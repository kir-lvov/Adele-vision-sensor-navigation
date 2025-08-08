import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import numpy as np
import json
import os
import math


class TreeDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None, seq_len=5):
        """Dataset class for tree images with avoidance points and wind sensor data

        Args:
            image_dir: Directory containing images
            json_path: Path to JSON annotations file
            transform: Image transformations
            seq_len: Length of sensor data sequence
        """
        self.image_dir = image_dir
        self.transform = transform
        self.seq_len = seq_len
        self.data = []
        self.image_size = 640  # Fixed image size for normalization

        # Load annotations from JSON file
        with open(json_path) as f:
            annotations = json.load(f)

        # Process each annotation item
        for item in annotations:
            img_path = os.path.join(self.image_dir, item['image'])

            # Generate random wind data for simulation
            wind_strength = np.random.uniform(0, 1, size=self.seq_len)
            wind_dir = np.random.uniform(0, 2 * np.pi, size=self.seq_len)

            # Convert wind data to x,y components
            wind_x = wind_strength * np.cos(wind_dir)
            wind_y = wind_strength * np.sin(wind_dir)

            # Process bounding box annotations
            boxes = []
            for ann in item['annotations']:
                # Normalize box coordinates (convert to YOLO format: center x,y + width,height)
                x = ann['coordinates']['x'] / self.image_size
                y = ann['coordinates']['y'] / self.image_size
                w = ann['coordinates']['width'] / self.image_size
                h = ann['coordinates']['height'] / self.image_size
                boxes.append([x + w / 2, y + h / 2, w, h])

            # Calculate avoidance point based on closest tree and wind influence
            if boxes:
                closest = min(boxes, key=lambda b: b[0] ** 2 + b[1] ** 2)
                wind_effect = 0.27  # Wind influence coefficient

                avoid_point = [
                    np.clip(closest[0] + 0.1 + np.mean(wind_x) * wind_effect, 0.05, 0.95),
                    np.clip(closest[1] + 0.1 + np.mean(wind_y) * wind_effect, 0.05, 0.95)
                ]
            else:
                avoid_point = [0.5, 0.5]  # Default center point if no trees

            # Stack and normalize sensor data (only wind_x and wind_y)
            sensor_data = np.vstack([
                wind_x,
                wind_y
            ]).T

            # Store processed data
            self.data.append({
                'image_path': img_path,
                'boxes': boxes,
                'avoid_point': avoid_point,
                'sensor_data': sensor_data
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get single data sample with error handling"""
        item = self.data[idx]

        try:
            # Load and transform image
            img = Image.open(item['image_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)

            return {
                'image': img,
                'boxes': torch.FloatTensor(item['boxes']) if item['boxes'] else torch.zeros(0, 4),
                'target': torch.FloatTensor(item['avoid_point']),
                'sensor': torch.FloatTensor(item['sensor_data'])  # Tensor of shape [seq_len, 2]
            }
        except Exception as e:
            None


def collate_fn(batch):
    """Custom collate function to handle variable-sized boxes and filter None values"""
    batch = [item for item in batch if item is not None]  # Filter None values
    if not batch:  # If all samples in batch are invalid
        return None

    # Stack batch elements
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    sensors = torch.stack([item['sensor'] for item in batch])
    return images, boxes, targets, sensors


class ADELE(nn.Module):
    """Attention-based Deep Learning for Environmental Learning and Evasion (ADELE) model"""

    def __init__(self, sensor_dim=2, seq_len=5):  # Changed sensor_dim to 2 (wind_x, wind_y)
        super().__init__()

        # Visual processing branch
        self.visual_backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.visual_proj = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Sensor data processing branch
        self.sensor_embed = nn.Linear(sensor_dim, 64)  # Now takes 2 parameters (wind_x, wind_y)
        self.pos_encoder = PositionalEncoding(64, seq_len)
        self.sensor_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 8, 256, dropout=0.2),
            num_layers=3
        )
        self.sensor_proj = nn.Linear(64 * seq_len, 256)

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(256, 8, dropout=0.2)

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # Normalization
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()  # Output normalized coordinates [0,1]
        )

    def forward(self, images, sensors):
        # Process visual input
        vis_features = self.visual_backbone(images)['0']  # Get FPN feature map
        vis_embed = self.visual_proj(vis_features)  # [B, 256]

        # Process sensor input
        sensor_embed = self.sensor_embed(sensors)  # [B, seq_len, 64]
        sensor_embed = self.pos_encoder(sensor_embed.permute(1, 0, 2))  # [seq_len, B, 64] + positional encoding
        sensor_embed = self.sensor_transformer(sensor_embed)  # [seq_len, B, 64]
        sensor_embed = sensor_embed.permute(1, 0, 2)  # [B, seq_len, 64]
        sensor_embed = sensor_embed.reshape(sensor_embed.size(0), -1)  # [B, seq_len*64]
        sensor_embed = self.sensor_proj(sensor_embed)  # [B, 256]

        # Cross-attention between visual and sensor features
        attn_out, _ = self.cross_attn(
            vis_embed.unsqueeze(0),  # [1, B, 256] (query)
            sensor_embed.unsqueeze(0),  # (key)
            sensor_embed.unsqueeze(0))  # (value)

        # Decode to final output (avoidance point coordinates)
        output = self.decoder(attn_out.squeeze(0))
        return output


class PositionalEncoding(nn.Module):
    """Transformer positional encoding for sequence data"""

    def __init__(self, d_model, max_len):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        """Add positional encoding to input tensor"""
        return x + self.pe[:x.size(0)]