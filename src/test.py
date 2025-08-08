import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import cv2
from model import ADELE


# 1. Model Loading
def load_model(ckpt_path, device='cpu'):
    """Load trained ADELE model from checkpoint

    Args:
        ckpt_path: Path to model checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    """
    model = ADELE().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model


def prepare_input(image_path, seq_len=5):
    """Prepare input tensors for model inference

    Args:
        image_path: Path to input image
        seq_len: Length of wind data sequence

    Returns:
        image_tensor: Preprocessed image tensor [1, 3, 640, 640]
        wind_data: Simulated wind data tensor [1, seq_len, 2]
    """
    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to model's expected input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Generate simulated wind data (random for demo purposes)
    wind_strength = torch.rand(seq_len)  # Random wind strengths [0,1]
    wind_dir = torch.rand(seq_len) * 2 * math.pi  # Random directions [0,2π]

    # Convert to x,y components
    wind_x = wind_strength * torch.cos(wind_dir)
    wind_y = wind_strength * torch.sin(wind_dir)

    # Combine and reshape wind data
    wind_data = torch.stack([wind_x, wind_y], dim=1).unsqueeze(0)  # [1, seq_len, 2]

    return image_tensor, wind_data


def visualize(image_tensor, pred_point, true_point=None):
    """Visualize prediction on image

    Args:
        image_tensor: Preprocessed image tensor
        pred_point: Predicted avoidance point coordinates [x,y] normalized [0,1]
        true_point: Optional ground truth point for comparison
    """
    # Convert tensor to displayable image format
    # 1. Remove batch dimension and rearrange channels
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    # 2. Reverse ImageNet normalization
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)  # Ensure valid pixel values
    image = (image * 255).astype(np.uint8)  # Convert to 0-255 range
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Scale normalized coordinates to image dimensions
    h, w = image.shape[:2]
    pred_px = (int(pred_point[0] * w), int(pred_point[1] * h))

    # Draw prediction point
    cv2.circle(image, pred_px, 10, (0, 0, 255), -1)  # Red circle for prediction

    # Display result
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Select device automatically
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # 1. Load trained model
        model = load_model("best_model.pth", device)

        # 2. Prepare input data
        test_image_path = "C:\\Users\\Felix\\Desktop\\проект\\Dataset\\Датасет\\test\\-20_jpg.rf.691e954146790fe9596530eddf7c096e.jpg"
        image_tensor, wind_data = prepare_input(test_image_path)

        # 3. Run model inference
        with torch.no_grad():  # Disable gradient calculation
            pred = model(image_tensor.to(device), wind_data.to(device))
            pred_point = pred[0].cpu().numpy()  # Get first (only) prediction
            pred_point_normalize = pred_point * 640  # Scale to original image size

        # 4. Output results
        print(f"Predicted point coordinates: {pred_point_normalize}")
        visualize(image_tensor, pred_point)

    except Exception as e:
        print(f"Error during inference: {e}")