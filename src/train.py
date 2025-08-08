import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import TreeDataset, ADELE, collate_fn
from paths import PATHS


def train():
    """Main training function for ADELE model"""

    # Configuration dictionary with training parameters
    config = {
        'batch_size': 16,  # Number of samples per batch
        'lr': 2e-4,  # Initial learning rate
        'epochs': 20,  # Maximum number of training epochs
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Auto-select device
        'val_ratio': 0.2,  # Percentage of data for validation
        'patience': 13  # Early stopping patience (epochs)
    }

    # Image transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to fixed dimensions
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Color augmentation
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Slight blur
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with transformations
    dataset = TreeDataset(
        image_dir=str(PATHS["train_images"]),  # Path to training images
        json_path=str(PATHS["train_annotations"]),  # Path to annotations
        transform=transform  # Apply defined transforms
    )

    # Split dataset into training and validation sets
    train_size = int((1 - config['val_ratio']) * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Create data loaders for efficient batch processing
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,  # Shuffle for better training
        collate_fn=collate_fn,  # Custom collate function
        num_workers=4  # Parallel data loading
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,  # No need to shuffle validation
        collate_fn=collate_fn,
        num_workers=4
    )

    # Initialize model and move to device
    model_6 = ADELE().to(config['device'])

    # Reset BatchNorm statistics for fresh training
    for m in model_6.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model_6.parameters(),
                                  lr=config['lr'],
                                  weight_decay=0.01)

    # Learning rate scheduler with one-cycle policy
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.3)  # Percentage of increasing LR phase

    # Training state variables
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Metrics tracking
    train_losses = []  # Training losses (Smooth L1)
    val_losses = []  # Validation losses
    train_distances = []  # Euclidean distances in pixels (train)
    val_distances = []  # Euclidean distances in pixels (val)

    # Main training loop
    for epoch in range(config['epochs']):
        model_6.train()  # Set model to training mode
        epoch_train_loss = 0.0
        epoch_train_distance = 0.0

        # Process batches with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # Unpack batch data
            images = batch[0].to(config['device'])
            targets = batch[2].to(config['device'])  # Normalized [0,1]
            sensors = batch[3].to(config['device'])

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_6(images, sensors)  # Predictions in [0,1]

            # Scale to pixels for metric calculation
            outputs_pixels = outputs * 640.0
            targets_pixels = targets * 640.0

            # Calculate loss (in pixel space)
            loss = F.smooth_l1_loss(outputs_pixels, targets_pixels)
            epoch_train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate Euclidean distance (pixel space)
            distance = torch.norm(outputs_pixels - targets_pixels, dim=1).mean().item()
            epoch_train_distance += distance

        # Validation phase
        model_6.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        epoch_val_distance = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for batch in val_loader:
                images = batch[0].to(config['device'])
                targets = batch[2].to(config['device'])
                sensors = batch[3].to(config['device'])

                outputs = model_6(images, sensors)
                outputs_pixels = outputs * 640.0
                targets_pixels = targets * 640.0

                # Accumulate validation metrics
                epoch_val_loss += F.smooth_l1_loss(outputs_pixels, targets_pixels).item()
                epoch_val_distance += torch.norm(outputs_pixels - targets_pixels, dim=1).mean().item()

        # Calculate epoch averages
        epoch_train_loss /= len(train_loader)
        epoch_train_distance /= len(train_loader)
        epoch_val_loss /= len(val_loader)
        epoch_val_distance /= len(val_loader)

        # Store metrics for visualization
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_distances.append(epoch_train_distance)
        val_distances.append(epoch_val_distance)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss (Smooth L1): {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"Train Euclidean distance: {epoch_train_distance:.2f} px | Val distance: {epoch_val_distance:.2f} px")

        # Early stopping and model checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model_6.state_dict(), "best_model.pth")
            print("Saved best model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print("Early stopping triggered")
                break

    # Training visualization
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train', color='blue')
    plt.plot(val_losses, label='Val', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_distances, label='Train', color='green')
    plt.plot(val_distances, label='Val', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Euclidean Distance (px)')
    plt.title('Training and Validation Distance Error')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train()