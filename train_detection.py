import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import ConfusionMatrix
import numpy as np

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

# Function to calculate mean IoU
def calculate_mIoU(all_preds, all_labels, num_classes=3):
    # Flatten the predictions and labels
    all_preds = all_preds.flatten()
    all_labels = all_labels.flatten()

    # Compute the confusion matrix
    cm = sklearn_confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # Calculate IoU for each class
    IoUs = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        if union == 0:
            IoUs.append(1)  # Perfect IoU if no true positives, false positives, or false negatives
        else:
            IoUs.append(intersection / union)

    # Calculate mean IoU
    mIoU = np.mean(IoUs)
    return mIoU

# Main training function
def train_detection(model_name="detector", num_epoch=20, batch_size=16, learning_rate=0.001):
    """
    Trains the detection model on the road dataset.
    
    Args:
        model_name (str): Name of the model to train.
        num_epoch (int): Number of epochs to train.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with augmentations for training and default for validation
    train_loader = load_data("road_data/train", transform_pipeline="aug", batch_size=batch_size, shuffle=True)
    val_loader = load_data("road_data/val", transform_pipeline="default", batch_size=batch_size, shuffle=False)

    # Initialize model, loss functions, and optimizer
    model = Detector(in_channels=3, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()  # For segmentation
    mae_loss = torch.nn.L1Loss()                      # For depth estimation

    # Training and validation loop
    for epoch in range(num_epoch):
        model.train()
        total_segmentation_loss, total_depth_loss = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            segmentation_labels = batch["track"].to(device)  # track is the segmentation mask
            depth_labels = batch["depth"].to(device)

            # Forward pass
            optimizer.zero_grad()
            segmentation_logits, depth_preds = model(images)

            # Compute losses
            seg_loss = cross_entropy_loss(segmentation_logits, segmentation_labels)
            depth_loss = mae_loss(depth_preds, depth_labels)
            loss = seg_loss + depth_loss  # Combine the two losses

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_segmentation_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()

        # Average losses for epoch
        avg_seg_loss = total_segmentation_loss / len(train_loader)
        avg_depth_loss = total_depth_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epoch}], Segmentation Loss: {avg_seg_loss:.4f}, Depth Loss: {avg_depth_loss:.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            total_val_depth_loss = 0

            for batch in val_loader:
                images = batch["image"].to(device)
                segmentation_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)

                # Forward pass
                segmentation_logits, depth_preds = model(images)

                # Accumulate predictions and labels for mIoU computation
                preds = segmentation_logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(segmentation_labels.cpu().numpy())

                # Calculate depth loss
                depth_loss = mae_loss(depth_preds, depth_labels)
                total_val_depth_loss += depth_loss.item()

            # Flatten predictions and labels for the entire validation set
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Calculate mIoU for segmentation
            mIoU = calculate_mIoU(all_preds, all_labels, num_classes=3)
            avg_val_depth_loss = total_val_depth_loss / len(val_loader)
            print(f"Validation - mIoU: {mIoU:.4f}, Depth MAE: {avg_val_depth_loss:.4f}")

        # Save model after each epoch directly using torch.save
        torch.save(model.state_dict(), "detector.th")