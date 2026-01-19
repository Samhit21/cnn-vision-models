# I did use Chatgpt and Copilot for reference
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets.classification_dataset import SuperTuxDataset  # Use SuperTuxDataset here
from homework.models import Classifier

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Set dataset paths
train_data_path = './classification_data/train'  # Path to the directory with images and labels.csv for training
val_data_path = './classification_data/val'      # Path to the directory with images and labels.csv for validation

# Load datasets using SuperTuxDataset
train_dataset = SuperTuxDataset(dataset_path=train_data_path, transform_pipeline="aug")
val_dataset = SuperTuxDataset(dataset_path=val_data_path, transform_pipeline="default")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, loss, and optimizer
model = Classifier(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print epoch stats
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), "classifier.th")

