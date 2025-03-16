import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file

# Define a custom MNIST dataset class to load from raw files
class CustomMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        
        if self.train:
            image_file = os.path.join(root, 'train-images.idx3-ubyte')
            label_file = os.path.join(root, 'train-labels.idx1-ubyte')
        else:
            image_file = os.path.join(root, 't10k-images.idx3-ubyte')
            label_file = os.path.join(root, 't10k-labels.idx1-ubyte')
            
        self.data = read_image_file(image_file)
        self.targets = read_label_file(label_file)

# Define the neural network architecture
class MNISTNet(nn.Module):
    def __init__(self, hidden_size=5000):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)  # Input layer: 28x28 pixels flattened
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)  # Output layer: 10 classes (digits 0-9)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load MNIST dataset from raw files
data_dir = '/Users/alberthyang/Code/gpu-hack/data'
train_dataset = CustomMNIST(
    root=data_dir,
    train=True,
    transform=transform
)

test_dataset = CustomMNIST(
    root=data_dir,
    train=False,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Initialize the model
model = MNISTNet(hidden_size=5000).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved to mnist_model.pth")

# Plot training metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Training metrics plot saved to training_metrics.png") 