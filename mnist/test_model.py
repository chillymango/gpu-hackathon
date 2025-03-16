import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
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

# Load the trained model
model = MNISTNet(hidden_size=5000).to(device)
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load test dataset
data_dir = 'data'
test_dataset = CustomMNIST(
    root=data_dir,
    train=False,
    transform=transform
)

# Get a batch of test images
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Make predictions
with torch.no_grad():
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Move tensors back to CPU for visualization
    images = images.cpu()
    predicted = predicted.cpu()

# Display images and predictions
fig = plt.figure(figsize=(15, 6))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    
    # Display the image
    img = images[i].squeeze().numpy()
    # Denormalize the image for display
    img = img * 0.3081 + 0.1307
    ax.imshow(img, cmap='gray')
    
    # Set the title with true and predicted labels
    ax.set_title(f'True: {labels[i]}, Pred: {predicted[i]}', 
                 color=('green' if predicted[i] == labels[i] else 'red'))
    ax.axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png')
print("Test predictions saved to test_predictions.png")

# Calculate overall accuracy on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total:.2f}%') 