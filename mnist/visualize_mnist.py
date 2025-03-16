import torch
import torchvision
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

# Data transformations - just convert to tensor without normalization for visualization
transform = transforms.ToTensor()

# Load MNIST dataset from raw files
data_dir = 'data'
train_dataset = CustomMNIST(
    root=data_dir,
    train=True,
    transform=transform
)

# Function to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')

# Get random training images
dataiter = iter(torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True))
images, labels = next(dataiter)

# Create a grid of images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    imshow(images[i])
    plt.title(f"Label: {labels[i].item()}")

plt.tight_layout()
plt.savefig('mnist_samples.png')
print("MNIST samples saved to mnist_samples.png")

# Show a single image in more detail
plt.figure(figsize=(5, 5))
img = images[0].squeeze().numpy()
plt.imshow(img, cmap='gray')
plt.title(f"Label: {labels[0].item()}")
plt.colorbar()
plt.savefig('mnist_single_sample.png')
print("Single MNIST sample saved to mnist_single_sample.png") 