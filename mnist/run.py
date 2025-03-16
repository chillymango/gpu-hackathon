import torch
import torch.nn as nn
import os
import torch
from torchvision.datasets.mnist import read_image_file, read_label_file

import torch_backend.webgpu_backend as webgpu_backend


def load_and_convert_model(model_path):
    # Load the model
    #state_dict = torch.load(model_path, map_location=torch.device('webgpu'))
    state_dict = torch.load(model_path)
    
    # Extract weights and biases
    fc1_weight = state_dict['fc1.weight'].T.to('webgpu')  # [5000, 784]
    fc1_bias = state_dict['fc1.bias'].to('webgpu')      # [5000]
    fc2_weight = state_dict['fc2.weight'].T.to('webgpu')  # [10, 5000]
    fc2_bias = state_dict['fc2.bias'].to('webgpu')      # [10]
    
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


def classify_image(image, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """
    Classify a 28x28 image using matrix multiplications and ReLU activations
    
    Args:
        image: torch.Tensor of shape [28, 28]
        fc1_weight, fc1_bias, fc2_weight, fc2_bias: model parameters
    
    Returns:
        predicted class (0-9)
    """
    # Flatten the image to [784]
    x = image.flatten().to('webgpu')
    
    # First matmul: [784] x [5000, 784]^T = [5000]
    z1 = torch.mm(x, fc1_weight) + fc1_bias

    # First ReLU
    a1 = torch.relu(z1)
    
    # Second matmul: [5000] x [10, 5000]^T = [10]
    z2 = torch.mm(a1, fc2_weight) + fc2_bias
    
    # Second ReLU
    a2 = torch.relu(z2)

    # Return the class with highest score
    return torch.argmax(a2).item()


def load_mnist_images(data_dir='data', train=False, num_samples=10):
    """
    Load MNIST images from raw files
    
    Args:
        data_dir: directory containing MNIST data files
        train: whether to load training or test data
        num_samples: number of samples to load
    
    Returns:
        images: torch.Tensor of shape [num_samples, 28, 28]
        labels: torch.Tensor of shape [num_samples]
    """
    if train:
        image_file = os.path.join(data_dir, 'train-images.idx3-ubyte')
        label_file = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    else:
        image_file = os.path.join(data_dir, 't10k-images.idx3-ubyte')
        label_file = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    # Load images and labels
    images = read_image_file(image_file).float() / 255.0  # Normalize to [0, 1]
    labels = read_label_file(label_file)
    
    # Select a subset of samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    return images, labels


def main():
    # Load the model parameters
    model_path = 'mnist/mnist_model.pth'
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = load_and_convert_model(model_path)
    
    try:
        # Try to load real MNIST images
        images, labels = load_mnist_images(data_dir='data', train=False, num_samples=10)
        print(f"Loaded {len(images)} MNIST test images")

        # Classify each image
        correct = 0
        for i, (image, label) in enumerate(zip(images, labels)):
            predicted = classify_image(image, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
            print(f"Image {i}: True label = {label.item()}, Predicted = {predicted}")
            if predicted == label.item():
                correct += 1
        
        print(f"Accuracy: {correct}/{len(images)} ({100.0 * correct / len(images):.2f}%)")
        
    except FileNotFoundError:
        print("MNIST data files not found. Using a random image instead.")
        # Create a random 28x28 image
        random_image = torch.rand(28, 28)
        
        # Classify the image
        predicted_class = classify_image(random_image, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        print(f"Predicted class for random image: {predicted_class}")


if __name__ == "__main__":
    main()
