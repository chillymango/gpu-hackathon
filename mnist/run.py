import torch
import torch.nn as nn
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets.mnist import read_image_file, read_label_file
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Button, Label, Frame
import PIL.Image, PIL.ImageDraw

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


class DrawingApp:
    def __init__(self, root, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        self.root = root
        self.root.title("MNIST Digit Classifier")
        self.root.geometry("600x700")  # Increased height to ensure labels are visible
        
        # Model parameters
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias
        
        # Same transformation as in train_mnist.py
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Flag to track if the canvas has changed since last prediction
        self.canvas_changed = False
        
        # Setup canvas
        self.setup_ui()
        
        # Bind keyboard events - 'r' to reset
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # Start the prediction timer
        self.schedule_prediction()
        
    def setup_ui(self):
        # PREDICTION LABEL FIRST - at the very top of the window
        # Result label - Make it extremely prominent
        self.result_label = Label(
            self.root, 
            text="Draw a digit (0-9)", 
            font=("Arial", 36, "bold"),
            bg="yellow",  # Very noticeable yellow background
            fg="blue",    # Blue text
            relief=tk.RAISED,
            borderwidth=4,
            height=1,
            width=20
        )
        self.result_label.pack(side=tk.TOP, padx=20, pady=20, fill=tk.X)
        
        # Controls frame
        controls_frame = Frame(self.root)
        controls_frame.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)
        
        # Clear button
        clear_btn = Button(controls_frame, text="Clear (or press 'R')", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Live prediction status label
        status_label = Label(controls_frame, text="Live prediction: every 1 second", font=("Arial", 10))
        status_label.pack(side=tk.LEFT, padx=10)
        
        # Frame for drawing area
        draw_frame = Frame(self.root)
        draw_frame.pack(side=tk.TOP, padx=10, pady=10)
        
        # Drawing canvas (280x280 px for better drawing, will be resized to 28x28 for prediction)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=draw_frame)
        self.canvas.get_tk_widget().pack()
        
        self.ax.set_xlim(0, 280)
        self.ax.set_ylim(0, 280)
        self.ax.invert_yaxis()  # Invert y-axis to match drawing coordinates
        self.ax.axis('off')
        
        # Create a blank image
        self.image = PIL.Image.new("L", (280, 280), color=0)
        self.draw = PIL.ImageDraw.Draw(self.image)
        
        # For tracking mouse movement
        self.prev_x = None
        self.prev_y = None
        
        # Connect events
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_down)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_up)
        
        # Display the blank image
        self.update_canvas()
        
    def on_key_press(self, event):
        # 'r' or 'R' to reset
        if event.char.lower() == 'r':
            self.clear_canvas()
        
    def on_mouse_down(self, event):
        self.prev_x = event.xdata
        self.prev_y = event.ydata
    
    def on_mouse_up(self, event):
        self.prev_x = None
        self.prev_y = None
    
    def on_mouse_move(self, event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None and self.prev_x is not None:
            # Draw line between previous and current position
            self.draw.line(
                [(self.prev_x, self.prev_y), (event.xdata, event.ydata)],
                fill=255,
                width=20
            )
            self.prev_x = event.xdata
            self.prev_y = event.ydata
            self.canvas_changed = True
            self.update_canvas()
    
    def update_canvas(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        self.ax.axis('off')
        self.canvas.draw()
    
    def clear_canvas(self):
        self.image = PIL.Image.new("L", (280, 280), color=0)
        self.draw = PIL.ImageDraw.Draw(self.image)
        self.canvas_changed = True
        self.update_canvas()
        self.result_label.config(text="Draw a digit (0-9)")
        self.top_predictions_label.config(text="")
    
    def schedule_prediction(self):
        """Schedule the next prediction in 1 second"""
        self.root.after(1000, self.predict_and_reschedule)
    
    def predict_and_reschedule(self):
        """Run prediction and schedule the next one"""
        self.predict()
        self.schedule_prediction()
    
    def predict(self):
        print("*** Prediction function is running ***")
        
        # Skip prediction if canvas hasn't changed
        if not self.canvas_changed and self.result_label.cget("text") != "Draw a digit (0-9)":
            print("Canvas hasn't changed, skipping prediction")
            return
        
        # Reset the flag
        self.canvas_changed = False
        
        # Check if the canvas is empty
        canvas_data = np.array(self.image)
        max_pixel = np.max(canvas_data)
        print(f"Canvas max pixel value: {max_pixel}")
        
        if max_pixel == 0:
            print("Canvas is empty, setting default text")
            self.result_label.config(text="Draw a digit (0-9)")
            return
            
        try:
            # Resize image to 28x28
            img_resized = self.image.resize((28, 28), PIL.Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Convert to tensor and apply transform
            img_tensor = self.transform(img_array)
            
            # Reshape to 28x28
            img_tensor = img_tensor.reshape(28, 28)
            
            # Classify
            prediction = classify_image(img_tensor, self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias)
            
            # Update label with large, clearly visible text
            print(f"Prediction result: {prediction} - Updating label now")
            prediction_text = f"PREDICTION: {prediction}"
            self.result_label.config(text=prediction_text)
            
            # Force update the UI
            self.root.update_idletasks()
            print(f"Label text is now: {self.result_label.cget('text')}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()


def main():
    # Load the model parameters
    model_path = 'mnist/mnist_model.pth'
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = load_and_convert_model(model_path)
    
    # Create the interactive drawing application
    root = tk.Tk()
    app = DrawingApp(root, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
    root.mainloop()
    
    # The code below is the original testing code, which is commented out since we're now using the interactive interface
    """
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
    """


if __name__ == "__main__":
    main()
