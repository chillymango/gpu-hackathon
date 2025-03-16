import torch
import test_backend

# Run a simple test with our custom PyTorch backend
def main():
    # Set our test backend as the default device
    torch.set_default_device("test")
    
    print("Creating tensors...")
    a = torch.ones(3, 3)  # Uses our "test" backend
    b = torch.ones(3, 3)  # Uses our "test" backend
    
    print("\nPerforming addition...")
    c = a + b
    
    print("\nPerforming multiplication...")
    d = a * b
    
    print("\nTensor info:")
    print("a:", a)
    print("b:", b)
    print("a + b:", c)
    print("a * b:", d)
    
if __name__ == "__main__":
    main()