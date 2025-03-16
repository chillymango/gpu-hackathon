import torch
import numpy as np
import webgpu_backend

def main():
    print("===== Testing WebGPU PyTorch Backend =====")
    
    # Set the WebGPU device as the default device
    torch.set_default_device("webgpu")
    
    # Create basic tensors
    print("\nCreating tensors on WebGPU...")
    a = torch.ones(4, 4)
    b = torch.ones(4, 4) * 2.0
    
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor b shape: {b.shape}")
    
    # Basic arithmetic operations
    print("\nPerforming basic operations...")
    c = a + b
    print("a + b completed")
    
    d = a * b
    print("a * b completed")
    
    e = torch.mm(a, b)
    print("a @ b (matrix multiply) completed")
    
    f = torch.relu(a - 0.5)
    print("relu(a - 0.5) completed")
    
    # Verify results by comparing with CPU
    print("\nVerifying results with CPU...")
    a_cpu = torch.ones(4, 4, device="cpu")
    b_cpu = torch.ones(4, 4, device="cpu") * 2.0
    
    # Compute expected results on CPU
    c_expected = a_cpu + b_cpu
    d_expected = a_cpu * b_cpu
    e_expected = torch.mm(a_cpu, b_cpu)
    f_expected = torch.relu(a_cpu - 0.5)
    
    # Get actual results from WebGPU by copying to CPU first
    c_actual = c.cpu().numpy()
    d_actual = d.cpu().numpy()
    e_actual = e.cpu().numpy()
    f_actual = f.cpu().numpy()
    
    # Compare results
    print("Addition correct:", np.allclose(c_expected.numpy(), c_actual))
    print("Multiplication correct:", np.allclose(d_expected.numpy(), d_actual))
    print("Matrix multiplication correct:", np.allclose(e_expected.numpy(), e_actual))
    print("ReLU correct:", np.allclose(f_expected.numpy(), f_actual))
    
    # Print a sample of the results
    print("\nSample results:")
    print("a[0,0]:", a_cpu[0,0].item())
    print("b[0,0]:", b_cpu[0,0].item())
    print("(a+b)[0,0] expected:", c_expected[0,0].item())
    print("(a+b)[0,0] actual:", c_actual[0,0])
    print("(a*b)[0,0] expected:", d_expected[0,0].item())
    print("(a*b)[0,0] actual:", d_actual[0,0])
    
    print("\n===== WebGPU Backend Test Complete =====")

if __name__ == "__main__":
    main()