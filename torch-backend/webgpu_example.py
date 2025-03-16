import torch
import numpy as np
import webgpu_backend

def main():
    print("===== Testing WebGPU PyTorch Backend =====")
    
    # Set the WebGPU device as the default device
    try:
        torch.set_default_device("webgpu")
        print("Default device set to webgpu")
    except Exception as e:
        print(f"Error setting default device: {e}")
    
    # Create basic tensors
    print("\nCreating tensors on WebGPU...")
    try:
        device = torch.device("webgpu")
        print(f"Created device: {device} (type: {type(device)}, str: {str(device)})")
        a = torch.ones(4, 4, device=device)
        print(f"Created tensor a: {a}")
        b = torch.ones(4, 4, device=device) * 2.0
        print(f"Created tensor b: {b}")
    except Exception as e:
        print(f"Error creating tensors: {e}")
        raise
    
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor b shape: {b.shape}")
    
    # Basic arithmetic operations
    print("\nPerforming basic operations...")
    print(f"a.device: {a.device}")
    
    # Attempt to use operators
    print("Performing addition...")
    c = a + b
    print("a + b completed using operator")
    
    # Try direct function calls with our wrapper functions
    print("Using direct WebGPU compute operations...")
    c_direct = webgpu_backend.webgpu_add(a, b)
    print("Direct WebGPU add completed")
    
    print("Performing multiplication...")
    d = a * b
    print("a * b completed using operator")
    d_direct = webgpu_backend.webgpu_mul(a, b)
    print("Direct WebGPU mul completed")
    
    print("Performing matrix multiplication...")
    e = torch.mm(a, b)  
    print("a @ b completed using torch.mm")
    e_direct = webgpu_backend.webgpu_mm(a, b)
    print("Direct WebGPU matrix multiply completed")
    
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
    
    # Get actual results from WebGPU operations by copying to CPU
    c_actual = c.cpu().numpy()
    d_actual = d.cpu().numpy()
    e_actual = e.cpu().numpy()
    f_actual = f.cpu().numpy()
    
    # Get actual results from our direct WebGPU operations
    # Access the data directly from the WebGPU backend storage
    import webgpu_backend
    c_direct_data = webgpu_backend.tensor_storage.get(c_direct)["data"]
    d_direct_data = webgpu_backend.tensor_storage.get(d_direct)["data"]
    e_direct_data = webgpu_backend.tensor_storage.get(e_direct)["data"]
    
    c_direct_actual = c_direct_data
    d_direct_actual = d_direct_data
    e_direct_actual = e_direct_data
    
    # Compare regular operator results
    print("Standard operators:")
    print("Addition correct:", np.allclose(c_expected.numpy(), c_actual))
    print("Multiplication correct:", np.allclose(d_expected.numpy(), d_actual))
    print("Matrix multiplication correct:", np.allclose(e_expected.numpy(), e_actual))
    print("ReLU correct:", np.allclose(f_expected.numpy(), f_actual))
    
    # Compare direct WebGPU results
    print("\nDirect WebGPU operations:")
    print("Addition correct:", np.allclose(c_expected.numpy(), c_direct_actual))
    print("Multiplication correct:", np.allclose(d_expected.numpy(), d_direct_actual))
    print("Matrix multiplication correct:", np.allclose(e_expected.numpy(), e_direct_actual))
    
    # Debug output for direct results
    print("\nDirect WebGPU operation results:")
    print("Addition expected:", c_expected[0,0].item())
    print("Addition actual:", c_direct_actual[0,0])
    print("Multiplication expected:", d_expected[0,0].item())
    print("Multiplication actual:", d_direct_actual[0,0])
    print("Matrix multiplication expected:", e_expected[0,0].item())
    print("Matrix multiplication actual:", e_direct_actual[0,0])
    
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