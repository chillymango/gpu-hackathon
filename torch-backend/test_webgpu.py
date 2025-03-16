import pytest
import torch
import numpy as np

# Import the WebGPU backend to register it with PyTorch
import webgpu_backend

def test_webgpu_tensor_higher_dimensions():
    """Test WebGPU operations with higher dimensional tensors."""
    
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create random 10x10x10 tensors on CPU
    shape = (10, 10, 10)
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)
    
    # Move tensors to WebGPU device
    device = torch.device("webgpu")
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    
    print(f"Created tensors with shape {shape}")
    
    # Perform addition on WebGPU
    c_gpu = a_gpu + b_gpu
    
    # Move result back to CPU for verification
    c_cpu = c_gpu.to("cpu")
    
    # Compute the expected result on CPU
    expected = a + b
    
    # Verify the result
    max_diff = torch.max(torch.abs(c_cpu - expected)).item()
    print(f"Maximum difference between WebGPU and CPU addition: {max_diff}")
    np.testing.assert_allclose(c_cpu.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
    
    print("WebGPU tensor addition test passed for higher dimensions!")
    
    # Try with scalar addition too
    scalar = 2.5
    d_gpu = a_gpu + scalar
    d_cpu = d_gpu.to("cpu")
    
    # Expected scalar addition result
    expected_scalar = a + scalar
    
    # Verify the scalar addition result
    max_diff_scalar = torch.max(torch.abs(d_cpu - expected_scalar)).item()
    print(f"Maximum difference between WebGPU and CPU scalar addition: {max_diff_scalar}")
    np.testing.assert_allclose(d_cpu.numpy(), expected_scalar.numpy(), rtol=1e-5, atol=1e-5)
    
    print("WebGPU scalar addition test passed for higher dimensions!")
    
    # Check matrix multiplication with 2D slices
    a_2d = a[:, 0, :]  # Shape: (10, 10)
    b_2d = b[:, 0, :]  # Shape: (10, 10)
    
    a_2d_gpu = a_2d.to(device)
    b_2d_gpu = b_2d.to(device)
    
    # Matrix multiplication on GPU
    mm_gpu = torch.mm(a_2d_gpu, b_2d_gpu)
    mm_cpu_from_gpu = mm_gpu.to("cpu")
    
    # Matrix multiplication on CPU
    mm_cpu = torch.mm(a_2d, b_2d)
    
    # Verify matrix multiplication
    max_diff_mm = torch.max(torch.abs(mm_cpu_from_gpu - mm_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU matrix multiplication: {max_diff_mm}")
    np.testing.assert_allclose(mm_cpu_from_gpu.numpy(), mm_cpu.numpy(), rtol=1e-4, atol=1e-4)
    
    print("WebGPU matrix multiplication test passed!")
    
    return (a, b, c_cpu, expected, d_cpu, expected_scalar, mm_cpu_from_gpu, mm_cpu)

if __name__ == "__main__":
    results = test_webgpu_tensor_higher_dimensions()