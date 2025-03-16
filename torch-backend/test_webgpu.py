import pytest
import torch
import numpy as np

# Import the WebGPU backend to register it with PyTorch
import webgpu_backend

@pytest.fixture
def webgpu_device():
    """Get the WebGPU device."""
    return torch.device("webgpu")

@pytest.fixture
def random_tensors():
    """Create random tensors for testing."""
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create random 10x10x10 tensors on CPU
    shape = (10, 10, 10)
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)
    
    return a, b, shape

@pytest.fixture
def tensors_on_device(random_tensors, webgpu_device):
    """Move tensors to WebGPU device."""
    a, b, _ = random_tensors
    a_gpu = a.to(webgpu_device)
    b_gpu = b.to(webgpu_device)
    return a_gpu, b_gpu, a, b

@pytest.fixture
def matrix_tensors(random_tensors, webgpu_device):
    """Create 2D tensors for matrix multiplication."""
    a, b, _ = random_tensors
    # Extract 2D slices for matrix multiplication
    a_2d = a[:, 0, :]  # Shape: (10, 10)
    b_2d = b[:, 0, :]  # Shape: (10, 10)
    
    a_2d_gpu = a_2d.to(webgpu_device)
    b_2d_gpu = b_2d.to(webgpu_device)
    
    return a_2d_gpu, b_2d_gpu, a_2d, b_2d

def test_device_assignment(tensors_on_device):
    """Test that tensors are correctly assigned to the WebGPU device."""
    a_gpu, b_gpu, _, _ = tensors_on_device
    
    assert str(a_gpu.device) == "webgpu:0"
    assert str(b_gpu.device) == "webgpu:0"
    
    print(f"Device assignment test passed! Tensor device: {a_gpu.device}")

def test_tensor_addition(tensors_on_device):
    """Test element-wise addition of tensors on WebGPU."""
    a_gpu, b_gpu, a, b = tensors_on_device
    
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
    print("Tensor addition test passed!")

def test_scalar_addition(tensors_on_device):
    """Test scalar addition on WebGPU."""
    a_gpu, _, a, _ = tensors_on_device
    
    # Try with scalar addition
    scalar = 2.5
    d_gpu = a_gpu + scalar
    d_cpu = d_gpu.to("cpu")
    
    # Expected scalar addition result
    expected_scalar = a + scalar
    
    # Verify the scalar addition result
    max_diff_scalar = torch.max(torch.abs(d_cpu - expected_scalar)).item()
    print(f"Maximum difference between WebGPU and CPU scalar addition: {max_diff_scalar}")
    
    np.testing.assert_allclose(d_cpu.numpy(), expected_scalar.numpy(), rtol=1e-5, atol=1e-5)
    print("Scalar addition test passed!")

def test_matrix_multiplication(matrix_tensors):
    """Test matrix multiplication on WebGPU."""
    a_2d_gpu, b_2d_gpu, a_2d, b_2d = matrix_tensors
    
    # Matrix multiplication on GPU
    mm_gpu = torch.mm(a_2d_gpu, b_2d_gpu)
    mm_cpu_from_gpu = mm_gpu.to("cpu")
    
    # Matrix multiplication on CPU
    mm_cpu = torch.mm(a_2d, b_2d)
    
    # Verify matrix multiplication
    max_diff_mm = torch.max(torch.abs(mm_cpu_from_gpu - mm_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU matrix multiplication: {max_diff_mm}")
    
    np.testing.assert_allclose(mm_cpu_from_gpu.numpy(), mm_cpu.numpy(), rtol=1e-4, atol=1e-4)
    print("Matrix multiplication test passed!")
    
def test_vector_matrix_multiplication(random_tensors, webgpu_device):
    """Test vector-matrix multiplication on WebGPU."""
    a, b, _ = random_tensors
    
    # Create a vector (1D tensor) and a matrix
    vector = a[0, 0, :]  # Shape: (10,)
    matrix = b[:, 0, :].t()  # Shape: (10, 10)
    
    # Move to GPU
    vector_gpu = vector.to(webgpu_device)
    matrix_gpu = matrix.to(webgpu_device)
    
    # We need to reshape the vector to 2D for mm operation
    vector_2d = vector.reshape(1, -1)  # Shape: (1, 10)
    vector_2d_gpu = vector_gpu.reshape(1, -1)
    
    # Matrix multiplication on GPU: (1, 10) @ (10, 10) = (1, 10)
    mm_gpu = torch.mm(vector_2d_gpu, matrix_gpu)
    mm_cpu_from_gpu = mm_gpu.to("cpu")
    
    # Matrix multiplication on CPU
    mm_cpu = torch.mm(vector_2d, matrix)
    
    # Verify vector-matrix multiplication
    max_diff_mm = torch.max(torch.abs(mm_cpu_from_gpu - mm_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU vector-matrix multiplication: {max_diff_mm}")
    
    np.testing.assert_allclose(mm_cpu_from_gpu.numpy(), mm_cpu.numpy(), rtol=1e-4, atol=1e-4)
    
    # Also test the @ operator, which should handle the broadcasting automatically
    matmul_gpu = vector_gpu @ matrix_gpu
    matmul_cpu_from_gpu = matmul_gpu.to("cpu")
    
    # Matrix multiplication on CPU using @ operator
    matmul_cpu = vector @ matrix
    
    # Verify @ operator multiplication
    max_diff_matmul = torch.max(torch.abs(matmul_cpu_from_gpu - matmul_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU @ operator multiplication: {max_diff_matmul}")
    
    np.testing.assert_allclose(matmul_cpu_from_gpu.numpy(), matmul_cpu.numpy(), rtol=1e-4, atol=1e-4)
    
    print("Vector-matrix multiplication test passed!")

def test_relu_activation(tensors_on_device):
    """Test ReLU activation on WebGPU."""
    a_gpu, _, a, _ = tensors_on_device
    
    # Create tensor with negative values
    neg_values = a - 0.5  # Half the values should be negative
    neg_values_gpu = neg_values.to(a_gpu.device)
    
    # Apply ReLU on GPU
    relu_gpu = torch.relu(neg_values_gpu)
    relu_cpu_from_gpu = relu_gpu.to("cpu")
    
    # Apply ReLU on CPU
    relu_cpu = torch.relu(neg_values)
    
    # Verify ReLU results
    max_diff_relu = torch.max(torch.abs(relu_cpu_from_gpu - relu_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU ReLU: {max_diff_relu}")
    
    np.testing.assert_allclose(relu_cpu_from_gpu.numpy(), relu_cpu.numpy(), rtol=1e-5, atol=1e-5)
    print("ReLU activation test passed!")
    
def test_as_strided(tensors_on_device):
    """Test as_strided operation on WebGPU."""
    a_gpu, _, a, _ = tensors_on_device
    
    # Original tensor shape is (10, 10, 10)
    # Extract a (2, 2) matrix from the first slice with different strides and offsets
    
    # Case 1: Extract a 2x2 submatrix with default strides
    size = (2, 2)
    stride = (1, 10)  # Move 1 element to the next row, 10 elements to the next column
    offset = 0
    
    # Apply as_strided on CPU
    a_strided_cpu = torch.as_strided(a, size, stride, offset)
    
    # Apply as_strided on GPU
    a_strided_gpu = torch.as_strided(a_gpu, size, stride, offset)
    a_strided_gpu_cpu = a_strided_gpu.to("cpu")
    
    # Verify results
    max_diff = torch.max(torch.abs(a_strided_gpu_cpu - a_strided_cpu)).item()
    print(f"Maximum difference between WebGPU and CPU as_strided (case 1): {max_diff}")
    np.testing.assert_allclose(a_strided_gpu_cpu.numpy(), a_strided_cpu.numpy(), rtol=1e-5, atol=1e-5)
    
    # Case 2: Extract a 2x3 submatrix with custom strides and offset
    size = (2, 3)
    stride = (10, 1)  # Move 10 elements to the next row, 1 element to the next column
    offset = 5  # Start from the 5th element
    
    # Apply as_strided on CPU
    a_strided_cpu2 = torch.as_strided(a, size, stride, offset)
    
    # Apply as_strided on GPU
    a_strided_gpu2 = torch.as_strided(a_gpu, size, stride, offset)
    a_strided_gpu_cpu2 = a_strided_gpu2.to("cpu")
    
    # Verify results
    max_diff2 = torch.max(torch.abs(a_strided_gpu_cpu2 - a_strided_cpu2)).item()
    print(f"Maximum difference between WebGPU and CPU as_strided (case 2): {max_diff2}")
    np.testing.assert_allclose(a_strided_gpu_cpu2.numpy(), a_strided_cpu2.numpy(), rtol=1e-5, atol=1e-5)
    
    print("as_strided test passed!")

if __name__ == "__main__":
    # When running as a script, we need to create the data directly instead of using fixtures
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create device
    device = torch.device("webgpu")
    
    # Create random tensors
    shape = (10, 10, 10)
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)
    
    # Move to device
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    
    # Create 2D tensors for matmul
    a_2d = a[:, 0, :]
    b_2d = b[:, 0, :]
    a_2d_gpu = a_2d.to(device)
    b_2d_gpu = b_2d.to(device)
    
    # Run tests
    test_device_assignment((a_gpu, b_gpu, a, b))
    test_tensor_addition((a_gpu, b_gpu, a, b))
    test_scalar_addition((a_gpu, b_gpu, a, b))
    test_matrix_multiplication((a_2d_gpu, b_2d_gpu, a_2d, b_2d))
    test_vector_matrix_multiplication((a, b, shape), device)
    test_relu_activation((a_gpu, b_gpu, a, b))
    test_as_strided((a_gpu, b_gpu, a, b))
    
    print("\nAll tests passed successfully!")
    
    # To run with pytest (which will use the fixtures properly), use:
    # pytest test_webgpu.py -v