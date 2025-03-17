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

def test_argmax(tensors_on_device):
    """Test argmax operation on WebGPU."""
    a_gpu, _, a, _ = tensors_on_device
    
    # Test global argmax (no dimension specified)
    global_argmax_gpu = torch.argmax(a_gpu)
    global_argmax_cpu = torch.argmax(a)
    
    # Move result back to CPU for verification
    global_argmax_gpu_cpu = global_argmax_gpu.to("cpu")
    
    # Verify global argmax
    assert global_argmax_gpu_cpu.item() == global_argmax_cpu.item()
    print(f"Global argmax test passed! GPU: {global_argmax_gpu_cpu.item()}, CPU: {global_argmax_cpu.item()}")

def test_device_guard(tensors_on_device):
    """Test device guard on WebGPU."""
    a_gpu, b_gpu, a, b = tensors_on_device

    # Test that operations between tensors on different devices fail
    try:
        _ = a_gpu + b
        assert False, "Addition between WebGPU and CPU tensors should fail"
    except RuntimeError as e:
        print(f"Addition between WebGPU and CPU tensors correctly failed with: {e}")
        pass
    
    # Test that operations between tensors on the same device succeed
    try:
        result = a_gpu + b_gpu
        assert result.device.type == "webgpu", "Result should be on WebGPU device"
        print("Addition between WebGPU tensors succeeded as expected")
    except Exception as e:
        assert False, f"Addition between WebGPU tensors should succeed, but failed with: {e}"
    
    # Test that operations with scalar values work
    try:
        result = a_gpu + 1.0
        assert result.device.type == "webgpu", "Result should be on WebGPU device"
        print("Addition with scalar value succeeded as expected")
    except Exception as e:
        assert False, f"Addition with scalar value should succeed, but failed with: {e}"
    
    # Test that device guard works for other operations
    try:
        _ = torch.mm(a_gpu[:, 0, :], b[:, 0, :])
        assert False, "Matrix multiplication between WebGPU and CPU tensors should fail"
    except RuntimeError as e:
        print(f"Matrix multiplication between WebGPU and CPU tensors correctly failed with: {e}")
        pass
    
    print("Device guard tests passed!")

def test_memory_optimization(webgpu_device):
    """Test that host memory is dropped after copying to the device."""
    import sys
    import gc
    
    # Create a large tensor to make memory usage noticeable
    shape = (1000, 1000)  # 4MB tensor (4 bytes per float32 * 1M elements)
    
    # Record memory usage before creating the tensor
    gc.collect()
    memory_before = sys.getsizeof(webgpu_backend.gpu_data.data_dict)
    
    # Create a tensor and move it to the device
    a = torch.rand(shape, dtype=torch.float32)
    a_gpu = a.to(webgpu_device)
    
    # Force garbage collection to clean up any temporary objects
    gc.collect()
    
    # Check that the tensor has a buffer but no host data
    tensor_id = id(a_gpu)
    entry = webgpu_backend.gpu_data.data_dict.get(tensor_id)
    
    assert entry is not None, "Tensor entry should exist in gpu_data"
    assert entry['buffer'] is not None, "Tensor should have a GPU buffer"
    assert entry['data'] is None, "Tensor should not have host data"
    
    # Verify we can still get data when needed
    data = webgpu_backend.get_data(a_gpu)
    assert data is not None, "Should be able to retrieve data from GPU"
    assert data.shape == shape, "Retrieved data should have the correct shape"
    
    # Verify the data matches the original tensor
    a_cpu = a_gpu.to("cpu")
    np.testing.assert_allclose(a_cpu.numpy(), a.numpy(), rtol=1e-5, atol=1e-5)
    
    print("Memory optimization test passed!")

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
    test_relu_activation((a_gpu, b_gpu, a, b))
    test_argmax((a_gpu, b_gpu, a, b))
    test_device_guard((a_gpu, b_gpu, a, b))
    test_memory_optimization(device)
    
    print("\nAll tests passed successfully!")
    
    # To run with pytest (which will use the fixtures properly), use:
    # pytest test_webgpu.py -v