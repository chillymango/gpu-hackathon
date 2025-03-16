import torch
import numpy as np
import pathlib
import wgpu
from wgpu.utils.compute import compute_with_buffers
from typing import Dict, Tuple, List, Optional, Union

# Cache for compiled shaders
shader_cache = {}

# Tensor tracking for WebGPU
class WebGPUTensorStorage:
    def __init__(self):
        self.tensors = {}
        self.next_id = 0
        # Use tensor object ID as the key for storage
        self.tensor_map = {}
    
    def register(self, tensor, tensor_data):
        tensor_id = id(tensor)
        self.tensors[tensor_id] = tensor_data
        return tensor_id
    
    def get(self, tensor):
        tensor_id = id(tensor)
        return self.tensors.get(tensor_id)
    
    def update(self, tensor, tensor_data):
        tensor_id = id(tensor)
        self.tensors[tensor_id] = tensor_data
        
    def remove(self, tensor):
        tensor_id = id(tensor)
        if tensor_id in self.tensors:
            del self.tensors[tensor_id]

# Global tensor storage
tensor_storage = WebGPUTensorStorage()

# WebGPU backend class for PyTorch
class WebGPUBackend:
    def __init__(self):
        print("Initializing WebGPU backend")
        
    def is_initialized(self):
        print("WebGPU: is_initialized called")
        return True

    def is_available(self):
        print("WebGPU: is_available called")
        return True

    def current_device(self):
        print("WebGPU: current_device called")
        return 0

    def _is_in_bad_fork(self):
        return False

    def manual_seed_all(self, seed: int):
        print(f"WebGPU: manual_seed_all({seed}) called")
        # WebGPU doesn't have built-in random functions that we control

    def device_count(self):
        print("WebGPU: device_count called")
        return 1

# Shader for matrix multiplication
def get_matmul_shader():
    if "matmul" not in shader_cache:
        shader_cache["matmul"] = """
        @group(0) @binding(0)
        var<storage, read> A: array<f32>;
        @group(0) @binding(1)
        var<storage, read> B: array<f32>;
        @group(0) @binding(2)
        var<storage, read_write> C: array<f32>;

        @group(0) @binding(3)
        var<storage, read> A_shape: array<u32>;
        @group(0) @binding(4)
        var<storage, read> B_shape: array<u32>;

        fn get_1d_index(row_ix: u32, col_ix: u32, n_cols: u32) -> u32 {
            return row_ix * n_cols + col_ix;
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let m: u32 = A_shape[0];
            let k: u32 = A_shape[1];
            let n: u32 = B_shape[1];
            
            // Check bounds
            if (gid.x >= n || gid.y >= m) {
                return;
            }

            var sum: f32 = 0.0;
            for (var i: u32 = 0; i < k; i++) {
                sum = sum + A[get_1d_index(gid.y, i, k)] * B[get_1d_index(i, gid.x, n)];
            }
            C[get_1d_index(gid.y, gid.x, n)] = sum;
        }
        """
    return shader_cache["matmul"]

# Shader for element-wise add
def get_add_shader():
    if "add" not in shader_cache:
        shader_cache["add"] = """
        @group(0) @binding(0)
        var<storage, read> A: array<f32>;
        @group(0) @binding(1)
        var<storage, read> B: array<f32>;
        @group(0) @binding(2)
        var<storage, read_write> C: array<f32>;
        @group(0) @binding(3)
        var<storage, read> alpha: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let index = gid.x;
            if (index >= arrayLength(&C)) {
                return;
            }
            C[index] = A[index] + alpha[0] * B[index];
        }
        """
    return shader_cache["add"]

# Shader for element-wise multiply
def get_mul_shader():
    if "mul" not in shader_cache:
        shader_cache["mul"] = """
        @group(0) @binding(0)
        var<storage, read> A: array<f32>;
        @group(0) @binding(1)
        var<storage, read> B: array<f32>;
        @group(0) @binding(2)
        var<storage, read_write> C: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let index = gid.x;
            if (index >= arrayLength(&C)) {
                return;
            }
            C[index] = A[index] * B[index];
        }
        """
    return shader_cache["mul"]

# Shader for ReLU
def get_relu_shader():
    if "relu" not in shader_cache:
        shader_cache["relu"] = """
        @group(0) @binding(0)
        var<storage, read> input: array<f32>;
        @group(0) @binding(1)
        var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let index = gid.x;
            if (index >= arrayLength(&output)) {
                return;
            }
            output[index] = max(0.0, input[index]);
        }
        """
    return shader_cache["relu"]

# Implementation of core operations
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    print(f"WebGPU: Creating empty tensor with size {size}")
    # Initialize with zeros for now
    shape = tuple(size)
    if dtype is None:
        dtype = torch.get_default_dtype()
    np_dtype = torch.empty(1, dtype=dtype).numpy().dtype
    
    # Create a CPU tensor with the correct shape and dtype
    cpu_tensor = torch.empty(size, dtype=dtype, device="cpu")
    
    # Register the tensor with our storage
    tensor_storage.register(cpu_tensor, {
        "data": cpu_tensor.numpy(),
        "shape": shape,
        "dtype": np_dtype
    })
    
    return cpu_tensor

# Register our backend with PyTorch
torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", WebGPUBackend())

# Register the implementations with PyTorch
@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format_impl(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    return empty_memory_format(size, dtype, layout, device, pin_memory, memory_format)

@torch.library.impl("aten::_to_copy", "privateuseone")
def _to_copy(tensor, **kwargs):
    print(f"WebGPU: Converting tensor with shape {tensor.shape} to CPU or other device")
    
    # Check the target device
    device_str = str(kwargs.get('device', 'cpu'))
    
    if device_str == 'cpu':
        # For CPU, we get the data from our tensor storage
        tensor_data = tensor_storage.get(tensor)
        if tensor_data:
            # Create a new CPU tensor with the data
            cpu_tensor = torch.tensor(tensor_data["data"], dtype=kwargs.get('dtype', tensor.dtype), device='cpu')
            return cpu_tensor
        
        # If we can't find the tensor, return a zeroed tensor as fallback
        return torch.zeros_like(tensor, device='cpu')
    else:
        # For non-CPU devices, use the default PyTorch implementation
        return torch.empty(tensor.shape, device=kwargs.get('device', 'cpu'), dtype=kwargs.get('dtype', tensor.dtype))

@torch.library.impl("aten::zeros", "privateuseone")
def zeros(size, dtype=None, layout=None, device=None, pin_memory=False):
    print(f"WebGPU: Creating zeros tensor with size {size}")
    # Create an empty tensor
    tensor = empty_memory_format(size, dtype, layout, device, pin_memory)
    
    # Fill with zeros
    tensor_data = tensor_storage.get(tensor)
    tensor_data["data"].fill(0)
    
    return tensor

@torch.library.impl("aten::ones", "privateuseone")
def ones(size, dtype=None, layout=None, device=None, pin_memory=False):
    print(f"WebGPU: Creating ones tensor with size {size}")
    # Create an empty tensor
    tensor = empty_memory_format(size, dtype, layout, device, pin_memory)
    
    # Fill with ones
    tensor_data = tensor_storage.get(tensor)
    tensor_data["data"].fill(1)
    
    return tensor

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(input, other, alpha=1):
    print(f"WebGPU: Adding tensors with shapes {input.shape} and {other.shape}")
    # Get the tensors from storage
    input_data = tensor_storage.get(input)["data"]
    other_data = tensor_storage.get(other)["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, input.device)
    
    # Run WebGPU compute operation
    bindings = {
        0: input_data.flatten(),
        1: other_data.flatten(),
        3: np.array([alpha], dtype=np.float32)
    }
    
    # Set up and run the compute operation
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(input.shape), "f")},
        shader=get_add_shader(),
        n=(np.prod(input.shape), 1, 1)
    )
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape(input.shape)
    
    return output

@torch.library.impl("aten::mul", "privateuseone")
def mul(input, other):
    print(f"WebGPU: Multiplying tensors with shapes {input.shape} and {other.shape}")
    # Get the tensors from storage
    input_data = tensor_storage.get(input)["data"]
    other_data = tensor_storage.get(other)["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, input.device)
    
    # Run WebGPU compute operation
    bindings = {
        0: input_data.flatten(),
        1: other_data.flatten()
    }
    
    # Set up and run the compute operation
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(input.shape), "f")},
        shader=get_mul_shader(),
        n=(np.prod(input.shape), 1, 1)
    )
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape(input.shape)
    
    return output

@torch.library.impl("aten::mm", "privateuseone")
def mm(input, other):
    print(f"WebGPU: Matrix multiplying tensors with shapes {input.shape} and {other.shape}")
    # Get the tensors from storage
    input_data = tensor_storage.get(input)["data"]
    other_data = tensor_storage.get(other)["data"]
    
    # Matrix shapes
    m, k = input.shape
    k2, n = other.shape
    
    if k != k2:
        raise ValueError(f"Incompatible matrix shapes for multiplication: {input.shape} and {other.shape}")
    
    # Prepare output tensor
    output = empty_memory_format((m, n), input.dtype, None, input.device)
    
    # Run WebGPU compute operation
    bindings = {
        0: input_data,
        1: other_data,
        3: np.array([m, k], dtype=np.uint32),
        4: np.array([k, n], dtype=np.uint32)
    }
    
    # Set up and run the compute operation
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (m * n, "f")},
        shader=get_matmul_shader(),
        n=(n, m, 1)  # n cols across "x dimension", m rows across "y dimension"
    )
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape((m, n))
    
    return output

@torch.library.impl("aten::relu", "privateuseone")
def relu(input):
    print(f"WebGPU: Computing ReLU of tensor with shape {input.shape}")
    # Get the tensor from storage
    input_data = tensor_storage.get(input)["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, input.device)
    
    # Run WebGPU compute operation
    bindings = {
        0: input_data.flatten()
    }
    
    # Set up and run the compute operation
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={1: (np.prod(input.shape), "f")},
        shader=get_relu_shader(),
        n=(np.prod(input.shape), 1, 1)
    )
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[1], dtype=np.float32).reshape(input.shape)
    
    return output

# Generate methods for the privateuse1 backend after we've registered our implementations
torch.utils.generate_methods_for_privateuse1_backend()

# Test function to demonstrate the WebGPU backend
def test_webgpu_backend():
    # Set the device
    device = torch.device("webgpu")
    
    print("\n=== Basic Tensor Creation ===")
    a = torch.ones(3, 3, device=device)
    b = torch.ones(3, 3, device=device)
    
    print("\n=== Basic Operations ===")
    c = a + b
    print("a + b shape:", c.shape)
    
    d = a * b
    print("a * b shape:", d.shape)
    
    e = torch.mm(a, b)
    print("a @ b shape:", e.shape)
    
    f = torch.relu(a - 0.5)
    print("relu(a - 0.5) shape:", f.shape)
    
    print("\n=== Testing with CPU verification ===")
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    
    c_cpu = a_cpu + b_cpu
    d_cpu = a_cpu * b_cpu
    e_cpu = torch.mm(a_cpu, b_cpu)
    f_cpu = torch.relu(a_cpu - 0.5)
    
    # Verify results
    c_webgpu = tensor_storage.get(c.webgpu_id)["data"]
    print("Add correct:", np.allclose(c_cpu.numpy(), c_webgpu))
    
    d_webgpu = tensor_storage.get(d.webgpu_id)["data"]
    print("Multiply correct:", np.allclose(d_cpu.numpy(), d_webgpu))
    
    e_webgpu = tensor_storage.get(e.webgpu_id)["data"]
    print("Matrix multiply correct:", np.allclose(e_cpu.numpy(), e_webgpu))
    
    f_webgpu = tensor_storage.get(f.webgpu_id)["data"]
    print("ReLU correct:", np.allclose(f_cpu.numpy(), f_webgpu))
    
    print("\nTest completed successfully!")
    return "WebGPU backend test completed"

if __name__ == "__main__":
    test_webgpu_backend()