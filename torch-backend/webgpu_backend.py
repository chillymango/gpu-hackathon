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
        # Add webgpu_id attribute to the tensor
        tensor.webgpu_id = tensor
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
    print("Get add shader")
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
    # We'll use the "default" device which should be "webgpu", but we can't directly modify device attribute
    cpu_tensor = torch.empty(size, dtype=dtype, device="cpu")
    
    # Register the tensor with our storage
    tensor_data = {
        "data": cpu_tensor.numpy(),
        "shape": shape,
        "dtype": np_dtype,
        "on_webgpu": True  # Flag to indicate this tensor should be treated as on WebGPU
    }
    tensor_storage.register(cpu_tensor, tensor_data)
    
    return cpu_tensor

# Register our backend with PyTorch
torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", WebGPUBackend())
torch.utils.generate_methods_for_privateuse1_backend()

# Register the implementations with PyTorch
@torch.library.impl("aten::empty.memory_format", "webgpu")
def empty_memory_format_impl(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    return empty_memory_format(size, dtype, layout, device, pin_memory, memory_format)

@torch.library.impl("aten::_to_copy", "webgpu")
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
        
        # If we can't find the tensor, raise an error
        raise RuntimeError(f"Tensor not found in WebGPU storage when converting to CPU")
    elif device_str == 'webgpu':
        # Already on webgpu, return as is
        return tensor
    else:
        # For other devices, use the default PyTorch implementation
        return torch.empty(tensor.shape, device=kwargs.get('device', 'cpu'), dtype=kwargs.get('dtype', tensor.dtype))

@torch.library.impl("aten::zeros", "webgpu")
def zeros(size, dtype=None, layout=None, device=None, pin_memory=False):
    print(f"WebGPU: Creating zeros tensor with size {size}")
    # Create an empty tensor
    tensor = empty_memory_format(size, dtype, layout, device, pin_memory)
    
    # Fill with zeros
    tensor_data = tensor_storage.get(tensor)
    tensor_data["data"].fill(0)
    
    return tensor

@torch.library.impl("aten::ones", "webgpu")
def ones(size, dtype=None, layout=None, device=None, pin_memory=False):
    print(f"WebGPU: Creating ones tensor with size {size}")
    # Create an empty tensor
    tensor = empty_memory_format(size, dtype, layout, device, pin_memory)
    
    # Fill with ones
    tensor_data = tensor_storage.get(tensor)
    tensor_data["data"].fill(1)
    
    return tensor

@torch.library.impl("aten::add.Tensor", "webgpu")
def add_tensor(input, other, alpha=1):
    print(f"WebGPU: Adding tensors with shapes {input.shape} and {other.shape}")
    
    # Handle both CPU and WebGPU tensors
    if hasattr(input, 'device') and str(input.device) in ['cpu', 'privateuseone:0', 'torch.privateuse1:0', 'webgpu']:
        # Get data directly from the tensor if it's CPU tensor
        if str(input.device) == 'cpu':
            print("Converting CPU tensor to WebGPU for addition")
            input_data = input.numpy()
        else:
            # Get from storage for WebGPU tensor
            input_tensor_data = tensor_storage.get(input)
            if input_tensor_data is None:
                print("Warning: Input tensor not found in WebGPU storage, falling back to CPU data")
                input_data = input.cpu().numpy()
            else:
                input_data = input_tensor_data["data"]
        
        # Same for other tensor
        if str(other.device) == 'cpu':
            print("Converting CPU tensor to WebGPU for addition")
            other_data = other.numpy()
        else:
            other_tensor_data = tensor_storage.get(other)
            if other_tensor_data is None:
                print("Warning: Other tensor not found in WebGPU storage, falling back to CPU data")
                other_data = other.cpu().numpy()
            else:
                other_data = other_tensor_data["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, torch.device("webgpu"))
    
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

@torch.library.impl("aten::mul", "webgpu")
def mul(input, other):
    print(f"WebGPU: Multiplying tensors with shapes {input.shape} and {other.shape}")
    # Get the tensors from storage
    input_tensor_data = tensor_storage.get(input)
    other_tensor_data = tensor_storage.get(other)
    
    if input_tensor_data is None or other_tensor_data is None:
        raise RuntimeError(f"Tensor not found in WebGPU storage. This is likely because the tensor was not created with device='webgpu'")
    
    input_data = input_tensor_data["data"]
    other_data = other_tensor_data["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, torch.device("webgpu"))
    
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

@torch.library.impl("aten::mm", "webgpu")
def mm(input, other):
    print(f"WebGPU: Matrix multiplying tensors with shapes {input.shape} and {other.shape}")
    # Get the tensors from storage
    input_tensor_data = tensor_storage.get(input)
    other_tensor_data = tensor_storage.get(other)
    
    if input_tensor_data is None or other_tensor_data is None:
        raise RuntimeError(f"Tensor not found in WebGPU storage. This is likely because the tensor was not created with device='webgpu'")
    
    input_data = input_tensor_data["data"]
    other_data = other_tensor_data["data"]
    
    # Matrix shapes
    m, k = input.shape
    k2, n = other.shape
    
    if k != k2:
        raise ValueError(f"Incompatible matrix shapes for multiplication: {input.shape} and {other.shape}")
    
    # Prepare output tensor
    output = empty_memory_format((m, n), input.dtype, None, torch.device("webgpu"))
    
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

@torch.library.impl("aten::relu", "webgpu")
def relu(input):
    print(f"WebGPU: Computing ReLU of tensor with shape {input.shape}")
    # Get the tensor from storage
    input_tensor_data = tensor_storage.get(input)
    
    if input_tensor_data is None:
        raise RuntimeError(f"Tensor not found in WebGPU storage. This is likely because the tensor was not created with device='webgpu'")
    
    input_data = input_tensor_data["data"]
    
    # Prepare output tensor
    output = empty_memory_format(input.shape, input.dtype, None, torch.device("webgpu"))
    
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

# Wrapper functions for direct calling
def webgpu_add(a, b, alpha=1):
    print(f"WebGPU: Direct add call with shapes {a.shape} and {b.shape}")
    # Get the tensors' data
    if hasattr(a, 'numpy'):
        a_data = a.numpy()
    else:
        a_data = tensor_storage.get(a)["data"] if tensor_storage.get(a) else a.cpu().numpy()
        
    if hasattr(b, 'numpy'):
        b_data = b.numpy()
    else:
        b_data = tensor_storage.get(b)["data"] if tensor_storage.get(b) else b.cpu().numpy()
    
    # Prepare output tensor
    output = empty_memory_format(a.shape, a.dtype, None, None)
    
    # Print input values for debugging
    print(f"a_data: shape={a_data.shape}, min={a_data.min()}, max={a_data.max()}")
    print(f"b_data: shape={b_data.shape}, min={b_data.min()}, max={b_data.max()}")
    
    # Run WebGPU compute operation
    bindings = {
        0: a_data.flatten(),
        1: b_data.flatten(),
        3: np.array([alpha], dtype=np.float32)
    }
    
    # Set up and run the compute operation
    print("Running compute_with_buffers for addition...")
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(a.shape), "f")},
        shader=get_add_shader(),
        n=(np.prod(a.shape), 1, 1)
    )
    
    # Debug the output buffer
    print(f"Output buffer keys: {list(out.keys())}")
    if 2 in out:
        print(f"Output buffer length: {len(out[2])}")
        output_array = np.frombuffer(out[2], dtype=np.float32)
        print(f"Output array shape: {output_array.shape}, min: {output_array.min()}, max: {output_array.max()}")
    else:
        print("No output buffer with key 2!")
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape(a.shape)
    print(f"Updated output tensor data: shape={output_data['data'].shape}, min={output_data['data'].min()}, max={output_data['data'].max()}")
    
    return output

def webgpu_mul(a, b):
    print(f"WebGPU: Direct multiply call with shapes {a.shape} and {b.shape}")
    # Get the tensors' data
    if hasattr(a, 'numpy'):
        a_data = a.numpy()
    else:
        a_data = tensor_storage.get(a)["data"] if tensor_storage.get(a) else a.cpu().numpy()
        
    if hasattr(b, 'numpy'):
        b_data = b.numpy()
    else:
        b_data = tensor_storage.get(b)["data"] if tensor_storage.get(b) else b.cpu().numpy()
    
    # Print input values for debugging
    print(f"a_data: shape={a_data.shape}, min={a_data.min()}, max={a_data.max()}")
    print(f"b_data: shape={b_data.shape}, min={b_data.min()}, max={b_data.max()}")
    
    # Prepare output tensor
    output = empty_memory_format(a.shape, a.dtype, None, None)
    
    # Run WebGPU compute operation
    bindings = {
        0: a_data.flatten(),
        1: b_data.flatten()
    }
    
    # Set up and run the compute operation
    print("Running compute_with_buffers for multiplication...")
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(a.shape), "f")},
        shader=get_mul_shader(),
        n=(np.prod(a.shape), 1, 1)
    )
    
    # Debug the output buffer
    print(f"Output buffer keys: {list(out.keys())}")
    if 2 in out:
        print(f"Output buffer length: {len(out[2])}")
        output_array = np.frombuffer(out[2], dtype=np.float32)
        print(f"Output array shape: {output_array.shape}, min: {output_array.min()}, max: {output_array.max()}")
    else:
        print("No output buffer with key 2!")
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape(a.shape)
    print(f"Updated output tensor data: shape={output_data['data'].shape}, min={output_data['data'].min()}, max={output_data['data'].max()}")
    
    return output

def webgpu_mm(a, b):
    print(f"WebGPU: Direct matrix multiply call with shapes {a.shape} and {b.shape}")
    # Get the tensors' data
    if hasattr(a, 'numpy'):
        a_data = a.numpy()
    else:
        a_data = tensor_storage.get(a)["data"] if tensor_storage.get(a) else a.cpu().numpy()
        
    if hasattr(b, 'numpy'):
        b_data = b.numpy()
    else:
        b_data = tensor_storage.get(b)["data"] if tensor_storage.get(b) else b.cpu().numpy()
    
    # Print input values for debugging
    print(f"a_data: shape={a_data.shape}, min={a_data.min()}, max={a_data.max()}")
    print(f"b_data: shape={b_data.shape}, min={b_data.min()}, max={b_data.max()}")
    
    # Matrix shapes
    m, k = a.shape
    k2, n = b.shape
    
    if k != k2:
        raise ValueError(f"Incompatible matrix shapes for multiplication: {a.shape} and {b.shape}")
    
    # Prepare output tensor
    output = empty_memory_format((m, n), a.dtype, None, None)
    
    # Run WebGPU compute operation
    bindings = {
        0: a_data,
        1: b_data,
        3: np.array([m, k], dtype=np.uint32),
        4: np.array([k, n], dtype=np.uint32)
    }
    
    # Set up and run the compute operation
    print("Running compute_with_buffers for matrix multiplication...")
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (m * n, "f")},
        shader=get_matmul_shader(),
        n=(n, m, 1)  # n cols across "x dimension", m rows across "y dimension"
    )
    
    # Debug the output buffer
    print(f"Output buffer keys: {list(out.keys())}")
    if 2 in out:
        print(f"Output buffer length: {len(out[2])}")
        output_array = np.frombuffer(out[2], dtype=np.float32)
        print(f"Output array shape: {output_array.shape}, min: {output_array.min()}, max: {output_array.max()}")
    else:
        print("No output buffer with key 2!")
    
    # Update the output tensor
    output_data = tensor_storage.get(output)
    output_data["data"] = np.frombuffer(out[2], dtype=np.float32).reshape((m, n))
    print(f"Updated output tensor data: shape={output_data['data'].shape}, min={output_data['data'].min()}, max={output_data['data'].max()}")
    
    return output

# Test function to demonstrate the WebGPU backend
def test_webgpu_backend():
    # Set the device
    device = torch.device("webgpu")
    
    print("\n=== Basic Tensor Creation ===")
    a = torch.ones(3, 3, device=device)
    b = torch.ones(3, 3, device=device)
    c = torch.ones(3, 3)
    test = a@c
    
    print("\n=== Basic Operations ===")
    c = webgpu_add(a, b)
    print("a + b shape:", c.shape)
    
    d = webgpu_mul(a, b)
    print("a * b shape:", d.shape)
    
    e = webgpu_mm(a, b)
    print("a @ b shape:", e.shape)
    
    print("\n=== Testing with CPU verification ===")
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    
    c_cpu = a_cpu + b_cpu
    d_cpu = a_cpu * b_cpu
    e_cpu = torch.mm(a_cpu, b_cpu)
    
    # Verify results
    c_data = tensor_storage.get(c)["data"]
    print("Add correct:", np.allclose(c_cpu.numpy(), c_data))
    
    d_data = tensor_storage.get(d)["data"]
    print("Multiply correct:", np.allclose(d_cpu.numpy(), d_data))
    
    e_data = tensor_storage.get(e)["data"]
    print("Matrix multiply correct:", np.allclose(e_cpu.numpy(), e_data))
    
    print("\nTest completed successfully!")
    return "WebGPU backend test completed"

if __name__ == "__main__":
    test_webgpu_backend()