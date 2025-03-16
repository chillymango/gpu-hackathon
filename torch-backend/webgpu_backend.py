# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
import torch
import numpy as np
import pathlib
import wgpu
from wgpu.utils.compute import compute_with_buffers
import torch.utils.cpp_extension
import math, operator, functools, inspect
from typing import Dict, Tuple, List, Optional, Union

# Enable debug output if needed
import os
TORCH_DEBUG = os.environ.get("TORCH_DEBUG", False)

torch.autograd.grad_mode.set_multithreading_enabled(False)

# Load C++ extension for device handling
mod = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")],
    verbose=False
)

# Cache for compiled shaders
shader_cache = {}

class WebGPUData:
    def __init__(self):
        self.data_dict = {}
    
    def store(self, tensor, data):
        self.data_dict[id(tensor)] = data
        
    def get(self, tensor):
        return self.data_dict.get(id(tensor))
    
    def remove(self, tensor):
        if id(tensor) in self.data_dict:
            del self.data_dict[id(tensor)]

# Global storage
gpu_data = WebGPUData()

# Device handling functions
def _to_torch_dtype(dtype):
    """Convert numpy dtype to torch dtype"""
    if dtype == np.float32:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    elif dtype == np.int32:
        return torch.int32
    elif dtype == np.int64:
        return torch.int64
    else:
        return torch.float32  # Default

def _from_torch_dtype(dtype):
    """Convert torch dtype to numpy dtype"""
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    else:
        return np.float32  # Default

def _from_torch_device(device: torch.device):
    return f"webgpu:{device.index or 0}"

def _to_torch_device(device: str):
    return torch.device("webgpu", int(device.partition(":")[2] or 0))

# Wrap and unwrap functions
def wrap(x) -> torch.Tensor:
    """Wrap a numpy array into a WebGPU tensor by first creating a CPU tensor"""
    # Use mod.wrap from the C++ extension, similar to what tinygrad does
    result = mod.wrap(x, _to_torch_dtype(x.dtype), 0)
    gpu_data.store(result, x)
    return result

def unwrap(x: torch.Tensor):
    """Unwrap a WebGPU tensor to get the underlying numpy array"""
    assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
    return gpu_data.get(x)

# ===== WebGPU Compute Shader Implementations =====

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

# WebGPU backend class for PyTorch
class WebGPUBackend:
    def __init__(self):
        print("Initializing WebGPU backend")
        
    def is_initialized(self):
        if TORCH_DEBUG:
            print("WebGPU: is_initialized called")
        return True

    def is_available(self):
        if TORCH_DEBUG:
            print("WebGPU: is_available called")
        return True

    def current_device(self):
        if TORCH_DEBUG:
            print("WebGPU: current_device called")
        return 0

    def _is_in_bad_fork(self):
        if TORCH_DEBUG:
            print("WebGPU: _is_in_bad_fork called")
        return False

    def manual_seed_all(self, seed: int):
        if TORCH_DEBUG:
            print(f"WebGPU: manual_seed_all called with seed {seed}")
        pass

    def device_count(self):
        if TORCH_DEBUG:
            print("WebGPU: device_count called")
        return 1

# Register our backend with PyTorch
torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", WebGPUBackend())

# Helper functions
def get_data(tensor):
    """Get the raw data from a tensor"""
    if tensor.device.type == 'cpu':
        return tensor.detach().numpy()
    
    data = gpu_data.get(tensor)
    if data is not None:
        return data
    
    # Fallback to CPU
    return tensor.cpu().detach().numpy()

# ===== WebGPU Operation Implementations =====

def webgpu_add(a, b, alpha=1):
    if TORCH_DEBUG:
        print(f"WebGPU add operation: {a.shape} + {b.shape}")
    
    # Get the tensor data
    a_data = get_data(a)
    b_data = get_data(b)
    
    # Set up WebGPU compute operation
    bindings = {
        0: a_data.flatten(),
        1: b_data.flatten(),
        3: np.array([alpha], dtype=np.float32)
    }
    
    # Execute compute shader
    output = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(a.shape), 'f')},
        shader=get_add_shader(),
        n=(np.prod(a.shape), 1, 1)
    )
    
    # Get the result
    result_data = np.frombuffer(output[2], dtype=np.float32).reshape(a.shape)
    
    # Create and return the wrapped tensor
    return wrap(result_data)

def webgpu_mul(a, b):
    if TORCH_DEBUG:
        print(f"WebGPU multiply operation: {a.shape} * {b.shape}")
    
    # Get the tensor data
    a_data = get_data(a)
    b_data = get_data(b)
    
    # Set up WebGPU compute operation
    bindings = {
        0: a_data.flatten(),
        1: b_data.flatten()
    }
    
    # Execute compute shader
    output = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (np.prod(a.shape), 'f')},
        shader=get_mul_shader(),
        n=(np.prod(a.shape), 1, 1)
    )
    
    # Get the result
    result_data = np.frombuffer(output[2], dtype=np.float32).reshape(a.shape)
    
    # Create and return the wrapped tensor
    return wrap(result_data)

def webgpu_mm(a, b):
    if TORCH_DEBUG:
        print(f"WebGPU matrix multiply operation: {a.shape} @ {b.shape}")
    
    # Get the tensor data
    a_data = get_data(a)
    b_data = get_data(b)
    
    # Matrix shapes
    m, k = a.shape
    k2, n = b.shape
    
    if k != k2:
        raise ValueError(f"Incompatible matrix shapes for multiplication: {a.shape} and {b.shape}")
    
    # Set up WebGPU compute operation
    bindings = {
        0: a_data.flatten(),
        1: b_data.flatten(),
        3: np.array([m, k], dtype=np.uint32),
        4: np.array([k, n], dtype=np.uint32)
    }
    
    # Execute compute shader
    output = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={2: (m * n, 'f')},
        shader=get_matmul_shader(),
        n=(n, m, 1)
    )
    
    # Get the result
    result_data = np.frombuffer(output[2], dtype=np.float32).reshape((m, n))
    
    # Create and return the wrapped tensor
    return wrap(result_data)

def webgpu_relu(a):
    if TORCH_DEBUG:
        print(f"WebGPU ReLU operation: {a.shape}")
    
    # Get the tensor data
    a_data = get_data(a)
    
    # Set up WebGPU compute operation
    bindings = {
        0: a_data.flatten()
    }
    
    # Execute compute shader
    output = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={1: (np.prod(a.shape), 'f')},
        shader=get_relu_shader(),
        n=(np.prod(a.shape), 1, 1)
    )
    
    # Get the result
    result_data = np.frombuffer(output[1], dtype=np.float32).reshape(a.shape)
    
    # Create and return the wrapped tensor
    return wrap(result_data)

# ===== Basic implementation of tensor creation =====

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    if TORCH_DEBUG:
        print(f"WebGPU empty: {size}")
    
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create a numpy array and wrap it
    np_array = np.empty(tuple(size), dtype=_from_torch_dtype(dtype))
    return wrap(np_array)

@torch.library.impl("aten::zeros", "privateuseone")
def zeros(size, dtype=None, layout=None, device=None, pin_memory=False):
    if TORCH_DEBUG:
        print(f"WebGPU zeros: {size}")
    
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create a numpy array and wrap it
    np_array = np.zeros(tuple(size), dtype=_from_torch_dtype(dtype))
    return wrap(np_array)

@torch.library.impl("aten::ones", "privateuseone")
def ones(size, dtype=None, layout=None, device=None, pin_memory=False):
    if TORCH_DEBUG:
        print(f"WebGPU ones: {size}")
    
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create a numpy array and wrap it
    np_array = np.ones(tuple(size), dtype=_from_torch_dtype(dtype))
    return wrap(np_array)

@torch.library.impl("aten::_to_copy", "privateuseone")
def _to_copy(tensor, **kwargs):
    device_str = str(kwargs.get('device', 'cpu'))
    if TORCH_DEBUG:
        print(f"WebGPU _to_copy: {tensor.shape} to {device_str}")
    
    data = get_data(tensor)
    dtype = kwargs.get('dtype', tensor.dtype)
    
    if device_str == 'cpu':
        # Create a CPU tensor
        return torch.tensor(data, dtype=dtype, device='cpu')
    else:
        # Create a WebGPU tensor
        np_array = data.copy().astype(_from_torch_dtype(dtype))
        return wrap(np_array)

# ===== Operation implementations =====

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(input, other, alpha=1):
    if TORCH_DEBUG:
        print(f"WebGPU add.Tensor: {input.shape} + {type(other)}")
    
    # Handle scalar addition
    if not isinstance(other, torch.Tensor):
        # Create a tensor of the same shape filled with the scalar value
        other_data = np.full(input.shape, other, dtype=_from_torch_dtype(input.dtype))
        other_tensor = wrap(other_data)
        return webgpu_add(input, other_tensor, alpha)
    
    return webgpu_add(input, other, alpha)

@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul_tensor(input, other):
    if TORCH_DEBUG:
        print(f"WebGPU mul.Tensor: {input.shape} * {type(other)}")
    
    # Handle scalar multiplication
    if not isinstance(other, torch.Tensor):
        # Create a tensor of the same shape filled with the scalar value
        other_data = np.full(input.shape, other, dtype=_from_torch_dtype(input.dtype))
        other_tensor = wrap(other_data)
        return webgpu_mul(input, other_tensor)
    
    return webgpu_mul(input, other)

@torch.library.impl("aten::mm", "privateuseone")
def mm(input, other):
    if TORCH_DEBUG:
        print(f"WebGPU mm: {input.shape} @ {other.shape}")
    return webgpu_mm(input, other)

@torch.library.impl("aten::relu", "privateuseone")
def relu(input):
    if TORCH_DEBUG:
        print(f"WebGPU relu: {input.shape}")
    return webgpu_relu(input)

# ===== In-place operations =====

def inplace_fn(outvars: str | list[str]):
    if type(outvars) is str:
        outvars = [outvars]

    def decorator(fn):
        sig = inspect.signature(fn)

        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            outs = [kwargs.get(v, bound.arguments.get(v)) for v in outvars]
            ret = fn(*args, **kwargs)
            return ret

        return wrapper

    return decorator

@torch.library.impl("aten::zero_", "privateuseone")
@inplace_fn("x")
def zero_(x):
    if TORCH_DEBUG:
        print(f"zero_ {x.shape}")
    data = unwrap(x)
    data.fill(0)
    return x

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
@inplace_fn("x")
def fill_scalar(x, y):
    if TORCH_DEBUG:
        print(f"fill_.Scalar {x.shape} {y}")
    data = unwrap(x)
    data.fill(y)
    return x

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor):
    data = unwrap(tensor)
    return data.item() if data.size > 0 else 0

# Generate methods for the privateuse1 backend
torch.utils.generate_methods_for_privateuse1_backend()