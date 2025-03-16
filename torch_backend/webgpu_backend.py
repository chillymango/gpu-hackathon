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

# Shader for argmax along a dimension
def get_argmax_shader():
    if "argmax" not in shader_cache:
        shader_cache["argmax"] = """
        @group(0) @binding(0)
        var<storage, read> input: array<f32>;  // Input tensor
        @group(0) @binding(1)
        var<storage, read_write> output: array<u32>;  // Output indices tensor
        @group(0) @binding(2)
        var<storage, read> dims: array<u32>;  // Dimensions of the input tensor
        @group(0) @binding(3)
        var<storage, read> params: array<u32>;  // [dim_to_reduce, keepdim]

        // Function to compute flat index from multi-dimensional coordinates
        fn compute_flat_index(coords: array<u32, 3>, dim0: u32, dim1: u32, dim2: u32, ndim: u32) -> u32 {
            var index: u32 = 0;
            // Handle up to 3D tensors
            if (ndim > 0) {
                index += coords[0];
            }
            if (ndim > 1) {
                index = index * dim1 + coords[1];
            }
            if (ndim > 2) {
                index = index * dim2 + coords[2];
            }
            return index;
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            // Get the thread ID
            let thread_id = gid.x;

            // Get the input shape and number of dimensions
            let ndim = arrayLength(&dims);

            // Get the dimension to reduce
            let dim_to_reduce = params[0];

            // Ensure we have all dimensions we need (up to 3D supported)
            let dim0 = dims[0];

            // WGSL doesn't support ternary operator, so use variable declarations
            var dim1: u32 = 1u;
            var dim2: u32 = 1u;

            if (ndim > 1) {
                dim1 = dims[1];
            }
            if (ndim > 2) {
                dim2 = dims[2];
            }

            // Calculate the output size (product of all dimensions except the reduced one)
            var output_size: u32 = 1;
            for (var i: u32 = 0; i < ndim; i++) {
                if (i != dim_to_reduce) {
                    output_size *= dims[i];
                }
            }

            // Check if this thread should process an element
            if (thread_id >= output_size) {
                return;
            }

            // Calculate the coordinates of this output element
            // We need to reverse-map from the flat output index to multi-dimensional coordinates
            var output_coords: array<u32, 3> = array<u32, 3>(0u, 0u, 0u);
            var remaining = thread_id;

            // This depends on the reduced dimension and requires different logic for each dimension
            if (dim_to_reduce == 0 && ndim >= 2) {
                // When reducing dim 0, output coords are for dims 1,2...
                output_coords[1] = remaining % dim1;
                if (ndim > 2) {
                    remaining /= dim1;
                    output_coords[2] = remaining;
                }
            } else if (dim_to_reduce == 1 && ndim >= 2) {
                // When reducing dim 1, output coords are for dims 0,2...
                output_coords[0] = remaining % dim0;
                if (ndim > 2) {
                    remaining /= dim0;
                    output_coords[2] = remaining;
                }
            } else if (dim_to_reduce == 2 && ndim >= 3) {
                // When reducing dim 2, output coords are for dims 0,1
                output_coords[0] = remaining % dim0;
                remaining /= dim0;
                output_coords[1] = remaining;
            }

            // Find the maximum value along the specified dimension
            var max_val: f32 = -3.402823e+38;  // Min float value
            var max_idx: u32 = 0;

            // Get the size of the dimension we're reducing
            let reduce_dim_size = dims[dim_to_reduce];

            // Iterate through the values along the dimension we're reducing
            for (var i: u32 = 0; i < reduce_dim_size; i++) {
                // Construct the coordinates for the current element
                var coords: array<u32, 3> = output_coords;
                coords[dim_to_reduce] = i;

                // Calculate flat index based on coordinates
                var flat_idx: u32 = 0;

                // Compute flat index based on the coordinates and dimensions
                if (ndim == 1) {
                    flat_idx = coords[0];
                } else if (ndim == 2) {
                    flat_idx = coords[0] * dim1 + coords[1];
                } else if (ndim == 3) {
                    flat_idx = (coords[0] * dim1 + coords[1]) * dim2 + coords[2];
                }

                // Get the value at this position
                let val = input[flat_idx];

                // Update max value if this is larger
                if (i == 0 || val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Store the result
            output[thread_id] = max_idx;
        }
        """
    return shader_cache["argmax"]

# Shader for global argmax (no dimension specified)
def get_global_argmax_shader():
    if "global_argmax" not in shader_cache:
        shader_cache["global_argmax"] = """
        @group(0) @binding(0)
        var<storage, read> input: array<f32>; // Input tensor
        @group(0) @binding(1)
        var<storage, read_write> output: array<u32>; // Output index (single value)

        // We'll use a single workgroup reduction
        var<workgroup> shared_data: array<f32, 256>;
        var<workgroup> shared_indices: array<u32, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>) {
            let global_id = gid.x;
            let local_id = lid.x;
            let input_length = arrayLength(&input);

            // Load data into shared memory
            if (global_id < input_length) {
                shared_data[local_id] = input[global_id];
                shared_indices[local_id] = global_id;
            } else {
                shared_data[local_id] = -3.402823e+38; // Min float value
                shared_indices[local_id] = 0u;
            }

            // Wait for all threads to load data
            workgroupBarrier();

            // Perform parallel reduction to find max value and its index
            for (var stride: u32 = 256u / 2u; stride > 0u; stride = stride / 2u) {
                if (local_id < stride) {
                    let other_idx = local_id + stride;
                    if (other_idx < 256u && global_id + stride < input_length) {
                        if (shared_data[other_idx] > shared_data[local_id]) {
                            shared_data[local_id] = shared_data[other_idx];
                            shared_indices[local_id] = shared_indices[other_idx];
                        }
                    }
                }
                // Wait for all threads to complete this round
                workgroupBarrier();
            }

            // The thread with local_id = 0 writes the final result
            if (local_id == 0u && wid.x == 0u) {
                output[0] = shared_indices[0];
            }
        }
        """
    return shader_cache["global_argmax"]

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
    if len(a.shape) == 1:
        m = 1
        k = a.shape[0]
    else:
        m, k = a.shape
    if len(b.shape) == 1:
        k2 = b.shape[0]
        n = 1
    else:
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

def webgpu_argmax(tensor, dim=None, keepdim=False):
    """
    Returns the indices of the maximum values of a tensor across a dimension.

    Args:
        tensor (torch.Tensor): The input tensor
        dim (int, optional): The dimension to reduce. If None, the argmax of the
            flattened input is returned.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.

    Returns:
        torch.Tensor: A tensor containing the indices of the maximum values.
    """
    if TORCH_DEBUG:
        print(f"WebGPU argmax operation: {tensor.shape}, dim={dim}, keepdim={keepdim}")

    # Get the tensor data
    tensor_data = get_data(tensor)

    # Handle the case of flattened argmax (no dimension specified)
    if dim is None:
        # We'll use a simpler approach for the global argmax
        flat_data = tensor_data.flatten()

        # Set up WebGPU compute operation
        bindings = {
            0: flat_data
        }

        # Execute global argmax shader
        output = compute_with_buffers(
            input_arrays=bindings,
            output_arrays={1: (1, 'I')},  # Single output integer index
            shader=get_global_argmax_shader(),
            n=(((len(flat_data) + 255) // 256) * 256, 1, 1)  # Ensure we have enough threads
        )

        # Get the result as an integer
        result_idx = np.frombuffer(output[1], dtype=np.uint32)[0]

        # Create and return a scalar tensor with the result
        return wrap(np.array([result_idx], dtype=np.int64))

    # Handle the case of argmax along a specific dimension
    else:
        # For dimensional argmax, use NumPy's argmax directly
        # This is a temporary solution until we fully optimize the WebGPU version

        # Ensure dim is valid and positive (normalize negative dims)
        ndim = tensor_data.ndim
        if dim < 0:
            dim = ndim + dim

        if dim >= ndim:
            raise ValueError(f"Dimension {dim} out of range for tensor with {ndim} dimensions")

        # Use NumPy's argmax directly
        result_data = np.argmax(tensor_data, axis=dim)

        # Handle keepdim if necessary
        if keepdim:
            # Insert a dimension of size 1 at the specified position
            result_data = np.expand_dims(result_data, dim)

        # Convert to int64 which is PyTorch's default index type
        result_data = result_data.astype(np.int64)

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

# Shader for strided tensor access
def get_strided_copy_shader():
    if "strided_copy" not in shader_cache:
        shader_cache["strided_copy"] = """
        @group(0) @binding(0)
        var<storage, read> input: array<f32>;
        @group(0) @binding(1)
        var<storage, read_write> output: array<f32>;
        @group(0) @binding(2)
        var<storage, read> dims: array<u32>;
        @group(0) @binding(3)
        var<storage, read> strides: array<u32>;
        @group(0) @binding(4)
        var<storage, read> params: array<u32>;  // [storage_offset]

        @compute @workgroup_size(8, 8, 8)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let ndim = arrayLength(&dims);

            // Check dimensions bounds
            if (ndim >= 1 && gid.x >= dims[0]) { return; }
            if (ndim >= 2 && gid.y >= dims[1]) { return; }
            if (ndim >= 3 && gid.z >= dims[2]) { return; }

            // Calculate source index using strides
            var src_idx: u32 = params[0];  // Start with storage_offset

            if (ndim >= 1 && gid.x < dims[0]) {
                src_idx = src_idx + gid.x * strides[0];
            }

            if (ndim >= 2 && gid.y < dims[1]) {
                src_idx = src_idx + gid.y * strides[1];
            }

            if (ndim >= 3 && gid.z < dims[2]) {
                src_idx = src_idx + gid.z * strides[2];
            }

            // Calculate destination index (linear layout in output tensor)
            var dst_idx: u32 = 0;
            if (ndim >= 1) { dst_idx = gid.x; }
            if (ndim >= 2) { dst_idx = dst_idx * dims[1] + gid.y; }
            if (ndim >= 3) { dst_idx = dst_idx * dims[2] + gid.z; }

            // Copy the data
            output[dst_idx] = input[src_idx];
        }
        """
    return shader_cache["strided_copy"]

def webgpu_strided_copy(tensor, size, stride, storage_offset=0):
    """
    Implementation of as_strided using WebGPU compute shader.

    Creates a new tensor from the input tensor using the specified strides.
    """
    # Get the tensor data
    tensor_data = get_data(tensor)

    # Ensure the size and stride are tuples
    size = tuple(size)
    stride = tuple(stride)

    # Calculate the number of dimensions
    ndim = len(size)

    # Pad the size and stride to 3D (WebGPU dispatch requires 3D)
    padded_size = size + (1,) * (3 - ndim)
    padded_stride = stride + (1,) * (3 - ndim)

    # Set up WebGPU compute operation
    bindings = {
        0: tensor_data.flatten(),
        2: np.array(size, dtype=np.uint32),
        3: np.array(stride, dtype=np.uint32),
        4: np.array([storage_offset], dtype=np.uint32)
    }

    # Calculate output size
    output_size = np.prod(size)

    # Execute compute shader
    output = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={1: (output_size, 'f')},
        shader=get_strided_copy_shader(),
        n=padded_size
    )

    # Get the result
    result_data = np.frombuffer(output[1], dtype=np.float32).reshape(size)

    # Create and return the wrapped tensor
    return wrap(result_data)

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor: torch.Tensor, size, stride, storage_offset=None):
    """
    Create a view of an existing torch.Tensor input with specified size, stride and storage_offset.

    Args:
        tensor: The input tensor
        size: The size (shape) of the output tensor
        stride: The stride of the output tensor
        storage_offset: The offset in the underlying storage of the output tensor

    Returns:
        A tensor with the strided data
    """
    if TORCH_DEBUG:
        print(f"WebGPU as_strided: {tensor.shape} -> {size}, stride={stride}, offset={storage_offset}")

    # Default storage offset is 0
    if storage_offset is None:
        storage_offset = 0

    # Execute strided copy operation
    return webgpu_strided_copy(tensor, size, stride, storage_offset)

@torch.library.impl("aten::view", "privateuseone")
def view(tensor, size):
    """
    Returns a new tensor with the same data but different shape.

    Args:
        tensor: The input tensor
        size: The new shape

    Returns:
        A tensor with the new shape
    """
    if TORCH_DEBUG:
        print(f"WebGPU view: {tensor.shape} -> {size}")

    # Get the original data
    data = get_data(tensor)

    # Ensure size is a tuple
    size = tuple(size)

    # Reshape the data
    reshaped_data = data.reshape(size)

    # Create and return the reshaped tensor
    return wrap(reshaped_data)

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

@torch.library.impl("aten::matmul", "privateuseone")
def matmul(input, other):
    """
    Matrix multiplication that handles broadcasting and different dimensions.
    This implements the @ operator behavior.
    """
    if TORCH_DEBUG:
        print(f"WebGPU matmul: {input.shape} @ {other.shape}")

    # Handle the case where input is a vector (1D tensor)
    if input.dim() == 1 and other.dim() == 2:
        # For vector @ matrix, reshape to (1, n) first, then mm
        input_reshaped = input.reshape(1, -1)
        result = webgpu_mm(input_reshaped, other)
        # Return result as a 1D tensor
        return result.reshape(-1)

    # Handle the case where other is a vector (1D tensor)
    elif input.dim() == 2 and other.dim() == 1:
        # For matrix @ vector, reshape vector to (n, 1), then mm
        other_reshaped = other.reshape(-1, 1)
        result = webgpu_mm(input, other_reshaped)
        # Return result as a 1D tensor
        return result.reshape(-1)

    # Standard matrix multiplication for 2D tensors
    elif input.dim() == 2 and other.dim() == 2:
        return webgpu_mm(input, other)

    # For higher dimensions, we'd need to implement broadcasting
    # For now, just handle the vector/matrix cases we need for the test
    else:
        raise NotImplementedError(f"WebGPU matmul not implemented for {input.dim()}D @ {other.dim()}D")

@torch.library.impl("aten::relu", "privateuseone")
def relu(input):
    if TORCH_DEBUG:
        print(f"WebGPU relu: {input.shape}")
    return webgpu_relu(input)

@torch.library.impl("aten::argmax", "privateuseone")
def argmax(input, dim=None, keepdim=False):
    """Implementation for torch.argmax"""
    if TORCH_DEBUG:
        print(f"WebGPU argmax: {input.shape}, dim={dim}, keepdim={keepdim}")
    return webgpu_argmax(input, dim, keepdim)

@torch.library.impl("aten::argmax.dim", "privateuseone")
def argmax_dim(input, dim, keepdim=False):
    """Implementation for torch.argmax with dimension"""
    if TORCH_DEBUG:
        print(f"WebGPU argmax.dim: {input.shape}, dim={dim}, keepdim={keepdim}")
    return webgpu_argmax(input, dim, keepdim)

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
