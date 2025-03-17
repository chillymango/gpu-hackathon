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

# Get the WebGPU device
device = wgpu.utils.get_default_device()

class WebGPUData:
    def __init__(self):
        self.data_dict = {}
        self.shape_dict = {}  # Store tensor shapes

    def store(self, tensor, data=None, buffer=None):
        """
        Store data associated with a tensor.
        
        Args:
            tensor: The PyTorch tensor
            data: The NumPy array data (optional if buffer is provided)
            buffer: Optional WebGPU buffer (if already created)
        """
        self.data_dict[id(tensor)] = {
            'data': data,
            'buffer': buffer
        }
        # Store the shape for later use
        self.shape_dict[id(tensor)] = tensor.shape
        
    def drop_host_data(self, tensor):
        """Drop the host memory copy after data has been transferred to GPU"""
        entry = self.data_dict.get(id(tensor))
        if entry and entry['buffer'] is not None:
            # Only drop data if we have a buffer
            entry['data'] = None

    def get(self, tensor):
        """Get the NumPy data associated with a tensor"""
        entry = self.data_dict.get(id(tensor))
        if entry and entry['data'] is not None:
            return entry['data']
        
        # If we have a buffer but no data, read from the buffer
        if entry and entry['buffer'] is not None:
            buffer = entry['buffer']
            # Create a staging buffer for reading
            staging_buffer = device.create_buffer(
                size=buffer.size,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
            )
            
            # Copy buffer to staging buffer
            command_encoder = device.create_command_encoder()
            command_encoder.copy_buffer_to_buffer(
                buffer, 0, staging_buffer, 0, buffer.size
            )
            device.queue.submit([command_encoder.finish()])
            
            # Read data from staging buffer
            staging_buffer.map_sync(wgpu.MapMode.READ)
            flat_data = np.frombuffer(staging_buffer.read_mapped(0, buffer.size), dtype=np.float32)
            staging_buffer.unmap()
            
            # Reshape the data to match the original tensor shape
            shape = self.shape_dict.get(id(tensor))
            if shape:
                data = flat_data.reshape(shape)
            else:
                data = flat_data  # Fallback to flat data if shape is unknown
            
            # Store the data
            entry['data'] = data
            return data
        
        return None

    def get_buffer(self, tensor):
        """Get the WebGPU buffer associated with a tensor"""
        entry = self.data_dict.get(id(tensor))
        if entry and 'buffer' in entry:
            return entry['buffer']
        return None

    def update_buffer(self, tensor, buffer):
        """Update the WebGPU buffer for a tensor"""
        if id(tensor) in self.data_dict:
            self.data_dict[id(tensor)]['buffer'] = buffer
        else:
            self.store(tensor, None, buffer)
        # Make sure we store the shape
        self.shape_dict[id(tensor)] = tensor.shape

    def remove(self, tensor):
        if id(tensor) in self.data_dict:
            del self.data_dict[id(tensor)]
        if id(tensor) in self.shape_dict:
            del self.shape_dict[id(tensor)]

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
def wrap(x, buffer=None) -> torch.Tensor:
    """
    Wrap a numpy array into a WebGPU tensor by first creating a CPU tensor
    
    Args:
        x: NumPy array to wrap
        buffer: Optional WebGPU buffer to associate with the tensor
    """
    # Use mod.wrap from the C++ extension, similar to what tinygrad does
    result = mod.wrap(x, _to_torch_dtype(x.dtype), 0)
    
    # If no buffer is provided, create one
    if buffer is None:
        buffer = device.create_buffer_with_data(
            data=x,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
    
    # Store the tensor data and buffer
    gpu_data.store(result, x, buffer)
    
    # Drop the host memory copy since we've copied to GPU
    gpu_data.drop_host_data(result)
    
    return result

def unwrap(x: torch.Tensor):
    """Unwrap a WebGPU tensor to get the underlying numpy array"""
    assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
    return gpu_data.get(x)

def get_or_create_buffer(tensor, data=None):
    """
    Get the WebGPU buffer for a tensor or create one if it doesn't exist
    
    Args:
        tensor: The PyTorch tensor
        data: Optional NumPy data to use if buffer needs to be created
    
    Returns:
        The WebGPU buffer
    """
    buffer = gpu_data.get_buffer(tensor)
    if buffer is not None:
        return buffer
    
    # If no buffer exists, create one
    if data is None:
        data = get_data(tensor)
    
    buffer = device.create_buffer_with_data(
        data=data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Store the buffer
    gpu_data.update_buffer(tensor, buffer)
    
    # Drop the host memory copy since we've copied to GPU
    gpu_data.drop_host_data(tensor)
    
    return buffer

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

    # Try to get data from WebGPUData
    data = gpu_data.get(tensor)
    
    # If we have data, return it
    if data is not None:
        return data

    # If we don't have data but have a buffer, this will trigger reading from the buffer
    buffer = gpu_data.get_buffer(tensor)
    if buffer is not None:
        return gpu_data.get(tensor)  # This will now read from the buffer

    # Fallback to CPU
    return tensor.cpu().detach().numpy()

def execute_webgpu_shader(shader_code, bindings, output_shape, workgroup_size=(256, 1, 1), dispatch_size=None):
    """
    Execute a WebGPU compute shader with the given bindings and return the result.
    
    Args:
        shader_code: The WGSL shader code to execute
        bindings: A list of (buffer, binding_type) tuples, where binding_type is one of
                 'read_only_storage' or 'storage'
        output_shape: The shape of the output tensor
        workgroup_size: The workgroup size to use (default: (256, 1, 1))
        dispatch_size: The number of workgroups to dispatch (default: calculated based on output size)
    
    Returns:
        The output buffer containing the result (data remains on GPU)
    """
    # Create bind group layout entries
    bind_group_layout_entries = []
    for i, (_, binding_type) in enumerate(bindings):
        bind_group_layout_entries.append({
            "binding": i,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": getattr(wgpu.BufferBindingType, binding_type)}
        })
    
    # Create bind group layout
    bind_group_layout = device.create_bind_group_layout(entries=bind_group_layout_entries)
    
    # Create pipeline layout
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    
    # Create compute pipeline
    shader_module = device.create_shader_module(code=shader_code)
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"}
    )
    
    # Create bind group entries
    bind_group_entries = []
    for i, (buffer, _) in enumerate(bindings):
        bind_group_entries.append({
            "binding": i,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}
        })
    
    # Create bind group
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=bind_group_entries
    )
    
    # Create command encoder
    command_encoder = device.create_command_encoder()
    
    # Create compute pass
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    
    # Calculate dispatch size if not provided
    if dispatch_size is None:
        output_size = np.prod(output_shape)
        x_groups = (output_size + workgroup_size[0] - 1) // workgroup_size[0]
        dispatch_size = (x_groups, 1, 1)
    
    # Dispatch workgroups
    compute_pass.dispatch_workgroups(*dispatch_size)
    compute_pass.end()
    
    # Find the output buffer (assumed to be the last one with binding_type 'storage')
    output_buffer = None
    for buffer, binding_type in reversed(bindings):
        if binding_type == 'storage':
            output_buffer = buffer
            break
    
    if output_buffer is None:
        raise ValueError("No output buffer found in bindings")
    
    # Submit commands
    device.queue.submit([command_encoder.finish()])
    
    # Return the output buffer (data remains on GPU)
    return output_buffer

# ===== WebGPU Operation Implementations =====

def webgpu_add(a, b, alpha=1):
    if TORCH_DEBUG:
        print(f"WebGPU add operation: {a.shape} + {b.shape}")

    # Get or create WebGPU buffers
    a_buffer = get_or_create_buffer(a)
    b_buffer = get_or_create_buffer(b)
    
    # Create alpha buffer
    alpha_data = np.array([alpha], dtype=np.float32)
    alpha_buffer = device.create_buffer_with_data(
        data=alpha_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    # Create output buffer
    output_size = np.prod(a.shape)
    output_buffer = device.create_buffer(
        size=output_size * 4,  # 4 bytes per float32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Set up bindings
    bindings = [
        (a_buffer, 'read_only_storage'),
        (b_buffer, 'read_only_storage'),
        (output_buffer, 'storage'),
        (alpha_buffer, 'read_only_storage')
    ]
    
    # Execute shader
    output_buffer = execute_webgpu_shader(
        shader_code=get_add_shader(),
        bindings=bindings,
        output_shape=a.shape
    )
    
    # Create a new tensor that references the output buffer
    result_tensor = mod.wrap(np.empty(a.shape, dtype=np.float32), _to_torch_dtype(np.float32), 0)
    gpu_data.store(result_tensor, None, output_buffer)
    
    return result_tensor

def webgpu_mul(a, b):
    if TORCH_DEBUG:
        print(f"WebGPU multiply operation: {a.shape} * {b.shape}")

    # Get or create WebGPU buffers
    a_buffer = get_or_create_buffer(a)
    b_buffer = get_or_create_buffer(b)
    
    # Create output buffer
    output_size = np.prod(a.shape)
    output_buffer = device.create_buffer(
        size=output_size * 4,  # 4 bytes per float32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Set up bindings
    bindings = [
        (a_buffer, 'read_only_storage'),
        (b_buffer, 'read_only_storage'),
        (output_buffer, 'storage')
    ]
    
    # Execute shader
    output_buffer = execute_webgpu_shader(
        shader_code=get_mul_shader(),
        bindings=bindings,
        output_shape=a.shape
    )
    
    # Create a new tensor that references the output buffer
    result_tensor = mod.wrap(np.empty(a.shape, dtype=np.float32), _to_torch_dtype(np.float32), 0)
    gpu_data.store(result_tensor, None, output_buffer)
    
    return result_tensor

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

    # Handle the case of flattened argmax (no dimension specified)
    if dim is None:
        # Get or create WebGPU buffer
        input_buffer = get_or_create_buffer(tensor)
        
        # Create output buffer
        output_buffer = device.create_buffer(
            size=4,  # Single uint32 value
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        
        # Set up bindings
        bindings = [
            (input_buffer, 'read_only_storage'),
            (output_buffer, 'storage')
        ]
        
        # Execute shader
        output_buffer = execute_webgpu_shader(
            shader_code=get_global_argmax_shader(),
            bindings=bindings,
            output_shape=(1,),
            workgroup_size=(256, 1, 1),
            dispatch_size=((np.prod(tensor.shape) + 255) // 256, 1, 1)
        )
        
        # Create a staging buffer to read the result
        staging_buffer = device.create_buffer(
            size=4,  # Single uint32 value
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
        )
        
        # Copy output to staging buffer
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            output_buffer, 0, staging_buffer, 0, 4
        )
        device.queue.submit([command_encoder.finish()])
        
        # Read the result
        staging_buffer.map_read()
        result_idx = np.frombuffer(staging_buffer.read_mapped_range(0, 4), dtype=np.uint32)[0]
        staging_buffer.unmap()
        
        # Create and return a scalar tensor with the result
        result_data = np.array([result_idx], dtype=np.int64)
        result_tensor = mod.wrap(result_data, _to_torch_dtype(result_data.dtype), 0)
        
        # Store the data but don't keep a host copy since it's small and we already have the value
        gpu_data.store(result_tensor, result_data)
        
        return result_tensor

    # Handle the case of argmax along a specific dimension
    else:
        # For dimensional argmax, use NumPy's argmax directly
        # This is a temporary solution until we fully optimize the WebGPU version
        tensor_data = get_data(tensor)

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
        result_tensor = mod.wrap(result_data, _to_torch_dtype(result_data.dtype), 0)
        
        # Create a buffer for the result
        buffer = device.create_buffer_with_data(
            data=result_data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        
        # Store the buffer and data
        gpu_data.store(result_tensor, result_data, buffer)
        
        # Drop the host memory copy
        gpu_data.drop_host_data(result_tensor)
        
        return result_tensor

# ===== Basic implementation of tensor creation =====

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    if TORCH_DEBUG:
        print(f"WebGPU empty: {size}")

    if dtype is None:
        dtype = torch.get_default_dtype()

    # Create a numpy array
    np_array = np.empty(tuple(size), dtype=_from_torch_dtype(dtype))
    
    # Create a tensor
    result_tensor = mod.wrap(np_array, dtype, 0)
    
    # Create a WebGPU buffer
    buffer = device.create_buffer_with_data(
        data=np_array,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Store the buffer
    gpu_data.store(result_tensor, np_array, buffer)
    
    # Drop the host memory copy
    gpu_data.drop_host_data(result_tensor)
    
    return result_tensor

@torch.library.impl("aten::zeros", "privateuseone")
def zeros(size, dtype=None, layout=None, device=None, pin_memory=False):
    if TORCH_DEBUG:
        print(f"WebGPU zeros: {size}")

    if dtype is None:
        dtype = torch.get_default_dtype()

    # Create a numpy array
    np_array = np.zeros(tuple(size), dtype=_from_torch_dtype(dtype))
    
    # Create a tensor
    result_tensor = mod.wrap(np_array, dtype, 0)
    
    # Create a WebGPU buffer
    buffer = device.create_buffer_with_data(
        data=np_array,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Store the buffer
    gpu_data.store(result_tensor, np_array, buffer)
    
    # Drop the host memory copy
    gpu_data.drop_host_data(result_tensor)
    
    return result_tensor

@torch.library.impl("aten::ones", "privateuseone")
def ones(size, dtype=None, layout=None, device=None, pin_memory=False):
    if TORCH_DEBUG:
        print(f"WebGPU ones: {size}")

    if dtype is None:
        dtype = torch.get_default_dtype()

    # Create a numpy array
    np_array = np.ones(tuple(size), dtype=_from_torch_dtype(dtype))
    
    # Create a tensor
    result_tensor = mod.wrap(np_array, dtype, 0)
    
    # Create a WebGPU buffer
    buffer = device.create_buffer_with_data(
        data=np_array,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Store the buffer
    gpu_data.store(result_tensor, np_array, buffer)
    
    # Drop the host memory copy
    gpu_data.drop_host_data(result_tensor)
    
    return result_tensor

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
        result_tensor = mod.wrap(np_array, dtype, 0)
        
        # Create a WebGPU buffer
        buffer = device.create_buffer_with_data(
            data=np_array,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        
        # Store the buffer
        gpu_data.store(result_tensor, np_array, buffer)
        
        # Drop the host memory copy
        gpu_data.drop_host_data(result_tensor)
        
        return result_tensor

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
    # Get or create WebGPU buffer for input tensor
    input_buffer = get_or_create_buffer(tensor)
    
    # Create output buffer
    output_size = np.prod(size)
    output_buffer = device.create_buffer(
        size=output_size * 4,  # 4 bytes per float32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Create dims buffer
    dims_data = np.array(size, dtype=np.uint32)
    dims_buffer = device.create_buffer_with_data(
        data=dims_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    # Create strides buffer
    strides_data = np.array(stride, dtype=np.uint32)
    strides_buffer = device.create_buffer_with_data(
        data=strides_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    # Create params buffer
    params_data = np.array([storage_offset], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    # Set up bindings
    bindings = [
        (input_buffer, 'read_only_storage'),
        (output_buffer, 'storage'),
        (dims_buffer, 'read_only_storage'),
        (strides_buffer, 'read_only_storage'),
        (params_buffer, 'read_only_storage')
    ]
    
    # Calculate padded size for dispatch
    ndim = len(size)
    padded_size = size + (1,) * (3 - ndim)
    
    # Execute shader
    output_buffer = execute_webgpu_shader(
        shader_code=get_strided_copy_shader(),
        bindings=bindings,
        output_shape=size,
        workgroup_size=(8, 8, 8),
        dispatch_size=padded_size
    )
    
    # Create a new tensor that references the output buffer
    result_tensor = mod.wrap(np.empty(size, dtype=np.float32), _to_torch_dtype(np.float32), 0)
    gpu_data.store(result_tensor, None, output_buffer)
    
    return result_tensor

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

    # Ensure size is a tuple
    size = tuple(size)
    
    # Get the buffer from the original tensor
    buffer = gpu_data.get_buffer(tensor)
    
    # Create a new tensor with the new shape
    result_tensor = mod.wrap(np.empty(size, dtype=_from_torch_dtype(tensor.dtype)), tensor.dtype, 0)
    
    # Store the same buffer with the new tensor, but no host data
    gpu_data.store(result_tensor, None, buffer)
    
    return result_tensor

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
    
    # Get the buffer
    buffer = gpu_data.get_buffer(x)
    
    # Create a command encoder
    command_encoder = device.create_command_encoder()
    
    # Fill the buffer with zeros
    device.queue.write_buffer(
        buffer, 0, np.zeros(np.prod(x.shape), dtype=np.float32)
    )
    
    # Submit the command
    device.queue.submit([command_encoder.finish()])
    
    # Update the data in the WebGPUData store
    data = unwrap(x)
    if data is not None:
        data.fill(0)
    
    return x

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
@inplace_fn("x")
def fill_scalar(x, y):
    if TORCH_DEBUG:
        print(f"fill_.Scalar {x.shape} {y}")
    
    # Get the buffer
    buffer = gpu_data.get_buffer(x)
    
    # Create a command encoder
    command_encoder = device.create_command_encoder()
    
    # Fill the buffer with the scalar value
    device.queue.write_buffer(
        buffer, 0, np.full(np.prod(x.shape), y, dtype=np.float32)
    )
    
    # Submit the command
    device.queue.submit([command_encoder.finish()])
    
    # Update the data in the WebGPUData store
    data = unwrap(x)
    if data is not None:
        data.fill(y)
    
    return x

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor):
    data = unwrap(tensor)
    return data.item() if data.size > 0 else 0

# Generate methods for the privateuse1 backend
torch.utils.generate_methods_for_privateuse1_backend()

def webgpu_mm(a, b):
    if TORCH_DEBUG:
        print(f"WebGPU matrix multiply operation: {a.shape} @ {b.shape}")
    
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
    
    # Get or create WebGPU buffers
    a_buffer = get_or_create_buffer(a)
    b_buffer = get_or_create_buffer(b)
    
    # Create shape buffers
    a_shape_data = np.array([m, k], dtype=np.uint32)
    b_shape_data = np.array([k, n], dtype=np.uint32)
    
    a_shape_buffer = device.create_buffer_with_data(
        data=a_shape_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    b_shape_buffer = device.create_buffer_with_data(
        data=b_shape_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    
    # Create output buffer
    output_size = m * n
    output_buffer = device.create_buffer(
        size=output_size * 4,  # 4 bytes per float32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Set up bindings
    bindings = [
        (a_buffer, 'read_only_storage'),
        (b_buffer, 'read_only_storage'),
        (output_buffer, 'storage'),
        (a_shape_buffer, 'read_only_storage'),
        (b_shape_buffer, 'read_only_storage')
    ]
    
    # Execute shader with custom workgroup size and dispatch size for matrix multiplication
    workgroup_size = (8, 8, 1)  # 8x8 workgroup size as defined in the shader
    dispatch_size = (
        (n + workgroup_size[0] - 1) // workgroup_size[0],
        (m + workgroup_size[1] - 1) // workgroup_size[1],
        1
    )
    
    output_buffer = execute_webgpu_shader(
        shader_code=get_matmul_shader(),
        bindings=bindings,
        output_shape=(m, n),
        workgroup_size=workgroup_size,
        dispatch_size=dispatch_size
    )
    
    # Create a new tensor that references the output buffer
    result_tensor = mod.wrap(np.empty((m, n), dtype=np.float32), _to_torch_dtype(np.float32), 0)
    gpu_data.store(result_tensor, None, output_buffer)
    
    return result_tensor

def webgpu_relu(a):
    if TORCH_DEBUG:
        print(f"WebGPU ReLU operation: {a.shape}")

    # Get or create WebGPU buffers
    a_buffer = get_or_create_buffer(a)
    
    # Create output buffer
    output_size = np.prod(a.shape)
    output_buffer = device.create_buffer(
        size=output_size * 4,  # 4 bytes per float32
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    
    # Set up bindings
    bindings = [
        (a_buffer, 'read_only_storage'),
        (output_buffer, 'storage')
    ]
    
    # Execute shader
    output_buffer = execute_webgpu_shader(
        shader_code=get_relu_shader(),
        bindings=bindings,
        output_shape=a.shape
    )
    
    # Create a new tensor that references the output buffer
    result_tensor = mod.wrap(np.empty(a.shape, dtype=np.float32), _to_torch_dtype(np.float32), 0)
    gpu_data.store(result_tensor, None, output_buffer)
    
    return result_tensor
