import torch
import torch.utils.cpp_extension
import pathlib
import inspect
import functools

# Register our backend with PyTorch
torch.utils.rename_privateuse1_backend("test")

# Define a minimal backend class
class TestBackend:
    def is_initialized(self):
        print("Hello world from is_initialized")
        return True

    def is_available(self):
        print("Hello world from is_available")
        return True

    def current_device(self):
        print("Hello world from current_device")
        return 0

    def _is_in_bad_fork(self):
        print("Hello world from _is_in_bad_fork")
        return False

    def manual_seed_all(self, seed):
        print(f"Hello world from manual_seed_all with seed {seed}")

    def device_count(self):
        print("Hello world from device_count")
        return 1

# Register our backend with PyTorch
torch.utils.rename_privateuse1_backend("test")
torch._register_device_module("test", TestBackend())
torch.utils.generate_methods_for_privateuse1_backend()

# Define some basic operations
@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    print(f"Hello world from empty_memory_format with size {size}")
    # Create a CPU tensor and pretend it's our device tensor
    return torch.empty(size, dtype=dtype, device="cpu")

@torch.library.impl("aten::add.Tensor", "privateuseone")
def add_tensor(input, other, alpha=1):
    print(f"Hello world from add_tensor")
    # Move to CPU, do the operation, then pretend to move back
    return torch.add(input.cpu(), other.cpu(), alpha=alpha)

@torch.library.impl("aten::mul", "privateuseone")
def mul(input, other):
    print(f"Hello world from mul")
    # Move to CPU, do the operation, then pretend to move back
    return torch.mul(input.cpu(), other.cpu())

# Add a simple test function
def test_backend():
    # Set the device
    device = torch.device("test")
    
    # Create tensors
    a = torch.ones(3, 3, device=device)
    b = torch.ones(3, 3, device=device)
    
    # Perform operations
    c = a + b
    d = a * b
    
    print("Tensor a:", a)
    print("Tensor b:", b)
    print("a + b:", c)
    print("a * b:", d)
    
    return "Test completed"

if __name__ == "__main__":
    test_backend()