# PyTorch WebGPU Backend Prototype

Demonstration of adding a WebGPU backend to PyTorch using [wgpu-py](https://github.com/pygfx/wgpu-py), allowing models to run on any GPU where [WebGPU](https://developer.chrome.com/blog/webgpu-io2023) is supported (Macbooks, Raspberry Pis, GH200s, refrigerators, etc). 

Built (mostly vibe-coded) at [SemiAnalysis Hackathon 2025](https://semianalysis.com/hackathon-2025/).

The backend integration borrows heavily from Tinygrad's [PyTorch backend](https://github.com/tinygrad/tinygrad/blob/master/extra/torch_backend/backend.py).

## Status

### Implemented

The implemented backend supports the following operations for fp32:

* addition
* elementwise multiplication
* matrix multiplication
* ReLU activation
* argmax

### Notable Missing Features

* argmax by dimension
* softmax
* every other operation...

## Getting Started

### Installation

There's no python package available right now. Install requirements:

```shell
pip install -r requirements.txt
```

### Usage

```py
# Import the WebGPU backend to register it with PyTorch
import torch
import webgpu_backend

device = torch.device("webgpu")
a = torch.rand((10, 10, 10), dtype=torch.float32)
b = torch.rand((10, 10, 10), dtype=torch.float32)

a_gpu = a.to(device)
b_gpu = b.to(device)

print(a_gpu.device)  # webgpu:0
c_gpu = a_gpu + b_gpu
c = c_gpu.to("cpu")

print(c.shape) # torch.Size([10, 10, 10])
```

We provide a simple use-case, running linear and ReLU for inference on a
MNIST model. The model itself is not trained using the WebGPU backend, but
the weights are loaded from `mnist/mnist_model.pth`.

To run the MNIST example, you can run the following command:

```shell
python -m mnist.run
```

### Testing

The following command will run the test suite:

```shell
cd torch_backend
pytest test_webgpu.py
```

The tests confirm that `torch` operations create buffers on the WebGPU device,
and invoke our shaders to compute the desired matrix operations. Additionally,
the tests confirm that the device guard works.
