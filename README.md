# PyTorch WebGPU Backend

SemiAnalysis Hackathon Mar 16 2025

Inspired by Tinygrad's Torch Backend: https://github.com/tinygrad/tinygrad/blob/master/extra/torch_backend/backend.py

## Status

### Implemented

The implemented backend supports the following operations:
* addition
* elementwise multiplication
* matrix multiplication
* ReLU activation
* argmax

### Notable Missing Features
* argmax by dimension
* softmax

## Getting Started

### Installation

Recommended to install with a virtual environment.

```shell
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$pwd
```

### Usage

We provide a simple use-case, running linear and ReLU for inference on a
MNIST model. The model itself is not trained using the WebGPU backend, but
the weights are loaded from `mnist/mnist_model.pth`.

To run the MNIST example, you can run the following command:

```shell
python mnist/run.py
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
