# Installation Guide (WSL2 / Linux)

Tested on WSL2 Ubuntu, Python 3.10, NVIDIA GPU (JAX on GPU, PyTorch on CPU).

## 1. Create environment

    conda create -n hyperlora python=3.10 -y
    conda activate hyperlora
    cd HyperLoRA

## 2. Install dependencies

**Order matters.** NumPy 1.x must go first, and PyTorch must be the
CPU build. Installing GPU PyTorch alongside `jax[cuda12]` causes a
cudnn version conflict (torch <2.2 pins cudnn 8.x, current JAX
requires cudnn 9.x). In this codebase PyTorch only runs the VMAS
physics, so the CPU build is sufficient; JAX handles all network
training on the GPU.

    # 1. NumPy 1.x FIRST (NumPy 2.x breaks torch/vmas compatibility)
    pip install "numpy>=1.23.0,<2.0.0"

    # 2. PyTorch CPU-only (avoids cudnn conflict with JAX)
    pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cpu

    # 3. JAX with CUDA (GPU) — or `pip install -U jax` for CPU-only machines
    pip install -U "jax[cuda12]"

    # 4. Remaining dependencies
    pip install flax optax distrax pyyaml gymnasium "vmas>=1.3.0" wandb moviepy

If any later install pulls in NumPy 2.x, re-pin it:
`pip install "numpy<2" --force-reinstall`

## 3. Verify install

    python -c "import numpy; print('numpy', numpy.__version__)"        # expect 1.26.x
    python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
    python -c "import jax; print('jax', jax.__version__, jax.devices())"
    python -c "import vmas, flax, distrax; print('vmas/flax/distrax ok')"

Expected:
- numpy 1.26.x (NOT 2.x)
- torch 2.1.2+cpu, cuda: False  ← intentional, VMAS runs on CPU
- jax showing [CudaDevice(id=0)] (or [CpuDevice(id=0)] on CPU-only installs)

## 4. Smoke test
``` python train.py --num-envs 8 --num-episodes 5 --no-logging ``` 

