# UDLM Environment Setup on CSCS Clariden (GH200 / aarch64)

This README documents a working installation procedure for the **UDLM** environment on **CSCS Clariden** using an **NVIDIA GH200 120GB** GPU on **aarch64**.

The procedure incorporates the issues encountered during installation and the fixes that worked successfully.

---

## 1. Target Environment

Recommended stack:

```text
OS / platform      Linux aarch64
GPU                NVIDIA GH200 120GB
Compute capability 9.0 (Hopper)

Python             3.11
CUDA toolkit       12.9
PyTorch            2.8.0+cu129
Torchvision        0.23.0
Triton             3.4.0

causal-conv1d      1.6.2.post1
mamba-ssm          1.2.0.post1
flash-attn         2.8.3.post1
```

Important changes compared with the original environment:

- Use **Python 3.11** instead of Python 3.9.
- Use CUDA-enabled PyTorch, not `torch+cpu`.
- Install a complete CUDA development stack so headers such as `cusparse.h`, `cublas_v2.h`, and `cusolverDn.h` are available.
- Keep system Python 3.12 Torch libraries out of `LD_LIBRARY_PATH`.
- Build `mamba-ssm==1.2.0.post1` from the GitHub source tag.
- Install FlashAttention from a prebuilt `aarch64 + cu129 + torch2.8` wheel instead of compiling 73 CUDA translation units from source.

---

# 2. Start from a Clean Environment

Deactivate the old environment:

```bash
conda deactivate
```

Remove the old environment if needed:

```bash
conda env remove -n udlm -y
```

Clean package caches:

```bash
conda clean --all -y
python -m pip cache purge 2>/dev/null || true
```

---

# 3. Use Scratch for Temporary Downloads and Builds

Clariden home quota can become a problem because PyTorch and CUDA packages are large.

```bash
mkdir -p "$SCRATCH/conda-pkgs"
mkdir -p "$SCRATCH/pip-tmp"
mkdir -p "$SCRATCH/udlm-build"

export CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs"
export TMPDIR="$SCRATCH/pip-tmp"
```

Optional space checks:

```bash
df -h "$HOME" "$SCRATCH"
quota -s 2>/dev/null || true
```

---

# 4. Create the Conda Environment

Create:

```bash
nano ~/NLU/environment-gh200.yml
```

Use:

```yaml
name: udlm

channels:
  - nvidia/label/cuda-12.9.2
  - defaults

dependencies:
  - python=3.11
  - pip
  - setuptools
  - wheel

  - cuda-nvcc=12.9.86
  - cuda-libraries-dev=12.9.2

  - ipykernel=6.29.5
  - ipython
  - ipywidgets=8.1.2
```

Create the environment:

```bash
export CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs"
export TMPDIR="$SCRATCH/pip-tmp"

conda env create -f ~/NLU/environment-gh200.yml
```

Activate:

```bash
conda activate udlm
```

Check:

```bash
python --version
uname -m
nvcc --version
```

Expected:

```text
Python 3.11.x
aarch64
CUDA 12.9
```

---

# 5. Protect the Conda Environment from System Torch Libraries

On Clariden, the system environment may contain:

```text
/usr/local/lib/python3.12/dist-packages/torch/lib
/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib
```

These can conflict with the Conda environment.

Create an activation script:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/udlm_cuda.sh" <<'EOF'
export PYTHONNOUSERSITE=1

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

export TORCH_CUDA_ARCH_LIST="9.0"

export CPATH="$CONDA_PREFIX/targets/sbsa-linux/include:$CONDA_PREFIX/include"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/sbsa-linux/include:$CONDA_PREFIX/include"

export LIBRARY_PATH="$CONDA_PREFIX/targets/sbsa-linux/lib:$CONDA_PREFIX/lib"

export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/sbsa-linux/lib:$CONDA_PREFIX/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
EOF
```

Reload:

```bash
conda deactivate
conda activate udlm
```

Check:

```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
```

The following path should **not** appear:

```text
/usr/local/lib/python3.12/dist-packages/torch/lib
```

---

# 6. Verify CUDA Development Headers

Check:

```bash
for h in \
    cuda_runtime.h \
    cusparse.h \
    cublas_v2.h \
    cusolverDn.h
do
    echo "===== $h ====="
    find "$CONDA_PREFIX" -name "$h" -print
done
```

They should normally exist under:

```text
$CONDA_PREFIX/targets/sbsa-linux/include/
```

Optional compile test:

```bash
cat >/tmp/test_cuda_headers.cpp <<'EOF'
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
int main() { return 0; }
EOF
```

Compile:

```bash
"$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-c++" \
  -I"$CONDA_PREFIX/targets/sbsa-linux/include" \
  -I"$CONDA_PREFIX/include" \
  -c /tmp/test_cuda_headers.cpp \
  -o /tmp/test_cuda_headers.o \
  && echo "CUDA HEADERS: ALL OK"
```

---

# 7. Install PyTorch 2.8 + CUDA 12.9

```bash
python -m pip install --no-cache-dir \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

Do **not** manually install the old `triton==2.2.0`; PyTorch 2.8 installs its matching Triton version automatically.

Verify:

```bash
python - <<'PY'
import torch
import platform

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

Expected:

```text
Torch: 2.8.0+cu129
Torch CUDA: 12.9
CUDA available: True
GPU: NVIDIA GH200 120GB
Capability: (9, 0)
ABI: True
```

---

# 8. Install Regular UDLM Dependencies

Create:

```bash
cat > ~/NLU/requirements-udlm-base.txt <<'EOF'
biopython==1.84
datasets==2.18.0
einops==0.8.0
fsspec==2024.2.0
git-lfs==1.6
h5py==3.10.0
huggingface-hub==0.26.2
hydra-core==1.3.2
ipdb==0.13.13
jupyter==1.1.1
jupyterlab==4.1.8
lightning==2.2.1
lightning-utilities==0.11.9
matplotlib==3.9.2
notebook==7.1.1
numpy==1.26.4
omegaconf==2.3.0
packaging==24.2
pandas==2.2.1
psutil
pytorch-image-generation-metrics==0.6.1
rdkit==2024.3.6
regex==2024.11.6
rich==13.7.1
safetensors==0.4.5
scikit-learn==1.4.0
scipy==1.13.1
seaborn==0.13.2
timm==0.9.16
tokenizers==0.15.2
torchmetrics==1.6.0
tqdm==4.67.0
transformers==4.38.2
wandb==0.13.5
ninja
EOF
```

Install:

```bash
python -m pip install \
  --no-cache-dir \
  -r ~/NLU/requirements-udlm-base.txt
```

Verify PyTorch was not replaced:

```bash
python - <<'PY'
import torch, torchvision, triton
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Triton:", triton.__version__)
print("CUDA:", torch.version.cuda)
PY
```

---

# 9. Install causal-conv1d

Use:

```text
causal-conv1d==1.6.2.post1
```

Build:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="9.0"

CAUSAL_CONV1D_FORCE_BUILD=TRUE \
CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE \
MAX_JOBS=4 \
python -m pip install \
  causal-conv1d==1.6.2.post1 \
  --no-build-isolation \
  --no-cache-dir
```

Verify:

```bash
python - <<'PY'
import torch
import causal_conv1d
import causal_conv1d_cuda

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("causal-conv1d:", causal_conv1d.__version__)
print("extension:", causal_conv1d_cuda.__file__)
print("IMPORT SUCCESS")
PY
```

Optional kernel test:

```bash
python - <<'PY'
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn

torch.manual_seed(0)

B, D, L, W = 2, 64, 128, 4

x = torch.randn(B, D, L, device="cuda")
weight = torch.randn(D, W, device="cuda")
bias = torch.randn(D, device="cuda")

y = causal_conv1d_fn(x, weight, bias, activation="silu")

ref = F.conv1d(
    x,
    weight.unsqueeze(1),
    bias,
    padding=W - 1,
    groups=D,
)[..., :L]

ref = F.silu(ref)

torch.cuda.synchronize()

print("max error:", (y - ref).abs().max().item())
print("mean error:", (y - ref).abs().mean().item())
print("CAUSAL-CONV1D GPU TEST: SUCCESS")
PY
```

---

# 10. Install mamba-ssm 1.2.0.post1

Do **not** install this version from its PyPI source tarball. Clone the complete GitHub tag.

```bash
mkdir -p "$SCRATCH/udlm-build"
cd "$SCRATCH/udlm-build"

rm -rf mamba-v1.2.0

git clone \
  --branch v1.2.0.post1 \
  --depth 1 \
  https://github.com/state-spaces/mamba.git \
  mamba-v1.2.0

cd mamba-v1.2.0
```

Verify:

```bash
ls -lh csrc/selective_scan/selective_scan.cpp
```

Build:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="9.0"

export MAMBA_FORCE_BUILD=TRUE
export MAMBA_FORCE_CXX11_ABI=TRUE
export MAX_JOBS=4

python -m pip install . \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps \
  -v 2>&1 | tee "$SCRATCH/mamba_build.log"
```

Large amounts of output such as:

```text
ptxas info : Compiling entry function ...
ptxas info : Used ... registers
ptxas info : Compile time = ...
```

are normal.

Warnings such as `LaneId() is deprecated` are also not build failures.

A successful build ends with something similar to:

```text
Successfully built mamba-ssm
Successfully installed mamba-ssm-1.2.0.post1
```

Verify:

```bash
python - <<'PY'
import torch
import mamba_ssm
import selective_scan_cuda

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Mamba:", mamba_ssm.__version__)
print("selective_scan:", selective_scan_cuda.__file__)
print("MAMBA IMPORT: SUCCESS")
PY
```

Optional GPU test:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba

x = torch.randn(2, 128, 64, device="cuda", dtype=torch.float32)

model = Mamba(
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
).cuda()

with torch.no_grad():
    y = model(x)

torch.cuda.synchronize()

print("input :", x.shape)
print("output:", y.shape)
print("finite:", torch.isfinite(y).all().item())
print("MAMBA GPU TEST: SUCCESS")
PY
```

---

# 11. Install FlashAttention

Compiling FlashAttention from source on GH200/aarch64 can take a very long time because it compiles many CUDA translation units.

Use the working prebuilt wheel instead:

```bash
python -m pip install \
  'flash-attn==2.8.3.post1+cu.12.9.torch.2.8' \
  --extra-index-url https://wheels.astral.sh/simple/cu129/ \
  --no-deps
```

This is the preferred installation for:

```text
aarch64
Python 3.11
CUDA 12.9
PyTorch 2.8
GH200 / Hopper
```

Verify:

```bash
python - <<'PY'
import torch
import flash_attn

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("FlashAttention:", flash_attn.__version__)
print("flash_attn file:", flash_attn.__file__)
PY
```

Optional FlashAttention GPU test:

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn_func(q, k, v)

torch.cuda.synchronize()

print("shape:", out.shape)
print("device:", out.device)
print("finite:", torch.isfinite(out).all().item())
print("FLASH ATTENTION GPU TEST: SUCCESS")
PY
```

---

# 12. Final Environment Audit

```bash
python - <<'PY'
import sys
import torch
import torchvision
import triton
import transformers
import causal_conv1d
import mamba_ssm
import flash_attn

print("Python          :", sys.version.split()[0])
print("Torch           :", torch.__version__)
print("Torchvision     :", torchvision.__version__)
print("Torch CUDA      :", torch.version.cuda)
print("Triton          :", triton.__version__)
print("Transformers    :", transformers.__version__)
print("causal-conv1d   :", causal_conv1d.__version__)
print("mamba-ssm       :", mamba_ssm.__version__)
print("flash-attn      :", flash_attn.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("GPU             :", torch.cuda.get_device_name(0))
print("Capability      :", torch.cuda.get_device_capability(0))
print("CXX11 ABI       :", torch._C._GLIBCXX_USE_CXX11_ABI)
PY

nvcc --version
```

Expected overall configuration:

```text
Python             3.11.x
Architecture       aarch64
GPU                NVIDIA GH200 120GB
Compute capability 9.0

CUDA toolkit       12.9
PyTorch            2.8.0+cu129
Torchvision        0.23.0
Triton             3.4.0

causal-conv1d      1.6.2.post1
mamba-ssm          1.2.0.post1
flash-attn         2.8.3.post1
```

---

# 13. Common Problems and Fixes

## Wrong `libtorch_python.so`

Symptom:

```text
ImportError:
/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_python.so
```

Cause: system Python 3.12 Torch library is ahead of the Conda environment in `LD_LIBRARY_PATH`.

Fix: use the activation script from Section 5.

---

## `cusparse.h`, `cublas_v2.h`, or `cusolverDn.h` missing

Cause: CUDA runtime/compiler exists but development libraries are incomplete.

Fix: install the complete CUDA development stack in the Conda environment:

```yaml
- cuda-libraries-dev=12.9.2
```

Then verify:

```bash
find "$CONDA_PREFIX" -name cusparse.h
find "$CONDA_PREFIX" -name cublas_v2.h
find "$CONDA_PREFIX" -name cusolverDn.h
```

---

## `mamba-ssm` says `selective_scan.cpp` is missing

Do not use the PyPI source tarball for `mamba-ssm==1.2.0.post1`.

Use:

```bash
git clone \
  --branch v1.2.0.post1 \
  --depth 1 \
  https://github.com/state-spaces/mamba.git \
  mamba-v1.2.0
```

---

## FlashAttention source build takes a very long time

Use the prebuilt wheel:

```bash
python -m pip install \
  'flash-attn==2.8.3.post1+cu.12.9.torch.2.8' \
  --extra-index-url https://wheels.astral.sh/simple/cu129/ \
  --no-deps
```

---

## Conda reports `Disk quota exceeded`

Move package cache and temporary builds to scratch:

```bash
export CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs"
export TMPDIR="$SCRATCH/pip-tmp"
```

Clean old caches:

```bash
conda clean --all -y
python -m pip cache purge
```

---

# 14. Recommended Compute-Node Initialization

After entering a compute node:

```bash
source ~/miniconda3/bin/activate
conda activate udlm

export CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs"
export TMPDIR="$SCRATCH/pip-tmp"
```

Quick GPU check:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Expected:

```text
2.8.0+cu129
12.9
True
NVIDIA GH200 120GB
```

---

# 15. Notes

- `nvidia-smi` may show a newer driver-supported CUDA version such as CUDA 13.x. PyTorch does not need to use that same toolkit version.
- PyTorch 2.8 + CUDA 12.9 works with newer compatible NVIDIA drivers.
- On Slurm, `torch.cuda.device_count()` reports only GPU(s) exposed to the job.
- Avoid reinstalling PyTorch after compiling CUDA extensions unless you plan to rebuild those extensions.
- Avoid mixing system Python packages with the Conda environment.








```bash
nano "$CONDA_PREFIX/etc/conda/activate.d/udlm_cuda.sh"
```

```bash
# Force Triton to use the CUDA toolkit belonging to this Conda environment.
# Clariden compute-node images may otherwise inject /usr/local/cuda (CUDA 13.x).

unset TRITON_PTXAS_BLACKWELL_PATH
unset TRITON_CUOBJDUMP_PATH
unset TRITON_NVDISASM_PATH
unset TRITON_CUDACRT_PATH
unset TRITON_CUDART_PATH
unset TRITON_CUPTI_INCLUDE_PATH
unset TRITON_CUPTI_LIB_PATH

export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
```