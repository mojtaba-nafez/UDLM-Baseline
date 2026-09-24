```bash
# Idiap Server Installation!


```bash
conda env remove -n udlm -y



module load CUDA/12.8

conda create -n udlm python=3.9.20 -y
conda activate udlm


conda install -y \
  -c nvidia/label/cuda-12.8.1 \
  cuda-nvcc


export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=2

which nvcc
nvcc --version


conda install -y -c pytorch -c nvidia pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=12.1
conda install -y ipykernel=6.29.5 ipython=8.15.0 ipywidgets=8.1.2 pip=23.3.1

pip install datasets==2.18.0 einops==0.8.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 \
  huggingface-hub==0.26.2 hydra-core==1.3.2 ipdb==0.13.13 jupyter==1.1.1 \
  jupyterlab==4.1.8 lightning==2.2.1 lightning-utilities==0.11.9 matplotlib==3.9.2 \
  notebook==7.1.1 numpy==1.26.4 omegaconf==2.3.0 pandas==2.2.1 \
  pytorch-image-generation-metrics==0.6.1 rdkit==2024.3.6 regex==2024.11.6 \
  rich==13.7.1 safetensors==0.4.5 scikit-learn==1.4.0 scipy==1.13.1 seaborn==0.13.2 \
  timm==0.9.16 tokenizers==0.15.2 torchmetrics==1.6.0 tqdm==4.67.0 \
  transformers==4.38.2 wandb==0.13.5
  
pip install biopython==1.84

pip install causal-conv1d==1.4.0
pip install mamba-ssm==1.2.0.post1
pip install flash-attn==2.7.2.post1

```




# verify we are still on the good torch
python - <<'PY'
import torch, platform
print("arch:", platform.machine())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("GPU:", torch.cuda.get_device_name(0))
PY