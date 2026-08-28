# Idiap Server Installation!

```bash
module load CUDA/12.8

conda create -n discdiff python=3.9.20 -y
conda activate discdiff

conda install -y -c pytorch -c nvidia pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=12.1
conda install -y ipykernel=6.29.5 ipython=8.15.0 ipywidgets=8.1.2 pip=23.3.1

pip install biopython==1.84
pip install causal-conv1d==1.4.0
pip install mamba-ssm==1.2.0.post1
pip install flash-attn==2.7.2.post1
pip install datasets==2.18.0 einops==0.8.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 \
  huggingface-hub==0.26.2 hydra-core==1.3.2 ipdb==0.13.13 jupyter==1.1.1 \
  jupyterlab==4.1.8 lightning==2.2.1 lightning-utilities==0.11.9 matplotlib==3.9.2 \
  notebook==7.1.1 numpy==1.26.4 omegaconf==2.3.0 pandas==2.2.1 \
  pytorch-image-generation-metrics==0.6.1 rdkit==2024.3.6 regex==2024.11.6 \
  rich==13.7.1 safetensors==0.4.5 scikit-learn==1.4.0 scipy==1.13.1 seaborn==0.13.2 \
  timm==0.9.16 tokenizers==0.15.2 torchmetrics==1.6.0 tqdm==4.67.0 \
  transformers==4.38.2 wandb==0.13.5
```

# Sample PPL GPT-Large

```bash
python -u -m main hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 mode="gen_ppl_eval" model.pretrained_model_name_or_path="kuleshov-group/udlm-lm1b" data=lm1b backbone=hf_dit model=hf model.length=128 zero_recon_loss=True training.guidance=null parameterization=d3pm diffusion=uniform time_conditioning=True T=0 sampling.num_sample_batches=32 sampling.batch_size=32 sampling.steps=256 sampling.use_cache=False sampling.use_float64=False eval.generated_samples_path="$PWD/outputs/lm1b/udlm/samples-lm1b-gen-ppl-eval-float64-False_add-CLS_T-256_seed-1-Original.json" +eval.generative_ppl_model_name_or_path="gpt2-large"
```



# Training

```bash
python -u -m main \
  diffusion="uniform" \
  parameterization="d3pm" \
  T=0 \
  time_conditioning=True \
  zero_recon_loss=True \
  data="openwebtext-split" \
  data.wrap=False \
  data.tokenizer_name_or_path=gpt2 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=1024 \
  loader.batch_size=64 \
  loader.eval_batch_size=128 \
  backbone="dit" \
  model=small \
  model.length=512 \
  optim.lr=3e-4 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100_000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=1_000_000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10_000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=True \
  sampling.steps=128 \
  wandb.name="lm1b_udlm" \
  hydra.run.dir="${PWD}/outputs/lm1b/udlm" \
  +training.clean_conf_lambda=0.1
```


```bash
sbatch -p gpu -A balm /idiap/temp/mnafez/research/discrete-diffusion-guidance/scripts/udlm_clean_loss_term_idiap.sh
```


```bash
sbatch -p gpu -A balm  /idiap/temp/mnafez/research/discrete-diffusion-guidance/scripts/udlm_idiap.sh
```


```bash
sbatch -p gpu -A balm  /idiap/temp/mnafez/research/discrete-diffusion-guidance/scripts/udlm_time_dependent_clean_loss_term_idiap.sh
```