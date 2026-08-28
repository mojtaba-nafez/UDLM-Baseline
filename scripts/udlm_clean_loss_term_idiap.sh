#!/usr/bin/env bash
#SBATCH --job-name=udlm-training-clean-term
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --time=0-09:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p logs-eval-slurm
echo "======= Conda and CUDA ======="

module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate udml

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

echo "================================"

python -u -m main \
  diffusion="uniform" \
  parameterization="d3pm" \
  T=0 \
  time_conditioning=True \
  zero_recon_loss=True \
  data="openwebtext-split" \
  data.wrap=False \
  data.tokenizer_name_or_path=gpt2 \
  loader.global_batch_size=128 \
  loader.eval_global_batch_size=128 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  backbone="dit" \
  model=small \
  model.length=128 \
  optim.lr=3e-4 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=30_000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=122000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10_000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=2 \
  sampling.batch_size=16 \
  sampling.use_cache=True \
  sampling.steps=128 \
  wandb.name="owt_udlm_clean_loss_term" \
  hydra.run.dir="${PWD}/outputs/owt/udlm_clean_loss_term" \
  +training.clean_conf_lambda=0.1