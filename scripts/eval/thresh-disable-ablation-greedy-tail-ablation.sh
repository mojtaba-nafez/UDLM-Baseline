#!/usr/bin/env bash
#SBATCH --job-name=greedy-tail-ablation-threshold-disable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p logs-eval-slurm
mkdir -p greedy-tail-ablation-result
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

B=512
for greedy_step in 0 1 2 3 4 5 6 8 10 12 14 16 18 20; do

    b_update=$((B - greedy_step))
    python -u -m main \
        hydra.output_subdir=null hydra.run.dir="$PWD" \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        seed=1 \
        mode="gen_ppl_eval" \
        eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/udlm_clean_loss_term_time_independent-v2-ablation/5-42000.ckpt" \
        data="openwebtext-split" \
        backbone=dit \
        model=small \
        model.length=512 \
        zero_recon_loss=True \
        training.guidance=null \
        parameterization=d3pm \
        diffusion=uniform \
        time_conditioning=True \
        T=0 \
        sampling.num_sample_batches=32 \
        sampling.batch_size=32 \
        sampling.steps="$b_update" \
        sampling.use_cache=False \
        sampling.use_float64=False \
        eval.generated_samples_path="$PWD/greedy-tail-ablation-result/threshold-disable-udlm-clean-term-step-512-greedy-${greedy_step}.json" \
        +eval.generative_ppl_model_name_or_path="gpt2-large" \
        +sampling.noise_removal="greedy" \
        +sampling.noise_removal_steps="$greedy_step" \

done