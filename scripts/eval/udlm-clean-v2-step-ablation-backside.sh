#!/usr/bin/env bash
#SBATCH --job-name=udlm-full-spectrogram-backside
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --time=0-06:00:00
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
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=16 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-16-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    


echo "================================"

python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=32 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-32-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    


echo "================================"


python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=64 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-64-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    


echo "================================"

python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=128 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-128-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    


echo "================================"

python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=256 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-256-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    


echo "================================"


python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=512 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-512-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0 
    


echo "================================"


python -u -m main \
    hydra.output_subdir=null hydra.run.dir="$PWD" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    seed=1 \
    mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
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
    sampling.steps=1024 \
    sampling.use_cache=False \
    sampling.use_float64=False \
    eval.generated_samples_path="$PWD/outputs/owt/step-ablation-main/udlm-clean-v2-term-1024-greedy-0.json" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +sampling.noise_removal="greedy" \
    +sampling.noise_removal_steps=0
    

