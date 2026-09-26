#!/usr/bin/env bash
#SBATCH --job-name=udlm-val-loss-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue


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


datasets=("scientific_papers_arxiv" "scientific_papers_pubmed" "wikitext103")

model="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt"

for data in "${datasets[@]}"; do
    python -u -m main \
        hydra.output_subdir=null \
        hydra.run.dir="${PWD}" \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled \
        seed=1 \
        mode="ppl_eval" \
        eval.checkpoint_path=$model \
        eval.generate_samples=False \
        loader.eval_batch_size=16 \
        data=$data \
        data.tokenizer_name_or_path=gpt2 \
        model.length=512 \
        backbone=dit \
        model=small \
        training.guidance=null \
        parameterization=d3pm \
        diffusion="uniform" \
        time_conditioning=True \
        zero_recon_loss=True \
        T=0 \
        +val_loss_save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/validation_loss_result/${data}_udlm_crt_reg"
done