#!/usr/bin/env bash

#SBATCH --job-name=udlm-owt-clean-term-v2
#SBATCH --account=a0236
#SBATCH --partition=normal

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

#SBATCH --cpus-per-task=8

#SBATCH --time=0-10:00:00

#SBATCH --output=logs-train-slurm/clean-term-time-dep-v2-%x-%j.out
#SBATCH --error=logs-train-slurm/clean-term-time-dep-v2-%x-%j.err

set -euo pipefail

PROJECT_DIR="/mnt/home/NLU/UDLM-Baseline"

cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)

export MASTER_PORT
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "UDLM distributed training"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Nodes:              ${SLURM_JOB_NUM_NODES}"
echo "Node list:          ${SLURM_JOB_NODELIST}"
echo "Tasks per node:     ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks:        ${SLURM_NTASKS}"
echo "MASTER_ADDR:        ${MASTER_ADDR}"
echo "MASTER_PORT:        ${MASTER_PORT}"
echo "============================================================"

scontrol show hostnames "${SLURM_JOB_NODELIST}"

srun -ul --cpu-bind=cores bash -c '

    source /mnt/home/miniconda3/etc/profile.d/conda.sh
    conda activate udlm

    cd /mnt/home/NLU/UDLM-Baseline

    echo "------------------------------------------------------------"
    echo "Host:                 $(hostname)"
    echo "SLURM_PROCID:         ${SLURM_PROCID}"
    echo "SLURM_LOCALID:        ${SLURM_LOCALID}"
    echo "SLURM_NODEID:         ${SLURM_NODEID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "Python:               $(which python)"
    echo "------------------------------------------------------------"

    python -u -m main \
        diffusion="uniform" \
        parameterization="d3pm" \
        T=0 \
        time_conditioning=True \
        zero_recon_loss=True \
        data="openwebtext-split" \
        data.wrap=False \
        data.tokenizer_name_or_path=gpt2 \
        loader.global_batch_size=1024 \
        loader.eval_global_batch_size=512 \
        loader.batch_size=64 \
        loader.eval_batch_size=64 \
        backbone="dit" \
        model=small \
        model.length=512 \
        optim.lr=3e-4 \
        training.guidance=null \
        training.compute_loss_on_pad_tokens=False \
        checkpointing.resume_from_ckpt=False \
        callbacks.checkpoint_every_n_steps.every_n_train_steps=6000 \
        trainer.log_every_n_steps=100 \
        trainer.max_steps=68000 \
        trainer.num_nodes=2 \
        trainer.precision=bf16-mixed \
        trainer.val_check_interval=1000 \
        eval.generate_samples=True \
        sampling.num_sample_batches=2 \
        sampling.batch_size=16 \
        sampling.use_cache=True \
        sampling.steps=512 \
        wandb.name="owt_udlm_clean_loss_term_time_dependent-v2" \
        hydra.run.dir="${PWD}/outputs/owt/udlm_clean_loss_term_time_dependent-v2" \
        +training.clean_conf_lambda=0.02 \
        +training.clean_conf_time_power=1.0 \
        +training.clean_conf_t_threshold=0.5 \
        loader.num_workers=8 \
        loader.persistent_workers=True
'