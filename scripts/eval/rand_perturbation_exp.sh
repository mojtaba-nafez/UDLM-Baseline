#!/usr/bin/env bash
#SBATCH --job-name=rand_perturb_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=60G
#SBATCH --time=0-06:00:00
#SBATCH --output=rand_perturb_exp/%x-%j.out
#SBATCH --error=rand_perturb_exp/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p rand_perturb_exp
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

python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm/5-42000.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.1 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/original_p-0-1.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm/5-42000.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.2 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/original_p-0-2.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm/5-42000.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.3 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/original_p-0-3.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm/5-42000.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.4 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/original_p-0-4.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm/5-42000.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.5 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/original_p-0-5.json"








python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.1 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/clean_loss_p-0-1.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.2 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/clean_loss_p-0-2.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.3 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/clean_loss_p-0-3.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.4 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/clean_loss_p-0-4.json"


python -u -m random_perturbation_experiment hydra.output_subdir=null hydra.run.dir="$PWD" hydra/job_logging=disabled hydra/hydra_logging=disabled seed=1 \
	mode="gen_ppl_eval" \
    eval.checkpoint_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/our-trained-checkpoints/owt/checkpoints-cscs-trained/udlm_clean_loss_term_time_dependent-v2/5-42000-v1.ckpt" \
    data=openwebtext-split \
    backbone=dit \
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
	+eval.generative_ppl_model_name_or_path="gpt2-large" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.5 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/clean_loss_p-0-5.json"

