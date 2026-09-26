#!/usr/bin/env bash
#SBATCH --job-name=rand_perturb_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=60G
#SBATCH --time=0-18:00:00
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

dataset_name="wikitext103"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/wikitext103_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_original_p-$p.json"

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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_clean_loss_p-$p.json"
      
done





dataset_name="pubmed"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/pubmed_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_original_p-$p.json"

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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_clean_loss_p-$p.json"
      
done



dataset_name="lm1b"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/lm1b_test.txt"
for p in 0.1 0.2 0.3 0.4
do
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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_original_p-$p.json"

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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_clean_loss_p-$p.json"
      
done




dataset_name="arxiv"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/arxiv_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_original_p-$p.json"

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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_clean_loss_p-$p.json"
      
done




dataset_name="ag_news"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/ag_news_test.txt"
for p in 0.1 0.2 0.3 0.4
do
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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_original_p-$p.json"

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
        +dataset_path="$dataset_path" \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/discrete-diffusion-guidance/rand_perturb_exp/${dataset_name}_clean_loss_p-$p.json"
      
done