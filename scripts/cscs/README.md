

# Main Training (2 Node):

```bash
sbatch --environment=gidd \
  --nodes=2 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  -A a0236 \
  --time=0-10:00:00 \
  /users/mojtaba_nafez/NLU/UDLM-Baseline/scripts/cscs/udml_owt_clean_term_loss_time_dependent.sh
```



```bash
sbatch --environment=gidd \
  --nodes=2 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  -A a0236 \
  --time=0-10:00:00 \
  /users/mojtaba_nafez/NLU/UDLM-Baseline/scripts/cscs/udml_owt_clean_term_loss_time_dependent-v2.sh
```



```bash
sbatch --environment=gidd \
  --nodes=2 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  -A a0236 \
  --time=0-10:00:00 \
  /users/mojtaba_nafez/NLU/UDLM-Baseline/scripts/cscs/udml_owt.sh
```


```bash
sbatch --nodes=2 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  -A a0236 \
  --time=0-10:00:00 \
  /users/mojtaba_nafez/NLU/UDLM-Baseline/scripts/cscs/udml_owt_clean_term_loss_time_dependent-ablation.sh
```







# 1 Node:
sbatch --environment=gidd --nodes=1 -A a0236 --time=0-12:00:00 /users/mojtaba_nafez/NLU/UDLM-Baseline/scripts/cscs/training_clean.sh



```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main \
  diffusion="uniform" \
  parameterization="d3pm" \
  T=0 \
  time_conditioning=True \
  zero_recon_loss=True \
  data="openwebtext-split" \
  data.wrap=False \
  data.tokenizer_name_or_path=gpt2 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=512 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  backbone="dit" \
  model=small \
  model.length=512 \
  optim.lr=3e-4 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=6000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=68000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=1000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=2 \
  sampling.batch_size=16 \
  sampling.use_cache=True \
  sampling.steps=512 \
  wandb.name="owt_udlm_clean_loss_term_time_dependent" \
  hydra.run.dir="${PWD}/outputs/owt/udlm_clean_loss_term_time_dependent" \
  +training.clean_conf_lambda=0.1 \
  +training.clean_conf_time_power=1.0 \
  +training.clean_conf_t_threshold=0.7 \
  loader.num_workers=16 \
  loader.persistent_workers=True
```

1 node
4 GPUs
batch_size per GPU = 64
accumulate_grad_batches = 2
global_batch_size = 512
sequence length = 512
max_steps = 136,000


"30914": micro-batch
Epoch 0: 1% 390/30914 [02:23<3:07:28, 2.71it/s, ...]

each GPU:       64 sequences
4 GPUs:    4 × 64 = 256 sequences per micro-batch


effective global batch is 512:
64 × 4 GPUs × 2 accumulation = 512


trainer.max_steps=68000
optimizer/global steps.