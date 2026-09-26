import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm

import classifier
import dataloader
import diffusion
import eval_utils
import utils
import numpy as np

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y
)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False).to('cuda')


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


import torch
from pathlib import Path


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    print("tokenizer.name_or_path", tokenizer.name_or_path)
    print("config.eval.checkpoint_path: ", config.eval.checkpoint_path)
    model = _load_from_checkpoint(
        config=config, tokenizer=tokenizer)
    model.eval()
    tok_bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.cls_token
    '''
    # print("tok_bos_token: ", tok_bos_token) #  [CLS]
    samples = []
    for _ in tqdm(range(config.sampling.num_sample_batches),
                    desc='Gen. batches', leave=False):
        sample = pretrained.sample()
        samples.extend(
        pretrained.tokenizer.batch_decode(sample))
    '''

    dataset_path = getattr(config, "dataset_path", None)
    chunk_size = getattr(config, "chunk_size", 512)
    batch_size = getattr(config, "batch_size", 32)
    corruption_prob = getattr(config, "corruption_prob", 0.2)
    print(f"Loading dataset from {dataset_path} with chunk_size={chunk_size}, batch_size={batch_size}, corruption_prob={corruption_prob}")

    sep = "<|endoftext|>"
    docs = [d for d in Path(dataset_path).read_text(encoding="utf-8").split(sep) if d]
    token_ids = []
    B = 2000
    for i in tqdm(range(0, len(docs), B), desc="Tokenizing"):
        for ids in tokenizer(docs[i:i+B], add_special_tokens=False)["input_ids"]:
            token_ids.extend(ids); token_ids.append(tokenizer.eos_token_id)

    vocab_size = tokenizer.vocab_size
    n_chunks = len(token_ids) // chunk_size
    token_ids = token_ids[: n_chunks * chunk_size]
    clean_dataset = torch.tensor(token_ids, dtype=torch.long).view(n_chunks, chunk_size)
    print(f"Loaded {len(token_ids)} tokens ({n_chunks} chunks of {chunk_size}, batch_size={batch_size})")
    mask_id = 50257
    eps = 1e-5
    steps = chunk_size
    i = chunk_size - 1

    # timesteps = model._get_sampling_time_profile(eps, steps)

    # Tracking Variables (Pass-1, Pass-5, and Pass-10 counters)
    total_correct = 0
    total_correct_top5 = 0
    total_correct_top10 = 0

    total_copy_correct = 0
    total_copy_correct_top5 = 0
    total_copy_correct_top10 = 0

    total_corrupted = 0
    total_clean = 0

    total_clean_token_correct = 0
    total_clean_token_correct_top5 = 0
    total_clean_token_correct_top10 = 0

    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_chunks, batch_size), desc="Evaluating", miniters=100, mininterval=0):
            batch_end = min(batch_start + batch_size, n_chunks)
            clean_tokens = clean_dataset[batch_start:batch_end].to(model.device)  # [B, chunk_size]

            x = clean_tokens.clone()
            corruption_mask = (torch.rand_like(x.float()) < corruption_prob)
            random_tokens = torch.randint(low=0, high=vocab_size, size=x.shape, device=x.device,)
            x[corruption_mask] = random_tokens[corruption_mask]
            # x[corruption_mask] = mask_id

            num_corrupted = corruption_mask.sum().item()
            num_clean = (~corruption_mask).sum().item()

            if num_corrupted == 0:
                continue

            # Match the manually applied corruption probability
            schedule_eps = model.noise.eps  # 1e-3 for LogLinearNoise
            t_value = corruption_prob / (1.0 - schedule_eps)

            t = torch.full(
                (x.shape[0],),
                t_value,
                device=model.device,
                dtype=torch.float32,
            )

            sigma, _ = model.noise(t)
            sigma = model._process_sigma(sigma)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                logits = model.backbone(
                    x,
                    sigma,
                    cond=None,
                    x_emb=None,
                )

            # logits = torch.where(~corruption_mask.unsqueeze(-1), logits, torch.tensor(-float('inf'), device=logits.device))
            # --- PASS-1 PREDICTIONS ---
            pred_tokens = logits.argmax(dim=-1)

            # --- TOP-K PREDICTIONS (K=10 covers both top-5 and top-10) ---
            top10_preds = logits.topk(k=10, dim=-1).indices  # Shape: [B, chunk_size, 10]

            # Broadcast masks for Top-5 and Top-10 evaluations
            # Slicing top10_preds[:, :, :5] gives the top 5 indices efficiently
            correct_top5_mask = (top10_preds[:, :, :5] == clean_tokens.unsqueeze(-1)).any(dim=-1)
            correct_top10_mask = (top10_preds == clean_tokens.unsqueeze(-1)).any(dim=-1)

            copy_top5_mask = (top10_preds[:, :, :5] == x.unsqueeze(-1)).any(dim=-1)
            copy_top10_mask = (top10_preds == x.unsqueeze(-1)).any(dim=-1)

            # --- TALLYING PASS-1 ---
            correct = ((pred_tokens == clean_tokens) & corruption_mask).sum().item()
            copy_correct = ((pred_tokens == x) & corruption_mask).sum().item()
            clean_token_correct = ((pred_tokens == clean_tokens) & (~corruption_mask)).sum().item()

            # --- TALLYING PASS-5 ---
            correct_top5 = (correct_top5_mask & corruption_mask).sum().item()
            copy_correct_top5 = (copy_top5_mask & corruption_mask).sum().item()
            clean_token_correct_top5 = (correct_top5_mask & (~corruption_mask)).sum().item()

            # --- TALLYING PASS-10 ---
            correct_top10 = (correct_top10_mask & corruption_mask).sum().item()
            copy_correct_top10 = (copy_top10_mask & corruption_mask).sum().item()
            clean_token_correct_top10 = (correct_top10_mask & (~corruption_mask)).sum().item()

            # --- ACCUMULATION ---
            total_correct += correct
            total_correct_top5 += correct_top5
            total_correct_top10 += correct_top10

            total_copy_correct += copy_correct
            total_copy_correct_top5 += copy_correct_top5
            total_copy_correct_top10 += copy_correct_top10

            total_corrupted += num_corrupted
            total_clean += num_clean

            total_clean_token_correct += clean_token_correct
            total_clean_token_correct_top5 += clean_token_correct_top5
            total_clean_token_correct_top10 += clean_token_correct_top10

    # Calculate Final Percentages
    model_acc = 100 * total_correct / max(total_corrupted, 1)
    model_acc_top5 = 100 * total_correct_top5 / max(total_corrupted, 1)
    model_acc_top10 = 100 * total_correct_top10 / max(total_corrupted, 1)

    copy_acc = 100 * total_copy_correct / max(total_corrupted, 1)
    copy_acc_top5 = 100 * total_copy_correct_top5 / max(total_corrupted, 1)
    copy_acc_top10 = 100 * total_copy_correct_top10 / max(total_corrupted, 1)

    clean_acc = 100 * total_clean_token_correct / max(total_clean, 1)
    clean_acc_top5 = 100 * total_clean_token_correct_top5 / max(total_clean, 1)
    clean_acc_top10 = 100 * total_clean_token_correct_top10 / max(total_clean, 1)

    overall_acc = 100 * (total_clean_token_correct + total_correct) / (total_corrupted + total_clean)
    overall_acc_top5 = 100 * (total_clean_token_correct_top5 + total_correct_top5) / (total_corrupted + total_clean)
    overall_acc_top10 = 100 * (total_clean_token_correct_top10 + total_correct_top10) / (total_corrupted + total_clean)

    print("===============Final Results===============")
    print(
        f"Denoising Accuracy (Corrupted Tokens):\n"
        f"  Pass-1:  {model_acc:.6f}% ({total_correct}/{total_corrupted})\n"
        f"  Pass-5:  {model_acc_top5:.6f}% ({total_correct_top5}/{total_corrupted})\n"
        f"  Pass-10: {model_acc_top10:.6f}% ({total_correct_top10}/{total_corrupted})"
    )
    print(
        f"Copying Rate (Corrupted Input -> Output):\n"
        f"  Pass-1:  {copy_acc:.6f}% ({total_copy_correct}/{total_corrupted})\n"
        f"  Pass-5:  {copy_acc_top5:.6f}% ({total_copy_correct_top5}/{total_corrupted})\n"
        f"  Pass-10: {copy_acc_top10:.6f}% ({total_copy_correct_top10}/{total_corrupted})"
    )
    print(
        f"Clean Accuracy (Clean Tokens):\n"
        f"  Pass-1:  {clean_acc:.6f}% ({total_clean_token_correct}/{total_clean})\n"
        f"  Pass-5:  {clean_acc_top5:.6f}% ({total_clean_token_correct_top5}/{total_clean})\n"
        f"  Pass-10: {clean_acc_top10:.6f}% ({total_clean_token_correct_top10}/{total_clean})"
    )
    print(
        f"Overall Accuracy (All Tokens):\n"
        f"  Pass-1:  {overall_acc:.6f}% ({(total_clean_token_correct + total_correct)}/{(total_corrupted + total_clean)})\n"
        f"  Pass-5:  {overall_acc_top5:.6f}% ({(total_clean_token_correct_top5 + total_correct_top5)}/{(total_corrupted + total_clean)})\n"
        f"  Pass-10: {overall_acc_top10:.6f}% ({(total_clean_token_correct_top10 + total_correct_top10)}/{(total_corrupted + total_clean)})"
    )

    results = {
        "dataset_path": dataset_path,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "corruption_prob": corruption_prob,
        "n_chunks": n_chunks,
        "total_corrupted": total_corrupted,
        "total_clean": total_clean,
        "denoising_accuracy": {"pass1": model_acc, "pass5": model_acc_top5, "pass10": model_acc_top10},
        "copying_rate": {"pass1": copy_acc, "pass5": copy_acc_top5, "pass10": copy_acc_top10},
        "clean_accuracy": {"pass1": clean_acc, "pass5": clean_acc_top5, "pass10": clean_acc_top10},
        "overall_accuracy": {"pass1": overall_acc, "pass5": overall_acc_top5, "pass10": overall_acc_top10},
    }
    save_path = getattr(config, "save_path", "perturbation_results.json")
    with open(save_path, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved results to {save_path}")


if __name__ == '__main__':
  main()