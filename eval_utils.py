import os

import torch
import transformers
from tqdm import tqdm

import diffusion

import json
import numpy as np
import json

@torch.no_grad()
def compute_ppl(pretrained_model, val_ds, split_path=None, setup=None):
  ppl_metrics = diffusion.Perplexity().to('cuda')
  pretrained_model.split_S = torch.zeros(4, 10, dtype=torch.float64, device='cuda'); pretrained_model.split_N = torch.zeros_like(pretrained_model.split_S)
  pbar = tqdm(val_ds, desc='PPL')
  for batch in pbar:
    input_ids = batch['input_ids'].to('cuda'); attention_mask = batch['attention_mask'].to('cuda') if 'attention_mask' in batch else None
    losses = pretrained_model._loss(input_ids, attention_mask)
    ppl_metrics.update(losses.nlls, losses.token_mask); pbar.set_postfix({'ppl': ppl_metrics.compute().item()})
  if split_path is not None: save_nelbo_split(pretrained_model.split_S, pretrained_model.split_N, split_path, setup or {}, 'UDLM continuous-time per-token NELBO (diffusion_loss, + reconstruction_loss unless zero_recon_loss=True)')
  pretrained_model.split_S = None
  return ppl_metrics.compute().item()


def save_nelbo_split(S, N, path, setup, estimator):
  S, N = S.cpu().double(), N.cpu().double(); nb = S.shape[1]; cats = ['clean_right', 'clean_wrong', 'corrupt_right', 'corrupt_wrong']
  tot = (S.sum() / N.sum()).item(); m = S.sum(1) / N.sum(1).clamp(min=1); mb = S / N.clamp(min=1)
  L = lambda v, n: [round(float(a), 6) if c > 0 else None for a, c in zip(v, n)]  # None where the bin has no tokens
  acc_c, acc_r = N[0] / (N[0] + N[1]).clamp(min=1), N[2] / (N[2] + N[3]).clamp(min=1)
  out = {
    'description': {
      'summary': 'Validation NELBO (likelihood upper bound, WITHOUT the CTR-Reg term) decomposed by position type (clean vs corrupted) and by whether the argmax prediction is correct, overall and per noise-level bin.',
      'units': 'Loss values in nats per token. PPL = exp(NELBO per token). BPD = NELBO per token / ln 2.',
      'estimator': f'One t ~ U[sampling_eps, 1] per sequence (antithetic within batch), one uniform corruption per sequence; per-token loss = {estimator}.',
      'categories': {'clean_right': 'z_t == x0 and argmax == x0 (token retained)', 'clean_wrong': 'z_t == x0 and argmax != x0 (retention error)',
                     'corrupt_right': 'z_t != x0 and argmax == x0 (successful revision)', 'corrupt_wrong': 'z_t != x0 and argmax != x0 (revision error)'},
      'penalty': 'mean NELBO of *_wrong minus mean NELBO of *_right, within the same position type (nats).',
      'accuracy': 'acc_clean: fraction of clean positions with argmax == x0 (retention); acc_corrupt: same for corrupted positions (revision).',
      't_bins': 'Bins over t in [0,1], low noise (t~0) to high noise (t~1); with the log-linear schedule, fraction corrupted ~ t.',
      'note': 'Uniform draws equal to the original token count as clean (matches the NELBO). Per-category means compared across models are subject to selection effects.'},
    'setup': setup,
    'overall': {'nelbo_per_token': tot, 'ppl': float(np.exp(tot)), 'bpd': tot / np.log(2), 'num_tokens': float(N.sum()), 'clean_share': float(S[:2].sum() / S.sum()),
                'acc_clean': float(N[0].sum() / N[:2].sum()), 'acc_corrupt': float(N[2].sum() / N[2:].sum()),
                'mean_nelbo_clean': float(S[:2].sum() / N[:2].sum()), 'mean_nelbo_corrupt': float(S[2:].sum() / N[2:].sum()),
                'penalty_clean': float(m[1] - m[0]), 'penalty_corrupt': float(m[3] - m[2])},
    'categories': {c: {'mean_nelbo_per_token': float(m[i]), 'total_nelbo': float(S[i].sum()), 'num_tokens': float(N[i].sum())} for i, c in enumerate(cats)},
    'per_t_bin': {'t_bin_edges': [i / nb for i in range(nb + 1)], 't_bin_centers': [(i + 0.5) / nb for i in range(nb)],
                  'acc_clean': L(acc_c, N[0] + N[1]), 'acc_corrupt': L(acc_r, N[2] + N[3]),
                  'penalty_clean': L(mb[1] - mb[0], torch.minimum(N[0], N[1])), 'penalty_corrupt': L(mb[3] - mb[2], torch.minimum(N[2], N[3])),
                  'mean_nelbo_per_token': {c: L(mb[i], N[i]) for i, c in enumerate(cats)}, 'num_tokens': {c: N[i].tolist() for i, c in enumerate(cats)}}}
  with open(f'{path}.json', 'w') as fp: json.dump(out, fp, indent=2)
  o, p, f = out['overall'], out['per_t_bin'], lambda v: np.array(v, dtype=float).round(3)
  print(f"NELBO/tok {o['nelbo_per_token']:.4f} PPL {o['ppl']:.2f} | clean share {o['clean_share']:.3f} | saved {path}.json")
  print(f"CLEAN   acc {o['acc_clean']:.3f} | right {m[0]:.4f} wrong {m[1]:.4f} penalty {o['penalty_clean']:.4f}")
  print(f"CORRUPT acc {o['acc_corrupt']:.3f} | right {m[2]:.4f} wrong {m[3]:.4f} penalty {o['penalty_corrupt']:.4f}")
  print('acc clean  ', f(p['acc_clean'])); print('acc corrupt', f(p['acc_corrupt'])); print('pen clean  ', f(p['penalty_clean'])); print('pen corrupt', f(p['penalty_corrupt'])); print('count/cat  ', f(N.sum(1)))

def compute_generative_ppl(
    sentences,
    eval_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=128):
  gen_ppl_metric = diffusion.Perplexity().to('cuda')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    eval_model_name_or_path)
  if eval_model_tokenizer.pad_token is None:
    eval_model_tokenizer.pad_token = \
      eval_model_tokenizer.eos_token
    eval_model_tokenizer.pad_token_id = \
      eval_model_tokenizer.eos_token_id
  eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name_or_path).eval()
  if max_length is None:
    max_length = max_length
  eval_model = eval_model.to('cuda')
  # Re-tokenize using eval model's tokenizer
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'truncation': True,
    'padding': True,
    'max_length': max_length,
  }
  eval_context_size = 1024
  samples = eval_model_tokenizer(
    sentences, **tokenizer_kwargs)
  attn_mask = samples['attention_mask']
  samples = samples['input_ids']
  attn_mask = attn_mask.to('cuda')
  samples = samples.to('cuda')
  num_batches = samples.shape[0] // gen_ppl_eval_batch_size
  for i in tqdm(range(num_batches),
                desc='Gen. PPL', leave=False):
    _samples = torch.split(
      samples[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    _attn_mask = torch.split(
      attn_mask[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
      logits = eval_model(
        sample_chunk, attention_mask=attn_mask_chunk)[0]
      logits = logits.transpose(-1, -2)

      nlls = torch.nn.functional.cross_entropy(
        logits[..., :-1],
        sample_chunk[..., 1:],
        reduction='none')
      # first_eos = (sample_chunk == eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
      # token_mask = (sample_chunk != eval_model_tokenizer.eos_token_id)
      # gen_ppl_metric.update(
      #   nlls, first_eos[..., 1:] + token_mask[..., 1:])
      gen_ppl_metric.update(
        nlls, attn_mask_chunk[..., 1:])
  return gen_ppl_metric.compute().item()
