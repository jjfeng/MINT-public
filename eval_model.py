"""Pick the validation patients, freeze their sequences, and sample rollouts of a trained MINT checkpoint: for each
target position p of each validation patient, the model sees the inputs up to p (everything before target p) and
samples the next J events, so rollout step 0 is the model's draw for target p itself.

Sequences are left-padded by J - 1 so that train_bayes.py can append a full rollout to any prefix within one row."""
import argparse
import logging

import numpy as np
import torch
import xarray as xr

from train_bayes import FROZEN_VARS, frozen_batch, left_pad, load_model
from utils import get_p2i, load_demographic_token_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--val-npy', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--n-val-obs', type=int, required=True, help='validation patients')
    parser.add_argument('--n-rollout', type=int, required=True, help='events J predicted after each prefix')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-rollouts', required=True, help='zarr directory with the val sequences and rollouts')
    parser.add_argument('--log', required=True)
    return parser.parse_args()


@torch.no_grad()
def sample_rollout(model, idx, age, n_rollout, gen):
    """Append n_rollout sampled events to (idx, age) of shape (b, t); returns their tokens and ages, (b, n_rollout)."""
    for _ in range(n_rollout):
        logits = model(idx, age)[0][:, -1]
        logits[:, model.config.ignore_tokens] = -torch.inf
        # as in Delphi.generate: every token draws an exponential waiting time and the first one to fire is the event
        u = torch.rand(logits.shape, generator=gen, device=logits.device)
        wait, token = (-torch.exp(-logits) * u.log()).min(dim=1)
        idx = torch.cat([idx, token[:, None]], dim=1)
        age = torch.cat([age, age[:, -1:] + wait[:, None]], dim=1)
    return idx[:, -n_rollout:], age[:, -n_rollout:]


def main():
    args = parse_args()
    logging.basicConfig(filename=args.log, filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    rng = np.random.default_rng(args.seed)
    model, train_cfg = load_model(args.ckpt, args.device)
    model.eval()

    val_data = np.load(args.val_npy).astype(np.uint32)
    val_p2i = get_p2i(val_data)
    val_ix = rng.choice(len(val_p2i), args.n_val_obs, replace=False)
    batch = frozen_batch(val_data, val_p2i, val_ix, train_cfg, load_demographic_token_ids(args.vocab))
    X, A, Y, B = [left_pad(t, args.n_rollout - 1).to(args.device) for t in batch]

    # padding sits at the start of each row, so every prefix ends at the patient's latest input
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    rollouts = [sample_rollout(model, X[:, :p + 1], A[:, :p + 1], args.n_rollout, gen) for p in range(X.shape[1])]
    tokens, ages = [torch.stack(r, dim=1).cpu().numpy() for r in zip(*rollouts)]
    out = xr.Dataset({name: (('batch', 'position'), t.cpu().numpy()) for name, t in zip(FROZEN_VARS, [X, A, Y, B])})
    out['rollout_tokens'] = (('batch', 'position', 'rollout_step'), tokens)
    out['rollout_ages'] = (('batch', 'position', 'rollout_step'), ages)
    out['patient'] = ('batch', val_data[val_p2i[val_ix, 0], 0])
    out.to_zarr(args.out_rollouts, mode='w')
    logging.info(f"rollouts {dict(out.sizes)} saved to {args.out_rollouts}")


if __name__ == "__main__":
    main()
