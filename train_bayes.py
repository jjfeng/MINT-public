"""Compute Bayesian Influence Functions (BIF) of training patients on validation patients for a trained
MINT checkpoint, using SGLD sampling from devinterp.

devinterp expects fixed-length `input_ids` and a loss `(model, input_ids) -> (batch, seq-1)`. Delphi needs
tokens and ages, so every patient's batch (X, A, Y, B) is built once and frozen, and each patient's
`input_ids` row is just its row index into those frozen tensors, repeated to the row length + 1.

The validation sequences and their J-event rollouts come from eval_model.py. With --rollout-bif, every
(validation patient, target position) also gets a frozen row of its history followed by its rollout, and the
BIF of the J rolled out events is saved as `rollout_influences`.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from datasets import Dataset
from devinterp.slt.bif import compute_bif
from devinterp.slt.sampling import sample

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch, load_demographic_token_ids

FROZEN_VARS = ['tokens', 'ages', 'target_tokens', 'target_ages']
MASK_TIME = -10000.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON file with SGLD and patient-count settings')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--train-npy', required=True)
    parser.add_argument('--rollouts', required=True, help='zarr directory written by eval_model.py')
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--correlation-method', required=True, choices=['token', 'sequence'])
    parser.add_argument('--rollout-bif', action='store_true', help='also compute the BIF of the rolled out events')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out-samples', required=True, help='zarr directory for the SGLD loss traces')
    parser.add_argument('--out-bif', required=True, help='zarr directory with train x val influences')
    parser.add_argument('--log', required=True)
    return parser.parse_args()


def load_model(ckpt_path, device):
    """Trained Delphi with dropout off, so loss traces vary only through the SGLD parameter draws."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**{**checkpoint['model_args'], 'dropout': 0.0}))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
    return model.to(device), checkpoint['config']


def frozen_batch(data, p2i, ix, train_cfg, demographic_token_ids):
    """Deterministic, uncut (X, A, Y, B) for patients `ix`, each of length block_size."""
    return get_batch(ix, data, p2i, block_size=train_cfg['model']['block_size'], padding='regular',
                     select=train_cfg['select'], no_event_token_rate=train_cfg['no_event_token_rate'],
                     cut_batch=False, demographic_token_ids=demographic_token_ids)


def left_pad(t, n):
    """Prepend n padding positions along the last dim: token 0, or the masked age for float tensors."""
    return F.pad(t, (n, 0), value=MASK_TIME if t.is_floating_point() else 0)


def rollout_rows(rollouts):
    """(X, A, Y, B) rows of each val history up to target position p followed by its rollout from p, for every
    real target p; the rolled out targets are the last J positions of each row. Also returns the (patient, p) pairs."""
    X, A, Y = [torch.from_numpy(rollouts[n].values) for n in ['tokens', 'ages', 'target_tokens']]
    R, RA = torch.from_numpy(rollouts['rollout_tokens'].values), torch.from_numpy(rollouts['rollout_ages'].values)
    length = X.shape[1] + 1
    vs, ps = np.nonzero(Y.numpy() > 0)

    def row(prefix, rollout):
        seq = torch.cat([prefix, rollout])
        # the leading positions cut to fit the row are padding, as eval_model.py left-pads by J - 1
        cut = seq[:-length]
        assert (cut == left_pad(cut[:0], len(cut))).all(), 'a rollout row would cut real events'
        return left_pad(seq, length)[-length:]
    tokens = torch.stack([row(X[v, :p + 1], R[v, p]) for v, p in zip(vs, ps)])
    ages = torch.stack([row(A[v, :p + 1], RA[v, p]) for v, p in zip(vs, ps)])
    return (tokens[:, :-1], ages[:, :-1], tokens[:, 1:], ages[:, 1:]), vs, ps


def index_dataset(rows, row_length):
    return Dataset.from_dict({'input_ids': np.repeat(rows[:, None], row_length + 1, axis=1)}).with_format('torch')


def observable_rows(rows, batch_size, empty_row):
    """Observables are evaluated in full batches, so top up `rows` with the empty row (constant zero loss)."""
    return np.concatenate([rows, np.full(-len(rows) % batch_size, empty_row)])


def make_loss_fn(X, A, Y, B):
    def loss_fn(model, input_ids):
        rows = input_ids[:, 0]
        return model.per_token_ce(X[rows], A[rows], Y[rows], B[rows])
    return loss_fn


def rollout_influences(samples, bif, obs_ranges, rollout_index, n_rollout, batch_size, device):
    """BIF of the train events on the last n_rollout targets of each rollout row, as (batch_1, batch_2, position,
    rollout_step, target_position) aligned with `bif`; NaN where a val position has no rollout. Computed in chunks
    of rollout rows, keeping only the rolled out targets, since the full token-level BIF would not fit in memory.
    rollout_index maps each rollout row to its (frozen val row, target position)."""
    out = np.full((bif.sizes['batch_1'], bif.sizes['batch_2'], bif.sizes['target_position_T'], n_rollout,
                   bif.sizes['target_position']), np.nan, dtype=np.float32)
    val_pos = {row: k for k, row in enumerate(bif['val_row'].values)}
    for start in range(obs_ranges['rollout'].start, obs_ranges['rollout'].stop, batch_size):
        chunk_range = range(start, start + batch_size)
        chunk = compute_bif(samples, correlation_method='token', loss_keys=list(obs_ranges),
                            batch_index_range_1=obs_ranges['train'], batch_index_range_2=chunk_range, device=device)
        chunk_rows = chunk['input_ids'].values[:, 0]
        assert (chunk_rows[obs_ranges['train']] == bif['train_row'].values).all(), 'train order changed'
        influences = chunk['influences'].isel(target_position_T=slice(-n_rollout, None)).transpose(
            'batch_1', 'batch_2', 'target_position_T', 'target_position').values
        for k, row in enumerate(chunk_rows[chunk_range]):
            if row in rollout_index:
                val_row, p = rollout_index[row]
                out[:, val_pos[val_row], p] = influences[:, k]
    return out


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    logging.basicConfig(filename=args.log, filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, train_cfg = load_model(args.ckpt, cfg['device'])
    demographic_token_ids = load_demographic_token_ids(args.vocab)
    rollouts = xr.open_zarr(args.rollouts).load()
    n_rollout, n_val = rollouts.sizes['rollout_step'], rollouts.sizes['batch']

    train_data = np.load(args.train_npy).astype(np.uint32)
    train_p2i = get_p2i(train_data)
    # SGLD posterior uses n_sgld training patients; the influencing patients are the first n_train_obs of them
    train_ix = rng.choice(len(train_p2i), cfg['n_sgld'], replace=False)
    train_batch = [left_pad(t, n_rollout - 1) for t in
                   frozen_batch(train_data, train_p2i, train_ix, train_cfg, demographic_token_ids)]
    val_batch = [torch.from_numpy(rollouts[n].values) for n in FROZEN_VARS]
    parts = [train_batch, val_batch]
    if args.rollout_bif:
        rollout_batch, vs, ps = rollout_rows(rollouts)
        parts.append(rollout_batch)
    # one all-padding row at the end tops up observables to full batches
    empty = [left_pad(t[:1, :0], t.shape[1]) for t in train_batch]
    X, A, Y, B = [torch.cat(ts).to(cfg['device']) for ts in zip(*parts, empty)]
    row_length = X.shape[1]
    empty_row = len(X) - 1

    # rows of the frozen tensors: training patients, validation patients, then rollout rows
    offsets = np.cumsum([0] + [len(p[0]) for p in parts])
    assert cfg['n_train_obs'] % cfg['batch_size'] == 0, 'train observables must fill whole batches, with no top-up'
    train_rows = np.arange(cfg['n_train_obs'])
    val_rows = offsets[1] + np.arange(n_val)
    rows = {'train': train_rows, 'val': val_rows}
    if args.rollout_bif:
        rows['rollout'] = offsets[2] + np.arange(len(vs))
    obs_rows = {name: observable_rows(r, cfg['batch_size'], empty_row) for name, r in rows.items()}
    observables = {name: (index_dataset(r, row_length), len(r) // cfg['batch_size']) for name, r in obs_rows.items()}
    obs_ranges = dict(zip(obs_rows, [range(s, s + len(r)) for s, r in
                                     zip(np.cumsum([0] + [len(r) for r in obs_rows.values()]), obs_rows.values())]))

    # scons decides when to rerun, so drop devinterp's cache rather than let it reject a changed config
    if Path(args.out_samples).exists():
        shutil.rmtree(args.out_samples)
    samples = sample(model=model, dataset=index_dataset(np.arange(cfg['n_sgld']), row_length),
                     observables=observables, loss_fn=make_loss_fn(X, A, Y, B),
                     init_seed=args.seed, obs_seed=args.seed, device=cfg['device'],
                     output_path=args.out_samples, save_metrics=True, **cfg['sgld'])
    step_loss = samples.dataset['sampling_loss_micro'].mean(['batch_sampling', 'token_pos']).values
    logging.info(f"mean SGLD sampling loss per step (chain x step):\n{np.array2string(step_loss, precision=3)}")

    bif = compute_bif(samples, correlation_method=args.correlation_method, loss_keys=list(obs_ranges),
                      batch_index_range_1=obs_ranges['train'], batch_index_range_2=obs_ranges['val'],
                      device=cfg['device'])
    # observables shuffle their patients, so map each BIF row back to its frozen row via input_ids (which covers
    # every observable); drop top-ups
    bif_rows = bif['input_ids'].values[:, 0]
    rows_1, rows_2 = bif_rows[obs_ranges['train']], bif_rows[obs_ranges['val']]
    bif = bif.drop_vars('input_ids').isel(batch_2=rows_2 != empty_row)
    rows_2 = rows_2[rows_2 != empty_row]
    assert set(rows_1) == set(train_rows) and set(rows_2) == set(val_rows)
    patient_ids = np.concatenate([train_data[train_p2i[train_ix, 0], 0], rollouts['patient'].values])
    bif = bif.assign_coords(patient_1=('batch_1', patient_ids[rows_1]), patient_2=('batch_2', patient_ids[rows_2]),
                            train_row=('batch_1', rows_1), val_row=('batch_2', rows_2))
    # inputs and targets of each patient, to read off which events the influences refer to
    for name, tensor in zip(FROZEN_VARS, [X, A, Y, B]):
        values = tensor.cpu().numpy()
        bif[f'{name}_1'] = (('batch_1', 'position'), values[rows_1])
        bif[f'{name}_2'] = (('batch_2', 'position'), values[rows_2])

    if args.rollout_bif:
        assert args.correlation_method == 'token', 'the rollout BIF is per rolled out token'
        rollout_index = {row: (val_rows[v], p) for row, v, p in zip(rows['rollout'], vs, ps)}
        bif['rollout_influences'] = (('batch_1', 'batch_2', 'position', 'rollout_step', 'target_position'),
                                     rollout_influences(samples, bif, obs_ranges, rollout_index, n_rollout,
                                                        cfg['batch_size'], cfg['device']))
    bif.to_zarr(args.out_bif, mode='w')
    logging.info(f"influences {dict(bif['influences'].sizes)} saved to {args.out_bif}")


if __name__ == "__main__":
    main()
