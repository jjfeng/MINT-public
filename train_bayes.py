"""Compute Bayesian Influence Functions (BIF) of training patients on validation patients for a trained
MINT checkpoint, using SGLD sampling from devinterp.

devinterp expects fixed-length `input_ids` and a loss `(model, input_ids) -> (batch, seq-1)`. Delphi needs
tokens and ages, so every patient's batch (X, A, Y, B) is built once and frozen, and each patient's
`input_ids` row is just its row index into those frozen tensors, repeated to length block_size + 1.
"""
import argparse
import json
import logging

import numpy as np
import torch
import xarray as xr
from datasets import Dataset
from devinterp.slt.bif import compute_bif
from devinterp.slt.sampling import sample

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch, load_demographic_token_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON file with SGLD and patient-count settings')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--train-npy', required=True)
    parser.add_argument('--val-npy', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--correlation-method', required=True, choices=['token', 'sequence'])
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


def index_dataset(rows, block_size):
    return Dataset.from_dict({'input_ids': np.repeat(rows[:, None], block_size + 1, axis=1)}).with_format('torch')


def make_loss_fn(X, A, Y, B):
    def loss_fn(model, input_ids):
        rows = input_ids[:, 0]
        return model.per_token_ce(X[rows], A[rows], Y[rows], B[rows])
    return loss_fn


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    logging.basicConfig(filename=args.log, filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, train_cfg = load_model(args.ckpt, cfg['device'])
    block_size = train_cfg['model']['block_size']
    demographic_token_ids = load_demographic_token_ids(args.vocab)

    train_data = np.load(args.train_npy).astype(np.uint32)
    val_data = np.load(args.val_npy).astype(np.uint32)
    train_p2i, val_p2i = get_p2i(train_data), get_p2i(val_data)
    # SGLD posterior uses n_sgld training patients; the influencing patients are the first n_train_obs of them
    train_ix = rng.choice(len(train_p2i), cfg['n_sgld'], replace=False)
    val_ix = rng.choice(len(val_p2i), cfg['n_val_obs'], replace=False)
    train_batch = frozen_batch(train_data, train_p2i, train_ix, train_cfg, demographic_token_ids)
    val_batch = frozen_batch(val_data, val_p2i, val_ix, train_cfg, demographic_token_ids)
    X, A, Y, B = [torch.cat([t, v]).to(cfg['device']) for t, v in zip(train_batch, val_batch)]
    assert X.shape == (cfg['n_sgld'] + cfg['n_val_obs'], block_size), X.shape

    # rows of the frozen tensors: [0, n_sgld) are training patients, [n_sgld, n_sgld + n_val_obs) validation
    train_rows = np.arange(cfg['n_train_obs'])
    val_rows = cfg['n_sgld'] + np.arange(cfg['n_val_obs'])
    for n in (cfg['n_train_obs'], cfg['n_val_obs']):
        assert n % cfg['batch_size'] == 0, 'observables are evaluated in full batches'
    observables = {name: (index_dataset(rows, block_size), len(rows) // cfg['batch_size'])
                   for name, rows in [('train', train_rows), ('val', val_rows)]}

    samples = sample(model=model, dataset=index_dataset(np.arange(cfg['n_sgld']), block_size),
                     observables=observables, loss_fn=make_loss_fn(X, A, Y, B),
                     init_seed=args.seed, obs_seed=args.seed, device=cfg['device'],
                     output_path=args.out_samples, save_metrics=True, **cfg['sgld'])
    step_loss = samples.dataset['sampling_loss_micro'].mean(['batch_sampling', 'token_pos']).values
    logging.info(f"mean SGLD sampling loss per step (chain x step):\n{np.array2string(step_loss, precision=3)}")

    n_train = cfg['n_train_obs']
    bif = compute_bif(samples, correlation_method=args.correlation_method, loss_keys=['train', 'val'],
                      batch_index_range_1=range(n_train), batch_index_range_2=range(n_train, n_train + cfg['n_val_obs']),
                      device=cfg['device'])

    # observables shuffle their patients, so map each BIF row back to its frozen row via input_ids
    rows = bif['input_ids'].values[:, 0]
    rows_1, rows_2 = rows[:n_train], rows[n_train:]
    assert set(rows_1) == set(train_rows) and set(rows_2) == set(val_rows)
    patient_ids = np.concatenate([train_data[train_p2i[train_ix, 0], 0], val_data[val_p2i[val_ix, 0], 0]])
    bif = bif.drop_vars('input_ids').assign_coords(
        patient_1=('batch_1', patient_ids[rows_1]), patient_2=('batch_2', patient_ids[rows_2]))
    # inputs and targets of each patient, to read off which events the influences refer to
    for name, tensor in zip(['tokens', 'ages', 'target_tokens', 'target_ages'], [X, A, Y, B]):
        values = tensor.cpu().numpy()
        bif[f'{name}_1'] = (('batch_1', 'position'), values[rows_1])
        bif[f'{name}_2'] = (('batch_2', 'position'), values[rows_2])
    bif.to_zarr(args.out_bif, mode='w')
    logging.info(f"influences {dict(bif['influences'].sizes)} saved to {args.out_bif}")


if __name__ == "__main__":
    main()
