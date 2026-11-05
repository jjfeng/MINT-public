"""HTML report of Bayesian Influence Functions from train_bayes.py: for each validation patient, its event
history; clicking a validation event shows the histories of its top-K most influential training patient-tokens.
Requires token-level BIF. A patient pair's influence (sidebar and violins) is its strongest token-pair influence."""
import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde

N_GRID = 200
TOKEN_DIMS = ['target_position', 'target_position_T']
# event category (by vocab prefix) -> categorical color slot; anything unlisted is 'Other'
CATEGORIES = {'Vital': '#2a78d6', 'Med': '#eb6834', 'CC': '#1baf7a', 'Lab': '#eda100',
              'Procedure': '#e87ba4', 'Encounter': '#008300', 'Demographic': '#6250d6', 'Other': '#9a9a96'}
PREFIX_CATEGORY = {'Vital': 'Vital', 'Med': 'Med', 'CC': 'CC', 'Lab': 'Lab', 'Procedure': 'Procedure',
                   'Arrival': 'Encounter', 'Acuity': 'Encounter', 'Admit': 'Encounter', 'ICU Start': 'Encounter',
                   'Discharge': 'Encounter', 'Age': 'Demographic', 'Sex': 'Demographic'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bif', required=True, help='zarr directory written by train_bayes.py')
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--template', required=True, help='HTML template with a __DATA__ placeholder')
    parser.add_argument('--rollouts', required=True, help='zarr directory written by eval_model.py')
    parser.add_argument('--top-k', type=int, required=True, help='training patient-tokens shown per validation event')
    parser.add_argument('--out-html', required=True)
    return parser.parse_args()


def token_influence(bif):
    """Token-level influences of shape (val patients, train patients, val target position, train target position).
    Padded positions have constant zero loss, so their correlations are NaN."""
    influences = bif['influences']
    assert influences.dims == ('batch_1', 'batch_2', *TOKEN_DIMS), f'expected token-level BIF, got {influences.dims}'
    # target_position indexes the train (batch_1) targets, target_position_T the val (batch_2) targets
    return influences.transpose('batch_2', 'batch_1', 'target_position_T', 'target_position').values


def category(name):
    matches = [cat for prefix, cat in PREFIX_CATEGORY.items() if name.strip('"').startswith(prefix)]
    return matches[0] if matches else 'Other'


def event(bif, side, i, pos, vocab_names):
    """Target event `pos` of patient i as [minutes since arrival, name, category]."""
    token = bif[f'target_tokens_{side}'].values[i, pos]
    return [float(bif[f'target_ages_{side}'].values[i, pos]), vocab_names[token], category(vocab_names[token])]


def events(bif, side, i, vocab_names):
    """Event sequence of patient i with the target position of each event (None for the first input, which is
    never predicted); padding dropped."""
    first = bif[f'tokens_{side}'].values[i, 0]
    head = [[float(bif[f'ages_{side}'].values[i, 0]), vocab_names[first], category(vocab_names[first]), None]] if first > 0 else []
    positions = np.flatnonzero(bif[f'target_tokens_{side}'].values[i] > 0)
    return head + [event(bif, side, i, p, vocab_names) + [int(p)] for p in positions]


def top_train_tokens(influence_v, top_k):
    """For each val target position, its top_k most influential train events over all train patients and
    positions, as a list of {'train': train row, 'pos': train target position, 'influence'}. influence_v has
    shape (train patients, val position, train position)."""
    n_train, n_val_pos, n_train_pos = influence_v.shape
    flat = influence_v.transpose(1, 0, 2).reshape(n_val_pos, -1)
    matches = {}
    for p in np.flatnonzero(~np.isnan(flat).all(axis=1)):
        # NaNs (padding) sort last
        best = np.argsort(-flat[p])[:top_k]
        assert np.isfinite(flat[p, best]).all(), f'val position {p} has fewer than {top_k} finite token influences'
        ts, qs = np.unravel_index(best, (n_train, n_train_pos))
        matches[int(p)] = [{'train': int(t), 'pos': int(q), 'influence': float(flat[p, b])}
                           for t, q, b in zip(ts, qs, best)]
    return matches


def violin_densities(influence, grid):
    """Per validation patient KDE of its influences over training patients, each scaled to a peak of 1."""
    densities = np.stack([gaussian_kde(row)(grid) for row in influence])
    return densities / densities.max(axis=1, keepdims=True)


def with_matches(val_events, matches, rollout):
    """Attach the top train tokens and the model's rollout to each val event that is a predicted target."""
    assert all(pos in matches for *_, pos in val_events if pos is not None), 'a real val event has no finite influence'
    return [e[:3] + ([matches[e[3]], rollout[e[3]]] if e[3] is not None else [None, None]) for e in val_events]


def rollout_events(rollouts, v, vocab_names, matches):
    """Sampled rollout of val patient v from each target position, as lists of [minutes, name, category, top
    train tokens]; matches are keyed by position * J + step (None when the rollout BIF was not computed)."""
    tokens, ages = rollouts['rollout_tokens'].values[v], rollouts['rollout_ages'].values[v]
    n_rollout = tokens.shape[1]
    return [[[float(a), vocab_names[t], category(vocab_names[t]),
              None if matches is None else matches.get(p * n_rollout + s)]
             for s, (t, a) in enumerate(zip(ts, as_))] for p, (ts, as_) in enumerate(zip(tokens, ages))]


def rollout_matches(bif, v, top_k):
    """Top train tokens for each rolled out event of val patient v, or None without a rollout BIF."""
    if 'rollout_influences' not in bif:
        return None
    influence = bif['rollout_influences'].isel(batch_2=v).transpose(
        'batch_1', 'position', 'rollout_step', 'target_position').values
    n_train, n_pos, n_rollout, n_train_pos = influence.shape
    return top_train_tokens(influence.reshape(n_train, n_pos * n_rollout, n_train_pos), top_k)


def report_data(tokens, bif, rollouts, vocab_names, top_k):
    # patients are ranked by their strongest token-pair influence
    influence = np.nanmax(tokens, axis=(2, 3))
    assert np.isfinite(influence).all(), 'some patient pair has no finite token influence'
    top_train = influence.argmax(axis=1)
    top_bif = influence.max(axis=1)
    median = np.median(influence, axis=1)
    z = (top_bif - influence.mean(axis=1)) / influence.std(axis=1)
    grid = np.linspace(influence.min(), influence.max(), N_GRID)
    densities = violin_densities(influence, grid)
    train = [{'patient': int(bif['patient_1'].values[t]), 'events': events(bif, 1, t, vocab_names)}
             for t in range(influence.shape[1])]
    val = [{'patient': int(bif['patient_2'].values[v]), 'median': float(median[v]), 'z': float(z[v]),
            'top_train': int(top_train[v]), 'top_bif': float(top_bif[v]), 'density': densities[v].round(4).tolist(),
            'events': with_matches(events(bif, 2, v, vocab_names), top_train_tokens(tokens[v], top_k),
                                   rollout_events(rollouts, v, vocab_names, rollout_matches(bif, v, top_k)))}
           for v in range(influence.shape[0])]
    return {'categories': CATEGORIES, 'grid': grid.tolist(), 'train': train, 'val': val}


def main():
    args = parse_args()
    bif = xr.open_zarr(args.bif).load()
    vocab = pd.read_csv(args.vocab)
    # model tokens are vocab indices shifted by one; model token 1 is the inserted "no event" token
    vocab_names = {**dict(zip(vocab['index'] + 1, vocab['name'])), 1: 'No event'}

    # the BIF shuffles the val patients, so put the rollouts in its order
    rollouts = xr.open_zarr(args.rollouts).load().swap_dims(batch='patient').sel(patient=bif['patient_2'].values)
    data = report_data(token_influence(bif), bif, rollouts, vocab_names, args.top_k)
    with open(args.template) as f:
        template = f.read()
    assert '__DATA__' in template, f'{args.template} has no __DATA__ placeholder'
    with open(args.out_html, 'w') as f:
        f.write(template.replace('__DATA__', json.dumps(data)))


if __name__ == "__main__":
    main()
