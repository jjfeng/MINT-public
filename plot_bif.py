"""HTML report of Bayesian Influence Functions from train_bayes.py: for each validation patient, its event
history next to the histories of its most influential training patients. Requires token-level BIF: each
validation event shows its most influential training event, and patient pairs are ranked by their strongest
token-pair influence."""
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

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BIF report</title>
<style>
  :root { --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --border: #e6e5e1; }
  body { margin: 0; font: 13px/1.4 system-ui, sans-serif; color: var(--text-primary); background: var(--surface);
         display: grid; grid-template-columns: 330px 1fr; height: 100vh; }
  #sidebar { overflow-y: auto; border-right: 1px solid var(--border); }
  #main { overflow: auto; padding: 16px 20px; }
  h1 { font-size: 15px; margin: 12px; }
  .note { color: var(--text-secondary); margin: 0 12px 10px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--border); }
  th { position: sticky; top: 0; background: var(--surface); cursor: pointer; color: var(--text-secondary); }
  th:first-child, td:first-child { text-align: left; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: #f0efec; }
  tbody tr.selected { background: #cde2fb; }
  #legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 14px; }
  #legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                         margin-right: 5px; background: var(--c); vertical-align: -1px; }
  #columns { display: flex; gap: 14px; align-items: flex-start; }
  .patient { flex: 0 0 290px; border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .patient.val { border: 2px solid var(--text-primary); }
  .patient h2 { font-size: 13px; margin: 0 0 2px; }
  .patient .meta { color: var(--text-secondary); margin-bottom: 8px; }
  .event { display: grid; grid-template-columns: 46px 1fr auto; gap: 6px; padding: 2px 4px 2px 7px; margin: 2px 0;
           border-left: 4px solid var(--c); border-radius: 2px; }
  .event .age, .event .side { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .event.clickable { cursor: pointer; }
  .event.clickable:hover { background: #f0efec; }
  .event.selected, .event.highlight { background: #cde2fb; font-weight: 600; }
  .patient.match { border: 2px solid #2a78d6; max-height: 80vh; overflow-y: auto; }
  .patient.placeholder { border-style: dashed; }
  #tabs { display: flex; gap: 4px; margin-bottom: 14px; border-bottom: 1px solid var(--border); }
  #tabs button { font: inherit; border: 0; background: none; padding: 6px 12px; cursor: pointer;
                 color: var(--text-secondary); border-bottom: 2px solid transparent; }
  #tabs button.active { color: var(--text-primary); border-bottom-color: var(--text-primary); }
  #violins text { font-size: 11px; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }
  #violins .row:hover .body { fill: #b3b2ad; }
  #tooltip { position: fixed; pointer-events: none; background: #fff; border: 1px solid var(--border);
             border-radius: 4px; padding: 6px 8px; box-shadow: 0 2px 6px rgba(0,0,0,.12); font-variant-numeric: tabular-nums; }
</style></head>
<body>
<div id="sidebar">
  <h1>Validation patients</h1>
  <p class="note">BIF = posterior correlation of losses. A patient pair's BIF is its strongest token-pair BIF.
  z = how far the top training patient sits above
  that validation patient's influences over all training patients. Click a header to sort.</p>
  <table><thead><tr><th data-key="patient">patient</th><th data-key="top_bif">top BIF</th>
  <th data-key="median">median</th><th data-key="z">z</th></tr></thead><tbody id="rows"></tbody></table>
</div>
<div id="main">
  <div id="tabs"><button data-tab="patients" class="active">Patient comparison</button><button data-tab="overview">Overview</button></div>
  <div id="patients" class="pane"><div id="legend"></div><div id="columns"></div></div>
  <div id="overview" class="pane" hidden>
    <p class="note" style="margin-left:0">Each violin is one validation patient's BIF over all training patients (tick = median,
    dot = top training patient). Rows follow the sidebar order. Click a row to open that patient.</p>
    <svg id="violins"></svg>
  </div>
  <div id="tooltip" hidden></div>
</div>
<script>
const DATA = __DATA__;
const fmt = x => x.toFixed(3);

document.getElementById('legend').innerHTML = Object.entries(DATA.categories)
  .map(([name, color]) => `<span style="--c:${color}">${name}</span>`).join('');

// one event row; `extra` holds attributes, `side` text shown at the right
function eventRow([age, name, cat], cls, extra, side) {
  return `<div class="event ${cls}" style="--c:${DATA.categories[cat]}" ${extra}><span class="age">${age.toFixed(2)}y</span>
    <span>${name}</span><span class="side">${side}</span></div>`;
}

function card(title, meta, rows, cls) {
  return `<div class="patient ${cls}"><h2>${title}</h2><div class="meta">${meta}</div>${rows.join('')}</div>`;
}

// training patient card; the event at target position `highlightPos` (if any) is highlighted
function trainCard(t, title, meta, highlightPos) {
  const rows = DATA.train[t].events.map(e => eventRow(e, e[3] === highlightPos ? 'highlight' : '', '', ''));
  return card(title, meta, rows, highlightPos === undefined ? '' : 'match');
}

function show(i, selected) {
  const v = DATA.val[i];
  document.querySelectorAll('#rows tr').forEach(tr => tr.classList.toggle('selected', +tr.dataset.i === i));
  const valRows = v.events.map((e, k) => e[3]
    ? eventRow(e, `clickable${k === selected ? ' selected' : ''}`, `data-k="${k}"`, fmt(e[3].influence))
    : eventRow(e, '', '', ''));
  const val = card(`Validation patient ${v.patient}`,
    'Click an event to see its most influential training token (BIF at right).', valRows, 'val');
  const m = selected === undefined ? null : v.events[selected][3];
  const match = m
    ? trainCard(m.train, `Top token: training patient ${DATA.train[m.train].patient}`,
        `BIF ${fmt(m.influence)} on "${v.events[selected][1]}"`, m.pos)
    : card('Top influential token', 'Click a validation event to show it here.', [], 'match placeholder');
  const top = v.top.map((t, r) =>
    trainCard(t.train, `#${r + 1} training patient ${DATA.train[t.train].patient}`, `max token-pair BIF ${fmt(t.influence)}`));
  document.getElementById('columns').innerHTML = val + match + top.join('');
  document.querySelectorAll('.event.clickable').forEach(el => el.onclick = () => show(i, +el.dataset.k));
  document.querySelector('.event.highlight')?.scrollIntoView({block: 'nearest'});
}

function setTab(name) {
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.pane').forEach(p => p.hidden = p.id !== name);
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => setTab(b.dataset.tab));

function renderViolins(order) {
  const rowH = 26, left = 70, right = 20, top = 10, axisH = 34, width = 900;
  const grid = DATA.grid, lo = grid[0], hi = grid[grid.length - 1];
  const x = b => left + (b - lo) / (hi - lo) * (width - left - right);
  const height = top + order.length * rowH + axisH;
  const ticks = Array.from({length: 6}, (_, k) => lo + k * (hi - lo) / 5);
  const axis = ticks.map(t => `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${height - axisH}" stroke="#e6e5e1"/>
    <text x="${x(t)}" y="${height - axisH + 14}" text-anchor="middle">${t.toFixed(2)}</text>`).join('')
    + `<text x="${(left + width - right) / 2}" y="${height - 4}" text-anchor="middle">BIF over all training patients</text>`;
  const rows = order.map((i, r) => {
    const v = DATA.val[i], cy = top + (r + 0.5) * rowH, half = rowH * 0.42;
    const upper = grid.map((b, k) => `${x(b)},${cy - v.density[k] * half}`);
    const lower = grid.map((b, k) => `${x(b)},${cy + v.density[k] * half}`).reverse();
    return `<g class="row" data-i="${i}" style="cursor:pointer">
      <rect x="0" y="${cy - rowH / 2}" width="${width}" height="${rowH}" fill="transparent"/>
      <polygon class="body" points="${upper.concat(lower).join(' ')}" fill="#c9c8c3"/>
      <line x1="${x(v.median)}" x2="${x(v.median)}" y1="${cy - half}" y2="${cy + half}" stroke="#6b6b66" stroke-width="1.5"/>
      <circle cx="${x(v.top_bif)}" cy="${cy}" r="4.5" fill="#2a78d6" stroke="#fcfcfb" stroke-width="2"/>
      <text x="${left - 8}" y="${cy + 4}" text-anchor="end">${v.patient}</text></g>`;
  }).join('');
  const svg = document.getElementById('violins');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.innerHTML = axis + rows;
  const tip = document.getElementById('tooltip');
  svg.querySelectorAll('.row').forEach(g => {
    const v = DATA.val[+g.dataset.i];
    g.onmousemove = e => {
      tip.hidden = false;
      tip.style.left = `${e.clientX + 12}px`;
      tip.style.top = `${e.clientY + 12}px`;
      tip.innerHTML = `<b>Validation patient ${v.patient}</b><br>top BIF ${fmt(v.top_bif)} (training patient ${DATA.train[v.top[0].train].patient})<br>median ${fmt(v.median)}<br>z ${v.z.toFixed(1)}`;
    };
    g.onmouseleave = () => { tip.hidden = true; };
    g.onclick = () => { tip.hidden = true; show(+g.dataset.i); setTab('patients'); };
  });
}

let sortKey = 'z', ascending = false;
function renderRows() {
  const order = DATA.val.map((v, i) => i).sort((a, b) => {
    const d = DATA.val[a][sortKey] - DATA.val[b][sortKey];
    return ascending ? d : -d;
  });
  renderViolins(order);
  document.getElementById('rows').innerHTML = order.map(i => {
    const v = DATA.val[i];
    return `<tr data-i="${i}"><td>${v.patient}</td><td>${fmt(v.top[0].influence)}</td><td>${fmt(v.median)}</td><td>${v.z.toFixed(1)}</td></tr>`;
  }).join('');
  document.querySelectorAll('#rows tr').forEach(tr => tr.onclick = () => show(+tr.dataset.i));
}
document.querySelectorAll('th').forEach(th => th.onclick = () => {
  ascending = sortKey === th.dataset.key ? !ascending : false;
  sortKey = th.dataset.key;
  renderRows();
});
DATA.val.forEach(v => { v.top_bif = v.top[0].influence; });
renderRows();
show(0);
</script>
</body></html>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bif', required=True, help='zarr directory written by train_bayes.py')
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--top-k', type=int, required=True)
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
    """Target event `pos` of patient i as [age in years, name, category]."""
    token = bif[f'target_tokens_{side}'].values[i, pos]
    return [float(bif[f'target_ages_{side}'].values[i, pos] / 365.25), vocab_names[token], category(vocab_names[token])]


def events(bif, side, i, vocab_names):
    """Event sequence of patient i with the target position of each event (None for the first input, which is
    never predicted); padding dropped."""
    first = bif[f'tokens_{side}'].values[i, 0]
    head = [[float(bif[f'ages_{side}'].values[i, 0] / 365.25), vocab_names[first], category(vocab_names[first]), None]] if first > 0 else []
    positions = np.flatnonzero(bif[f'target_tokens_{side}'].values[i] > 0)
    return head + [event(bif, side, i, p, vocab_names) + [int(p)] for p in positions]


def top_train_token(influence_v):
    """For each val target position, the most influential train event over all train patients and positions:
    {'train': train row, 'pos': train target position, 'influence'}. influence_v has shape
    (train patients, val position, train position)."""
    n_train, n_val_pos, n_train_pos = influence_v.shape
    flat = influence_v.transpose(1, 0, 2).reshape(n_val_pos, -1)
    matches = {}
    for p in np.flatnonzero(~np.isnan(flat).all(axis=1)):
        best = np.nanargmax(flat[p])
        t, q = np.unravel_index(best, (n_train, n_train_pos))
        matches[int(p)] = {'train': int(t), 'pos': int(q), 'influence': float(flat[p, best])}
    return matches


def violin_densities(influence, grid):
    """Per validation patient KDE of its influences over training patients, each scaled to a peak of 1."""
    densities = np.stack([gaussian_kde(row)(grid) for row in influence])
    return densities / densities.max(axis=1, keepdims=True)


def with_matches(val_events, matches):
    """Attach the top train token to each val event that is a predicted target."""
    assert all(pos in matches for *_, pos in val_events if pos is not None), 'a real val event has no finite influence'
    return [e[:3] + [matches[e[3]] if e[3] is not None else None] for e in val_events]


def report_data(tokens, bif, vocab_names, top_k):
    # patients are ranked by their strongest token-pair influence
    influence = np.nanmax(tokens, axis=(2, 3))
    assert np.isfinite(influence).all(), 'some patient pair has no finite token influence'
    order = np.argsort(-influence, axis=1)[:, :top_k]
    median = np.median(influence, axis=1)
    z = (influence[np.arange(len(order)), order[:, 0]] - influence.mean(axis=1)) / influence.std(axis=1)
    grid = np.linspace(influence.min(), influence.max(), N_GRID)
    densities = violin_densities(influence, grid)
    train = [{'patient': int(bif['patient_1'].values[t]), 'events': events(bif, 1, t, vocab_names)}
             for t in range(influence.shape[1])]
    val = [{'patient': int(bif['patient_2'].values[v]), 'median': float(median[v]), 'z': float(z[v]),
            'density': densities[v].round(4).tolist(),
            'events': with_matches(events(bif, 2, v, vocab_names), top_train_token(tokens[v])),
            'top': [{'train': int(t), 'influence': float(influence[v, t])} for t in order[v]]}
           for v in range(influence.shape[0])]
    return {'categories': CATEGORIES, 'grid': grid.tolist(), 'train': train, 'val': val}


def main():
    args = parse_args()
    bif = xr.open_zarr(args.bif).load()
    vocab = pd.read_csv(args.vocab)
    # model tokens are vocab indices shifted by one; model token 1 is the inserted "no event" token
    vocab_names = {**dict(zip(vocab['index'] + 1, vocab['name'])), 1: 'No event'}

    data = report_data(token_influence(bif), bif, vocab_names, args.top_k)
    with open(args.out_html, 'w') as f:
        f.write(HTML_TEMPLATE.replace('__DATA__', json.dumps(data)))


if __name__ == "__main__":
    main()
