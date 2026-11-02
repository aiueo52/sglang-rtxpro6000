# SPDX-License-Identifier: Apache-2.0
"""Graph-capturable, shape-only router prediction statistics (no prefetch)."""
import torch
import torch.nn.functional as F

COUNTERS = ('calls', 'dem', 'dem_cov10', 'dem_cov16', 'miss', 'miss_cov10',
            'miss_cov16', 'vol10', 'vol16', 'self_dem', 'self_cov', 'reserved')


def demand(topk_ids):
    # Fixed-size reduction: never compact indices using a data-dependent mask.
    experts = torch.arange(512, device=topk_ids.device)
    return (topk_ids.reshape(-1, 1) == experts).any(dim=0)


def before(runtime, state, topk_ids):
    layer = state.layer_number
    wanted = demand(topk_ids)
    if layer >= 1:
        rank = runtime.pred_rank[layer]
        absent = state.slot_of < 0
        row = runtime.predict_stats[layer]
        row[0].add_(1)
        row[1].add_(wanted.sum())
        row[2].add_((wanted & (rank < 10)).sum())
        row[3].add_((wanted & (rank < 16)).sum())
        row[7].add_((absent & (rank < 10)).sum())
        row[8].add_((absent & (rank < 16)).sum())
    return wanted


def ranked(hidden, weight, active, k=16):
    ids = F.linear(hidden, weight).topk(k, dim=-1).indices
    ranks = torch.arange(k, dtype=torch.int32, device=hidden.device).expand_as(ids)
    ranks = torch.where(active[:, None], ranks, 127)
    out = torch.full((512,), 127, dtype=torch.int32, device=hidden.device)
    out.scatter_reduce_(0, ids.reshape(-1), ranks.reshape(-1), reduce='amin', include_self=True)
    return out


def after(runtime, state, topk_ids, hidden, wanted):
    layer = state.layer_number
    row = runtime.predict_stats[layer]
    if layer >= 1:
        valid = torch.arange(state.k, device=topk_ids.device) < state.n_copy
        # Inactive copy_list rows may be uninitialized, so clamp before indexing.
        experts = state.copy_list[:, 1].long().clamp(0, 511)
        rank = runtime.pred_rank[layer][experts]
        row[4].add_(valid.sum())
        row[5].add_((valid & (rank < 10)).sum())
        row[6].add_((valid & (rank < 16)).sum())
    active = (topk_ids >= 0).any(dim=1)
    own = ranked(hidden, runtime.router_weights[layer], active, k=10)
    row[9].add_(wanted.sum())
    row[10].add_((wanted & (own < 10)).sum())
    if layer <= 46:
        runtime.pred_rank[layer + 1].copy_(ranked(hidden, runtime.router_weights[layer + 1], active))


def report(stats):
    rows = {str(i): dict(zip(COUNTERS, row)) for i, row in enumerate(stats.cpu().tolist())}
    totals = {name: sum(row[name] for row in rows.values()) for name in COUNTERS}

    def metrics(row):
        def ratio(a, b):
            return row[a] / row[b] if row[b] else None
        return {str(r): dict(miss_recall=ratio(f'miss_cov{r}', 'miss'),
                             dem_recall=ratio(f'dem_cov{r}', 'dem'),
                             vol_per_call=ratio(f'vol{r}', 'calls'),
                             self_recall=ratio('self_cov', 'self_dem')) for r in (10, 16)}

    return dict(layers=rows, totals=totals,
                metrics=dict(layers={i: metrics(row) for i, row in rows.items()}, totals=metrics(totals)))
