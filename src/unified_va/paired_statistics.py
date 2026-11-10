"""Paired macro-VA reporting and within-target record-cluster bootstrap.

No model inference, seed/recipe selection, file deletion, or data reads at import.
VA RMSE follows this project: sqrt(mean(error_v**2 + error_a**2)), without /2.
Bootstrap intervals condition on the supplied fitted models and fixed targets.
"""
from collections import Counter
import math
import numpy as np


def index_predictions(rows, targets):
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError('unique ordered target registry required')
    expected = set(targets)
    result = {}
    for row in rows:
        key = (row['target'], str(row['id']), int(row['aspect_index']))
        if key[0] not in expected or key in result:
            raise ValueError('unknown target or duplicate target/record/aspect identity')
        values = [float(row[k]) for k in ('v_pred', 'a_pred', 'v_true', 'a_true')]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('nonfinite prediction or label')
        result[key] = values
    if {k[0] for k in result} != expected:
        raise ValueError('incomplete evaluation targets')
    return result


def aligned_errors(pairs, targets, expected_counts=None):
    """Each pair is {seed, S:[prediction rows], T:[prediction rows]}."""
    if not pairs or len({p['seed'] for p in pairs}) != len(pairs):
        raise ValueError('unique seed pairs required')
    reference_keys = None
    reference_gold = None
    errors_s, errors_t = [], []
    for pair in pairs:
        s, t = index_predictions(pair['S'], targets), index_predictions(pair['T'], targets)
        keys = sorted(s)
        if set(s) != set(t):
            raise ValueError('S/T aspect identities differ')
        if reference_keys is not None and keys != reference_keys:
            raise ValueError('seed evaluation populations differ')
        a = np.asarray([s[k] for k in keys], dtype=np.float64)
        b = np.asarray([t[k] for k in keys], dtype=np.float64)
        if not np.array_equal(a[:, 2:], b[:, 2:]):
            raise ValueError('S/T gold labels differ')
        if reference_gold is not None and not np.array_equal(a[:, 2:], reference_gold):
            raise ValueError('gold labels differ between seeds')
        reference_keys, reference_gold = keys, a[:, 2:]
        errors_s.append((a[:, :2] - a[:, 2:]) ** 2)
        errors_t.append((b[:, :2] - b[:, 2:]) ** 2)
    counts = dict(Counter(k[0] for k in reference_keys))
    if expected_counts is not None and counts != expected_counts:
        raise ValueError('cohort counts differ from the formal evaluation manifest')
    return reference_keys, np.stack(errors_s), np.stack(errors_t)


def summarize(pairs, targets, training_targets, expected_counts=None):
    keys, es, et = aligned_errors(pairs, targets, expected_counts)
    if len(training_targets) != len(set(training_targets)) or not set(training_targets) < set(targets):
        raise ValueError('training group must be a proper subset of the evaluation registry')
    groups = {'full': list(targets), 'training_covered': list(training_targets),
              'held_out': [t for t in targets if t not in training_targets]}
    per_target = []
    group_scores = {g: {'S': [], 'T': [], 'T_minus_S': []} for g in groups}
    seed_groups = []
    for i, pair in enumerate(pairs):
        scores = {}
        for target in targets:
            mask = np.asarray([k[0] == target for k in keys])
            smse, tmse = es[i, mask].mean(0), et[i, mask].mean(0)
            sr, tr = float(np.sqrt(smse.sum())), float(np.sqrt(tmse.sum()))
            per_target.append({'seed': pair['seed'], 'target': target, 'aspects': int(mask.sum()),
                'S_rmse_va': sr, 'T_rmse_va': tr, 'T_minus_S': tr - sr,
                'S_rmse_v': float(np.sqrt(smse[0])), 'S_rmse_a': float(np.sqrt(smse[1])),
                'T_rmse_v': float(np.sqrt(tmse[0])), 'T_rmse_a': float(np.sqrt(tmse[1]))})
            scores[target] = (sr, tr)
        for group, members in groups.items():
            sr, tr = np.mean([scores[t] for t in members], axis=0)
            seed_groups.append({'seed': pair['seed'], 'group': group, 'S': float(sr), 'T': float(tr),
                                'T_minus_S': float(tr - sr)})
            for name, value in [('S', sr), ('T', tr), ('T_minus_S', tr - sr)]:
                group_scores[group][name].append(float(value))
    aggregate = []
    for group, values in group_scores.items():
        row = {'group': group, 'seeds': len(pairs)}
        for name, scores in values.items():
            row[name + '_mean'] = float(np.mean(scores))
            row[name + '_sample_std'] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
        aggregate.append(row)
    per_target_aggregate = []
    for target in targets:
        rows = [r for r in per_target if r['target'] == target]
        row = {'target': target, 'seeds': len(rows), 'aspects_per_seed': rows[0]['aspects']}
        for name in ['S_rmse_va', 'T_rmse_va', 'T_minus_S']:
            values = [r[name] for r in rows]
            row[name + '_mean'] = float(np.mean(values))
            row[name + '_sample_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        per_target_aggregate.append(row)
    return {'per_seed_groups': seed_groups, 'per_seed_targets': per_target,
            'group_seed_statistics': aggregate, 'target_seed_statistics': per_target_aggregate,
            'aggregation': 'arithmetic mean of per-target RMSE within each fitted seed; then seed mean and sample SD',
            'prediction_ensemble': False, 'difference_direction': 'T minus S; negative favors T'}


def cluster_target_arrays(keys, es, et, target):
    """All aspects of a target/record stay together, including unequal cluster sizes."""
    positions = [i for i, key in enumerate(keys) if key[0] == target]
    records = sorted({keys[i][1] for i in positions})
    record_index = {record: i for i, record in enumerate(records)}
    inverse = np.asarray([record_index[keys[i][1]] for i in positions])
    counts = np.bincount(inverse, minlength=len(records)).astype(np.float64)
    sums_s, sums_t = [], []
    for seed in range(es.shape[0]):
        sums_s.append(np.bincount(inverse, weights=es[seed, positions].sum(-1), minlength=len(records)))
        sums_t.append(np.bincount(inverse, weights=et[seed, positions].sum(-1), minlength=len(records)))
    return counts, np.stack(sums_s), np.stack(sums_t)


def bootstrap(pairs, targets, training_targets, expected_counts=None, *, resamples=10000, random_seed=20261101, batch_size=128):
    if type(resamples) is not int or resamples < 100 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive batching and at least100 prespecified bootstrap replicates required')
    report = summarize(pairs, targets, training_targets, expected_counts)
    keys, es, et = aligned_errors(pairs, targets, expected_counts)
    groups = {'full': list(targets), 'training_covered': list(training_targets),
              'held_out': [t for t in targets if t not in training_targets]}
    draws = {g: np.zeros((resamples, len(pairs)), dtype=np.float64) for g in groups}
    # Independent target streams make batching a memory choice, not a statistical change.
    streams = np.random.SeedSequence(random_seed).spawn(len(targets))
    for target, stream in zip(targets, streams):
        counts, sums_s, sums_t = cluster_target_arrays(keys, es, et, target)
        rng = np.random.default_rng(stream)
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            sampled = rng.integers(0, len(counts), size=(stop - start, len(counts)))
            denominator = counts[sampled].sum(1)
            sr = np.sqrt(sums_s[:, sampled].sum(-1) / denominator[None, :])
            tr = np.sqrt(sums_t[:, sampled].sum(-1) / denominator[None, :])
            for name, members in groups.items():
                if target in members:
                    draws[name][start:stop] += (tr - sr).T / len(members)
    intervals = []
    for group, values in draws.items():
        for i, pair in enumerate(pairs):
            lo, hi = np.quantile(values[:, i], [.025, .975])
            point = next(r['T_minus_S'] for r in report['per_seed_groups'] if r['group'] == group and r['seed'] == pair['seed'])
            intervals.append({'group': group, 'scope': 'one fixed fitted seed pair', 'seed': pair['seed'],
                              'T_minus_S': point, 'lower_95': float(lo), 'upper_95': float(hi)})
        lo, hi = np.quantile(values.mean(1), [.025, .975])
        point = next(r['T_minus_S_mean'] for r in report['group_seed_statistics'] if r['group'] == group)
        intervals.append({'group': group, 'scope': 'mean of the supplied fixed fitted seed pairs',
                          'seed': None, 'T_minus_S': point, 'lower_95': float(lo), 'upper_95': float(hi)})
    report.update(bootstrap_intervals=intervals, bootstrap_protocol={
        'replicates': resamples, 'random_seed': random_seed, 'resampling_unit': 'target/record; all aspects together',
        'paired': 'same sampled records for S/T and all supplied seeds', 'targets_resampled': False,
        'interval': 'two-sided percentile95', 'training_randomness_captured': False,
        'multiple_comparison_adjustment': False, 'recipe_selection_allowed': False})
    return report
