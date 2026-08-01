import os
import json
from collections import defaultdict

import numpy as np


def write_metrics_json(metrics, metrics_file_path, verbose=True):
    metrics_dir = os.path.dirname(metrics_file_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
        if verbose:
            print("Created the directory to store the output: " + str(metrics_dir))
    with open(metrics_file_path, "w") as metrics_file:
        metrics_file.write(json.dumps(metrics, indent=4))


class IdAlignedPredsActuals:
    """Pairs predictions and ground-truth actuals - both dicts keyed by an arbitrary id - and
    drops ids that don't appear on both sides, warning once with a count of how many were
    dropped instead of letting a downstream KeyError crash the run on a partial evaluation.

    Iteration/alignment order follows `unaligned_preds`' own (deterministic) key order.
    """
    def __init__(self, unaligned_preds, unaligned_actuals, verbose=True):
        self._unaligned_preds = unaligned_preds
        self._unaligned_actuals = unaligned_actuals
        self._verbose = verbose
        self._aligned_ids, rejected_count = self._align(lambda _: True)
        if self._verbose and rejected_count > 0:
            print("WARNING: Failed to match", rejected_count,
                  "predictions with actual labels. This could be due to running a partial evaluation.")

    def _align(self, predicate):
        valid_ids = []
        rejected_count = 0
        for pred_id in self._unaligned_preds:
            if pred_id in self._unaligned_actuals and predicate(pred_id):
                valid_ids.append(pred_id)
            else:
                rejected_count += 1
        return valid_ids, rejected_count

    def get_aligned_ids(self):
        return self._aligned_ids

    def get_pred(self, aligned_id):
        return self._unaligned_preds[aligned_id]

    def get_actual(self, aligned_id):
        return self._unaligned_actuals[aligned_id]

    def subset_ids(self, predicate):
        subset_ids, _ = self._align(predicate)
        return subset_ids

    def __len__(self):
        return len(self._aligned_ids)


def group_items(items, key_func):
    """Partition `items` into {key_func(item): [items...]}, preserving first-seen key order.

    This is the single place per-group ("bucket by a key") bucketing lives; every evaluate_* per-ortho
    / per-<field> breakdown routes through it and supplies its own key / metric functors.
    """
    groups = defaultdict(list)
    for item in items:
        groups[key_func(item)].append(item)
    return groups


def compute_metrics_per_group(items, group_key_func, metrics_func):
    """Bucket `items` by `group_key_func` and run `metrics_func` on each group's item list, returning
    {group_key: metrics_func(items_in_group)}. The shared per-ortho / per-<field> breakdown used by
    evaluate_BDA and evaluate_RDA - they differ only in the two functors passed in. (Empty groups
    can't occur, so no explicit skip is needed.)"""
    return {key: metrics_func(group) for key, group in group_items(items, group_key_func).items()}


def compute_per_sample_metric_bundle(aligned_ids,
                                     metric_func,
                                     primary_group_key_func,
                                     primary_group_universe,
                                     secondary_group_key_func,
                                     ignore_primary_groups=None,
                                     ignored_label_suffix="(Filtered)"):
    """Computes a per-id metric, then reports Average/Median overall, Average/Median excluding
    ids whose primary group is in `ignore_primary_groups` (labeled with `ignored_label_suffix`),
    and Average/Median broken down by both the primary group (pre-seeded with every key in
    `primary_group_universe`, so groups with zero samples still appear) and the secondary group
    (only groups that actually occur).
    """
    ignore_primary_groups = ignore_primary_groups or set()
    scored = [(aligned_id, metric_func(aligned_id)) for aligned_id in aligned_ids]

    grouped_primary = group_items(scored, lambda pair: primary_group_key_func(pair[0]))
    # Pre-seed and order by the universe so zero-sample classes still appear in a stable order.
    metric_by_primary_group = {key: grouped_primary.get(key, []) for key in primary_group_universe}
    metric_by_secondary_group = group_items(scored, lambda pair: secondary_group_key_func(pair[0]))

    metric_values = [value for _, value in scored]
    metric_values_filtered = [value for aligned_id, value in scored
                              if primary_group_key_func(aligned_id) not in ignore_primary_groups]

    def group_stat(grouped, stat_func):
        return {key: stat_func([value for _, value in pairs]) for key, pairs in grouped.items()}

    return {
        "Average": np.mean(metric_values),
        f"Average {ignored_label_suffix}": np.mean(metric_values_filtered),
        "Average By Class": group_stat(metric_by_primary_group, np.mean),
        "Average By File": group_stat(metric_by_secondary_group, np.mean),
        "Median": np.median(metric_values),
        f"Median {ignored_label_suffix}": np.median(metric_values_filtered),
        "Median By Class": group_stat(metric_by_primary_group, np.median),
        "Median By File": group_stat(metric_by_secondary_group, np.median),
    }
