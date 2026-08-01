from collections import defaultdict

from sklearn.metrics import auc
import numpy as np

from modeling.Models.OrthoInferenceWrapper import parse_pred_key

def evaluate_multiscale_predictions(gsd_keyed_preds, gsd_keyed_actuals, bucket_func, bucket_gsd_x_or_y="x"):
    gsd_pred_buckets = defaultdict(list)
    gsd_actuals_buckets = defaultdict(list)
    gsd_index = 2 if bucket_gsd_x_or_y == "x" else 3
    for gsd_keyed_key in gsd_keyed_actuals:
        key_values = parse_pred_key(gsd_keyed_key)
        gsd = key_values[gsd_index]
        gsd_pred_buckets[gsd].append(gsd_keyed_preds[gsd_keyed_key]["label"])
        gsd_actuals_buckets[gsd].append(gsd_keyed_actuals[gsd_keyed_key])

    metrics = {}
    for gsd in gsd_actuals_buckets:
        metrics[float(gsd)] = bucket_func(gsd_pred_buckets[gsd], gsd_actuals_buckets[gsd])

    return metrics

def AUC_multiscale_metrics(multiscale_metrics, log_space_area=False, normalize=False, auc_strat="avg"):
    gsd_floats = sorted(list(multiscale_metrics.keys()))
    gsd_floats_map = {gsd_float:gsd_float for gsd_float in gsd_floats}

    if log_space_area:
        gsd_floats_map = {gsd_float:np.log10(gsd_float) for gsd_float in gsd_floats}
    if normalize:
        max_gsd = max(gsd_floats)
        min_gsd = min(gsd_floats)
        if max_gsd-min_gsd <= 0:
            min_gsd = 0.0
            max_gsd = 1.0
        for gsd_float in gsd_floats:
            gsd_floats_map[gsd_float] = (gsd_float - min_gsd) / (max_gsd-min_gsd)

    metrics = [float(multiscale_metrics[gsd_float]) for gsd_float in gsd_floats]
    if len(metrics) > 1:
        if "trap" in auc_strat.lower():
            return auc([gsd_floats_map[gsd_float] for gsd_float in gsd_floats], metrics)
        if "avg" in auc_strat.lower():
            return np.mean(metrics)
        raise ValueError("Got auc_strat=" + str(auc_strat) + " but the allowed values are \"avg\" and \"trap\"")
    return 0.0
