import copy
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample

from modeling.utils.hyperparameters import parse_hyperparameters, add_hyperparameters_files_to_parse_args
from modeling.utils.evaluation_utils import write_metrics_json, compute_metrics_per_group
from modeling.utils.confidence_utils import confidence_scalar
from modeling.Models.OrthoInferenceWrapper import joint_file_pred_key, parse_pred_key
from modeling.Orthomosaic import MultisourceOrthomosaicFactory


def override_preds_with_random_baseline(preds, class_labels):
    # Random-performance reference, mirroring evaluate_RDA.py's --random_baseline: replace every
    # prediction's label with a uniform random class so the same metric machinery reports what chance
    # scores on this exact aligned set. For a two-class macro F1 this sits at ~0.5 regardless of class
    # balance -- the honest "no-skill" bar (the always-majority floor is lower and flatters real
    # models). class_preds is reset to a one-hot on the drawn label so confidence/AUC stay consistent.
    randomized = {}
    for pred_id, pred in preds.items():
        drawn = np.random.choice(class_labels)
        new_pred = dict(pred)
        new_pred["label"] = drawn
        new_pred["class_preds"] = {label: (1.0 if label == drawn else 0.0) for label in class_labels}
        randomized[pred_id] = new_pred
    return randomized


def restrict_actuals_to_event_phase(actuals, orthomosaic_stats, phase):
    # Keep only ground-truth entries whose anchor orthomosaic matches the requested event phase
    # ("pre" or "post") per the statistics.csv "Pre/Post Event" column. The CHANGE task anchors its
    # sample windows on satellite orthomosaics, many of which are pre-event baselines captured months
    # or years before the disaster; a model given a pre-event tile cannot see damage that only exists
    # post-event, so evaluating on those anchors measures priors, not change detection. Restricting to
    # "post" scores the model only where the anchor imagery actually carries a change signal. Alignment
    # is an inner join on the actuals, so dropping pre-event anchors here drops their predictions too.
    if phase == "all":
        return actuals
    want = phase.upper()
    kept = {}
    dropped_orthos = set()
    for actual_key, label in actuals.items():
        ortho_name, _, _, _ = parse_pred_key(actual_key)
        event_phase = str(orthomosaic_stats.loc[ortho_name]["Pre/Post Event"]).strip().upper()
        if event_phase == want:
            kept[actual_key] = label
        else:
            dropped_orthos.add(ortho_name)
    print(f"\tRestricting to {want}-event anchors: kept {len(kept)} of {len(actuals)} ground-truth buildings "
          f"({len(dropped_orthos)} anchor orthomosaics dropped).")
    return kept


def parse_reportable_metrics(metrics_payload, metric_paths_str):
    result = []
    for path in metric_paths_str.split(","):
        result.append(parse_reportable_metric(metrics_payload, path))
    return result

def parse_reportable_metric(metrics_payload, metric_path):
    metric_tmp = copy.deepcopy(metrics_payload)
    result = {}
    metric_path_aug = copy.deepcopy(metric_path)
    metric_keys = metric_path.split(">")
    for i, key in enumerate(metric_keys):
        if key == "*":
            sub_result = {}
            for matched_key in metric_tmp.keys():
                sub_keys = ">".join(metric_keys[i:]).replace("*", matched_key, 1)
                sub_result = sub_result | parse_reportable_metric(metric_tmp, sub_keys)
            metric_tmp = sub_result
            metric_path_aug = metric_path.split("*")[0]
            break
        # Requested breakdowns can legitimately be absent - e.g. --lightweight skips the
        # per_gsd/per_* metrics, so the default metrics_to_report's "per_gsd_metrics>..." path
        # has nothing to walk into. Reporting is best-effort console output, so skip a path whose
        # key isn't present instead of failing the whole evaluation with a KeyError.
        if not isinstance(metric_tmp, dict) or key not in metric_tmp:
            return {}
        metric_tmp = metric_tmp[key]
    result[metric_path_aug] = metric_tmp
    return result


def describe_buildings_for_change(orthomosaics, skip_same_source=True, change_decision_rule="all",
                                  anchor_source=None, anchor_event_phase=None):
    # Reconstruct CHANGE ground truth with the same decision rule the training annotator uses
    # (BuildingChangeSampleAnnotator): under "all" a building is CHANGED when every one of its
    # correlated parallel views (in other orthomosaics) disagrees with its label, under "any" a
    # single disagreeing view is enough; UNCHANGED otherwise.
    #
    # Returns the full description per building rather than only the CHANGE label, because the
    # damage class the building carries in its OWN view is what the class-conditional baselines
    # (see class_conditional_change_rates / class_oracle_scores) are built from, and relabelling
    # throws it away. relabel_buildings_for_change projects this down to just the label.
    #
    # anchor_source / anchor_event_phase optionally restrict which orthomosaics may anchor a
    # building. Callers scoring the CHANGE task pass Satellite + POST, matching
    # initialize_windowed_dataset: a pre-event anchor carries no damage signal, so anchoring on one
    # measures priors rather than observable change.
    if change_decision_rule not in ("all", "any"):
        raise ValueError("Unknown change_decision_rule " + str(change_decision_rule) + ". Options are ['all', 'any']")
    rule = all if change_decision_rule == "all" else any
    described = {}
    for ortho in orthomosaics:
        building_source = ortho.get_source_used_for_collection()
        if anchor_source is not None and building_source != anchor_source:
            continue
        if anchor_event_phase is not None and ortho.get_event_phase() != anchor_event_phase:
            continue
        for building in ortho.get_buildings() or []:
            correlation_id = building.getCorrelationId()
            parallel_views = []
            for comp_ortho in orthomosaics:
                if comp_ortho.get_name() == ortho.get_name():
                    continue
                # By default two views collected by the same source are not treated as parallel views.
                if skip_same_source and building_source == comp_ortho.get_source_used_for_collection():
                    continue
                try:
                    buildings = comp_ortho.get_buildings(correlation_ids=[correlation_id])
                    if buildings[0].getCorrelationId() == correlation_id:
                        parallel_views.extend(buildings)
                except (KeyError, TypeError):
                    # This comp_ortho has no building with that correlation id (KeyError) or holds
                    # no buildings at all (TypeError on the None return), so it contributes no view.
                    pass

            if len(parallel_views) > 0:
                label_changed_in_parallel_views = rule(v.getLabel() != building.getLabel() for v in parallel_views)
                described[joint_file_pred_key(ortho.get_name(), building.getId())] = {
                    "label": "CHANGED" if label_changed_in_parallel_views else "UNCHANGED",
                    "orthomosaic": ortho.get_name(),
                    "self_label": building.getLabel(),
                    "parallel_labels": [v.getLabel() for v in parallel_views],
                }
    return described

def relabel_buildings_for_change(orthomosaics, skip_same_source=True, change_decision_rule="all"):
    # The CHANGE label alone, keyed as the evaluator's actuals expect. See
    # describe_buildings_for_change for the pairing rule and for the richer per-building record.
    return {key: record["label"] for key, record
            in describe_buildings_for_change(orthomosaics, skip_same_source, change_decision_rule).items()}

def parse_model_name_and_predicted_labels(preds_path):
    with open(preds_path, "r") as f:
        preds_data = json.load(f)
        return preds_data["model_name"], preds_data["preds"]

def get_label_indicator_dict(label, labels):
    result = {l:0 for l in labels}
    result[label] = 1.0
    return result

class AlignedPredsActuals:
    def __init__(self, unaligned_preds, unaligned_actuals, orthomosaic_stats, channel_map, verbose=True):
        # Store the passed objects
        self._unaligned_preds = unaligned_preds
        self._unaligned_actuals = unaligned_actuals
        self._orthomosaic_stats = orthomosaic_stats
        self._channel_map = channel_map
        self._verbose = verbose
        self._deterministic_ordering, rejected_preds_count = self._get_aligned_pred_ids_on_field_value(None, None)
        if self._verbose and rejected_preds_count > 0:
            print("WARNING: Failed to match", rejected_preds_count, "predictions with actual labels. This could be due to running a partial evaluation.")

    def get_labels(self):
        result = list(self._channel_map["channel_maps"]["output_class_2_idx_map"].keys())
        result.remove("background")
        return result
    def get_preds_multilabel_dict(self):
        return self._get_fixed_ordered_data(self._unaligned_preds, lambda x:get_label_indicator_dict(x["label"], self.get_labels()))
    def get_actuals_multilabel_dict(self):
        return self._get_fixed_ordered_data(self._unaligned_actuals, lambda x:get_label_indicator_dict(x, self.get_labels()))
    def get_preds_labels(self):
        return self._get_fixed_ordered_data(self._unaligned_preds, lambda x:x["label"])
    def get_preds_confidences(self):
        def pred_confidence(x):
            # The normalized per-class probabilities are the calibratable confidence source;
            # fall back to the serialized confidence field for older prediction files.
            if "class_preds" in x:
                return x["class_preds"].get(x["label"], 0.0)
            return confidence_scalar(x.get("confidence", 0.0), x["label"])
        return self._get_fixed_ordered_data(self._unaligned_preds, pred_confidence)
    def get_actuals_labels(self):
        return self._get_fixed_ordered_data(self._unaligned_actuals, lambda x:x)
    def get_class_probabilities(self, label):
        # Per-class probability, renormalized over the real classes (see change_probability). Falls
        # back to the argmax indicator only for prediction files written before class_preds existed,
        # where no probability was serialized -- those cannot support a ranking metric.
        labels = self.get_labels()
        def probability(pred):
            if "class_preds" not in pred:
                return float(pred["label"] == label)
            total = sum(pred["class_preds"].get(l, 0.0) for l in labels)
            return pred["class_preds"].get(label, 0.0) / total if total > 0 else 1.0 / len(labels)
        return self._get_fixed_ordered_data(self._unaligned_preds, probability)
    def get_change_scores(self):
        # P(CHANGED) per row -- the score every AUC in the paper ranks by.
        return np.array(self.get_class_probabilities(CHANGE_POSITIVE_LABEL))
    def get_change_actuals(self):
        # 1 for CHANGED, 0 otherwise, aligned with get_change_scores.
        return np.array([1 if label == CHANGE_POSITIVE_LABEL else 0
                         for label in self.get_actuals_labels()])
    def get_pred_keys(self):
        return list(self._deterministic_ordering)
    def get_orthomosaic_codes(self):
        # Small integer code per row identifying its orthomosaic; see within_ortho_auc.
        names = [parse_pred_key(pred_id)[0] for pred_id in self._deterministic_ordering]
        codes = {name: index for index, name in enumerate(sorted(set(names)))}
        return np.array([codes[name] for name in names]), len(codes)
    def get_building_ids(self):
        return building_ids_from_pred_keys(self._deterministic_ordering)
    def _get_fixed_ordered_data(self, data, func):
        result = []
        for pred_id in self._deterministic_ordering:
            valid = False
            try:
                result.append(func(data[pred_id]))
                valid = True
            except KeyError:
                pass
            try:
                if not valid:
                    # Predictions are keyed with a GSD prefix but actuals are not - retry the
                    # lookup with the GSD stripped from the key.
                    ortho_name, building_id, _, _ = parse_pred_key(pred_id)
                    general_key = joint_file_pred_key(ortho_name, building_id)
                    result.append(func(data[general_key]))
                    valid = True
            except KeyError:
                pass
            if self._verbose and not valid:
                print("WARNING: Found prediction for a building that does not appear in actuals.")
        return result
    def _get_subset_dict(self, data, pred_ids, gsd_pred_ids=True):
        if gsd_pred_ids:
            return {pred_id:data[pred_id] for pred_id in pred_ids}
        pred_ids_without_gsds = []
        for pred_id in pred_ids:
            ortho_name, building_id, _, _ = parse_pred_key(pred_id)
            pred_ids_without_gsds.append(joint_file_pred_key(ortho_name, building_id))
        return {pred_id:data[pred_id] for pred_id in pred_ids_without_gsds}

    def _get_aligned_pred_ids_on_field_value(self, field_name, field_value):
        valid_pred_ids = []
        rejected_preds_count = 0
        found_field_value = None
        for pred_id in self._unaligned_preds:
            ortho_name, building_id, gsd_x, gsd_y = parse_pred_key(pred_id)
            if field_name == "gsd_x":
                found_field_value = gsd_x
            elif field_name == "gsd_y":
                found_field_value = gsd_y
            elif not field_name is None:
                found_field_value = self._orthomosaic_stats.loc[ortho_name][field_name]
            valid_field = field_name is None or field_value is None or found_field_value == field_value
            valid_actual = joint_file_pred_key(ortho_name, building_id) in self._unaligned_actuals
            if valid_field and valid_actual:
                valid_pred_ids.append(pred_id)
            else:
                rejected_preds_count += 1
        return valid_pred_ids, rejected_preds_count
    def field_value(self, pred_id, field):
        # The value of `field` for a single prediction id (an ortho-stats column, or a parsed GSD).
        ortho_name, _, gsd_x, gsd_y = parse_pred_key(pred_id)
        if field == "gsd_x":
            return gsd_x
        if field == "gsd_y":
            return gsd_y
        return self._orthomosaic_stats.loc[ortho_name][field]
    def get_predicted_values(self, field):
        return list({self.field_value(pred_id, field) for pred_id in self._unaligned_preds})
    def get_aligned_pred_ids(self):
        return self._deterministic_ordering
    def subset_by_pred_ids(self, pred_ids):
        subset_preds = self._get_subset_dict(self._unaligned_preds, pred_ids, gsd_pred_ids=True)
        subset_actuals = self._get_subset_dict(self._unaligned_actuals, pred_ids, gsd_pred_ids=False)
        return AlignedPredsActuals(subset_preds, subset_actuals, self._orthomosaic_stats, self._channel_map)
    def subset(self, field_name=None, field_value=None):
        subset_pred_ids, _ = self._get_aligned_pred_ids_on_field_value(field_name, field_value)
        return self.subset_by_pred_ids(subset_pred_ids)
    def resample(self, n_samples, replace):
        if len(self._deterministic_ordering) <= 0:
            return AlignedPredsActuals({}, {}, self._orthomosaic_stats, self._channel_map, verbose=False)
        y_true_aligned = self.get_actuals_labels()

        # Resample stratified by the true labels
        resampled_ids = resample(
            self._deterministic_ordering,
            replace=replace,
            stratify=y_true_aligned,
            n_samples=n_samples
        )

        result_preds = {}
        result_actuals = {}

        for pred_id in resampled_ids:
            ortho_name, building_id, _, _ = parse_pred_key(pred_id)
            lookup_id = joint_file_pred_key(ortho_name, building_id)

            result_preds[pred_id] = self._unaligned_preds[pred_id]
            result_actuals[lookup_id] = self._unaligned_actuals[lookup_id]

        return AlignedPredsActuals(result_preds, result_actuals, self._orthomosaic_stats, self._channel_map, verbose=False)
    def get_actual_class_counts(self):
        result = {l:0 for l in self.get_labels()}

        # Load the actual labels from the dataset
        for label in self._unaligned_actuals.values():
            result[label] += 1
        return result
    def get_predicted_class_counts(self):
        result = {l:0 for l in self.get_labels()}

        # Load the predicted labels from the predictions
        for pred_data in self._unaligned_preds.values():
            result[pred_data["label"]] += 1
        return result
    def __len__(self):
        return len(self._unaligned_preds)

def compute_AUCROC(aligned_preds_actuals_bundle):
    # One-vs-rest AUC per class, scored from the model's per-class PROBABILITIES.
    #
    # Scored from the probabilities, never from get_preds_multilabel_dict() -- that is a one-hot
    # indicator of the argmax label, and feeding a thresholded 0/1 decision to roc_auc_score does
    # not measure ranking quality at all: it collapses to a rescaled balanced accuracy, and cannot
    # distinguish a
    # model that ranks well but is mis-thresholded from one that ranks no better than chance. That
    # distinction is the whole point of reporting AUC on this task, where the decision threshold is
    # known not to transfer across events while the ranking does.
    change_auc_roc = {}
    for label in aligned_preds_actuals_bundle.get_labels():
        try:
            is_actual_label = [float(a[label]) for a in aligned_preds_actuals_bundle.get_actuals_multilabel_dict()]
            scores = aligned_preds_actuals_bundle.get_class_probabilities(label)
            change_auc_roc[label] = roc_auc_score(is_actual_label, scores)
        except ValueError:
            print("Warning: Unable to compute AUC ROC for", label, " due to insufficient positive or negative examples.")
    return change_auc_roc


# ---- ranking metrics ---------------------------------------------------------------------------
# The CHANGE task is binary, and the paper scores it by AUC rather than by a thresholded metric: the
# ranking transfers across disaster events while the threshold does not. These are the primitives
# behind every AUC the paper reports, and they live here rather than in the figure code so the
# evaluator and the figures cannot compute the same quantity two different ways.
CHANGE_POSITIVE_LABEL = "CHANGED"
CHANGE_NEGATIVE_LABEL = "UNCHANGED"


def change_probability(class_preds, positive_label=CHANGE_POSITIVE_LABEL,
                       negative_label=CHANGE_NEGATIVE_LABEL):
    """P(positive) renormalized over the two real classes.

    The stored per-class values are divided by a per-building pixel total that INCLUDES background,
    so they do not sum to 1 and the deflation varies per building. Renormalizing over the two real
    classes cancels it; ranking on the raw value would corrupt AUC. Returns 0.5 -- a deliberate tie
    -- when a building has no mass on either class, since there is no evidence to rank it by.
    """
    positive = class_preds.get(positive_label, 0.0)
    denominator = positive + class_preds.get(negative_label, 0.0)
    return positive / denominator if denominator > 0 else 0.5


def global_auc(actuals, scores):
    """AUC over all pairs. NaN when only one class is present."""
    actuals = np.asarray(actuals)
    if len(np.unique(actuals)) < 2:
        return np.nan
    return roc_auc_score(actuals, np.asarray(scores))


def within_ortho_auc(actuals, scores, group_codes, n_groups=None):
    """Mann-Whitney restricted to within-orthomosaic pairs: sum_o n1*n0*AUC_o / sum_o n1*n0.

    Between-site comparisons are the main confound on this task -- sites differ in damage prevalence
    and in GSD -- so this reports the ranking quality available to someone triaging a single scene.

    `group_codes` must be small integer codes, not strings: this runs inside the bootstrap tens of
    thousands of times and string comparison dominated the runtime otherwise.
    """
    actuals, scores = np.asarray(actuals), np.asarray(scores)
    group_codes = np.asarray(group_codes)
    numerator = denominator = 0.0
    for group in range(n_groups if n_groups is not None else int(group_codes.max()) + 1):
        mask = group_codes == group
        group_actuals = actuals[mask]
        positives = int(group_actuals.sum())
        negatives = int(mask.sum()) - positives
        if positives == 0 or negatives == 0:
            continue
        numerator += positives * negatives * roc_auc_score(group_actuals, scores[mask])
        denominator += positives * negatives
    return numerator / denominator if denominator else np.nan


def class_conditional_change_rates(described_buildings, restrict_to_classes=None):
    """P(CHANGED | the building's own damage class), from describe_buildings_for_change output.

    Fit this on a split you are NOT scoring. Fitting it on the evaluation split and then scoring
    that split with it leaks the labels and the resulting "baseline" is not a baseline.
    """
    changed, total = {}, {}
    for record in described_buildings.values():
        damage_class = record["self_label"]
        if restrict_to_classes is not None and damage_class not in restrict_to_classes:
            continue
        changed[damage_class] = changed.get(damage_class, 0) + (record["label"] == CHANGE_POSITIVE_LABEL)
        total[damage_class] = total.get(damage_class, 0) + 1
    return {damage_class: changed[damage_class] / count for damage_class, count in total.items()}


def class_oracle_scores(rate_table, damage_classes, default_rate=0.0):
    """Rank buildings purely by P(CHANGED | damage class), using each building's TRUE class.

    An upper bound on any two-stage "classify damage, then infer change" design rather than an
    achievable model: in deployment the class has to be inferred. Classes absent from the table
    score `default_rate`.
    """
    return np.array([rate_table.get(damage_class, default_rate) for damage_class in damage_classes])


def rank_average(*score_arrays):
    """Mean of the global percentile ranks of each input -- the paper's combiner.

    Averaging RANKS rather than raw scores is what makes combining a calibrated rate table with an
    uncalibrated model output meaningful: only the ordering of each input is used, so neither has to
    be on the other's scale. Ranking globally (not within orthomosaic) keeps the between-site
    component of both signals; handling the site confound is within_ortho_auc's job, and doing it a
    second time inside the score only discards information.
    """
    if not score_arrays:
        raise ValueError("rank_average needs at least one score array")
    n = len(score_arrays[0])
    if any(len(scores) != n for scores in score_arrays):
        raise ValueError("rank_average needs score arrays of equal length")
    return sum(rankdata(scores) / n for scores in score_arrays) / len(score_arrays)


def averaged_with_class_oracle_scores(model_scores, rate_table, damage_classes):
    """The "Class Oracle + Change Model Average" score for each building."""
    return rank_average(class_oracle_scores(rate_table, damage_classes), np.asarray(model_scores))


def building_ids_from_pred_keys(pred_keys):
    """Physical-building identity for each prediction key, for clustered resampling.

    A building seen under two satellite views yields two rows that share a building id. The id is
    the trailing hash of the joint key, which is stable across orthomosaics and sensors.
    """
    return np.array([parse_pred_key(key)[1] for key in pred_keys])


def building_clustered_bootstrap_ci(metric_func, building_ids, n_iterations=2000,
                                    confidence_level=0.95, seed=7):
    """Percentile CI as a (lower, upper) tuple, from resampling BUILDINGS rather than rows.

    All rows of a drawn building travel together. Most test buildings appear under more than one
    satellite view, and those views are nowhere near independent draws -- their CHANGED/UNCHANGED
    labels agree far more often than chance and their model scores are strongly correlated -- so
    resampling rows independently understates the variance. `metric_func` takes row indices and
    returns a scalar; non-finite draws are skipped.

    Reseeds its own generator from `seed` so an interval does not depend on how many other intervals
    were computed before it; sharing one stream makes every published number a function of call
    order. Returns None when no draw produced a finite value.
    """
    rng = np.random.RandomState(seed)
    building_ids = np.asarray(building_ids)
    _, inverse = np.unique(building_ids, return_inverse=True)
    n_buildings = int(inverse.max()) + 1
    members = [np.where(inverse == building)[0] for building in range(n_buildings)]

    scores = []
    for _ in range(n_iterations):
        drawn = rng.randint(0, n_buildings, n_buildings)
        value = metric_func(np.concatenate([members[building] for building in drawn]))
        if np.isfinite(value):
            scores.append(value)
    if not scores:
        return None
    alpha = (1.0 - confidence_level) / 2.0
    return (float(np.percentile(scores, alpha * 100)),
            float(np.percentile(scores, (1.0 - alpha) * 100)))


def named_confidence_bounds(interval, confidence_level=0.95):
    """(lower, upper) -> the {"lower_bound_<level>": ..., "upper_bound_<level>": ...} payload shape.

    Naming is a reporting concern, so the bootstrap primitive returns a plain tuple and this applies
    the metrics-file convention. Keeps the primitive reusable by callers that just want the numbers.
    """
    if interval is None:
        return None
    return {"lower_bound_" + str(confidence_level): interval[0],
            "upper_bound_" + str(confidence_level): interval[1]}

def compute_measure(aligned_preds_actuals_bundle, measure_func, measure_type="class_level", zero_division=None):
    if measure_type == "class_level":
        change_measure = {}
        for label in aligned_preds_actuals_bundle.get_labels():
            # sklearn convention is measure_func(y_true, y_pred): pass actuals then preds.
            preds = [a[label] for a in aligned_preds_actuals_bundle.get_preds_multilabel_dict()]
            actuals = np.around([p[label] for p in aligned_preds_actuals_bundle.get_actuals_multilabel_dict()])
            if zero_division is None:
                change_measure[label] = measure_func(actuals, preds)
            else:
                change_measure[label] = measure_func(actuals, preds, zero_division=zero_division)
        return change_measure
    if zero_division is None:
        return measure_func(aligned_preds_actuals_bundle.get_actuals_labels(),
                            aligned_preds_actuals_bundle.get_preds_labels(),
                            average=measure_type)
    return measure_func(aligned_preds_actuals_bundle.get_actuals_labels(),
                        aligned_preds_actuals_bundle.get_preds_labels(),
                        average=measure_type,
                        zero_division=zero_division)

def compute_f1(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, f1_score, measure_type, zero_division=np.nan)
def compute_precision(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, precision_score, measure_type, zero_division=np.nan)
def compute_recall(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, recall_score, measure_type, zero_division=np.nan)
def compute_accuracy(aligned_preds_actuals_bundle):
    return compute_measure(aligned_preds_actuals_bundle, accuracy_score, "class_level")
def compute_confusion_matrix(aligned_preds_actuals_bundle):
    return confusion_matrix(y_true=aligned_preds_actuals_bundle.get_actuals_labels(),
                            y_pred=aligned_preds_actuals_bundle.get_preds_labels(),
                            labels=aligned_preds_actuals_bundle.get_labels())

def compute_ece(aligned_preds_actuals_bundle, n_bins=10):
    y_true = np.array(aligned_preds_actuals_bundle.get_actuals_labels())
    y_pred = np.array(aligned_preds_actuals_bundle.get_preds_labels())
    confidences = np.array(aligned_preds_actuals_bundle.get_preds_confidences())

    # Calculate whether each prediction was accurate
    accuracies = (y_true == y_pred).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Determine elements in the current bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prob_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)

    return float(ece)

def compute_ranking_metrics(aligned_preds_actuals_bundle, do_confidence=False, n_iterations=2000,
                            confidence_level=0.95, seed=7):
    """Global and within-orthomosaic AUC for the CHANGE score, with building-clustered intervals.

    Reported alongside the thresholded metrics because they answer a different question: whether the
    model ORDERS buildings usefully, independent of whether its decision threshold happens to suit
    the event being scored.
    """
    actuals = aligned_preds_actuals_bundle.get_change_actuals()
    scores = aligned_preds_actuals_bundle.get_change_scores()
    ortho_codes, n_orthos = aligned_preds_actuals_bundle.get_orthomosaic_codes()
    building_ids = aligned_preds_actuals_bundle.get_building_ids()

    def interval(metric_func):
        if not do_confidence:
            return None
        return named_confidence_bounds(
            building_clustered_bootstrap_ci(metric_func, building_ids, n_iterations,
                                           confidence_level, seed),
            confidence_level)
    return {
        "global": {
            "score": global_auc(actuals, scores),
            "confidence_interval": interval(lambda idx: global_auc(actuals[idx], scores[idx])),
        },
        "within_orthomosaic": {
            "score": within_ortho_auc(actuals, scores, ortho_codes, n_orthos),
            "confidence_interval": interval(
                lambda idx: within_ortho_auc(actuals[idx], scores[idx], ortho_codes[idx], n_orthos)),
        },
    }


def compute_class_oracle_metrics(aligned_preds_actuals_bundle, rate_table, damage_classes_by_key,
                                 do_confidence=False, n_iterations=2000, confidence_level=0.95,
                                 seed=7):
    """The class-conditional baseline and the model averaged with it.

    `rate_table` must be fitted on a split other than the one being scored -- see
    class_conditional_change_rates. `damage_classes_by_key` maps a prediction key to that building's
    own damage class.
    """
    actuals = aligned_preds_actuals_bundle.get_change_actuals()
    model_scores = aligned_preds_actuals_bundle.get_change_scores()
    building_ids = aligned_preds_actuals_bundle.get_building_ids()
    # Prediction keys carry a GSD prefix ("<gsd_x>_<gsd_y>_<ortho>_<hash>") while
    # describe_buildings_for_change keys on the building alone ("__<ortho>_<hash>"), so the two key
    # spaces do not intersect at all. Normalize before the lookup -- the same round-trip
    # AlignedPredsActuals._get_fixed_ordered_data does. Without it every lookup misses, every
    # building falls back to default_rate, and a constant score array makes the oracle AUC exactly
    # 0.5 while rank_average leaves the model's ordering untouched, so the averaged AUC comes back
    # bit-identical to the model's own.
    classes = []
    for key in aligned_preds_actuals_bundle.get_pred_keys():
        ortho_name, building_id, _, _ = parse_pred_key(key)
        classes.append(damage_classes_by_key.get(joint_file_pred_key(ortho_name, building_id)))
    unresolved = sum(1 for damage_class in classes if damage_class is None)
    if unresolved == len(classes):
        raise ValueError(
            f"No building resolved to a damage class ({len(classes)} rows). The class-oracle "
            "baseline would be a constant, scoring exactly 0.5 and leaving the averaged score "
            "identical to the model's own. This means the damage-class map and the prediction keys "
            "are in different key spaces.")
    if unresolved:
        print(f"WARNING: {unresolved} of {len(classes)} buildings have no damage class; they score "
              "the default rate and cannot be ranked by the class-conditional baseline.")
    oracle_scores = class_oracle_scores(rate_table, classes)
    averaged_scores = rank_average(oracle_scores, model_scores)

    def interval(scores):
        if not do_confidence:
            return None
        return named_confidence_bounds(
            building_clustered_bootstrap_ci(lambda idx: global_auc(actuals[idx], scores[idx]),
                                            building_ids, n_iterations, confidence_level, seed),
            confidence_level)
    return {
        "class_oracle": {
            "score": global_auc(actuals, oracle_scores),
            "confidence_interval": interval(oracle_scores),
            "rate_table": rate_table,
        },
        "averaged_with_class_oracle": {
            "score": global_auc(actuals, averaged_scores),
            "confidence_interval": interval(averaged_scores),
        },
    }


def generate_metrics_payload(aligned_preds_actuals_bundle, do_confidence, n_iterations, n_samples,
                             confidence_level, auc_n_iterations=2000, auc_seed=7):
    # auc_* are separate from n_iterations/n_samples on purpose: the thresholded metrics' intervals
    # come from row-level stratified resampling of n_samples rows, while the AUC intervals resample
    # BUILDINGS (all views together) and take no n_samples. Sharing one knob would silently tie two
    # different resampling schemes to the same number.
    return {
        "samples": {
            "total": len(aligned_preds_actuals_bundle),
            "actual_class_counts": aligned_preds_actuals_bundle.get_actual_class_counts(),
            "predicted_class_counts": aligned_preds_actuals_bundle.get_predicted_class_counts()
        },
        "metrics": {
            "AUC_ROC": {
                "class_level": compute_AUCROC(aligned_preds_actuals_bundle),
                # Ranking metrics for the CHANGE score itself. The class_oracle and
                # averaged_with_class_oracle entries are merged in by main() when a rate table is
                # supplied, since fitting one requires a split this evaluator is not scoring.
                **compute_ranking_metrics(aligned_preds_actuals_bundle, do_confidence,
                                          auc_n_iterations, confidence_level, auc_seed),
            },
            "F1": {
                "class_level": compute_f1(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score":compute_f1(aligned_preds_actuals_bundle, "macro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_f1(x, "macro"))},
                "micro": {"score":compute_f1(aligned_preds_actuals_bundle, "micro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_f1(x, "micro"))}
            },
            "Accuracy": {
                "class_level": compute_accuracy(aligned_preds_actuals_bundle),
            },
            "Precision": {
                "class_level": compute_precision(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score":compute_precision(aligned_preds_actuals_bundle, "macro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_precision(x, "macro"))},
                "micro": {"score":compute_precision(aligned_preds_actuals_bundle, "micro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_precision(x, "micro"))}
            },
            "Recall": {
                "class_level": compute_recall(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score":compute_recall(aligned_preds_actuals_bundle, "macro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_recall(x, "macro"))},
                "micro": {"score":compute_recall(aligned_preds_actuals_bundle, "micro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                  n_iterations,
                                                                  n_samples,
                                                                  confidence_level,
                                                                  lambda x:compute_recall(x, "micro"))}
            },
            "Expected_Calibration_Error": compute_ece(aligned_preds_actuals_bundle),
            "Confusion_Matrix": {
                "matrix": compute_confusion_matrix(aligned_preds_actuals_bundle).tolist(),
                "class_labels": aligned_preds_actuals_bundle.get_labels(),
            },
        }
    }

def change_per_group_metrics(aligned_preds_actuals_bundle):
    # metrics_func passed to the shared compute_metrics_per_group for CHANGE per-group
    # breakdowns; confidence intervals are not computed per group (do_confidence=False).
    return generate_metrics_payload(aligned_preds_actuals_bundle, False, 0, 0, 0.0)

def compute_confidence_interval(aligned_preds_actuals_bundle, n_iterations, n_samples, confidence_level, metric_func):
    bootstrapped_scores = []

    for _ in range(n_iterations):
        # Perform stratified sampling with replacement
        resampled_aligned_bundle = aligned_preds_actuals_bundle.resample(replace=True, n_samples=n_samples)

        # Compute the metric
        score = metric_func(resampled_aligned_bundle)
        bootstrapped_scores.append(score)

    # Calculate the Percentile Confidence Interval
    alpha = (1.0 - confidence_level) / 2.0

    # Find the upper and lower bounds based on the observed percentiles
    lower_bound = np.percentile(bootstrapped_scores, alpha * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1.0 - alpha) * 100)

    return {"lower_bound_"+str(confidence_level):lower_bound,
            "upper_bound_"+str(confidence_level):upper_bound}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model trained on the CHANGE task.")
    add_hyperparameters_files_to_parse_args(parser,
                                            add_dataset_paths_file_path=True,
                                            add_data_source_config_parameters_file_path=True,
                                            add_model_hyperparameters_yaml_path=True)
    parser.add_argument("--preds_paths", type=str, nargs="+",
        help="The path to file that contains the model predictions.")
    parser.add_argument("--channels_hyperparameters_file_path", type=str,
        help="The path to the file that maps dataset labels to model outputs.")
    parser.add_argument("--metrics_file", type=str,
        help="The path to the file where the metrics will be stored.")
    parser.add_argument("--ortho_stats_file", type=str, default=None,
        help="The path to the statistics.csv file included with the dataset. Overrides the \"statistics\" key in the dataset paths yaml.")
    parser.add_argument("--dataset_subset", type=str, default="test", help="The key in the dataset path yaml to access.")
    parser.add_argument("--ortho_backend", type=str, default="auto",
        help="Backend used when loading orthomosaics to reconstruct CHANGE ground truth (only metadata/annotations are needed, not pixel data).")
    parser.add_argument("--include_same_source_views", action="store_true",
        help="By default, two views collected by the same source are never treated as parallel "
             "views of each other. Set this flag to also compare views that share a source.")
    parser.add_argument("--change_decision_rule", type=str, default="all", choices=["all", "any"],
        help="Ground-truth rule, matching the training annotator's datagen setting: 'all' labels a "
             "building CHANGED only when every parallel view disagrees with it; 'any' when at least "
             "one does.")
    parser.add_argument("--restrict_pre_post", type=str, default="all", choices=["all", "pre", "post"],
        help="Restrict the evaluation to anchor orthomosaics of a given event phase (statistics.csv "
             "'Pre/Post Event'). 'post' scores only where the anchor imagery carries a change signal; "
             "'all' (default) keeps every anchor.")
    parser.add_argument("--random_baseline", action="store_true",
        help="When set, the input predictions are overridden with a uniform-random label so the "
             "reported metrics are the random-performance (no-skill) reference for this scored set "
             "(macro F1 ~= 0.5 for two classes). Mirrors evaluate_RDA.py's --random_baseline.")
    parser.add_argument("--class_oracle_rate_table", type=str, default=None,
        help="JSON file mapping each satellite damage class to its P(CHANGED), FITTED ON ANOTHER "
             "SPLIT. Supplying it adds the class-conditional baseline and the model-averaged-with-it "
             "AUC to the reported metrics. Fit it with class_conditional_change_rates on a split you "
             "are not scoring -- fitting on the evaluation split leaks the labels.")
    parser.add_argument("--auc_bootstrap_iterations", type=int, default=2000,
        help="Resamples behind the AUC confidence intervals. These use a BUILDING-clustered "
             "bootstrap (all views of a drawn building travel together), unlike the row-level "
             "stratified resampling behind the thresholded metrics' intervals.")
    parser.add_argument("--auc_bootstrap_seed", type=int, default=7,
        help="Base seed for the AUC bootstrap; each interval reseeds from it so intervals do not "
             "depend on how many others were computed first.")
    parser.add_argument("--confidence_interval_N", type=int, default=1000, help="The number of samples to draw when bootstrapping confidence intervals.")
    parser.add_argument("--confidence_iterations", type=int, default=1000, help="The number of times to sample statistics from the evaluation.")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="The confidence level for the produced confidence interval.")
    parser.add_argument("--metrics_to_report", type=str,
        default="global_metrics>metrics>AUC_ROC>global,per_gsd_metrics>*>metrics>AUC_ROC>global",
        help="The metric that will be printed to the log for easier monitoring. Defaults to AUC "
             "rather than macro F1: on this task the ranking transfers across events and the "
             "decision threshold does not, so a thresholded metric can collapse while the model's "
             "ordering is intact. Watching macro F1 during a run therefore reports a number the "
             "results are not scored on.")
    parser.add_argument("--lightweight", action="store_true",
        help="Only emit model_name and global_metrics, skipping the per-mapper/ortho/GSD/platform/source breakdowns. Use for the CI evaluation files "
             "under evaluations/ so they stay reviewably small.")
    args = parser.parse_args()

    print("Parsing the channel parameters...")
    channel_parameters = parse_hyperparameters(args.channels_hyperparameters_file_path)

    print("Parsing model hyperparameters (needed so orthomosaic loading knows to pull BDA annotations)...")
    model_hyperparameters = parse_hyperparameters(args.model_hyperparameters_yaml_path, verbose=False)

    print("Parsing data paths and evaluation parameters...")
    dataset_paths = parse_hyperparameters(args.dataset_paths_file_path, verbose=False)
    data_source_config_parameters = parse_hyperparameters(args.data_source_config_parameters_file_path, verbose=False)

    print("Parsing the orthomosaics statistics metadata file...")
    ortho_stats_file = args.ortho_stats_file if args.ortho_stats_file else dataset_paths["statistics"]
    paresed_orthomosaic_stats = pd.read_csv(ortho_stats_file, header=0, index_col="Orthomosaic")
    paresed_orthomosaic_stats["Orthomosaic"] = paresed_orthomosaic_stats.index

    print("Parsing passed predictions files...")
    all_preds = {}
    model_names = []
    for passed_preds_path in args.preds_paths:
        parsed_model_name, parsed_preds = parse_model_name_and_predicted_labels(passed_preds_path)
        all_preds = all_preds | parsed_preds
        model_names.append(parsed_model_name)

    if args.random_baseline:
        change_class_labels = [c for c in channel_parameters["channel_maps"]["output_class_2_idx_map"] if c != "background"]
        print(f"Overriding predictions with a uniform-random baseline over {change_class_labels}...")
        all_preds = override_preds_with_random_baseline(all_preds, change_class_labels)
        model_names = [name + "_random_baseline" for name in model_names]

    print("Loading orthomosaics to relabel CHANGE...")
    change_orthomosaics = MultisourceOrthomosaicFactory(
        dataset_paths_dict=dataset_paths,
        data_source_config_parameters=data_source_config_parameters,
        model_hyperparameters=model_hyperparameters,
        boundary_folder=None,
        statistics_file_path=ortho_stats_file,
        train_validation_test=args.dataset_subset,
        backend=args.ortho_backend,
        required_channels=None,
    )

    print("Reconstructing CHANGE ground truth labels...")
    parsed_described_buildings = describe_buildings_for_change(change_orthomosaics,
                                                        skip_same_source=not args.include_same_source_views,
                                                        change_decision_rule=args.change_decision_rule)
    ground_truth_labels = {key: record["label"] for key, record in parsed_described_buildings.items()}

    if args.restrict_pre_post != "all":
        print(f"Restricting evaluation to {args.restrict_pre_post}-event anchor imagery...")
        ground_truth_labels = restrict_actuals_to_event_phase(ground_truth_labels, paresed_orthomosaic_stats, args.restrict_pre_post)

    print("Aligning predictions and actuals...")
    parsed_aligned_preds_actuals_bundle = AlignedPredsActuals(all_preds, ground_truth_labels, paresed_orthomosaic_stats, channel_parameters)

    print("Computing metrics...")
    metrics = {"model_name": list(set(model_names))}
    metrics = metrics | {"global_metrics": generate_metrics_payload(parsed_aligned_preds_actuals_bundle,
                                                                    True,
                                                                    args.confidence_iterations,
                                                                    args.confidence_interval_N,
                                                                    args.confidence_level,
                                                                    args.auc_bootstrap_iterations,
                                                                    args.auc_bootstrap_seed)}
    if args.class_oracle_rate_table:
        print(f"Adding class-conditional baseline metrics from {args.class_oracle_rate_table}...")
        with open(args.class_oracle_rate_table, "r", encoding="utf-8") as rate_table_file:
            parsed_rate_table = json.load(rate_table_file)
        parsed_damage_classes_by_key = {key: record["self_label"] for key, record in parsed_described_buildings.items()}
        metrics["global_metrics"]["metrics"]["AUC_ROC"] |= compute_class_oracle_metrics(
            parsed_aligned_preds_actuals_bundle,
            parsed_rate_table,
            parsed_damage_classes_by_key,
            do_confidence=True,
            n_iterations=args.auc_bootstrap_iterations,
            confidence_level=args.confidence_level,
            seed=args.auc_bootstrap_seed)

    if not args.lightweight:
        # output metric_key -> orthomosaic-stats field the global metrics are broken down by.
        per_group_breakdown_fields = {
            "per_mapper_metrics": "Mapper",
            "per_ortho_metrics": "Orthomosaic",
            "per_gsd_metrics": "gsd_x",
            "per_collection_platform_metrics": "Platform / Provider",
            "per_source_metrics": "Source",
        }
        for metric_key, stats_field in per_group_breakdown_fields.items():
            metrics[metric_key] = compute_metrics_per_group(
                parsed_aligned_preds_actuals_bundle.get_aligned_pred_ids(),
                lambda pred_id, field=stats_field: parsed_aligned_preds_actuals_bundle.field_value(pred_id, field),
                lambda pred_ids: change_per_group_metrics(parsed_aligned_preds_actuals_bundle.subset_by_pred_ids(pred_ids)))

    reported_metrics = parse_reportable_metrics(metrics, args.metrics_to_report)
    for reported_metric in reported_metrics:
        print(reported_metric)

    write_metrics_json(metrics, args.metrics_file)
    print("Done.")
