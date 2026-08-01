"""Stage 1: annotations + model predictions -> figure_data.json.

Emits one artifact holding everything the statistics stage needs:

  rows         one per satellite-anchored building on the scored (test) stripe:
               key, ortho, sat_class, drone_class, y (1 = CHANGED), and every arm's
               normalized CHANGED score
  class_rates  per-stripe P(CHANGED | satellite damage class) counts, for every stripe named by
               --rate_subsets. The training stripe is what the Class Oracle is fitted on and what
               the train-vs-test disagreement figure plots, and it has no model predictions -- which
               is why rates are collected per stripe while scored rows come only from the test one.
  arms         arm slug -> {pretty, pretrained}, read from arms.yaml

Ground truth is rebuilt from the annotations rather than read out of preds.json -- preds.json's
`label` field is the model's ARGMAX PREDICTION, not the truth. The pairing rule is
`evaluate_CHANGE.describe_buildings_for_change`, called rather than reimplemented, so this stage and
the evaluator cannot disagree about which buildings changed.

Scores likewise come from `evaluate_CHANGE.change_probability`: the stored `class_preds` are divided by
a per-building pixel total that includes background, so the raw values do not sum to 1 and the
deflation varies per building. Renormalizing over the two real classes cancels it; ranking on the raw
value would corrupt AUC.

Every path is a required argument -- see ortho_loader's module docstring.
"""
import argparse
import glob
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=wrong-import-position
# ortho_loader puts <repo>/src on sys.path as an import side effect, so it and everything below it
# must be imported after HERE is defined -- they cannot move to the top.
from ortho_loader import add_dataset_args, add_taxonomy_arg, load_orthos  # noqa: E402
from modeling.Models.OrthoInferenceWrapper import joint_file_pred_key, parse_pred_key
from modeling.evaluate_CHANGE import change_probability, describe_buildings_for_change
from modeling.utils.initialize_dataset import CHANGE_ANCHOR_EVENT_PHASE

# The CHANGE task anchors its sample windows on satellite imagery.
CHANGE_ANCHOR_SOURCE = "Satellite"


def load_arms(path):
    """arm slug -> {pretty, pretrained}, from arms.yaml."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["arms"]


def build_truth(orthos):
    """Satellite-anchored key -> {ortho, sat_class, drone_class, y}.

    Thin projection of evaluate_CHANGE.describe_buildings_for_change. The pairing rule -- skip the
    building's own orthomosaic, skip same-source views, CHANGED iff the label differs in every
    remaining parallel view -- is imported from the evaluator rather than restated, so this stage and
    the evaluator cannot disagree about which buildings changed.

    Anchors are restricted to POST-event satellite imagery, matching initialize_windowed_dataset's
    hardwired CHANGE_ANCHOR_EVENT_PHASE. This matters for the per-stripe rate tables: a pre-event tile
    carries no damage signal, so anchoring on one measures priors rather than observable change. The
    scored rows are implicitly post-event anyway -- they intersect with model predictions, and the
    models only ran on post-event anchors -- but the rate tables have no such intersection, and
    including pre-event anchors more than doubles the training population and shifts every
    P(CHANGED | class) in the Class Oracle.
    """
    described = describe_buildings_for_change(orthos, skip_same_source=True,
                                              change_decision_rule="all",
                                              anchor_source=CHANGE_ANCHOR_SOURCE,
                                              anchor_event_phase=CHANGE_ANCHOR_EVENT_PHASE)
    print(f"  anchored buildings with ground truth ({CHANGE_ANCHOR_SOURCE} + "
          f"{CHANGE_ANCHOR_EVENT_PHASE}): {len(described)}", flush=True)
    return {key: {"ortho": record["orthomosaic"],
                  "sat_class": record["self_label"],
                  "drone_class": record["parallel_labels"][0],
                  "y": 1 if record["label"] == "CHANGED" else 0}
            for key, record in described.items()}


def class_rate_counts(meta):
    """satellite class -> {changed, total} over a stripe's rebuilt ground truth."""
    counts = {}
    for rec in meta.values():
        bucket = counts.setdefault(rec["sat_class"], {"changed": 0, "total": 0})
        bucket["changed"] += rec["y"]
        bucket["total"] += 1
    return counts


def score_arms(logs_dir, run_glob, meta, arms):
    """arm slug -> {gt_key: renormalized P(CHANGED)} for every run under logs_dir."""
    files = sorted(glob.glob(os.path.join(logs_dir, run_glob, "test_inference", "preds.json")))
    print(f"  found {len(files)} preds files under {logs_dir}", flush=True)
    if not files:
        raise SystemExit(f"no preds.json matched {run_glob}/test_inference under {logs_dir}")

    per_arm = {}
    for preds_path in files:
        run = os.path.basename(os.path.dirname(os.path.dirname(preds_path)))
        # Run dirs are "<timestamp>_<user>_change-post-512-<slug>_<commit>".
        arm = run.split("change-post-512-")[1].rsplit("_", 1)[0]
        if arm not in arms:
            raise SystemExit(f"run '{run}' has arm slug '{arm}', which is not in arms.yaml. Add it "
                             f"(with its pretty name and pretrained flag) or exclude the run via "
                             f"--run_glob.")
        scores = {}
        with open(preds_path, encoding="utf-8") as handle:
            preds = json.load(handle)["preds"]
        for key, pred in preds.items():
            ortho_name, building_id, _, _ = parse_pred_key(key)
            gt_key = joint_file_pred_key(ortho_name, building_id)
            if gt_key not in meta:
                continue
            scores[gt_key] = change_probability(pred["class_preds"])
        per_arm[arm] = scores
        print(f"    {arm:>14}: {len(scores)} scored", flush=True)
    return per_arm


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs_dir", required=True,
                    help="run-log root; reads <logs_dir>/<run_glob>/test_inference/preds.json")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data"),
                    help="where figure_data.json is written (default: ./data next to this script)")
    add_dataset_args(ap)
    add_taxonomy_arg(ap)
    ap.add_argument("--arms_file", default=os.path.join(HERE, "arms.yaml"),
                    help="arm slug -> pretty name + pretrained flag")
    ap.add_argument("--scored_subset", default="test", choices=["train", "validation", "test"],
                    help="the stripe the models were scored on; supplies the per-building rows")
    ap.add_argument("--rate_subsets", nargs="+", default=["train", "test"],
                    help="stripes to collect P(CHANGED | class) rates for. The Class Oracle is fitted "
                         "on the training stripe, which has no model predictions.")
    ap.add_argument("--run_glob", default="*post*",
                    help="which run directories to include; the default selects the post-only wave")
    # Opt-in, and off by default: a hardcoded expected population is only meaningful for the exact
    # dataset snapshot this paper scored. Pass both to re-assert it when reproducing that snapshot.
    ap.add_argument("--expect_n", type=int, default=None,
                    help="assert the scored population size (before excluding obscured)")
    ap.add_argument("--expect_positives", type=int, default=None,
                    help="assert the number of CHANGED buildings in the scored population")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    arms = load_arms(args.arms_file)

    # Rates first, one stripe at a time, so only one stripe's orthomosaics are resident at once.
    class_rates, scored_meta = {}, None
    for subset in dict.fromkeys([*args.rate_subsets, args.scored_subset]):
        print(f"loading orthomosaics for stripe={subset} + rebuilding ground truth ...", flush=True)
        orthos = load_orthos(args, subset)
        meta = build_truth(orthos)
        print(f"  satellite-anchored keys with ground truth: {len(meta)}", flush=True)
        if subset in args.rate_subsets:
            class_rates[subset] = class_rate_counts(meta)
        if subset == args.scored_subset:
            scored_meta = meta

    print(f"scoring arms against stripe={args.scored_subset} ...", flush=True)
    per_arm = score_arms(args.logs_dir, args.run_glob, scored_meta, arms)
    keys = sorted(set.intersection(*[set(s) for s in per_arm.values()]))
    print(f"  keys common to all {len(per_arm)} arms: {len(keys)}", flush=True)

    rows = [{"key": k, **{f: scored_meta[k][f] for f in ("ortho", "sat_class", "drone_class", "y")},
             "scores": {a: s[k] for a, s in per_arm.items()}} for k in keys]

    out = os.path.join(args.out_dir, "figure_data.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "class_rates": class_rates,
                   "arms": {a: arms[a] for a in sorted(per_arm)}}, handle)

    positives = sum(r["y"] for r in rows)
    missing = sum(1 for r in rows if r["drone_class"] is None)
    print(f"wrote {out}: n={len(rows)}, positives={positives} ({positives/len(rows):.3f}), "
          f"rows missing a drone class={missing}", flush=True)
    for subset, counts in class_rates.items():
        total = sum(c["total"] for c in counts.values())
        print(f"  {subset} rate table: {len(counts)} classes over {total} buildings", flush=True)

    assert missing == 0, f"{missing} rows have no drone class"
    # If a rebuild does not reproduce the asserted population, ground truth came from a materially
    # different labeler and every downstream figure would be silently wrong.
    if args.expect_n is not None:
        assert len(rows) == args.expect_n, f"expected n={args.expect_n}, got {len(rows)}"
    if args.expect_positives is not None:
        assert positives == args.expect_positives, \
            f"expected {args.expect_positives} positives, got {positives}"


if __name__ == "__main__":
    main()
