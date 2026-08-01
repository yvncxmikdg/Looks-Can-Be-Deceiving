import json
import argparse

from modeling.utils.run_metadata import (build_run_metadata, current_timestamp, add_stage,
                                         METADATA_KEY, STAGE_LABEL_REMAP, PREDICTION_COUNT, REMAPPED_COUNT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--preds_path", type=str,
        help="The path to file that contains the model predicitons.")
    parser.add_argument("--target_label", type=str,
        help="The label value that needs to be changed.")
    parser.add_argument("--destination_label", type=str,
        help="The value that should be put in place of the target label.")
    parser.add_argument("--out_path", type=str,
        help="The path to file that contains the model predicitons.")
    args = parser.parse_args()

    # Mark the start of the relabeling pass before we touch the input predictions.
    remap_start_time = current_timestamp()

    with open(args.preds_path, "r") as f:
        preds_data = json.load(f)

    remapped_count = 0
    for pred in preds_data["preds"]:
        if preds_data["preds"][pred]["label"] == args.target_label:
            preds_data["preds"][pred]["label"] = args.destination_label
            remapped_count += 1

    # Add the relabeling as its own stage, alongside whatever stages produced the predictions we
    # relabeled (inference, and possibly a meta-model pass) rather than replacing them - so a
    # floating relabeled preds file can still be traced back to what produced it and how it was
    # modified. Older inputs with no metadata simply start the dict here.
    preds_data[METADATA_KEY] = add_stage(
        preds_data.get(METADATA_KEY),
        STAGE_LABEL_REMAP,
        build_run_metadata(
            model_path=None,
            hyperparameters={"target_label": args.target_label, "destination_label": args.destination_label},
            start_time=remap_start_time,
            end_time=current_timestamp(),
            extra={PREDICTION_COUNT: len(preds_data["preds"]), REMAPPED_COUNT: remapped_count},
        ),
    )

    with open(args.out_path, "w") as f:
        f.write(json.dumps(preds_data))
