def confidence_scalar(confidence, label):
    """Return the scalar confidence of the predicted ``label``.

    BDA/RDA base-model predictions serialize confidence as a per-class map whose values are the
    building's raw pixel counts per class (see ConfidenceResult.jsonify / fuse_bda_tiled_inference).
    Without the meta-model there is no downstream calibration step, so the reported confidence is the
    proportion of the building's pixel mass assigned to the predicted label: normalize the map by its
    total so we emit a 0-1 confidence instead of a raw pixel count. The meta-model path already stores
    a calibrated scalar probability, which passes through unchanged. Normalizing an already-normalized
    map (values summing to 1) is a no-op, so this is safe for either representation.
    """
    if isinstance(confidence, dict):
        total = sum(confidence.values())
        if total <= 0:
            return 0.0
        return confidence.get(label, 0.0) / total
    return confidence
