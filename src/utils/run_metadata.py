"""Shared run-provenance helpers for prediction files (issue #183).

Every prediction file we emit (BDA, BDA_META, BDA_ADJ, RDA) carries a ``metadata`` block describing
the run(s) that produced it. The block is a dict *indexed by pipeline stage* (see the ``STAGE_*``
constants): each step that touches the file - base inference, the meta-model refinement, a label
remap, and finally evaluation - adds its own entry with ``add_stage`` instead of overwriting the
previous one. A metrics file downstream of, say, a BDA -> BDA_META -> evaluate run therefore carries
the provenance of all three stages, so it can be traced back to a reproducible run even if the
predictions file goes missing.

Every key that crosses a file boundary is a named constant here so producers and consumers can never
drift on a bare string literal. All environment capture is best-effort: anything that can't be
resolved (git isn't on PATH, a package isn't installed, ...) is recorded as ``None`` rather than
failing the run - collecting provenance must never be the reason a long inference job dies right
before writing its output.
"""
import os
import sys
import platform
import subprocess
import datetime
import importlib.metadata

# Top-level key under which the provenance block lives in a predictions / metrics file.
METADATA_KEY = "metadata"

# Pipeline-stage keys. The metadata block is a dict indexed by these - one entry per step that
# touched the file - so a multi-step lineage accumulates instead of overwriting.
# STAGE_TRAINING is the upstream stage: it produces the checkpoint the inference stage later loads.
# It doesn't write a predictions file (see modeling/utils/provenance.py, which stamps it into the
# training run's manifest), but it lives here so training provenance shares this vocabulary and can
# be folded into a downstream predictions file's lineage.
STAGE_TRAINING = "training"
STAGE_INFERENCE = "inference"
STAGE_META_MODEL = "meta_model"
STAGE_LABEL_REMAP = "label_remap"
STAGE_EVALUATION = "evaluation"

# Fields inside a single stage's metadata block.
RUN_START_TIME = "run_start_time"
RUN_END_TIME = "run_end_time"
MODEL_PATH = "model_path"
HYPERPARAMETERS = "hyperparameters"
COMMAND = "command"
ENVIRONMENT = "environment"
# Optional per-stage extras.
PREDICTION_COUNT = "prediction_count"
REMAPPED_COUNT = "remapped_count"

# Fields inside the environment sub-block.
GIT_COMMIT = "git_commit"
PYTHON_VERSION = "python_version"
PLATFORM = "platform"
PACKAGE_VERSIONS = "package_versions"

# Packages whose versions plausibly affect inference numerics; captured for replication.
_REPLICATION_RELEVANT_PACKAGES = ("torch", "pytorch-lightning", "numpy", "xgboost")


def current_timestamp():
    """A timezone-aware ISO-8601 UTC timestamp, used to mark the start/end of a run."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _package_version(distribution_name):
    """The installed version of a distribution, or None if it isn't installed. Uses the package
    metadata (not an import) so we don't pay an import cost or trigger import side effects just to
    read a version string."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit():
    """The current git commit hash of the checkout the code is running from, or None if it can't be
    determined (git missing, or not inside a work tree)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment():
    """A best-effort snapshot of the runtime environment for replication: the git commit the code
    was at, the Python/OS it ran on, and the versions of the packages most likely to shift model
    numerics."""
    return {
        GIT_COMMIT: _git_commit(),
        PYTHON_VERSION: platform.python_version(),
        PLATFORM: platform.platform(),
        PACKAGE_VERSIONS: {name: _package_version(name) for name in _REPLICATION_RELEVANT_PACKAGES},
    }


def build_run_metadata(model_path, hyperparameters, start_time, end_time, extra=None):
    """Assemble a single stage's provenance block.

    Args:
        model_path: Path to the checkpoint loaded for the run (None for stages without one, e.g.
            evaluation or a label remap).
        hyperparameters: JSON-serializable dict of every hyperparameter/argument fed to the step.
        start_time: ISO timestamp (see ``current_timestamp``) marking when the step began.
        end_time: ISO timestamp marking when the step finished.
        extra: Optional dict of step-specific fields merged into the block (e.g. ``prediction_count``).

    Returns:
        A plain, JSON-serializable dict. Callers place it under its stage key with ``add_stage``.
    """
    metadata = {
        RUN_START_TIME: start_time,
        RUN_END_TIME: end_time,
        MODEL_PATH: model_path,
        HYPERPARAMETERS: hyperparameters,
        # Verbatim command that produced this stage - the exact invocation, argv-for-argv.
        COMMAND: list(sys.argv),
        ENVIRONMENT: capture_environment(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def add_stage(existing_metadata, stage, block):
    """Return a metadata dict with ``block`` recorded under ``stage``, preserving every other stage
    already present. This is the single place the "accumulate, don't overwrite" semantics live: a
    step reads whatever metadata the file already carries (or None for a fresh file), adds its own
    stage, and writes the result back - so provenance builds up across inference -> meta-model ->
    remap -> evaluation instead of each step clobbering the last."""
    metadata = dict(existing_metadata) if existing_metadata else {}
    metadata[stage] = block
    return metadata
