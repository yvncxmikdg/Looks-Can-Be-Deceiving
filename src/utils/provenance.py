"""Training-run provenance manifest.

Thin training/cluster-specific layer over ``modeling.utils.run_metadata`` (issue #183). That shared
util owns the vocabulary for provenance blocks -- timestamps, the loaded model path, hyperparameters,
the verbatim command, and the replication-relevant environment (git commit, python, platform, package
versions). We reuse it verbatim so a training run's provenance speaks the same language as the
inference/eval stages, and add only what's specific to a *training* run on a cluster: the git branch
and working-tree dirty state (a dirty tree means the run isn't reproducible from a commit alone), the
SLURM job identifiers, and who/where it ran. The assembled block is stamped as the ``training`` stage
so it can later be folded into a downstream predictions file's lineage.

The result is written to ``train_provenance.yaml`` in the run's output directory, early -- before any
data loading -- so even a run that dies on startup leaves a record. Every capture here is best-effort:
anything unresolved is recorded as ``None`` rather than failing the run, exactly as run_metadata does.
"""
import os
import getpass
import platform
import subprocess

import yaml

from modeling.utils.run_metadata import add_stage, build_run_metadata, STAGE_TRAINING

# git commands run from this file's directory, matching run_metadata's _git_commit -- any subdir of
# the work tree resolves the same repo, so we don't need to compute the repo root.
_GIT_CWD = os.path.dirname(os.path.abspath(__file__))

# SLURM environment variables worth recording to tie a run back to its scheduler job.
_SLURM_ENV_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                   "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")


def _run_git(git_args):
    try:
        return subprocess.check_output(["git"] + git_args, cwd=_GIT_CWD, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_git_details():
    """Branch and working-tree dirty state, on top of the commit hash run_metadata already captures.
    Training cares about these because a dirty tree can't be reconstructed from a commit; the
    inference/eval stages deliberately record only the commit, so this stays out of run_metadata."""
    porcelain = _run_git(["status", "--porcelain"])
    dirty_files = [line for line in porcelain.splitlines() if line.strip()] if porcelain is not None else []
    return {
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        # None (not False) when git state is unknowable, so it's never mistaken for a clean tree.
        "is_dirty": (len(dirty_files) > 0) if porcelain is not None else None,
        "dirty_files": dirty_files,
    }


def collect_slurm_info():
    """The SLURM job identifiers if this run is under the scheduler, else None (e.g. a local run)."""
    if "SLURM_JOB_ID" not in os.environ:
        return None
    return {key: os.environ[key] for key in _SLURM_ENV_KEYS if key in os.environ}


def _current_user():
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return None


def build_training_metadata(experiment_comment, args_dict, start_time, model_path=None):
    """Assemble the ``training`` provenance stage for a run.

    Reuses run_metadata.build_run_metadata for the shared core (command, environment, timestamps) and
    attaches the training/cluster extras. ``end_time`` is left open (None): the manifest is written at
    run start, and the run's completion is recorded by whatever monitors the job, not here.

    Returns a stage-indexed dict (``{"training": {...}}``) matching the shape produced elsewhere in
    the pipeline, ready to write out or fold into a downstream file's metadata.
    """
    block = build_run_metadata(
        model_path=model_path,
        hyperparameters=args_dict,
        start_time=start_time,
        end_time=None,
        extra={
            "experiment_comment": experiment_comment,
            "git_details": collect_git_details(),
            "slurm": collect_slurm_info(),
            "user": _current_user(),
            "hostname": platform.node(),
            "working_directory": os.getcwd(),
        },
    )
    return add_stage(None, STAGE_TRAINING, block)


def write_run_manifest(out_path, metadata, filename="train_provenance.yaml"):
    """Write the provenance metadata to ``filename`` under ``out_path`` and return the written path."""
    os.makedirs(out_path, exist_ok=True)
    manifest_path = os.path.join(out_path, filename)
    with open(manifest_path, "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False, default_flow_style=False)
    return manifest_path
