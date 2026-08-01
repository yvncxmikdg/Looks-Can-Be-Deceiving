"""Shared translation from a PyTorch Lightning ``--accelerator`` string to a torch device.

Across the modeling entrypoints ``--accelerator`` is a *Lightning accelerator* string:
``train.py`` and ``infer.py`` forward it straight into ``Trainer(accelerator=...)``, which speaks
``cpu``/``gpu``/``mps``/``tpu``/``auto``. That is the vocabulary every launcher standardizes on.

Some code paths, however, need a raw *torch device* instead of a Lightning Trainer -- e.g.
``calibrate_logit_temperature.py`` runs its own manual ``task_module.to(device)`` /
``torch.randperm(..., device=device)`` loop, and ``train.py`` hands a device to the
``MixedPrecision`` plugin. Torch speaks ``cpu``/``cuda``/``mps`` and raises on ``gpu`` (``gpu`` is
not a valid torch device), so passing the Lightning string straight through breaks.

``resolve_torch_device`` is that single translation point, so one ``--accelerator`` value works
everywhere and launchers no longer have to pass ``cuda`` to one script and ``gpu`` to its siblings.
See issue #215.
"""
import torch

# Lightning accelerator string -> torch device string. ``cuda`` is accepted as an alias of the
# Lightning ``gpu`` so that a launcher (or habit) still passing the torch spelling keeps working.
_ACCELERATOR_TO_TORCH_DEVICE = {
    "cpu": "cpu",
    "gpu": "cuda",
    "cuda": "cuda",
    "mps": "mps",
}


def _resolve_auto():
    """Resolve Lightning's ``auto`` to a concrete torch device, mirroring its selection order."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def resolve_torch_device(accelerator):
    """Translate a Lightning-style ``--accelerator`` string into a torch device string.

    Accepts the same vocabulary the Lightning ``Trainer(accelerator=...)`` used by ``train.py`` /
    ``infer.py`` does -- ``cpu``, ``gpu``, ``mps``, ``auto`` (plus ``cuda`` as an alias of ``gpu``) --
    and returns a value torch's ``.to(device)`` accepts (``cpu`` / ``cuda`` / ``mps``). ``auto``
    resolves to cuda if available, then mps, else cpu.

    Raises ``ValueError`` for anything else (e.g. ``tpu``, which has no plain torch device string),
    so a typo fails loudly here rather than deep inside a ``.to(device)`` call.
    """
    key = str(accelerator).strip().lower()
    if key == "auto":
        return _resolve_auto()
    if key not in _ACCELERATOR_TO_TORCH_DEVICE:
        supported = ", ".join(sorted(set(_ACCELERATOR_TO_TORCH_DEVICE) | {"auto"}))
        raise ValueError(
            f"Unsupported accelerator '{accelerator}'. Expected a Lightning accelerator string "
            f"(one of: {supported})."
        )
    return _ACCELERATOR_TO_TORCH_DEVICE[key]
