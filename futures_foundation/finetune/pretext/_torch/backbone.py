"""Version-aware Mantis backbone loading shared by SSL, probes, and downstream heads."""


def resolve_mantis_version(model_id, model_version=None):
    """Return ``v1`` or ``v2`` and reject ambiguous/unsupported versions."""
    normalized = str(model_id).replace('-', '').replace('_', '').lower()
    inferred = ('v2' if 'mantisv2' in normalized else
                'v1' if ('mantis8m' in normalized or 'mantisplus' in normalized) else None)
    if model_version is not None:
        version = str(model_version).strip().lower()
    else:
        version = inferred or 'v1'
    if version not in {'v1', 'v2'}:
        raise ValueError(f"model_version must be 'v1' or 'v2', got {model_version!r}")
    if inferred is not None and version != inferred:
        raise ValueError(f"model_id {model_id!r} is {inferred} but model_version={version!r}")
    return version


def load_mantis(model_id='paris-noah/Mantis-8M', *, model_version=None, device='cpu'):
    """Load the requested pretrained architecture instead of forcing every ID into MantisV1.

    MantisV1 and MantisV2 checkpoints are not state-compatible.  V2's package API is instance-
    based, so instantiate the matching class first and then call ``from_pretrained``.
    """
    from mantis.architecture import Mantis8M, MantisV2

    version = resolve_mantis_version(model_id, model_version)
    if version == 'v2':
        return MantisV2(device=device).from_pretrained(model_id)
    # Preserve the legacy V1 module names (notably ``vit_unit``) for existing checkpoints,
    # downstream code, and bundles. V2 uses its distinct ``transf_unit`` architecture.
    return Mantis8M.from_pretrained(model_id).to(device)
