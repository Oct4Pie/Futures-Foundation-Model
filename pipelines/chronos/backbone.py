"""Chronos-Bolt backbone seam — load, pristine reset, causal pooling.

The ONLY module that touches the Chronos library, so everything else is
backbone-swappable and strategy-agnostic. `fresh_model()` resets the
backbone to its pretrained weights so each fine-tune is independent
(REAL / SHUFFLE / seeds never contaminate one another).
"""
MODEL = 'amazon/chronos-bolt-tiny'
_PIPE = None
_PRISTINE = None                       # cloned pretrained state_dict


def pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained(
            MODEL, device_map='cpu', dtype=torch.float32)
    return _PIPE


def _model():
    p = pipeline()
    return getattr(p, 'inner_model', None) or p.model


def d_model():
    return _model().config.d_model


def fresh_model():
    """The backbone reset to pretrained weights (independent fine-tunes)."""
    global _PRISTINE
    m = _model()
    if _PRISTINE is None:
        _PRISTINE = {k: v.detach().clone()
                     for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(_PRISTINE)
    return m


def pool(m, ctx):
    """Masked-mean pool of the encoder hidden states. ctx: [B,L] tensor of
    causal log-price context (bars <= decision t — the caller's contract)."""
    h, _ls, _emb, mask = m.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)
