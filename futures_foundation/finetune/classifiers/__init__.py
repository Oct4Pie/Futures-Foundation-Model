"""Classifier plugin layer. Concrete Classifier backbones (torch-bearing) live in per-backbone
subpackages and are imported LAZILY via `finetune.classifier.get_classifier` (libomp isolation:
never import a backbone from the torch-free finetune parent). Each backbone module registers its
class with `@register_classifier(name)`.

This __init__ is the torch-free PLUGIN MANIFEST: it maps a classifier name -> the backbone package
to import (which self-registers), and names the default foundation backbone. Adding a backbone
(e.g. MOMENT) = add its package + one entry here; the generic harness (wf/produce/loop/tune) and
the Classifier interface never name a backbone.
"""
_PKG = __name__

LAZY_BACKBONES = {
    'mantis':        _PKG + '.mantis',
    'mantis_frozen': _PKG + '.mantis',
    'moment_frozen': _PKG + '.moment',
    'logistic':      _PKG + '.logistic',
}

# Default foundation backbone — whose BASE_CKPT a new strategy finetunes on top of. Config, not
# code: switch backbones here (or via the `backbone=` arg to base_backbone_ckpt / get_classifier).
DEFAULT_BACKBONE = 'mantis'
