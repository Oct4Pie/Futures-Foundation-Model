"""Concrete Classifier implementations (torch-bearing).

These modules import torch and are loaded LAZILY via
`futures_foundation.finetune.classifier.get_classifier` — never from the
torch-free `finetune` parent (libomp isolation). Each module registers its
class with `@register_classifier(name)`.
"""
