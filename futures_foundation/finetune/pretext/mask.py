"""Stage-1 pretext: BERT-style masked modeling (UNCHANGED). Gate = REAL encodes regime/vol/
structure better than vanilla (mean_core_delta > margin) and doesn't collapse."""
from .base import PretextTask


class MaskTask(PretextTask):
    name, trainer = 'mask', 'train_ssl_mask'
    primary_targets = ('trend_eff', 'range_expand')
