"""Stage-1 pretext: BERT-style masked modeling (UNCHANGED). Gate = REAL encodes regime/vol/
structure better than vanilla (mean_core_delta > margin) and doesn't collapse."""
from .base import PretextTask


class MaskTask(PretextTask):
    name, trainer = 'mask', 'train_ssl_mask'

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        return bool(probe_res['mean_core_delta'] > margin and no_collapse), detail
