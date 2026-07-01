"""PretextTask — base class for a pluggable SSL pretraining objective (a "stage").

Subclass to add a new pretrain experiment WITHOUT editing the orchestrator (ssl.py). Each task
owns four things: how much window to RESERVE per window, how to TRAIN (which _ssl_torch trainer),
its report-only GATE on the probe, and any pretext-specific verdict fields. Trainers swallow
unknown kwargs (**_ignore), so the shared cfg is safe to pass to any task.
"""


class PretextTask:
    name = 'base'
    trainer = None                                        # _ssl_torch trainer fn name

    def reserve(self, cfg):
        """Extra bars reserved per window beyond `seq` (0 = none)."""
        return 0

    def train(self, big, tr, va, cfg, control):
        """-> (best_encoder_state, history) under a control ('real'|'shuffle'|'random')."""
        from .. import _ssl_torch                          # lazy: keep module load torch-free
        kw = {k: v for k, v in cfg.items() if k != 'pretext'}
        return getattr(_ssl_torch, self.trainer)(big, tr, va, control=control, **kw)

    def gate(self, probe_res, std, margin, dir_margin):
        """Report-only gate on the PROBE (representation content), NOT the loss. Builds the shared
        detail dict, then defers the pass/fail to `_decide`."""
        no_collapse = bool(std > 0.01)
        detail = {'no_collapse': no_collapse}
        if probe_res is None:
            return no_collapse, {**detail, 'probe': None}
        detail.update({'mean_core_delta': float(probe_res['mean_core_delta']),
                       'descriptive_delta': float(probe_res.get('descriptive_delta', 0.0)),
                       'fwd_absmove_delta': float(probe_res.get('fwd_absmove_delta', 0.0)),
                       'fwd_dir_delta': float(probe_res.get('fwd_dir_delta', 0.0)),
                       'forward_score': float(probe_res.get('forward_score', 0.0)),
                       'learns_regime_vol_structure': bool(probe_res['learns_regime_vol_structure'])})
        return self._decide(probe_res, no_collapse, margin, dir_margin, detail)

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        raise NotImplementedError

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        """Add any pretext-specific fields to the saved verdict (default: none)."""
        return verdict
