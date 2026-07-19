"""PretextTask — base class for a pluggable SSL pretraining objective (a "stage").

Subclass to add a new pretrain experiment WITHOUT editing the orchestrator (ssl.py). Each task
owns four things: how much window to RESERVE per window, how to TRAIN (which _ssl_torch trainer),
its report-only GATE on the probe, and any pretext-specific verdict fields. Trainers swallow
unknown kwargs (**_ignore), so the shared cfg is safe to pass to any task.
"""


class PretextTask:
    name = 'base'
    trainer = None                                        # _ssl_torch trainer fn name
    # One promotion schema for every stage. Subclasses declare which targets must improve;
    # every listed target must at least be non-inferior to vanilla independently.
    noninferiority_targets = ('vol', 'trend_eff', 'range_expand', 'fwd_absmove')
    diagnostic_targets = ('direction', 'fwd_dir')
    primary_targets = ()
    min_consistent_fold_fraction = 0.6
    target_semantics_version = None

    def reserve(self, cfg):
        """Total parent-window length required by the task (0 = use seq+max_jitter)."""
        return 0

    def train(self, big, tr, va, cfg, control):
        """-> (best_encoder_state, history) under a control ('real'|'shuffle'|'random')."""
        # Resolve through sys.modules on every dispatch. Besides keeping import lazy, this avoids
        # a stale package attribute defeating controlled module replacement in tests/plugins.
        import importlib
        _ssl_torch = importlib.import_module('futures_foundation.finetune._ssl_torch')
        kw = {k: v for k, v in cfg.items() if k != 'pretext'}
        return getattr(_ssl_torch, self.trainer)(big, tr, va, control=control, **kw)

    def gate(self, probe_res, std, margin, dir_margin):
        """Single promotion gate: per-target means + fold consistency, never mixed-metric sums."""
        no_collapse = bool(std > 0.01)
        detail = {'gate_schema': 'ffm_ssl_promotion_v3', 'no_collapse': no_collapse,
                  'target_semantics_version': self.target_semantics_version,
                  'primary_targets': list(self.primary_targets),
                  'noninferiority_targets': list(self.noninferiority_targets),
                  'diagnostic_targets': list(self.diagnostic_targets),
                  'min_consistent_fold_fraction': self.min_consistent_fold_fraction}
        if probe_res is None:
            return False, {**detail, 'probe': None, 'reason': 'strict probe required'}
        detail.update({'mean_core_delta': float(probe_res['mean_core_delta']),
                       'descriptive_delta': float(probe_res.get('descriptive_delta', 0.0)),
                       'fwd_absmove_delta': float(probe_res.get('fwd_absmove_delta', 0.0)),
                       'fwd_dir_delta': float(probe_res.get('fwd_dir_delta', 0.0)),
                       'forward_score': float(probe_res.get('forward_score', 0.0)),
                       'learns_regime_vol_structure': bool(probe_res['learns_regime_vol_structure'])})
        per_target = probe_res.get('per_target') or {}
        missing = [name for name in self.noninferiority_targets if name not in per_target]
        if missing:
            return False, {**detail, 'missing_targets': missing}
        checks = {}
        for name in self.noninferiority_targets:
            result = per_target[name]
            limit = float(dir_margin if name == 'fwd_dir' else 0.0)
            fold_delta = result.get('fold_delta') or [float(result['delta'])]
            fraction = sum(float(x) >= limit for x in fold_delta) / len(fold_delta)
            checks[name] = {
                'delta': float(result['delta']), 'noninferiority_limit': limit,
                'mean_noninferior': bool(float(result['delta']) >= limit),
                'consistent_fold_fraction': float(fraction),
                'fold_consistent': bool(fraction >= self.min_consistent_fold_fraction),
            }
        diagnostic_checks = {}
        for name in self.diagnostic_targets:
            if name not in per_target:
                continue
            result = per_target[name]
            fold_delta = result.get('fold_delta') or [float(result['delta'])]
            diagnostic_checks[name] = {
                'delta': float(result['delta']),
                'consistent_nonnegative_fraction': float(
                    sum(float(x) >= 0 for x in fold_delta) / len(fold_delta)),
                'gated': False,
            }
        primary_checks = {}
        for name in self.primary_targets:
            gain = float(dir_margin if name == 'fwd_dir' else margin)
            result = per_target[name]
            fold_delta = result.get('fold_delta') or [float(result['delta'])]
            fraction = sum(float(x) > gain for x in fold_delta) / len(fold_delta)
            primary_checks[name] = {
                'gain_margin': gain, 'mean_gain': bool(float(result['delta']) > gain),
                'consistent_gain_fraction': float(fraction),
                'fold_consistent': bool(fraction >= self.min_consistent_fold_fraction),
            }
        noninferior = all(x['mean_noninferior'] and x['fold_consistent'] for x in checks.values())
        primary = all(x['mean_gain'] and x['fold_consistent'] for x in primary_checks.values())
        detail.update({
            'per_target_checks': checks, 'diagnostic_target_checks': diagnostic_checks,
            'primary_checks': primary_checks,
            'per_target_noninferior': bool(noninferior), 'primary_gains': bool(primary),
            # Backward-compatible report fields; these are now derived from per-target checks.
            'descriptive_ok': all(checks[x]['mean_noninferior'] for x in
                                  ('vol', 'trend_eff', 'range_expand')),
            'fwd_size_ok': checks['fwd_absmove']['mean_noninferior'],
            'fwd_dir_ok': bool((diagnostic_checks.get('fwd_dir') or {}).get('delta', 0.0)
                               >= float(dir_margin)),
        })
        return bool(no_collapse and noninferior and primary), detail

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        """Add any pretext-specific fields to the saved verdict (default: none)."""
        return verdict
