"""
Fold health monitoring — detects training pathologies after each fold and
prints actionable diagnosis with suggested config patches.

Seven signals checked after each fold:

  EARLY_EPOCH       best_epoch ≤ threshold (default 5)
                    → initialization anchor; model solved before training started
                    → suggest: increase LR by 3×, or reduce freeze_ratio

  WEIGHT_LOCK       feature importance cosine similarity vs previous fold > threshold
                    → model weights not adapting fold-to-fold
                    → suggest: add train_start sliding window (18 months)

  P80_DECLINE       P@0.80 declined for 2+ consecutive folds
                    → systematic regression, not one-off regime change
                    → suggest: add train_start sliding window to remaining folds

  VAL_TEST_GAP      val P@0.80 exceeds test P@0.80 by > threshold (default 10 ppts)
                    → model over-fit to validation window; test generalisation is poor
                    → suggest: reduce epochs or increase regularisation (focal_gamma)

  N_COLLAPSE        predicted positives above threshold drop > ratio vs prev fold
                    → model becoming over-conservative; trading frequency collapsing
                    → suggest: lower confidence threshold, check label distribution

  CONFIDENCE_FLAT   std of output confidences < threshold (default 0.05)
                    → model not discriminating; all outputs bunched near same value
                    → suggest: lower LR, check for feature scaling issues

  ZERO_SIGNAL_FOLD  predicted positives above threshold < min_signal_n (default 20)
                    → fold too thin for reliable P@80 estimate; results are noise
                    → suggest: widen fold date range or lower confidence threshold

Usage::

    from futures_foundation.finetune import FoldHealthMonitor
    monitor = FoldHealthMonitor()
    fold_results = run_finetune(..., health_monitor=monitor)
    monitor.summary()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HealthWarning:
    fold: str
    code: str        # EARLY_EPOCH | WEIGHT_LOCK | P80_DECLINE | VAL_TEST_GAP | N_COLLAPSE | CONFIDENCE_FLAT | ZERO_SIGNAL_FOLD
    severity: str    # warning | critical
    message: str
    suggestion: str


class FoldHealthMonitor:
    """
    Stateful health checker — call check() after each fold via run_finetune's
    health_monitor parameter.  After all folds, call summary() for a report.

    Parameters
    ----------
    early_epoch_threshold : int
        best_epoch at or below this triggers EARLY_EPOCH warning.
    weight_lock_threshold : float
        Cosine similarity of feature importance vectors between consecutive
        folds at or above this triggers WEIGHT_LOCK warning.
    p80_decline_window : int
        Number of consecutive P@80 declines before triggering P80_DECLINE.
    val_test_gap_threshold : float
        Val P@80 minus test P@80 above this triggers VAL_TEST_GAP warning.
    n_collapse_ratio : float
        Fractional drop in N-above-threshold between folds above this triggers
        N_COLLAPSE warning.  Only fires if the previous fold had viable N.
    conf_flat_threshold : float
        Confidence std below this triggers CONFIDENCE_FLAT warning.
    min_signal_n : int
        N-above-threshold below this triggers ZERO_SIGNAL_FOLD (critical).
        Also used as the minimum viable baseline for N_COLLAPSE comparisons.
    """

    def __init__(
        self,
        early_epoch_threshold: int = 5,
        weight_lock_threshold: float = 0.99,
        p80_decline_window: int = 2,
        val_test_gap_threshold: float = 0.10,
        n_collapse_ratio: float = 0.50,
        conf_flat_threshold: float = 0.05,
        min_signal_n: int = 20,
    ):
        self.early_epoch_threshold  = early_epoch_threshold
        self.weight_lock_threshold  = weight_lock_threshold
        self.p80_decline_window     = p80_decline_window
        self.val_test_gap_threshold = val_test_gap_threshold
        self.n_collapse_ratio       = n_collapse_ratio
        self.conf_flat_threshold    = conf_flat_threshold
        self.min_signal_n           = min_signal_n

        self._warnings: List[HealthWarning]          = []
        self._p80_history: List[tuple]               = []   # (fold_name, p80)
        self._prev_importance: Optional[np.ndarray]  = None
        self._prev_n_above: Optional[int]            = None

    # ------------------------------------------------------------------
    def check(
        self,
        fold_name: str,
        metrics: dict,
        fold_config: Optional[dict] = None,
    ) -> List[HealthWarning]:
        """
        Run all health checks for a completed fold.

        Parameters
        ----------
        fold_name : str
            Fold identifier, e.g. 'F1'.
        metrics : dict
            test_metrics dict from _train_fold — must include 'all_conf',
            'all_labels', and optionally 'best_epoch', 'feature_importance',
            'val_p80'.
        fold_config : dict, optional
            The fold definition dict (e.g. from the FOLDS list). Used to check
            whether train_start is already configured before suggesting it.

        Returns
        -------
        List[HealthWarning]
            Warnings found for this fold (also appended to self._warnings).
        """
        if metrics is None:
            return []

        fold_warnings: List[HealthWarning] = []

        p80 = self._compute_p80(metrics)
        self._p80_history.append((fold_name, p80))

        conf = np.array(metrics.get('all_conf', []))

        # ── 1. Early best epoch ──────────────────────────────────────────
        best_epoch = metrics.get('best_epoch')
        epochs_trained = metrics.get('epochs_trained')
        # Only fire if the model barely continued past best_epoch — if it ran
        # 10+ more epochs (e.g. full p80_patience), early peaking is a natural
        # result of the warm start / gamma dynamics, not an anchor pathology.
        _early_epoch_stalled = (
            epochs_trained is None
            or (epochs_trained - best_epoch) < 10
        )
        if best_epoch is not None and best_epoch <= self.early_epoch_threshold and _early_epoch_stalled:
            w = HealthWarning(
                fold       = fold_name,
                code       = 'EARLY_EPOCH',
                severity   = 'warning',
                message    = (
                    f'{fold_name}: best_epoch={best_epoch} '
                    f'(≤ {self.early_epoch_threshold}) and training stalled early — '
                    'model solved before meaningful training; '
                    'likely anchored to continue_from initialization'
                ),
                suggestion = (
                    'Increase LR by 3×, '
                    'or reduce FREEZE_RATIO so more backbone layers update'
                ),
            )
            fold_warnings.append(w)

        # ── 2. Feature importance lock ───────────────────────────────────
        importance = metrics.get('feature_importance')
        if importance is not None and self._prev_importance is not None:
            sim  = _cosine_similarity(importance, self._prev_importance)
            l2   = float(np.linalg.norm(importance - self._prev_importance))
            prev_fold = self._p80_history[-2][0] if len(self._p80_history) >= 2 else '?'
            if sim >= self.weight_lock_threshold:
                best_epoch = metrics.get('best_epoch')
                early_conv = best_epoch is not None and best_epoch <= 15
                has_train_start = (
                    fold_config is not None and fold_config.get('train_start') is not None
                )
                if early_conv:
                    suggestion = (
                        f'Model converged early (best_epoch={best_epoch}) before '
                        'strategy weights had pressure to adapt. Options: '
                        '(1) increase LR by 2× for the strategy head; '
                        '(2) reduce FREEZE_RATIO to allow more backbone layers to update'
                    )
                    if not has_train_start:
                        suggestion += '; (3) add 18-month sliding window via train_start'
                else:
                    if has_train_start:
                        suggestion = (
                            f'train_start is already configured — regime drift may be the cause. '
                            'Options: (1) increase strategy_lr_multiplier; '
                            '(2) reduce FREEZE_RATIO so more backbone layers update'
                        )
                    else:
                        suggestion = (
                            f'Add train_start to fold {fold_name} and later folds '
                            '(18-month sliding window): '
                            'train_start = train_end minus 18 months'
                        )
                w = HealthWarning(
                    fold       = fold_name,
                    code       = 'WEIGHT_LOCK',
                    severity   = 'warning',
                    message    = (
                        f'{fold_name}: feature importance cos_sim={sim:.4f} L2={l2:.4f} '
                        f'vs {prev_fold} (cos_sim ≥ {self.weight_lock_threshold}) — '
                        'feature attention pattern not shifting fold-to-fold'
                    ),
                    suggestion = suggestion,
                )
                fold_warnings.append(w)
            else:
                print(f'  ✅ Feature weights diverged vs {prev_fold}: '
                      f'cos_sim={sim:.4f} L2={l2:.4f} '
                      f'(cos_sim < {self.weight_lock_threshold} — adapting normally)')
        if importance is not None:
            self._prev_importance = importance.copy()

        # ── 3. Consecutive P@80 decline ──────────────────────────────────
        if len(self._p80_history) >= self.p80_decline_window + 1:
            recent = [p for _, p in self._p80_history[-(self.p80_decline_window + 1):]]
            if all(recent[i] > recent[i + 1] for i in range(self.p80_decline_window)):
                decline_folds = [fn for fn, _ in self._p80_history[-(self.p80_decline_window + 1):]]
                first_val = recent[0]
                last_val  = recent[-1]
                w = HealthWarning(
                    fold       = fold_name,
                    code       = 'P80_DECLINE',
                    severity   = 'critical',
                    message    = (
                        f'{fold_name}: P@80 declined for '
                        f'{self.p80_decline_window} consecutive folds '
                        f'({" → ".join(decline_folds)}: '
                        f'{first_val:.1%} → {last_val:.1%})'
                    ),
                    suggestion = (
                        'train_start is already configured — this is likely regime drift. '
                        'Options: (1) increase strategy_lr_multiplier to force feature re-weighting; '
                        '(2) reduce FREEZE_RATIO to allow backbone to adapt more; '
                        '(3) accept as regime headwind if val/test gap is small'
                        if fold_config is not None and fold_config.get('train_start') is not None
                        else
                        'Add train_start to remaining folds '
                        '(18-month sliding window). Example: '
                        'if train_end=2025-04-01, set train_start=2023-10-01'
                    ),
                )
                fold_warnings.append(w)

        # ── 4. Val / test P@80 gap ───────────────────────────────────────
        val_p80 = metrics.get('val_p80')
        if val_p80 is not None:
            gap = val_p80 - p80
            if gap > self.val_test_gap_threshold:
                w = HealthWarning(
                    fold       = fold_name,
                    code       = 'VAL_TEST_GAP',
                    severity   = 'warning',
                    message    = (
                        f'{fold_name}: val P@80={val_p80:.1%} vs test P@80={p80:.1%} '
                        f'(gap={gap:.1%} > {self.val_test_gap_threshold:.0%}) — '
                        'model over-fit to validation window'
                    ),
                    suggestion = (
                        'Reduce max epochs, increase focal_gamma, '
                        'or extend val window to cover more of the regime'
                    ),
                )
                fold_warnings.append(w)

        # ── Compute N above 0.80 (shared by N_COLLAPSE and ZERO_SIGNAL) ──
        # Use pre-computed n_at_80 from trainer (applies (conf>=0.80)&(pred>0) mask)
        # to match the threshold table definition exactly.
        n_above = int(metrics['n_at_80']) if 'n_at_80' in metrics else (
            int((conf >= 0.80).sum()) if len(conf) > 0 else 0
        )

        # ── 5. Confidence flat ───────────────────────────────────────────
        if len(conf) > 0 and float(np.std(conf)) < self.conf_flat_threshold:
            w = HealthWarning(
                fold       = fold_name,
                code       = 'CONFIDENCE_FLAT',
                severity   = 'warning',
                message    = (
                    f'{fold_name}: confidence std={np.std(conf):.4f} '
                    f'(< {self.conf_flat_threshold}) — '
                    'model not discriminating; all outputs bunched near same value'
                ),
                suggestion = (
                    'Check feature scaling; confirm signals are non-trivially separable; '
                    'try lower LR or longer training'
                ),
            )
            fold_warnings.append(w)

        # ── 6. Zero signal fold ──────────────────────────────────────────
        if n_above < self.min_signal_n:
            w = HealthWarning(
                fold       = fold_name,
                code       = 'ZERO_SIGNAL_FOLD',
                severity   = 'critical',
                message    = (
                    f'{fold_name}: only {n_above} predicted positives above 0.80 '
                    f'(< {self.min_signal_n} minimum) — '
                    'P@80 estimate is unreliable noise'
                ),
                suggestion = (
                    'Widen fold date range to increase test signal count, '
                    'or lower the confidence reporting threshold'
                ),
            )
            fold_warnings.append(w)

        # ── 7. N collapse ────────────────────────────────────────────────
        # Only meaningful when the previous fold had viable N as a baseline.
        if (
            self._prev_n_above is not None
            and self._prev_n_above >= self.min_signal_n
            and n_above < self._prev_n_above * (1 - self.n_collapse_ratio)
        ):
            prev_fold = self._p80_history[-2][0] if len(self._p80_history) >= 2 else '?'
            drop_pct  = 1 - n_above / self._prev_n_above
            w = HealthWarning(
                fold       = fold_name,
                code       = 'N_COLLAPSE',
                severity   = 'warning',
                message    = (
                    f'{fold_name}: N@0.80={n_above} vs {prev_fold} N={self._prev_n_above} '
                    f'(drop={drop_pct:.0%} > {self.n_collapse_ratio:.0%}) — '
                    'model becoming over-conservative; trading frequency collapsing'
                ),
                suggestion = (
                    'Lower deployment confidence threshold, '
                    'or check if label distribution shifted dramatically between folds'
                ),
            )
            fold_warnings.append(w)
        self._prev_n_above = n_above

        # ── Print immediately ────────────────────────────────────────────
        if fold_warnings:
            print(f'\n  {"=" * 58}')
            print(f'  FOLD HEALTH MONITOR — {fold_name}')
            print(f'  {"=" * 58}')
            for w in fold_warnings:
                icon = '🔴' if w.severity == 'critical' else '🟡'
                print(f'  {icon} [{w.code}] {w.message}')
                print(f'     → {w.suggestion}')
            print(f'  {"=" * 58}')

        self._warnings.extend(fold_warnings)
        return fold_warnings

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a consolidated summary of all warnings across all folds."""
        print(f'\n{"=" * 60}')
        print('  FOLD HEALTH SUMMARY')
        print(f'{"=" * 60}')

        if not self._warnings:
            print('  ✅ No health issues detected across all folds')
            print(f'{"=" * 60}')
            return

        codes_seen = {}
        for w in self._warnings:
            codes_seen.setdefault(w.code, []).append(w.fold)

        for code, folds in codes_seen.items():
            severity = next(w.severity for w in self._warnings if w.code == code)
            icon = '🔴' if severity == 'critical' else '🟡'
            print(f'  {icon} {code}: detected on {", ".join(folds)}')

        # P@80 trend table
        if self._p80_history:
            print(f'\n  P@80 per fold:')
            prev = None
            for fn, p80 in self._p80_history:
                arrow = ''
                if prev is not None:
                    arrow = '▲' if p80 > prev else ('▼' if p80 < prev else '—')
                print(f'    {fn}: {p80:.1%}  {arrow}')
                prev = p80

        # Consolidated suggestions
        if 'WEIGHT_LOCK' in codes_seen or 'P80_DECLINE' in codes_seen:
            print(
                '\n  Primary fix: see per-fold WEIGHT_LOCK suggestion above\n'
                '  (early convergence → raise LR; no train_start → add sliding window)'
            )
        if 'EARLY_EPOCH' in codes_seen and 'WEIGHT_LOCK' not in codes_seen:
            print('\n  Primary fix: increase LR or reduce FREEZE_RATIO')
        if 'VAL_TEST_GAP' in codes_seen:
            print('\n  Primary fix: reduce max epochs or increase focal_gamma')
        if 'CONFIDENCE_FLAT' in codes_seen:
            print('\n  Primary fix: check feature scaling or lower LR')
        if 'ZERO_SIGNAL_FOLD' in codes_seen:
            print('\n  Primary fix: widen fold date ranges or lower confidence threshold')
        if 'N_COLLAPSE' in codes_seen:
            print('\n  Primary fix: check label distribution shift between folds')

        print(f'{"=" * 60}')

    # ------------------------------------------------------------------
    @property
    def warnings(self) -> List[HealthWarning]:
        return list(self._warnings)

    def has_critical(self) -> bool:
        return any(w.severity == 'critical' for w in self._warnings)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_p80(metrics: dict) -> float:
        # prec_at_80 is pre-computed by the trainer with the correct
        # (conf >= 0.80) & (pred > 0) mask — same definition as the threshold table.
        if 'prec_at_80' in metrics:
            return float(metrics['prec_at_80'])
        conf   = np.array(metrics.get('all_conf', []))
        labels = np.array(metrics.get('all_labels', []))
        preds  = np.array(metrics.get('all_preds', []))
        if len(conf) == 0:
            return 0.0
        if len(preds) == len(conf):
            mask = (conf >= 0.80) & (preds > 0)
        else:
            mask = conf >= 0.80
        return float((labels[mask] > 0).mean()) if mask.sum() > 0 else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0
