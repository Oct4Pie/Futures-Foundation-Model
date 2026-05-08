"""
Fold health monitoring — detects training pathologies after each fold and
prints actionable diagnosis with suggested config patches.

Three signals checked after each fold:

  EARLY_EPOCH   best_epoch ≤ threshold (default 5)
                → initialization anchor; model solved before training started
                → suggest: increase LR by 3×, or reduce freeze_ratio

  WEIGHT_LOCK   feature importance cosine similarity vs previous fold > threshold
                → model weights not adapting fold-to-fold
                → suggest: add train_start sliding window (18 months)

  P80_DECLINE   P@0.80 declined for 2+ consecutive folds
                → systematic regression, not one-off regime change
                → suggest: add train_start sliding window to remaining folds

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
    code: str        # EARLY_EPOCH | WEIGHT_LOCK | P80_DECLINE
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
    """

    def __init__(
        self,
        early_epoch_threshold: int = 5,
        weight_lock_threshold: float = 0.99,
        p80_decline_window: int = 2,
    ):
        self.early_epoch_threshold = early_epoch_threshold
        self.weight_lock_threshold = weight_lock_threshold
        self.p80_decline_window    = p80_decline_window

        self._warnings: List[HealthWarning] = []
        self._p80_history: List[tuple]      = []   # (fold_name, p80)
        self._prev_importance: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def check(self, fold_name: str, metrics: dict) -> List[HealthWarning]:
        """
        Run all health checks for a completed fold.

        Parameters
        ----------
        fold_name : str
            Fold identifier, e.g. 'F1'.
        metrics : dict
            test_metrics dict from _train_fold — must include 'all_conf',
            'all_labels', and optionally 'best_epoch', 'feature_importance'.

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

        # ── 1. Early best epoch ──────────────────────────────────────────
        best_epoch = metrics.get('best_epoch')
        if best_epoch is not None and best_epoch <= self.early_epoch_threshold:
            w = HealthWarning(
                fold       = fold_name,
                code       = 'EARLY_EPOCH',
                severity   = 'warning',
                message    = (
                    f'{fold_name}: best_epoch={best_epoch} '
                    f'(≤ {self.early_epoch_threshold}) — '
                    'model solved before meaningful training; '
                    'likely anchored to continue_from initialization'
                ),
                suggestion = (
                    'Increase LR by 3× (e.g. 5e-5 → 1.5e-4), '
                    'or reduce FREEZE_RATIO so more backbone layers update'
                ),
            )
            fold_warnings.append(w)

        # ── 2. Feature importance lock ───────────────────────────────────
        importance = metrics.get('feature_importance')
        if importance is not None and self._prev_importance is not None:
            sim = _cosine_similarity(importance, self._prev_importance)
            if sim >= self.weight_lock_threshold:
                prev_fold = self._p80_history[-2][0] if len(self._p80_history) >= 2 else '?'
                w = HealthWarning(
                    fold       = fold_name,
                    code       = 'WEIGHT_LOCK',
                    severity   = 'warning',
                    message    = (
                        f'{fold_name}: feature importance cos_sim={sim:.4f} '
                        f'vs {prev_fold} (≥ {self.weight_lock_threshold}) — '
                        'weights not adapting fold-to-fold'
                    ),
                    suggestion = (
                        f'Add train_start to fold {fold_name} and later folds '
                        '(18-month sliding window): '
                        'train_start = train_end minus 18 months'
                    ),
                )
                fold_warnings.append(w)
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
                        'Add train_start to remaining folds '
                        '(18-month sliding window). Example: '
                        'if train_end=2025-04-01, set train_start=2023-10-01'
                    ),
                )
                fold_warnings.append(w)

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

        # Consolidated suggestion
        if 'WEIGHT_LOCK' in codes_seen or 'P80_DECLINE' in codes_seen:
            print(
                '\n  Primary fix: add train_start to fold dicts\n'
                '  (18-month sliding window — see suggestions above)'
            )
        if 'EARLY_EPOCH' in codes_seen and 'WEIGHT_LOCK' not in codes_seen:
            print('\n  Primary fix: increase LR or reduce FREEZE_RATIO')

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
        conf = np.array(metrics.get('all_conf', []))
        labels = np.array(metrics.get('all_labels', []))
        if len(conf) == 0:
            return 0.0
        mask = conf >= 0.80
        return float((labels[mask] > 0).mean()) if mask.sum() > 0 else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0
