"""DOWNSTREAM METRICS LEDGER — JSONL append with numpy sanitization (produce._ledger_append)."""
import json
import os

import numpy as np


def test_ledger_appends_jsonl_with_numpy_sanitized(tmp_path, monkeypatch):
    from futures_foundation.finetune.produce import _ledger_append
    p = tmp_path / 'ledger.jsonl'
    monkeypatch.setenv('RUN_LEDGER', str(p))
    rec = {'ts': '2026-07-16T00:00:00', 'kind': 'produce_streamed',
           'oos_auc': np.float64(0.5451), 'n_train': np.int64(1283214),
           'ops': [{'rate': np.int32(2), 'wr3R': np.float32(0.649)}],
           'per_ticker': {np.str_('NQ'): [np.float64(0.587)]},
           'arr': np.array([1.0, 2.0])}
    for _ in range(2):                                    # append-only: two runs -> two lines
        _ledger_append(dict(rec))
    lines = p.read_text().strip().split('\n')
    assert len(lines) == 2
    back = json.loads(lines[0])
    assert back['oos_auc'] == 0.5451 and back['n_train'] == 1283214
    assert back['per_ticker']['NQ'] == [0.587] and back['arr'] == [1.0, 2.0]


def test_ledger_is_opt_in_only(tmp_path, monkeypatch):
    """2026-07-17 (user directive): NO ledger unless RUN_LEDGER is set — the old
    output_path fallback wrote run_ledger.jsonl on every run, unwanted by default."""
    from futures_foundation.finetune.produce import _ledger_append
    monkeypatch.delenv('RUN_LEDGER', raising=False)
    out = tmp_path / 'exp' / 'model'
    assert _ledger_append({'kind': 'produce_streamed', 'oos_auc': 0.5}, str(out)) is None
    assert not (tmp_path / 'exp' / 'run_ledger.jsonl').exists()   # output_path alone -> NO file
    assert _ledger_append({'k': 1}, None) is None         # no path anywhere -> skip, no raise
