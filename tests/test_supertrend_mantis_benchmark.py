import numpy as np
import pandas as pd

from scripts.benchmark_supertrend_mantis import SuperTrendMantis


def _labeler_with_roll():
    n = 500
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC').to_numpy()
    close = np.linspace(100, 110, n)
    contract = np.where(np.arange(n) < 300, 'A', 'B')
    segment = (contract != contract[0]).astype(np.int32)
    b = dict(ts=ts, o=close, h=close + 1, l=close - 1, c=close,
             v=np.ones(n), atr=np.ones(n), st=np.ones(n), st_line=close,
             contract=contract, segment=segment)
    lab = SuperTrendMantis.__new__(SuperTrendMantis)
    lab._b = {('ES', '1min'): b}
    # 150 is clean; 200's outcome crosses the roll; 350's context crosses it.
    lab._signals = lambda _b, _lo, _hi: iter(((150, 1), (200, 1), (350, 1)))
    return lab, ts


def test_supertrend_labeler_purges_boundaries_and_contract_rolls():
    lab, ts = _labeler_with_roll()
    _, _, keys = lab.build(ts[0], ts[-1], None)
    assert [k[1] for k in keys] == [150]
    assert lab.label_end_times(keys)[0] == ts[150 + 1 + lab.VERT]

    # The clean candidate's forward label ends after this boundary, so it is purged.
    _, _, purged = lab.build(ts[0], ts[-1], ts[250])
    assert purged == []


def test_supertrend_context_is_strictly_backward_looking():
    lab, _ = _labeler_with_roll()
    key = ('ES@1min', 150, 1, 0.0, 0.0, 0.0, 0.0, 0.0)
    window = lab.mv_contexts([key])
    assert window.shape == (1, 5, lab.MV_SEQ)
    assert np.isclose(window[0, 3, -1], lab._b[('ES', '1min')]['c'][150])
    assert np.isclose(window[0, 3, 0],
                      lab._b[('ES', '1min')]['c'][150 - lab.MV_SEQ + 1])
