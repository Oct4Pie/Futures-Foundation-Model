import json
import hashlib

import pandas as pd

from scripts.prepare_ssl_corpus import build_corpus
from scripts.benchmark_ssl_checkpoints import _attest_run
from futures_foundation.finetune import ssl_data


def _day(path, start, contract, *, gap=False):
    ts = pd.date_range(start, periods=20, freq='1min', tz='UTC')
    if gap:
        ts = ts[:10].append(ts[10:] + pd.Timedelta('1min'))
    close = pd.Series(range(100, 120), dtype=float)
    pd.DataFrame({
        'timestamp': ts, 'open': close, 'high': close + 1, 'low': close - 1,
        'close': close + 0.25, 'volume': 10, 'contract_id': contract,
    }).to_parquet(path, index=False)


def test_build_corpus_is_sealed_causal_and_preserves_segments(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    src = source / 'F.US.ES' / '1m'
    src.mkdir(parents=True)
    _day(src / '2024-01-01.parquet', '2024-01-01', 'H24', gap=True)
    _day(src / '2024-01-02.parquet', '2024-01-02', 'M24')
    report = build_corpus(source, output, roots=('ES',), timeframes=(1, 5), verbose=False)
    assert report['purpose'].startswith('self-supervised')
    assert len(report['source_snapshot_sha256']) == 64
    assert report['roots_report']['ES']['contract_change_edges'] == 1
    assert report['roots_report']['ES']['one_minute_gap_edges'] >= 2  # intraday + overnight
    assert (output / 'ES_1min.csv').exists() and (output / 'ES_5min.csv').exists()
    assert json.loads((output / 'MANIFEST.json').read_text())['resample']['forward_fill'] is False

    streams = ssl_data.load_ohlcv(output, ['ES'], ['1min'], verbose=False)
    stream = streams[0]
    assert set(stream['contract_id']) == {'H24', 'M24'}
    starts = ssl_data.window_starts(range(len(stream['ts'])), 8, timestamps=stream['ts'],
                                    expected_delta='1min', segment_ids=stream['contract_id'])
    # A valid window can live inside either segment, but none may span the missing minute,
    # overnight boundary, or H24->M24 roll.
    for s in starts:
        assert len(set(stream['contract_id'][s:s + 8])) == 1
        delta = pd.DatetimeIndex(stream['ts'][s:s + 8]).to_series().diff().dropna()
        assert (delta == pd.Timedelta('1min')).all()


def test_strict_probe_attestation_is_hash_bound_and_stage_gated(tmp_path):
    ckpt = tmp_path / 'stage1.pt'
    ckpt.write_bytes(b'valid checkpoint bytes')
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    run_path = tmp_path / 'stage1.pt.run.json'
    run_path.write_text(json.dumps({
        'status': 'complete', 'stage': 'mask', 'output': str(ckpt),
        'val_start': '2024-01-01', 'holdout_start': '2025-07-01',
    }))
    report_path = tmp_path / 'benchmark.json'
    report = {
        'config': {'val_start': '2024-01-01', 'holdout_start': '2025-07-01'},
        'results': {ckpt.name: {
            'path': str(ckpt), 'sha256': digest,
            'probe': {'mean_core_delta': 0.01, 'descriptive_delta': 0.02,
                      'fwd_absmove_delta': -0.01, 'fwd_dir_delta': -0.01},
        }},
    }
    report_path.write_text(json.dumps(report))
    assert _attest_run(run_path, report_path, report)
    attested = json.loads(run_path.read_text())['strict_probe']
    assert attested['protocol'] == 'expanding_walk_forward'
    assert attested['passed'] and attested['checkpoint_sha256'] == digest
