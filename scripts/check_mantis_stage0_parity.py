#!/usr/bin/env python3
"""Verify MantisV2 training/deployment embedding parity on sealed futures windows.

This is a contract test, not a performance benchmark. It compares the exact clean-input path used
inside SSL with the frozen Python deployment path, a versioned deployment bundle, repeat/batch
extraction, and (for the 256-bar deployment context) ONNX Runtime. No future bars are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from futures_foundation.finetune.pretext._torch.common import (
    embed_windows, export_encoder_onnx, make_deployment_bundle, preprocess_windows,
)
from futures_foundation.finetune.pretext._torch.contrastive import ContrastiveTrendNet


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _difference(reference, candidate):
    delta = np.abs(np.asarray(reference) - np.asarray(candidate))
    return {'max_abs': float(delta.max(initial=0.0)),
            'mean_abs': float(delta.mean() if delta.size else 0.0)}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def run(args):
    artifact = Path(args.windows).resolve()
    with np.load(artifact, allow_pickle=False) as data:
        raw = np.asarray(data['context'][:args.samples], np.float32)
    if raw.ndim != 3:
        raise ValueError('context must have shape [rows,time,channels]')
    windows = np.transpose(raw, (0, 2, 1)).copy()
    if windows.shape[1] != 5 or windows.shape[2] < max(args.contexts):
        raise ValueError('sealed artifact must contain five channels and all requested contexts')

    device = args.device
    net = ContrastiveTrendNet(
        C=5, model_id='paris-noah/MantisV2', model_version='v2').to(device).eval()
    state = {key: value.detach().cpu() for key, value in net.encoder.state_dict().items()}
    results = {}
    with tempfile.TemporaryDirectory(prefix='mantis-stage0-') as tmpdir:
        tmpdir = Path(tmpdir)
        legacy = tmpdir / 'encoder.pt'
        torch.save(state, legacy)
        for length in args.contexts:
            current = windows[:, :, -int(length):]
            tensor = torch.from_numpy(current).to(device)
            with torch.no_grad():
                training_clean = net.embed(
                    preprocess_windows(tensor, args.preprocessing)).float().cpu().numpy()
            bundle = tmpdir / f'encoder-{length}.bundle.pt'
            torch.save(make_deployment_bundle(
                state, model_id='paris-noah/MantisV2', model_version='v2', channels=5,
                train_context_lengths=[int(length)], preprocessing=args.preprocessing), bundle)
            python_legacy = embed_windows(
                current, ckpt=legacy, model_id='paris-noah/MantisV2', model_version='v2',
                device=device, batch=args.samples, preprocessing=args.preprocessing)
            python_bundle = embed_windows(
                current, ckpt=bundle, model_id='paris-noah/MantisV2', model_version='v2',
                device=device, batch=args.samples)
            repeat = embed_windows(
                current, ckpt=bundle, model_id='paris-noah/MantisV2', model_version='v2',
                device=device, batch=args.samples)
            singleton = embed_windows(
                current, ckpt=bundle, model_id='paris-noah/MantisV2', model_version='v2',
                device=device, batch=1)
            checks = {
                'training_vs_python_legacy': _difference(training_clean, python_legacy),
                'training_vs_python_bundle': _difference(training_clean, python_bundle),
                'bundle_repeat_determinism': _difference(python_bundle, repeat),
                # GPU GEMM reduction order changes with batch shape. This is a bounded numerical
                # sensitivity diagnostic, not bitwise determinism; CPU is expected to be exact.
                'bundle_batch_vs_singleton': _difference(python_bundle, singleton),
            }
            results[str(length)] = {'embedding_shape': list(training_clean.shape),
                                    'checks': checks}

        deployment = windows[:, :, -256:]
        bundle = tmpdir / 'encoder-256.bundle.pt'
        onnx = tmpdir / 'encoder-256.onnx'
        export_encoder_onnx(
            onnx, ckpt=bundle, C=5, seq=256, model_id='paris-noah/MantisV2',
            model_version='v2', preprocessing=args.preprocessing)
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
        onnx_embedding = session.run(None, {'window': deployment})[0]
        python_embedding = embed_windows(
            deployment, ckpt=bundle, model_id='paris-noah/MantisV2', model_version='v2',
            device='cpu', batch=args.samples)
        results['256']['checks']['python_bundle_vs_onnx'] = _difference(
            python_embedding, onnx_embedding)

    torch_limit = float(args.torch_atol)
    batch_limit = float(args.batch_atol if device.startswith('cuda') else args.torch_atol)
    onnx_limit = float(args.onnx_atol)
    passed = True
    for length, entry in results.items():
        for name, values in entry['checks'].items():
            limit = (onnx_limit if name.endswith('onnx') else
                     batch_limit if name.endswith('singleton') else torch_limit)
            values['atol'] = limit
            values['passed'] = bool(values['max_abs'] <= limit)
            passed &= values['passed']
    report = {
        'schema_version': 'ffm_mantis_stage0_parity_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'windows_path': str(artifact), 'windows_sha256': _sha256(artifact),
        'rows': int(len(windows)), 'contexts': [int(x) for x in args.contexts],
        'model_id': 'paris-noah/MantisV2', 'model_version': 'v2',
        'preprocessing': args.preprocessing, 'device': device,
        'results': results, 'passed': bool(passed),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',
                        default='output/foundation_tournament/representation_apples/windows.npz')
    parser.add_argument('--output', default='output/mantis_v2_ssl_pilot/stage0_parity.json')
    parser.add_argument('--contexts', default='64,128,256')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preprocessing', default='per_window_per_channel_zscore_v1')
    parser.add_argument('--torch-atol', type=float, default=1e-5)
    parser.add_argument('--batch-atol', type=float, default=1e-2,
                        help='CUDA batch-shape numerical sensitivity bound')
    parser.add_argument('--onnx-atol', type=float, default=1e-4)
    args = parser.parse_args()
    args.contexts = tuple(int(value) for value in args.contexts.split(',') if value)
    if not args.contexts or 256 not in args.contexts or args.samples < 2:
        parser.error('contexts must include 256 and samples must be at least two')
    run(args)


if __name__ == '__main__':
    main()
