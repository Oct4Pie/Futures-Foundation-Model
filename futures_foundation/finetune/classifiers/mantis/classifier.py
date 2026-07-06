"""MantisClassifier — torch-free parent-side adapter.

featurize() builds the strategy's multivariate windows (numpy, via the labeler's
mv_contexts). fit_predict() spawns the isolated torch worker (_worker -> _mantis_torch)
so torch never shares a process with xgboost (libomp segfault). Each fit_predict is a
fresh subprocess → MPS/RAM freed on exit.

Registered as 'mantis'. Config kwargs (new_channels, ft_mode, unfreeze_blocks, epochs,
batch, lr, weight_decay, patience, threads, device, max_train, verbose) are forwarded
to the torch trainer.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ...classifier import Classifier, register_classifier


@register_classifier('mantis')
class MantisClassifier(Classifier):
    needs_standardize = True            # harness standardizes [N,C,seq] on train stats

    @staticmethod
    def suggest_params(trial):
        """Optuna search space for the Mantis fine-tune — consumed by finetune.tune via DI."""
        return dict(
            lr=trial.suggest_float('lr', 5e-5, 1e-3, log=True),
            unfreeze_blocks=trial.suggest_int('unfreeze_blocks', 1, 3),   # fewer = more reg
            new_channels=trial.suggest_int('new_channels', 6, 14),
            weight_decay=trial.suggest_float('weight_decay', 0.01, 0.3, log=True),
            batch=trial.suggest_categorical('batch', [64, 128]),
        )

    def __init__(self, **cfg):
        self.cfg = cfg

    def featurize(self, labeler, keys):
        return np.asarray(labeler.mv_contexts(keys), np.float32)

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        cfg = dict(self.cfg)
        log_path = cfg.pop('log_path', None)        # parent-side only (not a trainer arg)
        cmd = [sys.executable, '-u', '-m', 'futures_foundation.finetune.classifiers.mantis._worker']
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / 'ytr.npy', np.asarray(ytr))      # small: always to tempdir
            np.save(d / 'yval.npy', np.asarray(yval))
            paths = {}                                   # X: memmap path passthrough OR save
            for name, arr in [('Xtr', Xtr), ('Xval', Xval), ('Xeval', Xeval)]:
                if isinstance(arr, str):
                    paths[name] = arr                    # full-data memmap on disk — no copy
                else:
                    np.save(d / f'{name}.npy', np.asarray(arr))
                    paths[name] = str(d / f'{name}.npy')
            cfg['_paths'] = paths
            (d / 'cfg.json').write_text(json.dumps(dict(cfg, seed=int(seed))))
            if log_path:                            # STREAM worker output to a file (watchable)
                with open(log_path, 'a') as lf:
                    lf.write(f"\n--- mantis worker (seed={seed}) ---\n"); lf.flush()
                    r = subprocess.run(cmd + [str(d)], stdout=lf, stderr=subprocess.STDOUT, text=True)
                if r.returncode != 0:
                    raise RuntimeError("mantis worker failed:\n"
                                       + open(log_path).read()[-3000:])
            else:
                r = subprocess.run(cmd + [str(d)], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"mantis worker failed:\n{r.stderr[-3000:]}")
            meta = np.load(d / 'meta.npy')
            return np.load(d / 'p_val.npy'), np.load(d / 'p_eval.npy'), float(meta[0])
