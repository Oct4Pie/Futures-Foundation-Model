#!/usr/bin/env python3
"""Persistent, validation-only Optuna studies for the equal-history foundation tournament."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from futures_foundation.finetune.tournament import protocol


ROOT = Path(__file__).resolve().parents[1]
MANTIS_TRAIN = ROOT / "scripts" / "train_ssl_local.py"
MOMENT_TRAIN = ROOT / "scripts" / "train_moment_tournament.py"
KRONOS_TRAIN = ROOT / "scripts" / "train_kronos_tournament.py"
CHRONOS_TRAIN = ROOT / "scripts" / "train_chronos_tournament.py"
TTM_TRAIN = ROOT / "scripts" / "train_ttm_tournament.py"
TIMESFM_TRAIN = ROOT / "scripts" / "train_timesfm_tournament.py"
MOIRAI2_TRAIN = ROOT / "scripts" / "train_moirai2_tournament.py"
MANTIS_MODELS = {
    "mantis_v1": ("paris-noah/Mantis-8M", "v1"),
    "mantis_v2": ("paris-noah/MantisV2", "v2"),
}
CHRONOS_MODELS = ("chronos_v1", "chronos_bolt", "chronos_v2")
KRONOS_MODELS = ("kronos_mini", "kronos_small")
TRAIN_BATCH = {
    "mantis_v1": 128, "mantis_v2": 128, "moment": 16,
    "kronos_mini": 256, "kronos_small": 256,
    "ttm_r2": 1024,
    "timesfm25": 128,
    "moirai2_small": 512,
    "chronos_v1": 128, "chronos_bolt": 128, "chronos_v2": 128,
}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def build_command(model, params, *, data_dir, output, trial_steps, seed,
                  moment_repo=None, kronos_repo=None, ttm_repo=None, ttm_python=None,
                  timesfm_repo=None,
                  uni2ts_repo=None, moirai_python=None,
                  smoke=False, evaluation_seed=5400, mantis_stage="contrastive",
                  warm_checkpoint=None):
    common_dates = [
        "--train-start", "2019-07-01", "--val-start", "2024-07-01",
        "--holdout-start", "2025-07-01",
    ]
    if model in MANTIS_MODELS:
        model_id, model_version = MANTIS_MODELS[model]
        epochs = 1 if smoke else 8
        steps_per_epoch = max(1, int(trial_steps) // epochs)
        if mantis_stage == "forecast" and not warm_checkpoint:
            raise ValueError("Mantis forecast tuning requires a promoted contrastive checkpoint")
        lineage = "canonical" if mantis_stage == "forecast" else "vanilla"
        command = [
            sys.executable, str(MANTIS_TRAIN), "--stage", mantis_stage,
            "--lineage", lineage, "--protocol", "foundation_5y1y1y_v1",
            *common_dates, "--data-dir", str(data_dir), "--output", str(output),
            "--model-id", model_id, "--model-version", model_version,
            "--batch", "8" if smoke else "128", "--epochs", str(epochs),
            "--steps-per-epoch", str(2 if smoke else steps_per_epoch),
            "--lr", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--seq", str(params["context"]),
            "--preprocessing", params["preprocessing"],
            "--probe-folds", "5", "--controls", "", "--seed", str(seed),
            "--probe-seed", str(evaluation_seed),
            "--device", "cuda",
        ]
        if warm_checkpoint:
            command.extend(["--warm-checkpoint", str(warm_checkpoint)])
        if mantis_stage == "mask":
            command.extend([
                "--mask-ratio", str(params["mask_ratio"]),
                "--span-mean", str(params["span_mean"]),
                "--span-max", str(params["span_max"]),
                "--feature-anchor-weight", str(params.get("feature_anchor_weight", 0.0)),
            ])
        elif mantis_stage == "contrastive":
            command.extend([
                "--temperature", str(params["temperature"]),
                "--crop-max", str(params["crop_max"]),
                "--aug-noise", str(params["aug_noise"]),
                "--aug-scale", str(params["aug_scale"]),
                "--aug-tmask", str(params["aug_tmask"]),
            ])
        elif mantis_stage == "forecast":
            command.extend([
                "--context-lengths", params["context_lengths"],
                "--objective", params["objective"],
                "--dir-weight", str(params["dir_weight"]),
                "--freeze-encoder-layers", str(params["freeze_encoder_layers"]),
            ])
        else:
            raise ValueError("the tournament tuner currently supports Mantis mask/contrastive")
        if smoke:
            command.extend(["--smoke", "--no-probe", "--tickers", "ES", "--tfs", "60min"])
        return command
    if model == "moment":
        if not moment_repo:
            raise ValueError("MOMENT study requires --moment-repo")
        return [
            sys.executable, str(MOMENT_TRAIN), "--moment-repo", str(moment_repo),
            "--data-dir", str(data_dir), "--output", str(output),
            "--context", str(params["context"]), "--batch-size", "2" if smoke else "16",
            "--max-steps", str(2 if smoke else trial_steps),
            "--eval-every", str(1 if smoke else min(512, trial_steps)),
            "--val-batches", str(1 if smoke else 32),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--mask-ratio", str(params["mask_ratio"]), "--seed", str(seed),
            "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
    if model in KRONOS_MODELS or model == "kronos":
        if not kronos_repo:
            raise ValueError("Kronos study requires --kronos-repo")
        tokenizer_steps = 1 if smoke else int(params["tokenizer_steps"])
        predictor_steps = 1 if smoke else int(trial_steps) - tokenizer_steps
        return [
            sys.executable, str(KRONOS_TRAIN), "--kronos-repo", str(kronos_repo),
            "--arm", ("kronos_small" if model == "kronos" else model),
            "--data-dir", str(data_dir), "--output", str(output),
            "--batch-size", "2" if smoke else "256",
            "--tokenizer-steps", str(tokenizer_steps),
            "--predictor-steps", str(predictor_steps),
            "--eval-every", str(1 if smoke else min(512, max(1, predictor_steps))),
            "--val-batches", str(1 if smoke else 32),
            "--tokenizer-learning-rate", str(params["tokenizer_learning_rate"]),
            "--predictor-learning-rate", str(params["predictor_learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--clip", str(params["clip"]), "--seed", str(seed),
            "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
    if model in CHRONOS_MODELS:
        command = [
            sys.executable, str(CHRONOS_TRAIN), "--family", model,
            "--data-dir", str(data_dir), "--output", str(output),
            "--batch-size", "2" if smoke else "128",
            "--max-steps", str(2 if smoke else trial_steps),
            "--eval-every", str(1 if smoke else min(512, trial_steps)),
            "--val-batches", str(1 if smoke else 32),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--grad-clip", str(params["grad_clip"]),
            "--seed", str(seed), "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
        if model == "chronos_v2":
            command.extend(["--chronos2-mode", "joint_ohlcv"])
        else:
            command.extend(["--univariate-input", "channel_independent_ohlcv"])
        return command
    if model == "ttm_r2":
        if not ttm_repo:
            raise ValueError("TTM-R2 study requires --ttm-repo")
        return [
            str(ttm_python or sys.executable), str(TTM_TRAIN),
            "--ttm-repo", str(ttm_repo), "--data-dir", str(data_dir),
            "--output", str(output), "--batch-size", "2" if smoke else "1024",
            "--max-steps", str(2 if smoke else trial_steps),
            "--eval-every", str(1 if smoke else min(512, trial_steps)),
            "--val-batches", str(1 if smoke else 32),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--grad-clip", str(params["grad_clip"]), "--seed", str(seed),
            "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
    if model == "timesfm25":
        if not timesfm_repo:
            raise ValueError("TimesFM study requires --timesfm-repo")
        return [
            sys.executable, str(TIMESFM_TRAIN), "--timesfm-repo", str(timesfm_repo),
            "--data-dir", str(data_dir), "--output", str(output),
            "--batch-size", "1" if smoke else "128",
            "--max-steps", str(2 if smoke else trial_steps),
            "--eval-every", str(1 if smoke else min(512, trial_steps)),
            "--val-batches", str(1 if smoke else 32),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--grad-clip", str(params["grad_clip"]),
            "--lora-rank", str(params["lora_rank"]),
            "--lora-alpha", str(params["lora_alpha"]),
            "--lora-dropout", str(params["lora_dropout"]),
            "--seed", str(seed), "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
    if model == "moirai2_small":
        if not uni2ts_repo:
            raise ValueError("Moirai-2 study requires --uni2ts-repo")
        return [
            str(moirai_python or sys.executable), str(MOIRAI2_TRAIN),
            "--uni2ts-repo", str(uni2ts_repo), "--data-dir", str(data_dir),
            "--output", str(output), "--batch-size", "2" if smoke else "512",
            "--max-steps", str(2 if smoke else trial_steps),
            "--eval-every", str(1 if smoke else min(128, trial_steps)),
            "--val-batches", str(1 if smoke else 32),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--grad-clip", str(params["grad_clip"]),
            "--seed", str(seed), "--validation-seed", str(evaluation_seed),
            *( ["--tickers", "ES", "--timeframes", "60min"] if smoke else [] ),
        ]
    raise ValueError(f"unsupported tournament model: {model}")


def _suggest(trial, model, trial_steps, mantis_stage="contrastive"):
    if model in MANTIS_MODELS:
        if mantis_stage == "mask":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.5e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
                # Context changes the legal non-overlapping probe population. Keep it fixed so
                # every Optuna trial is scored on byte-identical validation rows.
                "context": 256,
                "preprocessing": trial.suggest_categorical("preprocessing", [
                    "per_window_per_channel_zscore_v1", "per_window_shared_ohlc_zscore_v1",
                    "per_window_log_price_rel_volume_zscore_v1",
                ]),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.20, 0.50),
                "span_mean": trial.suggest_categorical("span_mean", [8.0, 16.0, 32.0]),
                "span_max": trial.suggest_categorical("span_max", [32, 64, 96]),
                "feature_anchor_weight": trial.suggest_categorical(
                    "feature_anchor_weight", [0.01, 0.05, 0.2]
                ),
            }
        if mantis_stage != "contrastive":
            if mantis_stage == "forecast":
                objective = trial.suggest_categorical(
                    "objective", ["candle_mse", "candle_direction"])
                return {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 5e-6, 1e-4, log=True),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 0.01, 0.2, log=True),
                    "context": 128,
                    # The maximum forecast context changes the legal parent-window universe.
                    # Hold it fixed so every trial has byte-identical validation rows.
                    "context_lengths": "64,128,192",
                    "preprocessing": "per_window_per_channel_zscore_v1",
                    "objective": objective,
                    "dir_weight": (0.0 if objective == "candle_mse" else
                                   trial.suggest_float("dir_weight", 0.05, 1.0, log=True)),
                    "freeze_encoder_layers": trial.suggest_categorical(
                        "freeze_encoder_layers", [2, 3, 4]),
                }
            raise ValueError(f"unsupported Mantis tuning stage: {mantis_stage}")
        return {
            "learning_rate": trial.suggest_float("learning_rate", 2e-5, 3e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
            # Stage-2 needs up to two contexts inside one front-contract segment. seq=256 leaves
            # too few legal CL@60m validation parents for five walk-forward folds.
            "context": 128,
            "preprocessing": trial.suggest_categorical("preprocessing", [
                "per_window_per_channel_zscore_v1", "per_window_shared_ohlc_zscore_v1",
                "per_window_log_price_rel_volume_zscore_v1",
            ]),
            "temperature": trial.suggest_float("temperature", 0.05, 0.2, log=True),
            "crop_max": trial.suggest_float("crop_max", 0.10, 0.35),
            "aug_noise": trial.suggest_float("aug_noise", 0.02, 0.15),
            "aug_scale": trial.suggest_float("aug_scale", 0.05, 0.25),
            "aug_tmask": trial.suggest_float("aug_tmask", 0.05, 0.20),
        }
    if model == "moment":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 2e-6, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.2, log=True),
            "context": trial.suggest_categorical("context", [64, 128, 256]),
            "mask_ratio": trial.suggest_float("mask_ratio", 0.20, 0.50),
        }
    if model in CHRONOS_MODELS:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-7, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.2, log=True),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 3.0]),
        }
    if model == "ttm_r2":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.2, log=True),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 3.0]),
        }
    if model == "timesfm25":
        rank = trial.suggest_categorical("lora_rank", [4, 8, 16])
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 3.0]),
            "lora_rank": rank, "lora_alpha": 2 * rank,
            "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1]),
        }
    if model == "moirai2_small":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 2e-6, 2e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.2, log=True),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 3.0]),
        }
    tokenizer_choices = sorted({0, max(1, int(trial_steps) // 4)})
    return {
        "tokenizer_steps": trial.suggest_categorical("tokenizer_steps", tokenizer_choices),
        "tokenizer_learning_rate": trial.suggest_float(
            "tokenizer_learning_rate", 5e-6, 5e-5, log=True,
        ),
        "predictor_learning_rate": trial.suggest_float(
            "predictor_learning_rate", 2e-7, 5e-6, log=True,
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
        "clip": trial.suggest_categorical("clip", [3.0, 5.0, 8.0]),
    }


def _validation_fingerprint(model, report):
    """Return the immutable validation sample identity emitted by each trainer."""
    sampling = report.get("sampling", {})
    if model in MANTIS_MODELS:
        return sampling.get("probe_sample_sha256")
    return sampling.get("validation_schedule_sha256")


def _mantis_validation_values(report):
    """Paired adaptation deltas; never raw scores across preprocessing contracts."""
    targets = report["probe"]["per_target"]
    return (float(targets["fwd_absmove"]["delta"]),
            float(targets["fwd_dir"]["delta"]))


def _guard_validation_fingerprint(study, trial, model, report):
    fingerprint = _validation_fingerprint(model, report)
    if not fingerprint:
        raise RuntimeError("trainer report is missing its validation sample fingerprint")
    expected = study.user_attrs.get("validation_schedule_sha256")
    if expected is None:
        study.set_user_attr("validation_schedule_sha256", fingerprint)
    elif fingerprint != expected:
        raise RuntimeError(
            f"validation sample drift: expected {expected}, received {fingerprint}"
        )
    trial.set_user_attr("validation_schedule_sha256", fingerprint)


def _selected_trial(study, model):
    """Apply one deterministic validation-only winner rule."""
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not complete:
        raise RuntimeError("study has no completed trial")
    if model in MANTIS_MODELS:
        # A larger primary score cannot override a failed safety/promotion gate. If at least one
        # trial is promotable, finalist selection is restricted to that set. When none promote,
        # retain the best diagnostic trial in the report but mark it non-promotable explicitly.
        promoted = [trial for trial in complete
                    if bool(trial.user_attrs.get("promotion_passed", False))]
        candidates = promoted or complete
        # Forward absolute-move representation is primary; forward-direction AUC is an
        # exact-tie breaker.  Never pick an arbitrary member of Optuna's Pareto set.
        return max(candidates, key=lambda trial: (
            float(trial.values[0]), float(trial.values[1]),
        ))
    return min(complete, key=lambda trial: float(trial.value))


def tune(args):
    import optuna
    suffix = f"_{args.mantis_stage}" if args.model in MANTIS_MODELS else ""
    output_dir = Path(args.output_dir).resolve() / f"{args.model}{suffix}"
    model_key = "kronos_small" if args.model == "kronos" else args.model
    if args.trial_steps is None:
        batch = TRAIN_BATCH[model_key]
        if args.examples_per_trial % batch:
            raise ValueError(
                f"examples-per-trial must be divisible by {model_key} batch size {batch}"
            )
        trial_steps = args.examples_per_trial // batch
    else:
        trial_steps = int(args.trial_steps)
        if trial_steps < 1:
            raise ValueError("trial-steps must be positive")
        if not args.smoke and trial_steps * TRAIN_BATCH[model_key] != args.examples_per_trial:
            raise ValueError(
                "explicit trial-steps violates the equal sampled-anchor budget: "
                f"{trial_steps}*{TRAIN_BATCH[model_key]} != {args.examples_per_trial}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}"
    if args.model in MANTIS_MODELS:
        study = optuna.create_study(
            study_name=f"{args.model}_{args.mantis_stage}_5y1y1y_v4_delta", storage=storage,
            directions=("maximize", "maximize"), load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=4),
        )
    else:
        study = optuna.create_study(
            study_name=f"{args.model}_5y1y1y_v2", storage=storage,
            direction="minimize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=4),
        )

    def objective(trial):
        params = _suggest(trial, args.model, trial_steps, args.mantis_stage)
        trial_dir = output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = trial_dir / "checkpoint.pt"
        log = trial_dir / "train.log"
        command = build_command(
            args.model, params, data_dir=Path(args.data_dir).resolve(), output=checkpoint,
            trial_steps=trial_steps, seed=args.seed,
            moment_repo=args.moment_repo, kronos_repo=args.kronos_repo,
            ttm_repo=args.ttm_repo, ttm_python=args.ttm_python,
            timesfm_repo=args.timesfm_repo,
            uni2ts_repo=args.uni2ts_repo, moirai_python=args.moirai_python,
            evaluation_seed=args.evaluation_seed, smoke=args.smoke,
            mantis_stage=args.mantis_stage,
            warm_checkpoint=args.warm_checkpoint,
        )
        trial.set_user_attr("checkpoint", str(checkpoint))
        trial.set_user_attr("command", command)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"trial {trial.number} failed; inspect {log}")
        report = json.loads(Path(str(checkpoint) + ".report.json").read_text())
        _guard_validation_fingerprint(study, trial, args.model, report)
        if args.model in MANTIS_MODELS:
            # Different preprocessing contracts give vanilla Mantis materially different raw
            # probe scores. Comparing absolute adapted scores would reward an easy baseline, not
            # adaptation quality. The tournament objective is improvement over the paired vanilla
            # encoder under the IDENTICAL contract and fixed validation rows.
            move_delta, direction_delta = _mantis_validation_values(report)
            run_manifest = json.loads(
                Path(str(checkpoint) + ".run.json").read_text()
            )
            trial.set_user_attr("validation_fwd_absmove_delta_r2", move_delta)
            trial.set_user_attr("validation_fwd_dir_delta_auc", direction_delta)
            trial.set_user_attr(
                "promotion_passed", bool(run_manifest["strict_probe"]["passed"])
            )
            trial.set_user_attr("mean_core_delta", float(report["probe"]["mean_core_delta"]))
            return move_delta, direction_delta
        key = ("best_val_loss" if args.model in {
                   "moment", "ttm_r2", "timesfm25", "moirai2_small",
               } or args.model in CHRONOS_MODELS
               else "best_predictor_val_loss")
        value = float(report[key])
        trial.set_user_attr("native_validation_loss", value)
        return value

    study.optimize(objective, n_trials=args.n_trials)
    selected = _selected_trial(study, args.model)
    summary = {
        "schema_version": "ffm_foundation_optuna_study_v3_delta",
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "mantis_stage": (args.mantis_stage if args.model in MANTIS_MODELS else None),
        "protocol": protocol(), "trial_steps": trial_steps,
        "examples_per_trial": int(trial_steps * TRAIN_BATCH[model_key]),
        "training_seed": args.seed, "evaluation_seed": args.evaluation_seed,
        "sampler": {"name": "TPESampler", "n_startup_trials": 4},
        "study_storage": storage, "study_name": study.study_name,
        "trials": [
            {"number": row.number, "state": row.state.name, "params": row.params,
             "values": row.values, "user_attrs": row.user_attrs}
            for row in study.trials
        ],
        "best_trials": [row.number for row in study.best_trials],
        "selected_trial": selected.number,
        "selected_promotion_passed": bool(
            selected.user_attrs.get("promotion_passed", False)
            if args.model in MANTIS_MODELS else True),
        "selection_rule": ("maximize paired validation fwd_absmove delta-R2; break exact "
                           "ties with paired fwd_dir delta-AUC" if args.model in MANTIS_MODELS
                           else "minimize native validation loss within family"),
    }
    _atomic_json(output_dir / "study.json", summary)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(
        *KRONOS_MODELS, "moment", "mantis_v1", "mantis_v2", "ttm_r2", "timesfm25",
        "moirai2_small",
        *CHRONOS_MODELS,
    ),
                        required=True)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--mantis-stage", choices=("mask", "contrastive", "forecast"),
                        default="contrastive")
    parser.add_argument("--warm-checkpoint",
                        help="promoted Mantis contrastive checkpoint required by forecast stage")
    parser.add_argument("--output-dir", default="output/foundation_tournament/optuna")
    parser.add_argument("--moment-repo")
    parser.add_argument("--kronos-repo")
    parser.add_argument("--ttm-repo")
    parser.add_argument("--ttm-python")
    parser.add_argument("--timesfm-repo")
    parser.add_argument("--uni2ts-repo")
    parser.add_argument("--moirai-python")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--trial-steps", type=int)
    parser.add_argument("--examples-per-trial", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--evaluation-seed", type=int, default=5400)
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    tune(_parser().parse_args())
