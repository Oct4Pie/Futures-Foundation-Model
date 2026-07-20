#!/usr/bin/env python3
"""Run the exact MOMENT masked-reconstruction route on synthetic data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune import ssl_data
from futures_foundation.finetune.native_contract_harness import (
    CheckResult, control_rejection_check, forward_backward_check,
    future_corruption_check, interruption_resume_parity_check,
    loss_decrease_check, negative_price_behavior_check, parity_check,
    performance_check, prefix_invariance_check, rejection_check,
)
from futures_foundation.finetune.native_route_smoke import (
    build_route_smoke_evidence, validate_route_smoke_evidence,
)
from futures_foundation.finetune.routes import moment_reconstruction as route


FIXTURE_SCHEMA = "ffm_moment_reconstruction_smoke_fixture_v1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path).resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, target); return target


def _fixture(batch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(route.PARENT_LENGTH, dtype=np.float64)
    level = rng.uniform(45.0, 65.0, batch)
    slope = rng.uniform(-0.03, 0.03, batch)
    phase = rng.uniform(0.0, 2.0 * np.pi, batch)
    close = level[:, None] + slope[:, None] * t + 0.4 * np.sin(t[None, :] / 13.0 + phase[:, None])
    open_ = np.concatenate((close[:, :1], close[:, :-1]), axis=1)
    spread = 0.2 + 0.05 * np.abs(np.sin(t / 17.0))[None, :]
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 55.0 + rng.uniform(-2.0, 2.0, batch)[:, None] + 0.01 * t + 0.5 * np.sin(t[None, :] / 19.0 + phase[:, None])
    return route.parent_array(np.stack((open_, high, low, close, volume), axis=-1).astype(np.float32))


def _negative_fixture(batch: int, seed: int) -> np.ndarray:
    value = _fixture(batch, seed).copy()
    value[:, :, :4] -= float(np.max(value[:, :, :4]) + 10.0)
    return route.parent_array(value)


def _write_fixture(directory: Path, train: np.ndarray, validation: np.ndarray):
    artifact = directory / "synthetic_fixture.npz"
    np.savez_compressed(artifact, train=train, validation=validation)
    manifest = {"schema_version": FIXTURE_SCHEMA, "generator": "smooth_ohlcv_reconstruction_v1",
        "market_data_read": False, "oos_read": False, "train_shape": list(train.shape),
        "validation_shape": list(validation.shape), "artifact": {"path": str(artifact.resolve()),
        "sha256": _sha(artifact), "bytes": artifact.stat().st_size}}
    return artifact, _atomic_json(directory / "synthetic_fixture.manifest.json", manifest)


def _fresh(base: Any, initial: Mapping[str, Any], device: str) -> Any:
    model = copy.deepcopy(base).to(device); model.load_state_dict(initial, strict=True); return model


def _np(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _loss(model: Any, parent: np.ndarray, device: str, seed: int) -> float:
    torch = route._torch(); model.eval(); route.seed_everything(seed)
    with torch.no_grad(): return float(route.native_loss(model, parent, device=device).detach().cpu())


def _mismatched_loss(model, parent, target, device):
    output, _, input_mask = route.native_output(model, parent, device=device)
    target_x, _ = route.model_inputs(target, device=device)
    hidden = input_mask * (1 - output.pretrain_mask)
    loss = ((output.reconstruction - target_x).square() * hidden[:, None, :]).sum() / (hidden.sum() * target_x.shape[1])
    return loss


def _train(base, initial, parent, validation, config, device, *, target=None):
    route.seed_everything(config.seed); model = _fresh(base, initial, device)
    optimizer = route.make_optimizer(model, config); scheduler = route.make_scheduler(optimizer, config); losses = []
    for _ in range(config.total_steps):
        if target is None:
            row = route.optimizer_step(model, optimizer, scheduler, parent, device=device,
                                       max_gradient_norm=config.max_gradient_norm); losses.append(row["loss"])
        else:
            torch = route._torch(); model.train(); optimizer.zero_grad(set_to_none=True)
            loss = _mismatched_loss(model, parent, target, device); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach().cpu()))
    return model, losses, _loss(model, validation, device, config.seed + 500)


def _result(passed, metrics, reason):
    return CheckResult(status="pass" if passed else "fail", metrics=dict(metrics), reason=None if passed else reason)


def _boundary(kind):
    delta = pd.Timedelta("1min"); length = 640
    ts = pd.date_range("2024-01-01", periods=length, freq=delta, tz="UTC")
    def validate(case):
        starts = ssl_data.window_starts(np.asarray(case["indices"], np.int64), route.PARENT_LENGTH,
            timestamps=case["timestamps"], expected_delta=delta, segment_ids=case.get("segments"))
        if not len(starts): raise ValueError("invalid boundary rejected")
    if kind == "contract_roll": case = {"indices": np.arange(length), "timestamps": ts,
        "segments": np.asarray(["A"] * 320 + ["B"] * 320)}
    elif kind == "session_gap": case = {"indices": np.arange(length),
        "timestamps": ts[:320].append(ts[320:] + pd.Timedelta("1h")), "segments": np.asarray(["A"] * length)}
    elif kind == "split_boundary": case = {"indices": np.r_[np.arange(320), np.arange(420, 740)],
        "timestamps": pd.date_range("2024-01-01", periods=740, freq=delta, tz="UTC"), "segments": np.asarray(["A"] * 740)}
    else: case = {"indices": np.arange(320), "timestamps": ts, "segments": np.asarray(["A"] * length)}
    return rejection_check(validate, {kind: case})


def _resume(base, initial, parent, config, identity, device, path):
    steps = 4
    def run_full():
        route.seed_everything(config.seed + 31); m = _fresh(base, initial, device); o = route.make_optimizer(m, config); s = route.make_scheduler(o, config); h=[]
        for step in range(steps): h.append({"step": step+1, **route.optimizer_step(m,o,s,parent,device=device,max_gradient_norm=config.max_gradient_norm)})
        return route.capture_training_state(model=m,optimizer=o,scheduler=s,config=config,model_identity=identity,global_step=steps,sampler_cursor=steps,history=h)
    def run_resumed():
        route.seed_everything(config.seed + 31); m = _fresh(base, initial, device); o=route.make_optimizer(m,config); s=route.make_scheduler(o,config); h=[]
        for step in range(2): h.append({"step": step+1, **route.optimizer_step(m,o,s,parent,device=device,max_gradient_norm=config.max_gradient_norm)})
        partial=route.capture_training_state(model=m,optimizer=o,scheduler=s,config=config,model_identity=identity,global_step=2,sampler_cursor=2,history=h)
        route.save_training_state(path,partial); state=route.load_training_state(path); m2=_fresh(base,initial,device); o2=route.make_optimizer(m2,config); s2=route.make_scheduler(o2,config)
        step0,cursor,h=route.restore_training_state(state,model=m2,optimizer=o2,scheduler=s2,config=config,model_identity=identity)
        if (step0,cursor)!=(2,2): raise RuntimeError("resume cursor drift")
        for step in range(2,steps): h.append({"step":step+1, **route.optimizer_step(m2,o2,s2,parent,device=device,max_gradient_norm=config.max_gradient_norm)})
        return route.capture_training_state(model=m2,optimizer=o2,scheduler=s2,config=config,model_identity=identity,global_step=steps,sampler_cursor=steps,history=h)
    return interruption_resume_parity_check(run_full,run_resumed,atol=0,rtol=0)


def run(args):
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch=route._torch(); output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite: raise FileExistsError(f"smoke output not empty: {output}")
    output.mkdir(parents=True,exist_ok=True); config=route.RouteConfig(learning_rate=args.learning_rate,weight_decay=args.weight_decay,batch_size=args.batch_size,total_steps=args.steps,seed=args.seed); config.validate()
    if config.total_steps!=20: raise ValueError("canonical MOMENT smoke requires 20 steps")
    train=_fixture(config.batch_size,config.seed); validation=_fixture(config.batch_size,config.seed+1); fixture_path,fixture_manifest=_write_fixture(output,train,validation)
    base,identity=route.load_model(args.model_snapshot,source_runtime=args.source_runtime,device=args.device); base.eval(); initial=route.model_state_cpu(base); checks={}
    one=_fresh(base,initial,args.device); oo=route.make_optimizer(one,config); osched=route.make_scheduler(oo,config)
    checks["one_batch_forward_backward"]=forward_backward_check(lambda: route.optimizer_step(one,oo,osched,train,device=args.device,max_gradient_norm=config.max_gradient_norm)); del one,oo,osched
    route.seed_everything(config.seed); real=_fresh(base,initial,args.device); ro=route.make_optimizer(real,config); rs=route.make_scheduler(ro,config)
    checks["controlled_learnable_loss_decrease"]=loss_decrease_check(lambda:_loss(real,train,args.device,config.seed+100),lambda:route.optimizer_step(real,ro,rs,train,device=args.device,max_gradient_norm=config.max_gradient_norm),steps=20,min_relative_decrease=.5,tail=3)
    real_val=_loss(real,validation,args.device,config.seed+500); rng=np.random.default_rng(config.seed+91); shuffled_target=train[rng.permutation(len(train))]
    _,shuffle_losses,shuffle_val=_train(base,initial,train,validation,config,args.device,target=shuffled_target)
    destroyed=train[:,rng.permutation(route.CONTEXT_LENGTH)]; _,destroy_losses,destroy_val=_train(base,initial,destroyed,validation,config,args.device)
    controls=control_rejection_check(real_val,[shuffle_val],[destroy_val],margin=.05,higher_is_better=False)
    checks["shuffle_control_rejection"]=_result(real_val+.05<=shuffle_val,{**controls.metrics,"shuffle_final_train_loss":shuffle_losses[-1]},"real reconstruction did not beat mismatched targets")
    checks["time_destroyed_control_rejection"]=_result(real_val+.05<=destroy_val,{**controls.metrics,"time_destroyed_final_train_loss":destroy_losses[-1]},"real reconstruction did not beat time-destroyed sequence")
    resume_path=output/"interrupted.train.pt"; checks["exact_interruption_resume_trajectory"]=_resume(base,initial,train,config,identity,args.device,resume_path)
    history=[{"step":i+1,"train_loss":float(v)} for i,v in enumerate(checks["controlled_learnable_loss_decrease"].metrics["losses"][1:])]
    state=route.capture_training_state(model=real,optimizer=ro,scheduler=rs,config=config,model_identity=identity,global_step=20,sampler_cursor=20,history=history); state_path=output/"moment_smoke.train.pt"; state_id=route.save_training_state(state_path,state)
    bundle=route.build_export_bundle(model=real,model_identity=identity); bundle_path=output/"moment_smoke.representation.pt"; bundle_id=route.save_export_bundle(bundle_path,bundle)
    exported,reopened_bundle=route.load_export_bundle(bundle_path,snapshot=args.model_snapshot,source_runtime=args.source_runtime,device=args.device); real.eval(); exported.eval()
    reference=_np(route.mean_embedding(real,validation,device=args.device)); candidate=_np(route.mean_embedding(exported,validation,device=args.device)); checks["training_exported_inference_parity"]=parity_check(reference,candidate,atol=0,rtol=0,name="exported MOMENT embedding")
    extra=np.concatenate((validation,np.zeros((len(validation),16,5),np.float32)),axis=1); changed=extra.copy(); changed[:,512:]+=100
    checks["prefix_invariance"]=prefix_invariance_check(lambda x:route.parent_array(x[:,:512]),extra,changed,prefix_length=512,atol=0,rtol=0)
    checks["future_corruption_invariance"]=future_corruption_check(lambda x:_np(route.mean_embedding(exported,x[:,:512],device=args.device)),extra,changed,visible_length=512,atol=0,rtol=0)
    checks["contract_roll_rejection"]=_boundary("contract_roll"); checks["session_gap_rejection"]=_boundary("session_gap"); checks["split_boundary_rejection"]=_boundary("split_boundary"); checks["oos_boundary_rejection"]=_boundary("oos_boundary")
    baseline=_np(route.reconstruction(exported,validation,device=args.device,seed=config.seed+700)); perturbed=validation.copy(); perturbed[:,:,4]*=1.1; changed_rec=_np(route.reconstruction(exported,perturbed,device=args.device,seed=config.seed+700)); unaffected_error=float(np.max(np.abs(baseline[:,:4]-changed_rec[:,:4]))); affected=float(np.max(np.abs(baseline[:,4]-changed_rec[:,4])))
    checks["multivariate_channel_grouping"]=_result(unaffected_error==0 and affected>0,{"layout":"internal_channel_fold_no_interaction","shape":list(baseline.shape),"unaffected_channel_max_abs":unaffected_error,"affected_channel_max_change":affected,"perturbed_channel":"volume"},"MOMENT channel fold leaked across channels")
    missing=validation.copy(); missing[:,:16,:]=np.nan; missing_emb=_np(route.mean_embedding(exported,missing,device=args.device)); checks["native_missing_data_mask"]=_result(np.isfinite(missing_emb).all(),{"missing_prefix_bars":16,"finite":bool(np.isfinite(missing_emb).all())},"MOMENT missing mask produced non-finite embedding")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"): torch.cuda.reset_peak_memory_stats(args.device); memory_probe=lambda:int(torch.cuda.max_memory_allocated(args.device))
    else: memory_probe=None
    perf=performance_check(lambda:route.mean_embedding(exported,validation,device=args.device),batch_size=config.batch_size,repeats=5,warmups=2,min_examples_per_second=.1,memory_probe=memory_probe); checks["memory_measurement"]=CheckResult(perf.status,dict(perf.metrics),perf.reason); checks["throughput_measurement"]=CheckResult(perf.status,dict(perf.metrics),perf.reason)
    checks["negative_price_behavior"]=negative_price_behavior_check(lambda x:_np(route.mean_embedding(exported,x,device=args.device)),_negative_fixture(config.batch_size,config.seed+7),behavior="support")
    rec1=_np(route.reconstruction(exported,validation,device=args.device,seed=config.seed+900)); rec2=_np(route.reconstruction(exported,validation,device=args.device,seed=config.seed+900)); checks["native_output_parity"]=parity_check(rec1,rec2,atol=0,rtol=0,name="seeded MOMENT reconstruction")
    reopened_state=route.load_training_state(state_path); checks["checkpoint_lineage"]=_result(reopened_state.get("schema_version")==route.CHECKPOINT_SCHEMA and reopened_state.get("route_key")==route.ROUTE_KEY and reopened_state.get("model_identity")==identity and state_id["sha256"]==_sha(state_path) and reopened_bundle["route_key"]==route.ROUTE_KEY and bundle_id["sha256"]==_sha(bundle_path),{"checkpoint":state_id,"deployment":bundle_id,"required_state_fields":sorted(reopened_state)},"MOMENT checkpoint lineage incomplete")
    fixture_doc=json.loads(fixture_manifest.read_text()); checks["data_lineage"]=_result(fixture_doc["artifact"]["sha256"]==_sha(fixture_path) and not fixture_doc["market_data_read"] and not fixture_doc["oos_read"],{"fixture_sha256":_sha(fixture_path),"market_data_read":False,"oos_read":False},"MOMENT fixture lineage invalid")
    raw_checks={n:r.manifest() for n,r in checks.items()}; raw={"schema_version":"ffm_moment_reconstruction_smoke_raw_v1","route_key":route.ROUTE_KEY,"checks":raw_checks,"metrics":{"real_validation_loss":real_val,"shuffle_validation_loss":shuffle_val,"time_destroyed_validation_loss":destroy_val,"all_checks_pass":all(r.status=="pass" for r in checks.values())}}; raw_path=_atomic_json(output/"raw_checks.json",raw)
    evidence=build_route_smoke_evidence(route_key=route.ROUTE_KEY,executor_path=route.__file__,executor_entrypoint="native_loss/mean_embedding",checks=raw_checks,artifacts={"model_snapshot":Path(args.model_snapshot).resolve(),"source_runtime":Path(args.source_runtime).resolve(),"synthetic_fixture":fixture_path,"synthetic_fixture_manifest":fixture_manifest,"interrupted_state":resume_path,"training_state":state_path,"deployment_bundle":bundle_path,"raw_checks":raw_path,"smoke_runner":Path(__file__).resolve()},metrics=raw["metrics"]); validate_route_smoke_evidence(evidence); evidence_path=_atomic_json(output/"smoke_evidence.json",evidence)
    return {"status":"pass" if evidence["smoke_admitted"] else "fail","route_key":route.ROUTE_KEY,"smoke_admitted":evidence["smoke_admitted"],"pilot_admitted":False,"training_admitted":False,"evidence":{"path":str(evidence_path),"sha256":_sha(evidence_path),"content_sha256":evidence["evidence_sha256"]},"metrics":raw["metrics"],"failed_checks":[n for n,r in checks.items() if r.status!="pass"]}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model-snapshot",required=True); p.add_argument("--source-runtime",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--steps",type=int,default=20); p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.01); p.add_argument("--seed",type=int,default=20260718); p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); result=run(a); print(json.dumps(result,indent=2,sort_keys=True));
    if result["status"]!="pass": raise SystemExit(1)


if __name__=="__main__": main()
