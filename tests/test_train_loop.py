"""Process tests for the overfit-driven training loop (train_loop.train_loop).

These mock ev.run (the walk-forward) and TH.tune_head (the Optuna scan) so the
loop's CONTROL FLOW is verified deterministically, step by step, without the
heavy embed/XGBoost compute. The contract under test is exactly the process:

  default WF -> generalizes? keep defaults -> else Optuna -> rerun -> loop
  until it passes -> final full walk-forward.
"""
from futures_foundation.pipeline import train_loop as TL


class FakeLab:
    n_classes = 2


def _verdict(gen, all_pass=True, gap=-0.10):
    """A canned ev.run(return_verdict=True) result."""
    return dict(all_pass=all_pass, generalizes=gen, gap=gap, thr=0.60,
                test_meanR=0.50, test_wr=0.60, test_n=200, val_meanR=0.50,
                edge_shuffle=0.40, edge_random=0.40, edge_naive=0.30, records=[])


def _patch(monkeypatch, run_verdicts, tune_results):
    """Wire ev.run to yield run_verdicts in order, TH.tune_head from tune_results."""
    state = dict(run=0, tune=0, run_maxfolds=[])
    rv = list(run_verdicts)
    tr = list(tune_results)

    def fake_run(labeler, head_factory=None, seeds=None, max_folds=None,
                 auto_regularize=True, return_verdict=False, **kw):
        assert return_verdict is True         # train_loop must request the verdict
        state['run'] += 1
        state['run_maxfolds'].append(max_folds)
        return rv[min(state['run'] - 1, len(rv) - 1)]

    def fake_tune(labeler, **kw):
        state['tune'] += 1
        return tr[min(state['tune'] - 1, len(tr) - 1)]

    monkeypatch.setattr(TL.ev, 'run', fake_run)
    monkeypatch.setattr(TL.TH, 'tune_head', fake_tune)
    monkeypatch.setattr(TL.backbone, 'stamp_active_source', lambda **k: None)
    return state


def test_defaults_generalize_keeps_defaults(monkeypatch):
    # Step 2 path: iter-0 defaults generalize -> NO Optuna, keep defaults, final.
    state = _patch(monkeypatch,
                   run_verdicts=[_verdict(gen=True), _verdict(gen=True)],
                   tune_results=[dict(generalizes=False, params={}, guard_lift=0.0)])
    res = TL.train_loop(FakeLab(), max_iters=3, loop_max_folds=12)
    assert state['tune'] == 0                 # Optuna NOT triggered
    assert state['run'] == 2                  # iter0 + final only
    assert res['source'] == 'default'
    assert res['params'] == {}
    assert state['run_maxfolds'][0] == 12     # loop iter subsampled
    assert state['run_maxfolds'][-1] is None  # final = all folds


def test_overfit_then_tuned_generalizes(monkeypatch):
    # Step 3-5 path: iter-0 overfit -> Optuna finds generalizing params ->
    # rerun generalizes -> final.
    tuned = {'max_depth': 3, 'min_child_weight': 30}
    state = _patch(
        monkeypatch,
        run_verdicts=[_verdict(gen=False, all_pass=False, gap=0.55),  # iter0
                      _verdict(gen=True),                              # iter1 tuned
                      _verdict(gen=True)],                             # final
        tune_results=[dict(generalizes=True, params=tuned, guard_lift=0.12)])
    res = TL.train_loop(FakeLab(), max_iters=3, loop_max_folds=12)
    assert state['tune'] == 1                 # Optuna triggered once
    assert state['run'] == 3                  # iter0 + iter1 + final
    assert res['source'].startswith('tuned')
    assert res['params'] == tuned
    assert state['run_maxfolds'][-1] is None  # final = all folds


def test_optuna_cannot_find_generalizing_params_flags(monkeypatch):
    # iter-0 overfit -> Optuna's guard rejects (no generalizing params) ->
    # stop, FLAG, fall back to defaults, final.
    state = _patch(
        monkeypatch,
        run_verdicts=[_verdict(gen=False, all_pass=False, gap=0.55),  # iter0
                      _verdict(gen=False, all_pass=False, gap=0.55)],  # final
        tune_results=[dict(generalizes=False, params={}, guard_lift=-0.20)])
    res = TL.train_loop(FakeLab(), max_iters=3, loop_max_folds=12)
    assert state['tune'] == 1                 # tried once, rejected
    assert state['run'] == 2                  # iter0 + final (no tuned rerun)
    assert res['source'] == 'default(flagged)'
    assert res['params'] == {}


def test_overfit_persists_through_max_iters_flags(monkeypatch):
    # Optuna keeps returning "generalizing-on-guard" params but the walk-forward
    # stays overfit every iter -> exhaust max_iters -> FLAG + defaults.
    state = _patch(
        monkeypatch,
        run_verdicts=[_verdict(gen=False, gap=0.55)],   # every WF overfit
        tune_results=[dict(generalizes=True, params={'max_depth': 4},
                           guard_lift=0.10)])
    res = TL.train_loop(FakeLab(), max_iters=2, loop_max_folds=12)
    assert state['tune'] == 2                 # one scan per iter, capped
    # iter0 + iter1 WF + iter2 WF + final
    assert state['run'] == 4
    assert res['source'] == 'default(flagged)'
    assert res['params'] == {}


def test_returns_history_and_final(monkeypatch):
    state = _patch(monkeypatch,
                   run_verdicts=[_verdict(gen=True), _verdict(gen=True)],
                   tune_results=[dict(generalizes=False, params={}, guard_lift=0.0)])
    res = TL.train_loop(FakeLab(), max_iters=3, loop_max_folds=12)
    assert 'history' in res and res['history'][0]['source'] == 'default'
    assert res['final'] is not None and res['final']['generalizes'] is True
