"""Standalone RL trading pipeline (PPO).

Generic + public: this package holds ONLY the model-agnostic RL machinery —
the RLStrategy contract + registry, the PPO env, walk-forward reuse
(pipelines.common), shuffle + multi-seed gates, a generic context-head
precompute utility, and a device helper. It contains NO proprietary
strategy logic.

A concrete strategy (e.g. the proprietary CRT sweep detector) is supplied
as a plug-in that subclasses RLStrategy and registers itself — authored in
the private ffm-strategies repo, exactly as the CISD scripts plug into
futures_foundation.finetune. The framework stays generic; the IP stays
private.

    from pipelines.rl import RLStrategy, register, run_walkforward

    @register("my_strategy")
    class MyStrategy(RLStrategy):
        name = "my_strategy"
        entry_filter = True                 # PPO learns a chop-veto
        def detect_entries(self, df_raw, ctx_df, ticker): ...  # -> events df

    run_walkforward(MyStrategy(), RLConfig(...))   # loop/gates/seeds free
"""

from .base import RLStrategy, register, get_strategy, RL_STRATEGIES
from .pipeline import run_walkforward, RLConfig

__all__ = ["RLStrategy", "register", "get_strategy", "RL_STRATEGIES",
           "run_walkforward", "RLConfig"]
