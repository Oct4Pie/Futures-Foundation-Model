"""Pinned admission contract for the equal-history foundation-model tournament.

This registry records what can actually be compared.  A model may be a trainable
forecast provider, a frozen/zero-shot control, or a downstream control.  Those roles
must not be collapsed into one native-loss leaderboard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FoundationArm:
    key: str
    family: str
    model_id: str
    model_revision: str
    source_url: str
    source_revision: str
    license: str
    role: str
    adaptation: str
    ohlcv_mode: str
    supported_training: bool

    def manifest(self) -> dict:
        return asdict(self)


# Source/model revisions were resolved on 2026-07-15.  The non-commercial Moirai
# checkpoint is allowed only as research evidence and must never become a deployable
# project dependency without a separate license decision.
ARMS = {
    "moment_small": FoundationArm(
        "moment_small", "moment", "AutonLab/MOMENT-1-small",
        "411e288267f82cce86296dbe4d6c8bc533cc162f",
        "https://github.com/moment-timeseries-foundation-model/moment.git",
        "38f7310ad594100747ca2a8357e9c7ca7d323e0e", "MIT",
        "representation_and_forecast", "full_native", "channel_independent_ohlcv", True,
    ),
    "kronos_mini": FoundationArm(
        "kronos_mini", "kronos", "NeoQuasar/Kronos-mini",
        "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "https://github.com/shiyu-coder/Kronos.git",
        "67b630e67f6a18c9e9be918d9b4337c960db1e9a", "MIT",
        "forecast_and_tokenizer", "full_native", "joint_ohlcva", True,
    ),
    "kronos_small": FoundationArm(
        "kronos_small", "kronos", "NeoQuasar/Kronos-small",
        "901c26c1332695a2a8f243eb2f37243a37bea320",
        "https://github.com/shiyu-coder/Kronos.git",
        "67b630e67f6a18c9e9be918d9b4337c960db1e9a", "MIT",
        "forecast_and_representation", "full_native", "joint_ohlcva", True,
    ),
    "ttm_r2": FoundationArm(
        "ttm_r2", "ttm", "ibm-granite/granite-timeseries-ttm-r2",
        "b972f0c22190b7502764526004d16e2b4ed39e8c",
        "https://github.com/ibm-granite/granite-tsfm.git",
        "743e709b9edbfe1b59e31adb1621bdc98a57b91b", "Apache-2.0",
        "forecast_control", "full_native", "joint_ohlcv", True,
    ),
    "timesfm25": FoundationArm(
        "timesfm25", "timesfm", "google/timesfm-2.5-200m-transformers",
        "5a9806b9b291fad9233b5249d88263f1846304d3",
        "https://github.com/google-research/timesfm.git",
        "3dae50b20d7a724981e8ea36cda75578f80dd2dc", "Apache-2.0",
        "forecast_teacher", "lora_native", "channel_independent_ohlcv", True,
    ),
    "moirai2_small": FoundationArm(
        "moirai2_small", "moirai2", "Salesforce/moirai-2.0-R-small",
        "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
        "https://github.com/SalesforceAIResearch/uni2ts.git",
        "cfd46d4510ed8896f263116f32928eede05b0a75", "CC-BY-NC-4.0",
        "multivariate_forecast_control", "full_native", "joint_ohlcv", True,
    ),
    "toto2_22m": FoundationArm(
        "toto2_22m", "toto2", "Datadog/Toto-2.0-22m",
        "685e4ae3e2be8d8998025e53dd98e7fdcb296a89",
        "https://github.com/DataDog/toto.git",
        "44ea4e88852228039564aa3e76fac26aafac0803", "Apache-2.0",
        "multivariate_forecast_control", "zero_shot_only", "joint_ohlcv", False,
    ),
    "sundial_base": FoundationArm(
        "sundial_base", "sundial", "thuml/sundial-base-128m",
        "3212e42564493f520593e5414af4367fc4b49226",
        "https://github.com/thuml/Sundial.git",
        "3ef03b8c3804f64a506e57101a173470040aaece", "Apache-2.0",
        "generative_forecast_control", "zero_shot_only",
        "channel_independent_ohlcv", False,
    ),
    "tabpfn_ts": FoundationArm(
        "tabpfn_ts", "tabpfn_ts", "PriorLabs/TabPFN-TS-3", "package_managed",
        "https://github.com/PriorLabs/tabpfn-time-series.git",
        "a756ae3fb3af82c903c39e1cd71864ff5252bc4d", "Apache-2.0/model-terms",
        "downstream_forecast_control", "in_context_fit", "decomposed_univariate", False,
    ),
}


def get_arm(key: str) -> FoundationArm:
    try:
        return ARMS[str(key)]
    except KeyError as exc:
        raise ValueError(f"unknown foundation arm: {key}") from exc
