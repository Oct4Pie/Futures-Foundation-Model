"""Compatibility facade for the authoritative native-contract registry.

New code should import :mod:`futures_foundation.finetune.native_contracts` directly.
The old ``FoundationArm``/``ARMS``/``get_arm`` surface remains available so historical
scripts can be migrated without maintaining a second, contradictory roster.
"""
from __future__ import annotations

from futures_foundation.finetune.native_contracts import (
    FoundationArm,
    NativeContractError,
    all_arms,
    get_arm,
)


ARMS = all_arms()

__all__ = ["ARMS", "FoundationArm", "NativeContractError", "get_arm"]
