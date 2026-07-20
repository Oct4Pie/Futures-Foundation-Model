"""Shared mandatory check closure for route-specific native training smoke."""
from __future__ import annotations


REQUIRED_SMOKE_CHECKS = (
    "one_batch_forward_backward",
    "controlled_learnable_loss_decrease",
    "shuffle_control_rejection",
    "time_destroyed_control_rejection",
    "exact_interruption_resume_trajectory",
    "training_exported_inference_parity",
    "prefix_invariance",
    "future_corruption_invariance",
    "contract_roll_rejection",
    "session_gap_rejection",
    "split_boundary_rejection",
    "oos_boundary_rejection",
    "multivariate_channel_grouping",
    "native_missing_data_mask",
    "memory_measurement",
    "throughput_measurement",
    "negative_price_behavior",
    "native_output_parity",
    "checkpoint_lineage",
    "data_lineage",
)


__all__ = ["REQUIRED_SMOKE_CHECKS"]
