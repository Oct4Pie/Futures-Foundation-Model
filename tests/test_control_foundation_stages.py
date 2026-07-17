from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_control_foundation_stages as stages


def test_normalization_is_causal_and_channelwise():
    raw = np.arange(2 * 272 * 5, dtype=np.float32).reshape(2, 272, 5)
    normalized = stages._normalize(raw, 256)
    np.testing.assert_allclose(normalized[:, :256].mean(axis=1), 0.0, atol=2e-5)
    np.testing.assert_allclose(normalized[:, :256].std(axis=1), 1.0, atol=2e-5)
    changed = raw.copy(); changed[:, 256:] += 1e6
    np.testing.assert_allclose(stages._normalize(changed, 256)[:, :256], normalized[:, :256])
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) <= 10.0


def test_signature_excludes_only_output_resume_and_stop():
    common = dict(family="toto2_22m", stage=stages.STAGES[0], output="a", resume=False,
                  stop_after_step=1, learning_rate=1e-5)
    first = stages._signature(SimpleNamespace(**common))
    common.update(output="b", resume=True, stop_after_step=2)
    assert stages._signature(SimpleNamespace(**common)) == first
    common["learning_rate"] = 2e-5
    assert stages._signature(SimpleNamespace(**common)) != first


def test_stage_names_are_complete_and_ordered():
    assert stages.STAGES == (
        "stage1_reconstruction", "stage2_contrastive", "stage3_forecast",
    )
