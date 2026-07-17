from scripts.train_moirai2_tournament import PARENT_LENGTH
from futures_foundation.finetune.tournament import FORECAST_HORIZON, MAX_CONTEXT


def test_moirai_parent_is_exact_context_plus_future():
    assert PARENT_LENGTH == MAX_CONTEXT + FORECAST_HORIZON
    assert FORECAST_HORIZON == 16
