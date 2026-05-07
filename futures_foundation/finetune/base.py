from abc import ABC, abstractmethod

import pandas as pd


class StrategyLabeler(ABC):
    """
    Implement this to plug a new strategy into the fine-tuning framework.
    Cell 3 of any strategy script is a concrete StrategyLabeler subclass.

    The framework handles all I/O, caching, training, evaluation, and ONNX
    export. The only contract is:
      - name:         short slug used in log output
      - feature_cols: ordered list of strategy-specific column names
      - run():        map raw OHLCV + FFM features → (strategy_features, labels)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in log output (e.g. 'cisd_ote', 'orb')."""
        ...

    @property
    @abstractmethod
    def feature_cols(self) -> list:
        """Ordered column names of the strategy-specific feature array."""
        ...

    def config_dict(self) -> dict:
        """
        Return a JSON-serialisable dict of all parameters that affect labeling
        output.  The framework hashes this (plus tickers + feature_cols) to
        decide whether the cached parquet files are still valid.

        Override this in every concrete labeler.  Include the labeling version,
        all thresholds, and any other constants that, if changed, would produce
        different labels.  Do NOT include paths — only logical parameters.
        """
        return {}

    @abstractmethod
    def run(
        self,
        df_raw: pd.DataFrame,
        ffm_df: pd.DataFrame,
        ticker: str,
    ) -> tuple:
        """
        Compute strategy labels and features for one ticker.

        Args:
            df_raw:  Raw 5-min OHLCV DataFrame, indexed by tz-aware NY datetime.
                     Columns: open, high, low, close, volume (lowercase).
            ffm_df:  FFM-prepared features parquet for the same ticker,
                     indexed by tz-aware NY datetime.  Use it for any HTF
                     context (htf_1h_structure, vty_atr_raw, etc.) rather
                     than recomputing from raw bars.
            ticker:  Instrument symbol (e.g. 'MES', 'ES').

        Returns:
            strategy_features_df: pd.DataFrame with columns == self.feature_cols,
                                  aligned to ffm_df.index.
            labels_df:            pd.DataFrame with required columns:
                                    signal_label  int8   (0 = no trade, >0 = trade)
                                    max_rr        float32 (best RR achieved)
                                    sl_distance   float32 (stop distance, optional)
                                  aligned to ffm_df.index.
        """
        ...
