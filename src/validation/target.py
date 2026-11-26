"""
Non-Anticipatory Forward Return Target Formulation Module.

Constructs execution-lagged forward return targets (t + lag -> t + lag + h) from
cumulative total return indices and partitions cross-sections into deterministic
ordinal quantile labels for cross-sectional machine learning models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ForwardReturnTargetBuilder:
    """Builds non-anticipatory forward returns and cross-sectional quantile targets."""

    DEFAULT_HOLDING_PERIOD: int = 5
    DEFAULT_EXECUTION_LAG: int = 1
    DEFAULT_CUM_RET_COL: str = "cum_ret"
    FALLBACK_CUM_RET_COL: str = "cum_ret_index"
    DEFAULT_NUM_QUANTILES: int = 10
    DEFAULT_OUTPUT_QUANTILE_COL: str = "target_quantile"

    def __init__(
        self,
        holding_period: int = DEFAULT_HOLDING_PERIOD,
        execution_lag: int = DEFAULT_EXECUTION_LAG,
        cum_ret_col: str = DEFAULT_CUM_RET_COL,
        num_quantiles: int = DEFAULT_NUM_QUANTILES,
    ) -> None:
        """Initialize the forward return target builder.

        Args:
            holding_period: Forward holding horizon in trading days (default: 5).
            execution_lag: Trading days between signal close and execution close
                (default: 1, executing at t+1 close).
            cum_ret_col: Column name for cumulative total return index (default: 'cum_ret',
                automatically falling back to 'cum_ret_index' if default is absent).
            num_quantiles: Number of ordinal cross-sectional quantiles (default: 10).
        """
        if holding_period < 1:
            raise ValueError("holding_period must be >= 1.")
        if execution_lag < 0:
            raise ValueError("execution_lag must be >= 0.")
        if num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2.")

        self.holding_period = int(holding_period)
        self.execution_lag = int(execution_lag)
        self.cum_ret_col = cum_ret_col
        self.num_quantiles = int(num_quantiles)

    @staticmethod
    def _normalize_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Normalize input DataFrame into a chronologically sorted ['date', 'permno'] MultiIndex."""
        if isinstance(df.index, pd.MultiIndex) and {"date", "permno"}.issubset(
            set(df.index.names)
        ):
            was_multiindex = True
            panel = df.reset_index()
        elif {"date", "permno"}.issubset(set(df.columns)):
            was_multiindex = False
            panel = df.copy()
        else:
            raise ValueError(
                "Input DataFrame must contain ['date', 'permno'] as MultiIndex levels or columns."
            )

        if panel.empty:
            panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
            panel["permno"] = pd.Series(dtype="int64")
            panel = panel.set_index(["date", "permno"])
            return panel, was_multiindex

        panel = panel.dropna(subset=["date", "permno"]).copy()
        panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
        panel["permno"] = pd.to_numeric(panel["permno"], errors="coerce")
        panel = panel.dropna(subset=["permno"]).copy()
        panel["permno"] = panel["permno"].astype("int64")

        if panel.duplicated(subset=["date", "permno"]).any():
            panel = panel.drop_duplicates(subset=["date", "permno"], keep="last")

        panel = panel.set_index(["date", "permno"]).sort_index()
        return panel, was_multiindex

    @staticmethod
    def _restore_format(panel: pd.DataFrame, was_multiindex: bool) -> pd.DataFrame:
        return panel if was_multiindex else panel.reset_index()

    def _resolve_cum_ret_col(self, panel: pd.DataFrame, cum_ret_col: Optional[str]) -> str:
        col = self.cum_ret_col if cum_ret_col is None else cum_ret_col
        if col in panel.columns:
            return col
        if (
            col == self.DEFAULT_CUM_RET_COL
            and self.FALLBACK_CUM_RET_COL in panel.columns
        ):
            return self.FALLBACK_CUM_RET_COL
        raise ValueError(
            f"Cumulative return index column '{col}' not found in DataFrame."
        )

    def compute_forward_returns(
        self,
        df: pd.DataFrame,
        holding_period: Optional[int] = None,
        execution_lag: Optional[int] = None,
        cum_ret_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute non-anticipatory forward returns from cumulative wealth indices.

        Implements:
            R^{fwd}_{i, t+lag -> t+lag+h} = (I_{i, t+lag+h} - I_{i, t+lag}) / I_{i, t+lag}

        Args:
            df: Input panel with ['date', 'permno'] and cumulative return index column.
            holding_period: Override holding horizon h in trading days (default: 5).
            execution_lag: Override execution lag in trading days (default: 1).
            cum_ret_col: Override cumulative return index column name.

        Returns:
            DataFrame copy enriched with 'fwd_ret_<holding_period>d'.
        """
        h = self.holding_period if holding_period is None else int(holding_period)
        lag = self.execution_lag if execution_lag is None else int(execution_lag)
        if h < 1:
            raise ValueError("holding_period must be >= 1.")
        if lag < 0:
            raise ValueError("execution_lag must be >= 0.")

        panel, was_multiindex = self._normalize_panel(df)
        out_col = f"fwd_ret_{h}d"

        if panel.empty:
            panel[out_col] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        resolved_col = self._resolve_cum_ret_col(panel, cum_ret_col)
        cum_idx = pd.to_numeric(panel[resolved_col], errors="coerce").astype("float64")
        grouped = cum_idx.groupby(level="permno", sort=False)

        buy_cum = grouped.shift(-lag)
        sell_cum = grouped.shift(-(lag + h))
        valid_buy = buy_cum.where(buy_cum != 0.0)
        panel[out_col] = (sell_cum - valid_buy) / valid_buy

        return self._restore_format(panel, was_multiindex)

    @staticmethod
    def _rank_to_quantiles(series: pd.Series, num_quantiles: int) -> pd.Series:
        valid = series.dropna()
        if len(valid) < num_quantiles:
            return pd.Series(np.nan, index=series.index, dtype="float64")
        ranked = valid.rank(method="first")
        bins = pd.qcut(ranked, q=num_quantiles, labels=False).astype("float64")
        return bins.reindex(series.index)

    def discretize_quantiles(
        self,
        df: pd.DataFrame,
        target_col: str = "fwd_ret_5d",
        num_quantiles: Optional[int] = None,
        output_col: str = DEFAULT_OUTPUT_QUANTILE_COL,
    ) -> pd.DataFrame:
        """Partition continuous forward returns into ordinal cross-sectional quantiles per date.

        Uses deterministic first-occurrence rank sorting (rank(method='first')) before
        pd.qcut so tied returns receive unique, reproducible quantile assignments in
        {0, 1, ..., K-1}. Cross-sections with fewer than K valid observations yield NaN.

        Args:
            df: Input panel containing ['date', 'permno', target_col].
            target_col: Continuous forward return column to discretize.
            num_quantiles: Number of quantiles K (default: self.num_quantiles).
            output_col: Output column name (default: 'target_quantile').

        Returns:
            DataFrame copy enriched with ordinal quantile target column.
        """
        k = self.num_quantiles if num_quantiles is None else int(num_quantiles)
        if k < 2:
            raise ValueError("num_quantiles must be >= 2.")

        panel, was_multiindex = self._normalize_panel(df)
        if target_col not in panel.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        if panel.empty:
            panel[output_col] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        values = pd.to_numeric(panel[target_col], errors="coerce").astype("float64")
        panel[output_col] = (
            values.groupby(level="date", sort=False)
            .transform(self._rank_to_quantiles, num_quantiles=k)
            .astype("float64")
        )
        return self._restore_format(panel, was_multiindex)

    def build_target_panel(
        self,
        df: pd.DataFrame,
        holding_period: Optional[int] = None,
        num_quantiles: Optional[int] = None,
        execution_lag: Optional[int] = None,
        cum_ret_col: Optional[str] = None,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """Execute full target construction: forward returns followed by quantile binning.

        Args:
            df: Input panel with ['date', 'permno'] and cumulative return index.
            holding_period: Holding period h in trading days (default: 5).
            num_quantiles: Number of ordinal quantile bins K (default: 10).
            execution_lag: Execution lag in trading days (default: 1).
            cum_ret_col: Optional cumulative return index column override.
            dropna: If True, drops rows where 'target_quantile' is NaN.

        Returns:
            DataFrame copy containing 'fwd_ret_<h>d' and 'target_quantile'.
        """
        h = self.holding_period if holding_period is None else int(holding_period)
        panel_fwd = self.compute_forward_returns(
            df,
            holding_period=h,
            execution_lag=execution_lag,
            cum_ret_col=cum_ret_col,
        )
        panel_out = self.discretize_quantiles(
            panel_fwd,
            target_col=f"fwd_ret_{h}d",
            num_quantiles=num_quantiles,
        )
        if dropna:
            panel_out = panel_out.dropna(subset=[self.DEFAULT_OUTPUT_QUANTILE_COL])
        return panel_out
