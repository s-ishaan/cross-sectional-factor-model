"""
Dollar-Neutral Quantile Portfolio Allocation Engine.

Maps cross-sectional alpha predictions (e.g., out-of-fold ML scores or percentile ranks)
into strictly dollar-neutral long/short portfolio weights with configurable quantile
cutoffs and equal or rank-proportional intra-leg weighting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class QuantilePortfolioAllocator:
    """Constructs dollar-neutral long/short portfolios from quantile alpha sorts."""

    VALID_SCHEMES: tuple[str, ...] = ("equal", "rank_weighted")

    def __init__(
        self,
        num_quantiles: int = 10,
        top_quantile: Optional[int] = None,
        bottom_quantile: int = 0,
        weighting_scheme: str = "equal",
        signal_col: Optional[str] = None,
        gross_leverage: float = 1.0,
    ) -> None:
        """Initialize the quantile portfolio allocator.

        Args:
            num_quantiles: Number of cross-sectional quantile buckets (default: 10).
            top_quantile: Long bucket index in [0, num_quantiles - 1]. Defaults to
                num_quantiles - 1 (e.g., decile 9 for 10 quantiles).
            bottom_quantile: Short bucket index in [0, num_quantiles - 1] (default: 0).
            weighting_scheme: Intra-leg weighting rule: 'equal' or 'rank_weighted'.
            signal_col: Explicit prediction column name. When None, automatically
                selects 'predicted_rank' if present, falling back to 'pred'.
            gross_leverage: Total gross exposure sum(|w_t|) (default: 1.0, giving
                +0.5 long sum and -0.5 short sum).
        """
        if num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2.")
        resolved_top = (num_quantiles - 1) if top_quantile is None else int(top_quantile)
        resolved_bottom = int(bottom_quantile)

        if not (0 <= resolved_bottom < num_quantiles):
            raise ValueError(
                f"bottom_quantile must be in [0, {num_quantiles - 1}]; got {resolved_bottom}."
            )
        if not (0 <= resolved_top < num_quantiles):
            raise ValueError(
                f"top_quantile must be in [0, {num_quantiles - 1}]; got {resolved_top}."
            )
        if resolved_top == resolved_bottom:
            raise ValueError("top_quantile and bottom_quantile must be distinct.")
        if weighting_scheme not in self.VALID_SCHEMES:
            raise ValueError(
                f"weighting_scheme must be one of {self.VALID_SCHEMES}; got '{weighting_scheme}'."
            )
        if gross_leverage <= 0.0:
            raise ValueError("gross_leverage must be strictly positive.")

        self.num_quantiles = int(num_quantiles)
        self.top_quantile = resolved_top
        self.bottom_quantile = resolved_bottom
        self.weighting_scheme = weighting_scheme
        self.signal_col = signal_col
        self.gross_leverage = float(gross_leverage)

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
            return panel.set_index(["date", "permno"]), was_multiindex

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

    def _resolve_signal_col(
        self, panel: pd.DataFrame, signal_col: Optional[str]
    ) -> str:
        col = self.signal_col if signal_col is None else signal_col
        if col is not None:
            if col not in panel.columns:
                raise ValueError(f"Signal column '{col}' not found in DataFrame.")
            return col
        for candidate in ("predicted_rank", "pred"):
            if candidate in panel.columns:
                return candidate
        raise ValueError(
            "Predictions DataFrame must contain 'predicted_rank' or 'pred', "
            "or signal_col must be specified explicitly."
        )

    @staticmethod
    def _assign_quantiles(series: pd.Series, num_quantiles: int) -> pd.Series:
        valid = series.dropna()
        if len(valid) < num_quantiles:
            return pd.Series(np.nan, index=series.index, dtype="float64")
        ranked = valid.rank(method="first")
        bins = pd.qcut(ranked, q=num_quantiles, labels=False).astype("float64")
        return bins.reindex(series.index)

    def _compute_date_weights(
        self, signal_cs: pd.Series, quantile_cs: pd.Series
    ) -> pd.Series:
        weights = pd.Series(0.0, index=signal_cs.index, dtype="float64")
        long_mask = (quantile_cs == self.top_quantile) & signal_cs.notna()
        short_mask = (quantile_cs == self.bottom_quantile) & signal_cs.notna()

        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long == 0 or n_short == 0:
            return weights

        leg_exposure = 0.5 * self.gross_leverage
        if self.weighting_scheme == "equal":
            weights.loc[long_mask] = leg_exposure / n_long
            weights.loc[short_mask] = -leg_exposure / n_short
        else:
            # Higher signal gets strictly higher positive weight in the long leg;
            # lower signal gets strictly more negative weight in the short leg.
            long_ranks = signal_cs.loc[long_mask].rank(method="average", ascending=True)
            short_ranks = signal_cs.loc[short_mask].rank(method="average", ascending=False)
            weights.loc[long_mask] = leg_exposure * (long_ranks / long_ranks.sum())
            weights.loc[short_mask] = -leg_exposure * (short_ranks / short_ranks.sum())

        return weights

    def allocate(
        self,
        predictions_df: pd.DataFrame,
        signal_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Map cross-sectional predictions to dollar-neutral long/short weights.

        Args:
            predictions_df: DataFrame indexed by ['date', 'permno'] (or containing them
                as columns) with 'predicted_rank' or 'pred'.
            signal_col: Optional override for prediction signal column.

        Returns:
            DataFrame copy preserving input layout with added 'quantile' and 'weight'
            columns. On every active date, long weights sum to +0.5 * gross_leverage
            and short weights sum to -0.5 * gross_leverage.
        """
        panel, was_multiindex = self._normalize_panel(predictions_df)

        if panel.empty:
            panel["quantile"] = pd.Series(dtype="float64", index=panel.index)
            panel["weight"] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        resolved_col = self._resolve_signal_col(panel, signal_col)
        signals = pd.to_numeric(panel[resolved_col], errors="coerce").astype("float64")

        quantiles = (
            signals.groupby(level="date", sort=False)
            .transform(self._assign_quantiles, num_quantiles=self.num_quantiles)
            .astype("float64")
        )

        weights = pd.Series(0.0, index=panel.index, dtype="float64")
        for _, date_idx in panel.groupby(level="date", sort=False).groups.items():
            weights.loc[date_idx] = self._compute_date_weights(
                signals.loc[date_idx], quantiles.loc[date_idx]
            )

        panel["quantile"] = quantiles
        panel["weight"] = weights
        return self._restore_format(panel, was_multiindex)

    def allocate_matrix(
        self,
        predictions_df: pd.DataFrame,
        signal_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Allocate predictions and return a wide (date x permno) portfolio weight matrix."""
        allocated = self.allocate(predictions_df, signal_col=signal_col)
        panel, _ = self._normalize_panel(allocated)
        if panel.empty:
            return pd.DataFrame(dtype="float64")
        wide = panel["weight"].unstack(level="permno").fillna(0.0).astype("float64")
        wide.index = pd.DatetimeIndex(wide.index, name="date")
        wide.columns.name = "permno"
        return wide.sort_index().sort_index(axis=1)
