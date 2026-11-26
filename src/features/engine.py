"""
Multi-Factor Feature Engineering & Cross-Sectional Normalization Engine.

Computes survivorship-free cumulative total return indices, multi-horizon momentum,
short-term mean reversion, realized volatility, and liquidity dynamics, followed
by strict point-in-time cross-sectional winsorization and Z-score standardization.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_FACTOR_COLUMNS: tuple[str, ...] = (
    "mom_21d",
    "mom_63d",
    "mom_12m_1m",
    "rev_5d",
    "vol_21d",
    "adv_21d",
)


class CrossSectionalFeatureEngine:
    """Vectorized multi-factor feature engineering and cross-sectional normalization."""

    CUM_RET_COL: str = "cum_ret_index"
    RAW_FACTOR_COLS: tuple[str, ...] = DEFAULT_FACTOR_COLUMNS

    MOM_1M_WINDOW: int = 21
    MOM_3M_WINDOW: int = 63
    MOM_12M_WINDOW: int = 252
    MOM_SKIP_WINDOW: int = 21
    REV_WINDOW: int = 5
    VOL_WINDOW: int = 21
    ADV_WINDOW: int = 21
    TRADING_DAYS_PER_YEAR: float = 252.0

    def __init__(
        self,
        ret_col: str = "ret_adj",
        prc_col: str = "prc",
        vol_col: str = "vol",
        cum_ret_col: str = CUM_RET_COL,
        initial_index: float = 1.0,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        min_constituents: int = 2,
        eps: float = 1e-12,
    ) -> None:
        """Initialize the cross-sectional feature engine.

        Args:
            ret_col: Column name for daily delisting-adjusted returns (default: 'ret_adj').
            prc_col: Column name for closing price (default: 'prc').
            vol_col: Column name for daily share trading volume (default: 'vol').
            cum_ret_col: Output column name for cumulative total return index.
            initial_index: Starting wealth level I_{i,0} for return accumulation (default: 1.0).
            lower_q: Lower quantile for two-tailed cross-sectional winsorization (default: 0.01).
            upper_q: Upper quantile for two-tailed cross-sectional winsorization (default: 0.99).
            min_constituents: Minimum non-NaN observations per date required for
                cross-sectional Z-scoring (default: 2).
            eps: Minimum cross-sectional standard deviation threshold below which
                Z-scores evaluate to NaN to guard against zero-variance division.
        """
        self._validate_quantiles(lower_q, upper_q)
        if min_constituents < 2:
            raise ValueError("min_constituents must be at least 2 for sample std (ddof=1).")
        if initial_index <= 0.0:
            raise ValueError("initial_index must be strictly positive.")

        self.ret_col = ret_col
        self.prc_col = prc_col
        self.vol_col = vol_col
        self.cum_ret_col = cum_ret_col
        self.initial_index = float(initial_index)
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.min_constituents = int(min_constituents)
        self.eps = float(eps)

    @staticmethod
    def _validate_quantiles(lower_q: float, upper_q: float) -> None:
        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError(
                f"Quantiles must satisfy 0.0 <= lower_q < upper_q <= 1.0; got ({lower_q}, {upper_q})."
            )

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: Sequence[str], name: str = "DataFrame") -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

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
        """Restore MultiIndex or flat column layout matching caller's input format."""
        return panel if was_multiindex else panel.reset_index()

    @staticmethod
    def _safe_index_ratio(
        numerator: pd.Series, denominator: pd.Series
    ) -> pd.Series:
        """Compute (numerator / denominator) - 1 safely, returning NaN when denominator is 0."""
        valid_denom = denominator.where(denominator != 0.0)
        return (numerator / valid_denom) - 1.0

    def compute_cumulative_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct continuous cumulative total return index I_{i,t} per permno.

        Missing/NaN daily returns are treated as 0.0 during compounding so the
        accumulation index remains continuous across temporary trading halts.

        Args:
            df: Input panel with ['date', 'permno', ret_col].

        Returns:
            DataFrame copy with added cumulative return index column.
        """
        panel, was_multiindex = self._normalize_panel(df)
        self._validate_columns(panel, [self.ret_col], "Input DataFrame")

        if panel.empty:
            panel[self.cum_ret_col] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        ret_series = pd.to_numeric(panel[self.ret_col], errors="coerce").astype("float64")
        gross_ret = 1.0 + ret_series.fillna(0.0)
        panel[self.cum_ret_col] = (
            self.initial_index * gross_ret.groupby(level="permno", sort=False).cumprod()
        )
        return self._restore_format(panel, was_multiindex)

    def compute_raw_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw multi-horizon momentum, reversal, volatility, and liquidity factors.

        Computed columns:
            - mom_21d:    I_{i,t} / I_{i,t-21} - 1
            - mom_63d:    I_{i,t} / I_{i,t-63} - 1
            - mom_12m_1m: I_{i,t-21} / I_{i,t-252} - 1
            - rev_5d:     I_{i,t} / I_{i,t-5} - 1
            - vol_21d:    Std_{21}(ret_adj) * sqrt(252)
            - adv_21d:    Median_{21}(|prc| * vol)

        Args:
            df: Input panel containing ['date', 'permno', ret_col, prc_col, vol_col]
                and optionally precomputed cumulative return index.

        Returns:
            DataFrame copy enriched with raw factor columns.
        """
        panel, was_multiindex = self._normalize_panel(df)
        self._validate_columns(panel, [self.ret_col, self.prc_col, self.vol_col], "Input DataFrame")

        if panel.empty:
            if self.cum_ret_col not in panel.columns:
                panel[self.cum_ret_col] = pd.Series(dtype="float64", index=panel.index)
            for col in self.RAW_FACTOR_COLS:
                panel[col] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        if self.cum_ret_col not in panel.columns:
            panel_with_cum, _ = self._normalize_panel(self.compute_cumulative_returns(panel))
            panel[self.cum_ret_col] = panel_with_cum[self.cum_ret_col]

        cum_idx = pd.to_numeric(panel[self.cum_ret_col], errors="coerce").astype("float64")
        cum_grouped = cum_idx.groupby(level="permno", sort=False)

        lag_5 = cum_grouped.shift(self.REV_WINDOW)
        lag_21 = cum_grouped.shift(self.MOM_1M_WINDOW)
        lag_63 = cum_grouped.shift(self.MOM_3M_WINDOW)
        lag_252 = cum_grouped.shift(self.MOM_12M_WINDOW)

        panel["mom_21d"] = self._safe_index_ratio(cum_idx, lag_21)
        panel["mom_63d"] = self._safe_index_ratio(cum_idx, lag_63)
        panel["mom_12m_1m"] = self._safe_index_ratio(lag_21, lag_252)
        panel["rev_5d"] = self._safe_index_ratio(cum_idx, lag_5)

        ret_series = pd.to_numeric(panel[self.ret_col], errors="coerce").astype("float64")
        vol_21d = (
            ret_series.groupby(level="permno", sort=False)
            .rolling(window=self.VOL_WINDOW, min_periods=self.VOL_WINDOW)
            .std(ddof=1)
            .droplevel(0)
            .reindex(panel.index)
            * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        )
        panel["vol_21d"] = vol_21d

        prc_series = pd.to_numeric(panel[self.prc_col], errors="coerce").astype("float64")
        vol_series = pd.to_numeric(panel[self.vol_col], errors="coerce").astype("float64")
        dollar_vol = prc_series.abs() * vol_series
        adv_21d = (
            dollar_vol.groupby(level="permno", sort=False)
            .rolling(window=self.ADV_WINDOW, min_periods=self.ADV_WINDOW)
            .median()
            .droplevel(0)
            .reindex(panel.index)
        )
        panel["adv_21d"] = adv_21d

        return self._restore_format(panel, was_multiindex)

    def _resolve_factor_cols(
        self, panel: pd.DataFrame, factor_cols: Optional[Sequence[str]]
    ) -> list[str]:
        if factor_cols is None:
            cols = [c for c in self.RAW_FACTOR_COLS if c in panel.columns]
            if not cols:
                raise ValueError(
                    "No default raw factor columns found in DataFrame; pass factor_cols explicitly "
                    "or run compute_raw_factors() first."
                )
            return cols
        cols = list(factor_cols)
        self._validate_columns(panel, cols, "Input DataFrame")
        return cols

    def winsorize_cross_section(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[Sequence[str]] = None,
        lower_q: Optional[float] = None,
        upper_q: Optional[float] = None,
    ) -> pd.DataFrame:
        """Apply strict point-in-time two-tailed cross-sectional winsorization per date.

        Args:
            df: Input panel containing ['date', 'permno'] and factor_cols.
            factor_cols: Columns to winsorize. Defaults to present RAW_FACTOR_COLS.
            lower_q: Lower quantile bound override (default: self.lower_q).
            upper_q: Upper quantile bound override (default: self.upper_q).

        Returns:
            DataFrame copy with specified factor_cols clipped to [Q_t(lower_q), Q_t(upper_q)].
        """
        lq = self.lower_q if lower_q is None else float(lower_q)
        uq = self.upper_q if upper_q is None else float(upper_q)
        self._validate_quantiles(lq, uq)

        panel, was_multiindex = self._normalize_panel(df)
        cols = self._resolve_factor_cols(panel, factor_cols)

        if panel.empty or not cols:
            return self._restore_format(panel, was_multiindex)

        for col in cols:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float64")

        grouped = panel.groupby(level="date", sort=False)[cols]
        lower_bounds = grouped.transform("quantile", lq)
        upper_bounds = grouped.transform("quantile", uq)
        panel[cols] = panel[cols].clip(lower=lower_bounds, upper=upper_bounds, axis=0)

        return self._restore_format(panel, was_multiindex)

    def standardize_cross_section(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[Sequence[str]] = None,
        min_constituents: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply point-in-time cross-sectional Z-score standardization per date.

        Produces `<col>_z` columns enforcing zero mean and unit sample variance per date.
        Cross-sections with zero variance (std <= eps) or fewer than min_constituents
        valid observations safely yield NaN.

        Args:
            df: Input panel containing ['date', 'permno'] and factor_cols.
            factor_cols: Columns to standardize. Defaults to present RAW_FACTOR_COLS.
            min_constituents: Override minimum valid constituent count per date.

        Returns:
            DataFrame copy enriched with `<col>_z` columns.
        """
        min_obs = self.min_constituents if min_constituents is None else int(min_constituents)
        if min_obs < 2:
            raise ValueError("min_constituents must be at least 2 for sample std (ddof=1).")

        panel, was_multiindex = self._normalize_panel(df)
        cols = self._resolve_factor_cols(panel, factor_cols)
        z_cols = [f"{col}_z" for col in cols]

        if panel.empty or not cols:
            for z_col in z_cols:
                panel[z_col] = pd.Series(dtype="float64", index=panel.index)
            return self._restore_format(panel, was_multiindex)

        numeric_factors = panel[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        grouped = numeric_factors.groupby(level="date", sort=False)

        means = grouped.transform("mean")
        stds = grouped.transform("std", ddof=1)
        counts = grouped.transform("count")

        valid_cs = (counts >= min_obs) & (stds > self.eps) & stds.notna()
        z_values = (numeric_factors - means).div(stds.where(valid_cs))
        z_values.columns = z_cols
        panel[z_cols] = z_values

        return self._restore_format(panel, was_multiindex)

    def generate_feature_panel(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[Sequence[str]] = None,
        lower_q: Optional[float] = None,
        upper_q: Optional[float] = None,
        min_constituents: Optional[int] = None,
        dropna_z: bool = False,
    ) -> pd.DataFrame:
        """Execute full end-to-end feature engineering and normalization pipeline.

        Computes cumulative return index and raw factors, applies point-in-time
        cross-sectional winsorization followed by Z-score standardization, and returns
        both the raw factors and normalized `<factor>_z` features.

        Args:
            df: Input panel containing ['date', 'permno', ret_col, prc_col, vol_col].
            factor_cols: Subset of factors to normalize (defaults to RAW_FACTOR_COLS).
            lower_q: Lower quantile override for winsorization.
            upper_q: Upper quantile override for winsorization.
            min_constituents: Minimum valid constituents per date for Z-scoring.
            dropna_z: If True, drops rows where any `<factor>_z` column is NaN.

        Returns:
            Panel DataFrame containing cumulative index, raw factors, and normalized `*_z` features.
        """
        panel, was_multiindex = self._normalize_panel(df)
        panel, _ = self._normalize_panel(self.compute_raw_factors(panel))

        cols = self._resolve_factor_cols(panel, factor_cols)
        z_cols = [f"{col}_z" for col in cols]

        winsorized, _ = self._normalize_panel(
            self.winsorize_cross_section(panel, factor_cols=cols, lower_q=lower_q, upper_q=upper_q)
        )
        standardized, _ = self._normalize_panel(
            self.standardize_cross_section(
                winsorized, factor_cols=cols, min_constituents=min_constituents
            )
        )
        panel[z_cols] = standardized[z_cols]

        if dropna_z and z_cols:
            panel = panel.dropna(subset=z_cols)

        return self._restore_format(panel, was_multiindex)
