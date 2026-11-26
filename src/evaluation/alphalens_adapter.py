"""
Alphalens Factor Attribution Adapter.

Transforms out-of-fold cross-sectional predictions and asset price / cumulative return
panels into clean MultiIndex factor-and-forward-returns structures compatible with
`alphalens.utils.get_clean_factor_and_forward_returns` across 1-day, 5-day, and 10-day
evaluation horizons.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import alphalens as al
import pandas as pd


class AlphalensAdapter:
    """Formats model predictions and pricing panels for multi-horizon Alphalens analysis."""

    DEFAULT_PERIODS: tuple[int, ...] = (1, 5, 10)

    def __init__(
        self,
        periods: Sequence[int] = DEFAULT_PERIODS,
        quantiles: Optional[int] = None,
        filter_zscore: Optional[float] = None,
        max_loss: float = 0.35,
        zero_aware: bool = False,
    ) -> None:
        """Initialize the Alphalens adapter.

        Args:
            periods: Forward holding horizons in trading days (default: (1, 5, 10)).
            quantiles: Number of factor quantiles. When None, defaults to 10 if universe
                has >= 50 assets, else min(3, n_assets).
            filter_zscore: Optional outlier z-score filter passed to Alphalens (default: None).
            max_loss: Maximum fraction of dropped observations allowed by Alphalens (default: 0.35).
            zero_aware: Whether quantile binning separates positive/negative values (default: False).
        """
        if not periods or any(int(p) < 1 for p in periods):
            raise ValueError("periods must be a non-empty sequence of positive integers.")
        if quantiles is not None and int(quantiles) < 2:
            raise ValueError("quantiles must be >= 2 when specified.")
        if not (0.0 <= max_loss <= 1.0):
            raise ValueError("max_loss must be in [0.0, 1.0].")

        self.periods: tuple[int, ...] = tuple(int(p) for p in periods)
        self.quantiles: Optional[int] = int(quantiles) if quantiles is not None else None
        self.filter_zscore = filter_zscore
        self.max_loss = float(max_loss)
        self.zero_aware = bool(zero_aware)

    @staticmethod
    def prepare_factor(
        predictions_df: Union[pd.DataFrame, pd.Series],
        factor_col: Optional[str] = None,
    ) -> pd.Series:
        """Format predictions into a float64 Series with MultiIndex ['date', 'asset']."""
        if isinstance(predictions_df, pd.Series):
            if not isinstance(predictions_df.index, pd.MultiIndex) or len(
                predictions_df.index.levels
            ) != 2:
                raise ValueError(
                    "Series predictions must have a 2-level MultiIndex ['date', 'permno'/'asset']."
                )
            df = predictions_df.to_frame(name="_factor").reset_index()
            level_names = list(predictions_df.index.names)
            date_col = level_names[0] if level_names[0] is not None else "level_0"
            asset_col = level_names[1] if level_names[1] is not None else "level_1"
            col = "_factor"
        elif isinstance(predictions_df, pd.DataFrame):
            has_multi = isinstance(predictions_df.index, pd.MultiIndex) and len(
                predictions_df.index.levels
            ) == 2
            if has_multi:
                df = predictions_df.reset_index()
                date_col = (
                    "date"
                    if "date" in df.columns
                    else df.columns[0]
                )
                asset_col = (
                    "permno"
                    if "permno" in df.columns
                    else ("asset" if "asset" in df.columns else df.columns[1])
                )
            elif "date" in predictions_df.columns and (
                "permno" in predictions_df.columns or "asset" in predictions_df.columns
            ):
                df = predictions_df.copy()
                date_col = "date"
                asset_col = "permno" if "permno" in df.columns else "asset"
            else:
                raise ValueError(
                    "predictions_df must be MultiIndexed by ['date', 'permno'/'asset'] "
                    "or contain them as columns."
                )

            if factor_col is not None:
                if factor_col not in df.columns:
                    raise ValueError(f"Factor column '{factor_col}' not found in predictions_df.")
                col = factor_col
            else:
                col = next(
                    (c for c in ("predicted_rank", "pred", "factor") if c in df.columns),
                    None,
                )
                if col is None:
                    raise ValueError(
                        "predictions_df must contain 'predicted_rank', 'pred', or 'factor', "
                        "or factor_col must be provided."
                    )
        else:
            raise TypeError(f"Unsupported predictions_df type: {type(predictions_df)}")

        df = df.dropna(subset=[date_col, asset_col, col]).copy()
        df[date_col] = pd.to_datetime(df[date_col]).astype("datetime64[ns]")
        df[asset_col] = pd.to_numeric(df[asset_col], errors="coerce")
        df = df.dropna(subset=[asset_col]).copy()
        df[asset_col] = df[asset_col].astype("int64")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = df.dropna(subset=[col])

        if df.duplicated(subset=[date_col, asset_col]).any():
            df = df.drop_duplicates(subset=[date_col, asset_col], keep="last")

        df = df.rename(columns={date_col: "date", asset_col: "asset"})
        factor = df.set_index(["date", "asset"])[col].sort_index().astype("float64")
        factor.name = col
        return factor

    @staticmethod
    def prepare_prices(
        prices_data: pd.DataFrame,
        price_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Format wide or panel pricing data into a float64 (date x asset) DataFrame."""
        has_multi = isinstance(prices_data.index, pd.MultiIndex) and len(
            prices_data.index.levels
        ) == 2
        has_cols = "date" in prices_data.columns and (
            "permno" in prices_data.columns or "asset" in prices_data.columns
        )

        if has_multi or has_cols:
            df = prices_data.reset_index() if has_multi else prices_data.copy()
            date_col = "date" if "date" in df.columns else df.columns[0]
            asset_col = (
                "permno"
                if "permno" in df.columns
                else ("asset" if "asset" in df.columns else df.columns[1])
            )
            if price_col is not None:
                if price_col not in df.columns:
                    raise ValueError(f"Price column '{price_col}' not found in prices_data.")
                col = price_col
            else:
                col = next(
                    (
                        c
                        for c in ("cum_ret_index", "cum_ret", "close", "prc", "price")
                        if c in df.columns
                    ),
                    None,
                )
                if col is None:
                    raise ValueError(
                        "Cannot infer price column; pass price_col explicitly "
                        "(e.g. 'cum_ret_index', 'cum_ret', or 'prc')."
                    )

            df = df.dropna(subset=[date_col, asset_col]).copy()
            df[date_col] = pd.to_datetime(df[date_col]).astype("datetime64[ns]")
            df[asset_col] = pd.to_numeric(df[asset_col], errors="coerce")
            df = df.dropna(subset=[asset_col]).copy()
            df[asset_col] = df[asset_col].astype("int64")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            if col == "prc":
                df[col] = df[col].abs()
            if df.duplicated(subset=[date_col, asset_col]).any():
                df = df.drop_duplicates(subset=[date_col, asset_col], keep="last")

            wide = df.pivot(index=date_col, columns=asset_col, values=col)
            wide.index = pd.DatetimeIndex(wide.index, name="date")
            wide.columns = pd.Index(wide.columns.astype("int64"), name="asset")
            return wide.sort_index().sort_index(axis=1).astype("float64")

        wide = prices_data.copy()
        wide.index = pd.to_datetime(wide.index).astype("datetime64[ns]")
        wide.index.name = "date"
        wide.columns = pd.Index([int(c) for c in wide.columns], name="asset")
        return wide.apply(pd.to_numeric, errors="coerce").astype("float64").sort_index().sort_index(axis=1)

    def resolve_quantiles(self, n_assets: int, quantiles: Optional[int] = None) -> int:
        """Determine quantile count dynamically (10 for >= 50 assets, else min(3, n_assets))."""
        q = self.quantiles if quantiles is None else int(quantiles)
        if q is not None:
            return q
        return 10 if n_assets >= 50 else max(2, min(3, n_assets))

    def build_factor_data(
        self,
        predictions_df: Union[pd.DataFrame, pd.Series],
        prices_data: pd.DataFrame,
        factor_col: Optional[str] = None,
        price_col: Optional[str] = None,
        periods: Optional[Sequence[int]] = None,
        quantiles: Optional[int] = None,
        filter_zscore: Optional[float] = None,
        max_loss: Optional[float] = None,
        zero_aware: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Build Alphalens clean factor and forward returns MultiIndex DataFrame.

        Args:
            predictions_df: Predictions DataFrame/Series with 'predicted_rank' or 'pred'.
            prices_data: Wide (date x asset) price matrix or long panel with pricing/cum_ret.
            factor_col: Optional signal column override.
            price_col: Optional price column override for long panels.
            periods: Holding period tuple override (default: self.periods, i.e., (1, 5, 10)).
            quantiles: Quantile count override.
            filter_zscore: Z-score filter override.
            max_loss: Max loss tolerance override.
            zero_aware: Zero-aware binning override.

        Returns:
            DataFrame MultiIndexed by ['date', 'asset'] with forward return columns
            (e.g., '1D', '5D', '10D'), 'factor', and 'factor_quantile'.
        """
        factor = self.prepare_factor(predictions_df, factor_col=factor_col)
        prices = self.prepare_prices(prices_data, price_col=price_col)

        common_assets = factor.index.get_level_values("asset").unique().intersection(prices.columns)
        n_assets = len(common_assets)
        if n_assets < 2:
            raise ValueError("At least 2 overlapping assets between factor and prices are required.")

        resolved_periods = tuple(int(p) for p in periods) if periods is not None else self.periods
        resolved_quantiles = self.resolve_quantiles(n_assets, quantiles=quantiles)
        resolved_z = self.filter_zscore if filter_zscore is None else filter_zscore
        resolved_max_loss = self.max_loss if max_loss is None else float(max_loss)
        resolved_zero_aware = self.zero_aware if zero_aware is None else bool(zero_aware)

        return al.utils.get_clean_factor_and_forward_returns(
            factor=factor,
            prices=prices,
            periods=resolved_periods,
            quantiles=resolved_quantiles,
            filter_zscore=resolved_z,
            max_loss=resolved_max_loss,
            zero_aware=resolved_zero_aware,
        )

    @staticmethod
    def compute_factor_metrics(factor_data: pd.DataFrame) -> dict[str, Any]:
        """Extract summary IC, quantile mean returns, and long-short spread via Alphalens."""
        ic = al.performance.factor_information_coefficient(factor_data)
        mean_ret_by_q, std_err_by_q = al.performance.mean_return_by_quantile(
            factor_data, by_date=False, demeaned=True
        )
        mean_ret_daily, std_err_daily = al.performance.mean_return_by_quantile(
            factor_data, by_date=True, demeaned=True
        )
        spread, spread_err = al.performance.compute_mean_returns_spread(
            mean_returns=mean_ret_daily,
            upper_quant=int(factor_data["factor_quantile"].max()),
            lower_quant=int(factor_data["factor_quantile"].min()),
            std_err=std_err_daily,
        )
        return {
            "daily_ic": ic,
            "mean_ic": ic.mean(),
            "mean_return_by_quantile": mean_ret_by_q,
            "std_err_by_quantile": std_err_by_q,
            "top_minus_bottom_spread": spread,
            "top_minus_bottom_spread_err": spread_err,
        }
