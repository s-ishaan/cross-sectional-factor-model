"""
Execution Simulator & Three-Component Transaction Cost Engine.

Models passive weight drift between rebalancing dates, drift-adjusted rebalancing
turnover, linear commission/clearing fees, Almgren-Chriss square-root market impact,
short borrow financing drag, and institutional ADV participation constraints.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class TransactionCostModel:
    """Three-component institutional transaction cost & liquidity constraint model."""

    DEFAULT_FEE_RATE: float = 5e-4          # 5 bps linear broker/clearing/SEC fees
    DEFAULT_IMPACT_COEFF: float = 0.10      # Almgren-Chriss impact coefficient eta
    DEFAULT_AUM: float = 100_000_000.0      # $100M baseline portfolio AUM
    DEFAULT_BORROW_RATE: float = 0.0050     # 50 bps annualized general collateral borrow
    DEFAULT_MAX_PARTICIPATION: float = 0.015  # 1.5% ADV participation limit
    TRADING_DAYS_PER_YEAR: int = 252

    def __init__(
        self,
        fee_rate: float = DEFAULT_FEE_RATE,
        impact_coeff: float = DEFAULT_IMPACT_COEFF,
        aum: float = DEFAULT_AUM,
        borrow_rate: float = DEFAULT_BORROW_RATE,
        max_participation: float = DEFAULT_MAX_PARTICIPATION,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize the transaction cost model.

        Args:
            fee_rate: Linear transaction fee c_fee applied to traded notional sum(|dw|)
                (default: 5e-4, i.e., 5 bps).
            impact_coeff: Square-root market impact scalar eta (default: 0.10).
            aum: Portfolio assets under management in USD (default: $100M).
            borrow_rate: Annualized short borrow financing rate b_{i,t} (default: 0.0050,
                i.e., 50 bps/year).
            max_participation: Maximum single-day trade notional as a fraction of ADV
                rho_max (default: 0.015, i.e., 1.5% ADV).
            trading_days_per_year: Annualization factor for daily borrow accrual (default: 252).
        """
        if fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative.")
        if impact_coeff < 0.0:
            raise ValueError("impact_coeff must be non-negative.")
        if aum <= 0.0:
            raise ValueError("aum must be strictly positive.")
        if borrow_rate < 0.0:
            raise ValueError("borrow_rate must be non-negative.")
        if max_participation <= 0.0:
            raise ValueError("max_participation must be strictly positive.")
        if trading_days_per_year < 1:
            raise ValueError("trading_days_per_year must be >= 1.")

        self.fee_rate = float(fee_rate)
        self.impact_coeff = float(impact_coeff)
        self.aum = float(aum)
        self.borrow_rate = float(borrow_rate)
        self.max_participation = float(max_participation)
        self.trading_days_per_year = int(trading_days_per_year)

    def compute_linear_fee(self, delta_weights: np.ndarray | pd.Series) -> float:
        """Compute C_fee = c_fee * sum(|delta_w|)."""
        dw = np.asarray(delta_weights, dtype="float64")
        return float(self.fee_rate * np.nansum(np.abs(dw)))

    def compute_market_impact(
        self,
        delta_weights: np.ndarray | pd.Series,
        daily_volatility: Optional[np.ndarray | pd.Series],
        adv: Optional[np.ndarray | pd.Series],
    ) -> float:
        """Compute C_impact = eta * sum(sigma * sqrt(|delta_w| * AUM / ADV))."""
        if self.impact_coeff == 0.0 or daily_volatility is None or adv is None:
            return 0.0

        abs_dw = np.abs(np.asarray(delta_weights, dtype="float64"))
        sigma = np.asarray(daily_volatility, dtype="float64")
        adv_arr = np.asarray(adv, dtype="float64")

        valid = (abs_dw > 0.0) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(adv_arr) & (adv_arr > 0.0)
        if not np.any(valid):
            return 0.0

        participation = (abs_dw[valid] * self.aum) / adv_arr[valid]
        impact_terms = self.impact_coeff * sigma[valid] * np.sqrt(participation)
        return float(np.sum(impact_terms))

    def compute_borrow_cost(
        self,
        target_weights: np.ndarray | pd.Series,
        borrow_rates: Optional[np.ndarray | pd.Series] = None,
    ) -> float:
        """Compute C_borrow = sum_{w < 0} (b_{i,t} / 252) * |w_{i,t}|."""
        w = np.asarray(target_weights, dtype="float64")
        short_mask = np.isfinite(w) & (w < 0.0)
        if not np.any(short_mask):
            return 0.0

        if borrow_rates is None:
            rates = np.full_like(w, self.borrow_rate, dtype="float64")
        else:
            rates = np.asarray(borrow_rates, dtype="float64")
            rates = np.where(np.isfinite(rates) & (rates >= 0.0), rates, self.borrow_rate)

        daily_rates = rates[short_mask] / float(self.trading_days_per_year)
        return float(np.sum(daily_rates * np.abs(w[short_mask])))

    def find_liquidity_violations(
        self,
        delta_weights: np.ndarray | pd.Series,
        adv: Optional[np.ndarray | pd.Series],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Identify trades exceeding |delta_w| * AUM <= rho_max * ADV.

        Returns:
            Tuple (violation_mask, trade_notional, max_allowed_notional).
        """
        abs_dw = np.abs(np.asarray(delta_weights, dtype="float64"))
        trade_notional = abs_dw * self.aum
        if adv is None:
            empty_mask = np.zeros(abs_dw.shape, dtype=bool)
            return empty_mask, trade_notional, np.full_like(trade_notional, np.inf)

        adv_arr = np.asarray(adv, dtype="float64")
        max_allowed = np.where(
            np.isfinite(adv_arr) & (adv_arr > 0.0),
            self.max_participation * adv_arr,
            0.0,
        )
        violation_mask = (trade_notional > 0.0) & (trade_notional > max_allowed + 1e-12)
        return violation_mask, trade_notional, max_allowed

    def cap_trades_to_liquidity(
        self,
        delta_weights: np.ndarray | pd.Series,
        adv: Optional[np.ndarray | pd.Series],
    ) -> np.ndarray:
        """Clamp |delta_w| so |delta_w| * AUM <= rho_max * ADV while preserving direction."""
        dw = np.asarray(delta_weights, dtype="float64")
        if adv is None:
            return dw.copy()

        adv_arr = np.asarray(adv, dtype="float64")
        max_abs_dw = np.where(
            np.isfinite(adv_arr) & (adv_arr > 0.0),
            (self.max_participation * adv_arr) / self.aum,
            0.0,
        )
        return np.clip(dw, -max_abs_dw, max_abs_dw)

    def compute_costs(
        self,
        target_weights: np.ndarray | pd.Series,
        delta_weights: np.ndarray | pd.Series,
        daily_volatility: Optional[np.ndarray | pd.Series] = None,
        adv: Optional[np.ndarray | pd.Series] = None,
        borrow_rates: Optional[np.ndarray | pd.Series] = None,
    ) -> dict[str, float]:
        """Compute full 3-component transaction cost breakdown for one rebalance step."""
        fee_cost = self.compute_linear_fee(delta_weights)
        impact_cost = self.compute_market_impact(delta_weights, daily_volatility, adv)
        borrow_cost = self.compute_borrow_cost(target_weights, borrow_rates)
        return {
            "fee_cost": fee_cost,
            "impact_cost": impact_cost,
            "borrow_cost": borrow_cost,
            "total_cost": fee_cost + impact_cost + borrow_cost,
        }


class BacktestSimulator:
    """Simulates daily drift-adjusted execution, turnover, liquidity guards, and net returns."""

    VALID_LIQUIDITY_MODES: tuple[str, ...] = ("warn", "flag", "cap", "clamp", "ignore")

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        liquidity_mode: str = "warn",
        vol_window: int = 21,
        trading_days_per_year: int = 252,
    ) -> None:
        """Initialize the backtest execution simulator.

        Args:
            cost_model: TransactionCostModel instance (defaults to standard institutional
                parameters: 5 bps fee, eta=0.10, $100M AUM, 50 bps borrow, 1.5% ADV limit).
            liquidity_mode: Behavior when |delta_w| * AUM > rho_max * ADV:
                - 'warn' / 'flag': emit UserWarning and record violations without altering trades.
                - 'cap' / 'clamp': clamp trade sizes to max participation and record violations.
                - 'ignore': record violation count silently without warning or clamping.
            vol_window: Lookback window for realized daily volatility fallback when
                daily_volatility is not supplied explicitly (default: 21).
            trading_days_per_year: Trading days per year (default: 252).
        """
        if liquidity_mode not in self.VALID_LIQUIDITY_MODES:
            raise ValueError(
                f"liquidity_mode must be one of {self.VALID_LIQUIDITY_MODES}; got '{liquidity_mode}'."
            )
        if vol_window < 2:
            raise ValueError("vol_window must be >= 2.")

        self.cost_model = cost_model if cost_model is not None else TransactionCostModel(
            trading_days_per_year=trading_days_per_year
        )
        self.liquidity_mode = liquidity_mode
        self.vol_window = int(vol_window)
        self.trading_days_per_year = int(trading_days_per_year)

    @staticmethod
    def _to_wide_matrix(
        data: Union[pd.DataFrame, pd.Series],
        value_col: Optional[str] = None,
        candidate_cols: tuple[str, ...] = (),
        fill_value: Optional[float] = None,
        name: str = "data",
    ) -> pd.DataFrame:
        """Convert either a wide (date x permno) DataFrame or MultiIndex panel to wide float64."""
        if isinstance(data, pd.Series):
            if not isinstance(data.index, pd.MultiIndex) or not {"date", "permno"}.issubset(
                set(data.index.names)
            ):
                raise ValueError(
                    f"Series input for {name} must have MultiIndex ['date', 'permno']."
                )
            df = data.to_frame(name="_val").reset_index()
            col = "_val"
        elif isinstance(data, pd.DataFrame):
            has_multi = isinstance(data.index, pd.MultiIndex) and {"date", "permno"}.issubset(
                set(data.index.names)
            )
            has_cols = {"date", "permno"}.issubset(set(data.columns))
            if has_multi or has_cols:
                df = data.reset_index() if has_multi else data.copy()
                if value_col is not None:
                    if value_col not in df.columns:
                        raise ValueError(f"Column '{value_col}' not found in {name}.")
                    col = value_col
                else:
                    found = next((c for c in candidate_cols if c in df.columns), None)
                    if found is None:
                        non_key_cols = [c for c in df.columns if c not in ("date", "permno")]
                        if len(non_key_cols) == 1:
                            found = non_key_cols[0]
                        else:
                            raise ValueError(
                                f"Cannot infer value column for {name}; specify column explicitly."
                            )
                    col = found
            else:
                wide = data.copy()
                wide.index = pd.to_datetime(wide.index).astype("datetime64[ns]")
                wide.index.name = "date"
                wide.columns = pd.Index(
                    [int(c) for c in wide.columns], name="permno"
                )
                wide = wide.apply(pd.to_numeric, errors="coerce").astype("float64")
                wide = wide.sort_index().sort_index(axis=1)
                if fill_value is not None:
                    wide = wide.fillna(fill_value)
                return wide
        else:
            raise TypeError(f"Unsupported type for {name}: {type(data)}")

        if df.empty:
            return pd.DataFrame(dtype="float64")

        df = df.dropna(subset=["date", "permno"]).copy()
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
        df = df.dropna(subset=["permno"]).copy()
        df["permno"] = df["permno"].astype("int64")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        if df.duplicated(subset=["date", "permno"]).any():
            df = df.drop_duplicates(subset=["date", "permno"], keep="last")

        wide = df.pivot(index="date", columns="permno", values=col).sort_index().sort_index(axis=1)
        wide.index.name = "date"
        wide.columns.name = "permno"
        if fill_value is not None:
            wide = wide.fillna(fill_value)
        return wide

    def _resolve_optional_matrix(
        self,
        data: Optional[Union[pd.DataFrame, pd.Series, float]],
        dates: pd.DatetimeIndex,
        permnos: pd.Index,
        value_col: Optional[str],
        candidate_cols: tuple[str, ...],
        name: str,
    ) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if np.isscalar(data):
            return pd.DataFrame(
                float(data), index=dates, columns=permnos, dtype="float64"
            )
        wide = self._to_wide_matrix(
            data, value_col=value_col, candidate_cols=candidate_cols, name=name
        )
        return wide.reindex(index=dates, columns=permnos)

    def run(
        self,
        weights: Union[pd.DataFrame, pd.Series],
        returns: Union[pd.DataFrame, pd.Series],
        daily_volatility: Optional[Union[pd.DataFrame, pd.Series]] = None,
        adv: Optional[Union[pd.DataFrame, pd.Series]] = None,
        borrow_rates: Optional[Union[pd.DataFrame, pd.Series, float]] = None,
        weight_col: Optional[str] = None,
        return_col: Optional[str] = None,
        volatility_col: Optional[str] = None,
        adv_col: Optional[str] = None,
        volatility_is_annualized: bool = False,
    ) -> dict[str, Any]:
        """Execute multi-period portfolio simulation with passive drift and friction costs.

        Args:
            weights: Target portfolio weights (wide date x permno DataFrame or MultiIndex
                ['date', 'permno'] DataFrame/Series).
            returns: Daily asset total returns R_{i,t} (wide date x permno DataFrame or
                MultiIndex panel with 'ret_adj', 'ret', or 'return').
            daily_volatility: Optional 21-day realized daily volatility sigma_{i,t}.
                When None and adv is provided, computed from rolling std(returns, window=vol_window).
            adv: Optional 21-day median daily dollar volume ADV_{i,t} in USD.
            borrow_rates: Optional per-asset annualized borrow rates b_{i,t} (scalar, wide,
                or MultiIndex). Defaults to cost_model.borrow_rate.
            weight_col: Optional column name if weights is a multi-column panel
                (auto-detects 'weight' when None).
            return_col: Optional column name for returns panel.
            volatility_col: Optional column name for volatility panel.
            adv_col: Optional column name for ADV panel.
            volatility_is_annualized: Set True if passing annualized volatility (such as
                CrossSectionalFeatureEngine's 'vol_21d'); divides by sqrt(252).

        Returns:
            Dictionary containing:
                - 'daily_summary': DataFrame indexed by date with gross/net returns, cost
                  components, turnover, exposures, and liquidity violation telemetry.
                - 'executed_weights': Wide DataFrame of post-rebalance weights w_{i,t}.
                - 'drifted_weights': Wide DataFrame of pre-rebalance drifted weights w_{i,t^-}.
                - 'trades': Wide DataFrame of trades delta_w_{i,t} = w_{i,t} - w_{i,t^-}.
                - 'liquidity_violations': DataFrame detailing all trades breaching rho_max * ADV.
        """
        weights_wide = self._to_wide_matrix(
            weights,
            value_col=weight_col,
            candidate_cols=("weight",),
            fill_value=0.0,
            name="weights",
        )
        returns_wide = self._to_wide_matrix(
            returns,
            value_col=return_col,
            candidate_cols=("ret_adj", "ret", "return"),
            fill_value=None,
            name="returns",
        )

        if weights_wide.empty or returns_wide.empty:
            empty_summary = pd.DataFrame(
                columns=[
                    "gross_return",
                    "net_return",
                    "fee_cost",
                    "impact_cost",
                    "borrow_cost",
                    "total_cost",
                    "turnover",
                    "traded_notional",
                    "gross_exposure",
                    "net_exposure",
                    "long_exposure",
                    "short_exposure",
                    "n_liquidity_violations",
                    "max_participation_ratio",
                ]
            )
            empty_summary.index = pd.DatetimeIndex([], name="date")
            return {
                "daily_summary": empty_summary,
                "executed_weights": pd.DataFrame(dtype="float64"),
                "drifted_weights": pd.DataFrame(dtype="float64"),
                "trades": pd.DataFrame(dtype="float64"),
                "liquidity_violations": pd.DataFrame(
                    columns=[
                        "date",
                        "permno",
                        "abs_delta_weight",
                        "trade_notional",
                        "adv",
                        "participation_ratio",
                        "max_allowed_notional",
                    ]
                ),
            }

        start_date = weights_wide.index.min()
        end_date = weights_wide.index.max()
        sim_dates = returns_wide.index[
            (returns_wide.index >= start_date) & (returns_wide.index <= end_date)
        ]
        if len(sim_dates) == 0:
            raise ValueError(
                "No overlapping dates between weights and returns within the weight date range."
            )

        all_permnos = weights_wide.columns.union(returns_wide.columns).sort_values()
        returns_aligned = returns_wide.reindex(columns=all_permnos)
        weights_aligned = weights_wide.reindex(index=sim_dates, columns=all_permnos)
        rebalance_mask = sim_dates.isin(weights_wide.index)

        adv_aligned = self._resolve_optional_matrix(
            adv,
            sim_dates,
            all_permnos,
            value_col=adv_col,
            candidate_cols=("adv_21d", "adv"),
            name="adv",
        )

        if daily_volatility is not None:
            vol_aligned = self._resolve_optional_matrix(
                daily_volatility,
                sim_dates,
                all_permnos,
                value_col=volatility_col,
                candidate_cols=("daily_vol", "vol_21d", "volatility"),
                name="daily_volatility",
            )
            if volatility_is_annualized and vol_aligned is not None:
                vol_aligned = vol_aligned / np.sqrt(float(self.trading_days_per_year))
        elif adv_aligned is not None:
            rolling_vol = returns_aligned.rolling(
                window=self.vol_window, min_periods=min(2, self.vol_window)
            ).std(ddof=1)
            vol_aligned = rolling_vol.reindex(index=sim_dates, columns=all_permnos)
        else:
            vol_aligned = None

        borrow_aligned = self._resolve_optional_matrix(
            borrow_rates,
            sim_dates,
            all_permnos,
            value_col=None,
            candidate_cols=("borrow_rate", "rate"),
            name="borrow_rates",
        )

        # Map each simulation date to its integer position inside returns_aligned
        ret_pos_lookup = {d: i for i, d in enumerate(returns_aligned.index)}
        ret_values = returns_aligned.to_numpy(dtype="float64")
        n_sim = len(sim_dates)
        n_assets = len(all_permnos)

        exec_arr = np.zeros((n_sim, n_assets), dtype="float64")
        drift_arr = np.zeros((n_sim, n_assets), dtype="float64")
        trade_arr = np.zeros((n_sim, n_assets), dtype="float64")

        summary_rows: list[dict[str, Any]] = []
        violation_records: list[dict[str, Any]] = []

        prev_w = np.zeros(n_assets, dtype="float64")
        prev_gross_ret = 0.0

        for step, date in enumerate(sim_dates):
            pos = ret_pos_lookup[date]

            if step == 0:
                drifted_w = np.zeros(n_assets, dtype="float64")
            else:
                r_curr = np.nan_to_num(ret_values[pos], nan=0.0)
                denom = 1.0 + prev_gross_ret
                if not np.isfinite(denom) or denom <= 1e-12:
                    denom = 1e-12
                drifted_w = prev_w * ((1.0 + r_curr) / denom)

            if rebalance_mask[step]:
                target_w = np.nan_to_num(
                    weights_aligned.iloc[step].to_numpy(dtype="float64"), nan=0.0
                )
            else:
                target_w = drifted_w.copy()

            delta_w = target_w - drifted_w

            adv_row = (
                adv_aligned.iloc[step].to_numpy(dtype="float64")
                if adv_aligned is not None
                else None
            )
            vol_row = (
                vol_aligned.iloc[step].to_numpy(dtype="float64")
                if vol_aligned is not None
                else None
            )
            borrow_row = (
                borrow_aligned.iloc[step].to_numpy(dtype="float64")
                if borrow_aligned is not None
                else None
            )

            viol_mask, trade_notional, max_allowed = self.cost_model.find_liquidity_violations(
                delta_w, adv_row
            )
            n_violations = int(np.sum(viol_mask))
            if adv_row is not None:
                pos_adv = np.isfinite(adv_row) & (adv_row > 0.0) & (trade_notional > 0.0)
                if np.any(pos_adv):
                    max_part = float(np.max(trade_notional[pos_adv] / adv_row[pos_adv]))
                elif n_violations > 0:
                    max_part = float("inf")
                else:
                    max_part = 0.0
            else:
                max_part = 0.0

            if n_violations > 0:
                viol_indices = np.flatnonzero(viol_mask)
                for idx in viol_indices:
                    adv_val = float(adv_row[idx]) if adv_row is not None else float("nan")
                    part_ratio = (
                        float(trade_notional[idx] / adv_val)
                        if np.isfinite(adv_val) and adv_val > 0.0
                        else float("inf")
                    )
                    violation_records.append(
                        {
                            "date": date,
                            "permno": int(all_permnos[idx]),
                            "abs_delta_weight": float(abs(delta_w[idx])),
                            "trade_notional": float(trade_notional[idx]),
                            "adv": adv_val,
                            "participation_ratio": part_ratio,
                            "max_allowed_notional": float(max_allowed[idx]),
                        }
                    )

                if self.liquidity_mode in ("cap", "clamp"):
                    delta_w = self.cost_model.cap_trades_to_liquidity(delta_w, adv_row)
                    executed_w = drifted_w + delta_w
                else:
                    executed_w = target_w
            else:
                executed_w = target_w

            costs = self.cost_model.compute_costs(
                target_weights=executed_w,
                delta_weights=delta_w,
                daily_volatility=vol_row,
                adv=adv_row,
                borrow_rates=borrow_row,
            )

            traded_notional_frac = float(np.sum(np.abs(delta_w)))
            turnover = 0.5 * traded_notional_frac

            if pos + 1 < len(returns_aligned.index):
                r_next = np.nan_to_num(ret_values[pos + 1], nan=0.0)
                gross_ret = float(np.dot(executed_w, r_next))
                net_ret = gross_ret - costs["total_cost"]
            else:
                gross_ret = float("nan")
                net_ret = float("nan")

            drift_arr[step] = drifted_w
            exec_arr[step] = executed_w
            trade_arr[step] = delta_w

            long_exp = float(np.sum(executed_w[executed_w > 0.0]))
            short_exp = float(np.sum(executed_w[executed_w < 0.0]))

            summary_rows.append(
                {
                    "date": date,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "fee_cost": costs["fee_cost"],
                    "impact_cost": costs["impact_cost"],
                    "borrow_cost": costs["borrow_cost"],
                    "total_cost": costs["total_cost"],
                    "turnover": turnover,
                    "traded_notional": traded_notional_frac,
                    "gross_exposure": long_exp - short_exp,
                    "net_exposure": long_exp + short_exp,
                    "long_exposure": long_exp,
                    "short_exposure": short_exp,
                    "n_liquidity_violations": n_violations,
                    "max_participation_ratio": max_part,
                }
            )

            prev_w = executed_w
            prev_gross_ret = gross_ret if np.isfinite(gross_ret) else 0.0

        if violation_records and self.liquidity_mode in ("warn", "flag", "cap", "clamp"):
            action_msg = (
                "Capped to ADV limit."
                if self.liquidity_mode in ("cap", "clamp")
                else "Flagged without clamping."
            )
            warnings.warn(
                f"{len(violation_records)} trade(s) exceeded max ADV participation "
                f"({self.cost_model.max_participation:.2%}). {action_msg}",
                UserWarning,
                stacklevel=2,
            )

        daily_summary = pd.DataFrame(summary_rows).set_index("date")
        daily_summary.index = pd.DatetimeIndex(daily_summary.index, name="date")

        executed_df = pd.DataFrame(exec_arr, index=sim_dates, columns=all_permnos)
        drifted_df = pd.DataFrame(drift_arr, index=sim_dates, columns=all_permnos)
        trades_df = pd.DataFrame(trade_arr, index=sim_dates, columns=all_permnos)

        violations_df = (
            pd.DataFrame(violation_records)
            if violation_records
            else pd.DataFrame(
                columns=[
                    "date",
                    "permno",
                    "abs_delta_weight",
                    "trade_notional",
                    "adv",
                    "participation_ratio",
                    "max_allowed_notional",
                ]
            )
        )

        return {
            "daily_summary": daily_summary,
            "executed_weights": executed_df,
            "drifted_weights": drifted_df,
            "trades": trades_df,
            "liquidity_violations": violations_df,
        }
