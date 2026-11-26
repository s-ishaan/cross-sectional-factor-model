"""
Portfolio Performance Attribution & Risk Telemetry Engine.

Computes cumulative gross/net wealth trajectories, annualized risk-adjusted ratios
(Sharpe, Sortino, Calmar), peak-to-trough drawdown analytics, and multi-component
transaction cost drag decomposition.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR: int = 252


def compute_drawdowns(returns: pd.Series) -> dict[str, Any]:
    """Compute drawdown series, maximum drawdown, peak/trough/recovery dates, and duration.

    Wealth is anchored at initial equity 1.0 before the first observation so an immediate
    loss on the first period is accurately captured as a drawdown from inception.

    Args:
        returns: Chronologically ordered periodic return series.

    Returns:
        Dictionary containing:
            - 'drawdown_series': pd.Series of signed drawdowns in [-1, 0].
            - 'max_drawdown': float positive MDD magnitude in [0, 1].
            - 'peak_date': Timestamp of peak preceding max drawdown (or None).
            - 'trough_date': Timestamp of max drawdown trough (or None).
            - 'recovery_date': Timestamp when wealth recovered to peak (or None if unrecovered).
            - 'max_drawdown_duration': int longest underwater stretch in trading periods.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    if clean.empty:
        return {
            "drawdown_series": pd.Series(dtype="float64", name="drawdown"),
            "max_drawdown": float("nan"),
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "max_drawdown_duration": 0,
        }

    wealth = (1.0 + clean).cumprod()
    running_peak = np.maximum(1.0, wealth.cummax())
    drawdown = (wealth / running_peak) - 1.0
    drawdown.name = "drawdown"

    # Compute longest contiguous underwater streak
    max_duration = 0
    current_duration = 0
    for is_underwater in (drawdown < -1e-15).to_numpy():
        if is_underwater:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0

    min_dd = float(drawdown.min())
    max_dd = float(-min_dd)

    if max_dd <= 1e-15:
        return {
            "drawdown_series": drawdown,
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "max_drawdown_duration": 0,
        }

    trough_date = drawdown.idxmin()
    peak_val = float(running_peak.loc[trough_date])
    pre_trough_wealth = wealth.loc[:trough_date]
    peak_candidates = pre_trough_wealth[np.isclose(pre_trough_wealth, peak_val, rtol=0.0, atol=1e-12)]
    peak_date = peak_candidates.index[-1] if not peak_candidates.empty else clean.index[0]

    post_trough_wealth = wealth.loc[trough_date:]
    recovered = post_trough_wealth[post_trough_wealth >= peak_val - 1e-12]
    # Exclude trough itself if trough_date happened to equal peak_val (impossible since max_dd > 0)
    recovery_date = recovered.index[0] if not recovered.empty else None

    return {
        "drawdown_series": drawdown,
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "max_drawdown_duration": int(max_duration),
    }


def compute_risk_ratios(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """Compute comprehensive annualized risk and return ratios from daily returns.

    Args:
        returns: Periodic daily return series.
        risk_free_rate: Annualized risk-free rate for Sharpe ratio (default: 0.0).
        mar: Annualized minimum acceptable return for Sortino downside deviation (default: 0.0).
        trading_days_per_year: Annualization factor (default: 252).

    Returns:
        Dictionary with cumulative return, annualized return, annualized volatility,
        Sharpe ratio, downside deviation, Sortino ratio, max drawdown, Calmar ratio,
        drawdown dates/duration, hit rate, and observation count.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    n_days = int(len(clean))
    if n_days == 0:
        return {
            "cumulative_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "downside_deviation": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "calmar_ratio": float("nan"),
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "max_drawdown_duration": 0,
            "hit_rate": float("nan"),
            "n_days": 0,
        }

    ann_factor = float(trading_days_per_year)
    mean_daily = float(clean.mean())
    ann_return = mean_daily * ann_factor
    cum_return = float((1.0 + clean).prod() - 1.0)

    if n_days > 1:
        ann_vol = float(clean.std(ddof=1)) * np.sqrt(ann_factor)
        sharpe = (
            (ann_return - risk_free_rate) / ann_vol
            if np.isfinite(ann_vol) and ann_vol > 1e-15
            else float("nan")
        )
    else:
        ann_vol = float("nan")
        sharpe = float("nan")

    daily_mar = mar / ann_factor
    shortfall = np.minimum(clean.to_numpy(dtype="float64") - daily_mar, 0.0)
    downside_dev = float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(ann_factor))
    sortino = (
        (ann_return - mar) / downside_dev
        if np.isfinite(downside_dev) and downside_dev > 1e-15
        else float("nan")
    )

    dd_info = compute_drawdowns(clean)
    mdd = dd_info["max_drawdown"]
    calmar = (
        ann_return / mdd
        if np.isfinite(mdd) and mdd > 1e-15
        else float("nan")
    )

    return {
        "cumulative_return": cum_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "downside_deviation": downside_dev,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "peak_date": dd_info["peak_date"],
        "trough_date": dd_info["trough_date"],
        "recovery_date": dd_info["recovery_date"],
        "max_drawdown_duration": dd_info["max_drawdown_duration"],
        "hit_rate": float((clean > 0.0).mean()),
        "n_days": n_days,
    }


class PerformanceAttribution:
    """Computes gross/net performance attribution and 3-component cost decomposition."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        mar: float = 0.0,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize the performance attribution engine.

        Args:
            risk_free_rate: Annualized risk-free rate for Sharpe ratio (default: 0.0).
            mar: Annualized minimum acceptable return for Sortino ratio (default: 0.0).
            trading_days_per_year: Trading days per year (default: 252).
        """
        if trading_days_per_year < 1:
            raise ValueError("trading_days_per_year must be >= 1.")
        self.risk_free_rate = float(risk_free_rate)
        self.mar = float(mar)
        self.trading_days_per_year = int(trading_days_per_year)

    @staticmethod
    def _extract_daily_summary(
        backtest_output: Union[pd.DataFrame, dict[str, Any]]
    ) -> pd.DataFrame:
        if isinstance(backtest_output, dict):
            if "daily_summary" not in backtest_output:
                raise ValueError("Backtest output dict must contain 'daily_summary'.")
            df = backtest_output["daily_summary"].copy()
        elif isinstance(backtest_output, pd.DataFrame):
            df = backtest_output.copy()
        else:
            raise TypeError(
                f"Expected BacktestSimulator output dict or daily_summary DataFrame; got {type(backtest_output)}."
            )

        required = ("gross_return", "net_return")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"daily_summary is missing required columns: {missing}")

        return df.sort_index()

    def evaluate(
        self,
        backtest_output: Union[pd.DataFrame, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute full gross/net risk ratios, cumulative wealth curves, and cost drag breakdown.

        Args:
            backtest_output: Either the dict returned by BacktestSimulator.run() or its
                'daily_summary' DataFrame.

        Returns:
            Dictionary containing:
                - 'gross': risk & drawdown metrics dictionary for gross_return.
                - 'net': risk & drawdown metrics dictionary for net_return.
                - 'cumulative_gross_index': pd.Series I^{gross}_{p,t}.
                - 'cumulative_net_index': pd.Series I^{net}_{p,t}.
                - 'gross_drawdowns': drawdown telemetry dict for gross_return.
                - 'net_drawdowns': drawdown telemetry dict for net_return.
                - 'cost_decomposition': turnover & annualized bps drag breakdown.
                - 'summary_table': DataFrame comparing Gross vs. Net metrics side by side.
        """
        df = self._extract_daily_summary(backtest_output)

        gross_ret = pd.to_numeric(df["gross_return"], errors="coerce").dropna()
        net_ret = pd.to_numeric(df["net_return"], errors="coerce").dropna()

        # Align cost evaluation to active return periods where net_return is realized
        active_idx = gross_ret.index.intersection(net_ret.index)
        active_df = df.loc[active_idx]

        gross_metrics = compute_risk_ratios(
            gross_ret,
            risk_free_rate=self.risk_free_rate,
            mar=self.mar,
            trading_days_per_year=self.trading_days_per_year,
        )
        net_metrics = compute_risk_ratios(
            net_ret,
            risk_free_rate=self.risk_free_rate,
            mar=self.mar,
            trading_days_per_year=self.trading_days_per_year,
        )

        cum_gross = (1.0 + gross_ret).cumprod()
        cum_gross.name = "cum_gross_index"
        cum_net = (1.0 + net_ret).cumprod()
        cum_net.name = "cum_net_index"

        gross_dd = compute_drawdowns(gross_ret)
        net_dd = compute_drawdowns(net_ret)

        ann_factor = float(self.trading_days_per_year)
        if not active_df.empty:
            turnover_s = (
                pd.to_numeric(active_df["turnover"], errors="coerce").fillna(0.0)
                if "turnover" in active_df.columns
                else pd.Series(0.0, index=active_df.index)
            )
            fee_s = (
                pd.to_numeric(active_df["fee_cost"], errors="coerce").fillna(0.0)
                if "fee_cost" in active_df.columns
                else pd.Series(0.0, index=active_df.index)
            )
            impact_s = (
                pd.to_numeric(active_df["impact_cost"], errors="coerce").fillna(0.0)
                if "impact_cost" in active_df.columns
                else pd.Series(0.0, index=active_df.index)
            )
            borrow_s = (
                pd.to_numeric(active_df["borrow_cost"], errors="coerce").fillna(0.0)
                if "borrow_cost" in active_df.columns
                else pd.Series(0.0, index=active_df.index)
            )
            total_cost_s = (
                pd.to_numeric(active_df["total_cost"], errors="coerce").fillna(0.0)
                if "total_cost" in active_df.columns
                else (fee_s + impact_s + borrow_s)
            )

            mean_daily_turnover = float(turnover_s.mean())
            ann_turnover = mean_daily_turnover * ann_factor
            mean_fee = float(fee_s.mean())
            mean_impact = float(impact_s.mean())
            mean_borrow = float(borrow_s.mean())
            mean_total = float(total_cost_s.mean())
            total_cost_sum = float(total_cost_s.sum())
        else:
            mean_daily_turnover = float("nan")
            ann_turnover = float("nan")
            mean_fee = float("nan")
            mean_impact = float("nan")
            mean_borrow = float("nan")
            mean_total = float("nan")
            total_cost_sum = float("nan")

        bps_scale = ann_factor * 10_000.0
        cost_decomposition = {
            "mean_daily_turnover": mean_daily_turnover,
            "annualized_turnover": ann_turnover,
            "total_cost_drag": total_cost_sum,
            "mean_daily_fee_bps": mean_fee * 10_000.0,
            "mean_daily_impact_bps": mean_impact * 10_000.0,
            "mean_daily_borrow_bps": mean_borrow * 10_000.0,
            "mean_daily_total_cost_bps": mean_total * 10_000.0,
            "annualized_fee_drag_bps": mean_fee * bps_scale,
            "annualized_impact_drag_bps": mean_impact * bps_scale,
            "annualized_borrow_drag_bps": mean_borrow * bps_scale,
            "annualized_total_cost_drag_bps": mean_total * bps_scale,
            "gross_minus_net_annualized_return": (
                gross_metrics["annualized_return"] - net_metrics["annualized_return"]
                if np.isfinite(gross_metrics["annualized_return"])
                and np.isfinite(net_metrics["annualized_return"])
                else float("nan")
            ),
            "fee_share": (
                mean_fee / mean_total if np.isfinite(mean_total) and mean_total > 0.0 else 0.0
            ),
            "impact_share": (
                mean_impact / mean_total if np.isfinite(mean_total) and mean_total > 0.0 else 0.0
            ),
            "borrow_share": (
                mean_borrow / mean_total if np.isfinite(mean_total) and mean_total > 0.0 else 0.0
            ),
        }

        summary_table = pd.DataFrame(
            {
                "Gross": {
                    "Cumulative Return": gross_metrics["cumulative_return"],
                    "Annualized Return": gross_metrics["annualized_return"],
                    "Annualized Volatility": gross_metrics["annualized_volatility"],
                    "Sharpe Ratio": gross_metrics["sharpe_ratio"],
                    "Downside Deviation": gross_metrics["downside_deviation"],
                    "Sortino Ratio": gross_metrics["sortino_ratio"],
                    "Max Drawdown": gross_metrics["max_drawdown"],
                    "Calmar Ratio": gross_metrics["calmar_ratio"],
                    "Max Drawdown Duration": gross_metrics["max_drawdown_duration"],
                    "Hit Rate": gross_metrics["hit_rate"],
                },
                "Net": {
                    "Cumulative Return": net_metrics["cumulative_return"],
                    "Annualized Return": net_metrics["annualized_return"],
                    "Annualized Volatility": net_metrics["annualized_volatility"],
                    "Sharpe Ratio": net_metrics["sharpe_ratio"],
                    "Downside Deviation": net_metrics["downside_deviation"],
                    "Sortino Ratio": net_metrics["sortino_ratio"],
                    "Max Drawdown": net_metrics["max_drawdown"],
                    "Calmar Ratio": net_metrics["calmar_ratio"],
                    "Max Drawdown Duration": net_metrics["max_drawdown_duration"],
                    "Hit Rate": net_metrics["hit_rate"],
                },
            }
        )

        return {
            "gross": gross_metrics,
            "net": net_metrics,
            "cumulative_gross_index": cum_gross,
            "cumulative_net_index": cum_net,
            "gross_drawdowns": gross_dd,
            "net_drawdowns": net_dd,
            "cost_decomposition": cost_decomposition,
            "summary_table": summary_table,
        }
