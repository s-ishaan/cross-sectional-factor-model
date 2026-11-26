"""
Unit tests for src/evaluation (PerformanceAttribution, compute_drawdowns, compute_risk_ratios, AlphalensAdapter)
and end-to-end Phase 1 -> Phase 2 pipeline verification.
"""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from src.evaluation import (
    AlphalensAdapter,
    PerformanceAttribution,
    compute_drawdowns,
    compute_risk_ratios,
)
from src.evaluation.alphalens_adapter import AlphalensAdapter as DirectAdapter
from src.evaluation.metrics import (
    PerformanceAttribution as DirectAttribution,
    compute_drawdowns as direct_drawdowns,
    compute_risk_ratios as direct_risk_ratios,
)
from src.features import CrossSectionalFeatureEngine
from src.models import WalkForwardModelEngine
from src.portfolio import BacktestSimulator, QuantilePortfolioAllocator, TransactionCostModel
from src.validation import ForwardReturnTargetBuilder, PurgedWalkForwardCV


class TestPerformanceMetricsAndAttribution(unittest.TestCase):
    """Numerical verification of drawdown, risk ratios, and cost drag reconciliation."""

    def test_package_export_identity(self) -> None:
        self.assertIs(PerformanceAttribution, DirectAttribution)
        self.assertIs(AlphalensAdapter, DirectAdapter)
        self.assertIs(compute_drawdowns, direct_drawdowns)
        self.assertIs(compute_risk_ratios, direct_risk_ratios)

    def test_compute_drawdowns_exact_telemetry(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=5)
        # Wealth path: [1.10, 0.88, 0.792, 1.188, 1.188]
        # Peak = 1.10 at d0, Trough = 0.792 at d2 => MDD = (1.10 - 0.792)/1.10 = 0.28
        # Underwater days: d1, d2 (duration = 2), recovered at d3 (1.188 >= 1.10)
        rets = pd.Series([0.10, -0.20, -0.10, 0.50, 0.00], index=dates)
        dd = compute_drawdowns(rets)

        self.assertAlmostEqual(dd["max_drawdown"], 0.28, places=12)
        self.assertEqual(dd["peak_date"], dates[0])
        self.assertEqual(dd["trough_date"], dates[2])
        self.assertEqual(dd["recovery_date"], dates[3])
        self.assertEqual(dd["max_drawdown_duration"], 2)

        # Monotonically positive returns => zero drawdown
        pos_rets = pd.Series([0.01, 0.02, 0.03], index=dates[:3])
        dd_pos = compute_drawdowns(pos_rets)
        self.assertEqual(dd_pos["max_drawdown"], 0.0)
        self.assertEqual(dd_pos["max_drawdown_duration"], 0)
        self.assertIsNone(dd_pos["trough_date"])

        # Immediate inception loss (unrecovered)
        neg_rets = pd.Series([-0.10, -0.10], index=dates[:2])
        dd_neg = compute_drawdowns(neg_rets)
        self.assertAlmostEqual(dd_neg["max_drawdown"], 0.19, places=12)
        self.assertIsNone(dd_neg["recovery_date"])
        self.assertEqual(dd_neg["max_drawdown_duration"], 2)

    def test_compute_risk_ratios_numerical_verification(self) -> None:
        dates = pd.bdate_range("2022-02-01", periods=4)
        rets = pd.Series([0.02, -0.01, 0.03, -0.02], index=dates)

        ratios = compute_risk_ratios(rets, risk_free_rate=0.0, mar=0.0, trading_days_per_year=252)

        mean_d = float(rets.mean())  # 0.005
        std_d = float(rets.std(ddof=1))
        expected_ann_ret = mean_d * 252.0
        expected_ann_vol = std_d * np.sqrt(252.0)
        expected_sharpe = expected_ann_ret / expected_ann_vol

        shortfalls = np.minimum(rets.to_numpy(), 0.0)
        expected_downside_dev = float(np.sqrt(np.mean(shortfalls**2)) * np.sqrt(252.0))
        expected_sortino = expected_ann_ret / expected_downside_dev
        expected_cum = float(np.prod(1.0 + rets.to_numpy()) - 1.0)
        expected_mdd = 0.02  # peak 1.02 * 0.99 * 1.03 = 1.040094, trough -2% -> 0.02
        expected_calmar = expected_ann_ret / expected_mdd

        self.assertAlmostEqual(ratios["cumulative_return"], expected_cum, places=12)
        self.assertAlmostEqual(ratios["annualized_return"], expected_ann_ret, places=12)
        self.assertAlmostEqual(ratios["annualized_volatility"], expected_ann_vol, places=12)
        self.assertAlmostEqual(ratios["sharpe_ratio"], expected_sharpe, places=12)
        self.assertAlmostEqual(ratios["downside_deviation"], expected_downside_dev, places=12)
        self.assertAlmostEqual(ratios["sortino_ratio"], expected_sortino, places=12)
        self.assertAlmostEqual(ratios["max_drawdown"], expected_mdd, places=12)
        self.assertAlmostEqual(ratios["calmar_ratio"], expected_calmar, places=12)
        self.assertAlmostEqual(ratios["hit_rate"], 0.5, places=12)
        self.assertEqual(ratios["n_days"], 4)

        # Zero-variance / single-observation safety
        flat_ratios = compute_risk_ratios(pd.Series([0.0, 0.0, 0.0], index=dates[:3]))
        self.assertTrue(np.isnan(flat_ratios["sharpe_ratio"]))
        self.assertTrue(np.isnan(flat_ratios["sortino_ratio"]))
        self.assertTrue(np.isnan(flat_ratios["calmar_ratio"]))

    def test_gross_vs_net_cost_drag_reconciliation(self) -> None:
        dates = pd.bdate_range("2022-03-01", periods=6)
        permnos = [101, 102, 103, 104]
        weights = pd.DataFrame(
            np.tile([0.5, 0.0, 0.0, -0.5], (5, 1)),
            index=dates[:5],
            columns=permnos,
        )
        returns = pd.DataFrame(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.02, 0.00, 0.00, -0.01],
                [-0.01, 0.00, 0.00, 0.005],
                [0.015, 0.00, 0.00, -0.01],
                [0.01, 0.00, 0.00, -0.005],
                [0.02, 0.00, 0.00, -0.01],
            ],
            index=dates,
            columns=permnos,
        )
        vol = pd.DataFrame(0.015, index=dates[:5], columns=permnos)
        adv = pd.DataFrame(5e9, index=dates[:5], columns=permnos)

        sim = BacktestSimulator(TransactionCostModel())
        sim_out = sim.run(weights=weights, returns=returns, daily_volatility=vol, adv=adv)

        attr = PerformanceAttribution()
        report = attr.evaluate(sim_out)

        self.assertIn("summary_table", report)
        self.assertEqual(list(report["summary_table"].columns), ["Gross", "Net"])

        cost = report["cost_decomposition"]
        self.assertAlmostEqual(
            cost["fee_share"] + cost["impact_share"] + cost["borrow_share"],
            1.0,
            places=12,
        )
        # Reconcile Gross - Net annualized return with annualized total cost drag
        self.assertAlmostEqual(
            cost["gross_minus_net_annualized_return"],
            cost["annualized_total_cost_drag_bps"] / 10_000.0,
            places=12,
        )
        self.assertGreater(
            report["cumulative_gross_index"].iloc[-1],
            report["cumulative_net_index"].iloc[-1],
        )


class TestAlphalensAdapterAndIntegration(unittest.TestCase):
    """Tests for Alphalens MultiIndex compatibility and end-to-end Phase 1->2 flow."""

    @staticmethod
    def _build_synthetic_market(n_days: int = 80, n_permnos: int = 20, seed: int = 11):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        permnos = [20000 + i for i in range(n_permnos)]

        # Create cross-sectional factor where signal predicts future returns
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        raw_sig = rng.normal(0.0, 1.0, size=len(idx))
        panel = pd.DataFrame(
            {
                "signal_raw": raw_sig,
                "prc": 50.0 + rng.uniform(-5.0, 20.0, size=len(idx)),
                "vol": rng.uniform(500_000.0, 2_000_000.0, size=len(idx)),
            },
            index=idx,
        )
        lagged_sig = panel["signal_raw"].groupby(level="permno").shift(2).fillna(0.0)
        panel["ret_adj"] = 0.005 * lagged_sig + rng.normal(0.0, 0.008, size=len(idx))
        return panel

    def test_alphalens_adapter_multi_horizon_compatibility(self) -> None:
        panel = self._build_synthetic_market(n_days=60, n_permnos=15, seed=5)
        feat_engine = CrossSectionalFeatureEngine()
        panel = feat_engine.compute_cumulative_returns(panel)

        preds = pd.DataFrame(
            {"predicted_rank": panel.groupby(level="date")["signal_raw"].rank(pct=True)},
            index=panel.index,
        )

        adapter = AlphalensAdapter(periods=(1, 5, 10), quantiles=3)
        with redirect_stdout(io.StringIO()):
            factor_data = adapter.build_factor_data(preds, panel, price_col="cum_ret_index")
            metrics = adapter.compute_factor_metrics(factor_data)

        self.assertEqual(list(factor_data.index.names), ["date", "asset"])
        for col in ("1D", "5D", "10D", "factor", "factor_quantile"):
            self.assertIn(col, factor_data.columns)
        self.assertEqual(set(factor_data["factor_quantile"].unique()), {1, 2, 3})
        self.assertIn("mean_ic", metrics)
        self.assertIn("top_minus_bottom_spread", metrics)

        # Also test wide price matrix input
        wide_prices = panel["cum_ret_index"].unstack(level="permno")
        with redirect_stdout(io.StringIO()):
            factor_data_wide = adapter.build_factor_data(preds, wide_prices)
        pd.testing.assert_frame_equal(factor_data, factor_data_wide)

    def test_end_to_end_phase1_to_phase2_pipeline(self) -> None:
        panel = self._build_synthetic_market(n_days=140, n_permnos=30, seed=42)

        # Phase 1: Features -> Targets -> Purged Walk-Forward LightGBM
        feat_engine = CrossSectionalFeatureEngine()
        feat_panel = feat_engine.generate_feature_panel(panel)
        target_builder = ForwardReturnTargetBuilder(holding_period=5, execution_lag=1)
        full_panel = target_builder.build_target_panel(feat_panel, num_quantiles=5)

        cv = PurgedWalkForwardCV(n_splits=3, holding_period=5)
        model_engine = WalkForwardModelEngine(cv_splitter=cv)
        oof_preds = model_engine.fit_predict(full_panel)

        # Phase 2: Allocation -> Backtest Simulation -> Attribution & Alphalens
        allocator = QuantilePortfolioAllocator(num_quantiles=5, weighting_scheme="equal")
        weights = allocator.allocate(oof_preds)

        sim = BacktestSimulator(
            cost_model=TransactionCostModel(aum=25_000_000.0),
            liquidity_mode="warn",
        )
        sim_out = sim.run(
            weights=weights,
            returns=full_panel,
            daily_volatility=full_panel,
            adv=full_panel,
            return_col="ret_adj",
            volatility_col="vol_21d",
            adv_col="adv_21d",
            volatility_is_annualized=True,
        )

        report = PerformanceAttribution().evaluate(sim_out)
        self.assertGreater(report["gross"]["n_days"], 10)
        self.assertTrue(np.isfinite(report["net"]["sharpe_ratio"]))
        self.assertTrue(np.isfinite(report["cost_decomposition"]["annualized_turnover"]))

        adapter = AlphalensAdapter(periods=(1, 5, 10), quantiles=5, max_loss=0.5)
        with redirect_stdout(io.StringIO()):
            al_data = adapter.build_factor_data(oof_preds, full_panel, price_col="cum_ret_index")
        self.assertFalse(al_data.empty)
        self.assertTrue({"1D", "5D", "10D"}.issubset(set(al_data.columns)))


if __name__ == "__main__":
    unittest.main()
