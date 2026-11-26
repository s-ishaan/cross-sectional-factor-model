"""
Unit tests for src/portfolio (QuantilePortfolioAllocator, TransactionCostModel, BacktestSimulator).
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from src.portfolio import BacktestSimulator, QuantilePortfolioAllocator, TransactionCostModel
from src.portfolio.allocator import QuantilePortfolioAllocator as DirectAllocator
from src.portfolio.simulator import (
    BacktestSimulator as DirectSimulator,
    TransactionCostModel as DirectCostModel,
)


class TestQuantilePortfolioAllocator(unittest.TestCase):
    """Tests for dollar-neutral quantile portfolio construction."""

    @staticmethod
    def _build_predictions(
        n_dates: int = 5, n_assets: int = 20, seed: int = 123
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2022-01-03", periods=n_dates)
        permnos = [10000 + i for i in range(n_assets)]
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        preds = rng.normal(0.0, 1.0, size=len(idx))
        df = pd.DataFrame({"pred": preds}, index=idx)
        df["predicted_rank"] = df.groupby(level="date")["pred"].rank(pct=True)
        return df

    def test_package_export_identity(self) -> None:
        self.assertIs(QuantilePortfolioAllocator, DirectAllocator)
        self.assertIs(TransactionCostModel, DirectCostModel)
        self.assertIs(BacktestSimulator, DirectSimulator)

    def test_equal_weight_dollar_neutrality_and_gross_leverage(self) -> None:
        preds = self._build_predictions(n_dates=5, n_assets=20)
        allocator = QuantilePortfolioAllocator(
            num_quantiles=10, top_quantile=9, bottom_quantile=0, weighting_scheme="equal"
        )
        allocated = allocator.allocate(preds)

        self.assertEqual(list(allocated.index.names), ["date", "permno"])
        self.assertIn("quantile", allocated.columns)
        self.assertIn("weight", allocated.columns)

        for _, group in allocated.groupby(level="date"):
            w = group["weight"]
            self.assertAlmostEqual(float(w.sum()), 0.0, places=12)
            self.assertAlmostEqual(float(w[w > 0].sum()), 0.5, places=12)
            self.assertAlmostEqual(float(w[w < 0].sum()), -0.5, places=12)
            self.assertAlmostEqual(float(w.abs().sum()), 1.0, places=12)
            # 20 assets into 10 deciles => 2 longs (+0.25 each), 2 shorts (-0.25 each)
            self.assertEqual(int((w > 0).sum()), 2)
            self.assertEqual(int((w < 0).sum()), 2)
            self.assertTrue(np.allclose(w[w > 0], 0.25))
            self.assertTrue(np.allclose(w[w < 0], -0.25))

        # Verify flat DataFrame format preservation and allocate_matrix()
        flat_out = allocator.allocate(preds.reset_index())
        self.assertFalse(isinstance(flat_out.index, pd.MultiIndex))
        wide = allocator.allocate_matrix(preds)
        self.assertEqual(wide.shape, (5, 20))
        self.assertTrue(np.allclose(wide.sum(axis=1), 0.0))
        self.assertTrue(np.allclose(wide.abs().sum(axis=1), 1.0))

    def test_rank_weighted_monotonicity_and_neutrality(self) -> None:
        dates = pd.bdate_range("2022-02-01", periods=3)
        permnos = [1001 + i for i in range(12)]
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        # Strictly increasing signal across assets 1001..1012
        signals = np.tile(np.arange(1, 13, dtype=float), len(dates))
        preds = pd.DataFrame({"pred": signals}, index=idx)

        allocator = QuantilePortfolioAllocator(
            num_quantiles=3, weighting_scheme="rank_weighted"
        )
        allocated = allocator.allocate(preds)

        for d in dates:
            cs = allocated.loc[d]
            w = cs["weight"]
            self.assertAlmostEqual(float(w.sum()), 0.0, places=12)
            self.assertAlmostEqual(float(w[w > 0].sum()), 0.5, places=12)
            self.assertAlmostEqual(float(w[w < 0].sum()), -0.5, places=12)
            self.assertAlmostEqual(float(w.abs().sum()), 1.0, places=12)

            # Top tertile (assets 1009..1012): weights strictly increasing with signal
            long_w = w[w > 0].to_numpy()
            self.assertEqual(len(long_w), 4)
            self.assertTrue(np.all(np.diff(long_w) > 0.0))

            # Bottom tertile (assets 1001..1004): weakest signal gets most negative weight
            short_w = w[w < 0].to_numpy()
            self.assertEqual(len(short_w), 4)
            self.assertTrue(np.all(np.diff(short_w) > 0.0))  # -0.2 < -0.15 < -0.1 < -0.05

    def test_thin_cross_section_and_invalid_params(self) -> None:
        dates = pd.bdate_range("2022-03-01", periods=2)
        permnos = [1, 2, 3, 4]
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        preds = pd.DataFrame({"pred": np.arange(8, dtype=float)}, index=idx)

        allocator = QuantilePortfolioAllocator(num_quantiles=10)
        allocated = allocator.allocate(preds)
        self.assertTrue((allocated["weight"] == 0.0).all())
        self.assertTrue(allocated["quantile"].isna().all())

        with self.assertRaises(ValueError):
            QuantilePortfolioAllocator(num_quantiles=1)
        with self.assertRaises(ValueError):
            QuantilePortfolioAllocator(num_quantiles=5, top_quantile=2, bottom_quantile=2)
        with self.assertRaises(ValueError):
            QuantilePortfolioAllocator(weighting_scheme="invalid")


class TestTransactionCostAndSimulator(unittest.TestCase):
    """Tests for drift-adjusted turnover, 3-component costs, and liquidity limits."""

    def test_passive_weight_drift_and_exact_turnover(self) -> None:
        dates = pd.bdate_range("2022-04-01", periods=4)
        permnos = [101, 102]

        # Target weights on first 3 dates; 4th date provides t+1 returns for date 2
        weights = pd.DataFrame(
            [[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]],
            index=dates[:3],
            columns=permnos,
        )
        returns = pd.DataFrame(
            [
                [0.00, 0.00],   # d0
                [0.20, -0.10],  # d1: R_{p,0} = 0.5*0.20 + (-0.5)*(-0.10) = 0.15
                [-0.05, 0.05],  # d2: R_{p,1} = 0.5*(-0.05) + (-0.5)*(0.05) = -0.05
                [0.02, 0.01],   # d3: R_{p,2} = 0.5*0.02 + (-0.5)*0.01 = 0.005
            ],
            index=dates,
            columns=permnos,
        )

        # Zero out costs to isolate drift & turnover math
        cost_model = TransactionCostModel(fee_rate=0.0, impact_coeff=0.0, borrow_rate=0.0)
        sim = BacktestSimulator(cost_model=cost_model)
        out = sim.run(weights=weights, returns=returns)
        summary = out["daily_summary"]

        # Day 0: initial build from 0 -> [+0.5, -0.5], sum(|dw|) = 1.0 => turnover = 0.5
        self.assertAlmostEqual(summary.loc[dates[0], "turnover"], 0.5, places=12)
        self.assertAlmostEqual(summary.loc[dates[0], "gross_return"], 0.15, places=12)

        # Day 1 drifted weights: w_{1^-} = w_0 * (1 + R_1) / (1 + 0.15)
        expected_drift_0 = 0.5 * 1.20 / 1.15
        expected_drift_1 = -0.5 * 0.90 / 1.15
        self.assertAlmostEqual(
            out["drifted_weights"].loc[dates[1], 101], expected_drift_0, places=12
        )
        self.assertAlmostEqual(
            out["drifted_weights"].loc[dates[1], 102], expected_drift_1, places=12
        )

        expected_dw_0 = 0.5 - expected_drift_0
        expected_dw_1 = -0.5 - expected_drift_1
        expected_turnover_1 = 0.5 * (abs(expected_dw_0) + abs(expected_dw_1))
        self.assertAlmostEqual(
            summary.loc[dates[1], "turnover"], expected_turnover_1, places=12
        )
        self.assertAlmostEqual(summary.loc[dates[1], "gross_return"], -0.05, places=12)
        self.assertAlmostEqual(summary.loc[dates[2], "gross_return"], 0.005, places=12)

    def test_non_rebalance_days_drift_with_zero_trade_turnover(self) -> None:
        dates = pd.bdate_range("2022-05-02", periods=4)
        permnos = [101, 102]
        # Rebalance only on day 0 and day 2
        weights = pd.DataFrame(
            [[0.5, -0.5], [0.5, -0.5]],
            index=[dates[0], dates[2]],
            columns=permnos,
        )
        returns = pd.DataFrame(
            [[0.0, 0.0], [0.10, -0.05], [0.02, 0.03], [0.01, -0.01]],
            index=dates,
            columns=permnos,
        )
        cost_model = TransactionCostModel(fee_rate=5e-4, impact_coeff=0.0, borrow_rate=0.0050)
        sim = BacktestSimulator(cost_model=cost_model)
        out = sim.run(weights=weights, returns=returns)
        summary = out["daily_summary"]

        # Day 1 is a passive hold day => zero turnover and zero fee cost, positive borrow cost
        self.assertAlmostEqual(summary.loc[dates[1], "turnover"], 0.0, places=12)
        self.assertAlmostEqual(summary.loc[dates[1], "fee_cost"], 0.0, places=12)
        self.assertGreater(summary.loc[dates[1], "borrow_cost"], 0.0)
        # Executed weights on day 1 equal drifted weights
        self.assertTrue(
            np.allclose(
                out["executed_weights"].loc[dates[1]],
                out["drifted_weights"].loc[dates[1]],
            )
        )

    def test_three_component_cost_decomposition(self) -> None:
        dates = pd.bdate_range("2022-06-01", periods=2)
        permnos = [101, 102]

        weights = pd.DataFrame([[0.5, -0.5]], index=[dates[0]], columns=permnos)
        returns = pd.DataFrame([[0.0, 0.0], [0.04, -0.02]], index=dates, columns=permnos)
        daily_vol = pd.DataFrame([[0.02, 0.03]], index=[dates[0]], columns=permnos)
        adv = pd.DataFrame([[10_000_000_000.0, 10_000_000_000.0]], index=[dates[0]], columns=permnos)

        aum = 100_000_000.0
        c_fee = 5e-4
        eta = 0.10
        borrow_rate = 0.0050
        cost_model = TransactionCostModel(
            fee_rate=c_fee,
            impact_coeff=eta,
            aum=aum,
            borrow_rate=borrow_rate,
            max_participation=0.05,
        )
        sim = BacktestSimulator(cost_model=cost_model)
        out = sim.run(
            weights=weights,
            returns=returns,
            daily_volatility=daily_vol,
            adv=adv,
        )
        row = out["daily_summary"].loc[dates[0]]

        # Day 0 trades: dw = [+0.5, -0.5] => sum(|dw|) = 1.0
        expected_fee = c_fee * 1.0
        expected_impact = eta * (
            0.02 * np.sqrt(0.5 * aum / 1e10) + 0.03 * np.sqrt(0.5 * aum / 1e10)
        )
        expected_borrow = (borrow_rate / 252.0) * 0.5
        expected_total = expected_fee + expected_impact + expected_borrow
        expected_gross = 0.5 * 0.04 + (-0.5) * (-0.02)

        self.assertAlmostEqual(row["fee_cost"], expected_fee, places=12)
        self.assertAlmostEqual(row["impact_cost"], expected_impact, places=12)
        self.assertAlmostEqual(row["borrow_cost"], expected_borrow, places=12)
        self.assertAlmostEqual(row["total_cost"], expected_total, places=12)
        self.assertAlmostEqual(row["gross_return"], expected_gross, places=12)
        self.assertAlmostEqual(row["net_return"], expected_gross - expected_total, places=12)

        # Verify annualized volatility flag divides by sqrt(252) identically
        ann_vol = daily_vol * np.sqrt(252.0)
        out_ann = sim.run(
            weights=weights,
            returns=returns,
            daily_volatility=ann_vol,
            adv=adv,
            volatility_is_annualized=True,
        )
        self.assertAlmostEqual(
            out_ann["daily_summary"].loc[dates[0], "impact_cost"],
            expected_impact,
            places=12,
        )

    def test_liquidity_guard_warning_and_clamping(self) -> None:
        dates = pd.bdate_range("2022-07-01", periods=2)
        permnos = [101, 102]
        weights = pd.DataFrame([[0.5, -0.5]], index=[dates[0]], columns=permnos)
        returns = pd.DataFrame([[0.0, 0.0], [0.01, -0.01]], index=dates, columns=permnos)
        daily_vol = pd.DataFrame([[0.02, 0.02]], index=[dates[0]], columns=permnos)

        # AUM = 100M, trade notional = 50M per stock.
        # Stock 101 ADV = 1B (50M / 1B = 5% > 1.5% limit -> VIOLATION!)
        # Stock 102 ADV = 10B (50M / 10B = 0.5% <= 1.5% limit -> OK)
        adv = pd.DataFrame([[1_000_000_000.0, 10_000_000_000.0]], index=[dates[0]], columns=permnos)
        cost_model = TransactionCostModel(aum=100_000_000.0, max_participation=0.015)

        # 1. Warn mode: emits UserWarning, logs violation, keeps original weights
        sim_warn = BacktestSimulator(cost_model=cost_model, liquidity_mode="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out_warn = sim_warn.run(weights=weights, returns=returns, daily_volatility=daily_vol, adv=adv)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        self.assertEqual(int(out_warn["daily_summary"].loc[dates[0], "n_liquidity_violations"]), 1)
        self.assertEqual(len(out_warn["liquidity_violations"]), 1)
        self.assertEqual(int(out_warn["liquidity_violations"].iloc[0]["permno"]), 101)
        self.assertAlmostEqual(out_warn["executed_weights"].loc[dates[0], 101], 0.5, places=12)

        # 2. Cap mode: clamps trade on 101 to 0.015 * 1B / 100M = 0.15
        sim_cap = BacktestSimulator(cost_model=cost_model, liquidity_mode="cap")
        with warnings.catch_warnings(record=True) as caught_cap:
            warnings.simplefilter("always")
            out_cap = sim_cap.run(weights=weights, returns=returns, daily_volatility=daily_vol, adv=adv)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught_cap))
        self.assertAlmostEqual(out_cap["executed_weights"].loc[dates[0], 101], 0.15, places=12)
        self.assertAlmostEqual(out_cap["executed_weights"].loc[dates[0], 102], -0.50, places=12)
        # Verify post-cap participation constraint holds strictly
        capped_notional = out_cap["trades"].loc[dates[0]].abs() * cost_model.aum
        self.assertTrue((capped_notional <= cost_model.max_participation * adv.loc[dates[0]] + 1e-9).all())


if __name__ == "__main__":
    unittest.main()
