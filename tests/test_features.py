"""
Unit tests for CrossSectionalFeatureEngine using standard library unittest.
"""

import unittest

import numpy as np
import pandas as pd

from src.features import CrossSectionalFeatureEngine
from src.features.engine import CrossSectionalFeatureEngine as DirectEngine


class TestCrossSectionalFeatureEngine(unittest.TestCase):
    """Test suite for cumulative return indices, factor computation, and CS normalization."""

    @staticmethod
    def _build_panel(
        n_permnos: int = 10,
        n_days: int = 270,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2015-01-02", periods=n_days)
        permnos = [10000 + i for i in range(n_permnos)]

        records = []
        for p in permnos:
            rets = rng.normal(loc=0.0004, scale=0.02, size=n_days)
            prc = 50.0 * np.cumprod(1.0 + rets)
            vol = rng.integers(100_000, 1_000_000, size=n_days).astype(float)
            for d, r, px, v in zip(dates, rets, prc, vol):
                records.append(
                    {
                        "date": d,
                        "permno": p,
                        "ret_adj": float(r),
                        "prc": float(px),
                        "vol": float(v),
                    }
                )
        return pd.DataFrame(records)

    def test_package_export_identity(self) -> None:
        self.assertIs(CrossSectionalFeatureEngine, DirectEngine)

    def test_cumulative_return_compounding_and_nan_continuity(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=5)
        df = pd.DataFrame(
            [
                {"date": dates[3], "permno": 10001, "ret_adj": np.nan, "prc": 10.0, "vol": 100.0},
                {"date": dates[0], "permno": 10001, "ret_adj": 0.10, "prc": 10.0, "vol": 100.0},
                {"date": dates[2], "permno": 10001, "ret_adj": 0.05, "prc": 10.0, "vol": 100.0},
                {"date": dates[1], "permno": 10001, "ret_adj": -0.20, "prc": 10.0, "vol": 100.0},
                {"date": dates[4], "permno": 10001, "ret_adj": 0.02, "prc": 10.0, "vol": 100.0},
                {"date": dates[0], "permno": 10002, "ret_adj": 0.01, "prc": 20.0, "vol": 200.0},
                {"date": dates[1], "permno": 10002, "ret_adj": 0.02, "prc": 20.0, "vol": 200.0},
            ]
        )

        engine = CrossSectionalFeatureEngine()
        res = engine.compute_cumulative_returns(df)
        p1 = res[res["permno"] == 10001].sort_values("date")

        expected_p1 = [
            1.10,
            1.10 * 0.80,
            1.10 * 0.80 * 1.05,
            1.10 * 0.80 * 1.05,  # NaN treated as 0 return, index preserved
            1.10 * 0.80 * 1.05 * 1.02,
        ]
        np.testing.assert_allclose(p1["cum_ret_index"].to_numpy(), expected_p1, rtol=1e-10)

        # Also verify MultiIndex ['date', 'permno'] input preserves MultiIndex output
        mi_df = df.set_index(["date", "permno"])
        mi_res = engine.compute_cumulative_returns(mi_df)
        self.assertIsInstance(mi_res.index, pd.MultiIndex)
        self.assertEqual(list(mi_res.index.names), ["date", "permno"])
        self.assertAlmostEqual(mi_res.loc[(dates[1], 10002), "cum_ret_index"], 1.01 * 1.02, places=10)

    def test_momentum_lookbacks_and_skip_window_mechanics(self) -> None:
        # 260 trading days for permno 10001:
        # Days 0..238: +0.2% daily; Days 239..259 (final 21 days): -2.0% daily crash.
        # On day index 259 (t = 259):
        #   I_t       = I_259
        #   I_{t-5}   = I_254 -> 5D reversal reflects 5 days of -2.0%
        #   I_{t-21}  = I_238 -> 21D momentum reflects 21 days of -2.0%
        #   I_{t-63}  = I_196 -> 63D momentum spans +0.2% and -2.0%
        #   I_{t-252} = I_7   -> 12M-1M uses I_238 / I_7 - 1 (purely +0.2% regime, skipping crash!)
        n_days = 260
        dates = pd.bdate_range("2018-01-02", periods=n_days)
        rets = np.where(np.arange(n_days) < 239, 0.002, -0.02)
        df = pd.DataFrame(
            {
                "date": dates,
                "permno": 10001,
                "ret_adj": rets,
                "prc": 50.0,
                "vol": 100_000.0,
            }
        ).set_index(["date", "permno"])

        engine = CrossSectionalFeatureEngine()
        res = engine.compute_raw_factors(df)

        t_idx = 259
        t_date = dates[t_idx]
        cum_idx = res.xs(10001, level="permno")["cum_ret_index"].to_numpy()

        expected_rev_5d = (1.0 - 0.02) ** 5 - 1.0
        expected_mom_21d = (1.0 - 0.02) ** 21 - 1.0
        expected_mom_63d = cum_idx[259] / cum_idx[196] - 1.0
        expected_mom_12m_1m = (1.0 + 0.002) ** (238 - 7) - 1.0

        row = res.loc[(t_date, 10001)]
        self.assertAlmostEqual(row["rev_5d"], expected_rev_5d, places=10)
        self.assertAlmostEqual(row["mom_21d"], expected_mom_21d, places=10)
        self.assertAlmostEqual(row["mom_63d"], expected_mom_63d, places=10)
        self.assertAlmostEqual(row["mom_12m_1m"], expected_mom_12m_1m, places=10)
        self.assertGreater(row["mom_12m_1m"], 0.0)
        self.assertLess(row["mom_21d"], 0.0)

        # Prior to 253rd observation (index 251), 12M-1M must be NaN
        self.assertTrue(np.isnan(res.loc[(dates[251], 10001), "mom_12m_1m"]))

    def test_realized_volatility_and_adv(self) -> None:
        n_days = 30
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0, 0.015, size=n_days)
        # Negative prices represent CRSP bid-ask midpoints; |prc| must be used for DVol
        prcs = -np.linspace(20.0, 35.0, n_days)
        vols = np.linspace(1000.0, 3000.0, n_days)

        df = pd.DataFrame(
            {"date": dates, "permno": 20001, "ret_adj": rets, "prc": prcs, "vol": vols}
        )
        engine = CrossSectionalFeatureEngine()
        res = engine.compute_raw_factors(df)

        # First 20 days must be NaN; index 20 (21st observation) onward populated
        self.assertTrue(res["vol_21d"].iloc[:20].isna().all())
        self.assertTrue(res["adv_21d"].iloc[:20].isna().all())

        expected_vol_last = np.std(rets[-21:], ddof=1) * np.sqrt(252.0)
        expected_adv_last = np.median(np.abs(prcs[-21:]) * vols[-21:])
        self.assertAlmostEqual(res["vol_21d"].iloc[-1], expected_vol_last, places=10)
        self.assertAlmostEqual(res["adv_21d"].iloc[-1], expected_adv_last, places=10)

    def test_winsorize_cross_section_caps_extreme_outliers(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=2)
        n_stocks = 101
        records = []
        for d_idx, d in enumerate(dates):
            base_vals = np.linspace(0.0, 1.0, n_stocks)
            # Inject extreme high and low outliers on day 0; keep day 1 mild
            if d_idx == 0:
                base_vals[0] = -1000.0
                base_vals[-1] = 1000.0
            for i, v in enumerate(base_vals):
                records.append({"date": d, "permno": 1000 + i, "factor_x": float(v)})

        df = pd.DataFrame(records).set_index(["date", "permno"])
        engine = CrossSectionalFeatureEngine()
        win = engine.winsorize_cross_section(df, factor_cols=["factor_x"], lower_q=0.01, upper_q=0.99)

        d0_vals = win.loc[dates[0], "factor_x"]
        d1_vals = win.loc[dates[1], "factor_x"]

        # For 101 sorted points [0..100], 1% quantile = index 1 (0.01), 99% quantile = index 99 (0.99)
        self.assertAlmostEqual(d0_vals.min(), 0.01, places=8)
        self.assertAlmostEqual(d0_vals.max(), 0.99, places=8)
        # Interior points remain unclipped and strictly rank-monotonic
        self.assertAlmostEqual(d0_vals.loc[1050], 0.50, places=8)
        # Day 1 winsorization is independent of Day 0
        self.assertAlmostEqual(d1_vals.min(), 0.01, places=8)
        self.assertAlmostEqual(d1_vals.max(), 0.99, places=8)

    def test_standardize_cross_section_zero_mean_unit_variance(self) -> None:
        df = self._build_panel(n_permnos=30, n_days=270, seed=123)
        engine = CrossSectionalFeatureEngine()
        orig_copy = df.copy()

        panel = engine.generate_feature_panel(df)
        # Input DataFrame must not be mutated
        pd.testing.assert_frame_equal(df, orig_copy)

        z_cols = [f"{c}_z" for c in engine.RAW_FACTOR_COLS]
        for z_col in z_cols:
            self.assertIn(z_col, panel.columns)
            # Every date cross-section with valid Z-scores must have mean ~ 0 and std ~ 1
            by_date = panel.groupby("date")[z_col]
            means = by_date.mean().dropna()
            stds = by_date.std(ddof=1).dropna()
            self.assertFalse(means.empty)
            self.assertTrue((means.abs() < 1e-6).all(), f"{z_col} failed zero-mean check")
            self.assertTrue(((stds - 1.0).abs() < 1e-4).all(), f"{z_col} failed unit-std check")

        # Raw factors in generate_feature_panel remain unclipped/raw (not overwritten by Z-scores)
        raw_only = engine.compute_raw_factors(df)
        for col in engine.RAW_FACTOR_COLS:
            pd.testing.assert_series_equal(panel[col], raw_only[col])

    def test_edge_cases_short_history_constant_series_and_zero_variance(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=25)
        records = []
        # 3 stocks with constant identical returns -> zero cross-sectional variance on all days
        for p in [10001, 10002, 10003]:
            for d in dates:
                records.append(
                    {"date": d, "permno": p, "ret_adj": 0.01, "prc": 25.0, "vol": 5000.0}
                )
        # 1 unseasoned stock trading only the last 8 days with distinct return
        for d in dates[-8:]:
            records.append(
                {"date": d, "permno": 10004, "ret_adj": 0.05, "prc": 10.0, "vol": 1000.0}
            )

        df = pd.DataFrame(records)
        engine = CrossSectionalFeatureEngine()
        panel = engine.generate_feature_panel(df)

        # Unseasoned stock 10004 (8 days) has NaN for 21D/63D/12M-1M factors, valid rev_5d on last 3 days
        p4 = panel[panel["permno"] == 10004].sort_values("date")
        self.assertTrue(p4["mom_21d"].isna().all())
        self.assertTrue(p4["vol_21d"].isna().all())
        self.assertTrue(p4["adv_21d"].isna().all())
        self.assertTrue(p4["rev_5d"].iloc[:5].isna().all())
        self.assertTrue(p4["rev_5d"].iloc[5:].notna().all())

        # On dates[22], 10001..10003 have identical constant returns (zero variance),
        # while 10004 is still unseasoned (NaN) -> Z-scores must be NaN (never inf or ZeroDivisionError)
        early = panel[panel["date"] == dates[22]]
        self.assertTrue(early["mom_21d_z"].isna().all())
        self.assertTrue(early["vol_21d_z"].isna().all())
        early_seasoned = early[early["permno"].isin([10001, 10002, 10003])]
        self.assertTrue((early_seasoned["vol_21d"] == 0.0).all())
        self.assertFalse(np.isinf(panel["mom_21d_z"].fillna(0.0)).any())

        # Single-constituent cross-section yields NaN Z-score
        single = pd.DataFrame(
            [{"date": dates[0], "permno": 99999, "factor_a": 1.23}]
        )
        std_single = engine.standardize_cross_section(single, factor_cols=["factor_a"])
        self.assertTrue(np.isnan(std_single["factor_a_z"].iloc[0]))

    def test_zero_cumulative_index_after_wipeout_and_validation_errors(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=7)
        df = pd.DataFrame(
            [
                {"date": dates[0], "permno": 10001, "ret_adj": -1.0, "prc": 1.0, "vol": 100.0},
                *[{"date": d, "permno": 10001, "ret_adj": 0.0, "prc": 1.0, "vol": 100.0} for d in dates[1:]],
            ]
        )
        engine = CrossSectionalFeatureEngine()
        res = engine.compute_raw_factors(df)
        # Division by zero index produces NaN rather than inf
        self.assertTrue(np.isnan(res["rev_5d"].iloc[-1]))

        with self.assertRaises(ValueError):
            CrossSectionalFeatureEngine(lower_q=0.95, upper_q=0.05)

        with self.assertRaises(ValueError):
            engine.compute_raw_factors(pd.DataFrame({"date": dates, "permno": 1}))


if __name__ == "__main__":
    unittest.main()
