"""
Unit tests for ForwardReturnTargetBuilder and PurgedWalkForwardCV using unittest.
"""

import unittest

import numpy as np
import pandas as pd

from src.validation import ForwardReturnTargetBuilder, PurgedWalkForwardCV
from src.validation.splitters import PurgedWalkForwardCV as DirectSplitter
from src.validation.target import ForwardReturnTargetBuilder as DirectTargetBuilder


class TestForwardReturnTargetBuilder(unittest.TestCase):
    """Test suite for non-anticipatory forward returns and quantile binning."""

    def test_package_export_identity(self) -> None:
        self.assertIs(ForwardReturnTargetBuilder, DirectTargetBuilder)
        self.assertIs(PurgedWalkForwardCV, DirectSplitter)

    def test_forward_return_execution_lag_and_holding_horizon(self) -> None:
        # 10 trading days for 2 stocks; verify exact (I_{t+6} - I_{t+1}) / I_{t+1}
        dates = pd.bdate_range("2020-01-02", periods=10)
        cum_1 = np.array([1.0, 1.1, 1.21, 1.331, 1.4, 1.5, 1.65, 1.8, 2.0, 2.2])
        cum_2 = np.linspace(1.0, 1.9, 10)

        df = pd.DataFrame(
            {
                "date": np.concatenate([dates, dates]),
                "permno": np.concatenate([np.full(10, 10001), np.full(10, 10002)]),
                "cum_ret": np.concatenate([cum_1, cum_2]),
            }
        ).set_index(["date", "permno"])

        builder = ForwardReturnTargetBuilder(holding_period=5, execution_lag=1)
        res = builder.compute_forward_returns(df)

        # At t = index 0 (dates[0]): buys at t+1 (index 1, 1.1), sells at t+6 (index 6, 1.65)
        expected_t0 = (1.65 - 1.1) / 1.1
        self.assertAlmostEqual(res.loc[(dates[0], 10001), "fwd_ret_5d"], expected_t0, places=10)

        # At t = index 3 (dates[3]): buys at index 4 (1.4), sells at index 9 (2.2)
        expected_t3 = (2.2 - 1.4) / 1.4
        self.assertAlmostEqual(res.loc[(dates[3], 10001), "fwd_ret_5d"], expected_t3, places=10)

        # At t = index 4..9 (last 6 rows): insufficient future horizon -> must be NaN (no cross-permno bleed)
        p1_tail = res.xs(10001, level="permno")["fwd_ret_5d"].iloc[4:]
        self.assertTrue(p1_tail.isna().all())

    def test_cum_ret_index_fallback_and_zero_buy_guard(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=8)
        df = pd.DataFrame(
            {
                "date": dates,
                "permno": 10001,
                "cum_ret_index": [1.0, 0.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            }
        )
        builder = ForwardReturnTargetBuilder()
        res = builder.compute_forward_returns(df)
        # At t=0, buy index (t=1) is 0.0 -> must produce NaN instead of inf
        self.assertTrue(np.isnan(res.loc[0, "fwd_ret_5d"]))
        # At t=1, buy index (t=2) is 1.2, sell index (t=7) is 1.7
        self.assertAlmostEqual(res.loc[1, "fwd_ret_5d"], (1.7 - 1.2) / 1.2, places=10)

    def test_deterministic_quantile_binning_and_thin_cross_sections(self) -> None:
        dates = pd.bdate_range("2021-01-04", periods=2)
        # Date 0: 30 stocks with ties -> K=10 quantiles must have 3 items per bin (0..9)
        # Date 1: only 5 stocks (< K=10) -> must evaluate to NaN
        records = []
        for i in range(30):
            # Inject intentional ties every 5 stocks
            val = float(i // 5) * 0.01
            records.append({"date": dates[0], "permno": 1000 + i, "fwd_ret_5d": val})
        for i in range(5):
            records.append({"date": dates[1], "permno": 1000 + i, "fwd_ret_5d": float(i) * 0.01})

        df = pd.DataFrame(records).set_index(["date", "permno"])
        builder = ForwardReturnTargetBuilder()
        res = builder.discretize_quantiles(df, target_col="fwd_ret_5d", num_quantiles=10)

        d0_bins = res.loc[dates[0], "target_quantile"]
        self.assertEqual(set(d0_bins.astype(int)), set(range(10)))
        counts = d0_bins.value_counts()
        self.assertTrue((counts == 3).all())

        # Deterministic tie-breaking: permno 1000..1004 all tied at 0.0, lower permno -> lower bin
        self.assertLessEqual(d0_bins.loc[1000], d0_bins.loc[1004])

        # Date 1 has fewer assets than K=10 -> all NaN
        d1_bins = res.loc[dates[1], "target_quantile"]
        self.assertTrue(d1_bins.isna().all())


class TestPurgedWalkForwardCV(unittest.TestCase):
    """Test suite for expanding purged & embargoed walk-forward cross-validation."""

    def test_zero_leakage_purge_cutoff_across_all_folds(self) -> None:
        dates = pd.bdate_range("2018-01-02", periods=120)
        permnos = [10001, 10002, 10003]
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        df = pd.DataFrame({"feat": np.arange(len(idx), dtype=float)}, index=idx)

        holding_period = 5
        splitter = PurgedWalkForwardCV(n_splits=5, holding_period=holding_period, embargo_days=0)
        self.assertEqual(splitter.get_n_splits(), 5)

        splits = list(splitter.split(df))
        self.assertEqual(len(splits), 5)

        prev_val_max = None
        for train_idx, val_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            # Disjoint row indices
            self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)

            train_dates = df.iloc[train_idx].index.get_level_values("date").unique()
            val_dates = df.iloc[val_idx].index.get_level_values("date").unique()

            # Zero-leakage check in both trading-date index space and calendar-day space
            self.assertLessEqual(
                train_dates.max(),
                val_dates.min() - pd.Timedelta(days=holding_period),
            )
            train_end_pos = dates.get_loc(train_dates.max())
            val_start_pos = dates.get_loc(val_dates.min())
            self.assertLessEqual(train_end_pos, val_start_pos - holding_period - 1)

            if prev_val_max is not None:
                self.assertGreater(val_dates.min(), prev_val_max)
            prev_val_max = val_dates.max()

    def test_embargo_min_train_dates_and_flat_column_support(self) -> None:
        # Daily calendar dates (non-business) verify exact trading-day purge + embargo count
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "date": np.repeat(dates, 2),
                "permno": np.tile([10001, 10002], len(dates)),
                "x": np.arange(len(dates) * 2, dtype=float),
            }
        )

        splitter = PurgedWalkForwardCV(
            n_splits=4, holding_period=5, embargo_days=2, min_train_dates=15
        )
        splits = list(splitter.split(df))
        self.assertEqual(len(splits), 4)

        for train_idx, val_idx in splits:
            train_dates = pd.to_datetime(df.iloc[train_idx]["date"]).unique()
            val_dates = pd.to_datetime(df.iloc[val_idx]["date"]).unique()
            self.assertGreaterEqual(len(train_dates), 15)
            # Purge (5) + Embargo (2) = 7 days gap strictly enforced
            self.assertLessEqual(
                train_dates.max(),
                val_dates.min() - pd.Timedelta(days=8),  # >= 7 skipped dates -> diff >= 8 days
            )


if __name__ == "__main__":
    unittest.main()
