"""
Unit tests for CRSPDelistingAdjuster using standard library unittest.
"""

import unittest

import numpy as np
import pandas as pd

from src.data import CRSPDelistingAdjuster
from src.data.delisting import CRSPDelistingAdjuster as DirectCRSPDelistingAdjuster


class TestCRSPDelistingAdjuster(unittest.TestCase):
    """Test suite for CRSP delisting return imputation, alignment, and compounding."""

    def setUp(self) -> None:
        """Create baseline synthetic CRSP stocknames and daily trading data."""
        self.trading_dates = pd.to_datetime(
            ["2015-05-13", "2015-05-14", "2015-05-15", "2015-05-18", "2015-05-19"]
        )
        self.df_stocknames = pd.DataFrame(
            [
                {
                    "permno": 10001,
                    "namedt": pd.Timestamp("2010-01-01"),
                    "nameenddt": pd.Timestamp("2020-12-31"),
                    "shrcd": 11,
                    "exchcd": 1,  # NYSE
                    "siccd": 7372,
                },
                {
                    "permno": 10002,
                    "namedt": pd.Timestamp("2010-01-01"),
                    "nameenddt": pd.Timestamp("2015-05-15"),
                    "shrcd": 11,
                    "exchcd": 3,  # NASDAQ
                    "siccd": 3571,
                },
                {
                    "permno": 10003,
                    "namedt": pd.Timestamp("2010-01-01"),
                    "nameenddt": pd.Timestamp("2015-05-15"),
                    "shrcd": 10,
                    "exchcd": 2,  # AMEX
                    "siccd": 2834,
                },
            ]
        )

    def test_package_export_identity(self) -> None:
        self.assertIs(CRSPDelistingAdjuster, DirectCRSPDelistingAdjuster)

    def test_compounding_when_both_ret_and_dlret_exist(self) -> None:
        df_dsf = pd.DataFrame(
            [
                {"permno": 10001, "date": pd.Timestamp("2015-05-14"), "ret": 0.02},
                {"permno": 10001, "date": pd.Timestamp("2015-05-15"), "ret": 0.05},
            ]
        )
        df_dsedelist = pd.DataFrame(
            [
                {
                    "permno": 10001,
                    "dlstdt": pd.Timestamp("2015-05-15"),
                    "dlstcd": 500,
                    "dlret": -0.20,
                }
            ]
        )

        adjuster = CRSPDelistingAdjuster()
        result = adjuster.apply_delisting_returns(df_dsf, df_dsedelist)

        # Normal day retains raw ret
        day1 = result.loc[result["date"] == pd.Timestamp("2015-05-14")].iloc[0]
        self.assertFalse(bool(day1["is_delisting_day"]))
        self.assertAlmostEqual(day1["ret_adj"], 0.02, places=8)

        # Delisting day compounds (1 + 0.05) * (1 - 0.20) - 1 = -0.16
        day2 = result.loc[result["date"] == pd.Timestamp("2015-05-15")].iloc[0]
        self.assertTrue(bool(day2["is_delisting_day"]))
        self.assertAlmostEqual(day2["ret_adj"], -0.16, places=8)

    def test_fallback_when_ret_is_nan_on_delisting_date(self) -> None:
        df_dsf = pd.DataFrame(
            [
                {"permno": 10001, "date": pd.Timestamp("2015-05-14"), "ret": -0.03},
                {"permno": 10001, "date": pd.Timestamp("2015-05-15"), "ret": np.nan},
            ]
        )
        df_dsedelist = pd.DataFrame(
            [
                {
                    "permno": 10001,
                    "dlstdt": pd.Timestamp("2015-05-15"),
                    "dlstcd": 574,
                    "dlret": -0.42,
                }
            ]
        )

        adjuster = CRSPDelistingAdjuster(df_dsf, df_dsedelist)
        result = adjuster.apply_delisting_returns()

        day2 = result.loc[result["date"] == pd.Timestamp("2015-05-15")].iloc[0]
        self.assertTrue(bool(day2["is_delisting_day"]))
        self.assertAlmostEqual(day2["ret_adj"], -0.42, places=8)

    def test_shumway_distress_imputation_flat_and_exchange_sensitive(self) -> None:
        df_dsedelist = pd.DataFrame(
            [
                # NYSE missing dlret -> -0.30 under both modes
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 500, "dlret": np.nan},
                # NASDAQ missing dlret -> -0.30 flat, -0.55 exchange-sensitive
                {"permno": 10002, "dlstdt": pd.Timestamp("2015-05-18"), "dlstcd": 580, "dlret": np.nan},
                # AMEX missing dlret -> -0.30 under both modes
                {"permno": 10003, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 591, "dlret": np.nan},
                # Distress with observed dlret must NOT be overwritten
                {"permno": 10004, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 500, "dlret": -0.12},
            ]
        )

        # 1. Flat institutional -30% imputation
        flat_adjuster = CRSPDelistingAdjuster(exchange_sensitive=False)
        cleaned_flat = flat_adjuster.clean_delisting_returns(df_dsedelist)
        flat_map = cleaned_flat.set_index("permno")["dlret"].to_dict()
        self.assertAlmostEqual(flat_map[10001], -0.30, places=8)
        self.assertAlmostEqual(flat_map[10002], -0.30, places=8)
        self.assertAlmostEqual(flat_map[10003], -0.30, places=8)
        self.assertAlmostEqual(flat_map[10004], -0.12, places=8)

        # 2. Exchange-sensitive imputation (-30% NYSE/AMEX, -55% NASDAQ)
        # Note: permno 10002 has dlstdt 2015-05-18 > nameenddt 2015-05-15, verifying
        # backward exchange lookup still resolves exchcd=3 cleanly.
        exch_adjuster = CRSPDelistingAdjuster(
            df_stocknames=self.df_stocknames, exchange_sensitive=True
        )
        cleaned_exch = exch_adjuster.clean_delisting_returns(df_dsedelist)
        exch_map = cleaned_exch.set_index("permno")["dlret"].to_dict()
        self.assertAlmostEqual(exch_map[10001], -0.30, places=8)
        self.assertAlmostEqual(exch_map[10002], -0.55, places=8)
        self.assertAlmostEqual(exch_map[10003], -0.30, places=8)
        self.assertAlmostEqual(exch_map[10004], -0.12, places=8)

    def test_non_distress_merger_delisting_imputation(self) -> None:
        df_dsf = pd.DataFrame(
            [
                {"permno": 10001, "date": pd.Timestamp("2015-05-14"), "ret": 0.01},
                {"permno": 10001, "date": pd.Timestamp("2015-05-15"), "ret": 0.03},
                {"permno": 10002, "date": pd.Timestamp("2015-05-15"), "ret": np.nan},
            ]
        )
        df_dsedelist = pd.DataFrame(
            [
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 200, "dlret": np.nan},
                {"permno": 10002, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 301, "dlret": np.nan},
            ]
        )

        adjuster = CRSPDelistingAdjuster(df_stocknames=self.df_stocknames, exchange_sensitive=True)
        result = adjuster.apply_delisting_returns(df_dsf, df_dsedelist)

        r1 = result.loc[(result["permno"] == 10001) & (result["date"] == pd.Timestamp("2015-05-15"))].iloc[0]
        self.assertTrue(bool(r1["is_delisting_day"]))
        self.assertAlmostEqual(r1["dlret"], 0.0, places=8)
        self.assertAlmostEqual(r1["ret_adj"], 0.03, places=8)

        r2 = result.loc[(result["permno"] == 10002) & (result["date"] == pd.Timestamp("2015-05-15"))].iloc[0]
        self.assertTrue(bool(r2["is_delisting_day"]))
        self.assertAlmostEqual(r2["dlret"], 0.0, places=8)
        self.assertAlmostEqual(r2["ret_adj"], 0.0, places=8)

    def test_off_market_and_post_trading_date_alignment(self) -> None:
        # 10001 trades until Friday 2015-05-15, delists on Saturday 2015-05-16 (off-market)
        # 10002 trades until Thursday 2015-05-14, delists weeks later on 2015-06-01
        # 10003 remains active through 2015-05-19 (ensures market has later dates than 10001/10002)
        df_dsf = pd.DataFrame(
            [
                {"permno": 10001, "date": pd.Timestamp("2015-05-14"), "ret": 0.01},
                {"permno": 10001, "date": pd.Timestamp("2015-05-15"), "ret": -0.10},
                {"permno": 10002, "date": pd.Timestamp("2015-05-13"), "ret": 0.02},
                {"permno": 10002, "date": pd.Timestamp("2015-05-14"), "ret": -0.05},
                {"permno": 10003, "date": pd.Timestamp("2015-05-15"), "ret": 0.005},
                {"permno": 10003, "date": pd.Timestamp("2015-05-18"), "ret": 0.005},
                {"permno": 10003, "date": pd.Timestamp("2015-05-19"), "ret": 0.005},
            ]
        )
        df_dsedelist = pd.DataFrame(
            [
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-16"), "dlstcd": 500, "dlret": np.nan},
                {"permno": 10002, "dlstdt": pd.Timestamp("2015-06-01"), "dlstcd": 550, "dlret": np.nan},
            ]
        )

        adjuster = CRSPDelistingAdjuster(df_stocknames=self.df_stocknames, exchange_sensitive=True)
        result = adjuster.apply_delisting_returns(df_dsf, df_dsedelist)

        # Row count preserved exactly
        self.assertEqual(len(result), len(df_dsf))

        # 10001 aligned onto Friday 2015-05-15 with NYSE -30%: (1 - 0.10)*(1 - 0.30) - 1 = -0.37
        r1 = result.loc[(result["permno"] == 10001) & (result["date"] == pd.Timestamp("2015-05-15"))].iloc[0]
        self.assertTrue(bool(r1["is_delisting_day"]))
        self.assertEqual(r1["dlstdt"], pd.Timestamp("2015-05-16"))
        self.assertAlmostEqual(r1["ret_adj"], -0.37, places=8)

        # 10002 aligned onto Thursday 2015-05-14 with NASDAQ -55%: (1 - 0.05)*(1 - 0.55) - 1 = -0.5725
        r2 = result.loc[(result["permno"] == 10002) & (result["date"] == pd.Timestamp("2015-05-14"))].iloc[0]
        self.assertTrue(bool(r2["is_delisting_day"]))
        self.assertEqual(r2["dlstdt"], pd.Timestamp("2015-06-01"))
        self.assertAlmostEqual(r2["ret_adj"], -0.5725, places=8)

        # 10003 unaffected
        self.assertFalse(bool(result.loc[result["permno"] == 10003, "is_delisting_day"].any()))

    def test_duplicate_and_multiple_delisting_records_edge_cases(self) -> None:
        df_dsf = pd.DataFrame(
            [
                {"permno": 10001, "date": pd.Timestamp("2015-05-14"), "ret": 0.01, "retx": 0.01},
                {"permno": 10001, "date": pd.Timestamp("2015-05-15"), "ret": -0.10, "retx": -0.10},
                {"permno": 10003, "date": pd.Timestamp("2015-05-15"), "ret": 0.02, "retx": 0.02},
            ]
        )
        df_dsedelist = pd.DataFrame(
            [
                # Duplicate same-date entries (one NaN, one observed)
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 500, "dlret": np.nan, "dlretx": np.nan},
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 500, "dlret": -0.25, "dlretx": -0.25},
                # Second later settlement record mapping to same last trading date 2015-05-15
                {"permno": 10001, "dlstdt": pd.Timestamp("2015-05-20"), "dlstcd": 574, "dlret": -0.50, "dlretx": -0.50},
                # Active CRSP placeholder (dlstcd=100) must NOT trigger delisting
                {"permno": 10003, "dlstdt": pd.Timestamp("2015-05-15"), "dlstcd": 100, "dlret": np.nan, "dlretx": np.nan},
            ]
        )

        dsf_orig = df_dsf.copy()
        delist_orig = df_dsedelist.copy()

        adjuster = CRSPDelistingAdjuster()
        result = adjuster.apply_delisting_returns(df_dsf, df_dsedelist)

        # Inputs not mutated
        pd.testing.assert_frame_equal(df_dsf, dsf_orig)
        pd.testing.assert_frame_equal(df_dsedelist, delist_orig)

        # No row explosion from multiple delisting records
        self.assertEqual(len(result), len(df_dsf))

        # Latest valid delisting record (-0.50) wins on 2015-05-15: (1-0.10)*(1-0.50)-1 = -0.55
        r1 = result.loc[(result["permno"] == 10001) & (result["date"] == pd.Timestamp("2015-05-15"))].iloc[0]
        self.assertTrue(bool(r1["is_delisting_day"]))
        self.assertAlmostEqual(r1["ret_adj"], -0.55, places=8)
        self.assertAlmostEqual(r1["retx_adj"], -0.55, places=8)

        # Active security with dlstcd=100 is not flagged as delisting
        r3 = result.loc[result["permno"] == 10003].iloc[0]
        self.assertFalse(bool(r3["is_delisting_day"]))
        self.assertAlmostEqual(r3["ret_adj"], 0.02, places=8)

    def test_empty_inputs_and_missing_required_columns(self) -> None:
        adjuster = CRSPDelistingAdjuster()

        empty_dsf = pd.DataFrame(columns=["permno", "date", "ret"])
        empty_delist = pd.DataFrame(columns=["permno", "dlstdt", "dlstcd", "dlret"])
        res = adjuster.apply_delisting_returns(empty_dsf, empty_delist)
        self.assertTrue(res.empty)
        self.assertIn("ret_adj", res.columns)
        self.assertIn("is_delisting_day", res.columns)

        with self.assertRaises(ValueError):
            adjuster.clean_delisting_returns(pd.DataFrame({"permno": [1]}))

        with self.assertRaises(ValueError):
            adjuster.apply_delisting_returns(pd.DataFrame({"permno": [1]}), empty_delist)


if __name__ == "__main__":
    unittest.main()
