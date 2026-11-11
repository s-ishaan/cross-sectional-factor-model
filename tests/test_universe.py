"""
Unit tests for PointInTimeUniverse engine using standard library unittest.
"""

import unittest
import pandas as pd
import numpy as np

from src.data.universe import PointInTimeUniverse


class TestPointInTimeUniverse(unittest.TestCase):
    """Test suite for annual reconstitution, eligibility filters, and PIT membership."""

    def setUp(self):
        """Construct synthetic CRSP daily and metadata dataframes."""
        # 2 years of trading days: 2014 and 2015
        dates = pd.date_range("2014-01-02", "2015-12-31", freq="B")
        
        # 6 stocks with distinct characteristics:
        # 10001: Standard eligible mega-cap
        # 10002: Standard eligible large-cap
        # 10003: Penny stock (|prc| < 5 on recon date)
        # 10004: REIT (siccd = 6798)
        # 10005: Unseasoned IPO (only 100 trading days in 2014)
        # 10006: Delists in mid-2015 (trades only until 2015-06-30)
        
        records = []
        for d in dates:
            # 10001
            records.append({"permno": 10001, "date": d, "prc": 100.0, "ret": 0.001, "shrout": 10000.0})
            # 10002
            records.append({"permno": 10002, "date": d, "prc": 50.0, "ret": 0.0005, "shrout": 5000.0})
            # 10003: Price drops to $3.00 on Dec 2014 recon date
            p3 = 3.0 if d.year == 2014 and d.month == 12 else 10.0
            records.append({"permno": 10003, "date": d, "prc": p3, "ret": -0.01, "shrout": 8000.0})
            # 10004: REIT
            records.append({"permno": 10004, "date": d, "prc": 80.0, "ret": 0.002, "shrout": 4000.0})
            # 10005: Starts late (from 2014-08-01 onwards)
            if d >= pd.Timestamp("2014-08-01"):
                records.append({"permno": 10005, "date": d, "prc": 40.0, "ret": 0.001, "shrout": 6000.0})
            # 10006: Delists in mid-2015
            if d <= pd.Timestamp("2015-06-30"):
                records.append({"permno": 10006, "date": d, "prc": 30.0, "ret": -0.002, "shrout": 3000.0})

        self.df_dsf = pd.DataFrame(records)

        # Metadata
        self.df_stocknames = pd.DataFrame([
            {"permno": 10001, "namedt": pd.Timestamp("2010-01-01"), "nameenddt": pd.Timestamp("2020-12-31"), "shrcd": 11, "exchcd": 1, "siccd": 7372},
            {"permno": 10002, "namedt": pd.Timestamp("2010-01-01"), "nameenddt": pd.Timestamp("2020-12-31"), "shrcd": 11, "exchcd": 3, "siccd": 2834},
            {"permno": 10003, "namedt": pd.Timestamp("2010-01-01"), "nameenddt": pd.Timestamp("2020-12-31"), "shrcd": 10, "exchcd": 1, "siccd": 3571},
            {"permno": 10004, "namedt": pd.Timestamp("2010-01-01"), "nameenddt": pd.Timestamp("2020-12-31"), "shrcd": 11, "exchcd": 1, "siccd": 6798}, # REIT
            {"permno": 10005, "namedt": pd.Timestamp("2014-08-01"), "nameenddt": pd.Timestamp("2020-12-31"), "shrcd": 11, "exchcd": 2, "siccd": 5000},
            {"permno": 10006, "namedt": pd.Timestamp("2010-01-01"), "nameenddt": pd.Timestamp("2015-06-30"), "shrcd": 11, "exchcd": 1, "siccd": 4000},
        ])

    def test_reconstitution_dates(self):
        engine = PointInTimeUniverse(self.df_dsf, self.df_stocknames, target_size=2)
        schedule = engine.get_reconstitution_dates()
        
        self.assertEqual(len(schedule), 1)
        recon_date, eff_start, eff_year = schedule[0]
        self.assertEqual(recon_date.year, 2014)
        self.assertEqual(recon_date.month, 12)
        self.assertEqual(eff_start.year, 2015)
        self.assertEqual(eff_start.month, 1)
        self.assertEqual(eff_year, 2015)

    def test_filters_applied_before_ranking(self):
        engine = PointInTimeUniverse(self.df_dsf, self.df_stocknames, min_history_days=200, target_size=5)
        schedule = engine.get_reconstitution_dates()
        recon_date, _, _ = schedule[0]

        eligible = engine.filter_eligible_securities(recon_date)
        eligible_permnos = set(eligible["permno"])

        # 10001 (standard), 10002 (standard), 10006 (standard in 2014) should be eligible
        self.assertIn(10001, eligible_permnos)
        self.assertIn(10002, eligible_permnos)
        self.assertIn(10006, eligible_permnos)

        # 10003 excluded (prc < 5 on recon date)
        self.assertNotIn(10003, eligible_permnos)

        # 10004 excluded (REIT: siccd = 6798)
        self.assertNotIn(10004, eligible_permnos)

        # 10005 excluded (insufficient history: only ~110 days)
        self.assertNotIn(10005, eligible_permnos)

    def test_market_cap_ranking_and_selection(self):
        engine = PointInTimeUniverse(self.df_dsf, self.df_stocknames, min_history_days=200, target_size=2)
        constituents = engine.build_annual_constituents()

        self.assertIn(2015, constituents)
        selected = constituents[2015]
        self.assertEqual(len(selected), 2)
        # 10001 (Cap = 1B) and 10002 (Cap = 250M) top 2
        self.assertEqual(selected, [10001, 10002])

    def test_daily_membership_and_delisting_drop(self):
        engine = PointInTimeUniverse(self.df_dsf, self.df_stocknames, min_history_days=200, target_size=3)
        daily_mem = engine.get_daily_membership(start_year=2015, end_year=2015)

        self.assertFalse(daily_mem.empty)
        
        # In early 2015 (e.g. 2015-01-05), all 3 are active
        jan_date = pd.Timestamp("2015-01-05")
        self.assertIn(jan_date, daily_mem.index.get_level_values("date"))
        jan_permnos = set(daily_mem.loc[jan_date].index)
        self.assertEqual(jan_permnos, {10001, 10002, 10006})

        # In late 2015 (e.g. 2015-08-03), 10006 delisted on 2015-06-30 so it drops
        aug_date = pd.Timestamp("2015-08-03")
        self.assertIn(aug_date, daily_mem.index.get_level_values("date"))
        aug_permnos = set(daily_mem.loc[aug_date].index)
        self.assertEqual(aug_permnos, {10001, 10002})
        self.assertNotIn(10006, aug_permnos)


if __name__ == "__main__":
    unittest.main()
