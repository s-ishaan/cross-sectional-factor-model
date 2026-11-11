"""
Point-in-Time CRSP Top-500 Common Equity Universe Engine.

Constructs an objective, rule-based, capitalization-ranked US equity universe
with annual December reconstitution and strict point-in-time eligibility filtering,
eliminating survivorship bias and non-operating equity contamination.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class PointInTimeUniverse:
    """Constructs and manages the Point-in-Time CRSP Top-500 common stock universe."""

    def __init__(
        self,
        df_dsf: pd.DataFrame,
        df_stocknames: pd.DataFrame,
        target_size: int = 500,
        min_price: float = 5.0,
        min_history_days: int = 252,
    ) -> None:
        """Initialize the universe engine with raw CRSP datasets.

        Args:
            df_dsf: CRSP Daily Stock File with columns ['permno', 'date', 'prc', 'ret', 'shrout'].
            df_stocknames: CRSP historical metadata with columns ['permno', 'namedt', 'nameenddt',
                'shrcd', 'exchcd', 'siccd'].
            target_size: Number of constituents to select at annual reconstitution (default: 500).
            min_price: Minimum absolute closing price hurdle at reconstitution (default: $5.00).
            min_history_days: Minimum required active trading days in preceding year (default: 252).
        """
        self.df_dsf = df_dsf.copy()
        self.df_stocknames = df_stocknames.copy()
        self.target_size = target_size
        self.min_price = min_price
        self.min_history_days = min_history_days

        # Ensure datetime dtypes
        self.df_dsf["date"] = pd.to_datetime(self.df_dsf["date"])
        self.df_stocknames["namedt"] = pd.to_datetime(self.df_stocknames["namedt"])
        self.df_stocknames["nameenddt"] = pd.to_datetime(self.df_stocknames["nameenddt"])

        # Cache all market trading dates in sorted order
        self.all_trading_dates = sorted(self.df_dsf["date"].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self.all_trading_dates)}

    def get_reconstitution_dates(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        """Identify December reconstitution dates and corresponding January effective dates.

        Returns:
            List of tuples: (reconstitution_date, effective_start_date, effective_year).
        """
        dates_series = pd.Series(self.all_trading_dates)
        years = dates_series.dt.year.unique()

        reconstitution_schedule: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []

        for yr in sorted(years):
            # Final trading date of December in year yr
            dec_dates = dates_series[
                (dates_series.dt.year == yr) & (dates_series.dt.month == 12)
            ]
            if dec_dates.empty:
                continue
            recon_date = dec_dates.max()

            # First trading date of January in year yr + 1
            next_jan_dates = dates_series[
                (dates_series.dt.year == yr + 1) & (dates_series.dt.month == 1)
            ]
            if next_jan_dates.empty:
                continue
            effective_start = next_jan_dates.min()

            reconstitution_schedule.append((recon_date, effective_start, yr + 1))

        return reconstitution_schedule

    def filter_eligible_securities(self, recon_date: pd.Timestamp) -> pd.DataFrame:
        """Apply point-in-time eligibility filters strictly ON the reconstitution date.

        Filters applied BEFORE ranking:
            a. shrcd IN (10, 11) (US common equity)
            b. exchcd IN (1, 2, 3) (NYSE, AMEX, NASDAQ point-in-time)
            c. siccd != 6798 (Excludes REITs) AND siccd NOT BETWEEN 6720 AND 6726 (Excludes closed-end funds)
            d. Absolute closing price |prc| >= min_price ($5.00)
            e. History: At least min_history_days (252) valid daily returns in preceding 252 trading days

        Args:
            recon_date: The reconstitution date (last trading day of December).

        Returns:
            DataFrame of eligible candidates on recon_date with calculated market capitalization.
        """
        # 1. Trading snapshot on reconstitution date
        dsf_recon = self.df_dsf[self.df_dsf["date"] == recon_date].copy()
        if dsf_recon.empty:
            return pd.DataFrame()

        # 2. Filter d: Price hurdle |prc| >= $5.00
        dsf_recon["abs_prc"] = dsf_recon["prc"].abs()
        dsf_recon = dsf_recon[dsf_recon["abs_prc"] >= self.min_price]
        if dsf_recon.empty:
            return pd.DataFrame()

        # 3. Filter e: History requirement (>= 252 trading days of valid return in trailing 252 market days)
        recon_idx = self._date_to_idx.get(recon_date)
        if recon_idx is None or recon_idx < (self.min_history_days - 1):
            return pd.DataFrame()

        history_window_dates = self.all_trading_dates[
            recon_idx - self.min_history_days + 1 : recon_idx + 1
        ]
        history_slice = self.df_dsf[
            self.df_dsf["date"].isin(history_window_dates)
            & self.df_dsf["ret"].notna()
        ]
        valid_history_counts = history_slice.groupby("permno")["ret"].count()
        seasoned_permnos = set(
            valid_history_counts[valid_history_counts >= self.min_history_days].index
        )
        dsf_recon = dsf_recon[dsf_recon["permno"].isin(seasoned_permnos)]
        if dsf_recon.empty:
            return pd.DataFrame()

        # 4. Point-in-time join with stocknames valid on recon_date
        names_pit = self.df_stocknames[
            (self.df_stocknames["namedt"] <= recon_date)
            & (recon_date <= self.df_stocknames["nameenddt"])
        ].drop_duplicates(subset=["permno"])

        merged = dsf_recon.merge(names_pit, on="permno", how="inner")

        # 5. Filter a: shrcd IN (10, 11)
        merged = merged[merged["shrcd"].isin([10, 11])]

        # 6. Filter b: exchcd IN (1, 2, 3)
        merged = merged[merged["exchcd"].isin([1, 2, 3])]

        # 7. Filter c: Industry exclusions (REITs: 6798; Closed-end funds: 6720-6726)
        is_reit = merged["siccd"] == 6798
        is_cef = merged["siccd"].between(6720, 6726)
        merged = merged[~(is_reit | is_cef)]

        # 8. Calculate Market Capitalization: Cap = |prc| * shrout * 1000
        merged["mkt_cap"] = merged["abs_prc"] * merged["shrout"] * 1000.0
        merged = merged.dropna(subset=["mkt_cap"])

        return merged

    def select_top_n(self, eligible_candidates: pd.DataFrame) -> List[int]:
        """Rank eligible securities by market cap descending with deterministic tie-breaking.

        Args:
            eligible_candidates: DataFrame of eligible securities from filter_eligible_securities.

        Returns:
            List of chosen PERMNO integers (up to target_size).
        """
        if eligible_candidates.empty:
            return []

        # Sort descending by mkt_cap, ascending by permno for deterministic tie-breaking
        sorted_candidates = eligible_candidates.sort_values(
            by=["mkt_cap", "permno"], ascending=[False, True]
        )
        return sorted_candidates["permno"].head(self.target_size).tolist()

    def build_annual_constituents(self) -> Dict[int, List[int]]:
        """Compute the annual Top-N constituent list for each effective year.

        Returns:
            Dictionary mapping effective_year (int) -> list of constituent PERMNOs.
        """
        schedule = self.get_reconstitution_dates()
        annual_universe: Dict[int, List[int]] = {}

        for recon_date, _, effective_year in schedule:
            eligible = self.filter_eligible_securities(recon_date)
            constituents = self.select_top_n(eligible)
            annual_universe[effective_year] = constituents

        return annual_universe

    def get_daily_membership(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Construct the daily point-in-time universe membership panel.

        For each effective year, constituents selected at the preceding December
        reconstitution remain in the universe on day t if and only if they actively
        trade in crsp.dsf on day t. Midyear delistings naturally cause daily constituent
        counts to contract below target_size as permnos exit.

        Args:
            start_year: Optional first effective year to include (e.g. 2015).
            end_year: Optional final effective year to include (e.g. 2024).

        Returns:
            DataFrame with MultiIndex ['date', 'permno'] and boolean column 'in_universe'.
        """
        annual_universe = self.build_annual_constituents()
        all_membership_records: List[pd.DataFrame] = []

        # Filter effective years if specified
        active_years = sorted(annual_universe.keys())
        if start_year is not None:
            active_years = [y for y in active_years if y >= start_year]
        if end_year is not None:
            active_years = [y for y in active_years if y <= end_year]

        for yr in active_years:
            constituents = set(annual_universe[yr])
            if not constituents:
                continue

            # Slice dsf for the effective calendar year
            year_mask = self.df_dsf["date"].dt.year == yr
            dsf_year = self.df_dsf[year_mask]

            # Security is active in universe if selected and present in dsf today
            active_today = dsf_year[dsf_year["permno"].isin(constituents)][
                ["date", "permno"]
            ].copy()
            active_today["in_universe"] = True
            all_membership_records.append(active_today)

        if not all_membership_records:
            return pd.DataFrame(columns=["in_universe"], index=pd.MultiIndex.from_tuples([], names=["date", "permno"]))

        universe_df = pd.concat(all_membership_records, ignore_index=True)
        universe_df = universe_df.set_index(["date", "permno"]).sort_index()
        return universe_df
