"""
CRSP Delisting Return Adjustment Engine.

Eliminates survivorship bias in cross-sectional equity backtests by reconciling
CRSP Daily Stock File (dsf) returns with CRSP Delisting Events (dsedelist),
following Shumway (1997) and Beaver, McNichols & Rhie (2005).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class CRSPDelistingAdjuster:
    """Adjusts CRSP daily stock returns for terminal delisting settlements.

    Handles compounding of final trading-day returns with delisting returns,
    Shumway (1997) distress-return imputation (flat or exchange-sensitive),
    non-distress merger imputation, and off-calendar / post-trading date alignment.
    """

    DISTRESS_CODES_MIN: int = 500
    DISTRESS_CODES_MAX: int = 591
    NON_DISTRESS_CODES_MIN: int = 200
    NON_DISTRESS_CODES_MAX: int = 399
    ACTIVE_CODE_THRESHOLD: int = 200

    DEFAULT_DISTRESS_RET: float = -0.30
    NASDAQ_DISTRESS_RET: float = -0.55
    NON_DISTRESS_RET: float = 0.0
    NASDAQ_EXCH_CODE: int = 3

    def __init__(
        self,
        df_dsf: Optional[pd.DataFrame] = None,
        df_dsedelist: Optional[pd.DataFrame] = None,
        df_stocknames: Optional[pd.DataFrame] = None,
        exchange_sensitive: bool = False,
        distress_ret_default: float = DEFAULT_DISTRESS_RET,
        distress_ret_nasdaq: float = NASDAQ_DISTRESS_RET,
        non_distress_ret_default: float = NON_DISTRESS_RET,
    ) -> None:
        """Initialize the CRSP delisting adjuster.

        Args:
            df_dsf: Optional CRSP Daily Stock File DataFrame with columns
                ['permno', 'date', 'ret', ...] (and optionally 'retx').
            df_dsedelist: Optional CRSP Delistings DataFrame with columns
                ['permno', 'dlstdt', 'dlstcd', 'dlret', ...] (and optionally 'dlretx').
            df_stocknames: Optional CRSP metadata DataFrame with ['permno', 'exchcd']
                and optionally ['namedt', 'nameenddt'] for exchange-sensitive imputation.
            exchange_sensitive: If True, uses exchange-specific Shumway distress
                imputation (-0.55 for NASDAQ exchcd=3; -0.30 for NYSE/AMEX exchcd in {1, 2}).
            distress_ret_default: Imputed return for missing distress delisting returns
                (default: -0.30).
            distress_ret_nasdaq: Imputed return for missing NASDAQ distress delisting
                returns when exchange_sensitive=True (default: -0.55).
            non_distress_ret_default: Imputed return for missing non-distress delistings
                with dlstcd in [200, 399] (default: 0.0).
        """
        self.df_dsf = df_dsf.copy() if df_dsf is not None else None
        self.df_dsedelist = df_dsedelist.copy() if df_dsedelist is not None else None
        self.df_stocknames = df_stocknames.copy() if df_stocknames is not None else None
        self.exchange_sensitive = exchange_sensitive
        self.distress_ret_default = float(distress_ret_default)
        self.distress_ret_nasdaq = float(distress_ret_nasdaq)
        self.non_distress_ret_default = float(non_distress_ret_default)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
        """Raise ValueError if any required columns are missing from DataFrame."""
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    @staticmethod
    def _compound_returns(base_ret: pd.Series, dl_ret: pd.Series) -> pd.Series:
        """Compound daily returns with delisting returns vectorially.

        Rules:
            - Both valid: (1 + r) * (1 + dlret) - 1
            - Daily return NaN, dlret valid: dlret
            - Otherwise: retain daily return
        """
        both_valid = base_ret.notna() & dl_ret.notna()
        only_dlret = base_ret.isna() & dl_ret.notna()

        adjusted = base_ret.copy()
        adjusted.loc[both_valid] = (
            (1.0 + base_ret.loc[both_valid]) * (1.0 + dl_ret.loc[both_valid]) - 1.0
        )
        adjusted.loc[only_dlret] = dl_ret.loc[only_dlret]
        return adjusted

    @staticmethod
    def _attach_exchange_codes(
        df_delist: pd.DataFrame,
        df_stocknames: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Resolve point-in-time exchange codes (exchcd) for delisting events."""
        existing = (
            pd.to_numeric(df_delist["exchcd"], errors="coerce").astype("float64")
            if "exchcd" in df_delist.columns
            else pd.Series(np.nan, index=df_delist.index, dtype="float64")
        )
        if df_stocknames is None or df_stocknames.empty or df_delist.empty:
            return existing

        CRSPDelistingAdjuster._validate_columns(
            df_stocknames, {"permno", "exchcd"}, "df_stocknames"
        )
        names = df_stocknames.copy()
        names = names.dropna(subset=["permno"]).copy()
        names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("int64")
        names["exchcd"] = pd.to_numeric(names["exchcd"], errors="coerce").astype("float64")

        if "namedt" in names.columns:
            names["namedt"] = pd.to_datetime(names["namedt"]).astype("datetime64[ns]")
            names = names.dropna(subset=["namedt"])
            sort_cols = ["namedt", "nameenddt"] if "nameenddt" in names.columns else ["namedt"]
            if "nameenddt" in names.columns:
                names["nameenddt"] = pd.to_datetime(names["nameenddt"]).astype("datetime64[ns]")
            names = (
                names.sort_values(sort_cols, kind="mergesort")
                .drop_duplicates(subset=["permno", "namedt"], keep="last")
            )

            left = pd.DataFrame(
                {
                    "_orig_idx": df_delist.index,
                    "permno": df_delist["permno"].values,
                    "dlstdt": df_delist["dlstdt"].values,
                }
            ).sort_values(["dlstdt", "permno"], kind="mergesort")

            # Use backward asof match on namedt <= dlstdt so settlements occurring
            # after nameenddt still resolve to the security's final listing exchange.
            matched = pd.merge_asof(
                left,
                names[["permno", "namedt", "exchcd"]],
                left_on="dlstdt",
                right_on="namedt",
                by="permno",
                direction="backward",
            )
            looked_up = (
                matched.set_index("_orig_idx")["exchcd"]
                .reindex(df_delist.index)
                .astype("float64")
            )
        else:
            exch_map = names.drop_duplicates(subset=["permno"], keep="last").set_index("permno")[
                "exchcd"
            ]
            looked_up = df_delist["permno"].map(exch_map).astype("float64")

        return existing.fillna(looked_up)

    def _impute_missing_delist_col(
        self,
        values: pd.Series,
        is_distress: pd.Series,
        is_non_distress: pd.Series,
        distress_fill: np.ndarray,
    ) -> tuple[pd.Series, pd.Series]:
        """Impute missing delisting return values for distress and non-distress codes."""
        is_missing = values.isna()
        impute_distress = is_missing & is_distress
        impute_non_distress = is_missing & is_non_distress

        filled = values.copy()
        filled.loc[impute_distress] = distress_fill[impute_distress.to_numpy()]
        filled.loc[impute_non_distress] = self.non_distress_ret_default
        was_imputed = impute_distress | impute_non_distress
        return filled, was_imputed

    def clean_delisting_returns(
        self,
        df_dsedelist: Optional[pd.DataFrame] = None,
        df_stocknames: Optional[pd.DataFrame] = None,
        exchange_sensitive: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Deduplicate and impute missing CRSP delisting returns.

        Imputation rules for missing dlret (and dlretx when present):
            - Distress delistings (dlstcd in [500, 591]):
                - Standard: -0.30
                - Exchange-sensitive (when enabled): -0.55 for NASDAQ (exchcd == 3),
                  -0.30 for NYSE/AMEX (exchcd in {1, 2}) or unknown exchange.
            - Non-distress delistings (dlstcd in [200, 399]): 0.0.
            - Active placeholders (dlstcd < 200) or other codes: left unchanged.

        Args:
            df_dsedelist: CRSP delistings table. Defaults to instance df_dsedelist.
            df_stocknames: Optional CRSP stocknames table for exchange lookup.
                Defaults to instance df_stocknames.
            exchange_sensitive: Override instance exchange_sensitive flag if provided.

        Returns:
            Cleaned copy of df_dsedelist sorted by ['permno', 'dlstdt'] with
            imputed returns and a boolean 'dlret_imputed' audit flag.
        """
        raw_delist = df_dsedelist if df_dsedelist is not None else self.df_dsedelist
        if raw_delist is None:
            raise ValueError(
                "df_dsedelist must be provided either at initialization or to clean_delisting_returns()."
            )

        self._validate_columns(
            raw_delist, {"permno", "dlstdt", "dlstcd", "dlret"}, "df_dsedelist"
        )
        stocknames = df_stocknames if df_stocknames is not None else self.df_stocknames
        use_exchange = self.exchange_sensitive if exchange_sensitive is None else exchange_sensitive

        df = raw_delist.copy()
        if df.empty:
            df["permno"] = pd.Series(dtype="int64")
            df["dlstdt"] = pd.Series(dtype="datetime64[ns]")
            df["dlstcd"] = pd.Series(dtype="float64")
            df["dlret"] = pd.Series(dtype="float64")
            df["dlret_imputed"] = pd.Series(dtype="bool")
            return df

        df = df.dropna(subset=["permno", "dlstdt"]).copy()
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
        df = df.dropna(subset=["permno"]).copy()
        df["permno"] = df["permno"].astype("int64")
        df["dlstdt"] = pd.to_datetime(df["dlstdt"]).astype("datetime64[ns]")
        df["dlstcd"] = pd.to_numeric(df["dlstcd"], errors="coerce").astype("float64")
        df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce").astype("float64")
        if "dlretx" in df.columns:
            df["dlretx"] = pd.to_numeric(df["dlretx"], errors="coerce").astype("float64")

        # Deduplicate duplicate (permno, dlstdt) entries, preferring non-null dlret
        # and actual delisting codes (>= 200) over placeholders.
        df["_has_dlret"] = df["dlret"].notna()
        df["_is_delist_code"] = df["dlstcd"].fillna(-1) >= self.ACTIVE_CODE_THRESHOLD
        df = (
            df.sort_values(
                ["permno", "dlstdt", "_is_delist_code", "_has_dlret", "dlstcd"],
                kind="mergesort",
            )
            .drop_duplicates(subset=["permno", "dlstdt"], keep="last")
            .drop(columns=["_has_dlret", "_is_delist_code"])
            .reset_index(drop=True)
        )

        is_distress = df["dlstcd"].between(self.DISTRESS_CODES_MIN, self.DISTRESS_CODES_MAX)
        is_non_distress = df["dlstcd"].between(
            self.NON_DISTRESS_CODES_MIN, self.NON_DISTRESS_CODES_MAX
        )

        distress_fill = np.full(len(df), self.distress_ret_default, dtype="float64")
        if use_exchange:
            exchcd = self._attach_exchange_codes(df, stocknames)
            is_nasdaq = exchcd == self.NASDAQ_EXCH_CODE
            distress_fill = np.where(
                is_nasdaq.to_numpy(), self.distress_ret_nasdaq, self.distress_ret_default
            )

        prior_imputed = (
            df["dlret_imputed"].astype("bool")
            if "dlret_imputed" in df.columns
            else pd.Series(False, index=df.index, dtype="bool")
        )
        df["dlret"], newly_imputed = self._impute_missing_delist_col(
            df["dlret"], is_distress, is_non_distress, distress_fill
        )
        df["dlret_imputed"] = prior_imputed | newly_imputed

        if "dlretx" in df.columns:
            df["dlretx"], _ = self._impute_missing_delist_col(
                df["dlretx"], is_distress, is_non_distress, distress_fill
            )

        return df

    def apply_delisting_returns(
        self,
        df_dsf: Optional[pd.DataFrame] = None,
        df_dsedelist: Optional[pd.DataFrame] = None,
        df_stocknames: Optional[pd.DataFrame] = None,
        exchange_sensitive: Optional[bool] = None,
        include_retx: bool = True,
    ) -> pd.DataFrame:
        """Merge cleaned delisting returns onto CRSP daily stock data.

        Aligns off-market or post-trading delisting dates onto each security's last
        recorded trading date in df_dsf, computes compounded 'ret_adj' (and 'retx_adj'
        when 'retx' and 'dlretx' are present), and flags delisting dates with
        'is_delisting_day'.

        Args:
            df_dsf: CRSP Daily Stock File DataFrame. Defaults to instance df_dsf.
            df_dsedelist: CRSP Delistings DataFrame. Defaults to instance df_dsedelist.
            df_stocknames: Optional CRSP stocknames table for exchange-sensitive imputation.
            exchange_sensitive: Override instance exchange_sensitive flag if provided.
            include_retx: If True and both 'retx' and 'dlretx' are present, computes
                'retx_adj'.

        Returns:
            Copy of df_dsf enriched with 'dlstdt', 'dlstcd', 'dlret', 'is_delisting_day',
            and 'ret_adj' (plus 'dlretx' and 'retx_adj' when applicable).
        """
        raw_dsf = df_dsf if df_dsf is not None else self.df_dsf
        raw_delist = df_dsedelist if df_dsedelist is not None else self.df_dsedelist
        if raw_dsf is None:
            raise ValueError(
                "df_dsf must be provided either at initialization or to apply_delisting_returns()."
            )
        if raw_delist is None:
            raise ValueError(
                "df_dsedelist must be provided either at initialization or to apply_delisting_returns()."
            )

        self._validate_columns(raw_dsf, {"permno", "date", "ret"}, "df_dsf")

        dsf_out = raw_dsf.copy()
        compute_retx = include_retx and ("retx" in dsf_out.columns) and ("dlretx" in raw_delist.columns)

        # Remove stale output columns if caller passes an already-adjusted DataFrame
        stale_cols = ["dlstdt", "dlstcd", "dlret", "is_delisting_day", "ret_adj"]
        if compute_retx:
            stale_cols.extend(["dlretx", "retx_adj"])
        existing_stale = [c for c in stale_cols if c in dsf_out.columns]
        if existing_stale:
            dsf_out = dsf_out.drop(columns=existing_stale)

        if dsf_out.empty:
            dsf_out["permno"] = pd.Series(dtype="int64")
            dsf_out["date"] = pd.Series(dtype="datetime64[ns]")
            dsf_out["ret"] = pd.Series(dtype="float64")
            dsf_out["dlstdt"] = pd.Series(dtype="datetime64[ns]")
            dsf_out["dlstcd"] = pd.Series(dtype="float64")
            dsf_out["dlret"] = pd.Series(dtype="float64")
            dsf_out["is_delisting_day"] = pd.Series(dtype="bool")
            dsf_out["ret_adj"] = pd.Series(dtype="float64")
            if compute_retx:
                dsf_out["dlretx"] = pd.Series(dtype="float64")
                dsf_out["retx_adj"] = pd.Series(dtype="float64")
            return dsf_out

        dsf_out["permno"] = pd.to_numeric(dsf_out["permno"], errors="coerce")
        dsf_out = dsf_out.dropna(subset=["permno", "date"]).copy()
        dsf_out["permno"] = dsf_out["permno"].astype("int64")
        dsf_out["date"] = pd.to_datetime(dsf_out["date"]).astype("datetime64[ns]")
        dsf_out["ret"] = pd.to_numeric(dsf_out["ret"], errors="coerce").astype("float64")
        if "retx" in dsf_out.columns:
            dsf_out["retx"] = pd.to_numeric(dsf_out["retx"], errors="coerce").astype("float64")

        # Guard against duplicate (permno, date) rows in df_dsf
        if dsf_out.duplicated(subset=["permno", "date"]).any():
            dsf_out["_has_ret"] = dsf_out["ret"].notna()
            dsf_out = (
                dsf_out.sort_values(["permno", "date", "_has_ret"], kind="mergesort")
                .drop_duplicates(subset=["permno", "date"], keep="last")
                .drop(columns=["_has_ret"])
            )

        cleaned_delist = self.clean_delisting_returns(
            raw_delist,
            df_stocknames=df_stocknames,
            exchange_sensitive=exchange_sensitive,
        )

        # Exclude active non-delisting placeholders (dlstcd < 200)
        is_actual_delist = cleaned_delist["dlstcd"].isna() | (
            cleaned_delist["dlstcd"] >= self.ACTIVE_CODE_THRESHOLD
        )
        delist_events = cleaned_delist[is_actual_delist].copy()

        if delist_events.empty:
            dsf_out["dlstdt"] = pd.NaT
            dsf_out["dlstcd"] = np.nan
            dsf_out["dlret"] = np.nan
            dsf_out["is_delisting_day"] = False
            dsf_out["ret_adj"] = dsf_out["ret"]
            if compute_retx:
                dsf_out["dlretx"] = np.nan
                dsf_out["retx_adj"] = dsf_out["retx"]
            return dsf_out

        # Align each delisting date onto the latest recorded trading date <= dlstdt
        # for that permno. This captures exact-date delistings, weekend/holiday
        # delistings, and post-trading settlement dates in one vectorized pass.
        dsf_dates = (
            dsf_out[["permno", "date"]]
            .drop_duplicates()
            .sort_values(["date", "permno"], kind="mergesort")
        )
        delist_sorted = delist_events.sort_values(["dlstdt", "permno"], kind="mergesort")

        aligned = pd.merge_asof(
            delist_sorted,
            dsf_dates,
            left_on="dlstdt",
            right_on="date",
            by="permno",
            direction="backward",
        )
        aligned = aligned.dropna(subset=["date"]).copy()

        if aligned.empty:
            dsf_out["dlstdt"] = pd.NaT
            dsf_out["dlstcd"] = np.nan
            dsf_out["dlret"] = np.nan
            dsf_out["is_delisting_day"] = False
            dsf_out["ret_adj"] = dsf_out["ret"]
            if compute_retx:
                dsf_out["dlretx"] = np.nan
                dsf_out["retx_adj"] = dsf_out["retx"]
            return dsf_out

        # If multiple delisting records for the same permno map to the same trading date,
        # retain the record with a valid dlret and the latest dlstdt.
        aligned["_has_dlret"] = aligned["dlret"].notna()
        aligned = (
            aligned.sort_values(
                ["permno", "date", "_has_dlret", "dlstdt"], kind="mergesort"
            )
            .drop_duplicates(subset=["permno", "date"], keep="last")
            .drop(columns=["_has_dlret"])
        )

        merge_cols = ["permno", "date", "dlstdt", "dlstcd", "dlret"]
        if compute_retx:
            merge_cols.append("dlretx")

        dsf_out = dsf_out.merge(aligned[merge_cols], on=["permno", "date"], how="left")
        dsf_out["is_delisting_day"] = dsf_out["dlstdt"].notna()
        dsf_out["ret_adj"] = self._compound_returns(dsf_out["ret"], dsf_out["dlret"])
        if compute_retx:
            dsf_out["retx_adj"] = self._compound_returns(dsf_out["retx"], dsf_out["dlretx"])

        return dsf_out
