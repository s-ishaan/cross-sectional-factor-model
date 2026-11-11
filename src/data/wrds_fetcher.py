"""
WRDS Data Ingestion Module for CRSP US Stocks.

Provides automated, robust extraction of CRSP Daily Stock File (dsf),
delistings (dsedelist), security metadata (stocknames), and index returns (dsi)
from Wharton Research Data Services (WRDS), with local Parquet caching.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import wrds


class WRDSFetcher:
    """Connects to WRDS and extracts CRSP daily stock and index datasets."""

    DEFAULT_START_DATE = "2014-01-01"
    DEFAULT_END_DATE = "2024-12-31"

    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        wrds_username: Optional[str] = None,
    ) -> None:
        """Initialize the WRDS fetcher.

        Args:
            output_dir: Local directory path to cache Parquet files.
            wrds_username: Optional WRDS account username. If omitted, falls back to
                WRDS_USERNAME or WRDS_USER environment variables, or default wrds config.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.username = (
            wrds_username
            or os.environ.get("WRDS_USERNAME")
            or os.environ.get("WRDS_USER")
        )

    def _get_connection(self) -> wrds.Connection:
        """Establish an authenticated connection to WRDS."""
        if self.username:
            return wrds.Connection(wrds_username=self.username)
        return wrds.Connection()

    def fetch_stocknames(self, force_reload: bool = False) -> pd.DataFrame:
        """Extract crsp.stocknames metadata table.

        Args:
            force_reload: If True, re-queries WRDS even if local parquet cache exists.

        Returns:
            DataFrame containing PERMNO historical names, exchange codes, and share codes.
        """
        file_path = self.output_dir / "crsp_stocknames.parquet"
        if file_path.exists() and not force_reload:
            return pd.read_parquet(file_path)

        query = """
            SELECT 
                permno,
                namedt,
                nameenddt,
                shrcd,
                exchcd,
                siccd,
                ncusip,
                ticker,
                comnam,
                shrcls,
                permco,
                hexcd
            FROM crsp.stocknames
            ORDER BY permno, namedt;
        """
        db = self._get_connection()
        try:
            df = db.raw_sql(query, date_cols=["namedt", "nameenddt"])
        finally:
            db.close()

        df["permno"] = df["permno"].astype("int64")
        df["shrcd"] = pd.to_numeric(df["shrcd"], errors="coerce")
        df["exchcd"] = pd.to_numeric(df["exchcd"], errors="coerce")
        df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce")
        df.to_parquet(file_path, index=False)
        return df

    def fetch_dsedelist(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Extract crsp.dsedelist daily delisting table.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            force_reload: If True, queries WRDS regardless of local cache.

        Returns:
            DataFrame containing delisting events, codes, and settlement returns.
        """
        file_path = self.output_dir / "crsp_dsedelist.parquet"
        if file_path.exists() and not force_reload:
            return pd.read_parquet(file_path)

        query = f"""
            SELECT 
                permno,
                dlstdt,
                dlstcd,
                dlprc,
                dlpdt,
                dlret,
                dlretx
            FROM crsp.dsedelist
            WHERE dlstdt >= '{start_date}' AND dlstdt <= '{end_date}'
            ORDER BY permno, dlstdt;
        """
        db = self._get_connection()
        try:
            df = db.raw_sql(query, date_cols=["dlstdt", "dlpdt"])
        finally:
            db.close()

        df["permno"] = df["permno"].astype("int64")
        df["dlstcd"] = pd.to_numeric(df["dlstcd"], errors="coerce")
        df["dlprc"] = pd.to_numeric(df["dlprc"], errors="coerce")
        df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce")
        df["dlretx"] = pd.to_numeric(df["dlretx"], errors="coerce")
        df.to_parquet(file_path, index=False)
        return df

    def fetch_dsi(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Extract crsp.dsi daily index return table (Value-Weighted and S&P 500).

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            force_reload: If True, queries WRDS regardless of local cache.

        Returns:
            DataFrame containing daily market benchmark returns.
        """
        file_path = self.output_dir / "crsp_dsi.parquet"
        if file_path.exists() and not force_reload:
            return pd.read_parquet(file_path)

        query = f"""
            SELECT 
                date,
                vwretd,
                vwretx,
                ewretd,
                ewretx,
                sprtrn,
                spindx
            FROM crsp.dsi
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date;
        """
        db = self._get_connection()
        try:
            df = db.raw_sql(query, date_cols=["date"])
        finally:
            db.close()

        for col in ["vwretd", "vwretx", "ewretd", "ewretx", "sprtrn", "spindx"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_parquet(file_path, index=False)
        return df

    def fetch_dsf(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        permnos: Optional[Sequence[int]] = None,
        filter_common_stocks: bool = True,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Extract crsp.dsf daily stock file.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            permnos: Optional subset of PERMNO integers to extract.
            filter_common_stocks: If True and permnos is None, filters against
                crsp.stocknames to include securities with shrcd IN (10, 11) and
                exchcd IN (1, 2, 3), reducing transfer size.
            force_reload: If True, queries WRDS regardless of local cache.

        Returns:
            DataFrame containing daily stock trading data.
        """
        file_path = self.output_dir / "crsp_dsf.parquet"
        if file_path.exists() and not force_reload:
            return pd.read_parquet(file_path)

        db = self._get_connection()
        try:
            if permnos is not None:
                permno_list_str = ", ".join(str(int(p)) for p in permnos)
                where_clause = f"""
                    WHERE date >= '{start_date}' AND date <= '{end_date}'
                      AND permno IN ({permno_list_str})
                """
            elif filter_common_stocks:
                where_clause = f"""
                    WHERE date >= '{start_date}' AND date <= '{end_date}'
                      AND permno IN (
                          SELECT DISTINCT permno 
                          FROM crsp.stocknames 
                          WHERE shrcd IN (10, 11) 
                            AND exchcd IN (1, 2, 3)
                            AND namedt <= '{end_date}' 
                            AND nameenddt >= '{start_date}'
                      )
                """
            else:
                where_clause = f"WHERE date >= '{start_date}' AND date <= '{end_date}'"

            query = f"""
                SELECT 
                    permno,
                    date,
                    prc,
                    ret,
                    retx,
                    vol,
                    shrout,
                    cfacpr,
                    cfacshr
                FROM crsp.dsf
                {where_clause}
                ORDER BY permno, date;
            """
            df = db.raw_sql(query, date_cols=["date"])
        finally:
            db.close()

        df["permno"] = df["permno"].astype("int64")
        df["prc"] = pd.to_numeric(df["prc"], errors="coerce")
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        df["retx"] = pd.to_numeric(df["retx"], errors="coerce")
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
        df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
        df["cfacpr"] = pd.to_numeric(df["cfacpr"], errors="coerce")
        df["cfacshr"] = pd.to_numeric(df["cfacshr"], errors="coerce")

        df.to_parquet(file_path, index=False)
        return df

    def fetch_all(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        force_reload: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all required CRSP tables and return as a dictionary of DataFrames."""
        print(f"Ingesting CRSP tables from WRDS ({start_date} to {end_date})...")
        stocknames = self.fetch_stocknames(force_reload=force_reload)
        dsedelist = self.fetch_dsedelist(start_date=start_date, end_date=end_date, force_reload=force_reload)
        dsi = self.fetch_dsi(start_date=start_date, end_date=end_date, force_reload=force_reload)
        dsf = self.fetch_dsf(start_date=start_date, end_date=end_date, force_reload=force_reload)
        return {
            "stocknames": stocknames,
            "dsedelist": dsedelist,
            "dsi": dsi,
            "dsf": dsf,
        }
