"""
Person 2 — CRSP Daily Data (Kozak & Nagel 2024 replication, MGT 6078)

What this script does
---------------------
1. Loads the monthly top-500 universe (Person 1's monthly_panel.parquet).
   If that file isn't ready yet, it can build a provisional universe from CRSP
   monthly data so you're not blocked (use --build-universe).
2. Takes the UNION of every PERMNO that was ever in the universe (2018-01..2025-12)
   and downloads their daily returns for 2016-01-01..2025-12-31 from CRSP (CIZ / v2
   tables, which run through 2025; legacy crsp.dsf stops at 2024).
3. Checks coverage:
     - per stock: first/last date, # trading days, # missing returns, gaps
     - per (formation month, stock): # non-missing daily returns in the trailing
       window that ends ON OR BEFORE the formation date (no look-ahead)
     - per formation month: how many of the 500 have enough history for covariance
4. Writes outputs to an output folder (put that folder in Google Drive / approved
   storage, NOT in GitHub — CRSP data is licensed).

Rules followed (from the team's "Common data rules")
-----------------------------------------------------
- PERMNO is the stock identifier.
- Missing daily returns stay NaN. Nothing is filled with zero.
- Covariance windows only use dates <= formation date; holding returns start after it.
- CIZ daily returns (dlyret) already include delisting returns, so no separate
  delisting adjustment is applied here (avoids double counting — Person 3 audits this).

Usage
-----
    # with Person 1's file
    python person2_daily_returns.py --universe path/to/monthly_panel.parquet --out output/

    # provisional universe straight from CRSP (if Person 1's file isn't ready)
    python person2_daily_returns.py --build-universe --out output/

    # rerun checks without re-downloading (uses cached raw pull)
    python person2_daily_returns.py --universe ... --out output/ --use-cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
DAILY_START = "2016-01-01"
DAILY_END = "2025-12-31"
FORMATION_START = "2018-01-01"
FORMATION_END = "2025-12-31"
UNIVERSE_SIZE = 500
LOOKBACK_DAYS = 252          # trailing trading days used for covariance estimation
MIN_OBS_FRAC = 0.80          # stock needs >= 80% of the window non-missing to be "sufficient"
PERMNO_CHUNK = 500           # how many PERMNOs per SQL query
GAP_FLAG_DAYS = 5            # report runs of >= this many consecutive missing trading days


# --------------------------------------------------------------------------------------
# 1. Universe
# --------------------------------------------------------------------------------------
def load_universe(path: str, permno_col: str = "permno", date_col: str | None = None) -> pd.DataFrame:
    """Load Person 1's monthly panel and return columns [formation_date, permno].

    formation_date = month-end date at which the top-500 was chosen.
    If date_col isn't given, the first column with 'date' in its name is used.
    """
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    if date_col is None:
        candidates = [c for c in df.columns if "date" in c.lower() or c.lower() in ("mthcaldt", "yyyymm")]
        if not candidates:
            raise ValueError(f"Can't find a date column in {list(df.columns)}; pass --date-col")
        date_col = candidates[0]
        print(f"[universe] using '{date_col}' as formation date column")
    out = pd.DataFrame({
        "formation_date": pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0),
        "permno": df[permno_col].astype("int64"),
    }).drop_duplicates()
    out = out[(out.formation_date >= FORMATION_START) & (out.formation_date <= FORMATION_END)]
    return out.sort_values(["formation_date", "permno"]).reset_index(drop=True)


def build_universe_from_crsp(db) -> pd.DataFrame:
    """PROVISIONAL top-500 universe from CRSP CIZ monthly file, for use until Person 1's
    panel is ready. Filters = CIZ equivalent of shrcd 10/11 and exchcd 1/2/3.
    Person 1's file is the source of truth; swap to it as soon as it exists.
    """
    sql = f"""
        SELECT permno, mthcaldt, mthcap
        FROM crsp.msf_v2
        WHERE mthcaldt BETWEEN '{FORMATION_START}' AND '{FORMATION_END}'
          AND sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
          AND usincflg = 'Y' AND issuertype IN ('ACOR', 'CORP')
          AND primaryexch IN ('N', 'A', 'Q')
          AND conditionaltype IN ('RW', 'NW') AND tradingstatusflg = 'A'
          AND mthcap IS NOT NULL
    """
    m = db.raw_sql(sql, date_cols=["mthcaldt"])
    m["formation_date"] = m["mthcaldt"] + pd.offsets.MonthEnd(0)
    m = m.sort_values(["formation_date", "mthcap", "permno"], ascending=[True, False, True])
    top = m.groupby("formation_date", group_keys=False).head(UNIVERSE_SIZE)
    return top[["formation_date", "permno"]].astype({"permno": "int64"}).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. Download daily returns
# --------------------------------------------------------------------------------------
def download_daily(db, permnos: list[int]) -> pd.DataFrame:
    """Pull daily data for the given PERMNOs from crsp.dsf_v2 (CIZ format) in chunks."""
    frames = []
    for i in range(0, len(permnos), PERMNO_CHUNK):
        chunk = permnos[i:i + PERMNO_CHUNK]
        ids = ",".join(str(int(p)) for p in chunk)
        sql = f"""
            SELECT permno, dlycaldt AS date, dlyret AS ret, dlyretx AS retx,
                   dlyprc AS prc, dlycap AS cap, dlyvol AS vol,
                   dlyretmissflg AS ret_missing_flag, dlydelflg AS delist_flag
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{DAILY_START}' AND '{DAILY_END}'
              AND permno IN ({ids})
        """
        print(f"[download] permnos {i + 1}-{i + len(chunk)} of {len(permnos)}")
        frames.append(db.raw_sql(sql, date_cols=["date"]))
    return pd.concat(frames, ignore_index=True)


def download_market_calendar(db) -> pd.DatetimeIndex:
    """Trading-day calendar = every date on which ANY stock in CRSP (crsp.dsf_v2) has a
    record. This covers the whole market, not just our 853 stocks, so a day where one of
    our stocks is missing still shows up as a missing day."""
    cal = db.raw_sql(
        f"SELECT DISTINCT dlycaldt AS date FROM crsp.dsf_v2 "
        f"WHERE dlycaldt BETWEEN '{DAILY_START}' AND '{DAILY_END}'",
        date_cols=["date"],
    )
    return pd.DatetimeIndex(sorted(cal["date"].unique()))


def read_local_daily(path: str) -> pd.DataFrame:
    """Read a daily file someone already built (e.g. the placeholder in the Drive 'daily'
    folder) and rename its columns to permno / date / ret."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)
    rename = {}
    for want, options in {"date": ["date", "dlycaldt", "trade_date", "day"],
                          "ret": ["ret", "dlyret", "return", "daily_ret", "ret_d"]}.items():
        if want not in df.columns:
            hit = next((c for c in options if c in df.columns), None)
            if hit is None:
                raise ValueError(f"No '{want}' column in {list(df.columns)}; rename it in the file")
            rename[hit] = want
    df = df.rename(columns=rename)
    if "permno" not in df.columns:
        raise ValueError(f"No 'permno' column in {list(df.columns)}")
    return df


def clean_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, de-duplicate, sort. Missing returns are left as NaN on purpose."""
    df = raw.copy()
    df["permno"] = df["permno"].astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["ret", "retx", "prc", "cap", "vol"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    n_before = len(df)
    df = df.drop_duplicates(subset=["permno", "date"], keep="last")
    if len(df) != n_before:
        print(f"[clean] dropped {n_before - len(df)} duplicate (permno, date) rows")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 3. Coverage checks
# --------------------------------------------------------------------------------------
def stock_coverage(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-stock summary + list of missing-data gaps.

    A stock's 'expected' days are the trading days between its first and last CRSP
    record in the sample. Missing = no row on that day OR a row with NaN return.
    """
    rows, gaps = [], []
    for permno, g in daily.groupby("permno", sort=True):
        first, last = g["date"].min(), g["date"].max()
        expected = calendar[(calendar >= first) & (calendar <= last)]
        have = g.loc[g["ret"].notna(), "date"]
        missing = expected.difference(pd.DatetimeIndex(have))
        rows.append({
            "permno": permno,
            "first_date": first.date(),
            "last_date": last.date(),
            "expected_days": len(expected),
            "rows_in_crsp": len(g),
            "nonmissing_ret": len(have),
            "missing_ret_days": len(missing),
            "pct_missing": round(len(missing) / max(len(expected), 1), 4),
            "starts_after_sample_start": first > calendar[0],
            "ends_before_sample_end": last < calendar[-1],
        })
        # group consecutive missing trading days into runs
        if len(missing):
            pos = expected.get_indexer(missing)
            run_id = np.cumsum(np.r_[1, np.diff(pos) != 1])
            for _, idx in pd.Series(pos).groupby(run_id):
                if len(idx) >= GAP_FLAG_DAYS:
                    gaps.append({
                        "permno": permno,
                        "gap_start": expected[idx.iloc[0]].date(),
                        "gap_end": expected[idx.iloc[-1]].date(),
                        "trading_days_missing": len(idx),
                    })
    gap_cols = ["permno", "gap_start", "gap_end", "trading_days_missing"]
    return pd.DataFrame(rows), pd.DataFrame(gaps, columns=gap_cols)


def estimation_window(calendar: pd.DatetimeIndex, formation_date, lookback: int = LOOKBACK_DAYS) -> pd.DatetimeIndex:
    """Trading days used to estimate covariance at a formation date.

    The window ENDS on the last trading day <= formation_date (month-end). Nothing
    after the formation date is ever included. Holding-period returns start on the
    next trading day.
    """
    eligible = calendar[calendar <= pd.Timestamp(formation_date)]
    return eligible[-lookback:]


def formation_coverage(daily: pd.DataFrame, universe: pd.DataFrame, calendar: pd.DatetimeIndex,
                       lookback: int = LOOKBACK_DAYS, min_frac: float = MIN_OBS_FRAC) -> pd.DataFrame:
    """For each (formation month, stock in that month's top 500): how many non-missing
    daily returns exist in the trailing window, and is it enough?"""
    min_obs = int(np.ceil(lookback * min_frac))
    ret_wide = daily.pivot(index="date", columns="permno", values="ret").reindex(calendar)
    out = []
    for fdate, grp in universe.groupby("formation_date"):
        win = estimation_window(calendar, fdate, lookback)
        assert win.max() <= fdate, "look-ahead: window extends past formation date"
        cols = grp["permno"].tolist()
        sub = ret_wide.reindex(index=win, columns=cols)
        n_obs = sub.notna().sum(axis=0)
        out.append(pd.DataFrame({
            "formation_date": fdate,
            "permno": cols,
            "window_start": win.min(),
            "window_end": win.max(),
            "window_days": len(win),
            "n_obs": n_obs.values,
            "in_daily_file": [c in ret_wide.columns for c in cols],
        }))
    res = pd.concat(out, ignore_index=True)
    res["sufficient"] = (res["n_obs"] >= min_obs) & (res["window_days"] == lookback)
    res["min_obs_required"] = min_obs
    return res


def formation_summary(fc: pd.DataFrame) -> pd.DataFrame:
    s = fc.groupby("formation_date").agg(
        n_stocks=("permno", "size"),
        n_missing_from_daily=("in_daily_file", lambda x: int((~x).sum())),
        n_sufficient=("sufficient", "sum"),
        median_obs=("n_obs", "median"),
        min_obs=("n_obs", "min"),
        window_start=("window_start", "first"),
        window_end=("window_end", "first"),
    ).reset_index()
    s["pct_sufficient"] = (s["n_sufficient"] / s["n_stocks"]).round(4)
    return s


def missing_reasons(daily: pd.DataFrame) -> pd.DataFrame:
    """Count missing returns by CRSP's own reason code (dlyretmissflg), if available."""
    if "ret_missing_flag" not in daily.columns:
        return pd.DataFrame(columns=["ret_missing_flag", "n_rows"])
    m = daily[daily["ret"].isna()]
    return (m["ret_missing_flag"].fillna("(blank)").value_counts()
            .rename_axis("ret_missing_flag").reset_index(name="n_rows"))


def monthly_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month view: how many stocks have data, share of missing returns.
    Useful for spotting periods with poor coverage."""
    d = daily.assign(month=daily["date"].dt.to_period("M"))
    return d.groupby("month").agg(
        n_stocks=("permno", "nunique"),
        n_rows=("ret", "size"),
        n_missing_ret=("ret", lambda x: int(x.isna().sum())),
    ).assign(pct_missing=lambda x: (x.n_missing_ret / x.n_rows).round(4)).reset_index()


# --------------------------------------------------------------------------------------
# 4. Report
# --------------------------------------------------------------------------------------
def write_report(path: Path, daily, universe, stock_cov, gaps, fsum, mcov, source_note: str):
    worst = fsum.nsmallest(5, "pct_sufficient")
    lines = [
        "# Person 2 — Daily Returns: Validation Report",
        "",
        "## What was done",
        f"- Universe source: {source_note}",
        f"- Formation months: {universe.formation_date.min().date()} to {universe.formation_date.max().date()} "
        f"({universe.formation_date.nunique()} months)",
        f"- Union of PERMNOs ever in the universe: {universe.permno.nunique()}",
        f"- Daily data: CRSP CIZ `crsp.dsf_v2`, {DAILY_START} to {DAILY_END}; trading calendar = all distinct dates in `crsp.dsf_v2`",
        f"- Rows in daily_returns.parquet: {len(daily):,}; PERMNOs with any daily data: {daily.permno.nunique()}",
        "- Return field: `dlyret` (total return, decimal, includes delisting return — no separate delisting adjustment).",
        "- Missing returns kept as NaN. No filling, no imputation.",
        "",
        "## Covariance window convention",
        f"- Window = last {LOOKBACK_DAYS} trading days ending on the last trading day <= formation month-end.",
        "- Holding-period returns start the next trading day. Code asserts no window date > formation date.",
        f"- A stock is 'sufficient' if it has >= {int(np.ceil(LOOKBACK_DAYS * MIN_OBS_FRAC))} non-missing returns "
        f"({MIN_OBS_FRAC:.0%}) in the window.",
        "",
        "## Coverage results",
        f"- Average share of universe with sufficient history: {fsum.pct_sufficient.mean():.2%}",
        f"- Lowest month: {fsum.loc[fsum.pct_sufficient.idxmin(), 'formation_date'].date()} "
        f"({fsum.pct_sufficient.min():.2%})",
        f"- Universe PERMNOs with no daily data at all: {int(fsum.n_missing_from_daily.sum())} stock-months",
        f"- Stocks with any missing-return days inside their life: {int((stock_cov.missing_ret_days > 0).sum())}",
        f"- Gaps of >= {GAP_FLAG_DAYS} consecutive missing trading days: {len(gaps)}",
        f"- Worst calendar month for missing returns: {mcov.loc[mcov.pct_missing.idxmax(), 'month']} "
        f"({mcov.pct_missing.max():.2%})",
        "",
        "### Five weakest formation months",
        worst[["formation_date", "n_stocks", "n_sufficient", "pct_sufficient", "median_obs"]].to_string(index=False),
        "",
        "## Unresolved / for the team",
        "- Insufficient-history stocks are usually recent IPOs or spin-offs. Decide as a team: drop them from that "
        "month's covariance estimate, or use pairwise / shrinkage estimates.",
        "- Check any PERMNO listed as missing from the daily file against Person 1 (identifier mismatch?).",
    ]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", help="Person 1's monthly_panel.parquet")
    ap.add_argument("--permno-col", default="permno")
    ap.add_argument("--date-col", default=None, help="formation-date column in the universe file")
    ap.add_argument("--build-universe", action="store_true", help="build provisional top-500 from CRSP")
    ap.add_argument("--out", default="output", help="output folder (keep outside the git repo or gitignored)")
    ap.add_argument("--use-cache", action="store_true", help="reuse raw pulls saved in --out")
    ap.add_argument("--daily-file", help="use a local daily file (parquet/csv) instead of downloading from WRDS")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path, cal_path, uni_path = out / "_raw_dsf_v2.parquet", out / "_calendar.parquet", out / "_universe_used.parquet"

    db = None
    def conn():
        nonlocal db
        if db is None:
            import wrds
            db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
        return db

    # Universe
    if args.universe:
        universe = load_universe(args.universe, args.permno_col, args.date_col)
        source_note = f"Person 1 file `{Path(args.universe).name}`"
    elif args.build_universe:
        universe = build_universe_from_crsp(conn())
        source_note = "PROVISIONAL top-500 by market cap from `crsp.msf_v2` (replace with Person 1's panel)"
    else:
        ap.error("give --universe PATH or --build-universe")
    universe.to_parquet(uni_path, index=False)
    permnos = sorted(universe["permno"].unique().tolist())
    print(f"[universe] {universe.formation_date.nunique()} months, {len(permnos)} unique PERMNOs")

    # Guard: placeholder IDs (900000+) from the public/Yahoo panel don't exist in CRSP
    if not args.daily_file and any(p >= 900000 for p in permnos):
        raise SystemExit("Universe contains placeholder PERMNOs (900000+). These aren't in CRSP. "
                         "Use --build-universe, or wait for the CRSP version of the monthly panel.")

    # Daily data
    if args.daily_file:
        raw = read_local_daily(args.daily_file)
        calendar = pd.DatetimeIndex(sorted(pd.to_datetime(raw["date"]).unique()))
        source_note += f"; daily returns from local file `{Path(args.daily_file).name}` (calendar = dates in file)"
        print(f"[daily] loaded {len(raw):,} rows from {args.daily_file}")
    elif args.use_cache and raw_path.exists():
        raw = pd.read_parquet(raw_path)
        print("[download] using cached raw pull")
        if cal_path.exists():
            calendar = pd.DatetimeIndex(pd.read_parquet(cal_path)["date"])
        else:
            calendar = download_market_calendar(conn())
            pd.DataFrame({"date": calendar}).to_parquet(cal_path, index=False)
        print(f"[calendar] {len(calendar)} trading days")
    else:
        raw = download_daily(conn(), permnos)
        raw.to_parquet(raw_path, index=False)
        calendar = download_market_calendar(conn())
        pd.DataFrame({"date": calendar}).to_parquet(cal_path, index=False)

    daily = clean_daily(raw)
    daily = daily[daily["permno"].isin(permnos)]

    # Checks
    stock_cov, gaps = stock_coverage(daily, calendar)
    fc = formation_coverage(daily, universe, calendar)
    fsum = formation_summary(fc)
    mcov = monthly_coverage(daily)

    # Outputs
    daily.to_parquet(out / "daily_returns.parquet", index=False)
    stock_cov.to_csv(out / "daily_coverage_by_stock.csv", index=False)
    gaps.to_csv(out / "daily_gaps.csv", index=False)
    fc.to_csv(out / "daily_coverage_by_formation_stock.csv", index=False)
    fsum.to_csv(out / "daily_coverage_by_formation_month.csv", index=False)
    mcov.to_csv(out / "daily_coverage_by_calendar_month.csv", index=False)
    missing_reasons(daily).to_csv(out / "daily_missing_reasons.csv", index=False)
    write_report(out / "daily_returns_report.md", daily, universe, stock_cov, gaps, fsum, mcov, source_note)
    print(f"[done] outputs written to {out.resolve()}")
    print(fsum[["formation_date", "n_stocks", "n_sufficient", "pct_sufficient"]].tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
