"""
Purged and Embargoed Walk-Forward Cross-Validation Splitter.

Prevents forward-label horizon overlap across expanding-window time-series splits
for cross-sectional equity panels indexed by ['date', 'permno'].
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class PurgedWalkForwardCV:
    """Scikit-learn compatible expanding-window splitter with label purging and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        holding_period: int = 5,
        embargo_days: int = 0,
        min_train_dates: Optional[int] = None,
    ) -> None:
        """Initialize the purged walk-forward CV splitter.

        Args:
            n_splits: Number of expanding walk-forward validation folds (default: 5).
            holding_period: Forward target horizon h in trading days to purge from
                the tail of each training fold (default: 5).
            embargo_days: Additional buffer trading days removed before validation start
                (default: 0).
            min_train_dates: Optional minimum number of purged training dates required
                in the initial fold.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if holding_period < 0:
            raise ValueError("holding_period must be >= 0.")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0.")
        if min_train_dates is not None and min_train_dates < 1:
            raise ValueError("min_train_dates must be >= 1 when specified.")

        self.n_splits = int(n_splits)
        self.holding_period = int(holding_period)
        self.embargo_days = int(embargo_days)
        self.min_train_dates = int(min_train_dates) if min_train_dates is not None else None

    @staticmethod
    def _extract_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            return pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values("date")))
        if "date" in df.columns:
            return pd.DatetimeIndex(pd.to_datetime(df["date"]))
        raise ValueError("df must contain 'date' as a MultiIndex level or column.")

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged (train_idx, val_idx) integer position arrays for a panel.

        Enforces the purge condition across unique sorted trading dates:
            T_train_purged = { t in T_train | idx(t) <= idx(min(T_val)) - 1 - (h + embargo) }
        ensuring max(train_dates) <= min(val_dates) - holding_period in both
        trading-day index and calendar-day terms.

        Args:
            df: Panel DataFrame containing 'date' in its MultiIndex or columns.
            y: Ignored (present for scikit-learn API compatibility).
            groups: Ignored (present for scikit-learn API compatibility).

        Yields:
            Tuple of (train_indices, val_indices) integer numpy arrays into df.
        """
        if df.empty:
            raise ValueError("Cannot generate splits from an empty DataFrame.")

        row_dates = self._extract_dates(df)
        if row_dates.isna().any():
            raise ValueError("Input DataFrame contains NaT dates.")

        unique_dates = pd.DatetimeIndex(np.sort(row_dates.unique()))
        n_dates = len(unique_dates)
        purge_gap = self.holding_period + self.embargo_days

        if self.min_train_dates is not None:
            first_val_pos = self.min_train_dates + purge_gap
            remaining = n_dates - first_val_pos
            if remaining < self.n_splits:
                raise ValueError(
                    f"Insufficient unique dates ({n_dates}) for min_train_dates={self.min_train_dates}, "
                    f"purge_gap={purge_gap}, and n_splits={self.n_splits}."
                )
            val_blocks = np.array_split(np.arange(first_val_pos, n_dates), self.n_splits)
            date_splits = [
                (np.arange(0, int(block[0]) - purge_gap), block)
                for block in val_blocks
            ]
        else:
            if n_dates < self.n_splits + 1:
                raise ValueError(
                    f"Insufficient unique dates ({n_dates}) for n_splits={self.n_splits}."
                )
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            date_splits = []
            for train_pos, val_pos in tscv.split(unique_dates):
                purged_end = len(train_pos) - purge_gap
                if purged_end <= 0:
                    raise ValueError(
                        f"Purge gap ({purge_gap}) leaves zero training dates in fold; "
                        f"increase dataset span or reduce n_splits/holding_period."
                    )
                date_splits.append((train_pos[:purged_end], val_pos))

        row_date_values = row_dates.to_numpy()
        for train_pos, val_pos in date_splits:
            train_dates = unique_dates[train_pos].to_numpy()
            val_dates = unique_dates[val_pos].to_numpy()

            train_idx = np.flatnonzero(np.isin(row_date_values, train_dates))
            val_idx = np.flatnonzero(np.isin(row_date_values, val_dates))
            yield train_idx, val_idx
