"""
Walk-Forward Machine Learning Execution & Information Coefficient Engine.

Orchestrates expanding purged cross-validation over cross-sectional equity panels
using LightGBM or any scikit-learn compatible regressor, tracks daily Spearman
rank Information Coefficients (IC / ICIR), and packages out-of-fold predictions
with cross-sectional percentile ranks for downstream portfolio construction.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.validation.splitters import PurgedWalkForwardCV


class WalkForwardModelEngine:
    """Executes purged walk-forward training, OOF prediction, and Spearman IC evaluation."""

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        cv_splitter: Optional[PurgedWalkForwardCV] = None,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = "target_quantile",
        dropna_features: Optional[bool] = None,
    ) -> None:
        """Initialize the walk-forward ML engine.

        Args:
            estimator: Scikit-learn compatible regressor. Defaults to institutional
                LightGBM regressor (100 trees, lr=0.05, max_depth=4, min_child_samples=20).
            cv_splitter: Cross-validation splitter (default: PurgedWalkForwardCV()).
            feature_cols: Explicit sequence of feature column names. If None, auto-detects
                columns ending with '_z' at fit_predict time.
            target_col: Target column name (default: 'target_quantile').
            dropna_features: Controls dropping rows with NaN feature values before CV.
                If None (default), retains partial-NaN rows for LightGBM (which handles
                missing values natively) and drops feature NaNs for other estimators.
        """
        self.estimator: BaseEstimator = (
            estimator
            if estimator is not None
            else lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_child_samples=20,
                random_state=42,
                verbosity=-1,
            )
        )
        self.cv_splitter: PurgedWalkForwardCV = (
            cv_splitter if cv_splitter is not None else PurgedWalkForwardCV()
        )
        self.feature_cols: Optional[list[str]] = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.target_col = target_col
        self.dropna_features = dropna_features

        self.fold_models_: list[BaseEstimator] = []
        self.feature_cols_: list[str] = []

    @staticmethod
    def _normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df is sorted and MultiIndexed by ['date', 'permno']."""
        if isinstance(df.index, pd.MultiIndex) and {"date", "permno"}.issubset(
            set(df.index.names)
        ):
            panel = df.reset_index()
        elif {"date", "permno"}.issubset(set(df.columns)):
            panel = df.copy()
        else:
            raise ValueError(
                "Input DataFrame must contain ['date', 'permno'] as MultiIndex levels or columns."
            )

        panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
        panel["permno"] = pd.to_numeric(panel["permno"], errors="coerce")
        panel = panel.dropna(subset=["date", "permno"]).copy()
        panel["permno"] = panel["permno"].astype("int64")
        return panel.set_index(["date", "permno"]).sort_index()

    def _resolve_features(self, panel: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in panel.columns]
            if missing:
                raise ValueError(f"Missing feature columns in DataFrame: {missing}")
            return list(self.feature_cols)

        auto_cols = [c for c in panel.columns if c.endswith("_z")]
        if not auto_cols:
            raise ValueError(
                "No '*_z' feature columns found in DataFrame; pass feature_cols explicitly."
            )
        return auto_cols

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run expanding purged walk-forward cross-validation and collect OOF predictions.

        Args:
            df: Panel DataFrame indexed by ['date', 'permno'] (or containing them as columns)
                with feature columns and target_col.

        Returns:
            DataFrame MultiIndexed by ['date', 'permno'] containing columns:
                ['pred', 'actual', 'fold', 'predicted_rank'].
        """
        panel = self._normalize_panel(df)
        if self.target_col not in panel.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        features = self._resolve_features(panel)
        self.feature_cols_ = features

        clean_panel = panel.dropna(subset=[self.target_col]).copy()
        drop_feat_nans = (
            self.dropna_features
            if self.dropna_features is not None
            else not isinstance(self.estimator, lgb.LGBMRegressor)
        )
        if drop_feat_nans:
            clean_panel = clean_panel.dropna(subset=features)
        else:
            clean_panel = clean_panel[clean_panel[features].notna().any(axis=1)]

        if clean_panel.empty:
            raise ValueError("No valid training/validation rows remain after filtering NaNs.")

        self.fold_models_ = []
        fold_outputs: list[pd.DataFrame] = []

        for fold_num, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(clean_panel), start=1
        ):
            train_slice = clean_panel.iloc[train_idx]
            val_slice = clean_panel.iloc[val_idx]

            X_train = train_slice[features]
            y_train = train_slice[self.target_col].astype("float64")
            X_val = val_slice[features]
            y_val = val_slice[self.target_col].astype("float64")

            fold_model = clone(self.estimator)
            fold_model.fit(X_train, y_train)
            preds = np.asarray(fold_model.predict(X_val), dtype="float64")

            self.fold_models_.append(fold_model)
            fold_outputs.append(
                pd.DataFrame(
                    {
                        "pred": preds,
                        "actual": y_val.to_numpy(dtype="float64"),
                        "fold": np.full(len(val_slice), fold_num, dtype="int64"),
                    },
                    index=val_slice.index,
                )
            )

        results_df = pd.concat(fold_outputs).sort_index()
        results_df["predicted_rank"] = (
            results_df.groupby(level="date", sort=False)["pred"]
            .rank(pct=True)
            .astype("float64")
        )
        return results_df

    @staticmethod
    def _compute_daily_spearman_ic(results_df: pd.DataFrame) -> pd.Series:
        """Compute daily cross-sectional Spearman IC vectorially with zero-variance protection."""
        valid = results_df[["pred", "actual"]].dropna()
        if valid.empty:
            return pd.Series(dtype="float64", name="daily_ic")

        r_pred = valid.groupby(level="date", sort=False)["pred"].rank(method="average")
        r_actual = valid.groupby(level="date", sort=False)["actual"].rank(method="average")

        dp = r_pred - r_pred.groupby(level="date", sort=False).transform("mean")
        da = r_actual - r_actual.groupby(level="date", sort=False).transform("mean")

        cov = (dp * da).groupby(level="date", sort=False).sum()
        denom = np.sqrt(
            (dp**2).groupby(level="date", sort=False).sum()
            * (da**2).groupby(level="date", sort=False).sum()
        )
        daily_ic = (cov / denom.where(denom > 1e-12)).sort_index()
        daily_ic.name = "daily_ic"
        return daily_ic

    def evaluate_ic(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Compute daily Spearman rank IC series, fold breakdown, and summary statistics.

        Args:
            results_df: Output DataFrame from fit_predict() containing ['pred', 'actual']
                and optionally 'fold'.

        Returns:
            Dictionary containing:
                - 'daily_ic': pd.Series of daily Spearman IC values indexed by date
                - 'mean_ic': float mean daily IC across valid validation dates
                - 'std_ic': float sample standard deviation (ddof=1) of daily IC
                - 'icir': float Information Ratio (mean_ic / std_ic)
                - 'positive_ic_ratio': float share of valid dates with IC > 0
                - 'n_valid_dates': int count of non-NaN IC dates
                - 'fold_metrics': pd.DataFrame with per-fold IC summary (if 'fold' present)
        """
        if not isinstance(results_df.index, pd.MultiIndex) or "date" not in results_df.index.names:
            results_df = self._normalize_panel(results_df)

        missing = [c for c in ("pred", "actual") if c not in results_df.columns]
        if missing:
            raise ValueError(f"results_df is missing required columns: {missing}")

        daily_ic = self._compute_daily_spearman_ic(results_df)
        valid_ic = daily_ic.dropna()

        if valid_ic.empty:
            summary: dict[str, Any] = {
                "daily_ic": daily_ic,
                "mean_ic": float("nan"),
                "std_ic": float("nan"),
                "icir": float("nan"),
                "positive_ic_ratio": float("nan"),
                "n_valid_dates": 0,
            }
        else:
            mean_ic = float(valid_ic.mean())
            std_ic = float(valid_ic.std(ddof=1)) if len(valid_ic) > 1 else float("nan")
            icir = (
                mean_ic / std_ic
                if np.isfinite(std_ic) and std_ic > 0.0
                else float("nan")
            )
            pos_ratio = float((valid_ic > 0.0).mean())
            summary = {
                "daily_ic": daily_ic,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
                "positive_ic_ratio": pos_ratio,
                "n_valid_dates": int(len(valid_ic)),
            }

        if "fold" in results_df.columns:
            fold_rows = []
            for fold_id, fold_df in results_df.groupby("fold", sort=True):
                fold_daily = self._compute_daily_spearman_ic(fold_df).dropna()
                f_mean = float(fold_daily.mean()) if not fold_daily.empty else float("nan")
                f_std = float(fold_daily.std(ddof=1)) if len(fold_daily) > 1 else float("nan")
                f_icir = (
                    f_mean / f_std if np.isfinite(f_std) and f_std > 0.0 else float("nan")
                )
                f_pos = float((fold_daily > 0.0).mean()) if not fold_daily.empty else float("nan")
                fold_rows.append(
                    {
                        "fold": int(fold_id),
                        "mean_ic": f_mean,
                        "std_ic": f_std,
                        "icir": f_icir,
                        "positive_ic_ratio": f_pos,
                        "n_valid_dates": int(len(fold_daily)),
                    }
                )
            summary["fold_metrics"] = pd.DataFrame(fold_rows)

        return summary

    def get_out_of_fold_predictions(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Return cleaned out-of-fold predictions with cross-sectional percentile ranks.

        Args:
            results_df: Output DataFrame from fit_predict().

        Returns:
            Cleaned copy indexed by ['date', 'permno'] with 'predicted_rank' in (0, 1].
        """
        panel = (
            results_df.copy()
            if isinstance(results_df.index, pd.MultiIndex)
            and list(results_df.index.names) == ["date", "permno"]
            else self._normalize_panel(results_df)
        )
        missing = [c for c in ("pred", "actual") if c not in panel.columns]
        if missing:
            raise ValueError(f"results_df is missing required columns: {missing}")

        cleaned = panel.dropna(subset=["pred", "actual"]).sort_index().copy()
        cleaned["predicted_rank"] = (
            cleaned.groupby(level="date", sort=False)["pred"]
            .rank(pct=True)
            .astype("float64")
        )
        return cleaned
