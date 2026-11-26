"""
Unit tests for WalkForwardModelEngine using standard library unittest.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models import WalkForwardModelEngine
from src.models.walk_forward import WalkForwardModelEngine as DirectEngine
from src.validation import ForwardReturnTargetBuilder, PurgedWalkForwardCV


class TestWalkForwardModelEngine(unittest.TestCase):
    """Test suite for walk-forward LightGBM/sklearn execution and Spearman IC evaluation."""

    @staticmethod
    def _build_synthetic_signal_panel(
        n_days: int = 120,
        n_permnos: int = 30,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create synthetic cross-sectional panel with monotonic signal-to-return relationship."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-02", periods=n_days)
        permnos = [10000 + i for i in range(n_permnos)]

        rows = []
        for p in permnos:
            cum = 1.0
            for d in dates:
                signal = float(rng.normal(0.0, 1.0))
                noise = float(rng.normal(0.0, 0.005))
                # Positive linear/rank relationship between feature and subsequent returns
                ret = 0.004 * signal + noise
                cum *= 1.0 + ret
                rows.append(
                    {
                        "date": d,
                        "permno": p,
                        "alpha_sig_z": signal,
                        "vol_z": float(rng.normal(0.0, 1.0)),
                        "cum_ret": cum,
                    }
                )

        panel = pd.DataFrame(rows).set_index(["date", "permno"]).sort_index()
        # Shift feature forward relative to return? Here signal at t predicts t->t+5 naturally
        # by injecting signal_t into future t+1..t+6 returns:
        # Rebuild cum_ret cleanly so alpha_sig_z at t predicts R_{t+1 -> t+6}:
        sig = panel["alpha_sig_z"]
        lagged_sig = sig.groupby(level="permno").shift(3).fillna(0.0)
        daily_ret = 0.006 * lagged_sig + rng.normal(0.0, 0.004, size=len(panel))
        panel["cum_ret"] = (1.0 + daily_ret).groupby(level="permno").cumprod()

        target_builder = ForwardReturnTargetBuilder(holding_period=5, execution_lag=1)
        panel = target_builder.build_target_panel(panel, holding_period=5, num_quantiles=5)
        return panel

    def test_package_export_identity(self) -> None:
        self.assertIs(WalkForwardModelEngine, DirectEngine)

    def test_walk_forward_lightgbm_fit_predict_and_ic_metrics(self) -> None:
        panel = self._build_synthetic_signal_panel(n_days=110, n_permnos=30, seed=7)

        cv = PurgedWalkForwardCV(n_splits=5, holding_period=5)
        engine = WalkForwardModelEngine(
            cv_splitter=cv,
            feature_cols=["alpha_sig_z", "vol_z"],
            target_col="target_quantile",
        )
        results = engine.fit_predict(panel)

        self.assertEqual(len(engine.fold_models_), 5)
        self.assertEqual(set(results.columns), {"pred", "actual", "fold", "predicted_rank"})
        self.assertEqual(list(results.index.names), ["date", "permno"])
        self.assertTrue((results["predicted_rank"] > 0.0).all())
        self.assertTrue((results["predicted_rank"] <= 1.0).all())

        # Out-of-fold predictions must span all 5 folds without duplicate (date, permno)
        self.assertEqual(set(results["fold"].unique()), {1, 2, 3, 4, 5})
        self.assertFalse(results.index.duplicated().any())

        ic_stats = engine.evaluate_ic(results)
        self.assertIn("daily_ic", ic_stats)
        self.assertIn("mean_ic", ic_stats)
        self.assertIn("std_ic", ic_stats)
        self.assertIn("icir", ic_stats)
        self.assertIn("positive_ic_ratio", ic_stats)
        self.assertIn("fold_metrics", ic_stats)

        self.assertGreater(ic_stats["mean_ic"], 0.15)
        self.assertGreater(ic_stats["icir"], 0.5)
        self.assertGreater(ic_stats["positive_ic_ratio"], 0.7)
        self.assertEqual(len(ic_stats["fold_metrics"]), 5)

        oof = engine.get_out_of_fold_predictions(results)
        self.assertFalse(oof[["pred", "actual", "predicted_rank"]].isna().any().any())

    def test_arbitrary_sklearn_regressor_and_auto_feature_detection(self) -> None:
        panel = self._build_synthetic_signal_panel(n_days=90, n_permnos=25, seed=99)
        # Leave feature_cols=None to verify auto-detection of '*_z' columns
        engine = WalkForwardModelEngine(
            estimator=Ridge(alpha=1.0),
            cv_splitter=PurgedWalkForwardCV(n_splits=3, holding_period=5),
        )
        results = engine.fit_predict(panel)
        self.assertEqual(engine.feature_cols_, ["alpha_sig_z", "vol_z"])
        ic_stats = engine.evaluate_ic(results)
        self.assertGreater(ic_stats["mean_ic"], 0.10)

    def test_zero_variance_ic_protection(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=3)
        permnos = [10001, 10002, 10003, 10004]
        idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
        results_df = pd.DataFrame(
            {
                # Day 0: perfect rank correlation (+1.0)
                # Day 1: constant predictions (zero variance -> NaN)
                # Day 2: constant actuals (zero variance -> NaN)
                "pred": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0],
                "actual": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0],
                "fold": [1] * 12,
            },
            index=idx,
        )

        engine = WalkForwardModelEngine()
        ic_stats = engine.evaluate_ic(results_df)
        daily_ic = ic_stats["daily_ic"]

        self.assertAlmostEqual(daily_ic.loc[dates[0]], 1.0, places=10)
        self.assertTrue(np.isnan(daily_ic.loc[dates[1]]))
        self.assertTrue(np.isnan(daily_ic.loc[dates[2]]))
        self.assertEqual(ic_stats["n_valid_dates"], 1)
        self.assertAlmostEqual(ic_stats["mean_ic"], 1.0, places=10)
        self.assertTrue(np.isnan(ic_stats["std_ic"]))
        self.assertTrue(np.isnan(ic_stats["icir"]))


if __name__ == "__main__":
    unittest.main()
