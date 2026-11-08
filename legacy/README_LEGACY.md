# Legacy Academic Baseline (Quarantined)

This directory preserves the original academic course project codebase and its baseline implementation, frozen permanently for historical reference, verification, and performance comparison.

## 1. Overview of Original Submission
The original codebase implemented a prototype machine learning pipeline:
- **Sample Universe:** 5 hand-picked US mega-cap stocks (PERMNOs: 10107 [MSFT], 11404 [JNJ], 11850 [XOM], 14593 [AAPL], 22111 [JPM]).
- **Dataset:** `raw_crsp_data.parquet` (cached from 2015-01-02 to 2023-12-29, 11,320 rows).
- **Feature Set:** Single raw 1-month momentum factor (`mom_1m_z`), computed from a 21-day cumulative total return shift and standardized cross-sectionally.
- **Target:** Forward 5-day return discretized into 3 quantiles.
- **Model:** LightGBM regressor evaluated across 5 walk-forward splits with a trailing 5-day purge.
- **Baseline Performance:**
  - Fold 1 IC: -0.0112
  - Fold 2 IC: +0.0278
  - Fold 3 IC: -0.0214
  - Fold 4 IC: +0.0010
  - Fold 5 IC: -0.0140
  - **Mean Information Coefficient (IC):** -0.0036
  - **IC Standard Deviation:** 0.0173
  - **Information Ratio (ICIR):** -0.2064

## 2. Cryptographic Integrity Manifest
All files in this directory are frozen. Their cryptographic SHA-256 digests at baseline Git commit `5b9537c2945b03ce92d74ed108e9019cf2cd52b5` are:

| File | Size (Bytes) | SHA-256 Digest (64 Hex Characters) |
|---|---|---|
| `data.py` | 930 | `1a345236b7d5641f5bfd43f6abb5eb6e2aed386cd1fd142bfca0bed27ede8902` |
| `feature_eng.py` | 667 | `ead2d6786c05017832d2a2382678b875b6e1b40dd51c5f479d4b960a69c5e12d` |
| `target_def.py` | 1,126 | `c645936092cff8c65760225a7d4307888cbff8552543a1195e0f9a2bf78f2252` |
| `ml_arch.py` | 2,888 | `a4f10d97bc29c3c7bf581c4b1565becee80f1974fc473354e58b1f1db19b81a0` |
| `eval.py` | 964 | `0c071a2de3f94ff2f35eeea69e81e9c16ed8b9f5fc152335b84c7d16d33f84ae` |
| `dummy_run.py` | 2,185 | `efc0a32b3cf51cdebd58bab46d93df9232a86e94f29908b301f1d4bd066cf91a` |
| `raw_crsp_data.parquet` (local) | 283,244 | `6e4c938156588170aafaa0820c851e2014e9db46081bd046f137476bcbf6ff18` |

## 3. Quarantined Status
These scripts are retained solely as a historical baseline. Production quantitative research and development occurs exclusively within the modular `src/` package.
