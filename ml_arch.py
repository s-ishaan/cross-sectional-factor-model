import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

# Ensure feature engineering and target definition are run and available on df_train
import feature_eng
from target_def import df_train

features = ['mom_1m_z']
target = 'target_quantile' if 'target_quantile' in df_train.columns else 'target_decile'

# 1. Fix: Use sort_index() instead of deprecated/removed .sort()
df_train = df_train.sort_index()

# 2. Fix: Adjust LightGBM parameters for tree splitting (min_child_samples=5 for small sample)
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_child_samples=5,
    random_state=42,
    verbosity=-1
)

tscv = TimeSeriesSplit(n_splits=5)
unique_dates = df_train.index.get_level_values('date').unique()

cv_scores = []
all_predictions = []

print("Starting Walk-Forward Cross Validation...")

for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates)):
    train_dates = unique_dates[train_date_idx]
    val_dates = unique_dates[val_date_idx]

    # 3. Fix: Purge the last 5 trading days to prevent 5-day forward return leakage into validation
    train_dates_purged = train_dates[:-5]

    X_train = df_train.loc[pd.IndexSlice[train_dates_purged, :], features]
    y_train = df_train.loc[pd.IndexSlice[train_dates_purged, :], target]
    
    X_val = df_train.loc[pd.IndexSlice[val_dates, :], features]
    y_val = df_train.loc[pd.IndexSlice[val_dates, :], target]

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_results = pd.DataFrame({'pred': predictions, 'actual': y_val.values}, index=y_val.index)
    
    # 4. Fix: Safe daily Information Coefficient (handles zero-variance days without crashing)
    def calc_daily_ic(group):
        if group['pred'].nunique() > 1 and group['actual'].nunique() > 1:
            return group['pred'].corr(group['actual'], method='spearman')
        return np.nan

    daily_ic = val_results.groupby(level='date').apply(calc_daily_ic)
    ic = daily_ic.mean()
    
    # 5. Fix: Append fold IC to cv_scores
    cv_scores.append(ic)
    val_results['fold'] = fold + 1
    all_predictions.append(val_results)
    
    print(f"Fold {fold+1} Information Coefficient (IC): {ic:.4f}")

# Cross-validation summary metrics
mean_ic = np.nanmean(cv_scores)
std_ic = np.nanstd(cv_scores)
icir = mean_ic / std_ic if std_ic > 0 else np.nan

print("\n--- Walk-Forward CV Results ---")
print(f"Mean IC: {mean_ic:.4f}")
print(f"IC Std:  {std_ic:.4f}")
print(f"ICIR:    {icir:.4f}")

# Compile out-of-fold predictions and prices for downstream evaluation (e.g. Alphalens in eval.py)
predictions_df = pd.concat(all_predictions)
predictions_df['predicted_rank'] = predictions_df.groupby(level='date')['pred'].rank(pct=True)
prices_df = df_train['close'].unstack('permno')


# Cross-validation summary metrics
mean_ic = np.nanmean(cv_scores)
std_ic = np.nanstd(cv_scores)
icir = mean_ic / std_ic if std_ic > 0 else np.nan

print("\n--- Walk-Forward CV Results ---")
print(f"Mean IC: {mean_ic:.4f}")
print(f"IC Std:  {std_ic:.4f}")
print(f"ICIR:    {icir:.4f}")

# Compile out-of-fold predictions and prices for downstream evaluation (e.g. Alphalens in eval.py)
predictions_df = pd.concat(all_predictions)
predictions_df['predicted_rank'] = predictions_df.groupby(level='date')['pred'].rank(pct=True)
prices_df = df_train['close'].unstack('permno')