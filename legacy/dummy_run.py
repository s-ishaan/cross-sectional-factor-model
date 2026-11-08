import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("1. Generating Synthetic Data...")
dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='B')
permnos = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010]

idx = pd.MultiIndex.from_product([dates, permnos], names=['date', 'permno'])
df = pd.DataFrame(index=idx)

# Generate random prices and a random feature
df['open'] = np.random.uniform(10, 100, size=len(df))
df['close'] = df['open'] + np.random.normal(0, 1, size=len(df))
df['random_feature_z'] = np.random.normal(0, 1, size=len(df)) # The Noise Test

print("2. Engineering Target (T+1 to T+6 Forward Return)...")
def calc_fwd_ret(g):
    buy = g['open'].shift(-1)
    sell = g['open'].shift(-6)
    return (sell - buy) / buy

df['fwd_ret_5d'] = df.groupby(level='permno').apply(calc_fwd_ret).reset_index(level=0, drop=True)

# Cross-sectional decile ranking
def rank_deciles(s):
    valid = s.dropna()
    if len(valid) < 10: return pd.Series(np.nan, index=s.index)
    return pd.qcut(valid, 10, labels=False).reindex(s.index)

df['target_decile'] = df.groupby(level='date')['fwd_ret_5d'].transform(rank_deciles)
df = df.dropna()

print("3. Running Walk-Forward CV...")
unique_dates = df.index.get_level_values('date').unique()
tscv = TimeSeriesSplit(n_splits=3)
model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, random_state=42)

for fold, (train_idx, val_idx) in enumerate(tscv.split(unique_dates)):
    train_dates = unique_dates[train_idx]
    val_dates = unique_dates[val_idx]
    
    # The Purge
    purge_cutoff = train_dates[-6]
    train_dates_purged = train_dates[train_dates <= purge_cutoff]
    
    X_train = df.loc[(train_dates_purged, slice(None)), ['random_feature_z']]
    y_train = df.loc[(train_dates_purged, slice(None)), 'target_decile']
    X_val = df.loc[(val_dates, slice(None)), ['random_feature_z']]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print(f"Fold {fold+1} complete. Generated {len(preds)} predictions.")

print("Pipeline Dry Run Successful! No crashes.")