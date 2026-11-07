import numpy as np
from data import df

# Construct cumulative total return index per permno (adjusted for splits and dividends)
cum_ret = (1 + df['return'].fillna(0)).groupby(level='permno').cumprod()
df['mom_1m_raw'] = cum_ret.groupby(level='permno').pct_change(21)

def winsorize_series(s):
    return s.clip(lower = s.quantile(0.01), upper = s.quantile(0.99))

df['mom_1m_raw'] = df.groupby(level='date')['mom_1m_raw'].transform(winsorize_series)

def z_score(s):
    return (s-s.mean())/s.std()

df['mom_1m_z'] = df.groupby(level='date')['mom_1m_raw'].transform(z_score)

df = df.dropna(subset=['mom_1m_z'])

print(df[['close', 'mom_1m_raw', 'mom_1m_z']].tail())