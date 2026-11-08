import pandas as pd
import numpy as np
from data import df

# Construct cumulative total return index per permno (adjusted for splits and dividends)
cum_ret = (1 + df['return'].fillna(0)).groupby(level='permno').cumprod()

# Vectorized forward 5-day return:
# Simulates buying at t+1 close and holding until t+6 close (5 trading days)
buy_cum = cum_ret.groupby(level='permno').shift(-1)
sell_cum = cum_ret.groupby(level='permno').shift(-6)
df['fwd_ret_5d_raw'] = (sell_cum - buy_cum) / buy_cum

def rank_to_quantiles(s, q=3):
    valid = s.dropna()
    if len(valid) < q:
        return pd.Series(np.nan, index=s.index)
    # Using rank(method='first') guarantees unique inputs even if returns tie
    quantiles = pd.qcut(valid.rank(method='first'), q=q, labels=False)
    return quantiles.reindex(s.index)

df['target_quantile'] = df.groupby(level='date')['fwd_ret_5d_raw'].transform(lambda s: rank_to_quantiles(s, q=3))
df['target_decile'] = df['target_quantile']  # Alias for compatibility
df_train = df.dropna(subset=['target_quantile'])

print(df_train[['close', 'return', 'fwd_ret_5d_raw', 'target_quantile']].tail(10))
