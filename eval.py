import alphalens as al
import pandas as pd
import matplotlib.pyplot as plt
from ml_arch import predictions_df, prices_df

# Ensure clean float dtypes so Alphalens forward returns don't degrade to object arrays
prices = prices_df.astype(float)
factor = predictions_df['predicted_rank'].astype(float)

# Determine quantiles dynamically: 10 for broad universe (S&P 500), or 3 for small universe
num_assets = prices.shape[1]
n_quantiles = 10 if num_assets >= 50 else min(3, num_assets)

print(f"Running Alphalens analysis across {num_assets} assets with {n_quantiles} quantiles...")

factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=factor,
    prices=prices,
    periods=(1, 5, 10),     # Evaluate 1-day, 5-day, and 10-day holding periods
    quantiles=n_quantiles,  # Group into quantiles
    filter_zscore=None      # Outliers already handled in feature engineering
)

al.tears.create_full_tear_sheet(factor_data, long_short=True)
plt.show()
