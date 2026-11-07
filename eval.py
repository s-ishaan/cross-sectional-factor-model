import alphalens as al
import pandas as pd
from ml_arch import predictions_df, prices_df

factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=predictions_df['predicted_rank'],
    prices=prices_df,
    periods=(1, 5, 10), # Evaluate 1-day, 5-day, and 10-day holding periods
    quantiles=3,        # 3 quantiles for this sample universe (set to 10 for S&P 500)
    filter_zscore=None  # We already handled outliers in Phase 2
)

al.tears.create_full_tear_sheet(factor_data, long_short=True)
