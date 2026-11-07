import wrds
import pandas as pd
import os

data_path = 'raw_crsp_data.parquet'

if os.path.exists(data_path):
    print("Loading data from local Parquet file...")
    df = pd.read_parquet(data_path)
else:
    print("Connecting to WRDS...")
    db = wrds.Connection()

    sql_query = """
        SELECT 
            date, 
            permno, 
            prc as close, 
            ret as return, 
            vol as volume, 
            shrout as shares_outstanding
        FROM 
            crsp.dsf
        WHERE 
            date >= '2015-01-01' AND date <= '2024-01-01'
            AND permno IN (14593, 10107, 11404, 22111, 11850) 
    """

    print("Executing SQL query on CRSP...")
    df_raw = db.raw_sql(sql_query, date_cols=['date'])

    df = df_raw.set_index(['date', 'permno']).sort_index()

    df['close'] = df['close'].abs()

    print("Saving data to Parquet...")
    df.to_parquet(data_path)

print(df.head())