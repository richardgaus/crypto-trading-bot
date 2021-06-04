import os
from pathlib import Path
import click
import pandas as pd

@click.command()
@click.option('--data-dir',
              help='Directory of 1min OHLCV data',
              type=str)
@click.option('--new-interval',
              help='New OHLCV interval',
              type=str)
def main(data_dir, new_interval):
    ohlcv_1m_files = os.listdir(Path(data_dir) / '1min')
    (Path(data_dir) / new_interval).mkdir(parents=True, exist_ok=True)
    for file in ohlcv_1m_files:
        df = pd.read_csv(Path(data_dir) / '1min' / file)
        df['open time'] = pd.to_datetime(df['open time'] / 1e3, unit='s')
        df.index = df['open time']
        df = df[['open time', 'open', 'high', 'low', 'close', 'volume']]
        df_resampled = df.resample(new_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        df_resampled.to_csv(Path(data_dir) / new_interval / file)

if __name__ == '__main__':
    main()