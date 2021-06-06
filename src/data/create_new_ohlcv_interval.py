import os
from pathlib import Path
import click
import pandas as pd

from definitions import STABLECOINS, FIAT

@click.command()
@click.option('--data-dir',
              help='Directory of 1min OHLCV data',
              type=str)
@click.option('--new-interval',
              help='New OHLCV interval',
              type=str)
@click.option('--skip-stable',
              help='Remove stablecoin and fiat pairs',
              is_flag=True)
def main(data_dir, new_interval, skip_stable):
    ohlcv_1m_files = os.listdir(Path(data_dir))
    (Path(data_dir) / new_interval).mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(ohlcv_1m_files):
        print(f'{i+1:03}/{len(ohlcv_1m_files)} {file}')
        if skip_stable:
            quote = file[:-8]
            if quote in STABLECOINS:
                print(f'  {quote} is stablecoin. Skipping.')
                continue
            if quote in FIAT:
                print(f'  {quote} is fiat. Skipping.')
                continue
        df = pd.read_csv(Path(data_dir) / file)
        df['open time'] = pd.to_datetime(df['open time'] / 1e3, unit='s')
        df.index = df['open time']
        df = df[['open time', 'open', 'high', 'low', 'close', 'volume']]
        # Resample to new OHLCV interval
        df_resampled = df.resample(new_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        # Interpolate missing values
        df_resampled = df_resampled.interpolate()
        # Save to drive
        df_resampled.to_csv(Path(data_dir) / new_interval / file)

if __name__ == '__main__':
    main()