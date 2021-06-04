import click
import sys
import importlib.util
from pathlib import Path
from datetime import datetime as dt
import os
from zipfile import ZipFile
import shutil
import pandas as pd

from definitions import REPO_ROOT, RAW_DATA_DIR

# Load 'download-kline' module from 'binance-public-data' repository
binance_loader_dir = REPO_ROOT / 'binance-public-data' / 'python'
sys.path.append(str(binance_loader_dir))
spec = importlib.util.spec_from_file_location(
    name='binance_kline',
    location=binance_loader_dir / 'download-kline.py'
)
binance_kline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binance_kline)


@click.command()
@click.option('--interval',
              default='1m',
              help='Candlestick interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, '
                   '2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo',
              type=str)
@click.option('--destination',
              help='Destination folder for final OHLCV csv data.',
              type=str)
def main(interval,
         destination):
    print('Loading binance trading pairs...')
    symbols = binance_kline.get_all_symbols()
    usdt_pairs = [sym for sym in symbols if sym.endswith('USDT')]

    destination_content = os.listdir(destination)
    downloaded_pairs = [
        pair.replace('.csv', '') for pair in destination_content
    ]

    print('Fetching binance ohlcv data...')
    for i, pair in enumerate(usdt_pairs):
        print(f'  ({i+1:03}/{len(usdt_pairs)}) {pair}')
        if pair in downloaded_pairs:
            print('    Pair exists already at destinatoin')
            continue
        temp_path = Path.cwd() / 'temp'
        zipfiles_path = fetch_binance_ohlcv_data(
            symbol=pair,
            interval=interval,
            temp_path=temp_path
        )
        print('    Convert to dataframe...')
        make_ohlcv_dataframe(
            symbol=pair,
            zipfiles_path=zipfiles_path,
            destination_path=Path(destination)
        )
        print('    Remove temp folder...')
        shutil.rmtree(temp_path)


def fetch_binance_ohlcv_data(symbol: str,
                             interval: str,
                             temp_path: Path) -> str:
    start_date = '2000-01-01'
    end_date = str(dt.date(dt.now()))
    binance_kline.download_monthly_klines(
        symbols=[symbol],
        num_symbols=1,
        intervals=[interval],
        years=['2017', '2018', '2019', '2020', '2021'],
        months=list(range(1, 13)),
        start_date=start_date,
        end_date=end_date,
        folder=temp_path,
        checksum=0
    )
    date_range = start_date + "_" + end_date
    zipfiles_path = temp_path / 'data' / 'spot' / 'monthly' / 'klines' / f'{symbol.upper()}' \
                 / f'{interval}' / f'{date_range}'

    return zipfiles_path


def make_ohlcv_dataframe(symbol: str,
                         zipfiles_path: Path,
                         destination_path: Path):
    # Extract zip files
    zip_files = os.listdir(zipfiles_path)
    for zf in zip_files:
        with ZipFile(zipfiles_path / zf, 'r') as z:
            z.extractall(zipfiles_path / 'extracted')

    # Create and save pd.DataFrame
    column_names = [
        'open time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close time',
        'quote asset volume',
        'number of trades',
        'taker buy base asset volume',
        'taker buy quote asset volume',
        'ignore'
    ]
    csv_files = os.listdir(zipfiles_path / 'extracted')
    ohlcv_data = []
    for cf in csv_files:
        ohlcv_data.append(
            pd.read_csv(
                zipfiles_path / 'extracted' / cf,
                names=column_names
            )
        )
    df = pd.concat(ohlcv_data)
    df = df.sort_values(by='open time')
    destination_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination_path / f'{symbol}.csv', index=False)


if __name__ == '__main__':
    main()