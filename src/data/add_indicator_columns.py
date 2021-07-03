import os
from pathlib import Path
import click
import pandas as pd
from functools import partial

from src.data.indicators import ema, rsi, stochRSI

def add_indicator_cols(dataframe: pd.DataFrame,
                       indicator_dict: dict):
    df_out = dataframe.copy()
    for key, item in indicator_dict.items():
        if isinstance(key, tuple) or isinstance(key, list):
            arrays = item(dataframe)
            for col, arr in zip(key, arrays):
                df_out[col] = arr
        else:
            df_out[key] = item(dataframe)

    return df_out

@click.command()
@click.option('--data-dir',
              help='Directory OHLCV data',
              type=str)
def main(data_dir):
    ohlcv_files = os.listdir(Path(data_dir))
    for i, file in enumerate(ohlcv_files):
        print(f'{i+1:03}/{len(ohlcv_files)} {file}')
        df = pd.read_csv(Path(data_dir) / file)
        new_df = add_indicator_cols(
            dataframe=df,
            indicator_dict={
                'ema': partial(ema, period=200),
                'rsi': partial(rsi, period=14),
                ('stoch_k', 'stoch_d'): partial(
                    stochRSI, period=14, smoothK=3, smoothD=3
                )
            }
        )
        new_df.to_csv(Path(data_dir) / file, index=False)

if __name__ == '__main__':
    main()