import pandas as pd
import numpy as np

def ema(ohlc: pd.DataFrame,
        period: int,
        column: str='close')  -> float:
    """ Calculates exponential moving average (EMA) based on close prices.

    Args:
        ohlc: Past OHLC timeseries data
        period: Period of EMA
        columns: price type

    Returns:
        EMA
    """
    ema = ohlc[column].ewm(span=period, adjust=False).mean()

    return ema

def rsi(ohlc: pd.DataFrame,
        period: int,
        column: str='close') -> float:
    """ Calculates Relative Strength Index (RSI) based on close prices over last 'period' days

    Args:
        ohlc: Past OHLC timeseries data
        period: Period of RSI
        columns: price type

    Returns:
        Value of RSI for index idx
    """
    delta = ohlc[column].diff()

    up = delta.copy()
    up[up < 0] = 0
    up = pd.Series.ewm(up, alpha=1/period).mean()

    down = delta.copy()
    down[down > 0] = 0
    down *= -1
    down = pd.Series.ewm(down, alpha=1/period).mean()
    up = up
    down = down
    rsi = np.where(down == 0, 100, np.where(up == 0, 0, 100 - (100 / (1 + up / down))))
    return rsi



def stochRSI(ohlc: pd.DataFrame, period: int,  smoothK: int, smoothD: int, column: str='close'):
    """ Calculates Stochastic RSI based last 'period' days

      Args:
          ohlc: Past OHLC timeseries data
          period: Period of RSI
          smoothK: smoothing factor K index
          smoothD: smoothing factor D index
          columns: price type

      Returns:
          Value of stochRSI
      """
    rsi1 = rsi(ohlc, period, column)
    rsi2 = pd.DataFrame(rsi1)
    stochrsi  = (rsi2 - rsi2.rolling(period).min()) / (rsi2.rolling(period).max() - rsi2.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()
    return stochrsi_K.to_numpy(), stochrsi_D.to_numpy()
