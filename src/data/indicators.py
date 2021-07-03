import pandas as pd
import numpy as np

def ema(ohlc: pd.DataFrame,
        period: int,
        previous_ema: float = None,
        column: str='close') -> float:
    """ Calculates exponential moving average (EMA) based on close prices.

    Args:
        ohlc: Past OHLC timeseries data
        period: Period of EMA
        previous_ema: Previous value of EMA

    Returns:
        New value of EMA
    """
    # There are fewer data points than period length -> return None
    if len(ohlc) < period:
        return np.nan
    # No previous EMA given -> begin with SMA
    if previous_ema is None:
        previous_ema = sum(ohlc[column].iloc[-1-period:-1]) / period
    # Calculate new EMA
    mult = 2 / (period + 1)
    new_ema = ohlc[column].iloc[-1] * mult + previous_ema * (1 - mult)
    return new_ema

def rsi(ohlc: pd.DataFrame,
        period: int,
        column: str='close') -> float:
    """ Calculates Relative Strength Index (RSI) based on close prices over last 'period' days

    Args:
        ohlc: Past OHLC timeseries data
        period: Period of RSI
        idx: index of data frame

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

###bad
def stochastic(data, k_window, d_window, window):
    # input to function is one column from df
    # containing closing price or whatever value we want to extract K and D from

    min_val = data.rolling(window=window, center=False).min()
    max_val = data.rolling(window=window, center=False).max()

    #stoch = ((data - min_val) / (max_val - min_val)) * 100

    K = stoch.rolling(window=k_window, center=False).mean()
    # K = stoch

    D = K.rolling(window=d_window, center=False).mean()

    return K, D

def stoch(ohlc: pd.DataFrame,
          period: int) -> float:
    #calculate stochastic rsi
    # There are fewer data points than period length -> return None
    if len(ohlc) < period:
        return np.nan
    #try:
    stoch  = (ohlc['rsi'].iloc[-1] -min(ohlc['rsi'].iloc[-period:])) / (max(ohlc['rsi'].iloc[-period:]) - min(ohlc['rsi'].iloc[-period:]))
    #except:
       # stoch = 0
    return stoch

def stoch_k(ohlc: pd.DataFrame,
          smoothK: int) -> float:
    # There are fewer data points than period length -> return None
    if len(ohlc) < smoothK:
        return np.nan
    #calculate stochastic rsi k smoothed
    stoch_k = ohlc['stoch'].iloc[-smoothK:].mean()
    return stoch_k

def stoch_d(ohlc: pd.DataFrame,
          smoothD: int) -> float:
    # There are fewer data points than period length -> return None
    if len(ohlc) < smoothD:
        return None
    #calculate stochastic rsi k smoothed
    stoch_d = ohlc['stoch_k'].iloc[-smoothD:].mean()
    return stoch_d
