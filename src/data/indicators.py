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
    # Begin of Data Frame
    if len(ohlc) < period+1:
        return np.nan
    # calculating up and down moves
    #change = []
    #for idn in range(period, 0, -1):
    #    change.append(ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1])


    #up = max(change, 0).mean()
    #down = -min(change, 0).mean()

    #rsi = 0
    #if down == 0:
    #    rsi = 100
    #else:
    #    if up == 0:
    #        rsi = 0
    #    else:
    #        rsi = 100 - (100 / (1 + up / down))


    upPrices=[]
    downPrices=[]
    for idn in range(period, 0, -1): #period-1, -1, -1 -> 13....0 // period+1, +1, -1 -> 15....2
        if ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1] > 0:
            upPrices.append(ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1])
            downPrices.append(0)
        else:
            upPrices.append(0)
            downPrices.append(abs(ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1]))
    # averaging the advances and declines, expoentially weighted moving average



    up = 0
    down = 0
    alpha = 1 / period
    for idn in range(period-1, -1, -1):
        up = alpha * (1-alpha)**(period-idn-1) * upPrices[idn] + up
        down = alpha * (1 - alpha)**(period-idn-1) * downPrices[idn] + down

    #up = sum(upPrices)/period
    #down = sum(downPrices)/period
    #calculating relative strength
    rsi = 0
    if down == 0:
       rsi = 100
    else:
       if up == 0:
           rsi = 0
       else:
           rsi = 100 - (100 / (1 + up / down))

    #if avg_loss != 0:
    #    rs = avg_gain/avg_loss
    #else:
    #    rs = 50
    # calculating RSI
    #rsi = 100-(100/(1+rs))
    return rsi


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
    #try:
    stoch_k = ohlc['stoch'].iloc[-smoothK:].mean()
    #except:
    #   stoch_k = 0
    return stoch_k

def stoch_d(ohlc: pd.DataFrame,
          smoothD: int) -> float:
    # There are fewer data points than period length -> return None
    if len(ohlc) < smoothD:
        return None
    #calculate stochastic rsi k smoothed
    #try:
    stoch_d = ohlc['stoch_k'].iloc[-smoothD:].mean()
    #except:
    #    stoch_d = 0
    return stoch_d
