import pandas as pd

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
        return None
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
        return None
    # calculating up and down moves
    upPrices=[]
    downPrices=[]
    for idn in range(period, 0, -1):
        if ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1] > 0:
            upPrices.append(ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1])
            downPrices.append(0)
        else:
            upPrices.append(0)
            downPrices.append(abs(ohlc[column].iloc[-idn]-ohlc[column].iloc[-idn-1]))
    # averaging the advances and declines, rolling moving average
    avg_gain = sum(upPrices)/period
    avg_loss = sum(downPrices)/period
    #calculating relative strength
    if avg_loss != 0:
        rs = avg_gain/avg_loss
    else:
        rs = 50
    #calculating RSI
    rsi = 100-(100/(1+rs))
    return rsi


def stoch(ohlc: pd.DataFrame,
          period: int) -> float:
    #calculate stochastic rsi
    try:
        stoch  = (ohlc['rsi'].iloc[-1] -min(ohlc['rsi'].iloc[-period:])) / (max(ohlc['rsi'].iloc[-period:]) - min(ohlc['rsi'].iloc[-period:]))
    except:
        stoch = 0
    return stoch

def stoch_k(ohlc: pd.DataFrame,
          smoothK: int) -> float:
    #calculate stochastic rsi k smoothed
    try:
        stoch_k = ohlc['stoch'].iloc[-smoothK:].mean()
    except:
        stoch_k = 0
    return stoch_k

def stoch_d(ohlc: pd.DataFrame,
          smoothD: int) -> float:
    #calculate stochastic rsi k smoothed
    try:
        stoch_d = ohlc['stoch_k'].iloc[-smoothD:].mean()
    except:
        stoch_d = 0
    return stoch_d
