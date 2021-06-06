import pandas as pd

def ema(ohlc: pd.DataFrame,
        period: int,
        previous_ema: float = None) -> float:
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
        previous_ema = sum(ohlc['close'].iloc[-1-period:-1]) / period
    # Calculate new EMA
    mult = 2 / (period + 1)
    new_ema = ohlc['close'].iloc[-1] * mult + previous_ema * (1 - mult)
    return new_ema

def rsi(ohlc: pd.DataFrame,
        period: int) -> float:
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
        if ohlc['close'].iloc[-idn]-ohlc['close'].iloc[-idn-1] > 0:
            upPrices.append(ohlc['close'].iloc[-idn]-ohlc['close'].iloc[-idn-1])
            downPrices.append(0)
        else:
            upPrices.append(0)
            downPrices.append(abs(ohlc['close'].iloc[-idn]-ohlc['close'].iloc[-idn-1]))
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


def rsi_idx(ohlc: pd.DataFrame,
        period: int,
        idx: int = None) -> float:
    """ Calculates Relative Strength Index (RSI) based on close prices over last 'period' days

    Args:
        ohlc: Past OHLC timeseries data
        period: Period of RSI
        idx: index of data frame

    Returns:
        Value of RSI for index idx
    """
    # Begin of Data Frame
    if idx < period:
        rsi = 0
    # calculating up and down moves
    upPrices=[]
    downPrices=[]
    for idn in range(period, 0, -1):
        if ohlc['close'].iloc[idx-idn]-ohlc['close'].iloc[idx-idn-1] > 0:
            upPrices.append(ohlc['close'].iloc[idx-idn]-ohlc['close'].iloc[idx-idn-1])
            downPrices.append(0)
        else:
            upPrices.append(0)
            downPrices.append(abs(ohlc['close'].iloc[idx-idn]-ohlc['close'].iloc[idx-idn-1]))
    # averaging the advances and declines, rolling moving average
    avg_gain = sum(upPrices)/period
    avg_loss = sum(downPrices)/period
    #calculating relative strength
    if avg_loss != 0:
        rs = avg_gain/avg_loss
    else:
        rs = 0
    #calculating RSI
    if rs != -1:
        rsi = 100-(100/(1+rs))
    else: rsi=0
    return rsi

def stochrsi(ohlc: pd.DataFrame,
          period: int) -> float:
    #calculate stochastic rsi
    try:
        stochrsi  = (ohlc['rsi'].iloc[-1] -min(ohlc['rsi'].iloc[-period:])) / (max(ohlc['rsi'].iloc[-period:]) - min(ohlc['rsi'].iloc[-period:]))
    except:
        stochrsi = 0
    return stochrsi

def stochrsi_K(ohlc: pd.DataFrame,
          smoothK: int) -> float:
    #calculate stochastic rsi k smoothed
    try:
        stochrsi_K = ohlc['stochrsi'].iloc[-smoothK:].mean()
    except:
        stochrsi_K = 0
    return stochrsi_K

def stochrsi_D(ohlc: pd.DataFrame,
          smoothD: int) -> float:
    #calculate stochastic rsi k smoothed
    try:
        stochrsi_D = ohlc['stochrsi_K'].iloc[-smoothD:].mean()
    except:
        stochrsi_D = 0
    return stochrsi_D

def stoch_old(ohlc: pd.DataFrame,
          percentageK: int,
          percentageD: int) -> dict:

    #stochastic rsi k value
    try:
        stoch_rsi_k = (ohlc['rsi'].iloc[-1]-min(ohlc['rsi'].iloc[-percentageK:]))/(max(ohlc['rsi'].iloc[-percentageK:])-min(ohlc['rsi'].iloc[-percentageK:]))
    except:
        stoch_rsi_k = 0
    #stochastic rsi d value, moving average
    try:
        stoch_rsi_d = (sum(ohlc['stoch_rsi_k'].iloc[-percentageD + 1:]) + stoch_rsi_k) / percentageD
    except:
        stoch_rsi_d = stoch_rsi_k
    # merge stoch_rsi
    stoch_rsi = {'stoch_rsi_k': stoch_rsi_k, 'stoch_rsi_d': stoch_rsi_d }
    return stoch_rsi