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
    new_ema = (ohlc['close'].iloc[-1] - previous_ema) * mult + previous_ema
    return new_ema

def rsi(ohlc: pd.DataFrame) -> float:
    pass

def stoch(ohlc: pd.DataFrame) -> dict:
    pass
