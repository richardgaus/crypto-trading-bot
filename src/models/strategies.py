from abc import ABC
import pandas as pd

class Strategy(ABC):

    def apply(self,
              ohlcv_timeseries: pd.DataFrame) -> 'StrategyResults':
        pass

class StrategyResults(ABC):

    def __init__(self,
                 asset_name: str,
                 ohlcv_timeseries: pd.DataFrame):
        self.asset_name = asset_name
        self.ohlcv = ohlcv_timeseries
        self.trades = []

    def plot(self):
        pass

    def set_pnl(self,
                pnl: float):
        self._pnl = pnl

    @property
    def pnl(self):
        return self._pnl

    def add_trade(self,
                  entry_time,
                  entry_price: float,
                  take_profit: float,
                  stop_loss: float,
                  exit_time,
                  win: bool):
        _trade = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'exit_time': exit_time,
            'win': win
        }
        self.trades.append(_trade)

class RSIStoch200EMA(Strategy):

    def apply(self,
              ohlcv_timeseries: pd.DataFrame) -> 'RSIStoch200EMAResults':
        pass

class RSIStoch200EMAResults(StrategyResults):

    def __init__(self,
                 asset_name: str,
                 ohlcv_timeseries: pd.DataFrame):
        super().__init__(asset_name, ohlcv_timeseries)
        self.ema200 = None
        self.rsi = None
        self.stoch = None
        self.patterns = []

    def plot(self):
        pass

    def add_trade_and_signal(self,
                             entry_time,
                             entry_price: float,
                             take_profit: float,
                             stop_loss: float,
                             exit_time,
                             win: bool,
                             trend_line_price: tuple,
                             trend_line_rsi: tuple,
                             stoch_cross_time):
        self.add_trade(
            entry_time, entry_price, take_profit, stop_loss, exit_time, win
        )
        _signal = {
            'trend_line_price': trend_line_price,
            'trend_line_rsi': trend_line_rsi,
            'stoch_cross_time': stoch_cross_time
        }
        self.patterns.append(_signal)

    def set_indicators(self,
                       ema200: pd.Series,
                       rsi: pd.Series,
                       stoch: pd.DataFrame):
        self.ema200 = ema200
        self.rsi = rsi
        self.stoch = stoch
