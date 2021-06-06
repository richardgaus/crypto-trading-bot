from abc import ABC
import pandas as pd
import matplotlib as mpl
import mplfinance as mpf

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

    def plot(self,
             start_time: pd.Timestamp,
             end_time: pd.Timestamp):
        data = self.ohlcv[start_time:end_time]
        fig, axes = mpl.pyplot.subplots(
            4, 1,
            figsize=(15, 12),
            gridspec_kw={'height_ratios': [3, 1, 1, 1], 'hspace': 0},
            sharex=True
        )
        # OHLC candlesticks
        mpf.plot(
            data,
            ax=axes[0],
            volume=axes[1],
            style='yahoo',
            type='candle',
            show_nontrading=True,
            tight_layout=True
        )
        axes[0].grid(axis='x', ls='--')
        axes[1].grid(axis='y', ls='--')
        # EMA
        axes[0].plot(data.index, data['ema'], color='dodgerblue', zorder=-10)
        # RSI
        axes[2].plot(data.index, data['rsi'])
        axes[2].set_ylabel('RSI')
        axes[2].grid(axis='y', ls='--')
        # Stochastic RSI
        axes[3].plot(data.index, data['stoch_k'], color='dodgerblue')
        axes[3].plot(data.index, data['stoch_d'], color='darkorange')
        axes[3].set_ylabel('Stoch')
        axes[3].grid(axis='y', ls='--')
        # Format and rotate datetime labels
        axes[3].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d.%m.%y, %H:%M'))
        axes[3].tick_params(axis='x', rotation=45)

        # Draw trades and patterns
        for i, (trade, pattern) in enumerate(zip(self.trades, self.patterns)):
            # Draw divergence
            div_prices = (
                self.ohlcv.loc[pattern['divergence_timeframe'][0], 'close'],
                self.ohlcv.loc[pattern['divergence_timeframe'][1], 'close']
            )
            div_rsi = (
                self.ohlcv.loc[pattern['divergence_timeframe'][0], 'rsi'],
                self.ohlcv.loc[pattern['divergence_timeframe'][1], 'rsi']
            )

            axes[0].plot(  # Divergence line on price plot
                (mpl.dates.date2num(pattern['divergence_timeframe'][0]),
                 mpl.dates.date2num(pattern['divergence_timeframe'][1])),
                (div_prices[0], div_prices[1]),
                color='black'
            )
            axes[2].plot(  # Divergence line on RSI plot
                (mpl.dates.date2num(pattern['divergence_timeframe'][0]),
                 mpl.dates.date2num(pattern['divergence_timeframe'][1])),
                (div_rsi[0], div_rsi[1]),
                color='black'
            )

            # Draw trade rectangle
            axes[0].plot( # Entry price
                (trade['entry_time'], trade['exit_time']),
                (trade['entry_price'], trade['entry_price']),
                color='black'
            )
            axes[0].plot( # Take profit
                (trade['entry_time'], trade['exit_time']),
                (trade['take_profit'], trade['take_profit']),
                color='green'
            )
            axes[0].plot( # Stop loss
                (trade['entry_time'], trade['exit_time']),
                (trade['stop_loss'], trade['stop_loss']),
                color='red'
            )
            axes[0].add_patch(  # Background rectangle
                mpl.patches.Rectangle(
                    xy=(trade['entry_time'], trade['stop_loss']),
                    width=trade['exit_time'] - trade['entry_time'],
                    height=trade['take_profit'] - trade['stop_loss'],
                    color='green' if trade['win'] else 'red',
                    alpha=0.1,
                    zorder=10
                )
            )

    def add_trade_and_signal(self,
                             entry_time: pd.Timestamp,
                             entry_price: float,
                             take_profit: float,
                             stop_loss: float,
                             exit_time: pd.Timestamp,
                             win: bool,
                             divergence_timeframe: tuple):
        self.add_trade(
            entry_time, entry_price, take_profit, stop_loss, exit_time, win
        )
        _signal = {
            'divergence_timeframe': divergence_timeframe
        }
        self.patterns.append(_signal)

    def set_indicators(self,
                       ema200: pd.Series,
                       rsi: pd.Series,
                       stoch: pd.DataFrame):
        self.ema200 = ema200
        self.rsi = rsi
        self.stoch = stoch
