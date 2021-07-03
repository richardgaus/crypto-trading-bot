from abc import ABC

import numpy as np
import pandas as pd
import matplotlib as mpl
import mplfinance as mpf

class Strategy(ABC):

    def __init__(self,
                 max_number_open_trades: int,
                 *args, **kwargs):
        self.max_number_open_trades = max_number_open_trades

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

    def evaluation(self):
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

    def __init__(self,
                 max_number_open_trades: int,
                 min_period_to_last_low: int,
                 max_period_to_last_low: int,
                 min_absolute_slope_rsi: float,
                 swing_low_margin: int):
        super().__init__(max_number_open_trades)
        self.min_period_to_last_low = min_period_to_last_low
        self.max_period_to_last_low = max_period_to_last_low
        self.min_absolute_slope_rsi = min_absolute_slope_rsi
        self.swing_low_margin = swing_low_margin

    def apply(self,
              ohlcv_timeseries: pd.DataFrame,
              asset_name: str=None) -> 'RSIStoch200EMAResults':
        running_bullish_divergences = []
        running_bearish_divergences = []
        open_trades_exit_times = []
        true_divergence = None
        wait_for_stoch_cross = False
        current_pattern = {}
        current_trade = {}

        results = RSIStoch200EMAResults(
            asset_name=asset_name,
            ohlcv_timeseries=ohlcv_timeseries
        )

        for idx in range(1, len(ohlcv_timeseries) - 1):
            row = ohlcv_timeseries.iloc[idx]
            if np.isnan(row['ema']) or np.isnan(row['rsi']) or \
                    np.isnan(row['stoch_k']) or np.isnan(row['stoch_d']):
                continue
            if np.isnan(ohlcv_timeseries.iloc[idx - 1]['rsi']):
                continue

            # Remove closed trades
            open_trades_exit_times = [et for et in open_trades_exit_times if et > idx]

            for potential_div in running_bullish_divergences:
                # Update threshold
                threshold = potential_div['t1_rsi'] + (idx - potential_div['t1_idx']) * \
                            potential_div['slope']
                potential_div['threshold'] = threshold
                # Scan for lower RSI value
                if row['rsi'] < potential_div['threshold']:
                    # Update slope
                    delta_y = row['rsi'] - potential_div['t1_rsi']
                    delta_x = idx - potential_div['t1_idx']
                    potential_div['slope'] = delta_y / delta_x
                    # Is this a local low on RSI? -> potential bullish divergence spotted
                    if (row['rsi'] < ohlcv_timeseries.iloc[idx - 1]['rsi']) and (
                            row['rsi'] < ohlcv_timeseries.iloc[idx + 1]['rsi']):
                        if self.min_period_to_last_low <= idx - potential_div[
                            't1_idx'] <= self.max_period_to_last_low:
                            past_row = ohlcv_timeseries.loc[potential_div['t1']]
                            # Check if there is a higher low on price
                            if not wait_for_stoch_cross and \
                                    len(open_trades_exit_times) < self.max_number_open_trades and \
                                    min(past_row['close'], past_row['open']) < min(row['close'],
                                                                                   row['open']):
                                # Divergence found!
                                wait_for_stoch_cross = True
                                potential_div['t2'] = row.name
                                potential_div['t2_idx'] = idx
                                potential_div['t2_rsi'] = row['rsi']
                                true_divergence = potential_div

            for potential_div in running_bearish_divergences:
                # Update threshold
                threshold = potential_div['t1_rsi'] + (idx - potential_div['t1_idx']) * \
                            potential_div['slope']
                potential_div['threshold'] = threshold
                # Scan for higher RSI value
                if row['rsi'] > potential_div['threshold']:
                    # Update slope
                    delta_y = row['rsi'] - potential_div['t1_rsi']
                    delta_x = idx - potential_div['t1_idx']
                    potential_div['slope'] = delta_y / delta_x
                    # Is this a local high on RSI? -> potential bearish divergence spotted
                    if (row['rsi'] > ohlcv_timeseries.iloc[idx - 1]['rsi']) and (
                            row['rsi'] > ohlcv_timeseries.iloc[idx + 1]['rsi']):
                        if self.min_period_to_last_low <= idx - potential_div[
                            't1_idx'] <= self.max_period_to_last_low:
                            past_row = ohlcv_timeseries.loc[potential_div['t1']]
                            # Check if there is a higher high on price
                            if not wait_for_stoch_cross and \
                                    len(open_trades_exit_times) < self.max_number_open_trades and \
                                    max(past_row['close'], past_row['open']) > max(row['close'],
                                                                                   row['open']):
                                # Divergence found!
                                wait_for_stoch_cross = True
                                potential_div['t2'] = row.name
                                potential_div['t2_idx'] = idx
                                potential_div['t2_rsi'] = row['rsi']
                                true_divergence = potential_div

            # Remove all positions with threshold < 0 (bullish) or > 1 (bearish)
            running_bullish_divergences = [
                d for d in running_bullish_divergences \
                if d['threshold'] > 0 and idx - d['t1_idx'] <= self.max_period_to_last_low
            ]
            running_bearish_divergences = [
                d for d in running_bearish_divergences \
                if d['threshold'] < 100 and idx - d['t1_idx'] <= self.max_period_to_last_low
            ]

            # Save new lows / highs on RSI
            if (row['rsi'] < ohlcv_timeseries.iloc[idx - 1]['rsi']) and (
                    row['rsi'] < ohlcv_timeseries.iloc[idx + 1]['rsi']):
                new_potential_div = {}
                new_potential_div['t1'] = row.name
                new_potential_div['t1_idx'] = idx
                new_potential_div['t1_rsi'] = row['rsi']
                new_potential_div['slope'] = -self.min_absolute_slope_rsi
                new_potential_div['threshold'] = row['rsi']
                running_bullish_divergences.append(new_potential_div)
            if (row['rsi'] > ohlcv_timeseries.iloc[idx - 1]['rsi']) and (
                    row['rsi'] > ohlcv_timeseries.iloc[idx + 1]['rsi']):
                new_potential_div = {}
                new_potential_div['t1'] = row.name
                new_potential_div['t1_idx'] = idx
                new_potential_div['t1_rsi'] = row['rsi']
                new_potential_div['slope'] = self.min_absolute_slope_rsi
                new_potential_div['threshold'] = row['rsi']
                running_bearish_divergences.append(new_potential_div)

            if wait_for_stoch_cross:
                # Check for cross of stochastic RSI
                old_stoch_delta = \
                    ohlcv_timeseries.iloc[idx - 1]['stoch_k'] - ohlcv_timeseries.iloc[idx - 1]['stoch_d']
                new_stoch_delta = \
                    ohlcv_timeseries.iloc[idx]['stoch_k'] - ohlcv_timeseries.iloc[idx]['stoch_d']
                if old_stoch_delta * new_stoch_delta < 0:
                    # Cross found! Place trade.
                    current_pattern = {
                        'divergence_timeframe': (true_divergence['t1'], true_divergence['t2'])
                    }
                    current_trade = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'take_profit': None,
                        'stop_loss': None,
                        'exit_time': None,
                        'win': None
                    }
                    # Price below EMA 200 -> go short; above EMA 200 -> go long
                    go_long = True
                    if current_trade['entry_price'] < ohlcv_timeseries.iloc[idx]['ema']:
                        go_long = False
                    # Look for nearest swing low/high
                    idx_swing_low = idx
                    while True:
                        surrounding_idx = [
                            idx_swing_low + j for j in
                            range(-self.swing_low_margin, self.swing_low_margin + 1) if j != 0
                        ]
                        valid_idx = set(surrounding_idx).intersection(range(1, idx + 1))
                        if go_long:  # go long
                            if ohlcv_timeseries.iloc[idx_swing_low]['low'] < current_trade['entry_price'] \
                                    and ohlcv_timeseries.iloc[idx_swing_low]['low'] < \
                                    ohlcv_timeseries.iloc[list(valid_idx)]['low'].min():
                                # Swing low found. Place stop loss and take profit.
                                current_trade['stop_loss'] = 0.999 * ohlcv_timeseries.iloc[idx_swing_low][
                                    'low']
                                current_trade['take_profit'] = current_trade['entry_price'] \
                                                               + 2 * (current_trade['entry_price'] -
                                                                      current_trade['stop_loss'])
                                break
                        else:  # go short
                            if ohlcv_timeseries.iloc[idx_swing_low]['high'] > current_trade['entry_price'] \
                                    and ohlcv_timeseries.iloc[idx_swing_low]['high'] > \
                                    ohlcv_timeseries.iloc[list(valid_idx)]['high'].max():
                                # Swing high found. Place stop loss and take profit.
                                current_trade['stop_loss'] = 1.001 * ohlcv_timeseries.iloc[idx_swing_low][
                                    'high']
                                current_trade['take_profit'] = current_trade['entry_price'] \
                                                               + 2 * (current_trade['entry_price'] -
                                                                      current_trade['stop_loss'])
                                break
                        idx_swing_low = idx_swing_low - 1
                    # Find trade exit
                    for idx_future in range(idx, len(ohlcv_timeseries) - 1):
                        # Is trade long or short?
                        long = True
                        if current_trade['stop_loss'] > current_trade['take_profit']:
                            long = False
                        if (long and current_trade['stop_loss'] > ohlcv_timeseries.iloc[idx_future][
                            'low']) or \
                                (not long and current_trade['stop_loss'] <
                                 ohlcv_timeseries.iloc[idx_future]['high']):
                            # Stopp loss target hit
                            current_trade['exit_time'] = ohlcv_timeseries.iloc[idx_future].name
                            current_trade['win'] = False
                            open_trades_exit_times.append(idx_future)
                            break
                        if (long and current_trade['take_profit'] < ohlcv_timeseries.iloc[idx_future][
                            'high']) or \
                                (not long and current_trade['take_profit'] >
                                 ohlcv_timeseries.iloc[idx_future]['low']):
                            # Take profit target hit
                            current_trade['exit_time'] = ohlcv_timeseries.iloc[idx_future].name
                            current_trade['win'] = True
                            open_trades_exit_times.append(idx_future)
                            break
                        if idx_future == len(ohlcv_timeseries) - 2:
                            current_trade['exit_time'] = ohlcv_timeseries.iloc[idx_future].name
                            current_trade['win'] = False
                            open_trades_exit_times.append(idx_future)

                    results.add_trade_and_signal(**current_trade, **current_pattern)
                    wait_for_stoch_cross = False

        return results

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
            if pattern['divergence_timeframe'][0] > end_time or \
                trade['exit_time'] < start_time:
                continue
            # Draw divergence
            div_rsi = (
                self.ohlcv.loc[pattern['divergence_timeframe'][0], 'rsi'],
                self.ohlcv.loc[pattern['divergence_timeframe'][1], 'rsi']
            )
            # Bullish or bearish?
            fun = min
            if div_rsi[0] < div_rsi[1]:
                fun = max
            div_prices = (
                fun(
                    self.ohlcv.loc[pattern['divergence_timeframe'][0], 'close'],
                    self.ohlcv.loc[pattern['divergence_timeframe'][0], 'open']
                ), fun(
                    self.ohlcv.loc[pattern['divergence_timeframe'][1], 'close'],
                    self.ohlcv.loc[pattern['divergence_timeframe'][1], 'open']
                )
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

    def evaluation(self,
                   start_time: pd.Timestamp,
                   end_time: pd.Timestamp):
        data = self.ohlcv[start_time:end_time]
        #print(self.ohlcv['low'])
        buy_hold = (data['low'].iloc[-1] - data['low'].iloc[1])/data['low'].iloc[1]
        days = end_time - start_time
        total_trades = 0
        winners = 0
        losses = 0
        win_percentage = 0
        max_losses = 0
        max_wins = 0
        max_losses_tmp = 0
        max_wins_tmp = 0
        gain = 0
        for i, (trade, pattern) in enumerate(zip(self.trades, self.patterns)):
            total_trades = total_trades + 1
            if trade['win']:
                winners = winners + 1
                if (trade['take_profit'] - trade['entry_price']) > 0:
                    gain = gain + (trade['take_profit'] - trade['entry_price'])/trade['entry_price']
                else:
                    gain = gain + (trade['entry_price'] - trade['take_profit']) / trade['entry_price']
                max_wins_tmp = max_wins_tmp + 1
                max_losses_tmp = 0
            else:
                losses = losses + 1
                if (trade['entry_price'] - trade['stop_loss']) < 0:
                    gain = gain + (trade['entry_price'] - trade['stop_loss'])/trade['entry_price']
                else:
                    gain = gain + (trade['stop_loss'] - trade['entry_price']) / trade['entry_price']
                max_wins_tmp = 0
                max_losses_tmp = max_losses_tmp + 1

            max_wins = max(max_wins_tmp, max_wins)
            max_losses = max(max_losses_tmp, max_losses)
            #print(trade['win'])

        if total_trades > 0:
            win_percentage = winners/total_trades
        else:
            win_percentage = 0

        print(' Buy&Hold', buy_hold,'\n','Days:',days.days,'\n','Total Trades:', total_trades,'\n','Winners:', winners,'\n','Losses:', losses,'\n','Win Percentage:', win_percentage,'\n','Max Wins:', max_wins,'\n','Max Losses:', max_losses,'\n','Overall Gain:', gain,'\n',)
