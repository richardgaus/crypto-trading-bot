Crypto Trading Bot
==============================

Test different trading strategies based on technical indicators on historical 
cryptocurrency OHLCV data.

Getting started
===============

1. Download 1 minute OHLCV data for all USDT pairs from binance. For this we use
   the Binance Public Data repository (https://github.com/binance/binance-public-data).

   Run ``python src/data/make_binance_ohlcv_dataset.py`` to download the data. 
   Note: The complete 1 minute OHLCV dataset has size >10GB and will take a
   significant amount of time to fully download. Use the following options:

   ```
   --interval: Candlestick interval. Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo
   --destination: Destination folder for final OHLCV csv data.
   ```
   We recommend to download the full 1 minute candlestick data and convert it later
   to different intervals.
   

2. Convert 1 minute OHLCV data to desired interval.
   
   Run ``create_new_ohlcv_interval.py`` with the following options:
   ```
   --data-dir: Directory of 1min OHLCV data
   --new-interval: New OHLCV interval (e.g. 30min)
   ```