import click
import pandas as pd
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import cryptowatch as cw

from definitions import RAW_DATA_DIR, PROCESSED_DATA_DIR
from private import coinmarketcap_apikey

def get_cryptocurrency_table(apikey,
                             start=1,
                             limit=400,
                             convert='USD') -> pd.DataFrame:
    """
    Args:
        apikey: API key for CoinMarketCap API
        start: Table position of the first cryptocurrency to show
        limit: Table position of the last cryptocurrency to show
        convert: Currency unit for volume, etc.

    Returns:
        data_df: Dataframe with active cryptocurrencies, sorted by market cap
    """

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': start,
        'limit': limit,
        'convert': convert
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': apikey,
    }

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
        return

    return pd.DataFrame(data['data'])

@click.command()
@click.option('--cmc-table-length',
              default=500,
              help='Length of cryptocurrency table to load from '
                   'coinmarketcap.com',
              type=int)
@click.option('--candlestick-timeframe',
              default='1d',
              help='Timeframe of candlesticks.',
              type=str)
def main(cmc_table_length:int,
         candlestick_timeframe:str):

    # Load table of top cc_table_length cryptocurrencies from coinmarketcap.com
    crypto_table = get_cryptocurrency_table(
        apikey=coinmarketcap_apikey,
        start=1,
        limit=cmc_table_length,
        convert='USD'
    )
    crypto_table.to_csv(RAW_DATA_DIR / 'cryptocurrency_df.csv', index=False)
    print(f'Downloaded and saved top {cmc_table_length} cryptocurrency table '
          f'from coinmarketcap.com')

    # Create lists of non-stablecoin cryptocurrencies, stablecoins, and
    # exchanges from which data should preferredly be fetched
    cryptocurrencies = []
    stablecoins = []
    for row in crypto_table.iterrows():
        if 'stablecoin' in row[1]['tags']:
            stablecoins.append(row[1]['symbol'])
        else:
            cryptocurrencies.append(row[1]['symbol'])
    cryptocurrencies = [c.lower() for c in cryptocurrencies]
    stablecoins = [s.lower() for s in stablecoins]

    # We prefer USD quote since it has the longest time series in
    # some cases. After that, prefer stablecoins in volume order.
    preferred_quotes = ['usd'] + stablecoins

    # Kraken is most preferred since it is the oldest exchange and
    # it may be assumed that it has timeseries data reaching back
    # in time the farthest. The subsequent exchanges are the ones
    # ranked highest by CoinMarketCap. This selection is to ensure
    # best data quality.
    preferred_exchanges = [
        'kraken',
        'binance',
        'coinbase-pro',
        'bitfinex',
        'huobi',
        'uniswap-v2',
        'poloniex'
    ]

    # Load data about all trading pairs listed on cryptowat.ch and convert to
    # pd.DataFrame for convenience
    all_markets = cw.markets.list().markets
    all_markets_dic = {
        'id': [],
        'exchange': [],
        'pair': []
    }
    for market in all_markets:
        all_markets_dic['id'].append(market.id)
        all_markets_dic['exchange'].append(market.exchange)
        all_markets_dic['pair'].append(market.pair)
    all_markets_df = pd.DataFrame(all_markets_dic)

    # Search through trading pairs and select largest cryptocurrencies with USD
    # or stablecoins quotes from preferred exchanges. Save results in
    # chosen_pairs_df
    results = {
        'id': [],
        'exchange': [],
        'base': [],
        'quote': []
    }
    for cc in cryptocurrencies:
        next_cc = False
        cc_markets = all_markets_df.loc[all_markets_df['pair'].str.startswith(cc)]
        for exchange in preferred_exchanges:
            if next_cc:
                break
            exch_markets = cc_markets[cc_markets['exchange'] == exchange]
            for quote in preferred_quotes:
                wanted_pair = exch_markets.loc[exch_markets['pair'].str[len(cc):] == quote]
                if len(wanted_pair) > 0:
                    results['id'].append(wanted_pair.iloc[0]['id'])
                    results['exchange'].append(exchange)
                    results['base'].append(cc)
                    results['quote'].append(quote)
                    next_cc = True
                    break
        if next_cc:
            continue
        # Base-quote pair not found in preferred exchanges -> look in all others
        wanted_pair = cc_markets.loc[cc_markets['pair'].str[len(cc):] == quote]
        for quote in preferred_quotes:
            wanted_pair = exch_markets.loc[exch_markets['pair'].str[len(cc):] == quote]
            if len(wanted_pair) > 0:
                results['id'].append(wanted_pair.iloc[0]['id'])
                results['exchange'].append(wanted_pair.iloc[0]['exchange'])
                results['base'].append(cc)
                results['quote'].append(quote)
    chosen_pairs_df = pd.DataFrame(results)
    chosen_pairs_df.to_csv(RAW_DATA_DIR / 'chosen_trading_pairs.csv', index=False)
    print(f'Selected trading pairs and saved as dataframe')

    # Load historical candlestick data of chosen pairs
    print('Downloading candlestick data...')
    candlestick_dic = {}
    for i, row in enumerate(chosen_pairs_df.iterrows()):
        _id, exchange, base, quote = (
            row[1]['id'], row[1]['exchange'], row[1]['base'], row[1]['quote']
        )
        try:
            candles = cw.markets.get(
                f'{exchange}:{base}{quote}',
                ohlc=True,
                periods=[candlestick_timeframe]
            )
            candles_df = pd.DataFrame(
                getattr(candles, f'of_{candlestick_timeframe}'),
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'volume_quote'
                ]
            )
            candlestick_dic[f'{exchange}_{base}_{quote}'] = candles_df
            print(f'  {i + 1:03}/{len(chosen_pairs_df)} {exchange}:{base}{quote}')
        except Exception as e:
            print(e)

    for key, item in


if __name__ == '__main__':
    main()