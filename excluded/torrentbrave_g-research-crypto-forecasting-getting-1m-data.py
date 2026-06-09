!curl https://api.binance.com/api/v3/exchangeInfo


import json
import os
import random
import subprocess
import time
from datetime import date, datetime, timedelta
from datetime import date


import requests
import pandas as pd
import numpy as np

API_BASE = 'https://api.binance.com/api/v3/'

LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_asset_volume',
    'number_of_trades',
    'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume',
    'ignore'
]

METADATA = {
    'id': 'torrentbrave/G-research-binance-1m-data-BTC',
    'title': 'G-research-binance-1m-data-BTC',
    'isPrivate': False,
    'licenses': [{'name': 'other'}],
    'keywords': [
        'business',
        'finance',
        'investing',
        'currencies and foreign exchange'
    ],
    'collaborators': [],
    'data': []
}


def write_metadata(n_count):
    """Write the metadata file dynamically so we can include a pair count."""
    METADATA['subtitle'] = f'1 minute candlesticks for all {n_count} cryptocurrency pairs'
    METADATA['description'] = f"""### Introduction\n\nThis is a collection of all 1 minute candlesticks of all cryptocurrency pairs on [Binance.com](https://binance.com). All {n_count} of them are included. Both retrieval and uploading the data is fully automatedâ€”see [this GitHub repo](https://github.com/gosuto-ai/candlestick_retriever).\n\n### Content\n\nFor every trading pair, the following fields from [Binance's official API endpoint for historical candlestick data](https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#klinecandlestick-data) are saved into a Parquet file:\n\n```\n #   Column                        Dtype         \n---  ------                        -----         \n 0   open_time                     datetime64[ns]\n 1   open                          float32       \n 2   high                          float32       \n 3   low                           float32       \n 4   close                         float32       \n 5   volume                        float32       \n 6   quote_asset_volume            float32       \n 7   number_of_trades              uint16        \n 8   taker_buy_base_asset_volume   float32       \n 9   taker_buy_quote_asset_volume  float32       \ndtypes: datetime64[ns](1), float32(8), uint16(1)\n```\n\nThe dataframe is indexed by `open_time` and sorted from oldest to newest. The first row starts at the first timestamp available on the exchange, which is July 2017 for the longest running pairs.\n\nHere are two simple plots based on a single file; one of the opening price with an added indicator (MA50) and one of the volume and number of trades:\n\n![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2234678%2Fb8664e6f26dc84e9a40d5a3d915c9640%2Fdownload.png?generation=1582053879538546&alt=media)\n![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2234678%2Fcd04ed586b08c1576a7b67d163ad9889%2Fdownload-1.png?generation=1582053899082078&alt=media)\n\n### Inspiration\n\nOne obvious use-case for this data could be technical analysis by adding indicators such as moving averages, MACD, RSI, etc. Other approaches could include backtesting trading algorithms or computing arbitrage potential with other exchanges.\n\n### License\n\nThis data is being collected automatically from crypto exchange Binance."""
    with open('compressed/dataset-metadata.json', 'w') as file:
    #with open('dataset-metadata.json', 'w') as file:
        json.dump(METADATA, file, indent=4)

def get_batch(symbol, interval='1m', start_time=0, limit=1000):
    """Use a GET request to retrieve a batch of candlesticks. Process the JSON into a pandas
    dataframe and return it. If not successful, return an empty dataframe.
    """

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    try:
        # timeout should also be given as a parameter to the function
        response = requests.get(f'{API_BASE}klines', params, timeout=30)
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 5 mins...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except requests.exceptions.ConnectionResetError:
        print('Connection reset by peer, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)

    if response.status_code == 200:
        return pd.DataFrame(response.json(), columns=LABELS)
    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([])


def all_candles_to_csv(base, quote, interval='1m'):
    """Collect a list of candlestick batches with all candlesticks of a trading pair,
    concat into a dataframe and write it to CSV.
    """

    # see if there is any data saved on disk already
    try:
        batches = [pd.read_csv(f'data/{base}-{quote}.csv')]
        last_timestamp = batches[-1]['open_time'].max()
    except FileNotFoundError:
        batches = [pd.DataFrame([], columns=LABELS)]
        last_timestamp = 0
    old_lines = len(batches[-1].index)

    # gather all candlesticks available, starting from the last timestamp loaded from disk or 0
    # stop if the timestamp that comes back from the api is the same as the last one
    previous_timestamp = None

    while previous_timestamp != last_timestamp:
        # stop if we reached data from today
        if date.fromtimestamp(last_timestamp / 1000) >= date.today():
            break

        previous_timestamp = last_timestamp

        new_batch = get_batch(
            symbol=base+quote,
            interval=interval,
            start_time=last_timestamp+1
        )

        # requesting candles from the future returns empty
        # also stop in case response code was not 200
        if new_batch.empty:
            break

        last_timestamp = new_batch['open_time'].max()

        # sometimes no new trades took place yet on date.today();
        # in this case the batch is nothing new
        if previous_timestamp == last_timestamp:
            break

        batches.append(new_batch)
        last_datetime = datetime.fromtimestamp(last_timestamp / 1000)

        covering_spaces = 20 * ' '
        print(datetime.now(), base, quote, interval, str(last_datetime)+covering_spaces, end='\r', flush=True)

    # write clean version of csv to parquet
    parquet_name = f'{base}-{quote}.parquet'
    full_path = f'compressed/{parquet_name}'
    df = pd.concat(batches, ignore_index=True)
    df = quick_clean(df)
    write_raw_to_parquet(df, full_path)
    METADATA['data'].append({
        'description': f'All trade history for the pair {base} and {quote} at 1 minute intervals. Counts {df.index.size} records.',
        'name': parquet_name,
        'totalBytes': os.stat(full_path).st_size,
        'columns': []
    })

    # in the case that new data was gathered write it to disk
    if len(batches) > 1:
        df.to_csv(f'data/{base}-{quote}.csv', index=False)
        return len(df.index) - old_lines
    return 0

def set_dtypes(df):
    """
    set datetimeindex and convert all columns in pd.df to their proper dtype
    assumes csv is read raw without modifications; pd.read_csv(csv_filename)"""

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time', drop=True)

    df = df.astype(dtype={
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'close_time': 'datetime64[ms]',
        'quote_asset_volume': 'float64',
        'number_of_trades': 'int64',
        'taker_buy_base_asset_volume': 'float64',
        'taker_buy_quote_asset_volume': 'float64',
        'ignore': 'float64'
    })

    return df


def set_dtypes_compressed(df):
    """Create a `DatetimeIndex` and convert all critical columns in pd.df to a dtype with low
    memory profile. Assumes csv is read raw without modifications; `pd.read_csv(csv_filename)`."""

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time', drop=True)

    df = df.astype(dtype={
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32',
        'number_of_trades': 'uint16',
        'quote_asset_volume': 'float32',
        'taker_buy_base_asset_volume': 'float32',
        'taker_buy_quote_asset_volume': 'float32'
    })

    return df


def assert_integrity(df):
    """make sure no rows have empty cells or duplicate timestamps exist"""

    assert df.isna().all(axis=1).any() == False
    assert df['open_time'].duplicated().any() == False


def quick_clean(df):
    """clean a raw dataframe"""

    # drop dupes
    dupes = df['open_time'].duplicated().sum()
    if dupes > 0:
        df = df[df['open_time'].duplicated() == False]

    # sort by timestamp, oldest first
    df.sort_values(by=['open_time'], ascending=False)

    # just a doublcheck
    assert_integrity(df)

    return df


def write_raw_to_parquet(df, full_path):
    """takes raw df and writes a parquet to disk"""

    # some candlesticks do not span a full minute
    # these points are not reliable and thus filtered
    df = df[~(df['open_time'] - df['close_time'] != -59999)]

    # `close_time` column has become redundant now, as is the column `ignore`
    df = df.drop(['close_time', 'ignore'], axis=1)

    df = set_dtypes_compressed(df)

    # give all pairs the same nice cut-off
    df = df[df.index < str(date.today())]

    df.to_parquet(full_path)


def groom_data(dirname='data'):
    """go through data folder and perform a quick clean on all csv files"""

    for filename in os.listdir(dirname):
        if filename.endswith('.csv'):
            full_path = f'{dirname}/{filename}'
            quick_clean(pd.read_csv(full_path)).to_csv(full_path)


def compress_data(dirname='data'):
    """go through data folder and rewrite csv files to parquets"""

    os.makedirs('compressed', exist_ok=True)
    for filename in os.listdir(dirname):
        if filename.endswith('.csv'):
            full_path = f'{dirname}/{filename}'

            df = pd.read_csv(full_path)

            new_filename = filename.replace('.csv', '.parquet')
            new_full_path = f'compressed/{new_filename}'
            write_raw_to_parquet(df, new_full_path)



all_symbols = pd.DataFrame(requests.get(f'{API_BASE}exchangeInfo').json()['symbols'])


# !curl https://api.binance.com/api/v3/exchangeInfo


# æµ�è§ˆå™¨é‡Œæ‰‹åŠ¨è®¾ç½®ç«¯å�£ä¸ºæœ¬åœ°


dict_ticker = {'Bitcoin Cash':'BCH',
'Binance Coin':'BNB',
'Bitcoin':'BTC',
'EOS.IO':'EOS',
'Ethereum Classic':'ETC',
'Ethereum':'ETH',
'Litecoin':'LTC',
'Monero':'XMR',
'TRON':'TRX',
'Stellar':'XLM',
'Cardano':'ADA',
'IOTA':'IOTA',
'Maker':'MKR',
'Dogecoin':'DOGE'}


for a in dict_ticker:
    quoteAssetsa = all_symbols[all_symbols.baseAsset == dict_ticker[a]].quoteAsset.unique()
    USDquoteAssetsa = [qA for qA in quoteAssetsa if 'USD' in qA]
    print(USDquoteAssetsa)


# version 1: quote
"""
quote = 'BUSD'
all_pairs = [(dict_ticker[a],quote) for a in dict_ticker]
all_pairs
"""


# version 2: quote
"""
quote = 'USDT'
all_pairs = [(dict_ticker[a],quote) for a in dict_ticker]
all_pairs
"""


quote = 'USDT'
target_symbols = {'BTC', 'ETH'}

all_pairs = [(dict_ticker[a], quote) for a in dict_ticker if dict_ticker[a] in target_symbols]


all_pairs


import os
os.makedirs("/root/.kaggle", exist_ok=True)
!cp /kaggle/input/kaggle-json/kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json


# æ–°åŠ å�¡èŠ‚ç‚¹, æ—¶å·®8å°�æ—¶ï¼Œä½†ç”¨çš„æ•°æ�®éƒ½æ˜¯ä¸€æ ·çš„
# å®�é™…çš„å¤§é™†æ—¶é—´æ˜¯å‡Œæ™¨4ç‚¹å¤š
# 2025-06-27 20:17 2017-6


# make sure data folders exist
os.makedirs('data', exist_ok=True)
os.makedirs('compressed', exist_ok=True)

# do a full update on all pairs
n_count = len(all_pairs)
for n, pair in enumerate(all_pairs, 1):
    base, quote = pair
    new_lines = all_candles_to_csv(base=base, quote=quote)
    if new_lines > 0:
        print(f'{datetime.now()} {n}/{n_count} Wrote {new_lines} new lines to file for {base}-{quote}')
    else:
        print(f'{datetime.now()} {n}/{n_count} Already up to date with {base}-{quote}')


import os


n_count=1


from datetime import date, timedelta
import subprocess

def upload_to_kaggle(n_count):
    """åˆ›å»ºæˆ–æ›´æ–°æ•°æ�®é›†ï¼ˆæ ¹æ�® metadata æ–‡ä»¶æ˜¯å�¦å­˜åœ¨åˆ¤æ–­ï¼‰"""
    metadata_path = 'compressed/dataset-metadata.json'
    yesterday = date.today() - timedelta(days=1)
    message = f"full update of all {n_count} pairs up to {yesterday}"

    # å†™å…¥ metadataï¼ˆæ€»ä¼šå†™å…¥æœ€æ–° subtitle / description / filesï¼‰
    write_metadata(n_count)

    if not os.path.exists('.kaggle_created.flag'):
        # ç¬¬ä¸€æ¬¡ä¸Šä¼ 
        print("ğŸš€ Creating new Kaggle dataset...")
        subprocess.run(['kaggle', 'datasets', 'create', '-p', 'compressed'])
        with open('.kaggle_created.flag', 'w') as f:
            f.write('created')
    else:
        # å��ç»­æ›´æ–°
        print("ğŸ“¦ Uploading new version...")
        subprocess.run(['kaggle', 'datasets', 'version', 
                        '-p', 'compressed', 
                        '-m', message
                        ])

    # å�¯é€‰ï¼šä¸Šä¼ å®Œå��åˆ é™¤ metadata
    if os.path.exists(metadata_path):
        os.remove(metadata_path)


write_metadata(n_count)


upload_to_kaggle(n_count=n_count)


write_metadata(n_count)


%cd 


!kaggle datasets create


# clean the data folder and upload a new version of the dataset to kaggle
try:
    os.remove('compressed/.DS_Store')
except FileNotFoundError:
    pass
write_metadata(n_count)

yesterday = date.today() - timedelta(days=1)
subprocess.run(['kaggle', 'datasets', 'version', '-p', 'compressed/', '-m', f'full update of all {n_count} pairs up to {str(yesterday)}'])
os.remove('compressed/dataset-metadata.json')




