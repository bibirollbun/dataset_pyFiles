!pip install icecream --no-index --find-links=file:///kaggle/input/icecream/ 
!pip install airportsdata --no-index --find-links=file:///kaggle/input/airportsdata/ 
!pip install timezonefinder --no-index --find-links=file:///kaggle/input/timezonefinder/ 
!pip install polars --no-index --find-links=file:///kaggle/input/polars/ 
!pip install xgboost --no-index --find-links=file:///kaggle/input/xgboost/ 


from icecream import ic
import sys
import os
import pickle
import numpy as np
import polars as pl
import pandas as pd
from tqdm.auto import tqdm
import json
import glob
import logging
import time
import math
import polars.selectors as cs
from collections import OrderedDict
from itertools import chain
from IPython.display import display


import airportsdata
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo


logger = logging.getLogger('aeroclub')
#handler = logging.StreamHandler()
from rich.logging import RichHandler
handler = RichHandler(rich_tracebacks=True)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False   # 如果不想传到 root
logger.info('start!')
ic.configureOutput(prefix='', outputFunction=logger.info)


class FLAGS:
  root = '../input/aeroclub-recsys-2025'
  # tree model here can be 'xgb' or 'lgb' 
  # not used here as I will present 4 xgb models only which coul get similar score offline and online
  tm = 'xgb'
  # objective here can be ndcg,map,pariwse for xgb and lambdarank,xendcg for lgb
  obj = 'ndcg'
  # task could be ranking or classification
  task = 'ranking'
  # this is a bit hack as I created 5 folds and only use fold 4 wich is train on day 1-103 valid on day 104-166
  folds = 5
  fold = 4
  
  cat_method = 'count'
  # remove cast means all cat cols to be treated as numer cols 
  remove_cats = True
  reserve_cats = False
  # weather to use train/test for stats or only use train
  stats_all = True
  trees = 1000
  seed = 42
  
  # if false only use original csv existed cols as feats
  add_feats = True
  
  # weather to use history of selected data 
  # notice adding this could improve a lot on LB/PB but it is added after the game finished
  # and code copy from https://www.kaggle.com/code/mikhailgolubchik/sm-xgboost-single
  # also the feat generate using more time likely about 20-30 mintues and more memory needed about 400-500g
  history_avg = True
  # history_avg = False

  # model_dir = '../input/aeroclub-recsys-2025-model1'
  model_dir = '../input/aeroclub-recsys-2025-model2'
  
  external_dir = '../input/aeroclub-recsys-2025-external'
  out_dir = '../working'
    
  # wether use json files, not affect much
  use_ext = True
  # use_ext = False
  
  # online = False means offline train/valid mode, online = True means train on all train data for submission
  # for better LB/PB you need to set online = True, but if set online = True local valid is overly optimistic as we valid on data which also in train
  # online = False
  online = True
  
  # fast = True means for debug only which will run pipline using 0.01 ratio data and train model using 100 trees only
  # fast = True
  fast = False
  
  # n_models = 0 means not limit using all 4 models, n_models=1 means the best single model only with objective rank:ndcg
  # n_models = 1
  n_models = 0
  
  # mode = 'train' means train + eval + test this need CPU MEM more then 200-300g without his_avg
  # mode = 'infer'/'test' means only load from pretrained model and do infer online
  # mode = 'train'
  mode = 'infer'
  
  device = 'gpu'


def in_notebook():
  try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
      return False
  except Exception as e:
    return False
  return True

if not in_notebook():
  if len(sys.argv) > 1:
    FLAGS.fast = bool(int(sys.argv[1]))
    FLAGS.online = bool(int(sys.argv[2]))
    FLAGS.use_ext = bool(int(sys.argv[3]))
    FLAGS.history_avg = bool(int(sys.argv[4]))
  else:
    if 'fast' in os.environ:
      FLAGS.fast = bool(int(os.environ['fast']))
    if 'online' in os.environ:
      FLAGS.online = bool(int(os.environ['online']))
    if 'use_ext' in os.environ:
      FLAGS.use_ext = bool(int(os.environ['use_ext']))
    if 'history_avg' in os.environ:
      FLAGS.history_avg = bool(int(os.environ['history_avg']))


# online + not use_ext + not use history_avg about 250G memory needed after preprocess and neeed 360G before xgb train
FLAGS.out_dir = f'{FLAGS.out_dir}/fast{int(FLAGS.fast)}-online{int(FLAGS.online)}-use_ext{int(FLAGS.use_ext)}-history_avg{int(FLAGS.history_avg)}'
ic(FLAGS.fast)
ic(FLAGS.online)
ic(FLAGS.use_ext)
ic(FLAGS.history_avg)
ic(FLAGS.out_dir)
os.system(f'mkdir -p {FLAGS.out_dir}')


params_xgb = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@3',
    'max_depth': 12,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 100,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'seed': 42,
}

# this is what I used for lgb but it is fine to only train and ensemble xgb models
params_lgb = {
      'objective': 'lambdarank',
      'eval_metric': ['ndcg@3'],
      'ndcg_eval_at': [3], 
      'eval_at': [3],  
      'boosting_type': 'gbdt',
      'num_leaves': 63,
      'max_depth': 12,
      'min_data_in_leaf': 50,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'bagging_freq': 5,
      'lambda_l1': 0.1,
      'lambda_l2': 100,
      'learning_rate': 0.05,
      'n_estimators': 1000,
      'seed': 42,
}


if FLAGS.fast:
  FLAGS.trees = 100


COLS_TO_COMPARE = [
  "legs0_departureAt", 
  "legs0_arrivalAt", 
  "legs1_departureAt",
  "legs1_arrivalAt", 
  "legs0_segments0_flightNumber",
  "legs1_segments0_flightNumber"
]

IGNORE_COLS = [
  'Id', 
  'ranker_id',
  'selected',
  'fold',
  'uid',
  'day',
  'group_day',
  'sample_weight',
  'requestDate', #DateTime
  'bySelf', #train only 1 unique value, test half 0 half 1
  'pricingInfo_passengerCount', # nunique==1
  'legs0_departureAt',  # original time will be converted
  'legs0_arrivalAt',
  'legs1_departureAt',
  'legs1_arrivalAt',
  'legs0_segments3_baggage_count',
  'legs1_segments3_baggage_count',
  'legs0_segments3_baggage_weight',
  'legs1_segments3_baggage_weight',
  'requestReturnDate',
  'requestDepartureDate',
  'legs0_segments3_baggageAllowance_weightMeasurementType',
  'legs0_segments3_cabinClass',
  'legs1_segments3_baggageAllowance_quantity',
  'legs1_segments3_baggageAllowance_weightMeasurementType',
  'legs1_segments3_cabinClass',
  'legs1_segments3_seatsAvailable',
  'legs1_seg3_dep_offset', 
  'legs0_departureAirport',
  'legs1_departureAirport',
  'legs1_seg3_arr_offset',
  'isGlobal',
  'flight_hash',
]

rank_order = {
    'totalPrice': 'asc',
    'flight_duration_total': 'asc',
    'book_lead_time_hours': 'desc',
    'flight_duration_travel_ratio': 'asc',
    'seg_legs_all_count': 'asc',
    'avg_cabin_legs_all': 'desc',
    'avg_baggage_count_legs_all': 'desc',
    'avg_baggage_weight_legs_all': 'desc',
    'direct_price_per_km': 'asc',
}

source_cols = [
        'time_legs0_departureAt_hour',
        'time_legs1_departureAt_hour',
        'time_legs0_arrivalAt_hour',
        'time_legs1_arrivalAt_hour',
        'rank_totalPrice',
        'rank_flight_duration_total',
        'avg_cabin_legs_all',
        'avg_baggage_count_legs_all',
        'avg_baggage_weight_legs_all',
        'direct_price_per_km',
        'miniRules1_statusInfos',
        'miniRules0_statusInfos',
]


def load_external_data(n_files=0):
  datas = []
  json_files = glob.glob(f'{FLAGS.root}/raw/*.json')
  
  if FLAGS.fast:
    ic('fast mode just use 10 json files')
    json_files = json_files[:10]
  
  if n_files:
    json_files = json_files[:n_files]

  for json_file in tqdm(json_files, desc='json_files'):
    with open(json_file) as fh:
      data = json.load(fh)
      datas.append(data)
  
  l = []
  for data in tqdm(datas, desc='datas'):
    m = {
      'ranker_id': data['ranker_id']
    }
    routeData = data['routeData']
    m.update({
      'requestDepartureDate': routeData.get('requestDepartureDate', None),
      'requestReturnDate': routeData.get('requestReturnDate', None),
    })
    personalData = data['personalData']
    m['hasAssistant'] = personalData.get('hasAssistant', None)
    m['isGlobal'] = personalData.get('isGlobal', None)
    m['age'] = None
    if 'yearOfBirth' in personalData:
      try:
        m['age'] = 2024 - personalData['yearOfBirth']
      except Exception as e:
        m['age'] = None
    l.append(m)

  ic('to df_ext')
  df_ext = pl.DataFrame(l)
  ic(df_ext['requestReturnDate'].n_unique())
  return df_ext


def create_days(df):
  if df["requestDate"].dtype != pl.Datetime:
    df = df.with_columns(
        [pl.col("requestDate").str.to_datetime().alias("requestDate")])

  min_date = df["requestDate"].min()
  max_date = df["requestDate"].max()

  ic(f"Date range: {min_date} to {max_date}")

  df = df.with_columns([
      # 计算相对天数（从1开始）
      ((pl.col("requestDate") - min_date).dt.total_days() + 1
      ).cast(pl.Int32).alias("day")
  ])

  max_day = df["day"].max()
  min_day = df["day"].min()
  ic(f"Day range: {min_day} to {max_day}")

  group_day_mapping = (df.group_by("ranker_id", maintain_order=True).agg(
      pl.mean("day").alias("group_day")))

  df = df.join(group_day_mapping, on="ranker_id", how="left")
  return df


def get_bool_cols(df: pl.DataFrame) -> list[str]:
  return df.select(cs.boolean()).columns

def get_cat_cols(df: pl.DataFrame) -> list[str]:
  cat_cols = df.select(cs.string() | cs.categorical()).columns
  return cat_cols

def get_numer_cols(df: pl.DataFrame) -> list[str]:
  num_cols = df.select(cs.numeric() | cs.boolean()).columns
  return num_cols

def bool2int(df):
  return df.with_columns([pl.col(c).cast(pl.Int8) for c in get_bool_cols(df)])


def icl(lst, n=5):
  if not ic.enabled:
    return

  import inspect

  frame = inspect.currentframe().f_back
  vars_dict = frame.f_locals.items()
  # ic(vars_dict)

  var_names = [name for name, value in vars_dict if value is lst]
  var_name = var_names[0] if var_names else "unknown"

  if isinstance(lst, list):
    if len(lst) > n * 2:
      logger.info(f'{var_name} first {n}: {lst[:n]}')
      logger.info(f'{var_name} last {n}: {lst[-n:]}')
    else:
      logger.info(f'{var_name}: {lst}')
    logger.info(f'len({var_name}): {len(lst)}')
  else:
    logger.info(f'{var_name}: {lst}')


def timeit(info=''):

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      logger.info(f'{info} ---------------- {func.__name__} start')
      start = time.time()
      result = func(*args, **kwargs)
      end = time.time()
      logger.info(f'{info} ################ {func.__name__} elapsed: {end - start:.4f} seconds')
      return result

    return wrapper

  return decorator

import functools
def monitor_feats(info='', out_index=0):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
      original_columns = set(getattr(df, "columns", []))
      ret = func(df, *args, **kwargs)

      if isinstance(ret, (tuple, list)):
        if out_index >= len(ret):
          return ret
        new_df = ret[out_index]
      else:
        new_df = ret

      cols = getattr(new_df, "columns", None)
      if cols is None:
        return ret

      new_cols = [col for col in cols if col not in original_columns]
      logger.info(f"{info} {func.__name__} added:")
      icl(new_cols, 10)

      return ret
    return wrapper
  return decorator


def time_feats(info=''):
  def decorator(func):
    decorated_func = monitor_feats(info)(func)
    decorated_func = timeit(info)(decorated_func)
    return decorated_func
  return decorator


def dur_to_min(col: pl.Expr) -> pl.Expr:
  # extract day（ '3.05:10:00' -> 3）
  days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 24 * 60

  # extract time part
  time_str = pl.when(col.str.contains(r"^\d+\.")) \
                .then(col.str.replace(r"^\d+\.", "")) \
                .otherwise(col)

  hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
  minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)

  return (days + hours + minutes).fill_null(0)


@timeit()
def durs_to_unit(df):
  exprs = []

  for leg in (0, 1):
    col = f"legs{leg}_duration"
    assert col in df.columns
    exprs.append((dur_to_min(pl.col(col)) / 60))

    for s in range(4):
      col = f"legs{leg}_segments{s}_duration"
      assert col in df.columns
      exprs.append((dur_to_min(pl.col(col)) / 60))

  df = df.with_columns(exprs)
  return df


def load_df(use_ext=True):
  df_train = pl.read_parquet(f'{FLAGS.root}/train.parquet').drop('__index_level_0__')    
  df_test = pl.read_parquet(f'{FLAGS.root}/test.parquet').drop('__index_level_0__')
  df_test = df_test.with_columns(pl.lit(-1, dtype=pl.Int64).alias('selected'))
  df = pl.concat([df_train, df_test], how='vertical')
  df = create_days(df)
  
  df = df.with_columns(
    pl.col('profileId').alias('uid')
  )
  
  if use_ext:
    if os.path.exists(FLAGS.external_dir):
      logger.info('load external data from pre dumped external.parquet')
      df_ext = pl.read_parquet(f'{FLAGS.external_dir}/external.parquet')
    else:
      logger.info('load external data from json files')
      df_ext = load_external_data()
    
    ic(df_ext['requestReturnDate'].n_unique())
    display(df_ext)
    df = df.join(df_ext, on='ranker_id', how='left')
    assert 'age' in df.columns

  df = df.with_columns(
    pl.col('profileId').cast(pl.Utf8)
  )
  
  cat_cols = [
    'profileId',
    'companyID', 
    'corporateTariffCode', 
    'nationality',
  ]
    
  df = df.with_columns(
    pl.col(col).cast(pl.Utf8) for col in cat_cols 
  )
 
  df = df.with_columns([
      pl.concat_str([
          pl.col(c).cast(str).fill_null("NULL") 
          for c in COLS_TO_COMPARE
      ]).alias("flight_hash")
  ])

  df = bool2int(df)
  df = durs_to_unit(df)
  return df


def get_valid(df):
  df_valid = df.filter(pl.col('fold') <= FLAGS.fold)
  return df_valid

def get_train(df):
  if not FLAGS.online:
    df_train = df.filter(pl.col('fold') > FLAGS.fold)
  else:
    df_train = df
    
  return df_train

def get_test(df):
  return df.filter(pl.col('selected') == -1)

def get_nontest(df):
  # Keep only non-test data
  return df.filter(pl.col('selected') != -1)

def get_train_valid(df):
  # leave out test
  df = df.filter(pl.col('selected') != -1)

  df_train = get_train(df)
  df_valid = get_valid(df)

  return df_train, df_valid


def set_fold(df):
  df_train = df.filter(pl.col('selected') != -1)
  df_test = df.filter(pl.col('selected') == -1)
  # Group by ranker_id and find the latest requestDate for each
  # Original code that sorts by request date
  ranker_dates = (df_train
            .group_by('ranker_id', maintain_order=True)
            .agg(pl.col('requestDate').max().alias('last_request'))
            .sort('last_request'))
  
  # Modified code that preserves original ranker_id order
  unique_rankers = ranker_dates.select('ranker_id').unique(maintain_order=True)

  # Total number of unique ranker_ids
  n_rankers = unique_rankers.height
  
  # Initialize fold column with the last fold number
  unique_rankers = unique_rankers.with_columns(pl.lit(FLAGS.folds).alias('fold'))
  
  # Calculate how many ranker_ids for each fold (10% per fold)
  fold_size = int(n_rankers * 0.1)
  ic(fold_size)
  
  # Update fold values for the newest (FLAGS.folds - 1) groups
  for i in range(FLAGS.folds):
    # Get ranker_ids for current fold (10% of data)
    start_idx = n_rankers - (i + 1) * fold_size
    end_idx = n_rankers - i * fold_size
    len_ = end_idx - start_idx
    # Get the list of ranker_ids for this fold
    fold_rankers = unique_rankers.slice(start_idx, len_).get_column('ranker_id')
    
    # Update the fold value for these ranker_ids
    unique_rankers = unique_rankers.with_columns(
      pl.when(pl.col('ranker_id').is_in(fold_rankers))
      .then(i)
      .otherwise(pl.col('fold'))
      .alias('fold')
    )
  
  # Join fold assignments back to the training data
  df_train = df_train.join(
    unique_rankers.select('ranker_id', 'fold'),
    on='ranker_id'
  )

  df_train = df_train.with_columns(pl.col('fold').cast(pl.Int32))
  ic(len(df_train))
  df_test = df_test.with_columns(pl.lit(-1, dtype=pl.Int32).alias('fold'))
  df = pl.concat([df_train, df_test], how='vertical')

  ic(df.group_by('fold').agg(pl.len()).sort('fold'))
  
  df_valid = get_valid(df_train)
  ic(df_valid['day'].min(), df_valid['day'].max(), df_valid['day'].max() - df_valid['day'].min())
  
  return df


def filter(df, ratio=0.01, seed=42):
  unique_ids = df.select("ranker_id").unique()
  keep_ids = np.random.default_rng(seed).choice(
    unique_ids["ranker_id"].to_list(),
    size=int(len(unique_ids) * ratio),  
    replace=False
  )
  df = df.filter(pl.col("ranker_id").is_in(keep_ids))
  return df


@timeit()
def smart_fillnull(df: pl.DataFrame, numer_cols: list[str], cat_cols: list[str]) -> pl.DataFrame:
  zero_flags = df.select([
      (pl.col(col) == 0).any().alias(col)
      for col in numer_cols
  ])

  fill_map = {
      col: -1 if zero_flags[col][0] else 0
      for col in numer_cols
  }

  exprs = [
      pl.col(col).fill_null(fill_map[col]) for col in numer_cols
  ] + [
      pl.col(col).fill_null("missing") for col in cat_cols
  ]

  return df.with_columns(exprs)

def promote_dtype(dtypes):  
  unique_types = set(dtypes)
  is_float = lambda dt: pl.datatypes.is_float_dtype(dt)
  is_int = lambda dt: pl.datatypes.is_integer_dtype(dt)

  # int + float
  if any(is_float(dt) for dt in unique_types) and any(is_int(dt) for dt in unique_types):
    max_float_bits = max(dt.bit_width for dt in unique_types if is_float(dt))
    return pl.Float64 if max_float_bits > 32 else pl.Float32

  # all float
  if all(is_float(dt) for dt in unique_types):
    max_bits = max(dt.bit_width for dt in unique_types)
    return pl.Float64 if max_bits > 32 else pl.Float32

  # all int
  if all(is_int(dt) for dt in unique_types):
    max_bits = max(dt.bit_width for dt in unique_types)
    return {8: pl.Int8, 16: pl.Int16, 32: pl.Int32, 64: pl.Int64}[max_bits]

  # fallback
  return list(unique_types)[0]


def align_and_concat(dfs, verbose=True):
  from builtins import set

  if not dfs:
    raise ValueError("DataFrame empty")

  common_cols = set.intersection(*(set(df.columns) for df in dfs))

  for col in sorted(common_cols):
    dtypes = [df[col].dtype for df in dfs]
    if len(set(dtypes)) > 1:
      if verbose:
        print(f"[col type not same] {col}")
        for i, dt in enumerate(dtypes):
          print(f"  DF[{i}] dtype: {dt}")
      target_type = promote_dtype(dtypes)
      if verbose:
        print(f"  → convert to: {target_type}\n")
      dfs = [df.with_columns(pl.col(col).cast(target_type)) for df in dfs]

  return pl.concat(dfs, how="vertical")



unified_features = {
    'airport_iata': [],
    'airport_city_iata': [],
    'marketingCarrier_code': [],
    'operatingCarrier_code': [],
    'aircraft_code': [],
    'flightNumber': [],
    'searchRoute': [],
    'statusInfos': [],
    # 'baggageAllowance_weightMeasurementType': [],
    # # 'baggageAllowance_quantity': [],
    # 'cabinClass': [],
}

def get_unified_cat(col):
  for key in unified_features:
    if key in col:
      return key

  return col


def get_unified_cat_columns(cat_cols):
  for col in cat_cols:
    for key in unified_features:
      if key in col:
        unified_features[key].append(col)

  return unified_features

@timeit()
def encode_unified_cats(df_train,
                        unified_features,
                        id_col=None,
                        method='count',
                        num_workers=1):
  unified_cats = OrderedDict()

  for feature_type, columns in tqdm(unified_features.items(), desc='encode_unified_cats'):
    if not columns:
      continue

    all_values = []
    for col in columns:
      if col in df_train.columns:
        all_values.extend(
            df_train.select(pl.col(col)).drop_nulls().to_series().to_list())

    if method == 'count':
      from collections import Counter
      value_counts = Counter(all_values)
      sorted_values = [val for val, count in value_counts.most_common()]
      feature_dict = {val: idx for idx, val in enumerate(sorted_values)}
    elif method == 'val':
      unique_values = sorted(list(set(all_values)))
      feature_dict = {val: idx for idx, val in enumerate(unique_values)}
    elif method == 'seq':
      seen = set()
      unique_values = []
      for val in all_values:
        if val not in seen:
          unique_values.append(val)
          seen.add(val)
      feature_dict = {val: idx for idx, val in enumerate(unique_values)}

    unified_cats[feature_type] = feature_dict

  return unified_cats

@timeit()
def encode_cat_byseq(df_train, cat_cols, id_col=None):
  cats = OrderedDict()
  if id_col:
    df_train = sort_dataframe(df_train, id_col)

  for col in tqdm(cat_cols, desc='encode_cat_byseq'):
    categories = df_train[col].unique(maintain_order=True).to_list()
    category_dict = {category: idx for idx, category in enumerate(categories)}
    cats.update({col: category_dict})

  return cats

@timeit()
def encode_cat_bycount(df_train, cat_cols, id_col=None, num_workers=1):
  cats = OrderedDict()

  def process_column(col):
    dg = df_train.group_by(col).agg(pl.len().alias('count')).sort(
        'count', descending=True)

    return {col: {val: idx for idx, val in enumerate(dg[col].to_list())}}

  results = []
  for col in tqdm(cat_cols, desc='encode_cat_bycount'):
    results.append(process_column(col))

  for result in results:
    cats.update(result)

  return cats

@timeit()
def encode_cat_byval(df_train, cat_cols, num_workers=1):
  cats = OrderedDict()

  def process_column(col):
    categories = df_train.select(
        pl.col(col)).unique().sort(col).to_series().to_list()
    return {col: {val: idx for idx, val in enumerate(categories)}}

  for cat in tqdm(cat_cols, desc='encode_cat_byval'):
    results.append(process_column(cat))

  for result in results:
    cats.update(result)

  return cats

@timeit()
def encode_cat(df_train, cat_cols, id_col=None, method='count', num_workers=1):
  cats = None
  if method == 'seq':
    cats = encode_cat_byseq(df_train, cat_cols, id_col)
  elif method == 'count':
    cats = encode_cat_bycount(df_train, cat_cols, id_col, num_workers=num_workers)
  elif method == 'val':
    # same as rank('dense') in polars
    cats = encode_cat_byval(df_train, cat_cols, num_workers=num_workers)
  else:
    raise ValueError(f"Unsupported method: {method}")
  
  return cats

@timeit()
def encode_cat_unified(df_train,
                       cat_cols,
                       id_col=None,
                       method='count',
                       num_workers=1):

  unified_features = get_unified_cat_columns(cat_cols)

  unified_cols = set()
  for columns in unified_features.values():
    unified_cols.update(columns)

  unified_cat_cols = [col for col in cat_cols if col in unified_cols]
  regular_cat_cols = [col for col in cat_cols if col not in unified_cols]

  unified_cats = encode_unified_cats(df_train, unified_features, id_col, method, num_workers)

  regular_cats = encode_cat(df_train, regular_cat_cols, id_col, method,num_workers)

  all_cats = OrderedDict()
  all_cats.update(unified_cats)
  all_cats.update(regular_cats)

  return all_cats


@time_feats()
def add_group_feats(df: pl.DataFrame) -> pl.DataFrame:
  df = df.with_columns([
      pl.col("Id").count().over("ranker_id").alias(
          "group_size"),  
  ])
  return df

@time_feats()
def add_user_feats(df: pl.DataFrame) -> pl.DataFrame:
  df = df.with_columns(
      [pl.col("frequentFlyer").fill_null("").alias("frequentFlyer")])

  unique_combos = (
      df.select("frequentFlyer").unique().get_column("frequentFlyer").to_list())

  all_codes = sorted(
      set(chain.from_iterable(
          code.split("/") for code in unique_combos if code)))

  for code in all_codes:
    df = df.with_columns([
        pl.col("frequentFlyer").str.split("/").list.contains(code).cast(
            pl.Int8).alias(f"ff_{code}")
    ])

  df = df.with_columns(
      [pl.col("frequentFlyer").str.split("/").list.get(0).alias("ff_primary")])

  df = df.with_columns(
      [pl.col("frequentFlyer").str.count_matches("/").add(1).alias("ff_count")])

  df = df.with_columns([
      pl.when(pl.col("frequentFlyer") == "").then(0).otherwise(
          pl.col("ff_count")).alias("ff_count")
  ])

  df = df.drop("frequentFlyer")
  return df

@time_feats()
def add_searchRoute_feats(df):
  df = df.with_columns([
      pl.col("searchRoute").str.contains("/").not_().cast(
          pl.Int8).alias("isDirect")
  ])

  df = df.with_columns([
      pl.col("searchRoute").str.split("/").list.first().str.slice(
          0, 6).alias("base_searchRoute")
  ]).with_columns([
      pl.col("base_searchRoute").str.slice(0, 3).alias("p1"),
      pl.col("base_searchRoute").str.slice(3, 3).alias("p2"),
  ]).with_columns([
      pl.when(pl.col("p1") <= pl.col("p2")).then(
          pl.concat_str([pl.col("p1"), pl.col("p2")])).otherwise(
              pl.concat_str([pl.col("p2"),
                             pl.col("p1")])).alias("normed_searchRoute")
  ]).drop(["p1", "p2"])

  return df

@time_feats()
def add_segment_feats(df: pl.DataFrame) -> pl.DataFrame:
  exprs = []
  for leg in (0, 1):
    seg_cols = [
        f"legs{leg}_segments{s}_duration" for s in range(4)
        if f"legs{leg}_segments{s}_duration" in df.columns
    ]
    assert seg_cols, f"legs{leg} seg_cols is empty"
    exprs.append(
        pl.sum_horizontal([
            (pl.col(c) > 0).cast(pl.UInt8) for c in seg_cols
        ]).cast(pl.Int32).alias(f"seg_legs{leg}_count"))

  df = df.with_columns(exprs)

  df = df.with_columns([
      pl.sum_horizontal([
          pl.col(c) for c in ["seg_legs0_count", "seg_legs1_count"]
      ]).alias("seg_legs_all_count"),
  ])

  df = df.with_columns([
      pl.col("legs0_segments0_departureFrom_airport_iata").alias(
          "legs0_departureAirport"),
      pl.col("legs1_segments0_departureFrom_airport_iata").alias(
          "legs1_departureAirport"),
      pl.when(pl.col("seg_legs0_count") == 1
             ).then(pl.col("legs0_segments0_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs0_count") == 2).then(
                       pl.col("legs0_segments1_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs0_count") == 3).then(
                       pl.col("legs0_segments2_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs0_count") == 4).then(
                       pl.col("legs0_segments3_arrivalTo_airport_iata")
                   ).otherwise(None).alias("legs0_arrival_airport_iata"),

      pl.when(pl.col("seg_legs1_count") == 1
             ).then(pl.col("legs1_segments0_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs1_count") == 2).then(
                       pl.col("legs1_segments1_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs1_count") == 3).then(
                       pl.col("legs1_segments2_arrivalTo_airport_iata")
                   ).when(pl.col("seg_legs1_count") == 4).then(
                       pl.col("legs1_segments3_arrivalTo_airport_iata")
                   ).otherwise(None).alias("legs1_arrival_airport_iata"),
  ])

  return df

@time_feats()
def add_flight_duration_feats(df: pl.DataFrame) -> pl.DataFrame:
  df = df.with_columns([
      (pl.col("legs0_duration") +
       pl.col("legs1_duration")).alias("flight_duration_total"),
  ])
  df = df.with_columns([
      (pl.col("legs0_duration") /
       (pl.col("flight_duration_total") + 1e-5)).alias("legs0_duration_ratio"),
      (pl.col("legs1_duration") /
       (pl.col("flight_duration_total") + 1e-5)).alias("legs1_duration_ratio"),
  ])
  return df

def get_utc_offset(timezone_str):
  try:
    tz = ZoneInfo(timezone_str)
    now = datetime.now(tz)
    offset = now.utcoffset().total_seconds() / 3600
    return offset
  except Exception as e:
    print(f"Invalid timezone: {timezone_str} → {e}")
    return None


AIRPORTS_DB = None
TIMEZONE_FINDER = None

@timeit()
def get_airport_timezone_mapping():
  global AIRPORTS_DB, TIMEZONE_FINDER

  if AIRPORTS_DB is None:
    AIRPORTS_DB = airportsdata.load('IATA')
    TIMEZONE_FINDER = TimezoneFinder()

  airport_tz_map = {}
  for iata, info in AIRPORTS_DB.items():
    if info.get('lat') and info.get('lon'):
      timezone = TIMEZONE_FINDER.timezone_at(lat=info['lat'], lng=info['lon'])
      if timezone:
        airport_tz_map[iata] = timezone
      else:
        airport_tz_map[iata] = 'UTC'
    else:
      airport_tz_map[iata] = 'UTC'

  return airport_tz_map

def haversine_distance_expr(lat1, lon1, lat2, lon2):
  lat1_rad = lat1 * (math.pi / 180)
  lon1_rad = lon1 * (math.pi / 180)
  lat2_rad = lat2 * (math.pi / 180)
  lon2_rad = lon2 * (math.pi / 180)

  dlat = lat2_rad - lat1_rad
  dlon = lon2_rad - lon1_rad

  a = (dlat / 2
      ).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)
  c = 2 * a.sqrt().arcsin()

  return c * 6371  

@time_feats()
def add_segment_geography_time_feats(df: pl.DataFrame) -> pl.DataFrame:
  logger.info('Adding segment geography and time features - merged version')

  global AIRPORTS_DB
  if AIRPORTS_DB is None:
    AIRPORTS_DB = airportsdata.load('IATA')

  airport_info_data = []
  for iata, info in AIRPORTS_DB.items():
    timezone_str = info.get('tz', 'UTC')
    airport_info_data.append({
        'airport_code': iata,
        # 'country': info.get('country', ''),
        # 'city': info.get('city', ''),
        'lat': info.get('lat'),
        'lon': info.get('lon'),
        # 'elevation': info.get('elevation'),
        'timezone': timezone_str,
        'utc_offset_hours': get_utc_offset(timezone_str)
    })

  airport_info_df = pl.DataFrame(airport_info_data)

  geo_time_exprs = []

  for leg in [0, 1]:
    for seg in range(4):
      dep_airport_col = f"legs{leg}_segments{seg}_departureFrom_airport_iata"
      arr_airport_col = f"legs{leg}_segments{seg}_arrivalTo_airport_iata"
      duration_col = f"legs{leg}_segments{seg}_duration"

      if all(col in df.columns
             for col in [dep_airport_col, arr_airport_col, duration_col]):
        df = df.join(
            airport_info_df.select([
                'airport_code',
                # 'country',
                # 'city',
                'lat',
                'lon',
                'utc_offset_hours'
            ]).rename({
                'airport_code': dep_airport_col,
                # 'country': f"legs{leg}_seg{seg}_dep_country",
                # 'city': f"legs{leg}_seg{seg}_dep_city",
                'lat': f"legs{leg}_seg{seg}_dep_lat",
                'lon': f"legs{leg}_seg{seg}_dep_lon",
                'utc_offset_hours': f"legs{leg}_seg{seg}_dep_offset"
            }),
            on=dep_airport_col,
            how='left'
        ).with_columns([
            pl.when((pl.col(dep_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_dep_offset").fill_null(0)
                    ).otherwise(None).alias(f"legs{leg}_seg{seg}_dep_offset"),
            # pl.when((pl.col(dep_airport_col).is_not_null()) & (pl.col(duration_col) > 0))
            #   .then(pl.col(f"legs{leg}_seg{seg}_dep_country"))
            #   .otherwise(None)
            #   .alias(f"legs{leg}_seg{seg}_dep_country"),
            # pl.when((pl.col(dep_airport_col).is_not_null()) & (pl.col(duration_col) > 0))
            #   .then(pl.col(f"legs{leg}_seg{seg}_dep_city"))
            #   .otherwise(None)
            #   .alias(f"legs{leg}_seg{seg}_dep_city"),
            pl.when((pl.col(dep_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_dep_lat")
                    ).otherwise(None).alias(f"legs{leg}_seg{seg}_dep_lat"),
            pl.when((pl.col(dep_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_dep_lon")).otherwise(
                            None).alias(f"legs{leg}_seg{seg}_dep_lon"),
        ])

        df = df.join(
            airport_info_df.select([
                'airport_code',
                # 'country',
                # 'city',
                'lat',
                'lon',
                'utc_offset_hours'
            ]).rename({
                'airport_code': arr_airport_col,
                # 'country': f"legs{leg}_seg{seg}_arr_country",
                # 'city': f"legs{leg}_seg{seg}_arr_city",
                'lat': f"legs{leg}_seg{seg}_arr_lat",
                'lon': f"legs{leg}_seg{seg}_arr_lon",
                'utc_offset_hours': f"legs{leg}_seg{seg}_arr_offset"
            }),
            on=arr_airport_col,
            how='left'
        ).with_columns([
            pl.when((pl.col(arr_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_arr_offset").fill_null(0)).
            otherwise(None).alias(f"legs{leg}_seg{seg}_arr_offset"),
            # pl.when((pl.col(arr_airport_col).is_not_null()) & (pl.col(duration_col) > 0))
            #   .then(pl.col(f"legs{leg}_seg{seg}_arr_country"))
            #   .otherwise(None)
            #   .alias(f"legs{leg}_seg{seg}_arr_country"),
            # pl.when((pl.col(arr_airport_col).is_not_null()) & (pl.col(duration_col) > 0))
            #   .then(pl.col(f"legs{leg}_seg{seg}_arr_city"))
            #   .otherwise(None)
            #   .alias(f"legs{leg}_seg{seg}_arr_city"),
            pl.when((pl.col(arr_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_arr_lat")
                    ).otherwise(None).alias(f"legs{leg}_seg{seg}_arr_lat"),
            pl.when((pl.col(arr_airport_col).is_not_null()) &
                    (pl.col(duration_col) > 0)).then(
                        pl.col(f"legs{leg}_seg{seg}_arr_lon")).otherwise(
                            None).alias(f"legs{leg}_seg{seg}_arr_lon"),
        ])

        valid_coords_condition = (
            pl.col(f"legs{leg}_seg{seg}_dep_lat").is_not_null() &
            pl.col(f"legs{leg}_seg{seg}_dep_lon").is_not_null() &
            pl.col(f"legs{leg}_seg{seg}_arr_lat").is_not_null() &
            pl.col(f"legs{leg}_seg{seg}_arr_lon").is_not_null() &
            (pl.col(duration_col) > 0))

        valid_offset_condition = (
            (pl.col(duration_col) > 0) &
            (pl.col(f"legs{leg}_seg{seg}_dep_offset").is_not_null()) &
            (pl.col(f"legs{leg}_seg{seg}_arr_offset").is_not_null()))

        geo_time_exprs.extend([
            pl.when(valid_coords_condition).then(
                haversine_distance_expr(pl.col(f"legs{leg}_seg{seg}_dep_lat"),
                                        pl.col(f"legs{leg}_seg{seg}_dep_lon"),
                                        pl.col(f"legs{leg}_seg{seg}_arr_lat"),
                                        pl.col(f"legs{leg}_seg{seg}_arr_lon"))
            ).otherwise(0.0).alias(f"legs{leg}_seg{seg}_distance_km"),

            # pl.when(valid_coords_condition)
            #   .then((pl.col(f"legs{leg}_seg{seg}_dep_country") != pl.col(f"legs{leg}_seg{seg}_arr_country")).cast(pl.Int8))
            #   .otherwise(0)
            #   .alias(f"legs{leg}_seg{seg}_is_international"),

            # pl.when(valid_coords_condition)
            #   .then((pl.col(f"legs{leg}_seg{seg}_dep_city") == pl.col(f"legs{leg}_seg{seg}_arr_city")).cast(pl.Int8))
            #   .otherwise(0)
            #   .alias(f"legs{leg}_seg{seg}_is_same_city"),
        ])

  legs_airport_pairs = [
      ('legs0_departureAirport', 'legs0_dep_country', 'legs0_dep_city',
       'legs0_dep_lat', 'legs0_dep_lon', 'legs0_dep_offset'),
      ('legs0_arrival_airport_iata', 'legs0_arr_country', 'legs0_arr_city',
       'legs0_arr_lat', 'legs0_arr_lon', 'legs0_arr_offset'),
      ('legs1_departureAirport', 'legs1_dep_country', 'legs1_dep_city',
       'legs1_dep_lat', 'legs1_dep_lon', 'legs1_dep_offset'),
      ('legs1_arrival_airport_iata', 'legs1_arr_country', 'legs1_arr_city',
       'legs1_arr_lat', 'legs1_arr_lon', 'legs1_arr_offset'),
  ]

  for airport_col, country_col, city_col, lat_col, lon_col, offset_col in legs_airport_pairs:
    if airport_col in df.columns:
      df = df.join(
          airport_info_df.select([
              'airport_code',
              # 'country',
              # 'city',
              'lat',
              'lon',
              'utc_offset_hours'
          ]).rename({
              'airport_code': airport_col,
              # 'country': country_col,
              # 'city': city_col,
              'lat': lat_col,
              'lon': lon_col,
              'utc_offset_hours': offset_col,
          }),
          on=airport_col,
          how='left').with_columns([
              pl.when(pl.col(airport_col).is_not_null()
                     ).then(pl.col(offset_col).fill_null(0)
                           ).otherwise(0).alias(offset_col)
          ])

  if geo_time_exprs:
    df = df.with_columns(geo_time_exprs)

  legs_geo_exprs = []

  if all(col in df.columns for col in
         ['legs0_dep_lat', 'legs0_dep_lon', 'legs0_arr_lat', 'legs0_arr_lon']):
    legs_geo_exprs.extend([
        pl.when(
            pl.col('legs0_dep_lat').is_not_null() &
            pl.col('legs0_dep_lon').is_not_null() &
            pl.col('legs0_arr_lat').is_not_null() &
            pl.col('legs0_arr_lon').is_not_null()).then(
                haversine_distance_expr(pl.col('legs0_dep_lat'),
                                        pl.col('legs0_dep_lon'),
                                        pl.col('legs0_arr_lat'),
                                        pl.col('legs0_arr_lon'))
            ).otherwise(None).alias('legs0_direct_distance_km'),

        # (pl.col('legs0_dep_country') != pl.col('legs0_arr_country')).cast(pl.Int8).alias('legs0_is_international'),

        # (pl.col('legs0_dep_city') == pl.col('legs0_arr_city')).cast(pl.Int8).alias('legs0_is_same_city'),
    ])

  if all(col in df.columns for col in
         ['legs1_dep_lat', 'legs1_dep_lon', 'legs1_arr_lat', 'legs1_arr_lon']):
    legs_geo_exprs.extend([
        # legs1直线距离
        pl.when(
            pl.col('legs1_dep_lat').is_not_null() &
            pl.col('legs1_dep_lon').is_not_null() &
            pl.col('legs1_arr_lat').is_not_null() &
            pl.col('legs1_arr_lon').is_not_null()).then(
                haversine_distance_expr(pl.col('legs1_dep_lat'),
                                        pl.col('legs1_dep_lon'),
                                        pl.col('legs1_arr_lat'),
                                        pl.col('legs1_arr_lon'))
            ).otherwise(0.0).alias('legs1_direct_distance_km'),

        # (pl.col('legs1_dep_country') != pl.col('legs1_arr_country')).cast(pl.Int8).alias('legs1_is_international'),

        # (pl.col('legs1_dep_city') == pl.col('legs1_arr_city')).cast(pl.Int8).alias('legs1_is_same_city'),
    ])

  if legs_geo_exprs:
    df = df.with_columns(legs_geo_exprs)

  assert 'legs0_seg0_distance_km' in df.columns

  summary_exprs = []

  for leg in [0, 1]:
    seg_distance_cols = [
        f"legs{leg}_seg{seg}_distance_km" for seg in range(4)
        if f"legs{leg}_seg{seg}_distance_km" in df.columns
    ]
    assert seg_distance_cols
    summary_exprs.append(
        pl.sum_horizontal([pl.col(col) for col in seg_distance_cols
                          ]).alias(f"legs{leg}_total_segment_distance_km"))

  df = df.with_columns(summary_exprs)
  assert 'legs0_total_segment_distance_km' in df.columns

  summary_exprs = []
  for leg in [0, 1]:
    summary_exprs.append((pl.col(f"legs{leg}_total_segment_distance_km") /
                          (pl.col(f"legs{leg}_direct_distance_km") +
                           1)).alias(f"legs{leg}_detour_ratio"))

    # seg_distance_cols = [f"legs{leg}_seg{seg}_distance_km" for seg in range(4)
    #                       if f"legs{leg}_seg{seg}_distance_km" in df.columns]

    # # if len(seg_distance_cols) > 1:
    # summary_exprs.append(
    #   pl.concat_list([pl.col(col) for col in seg_distance_cols])
    #     .list.eval(pl.element().filter(pl.element() > 0))  # 过滤掉0距离
    #     .list.std()
    #     .alias(f"legs{leg}_segment_distance_std")
    # )

    # international_cols = [f"legs{leg}_seg{seg}_is_international" for seg in range(4)]
    # # if international_cols:
    # summary_exprs.append(
    #   pl.sum_horizontal([pl.col(col) for col in international_cols])
    #     .alias(f"legs{leg}_international_segments_count")
    # )

  df = df.with_columns(summary_exprs)

  assert 'legs0_total_segment_distance_km' in df.columns

  df = df.with_columns([
      (pl.col("legs0_total_segment_distance_km") +
       pl.col("legs1_total_segment_distance_km").fill_null(0)
      ).alias("total_flight_distance_km"),
      (pl.col("legs0_direct_distance_km") +
       pl.col("legs1_direct_distance_km").fill_null(0)
      ).alias("direct_flight_distance_km"),
  ])

  df = df.with_columns([
      (pl.col("total_flight_distance_km") /
       (pl.col("flight_duration_total") + 1e-5)).alias("avg_flight_speed_kmh"),

      (pl.col("totalPrice") / (pl.col("total_flight_distance_km") + 1)
      ).alias("flight_price_per_km"),

      (pl.col("direct_flight_distance_km") /
       (pl.col("flight_duration_total") + 1e-5)).alias("avg_direct_speed_kmh"),

      (pl.col("totalPrice") / (pl.col("direct_flight_distance_km") + 1)
      ).alias("direct_price_per_km"),
  ])

  # region_exprs = []
  # country_cols = []
  # for leg in [0, 1]:
  #   country_cols.extend([f"legs{leg}_dep_country", f"legs{leg}_arr_country"])
  #   for seg in range(4):
  #     country_cols.extend([f"legs{leg}_seg{seg}_dep_country", f"legs{leg}_seg{seg}_arr_country"])

  # region_exprs.append(
  #   pl.concat_list([pl.col(col) for col in country_cols if col in df.columns])
  #     .list.drop_nulls()
  #     .list.unique()
  #     .list.len()
  #     .alias("total_unique_countries")
  # )

  # df = df.with_columns(region_exprs)

  return df

@time_feats()
def add_travel_duration_feats(df: pl.DataFrame) -> pl.DataFrame:
  utc_time_exprs = []
  time_offset_pairs = [
      ('legs0_departureAt', 'legs0_dep_offset', 'legs0_departure_utc'),
      ('legs0_arrivalAt', 'legs0_arr_offset', 'legs0_arrival_utc'),
      ('legs1_departureAt', 'legs1_dep_offset', 'legs1_departure_utc'),
      ('legs1_arrivalAt', 'legs1_arr_offset', 'legs1_arrival_utc'),
  ]

  for time_col, offset_col, utc_col in time_offset_pairs:
    if time_col in df.columns and offset_col in df.columns:
      utc_time_exprs.append(
          (pl.col(time_col).str.to_datetime() -
           pl.duration(hours=pl.col(offset_col))).alias(utc_col))

  df = df.with_columns(utc_time_exprs)

  exprs = []

  exprs.extend([
      ((pl.col("legs0_arrival_utc") -
        pl.col("legs0_departure_utc")).dt.total_seconds() /
       3600).alias("travel_duration_legs0"),
      ((pl.col("legs1_arrival_utc") -
        pl.col("legs1_departure_utc")).dt.total_seconds() /
       3600).alias("travel_duration_legs1"),
      ((pl.col("legs1_departure_utc") -
        pl.col("legs0_arrival_utc")).dt.total_seconds() / 3600
      ).alias("travel_connection_duration"),
      ((pl.col("legs1_arrival_utc") -
        pl.col("legs0_departure_utc")).dt.total_seconds() / 3600
      ).alias("travel_duration_total"),
  ])

  df = df.with_columns(exprs)

  df = df.with_columns([
      pl.max_horizontal(pl.col("travel_connection_duration"),
                        1).alias("travel_connection_duration"),
      pl.max_horizontal(pl.col("travel_duration_total"),
                        1).alias("travel_duration_total"),
  ])

  df = df.with_columns([
      (pl.col("travel_connection_duration") /
       (pl.col("travel_duration_total") +
        1e-5)).alias("travel_connection_duration_ratio"),
      (pl.col("flight_duration_total") /
       (pl.col("travel_duration_total") +
        1e-5)).alias("flight_duration_travel_ratio"),
  ])

  exprs = [
      (((pl.col("legs0_departure_utc") -
         pl.col("requestDate")).dt.total_seconds()) / 3600
      ).alias("book_lead_time_hours"),
      (((pl.col("legs1_arrival_utc") -
         pl.col("requestDate")).dt.total_seconds()) /
       3600).alias("book_after_time_hours"),
  ]
  if 'requestDepartureDate' in df.columns:
    exprs += [
        (((pl.col("legs0_departureAt").str.to_datetime() -
           pl.col("requestDepartureDate").str.to_datetime()).dt.total_seconds())
         / 3600).alias("requestDepartureDate_diff_hours"),
        (((pl.col("legs1_departureAt").str.to_datetime() -
           pl.col("requestReturnDate").str.to_datetime()).dt.total_seconds()) /
         3600).alias("requestReturnDate_diff_hours"),
    ]
  df = df.with_columns(exprs)

  df = df.with_columns(
      (pl.col('travel_duration_total') / 24).alias('travel_total_days'),
      (pl.col('book_lead_time_hours') / 24).alias("book_lead_time_days"),
      (pl.col('book_after_time_hours') / 24).alias("book_after_time_days"),
  )

  temp_cols = [
      'legs0_departure_utc',
      'legs0_arrival_utc',
      'legs1_departure_utc',
      'legs1_arrival_utc',
      'legs0_dep_tz',
      'legs0_arr_tz',
      'legs1_dep_tz',
      'legs1_arr_tz',
      'legs0_departureAirport',
      # 'legs0_segments0_departureFrom_airport_iata',
      #  'legs0_arrivalAirport',
      'legs1_departureAirport',
      # 'legs1_segments0_departureFrom_airport_iata',
      #  'legs1_arrivalAirport'
  ]
  df = df.drop([col for col in temp_cols if col in df.columns])

  return df

@time_feats()
def add_time_feats(df: pl.DataFrame) -> pl.DataFrame:
  exprs = []
  time_cols = [
      "legs0_departureAt",
      "legs0_arrivalAt",
      "legs1_departureAt",
      "legs1_arrivalAt",
      "requestDepartureDate",
      "requestReturnDate",
  ]
  # for leg in [0, 1]:
  #   for seg in range(2):
  #     time_cols.extend([
  #       f'legs{leg}_seg{seg}_departure_local',
  #       f'legs{leg}_seg{seg}_arrival_local'
  #     ])
  ic(time_cols)

  for c in time_cols:
    # assert c in df.columns
    if c not in df.columns:
      logger.info(f"Column {c} is missing from DataFrame")
      continue
    if not c.endswith('_local'):
      dt = pl.col(c).str.to_datetime()
    else:
      dt = pl.col(c)
    hour_col = dt.dt.hour()
    weekday_col = dt.dt.weekday()
    month_col = dt.dt.month()
    exprs += [
        hour_col.alias(f"time_{c}_hour"),
        weekday_col.alias(f"time_{c}_weekday"),
        month_col.alias(f"time_{c}_month"),
        (dt.dt.weekday() >= 5).cast(pl.Int32).alias(f"time_{c}_is_weekend"),
        (dt.dt.hour().is_between(6, 9) | dt.dt.hour().is_between(17, 20)
        ).cast(pl.Int32).alias(f"time_{c}_is_peak"),
        (dt.dt.hour().is_between(0, 5)
        ).cast(pl.Int32).alias(f"time_{c}_is_red_eye"),
        pl.when(hour_col.is_between(5, 8)).then(0)  
        .when(hour_col.is_between(9, 11)).then(1)  
        .when(hour_col.is_between(12, 17)).then(2)  
        .when(hour_col.is_between(18, 22)).then(3)  
        .otherwise(4)  
        .alias(f"time_{c}_period"),

        pl.when(month_col.is_in([12, 1, 2])).then(0)
        .when(month_col.is_in([3, 4, 5])).then(1)  
        .when(month_col.is_in([6, 7, 8])).then(2)  
        .otherwise(3) 
        .alias(f"time_{c}_season"),

        (weekday_col.is_between(1, 5) & hour_col.is_between(8, 18)
        ).cast(pl.Int8).alias(f"time_{c}_is_business_hours"),

        (hour_col * (2 * np.pi / 24)).sin().alias(f"time_{c}_hour_sin"),
        (hour_col * (2 * np.pi / 24)).cos().alias(f"time_{c}_hour_cos"),
        (weekday_col * (2 * np.pi / 7)).sin().alias(f"time_{c}_weekday_sin"),
        (weekday_col * (2 * np.pi / 7)).cos().alias(f"time_{c}_weekday_cos"),
        (month_col * (2 * np.pi / 12)).sin().alias(f"time_{c}_month_sin"),
        (month_col * (2 * np.pi / 12)).cos().alias(f"time_{c}_month_cos"),
    ]
  df = df.with_columns(exprs)
  time_cols = [col for col in time_cols if col in df.columns]
  df = df.drop(time_cols)

  return df


@time_feats()
def add_cabin_feats(df: pl.DataFrame) -> pl.DataFrame:
  exprs = []
  for leg in (0, 1):
    cabin_cols = [
        f"legs{leg}_segments{s}_cabinClass" for s in range(4)
        if f"legs{leg}_segments{s}_cabinClass" in df.columns
    ]

    assert cabin_cols, f"leg{leg} cabin_cols is empty"
    exprs.append(
        pl.mean_horizontal(
            pl.col(c) for c in cabin_cols).alias(f"avg_cabin_legs{leg}"))

  df = df.with_columns(exprs)

  df = df.with_columns([
      pl.mean_horizontal(
          pl.col(c) for c in ["avg_cabin_legs0", "avg_cabin_legs1"]).alias(
              "avg_cabin_legs_all"),
  ])

  return df

@time_feats()
def add_baggage_feats(df: pl.DataFrame) -> pl.DataFrame:
  drop_cols = []
  exprs = []
  for leg in (0, 1):
    for s in range(4):
      exprs.extend([
          # baggage_count: only keep quantity if type is 'piece'
          pl.when(
              pl.col(
                  f"legs{leg}_segments{s}_baggageAllowance_weightMeasurementType"
              ) == 0
          ).then(
              pl.col(f"legs{leg}_segments{s}_baggageAllowance_quantity").cast(
                  pl.Int8)
          ).otherwise(None).alias(f"legs{leg}_segments{s}_baggage_count"),
          # baggage_weight: only keep quantity if type is 'weight'
          pl.when(
              pl.col(
                  f"legs{leg}_segments{s}_baggageAllowance_weightMeasurementType"
              ) == 1
          ).then(
              pl.col(f"legs{leg}_segments{s}_baggageAllowance_quantity").cast(
                  pl.Float32)
          ).otherwise(None).alias(f"legs{leg}_segments{s}_baggage_weight")
      ])
      drop_cols.append(f"legs{leg}_segments{s}_baggageAllowance_quantity")
  df = df.with_columns(exprs)
  df = df.drop(drop_cols)

  exprs = []
  for leg in (0, 1):
    baggage_cols = [
        f"legs{leg}_segments{s}_baggage_count" for s in range(4)
        if f"legs{leg}_segments{s}_baggage_count" in df.columns
    ]

    assert baggage_cols, f"leg{leg} baggage_cols is empty"
    exprs.append(
        pl.mean_horizontal([pl.col(c) for c in baggage_cols
                           ]).alias(f"avg_baggage_count_legs{leg}"))

  for leg in (0, 1):
    baggage_cols = [
        f"legs{leg}_segments{s}_baggage_weight" for s in range(4)
        if f"legs{leg}_segments{s}_baggage_weight" in df.columns
    ]

    assert baggage_cols, f"leg{leg} baggage_cols is empty"
    exprs.append(
        pl.mean_horizontal([pl.col(c) for c in baggage_cols
                           ]).alias(f"avg_baggage_weight_legs{leg}"))

  df = df.with_columns(exprs)

  df = df.with_columns([
      pl.mean_horizontal(
          pl.col(c)
          for c in ["avg_baggage_count_legs0", "avg_baggage_count_legs1"
                   ]).alias("avg_baggage_count_legs_all"),
      pl.mean_horizontal(
          pl.col(c)
          for c in ["avg_baggage_weight_legs0", "avg_baggage_weight_legs1"
                   ]).alias("avg_baggage_weight_legs_all"),
  ])

  return df

@time_feats()
def add_seats_feats(df: pl.DataFrame) -> pl.DataFrame:
  exprs = []
  for leg in (0, 1):
    seats_cols = [
        f"legs{leg}_segments{s}_seatsAvailable" for s in range(4)
        if f"legs{leg}_segments{s}_seatsAvailable" in df.columns
    ]

    assert seats_cols, f"leg{leg} seats_cols is empty"
    exprs.append(
        pl.mean_horizontal(
            pl.col(c) for c in seats_cols).alias(f"avg_seats_count_legs{leg}"))

  df = df.with_columns(exprs)

  df = df.with_columns([
      pl.mean_horizontal(
          pl.col(c) for c in ["avg_seats_count_legs0", "avg_seats_count_legs1"]
      ).alias("avg_seats_count_legs_all"),
  ])
  return df

@time_feats()
def add_carrier_feats(df: pl.DataFrame) -> pl.DataFrame:
  mc_cols = []
  for leg in (0, 1):
    mc_cols.extend([
        f"legs{leg}_segments{s}_marketingCarrier_code" for s in range(4)
        if f"legs{leg}_segments{s}_marketingCarrier_code" in df.columns
    ])

    assert mc_cols, f"leg{leg} mc_cols is empty"

  df = df.with_columns(
    pl.struct(mc_cols)
      .map_elements(lambda s: len(set(v for v in s.values() if v is not None)), return_dtype=pl.UInt8)
      .alias("num_unique_carriers")
  )

  df = df.with_columns([
      (pl.col('num_unique_carriers') / pl.max_horizontal(
          pl.col('seg_legs_all_count'), 1)).alias('carrier_diversity_ratio'),
  ])

  return df


@time_feats()
def add_ranking_feats(df, group_col, suffix=''):
  exprs = []
  for col, order in rank_order.items():
    if col in df.columns:
      exprs.append(
          pl.col(col).rank(method='average', descending=(
              order == 'desc')).over(group_col).alias(f'rank_{col}{suffix}'))
    else:
      logger.warning(f"Column {col} not found in DataFrame, skipping ranking.")
  df = df.with_columns(exprs)

  return df

@time_feats()
def add_flighthash_feats(df):
  df = (df.with_columns([
      pl.len().over(["ranker_id", "flight_hash"]).alias("flight_hash_count"),
  ]).with_columns([
      (pl.col("flight_hash_count") /
       pl.col("group_size")).alias("flight_hash_ratio"),
      pl.col("flight_hash_count").rank(
          "dense",
          descending=True).over("ranker_id").alias("rank_flight_hash_count"),
  ]))
  return df

#Notice not consider label/selected and df is (train and test) combined
@time_feats()
def add_stats_feats(df, group_col='profileId'):
  exprs = []
  for col in rank_order.keys():
    if col in df.columns:
      exprs.extend([
        pl.col(col).mean().over(group_col).alias(f"avg_{col}_{group_col}_stats"),
        pl.col(col).min().over(group_col).alias(f"min_{col}_{group_col}_stats"),
        pl.col(col).max().over(group_col).alias(f"max_{col}_{group_col}_stats"),
        pl.col(col).std().over(group_col).alias(f"std_{col}_{group_col}_stats"),
        pl.col(col).median().over(group_col).alias(f"median_{col}_{group_col}_stats"),
      ])
  df = df.with_columns(exprs)
  exprs = []
  for col in rank_order.keys():
    if col in df.columns:
      exprs.extend([
        ((pl.col(col) - pl.col(f"avg_{col}_{group_col}_stats")) / (pl.col(f"std_{col}_{group_col}_stats") + 1e-5)).alias(f"{col}_zscore_{group_col}_stats"),
        ((pl.col(col) - pl.col(f"min_{col}_{group_col}_stats")) / (pl.col(f"max_{col}_{group_col}_stats") - pl.col(f"min_{col}_{group_col}_stats") + 1e-5)).alias(f"{col}_minmax_{group_col}_stats"),
        (pl.col(col) / (pl.col(f"avg_{col}_{group_col}_stats") + 1e-5)).alias(f"{col}_{group_col}_stats_ratio"),
      ])

  df = df.with_columns(exprs)
  return df



@time_feats()
def make_history_avg(df, source_cols, group_col, suffix):
  ori_cols = [col for col in df.columns]
  selected_df = df.filter(
      pl.col("selected") == 1).select(["ranker_id", "requestDate", group_col] +
                                      source_cols)

  ranker_to_profile = dict(
      zip(selected_df["ranker_id"].to_list(), selected_df[group_col].to_list()))

  ranker_to_timestamp = dict(
      zip(selected_df["ranker_id"].to_list(),
          selected_df["requestDate"].to_list()))

  history_df = selected_df.select(["ranker_id", "requestDate", group_col] +
                                  source_cols)

  all_stats_dict = {}  # ranker_id -> {col_mean: val, col_std: val}

  unique_ranker_ids = df["ranker_id"].unique().to_list()

  for current_ranker_id in tqdm(unique_ranker_ids,
                                desc="Обработка ranker_id",
                                mininterval=10.0):
    current_profile_id = ranker_to_profile.get(current_ranker_id)

    current_timestamp = ranker_to_timestamp.get(current_ranker_id)

    profile_history = history_df.filter(
        (pl.col(group_col) == current_profile_id) &
        (pl.col("ranker_id") != current_ranker_id) &
        (pl.col("requestDate") < current_timestamp))

    agg_result = profile_history.select([
        *[
            pl.col(col).mean().alias(f"{col}{suffix}_mean")
            for col in source_cols
        ],
        *[pl.col(col).std().alias(f"{col}{suffix}_std") for col in source_cols],
        *[
            pl.col(col).count().alias(f"{col}{suffix}_count")
            for col in source_cols
        ],  
        *[
            pl.col(col).median().alias(f"{col}{suffix}_median")
            for col in source_cols
        ],
        *[
            pl.col(col).quantile(0.25).alias(f"{col}{suffix}_q25")
            for col in source_cols
        ],
        *[
            pl.col(col).quantile(0.75).alias(f"{col}{suffix}_q75")
            for col in source_cols
        ]
    ])

    if agg_result.height > 0:
      row = agg_result.row(0)
      n_cols = len(source_cols)
      all_stats_dict[current_ranker_id] = {
          **{
              f"{col}{suffix}_mean": row[i] for i, col in enumerate(source_cols)
          },
          **{
              f"{col}{suffix}_std": row[i + n_cols] for i, col in enumerate(source_cols)
          },
          **{
              f"{col}{suffix}_count": row[i + 2 * n_cols] for i, col in enumerate(source_cols)
          },
          **{
              f"{col}{suffix}_median": row[i + 3 * n_cols] for i, col in enumerate(source_cols)
          },
          **{
              f"{col}{suffix}_q25": row[i + 4 * n_cols] for i, col in enumerate(source_cols)
          },
          **{
              f"{col}{suffix}_q75": row[i + 5 * n_cols] for i, col in enumerate(source_cols)
          },
      }

  update_data = []
  for ranker_id, stats in all_stats_dict.items():
    row = {"ranker_id": ranker_id, **stats}
    update_data.append(row)

  schema = {"ranker_id": pl.Utf8} 
  for col in source_cols:
    schema[f"{col}{suffix}_mean"] = pl.Float32
    schema[f"{col}{suffix}_std"] = pl.Float32
    schema[f"{col}{suffix}_count"] = pl.Int32
    schema[f"{col}{suffix}_median"] = pl.Float32
    schema[f"{col}{suffix}_q25"] = pl.Float32
    schema[f"{col}{suffix}_q75"] = pl.Float32

  update_df = pl.DataFrame(update_data, schema=schema)

  df = df.join(update_df, on="ranker_id", how="left")

  agg_exprs = []
  for col in source_cols:
    agg_exprs.extend([
      pl.col(col).mean().alias(f"{col}{suffix}_mean"),
      pl.col(col).std().alias(f"{col}{suffix}_std"),
      pl.col(col).count().alias(f"{col}{suffix}_count"),
      pl.col(col).median().alias(f"{col}{suffix}_median"),
      pl.col(col).quantile(0.25).alias(f"{col}{suffix}_q25"),
      pl.col(col).quantile(0.75).alias(f"{col}{suffix}_q75")
    ])

  df_stats = (df.filter(
      pl.col("selected") == 1).group_by(group_col).agg(agg_exprs))

  stats_cols = [c for c in df.columns if c.endswith((
    "_mean", "_std", "_median", "_q25", "_q75")) and c not in ori_cols]
  count_cols = [c for c in df.columns if c.endswith("_count") and c not in ori_cols]

  df = df.with_columns([
    pl.col(stats_cols).cast(pl.Float32),
    pl.col(count_cols).cast(pl.Int32)
  ])
  df_stats = df_stats.with_columns([
    pl.col(stats_cols).cast(pl.Float32),
    pl.col(count_cols).cast(pl.Int32)
  ])

  return df, df_stats


@time_feats()
def gen_feats(df):
  df = add_group_feats(df)
  df = add_user_feats(df)
  df = add_searchRoute_feats(df)
  df = add_segment_feats(df)
  df = add_flight_duration_feats(df)
  df = add_segment_geography_time_feats(df)
  df = add_travel_duration_feats(df)
  df = add_time_feats(df)
  
  temp_geo_cols = []
  for leg in [0, 1]:
    temp_geo_cols.extend([
        f'legs{leg}_dep_lat', f'legs{leg}_dep_lon', f'legs{leg}_arr_lat',
        f'legs{leg}_arr_lon'
    ])
    for seg in range(4):
      temp_geo_cols.extend([
          f"legs{leg}_seg{seg}_dep_lat",
          f"legs{leg}_seg{seg}_dep_lon",
          f"legs{leg}_seg{seg}_arr_lat",
          f"legs{leg}_seg{seg}_arr_lon",

      ])

  drop_cols = [col for col in temp_geo_cols if col in df.columns]
  ic(drop_cols)
  df = df.drop(drop_cols)
  
  df = add_cabin_feats(df)
  df = add_baggage_feats(df)
  df = add_seats_feats(df)
  df = add_carrier_feats(df)
  
  df = add_flighthash_feats(df)
  df = add_ranking_feats(df, 'ranker_id')
  df = add_ranking_feats(df, ['ranker_id', 'flight_hash'], '_in_hash_group')
  
  df = add_stats_feats(df, 'uid')
  df = add_stats_feats(df, 'companyID')
  
  drop_cols = [
      col for col in df.columns if any(['_utc' in col, '_local' in col])
  ]
  ic(drop_cols)
  df = df.drop(drop_cols)
  
  if FLAGS.history_avg:
    test = get_test(df)
    
    df = get_nontest(df)
    train = get_train(df)
    
    # if not training using all train data, valid data need to merge stats similar as test
    if not FLAGS.online:
      valid = get_valid(df)
      test = pl.concat([valid, test], how='vertical')

    train, df_stats_pr = make_history_avg(train,
                                       source_cols=source_cols,
                                       group_col="uid",
                                       suffix='_uid')
    test = test.join(df_stats_pr, on='uid', how='left')
    
    train, df_stats_co = make_history_avg(train,
                                       source_cols=source_cols,
                                       group_col="companyID",
                                       suffix='_company')
    test = test.join(df_stats_co, on='companyID', how='left')

    # test = test.select(train.columns)
    
    df = align_and_concat([train, test])
  return df


def preprocess(add_feats=True):
  df = load_df(use_ext=FLAGS.use_ext)
  df = set_fold(df)
  if FLAGS.fast:
    df_train = get_nontest(df)
    df_test = get_test(df)
    df_train = filter(df_train, 0.01, FLAGS.seed)
    df = pl.concat([df_train, df_test], how='vertical')
       
  if add_feats:
    df = gen_feats(df)
  
  numer_cols = get_numer_cols(df)
  cat_cols = get_cat_cols(df)
  feat_cols = numer_cols + cat_cols
  
  df = smart_fillnull(df, numer_cols, cat_cols)
  
  ignore_cols = [col for col in IGNORE_COLS]
  cat_cols = [col for col in cat_cols if col not in ignore_cols]
  numer_cols = [col for col in numer_cols if col not in ignore_cols]
  feat_cols = [col for col in feat_cols if col not in ignore_cols]
  
  train = get_train(df) if not FLAGS.stats_all else df
  cats = encode_cat_unified(train, cat_cols, method=FLAGS.cat_method, num_workers=1)
  ic(cats.keys())
  df = df.with_columns([
      pl.col(col).replace_strict(cats[get_unified_cat(col)], default=-1) for col in cat_cols
  ])
  
  if FLAGS.remove_cats:  
    if not FLAGS.reserve_cats:
      numer_cols += cat_cols
      cat_cols = []
    else:
      reserve_cats = ['profileId', 'companyID', 'corporateTariffCode', 'nationality', 'companyCode']
      numer_cols += [col for col in cat_cols if col not in reserve_cats]
      cat_cols = [col for col in cat_cols if col in reserve_cats] 
   
  icl(numer_cols, 10) 
  icl(cat_cols, 10) 
  cols_dict = {
        'numer': numer_cols,  
        'cat': cat_cols,    
        'feat': feat_cols,           
  }  
  return df, cols_dict


df, cols_dict = preprocess(add_feats=FLAGS.add_feats)
df


def preprocess_cat_cols(df, cat_cols):
  if not cat_cols:
    return df
  
  for col in cat_cols:
    df[col] = df[col].astype('category')
  return df

def get_num_boost_round(params):
  if FLAGS.fast:
    return 100
  if FLAGS.trees:
    return FLAGS.trees
  if 'iterations' in params:
    return params['iterations']
  if 'num_iterations' in params:
    return params['num_iterations']
  if 'n_estimators' in params:
    return params['n_estimators']


cat_cols = cols_dict['cat']
feat_cols = cols_dict['feat']
df_train, df_valid = get_train_valid(df)  
df_test = get_test(df)

if FLAGS.mode == 'train':
  X_train = df_train.to_pandas()
  X_train = X_train[feat_cols]
  X_train = preprocess_cat_cols(X_train, cat_cols)
  
  y_train = df_train['selected'].to_pandas()
  group = df_train.select('ranker_id').group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
  ic(X_train.shape, group.shape)

  X_valid = df_valid.to_pandas()
  X_valid = X_valid[feat_cols]
  X_valid = preprocess_cat_cols(X_valid, cat_cols)

X_test = df_test.to_pandas()
X_test = X_test[feat_cols]
X_test = preprocess_cat_cols(X_test, cat_cols)


import xgboost as xgb
if FLAGS.mode == 'train':
  dtrain = xgb.DMatrix(
    X_train,
    y_train,
    group=group,
    enable_categorical=True)
# dvalid = xgb.DMatrix(X_valid, enable_categorical=True)
# dtest = xgb.DMatrix(X_test, enable_categorical=True)


def show_feat_importance(model):
  imp = model.get_score(importance_type="gain")

  imp_df = (
    pd.DataFrame({
      "feat": list(imp.keys()), 
      "importance": list(imp.values())
      })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
  )
  return imp_df


class XGBTQDMCallback(xgb.callback.TrainingCallback):

  def __init__(self, total_iterations, desc='', eval_name='valid'):
    self.pbar = tqdm(total=total_iterations, desc=desc)
    self.eval_name = eval_name

  def after_iteration(self, model, epoch, evals_log):
    self.pbar.update(1)
    for eval_name, metrics in evals_log.items():
      if eval_name == self.eval_name:
        m = {}
        for metric_name, values in metrics.items():
          m.update({metric_name: values[-1]})
        self.pbar.set_postfix(m)

    if epoch + 1 == self.pbar.total:
      self.pbar.close()

    return False


def batch_predict(model, X, batch_size=50_000, proba=False, inplace=False, progress=True):
  results = []
  n = len(X)
  iterator = range(0, n, batch_size)
  if progress:
    iterator = tqdm(iterator, desc="Batch Prediction")

  for start in iterator:
    end = min(start + batch_size, n)
    batch = X[start:end]

    if inplace and hasattr(model, "inplace_predict"):
      batch_pred = model.inplace_predict(batch)
    elif proba and hasattr(model, "predict_proba"):
      batch_pred = model.predict_proba(batch)
    else:
      batch_pred = model.predict(batch)

    results.append(batch_pred)

  first = results[0]
  if isinstance(first, np.ndarray):
    if first.ndim == 1:
      return np.concatenate(results)
    else:
      return np.vstack(results)
  else:
    return [item for sublist in results for item in sublist]



models = []
preds = []


params = params_xgb.copy()
objectives = [
  'rank:ndcg',
  'rank:map',
  'rank:pairwise',
  'binary:logistic'
]
if FLAGS.n_models > 0:
  objectives = objectives[:FLAGS.n_models]
params['device'] = FLAGS.device
# params['predictor'] = 'cpu_predictor'
ic(params)


if FLAGS.mode == 'train':
  for objective in tqdm(objectives, desc='objectives'):
    params['objective'] = objective
    model = xgb.train(
              params=params,
              dtrain=dtrain,
              num_boost_round=get_num_boost_round(params),
              evals=None,
              callbacks=[XGBTQDMCallback(get_num_boost_round(params), f'xgb_train_{objective}')],
              verbose_eval=100,
          )
    model.set_param({"device": "cpu"}) 
    # pred = model.predict(dvalid)
    # pred = model.inplace_predict(X_valid)
    pred = batch_predict(model, X_valid, inplace=True)
    ic(objective, pred)
    preds.append(pred)
    ic(objective)
    display(show_feat_importance(model))
    models.append(model)


def hitrate_at_3(y_true, y_pred, groups):
  df = pl.DataFrame({'group': groups, 'pred': y_pred, 'true': y_true})

  return (df.filter(pl.col("group").count().over("group") > 10).sort(
      ["group", "pred"], descending=[False, True]).group_by(
          "group", maintain_order=True).head(3).group_by("group").agg(
              pl.col("true").max()).select(pl.col("true").mean()).item())
  
def eval_df(df):
  score = hitrate_at_3(
      df['selected'].to_numpy(),
      df['pred'].to_numpy(),
      df['ranker_id'].to_numpy()
  )
  return score


def rerank(df: pl.DataFrame, penalty_factor=0.12):
  df = df.with_columns(
      pl.max("pred").over(["ranker_id", "flight_hash"]).alias("max_score_same_flight"))

  df = df.with_columns((pl.col("pred") - penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred"))).alias("pred"))

  return df


for objective, pred in zip(objectives, preds):
  df_valid = df_valid.with_columns(
    pl.Series("pred", pred)
  )
  score = eval_df(df_valid)
  ic(objective, score)


def ensemble(preds):
  preds = np.array(preds)
  i = 0
  for pred in preds:
    ic(pred.shape)
    min_val = pred.min()
    max_val = pred.max()
    if not (min_val >= 0 and max_val <= 1):
      pred = 1 / (1 + np.exp(-pred))
    preds[i] = pred
    i += 1
  pred = preds.mean(0)
  return pred


if FLAGS.mode == 'train':
  for i, model in tqdm(enumerate(models), total=len(models)):
    with open(f'{FLAGS.out_dir}/{i}.pkl', 'wb') as f:
      pickle.dump(model, f)


if FLAGS.mode != 'train':
  models = []
  preds = []
  model_dir = FLAGS.model_dir if os.path.exists(FLAGS.model_dir) else FLAGS.out_dir
  for i, objective in tqdm(enumerate(objectives), total=len(objectives)):
    with open(f'{model_dir}/{i}.pkl', 'rb') as f:
      model = pickle.load(f)
      models.append(model)
      #preds.append(model.inplace_predict(X_valid))


if preds:
  pred = ensemble(preds)
  ic(pred, pred.shape)
  df_valid = df_valid.with_columns(
    pl.Series("pred", pred)
  )
  score = eval_df(df_valid)
  ic('ori', score)
  df_valid = df_valid.with_columns(
    pl.Series("pred", pred)
  )
  df_valid = rerank(df_valid)
  score = eval_df(df_valid)
  ic('rerank', score)


preds = []
for model in tqdm(models):
  # pred = model.predict(dtest)
  # pred = model.inplace_predict(X_test)
  pred = batch_predict(model, X_test, inplace=True)
  preds.append(pred)
  
pred = ensemble(preds)
pred


def probs2rank(df):
  df = df.with_columns(
      pl.col('pred').rank(method='ordinal',
                          descending=True).over('ranker_id').cast(
                              pl.Int32).alias('selected')).select(
                                  ['Id', 'ranker_id', 'selected'])
  return df


df_test = df_test.with_columns(
    pl.Series("pred", pred)
)


sub = probs2rank(df_test)
sub


FLAGS.out_dir


df_test = df_test.with_columns(
    pl.Series("pred", preds[0])
)
sub_single = probs2rank(df_test)
sub_single


sub_single.write_parquet(f'{FLAGS.out_dir}/best_single.parquet')
sub.write_csv(f'submission.csv')




