try:
  from icecream import ic
except Exception:
    !pip install -q icecream --no-index --find-links=file:///kaggle/input/icecream


!pip install --upgrade transformers huggingface-hub tokenizers -q --no-cache-dir


import sys, os
import numpy as np
import pandas as pd
import json
import re
import six
import glob
import traceback
import inspect
from typing import Union
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from multiprocessing import cpu_count
from IPython.display import display
from tqdm.notebook import tqdm
from icecream import ic
import transformers
import tensorflow as tf
import torch
from torch import nn, einsum
import torch.nn.functional as F
ic(tf.__version__, torch.__version__)


tf.config.set_visible_devices([], 'GPU')


class FLAGS(object):
  # for tfrecords args, you could ignore
  seed = 1024
  batch_parse = False
  sparse_to_dense = True
  eval_keys = []
  incl_keys = []
  excl_keys = []
  recount_tfrecords = False  
  batch_sizes = []
  buffer_size = 1024
  buckets = None
  drop_remainder = None
  shard_by_files = True
  shuffle_batch = None
  shuffle_files = None
  num_dataset_threads = 0
  num_prefetch_batches = 1024
  repeat_then_shuffle = False
  length_index = 1
  length_key = None
  dynamic_pad = True
  cache = False
  cache_after_map = False
  fixed_random = False
  parallel_read_files = True
  padding_idx = 0
  dataset_keys = []
  dataset_excl_keys = []
  exclude_varlen_keys = False
  prefetch = None
  dataset_ordered = False
    
  torch = True
  keras = False
    
  # online==False means using n-fold split and train on fold 1,2, folds-1 while valid on fold 0
  # online==True means using all train data but still will valid on fold 0
  online = False  
  folds = 4
  fold = 0
  fold_seed = 1229
  root = '../input/asl-fingerspelling'
  working = '/kaggle/working'
  use_z = True  # use x,y,z if True
  norm_frames = True # norm frames using x - mean / std
  concat_frames = True # concat original and normalized frames
  add_pos = True # add abs frame pos, like 1/1000., 2/1000.
  sup_weight = 0.1 # for supplement dataset assigin weight 0.1
  
  train_files = []
  valid_files = []
      
  mix_sup = True # train & sup dataset
  vie = 5 # valid interval epochs 
  lr = 2e-3
  epochs = 400 
  batch_size = 256
  eval_batch_size = 256
  awp = False
  adv_start_epoch = None
  adv_lr = 0.2
  adv_eps = 0
  fp16 = True # notice fp16 could not be set True if using awp here, otherwise nan
  optimizer = 'Adam'
  opt_eps = 1e-6 
  scheduler = 'linear'
  # for model related configs
  encoder_layers = 17
  encoder_units = 200 
  n_frames = 320  
  distributed = False

  # for preprocess
  trunct_method = 'resize'

  # for aug
  flip_rate = 0.25
  resample_rate = 0.8
  temporal_mask_rate = 0.8
  temporal_mask_prob = 0.5
  temporal_mask_range = [0.1, 0.5]
  spatio_mask_rate = 0.
  spatio_mask_prob = 0.
  add_pos_before_resample = True
  shift_rate = 0.
  shift_range = [-0.05, 0.05]
  shift_method = 1
  temporal_seq_mask_rate = 0.5
  temporal_seq_mask_max = 2
  temporal_seq_mask_range = [0.1, 0.2]
  shift_rate = 0.75
  scale_method = 0
  scale_rate = 0.75
  rotate_rate = 0.75
  
  # for model
  ksize_vals = [15]
  skip_factor = 0.5
  mhatt_heads = 8
  mhatt_dimhead = 32

FLAGS.adv_start_epoch = int(FLAGS.epochs * 0.15)
    
def load_json(filename):
  with open(filename) as fh:
    obj = json.load(fh)
  return obj


FLAGS.batch_size = 128
FLAGS.acc_steps = 4 # gradient acc
FLAGS.batch_size = FLAGS.batch_size // FLAGS.acc_steps


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
ic(len(LIP))
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]
MID_LIP = [i for i in LIP if i not in LLIP + RLIP]
ic(len(LLIP), len(RLIP), len(MID_LIP))

NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
MID_NOSE = [i for i in NOSE if i not in LNOSE + RNOSE]

LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]

N_HAND_POINTS = 21
N_POSE_POINTS = len(LPOSE)
N_LIP_POINTS = len(LLIP)
N_EYE_POINTS = len(LEYE)
N_NOSE_POINTS = len(LNOSE)
N_MID_POINTS = len(MID_LIP + MID_NOSE)

SEL_COLS = []
for i in range(N_HAND_POINTS):
  SEL_COLS.extend([f'x_left_hand_{i}', f'y_left_hand_{i}', f'z_left_hand_{i}'])
for i in range(N_HAND_POINTS):
  SEL_COLS.extend([f'x_right_hand_{i}', f'y_right_hand_{i}', f'z_right_hand_{i}'])
for i in LPOSE:
  SEL_COLS.extend([f'x_pose_{i}', f'y_pose_{i}', f'z_pose_{i}'])
for i in RPOSE:
  SEL_COLS.extend([f'x_pose_{i}', f'y_pose_{i}', f'z_pose_{i}'])
for i in LLIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in RLIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])

for i in LEYE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in REYE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
  
for i in LNOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in RNOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
  
for i in MID_LIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in MID_NOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
    
N_COLS = len(SEL_COLS)
ic(N_COLS)
    
CHAR2IDX = load_json(f'../input/asl-fingerspelling/character_to_prediction_index.json')
CHAR2IDX = {k: v + 1 for k, v in CHAR2IDX.items()}
N_CHARS = len(CHAR2IDX)
ic(N_CHARS)

PAD_IDX = 0
SOS_IDX = PAD_IDX # Start Of Sentence
EOS_IDX = N_CHARS + 1 # End Of Sentence
ic(PAD_IDX, SOS_IDX, EOS_IDX)

PAD_TOKEN = '<PAD>'
SOS_TOKEN = PAD_TOKEN
EOS_TOKEN = '<EOS>'

CHAR2IDX[PAD_TOKEN] = PAD_IDX
CHAR2IDX[EOS_TOKEN] = EOS_IDX 

ADDRESS_TOKEN = '<ADDRESS>'
URL_TOKEN = '<URL>'
PHONE_TOKEN = '<PHONE>'
SUP_TOKEN = '<SUP>'

VOCAB_SIZE = len(CHAR2IDX)
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
ic(VOCAB_SIZE)
ic(len(IDX2CHAR))

STATS = {}
CLASSES = [
  'address', 
  'url', 
  'phone', 
  'sup',
  ]
PHRASE_TYPES = dict(zip(CLASSES, range(len(CLASSES))))
N_TYPES = len(CLASSES)
MAX_PHRASE_LEN = 32

def get_phrase_type(phrase):
  # Phone Number
  if re.match(r'^[\d+-]+$', phrase):
    return 'phone'
  # url
  elif any([substr in phrase for substr in ['www', '.', '/']
           ]) and ' ' not in phrase:
    return 'url'
  # Address
  else:
    return 'address'

def get_vocab_size():
  vocab_size = VOCAB_SIZE
  return vocab_size

def get_n_cols(no_motion=False, use_z=None):
  n_cols = N_COLS
  if use_z is None:
    use_z = FLAGS.use_z
  
  if FLAGS.concat_frames:
    assert FLAGS.norm_frames
    n_cols += N_COLS
  
  if not use_z:
    n_cols = n_cols // 3 * 2
    
  if FLAGS.add_pos:
    n_cols += 1
  
  return n_cols


def gen_inputs(files, 
           decode_fn, 
           batch_size=64,
           post_decode_fn=None,
           num_epochs = None, 
           num_threads=None, 
           buffer_size = 15000, #change from 1000 to 15000
           dynamic_pad=True,
           shuffle=True,
           shuffle_batch=None,
           shuffle_files=None,
           ordered=None,
           min_after_dequeue=None, #depreciated
           seed=None, 
           enqueue_many=False,  #depreciated
           fixed_random=False, 
           drop_remainder=False, 
           num_prefetch_batches=None, 
           bucket_boundaries=None,
           length_index=None,
           length_key=None,
           length_fn=None,
           bucket_batch_sizes=None,
           repeat=True,
           initializable=False,
           filter_fn=None,
           balance_pos_neg=False,
           pos_filter_fn=None,
           neg_filter_fn=None,
           count_fn=None,
           return_iterator=False,
           Dataset=None,
           batch_parse=False, #by default will be line parse
           hvd_shard=True,
           shard_by_files=False,
           training=False,
           simple_parse=False,
           repeat_then_shuffle=False,
           cache=False,
           cache_file='',
           cache_after_map=False,
           device=None,
           world_size=1,
           rank=0,
           parallel_read_files=False,
           use_feed_dict=False,
           feed_name=None,
           padding_values=None,
           distribute_strategy=None,
           torch=False,
           keras=False,
           subset=None,
           return_numpy=False,
           name='input'):
  Dataset = Dataset or tf.data.TFRecordDataset
  AUTO = tf.data.AUTOTUNE
  use_horovod = False

  def shard(d):
    return d.shard(hvd.size(), hvd.rank())

  # Choose to use cpu outside input function like in dataset.py
  #with tf.device('/cpu:0'):
  if isinstance(files, str):
    files = gezi.list_files(files)
  assert len(files) > 0

  if not num_threads:
    num_threads = 8

  if 'batch_size' in inspect.getfullargspec(decode_fn).args:
    decode_fn_ = decode_fn
    def decode_function(example):
      return decode_fn_(example, batch_size)
    decode_fn = decode_function
    
  if not num_epochs: 
    num_epochs = None

  if shuffle:
    if shuffle_files is None:
      shuffle_files = True
    if shuffle_batch is None:
      shuffle_batch = True
  else:
    if shuffle_files is None:
      shuffle_files = False
    if shuffle_batch is None:
      shuffle_batch = False
    # TDO 并行读取就会打乱顺序？
    if not shuffle_files:
      parallel_read_files = False

  if fixed_random:
    if seed is None:
      seed = 1024
  else:
    pass

  num_files = len(files)
  if use_feed_dict and feed_name:
    files = tf.compat.v1.placeholder(tf.string, [None], feed_name)
    gezi.set_global(feed_name, files)

  if not num_prefetch_batches:
    #num_prefetch_batches = num_threads + 3
    if buffer_size:
      num_prefetch_batches = int(buffer_size / batch_size)
    # else:
    #   num_prefetch_batches = 100
  
  if not buffer_size and num_prefetch_batches:
    buffer_size = num_prefetch_batches * batch_size
    
  options = tf.data.Options()
  try:
    options.threading.private_threadpool_size = num_threads
    options.threading.max_intra_op_parallelism = 1
  except Exception:
    options.experimental_threading.private_threadpool_size = num_threads
    options.experimental_threading.max_intra_op_parallelism = 1

  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  options.experimental_deterministic = True

  if shuffle and not fixed_random:
    options.experimental_deterministic = False

  if not ordered:
    options.experimental_deterministic = False

  if not parallel_read_files or num_files == 1:
    d = Dataset(files)
    d = d.with_options(options)
    if use_horovod and hvd_shard:
      d = shard(d)
    if not use_horovod and world_size > 1:
      d = d.shard(world_size, rank)
  else:
    try:
      if shffle_files and (use_horovod or world_size > 1):
        assert seed
      d = tf.data.Dataset.list_files(files, shuffle=shuffle_files, seed=seed)
      d = d.with_options(options)
    except Exception:
      d = tf.data.Dataset.from_tensor_slices(files)
      d = d.with_options(options)
    # here shard by files, not work good, especially for text line dataset with horovod
    if use_horovod and shard_by_files:
      d = shard(d)
    elif world_size > 1 and shard_by_files:
      d = d.shard(world_size, rank)

    d = d.interleave(Dataset,
                  #  cycle_length=min(len(files), 1000),  # in tf 1.14 must set and can not set as AUTOTUNE for tf 2.1 with default as AUTOTUNE
                  block_length=1,
                  num_parallel_calls=AUTO)

    if world_size > 1 and not shard_by_files:
      d = d.shard(world_size, rank)

  if repeat and repeat_then_shuffle:
    d = d.repeat(num_epochs)

  if cache and (not FLAGS.cache_after_map):
    d = d.cache(cache_file)
    
  # must batch then map if use pyfunc which you might use batch_parse, here batch_parse means batch parse otherwise slower but simple and powerfull...
  if not batch_parse:
    d = d.map(decode_fn, num_parallel_calls=AUTO)
    if cache and cache_after_map:
      d = d.cache(cache_file)
  
  if filter_fn is not None and not batch_parse:
    d = d.filter(filter_fn)

  if shuffle_batch:
    d = d.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

  # shuffle then repeat
  if repeat and not repeat_then_shuffle:
    d = d.repeat(num_epochs)
  
  if dynamic_pad:
    if not batch_parse: 
      d = d.padded_batch(batch_size, drop_remainder=drop_remainder)
    else:
      d = d.batch(batch_size, drop_remainder=drop_remainder)
  else:
    d = d.batch(batch_size, drop_remainder=drop_remainder)

  if batch_parse:
    d = d.map(decode_fn, num_parallel_calls=AUTO)
    if filter_fn is not None:
      try:
        d = d.unbatch()
        d = d.filter(filter_fn)
      except Exception:
        d = d.unbatch()

      d = d.batch(batch_size, drop_remainder=drop_remainder)

  if post_decode_fn is not None:
    d = d.map(post_decode_fn, num_parallel_calls=AUTO)

  if cache and FLAGS.cache_after_map:
    logging.debug('Cache datase after map')
    d = d.cache(cache_file)

  d = d.prefetch(FLAGS.prefetch or AUTO)

  if not return_numpy:    
    return d
  else:
    return d.as_numpy_iterator()



def decode_example(x):
  if tf.executing_eagerly():
    x = x.numpy()
  x = tf.train.Example.FromString(x).features.feature
  features = {}
  for key in x.keys():
    typenames = ['bytes_list', 'float_list', 'int64_list']
    dtypes = [object, np.float32, np.int64]
    for typename, dtype in zip(typenames, dtypes):
      value = getattr(x[key], typename).value
      if value:
        features[key] = np.array(value, dtype=dtype)
  return features

def first_example(record_file):
  if isinstance(record_file, (list, tuple)):
    record_file = record_file[0]
  if tf.executing_eagerly():
    for item in tf.data.TFRecordDataset(record_file):
      x = decode_example(item)
      return x
  else:
    for item in tf.compat.v1.python_io.tf_record_iterator(record_file):
      x = decode_example(item)
      return x

def npdtype2tfdtype(dtype, large=False):
  if dtype == np.float32:
    return tf.float32
  if dtype == np.int32:
    if not large:
      return tf.int32
    else:
      return tf.int64
  if dtype == np.int64:
    return tf.int64
  if dtype == np.float64:
    return tf.float32
  return tf.string

def sparse_tensor_to_dense(input_tensor, default_value=0):  
  return tf.sparse.to_dense(input_tensor, default_value=default_value, validate_indices=False)

def sparse2dense(features, key=None, default_value=0):

  def sparse2dense_(features, key, default_value):
    val = features[key]
    if val.values.dtype == tf.string:
      default_value = None
    val = sparse_tensor_to_dense(val, default_value)
    features[key] = val

  modified = False
  if key:
    sparse2dense_(features, key)
    modified = True
  else:
    from tensorflow.python.framework.sparse_tensor import SparseTensor
    for key, val in features.items():
      if isinstance(val, SparseTensor):
        sparse2dense_(features, key, default_value)
        modified = True
  return modified

def get_num_records_single(tf_record_file, recount=False):
  if not recount:
    filename = os.path.basename(tf_record_file)
    filename = filename.replace('-', '.').replace('_', '.')
    l = filename.split('.')

    for item in reversed(l):
      if item.isdigit():
        return int(item)

  # try:
  return sum(
      1 for _ in tf.compat.v1.python_io.tf_record_iterator(tf_record_file))
  # except Exception:
  #   return 0


def get_num_records(files, recount=False):
  if isinstance(files, str):
    files = gezi.list_files(files)
  res = sum([
      get_num_records_single(file, recount=recount)
      for file in tqdm(files, ascii=False, desc='get_num_records', leave=False)
  ])
  return res



# A wrapper base class for tfrecords related dataset 
class TfrecordsDataset(object):
  def __init__(self, 
               subset='valid',
               batch_size=None,
               Type=None, 
               files=None,
               num_instances=None,
               batch_parse=None,
               sparse_to_dense=None,
               hvd_shard=True,
               use_int32=True,
               is_info=False,
               eval_keys=[],
               incl_keys=[],
               excl_keys=[],
               str_keys=[],
               varlen_keys=[],
               use_tpu=False,
               recount=None):
    self.subset = subset
    self.filter_fn = None
    self.pos_filter_fn = None
    self.neg_filter_fn = None 
    self.count_fn = None
    self.Type = Type
    self.batch_parse = batch_parse if batch_parse is not None else FLAGS.batch_parse
    self.sparse_to_dense = sparse_to_dense if sparse_to_dense is not None else FLAGS.sparse_to_dense
    self.use_post_decode = None
    # if self.batch_parse:
    #   self.sparse_to_dense = False
    self.batch_size = batch_size or FLAGS.batch_size
    self.hvd_shard = hvd_shard
    self.indexes = {'train': -1, 'valid': -1, 'test': -1}
    self.is_info = is_info
    self.eval_keys = eval_keys or FLAGS.eval_keys
    if subset == 'test':
      self.eval_keys = gezi.get('test_keys') or self.eval_keys
    self.show_keys = set()  # 如果用户不指定eval_keys 可以用self.show_keys所有非变成以及长度为0,1的key 前提需要使用.adds不能自己外部定义
    self.excl_keys = excl_keys or FLAGS.excl_keys
    self.incl_keys = incl_keys or FLAGS.incl_keys
    self.str_keys = str_keys
    self.varlen_keys = varlen_keys

    self.parse_fn = tf.io.parse_single_example if not self.batch_parse else tf.io.parse_example

    self.features_dict = {}
    self.has_varlen_feats = False
    self.use_tpu = use_tpu
    try:
      # TPU detection. No parameters necessary if TPU_NAME environment variable is
      # set: this is always the case on Kaggle.
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      # print('Running on TPU ', tpu.master())
    except ValueError:
      tpu = None
    if tpu is not None:
      self.use_tpu = True
    self.use_int32 = use_int32
    if self.use_tpu:
      self.use_int32 = True

    self.num_instances_ = num_instances
    self.files_ = files

    # self.use_post_decode = use_post_decode
    self.recount = recount or FLAGS.recount_tfrecords

    assert self.subset in ['train', 'valid', 'test'], \
          'subset is {} but should in [train, valid, test]'.format(self.subset)

  @staticmethod
  def get_filenames_(subset=None, shuffle=False):
    try:
      if subset in ['train', 'valid', 'test']:
        if subset == 'train':
          return FLAGS.train_files
        elif subset == 'valid':
          return FLAGS.valid_files
      else:
        raise ValueError('Invalid data subset "%s"' % subset)
    except Exception:
      return None

  def get_filenames(self, subset=None, shuffle=False):
    subset = subset or self.subset
    return TfrecordsDataset.get_filenames_(subset, shuffle=False)

  def basic_parse(self, example):
    self.auto_parse(keys=self.incl_keys, exclude_keys=self.excl_keys)
    if self.varlen_keys:
      self.adds_varlens(self.varlen_keys)
    fe = self.parse_(serialized=example)
    return fe
  
  # override this
  def parse(self, example):
    return self.basic_parse(example)

  def decode(self, example):
    l = self.parse(example)
    
    if isinstance(l, (list, tuple)):
      features = l[0]
    else:
      features = l
    # self.use_tpu = True
    if isinstance(features, dict):
      if self.use_tpu:
        def decode_label(label):
          label = tf.io.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
          label = tf.reshape(label, [])  # label is a scalar
          return tf.cast(label, tf.int32) 
        for key in features.keys():
          if features[key].dtype in [tf.int64, tf.uint8, tf.uint16, tf.uint32]:
            features[key] = tf.cast(features[key], tf.int32)

        if not self.is_info:
          keys = list(features.keys())
          for key in keys:
            if features[key].dtype ==tf.string:
              del features[key]

              if key in self.eval_keys:
                FLAGS.use_info_dataset = True  # 因为训练model的dataset不再包含eval的某个信息 需要依赖再遍历一遍info_dataset
              # features[key] = tf.ones_like(features[key], tf.int32)
              # features[key] = decode_label(features[key]) ## not work TODO

      else:
        def _cast_dict(features):
          for key in features:
            if isinstance(features[key], dict):
              _cast_dict(features[key])
            else:
              # tf.print(key, features[key])
              if features[key].dtype == tf.int64 and self.use_int32:
                features[key] = tf.cast(features[key], tf.int32)
        _cast_dict(features)
 
      # is_info 只在tf2 keras模式下生效, 都创建 但是有可能不用 只有 FLAGS.use_info_dataset = True 才使用
      if self.is_info:
        keys = list(features.keys())
        if not FLAGS.predict_on_batch:
          if not self.eval_keys:
            for key in keys:
              dim = 1 if self.batch_parse else 0
              if not (len(features[key].shape) == dim or features[key].shape[dim] == 1):
                del features[key]
              else:
                self.show_keys.add(key)
          else:
            for key in keys:
              if key not in self.eval_keys:
                del features[key]
      else:
        keys = list(features.keys())
        for key in keys:
          if key in self.excl_keys:
            del features[key]

    return l

  def adjust(self, result):
    return result

  def parse_(self, serialized, features=None):
    features = features or self.features_dict
    # ic(features)
    features = self.parse_fn(serialized=serialized, features=features)
    if FLAGS.exclude_varlen_keys:
      from tensorflow.python.framework.sparse_tensor import SparseTensor
      sparse_keys = [key for key in features if isinstance(key, SparseTensor)]
      for key in sparse_keys:
        del features[key]
    else:
      if self.sparse_to_dense:
        modified = sparse2dense(features, default_value=FLAGS.padding_idx)
        self.has_varlen_feats = modified
    self.features = features
    return features
  
  def gen_example(self, files=None):
    if not files:
      files = self.get_filenames()
    if not isinstance(files, (list, tuple)):
      files = [files]
    example = {}
    if files:
      for file in files:
        try:
          example = first_example(file)
        except Exception:
          ic(traceback.format_exc())
          ic('bad tfrecord:', file)
        if example:
          self.example = example
          break
    self.example = example
    return example

  def gen_input(self, files=None):
    example = self.gen_example().copy()
    for key in example:
      example[key] = np.asarray([example[key]])
    return example

  def first_input(self, files=None):
    return self.gen_input(files)

  def add(self, key, dtype=None, length=None, features_dict=None):
    features_dict = features_dict or self.features_dict
    dtype_ = dtype
    if key in self.example:
      dtype = dtype_ or self.example[key].dtype 
      if length is None:
        features_dict[key] = tf.io.VarLenFeature(dtype)
      elif length > 0:
        features_dict[key] = tf.io.FixedLenFeature([length], dtype)
      else:
        features_dict[key] = tf.io.FixedLenFeature([], dtype)
    
  def adds(self, keys, dtype=None, length=None, features_dict=None):
    features_dict = features_dict or self.features_dict
    dtype_ = dtype
    for key in keys:
      if key in self.example:
        dtype = dtype_ or self.example[key].dtype 
        if length is None:
          features_dict[key] = tf.io.VarLenFeature(dtype)
        elif length > 0:
          features_dict[key] = tf.io.FixedLenFeature([length], dtype)
        else:
          features_dict[key] = tf.io.FixedLenFeature([], dtype)

  def auto_parse(self, keys=[], exclude_keys=[], features_dict=None):
    keys = keys or FLAGS.dataset_keys or self.example.keys()
    exclude_keys = exclude_keys or FLAGS.dataset_excl_keys
    keys = [key for key in keys if key not in exclude_keys]

    for key in keys:
      if key not in self.example:
        continue
      length = self.example[key].shape[0]
      
      if length == 1:
        # just to (bs,), tf keras will auto change to (bs,1), also for string 0 is ok
        length = 0 

      dtype = npdtype2tfdtype(self.example[key].dtype)
      # print(key, dtype, length, self.example[key])
      self.adds([key], dtype, length, features_dict)

  def adds_varlens(self, keys=[], exclude_keys=[], features_dict=None):
    keys = keys or self.example.keys()
    keys = [key for key in keys if key not in exclude_keys]

    for key in keys:
      if not key in self.example:
        continue
      length = self.example[key].shape[0]
      dtype = npdtype2tfdtype(self.example[key].dtype)
      length = None
      if dtype == tf.string:
        length = 1
      self.adds([key], dtype, length, features_dict)  
  
  def make_batch(self, 
                 batch_size=None, 
                 filenames=None,
                 subset=None,
                 initializable=False,
                 repeat=None,
                 shuffle=None,
                 return_iterator=True,
                 hvd_shard=None,
                 simple_parse=False,
                 num_epochs=None,
                 cache=False,
                 cache_file='',
                 buffer_size=None,
                 batch_sizes=None,
                 buckets=None,
                 drop_remainder=None,
                 world_size=1,
                 rank=0,
                 shard_by_files=None,
                 distribute_strategy=None,
                 return_numpy=False):
    # with tf.device('/cpu:0'):
    subset = subset or self.subset
    hvd_shard = hvd_shard if hvd_shard is not None else self.hvd_shard
    if batch_size is None:
      is_test = True
    else:
      is_test = False
    batch_size = batch_size or self.batch_size
    self.batch_size = batch_size
    batch_sizes = batch_sizes if batch_sizes is not None else FLAGS.batch_sizes
    buffer_size = buffer_size if buffer_size is not None else FLAGS.buffer_size
    buckets = buckets if buckets is not None else FLAGS.buckets
    drop_remainder = drop_remainder if drop_remainder is not None else FLAGS.drop_remainder
    shard_by_files = shard_by_files if shard_by_files is not None else FLAGS.shard_by_files

    self.return_numpy = return_numpy

    filenames = filenames or self.files_ or self.get_filenames(subset)
    
    self.gen_example(filenames)

    is_eager = tf.executing_eagerly()

    self.files_ = filenames

    self.indexes[self.subset] += 1
    
    if repeat is None:
      num_gpus = 1
      # if subset == 'train' or num_gpus > 1:
      if subset == 'train':
        repeat = True
      else:
        repeat = False
      if is_eager and num_gpus == 1 and tf.__version__ < '2':
        # let tf eager similary to pytorch
        repeat = False

    if shuffle is None:
      if subset == 'train':
        shuffle = FLAGS.shuffle 
      else:
        shuffle = FLAGS.shuffle_valid 

    if drop_remainder is None:
      if subset == 'train':
        drop_remainder = True
      else:
        drop_remainder = False

    balance_pos_neg=False
    ic(self.subset, repeat, drop_remainder)

    seed = FLAGS.seed 
    if seed is not None:
      FLAGS.seed += 1

    ## put on cpu or dummy
    with tf.device('/cpu'):
      result = gen_inputs(
        filenames, 
        decode_fn=self.decode,
        batch_size=batch_size,
        post_decode_fn=self.post_decode if hasattr(self, 'post_decode') and self.use_post_decode != False else None,
        shuffle=shuffle,
        shuffle_batch=FLAGS.shuffle_batch,
        shuffle_files=FLAGS.shuffle_files,
        ordered=FLAGS.dataset_ordered if subset == 'train' else True,
        num_threads=FLAGS.num_dataset_threads,
        buffer_size=buffer_size,
        num_prefetch_batches=FLAGS.num_prefetch_batches,
        initializable=initializable,
        repeat=repeat,
        repeat_then_shuffle=FLAGS.repeat_then_shuffle,
        drop_remainder=drop_remainder,
        bucket_boundaries=buckets,
        bucket_batch_sizes=batch_sizes,
        length_index=FLAGS.length_index,
        length_key=FLAGS.length_key,
        seed=seed,
        return_iterator=return_iterator,
        filter_fn=self.filter_fn,  # inside filter_fn judge subset train or valid or test
        balance_pos_neg=balance_pos_neg,
        pos_filter_fn=self.pos_filter_fn if subset == 'train' else None,
        neg_filter_fn=self.neg_filter_fn if subset == 'train' else None,
        count_fn=self.count_fn if subset == 'train' else None,
        name=subset,
        Dataset=self.Type,
        batch_parse=self.batch_parse,
        hvd_shard=hvd_shard,
        shard_by_files=shard_by_files,
        training=subset == 'train',
        simple_parse=simple_parse,
        num_epochs=num_epochs,
        dynamic_pad=FLAGS.dynamic_pad, #如果有varlen feats才需要 padded_batch 同时batch_parse模式其实也不需要因为sparse2dense就可以自动padd
        cache=cache,
        cache_file=cache_file,
        cache_after_map=FLAGS.cache_after_map,
        device='/gpu:0',
        world_size=world_size,
        rank=rank,
        fixed_random=FLAGS.fixed_random,
        parallel_read_files=FLAGS.parallel_read_files,
        #use_feed_dict=FLAGS.train_loop and FLAGS.rounds > 1 and not is_eager and FLAGS.feed_dataset and tf.__version__ < '2',
        feed_name=f'{self.subset}_{self.indexes[self.subset]}' if not is_test else None,
        padding_values=FLAGS.padding_idx, 
        distribute_strategy=distribute_strategy,
        torch=FLAGS.torch,
        keras=FLAGS.keras,
        subset=self.subset,
        return_numpy=return_numpy,
        ) 
      
    result = self.adjust(result)
    return result
    
  @staticmethod
  def num_examples_per_epoch(subset, dir=None):
    if subset == 'train':
      num_examples = get_num_records(FLAGS.train_files)
    elif subset == 'valid':
      num_examples = get_num_records(FLAGS.valid_files)
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
    
    assert num_examples
    return num_examples

  @staticmethod
  def num_examples(subset, dir=None):
    return Dataset.num_examples_per_epoch(subset, dir)

  @property
  def num_instances(self):
    if self.num_instances_:
      return self.num_instances_
    assert self.files_
    self.num_instances_ = get_num_records(self.files_, recount=self.recount)
    return self.num_instances_

  @property
  def files(self):
    return self.files_

  @property
  def records(self):
    return self.files_

  def __len__(self):
    return self.num_instances or Dataset.num_examples_per_epoch(self.subset)

  @property
  def num_steps(self):
    return -(-len(self) // self.batch_size)



def apply(func, x, p=0.5):
  if p <= 0:
    return x
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)

class Apply(object):
  def __init__(self, func, p):
    self.func = func
    self.p = p
  
  def __call__(self, x):
    return apply(self.func, x, self.p)

def oneof(funcs, x, p=1.):
  if p <= 0:
    return x
  num_choices = len(funcs)
  prob = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
  for i in range(num_choices):
    if prob <= float(i + 1) * p / float(num_choices):
      x = funcs[i](x)
  return x

class OneOf(object):
  def __init__(self, funcs, p=1.):
    self.funcs = funcs
    self.p = p

  def __call__(self, x):
    return oneof(self.funcs, x, self.p)

def reshape(data):
  data = tf.reshape(data, (1, -1, N_COLS // 3, 3))
  return data

def reshape_back(data):
  data = tf.reshape(data, (1, -1, N_COLS))
  return data

def try_reshape(data):
  if tf.shape(data)[-1] == N_COLS:
    data = reshape(data)
  return data

def splits(data):
  l = [
      N_HAND_POINTS, N_HAND_POINTS, 
      N_POSE_POINTS, N_POSE_POINTS, 
      N_LIP_POINTS, N_LIP_POINTS,
      N_EYE_POINTS, N_EYE_POINTS,
      N_NOSE_POINTS, N_NOSE_POINTS,
      N_MID_POINTS, 
  ]
  return tf.split(data, l, axis=-2)

def split(data):
  x, y, z = tf.split(data, 3, axis=-1)
  return x, y, z

def concat(x, y, z):
  data = tf.concat([x, y, z], axis=-1)
  return data

def flip_lr(data):
  x, y, z = split(data)
  x = 1. - x
  data = concat(x, y, z)
  # lhand, rhand, lpose, rpose, llip, rlip = splits(data)
  # data = tf.concat([rhand, lhand, rpose, lpose, rlip, llip], axis=-2)
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  data = tf.concat([rhand, lhand, rpose, lpose, rlip, llip, reye, leye, rnose, lnose, mid], axis=-2)
  return data

def zero_mean(x, axis=0, keepdims=False):
  upper = tf.reduce_sum(x, axis=axis, keepdims=keepdims)
  bottom = tf.reduce_sum(tf.where(tf.math.equal(x, 0.), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)
  return upper / bottom

def zero_std(x, center=None, axis=0, keepdims=False):
  if center is None:
    center = zero_mean(x, axis=axis, keepdims=True)
  d = x - center
  return tf.math.sqrt(zero_mean(d * d, axis=axis, keepdims=keepdims))

def interp1d(x, target_len, method='random'):
  target_len = tf.maximum(1, target_len)
  if method == 'random':
    if tf.random.uniform(()) < 0.33:
      x = tf.image.resize(x, (1, target_len), 'bilinear')
    else:
      if tf.random.uniform(()) < 0.5:
        x = tf.image.resize(x, (1, target_len), 'bicubic')
      else:
        x = tf.image.resize(x, (1, target_len), 'nearest')
  else:
    x = tf.image.resize(x, (1, target_len), method)
  x = tf.reshape(x, (1, target_len, -1))
  return x

# x [None, n_frames, n_cols]
def resample(x, rate=(0.5, 1.5)):
  rate = tf.random.uniform((), rate[0], rate[1])
  length = tf.shape(x)[1]
  new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
  new_x = interp1d(x, new_size)
  return new_x

# x [None, n_frames, n_cols]
def temporal_mask(x):
  mask_prob = FLAGS.temporal_mask_prob if not FLAGS.temporal_mask_range else tf.random.uniform((), *FLAGS.temporal_mask_range)
  mask = tf.random.uniform((tf.shape(x)[0], tf.shape(x)[1], 1))  > mask_prob
  mask = tf.cast(mask, dtype=x.dtype)
  new_x = x * mask
  return new_x

# x [1, n_frames, n_cols//3, 3]
def temporal_seq_mask(x):
  x = x[0]
  mask_value = float('nan')
  l = tf.shape(x)[0]
  mask_size = tf.random.uniform((), *FLAGS.temporal_seq_mask_range)
  mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
  mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l-mask_size,1,l), dtype=tf.int32)
  new_x = tf.tensor_scatter_nd_update(x, 
                                      tf.range(mask_offset, mask_offset+mask_size)[...,None],
                                      tf.fill([mask_size, tf.shape(x)[-2], tf.shape(x)[-1]],
                                      mask_value))
  new_x = new_x[None]
  return new_x


# x [None, n_frames, n_cols]
def spatio_mask(x):
  mask = tf.random.uniform((tf.shape(x)[0], 1, tf.shape(x)[2]))  > FLAGS.spatio_mask_prob
  mask = tf.cast(mask, dtype=x.dtype)
  new_x = x * mask
  return new_x

def shift(data):
  range_ = FLAGS.shift_range
  if FLAGS.shift_method == 0:
    shift_ = tf.random.uniform((),*range_)
    data += shift_
  elif FLAGS.shift_method == 1:
    shift_x = tf.random.uniform((),*range_)
    shift_y = tf.random.uniform((),*range_)
    shift_z = tf.random.uniform((),*range_)
    x, y, z = split(data)
    x += shift_x
    y += shift_y
    z += shift_z
    data = concat(x, y, z)
  else:
    raise ValueError('not supported shift method')
  return data

def scale(data):
  scale_ = tf.random.uniform((), *FLAGS.scale_range)
  if FLAGS.scale_method == 0:
    data *= scale_
  elif FLAGS.scale_method == 1:
    data -= 0.5
    data *= scale_
    data += 0.5
  return data

def rotate(data):
  data -= 0.5
  degree = tf.random.uniform((), *FLAGS.rotate_range)
  radian = degree / 180 * np.pi
  c = tf.math.cos(radian)
  s = tf.math.sin(radian)
  rotate_mat = tf.identity([
            [c,s],
            [-s, c],
        ])
  
  x, y, z = split(data)
  xy = tf.concat([x, y], axis=-1)
  # [None, 88, 2]
  # tf.print(xy.shape)
  xy = xy @ rotate_mat
  
  data = tf.concat([xy, z], axis=-1)  
  data += 0.5
  return data
  
def norm_frame(data):  
  x, y, z = split(data)
  x_mean = zero_mean(x, axis=-1, keepdims=True)
  y_mean = zero_mean(y, axis=-1, keepdims=True)
  z_mean = zero_mean(z, axis=-1, keepdims=True)
  x_std = zero_std(x, center=x_mean, axis=-1, keepdims=True)
  y_std = zero_std(y, center=y_mean, axis=-1, keepdims=True)
  z_std = zero_std(z, center=z_mean, axis=-1, keepdims=True)

  x = x - x_mean / x_std
  y = y - y_mean / y_std
  z = z - z_mean / z_std
  data = concat(x, y, z)
  return data

# [1, n_frames, n_cols//3, 3]
def norm_hands(data):  
  # lhand, rhand, lpose, rpose, llip, rlip = splits(data)
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  lhand = lhand[:,:,1:] - lhand[:,:,0:1]
  rhand = rhand[:,:,1:] - rhand[:,:,0:1]
  data = tf.concat([lhand, rhand], axis=2)
  if FLAGS.norm_hands_size:
    max_val = tf.math.maximum(tf.reduce_max(tf.abs(data)), 0.0001)
    data /= max_val
  return data

def add_motion(data):
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  hand = tf.concat([lhand, rhand], axis=2)
  hand_motion = tf.concat([tf.zeros_like(hand[:, :1]), hand[:, 1:] - hand[:, :-1]], axis=1)
  hand_motion2 = tf.concat([tf.zeros_like(hand[:, :2]), hand[:, 2:] - hand[:, :-2]], axis=1)
  return tf.concat([hand_motion, hand_motion2], axis=2)
  
def get_pos(n_frames, dtype):
  pos = tf.range(n_frames, dtype=dtype)
  pos /= 1000.
  pos = tf.reshape(pos, [1, n_frames, 1])
  return pos


"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
    [None, N_COLS] -> [1, NONE, N_COLS] -> [1, N_FRAMES, N_COLS]
"""
class PreprocessLayer(tf.keras.layers.Layer):

  def __init__(self, n_frames=128, training=False):
    super(PreprocessLayer, self).__init__()
    self.n_frames = n_frames
    self.n_cols = get_n_cols(no_motion=True, use_z=True)
    ic(self.n_cols)
    self.means = np.load(f'../input/3rd-place-step-3-gen-mean-and-std/means.npy')
    self.stds = np.load(f'../input/3rd-place-step-3-gen-mean-and-std/stds.npy')
    ic(self.means.shape, self.stds.shape)
    self.training = training

  ## seems not needed as training perf is similar
  @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_COLS],
                                              dtype=tf.float32),),)
  ## this will cause error
  # @tf.function()
  def call(self, data):
    trunct_method = FLAGS.trunct_method
    training = self.training

    # Hacky (add dim in front line shape [None,N_COLS] to shape [1, None, N_COLS])
    data = data[None]
    dtype = data.dtype
    
    # [1, None, N_COLS//3, 3]
    data = reshape(data)
    
    if training and FLAGS.use_aug:
      data = Apply(scale, FLAGS.scale_rate)(data)
      data = Apply(rotate, FLAGS.rotate_rate)(data)
      # shift not affect much..
      data = Apply(shift, FLAGS.shift_rate)(data)
      data = OneOf([scale, rotate, shift], FLAGS.affine_rate)(data)
      data = Apply(temporal_seq_mask, FLAGS.temporal_seq_mask_rate)(data)
    
    data_ = data
    data = reshape_back(data)
      
    # seems norm before resize produce better results
    # -------norm frames
    l = []
    if FLAGS.norm_frames:
      if FLAGS.concat_frames:
        l.append(data_)
      l.append(
          (data - self.means) / self.stds,
      )
      l[-1] = reshape(l[-1])
    else:
      l.append(data_)
            
    # [1, None, feats, 3]
    data = tf.concat(l, axis=-2)

    n_cols = self.n_cols if not FLAGS.add_pos else self.n_cols - 1
    data = tf.reshape(data, [1, -1, n_cols])
    # Fill NaN Values With 0
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
                       
    # add_pos helps a lot and add_pos_before_resample also +0.002 compare to add_pos_after_resample
    if FLAGS.add_pos and FLAGS.add_pos_before_resample:
      # n_frames = len(data[0])
      n_frames = tf.shape(data)[1]
      pos = get_pos(n_frames, dtype)
      data = tf.concat([data, pos], axis=-1)
      
    #  now basic features done, we resample for frames with basic features
    if training and FLAGS.use_aug:
      # resample for time range, like 120 frames to 135 or 90 frames
      # resample helps a lot
      data = Apply(resample, FLAGS.resample_rate)(data)
      data = tf.cast(data, dtype)
      
    # ------add addional info like add pos after resample, actually similar results before or after resample
    # n_frames = len(data[0])
    n_frames = tf.shape(data)[1]
    
    # do it after expand dim 0, and do not use expand_dims before...
    if FLAGS.add_pos and (not FLAGS.add_pos_before_resample):
      pos = get_pos(n_frames, dtype)
      data = tf.concat([data, pos], axis=-1)

    if not FLAGS.always_resize:
      # Pad Zeros  TODO dynamic type so not pad? dynanmic seq2seq input
      # n_frames = len(data[0])
      n_frames = tf.shape(data)[1]
      if n_frames < self.n_frames:
        if FLAGS.pad_frames:
          if FLAGS.pad_method == 'zero':
            data = tf.concat((data,
                              tf.zeros([1, self.n_frames - n_frames, self.n_cols],
                                        dtype=dtype)),
                            axis=1)
          else:
            data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=FLAGS.pad_resize_method,
            )
        elif n_frames < FLAGS.encode_pool_size:
          data = tf.concat(
              (data,
               tf.zeros([1, FLAGS.encode_pool_size - n_frames, self.n_cols],
                        dtype=dtype)),
              axis=1)
        elif n_frames < FLAGS.min_frames:
          data = tf.concat(
              (data,
               tf.zeros([1, FLAGS.min_frames - n_frames, self.n_cols],
                        dtype=dtype)),
              axis=1)

      # n_frames = len(data[0])
      n_frames = tf.shape(data)[1]
      if n_frames > self.n_frames:
        if trunct_method == 'resize':
          # Downsample
          data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=FLAGS.resize_method,
          )
          # For resize The return value has type float32, unless the method is ResizeMethod.NEAREST_NEIGHBOR
          data = tf.cast(data, dtype)
        else:
          data = data[:, :self.n_frames, :]
    else:
      data = tf.image.resize(
          data,
          [1, self.n_frames],
          method=FLAGS.resize_method,
      )
      data = tf.cast(data, dtype)

    if FLAGS.pad_frames:
      data = tf.reshape(data, [1, self.n_frames, self.n_cols])
      
    # 这个mask是否放在最后最好 是否放在前面 然后filter 全0的帧？那样比较麻烦 感觉这样还行...
    if training and FLAGS.use_aug:
      # temperal mask help a lot
      data = Apply(temporal_mask, FLAGS.temporal_mask_rate)(data)
      data = Apply(spatio_mask, FLAGS.spatio_mask_rate)(data)
      
    if not FLAGS.use_z:
      if FLAGS.add_pos:
        pos_data = data[...,-2:-1]
        data = data[...,:-1]
      data = tf.reshape(data, [1, self.n_frames, -1, 3])
      data = data[...,:2]
      data = tf.reshape(data, [1, self.n_frames, -1])
      if FLAGS.add_pos:
        data = tf.concat([data, pos_data], axis=-1)
             
    # Squeeze Batch Dimension
    data = tf.squeeze(data, axis=[0])
    # tf.print('----', data.shape)
    return data

class PreProcssor(object):
  def __init__(self, subset='valid', squeeze=False):
    training = subset == 'train'
    self.prepocess = PreprocessLayer(FLAGS.n_frames, training=training)
    self.squeeze = squeeze

  def __call__(self, fe):
    # # TODO HACK for hug datasets.. ds = ds.to_tf_dataset(batch_size=1)
    if self.squeeze:
      for key in fe:
        fe[key] = tf.squeeze(fe[key], axis=0)
    x = fe
    weights = fe['weight']
    fe['frames'] = tf.reshape(fe['frames'], [fe['n_frames'], N_COLS])
    fe['frames'] = self.prepocess(fe['frames']) 
    fe['phrase_len'] = tf.cond(tf.greater(fe['phrase_len'], MAX_PHRASE_LEN), lambda: tf.constant(MAX_PHRASE_LEN, fe['phrase_len'].dtype), lambda: fe['phrase_len'])
    if FLAGS.no_eos:
      # as in tfrecords we have eos, but in training if we don't have eos we change it to pad
      mask = tf.logical_or(fe['phrase_'] == PAD_IDX, fe['phrase_'] == EOS_IDX)
      mask = tf.cast(mask, fe['phrase_'].dtype)
      fe['phrase_'] = fe['phrase_'] * (1 - mask) + mask * PAD_IDX
        
    if FLAGS.mix_sup:
      weights_mask = tf.cast(weights != 1, tf.float32)
      weights = weights_mask * FLAGS.sup_weight + (1 - weights_mask)
      x['weight'] = weights
    
    y = fe['phrase_']
    return x, y
  


class TfDataset(TfrecordsDataset):
  def __init__(self, subset='valid', **kwargs):
    super().__init__(subset, **kwargs)
    assert not FLAGS.batch_parse, 'PreprocessLayer will be used as FLAGS.batch_parse = False'
    self.prepocess = PreProcssor(subset)

  def parse(self, example):
    dynamic_keys = ['frames']
    self.auto_parse(exclude_keys=dynamic_keys)
    self.adds(dynamic_keys)
    fe = self.parse_(serialized=example)
    return self.prepocess(fe)


FLAGS.use_aug = True
FLAGS.always_resize = False
FLAGS.affine_rate = 0
FLAGS.force_flip = False
FLAGS.scale_rate = 0.75
FLAGS.scale_range = [0.8, 1.2]
FLAGS.rotate_range = [-15.0, 15.0]
FLAGS.shift_range = [-0.05, 0.05]
FLAGS.temporal_seq_mask_range = [0.1, 0.2]
FLAGS.pad_frames = True
FLAGS.pad_method = 'zero'
FLAGS.resize_method = 'bilinear'
FLAGS.no_eos = False


from torch.utils.data import IterableDataset

class TorchDataset(IterableDataset):
  def __init__(self, subset='eval'):
    self.subset = subset

    if subset == 'train':
      dataset = TfDataset('train', files=FLAGS.train_files)
      datas = dataset.make_batch(FLAGS.batch_size, 
                             shuffle=True,
                             repeat=True,
                             drop_remainder=True,
                             return_numpy=True)
    else:
      dataset = TfDataset('valid', files=FLAGS.valid_files)
      datas = dataset.make_batch(FLAGS.eval_batch_size,  
                             shuffle=False,
                             repeat=False, 
                             drop_remainder=False,
                             return_numpy=False) # if set return numpy = True then you could only visit 1 time..
    
    self.num_instances = dataset.num_instances
    ic(self.num_instances)
    self.num_steps = dataset.num_steps
    ic(self.num_steps)
    
    self.datas = datas
    self.data_iter = iter(datas)
         
  def __iter__(self):
    if self.subset == 'train':
      while True:
        x, y = next(self.data_iter)
        del x['phrase_type']
        del x['phrase']
        if FLAGS.distributed:
          rank = gezi.get('RANK', 0)
          start = rank * FLAGS.batch_size
          end = start + FLAGS.batch_size
          for key in x:
            x[key] = x[key][start:end]
          y = y[start:end]
        yield x, y
    else:
      for x, y in self.datas:
        for key in x:
          x[key] = x[key].numpy()
        y = y.numpy()
        yield x, y
    
  def __len__(self):
    return self.num_instances
  
def get_dataloaders():
  kwargs = {
      'num_workers': 0, # >0 will hang...
      'pin_memory': True,
      'persistent_workers': False,
  }
  
  train_ds = TorchDataset('train')
  eval_ds = TorchDataset('eval')
  
  train_dl = torch.utils.data.DataLoader(train_ds,
                                         batch_size=None,
                                         sampler=None,
                                         **kwargs)
  eval_dl = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=None,
                                        sampler=None,
                                        **kwargs)

  return train_dl, eval_dl  



FLAGS.encode_pool_size = 2
FLAGS.emb_batchnorm = True
FLAGS.attn_drop = 0
FLAGS.ff_drop = 0
FLAGS.conv_drop = 0
FLAGS.inst_drop_rate = 0.2
FLAGS.cls_drop = 0.1
FLAGS.conv1d_ksize_vals = [15]
FLAGS.conv1d_expansion_factor = 2
FLAGS.ff_mult = 4
FLAGS.encoder_units = 200 
FLAGS.encoder_layers = 17
FLAGS.mhatt_dimhead = 32
FLAGS.mhatt_heads = 8
FLAGS.skip_factor = 0.5
# time reduce from layer 8, means 7 + 10 = 17 layers while the first 7 layers using 320 frames and the last 10 layers using 160 frames
FLAGS.time_reduce = True
FLAGS.time_reduce_idx = 7
FLAGS.time_kernel_size = 5 
FLAGS.time_stride = 2
# for ROPE
FLAGS.scaling_type = 'dynamic'
FLAGS.scaling_factor = 0.5


class BatchNorm(nn.Module):
  def __init__(self, num_features, momentum=0.1, eps=1e-5):
    super().__init__()
    self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)
  
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.bn(x)
    x = x.permute(0, 2, 1)
    return x

class InstanceDropout(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
    self.dropout = nn.Dropout(p)

  def forward(self, x):
    mask = torch.ones_like(x[:,:1,:1])
    mask = self.dropout(mask)
    return x * mask


class SimpleEmbedding(nn.Module):

  def __init__(self):
    super().__init__()

    self.emb_batchnorm = True if FLAGS.emb_batchnorm is None else FLAGS.emb_batchnorm
    self.embedding = nn.Linear(get_n_cols(), FLAGS.encoder_units, bias=False)
    if self.emb_batchnorm:
      self.batch_norm = BatchNorm(FLAGS.encoder_units, momentum=0.05, eps=1e-3)

  def forward(self, x):
    x = self.embedding(x)
    if self.emb_batchnorm:
      x = self.batch_norm(x)
    return x


class SqueezeformerEncoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding = SimpleEmbedding()
    attn_dropout, ff_dropout, conv_dropout = FLAGS.attn_drop, FLAGS.ff_drop, FLAGS.conv_drop
    self.encoder = Squeezeformer(
        dim=FLAGS.encoder_units,
        depth=FLAGS.encoder_layers,
        dim_head=FLAGS.mhatt_dimhead,
        heads=FLAGS.mhatt_heads,
        ff_mult=FLAGS.ff_mult,
        conv_expansion_factor=FLAGS.conv1d_expansion_factor,  # 2
        conv_kernel_size=FLAGS.conv1d_ksize_vals[0],
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        conv_dropout=conv_dropout)

  def forward(self, x):
    x = self.embedding(x)
    x = self.encoder(x)
    return x


class AvgPoolingModule(nn.Module):
  def __init__(self, pool_size, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.pooling = nn.AvgPool1d(pool_size)
    
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.pooling(x)
    x = x.permute(0, 2, 1)
    return x


class Encoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.encoder = SqueezeformerEncoder()
    if FLAGS.encode_pool_size > 1:
      self.pooling = AvgPoolingModule(FLAGS.encode_pool_size)

  def forward(self, frames):
    x = self.encoder(frames)  
    if FLAGS.encode_pool_size > 1:
      x = self.pooling(x)
    return x


class InferModel(nn.Module):
  def __init__(self, model, **kwargs):
    super().__init__(**kwargs)
    self.model = model
  
  def forward(self, frames):
    res = self.model.infer(frames)
    return res
  
  def infer(self, frames):
    return self.forward(frames)


class Model(nn.Module):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.encoder = Encoder()
    self.classifer = nn.Sequential(
            nn.Dropout(FLAGS.cls_drop),
            nn.Linear(FLAGS.encoder_units, get_vocab_size()),
        )
      
  # TODO check training flag ok
  def encode(self, frames):
    return self.encoder(frames)

  def forward_(self, frames):
    x = self.encode(frames)
    x = self.classifer(x)
    return x

  def forward(self, inputs):
    if self.training:
      self.input_ = inputs
    x = self.forward_(inputs['frames'])
    res = {
      'pred': x,
    }
    return res
  
  def infer(self, frames):
    return self.forward_(frames)

  def get_loss_fn(self):  
    def ctc_loss(loss_obj, preds, labels, labels_lengths, weights=None):
      preds = F.log_softmax(preds, dim=-1)
      preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
      loss = loss_obj(preds.transpose(0, 1), labels, preds_lengths, labels_lengths)
      if weights is not None:
        loss = torch.mean(loss * weights)
      return loss
      
    def loss_fn(res, labels, x, step=None, epoch=None, training=None):
      scalars = {}
      weights = None
      reduction = 'mean'
      if FLAGS.mix_sup:
        weights = x['weight']
        reduction = 'none'
      loss_obj = nn.CTCLoss(zero_infinity=True, reduction=reduction)
      preds = res['pred'].float()
      labels = labels.float()
      labels_lengths = torch.sum((labels != PAD_IDX).long(), dim=-1)
      loss = ctc_loss(loss_obj, preds, labels, labels_lengths, weights)      
      return loss

    return loss_fn
  
  def get_infer_model(self):
    return InferModel(self)


# simplfiy from NEMO
class TimeReductionModule(nn.Module):
    """
    Squeezeformer Time Reduction procedure. Downsamples the audio by `stride` in the time dimension.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, d_model: int, out_dim: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()

        self.d_model = d_model
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        ## NOTICE modify here so can dived by 2...
        # self.padding = max(0, self.kernel_size - self.stride) 
        ##  # like k=5, stride=2 here padding is 2 which make 320 -> 160 -> 80
        self.padding = (self.kernel_size + 1) // self.stride - 1

        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=d_model,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=d_model, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1,
        )

        self.reset_parameters()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return x

    def reset_parameters(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.d_model ** -0.5

        with torch.no_grad():
            torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
            torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
            torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
            torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """
    
    
class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache

class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model ** -0.5
        ffn2_max = self.d_ff ** -0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
    """

    def __init__(
        self, d_model, kernel_size, norm_type='batch_norm', conv_context_size=None, pointwise_activation='glu_'
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=True,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model ** -0.5
        dw_max = self.kernel_size ** -0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)
            
class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias

class DepthWiseConv1d(nn.Module):

  def __init__(self, chan_in, chan_out, kernel_size, padding):
    super().__init__()
    self.padding = padding
    self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

  def forward(self, x):
    x = F.pad(x, self.padding)
    return self.conv(x)
  
# attention, feedforward, and conv module


class Scale(nn.Module):

  def __init__(self, scale, fn):
    super().__init__()
    self.fn = fn
    self.scale = scale

  def forward(self, x, **kwargs):
    return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):

  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)

  def forward(self, x, **kwargs):
    x = self.norm(x)
    return self.fn(x, **kwargs)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # return torch.cat((-x2, x1), dim=-1)
    # return torch.cat((torch.neg(x2), x1), dim=-1)
    return torch.cat((x2 * (-1.), x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
  def __init__(self, dim, dim_head=64, 
                heads=8, dropout=0., max_pos_emb=512, 
                relpos_att=True, rope=False):
    super().__init__()
    self.scale = dim_head ** -0.5
    self.heads = heads
    inner_dim = heads * dim_head

    self.dropout = nn.Dropout(dropout)
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    self.to_out = nn.Linear(inner_dim, dim, bias = False)
    self.max_position_embeddings = max_pos_emb
    self.dim_head = dim_head
    self.relpos_att = relpos_att
    self.rope = rope
    if relpos_att:
      if rope:
        self._init_rope()
      else:
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

  def _init_rope(self):
    scaling_type = FLAGS.scaling_type
    if scaling_type is None:
        self.rotary_emb = LlamaRotaryEmbedding(self.dim_head, max_position_embeddings=self.max_position_embeddings)
    else:
        scaling_factor = FLAGS.scaling_factor
        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.dim_head, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.dim_head, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
      

  def forward(self, x):
    n, device, h = x.shape[-2], x.device, self.heads

    q, k, v = self.to_qkv(x).chunk(3, dim = -1)
    
    q = q.view(q.shape[0], q.shape[1], h, -1).permute(0, 2, 1, 3)
    k = k.view(k.shape[0], k.shape[1], h, -1).permute(0, 2, 1, 3)
    v = v.view(v.shape[0], v.shape[1], h, -1).permute(0, 2, 1, 3)

    if self.relpos_att:
      cos, sin = self.rotary_emb(v, seq_len=n)
      position_ids = torch.arange(0, n, dtype=torch.long, device=device)
      position_ids = position_ids.unsqueeze(0).view(-1, n)
      q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
      sim = torch.matmul(q, k.transpose(2, 3)) * self.scale
    else:
      sim = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
    
    attn = F.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = self.dropout(attn)
    
    out = torch.matmul(attn, v)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    out = self.to_out(out)
    out = self.dropout(out)
    return out


class SqueezeformerBlock(nn.Module):

  def __init__(self,
               *,
               dim,
               dim_head=64,
               heads=8,
               ff_mult=4,
               conv_expansion_factor=2,
               conv_kernel_size=31,
               attn_dropout=0.,
               ff_dropout=0.,
               conv_dropout=0.,
               conv_causal=False,
               relpos_att=True,
               rope=False,
               inst_drop=None,
               skip_factor=None):
    super().__init__()
    # first feed forward module
    self.norm_feed_forward1 = nn.LayerNorm(dim)
    self.feed_forward1 = ConformerFeedForward(d_model=dim, d_ff=dim * ff_mult, dropout=ff_dropout)
    self.feed_forward1_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    # convolution module
    self.norm_conv = nn.LayerNorm(dim)
    self.conv = ConformerConvolution(
        d_model=dim, kernel_size=conv_kernel_size, norm_type='batch_norm', pointwise_activation='swish'
    )
    self.conv_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    # multi-headed self-attention module
    self.norm_self_att = nn.LayerNorm(dim)
    self.self_attn = Attention(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          dropout=attn_dropout,
                          max_pos_emb=FLAGS.n_frames,
                          relpos_att=relpos_att,
                          rope=rope)
    self.self_attn_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)

    # second feed forward module
    self.norm_feed_forward2 = nn.LayerNorm(dim)
    self.feed_forward2 = ConformerFeedForward(d_model=dim, d_ff=dim * ff_mult, dropout=ff_dropout)
    self.feed_forward2_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    self.inst_drop = inst_drop if inst_drop is not None else FLAGS.inst_drop_rate
    # 0.2
    self.dropout = InstanceDropout(self.inst_drop)
    self.skip_factor = skip_factor if skip_factor is not None else FLAGS.skip_factor
    
    self.fc_factor = 0.5
    
    self.reset_parameters()
    
  def reset_parameters(self):
    # Used for Squeezeformer initialization only
    self.feed_forward1.reset_parameters_ff()
    self.feed_forward2.reset_parameters_ff()
    self.conv.reset_parameters_conv()

  def forward(self, x):
    residual = x
    x = self.self_attn_scale(x)
    x = self.self_attn(x)
    x = residual + self.dropout(x) * self.skip_factor
    
    x = self.norm_self_att(x)
    residual = x
    x = self.feed_forward1_scale(x)
    x = self.feed_forward1(x)
    x = residual + self.dropout(x) * self.skip_factor * self.fc_factor
    x = self.norm_feed_forward1(x)
    residual = x

    x = self.conv_scale(x)
    x = self.conv(x)
    x = residual + self.dropout(x) * self.skip_factor
    x = self.norm_conv(x)
    residual = x
    
    x = self.feed_forward2_scale(x)
    x = self.feed_forward2(x)
    x = residual + self.dropout(x) * self.skip_factor * self.fc_factor
    x = self.norm_feed_forward2(x)

    return x


class Squeezeformer(nn.Module):

  def __init__(self,
               dim,
               *,
               depth,
               dim_head=64,
               heads=8,
               ff_mult=4,
               conv_expansion_factor=2,
               conv_kernel_size=31,
               attn_dropout=0.,
               ff_dropout=0.,
               conv_dropout=0.,
               conv_causal=False):
    super().__init__()
    self.dim = dim
    self.layers = nn.ModuleList([])
    self.inst_drops = [None] * depth
    heads_ = heads
    conv_kernel_size_ = conv_kernel_size
    for i in range(depth):    
      relpos_att = True
      dim_head_ = dim_head
      rope = True
      heads_ = heads
      if i < FLAGS.time_reduce_idx:
        heads_ = heads // 2
      
      self.layers.append(SqueezeformerBlock(dim=dim,
                          dim_head=dim_head_,
                          heads=heads_,
                          ff_mult=ff_mult,
                          conv_expansion_factor=conv_expansion_factor,
                          conv_kernel_size=conv_kernel_size_,
                          conv_causal=conv_causal,
                          attn_dropout=attn_dropout,
                          ff_dropout=ff_dropout,
                          conv_dropout=conv_dropout,
                          relpos_att=relpos_att,
                          rope=rope,
                          inst_drop=self.inst_drops[i]))
      
    if FLAGS.time_reduce:
      if FLAGS.time_reduce_idx < depth - 1:
        reduction_module = TimeReductionModule(dim, dim, kernel_size=FLAGS.time_kernel_size, stride=FLAGS.time_stride) 
        self.layers.insert(FLAGS.time_reduce_idx, reduction_module)
      
  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = layer(x)
    return x


model = Model()
model


from torchinfo import summary
input_shape = (1, FLAGS.n_frames, get_n_cols())
summary(model.get_infer_model(), input_shape)


class AWP():

  def __init__(self,
               model,
               start_epoch=1,
               param_name="weight",
               lr=1.,
               eps=0.001):
    self.model = model
    self.param_name = param_name
    self.lr = 1 if lr is None else lr
    self.eps = 0.001 if eps is None else eps
    self.start_epoch = start_epoch
    ic(self.start_epoch, self.lr, self.eps)
    self.backup = {}
    self.backup_eps = {}

  def will_skip(self, epoch):
    return (self.lr == 0) or (epoch < self.start_epoch)

  def on_retrain_begin(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.save()
    self.attack()
    return True

  def on_retrain_end(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.restore()
    return True

  def attack(self):
    e = 1e-6
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        norm1 = torch.norm(param.grad)
        norm2 = torch.norm(param.data.detach())
        if norm1 != 0 and not torch.isnan(norm1):
          r_at = self.lr * param.grad / (norm1 + e) * (norm2 + e)
          param.data.add_(r_at)
          param.data = torch.min(
              torch.max(param.data, self.backup_eps[name][0]),
              self.backup_eps[name][1])
        # param.data.clamp_(*self.backup_eps[name])

  def save(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        if name not in self.backup:
          self.backup[name] = param.data.clone()
          grad_eps = self.eps * param.abs().detach()
          self.backup_eps[name] = (
              self.backup[name] - grad_eps,
              self.backup[name] + grad_eps,
          )

  def restore(self,):
    for name, param in self.model.named_parameters():
      if name in self.backup:
        param.data = self.backup[name]
    self.backup = {}
    self.backup_eps = {}


FLAGS.optimizer = 'Adam'
FLAGS.lr = 2e-3
FLAGS.opt_eps = 1e-6
FLAGS.warmup_proportion = 0.1
FLAGS.epochs = 400
FLAGS.pad_thre = -.3


Optimizer = getattr(torch.optim, FLAGS.optimizer)
kwargs = {
    'lr': FLAGS.lr,
    'eps': FLAGS.opt_eps,
}
optimizer = Optimizer(model.parameters(), **kwargs)
optimizer


records_pattern = f'../input/3rd-place-step1-gen-tfrecords-for-train/tfrecords/train/*.tfrec'
files = glob.glob(records_pattern) 

if FLAGS.online:
  FLAGS.train_files = files
else:
  FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]

FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]

if FLAGS.mix_sup:
  FLAGS.train_files += glob.glob(f'../input/3rd-place-step2-gen-tfrecords-for-supplement/tfrecords/sup/*.tfrec')
  np.random.shuffle(FLAGS.train_files)
ic(FLAGS.train_files[:3], FLAGS.valid_files[:2])

train_dl, eval_dl = get_dataloaders()


num_steps_per_epoch = train_dl.dataset.num_steps
total_steps = num_steps_per_epoch * FLAGS.epochs


def get_scheduler_infos():
  num_train_steps = int(num_steps_per_epoch * FLAGS.epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  ic(num_warmup_steps)
  return num_train_steps, num_warmup_steps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def to_device(x, y):
  for key in x:
    x[key] = x[key].to(device, non_blocking=True)
  y = y.to(device, non_blocking=True)
  return x, y


def simple_decode(pred):
  x = tf.argmax(pred, axis=-1)
  return x

def adjust_pad(pred):
  pred = tf.nn.softmax(pred, axis=-1)
  pred0, pred1 = pred[..., 0:1], pred[..., 1:]
  pred0 *= tf.cast(pred0 > FLAGS.pad_thre, pred.dtype)
  pred = tf.concat([pred0, pred1], axis=-1)
  return pred

def ctc_decode(x):
  x = tf.concat([x, tf.zeros((1), dtype=x.dtype)], axis=0)
  diff = tf.not_equal(x[:-1], x[1:])
  adjacent_indices = tf.where(diff)[:, 0]
  x = tf.gather(x, adjacent_indices)
  mask = x != PAD_IDX
  x = tf.boolean_mask(x, mask, axis=0)
  return x

@tf.function()
def decode_phrase(pred):
  pred = adjust_pad(pred)  
  x = tf.argmax(pred, axis=-1)
  x = ctc_decode(x)
  return x  


!pip install python-Levenshtein


from Levenshtein import distance
from sklearn.metrics import log_loss

def is_normal_char(idx):
  return not IDX2CHAR[idx].startswith('<')

def to_original_phrase(phrase_pred):
  phrase = simple_decode(phrase_pred).numpy()
  return phrase

def to_str(phrase):
  phrase = ''.join([IDX2CHAR[idx] for idx in phrase if is_normal_char(idx)])
  return phrase

def max_char_idx(phrase):
  idxs = [i + 1 for i, idx in enumerate(phrase) if is_normal_char(idx)]
  return max(idxs) if idxs else 0

def calc_score(phrase_true, phrase_pred):
  return distance(phrase_true, phrase_pred)


def evaluate(dataset, model, eval_step, steps, step, is_last, num_examples, loss_fn, outdir):
  return eval_seq(dataset, model, steps)


def try_numpy(x):
  if hasattr(x, 'numpy'):
    return x.numpy()
  return x

def eval_seq(dataset, model, steps):
  metrics = {}
  l = []
  outputs = []
  feats = []
      
  t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'eval_loop')
  for step_, (x, y) in t:
    if step_ == steps:
      break
    
    x['frames'] = x['frames'].cuda()
    frames = x['frames']

    with torch.no_grad():
      y_ = model.infer(frames)
    
    for key in x:
      if hasattr(x[key], 'cpu'):
        x[key] = x[key].cpu().detach()
        
    y_ = y_.cpu().detach().numpy()        
    outputs.append(y_)

    y = y.numpy()
    sequence_ids = try_numpy(x['sequence_id'])
    phrase_types = try_numpy(x['phrase_type'])
    phrase_dups = try_numpy(x['phrase_dup'])
    n_frames = try_numpy(x['n_frames'])
    frame_means = try_numpy(x['frame_mean'])
    idxes = try_numpy(x['idx'])
    cls_labels = try_numpy(x['cls_label'])
    for i, (sequence_id, phrase_true, phrase_pred, phrase_type, phrase_dup, n_frame, frame_mean, cls_label, idx) \
      in enumerate(zip(sequence_ids, y, y_, phrase_types, phrase_dups, n_frames, frame_means, cls_labels, idxes)):  
      phrase_ori = to_original_phrase(phrase_pred)
      char_max_idx = max_char_idx(phrase_ori)
      char_ori_rate = char_max_idx  / len(phrase_ori)
      phrase_ori = ''.join([IDX2CHAR[idx] for idx in phrase_ori])
      phrase_pred = decode_phrase(phrase_pred).numpy()
      phrase_true = to_str(phrase_true)
      phrase_pred = to_str(phrase_pred)
      # try:
      char_true_rate = 0. if not len(phrase_true) else char_max_idx / len(phrase_true)
      char_pred_rate = 0. if not len(phrase_pred) else char_max_idx / len(phrase_pred)

      cls_pred = np.zeros_like(cls_label)
      for ch in phrase_pred:
        cidx = CHAR2IDX[ch] - 1
        cls_pred[cidx] = 1
        
      phrase_type = phrase_type.decode('utf-8')
      phrase_type_pred = get_phrase_type(phrase_pred)
      distance = calc_score(phrase_true, phrase_pred)
      m = {
        'sequence_id': sequence_id,
        'phrase_type': phrase_type,
        'phrase_dup': phrase_dup,
        'phrase_true': phrase_true,
        'phrase_pred': phrase_pred,
        'phrase_ori': phrase_ori,
        'char/max_idx': char_max_idx,
        'char/ori_rate': char_ori_rate,
        'char/true_rate': char_true_rate,
        'char/pred_rate': char_pred_rate,
        'phrase_len_true': len(phrase_true),
        'phrase_len_pred': len(phrase_pred),
        'phrase_len_rate': len(phrase_pred) / len(phrase_true),
        'idx': idx,
        'distance': distance,
        'acc/char': (cls_label == cls_pred).mean(),
        'acc/type': phrase_type == phrase_type_pred,
        'acc/first': phrase_true[0] == phrase_pred[0] if phrase_pred else False,
        'acc/last': phrase_true[-1] == phrase_pred[-1] if phrase_pred else False,
        'n_frame': n_frame,
        'frame_mean': frame_mean,
        'score': max(len(phrase_true) - distance, 0.) / len(phrase_true),
      }
      l.append(m)
      
  outputs = np.concatenate(outputs, axis=0)
    
  df = pd.DataFrame(l)
  assert len(df) == len(df.idx.unique())
  def get_metrics(df):
    metrics = {
      'phrase_len_true': df['phrase_len_true'].mean(),
      'phrase_len_pred': df['phrase_len_pred'].mean(),
      'phrase_len_rate': df['phrase_len_rate'].mean(),
      'len/l1': (df['phrase_len_true'] - df['phrase_len_pred']).abs().mean(),
      'char/max_idx_max': df['char/max_idx'].max(),
      'char/max_idx': df['char/max_idx'].mean(),
      'char/ori_rate': df['char/ori_rate'].mean(),
      'char/true_rate': df['char/true_rate'].mean(),
      'char/pred_rate': df['char/pred_rate'].mean(),
      'distance': df['distance'].mean(),
      'acc/char': df['acc/char'].mean(),
      'acc/type': df['acc/type'].mean(),
      'acc/first': df['acc/first'].mean(),
      'acc/last': df['acc/last'].mean(),
      'score': max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum(),
    }
    return metrics
  
  metrics = get_metrics(df)
  
  df.to_csv(f'{FLAGS.working}/eval.csv', index=False)
  df_ = df
  df = df[df.idx < 20]
  df = df.sort_values(by=['idx'], ascending=True)
  df2 = df[['sequence_id', 'phrase_type', 'phrase_true', 'phrase_pred', 'phrase_ori', 'phrase_len_true', 'phrase_len_pred', 'distance', 'score']].head(20)
  # display(df2)
  # ic(metrics['score'])
  return metrics

TRAIN_LOSSES = []      # Training loss per step
EVAL_LOSSES = []       # Validation loss proxy per epoch
EVAL_SCORES = []       # Validation score per epoch
LRS = []               # Learning rate per step
GRAD_NORMS = []        # Gradient norm per accumulation step


# def train_loop():
#   global_step = 0
#   pbar = tqdm(range(FLAGS.epochs), total=FLAGS.epochs, desc='train_epochs')
#   ds = iter(train_dl)
#   for epoch in pbar:
#     pbar2 = tqdm(range(num_steps_per_epoch), total=num_steps_per_epoch, desc=f'train_epoch:{epoch}')
#     for _ in pbar2:
#       x, y = next(ds)
#       x, y = to_device(x, y)
      
#       def loss_fn_(x, y):
#         y_ = model(x)
#         loss = loss_fn(y_, y, x)
#         return loss
    
#       loss = loss_fn_(x, y)
#       loss = loss / FLAGS.acc_steps  
#       loss.backward()

#   #     awp.on_retrain_begin(epoch)
#   #     loss = loss_fn_(x, y)
#   #     loss = loss / FLAGS.acc_steps
#   #     loss.backward()
#   #     awp.on_retrain_end(epoch)

#       is_acc_step = ((global_step + 1) % FLAGS.acc_steps == 0) or (global_step + 1 == total_steps)
#       if is_acc_step:
#         optimizer.step()
#         model.zero_grad()
              
#       scheduler.step()
#       global_step += 1
#       pbar2.set_postfix({
#           'loss': loss.item()
#       })
        
#     metrics = eval_seq(eval_dl, model, eval_dl.dataset.num_steps)
#     pbar2.set_postfix({
#           'loss': loss.item(),
#           'score': metrics['score'],
#     })
    
#     pbar.set_postfix({
#         'loss': loss.item(),
#         'score': metrics['score'],
#     })
def train_loop(model, optimizer, scheduler, train_dl, eval_dl, num_steps_per_epoch, total_steps, num_epochs, loss_fn):
    global_step = 0
    ds = iter(train_dl)
    pbar = tqdm(range(num_epochs), total=num_epochs, desc='train_epochs')

    for epoch in pbar:
        pbar2 = tqdm(range(num_steps_per_epoch), total=num_steps_per_epoch, desc=f'train_epoch:{epoch}')

        for _ in pbar2:
            try:
                x, y = next(ds)
            except StopIteration:
                ds = iter(train_dl)
                x, y = next(ds)

            x, y = to_device(x, y)

            def loss_fn_(x, y):
                y_ = model(x)
                loss = loss_fn(y_, y, x)
                return loss

            loss = loss_fn_(x, y)
            loss = loss / FLAGS.acc_steps  
            loss.backward()

            is_acc_step = ((global_step + 1) % FLAGS.acc_steps == 0) or (global_step + 1 == total_steps)
            if is_acc_step:
                # gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                GRAD_NORMS.append(total_norm)

                optimizer.step()
                scheduler.step()   # must be after optimizer.step()
                model.zero_grad()

            LRS.append(scheduler.get_last_lr()[0])
            global_step += 1
            TRAIN_LOSSES.append(loss.item() * FLAGS.acc_steps)

            pbar2.set_postfix({'loss': loss.item() * FLAGS.acc_steps})

        metrics = eval_seq(eval_dl, model, eval_dl.dataset.num_steps)

        eval_loss_val = metrics.get('distance', 0)  # distance as proxy for "loss"
        eval_score_val = metrics['score']

        EVAL_LOSSES.append(eval_loss_val)
        EVAL_SCORES.append(eval_score_val)

        # save per-epoch CSV for later analysis
        pd.DataFrame([metrics]).to_csv(f"eval_epoch_{epoch+1}.csv", index=False)

        print(f"\n[Epoch {epoch+1}/{num_epochs}] Score: {eval_score_val:.4f}, Distance: {eval_loss_val:.4f}")

        pbar2.set_postfix({'loss': loss.item(), 'score': eval_score_val})
        pbar.set_postfix({'loss': loss.item(), 'score': eval_score_val})



# # FLAGS.epochs = 2
# FLAGS.epochs = 15
# # FLAGS.epochs = 400  
# FLAGS.lr = 2e-3
# Optimizer = getattr(torch.optim, FLAGS.optimizer)
# kwargs = {
#     'lr': FLAGS.lr,
#     'eps': FLAGS.opt_eps,
# }
# optimizer = Optimizer(model.parameters(), **kwargs)
# num_steps_per_epoch = train_dl.dataset.num_steps
# total_steps = num_steps_per_epoch * FLAGS.epochs
# num_train_steps, num_warmup_steps = get_scheduler_infos()
# scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, 
#                                                                    num_warmup_steps=num_warmup_steps, 
#                                                                    num_training_steps=num_train_steps + 1)

FLAGS.epochs = 30   # change to 50, 100, 150, 200 for different runs
FLAGS.lr = 2e-3

Optimizer = getattr(torch.optim, FLAGS.optimizer)
optimizer = Optimizer(model.parameters(), lr=FLAGS.lr, eps=FLAGS.opt_eps)

num_steps_per_epoch = train_dl.dataset.num_steps
total_steps = num_steps_per_epoch * FLAGS.epochs
num_train_steps, num_warmup_steps = get_scheduler_infos()

scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_train_steps + 1
)

loss_fn = model.get_loss_fn()
model = model.to(device)

print(f"--- STARTING MAIN TRAINING (Epochs={FLAGS.epochs}, LR={FLAGS.lr}) ---")
train_loop(model, optimizer, scheduler, train_dl, eval_dl, num_steps_per_epoch, total_steps, FLAGS.epochs, loss_fn)
print("--- MAIN TRAINING COMPLETE ---")



# model = model.to(device)
# ## p100 not support torch.compile
# # model = torch.compile(model)
# loss_fn = model.get_loss_fn()
# awp = AWP(model, start_epoch=int(FLAGS.epochs * 0.15), lr=0.2, eps=0)


# train_loop()


files = glob.glob(records_pattern) 

if FLAGS.online:
  FLAGS.train_files = files
else:
  FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]

FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]

ic(FLAGS.train_files[:3], FLAGS.valid_files[:2])
train_dl, eval_dl = get_dataloaders()


# # FLAGS.epochs = 2
# FLAGS.epochs = 4
# # FLAGS.epochs = 10
# FLAGS.lr = 1e-4
# Optimizer = getattr(torch.optim, FLAGS.optimizer)
# kwargs = {
#     'lr': FLAGS.lr,
#     'eps': FLAGS.opt_eps,
# }
# optimizer = Optimizer(model.parameters(), **kwargs)
# num_steps_per_epoch = train_dl.dataset.num_steps
# total_steps = num_steps_per_epoch * FLAGS.epochs
# num_train_steps, num_warmup_steps = get_scheduler_infos()
# scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, 
#                                                                    num_warmup_steps=num_warmup_steps, 
#                                                                    num_training_steps=num_train_steps + 1)

FLAGS.epochs = 5
FLAGS.lr = 1e-4

Optimizer = getattr(torch.optim, FLAGS.optimizer)
optimizer = Optimizer(model.parameters(), lr=FLAGS.lr, eps=FLAGS.opt_eps)

num_steps_per_epoch = train_dl.dataset.num_steps
total_steps = num_steps_per_epoch * FLAGS.epochs
num_train_steps, num_warmup_steps = get_scheduler_infos()

scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_train_steps + 1
)

print(f"--- STARTING FINETUNING (Epochs={FLAGS.epochs}, LR={FLAGS.lr}) ---")
train_loop(model, optimizer, scheduler, train_dl, eval_dl, num_steps_per_epoch, total_steps, FLAGS.epochs, loss_fn)
print("--- FINETUNING COMPLETE ---")

torch.save(model.state_dict(), './model.pt')



# torch.save(model.state_dict(), './model.pt')


import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, eval_losses, eval_scores, lrs, grad_norms):
    num_eval_epochs = len(eval_scores)

    # 1. Training vs Validation Loss
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss (per step)', alpha=0.7)
    plt.title('Training Loss vs Steps')
    plt.xlabel('Training Steps'); plt.ylabel('Loss'); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_eval_epochs+1), eval_losses, marker='o', color='red')
    plt.title('Validation Distance vs Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Distance'); plt.grid(True)
    plt.tight_layout(); plt.show()

    # 2. Score Curve
    plt.figure(figsize=(7,5))
    plt.plot(range(1, num_eval_epochs+1), eval_scores, marker='o', color='green')
    plt.title('Validation Score vs Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Score'); plt.grid(True)
    plt.show()

    # 3. LR and Gradient Norms
    plt.figure(figsize=(15,5))
    plt.plot(lrs, label='Learning Rate', color='purple', alpha=0.7)
    plt.plot(np.linspace(0, len(lrs), len(grad_norms)), grad_norms, '--', color='orange', label='Grad Norm')
    plt.title('Learning Rate & Gradient Norms')
    plt.xlabel('Training Steps'); plt.legend(); plt.grid(True)
    plt.show()

plot_training_metrics(TRAIN_LOSSES, EVAL_LOSSES, EVAL_SCORES, LRS, GRAD_NORMS)


def visualize_eval_csv(log_dir, show_worst=True, top_k=10):
    """
    Visualize metrics from eval.csv (per run) and eval_epoch_X.csv (if available).
    Args:
        log_dir (str): Directory where eval.csv / eval_epoch_X.csv are stored.
        show_worst (bool): Whether to print worst-case samples.
        top_k (int): Number of worst-case samples to display.
    """
    
    # --- Case 1: Latest eval.csv (per-run detailed results) ---
    try:
        df = pd.read_csv(f"{log_dir}/eval.csv")
        print(f"[INFO] Loaded eval.csv with {len(df)} samples.")

        # (a) Score Distribution
        plt.figure(figsize=(10,5))
        df['score'].hist(bins=50, alpha=0.7)
        plt.title("Distribution of Scores (Eval Set)")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()

        # (b) Distance vs Phrase Length
        plt.figure(figsize=(8,6))
        plt.scatter(df['phrase_len_true'], df['distance'], alpha=0.5)
        plt.xlabel("True Phrase Length")
        plt.ylabel("Edit Distance")
        plt.title("Edit Distance vs Phrase Length")
        plt.grid(True, alpha=0.3)
        plt.show()

        # (c) Score by Phrase Type
        if "phrase_type" in df.columns:
            acc_by_type = df.groupby("phrase_type")["score"].mean().sort_values()
            plt.figure(figsize=(12,6))
            acc_by_type.plot(kind="barh")
            plt.title("Average Score by Phrase Type")
            plt.xlabel("Score")
            plt.ylabel("Phrase Type")
            plt.show()

        # (d) Worst-case Predictions
        if show_worst:
            print(f"\n--- {top_k} WORST CASES (Lowest Scores) ---")
            worst_cases = df.sort_values("score").head(top_k)
            display(worst_cases[["sequence_id", "phrase_true", "phrase_pred", "score", "distance"]])

    except Exception as e:
        print(f"[WARN] Could not load {log_dir}/eval.csv. Error: {e}")

    # --- Case 2: Per-epoch eval logs ---
    eval_files = sorted(glob.glob(f"{log_dir}/eval_epoch_*.csv"))
    if eval_files:
        print(f"[INFO] Found {len(eval_files)} epoch-level eval logs.")
        scores, epochs = [], []
        for f in eval_files:
            df_epoch = pd.read_csv(f)
            scores.append(df_epoch['score'].mean())
            epoch = int(f.split("_")[-1].split(".")[0].replace(".csv", ""))
            epochs.append(epoch)

        plt.figure(figsize=(10,6))
        plt.plot(epochs, scores, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Mean Validation Score")
        plt.title("Validation Score per Epoch")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("[INFO] No per-epoch eval logs found.")
visualize_eval_csv(FLAGS.working, show_worst=True, top_k=10)

