import os
import pandas as pd
import numpy as np

from torch import nn as NN
import keras
import tensorflow as tf

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor , Booster

import warnings
warnings.filterwarnings('ignore')

random_seed=42


basedir= "/kaggle/input/jane-street-real-time-market-data-forecasting/"
train_dir = basedir+'train'
test_dir = basedir+'test'
features_dir = basedir+'features.csv'
resp_dir = basedir+'responders.csv'


tf.config.list_physical_devices()


def load_csv(df_dir):
    df = pd.read_csv(df_dir)
    return df
df_train = load_csv(features_dir)
df_train


df_train.info()


df_resp = load_csv(resp_dir)
df_resp


df_resp.info()



class Partition:
    def __init__(self, nums):
        self.nums = nums
        sorted = [_Num(nums[i], i) for i in range(len(nums))]
        sorted.sort()
        self.sorted = sorted

    def run(self):
        sorted = self.sorted[:]
        N = len(sorted)
        connections = [[] for i in range(N)]

        while len(sorted) > 1:
            bigger  = sorted.pop()
            smaller = sorted.pop()

