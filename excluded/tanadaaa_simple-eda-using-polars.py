import sys
# ====================================================
# Library
# ====================================================
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
import polars.selectors as cs
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path('/kaggle/input/2025-stellar-temperature-challenge')

df_train = pl.read_csv(DATA_PATH / 'train.csv')
df_test = pl.read_csv(DATA_PATH / 'test.csv')
df_sample_submission = pl.read_csv(DATA_PATH / 'sample_submission.csv')


# trainとtestを上下に結合して、is_train列を追加する
df_train_test = pl.concat([df_train.with_columns(pl.lit(1).alias('is_train')), df_test.with_columns(pl.lit(0).alias('is_train'))], how='diagonal_relaxed')


df_train_test


print('trainの行・列数', df_train.shape)
print('testの行・列数', df_test.shape)
print('df_train_testのの行・列数', df_train.shape)


df_train.describe()


df_test.describe()


sns.histplot(df_train, x='teff')


numeric_features = df_train_test.drop(['id','is_train','teff']).select(cs.numeric()).columns

for col in numeric_features:
    print(f'左:{col}のtrain,test間の分布の違い 右:{col}のtrainにおける目的変数との関係')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    sns.histplot(df_train_test, x=col, hue='is_train', ax=ax1)
    sns.scatterplot(df_train, x=col, y='teff',ax=ax2, alpha=0.5)
    plt.show()


def export_distribution_bycalcmethod(calculation_dict: dict) -> None:
    for col, calc_col in calculation_dict.items():
        # 計算方法はユニーク数が多いので、上位5つに絞る
        top_5 = df_train_test.get_column(calc_col).value_counts().sort("count", descending=True).head(5)[calc_col]
        df_limit_train_test =  df_train_test.filter(pl.col(calc_col).is_in(top_5))
        df_limit_train = df_limit_train_test.filter(pl.col('is_train') == 1)
    
        print(f'左:{col}のtrain_testにおける計算方法別の分布の違い 右:{col}のtrainにおける目的変数との関係')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        sns.histplot(df_limit_train_test, x=col, hue=calc_col, ax=ax1)
        sns.scatterplot(df_limit_train, x=col, y='teff', hue=calc_col,ax=ax2, alpha=0.5)
        plt.show()

def export_distribution_bycalcmethod_teff() -> None:
    # 計算方法はユニーク数が多いので、上位5つに絞る
    top_5 = df_train_test.get_column('teff_prov').value_counts().sort("count", descending=True).head(5)['teff_prov']
    df_limit_train_test =  df_train_test.filter(pl.col('teff_prov').is_in(top_5))
    df_limit_train = df_limit_train_test.filter(pl.col('is_train') == 1)

    print('teffのtrainにおける目的変数と計算方法の関係')
    sns.histplot(df_limit_train, x='teff', hue='teff_prov')


calc_dict = {
    'logg': 'logg_prov',
    'feh': 'feh_prov',
    'radius': 'prov_sec',
    'mass': 'prov_sec',
    'dens': 'prov_sec'
}

export_distribution_bycalcmethod(calc_dict)


export_distribution_bycalcmethod_teff()


corr = df_train.drop(['id']).select(cs.numeric()).to_pandas().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()


with pl.Config(tbl_rows=50):
    display(df_train.sample(50))


with pl.Config(tbl_rows=50):
    display(df_test.sample(50))

