!mkdir data
!mkdir models

!pip install -qq scikit-learn==1.6.1
!pip install pytorch_tabnet tabpfn


%%writefile data/data_class.py
from dataclasses import dataclass
from typing import Optional

import polars as pl


@dataclass
class DatasetX:
    X_train: pl.DataFrame
    X_valid: pl.DataFrame
    X_test: Optional[pl.DataFrame] = None

    def get(self) -> pl.DataFrame:
        return self.X_train, self.X_valid, self.X_test

    def to_dataset_xy(self, y_train: pl.Series, y_valid: pl.Series, y_test: Optional[pl.Series] = None) -> "DatasetXy":
        return DatasetXy(X_train=self.X_train, y_train=y_train, X_valid=self.X_valid, y_valid=y_valid, X_test=self.X_test, y_test=y_test)


@dataclass
class DatasetXy:
    X_train: pl.DataFrame
    y_train: pl.Series
    X_valid: pl.DataFrame
    y_valid: pl.Series
    X_test: Optional[pl.DataFrame] = None
    y_test: Optional[pl.Series] = None

    def get(self) -> pl.DataFrame:
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    def to_dataset_x(self) -> DatasetX:
        return DatasetX(X_train=self.X_train, X_valid=self.X_valid, X_test=self.X_test)


@dataclass
class Dfs:
    df_train: pl.DataFrame
    df_valid: pl.DataFrame
    df_test: Optional[pl.DataFrame] = None

    def get(self) -> pl.DataFrame:
        return self.df_train, self.df_valid, self.df_test



%%writefile data/data_process.py
import polars as pl

from config import cfg
from data.data_class import DatasetXy, Dfs
from data.feature_eng import add_original_cols, add_te, feature_eng, preprocess
from data.simple_feature_eng import standardize

_ = [DatasetXy, Dfs, add_te, feature_eng, preprocess, add_original_cols, standardize]


def get_dfs(cfg=cfg) -> Dfs:
    df_train = pl.read_csv(cfg.train_path)
    df_train = df_train.filter(pl.col("Number_of_Ads").is_not_null())

    df_test = None
    if hasattr(cfg, "predict") and cfg.predict:
        df_test = pl.read_csv(cfg.test_path)

    return Dfs(df_train=df_train, df_test=df_test)


def add_fold(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Publication_Day"]
    concat_expr = pl.col(cols[0]).cast(pl.Utf8)
    for col_name in cols[1:]:
        concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)

    df = df.with_columns(concat_expr.alias("fold").cast(pl.Categorical))
    return df


def get_Xy(dfs: Dfs) -> DatasetXy:
    df_train, df_valid, df_test = dfs.get()

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid, df_train)
    if df_test is not None:
        df_test = preprocess(df_test, df_train)

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col]
    X_train = df_train.drop(target_col)
    y_valid = df_valid[target_col]
    X_valid = df_valid.drop(target_col)
    X_test = df_test

    X_train = feature_eng(X_train, df_train)
    X_valid = feature_eng(X_valid, df_train)
    if X_test is not None:
        X_test = feature_eng(X_test, df_train)

    df_pltpd = pl.read_csv(cfg.pltpd_path)
    df_pltpd = df_pltpd.drop_nulls(subset=["Listening_Time_minutes"])
    df_pltpd = df_pltpd.filter(pl.col("Episode_Length_minutes").is_not_null())
    df_pltpd = df_pltpd.with_columns(
        pl.col("Number_of_Ads").cast(pl.Float64),
    )
    df_pltpd = add_fold(df_pltpd)
    df_pltpd = preprocess(df_pltpd)
    df_pltpd = feature_eng(df_pltpd, df_train)
    df_pltpd = df_pltpd.with_columns(
        (pl.lit(1000000).cast(pl.Int64) + pl.arange(0, len(df_pltpd))).cast(pl.Int64).alias("id"),
    )

    y_train = pl.concat([y_train, df_pltpd["Listening_Time_minutes"]], how="vertical")
    X_train = pl.concat([X_train, df_pltpd.select(X_train.columns)], how="vertical")

    X_train = add_original_cols(X_train, df_pltpd)
    X_valid = add_original_cols(X_valid, df_pltpd)
    if X_test is not None:
        X_test = add_original_cols(X_test, df_pltpd)

    datasetX = add_te(y_train, X_train, X_valid, X_test)
    X_train, X_valid, X_test = datasetX.get()

    X_train = standardize(X_train, df_train)
    X_valid = standardize(X_valid, df_train)
    if X_test is not None:
        X_test = standardize(X_test, df_train)

    X_train = X_train.drop(["id", "fold"])
    X_valid = X_valid.drop(["id", "fold"])
    if X_test is not None:
        X_test = X_test.drop(["id", "fold"])

    return DatasetXy(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=None)



%%writefile data/feature_eng.py
import gc
import random
from itertools import combinations

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import TargetEncoder
from tqdm import tqdm

from config import cfg
from data.data_class import DatasetX

default_selecteds = [
    'Number_of_Ads-ELen_Int', 'Episode_Sentiment-ELen_Int', 'Podcast_Name-Episode_Num-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Episode_Num-Length_per_Guest', 'Genre-Episode_Num-Length_per_Guest-HPperc_Dec', 'Genre-Host_Popularity_percentage-Episode_Num-Length_per_Guest', 'Podcast_Name-Publication_Day-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest', 'Genre-Publication_Day-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Publication_Time-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Dec', 'Genre-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest', 'Genre-Length_per_Ads-Length_per_Guest-HPperc_Dec', 'Genre-Guest_Popularity_percentage-Length_per_Ads-HPperc_Dec', 'Genre-Number_of_Ads-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Dec', 'Podcast_Name-Genre-Length_per_Guest-HPperc_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num-ELen_Dec', 'Podcast_Name-Publication_Day-Episode_Num-Length_per_Guest', 'Episode_Length_minutes-Genre-Guest_Popularity_percentage-HPperc_Dec', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-Length_per_Ads', 'Podcast_Name-Publication_Day-Length_per_Guest-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Episode_Sentiment-Length_per_Guest', 'Podcast_Name-Number_of_Ads-Episode_Num-Length_per_Guest', 'Genre-Guest_Popularity_percentage-ELen_Dec-HPperc_Dec', 'Podcast_Name-Episode_Sentiment-Episode_Num-Length_per_Guest', 'Podcast_Name-Publication_Day-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-Length_per_Ads', 'Publication_Day-Publication_Time-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Guest_Popularity_percentage-Length_per_Ads-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Length_per_Guest', 'Podcast_Name-Publication_Time-Length_per_Guest-HPperc_Int', 'Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Int', 'Podcast_Name-Host_Popularity_percentage-Episode_Num-Length_per_Ads', 'Podcast_Name-Genre-Episode_Num-Length_per_Guest', 'Podcast_Name-Publication_Time-Episode_Num-Length_per_Host', 'Publication_Time-Length_per_Ads-Length_per_Guest-HPperc_Dec', 'Publication_Time-Number_of_Ads-Length_per_Guest-HPperc_Dec', 'Publication_Time-Guest_Popularity_percentage-Length_per_Ads-HPperc_Dec', 'Genre-Publication_Day-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Dec', 'Publication_Time-Episode_Sentiment-Length_per_Guest-HPperc_Dec', 'Genre-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num', 'Guest_Popularity_percentage-Episode_Num-Length_per_Ads-HPperc_Int', 'Publication_Day-Length_per_Guest-HPperc_Dec', 'Host_Popularity_percentage-Publication_Time-Length_per_Guest', 'Podcast_Name-Genre-Length_per_Guest-HPperc_Int', 'Genre-Length_per_Ads-Length_per_Guest-HPperc_Int', 'Genre-Number_of_Ads-Length_per_Guest-HPperc_Int', 'Genre-Guest_Popularity_percentage-Length_per_Ads-HPperc_Int', 'Podcast_Name-Length_per_Guest-HPperc_Int', 'Podcast_Name-Genre-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads-ELen_Dec', 'Podcast_Name-Publication_Day-Publication_Time-Length_per_Guest', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Dec', 'Genre-Episode_Sentiment-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment-ELen_Dec', 'Number_of_Ads-Length_per_Guest-HPperc_Dec', 'Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment-HPperc_Dec', 'Publication_Time-Length_per_Guest-HPperc_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Host_Popularity_percentage-Episode_Num-ELen_Dec', 'Podcast_Name-Publication_Time-Guest_Popularity_percentage-Length_per_Ads', 'Podcast_Name-Publication_Time-Number_of_Ads-Length_per_Guest', 'Publication_Day-Episode_Num-Length_per_Ads-Length_per_Guest', 'Podcast_Name-Publication_Day-Publication_Time-Length_per_Host', 'Podcast_Name-Episode_Sentiment-Length_per_Ads-Length_per_Guest', 'Publication_Time-Guest_Popularity_percentage-ELen_Dec-HPperc_Dec', 'Episode_Length_minutes-Genre-Host_Popularity_percentage-Episode_Num', 'Podcast_Name-Publication_Time-Episode_Sentiment-Length_per_Guest', 'Genre-Episode_Num-Length_per_Host', 'Podcast_Name-Publication_Day-Episode_Sentiment-Length_per_Host', 'Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Dec', 'Genre-Publication_Day-Length_per_Ads-Length_per_Guest', 'Publication_Day-Episode_Sentiment-Episode_Num-Length_per_Guest', 'Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Dec', 'Podcast_Name-Publication_Day-Length_per_Guest', 'Genre-Publication_Day-Publication_Time-Length_per_Guest', 'Guest_Popularity_percentage-Episode_Num-ELen_Dec-HPperc_Int', 'Genre-Guest_Popularity_percentage-ELen_Int-HPperc_Dec', 'Genre-Guest_Popularity_percentage-ELen_Dec-HPperc_Int', 'Publication_Time-Number_of_Ads-Episode_Sentiment-ELen_Int', 'Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads-Episode_Num', 'Publication_Day-Episode_Num-Length_per_Ads-Length_per_Host', 'Genre-Host_Popularity_percentage-Episode_Num-ELen_Dec', 'Publication_Day-Publication_Time-Episode_Num-Length_per_Host', 'Publication_Day-Publication_Time-Length_per_Guest-HPperc_Int', 'Podcast_Name-Publication_Time-Episode_Sentiment-Length_per_Host', 'Podcast_Name-Genre-Length_per_Ads-Length_per_Guest', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-Episode_Num', 'Podcast_Name-Genre-Guest_Popularity_percentage-Length_per_Ads', 'Number_of_Ads-Episode_Sentiment-Episode_Num-Length_per_Guest', 'Podcast_Name-Genre-Publication_Day-Length_per_Host', 'Publication_Day-Episode_Sentiment-Episode_Num-Length_per_Host', 'Genre-Publication_Time-Length_per_Ads-Length_per_Guest', 'Publication_Time-Episode_Sentiment-Episode_Num-Length_per_Guest', 'Genre-Publication_Day-Publication_Time-Length_per_Host', 'Podcast_Name-Publication_Day-Length_per_Host', 'Genre-Publication_Time-Number_of_Ads-Length_per_Guest', 'Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage-Episode_Num', 'Publication_Day-Episode_Num-Length_per_Guest', 'Publication_Day-Episode_Num-Length_per_Guest-ELen_Int', 'Genre-Episode_Sentiment-Length_per_Ads-Length_per_Guest', 'Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment', 'Genre-Publication_Time-Episode_Sentiment-Length_per_Guest', 'Genre-Guest_Popularity_percentage-Episode_Sentiment-Length_per_Ads', 'Genre-Publication_Day-Episode_Sentiment-Length_per_Host', 'Podcast_Name-Episode_Sentiment-Length_per_Guest', 'Genre-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Int', 'Publication_Time-Guest_Popularity_percentage-Length_per_Ads-HPperc_Int', 'Publication_Time-Number_of_Ads-Length_per_Guest-HPperc_Int', 'Publication_Time-Number_of_Ads-Episode_Num-Length_per_Host', 'Podcast_Name-Publication_Time-Length_per_Ads-HPperc_Dec', 'Guest_Popularity_percentage-Episode_Sentiment-Length_per_Ads-HPperc_Int', 'Publication_Time-Episode_Sentiment-Episode_Num-Length_per_Host', 'Episode_Sentiment-Episode_Num-Length_per_Ads-Length_per_Host', 'Number_of_Ads-Episode_Sentiment-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Episode_Sentiment-Episode_Num-Length_per_Ads', 'Publication_Time-Episode_Sentiment-Length_per_Guest-HPperc_Int', 'Genre-Publication_Time-Length_per_Ads-Length_per_Host', 'Genre-Host_Popularity_percentage-Publication_Time-Length_per_Ads', 'Genre-Publication_Time-Number_of_Ads-Length_per_Host', 'Podcast_Name-Number_of_Ads-Length_per_Host', 'Publication_Day-Episode_Num-Length_per_Host', 'Podcast_Name-Publication_Time-Guest_Popularity_percentage-ELen_Dec', 'Publication_Time-Episode_Num-Length_per_Guest', 'Genre-Host_Popularity_percentage-Episode_Sentiment-Length_per_Ads', 'Genre-Episode_Sentiment-Length_per_Ads-Length_per_Host', 'Podcast_Name-Episode_Length_minutes-Genre-Guest_Popularity_percentage', 'Podcast_Name-Episode_Sentiment-Length_per_Host-ELen_Dec', 'Genre-Publication_Day-Length_per_Host-HPperc_Int', 'Genre-Publication_Day-Length_per_Host-ELen_Int', 'Episode_Length_minutes-Genre-Host_Popularity_percentage-Publication_Day', 'Genre-Publication_Time-Length_per_Guest-ELen_Int', 'Episode_Length_minutes-Genre-Publication_Time-Guest_Popularity_percentage', 'Genre-Publication_Time-Length_per_Guest-ELen_Dec', 'Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Dec', 'Podcast_Name-Episode_Num-Length_per_Ads-HPperc_Int', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Podcast_Name-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Genre-Number_of_Ads-Length_per_Guest-ELen_Int', 'Genre-Length_per_Ads-Length_per_Guest', 'Episode_Length_minutes-Genre-Guest_Popularity_percentage-Number_of_Ads', 'Publication_Time-Episode_Num-Length_per_Ads-HPperc_Dec', 'Genre-Guest_Popularity_percentage-Length_per_Ads', 'Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment-ELen_Int', 'Genre-Host_Popularity_percentage-Guest_Popularity_percentage', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Episode_Num', 'Guest_Popularity_percentage-Length_per_Ads-HPperc_Int', 'Number_of_Ads-Episode_Num-Length_per_Host', 'Publication_Time-Length_per_Guest-HPperc_Int', 'Episode_Length_minutes-Publication_Day-Episode_Num-HPperc_Dec', 'Podcast_Name-Episode_Length_minutes-Genre-Host_Popularity_percentage', 'Episode_Sentiment-Length_per_Guest-HPperc_Int', 'Publication_Day-Publication_Time-Length_per_Ads-Length_per_Guest', 'Publication_Day-Publication_Time-Guest_Popularity_percentage-Length_per_Ads', 'Genre-Number_of_Ads-Length_per_Host-ELen_Int', 'Genre-Number_of_Ads-Length_per_Host-HPperc_Int', 'Genre-Length_per_Ads-Length_per_Host', 'Publication_Day-Publication_Time-Number_of_Ads-Length_per_Guest', 'Podcast_Name-Host_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Length_per_Host-HPperc_Dec', 'Guest_Popularity_percentage-Number_of_Ads-ELen_Int-HPperc_Dec', 'Episode_Num-Length_per_Guest', 'Publication_Day-Episode_Sentiment-Length_per_Ads-Length_per_Guest', 'Publication_Day-Guest_Popularity_percentage-Episode_Sentiment-Length_per_Ads', 'Publication_Time-Guest_Popularity_percentage-ELen_Int-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Episode_Sentiment-ELen_Dec', 'Host_Popularity_percentage-Publication_Time-Episode_Num-ELen_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Int', 'Genre-Episode_Sentiment-Length_per_Host', 'Publication_Time-Guest_Popularity_percentage-ELen_Dec-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Episode_Num-HPperc_Int', 'Guest_Popularity_percentage-Episode_Sentiment-ELen_Int-HPperc_Dec', 'Genre-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Episode_Sentiment-HPperc_Dec', 'Guest_Popularity_percentage-Episode_Sentiment-ELen_Dec-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Episode_Num-HPperc_Dec', 'Publication_Day-Publication_Time-Length_per_Ads-Length_per_Host', 'Host_Popularity_percentage-Publication_Day-Publication_Time-Length_per_Ads', 'Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Number_of_Ads-Episode_Num-HPperc_Dec', 'Episode_Num-Length_per_Ads-HPperc_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads-Episode_Sentiment', 'Publication_Time-Episode_Sentiment-Length_per_Ads-Length_per_Guest', 'Length_per_Guest-HPperc_Int', 'Episode_Length_minutes-Episode_Sentiment-Episode_Num-HPperc_Dec', 'Publication_Time-Guest_Popularity_percentage-Episode_Sentiment-Length_per_Ads', 'Publication_Time-Number_of_Ads-Episode_Sentiment-Length_per_Guest', 'Genre-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage', 'Podcast_Name-Host_Popularity_percentage-Publication_Day-Episode_Num', 'Publication_Day-Number_of_Ads-Length_per_Guest-ELen_Int', 'Publication_Day-Number_of_Ads-Length_per_Guest-ELen_Dec', 'Podcast_Name-Publication_Time-Guest_Popularity_percentage-Episode_Num', 'Publication_Day-Publication_Time-Length_per_Guest', 'Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage-Episode_Sentiment', 'Publication_Day-Guest_Popularity_percentage-Episode_Num-HPperc_Int', 'Episode_Length_minutes-Genre-Host_Popularity_percentage', 'Publication_Day-Episode_Sentiment-Length_per_Guest-ELen_Int', 'Podcast_Name-Publication_Day-Guest_Popularity_percentage-ELen_Int', 'Podcast_Name-Publication_Day-Length_per_Ads-HPperc_Int', 'Episode_Length_minutes-Genre-Episode_Num-HPperc_Int', 'Publication_Time-Number_of_Ads-Episode_Sentiment-Length_per_Host', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Publication_Time', 'Podcast_Name-Publication_Time-Episode_Num-Length_per_Ads', 'Publication_Day-Publication_Time-Length_per_Host-HPperc_Int', 'Publication_Day-Publication_Time-Length_per_Host-ELen_Dec', 'Publication_Day-Publication_Time-Length_per_Host-ELen_Int', 'Publication_Day-Number_of_Ads-Length_per_Host-ELen_Dec', 'Publication_Day-Length_per_Ads-Length_per_Host', 'Publication_Day-Number_of_Ads-Length_per_Host-ELen_Int', 'Host_Popularity_percentage-Publication_Day-Length_per_Ads', 'Publication_Day-Publication_Time-Length_per_Host', 'Publication_Time-Guest_Popularity_percentage-Length_per_Ads', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Sentiment', 'Publication_Day-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Publication_Time-Number_of_Ads-Length_per_Guest', 'Publication_Time-Episode_Sentiment-Length_per_Guest-ELen_Int', 'Episode_Length_minutes-Guest_Popularity_percentage-Number_of_Ads-Episode_Sentiment', 'Publication_Day-Episode_Num-Length_per_Ads-HPperc_Int', 'Number_of_Ads-Episode_Sentiment-Length_per_Guest-ELen_Int', 'Number_of_Ads-Episode_Sentiment-Length_per_Guest-ELen_Dec', 'Podcast_Name-Host_Popularity_percentage-Publication_Day-ELen_Int', 'Guest_Popularity_percentage-Episode_Sentiment-Length_per_Ads', 'Podcast_Name-Publication_Time-Length_per_Ads-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Genre-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Episode_Sentiment-Episode_Num', 'Podcast_Name-Episode_Length_minutes-HPperc_Dec', 'Episode_Length_minutes-Episode_Num-HPperc_Dec', 'Podcast_Name-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Podcast_Name-Guest_Popularity_percentage-Number_of_Ads-ELen_Int', 'Genre-Publication_Day-Episode_Num-Length_per_Ads', 'Publication_Day-Length_per_Guest-ELen_Int', 'Publication_Day-Length_per_Guest-ELen_Dec', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Number_of_Ads', 'Publication_Time-Number_of_Ads-Length_per_Host-ELen_Dec', 'Publication_Time-Number_of_Ads-Length_per_Host-HPperc_Dec', 'Publication_Time-Length_per_Ads-Length_per_Host', 'Publication_Time-Number_of_Ads-Length_per_Host-ELen_Int', 'Publication_Time-Number_of_Ads-Length_per_Host-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Publication_Day-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Episode_Num-ELen_Int', 'Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Publication_Time-Episode_Sentiment-Length_per_Host-HPperc_Dec', 'Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads-Episode_Sentiment', 'Publication_Time-Number_of_Ads-Length_per_Host', 'Episode_Sentiment-Length_per_Ads-Length_per_Host-ELen_Dec', 'Number_of_Ads-Episode_Sentiment-Length_per_Host-HPperc_Int', 'Number_of_Ads-Episode_Sentiment-Length_per_Host-HPperc_Dec', 'Publication_Time-Episode_Num-Length_per_Ads-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int', 'Genre-Publication_Day-Length_per_Ads-HPperc_Int', 'Publication_Time-Episode_Sentiment-Length_per_Host', 'Publication_Day-Length_per_Host-ELen_Dec', 'Publication_Day-Length_per_Host-HPperc_Dec', 'Episode_Sentiment-Episode_Num-Length_per_Ads-HPperc_Int', 'Podcast_Name-Host_Popularity_percentage-Publication_Time-ELen_Int', 'Podcast_Name-Host_Popularity_percentage-Number_of_Ads-ELen_Int', 'Guest_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int', 'Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage', 'Episode_Length_minutes-Publication_Day-Episode_Num-HPperc_Int', 'Publication_Time-Length_per_Guest-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Episode_Sentiment-Episode_Num', 'Number_of_Ads-Length_per_Guest-ELen_Int', 'Publication_Day-Length_per_Host', 'Host_Popularity_percentage-Publication_Time-Episode_Num-ELen_Int', 'Host_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Length_per_Ads', 'Guest_Popularity_percentage-Number_of_Ads-ELen_Int-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment', 'Episode_Sentiment-Length_per_Guest-ELen_Dec', 'Podcast_Name-Episode_Length_minutes-Number_of_Ads-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Publication_Time-HPperc_Int', 'Host_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Episode_Sentiment-ELen_Int-HPperc_Int', 'Podcast_Name-Genre-Length_per_Ads-HPperc_Int', 'Genre-Episode_Sentiment-Episode_Num-Length_per_Ads', 'Host_Popularity_percentage-Guest_Popularity_percentage', 'Number_of_Ads-Length_per_Guest', 'Publication_Time-Length_per_Host-ELen_Dec', 'Episode_Length_minutes-Publication_Time-Episode_Num-HPperc_Int', 'Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads', 'Number_of_Ads-Length_per_Host-ELen_Int', 'Number_of_Ads-Length_per_Host-HPperc_Dec', 'Publication_Time-Episode_Sentiment-Length_per_Ads-HPperc_Dec', 'Genre-Host_Popularity_percentage-Publication_Day-ELen_Int', 'Episode_Length_minutes-Host_Popularity_percentage-Episode_Sentiment', 'Host_Popularity_percentage-Length_per_Ads', 'Episode_Sentiment-Length_per_Host-ELen_Dec-HPperc_Dec', 'Episode_Sentiment-Length_per_Host-HPperc_Dec', 'Episode_Sentiment-Length_per_Host-HPperc_Int', 'Episode_Sentiment-Length_per_Host-ELen_Int-HPperc_Int', 'Episode_Sentiment-Length_per_Host-ELen_Int', 'Episode_Length_minutes-Episode_Sentiment-Episode_Num-HPperc_Int', 'Podcast_Name-Genre-Host_Popularity_percentage-Episode_Num', 'Podcast_Name-Genre-Guest_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Publication_Day-Publication_Time-HPperc_Dec', 'Podcast_Name-Guest_Popularity_percentage-ELen_Int', 'Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-HPperc_Dec', 'Publication_Day-Length_per_Ads-ELen_Dec-HPperc_Dec', 'Genre-Host_Popularity_percentage-Publication_Time-Episode_Num', 'Genre-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Genre-Publication_Day-HPperc_Int', 'Genre-Guest_Popularity_percentage-Number_of_Ads-ELen_Int', 'Host_Popularity_percentage-Episode_Num-ELen_Int', 'Episode_Length_minutes-Publication_Day-Episode_Sentiment-HPperc_Dec', 'Publication_Day-Publication_Time-Episode_Num-Length_per_Ads', 'Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Podcast_Name-Genre-Host_Popularity_percentage-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Episode_Num', 'Episode_Length_minutes-Genre-Publication_Time-Episode_Num', 'Podcast_Name-Host_Popularity_percentage-ELen_Int', 'Podcast_Name-Episode_Num-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Genre-Number_of_Ads-Episode_Num', 'Genre-Host_Popularity_percentage-Publication_Time-ELen_Int', 'Publication_Day-Episode_Sentiment-Episode_Num-Length_per_Ads', 'Publication_Day-Publication_Time-Guest_Popularity_percentage-Episode_Num', 'Genre-Host_Popularity_percentage-Number_of_Ads-ELen_Int', 'Publication_Time-Guest_Popularity_percentage-Episode_Sentiment-HPperc_Dec', 'Podcast_Name-Episode_Sentiment-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Genre-HPperc_Int', 'Episode_Length_minutes-Episode_Num-HPperc_Int', 'Length_per_Host-ELen_Dec-HPperc_Dec', 'Episode_Length_minutes-Host_Popularity_percentage', 'Length_per_Host-ELen_Int', 'Length_per_Host-ELen_Int-HPperc_Int', 'Podcast_Name-Publication_Day-Episode_Sentiment-Length_per_Ads', 'Episode_Length_minutes-Genre-Episode_Sentiment-Episode_Num', 'Genre-Episode_Num-Length_per_Ads', 'Episode_Length_minutes-Genre-Publication_Time-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-HPperc_Dec', 'Genre-Host_Popularity_percentage-Episode_Sentiment-ELen_Int', 'Episode_Length_minutes-Genre-Number_of_Ads-HPperc_Int', 'Genre-Length_per_Ads-ELen_Dec-HPperc_Int', 'Publication_Day-Publication_Time-Guest_Popularity_percentage-HPperc_Int', 'Length_per_Guest', 'Publication_Day-Guest_Popularity_percentage-Number_of_Ads-ELen_Int', 'Publication_Day-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Host_Popularity_percentage-Publication_Time-Episode_Sentiment-ELen_Dec', 'Episode_Length_minutes-Genre-HPperc_Dec', 'Episode_Length_minutes-Publication_Time-Episode_Sentiment-HPperc_Dec', 'Episode_Length_minutes-Genre-Episode_Sentiment-HPperc_Int', 'Episode_Length_minutes-Publication_Day-Publication_Time-Episode_Num', 'Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-HPperc_Dec', 'Publication_Day-Guest_Popularity_percentage-Episode_Sentiment-ELen_Int', 'Episode_Sentiment-Length_per_Ads-ELen_Dec-HPperc_Dec', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-Episode_Num', 'Host_Popularity_percentage-Publication_Day-Publication_Time-ELen_Int', 'Host_Popularity_percentage-Publication_Day-Number_of_Ads-ELen_Int', 'Publication_Day-Episode_Num-Length_per_Ads-ELen_Dec', 'Publication_Time-Episode_Sentiment-Length_per_Ads-HPperc_Int', 'Length_per_Host', 'Host_Popularity_percentage-Publication_Day-Episode_Sentiment-ELen_Int', 'Episode_Length_minutes-Publication_Day-Episode_Sentiment-Episode_Num', 'Podcast_Name-Publication_Time-Episode_Sentiment-Length_per_Ads', 'Podcast_Name-Number_of_Ads-ELen_Int', 'Publication_Day-Episode_Num-Length_per_Ads', 'Episode_Length_minutes-Publication_Day-Publication_Time-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-Number_of_Ads-ELen_Int', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-HPperc_Int', 'Episode_Sentiment-Length_per_Ads-HPperc_Dec', 'Publication_Day-Length_per_Ads-ELen_Dec-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Publication_Day-Number_of_Ads', 'Podcast_Name-Episode_Length_minutes-Publication_Day-Publication_Time', 'Podcast_Name-Publication_Day-Length_per_Ads-ELen_Dec', 'Publication_Time-Guest_Popularity_percentage-Number_of_Ads-HPperc_Int', 'Episode_Length_minutes-Publication_Day-Episode_Sentiment-HPperc_Int', 'Episode_Length_minutes-Publication_Day-HPperc_Dec', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-Episode_Num', 'Publication_Time-Episode_Num-Length_per_Ads-ELen_Dec', 'Host_Popularity_percentage-Publication_Time-Number_of_Ads-ELen_Int', 'Guest_Popularity_percentage-Number_of_Ads-Episode_Sentiment-ELen_Int', 'Publication_Time-Episode_Num-ELen_Int-HPperc_Dec', 'Podcast_Name-Episode_Length_minutes-Publication_Day-Episode_Sentiment', 'Podcast_Name-Publication_Day-ELen_Int-HPperc_Dec', 'Publication_Time-Guest_Popularity_percentage-Episode_Sentiment-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Episode_Sentiment-Episode_Num', 'Episode_Length_minutes-Genre-Episode_Num', 'Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-Episode_Num', 'Episode_Sentiment-Episode_Num-Length_per_Ads-ELen_Dec', 'Host_Popularity_percentage-Publication_Time-Episode_Sentiment-ELen_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-HPperc_Int', 'Publication_Time-Length_per_Ads-ELen_Dec-HPperc_Int', 'Host_Popularity_percentage-Number_of_Ads-Episode_Sentiment-ELen_Int', 'Publication_Time-Episode_Num-ELen_Dec-HPperc_Int', 'Genre-Publication_Day-Publication_Time-Length_per_Ads', 'Episode_Length_minutes-Genre-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Episode_Sentiment-HPperc_Int', 'Episode_Sentiment-Length_per_Ads-ELen_Dec-HPperc_Int', 'Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-HPperc_Int', 'Publication_Day-Episode_Num-ELen_Int-HPperc_Int', 'Publication_Day-Guest_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Publication_Time-HPperc_Dec', 'Genre-Publication_Day-Number_of_Ads-ELen_Int', 'Episode_Length_minutes-Number_of_Ads-HPperc_Dec', 'Episode_Length_minutes-Publication_Day-Episode_Num', 'Length_per_Ads-ELen_Dec-HPperc_Dec', 'Host_Popularity_percentage-Publication_Day-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Number_of_Ads-Episode_Sentiment', 'Podcast_Name-Episode_Length_minutes-Publication_Time-Episode_Sentiment', 'Episode_Sentiment-Length_per_Ads-HPperc_Int', 'Number_of_Ads-ELen_Int-HPperc_Int', 'Podcast_Name-Number_of_Ads-ELen_Int-HPperc_Dec', 'Podcast_Name-Publication_Day-Episode_Num-ELen_Int', 'Episode_Length_minutes-Publication_Day-HPperc_Int', 'Episode_Length_minutes-Episode_Sentiment-HPperc_Dec', 'Episode_Length_minutes-Number_of_Ads', 'Length_per_Ads-ELen_Dec', 'Publication_Time-Episode_Num-ELen_Int-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Podcast_Name-Publication_Day-ELen_Int-HPperc_Int', 'Number_of_Ads-Episode_Num-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Episode_Num', 'Episode_Length_minutes-Number_of_Ads-Episode_Num', 'Podcast_Name-Episode_Length_minutes-Genre-Publication_Day', 'Podcast_Name-Episode_Length_minutes-Publication_Day', 'Host_Popularity_percentage-Publication_Time-ELen_Int', 'Episode_Sentiment-Episode_Num-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Number_of_Ads-HPperc_Int', 'Episode_Length_minutes-Publication_Time-HPperc_Int', 'Episode_Length_minutes-Episode_Sentiment-Episode_Num', 'Length_per_Ads-ELen_Dec-HPperc_Int', 'Host_Popularity_percentage-Episode_Sentiment-ELen_Int', 'Podcast_Name-Publication_Time-Episode_Num-ELen_Int', 'Episode_Length_minutes-Episode_Sentiment-HPperc_Int', 'Episode_Num-ELen_Int-HPperc_Dec', 'Podcast_Name-Number_of_Ads-ELen_Int-HPperc_Int', 'Podcast_Name-Publication_Time-ELen_Int-HPperc_Int', 'Podcast_Name-Episode_Sentiment-Episode_Num-ELen_Int', 'Podcast_Name-Episode_Length_minutes-Publication_Time', 'Podcast_Name-Episode_Length_minutes-Genre-Number_of_Ads', 'Podcast_Name-Genre-Length_per_Ads-ELen_Dec', 'Podcast_Name-Episode_Length_minutes-Number_of_Ads', 'Podcast_Name-Length_per_Ads-ELen_Dec', 'Episode_Length_minutes-HPperc_Dec', 'Podcast_Name-Publication_Time-Episode_Sentiment-ELen_Int', 'Episode_Length_minutes-Genre-Publication_Time-Number_of_Ads', 'Podcast_Name-Episode_Sentiment-ELen_Int-HPperc_Int', 'Podcast_Name-Episode_Length_minutes-Genre-Episode_Sentiment', 'Podcast_Name-Episode_Length_minutes-Episode_Sentiment', 'Episode_Length_minutes-Genre', 'Publication_Day-Episode_Num-ELen_Int', 'Episode_Length_minutes-Genre-Number_of_Ads-Episode_Sentiment', 'Episode_Length_minutes-Genre-Publication_Time-Episode_Sentiment', 'Episode_Length_minutes-Publication_Day-Publication_Time-Number_of_Ads', 'Publication_Day-Publication_Time-Length_per_Ads-ELen_Dec', 'Episode_Num-ELen_Int-HPperc_Int', 'Guest_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Episode_Num', 'Episode_Length_minutes-Number_of_Ads-Episode_Sentiment', 'Host_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-Episode_Sentiment', 'Publication_Day-Episode_Sentiment-Length_per_Ads-ELen_Dec', 'Episode_Length_minutes-Publication_Day-Publication_Time-Episode_Sentiment', 'Number_of_Ads-Episode_Sentiment-ELen_Int-HPperc_Int', 'Episode_Length_minutes-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads', 'Publication_Time-Length_per_Ads-ELen_Dec', 'Genre-Episode_Num-ELen_Int', 'Episode_Length_minutes-Genre-Publication_Day', 'Genre-Publication_Day-ELen_Int-HPperc_Int', 'Publication_Day-ELen_Int-HPperc_Dec', 'Publication_Time-Episode_Sentiment-Length_per_Ads-ELen_Dec', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-Episode_Sentiment', 'Podcast_Name-Episode_Length_minutes-Genre', 'Episode_Length_minutes-Publication_Day-Episode_Sentiment', 'Podcast_Name-Episode_Num-ELen_Int', 'Podcast_Name-Episode_Length_minutes', 'Publication_Day-Length_per_Ads-ELen_Dec', 'Publication_Time-Number_of_Ads-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Genre-Number_of_Ads', 'Episode_Length_minutes-Publication_Day-Number_of_Ads', 'Episode_Length_minutes-Publication_Day-Publication_Time', 'Podcast_Name-Publication_Day-Number_of_Ads-ELen_Int', 'Genre-Number_of_Ads-Episode_Num-ELen_Int', 'Podcast_Name-Genre-ELen_Int-HPperc_Int', 'Number_of_Ads-Episode_Sentiment-ELen_Int-HPperc_Dec', 'Podcast_Name-ELen_Int-HPperc_Int', 'Publication_Time-Number_of_Ads-Episode_Num-ELen_Int', 'Genre-Number_of_Ads-ELen_Int-HPperc_Int', 'Publication_Day-Number_of_Ads-Episode_Num-ELen_Int', 'Publication_Day-Number_of_Ads-ELen_Int-HPperc_Int', 'Publication_Day-Publication_Time-Episode_Num-ELen_Int', 'Publication_Day-Publication_Time-ELen_Int-HPperc_Int'
]

# default_selecteds += ["Length_per_Host", "Length_per_Guest", "Episode_Length_minutes", "ELen_Int", "Length_per_Ads", "ELen_Dec"]

# default_selecteds += [
#     "Episode_Num-Host_Popularity_percentage-Guest_Popularity_percentage",
#     "Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Int",
#     "Guest_Popularity_percentage-Episode_Num-HPperc_Int-ELen_Int",
#     "Guest_Popularity_percentage-Episode_Num-HPperc_Int-ELen_Int",
# ]

# Delete duplicate
default_selecteds = list(set(default_selecteds))

if hasattr(cfg, "eval") and cfg.eval:
    print("Length of default_selecteds:", len(default_selecteds), "-> Sampling 5")
    random.seed(42)
    default_selecteds = random.sample(default_selecteds, 5)


re_dict = {}
re_dict["podc_dict"] = {
    "Mystery Matters": 0,
    "Joke Junction": 1,
    "Study Sessions": 2,
    "Digital Digest": 3,
    "Mind & Body": 4,
    "Fitness First": 5,
    "Criminal Minds": 6,
    "News Roundup": 7,
    "Daily Digest": 8,
    "Music Matters": 9,
    "Sports Central": 10,
    "Melody Mix": 11,
    "Game Day": 12,
    "Gadget Geek": 13,
    "Global News": 14,
    "Tech Talks": 15,
    "Sport Spot": 16,
    "Funny Folks": 17,
    "Sports Weekly": 18,
    "Business Briefs": 19,
    "Tech Trends": 20,
    "Innovators": 21,
    "Health Hour": 22,
    "Comedy Corner": 23,
    "Sound Waves": 24,
    "Brain Boost": 25,
    "Athlete's Arena": 26,
    "Wellness Wave": 27,
    "Style Guide": 28,
    "World Watch": 29,
    "Humor Hub": 30,
    "Money Matters": 31,
    "Healthy Living": 32,
    "Home & Living": 33,
    "Educational Nuggets": 34,
    "Market Masters": 35,
    "Learning Lab": 36,
    "Lifestyle Lounge": 37,
    "Crime Chronicles": 38,
    "Detective Diaries": 39,
    "Life Lessons": 40,
    "Current Affairs": 41,
    "Finance Focus": 42,
    "Laugh Line": 43,
    "True Crime Stories": 44,
    "Business Insights": 45,
    "Fashion Forward": 46,
    "Tune Time": 47,
}
re_dict["genr_dict"] = {
    "True Crime": 0,
    "Comedy": 1,
    "Education": 2,
    "Technology": 3,
    "Health": 4,
    "News": 5,
    "Music": 6,
    "Sports": 7,
    "Business": 8,
    "Lifestyle": 9,
}
re_dict["week_dict"] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
re_dict["time_dict"] = {"Morning": 10, "Afternoon": 14, "Evening": 17, "Night": 21}
re_dict["sent_dict"] = {"Negative": 0, "Neutral": 1, "Positive": 2}


pl_i_type = pl.Int64
pl_f_type = pl.Float64


def cast_numeric_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [col for col in df.columns if df.schema[col] == pl.Float64 or df.schema[col] == pl.Float32]
    int_cols = [col for col in df.columns if df.schema[col] == pl.Int64 or df.schema[col] == pl.Int32]

    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl_f_type) for col in float_cols])
    if int_cols:
        df = df.with_columns([pl.col(col).cast(pl_i_type) for col in int_cols])

    return df


def preprocess(df: pl.DataFrame, df_train: pl.DataFrame = None) -> pl.DataFrame:
    df = cast_numeric_dtypes(df)
    df = df.with_columns(pl.col("Episode_Title").str.slice(8).cast(pl.Int32).alias("Episode_Num")).drop("Episode_Title")

    # Convert categorical variables using mapping
    for col, mapping in [
        ("Genre", re_dict["genr_dict"]),
        ("Podcast_Name", re_dict["podc_dict"]),
        ("Publication_Day", re_dict["week_dict"]),
        ("Publication_Time", re_dict["time_dict"]),
        ("Episode_Sentiment", re_dict["sent_dict"]),
    ]:
        df = df.with_columns(pl.col(col).replace(mapping).alias(col))

    # Cap extreme values
    df = df.with_columns(
        pl.when(pl.col("Episode_Length_minutes") > 121.0).then(121.0).otherwise(pl.col("Episode_Length_minutes")).alias("Episode_Length_minutes"),
        pl.when(pl.col("Number_of_Ads") > 103.91).then(103.91).otherwise(pl.col("Number_of_Ads")).alias("Number_of_Ads"),
    )

    # Create NaN indicator columns
    df = df.with_columns(
        pl.col("Episode_Length_minutes").is_null().cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Length_minutes_NaN"),
        pl.col("Guest_Popularity_percentage").is_null().cast(pl.Utf8).cast(pl.Categorical).alias("Guest_Popularity_percentage_NaN"),
    )

    # Fill NA values with median
    if df_train is None:
        df_train = df.clone()

    # e_median = df_train.select(pl.col("Episode_Length_minutes").median()).item()
    # g_median = df_train.select(pl.col("Guest_Popularity_percentage").median()).item()
    # n_median = df_train.select(pl.col("Number_of_Ads").median()).item()

    df = df.with_columns(
        pl.col("Episode_Length_minutes").fill_null(-2),
        pl.col("Guest_Popularity_percentage").fill_null(-2),
        pl.col("Number_of_Ads").fill_null(-2),
    )

    return df


def feature_eng(df: pl.DataFrame, df_train: pl.DataFrame) -> pl.DataFrame:
    global selected
    # Cyclical features for day and time
    df = df.with_columns(
        # Day features
        pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).sin().alias("Day_sin"),
        pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).cos().alias("Day_cos"),
        pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).sin().alias("Day_sin2"),
        pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).cos().alias("Day_cos2"),
        # Time features
        pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).sin().alias("Time_sin"),
        pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).cos().alias("Time_cos"),
        pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).sin().alias("Time_sin2"),
        pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).cos().alias("Time_cos2"),
        # Ratio features
        (pl.col("Episode_Length_minutes") / (pl.col("Number_of_Ads") + 1)).fill_null(0).alias("Length_per_Ads"),
        (pl.col("Episode_Length_minutes") / (pl.col("Host_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Host"),
        (pl.col("Episode_Length_minutes") / (pl.col("Guest_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Guest"),
        # Episode length features
        pl.col("Episode_Length_minutes").floor().alias("ELen_Int"),
        (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").floor()).alias("ELen_Dec"),
        pl.col("Host_Popularity_percentage").floor().alias("HPperc_Int"),
        (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").floor()).alias("HPperc_Dec"),
        # Sentiment features
        (pl.col("Episode_Sentiment") == "2").cast(pl.Int8).alias("Is_Positive_Sentiment"),
        pl.when(pl.col("Episode_Sentiment") == "2").then(0.75).otherwise(0.717).cast(pl_f_type).alias("Sentiment_Multiplier"),
        # Squared features
        (pl.col("Episode_Length_minutes") ** 2).alias("Episode_Length_squared"),
        (pl.col("Episode_Length_minutes") ** 3).alias("Episode_Length_squared2"),
    )

    df = df.with_columns(
        (np.sin(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Sin"),
        (np.cos(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Cos"),
        (pl.col("Episode_Length_minutes") * pl.col("Sentiment_Multiplier")).alias("Expected_Listening_Time_Sentiment"),
    )

    df = df.with_columns(
        (
            (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").median()).pow(2)
            + (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").median()).pow(2)
            + (pl.col("Guest_Popularity_percentage") - pl.col("Guest_Popularity_percentage").median()).pow(2)
            + (pl.col("Number_of_Ads") - pl.col("Number_of_Ads").median()).pow(2)
        ).alias("Diff_Squared")
    )

    # Convert columns to categorical
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))

    return df


def get_combinations(df: pl.DataFrame, columns_to_encode: list, pair_sizes: list) -> list:
    df_length = len(df)

    target_ratios = []
    target_ratios.extend(np.arange(0.01, 0.3, 0.05).tolist())
    target_ratios.extend(np.arange(0.3, 1.01, 0.005).tolist())

    if hasattr(cfg, "eval") and cfg.eval:
        target_ratios = np.arange(0.001, 0.999, 0.05).tolist()

    all_combinations = []
    for r in pair_sizes:
        for cols in combinations(columns_to_encode, r):
            group_counts = len(df.group_by(cols).count())
            ratio = group_counts / df_length
            all_combinations.append((cols, ratio))

    unique_combinations = set()
    for target in target_ratios:
        closest_combination = min(all_combinations, key=lambda x: abs(x[1] - target))
        unique_combinations.add(closest_combination[0])

    return list(unique_combinations)


def cols_encode(df: pl.DataFrame, combinations_list: list, round_num: int = 10) -> pl.DataFrame:
    batch_size = 20
    for i in range(0, len(combinations_list), batch_size):
        batch = combinations_list[i : i + batch_size]

        for cols in tqdm(batch):
            new_col_name = "colen_" + "_".join(cols)
            if cols[0] in df.select(cs.numeric()).columns:
                concat_expr = pl.col(cols[0]).round(round_num).cast(pl.Utf8)
            else:
                concat_expr = pl.col(cols[0]).cast(pl.Utf8)

            for col_name in cols[1:]:
                if col_name in df.select(cs.numeric()).columns:
                    concat_expr = concat_expr + "_" + pl.col(col_name).round(round_num).cast(pl.Utf8)
                else:
                    concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)

            df = df.with_columns(concat_expr.alias(new_col_name).cast(pl.Categorical))

        gc.collect()

        mem_usage = sum(df.estimated_size() for col in df.columns) / (1024 * 1024)
        print(f"Memory usage: {mem_usage:.2f} MB")

    return df


def encode_target(
    target: pl.Series, encode_columns: list, X_train: pl.DataFrame, X_valid: pl.DataFrame, X_test: pl.DataFrame = None, random_state: int = cfg.random_state
) -> DatasetX:
    if isinstance(target, pl.Series):
        target_values = target.to_numpy()
    else:
        target_values = target

    encoder = TargetEncoder(random_state=random_state)

    for col in tqdm(encode_columns, desc="Encoding cols"):
        encoded_col_name = f"{col}_{target.name}_encoded"

        X_train_col = X_train[col].to_numpy().reshape(-1, 1)
        encoded_train = encoder.fit_transform(X_train_col, target_values)
        X_train = X_train.with_columns(pl.Series(encoded_col_name, encoded_train.flatten()))

        X_valid_col = X_valid[col].to_numpy().reshape(-1, 1)
        encoded_valid = encoder.transform(X_valid_col)
        X_valid = X_valid.with_columns(pl.Series(encoded_col_name, encoded_valid.flatten()))

        if X_test is not None:
            X_test_col = X_test[col].to_numpy().reshape(-1, 1)
            encoded_test = encoder.transform(X_test_col)
            X_test = X_test.with_columns(pl.Series(encoded_col_name, encoded_test.flatten()))

        gc.collect()

    return DatasetX(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
    )


def add_te(y_train: pl.Series, X_train: pl.DataFrame, X_valid: pl.DataFrame, X_test: pl.DataFrame = None) -> DatasetX:
    before_encode_len = len(X_train.columns)

    combinations_list = [item.split("-") for item in default_selecteds]

    combinations_list = list(set(tuple(sorted(combo)) for combo in combinations_list))

    print("Combinations list length:", len(combinations_list))

    X_train = cols_encode(X_train, combinations_list)
    X_valid = cols_encode(X_valid, combinations_list)
    if X_test is not None:
        X_test = cols_encode(X_test, combinations_list)

    encoded_columns = X_train.columns[before_encode_len:]

    datasetX = encode_target(y_train, encoded_columns, X_train, X_valid, X_test=X_test)
    X_train, X_valid, X_test = datasetX.get()

    # encoded_columns = [col for col in encoded_columns if "Episode_Length_minutes" not in col and "ELen" not in col and "Length_per" not in col]
    # datasetX = encode_target(X_train["Episode_Length_minutes"], encoded_columns, X_train, X_valid, X_test=X_test)
    # X_train, X_valid, X_test = datasetX.get()

    X_train = X_train.drop(encoded_columns)
    X_valid = X_valid.drop(encoded_columns)
    if X_test is not None:
        X_test = X_test.drop(encoded_columns)

    return DatasetX(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
    )


def add_original_cols(df: pl.DataFrame, df_pltpd: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = df.select(cs.numeric()).columns
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    pte_selecteds = [
        'Podcast_Name-Episode_Num-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest', 'Podcast_Name-Host_Popularity_percentage-Episode_Num-Length_per_Guest', 'Genre-Publication_Day-Length_per_Guest-HPperc_Dec', 'Genre-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest', 'Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Genre-Length_per_Guest-HPperc_Dec', 'Podcast_Name-Guest_Popularity_percentage-Length_per_Ads-HPperc_Int', 'Podcast_Name-Publication_Day-Length_per_Guest-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Episode_Sentiment-Length_per_Guest', 'Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Dec', 'Podcast_Name-Publication_Time-Length_per_Guest-HPperc_Int', 'Host_Popularity_percentage-Publication_Time-Length_per_Guest', 'Podcast_Name-Publication_Day-Publication_Time-Length_per_Guest', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads-ELen_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Publication_Time-Guest_Popularity_percentage-Length_per_Ads', 'Podcast_Name-Publication_Time-Number_of_Ads-Length_per_Guest', 'Podcast_Name-Publication_Time-Episode_Sentiment-Length_per_Guest', 'Podcast_Name-Publication_Day-Episode_Sentiment-Length_per_Host', 'Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Dec', 'Publication_Day-Episode_Num-Length_per_Ads-Length_per_Host', 'Publication_Day-Publication_Time-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Publication_Day-Episode_Num-Length_per_Ads', 'Publication_Day-Number_of_Ads-Episode_Num-Length_per_Host', 'Genre-Publication_Day-Publication_Time-Length_per_Host', 'Podcast_Name-Publication_Day-Length_per_Host', 'Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment', 'Publication_Time-Episode_Sentiment-Episode_Num-Length_per_Host', 'Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Int', 'Publication_Day-Episode_Num-Length_per_Host-ELen_Int', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Num', 'Publication_Day-Episode_Num-Length_per_Host', 'Genre-Publication_Day-Length_per_Host-HPperc_Int', 'Genre-Publication_Day-Length_per_Host-ELen_Int', 'Episode_Length_minutes-Genre-Host_Popularity_percentage-Publication_Day', 'Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Podcast_Name-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Publication_Day-Episode_Sentiment-Length_per_Ads-Length_per_Guest', 'Podcast_Name-Episode_Length_minutes-Episode_Num-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-ELen_Int-HPperc_Dec', 'Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Int', 'Genre-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Int', 'Publication_Day-Publication_Time-Length_per_Guest-ELen_Int', 'Publication_Day-Episode_Sentiment-Length_per_Host-HPperc_Int', 'Publication_Day-Episode_Sentiment-Length_per_Host-ELen_Int', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Sentiment', 'Publication_Day-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Publication_Day-Episode_Num-Length_per_Ads-HPperc_Int', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Number_of_Ads', 'Publication_Time-Number_of_Ads-Length_per_Host-ELen_Dec', 'Publication_Time-Number_of_Ads-Length_per_Host-HPperc_Dec', 'Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Episode_Num-ELen_Int', 'Publication_Time-Episode_Num-Length_per_Ads-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int', 'Host_Popularity_percentage-Publication_Time-Episode_Num-ELen_Int', 'Episode_Length_minutes-Publication_Day-Episode_Num-HPperc_Int', 'Host_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Number_of_Ads-ELen_Int-HPperc_Int', 'Publication_Time-Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Publication_Time-Length_per_Host-ELen_Dec', 'Publication_Time-Length_per_Host-ELen_Dec-HPperc_Dec', 'Episode_Length_minutes-Host_Popularity_percentage-Publication_Time', 'Episode_Length_minutes-Publication_Time-Episode_Num-HPperc_Int', 'Episode_Length_minutes-Episode_Sentiment-Episode_Num-HPperc_Int', 'Episode_Length_minutes-Genre-Publication_Day-HPperc_Int', 'Host_Popularity_percentage-Episode_Num-ELen_Int', 'Guest_Popularity_percentage-Episode_Num-ELen_Int', 'Podcast_Name-Episode_Num-ELen_Int-HPperc_Int', 'Publication_Day-Publication_Time-Episode_Num-Length_per_Ads', 'Guest_Popularity_percentage-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Episode_Num-HPperc_Int', 'Episode_Length_minutes-Host_Popularity_percentage', 'Length_per_Host-ELen_Dec-HPperc_Dec', 'Publication_Day-Guest_Popularity_percentage-Number_of_Ads-ELen_Int', 'Episode_Length_minutes-Genre-Publication_Time-HPperc_Int', 'Publication_Day-Publication_Time-Guest_Popularity_percentage-ELen_Int', 'Episode_Length_minutes-Publication_Day-Publication_Time-Episode_Num', 'Publication_Day-Guest_Popularity_percentage-Episode_Sentiment-ELen_Int', 'Host_Popularity_percentage-Publication_Day-Publication_Time-ELen_Int', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-HPperc_Int', 'Episode_Length_minutes-Publication_Day-Publication_Time-HPperc_Int', 'Host_Popularity_percentage-Publication_Day-Episode_Sentiment-ELen_Int', 'Host_Popularity_percentage-Publication_Time-Number_of_Ads-ELen_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-Episode_Num', 'Episode_Length_minutes-Publication_Day-Episode_Sentiment-HPperc_Int', 'Host_Popularity_percentage-Publication_Time-Episode_Sentiment-ELen_Int', 'Publication_Time-Length_per_Ads-ELen_Dec-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-HPperc_Int', 'Episode_Sentiment-Length_per_Ads-ELen_Dec-HPperc_Int', 'Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-HPperc_Int', 'Publication_Day-Episode_Num-ELen_Int-HPperc_Int', 'Publication_Time-Episode_Num-ELen_Int-HPperc_Int', 'Number_of_Ads-Episode_Num-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Publication_Day-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Episode_Num', 'Episode_Sentiment-Episode_Num-ELen_Int-HPperc_Int', 'Host_Popularity_percentage-Publication_Time-ELen_Int', 'Podcast_Name-Number_of_Ads-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Episode_Sentiment-HPperc_Int', 'Podcast_Name-Publication_Time-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Genre-Publication_Time-Number_of_Ads', 'Episode_Num-ELen_Int-HPperc_Int', 'Podcast_Name-Episode_Sentiment-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Episode_Num', 'Episode_Length_minutes-HPperc_Int', 'Episode_Length_minutes-Publication_Day-Number_of_Ads-Episode_Sentiment', 'Publication_Time-Episode_Sentiment-Length_per_Ads-ELen_Dec', 'Publication_Time-Number_of_Ads-ELen_Int-HPperc_Int', 'Episode_Length_minutes-Publication_Time-Number_of_Ads-Episode_Sentiment', 'Publication_Day-Number_of_Ads-Episode_Num-ELen_Int', 'Publication_Day-Number_of_Ads-ELen_Int-HPperc_Int', 'Genre-Number_of_Ads-ELen_Int-HPperc_Int'
    ]

    # selecteds = random.sample(before_fe_selecteds, min(len(before_fe_selecteds), 50))
    combinations_list = []
    # combinations_list += [item.split("-") for item in default_selecteds]
    combinations_list += [item.split("-") for item in pte_selecteds]

    m = df_pltpd["Listening_Time_minutes"].mean()

    for cols in combinations_list:
        n = f"pte_{'_'.join(cols)}"
        means = df_pltpd.group_by(cols).agg(pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"))
        df = df.join(means, on=cols, how="left").with_columns(pl.col("mean_listening_time").fill_null(m).alias(n)).drop("mean_listening_time")

    return df



%%writefile data/simple_feature_eng.py
import polars as pl
from sklearn.model_selection import GroupKFold

GROUP_SPLIT = 2


def standardize(df: pl.DataFrame, df_train: pl.DataFrame, n_splits: int = GROUP_SPLIT) -> pl.DataFrame:
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df_train = df_train.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))
    df_train = df_train.with_columns(
        pl.col("Episode_Num").cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Num_Cat"),
    )
    df = df.with_columns(
        pl.col("Episode_Num").cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Num_Cat"),
    )

    # numeric_cols = df_train.select(cs.numeric()).columns
    numeric_cols = ["Listening_Time_minutes", "Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    categorical_cols = [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Num_Cat",
        "Episode_Sentiment",
        "Episode_Length_minutes_NaN",
        "Guest_Popularity_percentage_NaN",
    ]

    group_kfold = GroupKFold(n_splits=n_splits)
    df_update = pl.DataFrame()
    for (_, idx_valid), (t_idx_train, _) in zip(group_kfold.split(df, groups=df["fold"]), group_kfold.split(df_train, groups=df_train["fold"])):
        df_train_part = df_train[t_idx_train]
        stats = {
            col: {
                "mean": df_train_part.select(pl.col(col).mean()).item(),
                "std": df_train_part.select(pl.col(col).std()).item(),
            }
            for col in numeric_cols
        }

        # transformations = []
        # transform_cols = [col for col in numeric_cols if col != "Listening_Time_minutes"]
        # for col in transform_cols:
        #     transformations.append(((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).alias(f"{col}"))
        #     # transformations.append((pl.col(col) - stats[col]["mean"]).alias(f"{col}"))
        # df_update_part = df[idx_valid].with_columns(transformations)

        df_update_part = df[idx_valid]
        df_update_part = df_update_part.with_columns(
            pl.lit(stats["Listening_Time_minutes"]["mean"]).alias("Listening_Time_minutes_mean"),
            # pl.lit(stats["Listening_Time_minutes"]["std"]).alias("Listening_Time_minutes_std"),
        )

        for col in categorical_cols:
            # mean_target = df_train_part.group_by(col).agg(pl.col("Listening_Time_minutes").mean().alias(f"{col}_mean"))

            # df_update_part = df_update_part.join(mean_target, on=col, how="left").with_columns(
            #     pl.col(f"{col}_mean").fill_null(stats["Listening_Time_minutes"]["mean"]).alias(f"{col}_mean")
            # )

            # smoothing = np.random.randint(0, 5)
            smoothing = 0
            target_stats = df_train_part.group_by(col).agg(
                pl.col("Listening_Time_minutes").mean().alias("mean"), pl.col("Listening_Time_minutes").count().alias("count")
            )

            global_mean = stats["Listening_Time_minutes"]["mean"]
            target_stats = target_stats.with_columns(
                ((pl.col("count") * pl.col("mean") + smoothing * global_mean) / (pl.col("count") + smoothing)).alias(f"{col}_mean")
            )
            target_stats = target_stats.select([col, f"{col}_mean"])
            df_update_part = df_update_part.join(target_stats, on=col, how="left").with_columns(
                pl.col(f"{col}_mean").fill_null(stats["Listening_Time_minutes"]["mean"]).alias(f"{col}_mean")
            )

        df_update = pl.concat([df_update, df_update_part], how="vertical")

    df_update = df_update.sort("id")
    df = df.with_columns(df_update)
    # df = df.drop(categorical_cols)

    return df



%%writefile models/tabnet.py
import os
import warnings

import numpy as np
import polars as pl
import gc
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetRegressor

import wandb
from config import cfg
from data.data_class import DatasetXy

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# Create a proper callback class
class WandbCallback(Callback):
    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            wandb.log({metric_name: metric_value})
        return False


def train_model(fold: int, datasetXy: DatasetXy):
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    # Handle categorical columns
    cat_cols = [col for col in X_train.columns if pl.Categorical in X_train[col].dtype.base_type().__mro__ or X_train[col].dtype == pl.Utf8]
    cat_cols_idx = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
    category_mappings = {}
    cat_dims = []

    combined_data = pl.concat([X_train, X_valid], how="vertical")
    for col in cat_cols:
        cat_dims.append(combined_data[col].unique().count())

        unique_values = combined_data[col].unique().sort()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        category_mappings[col] = mapping

        X_train = X_train.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))
        X_valid = X_valid.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))

    print(X_train)
    print(X_train.null_count())
    
    # Convert Polars DataFrames to numpy arrays
    X_train_np = X_train.to_numpy()
    X_valid_np = X_valid.to_numpy()

    # Reshape target variables
    y_train_np = y_train.to_numpy().reshape(-1, 1) if hasattr(y_train, "to_numpy") else np.array(y_train).reshape(-1, 1)
    y_valid_np = y_valid.to_numpy().reshape(-1, 1) if hasattr(y_valid, "to_numpy") else np.array(y_valid).reshape(-1, 1)

    print(X_train_np.shape, y_train_np.shape, X_valid_np.shape, y_valid_np.shape)
    print(X_train_np)

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {"learning_rate": 2e-2, "n_iter": 200, "early_stopping": 10, "metric": "rmse", "n_d": 64, "n_a": 64, "n_steps": 5}
    wandb_run = wandb.init(project="playground-series-s5e4", config=config)

    # Set up model with proper categorical indices and dimensions
    tabnet_params = {
        # Architecture parameters
        "n_d": 64,  # Width of the decision prediction layer (increased from default 8)
        "n_a": 64,  # Width of the attention embedding for each step (increased from default 8)
        "n_steps": 5,  # Number of steps in the architecture (increased from default 3)
        "gamma": 1.5,  # Coefficient for feature reusage in the masks
        # For categorical features from your feature engineering
        "cat_idxs": cat_cols_idx,
        "cat_dims": cat_dims,
        "cat_emb_dim": 3,  # Embedding dimension for categorical features (increased slightly)
        # Feature selection parameters
        "n_independent": 2,  # Number of independent Gated Linear Units layers
        "n_shared": 3,  # Number of shared Gated Linear Units (increased from default)
        # Regularization parameters
        "lambda_sparse": 0.005,  # Sparsity regularization (increased slightly)
        "momentum": 0.3,  # Ghost Batch Norm momentum (increased)
        "clip_value": 2,  # Gradient clipping value (increased)
        # Training parameters
        "optimizer_fn": torch.optim.AdamW,
        "optimizer_params": {
            "lr": 0.01,
            "weight_decay": 1e-5,
        },
        # Learning rate scheduler
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 10, "verbose": True},
        # Other parameters
        "mask_type": "entmax",
        "verbose": 1,
        "seed": cfg.random_state,
        "device_name": cfg.device,
    }
    model = TabNetRegressor(**tabnet_params)

    # Train the model with our proper callback class
    model.fit(
        X_train=X_train_np,
        y_train=y_train_np,
        eval_set=[(X_train_np, y_train_np), (X_valid_np, y_valid_np)],
        eval_name=["train", "valid"],
        eval_metric=["rmse", "rmse"],
        max_epochs=500,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        callbacks=[WandbCallback()],
    )

    del X_train_np, y_train_np, X_valid_np, y_valid_np
    gc.collect()
    torch.cuda.empty_cache()

    # Get validation score
    val_score = min(model.history["valid_rmse"])
    print(f"Validation score: {val_score}")
    wandb.summary["best_val_score"] = val_score

    # feature_importances = model.feature_importances_
    # importance_dict = {col: imp for col, imp in zip(X_train.columns, feature_importances)}

    # # Log feature importances as a wandb Table
    # feature_importance_table = wandb.Table(columns=["Feature", "Importance"])
    # for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    #     feature_importance_table.add_data(feature, importance)

    # # Log the table and also as a bar chart for visualization
    # wandb.log({"feature_importances": feature_importance_table})
    # wandb.log({"feature_importance_plot": wandb.plot.bar(feature_importance_table, "Feature", "Importance", title="Feature Importances")})

    # # Also log as simple key-value pairs for easy access
    # wandb.log({"importance/" + feature: importance for feature, importance in importance_dict.items()})

    if hasattr(cfg, "predict") and cfg.predict:
        for col, mapping in category_mappings.items():
            X_test = X_test.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))
        X_test_np = X_test.to_numpy()
        y_test = model.predict(X_test_np)
        
        del X_test_np
        gc.collect()
        torch.cuda.empty_cache()
        
        return val_score, y_test.flatten().tolist()

    return val_score, None




%%writefile main.py
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import gc
import warnings
from dataclasses import asdict

import numpy as np
import polars as pl
# from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

import wandb
from config import cfg
from data.data_class import Dfs
from data.data_process import add_fold, get_Xy

from kaggle_secrets import UserSecretsClient

# from data.simple_data_process import add_fold, get_Xy
# from models.lgb import train_model
from models.tabnet import train_model
# from models.hgbr import train_model
# from models.svr import train_model
# from models.xgb import train_model

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Initialize wandb
user_secrets = UserSecretsClient()
WANDB_API_KEY = user_secrets.get_secret("wandb_api")
wandb.login(key=WANDB_API_KEY)
wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(cfg))

df = pl.read_csv(cfg.train_path)
df = df.filter(pl.col("Number_of_Ads").is_not_null())
df = add_fold(df)

df_test = None
if hasattr(cfg, "predict") and cfg.predict:
    df_test = pl.read_csv(cfg.test_path)
    df_test = add_fold(df_test)


def save_sub(test_preds):
    if hasattr(cfg, "predict") and cfg.predict:
        test_pred = np.array(test_preds).mean(axis=0)

        test_df = pl.read_csv(cfg.test_path)
        test_df = test_df.with_columns(pl.Series(test_pred).alias("Listening_Time_minutes"))
        test_df = test_df[["id", "Listening_Time_minutes"]]
        wandb_num = wandb_run.name.split("-")[-1]
        test_df.write_csv(f"/kaggle/working/sub-{wandb_num}.csv")
        print(f"Test predictions saved to /kaggle/working/sub-{wandb_num}.csv")


val_score = None
test_preds = []
# group_kfold = GroupKFold(n_splits=cfg.num_fold, shuffle=True, random_state=cfg.random_state)
group_kfold = GroupKFold(n_splits=cfg.num_fold)
for fold, (idx_train, idx_valid) in enumerate(group_kfold.split(df, groups=df["fold"])):
    df_train = df[idx_train]
    df_valid = df[idx_valid]

    datasetXy = get_Xy(Dfs(df_train=df_train, df_valid=df_valid, df_test=df_test))
    print(f"Fold {fold} - Train shape: {datasetXy.X_train.shape}")
    print(datasetXy.X_train)

    val_score, test_pred = train_model(fold, datasetXy)
    test_preds += [test_pred]

    gc.collect()
    if hasattr(cfg, "eval") and cfg.eval and fold >= 1:
        break

    save_sub(test_preds)

# git_info = commit_results(val_score, wandb_run.name)
# wandb.config.update(git_info)
wandb.finish()




%%writefile config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    train_path: Path = Path("/kaggle/input/playground-series-s5e4/train.csv")
    test_path: Path = Path("/kaggle/input/playground-series-s5e4/test.csv")
    sub_path: Path = Path("/kaggle/input/playground-series-s5e4/sample_submission.csv")
    pltpd_path: Path = Path("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

    model_path: Path = Path("./")
    test_output_path: Path = Path("./data/submission.csv")

    # num_fold: int = 5
    num_fold: int = 7
    dev_mode: bool = False

    # Model parameters
    n_iter: int = 10000
    max_depth: int = -1

    # num_leaves: int = 1024
    # colsample_bytree: float = 0.5
    # learning_rate: float = 0.04
    # random_state: int = 42

    num_leaves: int = 4096
    colsample_bytree: float = 0.5
    learning_rate: float = 0.02
    random_state: int = 142

    objective: str = "l2"
    metric: str = "rmse"
    verbosity: int = -1

    shuffle: bool = True
    log_eval: int = 100
    early_stopping: int = 200

    # default_combinations: bool = True

    # debug: bool = True
    # eval: bool = True
    predict: bool = True

    device: str = "cuda:0"


cfg = CFG()

if hasattr(cfg, "eval") and cfg.eval:
    cfg.n_iter = 2000
if hasattr(cfg, "debug") and cfg.debug:
    cfg.n_iter = 5



!python main.py







