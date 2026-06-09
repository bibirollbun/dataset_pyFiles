!git clone https://github.com/Loi-ND/recsys.git


# !pip install -U lightgbm==3.3.2


!pip install implicit


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

import pickle
from tqdm import tqdm
import gc
from pathlib import Path


import warnings
import sys
import os
from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings("ignore")
sys.path.append("/kaggle/working/recsys") # path to the `src`` folder
InteractiveShell.ast_node_interactivity = "all"
tqdm.pandas()


!touch /kaggle/working/recsys/src/__init__.py


from src.data import DataHelper
from src.data.metrics import map_at_k, hr_at_k, recall_at_k

from src.retrieval.rules import (
    OrderHistory,
    OrderHistoryDecay,
    ItemPair,
    UserGroupTimeHistory,
    UserGroupSaleTrend,
    TimeHistory,
    TimeHistoryDecay,
    SaleTrend,
    OutOfStock,
    
)

from src.retrieval.rules import *


from src.retrieval.collector import RuleCollector

from src.features import full_sale, week_sale, repurchase_ratio, popularity, period_sale

from src.utils import (
    calc_valid_date,
    merge_week_data,
    reduce_mem_usage,
    calc_embd_similarity,
)


data_dir = Path("/kaggle/working/recsys/data/")
model_dir = Path("/kaggle/working/recsys/models/")


TRAIN_WEEK_NUM = 4
WEEK_NUM = TRAIN_WEEK_NUM + 2

VERSION_NAME = "UG_Agebin_club_u"
TEST = True # * Set as `False` when do local experiments to save time
params_01 = {
    'days' : [3, 7, 14, 30],
    'n' : [100]
}

params_02 = {
    'days' : [3, 7, 14, 30],
    'n' : [100]
}
'FN','Active','club_member_status','fashion_news_frequency','user_gender','age_bins'
name_col = 'Agebin_club'
cols_list = ['age_bins', 'club_member_status']


os.listdir('/kaggle/input/h-and-m-personalized-fashion-recommendations/')


import os
if not os.path.exists(data_dir/"interim"/VERSION_NAME):
    os.makedirs(data_dir/"interim"/VERSION_NAME)
if not os.path.exists(data_dir/"processed"/VERSION_NAME):
    os.makedirs(data_dir/"processed"/VERSION_NAME)
os.makedirs(data_dir/"raw", exist_ok=True)


dh = DataHelper(data_dir, 'raw')
dh.raw_dir  = data_dir / '../../../input/h-and-m-personalized-fashion-recommendations/'
dh.raw_dir


!pip install pyarrow fastparquet


data = dh.preprocess_data(save=True, name="encoded_full") # * run only once, processed data will be saved


data = dh.load_data(name="encoded_full")


uid2idx = pickle.load(open(data_dir/"index_id_map/user_id2index.pkl", "rb"))
submission = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
submission['customer_id'] = submission['customer_id'].map(uid2idx)


listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
data['user']['age_bins'] = pd.cut(data['user']['age'], listBin)


inter = data["inter"].copy()
users = data["user"][["customer_id", "club_member_status", 'Active']].copy()
numbers = users.groupby(["club_member_status", 'Active']).size()
# Gộp để inter có thông tin Active
inter = inter.merge(users, on="customer_id", how="left")
trans_per_active = inter.groupby(["club_member_status", 'Active']).size()



def evaluate_rule(rule_name, rules, trans, customer_list, valid_exploded, week):
    print(f"\n====== TEST RULE: {rule_name} (week {week}) ======")

    candidates = RuleCollector().collect(
        week_num = week,
        trans_df = trans,
        customer_list=customer_list,
        rules=rules,
        min_pos_rate=0.00,
        compress=False,
    )
    if candidates is None:
        result = {
            "rule": rule_name,
            "num_candidates": 0,
            "num_matched": 0,
            "precision": 0,
            "recall": 0
        }
        return result, None
    # pivot
    candidates = (
        pd.pivot_table(
            candidates,
            values="score",
            index=["customer_id", "article_id"],
            columns=["method"],
            aggfunc=np.sum,
            observed=True
        )
        .reset_index()
    )

    # ---- CALCULATE PRECISION & RECALL ----
    matched = candidates.merge(valid_exploded, on=['customer_id','article_id'], how='inner')
    num_matched = len(matched)
    total_valid = len(valid_exploded)

    recall = num_matched / total_valid
    precision = num_matched / len(candidates)

    result = {
        "rule": rule_name,
        "num_candidates": len(candidates),
        "num_matched": num_matched,
        "precision": precision,
        "recall": recall
    }

    print(result)
    return result, candidates



class ItemGroupTimeHistory(ItemGroupRetrieveRule):
    """Retrieve popular items of each **item group** in a specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        item_df: pd.DataFrame,
        cat_cols: List[str],
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
        days: int = 7,
    ):
        """
        Parameters
        ----------
        customer_list : List
            List of target customer IDs.
        trans_df : pd.DataFrame
            Transaction dataframe with at least [customer_id, article_id, t_dat].
        cat_cols : List[str]
            Columns defining item groups (e.g., ['product_type_no', 'colour_group_code']).
        n : int
            Number of top items to retrieve per group.
        name : str
            Name of the rule, for method labeling.
        t : float
            Trend threshold ratio (optional, can be used if comparing two periods).
        item_id : str
            Column name for item ID.
        days : int
            Length of time window for popularity calculation.
        """
        self.customer_list = customer_list
        self.trans_df = trans_df[['customer_id', item_id, "t_dat"]].copy()
        self.item_df = item_df[[item_id, *cat_cols]].copy()
        self.trans_df = self.trans_df.merge(
            self.item_df, on=item_id, how="left"
        )
        self.trans_df['t_dat'] = pd.to_datetime(self.trans_df['t_dat'])

        # 1. Lấy dữ liệu gần nhất
        max_date = self.trans_df['t_dat'].max()
        min_date = max_date - pd.Timedelta(days=days)
        self.trans_df = self.trans_df[self.trans_df['t_dat'] >= min_date]
        self.cat_cols = cat_cols
        self.iid = item_id
        self.n = n
        self.name = name
        self.t = t
        self.days = days

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df.copy()
        df['t_dat'] = pd.to_datetime(df['t_dat'])

        # 1. Lấy dữ liệu gần nhất
        max_date = df['t_dat'].max()
        min_date = max_date - pd.Timedelta(days=self.days)
        df = df[df['t_dat'] >= min_date].copy()

        df["count"] = 1
        df = df.groupby([*self.cat_cols, self.iid], as_index=False)["count"].sum()
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df["rank"] = df.groupby([*self.cat_cols])["index"].rank(
            ascending=True, method="first"
        )

        # if self.scale:
        #     df["score"] = df["count"] / df["count"].max()
        # else:
        df["score"] = df["count"]
        df["method"] = "IGTimeHistory_" + self.name
        df = df[df["rank"] <= self.n][[*self.cat_cols, self.iid, "score", "method"]]

        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score", 'rank']]
        
   
class ItemGroupSaleTrend(ItemGroupRetrieveRule):
    """Retrieve trending items in a specified time window for each item group."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        item_df: pd.DataFrame,
        cat_cols: List[str],
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """
        Parameters
        ----------
        customer_list : List
            List of target customer IDs.
        trans_df : pd.DataFrame
            Transaction dataframe with at least ['customer_id', 'article_id', 't_dat'] + item group columns.
        cat_cols : List[str]
            Columns defining item groups (e.g., ['product_type_no', 'colour_group_code']).
        days : int
            Length of time window to calculate trends.
        n : int
            Top N items per item group.
        name : str
            Name of the rule for labeling.
        t : float
            Minimum trend ratio to consider an item trending.
        item_id : str
            Column name for item ID.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans[["customer_id", self.iid, "t_dat"]]
        self.item_df = item_df[[item_id, *cat_cols]].copy()
        self.trans_df = self.trans_df.merge(
            self.item_df, on=item_id, how="left"
        )
        self.cat_cols = cat_cols
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df  # KHÔNG copy

        t_dat = pd.to_datetime(df["t_dat"])
        dat_gap = (t_dat.max() - t_dat).dt.days

        # lọc 2 * days gần nhất
        df = df.loc[dat_gap <= 2 * self.days - 1].copy()
        dat_gap = dat_gap.loc[df.index]

        group_a = df.loc[dat_gap > self.days - 1].copy()   # cũ hơn
        group_b = df.loc[dat_gap <= self.days - 1].copy()  # gần đây

        del t_dat 
        del dat_gap
        group_a["count"] = 1
        group_b["count"] = 1

        group_a = group_a.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        group_b = group_b.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[*self.cat_cols, self.iid], how="left")
        log["count_y"] = log["count_y"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_x"]

        log = log[log["trend"] > self.t]
        log["rank"] = log.groupby(self.cat_cols)["trend"].rank(
            ascending=False, method="first"
        )
        log = log[log["rank"] <= self.n]

        log["method"] = f"ItemGroupSaleTrend_{self.name}"
        log["score"] = log["trend"]

        log = log[[*self.cat_cols, self.iid, "method", "score"]]

        result_df = self.merge(log)
        return result_df[["customer_id", self.iid, "method", "score", 'rank']]


all_results = []
for week in range(1, WEEK_NUM):

    trans = data["inter"]
    items = data["item"]
    start_date, end_date = calc_valid_date(week)
    train, valid = dh.split_data(trans, start_date, end_date)
    customer_cols = ['customer_id','Active','club_member_status','age_bins']

    item_cols = ['product_type_no']

    train = train.merge(data['user'][customer_cols], on='customer_id', how='left')

    
    last_week_start = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    last_week = train.loc[train.t_dat >= last_week_start]

    last_3day_start = (pd.to_datetime(start_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    last_3days = train.loc[train.t_dat >= last_3day_start]

    valid_exploded = valid.explode('article_id')
    customer_list = valid["customer_id"].values
    
    RULE_SET = {
        f"UGTH_{name_col}_{days}_{n}" : [UserGroupTimeHistory(data, customer_list, train, cols_list, days=days, n=n, name='1')] for days in params_01['days'] for n in params_01['n']
        # "UGTH3_10" : [UserGroupTimeHistory(data, customer_list, train, ['age_bins'], days=3, n=10, name='1')]
    }

    RULE_SET.update({
        f"UGST_{name_col}_{days}_{n}" : [UserGroupSaleTrend(data, customer_list, train, cat_cols=cols_list, days=days, n=n, name='1')] for days in params_02['days'] for n in params_02['n']
    })
    
    print(f"\n========= WEEK {week} | valid size = {len(valid_exploded)} =========")

    # ⚡ Chạy lần lượt từng rule
    for rule_name, rule_obj in RULE_SET.items():
        result, candidates = evaluate_rule(
            rule_name=rule_name,
            rules=rule_obj,
            trans=trans,
            customer_list=customer_list,
            valid_exploded=valid_exploded,
            week=week
        )

        # lưu result
        all_results.append(result)
        # xuất candidate từng rule
        if candidates is None:
            continue
        candidates.to_parquet(
            data_dir/"interim"/VERSION_NAME/f"week{week}_{rule_name}_candidates.pqt"
        )

    valid.to_parquet(data_dir/"processed"/VERSION_NAME/f"week{week}_label.pqt")
    gc.collect()

df_result = pd.DataFrame(all_results)
df_result.to_csv(data_dir/"interim"/VERSION_NAME/"all_rule_results.csv", index=False)
df_result.sort_values("precision", ascending=False)



