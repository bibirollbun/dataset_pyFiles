# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install duckdb
# !pip install hmmlearn
# !pip install ipykernel
# !pip install lightgbm
# !pip install matplotlib
# !pip install mlflow
# !pip install numpy
# !pip install pandas
# !pip install pandera
# !pip install plotly
# !pip install polars
# !pip install pyarrow
# !pip install pydantic
# !pip install scikit-learn
# !pip install shap
# !pip install tqdm
# !pip install xgboost


import sys

sys.path.append("/kaggle/input/dal-jet-engine-mabe-challenge/src")

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from helper import Solution, list_parquet_ids, get_df_test, get_labid_from_videoid
from lgbm import LGBM
from metamodel import MetaLR, default_metalr_params
# from postprocess import majority_filter_3frame
from xgbm import XGBM
# from dnn import STGCN
# from hmm_supervised import HMM
from preprocess import preprocess, get_label

import polars as pl


def get_mouse_indices(df_metadata: pd.DataFrame, video_id: int) -> list[int]:
    """指定したメタデータとvideo_idから動画に登場するマウスのIDを取得

    Args:
        df (pd.DataFrame): メタデータのデータフレーム(train.csv, test.csv)
        video_id (int): マウスIDを取得する動画のID

    Returns:
        list[int]: マウスIDのリスト(例:[1,2,3])

    """
    row = df_metadata.loc[df_metadata["video_id"] == video_id].iloc[0]
    cols = ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]

    # colsのうち欠損していないものを探し、それに応じてマウスIDを抽出(1~4)
    mouse_id_list = [
        i + 1
        for i, c in enumerate(cols)
        if pd.notna(row[c]) and row[c] != ""  # 念のため空文字も除外
    ]
    if video_id in [
        1260392287,
        1351098077,
        1643942986,
    ]:  # これらの動画データだけマウス3がtrackingデータで4と表記されている
        mouse_id_list = [1, 2, 4]

    return mouse_id_list


# モデルのリスト
model_dict = {"lgbm": LGBM, 
              "xgbm": XGBM, 
              "hmm":LGBM,
              # "stgcn":STGCN
             }

# モデルのパス
lgbm_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/LGBM_202512151703"
xgbm_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/XGBM_202512151744"
# hmm_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/hmm_202512151804"
# stgcn_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/STGCN_202512152210"
model_path_dict = {"lgbm":lgbm_results_path,
                   "xgbm":xgbm_results_path,
                   "hmm":lgbm_results_path,
                   # "stgcn":stgcn_results_path,
                  }

# メタモデル・後処理
# postprocess_list = [
#     "MetaLR",
#     # "MetaHnn",
#     "HMMPostProcessor",
#     "majority_filter_3frame",
# ]

postprocess_model_dict = {
    "MetaLR": MetaLR,
    # "MetaHnn": MetaHnn,
    # "HMMPostProcessor": HMMPostProcessor,
    # "majority_filter_3frame": majority_filter_3frame,
}

metalr_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/MetaLR_202512160002"
# hmmpp_results_path = "/kaggle/input/dal-jet-engine-mabe-challenge/model/HMMPP_202512160343"

# postprocess_path_dict = {"MetaLR":metalr_results_path}
metalr_path_dict = {"MetaLR":metalr_results_path, 
                    # "HMMPostProcessor":hmmpp_results_path,
                   }

# 行動リスト
action_list = ["submit", "chaseattack", "chase", "approach", "attack"]


video_list = list_parquet_ids("/kaggle/input/MABe-mouse-behavior-detection/test_tracking")
df_test = get_df_test(video_list)


preprocessed_path = Path("/kaggle/working/preprocessed/")
preprocessed_path.mkdir(parents=True, exist_ok=True)


for i in range(len(df_test)):
    video_id = int(df_test.iloc[i]["video_id"])
    agent_id = int(df_test.iloc[i]["agent_id"])
    target_id = int(df_test.iloc[i]["target_id"])
    lab_id = get_labid_from_videoid(video_id)

    mouse_list = list(
        pd.read_parquet(
            f"/kaggle/input/MABe-mouse-behavior-detection/test_tracking/{lab_id}/{video_id}.parquet"
        )["mouse_id"].unique()
    )

    X = preprocess(video_id, agent_id, target_id)

    X = X.filter(
        ~pl.any_horizontal(
            pl.all().is_null()
            | pl.all().is_nan()
            | pl.all().is_infinite()
        )
    )

    X = X.to_pandas()
    X["video_id"] = video_id
    X["agent_id"] = agent_id
    X["target_id"] = target_id
    X = X.rename(columns={
        "nose_centroid_costheta_agent_to_target": "head_centroid_costheta_agent_to_target"
    })

    X.to_parquet("/kaggle/working/preprocessed/" + f"{video_id}_{agent_id}_{target_id}.parquet")


solution = Solution()
for action in tqdm(action_list):
    solution_test_each_model = {}

    # 各モデルの予測結果を取得
    for model_name, model in model_dict.items():
        if model_name is "hmm":
            m = model.load(model_path_dict[model_name] + f"/lgbm_{action}.pkl")
        else:
            m = model.load(model_path_dict[model_name] + f"/{model_name}_{action}.pkl")

        sol_test = m.predict(df_test)
        solution_test_each_model[model_name] = sol_test

    # メタモデル(ロジスティック回帰)でスタッキング
    metalr = MetaLR.load(metalr_results_path + f"/metalr_{action}.pkl")
    if not hasattr(metalr.model, "multi_class"):
        metalr.model.multi_class = "auto" 
    meta_prediction_test = metalr.predict(solution_test_each_model)

    # # HMMでスムージング
    # hmmpp = HMMPostProcessor.load(hmmpp_results_path + f"/hmm_postprocessor_{action}.pkl")
    # meta_prediction_test = Solution().add_df(hmmpp.smooth(meta_prediction_test.df))

    # # majority filterでスムージング
    # meta_prediction_test.df = majority_filter_3frame(meta_prediction_test.df)
    solution.add_df(meta_prediction_test.df)

df = solution.make_submission_df()

df_out = (
    df
    .assign(
        # row_id を新規作成
        row_id=lambda x: range(len(x)),

        # agent_id / target_id を mouseX 形式に変換
        agent_id=lambda x: "mouse" + x["agent_id"].astype(str),
        target_id=lambda x: "mouse" + x["target_id"].astype(str),
    )
    # 列順を指定
    [["row_id", "video_id", "agent_id", "target_id",
      "action", "start_frame", "stop_frame"]]
)

df_out.to_csv("/kaggle/working/submission.csv", index=False)


df_out

