from typing import List, Any

import polars as pl
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datetime import datetime
from tqdm.notebook import tqdm
import joblib
import gc

import sys
import os
import kaggle_evaluation.jane_street_inference_server
sys.path.append(r"/kaggle/input/dnn-online-config")
from eval_metric import *
from config import Config


def same_seed(seed: int) -> None:
    """设置随机数种子（便于复现）

    Parameters
    ----------
    seed : int
        要设置的随机数种子的值
    """
    # 设置 CUDNN 为确定性操作，确保每次运行结果相同（有助于复现）
    torch.backends.cudnn.deterministic = True
    
    # 禁用 CUDNN 的自动优化功能，以避免影响确定性行为
    torch.backends.cudnn.benchmark = False
    
    # 设置 numpy 的随机种子，确保 numpy 的随机数生成器是可复现的
    np.random.seed(seed)
    
    # 设置 Pytorch 的 CPU 随机种子，确保 CPU 上的操作是可复现的
    torch.manual_seed(seed)
    
    # 如果有 GPU 可用，设置所有 GPU 设备的随机数种子，确保 GPU 上的操作也是可复现的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    pl.set_random_seed(seed)
        
    # 打印随机数种子值，便于追踪
    print(f"Set Seed = {seed}")

def r2_score(y_true: torch.Tensor, 
             y_pred: torch.Tensor, 
             sample_weight: torch.Tensor):
    numerator = torch.sum(sample_weight * (y_pred - y_true) ** 2) / torch.sum(sample_weight)
    denominator = torch.sum(sample_weight * (y_true ** 2)) / torch.sum(sample_weight) + 1e-38
    return 1 - numerator / denominator

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, y_true, y_pred, sample_weight):
        numerator = torch.mean(sample_weight*((y_pred - y_true) ** 2))
        denominator = torch.mean(sample_weight*(y_true ** 2)) + 1e-38
        return numerator / denominator

class JsDNNDataset(Dataset):
    def __init__(self, config: Config, df: pl.DataFrame):
        self.X: torch.Tensor = torch.tensor(df.select(config.features).to_numpy(), dtype=torch.float32)
        self.target: torch.Tensor = torch.tensor(df.select(config.target).to_numpy().squeeze(), dtype=torch.float32)
        self.weight: torch.Tensor = torch.tensor(df.select(config.sample_weight).to_numpy().squeeze(), dtype=torch.float32)
        
        # 打印形状以检查数据是否正确
        # print(f"X shape: {self.X.shape}")
        # print(f"target shape: {self.target.shape}")
        # print(f"weight shape: {self.weight.shape}")
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        # print(f"Accessing index {idx}")  # 打印索引调试信息
        return self.X[idx], self.target[idx], self.weight[idx]
    
class DNN(nn.Module):
    
    def __init__(self, input_dim: int,
                       hidden_dims: List[int],
                       dropouts: List[float],):
        super(DNN, self).__init__()
        
        assert len(hidden_dims) == len(dropouts)
        
        layers: List[Any] = []
        self.input_dim = input_dim
        
        # 设置隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.SiLU())
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(dropouts[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.SiLU())
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(dropouts[i]))
              
        # 设置输出层  
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze(-1)


def create_agg_list(day, columns, agg_type: str, last=True):
    agg_mean_list = [pl.col(c).mean().name.suffix(f"_mean_{agg_type}_{day}d") for c in columns]
    agg_std_list = [pl.col(c).std().name.suffix(f"_std_{agg_type}_{day}d") for c in columns]
    agg_max_list = [pl.col(c).max().name.suffix(f"_max_{agg_type}_{day}d") for c in columns]
    agg_min_list = [pl.col(c).min().name.suffix(f"_min_{agg_type}_{day}d") for c in columns]
    
    agg_list = agg_mean_list + agg_std_list + agg_max_list + agg_min_list
    
    if last:
        agg_last_list = [pl.col(c).last().name.suffix(f"_last_{agg_type}_{day}d") for c in columns]
        agg_list += agg_last_list
        
    return agg_list


class JaneStreetOnlinePredictor:
    
    def __init__(self, 
                 config: Config,
                 dnn_model,
                 gbdt_models, 
                 train_data: pl.DataFrame,
                 history_data: pl.DataFrame,
                 retrain_interval: int):
        self.config: Config = config
        
        # 配置 nn model 训练器的参数
        self.dnn_model = dnn_model
        self.criterion = WeightedMSELoss()
        self.optimizer = optim.Adam(self.dnn_model.parameters(), lr=5e-6, weight_decay=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dnn_model.to(self.device)
        
        self.train_data: pl.DataFrame = train_data
        self.retrain_data: pl.DataFrame = None
        self.scored_data: pl.DataFrame = None # 用于计算测试 score 的数据
        self.lags: pl.DataFrame = None
        self.new_day_test: List[pl.DataFrame] = []
        
        self.retrain_interval: int = retrain_interval
        self.count_date: int =  0
        self.max_date_id: int = (
            self.train_data
            .select("date_id")
            .max()
            .to_series()
            .cast(pl.Int32)
            .to_numpy()[0]
        )
        self.min_date_id: int = (
            self.train_data
            .select("date_id")
            .min()
            .to_series()
            .cast(pl.Int32)
            .to_numpy()[0]
        )
    
        self.is_zero_day = True 
        self.retrain = False

        # 配置 gbdt model 需要的数据
        self.gbdt_models = gbdt_models
        self.history_data = history_data
        
        self.lags_infer: pl.DataFrame = None
    
    def retrain_dnn(self):

        self.dnn_model.train()
        retrain_dataloader = DataLoader(JsDNNDataset(self.config, self.retrain_data), 
                                        batch_size=8196, shuffle=True)
        epochs = 5
        for epoch in range(epochs):
            print(f"epoch: {epoch+1}")
            
            if self.config.debug:
                y_train_true_list = []
                y_train_pred_list = []
                weight_train_list = []
                
            retrain_pbar= tqdm(retrain_dataloader, 
                               total=len(retrain_dataloader), 
                               desc=f"Epoch [{epoch + 1} / {epochs}]")
            for X, y, weight in retrain_pbar:
                self.optimizer.zero_grad()
                X, y, w = X.to(self.device), y.to(self.device), weight.to(self.device)
                y_train_pred = self.dnn_model(X)
                loss = self.criterion(y, y_train_pred, w)
                loss.backward()
                self.optimizer.step()

                if self.config.debug: 
                    y_train_true_list.append(y.cpu())
                    y_train_pred_list.append(y_train_pred.detach().cpu())
                    weight_train_list.append(w.cpu())
            if self.config.debug:
                train_score = r2_score(y_true=torch.cat(y_train_true_list),
                                        y_pred=torch.cat(y_train_pred_list),
                                        sample_weight=torch.cat(weight_train_list))
                print(f"目前的训练分数：{train_score}")
    
    def update_data(self, lags: pl.DataFrame):
        
        # 将前一天的 test 数据和 lags(label) 合并，并统一数据类型
        test = self.join_lags_with_test(lags)
        
        # 从历史的 train_data 中采样数据，然后和前一天的 test data 合并, 并填充缺失值
        sampled_data = (
            self.train_data
            .sample(fraction=1/48, with_replacement=False)
            .cast(self.config.row_column_tpyes)
        )
        self.retrain_data = pl.concat([sampled_data, test]).fill_null(value=0)
        
        # 新旧数据的 push 和 pop
        self.train_data = (
            pl.concat([self.train_data, test]) # push 
            .filter(pl.col("date_id") > self.min_date_id) # pop
        )

        # 在 debug 时实时计算每一天累计的测试数据的 score
        if self.config.debug:
            if self.scored_data is None:
                self.scored_data = test.select(["responder_6", "pred", "weight"])
            else:
                self.scored_data = pl.concat([self.scored_data,test.select(["responder_6", "pred", "weight"])])
            
        # 数据更新完成后，将 new_day_test 重新弄设置为空列表，接收下一个 date 的数据
        self.new_day_test = []
        
        gc.collect()
        return
    
    def cal_current_test_score(self):
        self.current_score = self.config.score(
            y_true=self.scored_data.select(self.config.target).to_numpy().squeeze(),
            y_pred=self.scored_data.select("pred").to_numpy().squeeze(),
            sample_weight = self.scored_data.select(self.config.sample_weight).to_numpy().squeeze())
    
    def join_lags_with_test(self, lags: pl.DataFrame) -> pl.DataFrame:
        
        test: pl.DataFrame = pl.concat(self.new_day_test)
        test = (
            test
            .with_columns(
                pl.lit(self.max_date_id)
                .alias("date_id")
            )
        )
        
        # lags 2 label
        lags2label = (
            lags
            .with_columns(
                pl.lit(self.max_date_id)
                .alias("date_id")
            )
            .rename({f"{resp}_lag_1": resp for resp in [f"responder_{i}" for i in range(9)]})
        )
        
        # join test with label
        test = test.join(lags2label, on=["symbol_id", "date_id", "time_id"], how="left")
        test = test.cast(self.config.row_column_tpyes)

        # join test with lag features
        self.lags_infer = (
            self.lags_infer
            .with_columns(
                pl.lit(self.max_date_id)
                .alias("date_id")
            )
            .cast({"date_id": pl.Int16})
        )
        test = test.join(self.lags_infer, on=["symbol_id", "date_id"], how="left")
        
        return test.select(self.train_data.columns)

    def feature_engineering(self, lags: pl.DataFrame, current_date: int) :
        
        # 原始lags先存储到history更新历史数据
        lags = lags.rename(self.config.lag_cols_rename)
        lags = lags.cast(self.config.id_column_types)
        lags = lags.cast(self.config.responder_column_types)

        self.history_data = pl.concat([self.history_data, lags])
        
        # 只储存最近N天的历史数据
        self.history_data = (
            self.history_data
            .filter(pl.col("date_id") > (current_date - self.config.lag_n_days))
            .sort(["symbol_id", "date_id", "time_id"])
        )

        # shift 1天的统计值
        shift_1d_agg_list = create_agg_list(1, self.config.responder_cols, agg_type="shift", last=True)
        shift_1d_data = self.history_data.filter(pl.col("date_id") == current_date)
        shift_1d_lags_infer = (
            shift_1d_data
            .group_by(["symbol_id", "date_id"], maintain_order=True)
            .agg(shift_1d_agg_list)
        )
        
        self.lags_infer = shift_1d_lags_infer
    
    def predict(self, 
                test: pl.DataFrame,
                lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
        current_date = test.select("date_id").to_numpy()[:, 0][0]
        if lags is not None:
            
            self.count_date += 1
            
            if not self.is_zero_day:
                print(self.min_date_id)
                # ============================================== 更新数据 ====================================
                start1 = datetime.now()
                
                # lags 和 lags2label 都在第 1 天才开始处理（date 从第 0 天开始）
                # lags 要在前一天保存，后一天统一处理，处理完之后，新的 lags 覆盖旧的 lags
                self.update_data(lags)

                end1 = datetime.now()
                print(f"更新数据用时：{end1 - start1}秒！")
                # ===========================================================================================
                
                # ============================================== 计算score ====================================
                # 仅在 debug 时计算
                if self.config.debug:
                    self.cal_current_test_score()
                    print(f"current_test_score: {self.current_score}")
                # ===========================================================================================
                
                # ============================================== 再训练 =======================================
                if self.count_date % self.retrain_interval == 0:
                    start2 = datetime.now()
                    
                    self.retrain = True
                    self.retrain_dnn()

                    end2 = datetime.now()
                    print(f"再训练用时：{end2 - start2}秒！")
                # ============================================================================================
                
            self.is_zero_day = False # 他在第 0 天之后都是 False
            self.max_date_id += 1   # 用来替换新数据的日期相当于 push
            self.min_date_id += 1   # 用来过滤旧数据：增加一天的新数据，删除一天的旧数据, 相当于 pop
            
            # 前一天的数据更新完之后，保存 lags，用于下次更新
            self.lags = None # 为了释放旧的 lags 内存
            gc.collect() 
            self.lags = lags
            
            # ====================================== feature engineering  ====================================
            self.feature_engineering(lags, current_date)
            # ================================================================================================
        else:
            self.retrain = False
        
        # =========================================== gbdt offline预测 ===============================
        gbdt_test = test.cast(self.config.id_column_types)
        gbdt_test = gbdt_test.cast(self.config.feature_column_types)

        # 将新来的 test 数据和当天的 lags 数据合并
        my_X_test = (
            gbdt_test
            .join(self.lags_infer, on=["symbol_id", "date_id"], how="left")
        )
        # gbdt_X_test = my_X_test.select(self.config.features).to_numpy()
        # gbdt_y_test_pred = np.mean([model.predict(gbdt_X_test) for model in self.gbdt_models], axis=0)
        # ===========================================================================================
        
        # ========================================== nn online 预测 ==================================
        self.dnn_model.eval()
        dnn_X_test = torch.tensor( # 将数据填充缺失值后转换为 tensor
            my_X_test
            .select(self.config.features)
            .fill_null(0)
            .to_numpy(),
            dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            self.optimizer.zero_grad()
            dnn_y_test_pred = self.dnn_model(dnn_X_test)
        
        dnn_y_test_pred = dnn_y_test_pred.cpu().numpy()
        # ===========================================================================================

        # =============================================== 组合预测 ===================================
        y_test_pred = dnn_y_test_pred # * 0.5 + gbdt_y_test_pred * 0.5
        y_test_pred = np.clip(y_test_pred, a_min=-5, a_max=5)
        # print(y_test_pred)
        # ===========================================================================================
        
        # ========================================= 保存 test data ===================================
        test = test.with_columns(pl.Series(y_test_pred).alias("pred"))
        # 将 test 数据添加到 new_day_test 列表中，并在 next_date 的 time_0 进行合并，然后将被重置为空列表
        selected_test = test.select(pl.all().exclude("row_id", "is_scored"))
        self.new_day_test.append(selected_test)
        # ===========================================================================================
        
        # ============================================== 交卷 ========================================
        predictions = (
            test
            .select(
                "row_id",
                pl.lit(0.0).alias("responder_6")
            )
            .with_columns(pl.Series(y_test_pred).alias("responder_6"))
        )
        assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
        assert list(predictions.columns) == ['row_id', 'responder_6']
        assert len(predictions) == len(test)

        if self.retrain:
            print(f"总用时：{datetime.now() - start1}秒！")

        return predictions


config = Config


config.lag_n_days = 2
history_data = (
    pl.scan_parquet(r"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")
    .select(['date_id','time_id','symbol_id'] + config.responder_cols)
    .filter(
        (pl.col("date_id")>=(1698 - config.lag_n_days))&(pl.col("date_id")<1698)
    )
)
history_data = history_data.with_columns(
    date_id = (pl.col("date_id") - pl.lit(1698))
              .cast(pl.Int16)
)
history_data = history_data.collect()

history_data = history_data.cast(config.id_column_types)
history_data = history_data.cast(config.responder_column_types)


train_data = (
    pl.scan_parquet(r"/kaggle/input/js-generate-retrain-data/data.parquet")
    .filter(pl.col("date_id").is_between(1219, 1698))
    .fill_null(value=0)
    .with_columns([
        pl.lit(0, dtype=pl.Float32).alias("pred")
    ])
    .cast(config.row_column_tpyes)
    .collect()
)


same_seed(42)
dnn_model_params_path = r"/kaggle/input/50_0.039666_0.022241/pytorch/default/1/50_0.039666_0.022241.pkl"
dnn_model_params_dict = joblib.load(dnn_model_params_path)
print("epoch: ", dnn_model_params_dict["epoch"])
dnn_model = DNN(
    input_dim=dnn_model_params_dict["input_dim"],
    hidden_dims=dnn_model_params_dict["hidden_dims"],
    dropouts=dnn_model_params_dict["dropouts"]
)
dnn_model.load_state_dict(dnn_model_params_dict["model_state_dict"])

gbdt_models = []
# xgb_models = joblib.load(
#     r"/kaggle/input/xgb_0.015296_800_1698/scikitlearn/default/1/2025-01-01_19_13_XGB_0.015296_800_1698.pkl"
# )
# gbdt_models += xgb_models

# lgb_models = joblib.load(
#     r"/kaggle/input/lgb_0.017605_800_1698/scikitlearn/default/1/2025-01-01_19_14_LGB_0.017605_800_1698.pkl"
# )
# gbdt_models += lgb_models


# 生成 lags 特征列表
new_features = []
stat_names = ["mean", "std", "max", "min",]
last = True
if last:
    stat_names.append("last")
for col in list(config.responder_cols):
    for stat_name in stat_names:
        new_features.append(f"{col}_{stat_name}_shift_{1}d")
 
config.features += new_features

print(len(config.features))


same_seed(42)
config.debug = False

js_predict = JaneStreetOnlinePredictor(config, 
                                       dnn_model, 
                                       gbdt_models, 
                                       train_data.drop("partition_id"), 
                                       history_data, 
                                       1)


%%time
test = False

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(js_predict.predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    if test:
         inference_server.run_local_gateway(
            (
                "/kaggle/input/js24-create-simulate-data/test.parquet",
                "/kaggle/input/js24-create-simulate-data/lags.parquet",
            )
        )
    else:
        inference_server.run_local_gateway(
            (
                "/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet",
                "/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet",
            )
        )










