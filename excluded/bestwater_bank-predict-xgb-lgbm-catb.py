"Good Luck!"


import pandas as pd
import numpy as np

# 读取 OOF 预测
oof_preds = pd.read_csv("/kaggle/input/bank-predict-xgb-lgbm-catb/xgb_oof.csv")
oof_preds



# 提取 P(1)
p1 = oof_preds["xgb_oof"]

# 计算 P(0)
p0 = 1 - p1

# 拼接成 DataFrame（和 predict_proba 格式一致）
oof_predictions = pd.DataFrame({
    0: p0,
    1: p1
})
 
# 保存为 .npy
np.save("xgb_oof.npy", oof_predictions.to_numpy())

# 查看前几行
oof_predictions


!cp -r /kaggle/input/bank-predict-xgb-lgbm-catb/xgb.csv submission.csv

