import pandas as pd
from scipy.stats import mode

# 提出ファイルをリストでまとめて読み込み
dfs = [
    pd.read_csv("/kaggle/input/super-multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/calorie-expenditure-cat-xgb-lgbm-ydf-hgb/submission.csv"),
    pd.read_csv("/kaggle/input/ps-s5e5-triple-ensemble-with-catboost-xgboost/submission_median.csv"),
    pd.read_csv("/kaggle/input/s5e5-tabnet-with-pretraining/submission.csv"),
]

# idは全て共通前提なので1つ目のデータフレームから取得
submission = dfs[0][['id']].copy()

# すべてのCalories列をまとめて2D配列にする
calories_array = pd.concat([df['Calories'] for df in dfs], axis=1).values

# scipy.stats.modeで各行の最頻値を取得
# mode_result.modeは最頻値の配列。reshapeで1次元に。
mode_values = mode(calories_array, axis=1).mode.reshape(-1)

# 最頻値を結果のCalories列にセット
submission['Calories'] = mode_values

# CSV出力
submission.to_csv('submission_mode.csv', index=False)


import pandas as pd
import numpy as np
from scipy.stats import mode

# 提出ファイルの読み込み
dfs = [
    pd.read_csv("/kaggle/input/super-multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/calorie-expenditure-cat-xgb-lgbm-ydf-hgb/submission.csv"),
    pd.read_csv("/kaggle/input/ps-s5e5-triple-ensemble-with-catboost-xgboost/submission_median.csv"),
    pd.read_csv("/kaggle/input/s5e5-tabnet-with-pretraining/submission.csv"),
]

submission = dfs[0][['id']].copy()

calories_array = pd.concat([df['Calories'] for df in dfs], axis=1).values
n_models = calories_array.shape[1]
n_samples = calories_array.shape[0]

np.random.seed(42)  # 再現性のため固定シード

bagging_k = 5  # 各行でランダムに選ぶモデル数（バギング数）

bagged_mode_values = []
for i in range(n_samples):
    # i行目のモデル予測からランダムにbagging_k個抽出
    sampled_preds = np.random.choice(calories_array[i], size=bagging_k, replace=True)
    # 最頻値を計算
    mode_val = mode(sampled_preds).mode
    bagged_mode_values.append(mode_val)

submission['Calories'] = bagged_mode_values
submission.to_csv('submission_bagged.csv', index=False)


import pandas as pd
import numpy as np
from collections import Counter

# すべての提出ファイルを読み込み
dfs = [
    pd.read_csv("/kaggle/input/super-multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/multiple-ensemble-strategies/submission_adaptive.csv"),
    pd.read_csv("/kaggle/input/calorie-expenditure-cat-xgb-lgbm-ydf-hgb/submission.csv"),
    pd.read_csv("/kaggle/input/ps-s5e5-triple-ensemble-with-catboost-xgboost/submission_median.csv"),
    pd.read_csv("/kaggle/input/s5e5-tabnet-with-pretraining/submission.csv"),
    pd.read_csv("/kaggle/input/what-is-automl/submission.csv"),
    pd.read_csv("/kaggle/input/best-fe-ensemble-of-3-models/submission.csv"),
    pd.read_csv("/kaggle/input/calories-ensemble/submission.csv"),
    pd.read_csv("/kaggle/input/s5e5-mean-ensemble-of-multiplemodels/trippleEnsemble_submission.csv"),
    pd.read_csv("/kaggle/input/s5e5-mean-ensemble-of-multiplemodels/XG_Catboost_FE_submission.csv"),
    pd.read_csv("/kaggle/input/s5e5-mean-ensemble-of-multiplemodels/Ensemble_submission.csv"),
]

submission = dfs[0][['id']].copy()

# 各モデルの予測（N, M）配列
preds = np.stack([df['Calories'].values for df in dfs], axis=1)  # shape: (N, M)

# モデル数
model_indices = list(range(preds.shape[1]))

# 1. 初期モデル：1つ目のモデル
current_ensemble = [model_indices[0]]
current_preds = preds[:, current_ensemble]
best_score = np.mean([
    Counter(row).most_common(1)[0][1] for row in current_preds
])  # 一致数（最頻値出現回数の平均）をスコアと仮定

# 2. 残りのモデルを1つずつ試す
remaining_models = model_indices[1:]
for m in remaining_models:
    temp_ensemble = current_ensemble + [m]
    temp_preds = preds[:, temp_ensemble]
    
    # 各行で最頻値を取り、その一致率をスコアとする
    temp_score = np.mean([
        Counter(row).most_common(1)[0][1] for row in temp_preds
    ])

    if temp_score > best_score:
        current_ensemble = temp_ensemble
        best_score = temp_score
        print(f"✔ モデル {m} を追加 → スコア: {best_score:.4f}")
    else:
        print(f"✘ モデル {m} は追加しない（スコア: {temp_score:.4f}）")

# 3. 最終的なアンサンブルで最頻値予測
final_preds = preds[:, current_ensemble]
submission['Calories'] = [
    Counter(row).most_common(1)[0][0] for row in final_preds
]
submission.to_csv("submission_hill-climbing.csv", index=False)

