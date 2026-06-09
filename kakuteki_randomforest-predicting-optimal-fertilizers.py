import warnings
warnings.simplefilter('ignore')

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

PATH = '../input/playground-series-s5e6/'
train = pl.read_csv(PATH + 'train.csv')
test = pl.read_csv(PATH + 'test.csv')

def base_encoder(input_df):
    out_df = input_df.select(['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']).with_columns(
        [pl.Series(input_df[col]).cast(pl.Categorical) for col in ['Soil Type', 'Crop Type']]
    )
    return out_df

x0 = base_encoder(train)
test_x0 = base_encoder(test)

le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])

N_CLASS = train['Fertilizer Name'].n_unique()

# convert pl.DataFrame to numpy
feature_names = x0.columns
cat_columns = [name for name, dtype in x0.schema.items() if dtype == pl.Categorical]

# convert categorical features to integer
all_x = pl.concat([x0, test_x0], how='vertical').with_columns(
    [pl.col(c).to_physical() for c in cat_columns]
)

x = all_x[:len(train)].to_numpy()
test_x = all_x[len(train):].to_numpy()

print("Data types:")
for i in range(x.shape[1]):
    print(f"{feature_names[i]}: {x[:,i].dtype}")

def single_apk(y, oof):
    sorted_oof = np.argsort(oof, axis=1)[:, ::-1][:, :3]
    
    score = 0
    for i in range(3):
        score += accuracy_score(y, sorted_oof[:, i]) / (i+1)
    
    return score

N_FOLDS = 5

oof = np.zeros((len(train), N_CLASS))
pred = np.zeros((len(test), N_CLASS))

logloss = []
map3 = []

fi_df = pl.DataFrame()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

# RandomForestClassifier with optimized parameters
model = RandomForestClassifier(
    n_estimators=500,           # 木の数
    max_depth=10,              # 最大深度
    min_samples_split=5,       # 分割に必要な最小サンプル数
    min_samples_leaf=2,        # 葉ノードの最小サンプル数
    max_features='sqrt',       # 特徴量の選択数
    bootstrap=True,            # ブートストラップサンプリング
    oob_score=True,           # Out-of-bag score計算
    random_state=0,
    n_jobs=-1,                # 並列処理
    verbose=0
)

for i, (train_idx, valid_idx) in tqdm(enumerate(skf.split(x, y)), total=N_FOLDS):
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]
    
    # RandomForestの学習（early_stoppingは不要）
    model.fit(x_train, y_train)
    
    # 予測
    oof[valid_idx, :] = model.predict_proba(x_valid)
    pred += model.predict_proba(test_x) / N_FOLDS
    
    # 特徴量重要度の記録
    fi_df = pl.concat([fi_df,
                       pl.DataFrame({'feature': feature_names, 
                                   'importance': model.feature_importances_, 
                                   'fold': i})])
    
    # スコア計算
    logloss.append(log_loss(y_valid, oof[valid_idx, :]))
    map3.append(single_apk(y_valid, oof[valid_idx, :]))
    
    print(f"Fold {i}: Logloss={logloss[-1]:.4f}, MAP@3={map3[-1]:.4f}, OOB Score={model.oob_score_:.4f}")

# 結果のまとめ
fold_df = pl.DataFrame({
    'fold': range(N_FOLDS),
    'Logloss': logloss,
    'MAP@3': map3
})

print("\nFold Results:")
print(fold_df)

total_logloss = log_loss(y, oof)
total_map3 = single_apk(y, oof)
print(f"\nTotal: Logloss={total_logloss:.4f}, MAP@3={total_map3:.4f}")

# 特徴量重要度の可視化
fi_summary = fi_df.group_by('feature').agg([
    pl.col('importance').mean().alias('mean_importance'),
    pl.col('importance').std().alias('std_importance')
]).sort('mean_importance', descending=True)

print("\nFeature Importance:")
print(fi_summary)

# 予測結果の生成
sorted_pred = np.argsort(pred, axis=1)[:, ::-1]

submission = pl.DataFrame({
    'id': test['id'],
    'pred1': le.inverse_transform(sorted_pred[:, 0]),
    'pred2': le.inverse_transform(sorted_pred[:, 1]),
    'pred3': le.inverse_transform(sorted_pred[:, 2])
}).with_columns(
    Fertilizer=pl.col('pred1') + ' ' + pl.col('pred2') + ' ' + pl.col('pred3')
).select(['id', 'Fertilizer']).rename({'Fertilizer': 'Fertilizer Name'})

submission.write_csv('submission.csv')
print("\nSubmission file created: submission.csv")

