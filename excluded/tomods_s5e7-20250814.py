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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings 
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
print("train/test:", train_df.shape,"/", test_df.shape)


train_df.head()


train_df.info()


y_train = train_df["Personality"].map({"Introvert":0,"Extrovert":1})
y_train.head()


plt.figure(figsize=(18, 9))
sns.heatmap(train_df.isnull(), cbar=False)


plt.figure(figsize=(18, 9))
sns.heatmap(test_df.isnull(), cbar=False)


train_df.head()


merged_df = pd.concat([train_df, test_df],axis=0).drop(["id","Personality"],axis=1)
print("merged_df.shape",merged_df.shape)


merged_df.head()


cat_cols = []
num_cols = []

for col  in merged_df.columns:
    if merged_df[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)

print("Categorical Columns", cat_cols)
print("Numerical Columns", num_cols)


for col in cat_cols:
    print(train_df[col].value_counts())


for col in cat_cols:
    print(test_df[col].value_counts())


from sklearn.impute import SimpleImputer

imp_num = SimpleImputer(missing_values=np.nan, strategy='median')
#imp = SimpleImputer(missing_values=np.nan, strategy='median')

merged_df_num = merged_df[num_cols]
merged_df_num = pd.DataFrame(imp_num.fit_transform(merged_df_num))
merged_df_num.columns = num_cols
merged_df_num


from sklearn.impute import SimpleImputer

imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

merged_df_cat = merged_df[cat_cols]
merged_df_cat = pd.DataFrame(imp_cat.fit_transform(merged_df_cat))
merged_df_cat.columns = cat_cols
merged_df_cat.head()


merged_df_cat.head()


le = LabelEncoder()
for col in merged_df_cat.columns:
    merged_df_cat[col] = le.fit_transform(merged_df_cat[col])

merged_df_cat.head()


merged_df_num.head()


merged_df_preprocessed = pd.concat([merged_df_num,merged_df_cat],axis=1)
print("merged_df_preprocessed.shape",merged_df_preprocessed.shape)


X_train = merged_df_preprocessed.iloc[:train_df.shape[0],:]
X_test = merged_df_preprocessed.iloc[train_df.shape[0]:,:]


X_train_sample = X_train.sample(10000)
y_train_sample = y_train.iloc[X_train_sample.index]


X_tra, X_val, y_tra, y_val = train_test_split(X_train_sample, y_train_sample, test_size=0.3, random_state=42)


y_train_sample


%%time
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_tra, y_tra)
y_pred = model.predict(X_val)
print("ACC:", accuracy_score(y_pred, y_val))


# ROC曲線をプロットするためのライブラリをインポート
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. アンサンブルモデルを使用してテストデータの予測確率を取得
y_pred_proba = model.predict_proba(X_val)[:, 1]

# 2. ROC曲線の値を計算
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 3. ROC曲線をプロット
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import confusion_matrix
import pandas as pd
pd.options.display.precision = 2 # 表示桁数の設定

conf_mat = confusion_matrix(y_val, y_pred, normalize='true')
display(pd.DataFrame(conf_mat))

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, cmap = 'Blues', annot=True)
plt.show()


y_test_predict = model.predict(X_test)


submission = pd.DataFrame({"id":test_df["id"],"Personality":y_test_predict})
submission["Personality"] = submission["Personality"].map({0:"Introvert",1:"Extrovert"})
submission.head()


submission.to_csv("submission_only1000samples.csv",index=False)


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import make_scorer, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

X = X_train
y = y_train

# モデル
model = lgb.LGBMClassifier(random_state=42)

# スコア関数（accuracy）
scorer = make_scorer(accuracy_score)

# StratifiedKFold 5分割
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# learning_curveの計算
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=cv,
    scoring=scorer,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# 各サイズごとの平均・標準偏差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Train Accuracy の平均と分散（最大データサイズ時）
print("=== Train Accuracy (最大データサイズ時) ===")
print(f"平均: {train_mean[-1]:.4f}")
print(f"分散: {np.var(train_scores[-1]):.6f}")

# プロット
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Train Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

plt.plot(train_sizes, test_mean, 'o-', color="green", label="Test Accuracy")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")

plt.title("Learning Curve (LightGBM Classifier)")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc="best")
plt.show()



%%time
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score

# 訓練データからターゲットと特徴量を分ける
X_train = X_train
y_train = y_train
X_test = X_test

# Optunaの目的関数
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
        'verbose': -1  # ← ログを抑制する
    }

    # LightGBMモデルをcross_val_scoreで評価
    model = lgb.LGBMClassifier(**param)

    # cross_val_scoreで交差検証を行い、各foldの精度を算出
    accuracies = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # accuracyの平均と分散を計算
    accuracy_mean = np.mean(accuracies)
    accuracy_var = np.var(accuracies)
    print(f"Accuracy mean: {accuracy_mean}, Accuracy variance: {accuracy_var}")

    return accuracy_mean

# Optunaによるハイパーパラメータ最適化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 最適なハイパーパラメータを取得
best_params = study.best_params
print("Best parameters:", best_params)

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)





y_pred_opputuna = final_model.predict(X_test)

submission2 = pd.DataFrame({"id":test_df["id"],"Personality":y_pred_opputuna})
submission2["Personality"] = submission2["Personality"].map({0:"Introvert",1:"Extrovert"})
submission2.head()


submission2.to_csv("submission.csv",index=False)

