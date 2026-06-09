import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import math
import scipy.stats as stats

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('../input/playground-series-s5e3/train.csv')
test = pd.read_csv('../input/playground-series-s5e3/test.csv')
submit = pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')


train.head()


train.tail()


train.shape


train.isnull().sum()


train.info()


train['rainfall'].value_counts()


###


num_cols = train.select_dtypes(include=np.number).columns
num_cols


def plot_distribution(data, target_col=None):
    n_cols = 4
    n_rows = math.ceil(len(data.columns) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 4*n_rows))
    sns.set(font_scale=1.2, style="whitegrid")

    # Flatten axes array if multiple rows exist
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col_name in enumerate(data.columns):
        ax = axes[i]
        
        # 歪度（Skewness）と尖度（Kurtosis）を計算
        skewness = stats.skew(data[col_name].dropna())
        kurt = stats.kurtosis(data[col_name].dropna(), fisher=True)  # Fisher's definition（正規分布なら0）

        # **計算結果を出力**
        print(f"{col_name}: Skewness = {skewness:.2f}, Kurtosis = {kurt:.2f}")

        # ヒストグラムを描画
        ax.hist(data[col_name], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{col_name.title()} Distribution', fontsize=14)
        ax.set_xlabel(col_name.title(), fontsize=12)
        ax.set_ylabel("Volume", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # グラフ上にスコアを表示
        ax.text(0.95, 0.95, f"Skew: {skewness:.2f}\nKurt: {kurt:.2f}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # 余ったサブプロットを削除
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 使用例
plot_distribution(train[num_cols], target_col="Exited")


corr_matrix = train.corr()

plt.figure(figsize=(12, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


features = ['humidity', 'sunshine', 'pressure', 'dewpoint',]

X_train = train[features]
y_train = train['rainfall']


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.10,
    'num_leaves': 8,
    'n_estimators': 2000,
    'random_state': 123,
    'importance_type': 'gain',
    'min_child_samples': 20,
    'colsample_bytree': 0.7,
    'subsample_freq': 1,
    'subsample': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0
}


n_splits = 10

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

metrics = []
imp = pd.DataFrame()
models = []

for nfold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    x_tr, y_tr = X_train.loc[train_idx], y_train.loc[train_idx]
    x_va, y_va = X_train.loc[val_idx], y_train.loc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr,
              y_tr,
              eval_set=[(x_tr, y_tr), (x_va, y_va)],
              callbacks=[
                  lgb.early_stopping(stopping_rounds=100, verbose=True),
                  lgb.log_evaluation(0)
              ])

    y_tr_pred = model.predict_proba(x_tr)[:, 1]
    y_va_pred = model.predict_proba(x_va)[:, 1]

    metric_tr = roc_auc_score(y_tr, y_tr_pred)
    metric_va = roc_auc_score(y_va, y_va_pred)

    print("tr: {:.4f}, va: {:.4f}".format(metric_tr, metric_va))

    metrics.append([nfold, metric_tr, metric_va])
    models.append(model)

    _imp = pd.DataFrame({
        "col": X_train.columns,
        "imp": model.feature_importances_,
        "nfold": nfold,
    })

    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)


metrics = np.array(metrics)
metrics


print("tr: {:.5f}+-{:.3f}, va: {:.5f}+-{:.3f}".format(metrics[:,1].mean(), metrics[:,1].std(), metrics[:,2].mean(), metrics[:,2].std()))


imp_df = imp.groupby('col')['imp'].agg(['mean', 'std'])
imp_df.columns = ["imp", "imp_std"]

imp_df.sort_values(by="imp", ascending=False)


def models_predict_proba(x, models):
    preds = []

    print("N models:", len(models))

    for model in models:
        pred = model.predict_proba(x)[:, 1]
        preds.append(pred)

    preds = np.array(preds)
    preds = np.mean(preds, axis=0)

    return preds


y_pred_proba = models_predict_proba(X_train, models)


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_train, y_pred_proba)

plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr, color='r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()


test.head()


test.shape


X_test = test[features]
X_test.shape


submit.head()


y_test_pred = models_predict_proba(X_test, models)

df_submit = pd.DataFrame({
    'id': test['id'],
    'rainfall': y_test_pred,
})
df_submit.head()


df_submit.to_csv("submission.csv", index=False)

