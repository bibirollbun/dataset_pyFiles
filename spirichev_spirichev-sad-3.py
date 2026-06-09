import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import os
import seaborn as sns
import warnings
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_sample_submission =  pd.read_csv('/kaggle/input/playground-series-s3e13/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s3e13/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e13/test.csv')


df_train.isna().sum()


df_sample_submission.head()


df_train.head()


df_train.describe()


warnings.filterwarnings("ignore", category=FutureWarning)

# Числовые столбцы
numeric_cols = df_train.select_dtypes(include='number').columns
n_cols = 4  # количество графиков в строке
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

# Размер всей фигуры
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.kdeplot(df_train[col].dropna(), color='blue', label='train', fill=False, ax=ax)
    sns.kdeplot(df_test[col].dropna(), color='red', label='test', fill=False, ax=ax)
    ax.set_title(col)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

# Отключаем пустые подграфики
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Cчитаем количество вхождений каждого класса
class_counts = df_train['prognosis'].value_counts()

# Строим столбчатую диаграмму по количеству классов:
class_counts.plot(kind='bar')

# Задаём заголовок графика и подписи осей:
plt.title('Частота каждого класса')
plt.xlabel('Класс')
plt.ylabel('Частота')

plt.show()


enc = LabelEncoder()
df_train['prognosis'] = enc.fit_transform(df_train['prognosis'])


def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def selecting_top_3(arr):
    n = arr.shape[0]
    out = np.zeros((n, 3))
    for i in range(n):
        out[i, ] = arr[i,].argsort()[::-1][:3]
    return out.astype('int32')


class Pipeline:

    def __init__(self, model='XGB'):
        self.model_type = model
        if model == 'LGB':
            self.model = lgb.LGBMClassifier(
                num_leaves=16,
                learning_rate=0.01,
                n_estimators=1000,
                objective="multiclass",
                verbose=-1  # подавляем вывод
            )
        elif model == 'CatBoost':
            self.model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.01,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                depth=2,
                verbose=0  # подавляем вывод
            )
        else:  # XGBoost
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                colsample_bytree=0.6,
                gamma=0.8,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=5,
                n_estimators=1000,
                subsample=0.7,
                verbosity=0,   # полностью подавляем лог XGBoost
                use_label_encoder=False,
                eval_metric='mlogloss'  # задаем напрямую, чтобы не было warning
            )

    def fit(self, X, y, X_val, y_val):
        if self.model_type == 'LGB':
            self.model.fit(
                X, y.ravel(),
                eval_set=[(X_val, y_val.ravel())],
                eval_metric='multi_logloss',
                callbacks=[lgb.log_evaluation(0)]  # отключаем лог LightGBM
            )
        elif self.model_type == 'XGB':
            self.model.fit(
                X, y.ravel(),
                eval_set=[(X_val, y_val.ravel())],
                verbose=False  # отключаем лог XGBoost
            )
        else:  # CatBoost
            self.model.fit(
                X, y.ravel(),
                eval_set=(X_val, y_val.ravel()),
                verbose=False  # отключаем лог CatBoost
            )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def grid_search(self, X, y, X_eval, y_eval):
        pass



def train(model_type):
    X = df_train.drop(['prognosis'], axis=1)
    y = df_train['prognosis']
    SKFs = KFold(n_splits=5, shuffle=True, random_state=1)
    losses = []
    pipelines = []
    for fold, (idx_tr, idx_vl) in enumerate(SKFs.split(X, y)):
        X_train, y_train = X.iloc[idx_tr], y.iloc[idx_tr]
        X_dev, y_dev = X.iloc[idx_vl], y.iloc[idx_vl]

        pipelineMy = Pipeline(model=model_type)
        pipelineMy.fit(X_train, y_train, X_dev, y_dev)

        pipelines.append(pipelineMy)
        prediction = selecting_top_3(pipelineMy.predict_proba(X_dev))
        loss = mapk(y_dev.values.reshape(-1, 1), prediction, k=3)
        losses.append(loss)
        print(f'Fold {fold} loss: {loss:.4f}')

    print(f'Mean loss: {np.mean(losses):.4f}')
    return losses, pipelines



lossesLGB, pipelinesLGB = train('LGB')


lossesCB, pipelinesCB = train('CatBoost')


lossesXGB, pipelinesXGB = train('XGB')


n_lgb = len(pipelinesLGB)
n_xgb = len(pipelinesXGB)
n_cb = len(pipelinesCB)


w_lgb = 1 - np.mean(lossesLGB)
w_xgb = 1 - np.mean(lossesXGB)
w_cb  = 1 - np.mean(lossesCB)


total_weight = w_lgb + w_xgb + w_cb
w_lgb /= total_weight
w_xgb /= total_weight
w_cb  /= total_weight


preds = np.zeros((df_test.shape[0], pipelinesLGB[0].predict_proba(df_test).shape[1]))


for model in pipelinesLGB:
    preds += w_lgb * model.predict_proba(df_test)


for model in pipelinesXGB:
    preds += w_xgb * model.predict_proba(df_test)


for model in pipelinesCB:
    preds += w_cb * model.predict_proba(df_test)


preds_top3 = selecting_top_3(preds)
preds_df = pd.DataFrame(preds_top3)


for i in range(3):
    preds_df[i] = enc.inverse_transform(preds_df[i])


submission = pd.DataFrame()
submission['prognosis'] = preds_df[0] + " " + preds_df[1] + " " + preds_df[2]
submission.index += 707  # сдвигаем ID
submission.to_csv("submission.csv", index=True, header=True, index_label="id")

