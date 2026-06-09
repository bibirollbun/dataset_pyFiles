# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from itertools import combinations
from tqdm import tqdm
from colorama import Fore, Style

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


train.isnull().sum()


# def encode_data(df):
#     df_copy = df.copy()

#     cat_cols = df_copy.select_dtypes(include='object').columns
#     for col in cat_cols:
#         encoder = LabelEncoder()
#         df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))

#     return df_copy

# train = encode_data(train)
# test = encode_data(test)


cat_cols = train.select_dtypes(include=['object','category']).columns
num_cols = train.select_dtypes(include=['int64','float64']).columns


for col in cat_cols:
    le = LabelEncoder()
    train[col ]= le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])



for col in num_cols:
    if col == 'y' or col == 'id':
        continue
    else:
        scaler = StandardScaler()
        train_vals = train[[col]]
        test_vals = test[[col]]
        train_scaled = scaler.fit_transform(train_vals).ravel()
        test_scaled = scaler.transform(test_vals).ravel()
        train[col] = train_scaled
        test[col] = test_scaled


sig_features = ['previous', 'pdays', 'duration', 'balance', 'education', 'marital', 'job', 'age']


MAX_BATCH = int(1e3)

class FeatureComb:
    def __init__(self, dataFrame, pairSize, columnsToEncode=None, maximumBatch=MAX_BATCH):
        self.dataFrame        = dataFrame
        self.columnsToEncode  = list(columnsToEncode) if columnsToEncode is not None else dataFrame.columns.tolist()
        self.pairSize         = pairSize if isinstance(pairSize, list) else [pairSize]
        self.maximumBatch     = maximumBatch

    def process(self):
        # Sadece seçilen sütunları string olarak al
        data = self.dataFrame[self.columnsToEncode].astype(str)
        newColumns = 0

        for pair in self.pairSize:
            total = np.math.comb(len(self.columnsToEncode), pair)
            print(f"\n {Fore.CYAN + Style.BRIGHT}⤷ İşlenecek kombinasyon sayısı: {total}{Style.RESET_ALL} "
                  f"({pair}-li, toplam {len(self.columnsToEncode)} sütundan)")

            combo     = combinations(self.columnsToEncode, pair)
            colsList  = []
            namesList = []

            with tqdm(total=total) as bar:
                while True:
                    colsList.clear()
                    namesList.clear()

                    # Her döngüde en fazla maximumBatch kadar kombinasyon al
                    for _ in range(self.maximumBatch):
                        try:
                            cols = next(combo)
                            colsList.append(list(cols))
                            namesList.append('+'.join(cols))
                        except StopIteration:
                            break

                    if not colsList:
                        break

                    # Alınan her kombinasyon için stringleri birleştir, label-encode et, yeni sütun olarak ekle
                    for cols, name in zip(colsList, namesList):
                        result = data[cols[0]].copy()
                        for col in cols[1:]:
                            result += data[col]  # string birleştirme

                        encoded = LabelEncoder().fit_transform(result) + 1
                        self.dataFrame[name] = encoded
                        bar.update(1)

                    newColumns += len(colsList)
                    if len(colsList) == self.maximumBatch:
                        print(f"{Fore.MAGENTA}⤷ İlerleme:{Style.RESET_ALL} {newColumns} / {total}")

            print(f"{Fore.GREEN + Style.BRIGHT}➤ {pair}-li kombinasyonlar tamamlandı. "
                  f"Şu anki sütun sayısı: {len(self.dataFrame.columns)}{Style.RESET_ALL}")

        return self.dataFrame

    def transform(self):
        return self.process()

# Örnek kullanım:
comb    = FeatureComb(dataFrame=train, pairSize=[2, 3], columnsToEncode=sig_features)
comb_ts = FeatureComb(dataFrame=test,  pairSize=[2, 3], columnsToEncode=sig_features)

train = comb.transform()
test  = comb_ts.transform()



train.info()


train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


X = train.drop(columns=['y'])
y = train['y']

X_train, X_valid, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state =41, stratify=y)


def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 10),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "max_delta_step":    trial.suggest_int("max_delta_step", 0, 10),


        "grow_policy":       trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "random_state":      42,
        "use_label_encoder": False,
        "eval_metric":       "logloss",
    }

    model = XGBClassifier(**params)

    scores = cross_val_score(
        model, X, y,
        cv=2,
        scoring="accuracy",
        n_jobs=2,
        verbose=0
    )
    return scores.mean()

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=25)

best_model = XGBClassifier(**study.best_params)
best_model.fit(X, y)

joblib.dump(best_model, "best_xgb_model.pkl")

print("Best Accuracy:", study.best_value)
print("Best Params:", study.best_params)



# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)


test_ids = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


feature_columns = [col for col in train.columns if col not in ["y",'id']]
best_params = study.best_params.copy()

model = XGBClassifier(**best_params)
model.fit(train[feature_columns], train["y"])


y_pred_proba = model.predict_proba(test[feature_columns])[:, 1]

submission = pd.DataFrame({
    "id": test_ids["id"],
    "y": y_pred_proba
})

submission.to_csv("submission.csv", index=False)

print("submission.csv successfull.")



best_params


submission




