import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df.head()


df.drop('id', inplace=True, axis=1)
df.shape


df.info()


df.describe()


df.columns


numeric = df.describe().columns.tolist()
categorical = ['Soil Type', 'Crop Type', 'Fertilizer Name']


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

axes = axes.flatten()
for ax, col in zip(axes, numeric):
    sb.distplot(df[col], ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


for col in categorical:
  print(df.groupby(col)['Soil Type'].count())
  print()


for col in categorical[:2]:
  temp = pd.get_dummies(df[col]).astype('int')
  df = pd.concat([df, temp], axis=1)
  df.drop(col, axis=1, inplace=True)
df.head()


df["N_K_ratio"]    = df["Nitrogen"] / (df["Potassium"]  + 1e-6)
df["N_P_ratio"]    = df["Nitrogen"] / (df["Phosphorous"]+ 1e-6)
df["total_npk"]    = df["Nitrogen"] + df["Potassium"] + df["Phosphorous"]
df["temp_humid"]   = df["Temparature"] * df["Humidity"]
df.head()


# Environmental Stress Indicators
df["moisture_deficit"] = 100 - df["Moisture"]
df["vpd_proxy"] = (100 - df["Humidity"]) * (df["Temparature"]/100)

 # Polynomial Features on Continous Variables
for col in ["Temparature", "Humidity", "Moisture"]:
    df[f"{col}_sq"]   = df[col] ** 2
    df[f"{col}_sqrt"] = np.sqrt(df[col])
df.head()


features = df.drop('Fertilizer Name', axis=1)
target = df['Fertilizer Name']


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size = 0.2,
                                                  random_state=10)
y_train = pd.get_dummies(Y_train).astype('int')
y_val = pd.get_dummies(Y_val).astype('int')
y_train.head()


target_labels = np.array(y_train.columns)
target_labels


from sklearn.preprocessing import StandardScaler
# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


!pip install -q optuna


target_labels


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(Y_train)
y_val_encoded = encoder.transform(Y_val)
labels = encoder.classes_
labels


import numpy as np
import optuna
from xgboost import XGBClassifier

def apk(actual, predicted, k=3):
    """Average precision at k for one sample."""
    if actual in predicted:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0

def mapk(y_true, y_pred_topk, k=3):
    """Mean average precision at k over all samples."""
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred_topk)])


def objective(trial):
    param = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'gamma': trial.suggest_uniform('gamma', 0.0, 5.0),
    }

    # --- 2. train the classifier
    model = XGBClassifier(**param,
                          use_label_encoder=False,
                          eval_metric='mlogloss')
    model.fit(
        X_train, y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False
    )

    # --- 3. get top-3 predictions on validation set
    probs = model.predict_proba(X_val)  # shape=(n_val, num_class)
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_preds = [[labels[i] for i in row] for row in top3_idx]

    # --- 4. compute MAP@3
    score = mapk(Y_val, top3_preds, k=3)
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best MAP@3:", study.best_value)
print("Best params:", study.best_params)


print("Best MAP@3:", study.best_value)
print("Best params:", study.best_params)


model = XGBClassifier(tree_method='gpu_hist',
                      predictor = 'gpu_predictor',
                      **study.best_params)
model.fit(X_train, y_train_encoded)
y_train_pred = model.predict_proba(X_train)
y_val_pred = model.predict_proba(X_val)


def get_top3_preds(probs):
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_preds = [[labels[i] for i in row] for row in top3_idx]
    return top3_preds


train_pred_strs = get_top3_preds(y_train_pred)
val_pred_strs = get_top3_preds(y_val_pred)

train_map = mapk(Y_train, train_pred_strs)
val_map = mapk(Y_val, val_pred_strs)

print('MAP@3 Value for Training Data is :', train_map)
print('MAP@3 Value for Validation Data is :', val_map)


test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test.head()


test.drop('id', axis=1, inplace=True)
test.shape


for col in categorical[:2]:
  temp = pd.get_dummies(test[col]).astype('int')
  test = pd.concat([test, temp], axis=1)
  test.drop(col, axis=1, inplace=True)
test.head()


test["N_K_ratio"]    = test["Nitrogen"] / (test["Potassium"]  + 1e-6)
test["N_P_ratio"]    = test["Nitrogen"] / (test["Phosphorous"]+ 1e-6)
test["total_npk"]    = test["Nitrogen"] + test["Potassium"] + test["Phosphorous"]
test["temp_humid"]   = test["Temparature"] * test["Humidity"]
test.head()


# Environmental Stress Indicators
test["moisture_deficit"] = 100 - test["Moisture"]
test["vpd_proxy"] = (100 - test["Humidity"]) * (test["Temparature"]/100)

 # Polynomial Features on Continous Variables
for col in ["Temparature", "Humidity", "Moisture"]:
    test[f"{col}_sq"]   = test[col] ** 2
    test[f"{col}_sqrt"] = np.sqrt(test[col])
test.head()


test_preds = model.predict_proba(scaler.transform(test))
test_str_preds = get_top3_preds(test_preds)
test_str_preds[:3]


ss = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
ss['Fertilizer Name'] = [' '.join(row) for row in test_str_preds]
ss.head()


ss.to_csv('Submission.csv', index=False)




