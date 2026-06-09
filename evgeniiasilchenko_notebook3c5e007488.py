import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print(train_df.head())
print(test_df.head())


target = "loan_paid_back"

X = train_df.drop(columns=[target])
y = train_df[target]

X_test = test_df.copy()


full = pd.concat([X, X_test], axis=0)

cat_cols = full.select_dtypes(include=["object"]).columns
print("Категориальные признаки:", list(cat_cols))

full_encoded = pd.get_dummies(full, columns=cat_cols)

# Возвращаем части обратно
X_encoded = full_encoded.iloc[:len(X)]
X_test_encoded = full_encoded.iloc[len(X):]



X_train, X_valid, y_train, y_valid = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)



lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[valid_data],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=200),
    ]
)



pred_valid = model.predict(X_valid)
auc = roc_auc_score(y_valid, pred_valid)
print("AUC =", auc)



full_train_data = lgb.Dataset(X_encoded, label=y)

final_model = lgb.train(
    params=lgb_params,
    train_set=full_train_data,
    num_boost_round=model.best_iteration
)



pred_test = final_model.predict(X_test_encoded)


submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_paid_back": pred_test
})

submission.to_csv("submission.csv", index=False)
print("Файл submission.csv сохранён!")



# 6. Создаём валид_df с предсказаниями и target
valid_df = X_valid.copy()
valid_df["loan_paid_back"] = y_valid.values
valid_df["pred_prob"] = pred_valid
valid_df["pred_class"] = (pred_valid > 0.5).astype(int)

# 7. Определяем тип ошибки
def error_type(row):
    if row["loan_paid_back"] == 1 and row["pred_class"] == 1:
        return "TP"
    elif row["loan_paid_back"] == 0 and row["pred_class"] == 0:
        return "TN"
    elif row["loan_paid_back"] == 1 and row["pred_class"] == 0:
        return "FN"
    elif row["loan_paid_back"] == 0 and row["pred_class"] == 1:
        return "FP"
    else:
        return "UNKNOWN"

valid_df["error_type"] = valid_df.apply(error_type, axis=1)

# 8. Новый столбец с правильностью предсказания
valid_df["correct"] = valid_df["error_type"].apply(lambda x: "Correct" if x in ["TP", "TN"] else "Incorrect")

# 9. Просмотр случайных 50 строк
valid_df.sample(50, random_state=42)

