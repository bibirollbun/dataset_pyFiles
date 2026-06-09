import os
import glob
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm



DATA_PATH = "/kaggle/input/pump-fun-graduation-february-2025"
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df = pd.read_csv(f"{DATA_PATH}/test_unlabeled.csv")

#Объединяю train и test, чтобы изменять сразу оба массива 
train_df["is_test"] = False
test_df["is_test"] = True
full_df = pd.concat([train_df, test_df], ignore_index=True)


chunk_files = glob.glob(f"{DATA_PATH}/chunk*.csv")
chunks = pd.concat([pd.read_csv(f) for f in tqdm(chunk_files, desc="Загрузка chunk-файлов")])



agg_features = chunks.groupby("base_coin").agg({
    "quote_coin_amount": ["sum", "mean", "max", "min", "std"],
    "base_coin_amount": ["sum", "mean", "max", "min", "std"],
    "signing_wallet": ["nunique", "count"],
    "direction": lambda x: (x == 'buy').sum() / (len(x) + 1e-6),
    "virtual_token_balance_after": ["mean", "std"],
    "virtual_sol_balance_after": ["mean", "std"]
})

agg_features.columns = list(tqdm(["_".join(col) if isinstance(col, tuple) else col for col in agg_features.columns], 
                                 total=len(agg_features.columns), 
                                 desc="Формирование имён колонок"))
agg_features.reset_index(inplace=True)
agg_features.rename(columns={"base_coin": "mint"}, inplace=True)
full_df = full_df.merge(agg_features, on="mint", how="left")


# Удаление ненужных таблиц
drop_cols = ["mint", "has_graduated", "slot_min", "slot_graduated", "is_test"]
features = list(tqdm((col for col in full_df.columns if col not in drop_cols),
                     total=len(full_df.columns),
                     desc="Отбор признаков"))
cat_features = [col for col in features if full_df[col].dtype == "object"]



train_data = full_df[~full_df.is_test]
test_data = full_df[full_df.is_test]

X = train_data[features].copy()
y = train_data["has_graduated"].astype(int)
X_test = test_data[features].copy()

# Заполнение пропусков ( -1 потому что catboost нормально определяет -1 как NULL)
X.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=228
)



model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=7,
    task_type="GPU",
    eval_metric="Logloss",
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features
)



feature_importances = model.get_feature_importance(prettified=True)
top_feats = feature_importances.sort_values("Importances", ascending=False).head(30)
top_feats.plot(kind="bar", x="Feature Id", y="Importances", figsize=(14, 6), legend=False)
plt.title("Важность признаков (топ 30)")
plt.tight_layout()
plt.show()




# Предсказание
test_df["has_graduated"] = model.predict_proba(X_test)[:, 1]

# Сохранение submission
submission = test_df[["mint", "has_graduated"]]
submission.to_csv("submission.csv", index=False)
print("Сохранилось.")

