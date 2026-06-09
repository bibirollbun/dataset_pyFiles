import pandas as pd
import lightgbm as lgbm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


# 1. Приведение категориальных признаков
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


# 2. фиче инжиниринг
train_df['LDL_to_HDL'] = (train_df['ldl_cholesterol'] / train_df['hdl_cholesterol'])
train_df['Non_HDL'] = (train_df['cholesterol_total'] - train_df['hdl_cholesterol'])

test_df['LDL_to_HDL'] = (test_df['ldl_cholesterol'] / test_df['hdl_cholesterol'])
test_df['Non_HDL'] = (test_df['cholesterol_total'] - test_df['hdl_cholesterol'])

train_df["log_activity"] = np.log1p(train_df['physical_activity_minutes_per_week'])
test_df["log_activity"] = np.log1p(test_df['physical_activity_minutes_per_week'])


# 3. Подготовка X и y
target_col = 'diagnosed_diabetes'
drop_cols = [target_col, 'id'] # Удаляем целевую переменную и ID

X = train_df.drop(columns=drop_cols)
y = train_df[target_col]

X_test = test_df.drop(columns=['id'])


# Настройки валидации, стратифает к фолдс, как и голосовали
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=21)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []


# Цикл обучения с валидацией по фолдам
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Модель ЛГБ(Т)МКлассифаер
    model = lgbm.LGBMClassifier(
        n_estimators=5000, 
        learning_rate=0.02,
        random_state=21,
        n_jobs=-1,
        verbose=-1
    )
    
    callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)]
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='auc', 
        callbacks=callbacks
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_val, val_preds)
    scores.append(fold_auc)

    print(f"Fold {fold+1}: AUC = {fold_auc:.5f}")

print("\n")
print(f"Общий AUC на кросс-валидации: {np.mean(scores):.5f}")


submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)

