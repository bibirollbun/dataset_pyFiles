import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)


le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

train_df['Soil Type'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop Type'] = le_crop.fit_transform(train_df['Crop Type'])
train_df['Fertilizer Name'] = le_fert.fit_transform(train_df['Fertilizer Name'])

test_df['Soil Type'] = le_soil.transform(test_df['Soil Type'])
test_df['Crop Type'] = le_crop.transform(test_df['Crop Type'])


train_df['Soil_Crop'] = train_df['Soil Type'].astype(str) + '_' + train_df['Crop Type'].astype(str)
test_df['Soil_Crop'] = test_df['Soil Type'].astype(str) + '_' + test_df['Crop Type'].astype(str)

le_soil_crop = LabelEncoder()
train_df['Soil_Crop'] = le_soil_crop.fit_transform(train_df['Soil_Crop'])
test_df['Soil_Crop'] = le_soil_crop.transform(test_df['Soil_Crop'])


train_df['N_to_P'] = train_df['Nitrogen'] / (train_df['Phosphorous'] + 1)
test_df['N_to_P'] = test_df['Nitrogen'] / (test_df['Phosphorous'] + 1)

train_df['Temp_bin'] = pd.cut(train_df['Temperature'], bins=[0, 15, 25, 35, 50], labels=False)
test_df['Temp_bin'] = pd.cut(test_df['Temperature'], bins=[0, 15, 25, 35, 50], labels=False)


features = [
    'Temperature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type', 'Soil_Crop',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'N_to_P', 'Temp_bin'
]

X = train_df[features]
y = train_df['Fertilizer Name']
X_test = test_df[features]


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((X_test.shape[0], len(le_fert.classes_)))
map3_scores = []


def apk(actual, predicted, k=3):
    if actual in predicted[:k]:
        return 1.0 / (list(predicted[:k]).index(actual) + 1)
    return 0.0

def mapk(y_true, y_pred, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n----- Fold {fold+1} / {n_splits} -----")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(le_fert.classes_),
        learning_rate=0.02,
        n_estimators=3000,
        max_depth=12,
        num_leaves=64,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    val_probs = model.predict_proba(X_val)
    val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
    fold_map3 = mapk(y_val.values, val_top3)
    print(f"Fold {fold+1} Validation MAP@3: {fold_map3:.4f}")
    map3_scores.append(fold_map3)

    test_preds += model.predict_proba(X_test) / n_splits


print(f"\nâœ… Mean Validation MAP@3: {np.mean(map3_scores):.4f}")


test_top3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
test_top3_labels = [' '.join(le_fert.inverse_transform(row)) for row in test_top3]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': test_top3_labels
})

submission.to_csv('submission.csv', index=False)
print("ğŸ“� submission.csv saved successfully!")

