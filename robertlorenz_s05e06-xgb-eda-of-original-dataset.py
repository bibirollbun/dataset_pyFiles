import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
original_data=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


print("------- Original Dataset -------")
print(original_data.info())
print(original_data.shape)
print(original_data.head())


df_numeric = original_data.drop(columns=["Soil Type", "Crop Type", "Fertilizer Name"] , axis=1)
corr = df_numeric.corr()

sns.heatmap(corr, cmap="coolwarm", annot=True)


fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 18))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i, col in enumerate(df_numeric.columns):
    sns.barplot(x="Fertilizer Name", y=col, data=original_data, ax=axes[i, 0], hue="Soil Type")
    axes[i, 0].set_title(f"{col} vs Fertilizer Name colored by Soil Type")
    sns.barplot(x="Fertilizer Name", y=col, data=original_data, ax=axes[i, 1], hue="Crop Type")
    axes[i, 1].set_title(f"{col} vs Fertilizer Name colored by Crop Type")
    sns.move_legend(axes[i, 0], "upper left", bbox_to_anchor=(1, 1))
    sns.move_legend(axes[i, 1], "upper left", bbox_to_anchor=(1, 1))


fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(5, 16))
plt.subplots_adjust(hspace=0.4)

for i, col in enumerate(df_numeric.columns):
    sns.boxplot(x="Fertilizer Name", y=col, data=original_data, ax=axes[i])
    axes[i].set_title(f"{col} vs Fertilizer Name")


cat_cols = [col for col in train.select_dtypes(include=['object']).columns if col != "Fertilizer Name"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    original_data[col]=le.transform(original_data[col])

label_encoder = LabelEncoder()
train['Fertilizer Name'] = label_encoder.fit_transform(train['Fertilizer Name'])
original_data['Fertilizer Name'] = label_encoder.transform(original_data['Fertilizer Name'])


original_data["Soil Type"]


K = 5

X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

kcv = KFold(n_splits=K, shuffle=True, random_state=42)

oof = np.zeros(shape=(len(X), y.nunique()))
oof_preds = np.zeros(len(X), dtype=int)
test_preds_proba = np.zeros((len(test), len(np.unique(y)))) 

for i, (train_idx, val_idx) in enumerate(kcv.split(X, y)):
    print(f"\n{'-'*5}Fold {i+1}/{K}{'-'*5}")

    X_train_ = X.iloc[train_idx].reset_index(drop=True)
    y_train_ = y.iloc[train_idx].reset_index(drop=True)
    X_val_ = X.iloc[val_idx].reset_index(drop=True)
    y_val_ = y.iloc[val_idx].reset_index(drop=True)

    X_train_ = pd.concat([X_train_, original_data.drop('Fertilizer Name', axis=1)])
    y_train_ = pd.concat([y_train_, original_data['Fertilizer Name']])

    xgboost_params = {
        'alpha': 4.93, 
        'colsample_bytree': 0.6, 
        'early_stopping_rounds': 328, 
        'eta': 0.0127, 
        'gamma': 0.232, 
        'max_delta_step': 5.62, 
        'max_depth': 22, 
        'min_child_weight': 6.916, 
        'n_estimators': 7206, 
        'reg_lambda': 1.28, 
        'subsample': 0.94,
        'device': 'cuda',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'n_jobs': -1,
        'enable_categorical': True
    }

    model = xgb.XGBClassifier(**xgboost_params)

    model.fit(X_train_, y_train_, eval_set=[(X_val_, y_val_)], verbose=100)

    oof[val_idx] = model.predict_proba(X_val_)
    test_pred_proba = model.predict_proba(test)
    
    # MAP@3 evaluation
    map_score = 0.0
    top_3_preds = np.argsort(oof[val_idx], axis=1)[:, -3:][:, ::-1]  
    for j in range(len(val_idx)):
        # Get top 3 predictions for this sample
        top_3_preds_j = top_3_preds[j]
        correct = 0
        precision = 0.0
        
        for k, pred in enumerate(top_3_preds_j):
            if pred == y_val_[j]:
                correct += 1
                precision += correct / (k + 1)
        
        # Average precision for this sample
        if correct > 0:
            map_score += precision / min(1, correct)
    
    print(f'Fold {i+1}: Map@3 score: {map_score / len(val_idx)}')


test_ids = test.index
top_3_test_preds = np.argsort(test_pred_proba, axis=1)[:, -3:][:, ::-1] 
labels = label_encoder.inverse_transform(top_3_test_preds.ravel())
print(labels)
pred_names = labels.reshape(top_3_test_preds.shape)

submission_format = [' '.join(row) for row in pred_names]
print(len(submission_format))
print(len(test_ids))
submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': submission_format})
submission.to_csv('submission.csv', index=False)
submission.head()

