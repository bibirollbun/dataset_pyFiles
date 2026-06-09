import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train_copy, test_copy = train_data, test_data


train_data.sample(5)


test_data.sample(5)


train_data.shape


test_data.shape


train_data.isnull().sum()


test_data.isnull().sum()


train_data.info()


categorical_features = ['Soil Type', 'Crop Type']
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
le=LabelEncoder()
train_data.drop("id",axis=1,inplace=True)
test_data.drop("id",axis=1,inplace=True)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

train_data["Fertilizer Name"]=le.fit_transform(train_data["Fertilizer Name"])

X=train_data.drop(columns=["Fertilizer Name"])
y=train_data["Fertilizer Name"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OrdinalEncoder(), categorical_features)
    ]
)


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time


FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X), dtype=int)
test_preds_proba = np.zeros((len(test_data), len(np.unique(y)))) 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    test_scaled=preprocessor.transform(test_data)

    model = XGBClassifier(
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=4000,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        early_stopping_rounds=100,
        objective='multi:softprob',
        random_state=13,
        enable_categorical=True,
        tree_method='hist',     
        device='cuda'
    )
    start = time.time()

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )

    val_preds = model.predict(X_val_scaled)
    oof_preds[val_idx] = val_preds

    test_preds_proba += model.predict_proba(test_scaled)

    acc = accuracy_score(y_val, val_preds)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Time: {time.time() - start:.1f} sec")

test_preds_proba /= FOLDS

oof_acc = accuracy_score(y, oof_preds)
print(f"\n Final OOF Accuracy: {oof_acc:.4f}")


top_3_preds = np.argsort(test_preds_proba, axis=1)[:, -3:][:, ::-1]  
top3_labels = np.array([le.inverse_transform(row) for row in top_3_preds])

top3_joined = [" ".join(row) for row in top3_labels]
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission["Fertilizer Name"] = top3_joined
submission


submission.to_csv("submission.csv",index=False)

