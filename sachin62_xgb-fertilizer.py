import numpy as np 
import pandas as pd
import time 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split , KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
original_data=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


for k in range(4):
    train = pd.concat([train, original_data], ignore_index=True)
    
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test.drop("id",axis=1,inplace=True)


le = LabelEncoder()
train['Fertilizer Encoded'] = le.fit_transform(train['Fertilizer Name'])


table_feature=[ 'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
       'Nitrogen', 'Potassium', 'Phosphorous',]


X=train[table_feature]
y=train['Fertilizer Encoded']


num_cols=['Temparature', 'Humidity', 'Moisture',
       'Nitrogen', 'Potassium', 'Phosphorous',]


cat_cols=[ 'Soil Type', 'Crop Type']


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(), cat_cols)
    ]
)



def mapk(true_labels, pred_probs, k=3):
    """Compute Mean Average Precision at k"""
    score = 0.0
    for true, preds in zip(true_labels, pred_probs):
        try:
            pred_top_k = preds[:k]
            if true in pred_top_k:
                rank = pred_top_k.index(true) + 1
                score += 1.0 / rank
        except:
            continue
    return score / len(true_labels)



# Cross-validation setup
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X), dtype=int)
oof_probs = np.zeros((len(X), len(np.unique(y))))
test_preds_proba = np.zeros((len(test), len(np.unique(y))))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n --> Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scaling
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    test_scaled = preprocessor.transform(test)

    # Model
    model = XGBClassifier(
        max_depth=16,
        colsample_bytree=0.4,
        subsample=0.86,
        n_estimators=6000,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=5,
        reg_alpha=3,
        reg_lambda=1.4,
        early_stopping_rounds=400,
        objective='multi:softprob',
        random_state=42,
        enable_categorical=True,
        min_child_weight=5,
        device='cuda',
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    start = time.time()

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )

    # Predict probabilities
    val_probs = model.predict_proba(X_val_scaled)
    test_probs = model.predict_proba(test_scaled)

    # Store out-of-fold probabilities
    oof_probs[val_idx] = val_probs
    test_preds_proba += test_probs

    # Top 3 predictions for MAP@3
    val_top3_preds = np.argsort(val_probs, axis=1)[:, ::-1][:, :3]
    val_top3_preds_list = val_top3_preds.tolist()
    map3 = mapk(y_val.values, val_top3_preds_list, k=3)

    print(f"Fold {fold} MAP@3: {map3:.4f}")
    print(f"Time: {time.time() - start:.1f} sec")

# Average test predictions
test_preds_proba /= FOLDS

# Final OOF MAP@3
oof_top3 = np.argsort(oof_probs, axis=1)[:, ::-1][:, :3].tolist()
final_map3 = mapk(y.values, oof_top3, k=3)
print(f"\nFinal OOF MAP@3: {final_map3:.4f}")





# FOLDS = 5
# kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# oof_preds = np.zeros(len(X), dtype=int)
# test_preds_proba = np.zeros((len(test), len(np.unique(y)))) 

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
#     print(f"\n --> Fold {fold}")

#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     X_train_scaled = preprocessor.fit_transform(X_train)
#     X_val_scaled = preprocessor.transform(X_val)
#     test_scaled=preprocessor.transform(test)
    
#     model = XGBClassifier(
#         max_depth=16,
#         colsample_bytree=0.4,
#         subsample=0.86,
#         n_estimators=6000,
#         learning_rate=0.005,
#         gamma=0.26,
#         max_delta_step=5,
#         reg_alpha=3,
#         reg_lambda=1.4,
#         early_stopping_rounds=400,
#         objective='multi:softprob',
#         random_state=42,
#         enable_categorical=True,
#         min_child_weight=5,     
#         device='cuda',
#         n_jobs=-1,
#         eval_metric='mlogloss'
#     )


#     start = time.time()
    
#     model.fit(
#         X_train_scaled, y_train,
#         eval_set=[(X_val_scaled, y_val)],
#         verbose= 100
#     )

#     val_preds = model.predict(X_val_scaled)
#     oof_preds[val_idx] = val_preds

#     test_preds_proba += model.predict_proba(test_scaled)

#     acc = accuracy_score(y_val, val_preds)
#     print(f"Fold {fold} Accuracy: {acc:.4f}")
#     print(f"Time: {time.time() - start:.1f} sec")

# test_preds_proba /= FOLDS

# oof_acc = accuracy_score(y, oof_preds)
# print(f"\n Final OOF Accuracy: {oof_acc:.4f}")


top_3_preds = np.argsort(test_preds_proba, axis=1)[:, -3:][:, ::-1]  
top3_labels = np.array([le.inverse_transform(row) for row in top_3_preds])

top3_joined = [" ".join(row) for row in top3_labels]
sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
sub["Fertilizer Name"]=top3_joined
sub.head()
sub.to_csv("submission.csv",index=False)







