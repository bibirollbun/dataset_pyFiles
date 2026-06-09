import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from lightgbm import LGBMClassifier



train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train_df.info()


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TARGET = "loan_paid_back"

train = train_df.copy()
test = test_df.copy()

for data in [train, test]:
    data.drop(columns=["id"], errors="ignore", inplace=True)



for data in [train, test]:
    data["loan_to_income"] = data["loan_amount"] / (data["annual_income"] + 1e-6)
    data["loan_to_credit"] = data["loan_amount"] / (data["credit_score"] + 1e-6)
    data["risk_ratio"] = data["debt_to_income_ratio"] * data["interest_rate"]
    data["payment_to_income"] = (data["loan_amount"] * data["interest_rate"]) / (data["annual_income"] + 1e-6)



for data in [train, test]:
    data["income_credit"] = data["annual_income"] * data["credit_score"]
    data["loan_interest"] = data["loan_amount"] * data["interest_rate"]
    data["interest_dti"] = data["interest_rate"] * data["debt_to_income_ratio"]


train["credit_bin"] = pd.qcut(train["credit_score"], q=5, duplicates="drop")
test["credit_bin"] = pd.qcut(test["credit_score"], q=5, duplicates="drop")

train["income_bin"] = pd.qcut(train["annual_income"], q=5, duplicates="drop")
test["income_bin"] = pd.qcut(test["annual_income"], q=5, duplicates="drop")


cat_cols = ["gender", "marital_status", "education_level",
            "employment_status", "loan_purpose", "grade_subgrade"]

for c in cat_cols:
    freq = train[c].value_counts(dropna=False)
    train[c + "_freq"] = train[c].map(freq)
    test[c + "_freq"] = test[c].map(freq)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

for c in cat_cols:
    train[c + "_te"] = 0

    for tr_idx, val_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]

        enc = tr.groupby(c)[TARGET].mean()
        train.loc[val_idx, c + "_te"] = val[c].map(enc)

    enc = train.groupby(c)[TARGET].mean()
    test[c + "_te"] = test[c].map(enc)


for c in cat_cols:
    stats = train.groupby(c).agg({
        "annual_income": "mean",
        "loan_amount": "mean",
        "credit_score": "mean"
    }).rename(columns={
        "annual_income": f"{c}_income_mean",
        "loan_amount": f"{c}_loan_mean",
        "credit_score": f"{c}_credit_mean"
    })

    train = train.merge(stats, on=c, how="left")
    test = test.merge(stats, on=c, how="left")



print("Train:", train.shape)
print("Test :", test.shape)



train_df['annual_income_log'] = np.log1p(train_df['annual_income'])
train_df['debt_to_income_ratio_log'] = np.log1p(train_df['debt_to_income_ratio'])
train_df.drop(columns=['annual_income', 'debt_to_income_ratio'], inplace=True)

test_df['annual_income_log'] = np.log1p(test_df['annual_income'])
test_df['debt_to_income_ratio_log'] = np.log1p(test_df['debt_to_income_ratio'])
test_df.drop(columns=['annual_income', 'debt_to_income_ratio'], inplace=True)


education_order = [
    "Other",
    "High School",
    "Bachelor's",
    "Master's",
    "PhD"
]
grade_order = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5',
]
ordinal_cols = ['education_level', 'grade_subgrade']
ordinal_categories = [education_order, grade_order]

nominal_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']

preprocessor = ColumnTransformer([
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols)
], remainder='passthrough')


X_train = train_df.drop(columns=['loan_paid_back', 'id'])
y_train = train_df['loan_paid_back']
X_test = test_df.drop(columns=['id'])


X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train), 1):
    print(f"\n----- Fold {fold} -----")

    X_tr, X_val = X_train_encoded[train_idx], X_train_encoded[val_idx]
    y_trn, y_val = y_train[train_idx], y_train[val_idx]


    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

   
    model = Sequential([

         Input(shape=(X_train_encoded.shape[1],)),

        Dense(256, activation="gelu"),
        Dropout(0.20),

        Dense(128, activation="gelu"),
        Dropout(0.15),

        Dense(64, activation="gelu"),
        Dropout(0.10),

        Dense(32, activation="gelu"),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    es = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_tr,
        y_trn,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=1024,
        verbose=1,
        callbacks=[es]
    )

    
    preds = model.predict(X_val).ravel()

    fold_auc = roc_auc_score(y_val, preds)
    roc_auc.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.4f}")
print("Average ROC-AUC: ", np.mean(roc_auc))


scaler = StandardScaler()
X_train_encoded = scaler.fit_transform(X_train_encoded)

X_test_encoded = scaler.transform(X_test_encoded)


model = Sequential([
        Dense(512, activation="relu", input_shape=(X_train_encoded.shape[1],)),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

es = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )
model.fit(
        X_train_encoded,
        y_train,
        epochs=50,
        batch_size=1024,
        verbose=1,
        # callbacks=[es]
    )


lgb_params = { 'verbosity':-1, 'n_estimators': 1942, 'max_depth': 3, 'learning_rate': 0.12093539056257775, 'subsample': 0.9643697539245966, 'colsample_bytree': 0.6138587381273723, 'min_child_weight': 4, 'reg_alpha': 3.9731738630617075, 'reg_lambda': 9.182017682059731}
lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(X_train_encoded, y_train)



submission = pd.DataFrame({
    'id':test_df['id'],
    'loan_paid_back': (model.predict(X_test_encoded).ravel())
})
submission.to_csv("submission.csv", index=False)

submission.head()

