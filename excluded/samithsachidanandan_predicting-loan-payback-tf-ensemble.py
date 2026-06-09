import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'







train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train_ids = train['id']
test_ids = test['id']


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


train.head()


test.head()


train.shape


train.info()


train.dtypes


print("Target column statistics (loan_paid_back):")

train['loan_paid_back'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train.describe().T


def remove_outliers(train_df, test_df=None):

    train_df = train_df.copy()


    credit_upper = train_df['credit_score'].mean() + 3 * train_df['credit_score'].std()
    credit_lower = train_df['credit_score'].mean() - 3 * train_df['credit_score'].std()
    rate_upper = train_df['interest_rate'].mean() + 3 * train_df['interest_rate'].std()
    rate_lower = train_df['interest_rate'].mean() - 3 * train_df['interest_rate'].std()


    train_df['credit_score'] = np.clip(train_df['credit_score'], credit_lower, credit_upper)
    train_df['interest_rate'] = np.clip(train_df['interest_rate'], rate_lower, rate_upper)


    features = ['annual_income', 'debt_to_income_ratio', 'loan_amount']
    limits = {}

    for feature in features:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        limits[feature] = {
            'lower': Q1 - 1.5 * IQR,
            'upper': Q3 + 1.5 * IQR
        }
        train_df[feature] = np.clip(train_df[feature], limits[feature]['lower'], limits[feature]['upper'])


    if test_df is not None:
        test_df = test_df.copy()
        test_df['credit_score'] = np.clip(test_df['credit_score'], credit_lower, credit_upper)
        test_df['interest_rate'] = np.clip(test_df['interest_rate'], rate_lower, rate_upper)

        for feature in features:
            test_df[feature] = np.clip(test_df[feature], limits[feature]['lower'], limits[feature]['upper'])

        return train_df, test_df

    return train_df








def engineer_features(df):

    df = df.copy()


    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']

    df['total_debt'] = df['loan_amount'] * df['debt_to_income_ratio']
    df['monthly_debt'] = df['total_debt'] / 12
    df['remaining_income'] = df['monthly_income'] - df['monthly_debt']

    df['credit_efficiency'] = df['credit_score'] / (df['debt_to_income_ratio'] + 0.001)
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']


    df['risk_score'] = (df['debt_to_income_ratio'] * df['interest_rate']) / (df['credit_score'] + 1)


    df['income_credit_interaction'] = df['annual_income'] * df['credit_score']
    df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_score']


    df['credit_score_squared'] = df['credit_score'] ** 2
    df['debt_ratio_squared'] = df['debt_to_income_ratio'] ** 2
    df['income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])


    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['education_employment'] = df['education_level'] + '_' + df['employment_status']


    df['high_risk_flag'] = ((df['debt_to_income_ratio'] > 0.4) |
                            (df['credit_score'] < 650) |
                            (df['interest_rate'] > 15)).astype(int)

    df['excellent_credit_flag'] = (df['credit_score'] >= 750).astype(int)
    df['high_income_flag'] = (df['annual_income'] >= 50000).astype(int)
    df['has_advanced_degree'] = (df['education_level'].isin(["Master's", "PhD"])).astype(int)

    return df



train, test = remove_outliers(train, test)


train.columns


train_df = engineer_features(train)
test_df = engineer_features(test)


y_train = train_df['loan_paid_back']
X_train = train_df.drop('loan_paid_back', axis=1)

X_test = test_df.copy()


cols_to_drop = [col for col in X_train.columns if col.startswith('_')]
if cols_to_drop:
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
print(f"Dropped temporary columns: {cols_to_drop}")


numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()


categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()



print("*"*180)
print("Numeric:", numeric_cols)

print("*"*180)

print("Categorical:", categorical_cols)
print("*"*180)




preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
    ('scale', MinMaxScaler(), numeric_cols)
])




def build_nn(input_dim):
    model = models.Sequential([

        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='roc_auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


print("Starting K-Fold Cross-Validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)

    class_weights_fold = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_tr),
        y=y_tr
    )
    class_weight_dict = {0: class_weights_fold[0], 1: class_weights_fold[1]}

    model = build_nn(input_dim=X_tr_trans.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_roc_auc', patience=15, mode='max',
                     restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=5,
                         mode='max', min_lr=1e-7, verbose=0)
    ]

    history = model.fit(
        X_tr_trans, y_tr,
        validation_data=(X_val_trans, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0
    )


    y_val_pred = model.predict(X_val_trans, verbose=0)
    auc = roc_auc_score(y_val, y_val_pred)
    roc_aucs.append(auc)
    fold_models.append(model)


    best_epoch = np.argmax(history.history['val_roc_auc']) + 1
    best_val_auc = max(history.history['val_roc_auc'])
    total_epochs = len(history.history['val_roc_auc'])

    print(f"Fold {fold + 1} ROC-AUC: {auc:.4f}")
    print(f"Best epoch: {best_epoch}/{total_epochs} (Val ROC-AUC: {best_val_auc:.4f})")

print(f"\n{'='*50}")
print(f"Mean CV ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")
print(f"{'='*50}\n")



print("Training final model on full training data...")


X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}


callbacks = [
    EarlyStopping(monitor='val_roc_auc', patience=20, mode='max',
                 restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=7,
                     mode='max', min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_roc_auc', mode='max',
                   save_best_only=True, verbose=1)
]


history = model.fit(
    X_train_trans, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


best_val_auc = max(history.history['val_roc_auc'])
print(f"\nBest Val ROC-AUC: {best_val_auc:.4f}")


final_pred = model.predict(X_test_trans, verbose=0)


print("Loading XGBoost model...")

try:
    with open('/kaggle/input/predicting-loan-payback-xgboost-lb-0-92331/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print("Could not load XGBoost model:", e)
    xgb_model = None



if xgb_model is not None:

    print("\n" + "="*60)
    print("Selecting Best Ensemble")
    print("="*60)

    print("Predicting on validation set...")

  
    xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]

 
    nn_val_pred = model.predict(X_val_trans, verbose=0).flatten()


    print("Predicting on test set...")

    xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
    nn_test_pred = final_pred.flatten()


    assert len(xgb_test_pred) == len(nn_test_pred), (
        f"Size mismatch! XGB={len(xgb_test_pred)} NN={len(nn_test_pred)}"
    )


    weights = [
        (0.70, 0.30),
        (0.75, 0.25),
        (0.65, 0.35),
        (0.80, 0.20),
    ]

    results = []

    for w_xgb, w_nn in weights:
        ensemble_val = w_xgb * xgb_val_pred + w_nn * nn_val_pred
        auc = roc_auc_score(y_val, ensemble_val)

        results.append({"xgb_w": w_xgb, "nn_w": w_nn, "auc": auc})
        print(f"Weights XGB {w_xgb:.0%} / NN {w_nn:.0%} → Val ROC-AUC = {auc:.5f}")

 
    best = max(results, key=lambda x: x["auc"])
    print("\nBest Ensemble on validation set:", best)


    final_ensemble = best["xgb_w"] * xgb_test_pred + best["nn_w"] * nn_test_pred

   
    submission['loan_paid_back'] = final_ensemble
    submission.to_csv("submission.csv", index=False)
    print("\nBest submission saved as submission.csv")
    print(f"Prediction range: [{final_ensemble.min():.4f}, {final_ensemble.max():.4f}]")


else:
    print("\nXGBoost model not found. Using NN only.")
    submission['loan_paid_back'] = final_pred.flatten()
    submission.to_csv("submission.csv", index=False)
    print("Saved as submission.csv")





