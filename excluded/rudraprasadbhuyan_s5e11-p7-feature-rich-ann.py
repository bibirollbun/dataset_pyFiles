""" 
Goal: Feature-rich dataset and NN
 
Author: Rudra Prasad Bhuyan
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, 
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc
)
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers, regularizers, callbacks
)

from sklearn import set_config
set_config(display="diagram")

import warnings
warnings.simplefilter('ignore')


SUB_PATH    = r"/kaggle/input/playground-series-s5e11/sample_submission.csv" 
TRAIN_PATH  = r"/kaggle/input/playground-series-s5e11/train.csv"
TEST_PATH   = r"/kaggle/input/playground-series-s5e11/test.csv"

RANDOM_SEED = 42
BATCH_SIZE  = 4096
EPOCHS      = 100
VALID_SIZE  = 0.2
MODEL_OUT   = "best_ann.h5"
SUB_OUT     = "submission.csv"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)


def advanced_feature_engineering(df, is_train=True):
    """
    Comprehensive Feature Engineering for Loan Prediction

    Creates multiple groups of features:
      1. Financial ratios and metrics
      2. Credit score features
      3. Interest rate features
      4. Composite risk & affordability scores
      5. Loan amount transformations
      6. Binned (quantile) features
      7. Feature interactions
      8. Grade/subgrade parsing
      9. Statistical aggregations
      10. Combined categorical features
      11. Risk/anomaly flags

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (train or test)
    is_train : bool, default=True
        Whether this dataframe is training data

    Returns
    -------
    df : pd.DataFrame
        DataFrame with engineered features
    """

    df = df.copy()
    print("\nFEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    print(f"Starting features: {df.shape[1]}")

    # 1. FINANCIAL RATIOS -----------------------------------------------------
    print("\n[1/11] Financial ratio features...")
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment_estimate'] = (df['loan_amount'] * df['interest_rate']) / 1200
    df['payment_to_income_ratio'] = df['monthly_payment_estimate'] / (df['monthly_income'] + 1)
    df['current_debt_amount'] = df['debt_to_income_ratio'] * df['annual_income']
    df['total_debt_with_loan'] = df['current_debt_amount'] + df['loan_amount']
    df['new_debt_to_income'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    df['debt_increase_ratio'] = df['new_debt_to_income'] / (df['debt_to_income_ratio'] + 0.01)
    df['disposable_income'] = df['annual_income'] - df['current_debt_amount']
    df['disposable_income_ratio'] = df['disposable_income'] / (df['annual_income'] + 1)
    df['loan_to_disposable_income'] = df['loan_amount'] / (df['disposable_income'] + 1)
    df['monthly_disposable_income'] = df['disposable_income'] / 12
    df['payment_to_disposable_ratio'] = df['monthly_payment_estimate'] / (df['monthly_disposable_income'] + 1)
    df['annual_payment_burden'] = df['monthly_payment_estimate'] * 12
    df['payment_burden_ratio'] = df['annual_payment_burden'] / (df['annual_income'] + 1)
    print("✓ Created 15 features")

    # 2. CREDIT SCORE FEATURES -------------------------------------------------
    print("[2/11] Credit score features...")
    df['credit_score_normalized'] = df['credit_score'] / 850
    df['credit_risk_score'] = 1 - df['credit_score_normalized']
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['credit_score_log'] = np.log1p(df['credit_score'])
    df['credit_category'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 800, 850],
        labels=['poor', 'fair', 'good', 'very_good', 'excellent']
    )
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income']
    df['credit_times_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['credit_loan_interaction'] = df['credit_score'] * df['loan_amount']
    print("✓ Created 8 features")

    # 3. INTEREST RATE FEATURES -----------------------------------------------
    print("[3/11] Interest rate features...")
    df['high_interest_flag'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    df['very_high_interest'] = (df['interest_rate'] > df['interest_rate'].quantile(0.75)).astype(int)
    df['low_interest_flag'] = (df['interest_rate'] < df['interest_rate'].quantile(0.25)).astype(int)
    df['total_interest_cost'] = df['loan_amount'] * df['interest_rate'] / 100
    df['interest_burden'] = df['total_interest_cost'] / (df['annual_income'] + 1)
    df['interest_credit_mismatch'] = df['interest_rate'] * (1 - df['credit_score_normalized'])
    df['interest_credit_ratio'] = df['interest_rate'] / (df['credit_score'] / 100)
    df['interest_rate_squared'] = df['interest_rate'] ** 2
    df['interest_rate_log'] = np.log1p(df['interest_rate'])
    print("✓ Created 9 features")

    # 4. COMPOSITE RISK SCORES -------------------------------------------------
    print("[4/11] Composite risk & affordability scores...")
    df['risk_score_v1'] = (
        df['debt_to_income_ratio'] * 0.25 +
        df['loan_to_income_ratio'] * 0.25 +
        df['credit_risk_score'] * 0.30 +
        (df['interest_rate'] / 100) * 0.20
    )
    df['risk_score_v2'] = (
        df['payment_to_income_ratio'] * 0.40 +
        df['new_debt_to_income'] * 0.35 +
        df['interest_burden'] * 0.25
    )
    df['affordability_score'] = (
        df['credit_score_normalized'] * 0.40 +
        (1 - df['debt_to_income_ratio']) * 0.30 +
        df['disposable_income_ratio'] * 0.30
    )
    df['financial_health_score'] = df['affordability_score'] * 0.60 - df['risk_score_v1'] * 0.40
    print("✓ Created 4 features")

    # 5. LOAN AMOUNT FEATURES --------------------------------------------------
    print("[5/11] Loan amount transformations...")
    df['loan_size'] = pd.cut(df['loan_amount'],
        bins=[0, 10000, 20000, 30000, np.inf],
        labels=['small', 'medium', 'large', 'very_large']
    )
    df['loan_amount_squared'] = df['loan_amount'] ** 2
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    df['annual_income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_sqrt'] = np.sqrt(df['loan_amount'])
    print("✓ Created 5 features")

    # 6. BINNED FEATURES -------------------------------------------------------
    print("[6/11] Quantile-binned features...")
    df['income_decile'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop')
    df['credit_decile'] = pd.qcut(df['credit_score'], q=10, labels=False, duplicates='drop')
    df['loan_decile'] = pd.qcut(df['loan_amount'], q=10, labels=False, duplicates='drop')
    df['dti_decile'] = pd.qcut(df['debt_to_income_ratio'], q=10, labels=False, duplicates='drop')
    df['interest_decile'] = pd.qcut(df['interest_rate'], q=10, labels=False, duplicates='drop')
    print("✓ Created 5 features")

    # 7. INTERACTION FEATURES --------------------------------------------------
    print("[7/11] Interaction features...")
    df['income_x_credit'] = df['annual_income'] * df['credit_score']
    df['dti_x_interest'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    df['income_x_dti'] = df['annual_income'] * df['debt_to_income_ratio']
    df['income_credit_loan'] = df['annual_income'] * df['credit_score'] * df['loan_amount']
    df['dti_interest_credit'] = df['debt_to_income_ratio'] * df['interest_rate'] * df['credit_score']
    print("✓ Created 6 features")

    # 8. GRADE / SUBGRADE ------------------------------------------------------
    print("[8/11] Grade/subgrade parsing...")
    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].str[0]
        df['subgrade_num'] = df['grade_subgrade'].str[1:].astype(int)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df['grade_numeric'] = df['grade'].map(grade_map)
        df['full_grade_score'] = df['grade_numeric'] * 10 + df['subgrade_num']
        df['grade_credit_ratio'] = df['full_grade_score'] / (df['credit_score'] / 100)
        print("✓ Created 5 features")
    else:
        print("⚠️ Skipped: 'grade_subgrade' not found")

    # 9. STATISTICAL AGGREGATIONS ----------------------------------------------
    print("[9/11] Statistical aggregations...")
    base_cols = ['debt_to_income_ratio', 'loan_to_income_ratio', 'payment_to_income_ratio']
    df['mean_financial_metrics'] = df[base_cols].mean(axis=1)
    df['max_financial_burden'] = df[base_cols].max(axis=1)
    df['min_financial_burden'] = df[base_cols].min(axis=1)
    df['std_financial_metrics'] = df[base_cols].std(axis=1)
    print("✓ Created 4 features")

    # 10. CATEGORICAL COMBINATIONS --------------------------------------------
    print("[10/11] Combined categorical features...")
    if all(col in df.columns for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']):
        df['gender_marital'] = df['gender'] + '_' + df['marital_status']
        df['education_employment'] = df['education_level'] + '_' + df['employment_status']
        df['gender_education'] = df['gender'] + '_' + df['education_level']
        df['marital_employment'] = df['marital_status'] + '_' + df['employment_status']
        df['purpose_grade'] = df['loan_purpose'] + '_' + df.get('grade', '')
        df['employment_purpose'] = df['employment_status'] + '_' + df['loan_purpose']
        print("✓ Created 6 features")
    else:
        print("⚠️ Skipped categorical combinations (missing columns)")

    # 11. ANOMALY FLAGS --------------------------------------------------------
    print("[11/11] Risk/anomaly flags...")
    df['extreme_dti'] = (df['debt_to_income_ratio'] > df['debt_to_income_ratio'].quantile(0.90)).astype(int)
    df['low_income'] = (df['annual_income'] < df['annual_income'].quantile(0.25)).astype(int)
    df['large_loan'] = (df['loan_amount'] > df['loan_amount'].quantile(0.75)).astype(int)
    df['risky_combo_1'] = ((df['debt_to_income_ratio'] > 0.4) & (df['credit_score'] < 650)).astype(int)
    df['risky_combo_2'] = ((df['loan_to_income_ratio'] > 0.5) & (df['interest_rate'] > 15)).astype(int)
    df['safe_combo'] = ((df['credit_score'] > 750) & (df['debt_to_income_ratio'] < 0.3)).astype(int)
    df['high_risk_all'] = (df['extreme_dti'] & df['risky_combo_1']).astype(int)
    print("✓ Created 7 features")

    print("\n" + "=" * 80)
    print("Feature Engineering Complete!")
    print(f"Final features: {df.shape[1]}")
    print("=" * 80)

    return df



train_df = advanced_feature_engineering(train_df, is_train=True)


test_df = advanced_feature_engineering(test_df, is_train=False)


TARGET = "loan_paid_back"
ID_COL = "id"

NUMERIC_COLS = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate"
]
CAT_COLS = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]


numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore",))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CAT_COLS)
    ],
    remainder="drop"
)

preprocessor


X_all = train_df[NUMERIC_COLS + CAT_COLS]
y_all = train_df[TARGET].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=VALID_SIZE, 
    random_state=RANDOM_SEED, stratify=y_all
)


preprocessor.fit(X_train)

X_train_p = preprocessor.transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(test_df[NUMERIC_COLS + CAT_COLS])

print("Preprocessed shapes:", X_train_p.shape, X_val_p.shape, X_test_p.shape)


input_dim = X_train_p.shape[1]

def build_model(input_dim,
                l2=1e-5,
                dropout_rate=0.3,
                hidden_units=[256, 128, 64]):
    
    inp = layers.Input(shape=(input_dim,), name="input")
    x = inp
    
    for i, h in enumerate(hidden_units):
        x = layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
            name=f"dense_{i}"
        )(x)
        
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
        
    # Final layers
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2), name="dense_final")(x)
    x = layers.BatchNormalization(name="bn_final")(x)
    x = layers.Dropout(dropout_rate, name="drop_final")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    
    return model


model = build_model(input_dim,
                    l2=1e-4,
                    dropout_rate=0.25,
                    hidden_units=[512, 256, 128])  # modify to make deeper/wider if desired

model.summary()


class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)



es = callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_auc", mode="max", save_best_only=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4, mode="max", verbose=1)


history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[es, mc, rlr],
    verbose=2
)


val_preds = model.predict(X_val_p, batch_size=8192).ravel()
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC: {val_auc:.5f}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

val_preds = model.predict(X_val_p, batch_size=8192).ravel()

fpr, tpr, thresholds = roc_curve(y_val, val_preds)
roc_auc = auc(fpr, tpr)
print(f"Validation ROC AUC: {roc_auc:.5f}")


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Optional: Save history as CSV
# hist_df = pd.DataFrame(history.history)
# hist_df.to_csv("training_history.csv", index=False)


test_preds = model.predict(X_test_p, batch_size=8192).ravel()

# Create submission file: id + probability
submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    "loan_paid_back_prob": test_preds
})


if sub_df is not None:
    possible_targets = [c for c in sub_df.columns if c != ID_COL]
    if len(possible_targets) == 1:
        submission = pd.DataFrame({
            ID_COL: test_df[ID_COL],
            possible_targets[0]: test_preds
        })

submission.to_csv(SUB_OUT, index=False)
print(f"Saved submission to {SUB_OUT}")


pd.read_csv(r"/kaggle/working/submission.csv").head()

