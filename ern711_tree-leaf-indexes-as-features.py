import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
NUMS = [col for col in train.columns if col not in CATS and col not in ['id', 'loan_paid_back']]
FEATURES = [f for f in train.columns if f not in ['id', 'loan_paid_back']]
TARGET = 'loan_paid_back'


# Run for different seeds...
for seed in range(10):

    print(f'<----Round {seed+1}---->')
    
    # 1) Train / test split on original data
    df_train, df_test = train_test_split(
        train, test_size=0.2, random_state=seed, stratify=train[TARGET]
    )
    
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_test  = df_test[FEATURES]
    y_test  = df_test[TARGET]
    
    # 2) First CatBoost model on original features
    cb1 = CatBoostClassifier(
        iterations=500,
        loss_function="Logloss",
        logging_level="Silent",
        random_state=42,
        task_type='GPU'
    )
    
    cb1.fit(X_train, y_train, cat_features=CATS)
    
    # Baseline performance of first model
    pred1 = cb1.predict_proba(X_test)[:, 1]
    print("CatBoost #1 AUC (original features):", roc_auc_score(y_test, pred1))

    # 3) Extract leaf indexes from cb1
    train_pool = Pool(X_train, label=y_train, cat_features=CATS)
    test_pool  = Pool(X_test,  label=y_test,  cat_features=CATS)
    
    leaf_train = cb1.calc_leaf_indexes(train_pool)  # shape (n_train, n_trees)
    leaf_test  = cb1.calc_leaf_indexes(test_pool)   # shape (n_test,  n_trees)
    
    leaf_train = leaf_train.astype(np.int64)
    leaf_test  = leaf_test.astype(np.int64)
    
    n_trees = leaf_train.shape[1]
    print("Leaf train shape:", leaf_train.shape)
    print("Leaf test shape:", leaf_test.shape)
    
    # 4) Create DataFrames of leaf indexes
    leaf_cols = [f"leaf_{t}" for t in range(n_trees)]
    
    leaf_train_df = pd.DataFrame(leaf_train, index=df_train.index, columns=leaf_cols)
    leaf_test_df  = pd.DataFrame(leaf_test,  index=df_test.index,  columns=leaf_cols)
    
    
    # For the second CatBoost, all leaf columns are categorical features
    leaf_FEATURES = leaf_cols
    leaf_CATS = list(range(len(leaf_FEATURES)))  # all columns are categorical
    
    # 5) Second CatBoost model using leaf-index features as categorical features
    cb2 = CatBoostClassifier(
        iterations=500,
        loss_function="Logloss",
        logging_level="Silent",
        random_state=42,
        task_type='GPU'
    )
    
    cb2.fit(
        leaf_train_df[leaf_FEATURES],
        y_train,
        cat_features=leaf_CATS
    )
    
    pred2 = cb2.predict_proba(leaf_test_df[leaf_FEATURES])[:, 1]
    print("CatBoost #2 AUC (on leaf-index features):", roc_auc_score(y_test, pred2))

    





