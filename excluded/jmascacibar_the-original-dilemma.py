import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from category_encoders import OneHotEncoder
import warnings
import gc

pd.set_option("display.max_columns", 100)



def add_original_cols(df_train, df_test, df_orig, feats, target_col):
    """
    Add original features groupby original target to the synthetic data
    """
    train = df_train.copy()
    test = df_test.copy()
    tm = original[target_col].mean()
    add_feats = []
    for c in feats:
        if c in df_orig.columns:
            nm = f"{c}_OTmean"
            mapping = df_orig.groupby(c)[target_col].mean()
            train[nm] = train[c].map(mapping)
            train[nm] = train[nm].fillna(tm)
            test[nm] = test[c].map(mapping)
            test[nm] = test[nm].fillna(tm)
            add_feats.append(nm)
            print(f"Added {nm} feature to train and test sets")
    print("\n---- Complete ----\n")
    print(f"Train shape is: {train.shape}")
    print(f"Test shape is: {test.shape}")
    return train, test, add_feats

def preprocess(X_tr, X_val, x_test, cat_feats, enc_type ="ohe"):
    """
    Apply preprocessing
    """
    if enc_type == "ohe":
        ohecols = [c for c in cat_feats if c in X_tr.columns]
        from category_encoders import OneHotEncoder
        ohe = OneHotEncoder(
            cols=ohecols,
            use_cat_names=True,
            handle_unknown="return_nan"
        )
        X_tr = ohe.fit_transform(X_tr)
        X_val = ohe.transform(X_val)
        x_test = ohe.transform(x_test)
    else:
        print(f"Encoder type '{enc_type}' is not in the function. Skipping Encoding!")
    print(f"Preprocessing done!\n")
    print(f"X_tr shape: {X_tr.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    return X_tr, X_val, x_test

def run_lgbm_cv(df_train, 
                df_test, 
                basefeats, 
                cat_feats, 
                target, 
                params, 
                name = "LGBM_Starter",
                use_org_cols=False, 
                use_org_rows=False,
                orig_df=None, 
                prepro="ohe", 
                folds=5, 
                seed=2020, 
                save=True,
                sub=None):
    """
    LGBM training pipeline
    """
    print(f"{'##'*25}")
    print(f"## Starting Model: {name} ")
    print(f"{'##'*25}\n")

    if use_org_cols and use_org_rows:
        raise ValueError(
            "You are using `use_org_cols` and `use_org_rows` simultaneously. Choose one or the other"
        )
    if save and sub is None:
        raise ValueError(
            "You are using `save` without `sub`. Add a submission DF"
        )

    # original_cols
    cvfeats = basefeats.copy()
    train, test = df_train.copy(), df_test.copy()
    if use_org_cols:
        if orig_df is None:
            raise ValueError("`orig_df` must be provided when `use_org_cols` is True.")
        train, test, orig_feats = add_original_cols(train, test, orig_df, basefeats, target)
        cvfeats.extend(orig_feats)

    # CV Setup
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    f_aucs = []

    # CV Loop
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train[target]), 1):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        if use_org_rows:
            if orig_df is None:
                raise ValueError("`orig_df` must be provided when `use_org_rows` is True.")
            print("Augmenting training data with rows from original.")
            
            X_tr = pd.concat([train.loc[tr_idx, cvfeats], orig_df[cvfeats]], axis=0)
            X_val= train.loc[val_idx, cvfeats].copy()
            y_tr = pd.concat([train.loc[tr_idx, target], orig_df[target]], axis=0) 
            y_val = train.loc[val_idx, target]
            x_test = test[cvfeats].copy()
            
        else:
            X_tr = train.loc[tr_idx, cvfeats].copy()
            X_val = train.loc[val_idx, cvfeats].copy()
            y_tr = train.loc[tr_idx, target]
            y_val = train.loc[val_idx, target]
            x_test = test[cvfeats].copy()

        # preprocessing
        X_tr, X_val, x_test = preprocess(X_tr, X_val, x_test, cat_feats, enc_type=prepro)

        # training
        print("Starting training...")
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        clf = lgb.train(
            params,
            train_data,
            num_boost_round=10000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        # fi
        fi_df = pd.DataFrame({
            'Feature': X_tr.columns,
            'Gain': clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)}
            ).sort_values('Gain', ascending=False)

        topN_fi = 25
        fi_df_head = fi_df.head(topN_fi)

        plt.figure(figsize=(5, 4)) 
        sns.barplot(x='Gain', y='Feature', data=fi_df_head)
        plt.title(f'Top {topN_fi} Feature Importances (Gain) - Fold {fold}')
        plt.tight_layout()
        plt.show()
        
        # inferences and eval
        val_preds = clf.predict(X_val)
        oof_preds[val_idx] = val_preds
        test_preds += clf.predict(x_test) / folds
        
        f_auc = roc_auc_score(y_val, val_preds)
        f_aucs.append(f_auc)
        print(f"Fold {fold} ROC AUC: {f_auc:.5f}")
        
        del X_tr, X_val, y_tr, y_val, x_test, train_data, valid_data, clf
        gc.collect()
    
    print(f"\n\n{'-#'*15} Final Summary for {name} {'#-'*15}")
    final_auc = roc_auc_score(train[target], oof_preds)
    
    print(f"\nOverall OOF CV ROC AUC: {final_auc:.5f}")
    print("\nFold-wise ROC AUC scores:")
    for i, auc_score in enumerate(f_aucs, 1):
        print(f"  Fold {i}: {auc_score:.5f}")
    print(f"\nMean ROC AUC: {np.mean(f_aucs):.5f} ± {np.std(f_aucs):.5f}\n")
    
    print("Final Classification Report (on OOF preds):")
    print(classification_report(train[target], (oof_preds > 0.5).astype(int), digits=4))
    
    results = {
        'name': name,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'auc': final_auc
    }

    if save:
        print(f"{'-#'*15} Saving OOF and TEST predictions for {name} {'#-'*15}")
        
        os.makedirs("oof_preds", exist_ok=True)
        os.makedirs("test_preds", exist_ok=True)
        os.makedirs("submission", exist_ok=True)

        oof_fn = f"oof_preds/{name}_oof.npy"
        test_fn = f"test_preds/{name}_test.npy"
        np.save(oof_fn, oof_preds)
        np.save(test_fn, test_preds)
        print(f"Saved OOF predictions to: '{oof_fn}'")
        print(f"Saved Test predictions to: '{test_fn}'")

        sub = sub.copy()
        sub[target] = test_preds
        sub_nm = f"submission/{name}_sub.csv"
        sub.to_csv(sub_nm, index=False)
        print(f"Saved submission file to: '{sub_nm}'")
        print("\nSubmission file head:")
        print(sub.head())
    
    return results


train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path = "/kaggle/input/playground-series-s5e8/test.csv"
original_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"
sub_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"

train  = pd.read_csv(train_path).drop("id", axis=1)
test = pd.read_csv(test_path).drop("id", axis=1)
original = pd.read_csv(original_path, sep= ";")
original["y"] = original["y"].map({"no": 0, "yes":1})
sub = pd.read_csv(sub_path)

print(f"Train shape is: {train.shape}")
print(f"Test shape is: {test.shape}")
print(f"Original shape is: {test.shape}")
display(train.head())
display(test.head())
display(original.head())
print(f"Class Positive rate: {train.y.sum()/len(train):.3%}")
cat_feats = ["job", "marital", "education", "housing", "default", "loan" ,"contact", "month", "poutcome"]
num_feats = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
features = cat_feats + num_feats
print(f"Total features: {len(features)}")
print(f"Numerical features: {len(num_feats)}")
print(f"Categorical features: {len(cat_feats)}")

TARGET_COL = 'y'
BASE_FEATURES = [col for col in train.columns if col != TARGET_COL]
CAT_FEATURES = ["job", "marital", "education", "housing", "default", "loan" ,"contact", "month", "poutcome"]
SEED = 2020
FOLDS = 5

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "max_depth": 4, 
    "learning_rate": 0.08,
    "colsample_bytree": 0.7,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
    "device": "gpu" 
    }

ALL_RESULTS = []


# Baseline
lgbm_bs = run_lgbm_cv(
    train,
    test,
    basefeats=BASE_FEATURES,
    cat_feats=CAT_FEATURES,
    target=TARGET_COL,
    params=LGBM_PARAMS,
    name="LGBM BASELINE",
    use_org_cols=False,
    folds=FOLDS,
    seed=SEED,
    save=True,
    sub=sub)
ALL_RESULTS.append(lgbm_bs)


# Adding original rows
lgbm_org_rows = run_lgbm_cv(
    train,
    test, 
    orig_df=original,
    basefeats=BASE_FEATURES, 
    cat_feats=CAT_FEATURES,
    target=TARGET_COL, 
    params=LGBM_PARAMS,
    name="LGBM_with_Orig_Rows", 
    use_org_cols=False,
    use_org_rows=True,
    seed=SEED, 
    save=True,
    sub=sub
)
ALL_RESULTS.append(lgbm_org_rows)


# Adding original cols
lgbm_org_cols = run_lgbm_cv(
    train, 
    test, 
    orig_df=original,
    basefeats=BASE_FEATURES, 
    cat_feats=CAT_FEATURES,
    target=TARGET_COL, 
    params=LGBM_PARAMS,
    name="LGBM_with_Orig_Features", 
    use_org_cols=True, 
    seed=SEED,
    save=True,
    sub=sub
)
ALL_RESULTS.append(lgbm_org_cols)


print(f"\n{'**'*25}")
print(f"** Final Model Comparison **")
print(f"{'**'*25}\n")
for result in ALL_RESULTS:
    print(f"Model: {result['name']:<30} | OOF AUC: {result['auc']:.5f}")

