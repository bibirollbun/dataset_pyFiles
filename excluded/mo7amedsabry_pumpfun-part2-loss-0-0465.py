import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, log_loss)
from sklearn.model_selection import StratifiedKFold
import catboost as cb
from sklearn.metrics import precision_score



data=pd.read_csv("train_v3.csv")


data


data=data.drop(columns=['mint','token_mint_address', 'name_x', 'symbol_x'],axis=1)


data.info()


data=data.dropna()


symbol_features = ['name_length', 'name_word_count', 'symbol_length', 'symbol_has_digits']
data[symbol_features].hist(bins=20, figsize=(12, 8))
plt.suptitle("Name and Symbol Features Distribution")
plt.show()



def scale_with_robust_scaler(df):
    """
    Automatically detects numeric columns and applies RobustScaler to them.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Scaled DataFrame with same column names
    """
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    scaler = RobustScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_scaled



scaled_df = scale_with_robust_scaler(data)


X = scaled_df.drop(columns=['has_graduated'])
y = scaled_df['has_graduated']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10,random_state=42,stratify=y
)


def balance_with_smote_tomek(X, y, random_state=42):
    smt = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    return X_resampled, y_resampled



X_train_balanced, y_train_balanced = balance_with_smote_tomek(X_train, y_train)


class_distribution = y_train_balanced.value_counts()
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='pie', 
                       autopct='%1.1f%%', 
                       colors=['skyblue', 'salmon'],
                       startangle=90,
                       labels=['Not Graduated (0)', 'Graduated (1)'])
plt.title('Class Distribution of has_graduated')
plt.ylabel('')  # Remove y-axis label
plt.show()


def train_lgb_with_cv(X, y, N_SPLITS=5, RANDOM_SEED=42):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(len(X))
    oof_true = np.zeros(len(X))
    models = []
    thresholds = []
    aucs = []
    reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        params = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 5,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'class_weight': {0: 1, 1: 20}  
             }


        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        oof_true[val_idx] = y_val

        precisions, recalls, thresholds_fold = precision_recall_curve(y_val, val_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds_fold[np.argmax(f1_scores)]
        thresholds.append(best_threshold)

        y_val_pred = (val_proba >= best_threshold).astype(int)

        report = classification_report(y_val, y_val_pred, output_dict=True)
        auc = roc_auc_score(y_val, val_proba)

        print(f"Fold {fold+1} AUC: {auc:.5f}")
        print(f"Fold {fold+1} Best Threshold: {best_threshold:.4f}")

        models.append(model)
        aucs.append(auc)
        reports.append(report)

    logloss_score = log_loss(oof_true, oof_preds)
    overall_auc = roc_auc_score(oof_true, oof_preds)

    print(f"\nğŸ“‰ Overall LogLoss: {logloss_score:.5f}")
    print(f"ğŸ“Š Overall AUC: {overall_auc:.5f}")

    return models, thresholds, oof_preds, oof_true, aucs, reports



lgb_models, thresholds, oof_preds, y_true, aucs, reports = train_lgb_with_cv(X_train, y_train)


def evaluate_lgb_models_on_test(models, thresholds, X_test, y_test):
    test_preds = np.zeros(len(X_test))
    for model in models:
        test_preds += model.predict_proba(X_test)[:, 1] / len(models)

    avg_threshold = np.mean(thresholds)
    y_pred_binary = (test_preds >= avg_threshold).astype(int)

    test_logloss = log_loss(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, y_pred_binary, digits=4)

    print(f"ğŸ“‰ Test LogLoss: {test_logloss:.5f}")
    print(f"ğŸ“Š Test AUC: {test_auc:.5f}")
    print("ğŸ“‹ Classification Report:")
    print(report)

    return  test_logloss, test_auc, report



evaluate_lgb_models_on_test(lgb_models, thresholds, X_test, y_test)


def train_catboost_with_cv(X, y, N_SPLITS=5, RANDOM_SEED=42):

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cb_oof_preds = np.zeros(len(X))  
    cb_test_preds = np.zeros(len(X))  
    
    cb_models = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = cb.CatBoostClassifier(
    iterations=2500,
    learning_rate=0.01,
    loss_function='Logloss',
    eval_metric='Logloss',
    task_type='GPU',
    depth=6,  # Reduced depth â†’ simpler model generalizes better
    l2_leaf_reg=3,  
    random_seed=RANDOM_SEED + fold,
    verbose=0,
    early_stopping_rounds=100,
    class_weights={0: 1, 1: 20},
    grow_policy='Lossguide',  # Encourages smaller trees based on loss
    random_strength=1,  # Add randomness to tree splits
    bagging_temperature=1.0,  
    od_type='Iter',  
)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  use_best_model=True)

        val_preds = model.predict_proba(X_val)[:, 1]
        cb_oof_preds[val_idx] = val_preds  
        cb_models.append(model)
        print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds):.5f}")
    
    return cb_models, cb_oof_preds, cb_test_preds


cb_models, cb_oof_preds, cb_test_preds = train_catboost_with_cv(
    X=X_train, 
    y=y_train, 
    N_SPLITS=5, 
    RANDOM_SEED=42
)


logloss_score = log_loss(y_train, cb_oof_preds)
print(f"ğŸ“‰ Overall OOF LogLoss: {logloss_score:.5f}")
auc_score = roc_auc_score(y_train, cb_oof_preds)
print(f"ğŸ“Š AUC: {auc_score:.5f}")
y_pred_binary = (cb_oof_preds >= 0.5).astype(int)
print("ğŸ“Š Classification Report (OOF Predictions):")
print(classification_report(y_train, y_pred_binary, digits=4))


for t in [0.3, 0.5, 0.6, 0.7, 0.8]:
    preds = (cb_oof_preds > t).astype(int)
    precision = precision_score(y_train, preds)
    print(f"Threshold {t}: Precision for class 1 = {precision:.4f}")


important_features = cb_models[0].get_feature_importance(prettified=True)
important_features = important_features[important_features['Importances'] > 0.1]



important_feature_names = important_features['Feature Id'].tolist()


X = scaled_df[important_feature_names]
y = scaled_df['has_graduated']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10,random_state=42,stratify=y
)


cb_models, cb_oof_preds, cb_test_preds = train_catboost_with_cv(
    X=X_train, 
    y=y_train, 
    N_SPLITS=5, 
    RANDOM_SEED=42
)



def test_logloss_over_factors(models, X_test, y_test, factors=np.linspace(1.0, 2.5, 16)):
    '''''''''
    ''''''''''''''''''''''
    '''''''''''''''''''''''''''
    '''''''''''''''''''''''''''''
    '''''''''''''''''''''''''''''''

    print(f"\nâœ… Best Factor: {best_factor:.2f} â�¤ Best LogLoss: {best_logloss:.5f}")
    return best_factor, best_logloss



best_factor, best_test_logloss = test_logloss_over_factors(
    models=cb_models,
    X_test=X_test,
    y_test=y_test
)


test_data=pd.read_csv("test_v3.csv")


test_data = test_data[list(X_train.columns) + ['mint']]


def create_submission(input_data: pd.DataFrame, model, id_column='mint'):
    mint_ids = input_data[id_column].values
    features = input_data.drop(columns=[id_column])
    preds_proba = model.predict_proba(features)[:, 1]  # Clip probabilities if needed
    submission = pd.DataFrame({
        id_column: mint_ids,
        'has_graduated': preds_proba
    })

    return submission



def adjust_submission_probs(submission: pd.DataFrame, factor: float = 1.80, clip_eps: float = 1e-15) -> pd.DataFrame:
    ''''''''
    ''''''''
    '''''''''
    ''''''''''

    return adjusted


submission_df = create_submission(test_data, [cb_models[0]])
submission_df


adjusted_submission = adjust_submission_probs(submission_df, factor=2)
adjusted_submission


adjusted_submission.to_csv("sub_4.csv", index=False)


## get  0.0465 for submit by this model only




