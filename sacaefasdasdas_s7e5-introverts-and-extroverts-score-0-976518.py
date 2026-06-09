# *** IMPORTS ***
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


# *** PREPROCESSING ***
def preprocessing(original, train, test):
    print("\n[INFO] Starting preprocessing...")
    df_original = original.rename(columns={'Personality': 'match_p'})
    drop_cols = [col for col in df_original.columns if col != 'match_p']
    df_original = df_original.drop_duplicates(subset=drop_cols)

    print(f"Original train shape: {train.shape}, test shape: {test.shape}")
    train = train.merge(df_original, how='left')
    test = test.merge(df_original, how='left')
    print(f"Merged train shape: {train.shape}, test shape: {test.shape}")

    X = train.drop(columns=['Personality'])
    y = train['Personality'] 

    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(y))
    print("[INFO] Label encoding completed. Classes:", target_encoder.classes_)
    return X, test, y, target_encoder 



def preprocess_fold_catboost(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    for df in [X_train, X_val]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.drop(columns=['id'], inplace=True, errors='ignore')
        df['stage_fear'] = df['stage_fear'].fillna('unknown')
        df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
        df['match_p_is_null'] = df['match_p'].isna().astype(int)
        df['match_p'] = df['match_p'].fillna('unknown')
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].mean())

    
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    return X_train, X_val, cat_cols


def preprocess_final_test_catboost(X_test, cat_cols):
    df = X_test.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['stage_fear'] = df['stage_fear'].fillna('unknown')
    df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
    df['match_p_is_null'] = df['match_p'].isna().astype(int)
    df['match_p'] = df['match_p'].fillna('unknown')
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    
    return df, cat_cols



def fit_cv_model_catboost(X, y, model_params, n_splits=5):
    models, scores = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n[INFO] Fold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_prep, X_val_prep, cat_cols = preprocess_fold_catboost(X_train, X_val)

        model = CatBoostClassifier(**model_params)
        model.fit(X_train_prep, y_train, cat_features=cat_cols, verbose=100)

        preds = model.predict(X_val_prep)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
        models.append(model)
        print(f"[INFO] Accuracy: {acc:.6f}")

    print("\n[INFO] Final CV Accuracy:", round(np.mean(scores), 6))
    return models, cat_cols


def make_submission_catboost(models, X_test, cat_cols, target_encoder, test_ids, file='submission.csv'):
    preds_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
    preds = target_encoder.inverse_transform(np.argmax(preds_proba, axis=1))

    submission = pd.DataFrame({'id': test_ids, 'personality': preds})
    submission.to_csv(file, index=False)
    print(f"[INFO] Submission saved as '{file}':")
    display(submission.head())

    # Feature importances
    importances = np.mean([m.get_feature_importance() for m in models], axis=0)
    feat = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances
                        }).sort_values('Importance', ascending=False)
    print("[INFO] Feature Importances:")
    display(feat.head(10))



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')

X, test, y, target_encoder = preprocessing(original, train, test)

catboost_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 100
}

print('\n' + '#' * 20 + ' MODELLING ' + '#' * 20)
models, cat_cols = fit_cv_model_catboost(X, y, catboost_params)

X_test, cat_cols = preprocess_final_test_catboost(test, cat_cols)

print('\n' + '#' * 20 + ' SUBMISSION ' + '#' * 20)
make_submission_catboost(models, X_test, cat_cols, target_encoder, test['id'])

