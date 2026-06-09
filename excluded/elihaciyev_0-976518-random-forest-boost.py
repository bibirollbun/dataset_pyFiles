# *** IMPORTS ***
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

def preprocess_fold(X_train, X_val):
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

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])
    print(f"[INFO] Fold preprocessing complete. Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, encoder

def preprocess_final_test(X_test, encoder):
    df = X_test.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['stage_fear'] = df['stage_fear'].fillna('unknown')
    df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
    df['match_p_is_null'] = df['match_p'].isna().astype(int)
    df['match_p'] = df['match_p'].fillna('unknown')
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = encoder.transform(df[cat_cols])
    print(f"[INFO] Test preprocessing complete. Shape: {df.shape}")
    return df

# *** MODELLING ***
def fit_cv_model(X, y, model, n_splits=5):
    models, scores = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n[INFO] Fold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_prep, X_val_prep, encoder = preprocess_fold(X_train, X_val)
        model.fit(X_train_prep, y_train)
        acc = accuracy_score(y_val, model.predict(X_val_prep))
        scores.append(acc), models.append(model)
        print(f"[INFO] Accuracy: {acc:.6f}")

    print("\n[INFO] Final CV Accuracy:", round(np.mean(scores), 6))
    return models, encoder

# *** SUBMISSION ***
def make_submission(models, X_test, target_encoder, test_ids, file='submission.csv'):
    importances = np.mean([m.feature_importances_ for m in models if hasattr(m, 'feature_importances_')], axis=0)
    submission = pd.DataFrame({'id': test_ids,
        'personality': target_encoder.inverse_transform(np.argmax(
            sum(m.predict_proba(X_test) for m in models) / len(models), axis=1))})
    submission.to_csv(file, index=False)
    print(f"[INFO] Submission saved as '{file}':"); display(submission.head())
    feat = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances
                        }).sort_values('Importance', ascending=False)
    print("[INFO] Feature Importances:"); display(feat.head(10))

# *** RUN PIPELINE ***
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
X, test, y, target_encoder = preprocessing(original, train, test)

rf = RandomForestClassifier(n_estimators=344, max_depth=11, max_features=None,
    min_samples_split=11, min_samples_leaf=1, random_state=42, n_jobs=-1 )

print('\n' + '#' * 20 + ' MODELLING ' + '#' * 20)
models, encoder = fit_cv_model(X, y, rf)
X_test = preprocess_final_test(test, encoder)

print('\n' + '#' * 20 + ' SUBMISSION ' + '#' * 20)
make_submission(models, X_test, target_encoder, test['id'])




