import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


main_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
main_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
extra_data = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')


extra_data = extra_data.rename(columns={'Personality': 'known_personality'})
key_cols = [col for col in extra_data.columns if col != 'known_personality']
extra_data = extra_data.drop_duplicates(subset=key_cols)

main_train = main_train.merge(extra_data, how='left')
main_test = main_test.merge(extra_data, how='left')


target_raw = main_train['Personality']
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target_raw)


def build_features(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['known_personality'] = df['known_personality'].fillna('unknown')
    df['null_personality_flag'] = df['known_personality'].eq('unknown').astype(int)
    df['stage_fear'] = df['stage_fear'].fillna('unknown')
    df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
    return df

X_all = build_features(main_train.drop(columns=['id', 'Personality']))
X_test = build_features(main_test.drop(columns=['id']))
test_ids = main_test['id']


def encode_and_fill(X_train, X_val):
    for col in X_train.select_dtypes(include='number').columns:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_val[col] = X_val[col].fillna(X_train[col].mean())

    cat_cols = X_train.select_dtypes(include='object').columns
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = ord_enc.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = ord_enc.transform(X_val[cat_cols])

    return X_train, X_val, ord_enc


model = RandomForestClassifier(
    n_estimators=344,
    max_depth=11,
    max_features=None,
    min_samples_split=11,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ensemble_models = []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, target_encoded)):
    print(f"---- Fold {fold + 1} ----")
    X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_tr, y_val = target_encoded[train_idx], target_encoded[val_idx]

    X_tr_enc, X_val_enc, final_encoder = encode_and_fill(X_tr.copy(), X_val.copy())
    model.fit(X_tr_enc, y_tr)

    preds_val = model.predict(X_val_enc)
    acc = accuracy_score(y_val, preds_val)
    print(f"Validation Accuracy: {acc:.4f}")

    ensemble_models.append(model)
    fold_accuracies.append(acc)

print(f"Average Accuracy Across Folds: {np.mean(fold_accuracies):.4f}")


def prepare_test_data(df, encoder):
    df = df.copy()
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = encoder.transform(df[cat_cols])
    return df

X_test_final = prepare_test_data(X_test.copy(), final_encoder)


probs_all_models = np.mean(
    [model.predict_proba(X_test_final) for model in ensemble_models],
    axis=0
)
final_preds = label_encoder.inverse_transform(np.argmax(probs_all_models, axis=1))


submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_preds})
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission file 'submission.csv' generated.")




