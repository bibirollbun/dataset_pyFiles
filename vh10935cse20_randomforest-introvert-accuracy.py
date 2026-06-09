import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
org=pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')


org = org.rename(columns={'Personality': 'match_p'})
drop_cols = [col for col in org.columns if col != 'match_p']
org = org.drop_duplicates(subset=drop_cols)


train = train.merge(org, how='left')
test = test.merge(org, how='left')


train.head(3)


from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


y=train['Personality']
x=train.drop(columns=['id','Personality'])


X_test_final = test.drop(columns=['id']).copy()
test_ids = test['id']

# --- Feature Engineering applied globally to x and X_test_final ---
for df_ref in [x, X_test_final]:
    df_ref.columns = df_ref.columns.str.lower().str.replace(' ', '_')
    df_ref['match_p_is_null'] = df_ref['match_p'].isna().astype(int)
    df_ref['stage_fear'] = df_ref['stage_fear'].fillna('unknown')
    df_ref['drained_after_socializing'] = df_ref['drained_after_socializing'].fillna('unknown')
    df_ref['match_p'] = df_ref['match_p'].fillna('unknown')


target_encoder=LabelEncoder()
y=pd.Series(target_encoder.fit_transform(y)).values.ravel()


def preprocess(X_train,X_val):
    for df in [X_train,X_val]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df['stage_fear']=df['stage_fear'].fillna('unknown')
        df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
        df['match_p'] = df['match_p'].fillna('unknown')
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].mean())
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])
    return X_train, X_val, encoder


rf = RandomForestClassifier(n_estimators=344, max_depth=11, max_features=None,
                            min_samples_split=11, min_samples_leaf=1,
                            random_state=42, n_jobs=-1)


models, scores = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    X_train_prep, X_val_prep, encoder = preprocess(X_train.copy(), X_val.copy())
    rf.fit(X_train_prep, y_train)
    acc = accuracy_score(y_val, rf.predict(X_val_prep))
    print(f"[INFO] Accuracy: {acc:.6f}")
    models.append(rf)
    scores.append(acc)

print("\n[INFO] Mean CV Accuracy:", np.mean(scores))


def preprocess_final_test(df, encoder):
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
    return df


X_test = preprocess_final_test(test.copy(), encoder)

# Prediction
probas = sum(model.predict_proba(X_test) for model in models) / len(models)
preds = target_encoder.inverse_transform(np.argmax(probas, axis=1))

# Submission
submission = pd.DataFrame({'id': test['id'], 'Personality': preds})
submission.to_csv('submission.csv', index=False)
print("[INFO] Submission saved to 'submission.csv'")




