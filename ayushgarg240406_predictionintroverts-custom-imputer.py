%config IPCompleter.greedy=True


import numpy as np
import pandas as pd


df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')



print(df_train.head())
print(df_test.head())


print(df_train.info())


print(df_train.describe())


for df in [df_train,df_test] :
    df.columns = df.columns.str.strip().str.lower()


print(df_train.columns)


cat_types = ['stage_fear','drained_after_socializing']
for column in df_train.columns :
    print(f" Describing feature {column}")
    print(df_train[column].unique())
    print(df_train[column].value_counts())


print(df_train['personality'].isnull().sum())


from sklearn.preprocessing import OrdinalEncoder
label_types = ['stage_fear','drained_after_socializing']
lbl_encode = OrdinalEncoder()
df_train[label_types] = lbl_encode.fit_transform(df_train[label_types])



label_test_types = ['stage_fear','drained_after_socializing']
df_test[label_test_types] = lbl_encode.transform(df_test[label_test_types])



from sklearn.preprocessing import LabelEncoder

y_encode = LabelEncoder()
df_train['personality'] = y_encode.fit_transform(df_train['personality'])



print(df_train.shape)


print(df_train.corr())


df_train.drop(columns='id',inplace=True)


df_test.drop(columns='id',inplace=True)


#custom Imputer


def get_unique_quantile(val, series):
    unique_vals = sorted(set(series.dropna()))
    below = [u for u in unique_vals if u < val]
    return len(below) / len(unique_vals) if unique_vals else 0.5


def smart_ordered_imputer(df, num_features, top_k=3):
    df_filled = df.copy()
    corr_matrix = df[num_features].corr() 

 
    top_corr = {
        col: sorted(
            [(other, corr_matrix[col][other]) for other in corr_matrix.columns if other != col],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        for col in num_features
    }

    for col in num_features:
        correlated_features = top_corr[col]

        for idx, row in df.iterrows():
            if pd.isna(row[col]):
                fill_val = None

                for c, corr_val in correlated_features:
                    val = row[c]
                    if not pd.isna(val):
                        ref_series = df[c].dropna()
                        if not ref_series.empty:
                            q = get_unique_quantile(val, ref_series)

                        
                            if corr_val < 0:
                                q = 1 - q

                            q = np.clip(q, 0.01, 0.99)
                            target_series = df[col].dropna()
                            if not target_series.empty:
                                fill_val = target_series.quantile(q)
                                break  

                if fill_val is None:
                    fill_val = df[col].median()

                df_filled.at[idx, col] = fill_val

    return df_filled



numerical_cols = ['time_spent_alone', 'stage_fear', 'social_event_attendance',
       'going_outside', 'drained_after_socializing', 'friends_circle_size',
       'post_frequency']
df_filled = smart_ordered_imputer(df_train, numerical_cols , top_k=3) 





for column in df_filled.columns :
    print(f" NaN values in column {column} :")
    print(df_filled[column].isnull().sum())


df_train = df_filled


print(df_train.shape)


print(df_train.head())


print(df_train.describe())


X = df_train.drop(columns='personality')
y = df_train['personality']


from sklearn.model_selection import StratifiedKFold


print(df_train.corr())


df_test = smart_ordered_imputer(df_test, numerical_cols)


print(df_test.head())


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
logreg = LogisticRegression(max_iter = 1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(logreg, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


df_sub = pd.read_csv(r"/kaggle/input/playground-series-s5e7/sample_submission.csv")


logreg.fit(X,y)
preds = logreg.predict(df_test)
preds_labels = y_encode.inverse_transform(preds)
submission_df = pd.DataFrame({
    'id': df_sub['id'],
    'Personality': preds_labels
})
from joblib import dump, load
dump(logreg, 'logReg_new.joblib')



submission_df.to_csv("/kaggle/working/submission_new_log.csv", index=False)


import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
def objective(trial):

    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # Pipeline with imputation and RandomForest
    rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

    return scores.mean()

# --- 3. Run Optuna Study ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # You can increase to 50–100 for deeper search

print("✅ Best CV Accuracy: {:.4f}".format(study.best_value))
print("✅ Best Params:", study.best_params)




best_params = study.best_params
rf_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
rf_model.fit(X, y)



from joblib import dump, load
dump(rf_model, 'rf_model.joblib')


# Predict on test set
y_test_preds = rf_model.predict(df_test)

# Decode back to original labels
y_test_labels = y_encode.inverse_transform(y_test_preds)

# Create submission file
submission_df = pd.DataFrame({
    'id': df_sub['id'],
    'Personality': y_test_labels
})

submission_df.to_csv("/kaggle/working/submission_rf.csv", index=False)
print("✅ submission.csv created using Optuna-optimized RandomForest.")



import xgboost as xgb


from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(accuracy_score)).mean()

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)


# Use best parameters from Optuna
best_params = study.best_params
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'

# Train the model on the full data
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y)




final_model.save_model("xgbModel_new.json")  


preds = final_model.predict(df_test)
preds_labels = y_encode.inverse_transform(preds)
submission_df = pd.DataFrame({
    'id': df_sub['id'],
    'Personality': preds_labels
})


submission_df.to_csv("/kaggle/working/submission_new_xgb.csv", index=False)

print("✅ Kaggle submission file saved as submission.csv")




