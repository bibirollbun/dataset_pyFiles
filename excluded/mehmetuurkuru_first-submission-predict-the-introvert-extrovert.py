import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import optuna


# Eğitim verisi yükleme ve ön işleme
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_train.drop(['id'], axis=1, inplace=True)

y_train = df_train['Personality']
X_train = df_train.drop('Personality', axis=1)


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


numerical_features_train = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_train = X_train.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features_train),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_train)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('imputer', KNNImputer(n_neighbors=5))])

X_train_imputed_array = pipeline.fit_transform(X_train)


# all_feature_names_train
encoded_feature_names_train = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features_train)
all_feature_names_train = numerical_features_train + list(encoded_feature_names_train)

X_train_processed = pd.DataFrame(X_train_imputed_array, columns=all_feature_names_train, index=X_train.index)

print("--- Train (X_train_processed) Head ---")
print(X_train_processed.head())
print("\n--- Train (X_train_processed) Isnull? ---")
print(X_train_processed.isnull().sum())
print("\n--- Train (X_train_processed) Info ---")
X_train_processed.info()


print("\n\n--- Test Preprocessing ---")

df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_test_ids = df_test['id']
df_test.drop(['id'], axis=1, inplace=True)

X_test_imputed_array = pipeline.transform(df_test)


X_test_processed = pd.DataFrame(X_test_imputed_array, columns=all_feature_names_train, index=df_test.index)

print("\n--- DataFrame Head ---")
print(X_test_processed.head())
print("\n--- Isnull (X_test) ---")
print(X_test_processed.isnull().sum())
print("\n--- Info ---")
X_test_processed.info()

print("\nTest preprocessing is completed.")



print("\n\n--- Modeling and OPTUNA are starting ---")

def objective(trial, model_name):

    early_stopping_rounds = 50 

    if model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 5.0, log=True),
            'random_seed': 42,
            'verbose': 0,
            'eval_metric': 'Accuracy'
        }
        model = CatBoostClassifier(**params)
    else: # This 'else' will now catch any model_name that isn't 'CatBoost'
        raise ValueError(f"Unknown model: {model_name}. Only 'CatBoost' is supported in this objective function.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train_encoded)):
        X_train_fold, X_val_fold = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_encoded[train_idx], y_train_encoded[val_idx]

        if model_name == 'CatBoost':
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      early_stopping_rounds=early_stopping_rounds,
                      verbose=0)
       
        y_val_pred = model.predict(X_val_fold)
        accuracies.append(accuracy_score(y_val_fold, y_val_pred))

    return np.mean(accuracies)


best_params_found = {}
best_trained_models = {}

model_names_to_tune = ['CatBoost']

for model_name in model_names_to_tune:
    print(f"\n--- {model_name} için Optuna Optimizasyonu ---")
    study = optuna.create_study(direction='maximize', study_name=f'{model_name}_optimization')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=50)

    print(f"\n{model_name} the best trial:")
    print(f"  Best_value: {study.best_value:.4f}")
    print(f"  Best_params: {study.best_params}")
    
    best_params_found[model_name] = study.best_params

    print(f"\n{model_name} Training model with best params...")

    X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
        X_train_processed, y_train_encoded, test_size=0.10, random_state=42, stratify=y_train_encoded
    )
    
    early_stopping_rounds_final = 75

    if model_name == 'CatBoost':
        final_model = CatBoostClassifier(**study.best_params, random_state=42, verbose=0, eval_metric='Accuracy')
        final_model.fit(X_final_train, y_final_train,
                        eval_set=[(X_final_val, y_final_val)],
                        early_stopping_rounds=early_stopping_rounds_final,
                        verbose=0)
   
    best_trained_models[model_name] = final_model

print("\nModeling and Optimization is completed.")


print("\n\n--- Prediction ---")

for name, model in best_trained_models.items():
    print(f"\n--- {name} Predictions ---")
    predictions_encoded = model.predict(X_test_processed)
    predictions_original = le.inverse_transform(predictions_encoded)

    submission_df = pd.DataFrame({
        'id': df_test_ids,
        'Personality': predictions_original
    })

    output_filename = f'submission_{name.lower()}_optimized_es.csv' 
    submission_df.to_csv(output_filename, index=False)
    print(f"Predictions saved in '{output_filename}")

