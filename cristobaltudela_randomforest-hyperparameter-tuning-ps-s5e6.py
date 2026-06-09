import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, make_scorer
from sklearn.pipeline import Pipeline
import time
import os
import warnings
import itertools
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


seed = 42


def mapk(y_true, y_pred, k=3):
    actual = [[label] for label in y_true]
    
    apk_values = []
    for a, p in zip(actual, y_pred):
        p = p[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if not a:
            apk_values.append(0.0)
        else:
            apk_values.append(score / min(len(a), k))
    
    return np.mean(apk_values)


def feature_engineering(df):
    df = df.copy()
    
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    
    # Convert categorical features
    for col in ['Soil Type', 'Crop Type']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Base numeric features list
    numeric_features = []
    
    # NPK related features
    if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
        npk_features = ['Nitrogen', 'Phosphorous', 'Potassium']
        numeric_features.extend(npk_features)
    
    # Temperature, humidity and moisture features
    if all(col in df.columns for col in ['Temparature', 'Humidity', 'Moisture']):
        climate_features = ['Temparature', 'Humidity', 'Moisture']
        numeric_features.extend(climate_features)
    
    # Categorical features processing
    if all(col in df.columns for col in ['Soil Type', 'Crop Type']):
        # Create numerical representations
        df['Soil_Code'] = df['Soil Type'].cat.codes
        df['Crop_Code'] = df['Crop Type'].cat.codes
        
        # Add these to numeric features list
        numeric_features.extend(['Soil_Code', 'Crop_Code'])
        
        # Combined categorical feature
        df['Combined_Cat'] = ''
        for cat_col in ['Soil Type', 'Crop Type']:
            df['Combined_Cat'] += df[cat_col].astype(str) + '_'
        df['Combined_Cat'] = LabelEncoder().fit_transform(df['Combined_Cat'])
        numeric_features.append('Combined_Cat')
    
    # Get all numeric columns that aren't already in our list
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if col not in numeric_features:
            numeric_features.append(col)
    
    # Automatic feature combinations using itertools
    print(f"Generating combinations of {len(numeric_features)} numeric features...")
    
    # To avoid explosion of features, limit combinations to pairs (2-way)
    for comb in itertools.combinations(numeric_features, 2):
        feature_name = f"{comb[0]}*{comb[1]}"
        # Skip if columns have same prefix to avoid redundancy
        if comb[0].split('_')[0] == comb[1].split('_')[0]:
            continue
        df[feature_name] = df[list(comb)].prod(axis=1)
    
    # Add squared terms for all features
    for feat in numeric_features:
        df[f"{feat}_squared"] = df[feat] ** 2
    
    # Print number of features created
    print(f"Total number of features after engineering: {df.shape[1]}")
    
    return df


print("Loading data...")
df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

test_ids = df_test['id'].values

# Guarda la variable target antes de aplicar ingeniería de características
target_column = 'Fertilizer Name'

print("Applying feature engineering...")
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

# Extrae la variable target después de la ingeniería
y_train = df_train.pop(target_column)

# Encode categorical features
for col in df_train.select_dtypes(include=['category']).columns:
    df_train[col] = df_train[col].cat.codes
    if col in df_test.columns:
        df_test[col] = df_test[col].cat.codes

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Get number of classes and fertilizer types
num_classes = len(label_encoder.classes_)
fertilizer_classes = label_encoder.classes_
print(f"Number of fertilizer classes: {num_classes}")
print(f"Fertilizer classes: {fertilizer_classes}")

# Variables globales para almacenar resultados
df_train_features = df_train  # Mantener referencia clara
df_test_features = df_test    # Mantener referencia clara


def map3_scorer(estimator, X, y):
    """Función custom para calcular MAP@3."""
    # Predecir probabilidades
    probs = estimator.predict_proba(X)
    
    # Convertir probabilidades a listas de top 3 predicciones
    pred_labels = []
    for pred_probs in probs:
        top_3_indices = np.argsort(pred_probs)[::-1][:3]
        pred_labels.append([fertilizer_classes[idx] for idx in top_3_indices])
    
    # Convertir etiquetas numéricas a nombres de fertilizantes
    true_labels = label_encoder.inverse_transform(y)
    
    # Calcular y devolver MAP@3
    return mapk(true_labels, pred_labels, k=3)


print("\nRunning hyperparameter search...")
tuning_start = time.time()

# Define parameter grid as arrays
n_estimators_array = [300, 350, 400, 450, 500, 550, 600]
max_depth_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
min_samples_split_array = [2, 3, 4, 5, 6, 8]
min_samples_leaf_array = [1, 2, 3, 4, 5]
max_features_array = ['sqrt', 'log2']

param_grid = {
    'n_estimators': n_estimators_array,
    'max_depth': max_depth_array,
    'min_samples_split': min_samples_split_array,
    'min_samples_leaf': min_samples_leaf_array,
    'max_features': max_features_array
}

# Create base model with fixed parameters
base_model = RandomForestClassifier(
    bootstrap=True,
    oob_score=True,
    class_weight='balanced',
    random_state=seed,
    n_jobs=1,  # Set to 1 since GridSearchCV will handle parallelism
    verbose=0
)

# Create a smaller subset for hyperparameter tuning (for speed)
print("Creating a subset of the data for faster hyperparameter tuning...")
from sklearn.model_selection import train_test_split
X_tune, _, y_tune, _ = train_test_split(df_train_features, y_train_encoded, 
                                        test_size=0.9,
                                        random_state=seed, 
                                        stratify=y_train_encoded)
print(f"Tuning subset size: {len(X_tune)} samples (5% de los datos originales)")

# Configure GridSearchCV
print("Configurando GridSearchCV con verbosidad reducida...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=map3_scorer,
    cv=10,  # 10-fold CV
    n_jobs=4,  # Fijo a 4 núcleos para GridSearchCV
    verbose=0,  # Sin verbosidad detallada
    return_train_score=True  # Para poder mostrar resultados por fold
)

# Run grid search
print("Running grid search...")
import time
start_time = time.time()

# Ejecutar la búsqueda
print("Buscando mejores hiperparámetros en silencio...")
grid_search.fit(X_tune, y_tune)

# Mostrar resumen por combinación de parámetros (agrupado por fold)
results = grid_search.cv_results_
total_time = time.time() - start_time

print(f"\nBusqueda de hiperparámetros completada en {total_time:.2f} segundos\n")


# Mostrar solo los mejores resultados por configuración
top_configs = 5
sorted_idx = np.argsort(results['mean_test_score'])[::-1][:top_configs]

print(f"Top {top_configs} configuraciones:")
print("=====================================================")
for i, idx in enumerate(sorted_idx):
    params = results['params'][idx]
    mean_score = results['mean_test_score'][idx]
    std_score = results['std_test_score'][idx]
    
    # Formato bonito para params
    params_str = ', '.join(f"{k}={v}" for k, v in params.items())
    
    print(f"Rank {i+1}: {params_str}")
    print(f"   MAP@3 Score: {mean_score:.4f} (±{std_score:.4f})")
    print("-----------------------------------------------------")

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest Parameters: {best_params}")
print(f"Best MAP@3 Score: {best_score:.6f}")
print(f"Hyperparameter tuning took {time.time() - tuning_start:.2f} seconds")


pd.DataFrame([best_params]).to_csv('/kaggle/working/best_params.csv', index=False)


best_n_estimators = best_params['n_estimators']
best_max_depth = best_params['max_depth']
best_min_samples_split = best_params['min_samples_split']
best_min_samples_leaf = best_params['min_samples_leaf']
best_max_features = best_params['max_features']

print("\nUsing best parameters for cross-validation...")

# Cross validation setup
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Inicialización de arrays para almacenar predicciones
oof_predictions = np.zeros((len(df_train_features), num_classes))
test_predictions = np.zeros((len(df_test_features), num_classes))

# Store metrics for analysis
fold_scores = []
fold_features = []
fold_times = []
fold_best_trees = []
fold_logloss = []


print(f"\nStarting training with {n_folds} folds...")
start_time = time.time()

for fold, (train_idx, valid_idx) in enumerate(kf.split(df_train_features, y_train_encoded)):
    fold_start = time.time()
    print(f"\n=== Fold {fold+1}/{n_folds} ===")
    X_tr = df_train_features.iloc[train_idx]
    y_tr = y_train_encoded[train_idx]
    X_val = df_train_features.iloc[valid_idx]
    y_val = y_train_encoded[valid_idx]
    
    print(f"Training set size: {len(X_tr)} rows")
    print(f"Validation set size: {len(X_val)} rows")
    
    # Configure Random Forest model with best parameters from hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        max_features=best_max_features,
        bootstrap=True,
        oob_score=True,
        class_weight='balanced',
        random_state=seed,
        n_jobs=4,
        verbose=0
    )
    
    print("Training model...")
    model.fit(X_tr, y_tr)
    
    # Get validation predictions
    val_preds_proba = model.predict_proba(X_val)
    oof_predictions[valid_idx] = val_preds_proba
    
    # Get test predictions for this fold
    test_preds_proba = model.predict_proba(df_test_features)
    test_predictions += test_preds_proba / n_folds
    
    # Calculate MAP@3
    val_pred_labels = []
    for pred_probs in val_preds_proba:
        top_3_indices = np.argsort(pred_probs)[::-1][:3]
        val_pred_labels.append([fertilizer_classes[idx] for idx in top_3_indices])
    
    map3_score = mapk(y_train.iloc[valid_idx].values, val_pred_labels, k=3)
    fold_scores.append(map3_score)
    
    # Calculate log loss
    ll_score = log_loss(y_val, val_preds_proba)
    fold_logloss.append(ll_score)
    
    # Record fold metrics
    fold_time = time.time() - fold_start
    fold_times.append(fold_time)
    fold_best_trees.append(model.n_estimators)
    
    print(f"MAP@3 for fold {fold+1}: {map3_score:.6f}")
    print(f"Log Loss for fold {fold+1}: {ll_score:.6f}")
    print(f"OOB score for fold {fold+1}: {model.oob_score_:.6f}")
    print(f"Fold training time: {fold_time:.2f} seconds")

total_time = time.time() - start_time

print("\n=== Final Results ===")
oof_pred_labels = []
for pred_probs in oof_predictions:
    top_3_indices = np.argsort(pred_probs)[::-1][:3]
    oof_pred_labels.append([fertilizer_classes[idx] for idx in top_3_indices])

map3_global = mapk(y_train.values, oof_pred_labels, k=3)
print(f"Global MAP@3 (OOF): {map3_global:.6f}")
print(f"Average MAP@3 per fold: {np.mean(fold_scores):.6f}")
print(f"MAP@3 standard deviation: {np.std(fold_scores):.6f}")
print(f"Average Log Loss: {np.mean(fold_logloss):.6f}")
print(f"Total training time: {total_time:.2f} seconds")


plt.figure(figsize=(10, 6))
plt.bar(range(1, n_folds+1), fold_scores, color='skyblue')
plt.axhline(y=map3_global, color='r', linestyle='--', label=f'Global MAP@3: {map3_global:.6f}')
plt.title('MAP@3 by Fold')
plt.xlabel('Fold Number')
plt.ylabel('MAP@3')
plt.xticks(range(1, n_folds+1))
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(10, 6))
plt.bar(range(1, n_folds+1), fold_logloss, color='salmon')
plt.axhline(y=np.mean(fold_logloss), color='b', linestyle='--', label=f'Average LogLoss: {np.mean(fold_logloss):.6f}')
plt.title('LogLoss by Fold')
plt.xlabel('Fold Number')
plt.ylabel('LogLoss')
plt.xticks(range(1, n_folds+1))
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(10, 6))
plt.bar(range(1, n_folds+1), fold_times, color='lightgreen')
plt.axhline(y=np.mean(fold_times), color='g', linestyle='--', label=f'Average time: {np.mean(fold_times):.2f}s')
plt.title('Training Time by Fold')
plt.xlabel('Fold Number')
plt.ylabel('Time (seconds)')
plt.xticks(range(1, n_folds+1))
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


print("\nTraining final model on all data for feature importance...")
final_model_start = time.time()

# Configure final Random Forest model with best parameters
final_model = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    max_features=best_max_features,
    bootstrap=True,
    oob_score=True,
    class_weight='balanced',
    random_state=seed,
    n_jobs=22,
    verbose=1
)

# Fit final model
final_model.fit(df_train_features, y_train_encoded)


feature_importance = pd.DataFrame({
    'Feature': df_train_features.columns,
    'Importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance from final model
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(30), palette='viridis')
plt.title('Feature Importance - Final Model')
plt.tight_layout()
plt.show()
plt.close()


final_model_time = time.time() - final_model_start
print(f"Final model OOB score: {final_model.oob_score_:.6f}")
print(f"Final model training time: {final_model_time:.2f} seconds")

# Export metrics to CSV
metrics_df = pd.DataFrame({
    'Fold': range(1, n_folds+1),
    'MAP@3': fold_scores,
    'LogLoss': fold_logloss,
    'Training_Time': fold_times,
    'Best_Trees': fold_best_trees
})
metrics_df.loc[len(metrics_df)] = [n_folds+1, map3_global, np.mean(fold_logloss), total_time, np.nan]
metrics_df.iloc[-1, 0] = 'Overall'
metrics_df.to_csv('/kaggle/working/rf_metrics.csv', index=False)


print("\nPreparing submission file...")
submission = pd.DataFrame({'id': test_ids})
test_pred_labels = []
for pred_probs in test_predictions:
    top_3_indices = np.argsort(pred_probs)[::-1][:3]
    test_pred_labels.append(' '.join([fertilizer_classes[idx] for idx in top_3_indices]))

submission['Fertilizer Name'] = test_pred_labels

submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")

print("\nProcess completed successfully!")

