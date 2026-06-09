## ğŸ“¦ Imports, seed setting, logging, TensorBoard setup, etc.

# âš™ï¸� Core
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ğŸ§  TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ğŸ§ª Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

from sklearn.model_selection import RandomizedSearchCV

# ğŸ”¥ XGBoost
#from xgboost.callback import EarlyStopping

# ğŸ�± CatBoost
from catboost import CatBoostRegressor, Pool



# Criando o train_df
train_df = pd.read_csv("/kaggle/input/predagedatasets/Train.csv")
train_df.head()


# Criando o test_df
test_df = pd.read_csv("/kaggle/input/predagedatasets/Test.csv")
test_df.head()


# =============================================
# ğŸ“Š Exploratory Data Analysis (EDA)
# =============================================

target_col = "Age (years)"

# 1. InformaÃ§Ãµes gerais
print("ğŸ”� InformaÃ§Ãµes bÃ¡sicas:")
train_df.info()
print("\nğŸ“ˆ EstatÃ­sticas descritivas:")
display(train_df.describe())

# 2. Verifica valores ausentes
print("\nğŸ”� Valores ausentes por coluna:")
display(train_df.isnull().sum().sort_values(ascending=False))

# 3. DistribuiÃ§Ã£o da variÃ¡vel alvo (idade)
plt.figure(figsize=(8, 4))
sns.histplot(train_df[target_col], bins=40, kde=True, color='dodgerblue')
plt.title("DistribuiÃ§Ã£o da idade (variÃ¡vel alvo)")
plt.xlabel("Idade (anos)")
plt.ylabel("FrequÃªncia")
plt.grid(True)
plt.show()

# 4. CorrelaÃ§Ãµes com a variÃ¡vel alvo
correlations = train_df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
print("\nğŸ“Š CorrelaÃ§Ãµes com a variÃ¡vel 'Age (years)':")
display(correlations)

# 5. Heatmap com as top correlaÃ§Ãµes
top_corr = train_df[correlations.head(10).index]
plt.figure(figsize=(10, 6))
sns.heatmap(top_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("ğŸ”— CorrelaÃ§Ã£o entre variÃ¡veis mais relacionadas Ã  idade")
plt.show()


# Checking object Columns:

object_cols = train_df.select_dtypes(include='object').columns

# Unique Values
for col in object_cols:
    print(f"\nğŸ“Œ Coluna: {col}")
    print(f"Valores Ãºnicos ({train_df[col].nunique()}):")
    print(train_df[col].unique())


# 3.1 Missing Value Imputation
categorical_cols = [
    "Family History",
    "Chronic Diseases",
    "Alcohol Consumption",
    "Medication Use",
    "Education Level"
]

for col in categorical_cols:
    train_df[col] = train_df[col].fillna("Missing")
    test_df[col] = test_df[col].fillna("Missing")


# 3.2 Blood Pressure Split
bp_split = train_df["Blood Pressure (s/d)"].str.split("/", expand=True)
train_df["Systolic BP"] = pd.to_numeric(bp_split[0], errors="coerce")
train_df["Diastolic BP"] = pd.to_numeric(bp_split[1], errors="coerce")
train_df = train_df.drop(columns=["Blood Pressure (s/d)"])


# Adjusting test
bp_split_test = test_df["Blood Pressure (s/d)"].str.split("/", expand=True)
test_df["Systolic BP"] = pd.to_numeric(bp_split_test[0], errors="coerce")
test_df["Diastolic BP"] = pd.to_numeric(bp_split_test[1], errors="coerce")
test_df.drop(columns=["Blood Pressure (s/d)"], inplace=True)



# ============================================================
# ğŸ”§ 4.1 Define Categorical and Numerical Columns
#.       One-Hot Encode categorical, standardize numerical
# ============================================================
target_col = "Age (years)"
categorical_cols = [
    "Gender",
    "Family History",
    "Chronic Diseases",
    "Alcohol Consumption",
    "Medication Use",
    "Education Level"
]


numerical_cols = [col for col in train_df.select_dtypes(include=np.number).columns if col != target_col]

# ================================
# ğŸ§¼ Pipeline
# ================================

# Pipeline (z-score)
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Pipeline categorical
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Categorical + Numeric
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# ================================
# ğŸ§ª features and target
# ================================
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# 4.2 Train/Validation Split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# ğŸš€ Pre Processing
# ================================
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_val_processed = preprocessor.transform(X_val_raw)

print("âœ… X_train shape:", X_train_processed.shape)
print("âœ… X_val shape:", X_val_processed.shape)


# Aplica a transformaÃ§Ã£o no X_train_raw
X_train_processed = preprocessor.transform(X_train_raw)


# Pega os nomes das colunas finais
num_features = numerical_cols
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_features = list(num_features) + list(cat_features)

# Converte para DataFrame
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_features, index=X_train_raw.index)

# Visualiza
print("âœ… Dados transformados:")
display(X_train_processed_df.head())


# Escolhe uma coluna numÃ©rica e uma categÃ³rica
original_cols = ['BMI', 'Gender']

# Mostra valores originais
print("ğŸ”� Originais:")
display(X_train_raw[original_cols].head())

# Mostra valores transformados correspondentes
print("ğŸ”� Transformados:")
display(X_train_processed_df[[col for col in all_features if 'BMI' in col or 'Gender' in col]].head())


# Checking the results

print(type(X_train_processed))  # <class 'numpy.ndarray'>
print(X_train_processed.shape)
print(X_train_processed[:5])    # Primeiras 5 linhas


# ================================================
# ğŸš€ 4.3 Prepare tf.data.Dataset
#    Convert NumPy arrays to TensorFlow
#    pipeline with batching and prefetching
# ================================================
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_processed, y_train.values))
train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_processed, y_val.values))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =====================================
# Round 1: Simple DNN (no callbacks)
# =====================================

config = {
    "dropout": 0.2,
    "l2": 0.0,
    "num_units": [128, 64],
    "epochs": 100,
    "batch_size": 32
}

model = Sequential([
    Dense(config['num_units'][0], activation='relu', input_shape=(X_train_processed.shape[1],)),
    BatchNormalization(),
    Dropout(config['dropout']),
    
    Dense(config['num_units'][1], activation='relu'),
    BatchNormalization(),
    Dropout(config['dropout'] / 2),
    
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# === No callbacks ===
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config['epochs'],
    verbose=1  # keeps logging per epoch
)

metrics = {
    'round': 'DNN-1',
    'train_loss': history.history['loss'][-1],
    'val_loss': history.history['val_loss'][-1],
    'train_mae': history.history['mae'][-1],
    'val_mae': history.history['val_mae'][-1],
    'train_mse': history.history['loss'][-1] ** 2,
    'val_mse': history.history['val_loss'][-1] ** 2
}

log_file = Path('metrics_log.csv')
if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("âœ… Metrics saved in metrics_log.csv")





# ========================================
# ğŸ§  Round 2
# ========================================

# ========================================
# ConfiguraÃ§Ãµes do modelo
# ========================================
config = {
    "dropout": 0.2,
    "l2": 0.0,
    "num_units": [256, 128, 64, 32],
    "epochs": 100,
    "batch_size": 32
}

# ========================================
# DefiniÃ§Ã£o do modelo
# ========================================
model = Sequential([
    Dense(config['num_units'][0], activation='relu', input_shape=(X_train_processed.shape[1],)),
    BatchNormalization(),
    Dropout(config['dropout']),
    
    Dense(config['num_units'][1], activation='relu'),
    BatchNormalization(),
    Dropout(config['dropout']),
    
    Dense(config['num_units'][2], activation='relu'),
    BatchNormalization(),
    Dropout(config['dropout'] / 2),
    
    Dense(config['num_units'][3], activation='relu'),
    BatchNormalization(),
    Dropout(config['dropout'] / 4),
    
    Dense(1)
])

# ========================================
# Compila o modelo
# ========================================
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# ========================================
# Callbacks
# ========================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# ========================================
# Treinamento
# ========================================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config['epochs'],
    callbacks=[early_stop, reduce_lr]
)

# ========================================
# Log manual em CSV (Ãºltima epoch)
# ========================================
metrics = {
    'round': 'DNN-2',
    'train_loss': history.history['loss'][-1],
    'val_loss': history.history['val_loss'][-1],
    'train_mae': history.history['mae'][-1],
    'val_mae': history.history['val_mae'][-1]
}
# Including MSE, competition metric
metrics['train_mse'] = metrics['train_loss'] ** 2
metrics['val_mse'] = metrics['val_loss'] ** 2

#
log_file = Path('metrics_log.csv')

if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)

print("âœ… MÃ©tricas do Round 2 salvas em metrics_log.csv")





import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from pathlib import Path

# ========================================
# ConfiguraÃ§Ãµes
# ========================================
config = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# ========================================
# XGBoost Model
# ========================================
model = xgb.XGBRegressor(
    max_depth=config['max_depth'],
    learning_rate=config['learning_rate'],
    n_estimators=config['n_estimators'],
    subsample=config['subsample'],
    colsample_bytree=config['colsample_bytree'],
    objective='reg:squarederror',
    tree_method='hist'  # para rodar mais rÃ¡pido
)

# ========================================
# Fit
# ========================================
model.fit(X_train_processed, y_train)

# ========================================
# Predict
# ========================================
y_pred_train = model.predict(X_train_processed)
y_pred_val = model.predict(X_val_processed)

# ========================================
# Metrics
# ========================================
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"Train MAE: {mae_train:.4f}")
print(f"Validation MAE: {mae_val:.4f}")
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Validation RMSE: {rmse_val:.4f}")

# ========================================
# Log manual em CSV
# ========================================
metrics = {
    'round': 'XGB-Basic',
    'train_mae': mae_train,
    'val_mae': mae_val,
    'train_loss': rmse_train,  # usando RMSE como loss (para manter compatÃ­vel com colunas)
    'val_loss': rmse_val
}
# Adiciona MSE baseado no RMSE (val_loss = RMSE)
metrics['train_mse'] = metrics['train_loss'] ** 2
metrics['val_mse'] = metrics['val_loss'] ** 2

log_file = Path('metrics_log.csv')

if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]  # remove duplicadas da mesma rodada
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)

print("âœ… MÃ©tricas do Round 4 salvas/atualizadas no metrics_log.csv")




import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from pathlib import Path

# ========================================
# ConfiguraÃ§Ãµes
# ========================================
config = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.9,
    "colsample_bytree": 0.9
}

# ========================================
# DMatrix
# ========================================
dtrain = xgb.DMatrix(X_train_processed, label=y_train)
dval = xgb.DMatrix(X_val_processed, label=y_val)

# ========================================
# Params
# ========================================
params = {
    'max_depth': config['max_depth'],
    'learning_rate': config['learning_rate'],
    'objective': 'reg:squarederror',
    'subsample': config['subsample'],
    'colsample_bytree': config['colsample_bytree'],
    'eval_metric': 'mae'
}

# ========================================
# Treino com early stopping
# ========================================
model = xgb.train(
    params,
    dtrain,
    num_boost_round=config['n_estimators'],
    evals=[(dval, 'validation')],
    early_stopping_rounds=20,
    verbose_eval=True
)

# ========================================
# Predict
# ========================================
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)

# ========================================
# Metrics
# ========================================
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"Train MAE: {mae_train:.4f}")
print(f"Validation MAE: {mae_val:.4f}")
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Validation RMSE: {rmse_val:.4f}")
print(f"Best iteration: {model.best_iteration}")

# ========================================
# Log manual em CSV
# ========================================
metrics = {
    'round': 'XGB-tuned',
    'train_mae': mae_train,
    'val_mae': mae_val,
    'train_loss': rmse_train,  # usa rmse no campo loss (para compatibilidade com grÃ¡fico)
    'val_loss': rmse_val,
    'best_iteration': model.best_iteration
}

# Adiciona MSE baseado no RMSE (val_loss = RMSE)
metrics['train_mse'] = metrics['train_loss'] ** 2
metrics['val_mse'] = metrics['val_loss'] ** 2
log_file = Path('metrics_log.csv')

if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]  # remove duplicada
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)

print("âœ… MÃ©tricas do Round 5 salvas/atualizadas no metrics_log.csv")




# ----------------------------------
# FunÃ§Ã£o de mÃ©trica customizada (MSE)
# ----------------------------------
def mse_scorer(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

mse_scorer_sklearn = make_scorer(mse_scorer, greater_is_better=False)

# ----------------------------------
# ParÃ¢metros para tuning
# ----------------------------------
param_grid = {
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# ----------------------------------
# Modelo base
# ----------------------------------
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist')

# ----------------------------------
# Randomized SearchCV
# ----------------------------------
random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_grid,
    n_iter=20,
    scoring=mse_scorer_sklearn,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# ----------------------------------
# Treino
# ----------------------------------
random_search.fit(X_train_processed, y_train)

# ----------------------------------
# Melhor modelo â†’ salvando como model_rand
# ----------------------------------
model_rand = random_search.best_estimator_
print("\nâœ… Melhor configuraÃ§Ã£o encontrada:")
print(random_search.best_params_)

print("\nğŸ“‰ Melhor MSE (negativo):")
print(random_search.best_score_)

# ----------------------------------
# AvaliaÃ§Ã£o no conjunto de validaÃ§Ã£o
# ----------------------------------
y_pred_val = model_rand.predict(X_val_processed)
val_mse = mean_squared_error(y_val, y_pred_val)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_pred_val)

print(f"\nğŸ“Š ValidaÃ§Ã£o: MSE={val_mse:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")

# ----------------------------------
# Log em CSV
# ----------------------------------
metrics = {
    'round': 'XGB-Random',
    'train_loss': abs(random_search.best_score_),
    'val_loss': val_rmse,
    'train_mae': np.nan,
    'val_mae': val_mae,
    'train_mse': np.nan,
    'val_mse': val_mse,
    'best_iteration': getattr(model_rand, 'best_iteration', np.nan)
}

log_file = Path("metrics_log.csv")

if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]  # remove duplicada
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("âœ… MÃ©tricas salvas em metrics_log.csv")







# ===================================================
# ğŸ“� GeraÃ§Ã£o do arquivo de submission CORRIGIDO
# ===================================================

# ğŸ”� Seleciona apenas as colunas de features do test_df
X_test_raw = test_df[X_train_raw.columns]  # garante mesmo conjunto de colunas do treino

# ğŸŒ€ Aplica o mesmo preprocessor usado no treino
X_test_processed = preprocessor.transform(X_test_raw)

print(f"âœ… X_test_processed gerado com shape: {X_test_processed.shape}")

# ğŸ‘‰ PrevisÃ£o no conjunto de teste
y_pred_test = model_rand.predict(X_test_processed)

# ğŸ‘‰ Cria DataFrame de submission diretamente dos IDs do test_df
submission_final = pd.DataFrame({
    'ID': test_df['ID'],            # pega os IDs diretamente do test_df
    'Age (years)': y_pred_test      # insere as previsÃµes
})

# ğŸ‘‰ Salva o CSV final
submission_final.to_csv('submission_final.csv', index=False)
print("âœ… Arquivo 'submission_final.csv' gerado com", submission_final.shape[0], "linhas e pronto para upload no Kaggle!")



# ===============================
# Optuna implementation
# ===============================
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import pandas as pd
import numpy as np

# Prepare DMatrix
dtrain = xgb.DMatrix(X_train_processed, label=y_train)
dval = xgb.DMatrix(X_val_processed, label=y_val)

# Objective function for Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 2.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist'
    }
    
    evals_result = {}
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int('n_estimators', 100, 500),
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    preds = bst.predict(dval)
    mse = mean_squared_error(y_val, preds)
    
    return mse

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Show best params
print("\nâœ… Best hyperparameters found:")
print(study.best_trial.params)
print(f"\nğŸ“‰ Best validation MSE from Optuna: {study.best_value:.4f}")

# Retrain final model with best params
best_params = study.best_trial.params.copy()
best_num_round = best_params.pop('n_estimators')

best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist'
})

best_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_num_round,
    evals=[(dval, 'validation')],
    early_stopping_rounds=20,
    verbose_eval=False
)

# Predict on validation set
y_pred_val = best_model.predict(dval)

# Metrics
val_mse = mean_squared_error(y_val, y_pred_val)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_pred_val)

print(f"\nğŸ“Š Final Validation: MSE={val_mse:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")

# Save metrics to CSV
metrics = {
    'round': 'XGB-OPTUNA-DMATRIX',
    'train_loss': np.nan,
    'val_loss': val_rmse,
    'train_mae': np.nan,
    'val_mae': val_mae,
    'train_mse': np.nan,
    'val_mse': val_mse,
    'best_iteration': best_model.best_iteration
}

log_file = Path('metrics_log.csv')

if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("âœ… Metrics logged in metrics_log.csv")

# Save model for future ensemble
xgb_model_optuna = best_model





# =========================================================
# Criar Pool
# ==========================================================
train_pool = Pool(data=X_train_processed, label=y_train)
val_pool = Pool(data=X_val_processed, label=y_val)

# 4ï¸�âƒ£ Definir modelo
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=50,
    verbose=100,
    random_seed=42
)

# 5ï¸�âƒ£ Treinar
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# 6ï¸�âƒ£ PrevisÃ£o
y_pred = model.predict(X_val_processed)

# 7ï¸�âƒ£ MÃ©tricas
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f'\nğŸ“Š Resultados CatBoost (sem novas features):')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# ğŸ”Ÿ Feature importance
import matplotlib.pyplot as plt
import seaborn as sns

# âœ… CORREÃ‡ÃƒO AQUI
feature_names = preprocessor.get_feature_names_out()  # <-- pega os nomes reais das features!

# Obter importÃ¢ncia
importances = model.get_feature_importance()

# Criar DataFrame
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Mostrar no print
print(feat_imp_df)

# Plot manual
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - CatBoost')
plt.show()

# ğŸ”µ 8ï¸�âƒ£ Salvar mÃ©tricas no log
metrics = {
    'round': 'CatBoost-original-features',
    'train_loss': model.best_score_['learn']['RMSE'],
    'val_loss': model.best_score_['validation']['RMSE'],
    'train_mae': np.nan,
    'val_mae': mae,
    'train_mse': np.nan,
    'val_mse': mse,
    'best_iteration': model.get_best_iteration()
}

log_file = Path("metrics_log.csv")
if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("ğŸ“� MÃ©tricas salvas em metrics_log.csv")




# ===================
# catboost tuned
# ===================

# 1ï¸�âƒ£ Dataset
train_pool = Pool(data=X_train_processed, label=y_train)
val_pool = Pool(data=X_val_processed, label=y_val)

# 2ï¸�âƒ£ Modelo tunado (parÃ¢metros usados anteriormente, pode ajustar)
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)

# 3ï¸�âƒ£ Treinar
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# 4ï¸�âƒ£ PrevisÃ£o
y_pred = model.predict(X_val_processed)

# 5ï¸�âƒ£ Avaliar
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f'\nğŸ“Š Resultados CatBoost-TUNED:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# 6ï¸�âƒ£ ImportÃ¢ncia das features
importances = model.get_feature_importance()
feature_names = all_features

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nâœ… ImportÃ¢ncia das features (CatBoost-TUNED):")
print(feat_imp_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - CatBoost-TUNED')
plt.show()

# 7ï¸�âƒ£ Log
metrics = {
    'round': 'CatBoost-TUNED',
    'train_loss': model.best_score_['learn']['RMSE'],
    'val_loss': model.best_score_['validation']['RMSE'],
    'train_mae': np.nan,
    'val_mae': mae,
    'train_mse': np.nan,
    'val_mse': mse,
    'best_iteration': model.get_best_iteration()
}

log_file = Path("metrics_log.csv")
if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("ğŸ“� MÃ©tricas salvas em metrics_log.csv")

catboost_model = model




from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 1ï¸�âƒ£ Modelo
model = LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# 2ï¸�âƒ£ Treinar com callbacks (early stopping + log a cada 100 rounds)
model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)]
)

# 3ï¸�âƒ£ PrevisÃ£o
y_pred = model.predict(X_val_processed, num_iteration=model.best_iteration_)

# 4ï¸�âƒ£ Avaliar
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f'\nğŸ“Š Resultados LightGBM:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# 5ï¸�âƒ£ Feature importance
importance = model.feature_importances_
feature_names = all_features

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nâœ… ImportÃ¢ncia das features (LightGBM):")
print(feat_imp_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - LightGBM')
plt.show()

# 6ï¸�âƒ£ Log
metrics = {
    'round': 'LightGBM-original-features',
    'train_loss': np.nan,
    'val_loss': rmse,
    'train_mae': np.nan,
    'val_mae': mae,
    'train_mse': np.nan,
    'val_mse': mse,
    'best_iteration': model.best_iteration_
}

log_file = Path("metrics_log.csv")
if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("ğŸ“� MÃ©tricas salvas em metrics_log.csv")

lightgbm_model = model 


# 3 modelos
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ”µ PrevisÃµes individuais
y_pred_cat = catboost_model.predict(X_val_processed)
y_pred_xgb = model_rand.predict(X_val_processed)
y_pred_lgb = lightgbm_model.predict(X_val_processed)

# ğŸ”µ Ensemble - mÃ©dia simples
y_pred_ensemble = (y_pred_cat + y_pred_xgb + y_pred_lgb) / 3

# ğŸ”µ Avaliar
mae = mean_absolute_error(y_val, y_pred_ensemble)
mse = mean_squared_error(y_val, y_pred_ensemble)
rmse = np.sqrt(mse)

print(f'\nğŸ“Š Resultados Ensemble (mÃ©dia simples):')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# ğŸ”µ Visualizar distribuiÃ§Ã£o dos erros
errors = pd.DataFrame({
    'y_true': y_val,
    'y_pred': y_pred_ensemble,
    'abs_error': abs(y_val - y_pred_ensemble)
}).sort_values(by='abs_error', ascending=False)

plt.figure(figsize=(10,6))
sns.histplot(errors['abs_error'], bins=30, kde=True)
plt.xlabel('Erro Absoluto')
plt.title('DistribuiÃ§Ã£o dos Erros Absolutos - Ensemble')
plt.show()

# ğŸ”µ Salvar log
metrics = {
    'round': 'Ensemble-XGB+CatBoost+LGB',
    'train_loss': np.nan,
    'val_loss': rmse,
    'train_mae': np.nan,
    'val_mae': mae,
    'train_mse': np.nan,
    'val_mse': mse,
    'best_iteration': np.nan
}

log_file = Path("metrics_log.csv")
if log_file.exists():
    df_log = pd.read_csv(log_file)
    df_log = df_log[df_log['round'] != metrics['round']]
    df_log = pd.concat([df_log, pd.DataFrame([metrics])], ignore_index=True)
else:
    df_log = pd.DataFrame([metrics])

df_log.to_csv(log_file, index=False)
print("ğŸ“� MÃ©tricas salvas em metrics_log.csv")



# ===================================================
# ğŸ“� Generate Submission File (Ensemble Model)
# ===================================================

# ğŸ”� Select only the feature columns (same as train)
X_test_raw = test_df[X_train_raw.columns]  # ensures same features as training

# ğŸŒ€ Apply the same preprocessing pipeline
X_test_processed = preprocessor.transform(X_test_raw)

print(f"âœ… X_test_processed generated with shape: {X_test_processed.shape}")

# ğŸ‘‰ Individual predictions from each model
y_pred_cat = catboost_model.predict(X_test_processed)
y_pred_xgb = model_rand.predict(X_test_processed)
y_pred_lgb = lightgbm_model.predict(X_test_processed)

# ğŸ‘‰ Ensemble prediction: simple mean
y_pred_test = (y_pred_cat + y_pred_xgb + y_pred_lgb) / 3

# ğŸ‘‰ Create submission DataFrame directly from test_df IDs
submission_final = pd.DataFrame({
    'ID': test_df['ID'],            # takes IDs directly from test_df
    'Age (years)': y_pred_test      # inserts predictions
})

# ğŸ‘‰ Save final CSV
submission_final.to_csv('submission_ensemble.csv', index=False)
print(f"âœ… Submission file 'submission_ensemble.csv' generated with {submission_final.shape[0]} rows, ready for Kaggle upload!")



# 1ï¸�âƒ£ Obter as prediÃ§Ãµes no conjunto de validaÃ§Ã£o:
dval = xgb.DMatrix(X_val_processed)
y_pred_val = best_model.predict(dval)



# 2ï¸�âƒ£ Calcular o erro absoluto:
erros = np.abs(y_val.values - y_pred_val)


# 3ï¸�âƒ£ Criar um DataFrame com os erros e as features originais:
df_erros = X_val_raw.copy()  # features antes da transformaÃ§Ã£o
df_erros['Idade Real'] = y_val.values
df_erros['Idade Prevista'] = y_pred_val
df_erros['Erro Absoluto'] = erros


# 4ï¸�âƒ£ Ordenar do maior para o menor erro:
df_erros_sorted = df_erros.sort_values(by='Erro Absoluto', ascending=False)


# 5ï¸�âƒ£ Mostrar as 10 piores previsÃµes:
print(df_erros_sorted.head(10))

