import os
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras

import optuna
import mlflow
from mlflow.models import infer_signature
from tensorflow.keras.callbacks import EarlyStopping

# Configurações globais do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.options.display.expand_frame_repr = False

# Desativa notação científica do numpy
np.set_printoptions(suppress=True)

# Suprime logs do optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['MLFLOW_DISABLE_RUN_STATUS_MESSAGES'] = 'true'


def set_all_seeds(seed=42):
    """
    Define todas as seeds necessárias para reprodutibilidade
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Para operações em GPU (se disponível)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Configurações adicionais do TensorFlow
    tf.config.experimental.enable_op_determinism()


def custom_transform_v1(df):
  df = df.copy()

  sleep_map = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
  }
  df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

  df['Gender'] = (df['Gender'] == 'Male').astype(int)
  df['Dietary Habits'] = (df['Dietary Habits'] == 'Healthy').astype(int)
  df['Family History of Mental Illness'] = (df['Family History of Mental Illness'] == 'Yes').astype(int)
  df['Have you ever had suicidal thoughts ?'] = (df['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)
  # df['Depression'] = (df['Depression'] == 'Yes').astype(int)

  return df


def custom_transform_v2(df):
  df = df.copy()

  categorical_cols_to_convert = [
    'Sleep Duration',
    'Gender',
    'Dietary Habits',
    'Family History of Mental Illness',
    'Have you ever had suicidal thoughts ?'
  ]

  for col in categorical_cols_to_convert:
    df[col] = df[col].astype('category')

  return df


def run_model(X, y, model, params, kf, pipeline):

    y_pred = np.zeros(len(y))
    accuracies = np.zeros(K)
    precisions = np.zeros(K)
    recalls = np.zeros(K)
    f1_scores = np.zeros(K)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        # print(f'Fold {fold + 1}/{kf.get_n_splits()}', end='\r')

        # Divisão dos dados
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Transformação dos dados
        pipeline.fit(X_train)
        X_train = pipeline.transform(X_train)
        X_val = pipeline.transform(X_val)

        # Treinamento do modelo
        current_model = model(**params)
        current_model.fit(X_train, y_train)

        # Previsões
        val_predictions = current_model.predict(X_val)
        y_pred[val_idx] = val_predictions

        # Store metrics for averaging
        accuracies[fold] = current_model.score(X_val, y_val)
        precisions[fold] = precision_score(y_val, val_predictions)
        recalls[fold] = recall_score(y_val, val_predictions)
        f1_scores[fold] = f1_score(y_val, val_predictions)

    # # Print average metrics after all folds
    # print(f'\nDesempenho médio do modelo:')
    # print(f'  Acurácia média: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    # print(f'  Precisão média: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
    # print(f'  Revocação média: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
    # print(f'  F1 médio: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')

    return accuracies.mean(), precisions.mean(), recalls.mean(), f1_scores.mean()



def log_model_run(
      X, y,
      model_class,
      model_parameters,
      kfold_splitter,
      transformation_function,
      run_prefix,
      run_description
      ):

  with mlflow.start_run(run_name=f"{run_prefix} - {model_class.__name__}"):

      # Define pipeline
      pipeline = Pipeline([('custom', FunctionTransformer(transformation_function, validate=False))])
      mlflow.log_param("pipeline", transformation_function.__name__)

      # Run model
      acc, precision, recall, f1 = run_model(X, y, model_class, model_parameters, kfold_splitter, pipeline)

      # Log hyperparameters
      mlflow.log_param("model", model_class.__name__)
      for param_name, param_value in model_parameters.items():
          mlflow.log_param(param_name, param_value)

      # Log metrics
      mlflow.log_metric("accuracy", acc)
      mlflow.log_metric("precision", precision)
      mlflow.log_metric("recall", recall)
      mlflow.log_metric("f1_score", f1)

      # Info
      mlflow.set_tag("info", run_description)


# Configure no início do notebook, antes de qualquer operação
SEED = 42
set_all_seeds(SEED)

# Configure o TensorFlow para usar menos threads para maior determinismo
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


train = pd.read_csv('kaggle/input/depressed-people/train.csv', index_col='index')
test = pd.read_csv('kaggle/input/depressed-people/test.csv', index_col='index')
sub = pd.read_csv('kaggle/input/depressed-people/sample_submission.csv')


y = (train['Depression'] == 'Yes').astype(int)
X = train.drop(columns=['Depression'])
X_test = test.copy()


# (opcional) definir URI de tracking explicitamente
mlflow.set_tracking_uri("http://localhost:5000")

# Cria ou seleciona um experimento
mlflow.set_experiment("Mental Health Classification")


K = 5
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)


# Check if there's an active run before ending it
if mlflow.active_run():
	mlflow.end_run()


transformation_function = custom_transform_v1
model_class = DummyClassifier
model_parameters = {'strategy': 'constant', 'constant': 1}
run_prefix = "Baseline"
run_description = "Baseline - Retorna 'Não' para todos os casos"

log_model_run(X, y, model_class, model_parameters, kf, transformation_function, run_prefix, run_description)


transformation_function = custom_transform_v1
model_class = LogisticRegression
model_parameters = {'penalty': 'l1', 'max_iter': 100, 'random_state': SEED, 'solver': 'liblinear', 'C': 1.0}
run_prefix = "Baseline"
run_description = "LogisticRegression Default"

log_model_run(X, y, model_class, model_parameters, kf, transformation_function, run_prefix, run_description)


transformation_function = custom_transform_v1
model_class = LogisticRegression
run_prefix = "BayesOpt"
run_description = "LogisticRegression ElasticNet com Optuna"

def objective(trial):
    # Define the hyperparameter search space
    model_parameters = {
        'penalty': 'elasticnet',
        'max_iter': 5000, # 'saga' pode precisar de mais iterações para convergir
        'random_state': SEED,
        'solver': 'saga',
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True)
        }

    with mlflow.start_run(nested=True):

        # Define pipeline
        pipeline = Pipeline([('custom', FunctionTransformer(transformation_function, validate=False))])
        mlflow.log_param("pipeline", transformation_function.__name__)

        # Run model
        acc, precision, recall, f1 = run_model(X, y, model_class, model_parameters, kf, pipeline)

        # Log hyperparameters
        mlflow.log_param("model", model_class.__name__)
        for param_name, param_value in model_parameters.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Info
        mlflow.set_tag("info", run_description)

        return f1

study = optuna.create_study(direction="maximize", study_name="ElasticNet Study")
study.optimize(objective, n_trials=10)


# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_param_importances(study).show()


# best_params_from_optuna = study.best_params
# final_model_params = {
#     'penalty': 'elasticnet',
#     'max_iter': 5000,
#     'random_state': SEED,
#     'solver': 'saga',
#     **best_params_from_optuna  # Adiciona os parâmetros otimizados
# }

# final_model = LogisticRegression(**final_model_params)
# pipeline = Pipeline([
#     ('custom', FunctionTransformer(transformation_function, validate=False)),
#     ('model', final_model)
# ])
# pipeline.fit(X, y)

# signature = infer_signature(X, pipeline.predict(X))

# with mlflow.start_run(run_name="Best LogisticRegression Final Model"):
#     mlflow.log_params(final_model_params)
#     model_info = mlflow.sklearn.log_model(
#         sk_model=pipeline,
#         artifact_path="logistic_regression_model",
#         signature=signature,
#         input_example=X,
#         registered_model_name="tracking-quickstart",
#     )


transformation_function = custom_transform_v1
model_class = XGBClassifier
model_parameters = {'random_state': SEED, 'enable_categorical': False}
run_prefix = "XGBoost"
run_description = "XGBoost sem suporte a variáveis categóricas"

log_model_run(X, y, model_class, model_parameters, kf, transformation_function, run_prefix, run_description)


transformation_function = custom_transform_v2
model_class = XGBClassifier
model_parameters = {'random_state': SEED, 'enable_categorical': True}
run_prefix = "XGBoost"
run_description = "XGBoost com suporte a variáveis categóricas"

log_model_run(X, y, model_class, model_parameters, kf, transformation_function, run_prefix, run_description)


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models, callbacks
import mlflow
import mlflow.keras

# Ativa autolog do MLflow para Keras
mlflow.keras.autolog(log_models=True)

transformation_function = custom_transform_v1
pipeline = Pipeline([('custom', FunctionTransformer(transformation_function, validate=False))])
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train_split = pipeline.fit_transform(X_train_split)
X_val_split = pipeline.transform(X_val_split)

def build_simple_nn(input_dim, hidden_units, dropout_rate, l2_lambda, n_layers, seed=42):
    tf.random.set_seed(seed)

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    if hidden_units > 0:
        for _ in range(n_layers):
            model.add(layers.Dense(
                hidden_units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None,
                # Inicializadores determinísticos
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                bias_initializer=keras.initializers.Zeros()
            ))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, seed=seed))

    model.add(layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=keras.initializers.Zeros()
    ))

    # Otimizador com seed
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    # Redefine seeds no início de cada trial
    trial_seed = SEED + trial.number  # Seed único para cada trial
    set_all_seeds(trial_seed)

    # Hiperparâmetros otimizáveis
    hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, .1, .2, .3, .4, .5])
    l2_lambda = trial.suggest_float("l2_lambda", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32])
    epochs = trial.suggest_categorical("epochs", [200])

    with mlflow.start_run(nested=True):
        mlflow.log_param("model", "Bayesian_NN")
        mlflow.log_param("trial_seed", trial_seed)  # Log da seed utilizada
        mlflow.set_tag("info", "NN Bayesian")
        mlflow.log_param("pipeline", transformation_function.__name__)
        mlflow.log_params({
            "hidden_units": hidden_units,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "l2_lambda": l2_lambda,
            "batch_size": batch_size,
            "epochs": epochs
        })

        model = build_simple_nn(X_train_split.shape[1], hidden_units, dropout_rate, l2_lambda, n_layers)

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,  # Importante para reprodutibilidade
            callbacks=[PlotLossesKeras(figsize=(12, 4)), early_stop],
        )

        y_pred_prob = model.predict(X_val_split).flatten()

        # Threshold ótimo via F1
        thresholds = np.linspace(0.1, 0.9, 100)
        f1_scores = [f1_score(y_val_split, y_pred_prob > t) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred = (y_pred_prob > best_threshold).astype(int)

        # Métricas
        acc = accuracy_score(y_val_split, y_pred)
        prec = precision_score(y_val_split, y_pred)
        rec = recall_score(y_val_split, y_pred)
        f1 = f1_score(y_val_split, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("best_threshold", best_threshold)

    return f1  # Objetivo: maximizar F1

# Otimização
study = optuna.create_study(direction="maximize", study_name="NN_optuna_f1")
study.optimize(objective, n_trials=10)


optuna.visualization.plot_optimization_history(study).show()


optuna.visualization.plot_param_importances(study).show()


final_model_params = study.best_params

final_model = build_simple_nn(
    X_train_split.shape[1],
    final_model_params['hidden_units'],
    final_model_params['dropout_rate'],
    final_model_params['l2_lambda'],
    final_model_params['n_layers']
    )

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

final_model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=final_model_params['epochs'],
    batch_size=final_model_params['batch_size'],
    verbose=0,
    shuffle=False,  # Importante para reprodutibilidade
    callbacks=[PlotLossesKeras(figsize=(12, 4)), early_stop],
)

y_pred_prob = final_model.predict(X_val_split).flatten()

 # Threshold ótimo via F1
thresholds = np.linspace(0.1, 0.9, 100)
f1_scores = [f1_score(y_val_split, y_pred_prob > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_prob > best_threshold).astype(int)

f1 = f1_score(y_val_split, y_pred)
print(f'  F1-Score: {f1:.4f}')
print(f'  Threshold ótimo: {best_threshold:.4f}')


  # F1-Score: 0.9880
  # Threshold ótimo: 0.6576


X_test = pipeline.transform(test)
y_pred_prob = final_model.predict(X_test).flatten()


y_pred = (y_pred_prob > best_threshold).astype(int)


sub['Depression'] = np.where(y_pred==1, 'Yes', 'No')
sub


sub.Depression.value_counts(normalize=True)


train.Depression.value_counts(normalize=True)


sub.to_csv('submission_v2.csv', index=False)




