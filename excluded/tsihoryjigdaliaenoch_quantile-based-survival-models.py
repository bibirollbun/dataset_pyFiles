!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install optuna category_encoders


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
sns.set_theme()
warnings.filterwarnings('ignore')


import lightgbm as lgb
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import category_encoders
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, KBinsDiscretizer
from category_encoders import HashingEncoder, TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans


path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
input_data = pd.read_csv(path)
input_data.shape


_ = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
test_data = pd.read_csv(_)


num_cols = input_data.select_dtypes(include='number').columns
cat_cols = input_data.select_dtypes(include='object').columns


%%time
# Convert raw data to float data and apply log transformation to target column

num_imputer = IterativeImputer(KNeighborsRegressor(weights='distance'), max_iter=10, random_state=42)

num_transformer = Pipeline(
      [
        ('iterative_impute', num_imputer), # KNN imputation
        ('scaler', MinMaxScaler()),
      ]
)

cat_transformer = Pipeline(
      [
        ('target_encoder', TargetEncoder()),
        ('hashing_encoder',  HashingEncoder(n_components=8)),
        ('imputer', SimpleImputer(strategy="constant",fill_value=0)), # Replace NaN values with 0
      ]
)
preproc = ColumnTransformer(
      [
        ('num', num_transformer, num_cols[:-2]),
        ('cat', cat_transformer, cat_cols),
      ]
)
y_transformed = input_data[['efs_time']].apply(lambda x: np.log(x+1)).values
X_transformed = preproc.fit_transform(
    input_data,
    y_transformed
)
X_transformed = X_transformed.astype(np.float32)


X_test_transformed = preproc.transform(test_data)
X_test_transformed = X_test_transformed.astype(np.float32)


df = input_data[['efs','efs_time']]
df = df.sort_values(by='efs_time')
df['Cumulative events'] = df['efs'].cumsum()
df['Overall survival proba.'] = 1- (df['Cumulative events']/len(df))


fig, (ax0,ax1) = plt.subplots(1,2, sharex=True, figsize=(12,4))
fig.subplots_adjust(wspace=0.5)
ax0.step(df['efs_time'],df['Overall survival proba.'],where='post')
ax1.step(df['efs_time'],df['Cumulative events'],where='post')
ax0.set_title(r"Survival function of patients post HCT")
ax0.set_xlabel("Time since HCT (months)")
ax0.set_ylabel("Estimated probability")
ax1.set_title("Cumulative events function")
ax1.set_xlabel("Time since HCT (months)")
ax1.set_ylabel("Events count")
ax0.set_xlim([0,48])
ax1.set_xlim([0,48])
plt.show()


def plot_kaplan_meir_curve(title:str, feature:str, labels:list):
  plt.figure(figsize=(10,5))
  kmf = KaplanMeierFitter()

  for label in labels:
    df = input_data.loc[input_data[feature] == label]
    kmf.fit(df["efs_time"], event_observed=df["efs"])
    kmf.plot(ax=plt.gca(), label=f"{label}")

  plt.ylim(0, 1)
  plt.title(title)
  plt.xlabel("Time in months")
  plt.ylabel("Est. survival probability")
  plt.legend(loc="best")
  plt.show()


plot_kaplan_meir_curve(
    title = "Primary diseases are key predictors of HCT outcomes",
    feature = 'prim_disease_hct',
    labels = list(input_data['prim_disease_hct'].value_counts().index)
)


fig = sns.catplot(
    input_data.loc[input_data['efs_time']<= 12.0],
    x='prim_disease_hct', y='efs_time', kind='boxen',
    order=list(input_data['prim_disease_hct'].value_counts().index),
    height=10, aspect=1.5
)
fig.figure.suptitle("Observed lifetimes by primary disease")
fig.set_xticklabels(rotation=90)
fig.set_axis_labels("", "EFS time in months")
plt.show()


hla_match_features = [
       'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6','hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
       'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low','hla_match_dqb1_low', 'hla_match_a_high','hla_match_b_low', 'hla_match_a_low',
       'hla_match_b_high','hla_low_res_8','hla_match_drb1_high', 'hla_low_res_10'
]


df = pd.DataFrame(X_transformed, columns = input_data.columns[:-2])
df = df[hla_match_features]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(df)


input_data['hla_match_cluster'] = kmeans.labels_
input_data['hla_match_cluster'] = input_data['hla_match_cluster'].astype('category')
test_data = pd.DataFrame(X_test_transformed, columns=test_data.columns)
test_data['hla_match_cluster'] = kmeans.predict(test_data[hla_match_features].astype(np.float32))


plot_kaplan_meir_curve(
    title = "Survival patterns of HLA-match clusters",
    feature = 'hla_match_cluster',
    labels = np.unique(kmeans.labels_)
)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.regularizers import L1L2


def build_model(input_dim:int, output_layer:keras.layers,
                dropout_rate:float, lr:float, loss:object):
  """Builds a neural network for survival analysis using a specific loss."""

  regularizer = L1L2(l1=0.01, l2=0.01)

  model = tf.keras.Sequential(
    [
      Dense(96, activation='relu',kernel_regularizer=regularizer,input_shape=(input_dim,)),
      BatchNormalization(),
      Dropout(dropout_rate),

      Dense(64, activation='relu', kernel_regularizer=regularizer),
      BatchNormalization(),
      Dropout(dropout_rate),

      output_layer
    ]
  )

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss)
  return model


def ranking_loss(y_true, y_pred):
    """Pairwise ranking loss for survival analysis."""
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Compute pairwise differences
    diff_matrix = tf.expand_dims(y_pred, axis=0) - tf.expand_dims(y_pred, axis=1)
    true_diff = tf.expand_dims(y_true, axis=0) - tf.expand_dims(y_true, axis=1)

    # Mask for valid pairs
    mask = tf.cast(true_diff > 0, tf.float32)

    # Compute log-loss
    loss = tf.reduce_mean(mask * tf.math.log_sigmoid(diff_matrix))

    return -loss  # Minimize the negative log-likelihood


import tensorflow.keras.backend as K

def cox_loss(y_true, y_pred):
    """ Negative Cox partial likelihood loss function """
    
    time = y_true[:, 0]  
    event = tf.ones_like(time)

    # Sort by time in descending order
    sorting = tf.argsort(time, direction='DESCENDING')
    y_pred_sorted = tf.gather(y_pred, sorting)
    event_sorted = tf.gather(event, sorting)

    # Compute cumulative hazard ratio
    risk = K.exp(y_pred_sorted)
    log_risk = K.log(K.cumsum(risk))
    log_likelihood = y_pred_sorted - log_risk
    loss = -K.sum(log_likelihood * event_sorted)
    return loss


def train_eval_model(model,X_train,X_test,y_train,y_test,epochs,batch_size,callbacks=None):

  history = model.fit(
      X_train, y_train,
      epochs=epochs, batch_size=batch_size, verbose=1,
      validation_data=(X_test,y_test), callbacks=callbacks
  )
  # Predict survival scores
  Y_pred = model.predict(X_test).flatten()

  return history, Y_pred


DNNCoxRegression = build_model(
    input_dim= X_transformed.shape[1],
    output_layer= layers.Dense(1, activation="linear"),
    dropout_rate= 0.1,
    lr= 0.0001,
    loss= cox_loss)

X_train,X_test,y_train,y_test,e_train,e_test = train_test_split(
      X_transformed, y_transformed, input_data[['efs']].values, test_size=0.2, random_state=42)

hist, y_pred = train_eval_model(
    DNNCoxRegression,X_train,X_test,y_train,y_test,epochs = 20,batch_size = 64,
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )


discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
input_data['cox_risk_group'] = discretizer.fit_transform(DNNCoxRegression.predict(X_transformed).reshape(-1, 1))
test_data['cox_risk_group'] = discretizer.transform(DNNCoxRegression.predict(X_test_transformed).reshape(-1, 1))


# Lower values correspond to higher survival probabilities
plot_kaplan_meir_curve(
    title = "Survival function by quantiles of CoxPH risk scores",
    feature = 'cox_risk_group',labels = np.unique(input_data['cox_risk_group'])
)


# Feed Cox PH risk groups into training data
encoder = LabelEncoder()
X_transformed = np.concatenate(
  [
      X_transformed[:,1:],
      encoder.fit_transform(input_data[['hla_match_cluster']]).reshape(-1, 1),
      input_data['cox_risk_group'].values.reshape(-1, 1)
  ],
  axis = 1
)

# Train and test split
X_train,X_test,Y_train,Y_test,E_train,E_test = train_test_split(
    X_transformed,
    y_transformed,
    input_data[['efs']].values,
    test_size=0.2, random_state=42
)

X_test_transformed = np.concatenate(
  [
      X_test_transformed[:,1:], 
      encoder.transform(test_data[['hla_match_cluster']]).reshape(-1, 1),
      test_data['cox_risk_group'].values.reshape(-1, 1)
  ],
  axis = 1
)


cindex_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

DNNRankingLoss = build_model(
    input_dim= X_train.shape[1],
    output_layer= layers.Dense(1, activation="linear"), # Compute risk score
    dropout_rate= 0.1,
    lr= 0.0001,
    loss= ranking_loss)

fold = 1
for train_idx, val_idx in kf.split(X_train):
    print(f"Training fold {fold}/5")
    x_train, x_val = X_train[train_idx], X_train[val_idx]
    y_train, y_val = Y_train[train_idx], Y_train[val_idx]

    history,y_pred = train_eval_model(
      model = DNNRankingLoss,
      X_train = x_train,
      X_test = x_val,
      y_train = y_train,
      y_test = y_val,
      epochs = 20,
      batch_size = 64,
      callbacks = [
          EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
          ModelCheckpoint(f'model_fold_{fold}.keras', monitor='val_loss',save_best_only=True)
      ]
    )
    cindex = concordance_index(E_train[val_idx].flatten(), -y_pred.flatten())
    cindex_scores.append(cindex)
    fold += 1


input_data['dnn_survival_scores'] = DNNRankingLoss.predict(X_transformed)
test_data['dnn_survival_scores'] = DNNRankingLoss.predict(X_test_transformed)


dnn_y_pred = DNNRankingLoss.predict(X_test).flatten()
c_index = concordance_index(E_test.flatten(), -dnn_y_pred.flatten())
print(f"DNNRankingLoss cross-validated score : {np.mean(cindex_scores):.4f}")
print(f"DNNRankingLoss final score : {c_index:.4f}")


# Add DNN Survival scores to training data
X_transformed = np.concatenate(
  [
      X_transformed,
      input_data['dnn_survival_scores'].values.reshape(-1, 1)
  ],
  axis = 1
)

# Train and test split
X_train,X_test,Y_train,Y_test,E_train,E_test = train_test_split(
    X_transformed,
    y_transformed,
    input_data[['efs']].values,
    test_size=0.2, random_state=42
)


X_test_transformed = np.concatenate(
  [
      X_test_transformed,
      test_data['dnn_survival_scores'].values.reshape(-1, 1)
  ],
  axis = 1
)


models = {}
def objective(trial):
  params = {
    'objective': 'quantile',  # Predict quantiles of survival time
    'alpha': 0.5,  # Median survival time
    'metric': 'l1',  # MAE
    'boosting_type': 'gbdt',
    "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
    "num_leaves": trial.suggest_int("num_leaves", 20, 80),
    "max_depth": trial.suggest_int("max_depth", 3, 7),
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
    'lambda_l1': trial.suggest_loguniform("lambda_l1", 1e-4, 1),
    'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-4, 1),
    'feature_fraction': trial.suggest_float("feature_fraction", 0.6, 1.0),
    'verbose': -1,
    'seed': 42
  }

  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  cindex_scores = []

  for train_idx, val_idx in kf.split(X_train):
    x_train, x_val = X_train[train_idx], X_train[val_idx]
    y_train, y_val = Y_train[train_idx], Y_train[val_idx]

    train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(x_val, label=y_val, free_raw_data=False)

    lgbm_model = lgb.train(params, train_data, valid_sets=[val_data])

    preds = lgbm_model.predict(x_val)
    cindex = concordance_index(E_train[val_idx].flatten(), -preds)
    cindex_scores.append(cindex)

  return np.mean(cindex_scores)


import optuna
quantiles = [0.1, 0.5]
best_params = {}
for q in quantiles:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params[q] = study.best_params


# Parallel Training for Multiple Quantiles
from joblib import Parallel, delayed

def train_quantile(q, X_train, X_test):
    params = {
        'objective': 'quantile',
        'alpha': q,
        'metric': 'l1',
        'boosting_type': 'gbdt',
        **best_params[q]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        x_train, x_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = Y_train[train_idx], Y_train[val_idx]

        train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(x_val, label=y_val, free_raw_data=False)

        model = lgb.train(params, train_data, valid_sets=[val_data], callbacks=[lgb.early_stopping(10)])
        models[q] = model
        
        return q, model

results = Parallel(n_jobs=-1)(delayed(train_quantile)(q, X_train, X_test) for q in quantiles)
models = dict(results)


predictions_test = {q: models[q].predict(X_test) for q in quantiles}


weights = [0.4, 0.6]  # Emphasize median (0.5 quantile)
weighted_avg_test = sum(w * predictions_test[q] for w, q in zip(weights, quantiles))
c_index = concordance_index(E_test.flatten(), -weighted_avg_test)
print(f"Weighted averaging C-index : {c_index:0.4f}" )


predictions_test_data = {q: models[q].predict(X_test_transformed) for q in quantiles}


predicted_scores = sum(w * predictions_test_data[q] for w, q in zip(weights, quantiles))


predicted_scores = [-x for x in predicted_scores]


submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
submission['prediction'] = predicted_scores
submission.to_csv("submission.csv",index=False)
submission.head()

