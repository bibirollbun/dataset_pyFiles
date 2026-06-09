import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

import lightgbm as lgb
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models

import optuna



sns.set_style("whitegrid")
sns.set_palette("Blues")


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.head()


# Spelling correction
train = train.rename(columns={'temparature': 'temperature'})
test = test.rename(columns={'temparature': 'temperature'})


def create_summary(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    # Add descriptive statistics
    desc = df.describe().transpose()
    desc = desc.reset_index()
    desc = desc.rename(columns={'index': 'Name'})
    summary = pd.merge(summary, desc, on='Name', how='left')

    return summary


create_summary(train).style.background_gradient(cmap='Blues')


test.isna().sum()


# Fill missing value in test set
test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


train.head()


def feat_eng(df):

    # Cyclic conversion of day_of_year
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)

    # Cyclic conversion winddirection
    df['winddirection_sin'] = np.sin(2 * np.pi * df['winddirection'] / 360.0)
    df['winddirection_cos'] = np.cos(2 * np.pi * df['winddirection'] / 360.0)
    
    # Height of cloudbase
    df['cloudbaseheight'] = (df['temperature'] - df['dewpoint']) * 125
    df['cloudbaseheight'] = df['cloudbaseheight'].clip(lower=0)

    # Add cloud cover in oktas
    bins = [-1, 0, 12, 25, 38, 50, 62, 75, 87, 100]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    df['cloud_oktas'] = pd.cut(df['cloud'], bins=bins, labels=labels, right=True).astype(int)

    # Temperature range
    df['temperature_range'] = df['maxtemp'] - df['mintemp']

    # Temperature anomaly
    df['temp_anomaly'] = df['temperature'] - df['temperature'].mean()
    
    # Vapor pressure & apparent temperature
    df['vapor_pressure'] = 6.11 * np.exp((17.27 * df['dewpoint']) / (237.7 + df['dewpoint']))
    df['apparent_temperature'] = df['temperature'] + (0.33 * df['vapor_pressure']) - (0.7 * df['windspeed']) - 4.0

    # Dewpoint depression
    df['dewpoint_depression'] = df['temperature'] - df['dewpoint']
    
    # Saturation Vapor Pressure (e_s)
    df['e_s'] = 6.112 * np.exp((17.67 * df['temperature']) / (df['temperature'] + 243.5))

    # Relative Humidity Calculation
    df['relative_humidity'] = (df['vapor_pressure'] / df['e_s']) * 100

    # Windspeed squared
    df['windspeed_squared'] = df['windspeed'] ** 2

    # Wind vectors
    df['u'] = df['windspeed'] * np.cos(df['winddirection'])
    df['v'] = df['windspeed'] * np.sin(df['winddirection'])
    
    # Pressure & temperature change
    df['pressure_change'] = df['pressure'].diff()
    
    # Fill NA values
    df['pressure_change'] = df['pressure_change'].fillna(0)
    
    # New feature engineering ideas
    # Dewpoint anomaly
    df['dewpoint_anomaly'] = df['dewpoint'] - df['dewpoint'].mean()

    # Humidity anomaly
    df['humidity_anomaly'] = df['humidity'] - df['humidity'].mean()

    # Wind speed anomaly
    df['windspeed_anomaly'] = df['windspeed'] - df['windspeed'].mean()

    # Cloud cover anomaly
    df['cloud_anomaly'] = df['cloud'] - df['cloud'].mean()

    # Sunshine anomaly
    df['sunshine_anomaly'] = df['sunshine'] - df['sunshine'].mean()

    # Temperature difference between max and min temperature
    df['temp_diff'] = df['maxtemp'] - df['mintemp']

    # Dewpoint difference between max and min dewpoint
    df['dewpoint_diff'] = df['dewpoint'].max() - df['dewpoint'].min()

    # Drop columns
    df.drop('day', axis=1, inplace=True)
    df.drop('winddirection', axis=1, inplace=True)
    
    return df


train = feat_eng(train)
test = feat_eng(test)


# Plot feature distribution
train_plot = train.drop('rainfall', axis=1)
ncols = 5
nrows = int(np.ceil(len(train_plot.columns) / ncols))

fig, ax = plt.subplots(nrows, ncols, figsize=(14, 14))
ax = ax.ravel()
plt.suptitle('Feature distribution', fontsize=20)

for idx, col in enumerate(train_plot.columns):
    
    sns.histplot(data=train_plot, x=col,ax=ax[idx], kde=True)
    # plt.title(f'{col} distribution')

# sns.histplot(data=train, x='rainfall', ax=ax[-1], kde=True)

plt.tight_layout()
plt.show()


# Convert skewed columns
skew_cols = []
normal_cols = []

upper_bound = 1
lower_bound = -1

for col in train_plot.columns:
    skew = train_plot[col].skew().round(2)

    if skew > upper_bound or skew < lower_bound:
        skew_cols.append(col)
    else:
        normal_cols.append(col)


# Define data and labels
X = train.drop('rainfall', axis=1)
y = train['rainfall']


sns.countplot(data=train, x='rainfall')
plt.title('Unbalanced Rainfall distribution')
plt.show()


# Calculate class weights
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, class_weights))

# Calculate class distribution
class_counts = np.bincount(y.astype(int))
class_priors = class_counts / len(y)

class_weights, class_priors


# Feature importance
X_feat = X.copy()
y_feat = y.copy()

col_names = X_feat.columns

rfc = SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=51), threshold=0.01)

rfc.fit(X_feat, y_feat)

# Feature importances
imp = rfc.estimator_.feature_importances_

# Map feature importances to feature names
df = pd.DataFrame(imp, index=col_names, columns=["Importance"])

# Plot feature importances
fig, ax = plt.subplots(figsize=(15, 10))
sorted_idx = imp.argsort()
ax.barh(df.index[sorted_idx], df["Importance"][sorted_idx], height=0.8)
ax.set_xlabel("Importance score")
ax.set_title("Feature Importance")
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

# Select features to keep based on threshold
to_drop = list(df.index[~rfc.get_support()])
print(f"Features suggested to drop: {to_drop}")


X = X.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)


# Make pipeline
# preprocessor = ColumnTransformer(transformers=[
#     ('power', PowerTransformer(method='yeo-johnson'), skew_cols),
#     ('scaler', StandardScaler(), normal_cols),
#     ],
#     remainder='passthrough'
# )
preprocessor = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
])

pipeline = Pipeline(steps=[
    # ('poly', PolynomialFeatures(degree=2)),
    ('preprocessor', preprocessor),
])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=51, stratify=y)


X_train = tf.convert_to_tensor(pipeline.fit_transform(X_train), dtype=tf.float32)
X_val = tf.convert_to_tensor(pipeline.transform(X_val), dtype=tf.float32)
X_test = tf.convert_to_tensor(pipeline.transform(test), dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)


# Define your objective function
def objective(trial):
    
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    activation = trial.suggest_categorical("activation", ["relu"])  # , "gelu", "tanh"
    l1_reg = trial.suggest_float("l1_reg", 1e-5, 1e-2, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
    
    # Define lists to store hyperparameters for each layer
    num_units_list = []
    dropout_rate_list = []
    
    for i in range(num_layers):
        num_units = trial.suggest_int(f"num_units_{i}", 32, 512, step=32)
        dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.2, 0.4, step=0.1)
        num_units_list.append(num_units)
        dropout_rate_list.append(dropout_rate)
    
    # Build model
    model = models.Sequential()
    model.add(layers.Dense(num_units_list[0], activation=activation, input_shape=(X_train.shape[1],),
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(layers.Dropout(dropout_rate_list[0]))
    for i in range(1, num_layers):
        model.add(layers.Dense(num_units_list[i], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(dropout_rate_list[i]))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='roc_auc')])
    
    # Train model
    es = tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc",patience=3)
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=3, min_lr=1e-6)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
              validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[es])
    
    # Evaluate model
    y_pred_proba = model.predict(X_val)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc


# Define Optuna study and optimize
np.random.seed(51)

# Set verbosity level
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)


np.random.seed(51)

# Get best hyperparameters
best_params = study.best_params
best_learning_rate = best_params["learning_rate"]
best_num_layers = best_params["num_layers"]
best_activation = best_params["activation"]
best_l1_reg = best_params["l1_reg"]
best_l2_reg = best_params["l2_reg"]
best_num_units_list = [best_params[f"num_units_{i}"] for i in range(best_num_layers)]
best_dropout_rate_list = [best_params[f"dropout_rate_{i}"] for i in range(best_num_layers)]

# Train final model with best hyperparameters
final_model = models.Sequential()
final_model.add(layers.Dense(best_num_units_list[0], activation=best_activation, input_shape=(X_train.shape[1],),
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=best_l1_reg,l2=best_l2_reg)))

final_model.add(layers.Dropout(best_dropout_rate_list[0]))

for i in range(1, best_num_layers):
    final_model.add(layers.Dense(best_num_units_list[i], kernel_regularizer=tf.keras.regularizers.l1_l2(l1=best_l1_reg,l2=best_l2_reg)))
    final_model.add(layers.BatchNormalization())
    final_model.add(layers.Activation(best_activation))
    final_model.add(layers.Dropout(best_dropout_rate_list[i]))
    
final_model.add(layers.Dense(1, activation='sigmoid'))

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                    loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.AUC(name='roc_auc')])

final_stop = tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc",patience=12, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=6, min_lr=1e-6)

history = final_model.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_data=(X_val, y_val), class_weight=class_weights, verbose=1,
                          callbacks=[lr_reduction, final_stop])  # , callbacks=[final_stop]

# Evaluate final model
final_score = final_model.evaluate(X_val, y_val)
print("ROC_AUC:", final_score[1])


# Plot training history

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot Loss
ax[0].plot(history.history['loss'], label='train')
ax[0].plot(history.history['val_loss'], label='validation')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].set_title('Loss')

# Plot ROC AUC
ax[1].plot(history.history['roc_auc'], label='train')
ax[1].plot(history.history['val_roc_auc'], label='validation')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('ROC AUC')
ax[1].legend()
ax[1].set_title('ROC AUC')

plt.tight_layout()
plt.show()


y_pred_proba = final_model.predict(X_val)
y_pred = np.where(y_pred_proba >= 0.5, 1, 0)


fig, ax = plt.subplots(1, 2, figsize=(15, 7))
plt.suptitle('ROC Curve and Confusion Matrix')

# Plot ROC curve
RocCurveDisplay.from_predictions(y_val, y_pred_proba, ax=ax[0])
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax[1], cmap='Blues', display_labels=['No Rainfall', 'Rainfall'])
ax[1].grid(False)

plt.show()


# Fit all data
X_final = tf.convert_to_tensor(pipeline.fit_transform(X), dtype=tf.float32)

final_model.fit(X_final, y)
y_pred_proba = final_model.predict(X_test)


plt.figure(figsize=(10, 6))
sns.histplot(y_pred_proba, kde=True)
plt.title('Predicted Rainfall distribution')
plt.legend([], [], frameon=False)
plt.show()


submit['rainfall'] = y_pred_proba


submit.to_csv('submission.csv', index=False)




