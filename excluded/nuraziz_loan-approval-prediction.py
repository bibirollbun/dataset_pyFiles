# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
df


df.info()


# Target distribution
sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status (0 = Approved, 1 = Rejected)")
plt.show()

# Class balance
print(df['loan_status'].value_counts(normalize=True))


missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)


categorical = df.select_dtypes(include='object').columns.tolist()
numerical = df.select_dtypes(include=['int64', 'float64']).drop(['loan_status', 'id'], axis=1).columns.tolist()
print("Categorical:", categorical)
print("Numerical:", numerical)


df[numerical].describe()
df[numerical].hist(bins=30, figsize=(14, 10))
plt.suptitle("Histograms of Numerical Features")
plt.show()


for col in categorical:
    plt.figure(figsize=(6, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()


import numpy as np
from scipy.stats import pointbiserialr

corrs = {}
for col in numerical:
    corr, _ = pointbiserialr(df[col], df['loan_status'])
    corrs[col] = corr

corr_df = pd.Series(corrs).sort_values(key=abs, ascending=False)
print(corr_df)

# Plot
corr_df.plot(kind='barh', title="Point-Biserial Correlation with Loan Status")
plt.xlabel("Correlation")
plt.show()


for col in categorical:
    ct = pd.crosstab(df[col], df['loan_status'], normalize='index')
    ct.plot(kind='bar', stacked=True, title=f"{col} vs Loan Status", figsize=(6,4))
    plt.ylabel("Proportion")
    plt.show()


import plotly.express as px

fig = px.scatter_3d(
    df,
    x='person_income',
    y='loan_amnt',
    z='loan_status',
    color='loan_status',
    title='Interactive 3D Scatter Plot: Income vs Loan Amount vs Age',
    opacity=0.8
)

fig.update_layout(
    scene=dict(
        zaxis=dict(range=[0, 5])  # Compresses Z-axis
    )
)

# Show the figure
fig.show()


sns.boxplot(x='loan_grade', y='loan_int_rate', hue='loan_status', data=df)
plt.title("Interest Rate by Loan Grade & Loan Status")
plt.show()


from scipy.stats import zscore
z_scores = df[numerical].apply(zscore)
outliers = (z_scores.abs() > 3).sum()
print("Number of outliers per feature:\n", outliers)


df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
sns.boxplot(x='loan_status', y='loan_to_income_ratio', data=df)
plt.title("Loan-to-Income Ratio by Loan Status")
plt.show()


X = df.drop('loan_status', axis=1)
y = df['loan_status']
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()


!pip install scikit-learn==1.3.0 imbalanced-learn==0.11.0 --quiet


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
# from sklearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomOverSampler

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


# # Create pipeline
# model_pipeline = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     # ('smote', SMOTE(random_state=42)),
#     ('classifier', XGBClassifier(
#         random_state=42,
#         scale_pos_weight=6,  # Because class imbalance ~ 86:14
#         use_label_encoder=False,
#         eval_metric='logloss'
#     ))
# ])

# # Train the model
# model_pipeline.fit(X_train, y_train)


# # Predictions
# y_pred = model_pipeline.predict(X_test)
# y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# # Metrics
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("AUC Score:", roc_auc_score(y_test, y_proba))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# One-Hot Cardinal
# 2) Separate label
y = df["loan_status"]
X = df.drop(columns=["loan_status", "id"], errors="ignore")

# 3) Feature Engineering
def add_features(df):
    df = df.copy()
    # Loanâ€‘toâ€‘Income ratio
    df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
    # Employment length buckets
    df["emp_length_bin"] = pd.cut(
        df["person_emp_length"],
        bins=[-1, 0, 2, 5, 10, np.inf],
        labels=["<0yr","0â€“2yr","2â€“5yr","5â€“10yr","10+yr"]
    )
    return df

featurizer = FunctionTransformer(add_features)

# 4) Column lists after engineering
numeric_feats = [
    "person_age", "person_income", "loan_amnt",
    "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "loan_to_income"
]
ordinal_feats = ["loan_grade"]      # assume A,B,Câ€¦ maps to order
ordinal_mapping = [["A","B","C","D","E","F","G"]]
categorical_feats = [
    "person_home_ownership", "loan_intent",
    "cb_person_default_on_file", "emp_length_bin"
]

# 5) Preprocessing pipelines
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

ord_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="A")),
    ("encode", OrdinalEncoder(categories=ordinal_mapping))
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer([
    ("num",  num_pipeline, numeric_feats),
    ("ord",  ord_pipeline, ordinal_feats),
    ("cat",  cat_pipeline, categorical_feats),
])


# Embedding. Skip, use either One-Hot cardinal or Target.

y = df["loan_status"]
X = df.drop(columns=["loan_status", "id"], errors="ignore")

# 3) Feature Engineering
def add_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["emp_length_bin"] = pd.cut(
        df["person_emp_length"],
        bins=[-1, 0, 2, 5, 10, np.inf],
        labels=["<0yr","0â€“2yr","2â€“5yr","5â€“10yr","10+yr"]
    )
    return df

featurizer = FunctionTransformer(add_features)
X = featurizer.fit_transform(X)

# 4) Column lists after engineering
numeric_feats = [
    "person_age", "person_income", "loan_amnt",
    "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "loan_to_income"
]
ordinal_feats = ["loan_grade"]
ordinal_mapping = [["A","B","C","D","E","F","G"]]
categorical_feats = [
    "person_home_ownership", "loan_intent",
    "cb_person_default_on_file", "emp_length_bin"
]

# 5) Preprocessing
# Only preprocess numeric + ordinal
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

ord_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="A")),
    ("encode", OrdinalEncoder(categories=ordinal_mapping))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_feats),
    ("ord", ord_pipeline, ordinal_feats),
    # Categorical features are excluded here to be handled with embedding layers
])

# Transform numeric and ordinal features
X_numord = preprocessor.fit_transform(X)

# Raw categorical features (to be integer encoded separately for embeddings)
X_cat_raw = X[categorical_feats].astype(str)


#Target. Skip, use either One-Hot cardinal or Embedding.
y = df["loan_status"]
X = df.drop(columns=["loan_status", "id"], errors="ignore")

# 2) Feature Engineering
def add_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["emp_length_bin"] = pd.cut(
        df["person_emp_length"],
        bins=[-1, 0, 2, 5, 10, np.inf],
        labels=["<0yr","0â€“2yr","2â€“5yr","5â€“10yr","10+yr"]
    )
    return df

featurizer = FunctionTransformer(add_features)

# 3) Column lists after engineering
numeric_feats = [
    "person_age", "person_income", "loan_amnt",
    "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "loan_to_income"
]
ordinal_feats = ["loan_grade"]
ordinal_mapping = [["A", "B", "C", "D", "E", "F", "G"]]
categorical_feats = [
    "person_home_ownership", "loan_intent",
    "cb_person_default_on_file", "emp_length_bin"
]

# 4) Pipelines
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

ord_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="A")),
    ("encode", OrdinalEncoder(categories=ordinal_mapping))
])

# 5) ColumnTransformer (numeric + ordinal only)
preprocessor = ColumnTransformer([
    ("num",  num_pipeline, numeric_feats),
    ("ord",  ord_pipeline, ordinal_feats),
])

# 6) Fit preprocessing
X_fe = featurizer.fit_transform(X)
X_proc = preprocessor.fit_transform(X_fe)

# 7) Target Encoding (done separately!)
target_enc = ce.TargetEncoder(cols=categorical_feats)
X_cat_te = target_enc.fit_transform(X_fe[categorical_feats], y)

# 8) Final X matrix
from numpy import hstack
import pandas as pd

X_final = hstack([X_proc, X_cat_te.values])


# 6) Create full pipeline
full_pipeline = Pipeline([
    ("feature_engineering", featurizer),
    ("preprocessing", preprocessor)
])

# 7) Preprocess the data
X_processed = full_pipeline.fit_transform(X)

# 8) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)


# 9) Convert to TensorFlow format
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)


# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')  # Output for binary classification
# ])

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
# )

# # 11) Add early stopping
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # 12) Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )


# # 13) Evaluate
# y_pred_prob = model.predict(X_test).flatten()
# y_pred = (y_pred_prob >= 0.5).astype(int)

# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_prob))


# test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
# test


# test['loan_to_income_ratio'] = test['loan_amnt'] / (test['person_income'] + 1)


# test_processed = full_pipeline.fit_transform(test)


# output = model.predict(test_processed).flatten()


# output


# result = pd.DataFrame({
#     'id': test['id'],
#     'loan_status': output
# })
# result


# output = model_pipeline.predict(test)
# output


# result = pd.DataFrame({
#     'id': test['id'],
#     'loan_status': output
# })
# result


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam


# X_processed


# Random Search
# units_input: 256
# bn_input: False
# act_input: relu
# dropout_input: 0.2
# num_layers: 3
# units_0: 32
# bn_0: False
# dropout_0: 0.30000000000000004
# lr: 0.005118522483814269
# units_1: 256
# bn_1: False
# dropout_1: 0.30000000000000004
# units_2: 128
# bn_2: True
# dropout_2: 0.4

# Optuna
# Best AUC: 0.9215285778045654
# Best Hyperparameters:
# units_input: 128
# bn_input: True
# act_input: tanh
# dropout_input: 0.4
# num_layers: 1
# units_0: 32
# bn_0: True
# dropout_0: 0.30000000000000004
# lr: 0.0009785132954243556

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

auc_scores = []
oof_preds = np.zeros(len(X))  # Same length as original data
models= []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    optimizer = Adam(learning_rate=0.005118522483814269)

    # Define model inside loop to reset weights each fold
    model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_processed.shape[1],)),  # ðŸ‘ˆ clean way to declare input shape
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train_fold, y_train_fold,
              validation_data=(X_val_fold, y_val_fold),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop],
              verbose=1)
    
    # Predictions
    val_preds = model.predict(X_val_fold).ravel()
    oof_preds[val_idx] = val_preds
    auc = roc_auc_score(y_val_fold, val_preds)
    auc_scores.append(auc)
    models.append((model, auc))
    models_sorted = sorted(models, key=lambda x: x[1], reverse=True)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

best_model, best_auc = models_sorted[0]
print(f"Best AUC: {best_auc:.4f}")


# test = models[6]
# model = test[0]


# output = model.predict(test_processed).flatten()
# output


# result = pd.DataFrame({
#     'id': test['id'],
#     'loan_status': output
# })
# result


# result.to_csv('submission.csv', index=False)


# import keras_tuner as kt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
# from tensorflow.keras.optimizers import Adam, RMSprop

# def build_model(hp):
#     model = Sequential()
    
#     # First Dense Layer
#     model.add(Dense(
#         units=hp.Choice('units_input', [32, 64, 128, 256]),
#         input_shape=(X_train.shape[1],)
#     ))
#     if hp.Boolean('bn_input'):
#         model.add(BatchNormalization())
#     model.add(Activation(hp.Choice('act_input', ['relu', 'tanh'])))
#     model.add(Dropout(hp.Float('dropout_input', 0.2, 0.5, step=0.1)))

#     # Hidden Layers
#     for i in range(hp.Int('num_layers', 1, 3)):
#         model.add(Dense(
#             units=hp.Choice(f'units_{i}', [32, 64, 128, 256])
#         ))
#         if hp.Boolean(f'bn_{i}'):
#             model.add(BatchNormalization())
#         model.add(Activation('relu'))
#         model.add(Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))

#     # Output Layer
#     model.add(Dense(1, activation='sigmoid'))

#     # Optimizer
#     learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')
#     optimizer = Adam(learning_rate=learning_rate)

#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=30,
#     executions_per_trial=2,
#     directory='my_dir',
#     project_name='tune_with_bn'
# )

# # tuner.search(X_processed, y, epochs=20, validation_split=0.2)


# # Get the best hyperparameter configuration
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# # Print each hyperparameter value
# for param in best_hps.values:
#     print(f"{param}: {best_hps.get(param)}")


# Run to find the hyperparameter
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import AUC

def build_model(hp):
    model = Sequential()
    
    # First Dense Layer
    model.add(Dense(
        units=hp.Choice('units_input', [32, 64, 128, 256]),
        input_shape=(X_train.shape[1],)
    ))
    if hp.Boolean('bn_input'):
        model.add(BatchNormalization())
    model.add(Activation(hp.Choice('act_input', ['relu', 'tanh'])))
    model.add(Dropout(hp.Float('dropout_input', 0.2, 0.5, step=0.1)))

    # Hidden Layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Choice(f'units_{i}', [32, 64, 128, 256])
        ))
        if hp.Boolean(f'bn_{i}'):
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer
    learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) #this is loss/accuracy. try recording auc.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='auc',
    max_trials=30,
    executions_per_trial=2,
    directory='my_dir',
    project_name='tune_with_bn'
)

tuner.search(X_processed, y, epochs=20, validation_split=0.2)


# Get the best hyperparameter configuration
 best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print each hyperparameter value
for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")


import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

def objective(trial):
    model = Sequential()

    # First Dense Layer
    units_input = trial.suggest_categorical('units_input', [32, 64, 128, 256])
    model.add(Dense(units_input, input_shape=(X_train.shape[1],)))
    
    if trial.suggest_categorical('bn_input', [True, False]):
        model.add(BatchNormalization())

    act_input = trial.suggest_categorical('act_input', ['relu', 'tanh'])
    model.add(Activation(act_input))

    dropout_input = trial.suggest_float('dropout_input', 0.2, 0.5, step=0.1)
    model.add(Dropout(dropout_input))

    # Hidden Layers
    num_layers = trial.suggest_int('num_layers', 1, 3)
    for i in range(num_layers):
        units_i = trial.suggest_categorical(f'units_{i}', [32, 64, 128, 256])
        model.add(Dense(units_i))

        if trial.suggest_categorical(f'bn_{i}', [True, False]):
            model.add(BatchNormalization())

        model.add(Activation('relu'))

        dropout_i = trial.suggest_float(f'dropout_{i}', 0.2, 0.5, step=0.1)
        model.add(Dropout(dropout_i))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    # Return the best AUC from validation
    return max(history.history['val_auc'])


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)


print("Best AUC:", study.best_value)
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")




