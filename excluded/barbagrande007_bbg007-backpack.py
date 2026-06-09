import warnings
warnings.filterwarnings("ignore")

import os
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, pairwise_distances, silhouette_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.keras import layers, models

import category_encoders as ce


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# plt.style.use('dark_background')

SEED = 51


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col="id")
submit = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col="id")


train.shape, train_extra.shape


train = pd.concat([train, train_extra], axis=0)


train.head()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


train.info()


train.isna().sum()


test.isna().sum()


# Identify categorical and numerical columns
def col_dtypes(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns
    col_names = df.columns
    return cat_cols, num_cols, col_names


# Pipeline for preprocessing (scaling + encoding)
def preprocess_pipeline(df_train, df_test):
    cat_cols, num_cols, col_names = col_dtypes(df_train)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False), cat_cols) 
        ], remainder='passthrough'
    )
    train_array = preprocessor.fit_transform(df_train)
    test_array = preprocessor.transform(df_test)
    
    return train_array, test_array


# Pipeline for feature selection (ordinal encoding)
def feature_importance(df):
    cat_cols, num_cols, col_names = col_dtypes(df)

    oenc = ColumnTransformer([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough')

    df = oenc.fit_transform(df)
    df = pd.DataFrame(df, columns=col_names)

    return df, col_names


X = train.drop("Price", axis=1)
y = train["Price"]

X_test = test.copy()


# Impute missing values

cat_cols, num_cols, col_names = col_dtypes(X)

X[cat_cols] = X[cat_cols].fillna("missing")
X_test[cat_cols] = X_test[cat_cols].fillna("missing")

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X_test[num_cols] = X_test[num_cols].fillna(X_test[num_cols].median())


X.head()


def feat_eng(df, df_test):

    # create new features
    df['weight/compartments'] = df['Weight Capacity (kg)'] / df['Compartments']
    df_test['weight/compartments'] = df_test['Weight Capacity (kg)'] / df_test['Compartments']

    df['Weight_Capacity_x_Compartments'] = df['Weight Capacity (kg)'] * df['Compartments']
    df_test['Weight_Capacity_x_Compartments'] = df_test['Weight Capacity (kg)'] * df_test['Compartments']

    df['Weight_Capacity_minus_Compartments'] = df['Weight Capacity (kg)'] - df['Compartments']
    df_test['Weight_Capacity_minus_Compartments'] = df_test['Weight Capacity (kg)'] - df_test['Compartments']

    cat_cols, num_cols, col_names = col_dtypes(df)
    
    # Target encoding
    target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=25)
    df = target_encoder.fit_transform(df, y)
    df_test = target_encoder.transform(df_test)

    return df, df_test


X, X_test = feat_eng(X, X_test)


cols = 6
rows = int(np.ceil(len(X.columns) / cols))

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(20,15))
ax = ax.flatten()

plt.suptitle("Visualize all features",size=24, y=1.01)

for i,col in enumerate(X.columns):
    if X[col].dtype == float or X[col].dtype == int:
        sns.boxplot(data=X,y=col,ax=ax[i],orient="vertical")
        ax[i].set_title(f"{col}")
    else:
        sns.countplot(data=X,x=col,ax=ax[i])
        ax[i].set_title(f"{col}")
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)

# sns.boxplot(data=y, orient="vertical", ax=ax[-1])
# ax[-1].set_title("Price")

plt.tight_layout()
plt.show()


X.head()


X, X_test = preprocess_pipeline(X, X_test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=SEED)


X_train.shape, X_val.shape


# Define model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(1)
])

# Compile model
model.compile(
    optimizer='adam', loss='mse', 
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

# Callbacks
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, 
    patience=6, verbose=1, mode='min', 
    min_delta=0.0001, cooldown=0, min_lr=0
    )

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_rmse', patience=12, verbose=1, 
    mode='min', restore_best_weights=True
    )

# Fit model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=512, 
    validation_data=(X_val, y_val), 
    verbose=1, callbacks=[lr_schedule, early_stopping]
    )

best_rmse = min(history.history['val_rmse'])
print(f"Best RMSE: {best_rmse}")


# Plot training & validation loss and RMSE values
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
ax[0].plot(history.history['loss'], label='Training Loss', lw=0.5)
ax[0].plot(history.history['val_loss'], label='Validation Loss', lw=0.5)
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='upper right')
ax[0].grid(True)  # Add grid

# Plot RMSE
ax[1].plot(history.history['rmse'], label='Training RMSE', lw=0.5)
ax[1].plot(history.history['val_rmse'], label='Validation RMSE', lw=0.5)
ax[1].set_title('Model RMSE')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('RMSE')
ax[1].legend(loc='upper right')
ax[1].grid(True)  # Add grid

plt.tight_layout()
plt.show()


lr_schedule_final = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='rmse', factor=0.1, 
    patience=6, verbose=1, mode='min', 
    min_delta=0.0001, cooldown=0, min_lr=0
    )

early_stopping_final = tf.keras.callbacks.EarlyStopping(
    monitor='rmse', patience=12, verbose=1, 
    mode='min', restore_best_weights=True
    )

history = model.fit(X, y, epochs=50, batch_size=512, verbose=1, callbacks=[lr_schedule_final, early_stopping_final])

best_rmse = min(history.history['rmse'])
print(f"Best RMSE: {best_rmse}")


y_pred = model.predict(X_test)


plt.figure(figsize=(10, 5))
sns.kdeplot(y_pred,color="red", label='Predictions')
sns.kdeplot(y, color="blue",label='True values')
plt.title("Predictions vs True values")
plt.legend()
plt.show()


submit['Price'] = y_pred


submit.to_csv("submission.csv", index=False)







