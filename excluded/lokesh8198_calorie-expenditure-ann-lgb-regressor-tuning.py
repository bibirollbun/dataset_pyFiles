import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import pandas as pd
import numpy as np

# Scikit-learn for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error


# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Hyperparameter optimization
import optuna

# Deep learning with TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TerminateOnNaN

# Keras Tuner for deep learning hyperparameter tuning
import keras_tuner as kt


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


train_file=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_file.head()



train_file[[i for i in train_file.columns if train_file[i].isnull().sum()>0]].isnull().sum()




#Table Information
train_file.info()



train_file.describe()


numeric_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']


# Set plot layout
n_cols = 3  # number of plots per row
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols 

plt.figure(figsize=(6 * n_cols, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train_file[col], kde=True,bins=30, color='steelblue')
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Density")

plt.tight_layout()
plt.show()



def plot_sex_pie(train_file):
    sex_counts = train_file['Sex'].value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of Sex')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

plot_sex_pie(train_file)



# Mapping dictionary
sex = {'male': 0, 'female': 1}

# Apply map
train_file['Sex_Numeric'] = train_file['Sex'].map(sex)
train_file.head()


train, validation, y_train, y_val = train_test_split(train_file[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Sex_Numeric']],train_file['Calories'], test_size=0.2, random_state=42)


# StandardScaler transformation of train data
def standardtrain(x):
    scaler = StandardScaler()
    column = x.columns
    sclaingdata = pd.DataFrame(scaler.fit_transform(x), columns=column)
    return sclaingdata

train_scale = standardtrain(train)
validation_scale = standardtrain(validation)




model_names = {
    "Linear Regressor":
    {
        "model": LinearRegression()
    },
    "Ridge Regressor":
    {
        "model": Ridge()
    },
    "Lasso Regressor":
    {
        "model": Lasso()
    },
    "K-Nearest Neighbors Regressor":
    {
        "model": KNeighborsRegressor()
    }
}

model_noscale_required = {
    "Decision Tree Regressor": {
        "model": DecisionTreeRegressor()
    },
    "Random Forest Regressor": {
        "model": RandomForestRegressor()
    },
    "Gradient Boosting Regressor": {
        "model": GradientBoostingRegressor()
    },
    "XGBoost Regressor": {
        "model": xgb.XGBRegressor()
    },
    
    "LightGBM Regressor": {
        "model": lgb.LGBMRegressor()
    }
}


def model_checking(model_name, model_obj, train_data, validation_data, y_train, y_val):
    model = model_obj['model']
    model.fit(train_data, y_train)
    y_pred= model.predict(validation_data)
    print(model_name)
    print("Mean Absolute Error: ",mean_absolute_error(y_val, y_pred))
    print("Mean Square Error: ",mean_squared_error(y_val, y_pred))
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("Root Mean Squared Error (RMSE): ",rmse)
  
    print("R² : ",r2_score(y_val, y_pred))

    # For Adjusted R², you need the number of samples (n) and features (p)
    n = len(y_val)  # Number of samples
    p = validation_data.shape[1]  # Number of features
    adjusted_r2 = 1 - (1 - r2_score(y_val, y_pred)) * (n - 1) / (n - p - 1)
    print("Adjusted R²: ",adjusted_r2)
    
    # Clip negatives to 0
    y_val_clip = np.clip(y_val, 0, None)
    y_pred_clip = np.clip(y_pred, 0, None)
    rmsle = mean_squared_log_error(y_val_clip, y_pred_clip, squared=False)
    print("Root Mean Squared Log Error (RMSLE): ", rmsle)
    print('\n \n')


for model_name, mn in model_names.items():
    model_checking(model_name, mn, train_scale, validation_scale, y_train, y_val)
for model_name, mn in model_noscale_required.items():
    model_checking(model_name, mn, train, validation, y_train, y_val)



import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    params = {
        'boosting_type': 'rf',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9), 
        'subsample_freq': 1, 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'device': 'gpu',
        'gpu_use_dp': False,
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': -1
    }

    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train, y_tr)
        preds = model.predict(X_val)
        y_val_clip = np.clip(y_val, 0, None)
        y_pred_clip = np.clip(preds, 0, None)
        rmsle = mean_squared_log_error(y_val_clip, y_pred_clip, squared=False)
        scores.append(rmsle)

    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best Hyperparameters:", study.best_params)




best_params = study.best_params
best_params["verbosity"] = -1 

best_model_lgb = lgb.LGBMRegressor(**best_params)
best_model_lgb.fit(train, y_train)
predictions = best_model_lgb.predict(validation)

mse = mean_squared_error(y_val, predictions)
mae = mean_absolute_error(y_val, predictions)
r2 = r2_score(y_val, predictions)
print(f'Best Model Mean Squared Error: {mse} \n R² Score: {r2} \n Mean Absolute Error: {mae}')
y_val_clip = np.clip(y_val, 0, None)
y_pred_clip = np.clip(predictions, 0, None)
rmsle = mean_squared_log_error(y_val_clip, y_pred_clip, squared=False)
print("Root Mean Squared Log Error (RMSLE): ", rmsle)


# Apply Power Transformation to normalize data
pt = PowerTransformer(method='yeo-johnson')
train_pow_trans = pt.fit_transform(train)
validation_pow_trans = pt.transform(validation)


# Standardize after Power Transform
scaler = StandardScaler()
train_pow_trans = scaler.fit_transform(train_pow_trans)
validation_pow_trans = scaler.transform(validation_pow_trans)


def rmsle(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0, tf.reduce_max(y_true))
    y_pred = tf.clip_by_value(y_pred, 0, tf.reduce_max(y_pred))
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

# Define function to build ANN model with hyperparameters
def build_model(hp):
    inputs = Input(shape=(train_pow_trans.shape[1],))
    x = inputs
    
    # Create hidden layers dynamically based on hyperparameters
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4, step=1)):
        x = Dense(
            hp.Int(f'units_{i+1}', min_value=32, max_value=128, step=32),
            activation=hp.Choice(f'activation_{i+1}', ['relu', 'tanh', 'elu', 'selu']),
            kernel_regularizer=l2(hp.Float(f'regularization_{i+1}', min_value=0.0001, max_value=0.01, step=0.0005))
        )(x)
        x = Dropout(hp.Float(f'dropout_rate_{i+1}', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    # Select optimizer dynamically
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
    
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=rmsle, metrics=['mae'])
    
    return model


# Perform hyperparameter tuning using Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='tuning_results',
    project_name='ann_regression_tuning'
)



# Run hyperparameter search
tuner.search(train_pow_trans, y_train,
             epochs=20,
             validation_data=(validation_pow_trans, y_val),
             callbacks=[TerminateOnNaN()])

# Retrieve best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")




# Train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(train_pow_trans, y_train, epochs=20, validation_data=(validation_pow_trans, y_val))



# Model evaluation
y_pred = best_model.predict(validation_pow_trans)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f'Best Model Mean Squared Error: {mse} \n R² Score: {r2} \n Mean Absolute Error: {mae}')
y_val_clip = np.clip(y_val, 0, None)
y_pred_clip = np.clip(y_pred, 0, None)
rmsle = mean_squared_log_error(y_val_clip, y_pred_clip, squared=False)
print("Root Mean Squared Log Error (RMSLE): ", rmsle)


test_file=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Apply map
test_file['Sex_Numeric'] = test_file['Sex'].map(sex)
test_file.head()


predictions = best_model_lgb.predict(test_file[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Sex_Numeric']])
predictions = np.clip(predictions, 0, None)


test_file['Calories']= predictions
test_file[['id','Calories']].to_csv('submission.csv', index = False)

