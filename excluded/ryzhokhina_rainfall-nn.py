# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from matplotlib import pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col = 0)


def plot_roc_curve(y_true, y_pred):
    # Evaluate Model
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_pred):.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Rainfall Prediction", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid()
    plt.show()


train.describe()


from scipy import stats

# Function to detect outliers using Z-Score
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    print(f"ğŸ”� Outliers detected in {column}: {len(outliers)}")
    return outliers


feature_columns = ['pressure', 'temparature',  'dewpoint','humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

for col in feature_columns:
    detect_outliers_zscore(train, col, threshold=3)    


win_size = 7


def add_rolling_stat(data, columns, win_size):
    for col in columns:
        # Rolling Mean 
        data[f'{col}_rol_mean'] = data[col].rolling(window=win_size).mean()
        data[f'{col}_rol_mean'] = data[f'{col}_rol_mean'].bfill()
        # Rolling Standard Deviation 
        data[f'{col}_rol_std'] = data[col].rolling(window=win_size).std()
        data[f'{col}_rol_std'] = data[f'{col}_rol_std'].bfill()
    return data


# procedure to add_extention_features

def transform_dataset(data, win_size):
    # sin of day
    day_sin = np.sin((2.0*np.pi*data['day'])/365)
    extend_data = data.copy()
    extend_data['day_sin'] = day_sin

    # add rollings features
    feature_columns = ['pressure', 'temparature', 'maxtemp','mintemp', 'dewpoint','humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    extend_data = add_rolling_stat(extend_data, feature_columns, win_size)

    # drop not used features
    extend_data.drop('day', axis = 1, inplace = True)
    #extend_data.drop('maxtemp', axis = 1, inplace = True)
    #extend_data.drop('mintemp', axis = 1, inplace = True)

    return extend_data    


extend_train = transform_dataset(train, win_size)


extend_train.columns


feature_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'winddirection', 'windspeed',
       'day_sin', 'pressure_rol_mean', 'pressure_rol_std',
       'temparature_rol_mean', 'temparature_rol_std', 'maxtemp_rol_mean',
       'maxtemp_rol_std', 'mintemp_rol_mean', 'mintemp_rol_std',
       'dewpoint_rol_mean', 'dewpoint_rol_std', 'humidity_rol_mean',
       'humidity_rol_std', 'cloud_rol_mean', 'cloud_rol_std',
       'sunshine_rol_mean', 'sunshine_rol_std', 'winddirection_rol_mean',
       'winddirection_rol_std', 'windspeed_rol_mean', 'windspeed_rol_std']
target_column = 'rainfall'


x = extend_train[feature_columns]
y = extend_train[target_column]


# Split Data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


x_train.isna().sum()



x_val.isna().sum()


# Scale Numerical Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def plot_training_history(history):
    # Extract values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


def create_nn_model(input_shape):
    
    model = Sequential([
        Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=input_shape),  # Input Layer
        Dropout(0.2),  # Dropout for regularization
        Dense(64, activation='relu', kernel_initializer='he_normal'),  # Hidden Layer
        Dropout(0.5),
        Dense(32, activation='relu', kernel_initializer='he_normal'),  # Hidden Layer
        Dropout(0.3),
        Dense(16, activation='relu', kernel_initializer='he_normal'),  # Another Hidden Layer
        Dense(1, activation='sigmoid')  # Output Layer for Binary Classification
    ])
    return model


INPUT_SHARE = (x_train.shape[1],)
print(f'Input shape is{INPUT_SHARE}')


# Define the Neural Network Model
model = create_nn_model(INPUT_SHARE)

early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor the validation loss
    patience=15,                   # Wait for 7 epochs after the last improvement
    restore_best_weights=True,    # Restore the best weights from the epoch with the lowest validation loss
    verbose=1                     # Print information about early stopping
)

# Compile the Model
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    x_train, 
    y_train, 
    epochs=250, batch_size=32, 
    validation_data=(x_val, y_val), 
    callbacks = [early_stopping],
    verbose=1)


plot_training_history(history)


# Predict Probabilities & Convert to Binary Labels
y_prob = model.predict(x_val)
# Evaluate Model
y_pred = (y_prob > 0.5).astype(int)
plot_roc_curve(y_val, y_pred)


from sklearn.linear_model import LogisticRegression

# Train Logistic Regression Model
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(x_train, y_train)

# Predictions
y_pred = logreg.predict(x_val)
y_prob = logreg.predict_proba(x_val)[:, 1]  # Probability of rainfall (class 1)

plot_roc_curve(y_val, y_pred)


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import optuna
from functools import partial


# Define the Optuna Optimization Function
def objective(trial, x_train, x_test, y_train, y_test):
    # Tune hyperparameters for Logistic Regression
    lr_C = trial.suggest_float('lr_C', 1e-4, 10, log=True)

    # Tune hyperparameters for Random Forest
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
    rf_max_depth = trial.suggest_int('rf_max_depth', 3, 20)

    # Tune hyperparameters for XGBoost
    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 20)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)

    # Define the models with suggested parameters
    model1 = LogisticRegression(C=lr_C, max_iter=500)
    model2 = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    model3 = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                           learning_rate=xgb_learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Create Voting Classifier (Soft Voting)
    voting_clf = VotingClassifier(estimators=[
        ('lr', model1),
        ('rf', model2),
        ('xgb', model3)
    ], voting='soft')

    # Train the model
    voting_clf.fit(x_train, y_train)

    # Evaluate the model
    y_pred = voting_clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return roc_auc  # Optuna will try to maximize roc_auc


study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val), n_trials=20)

# Best Parameters
print("\n Best Hyperparameters Found:")
print(study.best_params)

# Train the Final Model with Best Parameters
best_params = study.best_params

final_lr = LogisticRegression(C=best_params['lr_C'], max_iter=500)
final_rf = RandomForestClassifier(n_estimators=best_params['rf_n_estimators'], max_depth=best_params['rf_max_depth'], random_state=42)
final_xgb = XGBClassifier(n_estimators=best_params['xgb_n_estimators'], max_depth=best_params['xgb_max_depth'],
                          learning_rate=best_params['xgb_learning_rate'], use_label_encoder=False, eval_metric='logloss', random_state=42)

# Create Final Voting Classifier
final_voting_clf = VotingClassifier(estimators=[
    ('lr', final_lr),
    ('rf', final_rf),
    ('xgb', final_xgb)
], voting='soft')

# Train the Final Optimized Model
final_voting_clf.fit(x_train, y_train)

# Evaluate Final Model
y_pred_final = final_voting_clf.predict(x_val)
final_accuracy = accuracy_score(y_val, y_pred_final)
final_roc_auc = roc_auc_score(y_val, y_pred_final)
print(f"\n Final Optimized Accuracy: {final_accuracy:.4f}")
print(f"\n Final Optimized ROC_AUC: {final_roc_auc:.4f}")


plot_roc_curve(y_val, y_pred_final)


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col = 0)


print(test.shape)
test.head()


mode_per_day = train.groupby('day')['winddirection'].agg(lambda x: x.mode()[0]).reset_index()
mode_per_day.columns = ['day', 'winddirection_mode']
mode_per_day.head()


# Merge mode information into test dataset
test_data = test.merge(mode_per_day, on='day', how='left')

# Fill missing values in test_data using the mode from train_data
test_data['winddirection'] = test_data['winddirection'].fillna(test_data['winddirection_mode'])

# Drop the extra mode column after filling
test_data.drop(columns=['winddirection_mode'], inplace=True)
test_data.head()


test_data.isna().sum()


test_data = transform_dataset(test_data, win_size)


print(test_data.shape)
test_data.head(10)


x_test = test_data[feature_columns]
x_test.head()


x_test = scaler.transform(x_test)
x_test


y_test_pred = model.predict(x_test)


y_test_pred.shape


y_test_pred = y_test_pred.squeeze()


len(y_test_pred)


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head() 


submission['rainfall'] = y_test_pred


submission.to_csv("submission.csv", index=False)




