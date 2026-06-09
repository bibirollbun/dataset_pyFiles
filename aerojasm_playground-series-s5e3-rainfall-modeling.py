# Import packages and show directory files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm
import warnings

import os
# Deactivate for local execution
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Deactivate for execution in Kaggle
#dirname = os.getcwd()

pd.set_option('display.max_columns', None)
random_state = 42


# Import datasets
train = pd.read_csv(dirname + '/train.csv')
train = train.rename(columns={'temparature': 'temperature'})
test = pd.read_csv(dirname + '/test.csv')
test = test.rename(columns={'temparature': 'temperature'})

target = 'rainfall'

print('Datasets imported correctly')


# Define Custom Functions
def plot_feature_distribution(data, features):
    n_plots = len(features)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 7))
    fig.set_layout_engine('tight')
    axs = axs.ravel()

    for idx, col in enumerate(features):
        axs[idx].hist(data[col], bins=30, color='cornflowerblue')
        axs[idx].set_title(col)
        axs[idx].tick_params(axis='both', labelsize=8)
        axs[idx].tick_params(axis='x', rotation=20)
        axs[idx].yaxis.set_major_formatter(mtick.FuncFormatter(human_format))
        axs[idx].set_ylabel('Samples')

    for idx in range(n_plots, len(axs)):
        axs[idx].set_visible(False)
        
    return fig

def plot_binning_analyzer(data, features, bins=10):
    n_plots = len(features)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 12), sharey=True)
    fig.set_layout_engine('tight')
    axs = axs.ravel()

    for idx, col in enumerate(features):
        data["bin"] = pd.qcut(data[col], q=bins, duplicates='drop')
        data["bin"] = data["bin"].apply(lambda x: f"({round(x.left, 2)}, {round(x.right, 2)}]")
        bin_summary = data.groupby("bin", observed=False)["rainfall"].mean()
        bin_summary.plot(kind='bar', ax=axs[idx], color='teal', alpha=0.7)
        axs[idx].set_title(col)
        axs[idx].set_ylabel('Event rate')
        axs[idx].set_xlabel('')
        axs[idx].yaxis.set_major_formatter(mtick.PercentFormatter(1))

    for idx in range(n_plots, len(axs)):
        axs[idx].set_visible(False)

    data = data.drop(columns=['bin'])

    return fig

def human_format(num, pos):
    if num >= 1e9:
        return f'{num/1e9:.1f}B'
    elif num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.0f}K'
    else:
        return f'{num:.0f}'

def plot_confusion_matrix(model, X_1, y_1, X_2, y_2, data1=None, data2=None, threshold=0.5):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.set_layout_engine('tight')
    fig.patch.set_facecolor('#f2ebe3ff')

    # Get probability predictions
    y_pred_1 = (model.predict_proba(X_1)[:, 1] >= threshold).astype(int)
    y_pred_2 = (model.predict_proba(X_2)[:, 1] >= threshold).astype(int)

    # Compute confusion matrices
    cm1 = confusion_matrix(y_1, y_pred_1)
    cm2 = confusion_matrix(y_2, y_pred_2)

    # Display confusion matrices
    disp1 = ConfusionMatrixDisplay(cm1).plot(cmap='Blues', ax=axs[0])
    disp2 = ConfusionMatrixDisplay(cm2).plot(cmap='Blues', ax=axs[1])

    # Remove colorbars
    for disp in [disp1, disp2]:
        for img in disp.ax_.get_images():
            img.colorbar.remove()

    # Add titles
    axs[0].set_title(f'Confusion Matrix - {data1}')
    axs[1].set_title(f'Confusion Matrix - {data2}')
    
    return fig

print('Custom functions defined')


# Train data sample
train.sample(5)


# Define relevant columns
avoid_cols = ['id', 'day', 'rainfall']
base_features = [col for col in train.columns if col not in avoid_cols]
base_features


# Data shape
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)


# Descriptive statistics
train.describe()


# Check for missing values
train.isnull().sum()


# Assess numerical columns distributions
numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns
numerical_plots = plot_feature_distribution(train, numerical_columns)


# There are no categorical columns (beside the rainfall label)
categorical_columns = train.select_dtypes(include=['object']).columns
print('Number of categorical columns: ', len(categorical_columns))


# Label distribution (%)
train[target].value_counts(normalize=True)


# Label distribution (#)
train[target].value_counts()


# Binning Analyzer
rainfall_binning = plot_binning_analyzer(train, base_features, 8)


# As mentioned before, some features will probably have to be transformed and scaled
skewed_features = ['maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'sunshine', 'cloud']

for feature in skewed_features:
    # Train set
    scaler = StandardScaler()
    scaled_feature_name = f'{feature}_scaled'
    train[scaled_feature_name] = np.log1p(train[feature])
    train[scaled_feature_name] = scaler.fit_transform(train[scaled_feature_name].values.reshape(-1, 1))
    
    # Test set
    test[scaled_feature_name] = np.log1p(test[feature])
    test[scaled_feature_name] = scaler.transform(test[scaled_feature_name].values.reshape(-1, 1))

print('Features scaled and log transformed')


# Additional features
train['windspeed_squared'] = train['windspeed'] ** 2
train['temperature_difference'] = train['maxtemp'] - train['mintemp']


X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=[target]), train[target], test_size=0.3, stratify=train[target], random_state=42)

print('Train set: ', X_train.shape)
print('Validation set: ', X_val.shape)
print('Train labels: ', y_train.shape)
print('Validation labels: ', y_val.shape)


# Model: LightGBM
features = [
    'humidity',
    'cloud',
    'sunshine',
    'temperature',
    'windspeed'
    #'pressure'
    #'winddirection',
]

base_params = {
    "boosting": "dart",
    "learning_rate": 0.1,
    "max_bin": 200,
    "max_depth": 5,
    "num_leaves": 40,
    #"is_unbalance": True,
    "min_data_in_leaf": 200,
    "min_gain_to_split": 2.5,
    "reg_alpha": 0.1,
    #"reg_lambda": 0.1,
    "objective": "binary",
    "seed": random_state,
    "metric": "auc",
    #"feature_fraction": 0.8,
    "verbose": -1,
}

training_rounds = 250
dtrain = lgbm.Dataset(data=X_train[features], label=y_train)
dval = lgbm.Dataset(data=X_val[features], label=y_val, reference=dtrain)

lgbm_base_model = lgbm.train(
    params=base_params,
    train_set=dtrain,
    num_boost_round=training_rounds,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
)

y_val_pred_proba = lgbm_base_model.predict(X_val[features])
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# Check model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print('Validation Accuracy: ', val_accuracy)
print('Validation Precision: ', val_precision)
print('Validation Recall: ', val_recall)
print('Validation AUC ', val_roc_auc)


# Logistic regression
features = [
    'cloud_scaled',
    'humidity_scaled',
    'sunshine_scaled',
    #'temperature_difference',
    #'windspeed_squared'
    #'temperature_scaled',
    #'windspeed'
    #'pressure'
    #'winddirection',
]

lr = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',
    random_state=random_state, 
    max_iter=500
    #n_jobs=-1
)
lr.fit(X_train[features], y_train)

y_val_pred_proba = lr.predict_proba(X_val[features])[:, 1]
y_val_pred = lr.predict(X_val[features])

# Check model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print('Validation Accuracy: ', val_accuracy)
print('Validation Precision: ', val_precision)
print('Validation Recall: ', val_recall)
print('Validation AUC ', val_roc_auc)


# Random Forest Classifier
features = [
    'cloud',
    'humidity',
    'sunshine',
    'temperature_difference',
    #'windspeed_squared'
    #'temperature_scaled',
    #'windspeed',
    #'pressure',
    #'winddirection',
]

rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf.fit(X_train[features], y_train)

y_val_pred_proba = rf.predict_proba(X_val[features])[:, 1]
y_val_pred = rf.predict(X_val[features])

# Check model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print('Validation Accuracy: ', val_accuracy)
print('Validation Precision: ', val_precision)
print('Validation Recall: ', val_recall)
print('Validation AUC ', val_roc_auc)


# XGBoost + GridSearchCV
param_grid = {
    'objective': ['binary:logistic'],
    'scale_pos_weight': [1, 4],
    'min_child_weight': [1, 10],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.01],
    'eval_metric': ['auc'],
    #'reg_alpha': [0.1],
    #'reg_lambda': [0.1]
}

kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

xgb = XGBClassifier(
    random_state=random_state
)

grid_search_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=kf,
    scoring='roc_auc',
    verbose=1
)

grid_search_xgb.fit(
    X_train[features], y_train,
    eval_set=[(X_val[features], y_val)],
    #early_stopping_rounds=10,
    verbose=False
)

y_val_pred_proba = grid_search_xgb.predict_proba(X_val[features])[:, 1]
y_val_pred = grid_search_xgb.predict(X_val[features])

# Check model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print('Validation Accuracy: ', val_accuracy)
print('Validation Precision: ', val_precision)
print('Validation Recall: ', val_recall)
print('Validation AUC ', val_roc_auc)


features = [
    'humidity',
    'cloud',
    'sunshine',
    'temperature',
    'windspeed'
]
y_test_pred = lgbm_base_model.predict(test[features])
test['rainfall'] = y_test_pred
test[['id', 'rainfall']].to_csv('submission.csv', index=False)
print('Test submissing exported correctly')

