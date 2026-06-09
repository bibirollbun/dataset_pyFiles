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


df= pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.info()


df.columns


df['rainfall'].unique


df = df.drop('id', axis=1)


import matplotlib.pyplot as plt
import seaborn as sns



# Calculate correlation matrix
corr_matrix = df[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'rainfall']].corr()

# Create heatmap
plt.figure(figsize=(10, 8))  # Adjust size as needed
sns.heatmap(corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center the colormap at 0
            square=True,  # Make the plot square-shaped
            linewidths=.5,  # Width of the lines between cells
            fmt='.2f')  # Format annotation with 2 decimal places

plt.title('Correlation Matrix Heatmap')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# Assuming you have a DataFrame 'df' with these columns
columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
          'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
          'windspeed', 'rainfall']

# Create figure with subplots
fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(12, 4*len(columns)))

# Iterate through columns and create plots
for i, col in enumerate(columns):
    # Distribution plot (histogram + KDE)
    sns.histplot(data=df, x=col, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


#WINSORIZATION

from scipy.stats.mstats import winsorize

# List of numerical columns to apply Winsorization
outlier_cols = ['dewpoint', 'humidity', 'cloud']

# Apply Winsorization (Capping at 1st and 99th percentile)
for col in outlier_cols:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])  # 1% from both tails

# Check if outliers are reduced
df[outlier_cols].describe()



from scipy.stats import boxcox

# Define transformations for each type of skewness
right_skewed_cols = ['pressure', 'sunshine', 'winddirection', 'windspeed']
left_skewed_cols = ['maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud']

# Apply log transformation for right-skewed columns (add 1 to avoid log(0) errors)
for col in right_skewed_cols:
    df[col] = np.log1p(df[col])  # log1p is log(1 + x), safer for small values

# Apply power transformation (square) for left-skewed columns
for col in left_skewed_cols:
    df[col] = np.power(df[col], 2)  # Square the values to normalize left skew

# Check skewness again
df.skew()



df['cloud']= np.power(df['cloud'], 2)
df['cloud'].skew()


# Assuming you have a DataFrame 'df' with these columns
columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
          'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
          'windspeed', 'rainfall']

# Create figure with subplots
fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(12, 4*len(columns)))

# Iterate through columns and create plots
for i, col in enumerate(columns):
    # Distribution plot (histogram + KDE)
    sns.histplot(data=df, x=col, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Define features (X) and target (y)
X = df.drop(columns=['rainfall'])  # Replace with your actual target column
y = df['rainfall']

# Split data (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



def objective(trial, model_name):
    """Defines the objective function for Optuna hyperparameter tuning based on the given model."""

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=42
        )
    
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            random_state=42
        )
    
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 300),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            verbose=0,
            random_state=42
        )

    # Perform 5-fold cross-validation (Training Set: ~1,533 samples per fold)
    auc = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1).mean()
    
    return auc



!pip install optuna


import optuna
from sklearn.model_selection import cross_val_score

models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
best_params = {}

for model in models:
    study = optuna.create_study(direction="maximize")  # Maximize AUC
    study.optimize(lambda trial: objective(trial, model), n_trials=20)
    best_params[model] = study.best_params
    print(f"Best params for {model}: {study.best_params}")



rf = RandomForestClassifier(**best_params["RandomForest"], random_state=42)
xgb = XGBClassifier(**best_params["XGBoost"], use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm = LGBMClassifier(**best_params["LightGBM"], random_state=42)
cat = CatBoostClassifier(**best_params["CatBoost"], verbose=0, random_state=42)

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
cat.fit(X_train, y_train)



models_dict = {"RandomForest": rf, "XGBoost": xgb, "LightGBM": lgbm, "CatBoost": cat}

for name, model in models_dict.items():
    y_pred_prob = model.predict_proba(X_val)[:, 1]  # Get probability scores
    auc_score = roc_auc_score(y_val, y_pred_prob)
    
    print(f"{name} AUC: {auc_score:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")

# Final ROC Plot
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()



from sklearn.metrics import ConfusionMatrixDisplay
for name, model in models_dict.items():
    y_pred = model.predict(X_val)  # Get final predictions
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n{name} Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()


df_test= pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test.info()


df_test= df_test.drop('id', axis=1)


df_test["winddirection"].fillna(df_test["winddirection"].mean(), inplace=True)


# Assuming you have a DataFrame 'df' with these columns
columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
          'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
          'windspeed']

# Create figure with subplots
fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(12, 4*len(columns)))

# Iterate through columns and create plots
for i, col in enumerate(columns):
    # Distribution plot (histogram + KDE)
    sns.histplot(data=df_test, x=col, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


df_test.skew()


from scipy.stats import boxcox

# Define transformations for each type of skewness
right_skewed_cols = ['pressure', 'sunshine', 'winddirection', 'windspeed']
left_skewed_cols = ['maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud']

# Apply log transformation for right-skewed columns (add 1 to avoid log(0) errors)
for col in right_skewed_cols:
    df_test[col] = np.log1p(df_test[col])  # log1p is log(1 + x), safer for small values

# Apply power transformation (square) for left-skewed columns
for col in left_skewed_cols:
    df_test[col] = np.power(df_test[col], 2)  # Square the values to normalize left skew

# Check skewness again
df_test.skew()


df_test['cloud'] = np.power(df_test['cloud'], 2)
df_test['cloud'].skew()


# Standardize features
scaler = StandardScaler()
df_test = scaler.fit_transform(df_test)


pred_prob= cat.predict_proba(df_test)


test_data_id = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')  # Replace with your test file path


submission_df = pd.DataFrame({
    'id': test_data_id['id'],    # Extract 'id' column from df_test
    'rainfall': pred_prob[:,1]       # Assign predictions to 'rainfall' column
})


submission_df.head()


submission_df.to_csv('submission.csv', index= False)

