#IMPORTING BASE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Notebook settings
import warnings
warnings.filterwarnings('ignore')


# IMPORT DATASETS (TRAIN & TEST)

train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

test_ids = test_data['id']
train_data = train_data.drop('id', axis=1)
test_data = test_data.drop('id', axis=1)


# Display the first few rows of each dataset
print("\n" + "#"*50 + "Train Data Head"+ "#"*50 + "\n")
display(train_data.head())
print("\n" + "#"*50 + "Test Data Head"+ "#"*50 + "\n")
display(test_data.head())


# DATA INFO
print("\n" + "#"*100 + "\n")
display(train_data.info())
print("\n" + "#"*100 + "\n")
display(test_data.info())


# SUMMARY STATS
print("\n" + "#"*100 + "\n")
display(train_data.describe())
print("\n" + "#"*100 + "\n")
display(test_data.describe())


plt.figure(figsize=(10, 6))
ax = sns.countplot(data=train_data, x='Personality', palette='viridis')

plt.title('Distribution of the Target Variable (Personality)', fontsize=16, weight='bold')
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add annotations
for p in ax.patches:
    ax.annotate(f'{p.get_height()} ({p.get_height()/len(train_data)*100:.2f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()


# NUMERICAL VARIABLES
#----------------------------------------------
# Histogram and Boxplot Ploting Helper Function
def hist_box(dataset, target):
    num_cols = dataset.select_dtypes(include=np.number).columns
    # Layout of the plots
    fig, axes = plt.subplots(len(num_cols), 2, figsize=[15, 20])

    for i, var in enumerate(num_cols):
        sns.histplot(data=dataset, x=var, ax=axes[i, 0], hue=target, palette='Spectral', kde=True)
        sns.boxplot(data=dataset, x=target, y=var, ax=axes[i, 1], hue=target, palette='Spectral')
    plt.show()


hist_box(train_data, 'Personality')


# CORRELATION ANALYSIS
plt.figure(figsize=(10, 8))
# We compute the correlation on the raw numerical data
corr_matrix = train_data.select_dtypes(include=np.number).corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    linewidths=.5
)

plt.title('Correlation Matrix of Numerical Variables', fontsize=16, weight='bold')
plt.show()


# CATEGORICAL VARIABLES
#----------------------------------------------

def cat_plot(dataset, target):
    cat_cols = dataset.select_dtypes(exclude=np.number).columns
    # Canvas of the plots
    fig, axes = plt.subplots(1, len(cat_cols), figsize=[15, 5])
    axes = axes.flatten()

    for i, var in enumerate(cat_cols):
        sns.countplot(data=dataset, x=var, ax=axes[i], palette='Spectral', hue=target)
        if var == 'Personality':
            sns.countplot(data=dataset, x=var, ax=axes[i], palette='Spectral')
            
        axes[i].set_title(f'{var}')
    plt.tight_layout()
    plt.show()
    


cat_plot(dataset=train_data, target='Personality')


# Visualizing missing values in the rain and test data
#=====================================================

# Helper function
def viz_missing(data_1, data_2):
    data_list = [data_1, data_2]
    fig, axes = plt.subplots(1, len(data_list), figsize=[15, 7]) # set plot canvas
    axes = axes.flatten()

    for i, data in enumerate(data_list):
        sns.heatmap(
            data.isnull(), cbar=False,
            cmap='viridis',
            yticklabels=False,
            ax=axes[i]
        )
        if 'Personality' in data.columns.tolist():
            axes[i].set_title('Missing values (yellow strips) in train data')
        else:
            axes[i].set_title('Missing values (yellow strips) in test data')


viz_missing(train_data, test_data)


# Calculate missing value percentages
missing_train = train_data.isnull().sum() / len(train_data) * 100
missing_test = test_data.isnull().sum() / len(test_data) * 100

missing_df = pd.DataFrame({'Train Missing %': missing_train, 'Test Missing %': missing_test})
missing_df = missing_df.drop('Personality', errors='ignore').sort_values(by='Train Missing %', ascending=False)

# Plotting
missing_df.plot(kind='barh', figsize=(12, 8), color=['#3498db', '#e74c3c'])
plt.title('Percentage of Missing Values per Feature', fontsize=16, weight='bold')
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.legend(title='Dataset')
plt.tight_layout()
plt.show()


# Import necessary libraries for this section
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def create_features(df):
    """
    Creates new features to enhance the predictive power of the model.
    """
    df_copy = df.copy()

    # Interaction Features
    # We hypothesize that the ratio of alone time to social time is important.
    # We add a small epsilon (1e-6) to avoid division by zero.
    df_copy['Alone_vs_Social'] = df_copy['Time_spent_Alone'] / (df_copy['Social_event_attendance'] + 1e-6)
    
    # Social Engagement Score
    # Combine features related to social activity. We are simply summing them up here.
    # A more complex version could scale them first.
    social_cols = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
    df_copy['Social_Engagement_Score'] = df_copy[social_cols].sum(axis=1)

    # Digital vs Physical Social Ratio
    df_copy['Digital_vs_Physical_Social'] = df_copy['Post_frequency'] / (df_copy['Friends_circle_size'] + 1e-6)
    
    return df_copy


train_featured = create_features(train_data)
test_featured = create_features(test_data)

display(train_featured.head())


# Separate target variable from features
X = train_featured.drop('Personality', axis=1)
y = train_featured['Personality']

# The test set for submission
X_test_final = test_featured.copy()

# Encode the target variable (e.g., Introvert -> 0, Extrovert -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify numerical and categorical columns from the FEATURED dataframe
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing pipelines for both numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fills NaNs with the most common value
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Converts categories to numerical format
])

# Create a single preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ],
    remainder='passthrough'
)

print("âœ… Preprocessing pipeline created successfully.")


# Import necessary libraries for this section
import optuna
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# This will hold the results of experiments
model_results = {}

# We define the preprocessed data that will be used in the experiments
# The preprocessor is applied INSIDE the cross-validation loop for purity,
# but for simplicity in this script, we apply it once. Applying
# it inside the loop is the gold standard to prevent any data leakage.

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test_final)

# Let's get the new feature names after OHE
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + ohe_feature_names.tolist()

# Convert processed arrays back to DataFrames (useful for some models and inspection)
X_processed = pd.DataFrame(X_processed, columns=all_feature_names)
X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names)

print("Data has been preprocessed.")
print("Shape of processed training data:", X_processed.shape)
print("Shape of processed test data:", X_test_processed.shape)


def lgbm_objective(trial):
    # Define parameters for Optuna to tune.
    params = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    cv_scores = []
    for train_idx, val_idx in kf.split(X_processed, y_encoded):
        X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Define Model
        model = lgb.LGBMClassifier(**params, random_state=42)
        # Fit Model
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='accuracy', # Accuracy score
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        cv_scores.append(accuracy)

    return np.mean(cv_scores)

# Create and run the Optuna study
lgbm_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
print("--- Starting LightGBM Hyperparameter Search ---")
lgbm_study.optimize(lgbm_objective, n_trials=50) 

print("\n--- LightGBM Tuning Complete ---")
print("Best trial for LightGBM:", lgbm_study.best_trial.value)
print("Best params for LightGBM:", lgbm_study.best_params)

# Add the 'metric' back into the best_params dictionary for later use
best_lgbm_params = lgbm_study.best_params
best_lgbm_params['metric'] = 'accuracy'
best_lgbm_params['objective'] = 'binary'

model_results['LGBM'] = {'score': lgbm_study.best_trial.value, 'params': best_lgbm_params}


def catboost_objective(trial):
    # Parameter space
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'task_type': 'GPU'
    }
    
    cv_scores = []
    for train_idx, val_idx in kf.split(X_processed, y_encoded):
        X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = cb.CatBoostClassifier(**params, random_seed=42)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  use_best_model=True)
        
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        cv_scores.append(accuracy)

    return np.mean(cv_scores)

# Create and run the study
catboost_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
catboost_study.optimize(catboost_objective, n_trials=50)

print("Best trial for CatBoost:", catboost_study.best_trial.value)
print("Best params for CatBoost:", catboost_study.best_params)
model_results['CatBoost'] = {'score': catboost_study.best_trial.value, 'params': catboost_study.best_params}


for model_name, result in model_results.items():
    print(f"{model_name}: Best CV Score = {result['score']:.5f}")

# Create instances of the best models with their optimal parameters
best_lgbm = lgb.LGBMClassifier(**model_results['LGBM']['params'], random_state=42)
best_catboost = cb.CatBoostClassifier(**model_results['CatBoost']['params'], random_state=42, verbose=0)

# Create a soft-voting ensemble
# 'soft' voting averages the predicted probabilities
ensemble_model = VotingClassifier(
    estimators=[
        ('lgbm', best_lgbm),
        ('catboost', best_catboost)
    ],
    voting='soft',
    weights=[0.5, 0.5] 
)

print("\n--- Training Final Ensemble Model on Full Dataset ---")
ensemble_model.fit(X_processed, y_encoded)
print("âœ… Final model trained successfully!")


# Predict probabilities on the processed test data
ensemble_preds_proba = ensemble_model.predict_proba(X_test_processed)[:, 1]
# We can set a threshold if needed, but 0.5 is standard for binary classification
ensemble_preds = (ensemble_preds_proba > 0.5).astype(int)


# Inverse transform the numerical predictions back to original labels ('Introvert'/'Extrovert')
final_predictions = label_encoder.inverse_transform(ensemble_preds)



# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': final_predictions
})




# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission file created successfully!")
display(submission_df.head())




