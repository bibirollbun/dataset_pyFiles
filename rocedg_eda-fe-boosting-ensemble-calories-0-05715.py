# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Set plotting style and parameters
sns.set_theme(style="whitegrid")
sns.set_palette('pastel')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


# Function to load data
def load_data(base_path="/kaggle/input/playground-series-s5e5/"):
    """Load and return train, test and sample submission dataframes"""
    train = pd.read_csv(f"{base_path}train.csv")
    test = pd.read_csv(f"{base_path}test.csv")
    submission = pd.read_csv(f"{base_path}sample_submission.csv")
    
    # Convert 'Sex' to boolean: male=1, female=0
    train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
    test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
    
    return train, test, submission

# List input files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the data
train_data, test_data, sample_submission = load_data()


# Create reusable plotting function for efficiency
def plot_feature_target_relationship(df, features, target, figsize=(20, 10)):
    """Plot relationship between features and target with correlation coefficient"""
    fig, axes = plt.subplots(nrows=(len(features)+2)//3, ncols=3, figsize=figsize,
                           constrained_layout=True)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        # Scatterplot
        sns.scatterplot(x=df[feature], y=df[target], alpha=0.5, ax=axes[i])
        axes[i].set_title(f'{feature} vs {target}', fontsize=14)
        
        # Calculate and display correlation
        r = df[feature].corr(df[target])
        axes[i].text(0.05, 0.90, f'$r={r:.2f}$', transform=axes[i].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].axis('off')
    
    plt.show()


# Function to check data overview
def data_overview(df, name="Dataset"):
    """Print basic information about the dataset"""
    print(f"\n{name} Overview:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    display(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nStatistical summary:")
    display(df.describe())

# Check train data
data_overview(train_data, "Training Data")


# Visualize the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(train_data['Calories'], kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Calories Burned', fontsize=16)
plt.xlabel('Calories', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(alpha=0.3)
plt.show()


# Correlation analysis with heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Print correlation with target in descending order
print("Correlation with Calories:")
print(correlation_matrix['Calories'].sort_values(ascending=False))



# Analyze relationships between key features and target variable
features = ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plot_feature_target_relationship(train_data, features, 'Calories')


plt.figure(figsize=(12, 8))

# Create heart rate intervals for coloring
hr_bins = [0, 80, 90, 100, 110, 120, 130]
hr_labels = ['<80', '80-90', '90-100', '100-110', '110-120', '120+']
train_data['HR_Group'] = pd.cut(train_data['Heart_Rate'], bins=hr_bins, labels=hr_labels, right=False)

# Define a color palette
colors = ['purple', 'blue', 'teal', 'green', 'yellow', 'orange']
color_dict = dict(zip(hr_labels, colors))

# Plot each heart rate group with different color
for hr_group in hr_labels:
    group_data = train_data[train_data['HR_Group'] == hr_group]
    plt.scatter(group_data['Duration'], group_data['Calories'], 
                alpha=0.5, label=f'HR {hr_group} bpm', 
                color=color_dict[hr_group])

plt.title('Calorie Expenditure by Duration and Heart Rate', fontsize=14)
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Calories', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(title='Heart Rate Groups')
plt.show()


# 2. Body Temperature Relationship Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))

# Scatter plot with polynomial fit
sns.regplot(x='Body_Temp', y='Calories', data=train_data, 
           order=2, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax1)
ax1.set_title('Quadratic Relationship: Body Temp vs Calories', fontsize=14)

# Residual plot
sns.residplot(x='Body_Temp', y='Calories', data=train_data,
             order=2, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax2)
ax2.set_title('Residuals of Quadratic Fit', fontsize=14)
plt.tight_layout()
plt.show()


# Create age groups for analysis
def create_age_groups(df):
    """Create age group categories"""
    df = df.copy()
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 100], 
                           labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])
    return df

# Add age groups to data
train_data = create_age_groups(train_data)

# Analyze calories by sex
plt.figure(figsize=(4, 3))
sns.violinplot(x="Sex", y="Calories", data=train_data, palette="pastel", inner="quartile")
plt.title("Calories Burned by Sex")
plt.show()

# Analyze calories by age group
plt.figure(figsize=(12, 3))
sns.violinplot(x='Age_Group', y='Calories', data=train_data, palette="pastel", inner="quartile")
plt.title('Calories Burned by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Calories')
plt.show()



def engineer_features_with_activity(df):
    """Create clinically-relevant features with numerical/categorical separation"""
    df_fe = df.copy()
    
    # ======================
    # Numerical Features
    # ======================
    
    # 1. Anthropometric Features
    df_fe['BMI'] = df_fe['Weight'] / ((df_fe['Height']/100) ** 2)
    
    # 2. Metabolic Features (gender-specific formulas)
    # Resting Metabolic Rate (Mifflin-St Jeor)
    df_fe['RMR'] = np.where(
        df_fe['Sex'] == 1,  # Male
        (10 * df_fe['Weight']) + (6.25 * df_fe['Height']) - (5 * df_fe['Age']) + 5,
        (10 * df_fe['Weight']) + (6.25 * df_fe['Height']) - (5 * df_fe['Age']) - 161  # Female
    )
    
    # Gender-specific Max Heart Rate
    df_fe['Max_HR'] = np.where(
        df_fe['Sex'] == 1,
        208 - 0.7 * df_fe['Age'],  # Male formula
        206 - 0.88 * df_fe['Age']  # Female formula
    )
    
    # 3. Cardiovascular Features
    # Resting Heart Rate estimation
    df_fe['Estimated_RHR'] = np.where(
        df_fe['Sex'] == 1,
        np.select(
            [df_fe['Age'] < 30, df_fe['Age'] < 50, df_fe['Age'] >= 50],
            [60, 62, 65]
        ),
        np.select(
            [df_fe['Age'] < 30, df_fe['Age'] < 50, df_fe['Age'] >= 50],
            [65, 67, 70]
        )
    )
    
    # Heart Rate Reserve Percentage (clipped at 0)
    df_fe['Effort%'] = ((df_fe['Heart_Rate'] - df_fe['Estimated_RHR']) / (df_fe['Max_HR'] - df_fe['Estimated_RHR'])).clip(lower=0.01)
    
    # 4. Thermal Features
    df_fe['Temp_Deviation'] = df_fe['Body_Temp'] - 36.5
    df_fe['Temp_Deviation_Squared'] = df_fe['Temp_Deviation'] ** 2
    
    # 5. Activity Features (duration-adjusted)
    df_fe['Fitness_Score'] = df_fe['Duration'] / (((1 + df_fe['Effort%'])**2) * (1 + df_fe['Temp_Deviation']/6))
    
    # Physical Activity Level (updated bins)
    activity_bins = [0, 2.5, 5, 7.5, 10, 15]
    df_fe['PAL_Multiplier'] = pd.cut(df_fe['Fitness_Score'], bins=activity_bins,
                                   labels=[1.2, 1.375, 1.55, 1.725, 1.9])
    
    # 6. Energy Expenditure
    df_fe['TDEE'] = df_fe['RMR'] * df_fe['PAL_Multiplier'].astype(float)
    
    # 7. Exercise Response
    df_fe['Duration_HR'] = df_fe['Duration'] * df_fe['Heart_Rate']
    
    # 8. Interaction Terms
    df_fe['BMI_Duration'] = df_fe['BMI'] * df_fe['Duration']
    df_fe['BMI_Heart_Rate'] = df_fe['BMI'] * df_fe['Heart_Rate']
    df_fe['BMI_Age'] = df_fe['BMI'] * df_fe['Age']
    df_fe['Age_Duration'] = df_fe['Age'] * df_fe['Duration']

    # ======================
    # Categorical Features
    # ======================
    
    # 1. Demographic Categories
    bmi_bins = [0, 18.5, 25, 30, 35, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese','Severely_Obese']
    df_fe['BMI_Cat'] = pd.cut(df_fe['BMI'], bins=bmi_bins, labels=bmi_labels,right=False)

    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['-20', '20-30', '30-40', '40-50', '50-60', '60+']
    df_fe['Age_Cat'] = pd.cut(df_fe['Age'], bins=age_bins, labels=age_labels,right=False)

    tdee_bins = [0, 1800, 2200, 2600, 3000, 5000]
    tdee_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    df_fe['TDEE_Cat'] = pd.cut(df_fe['TDEE'], bins=tdee_bins,labels=tdee_labels, right=False)

    # Individual category encoding
    age_dummies = pd.get_dummies(df_fe['Age_Cat'], prefix='Age', dtype=int)
    bmi_dummies = pd.get_dummies(df_fe['BMI_Cat'], prefix='BMI', dtype=int)
    tdee_dummies = pd.get_dummies(df_fe['TDEE_Cat'], prefix='TDEE', dtype=int)
    # Concatenate all encoded features
    df_fe = pd.concat([df_fe, age_dummies, bmi_dummies, tdee_dummies], axis=1)

    # 2. Group Statistics
    # â€” Gender HR norms & Z-score
    df_fe['TDEE_Mean_Sex'] = df_fe.groupby('Sex')['TDEE'].transform('mean')
    df_fe['TDEE_Std_Sex']  = df_fe.groupby('Sex')['TDEE'].transform('std')
    df_fe['TDEE_ZScore_Sex'] = (df_fe['TDEE'] - df_fe['TDEE_Mean_Sex']) / df_fe['TDEE_Std_Sex']

    # â€” Gender DurHR norms & Z-score
    df_fe['DurHR_Mean_Sex'] = df_fe.groupby('Sex')['Duration_HR'].transform('mean')
    df_fe['DurHR_Std_Sex']  = df_fe.groupby('Sex')['Duration_HR'].transform('std')
    df_fe['DurHR_ZScore_Sex'] = (df_fe['Duration_HR'] - df_fe['DurHR_Mean_Sex']) / df_fe['DurHR_Std_Sex']

    # â€” Cleanup temps
    df_fe.drop(columns=['TDEE_Mean_Sex','TDEE_Std_Sex','DurHR_Mean_Sex','DurHR_Std_Sex'], inplace=True)
    
    return df_fe

# Create feature-engineered datasets
train_data_fe = engineer_features_with_activity(train_data)
test_data_fe = engineer_features_with_activity(test_data)


# Extract and clean
x = train_data_fe['TDEE'].dropna()
y = train_data_fe['Fitness_Score'].loc[x.index]

# Compute Pearson correlation
corr = x.corr(y)
print(f'Correlation between TDEE and Fitness Score: {corr:.3f}')

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('TDEE (kcal)')
plt.ylabel('Fitness Score')
plt.title(f'TDEE vs. Fitness Score (r = {corr:.3f})')
plt.tight_layout()
plt.show()


# Analyze relationships between key features and target variable
features = ['BMI','BMI_Age', 'RMR', 'TDEE' , 'Temp_Deviation_Squared','Duration_HR']
plot_feature_target_relationship(train_data_fe, features, 'Calories')


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def indirect_influence(df, variable, target='Calories', predictor='Duration_HR',
                       bins=None, labels=None, title=None, color_cycle=None, ax=None):
    """
    Stratified regression to assess how `variable` indirectly influences the 
    predictor-target relationship. Returns group coefficients and plots all fits.
    """
    group_col = f'{variable}_Group'
    df[group_col] = pd.cut(df[variable], bins=bins, labels=labels, right=False)
    x_vals = np.linspace(df[predictor].min(), df[predictor].max(), 300)
    # Create DataFrame with proper feature name for prediction
    x_vals_df = pd.DataFrame(x_vals, columns=[predictor])
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df[predictor], df[target], alpha=0.15, color='lightgray', label='Data')
    # Overall fit
    model = LinearRegression().fit(df[[predictor]], df[target])
    ax.plot(x_vals, model.predict(x_vals_df), 'k-', lw=2, label='Overall')
    print(f"Overall: coef={model.coef_[0]:.4f}  intercept={model.intercept_:.2f}")
    
    # Group fits
    coeffs, names = [], []
    if color_cycle is None:
        color_cycle = plt.cm.Set1.colors
    for i, group in enumerate(labels):
        sub = df[df[group_col]==group]
        if len(sub) > 0:
            m = LinearRegression().fit(sub[[predictor]], sub[target])
            ax.plot(x_vals, m.predict(x_vals_df), color=color_cycle[i%len(color_cycle)],
                   lw=2, label=f'{variable}: {group}')
            coeffs.append(m.coef_[0])
            names.append(group)
            print(f"{variable} {group}: coef={m.coef_[0]:.4f}  intercept={m.intercept_:.2f}")
    
    # Coefficient of variation
    if len(coeffs) > 1:
        cv = np.std(coeffs) / np.mean(coeffs)
        print(f"Coefficient Variation for {variable}: {cv:.4f}")
    
    ax.set_title(title or f"{variable} as Indirect Influencer")
    ax.set_xlabel(predictor)
    ax.set_ylabel(target)
    ax.legend(loc='upper right', fontsize='small')
    
    # Only show the plot if ax was None (meaning we created a new figure)
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return pd.DataFrame({f'{variable}_Group': names, 'Coefficient': coeffs})


# Create a 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

# Define bins/labels
age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['-20', '20-30', '30-40', '40-50', '50-60', '60+']
bmi_bins = [0, 18.5, 25, 30, 35, 50]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese']
rmr_bins = [0, 1400, 1700, 2000, 2300, 3000]
rmr_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
tdee_bins = [0, 1800, 2200, 2600, 3000, 5000]
tdee_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Run analysis for each variable on its own subplot
print("\n=== Age Analysis ===")
age_res = indirect_influence(train_data_fe, 'Age', bins=age_bins, labels=age_labels,
                           title='Age as Indirect Influencer', ax=axes[0])

print("\n=== BMI Analysis ===")
bmi_res = indirect_influence(train_data_fe, 'BMI', bins=bmi_bins, labels=bmi_labels,
                           title='BMI as Indirect Influencer', ax=axes[1])

print("\n=== RMR Analysis ===")
rmr_res = indirect_influence(train_data_fe, 'RMR', bins=rmr_bins, labels=rmr_labels,
                           title='RMR as Indirect Influencer', ax=axes[2])

print("\n=== TDEE Analysis ===")
tdee_res = indirect_influence(train_data_fe, 'TDEE', bins=tdee_bins, labels=tdee_labels,
                             title='TDEE as Indirect Influencer', ax=axes[3])

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

# Compare coefficient of variation for all
def coeff_var(df): return np.std(df['Coefficient']) / np.mean(df['Coefficient'])
influence_comparison = pd.DataFrame({
    'Variable': ['Age', 'BMI', 'RMR', 'TDEE'],
    'Coefficient_Variation': [coeff_var(age_res), coeff_var(bmi_res),
                             coeff_var(rmr_res), coeff_var(tdee_res)]
}).sort_values('Coefficient_Variation', ascending=False)
print("\nIndirect Influence Strength (by coefficient variation):")
print(influence_comparison)



# Make copies to avoid modifying the original DataFrames
train_data_fe_copy = train_data_fe.copy()
test_data_fe_copy = test_data_fe.copy()

train_data_fe_copy.columns


drop_cols = ['HR_Group', 'Age_Group', 'BMI', 'RMR',
       'Estimated_RHR', 'HRR_Percent','Activity_Score', 'PAL_Multiplier',
        'BMI_Cat', 'Age_Cat', 'TDEE_Cat','BMI_Group', 'RMR_Group', 'TDEE_Group']

# Prepare X_train and X_test for modeling, ignoring errors for missing columns
X_train = train_data_fe_copy.drop(['id', 'Calories'] + drop_cols, axis=1, errors='ignore')
y_train = train_data_fe_copy['Calories']
X_test = test_data_fe_copy.drop(['id'] + drop_cols, axis=1, errors='ignore')

# Display the first few rows of both sets
print("X_train sample:")
pd.set_option('display.max_columns', None)
X_train.head()


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

# Model imports
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# RMSLE metric for evaluation
def rmsle(y_true, y_pred):
    # Handle negative values
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# Create folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


"""from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

#ExtremeGBR
from xgboost import XGBRegressor
def tune_xgboost(X, y, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'tree_method': 'hist'
        }
        scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            scores.append(rmsle(y_val, val_pred))
        mean_score = np.mean(scores)
        print(f"[XGB][Trial {trial.number}] RMSLE: {mean_score:.5f}, Params: {params}")
        return mean_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best XGBoost params:", study.best_params)
    return study.best_params

# LightGBM
from lightgbm import LGBMRegressor
def tune_lightgbm(X, y, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 255),  
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'verbose': -1  # Suppress warnings[3]
        }
        scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            scores.append(rmsle(y_val, val_pred))
        mean_score = np.mean(scores)
        print(f"[LGBM][Trial {trial.number}] RMSLE: {mean_score:.5f}, Params: {params}")
        return mean_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best LightGBM params:", study.best_params)
    return study.best_params

# CatBoost
from catboost import CatBoostRegressor
def tune_catboost(X, y, n_trials=20):
    def objective(trial):
        cat_features = [
        'Sex', 'Age_-20', 'Age_20-30', 'Age_30-40',
        'Age_40-50', 'Age_50-60', 'Age_60+', 'BMI_Underweight', 'BMI_Normal',
        'BMI_Overweight', 'BMI_Obese', 'BMI_Severely_Obese', 'TDEE_Very_Low',
        'TDEE_Low', 'TDEE_Medium', 'TDEE_High', 'TDEE_Very_High']
        
        params = {
            'random_seed': 42,
            'cat_features': cat_features,
            'iterations': trial.suggest_int('iterations', 300, 2000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'verbose': 0
        }
        scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            scores.append(rmsle(y_val, val_pred))
        mean_score = np.mean(scores)
        print(f"[CAT][Trial {trial.number}] RMSLE: {mean_score:.5f}, Params: {params}")
        return mean_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best CatBoost params:", study.best_params)
    return study.best_params"""


import time
from scipy.optimize import minimize

# Cross-validation setup
FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize models with optimized parameters obtained in the previous tunnings
models = {
    'XGBoost': XGBRegressor(
        n_estimators=1351,
        max_depth=11,
        learning_rate=0.0164,
        subsample=0.8687,
        colsample_bytree=0.7103,
        reg_alpha=2.2751,
        reg_lambda=7.5201,
        tree_method='hist',
        random_state=42,
        enable_categorical=True
    ),

    'LightGBM': LGBMRegressor(
        n_estimators=818,
        max_depth=10,
        num_leaves=112,
        learning_rate=0.0648,
        subsample=0.8196,
        colsample_bytree=0.6699,
        reg_alpha=9.7294,
        reg_lambda=5.2392,
        min_child_samples=100,
        verbose=-1,
        random_state=42
    ),

    'CatBoost': CatBoostRegressor(
        iterations=1992,
        depth=9,
        learning_rate=0.1023,
        l2_leaf_reg=3.8577,
        bagging_temperature=0.9790,
        random_strength=0.0400,
        cat_features=[
            'Sex', 'Age_-20', 'Age_20-30', 'Age_30-40',
            'Age_40-50', 'Age_50-60', 'Age_60+', 'BMI_Underweight', 'BMI_Normal',
            'BMI_Overweight', 'BMI_Obese', 'BMI_Severely_Obese', 'TDEE_Very_Low',
            'TDEE_Low', 'TDEE_Medium', 'TDEE_High', 'TDEE_Very_High'
        ],
        random_seed=42,
        verbose=0
    )
}


from sklearn.metrics import mean_squared_log_error
import time

# Initialize dictionary to store results
results = {}

def train_xgboost(model, X_train, y_train, X_test, kf, FOLDS):
    name = 'XGBoost'
    print(f"\n=== Training {name} Model ===")
    start_time = time.time()
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{FOLDS}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)
        
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_val, y_val_log)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        val_pred = model.predict(X_val)
        oof_preds[valid_idx] = val_pred
        test_preds += model.predict(X_test) / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(y_val, np.expm1(val_pred)))
        fold_scores.append(rmsle)
        print(f"Fold {fold + 1} RMSLE: {rmsle:.5f}")
    
    mean_fold_rmsle = np.mean(fold_scores)
    std_fold_rmsle = np.std(fold_scores)
    
    print(f"{name} CV RMSLE: {mean_fold_rmsle:.5f} (Â±{std_fold_rmsle:.5f})")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
    
    results[name] = {
        'oof': oof_preds,
        'pred': test_preds,
        'scores': fold_scores,
        'overall_score': mean_fold_rmsle
    }

def train_catboost(model, X_train, y_train, X_test, kf, FOLDS):
    name = 'CatBoost'
    print(f"\n=== Training {name} Model ===")
    start_time = time.time()
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{FOLDS}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)
        
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_val, y_val_log)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        val_pred = model.predict(X_val)
        oof_preds[valid_idx] = val_pred
        test_preds += model.predict(X_test) / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(y_val, np.expm1(val_pred)))
        fold_scores.append(rmsle)
        print(f"Fold {fold + 1} RMSLE: {rmsle:.5f}")
    
    mean_fold_rmsle = np.mean(fold_scores)
    std_fold_rmsle = np.std(fold_scores)
    
    print(f"{name} CV RMSLE: {mean_fold_rmsle:.5f} (Â±{std_fold_rmsle:.5f})")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
    
    results[name] = {
        'oof': oof_preds,
        'pred': test_preds,
        'scores': fold_scores,
        'overall_score': mean_fold_rmsle
    }

def train_lightgbm(model, X_train, y_train, X_test, kf, FOLDS):
    name = 'LightGBM'
    print(f"\n=== Training {name} Model ===")
    start_time = time.time()
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{FOLDS}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)
        
        # Fix for LightGBM: Use callbacks instead of early_stopping_rounds
        try:
            # First try with callbacks if lightgbm version supports it
            from lightgbm import early_stopping
            model.fit(
                X_tr, y_tr_log,
                eval_set=[(X_val, y_val_log)],
                callbacks=[early_stopping(stopping_rounds=50)],
                verbose=0
            )
        except (ImportError, TypeError):
            # Fallback to simpler fit without early stopping
            model.fit(X_tr, y_tr_log)
        
        val_pred = model.predict(X_val)
        oof_preds[valid_idx] = val_pred
        test_preds += model.predict(X_test) / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(y_val, np.expm1(val_pred)))
        fold_scores.append(rmsle)
        print(f"Fold {fold + 1} RMSLE: {rmsle:.5f}")
    
    mean_fold_rmsle = np.mean(fold_scores)
    std_fold_rmsle = np.std(fold_scores)
    
    print(f"{name} CV RMSLE: {mean_fold_rmsle:.5f} (Â±{std_fold_rmsle:.5f})")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
    
    results[name] = {
        'oof': oof_preds,
        'pred': test_preds,
        'scores': fold_scores,
        'overall_score': mean_fold_rmsle
    }


train_xgboost(models['XGBoost'], X_train, y_train, X_test, kf, FOLDS)


train_catboost(models['CatBoost'], X_train, y_train, X_test, kf, FOLDS)


train_lightgbm(models['LightGBM'], X_train, y_train, X_test, kf, FOLDS)


# Print model comparison
print("\n=== Model Comparison ===")
for name in models.keys():
    mean_score = results[name]['overall_score']
    print(f"{name}: RMSLE = {mean_score:.5f}")


# Optimize ensemble weights
print("\n=== Optimizing Ensemble Weights ===")
oof_preds = {name: np.expm1(results[name]['oof']) for name in models}
y_true = np.expm1(y_train)

def rmsle_loss(weights):
    blended = sum(weights[i] * oof_preds[name] for i, name in enumerate(models))
    return np.sqrt(mean_squared_log_error(y_true, blended))

initial_weights = [1/len(models)] * len(models)
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0,1)] * len(models)

res = minimize(rmsle_loss, initial_weights, 
               method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print("\nOptimized Weights:")
for name, weight in zip(models.keys(), best_weights):
    print(f"{name}: {weight:.4f}")


# Generate final predictions
blended_preds = sum(weight * np.expm1(results[name]['pred']) 
                    for name, weight in zip(models.keys(), best_weights))
blended_preds = np.clip(blended_preds, 1, 314)


# Create submission
submission = pd.DataFrame({'id': test_data_fe_copy['id'], 'Calories': blended_preds})
submission.to_csv('ensemble_submission.csv', index=False)

print("\n=== Final Submission ===")
print(submission.head())
print(f"\nMean prediction: {blended_preds.mean():.2f}")
print(f"Median prediction: {np.median(blended_preds):.2f}")


# Get the XGBoost model from the models dictionary
xgb_model = models['XGBoost']

# Plot feature importance
fig, ax = plt.subplots(figsize=(24, 15))  # Adjust the figure size if needed
xgb.plot_importance(
    xgb_model,
    ax=ax,
    max_num_features=40, 
    importance_type="weight",
)
plt.title("XGB Top Feature Importances")
plt.show()


# Calculate ensemble CV score
print("\n=== Ensemble CV Score ===")
# Calculate the ensemble OOF predictions
ensemble_oof_preds = np.zeros(len(y_train))
for name, weight in zip(models.keys(), best_weights):
    ensemble_oof_preds += weight * np.expm1(results[name]['oof'])

# Calculate RMSLE for the ensemble
ensemble_rmsle = np.sqrt(mean_squared_log_error(y_train, ensemble_oof_preds))
print(f"Ensemble CV RMSLE: {ensemble_rmsle:.5f}")

# Compare with individual model scores
print("\nModel vs Ensemble Performance:")
for name in models.keys():
    mean_score = results[name]['overall_score']
    print(f"{name}: RMSLE = {mean_score:.5f}")
print(f"Ensemble: RMSLE = {ensemble_rmsle:.5f}")


# Create predicted vs real plot
plt.figure(figsize=(12, 8))
plt.scatter(y_train, ensemble_oof_preds, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Predicted vs Actual Calories')
plt.show()


# Identify and display outliers
print("\n=== Outlier Analysis ===")
# Calculate absolute percentage error
abs_pct_error = np.abs((ensemble_oof_preds - y_train) / y_train) * 100

# Create a DataFrame with actual, predicted, and error information
outlier_df = pd.DataFrame({
    'Actual_Calories': y_train,
    'Predicted_Calories': ensemble_oof_preds,
    'Absolute_Error': np.abs(ensemble_oof_preds - y_train),
    'Percentage_Error': abs_pct_error
})

# Sort by percentage error to find the worst predictions
worst_predictions = outlier_df.sort_values('Absolute_Error', ascending=False).head(20)
print("Top 20 worst predictions (by abs error):")
print(worst_predictions)


# Create bins of prediction ranges
bins = np.linspace(y_train.min(), y_train.max(), 10)
bin_labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]

# Assign each prediction to a bin
bin_indices = np.digitize(y_train, bins) - 1
bin_indices[bin_indices == len(bin_labels)] = len(bin_labels) - 1  # Handle edge case

# Calculate mean percentage error for each bin
bin_errors = []
bin_counts = []
for i in range(len(bin_labels)):
    mask = bin_indices == i
    if mask.sum() > 0:  # Ensure we have data in this bin
        bin_errors.append(outlier_df.loc[mask, 'Percentage_Error'].mean())
        bin_counts.append(mask.sum())
    else:
        bin_errors.append(0)
        bin_counts.append(0)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

# Plot mean percentage error by range
bars = ax1.bar(bin_labels, bin_errors, color='skyblue', edgecolor='navy')
ax1.set_title('Mean Percentage Error by Calorie Range', fontsize=16)
ax1.set_ylabel('Mean Percentage Error (%)', fontsize=14)
ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add error values on top of bars
for bar, error in zip(bars, bin_errors):
    if error > 0:  # Only show for non-zero values
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{error:.2f}%', ha='center', fontsize=10)

# Plot sample count per bin
ax2.bar(bin_labels, bin_counts, color='lightgreen', edgecolor='darkgreen')
ax2.set_title('Sample Count per Calorie Range', fontsize=14)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


