import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")

df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head(1)


df.shape


df.dtypes


def plot_eda_grid(df, output_file='eda_plots.png', figsize=(12,8), cols_per_row=4, palette='muted'):
    """
    Generate a grid of plots for all categorical and numerical columns in a DataFrame with a consistent theme.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - output_file: str, filename to save the plot (e.g., 'eda_plots.png')
    - figsize: tuple, figure size (width, height)
    - cols_per_row: int, number of plots per row in the grid
    - palette: str, seaborn color palette name (e.g., 'muted', 'bright', 'deep')
    """
    # Set seaborn theme for consistent, modern visuals
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['font.family'] = 'Arial'
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    all_cols = list(categorical_cols) + list(numerical_cols)
    
    if not all_cols:
        print("No columns to plot.")
        return
    
    # Calculate grid dimensions
    n_cols = min(cols_per_row, len(all_cols))
    n_rows = math.ceil(len(all_cols) / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Plot each column
    for i, col in enumerate(all_cols):
        ax = axes[i]
        if col in categorical_cols:
            # Bar plot for categorical columns
            sns.countplot(data=df, x=col, ax=ax, palette=palette)
            ax.set_title(f'{col} (Categorical)', fontsize=12, pad=10)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        else:
            # Histogram with KDE for numerical columns
            sns.histplot(data=df, x=col, ax=ax, kde=True, color=sns.color_palette(palette)[0])
            ax.set_title(f'{col} (Numerical)', fontsize=12, pad=10)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.tick_params(axis='both', labelsize=9)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add a super title
    fig.suptitle('Exploratory Data Analysis', fontsize=16, y=1.05)
    
    # Save and show plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


a = plot_eda_grid(df, output_file='iris_eda_plots.png', palette='muted')
a


df.head(1)


# 1. BMI: weight (kg) / height (m)^2
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 2. Age Group: Categorical bins
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    
# 3. Heart Rate x Weight
df['Heart_Rate_x_Weight'] = df['Heart_Rate'] * df['Weight']
    
# 4. Age x Heart Rate
df['Age_x_Heart_Rate'] = df['Age'] * df['Heart_Rate']
    
# 5. Duration x Heart Rate
df['Duration_x_Heart_Rate'] = df['Duration'] * df['Heart_Rate']
    
# 6. RMR: Mifflin-St Jeor Equation
# Men: RMR = 10*weight + 6.25*height*100 - 5*age + 5
# Women: RMR = 10*weight + 6.25*height*100 - 5*age - 161
df['RMR'] = np.where(
        df['Sex'] == 'male',
        10 * df['Weight'] + 6.25 * df['Height'] * 100 - 5 * df['Age'] + 5,
        10 * df['Weight'] + 6.25 * df['Height'] * 100 - 5 * df['Age'] - 161
)
    
# 7. VO2 Max: Using heart rate (simplified formula: VO2max = 15 * (HRmax/HRrest))
# Assume heart_rate is resting for simplicity; adjust if you have max HR data
df['HR_max'] = 220 - df['Age']  # Max heart rate
df['VO2_Max'] = 15 * (df['HR_max'] / df['Heart_Rate'].replace(0, np.nan))  # Avoid division by zero
    
# 8. Heart Rate Zone: Low (<60% HRmax), Moderate (60-80% HRmax), High (>80% HRmax)
df['HR_Percentage'] = df['Heart_Rate'] / df['HR_max'] * 100
bins = [0, 60, 80, 100]
labels = ['Low', 'Moderate', 'High']
df['Heart_Rate_Zone'] = pd.cut(df['HR_Percentage'], bins=bins, labels=labels, include_lowest=True)
    
# 9. Polynomial Duration: log(duration + 1)
df['Log_Duration'] = np.log1p(df['Duration'])
    
# 10. Lean Body Mass (Boer Formula)
# Men: LBM = 0.407*weight + 0.267*height*100 - 19.2
# Women: LBM = 0.252*weight + 0.473*height*100 - 48.3
df['Lean_Body_Mass'] = np.where(
        df['Sex'] == 'male',
        0.407 * df['Weight'] + 0.267 * df['Height'] * 100 - 19.2,
        0.252 * df['Weight'] + 0.473 * df['Height'] * 100 - 48.3
)
    
# 11. Body Fat Percentage (BMI-based, Deurenberg formula)
# BF% = 1.2*BMI + 0.23*age - 10.8*(1 if male, 0 if female) - 5.4
df['Body_Fat_Percentage'] = 1.2 * df['BMI'] + 0.23 * df['Age'] - 10.8 * (df['Sex'] == 'male').astype(int) - 5.4
    
# 12. Heart Rate x Body Temperature
df['Heart_Rate_x_Body_Temp'] = df['Heart_Rate'] * df['Body_Temp']
    
# Handle missing values (e.g., from division by zero or invalid calculations)
df = df.fillna(df.mean(numeric_only=True))


df.head(1)


df_eda = df.copy()


df_eda.drop(['Sex', 'Age', 'id', 'Duration', 'Height', 'Weight', 'Heart_Rate', 'Body_Temp'], axis=1, inplace=True)
df_eda.head(1)


a = plot_eda_grid(df_eda, output_file='iris_eda_plots.png',palette='muted')
a


numerical_cols = ['BMI', 'Heart_Rate_x_Weight', 'Age_x_Heart_Rate', 'Duration_x_Heart_Rate',
                     'RMR', 'VO2_Max', 'HR_max', 'HR_Percentage', 'Log_Duration',
                     'Lean_Body_Mass', 'Body_Fat_Percentage', 'Heart_Rate_x_Body_Temp']
categorical_cols = ['Age_Group', 'Heart_Rate_Zone', 'gender']


numerical_cols = [col for col in numerical_cols if col in df.columns]
categorical_cols = [col for col in categorical_cols if col in df.columns]


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()
    
# Define which scaler to use for each column
minmax_scale_cols = ['Duration', 'HR_Percentage']
robust_scale_cols = ['Age', 'Height', 'Weight', 'Heart_Rate', 'Heart_Rate_x_Weight', 
                        'Age_x_Heart_Rate', 'Duration_x_Heart_Rate', 'RMR', 'HR_max', 
                        'Lean_Body_Mass', 'Body_Fat_Percentage', 'Heart_Rate_x_Body_Temp', 'Body_Temp']
log_scale_cols = ['BMI', 'VO2_Max', 'Log_Duration']
    
    # Apply standardization
for col in numerical_cols:
    if col in minmax_scale_cols:
        df[[col]] = minmax_scaler.fit_transform(df[[col]])
    elif col in robust_scale_cols:
        df[[col]] = robust_scaler.fit_transform(df[[col]])
    elif col in log_scale_cols:
            # Apply log transformation and then StandardScaler
            # Use np.log1p to handle zero values safely
        df[col] = np.log1p(df[col].clip(lower=0))  # Ensure no negative values
        df[[col]] = standard_scaler.fit_transform(df[[col]])
    
    # Handle missing values (if any)
df[numerical_cols] = df[numerical_cols].fillna(0)  # Fill NaNs with 0 for standardized columns


one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = one_hot_encoder.fit_transform(df[['Age_Group', 'Heart_Rate_Zone', 'Sex']])
encoded_col_names = one_hot_encoder.get_feature_names_out(['Age_Group', 'Heart_Rate_Zone', 'Sex'])
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)

# Step 2: Scale numeric features
numeric_cols = ['Heart_Rate', 'Weight', 'Height', 'Duration', 'Body_Temp', 'BMI']
min_max_scaler = MinMaxScaler()
df[numeric_cols] = min_max_scaler.fit_transform(df[numeric_cols])

# Step 3: Combine encoded and scaled features, drop original categorical columns
df = pd.concat([df.drop(['Age_Group', 'Heart_Rate_Zone', 'Sex'], axis=1), encoded_df], axis=1)


X = df.drop(columns=['Calories'])  # Features
y = df['Calories']  # Target


from sklearn.model_selection import KFold

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# Define RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE).
    
    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    
    Returns:
    - rmsle: float, RMSLE score
    """
    # Clip predictions to avoid negative values
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    # Compute RMSLE
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


import optuna
from catboost import CatBoostRegressor, Pool

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False
    }

    # K-Fold Cross-Validation
    fold_rmsle_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Create CatBoost Pool
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        # Initialize CatBoost model
        modelCat = CatBoostRegressor(**params)

        # Train the model
        modelCat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        # Evaluate the model
        y_pred = modelCat.predict(X_val)
        score = rmsle(y_val, y_pred)

        fold_rmsle_scores.append(score)

    # Return the mean RMSLE across folds
    return np.mean(fold_rmsle_scores)


# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
final_rmsle_scores = []


for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    modelCat = CatBoostRegressor(**best_params, verbose=False)
    modelCat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    y_pred = modelCat.predict(X_val)
    score = rmsle(y_val, y_pred)

    final_rmsle_scores.append(score)

    print(f"Fold: Validation RMSLE = {score}")


mean_rmsle = np.mean(final_rmsle_scores)
std_rmsle = np.std(final_rmsle_scores)

print(f"\nCross-Validation Results:")
print(f"Mean RMSLE: {mean_rmsle} (±{std_rmsle})")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
X = df_test.drop(columns=['id'])
y = df_test['id']
df_test.head(1)


# 1. BMI: weight (kg) / height (m)^2
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] ** 2)

# 2. Age Group: Categorical bins
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']
df_test['Age_Group'] = pd.cut(df_test['Age'], bins=bins, labels=labels, include_lowest=True)
    
# 3. Heart Rate x Weight
df_test['Heart_Rate_x_Weight'] = df_test['Heart_Rate'] * df_test['Weight']
    
# 4. Age x Heart Rate
df_test['Age_x_Heart_Rate'] = df_test['Age'] * df_test['Heart_Rate']
    
# 5. Duration x Heart Rate
df_test['Duration_x_Heart_Rate'] = df_test['Duration'] * df_test['Heart_Rate']
    
# 6. RMR: Mifflin-St Jeor Equation
# Men: RMR = 10*weight + 6.25*height*100 - 5*age + 5
# Women: RMR = 10*weight + 6.25*height*100 - 5*age - 161
df_test['RMR'] = np.where(
        df_test['Sex'] == 'male',
        10 * df_test['Weight'] + 6.25 * df_test['Height'] * 100 - 5 * df_test['Age'] + 5,
        10 * df_test['Weight'] + 6.25 * df_test['Height'] * 100 - 5 * df_test['Age'] - 161
)
    
# 7. VO2 Max: Using heart rate (simplified formula: VO2max = 15 * (HRmax/HRrest))
# Assume heart_rate is resting for simplicity; adjust if you have max HR data
df_test['HR_max'] = 220 - df_test['Age']  # Max heart rate
df_test['VO2_Max'] = 15 * (df_test['HR_max'] / df_test['Heart_Rate'].replace(0, np.nan))  # Avoid division by zero
    
# 8. Heart Rate Zone: Low (<60% HRmax), Moderate (60-80% HRmax), High (>80% HRmax)
df_test['HR_Percentage'] = df_test['Heart_Rate'] / df_test['HR_max'] * 100
bins = [0, 60, 80, 100]
labels = ['Low', 'Moderate', 'High']
df_test['Heart_Rate_Zone'] = pd.cut(df_test['HR_Percentage'], bins=bins, labels=labels, include_lowest=True)
    
# 9. Polynomial Duration: log(duration + 1)
df_test['Log_Duration'] = np.log1p(df_test['Duration'])
    
# 10. Lean Body Mass (Boer Formula)
# Men: LBM = 0.407*weight + 0.267*height*100 - 19.2
# Women: LBM = 0.252*weight + 0.473*height*100 - 48.3
df_test['Lean_Body_Mass'] = np.where(
        df_test['Sex'] == 'male',
        0.407 * df_test['Weight'] + 0.267 * df_test['Height'] * 100 - 19.2,
        0.252 * df_test['Weight'] + 0.473 * df_test['Height'] * 100 - 48.3
)
    
# 11. Body Fat Percentage (BMI-based, Deurenberg formula)
# BF% = 1.2*BMI + 0.23*age - 10.8*(1 if male, 0 if female) - 5.4
df_test['Body_Fat_Percentage'] = 1.2 * df_test['BMI'] + 0.23 * df_test['Age'] - 10.8 * (df_test['Sex'] == 'male').astype(int) - 5.4
    
# 12. Heart Rate x Body Temperature
df_test['Heart_Rate_x_Body_Temp'] = df_test['Heart_Rate'] * df_test['Body_Temp']
    
# Handle missing values (e.g., from division by zero or invalid calculations)
df_test = df_test.fillna(df_test.mean(numeric_only=True))



for col in numerical_cols:
    if col in minmax_scale_cols:
        df_test[[col]] = minmax_scaler.fit_transform(df_test[[col]])
    elif col in robust_scale_cols:
        df_test[[col]] = robust_scaler.fit_transform(df_test[[col]])
    elif col in log_scale_cols:
            # Apply log transformation and then StandardScaler
            # Use np.log1p to handle zero values safely
        df_test[col] = np.log1p(df_test[col].clip(lower=0))  # Ensure no negative values
        df_test[[col]] = standard_scaler.fit_transform(df_test[[col]])


one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = one_hot_encoder.fit_transform(df_test[['Age_Group', 'Heart_Rate_Zone', 'Sex']])
encoded_col_names = one_hot_encoder.get_feature_names_out(['Age_Group', 'Heart_Rate_Zone', 'Sex'])
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df_test.index)

# Step 2: Scale numeric features
numeric_cols = ['Heart_Rate', 'Weight', 'Height', 'Duration', 'Body_Temp', 'BMI']
min_max_scaler = MinMaxScaler()
df_test[numeric_cols] = min_max_scaler.fit_transform(df_test[numeric_cols])

# Step 3: Combine encoded and scaled features, drop original categorical columns
df_test = pd.concat([df_test.drop(['Age_Group', 'Heart_Rate_Zone', 'Sex'], axis=1), encoded_df], axis=1)


predictions = modelCat.predict(df_test)
predictions_df = pd.DataFrame(predictions, columns=['Calories'])
result = pd.concat([y, predictions_df], axis=1)
result.to_csv('submission.csv', index=False)




