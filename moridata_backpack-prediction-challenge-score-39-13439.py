# Essential Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# ML
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


import optuna


# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")


# Load the train and test datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Add a 'dataset' column to distinguish between train and test data
train['dataset'] = 'train'
test['dataset'] = 'test'

# Concatenate the train and test datasets and reset the index
df = pd.concat([train, test], axis=0).reset_index(drop=True)


df.head()


# Data Exploration
df.shape


df.info()


df.describe()


df.columns


#Identifying Missing Values
# Count missing values per column
df.isnull().sum()


# Percentage of missing values
df.isnull().mean() * 100


# Get the list of numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


#Data Cleaning
# Fill numerical columns with mean
df[['Compartments', 'Weight Capacity (kg)']] = df[['Compartments', 'Weight Capacity (kg)']].fillna(df[['Compartments', 'Weight Capacity (kg)']].mean())

# Fill categorical columns with mode correctly
# df[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']] = df[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']].apply(lambda x: x.fillna(x.mode()[0]))
df[['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']] = df[['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']].apply(lambda x: x.fillna('unknown'))
df['Size'] = df['Size'].fillna(df['Size'].mode()[0])


#Basic Visual Exploration
# Histograms - Great for understanding the distribution of numeric columns.
import matplotlib.pyplot as plt
df[['Compartments', 'Weight Capacity (kg)', 'Price']].hist(bins=30)
plt.title('Distribution of some_numeric_col')
plt.show()


# List of numerical features you want to plot
num_features = ['Compartments', 'Weight Capacity (kg)', 'Price']

# Create boxplots
plt.figure(figsize=(10, 5))
for i, col in enumerate(num_features):
    plt.subplot(1, len(num_features), i + 1)  # Adjust for multiple columns
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()



# Correlation Heatmap - Identify highly correlated features.
plt.figure(figsize=(8,6))
sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm')
plt.show()


df['Brand'].value_counts()


df['Material'].value_counts()


df['Size'].value_counts()


df['Laptop Compartment'].value_counts()


df['Waterproof'].value_counts()


df['Style'].value_counts()


df['Color'].value_counts()


# # Weight per Compartment (Feature Engineering)
# df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']


#ordinal encoding for ordinal
size_order = ['Small', 'Medium', 'Large']  # Define order manually
ordinal_encoder = OrdinalEncoder(categories=[size_order])
df['Size'] = ordinal_encoder.fit_transform(df[['Size']])



from category_encoders import TargetEncoder
# Function for Target Encoding multiple categorical columns
def target_encoding_multiple_columns(df, target_column, categorical_columns, dataset_column='dataset'):
    for col in categorical_columns:
        # Calculate target mean for each category in the column using training data
        target_mean = df[df[dataset_column] == 'train'].groupby(col)[target_column].mean()

        # Map the target mean encoding to the entire dataset
        df[f'{col}_encoded'] = df[col].map(target_mean)

        # Handle missing values for categories not present in training
        df[f'{col}_encoded'].fillna(df[target_column].mean(), inplace=True)

    return df
# Example usage: Apply Target Encoding to multiple columns
categorical_columns = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
df = target_encoding_multiple_columns(df, target_column='Price', categorical_columns=categorical_columns)
# # Drop unnecessary columns after encoding
# columns_to_drop = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
# df.drop(columns=columns_to_drop, errors='ignore', inplace=True)


# onehotencoding
categorical_cols = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)


# df['Price'] = np.log(df['Price'])
df


# Separate train and test datasets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df = df[df['dataset'] == 'test'].drop(columns=['dataset'], errors='ignore')


# Drop unnecessary columns from both datasets
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['Price'], errors='ignore')


# Separate features and target
X = train_df.drop(['Price'], axis=1)
y = train_df['Price']


# X['Size']= X['Size'].astype(int)


X.columns


# List categorical features (CatBoost will auto-encode these)
categorical_features = ['Brand_Jansport', 'Brand_Nike',
       'Brand_Puma', 'Brand_Under Armour', 'Brand_unknown', 'Material_Leather',
       'Material_Nylon', 'Material_Polyester', 'Material_unknown',
       'Laptop Compartment_Yes', 'Laptop Compartment_unknown',
       'Waterproof_Yes', 'Waterproof_unknown', 'Style_Messenger', 'Style_Tote',
       'Style_unknown', 'Color_Blue', 'Color_Gray', 'Color_Green',
       'Color_Pink', 'Color_Red', 'Color_unknown']

# Initialize parameters
n_splits = 5  # Adjust based on dataset size
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# CatBoost model configuration
model = CatBoostRegressor(
    iterations=1000,  # Use early stopping to avoid overfitting
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    verbose=False,  # Set to True for training logs
    cat_features=categorical_features  # Specify categorical columns
)

# Store scores
rmse_scores = []
r2_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Create CatBoost Pool objects for efficiency
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    
    # Train with early stopping
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # Predict and score
    y_pred = model.predict(val_pool)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)

# Average performance
print(f"Mean RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
print(f"Mean R²: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")


feature_importance = model.get_feature_importance(prettified=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importances', y='Feature Id', data=feature_importance)
plt.title('CatBoost Feature Importance')
plt.show()


!pip install optuna


def objective(trial):
    # Hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10),
        'cat_features': categorical_features,
        'verbose': False
    }
    
    # K-Fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        # Predict and score
        y_pred = model.predict(X_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(fold_rmse)
        
        # Prune underperforming trials early
        trial.report(fold_rmse, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(rmse_scores)

# Optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, timeout=3600)  # Adjust based on compute resources

# Results
print("Best trial:")
trial = study.best_trial
print(f"RMSE: {trial.value:.2f}")
print("Params:")
for key, value in trial.params.items():
    print(f"{key}: {value}")


# Split 90% for final training, 10% for validation during early stopping
X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(X, y, test_size=0.1, random_state=42)


best_params = study.best_trial.params
best_params['cat_features'] = categorical_features

final_model = CatBoostRegressor(**best_params)
final_model.fit(
    X_final_train, y_final_train,
    eval_set=(X_final_val, y_final_val),  # Monitor validation performance
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose=100
)


# Preprocess test data
test_features = test_df.drop(columns=['id'], errors='ignore')  # Drop unnecessary columns
test_features = test_features.reindex(columns=X.columns, fill_value=0)  # Align columns with training data


test_df['Price'] = model.predict(test_features)


# Create submission file
submission = test_df[['id', 'Price']]  # Include 'id' and the predicted target column
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")


submission

