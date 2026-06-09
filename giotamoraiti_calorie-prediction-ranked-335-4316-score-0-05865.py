import os
print(os.listdir("/kaggle/input"))

import warnings
# Suppress specific FutureWarnings related to inf handling in Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

import optuna
import scipy.stats as stats

from fastai.tabular.all import *
from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from fastai.callback.tracker import EarlyStoppingCallback


# Adjust the filename based on the competition dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df


test_df


# Save original dataframes
train_df_original = train_df.copy()
test_df_original = test_df.copy()


# See Features
print(train_df.columns)
print("Number of useful features:", len(train_df.columns)-2) # Exclude id and target


print("\nUnique Values for Categorical Features:")
for column in train_df.columns:
    if train_df[column].dtype == 'object':  # Check if the column is categorical
        unique_values = train_df[column].unique()
        print(f"{column}: {unique_values}")

print("\nSmallest and Largest Values for Numerical Features:")
for column in train_df.columns:
    if train_df[column].dtype != 'object':  # Check if the column is numerical
        min_value = train_df[column].min()
        max_value = train_df[column].max()
        print(f"{column}: Min = {min_value}, Max = {max_value}")


# Check for NaN or infinite values in the DataFrame
print(train_df.isna().sum())  # Count of NaN values per column


# Check for NaN or infinite values in the DataFrame
print(test_df.isna().sum())  # Count of NaN values per column


train_df.describe()


train_df.describe(include='object')


# Map 'sex' column: assume 'Male' -> 0 and 'Female' -> 1
sex_mapping = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(sex_mapping)
test_df['Sex'] = test_df['Sex'].map(sex_mapping)

# Copy of train_df for plots
train_df_original_no_id = train_df.copy()
train_df_original_no_id = train_df_original_no_id.drop(['id'], axis=1)

# Drop 'id'
train_df = train_df.drop(['id'], axis=1)
test_df = test_df.drop(['id'], axis=1)

y_train = train_df['Calories']

# Drop 'target'
train_df = train_df.drop(['Calories'], axis=1)


# Get columns
columns = train_df_original_no_id.columns

# Define the number of rows and columns for the grid
rows, cols = 4, 2

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(8, 10))

# Flatten axes array to make indexing easier
axes = axes.flatten()
plt.suptitle('Distribution of Categorical Features')

# Loop through each categorical column and plot
for i, col in enumerate(columns):
    axes[i].bar(train_df_original_no_id[col].value_counts().index, train_df_original_no_id[col].value_counts().values)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    
    # Rotate the x-axis labels for better readability
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


columns = train_df_original_no_id.columns

plt.figure(figsize=(15, 7))
for i, feature in enumerate(columns, 1):
    plt.subplot(2, 4, i)
    sns.histplot(train_df_original_no_id[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# Print sex distribution
print("\nSex Distribution:")
print(train_df_original_no_id['Sex'].value_counts())

# Create subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Countplot of 'Sex'
sns.countplot(x='Sex', data=train_df_original_no_id, ax=axes[0])
axes[0].set_title('Distribution of Sex')

# Boxplot of Calories by Sex
sns.boxplot(x='Sex', y='Calories', data=train_df_original_no_id, ax=axes[1])
axes[1].set_title('Calories by Sex')

# Adjust layout
plt.tight_layout()
plt.show()


# Scatterplot Matrix - Visualize the pairwise relationships between features to understand any correlation
features = train_df_original_no_id.columns
sns.pairplot(train_df_original_no_id[features])
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()


# Correlation Heatmap to understand the linear relationships between numerical variables
plt.figure(figsize=(10, 6))
corr = train_df_original_no_id.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Get categorical features
numerical_features = train_df.select_dtypes(include=['number']).columns

# Remove 'Sex'
numerical_features = numerical_features.drop('Sex', errors='ignore')

# Create subplots
num_plots = len(numerical_features)
fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(10, 2 * (num_plots // 3 + 1)))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each numerical feature
for i, feature in enumerate(numerical_features):
    # Dynamically determine the number of bins based on the feature's range or standard deviation
    range_feature = train_df[feature].max() - train_df[feature].min()
    num_bins = max(5, int(range_feature // 10))  # Adjust the number of bins based on the range (you can change this logic)
    
    # Create bins for the feature values
    bins = np.linspace(train_df[feature].min(), train_df[feature].max(), num_bins)
    
    # Digitize the feature values into bins
    bin_indices = np.digitize(train_df[feature], bins)
    
    # Calculate the mean of y_train for each bin
    bin_means = [y_train[bin_indices == i].mean() for i in range(1, len(bins))]
    
    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot on the respective subplot
    ax = axes[i]
    ax.plot(bin_centers, bin_means, marker='o', linestyle='-', color='b')
    ax.set_title(f"y_train vs {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Calories")
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

train_df = add_feature_cross_terms(train_df, numerical_features)
test_df = add_feature_cross_terms(test_df, numerical_features)

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})

y_train = np.log1p(train_df["Calories"]) # To simulate RMSLE with RMSE
train_df = train_df.drop(columns=["id", "Calories"])
test_df = test_df.drop(columns=["id"])


train_df


y_train


# Initialize the StandardScaler
scaler = StandardScaler()

# Separate the 'Sex' column from the rest
sex_column_train = train_df['Sex']
sex_column_test = test_df['Sex']

# Drop 'Sex' column from the data before scaling
train_df_no_sex = train_df.drop(columns=['Sex'])
test_df_no_sex = test_df.drop(columns=['Sex'])

# Fit and transform the training data (excluding 'Sex')
train_df_scaled = scaler.fit_transform(train_df_no_sex)

# Transform the test data (use the same scaler fitted on train data)
test_df_scaled = scaler.transform(test_df_no_sex)

# Convert the scaled train and test data to DataFrames
train_df_scaled = pd.DataFrame(train_df_scaled, columns=train_df_no_sex.columns)
test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df_no_sex.columns)

# Reattach the 'Sex' column back to the scaled data
train_df_scaled['Sex'] = sex_column_train
test_df_scaled['Sex'] = sex_column_test


train_df_scaled


def evaluate_models(X_train, y_train, random_state=26):
    # Define KFold cross-validation
    cv = KFold(n_splits=2, shuffle=True, random_state=random_state)
    
    # Initialize the models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Bayesian Ridge': BayesianRidge(),

        'XGBoost': xgb.XGBRegressor(),
        'CatBoost': CatBoostRegressor(silent=True),
        'LightGBM': lgbm.LGBMRegressor(verbose=-1),

        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Extra Trees': ExtraTreesRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        
        'K-Nearest Neighbors': KNeighborsRegressor(),
        # 'SVM': SVR(),
        # 'Gaussian Process': GaussianProcessRegressor()
    }

    # Dictionary to store results
    model_scores = {}
    
    # Loop through each model
    for model_name, model in models.items():
        # Perform cross-validation and compute the negative RMSE
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        
        # Convert to RMSE
        rmse_scores = np.sqrt(-scores)
        
        # Convert MSE to positive RMSE
        mean_rmse = np.mean(rmse_scores)  # Compute the mean RMSE across folds
        
        # Print the mean RMSE for this model
        # print(f'{model_name} - Mean RMSE: {mean_rmse:.4f}')
        print(f'Evaluating {model_name}...')

        # Store the result
        model_scores[model_name] = mean_rmse

    # Sort models by RMSE in ascending order (lower RMSE is better)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

    # Print results in descending order (best models first)
    print("\nModels sorted by RMSE:")
    for model, score in sorted_models:
        print(f"{model}: Mean RMSE = {score:.4f}")


# No Feature Engineering
# evaluate_models(train_df_scaled, y_train)


# With Feature Engineering
evaluate_models(train_df_scaled, y_train)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

train_df = add_feature_cross_terms(train_df, numerical_features)
test_df = add_feature_cross_terms(test_df, numerical_features)

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})

train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])


def train_and_evaluate_kfold(df, test_df, dep_var, cat_names, cont_names, config, layers, epochs, loss, metric, n_splits=5, bs=256):
    error_scores = []
    y_pred_test = np.zeros(len(test_df))
    y_oof = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=26)
    
    # Create TabularPandas object
    procs = [Categorify, FillMissing, Normalize]

    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
        print(f"Training fold {fold+1}...")

        # Create TabularPandas using full df (not train_df)
        to = TabularPandas(df, procs=procs, 
                           cat_names=cat_names, 
                           cont_names=cont_names, 
                           y_names=dep_var, 
                           y_block=RegressionBlock(),
                           splits=(list(train_idx), list(valid_idx)))
        
        # Convert to DataLoaders
        dls = to.dataloaders(bs=bs)
        # dls.show_batch()

        # Initialize TabularLearner
        learn = tabular_learner(dls, loss_func=loss, metrics=metric, layers=layers, config=config)
        # print(learn.dls.device) 
        
        # Train model
        learn.fit_flat_cos(epochs, lr=1e-3, div_final=1000000, pct_start=0.15, 
                           cbs=[EarlyStoppingCallback(monitor='rmsle', comp=np.less, patience=25),
                                SaveModelCallback(monitor='rmsle', comp=np.less, fname=f'best_model_fold_{fold+1}')])
        
        # Save plots
        learn.recorder.plot_loss()
        plt.title(f"Fold {fold+1} Loss Curve")
        # plt.savefig(f"fold_{fold+1}_loss_curve.png")
        plt.show()
        plt.close()

        # Reload best model weights
        learn.load(f'best_model_fold_{fold+1}', with_opt=False)
        learn.model.eval()
        
        # Get predictions
        preds, targets = learn.get_preds(dl=dls.valid)
        
        # Save OOF predictions for this fold into the right indices
        y_oof[valid_idx] = preds.numpy().flatten()

        # Compute RMSE
        # error_score = rmsle(preds, targets)
        rmsle_idx = learn.recorder.metric_names.index('rmsle') - 1
        error_score = learn.recorder.log[rmsle_idx]

        print(f"Fold {fold+1} - RMSLE: {error_score:.5f}")
        error_scores.append(error_score)

        # Apply same transforms to test data
        test_df_copy = test_df.copy()
        to_test = to.new(test_df_copy)
        test_dl = learn.dls.test_dl(to_test.items)

        # Get predictions for the test set
        test_preds, _ = learn.get_preds(dl=test_dl)
        y_pred_test += test_preds.numpy().flatten() 

     # Average the predictions from all folds
    y_pred_test /= n_splits
    
    # Compute final RMSE
    mean_error = np.mean(error_scores)
    print(f"Average RMSLE across folds: {mean_error:.5f}")

    # Save OOF and test predictions as .npy files
    np.save('oof_preds_nn.npy', y_oof)
    np.save('y_test_preds_nn.npy', y_pred_test)
    
    return y_pred_test, y_oof


def rmse(inp, targ):
    return mean_squared_error(targ.cpu().numpy(), inp.cpu().numpy(), squared=False)

def rmsle(preds, targets):
    # Clamp predictions to avoid log(0) or negative values
    preds = torch.clamp(preds, min=1e-6)
    targets = torch.clamp(targets, min=1e-6)

    return torch.sqrt(F.mse_loss(torch.log1p(preds), torch.log1p(targets))).item()


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


continuous_columns = train_df.select_dtypes(include=['number']).columns.tolist()
if 'Sex' in continuous_columns:
     continuous_columns.remove('Sex')
continuous_columns.remove('Calories')
continuous_columns


train_df


test_df


# Define target and column types
dep_var = 'Calories'  # Set your target

# Categorical columns
cat_names = ['Sex']  

# Continuous columns (excluding 'Calories' since it's the target)
cont_names = continuous_columns  

# Loss Function
# loss = MSELoss()
# loss = L1Loss()
loss = SmoothL1Loss()

# Evaluation Metrics
# metric_to_monitor = [rmse, rmsle]
metric_to_monitor = [rmsle]

# NN Architecture
config = tabular_config(
    act_cls=nn.GELU(),
    ps=[0.0, 0.0, 0.0, 0.0, 0.0]
)

# Hyperparameters
layers = [32, 64, 128, 64, 32]
epochs = 150
batch = 512

# NN
y_pred_test_nn, y_oof_nn = train_and_evaluate_kfold(train_df, test_df, dep_var, 
                                                    cat_names, cont_names, config, layers, epochs, 
                                                    loss, metric_to_monitor, bs=batch)


# Feature Engineering
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

train_df = add_feature_cross_terms(train_df, numerical_features)
test_df = add_feature_cross_terms(test_df, numerical_features)

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})

y_train = np.log1p(train_df["Calories"]) # To simulate RMSLE with RMSE
train_df = train_df.drop(columns=["id", "Calories"])
test_df = test_df.drop(columns=["id"])


def rmse(preds, targets):
    return np.sqrt(mean_squared_error(preds, targets))
    
def train_and_plot_model(model, X, y, test_df, y_test=None, n_splits=5, random_state=26, model_name="catboost"):
    error_scores = []
    y_pred_test = np.zeros(len(test_df))
    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold+1}...")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        if model_name == "catboost":
            # Catboost
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        elif model_name == "xgboost":
            # XGBoost
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)
        elif model_name == "lgbm":
            # lgbm
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[lgbm.early_stopping(stopping_rounds=250), lgbm.log_evaluation(period=500)])
        else:
            model.fit(X_train, y_train)
        
        preds = model.predict(X_valid)
        error_score = rmse(preds, y_valid)
        print(f"Fold {fold+1} - RMSLE: {error_score:.5f}\n")
        error_scores.append(error_score)
        oof_preds[valid_idx] = preds

        test_preds = model.predict(test_df)
        test_preds = np.expm1(test_preds)
        y_pred_test += test_preds

    # Average predictions from all folds
    y_pred_test /= n_splits
    mean_error = np.mean(error_scores)
    print(f"Average RMSLE across folds: {mean_error:.5f}")

    # Plot diagnostics using OOF predictions
    residuals = y - oof_preds

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # 1. Predicted vs Actual
    axes[0, 0].scatter(y, oof_preds, alpha=0.6, edgecolors='k')
    min_val = min(min(y), min(oof_preds))
    max_val = max(max(y), max(oof_preds))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
    axes[0, 0].set_title("Predicted vs Actual Values")
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")

    # 2. Residuals Plot
    axes[0, 1].scatter(oof_preds, residuals, alpha=0.6, edgecolors='k')
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_title("Residuals vs Predicted")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")

    # 3. Residual Histogram & KDE
    sns.histplot(residuals, kde=True, ax=axes[1, 0], bins=25, color="blue")
    axes[1, 0].set_title("Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals")

    # 4. QQ Plot (Normality Check)
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot of Residuals")

    plt.show()

    if model_name == "catboost":
        feature_importance = model.get_feature_importance()
    else:
        feature_importance = model.feature_importances_

    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title(f'{model_name} Feature Importance')
    plt.show()

    # Convert to original scale
    oof_preds =  np.expm1(oof_preds)

    # Clip values
    oof_preds = np.clip(oof_preds, 1, 314)
    y_pred_test = np.clip(y_pred_test, 1, 314)

    np.save(f"y_test_preds_{model_name}.npy", y_pred_test)
    np.save(f"oof_preds_{model_name}.npy", oof_preds)

    return y_pred_test, oof_preds


catboost_model = CatBoostRegressor(iterations=2500, 
                                   depth=10,
                                   l2_leaf_reg=3,
                                   max_bin=1024, 
                                   verbose=500, 
                                   random_seed=26, 
                                   learning_rate=0.02,
                                   early_stopping_rounds=250, 
                                   loss_function='RMSE', 
                                   eval_metric='RMSE')

test_preds_cb, oof_preds_cb = train_and_plot_model(catboost_model, train_df, y_train, test_df, 
                                                   model_name="catboost")


xgboost_model = xgb.XGBRegressor(
    max_depth=10,
    colsample_bytree=0.75,
    subsample=0.9,
    n_estimators=2500,
    learning_rate=0.02,
    gamma=0.01, 
    max_delta_step=2,
    early_stopping_rounds=250,
    eval_metric="rmse",
    random_state=26)

test_preds_xgb, oof_preds_xgb = train_and_plot_model(xgboost_model, train_df, y_train, test_df, 
                                                     model_name="xgboost")


lgbm_model = lgbm.LGBMRegressor(n_estimators=2500, 
                                learning_rate=0.02, 
                                max_depth=10, 
                                colsample_bytree=0.75,
                                subsample=0.9, 
                                random_state=26, 
                                verbose=-1,
                                eval_metric='rmse',
                                objective='regression')

test_preds_lgbm, oof_preds_lgbm = train_and_plot_model(lgbm_model, train_df, y_train, test_df, model_name="lgbm")


rf_model = RandomForestRegressor(
    n_estimators=250,
    random_state=26)

test_preds_rf, oof_preds_rf = train_and_plot_model(rf_model, train_df, y_train, test_df, 
                                                   model_name="random_forest")


extra_trees_model = ExtraTreesRegressor(
    n_estimators=250,
    random_state=26)

test_preds_et, oof_preds_et = train_and_plot_model(extra_trees_model, train_df, y_train, test_df, 
                                                   model_name="extra_trees")


gradient_boosting_model = GradientBoostingRegressor(
    n_estimators=250,
    random_state=26)

test_preds_gb, oof_preds_gb = train_and_plot_model(gradient_boosting_model, train_df, y_train, test_df, 
                                                   model_name="gradient_boosting")


def generate_predictions_and_save(test_df, predictions, output_filename='predictions.csv'):   
    # Prepare the DataFrame with 'id' and 'Listening_Time_minutes'
    output_df = pd.DataFrame({
        'id': test_df['id'],  # Assuming 'id' starts from 0, adjust if needed
        'Calories': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_filename, index=False)
    print(f'Predictions saved to {output_filename}')

    # Visualize predicted values
    plt.hist(predictions, bins=100)
    plt.title("Test Predictions")
    plt.show()


# For ids
test_df_original = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Save predictions to csv
generate_predictions_and_save(test_df_original, test_preds_cb, output_filename='cb.csv')
generate_predictions_and_save(test_df_original, test_preds_xgb, output_filename='xgb.csv')
generate_predictions_and_save(test_df_original, test_preds_lgbm, output_filename='lgbm.csv')
generate_predictions_and_save(test_df_original, test_preds_rf, output_filename='rf.csv')
generate_predictions_and_save(test_df_original, test_preds_et, output_filename='et.csv')
generate_predictions_and_save(test_df_original, test_preds_gb, output_filename='gb.csv')


# See dataframe with predections
y_pred_df_nn = pd.DataFrame(y_pred_test_nn, columns=["Calories"])
y_pred_df_nn


# Print max and min values
print("Min predicted value:", y_pred_df_nn["Calories"].min())
print("Max predicted value:", y_pred_df_nn["Calories"].max())

# Clip values to a specific range 
y_pred_test_nn = np.clip(y_pred_test_nn, 1, 314)

# Get original test dataframe (for ids extraction)
test_df_original = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Generate predictions and save into a csv file
generate_predictions_and_save(test_df_original, y_pred_test_nn.flatten(), output_filename='NN.csv')


x_train = []
x_test = []
files = []
PATH = "/kaggle/input/calorie-expenditure-predictions/"

print("Loading files...") 
for c in ['catboost','xgboost','lgbm', 'nn', 'random_forest', 'gradient_boosting', 'extra_trees']:
    print(f"=> {c} ")
    oof = np.load(f"{PATH}oof_preds_{c}.npy")
    # IF NOT LOG1P THEN APPLY LOG1P
    if oof.mean()>10: 
        oof = np.log1p(oof)
    x_train.append(oof)
    files.append(f"oof_preds_{c}")

    y_test = np.load(f"{PATH}y_test_preds_{c}.npy")
    # IF NOT LOG1P THEN APPLY LOG1P
    if y_test.mean()>10: 
        y_test = np.log1p(y_test)
    x_test.append(y_test)

x_train = np.stack(x_train).T
print("Our combined OOF have shape:",x_train.shape)

x_test = np.stack(x_test).T
print("Our combined PRED have shape:",x_test.shape)

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
true = np.log1p(train_df["Calories"]) # To simulate RMSLE with RMSE


def compute_metric_rmse(p):
    m = np.sqrt(np.mean( (p-true)**2.0 ) )
    return m

# COMPUTE METRIC FOR EACH OOF
best_score = 40
best_index = -1

for k,name in enumerate( files ):
    s = compute_metric_rmse(x_train[:,k])
    if s < best_score:
        best_score = s
        best_index = k
    print(f'RMSE {s:0.5f} {name}') 
print()
print(f'Best single model is {files[best_index]} with RMSE = {best_score:0.5f}')


import cupy as cp, gc

def multiple_rmse_scores(actual, predicted):
    """
    Computes multiple approximate AUC scores using GPU.
    
    This function calculates K approximate AUC scores simultaneously for a binary classification 
    problem. The implementation does not handle ties in predictions correctly, making it an 
    approximate AUC computation. The function is based on the algorithm outlined in:
    https://github.com/benhamner/Metrics/blob/master/R/R/metrics.r

    Parameters:
    ----------
    actual : cupy.ndarray
        A 1D GPU array of shape (N,), where N is the number of samples. 
        Contains binary values (0 or 1) indicating the true labels.
    
    predicted : cupy.ndarray
        A 2D GPU array of shape (N, K), where K is the number of classifiers.
        Each column contains predicted scores for the corresponding classifier.

    Returns:
    -------
    cupy.ndarray
        A 1D GPU array of shape (K,) containing the AUC scores for each classifier.

    """
    if len(actual.shape)==1: 
        actual = actual[:,cp.newaxis]
    m = cp.sqrt(cp.mean(  (actual-predicted)**2.0,axis=0 ))
    return m


USE_NEGATIVE_WGT = True
MAX_MODELS = 1000
TOL = 1e-7

indices = [best_index]
old_best_score = best_score
print(f'0 We begin with best single model RMSE {best_score:0.5f} from "{files[best_index]}"')

# PREPARE/MOVE VARIABLES TO GPU FOR SPEED UP
x_train2 = cp.array( x_train ) #GPU
best_ensemble = x_train2[:,best_index] # GPU
truth = cp.array( true ) # GPU
start = -0.50
if not USE_NEGATIVE_WGT: start = 0.01
ww = cp.arange(start,0.51,0.01) # GPU
nn1 = len(ww)

# BEGIN HILL CLIMBING
models = [best_index]
weights = []
metrics = [best_score]

for kk in range(1_000_000):

    best_score = 40
    best_index = -1
    best_weight = 0

    # TRY ADDING ONE MORE MODEL
    for k,ff in enumerate(files):
        new_model = x_train2[:,k] # GPU
        m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn1, axis=1) * (1-ww) # GPU
        m2 = cp.repeat(new_model[:, cp.newaxis], nn1, axis=1) * ww # GPU
        mm = m1+m2 # GPU
        new_aucs = multiple_rmse_scores(truth, mm)
        new_score = cp.min(new_aucs).item() # GPU -> CPU
        if new_score < best_score:
            best_score = new_score # CPU
            best_index = k # CPU
            ii = np.argmin(new_aucs).item() # GPU -> CPU
            best_weight = ww[ii].item() # GPU -> CPU
            potential_ensemble = mm[:,ii] # GPU
    del new_model, m1, m2, mm, new_aucs, new_score
    gc.collect()

    # STOPPING CRITERIA
    indices.append(best_index)
    indices = list(np.unique(indices))
    if len(indices)>MAX_MODELS:
        print(f'=> We reached {MAX_MODELS} models')
        indices = indices[:-1]
        break
    if -1*(best_score - old_best_score) < TOL: 
        print(f'=> We reached tolerance {TOL}')
        break

    # RECORD NEW RESULT
    print(kk+1,'New best RMSE',best_score,f'adding "{files[best_index]}"','with weight',f'{best_weight:0.3f}')
    models.append(best_index)
    weights.append(best_weight)
    metrics.append(best_score)
    best_ensemble = potential_ensemble
    old_best_score = best_score


wgt = np.array([1])
for w in weights:
    wgt = wgt*(1-w)
    wgt = np.concatenate([wgt,np.array([w])])
    
rows = []
t = 0
for m,w,s in zip(models,wgt,metrics):
    name = files[m]
    dd = {}
    dd['weight'] = w
    dd['model'] = name
    rows.append(dd)
    t += float( f'{w:.3f}' )

# DISPLAY WEIGHT PER MODEL
df = pd.DataFrame(rows)
df = df.groupby('model').agg('sum').reset_index().sort_values('weight',ascending=False)
df = df.reset_index(drop=True)
df


# SANITY CHECK
print('Ensemble weights sum to',df.weight.sum())


# COMBINE OOF PREDITIONS (using weights from hill climbing)
x_map = {x:y for x,y in zip(files,np.arange(len(files)))}
x_train3 = x_train2.get()
ensemble = x_train3[:, x_map[df.model.iloc[0]] ] * df.weight.iloc[0]
for k in range(1,len(df)):
    ensemble += x_train3[:, x_map[df.model.iloc[k]] ] * df.weight.iloc[k]
m = compute_metric_rmse(ensemble)
print(f'Overall Hill climbing RMSE = {m:0.6f}')

np.save(f'oof_hill_climbing',ensemble)


# COMBINE TEST PREDITIONS (using weights from hill climbing)
x_map = {x:y for x,y in zip(files,np.arange(len(files)))}
pred = x_test[:, x_map[df.model.iloc[0]] ] * df.weight.iloc[0]
for k in range(1,len(df)):
    pred += x_test[:, x_map[df.model.iloc[k]]] * df.weight.iloc[k]


# WRITE SUB TO CSV
sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# CLIP TO TRAIN MIN AND MAX
mn = train_df.Calories.min(); mx = train_df.Calories.max()
sub.Calories = np.clip(np.expm1(pred),mn,mx)

print("Test shape", sub.shape)
print("Test target mean is", sub.Calories.mean())
sub.to_csv(f"submission_hill_climbing.csv",index=False)
sub.head()


plt.hist(sub.Calories,bins=100)
plt.show()


sub = sub['Calories'].to_numpy().squeeze()


sub


def generate_predictions_and_save(test_df, predictions, output_filename='predictions.csv'):   
    # Prepare the DataFrame with 'id' and 'Listening_Time_minutes'
    output_df = pd.DataFrame({
        'id': test_df['id'],  # Assuming 'id' starts from 0, adjust if needed
        'Calories': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_filename, index=False)
    print(f'Predictions saved to {output_filename}')

    # Visualize predicted values
    plt.hist(predictions, bins=100)
    plt.title("Test Predictions")
    plt.show()

generate_predictions_and_save(test_df, sub, output_filename='hill_climbing_cb_xgb_lgbm_nn.csv')

