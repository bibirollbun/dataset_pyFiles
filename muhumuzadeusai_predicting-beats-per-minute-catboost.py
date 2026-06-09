# IMPORT BASE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Import Datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# List of both datasets
datasets = [('TRAIN DATASET', train_data), ('TEST DATASET', test_data)]

print("="*40 + "TRAIN DATA" + "="*40)
display(train_data.head())
print("="*40 + "TEST DATA" + "="*40)
display(test_data.head())


# Dataset info
print("="*40 + "INFO (TRAIN DATA)" + "="*40)
display(train_data.info())
print("="*40 + "INFO (TEST DATA)" + "="*40)
display(test_data.info())


# Summary stats
print("="*40 + "SUMMARY STATISTICS (TRAIN DATA)" + "="*40)
display(train_data.describe())
print("="*40 + "SUMMARY STATISTICS (TEST DATA)" + "="*40)
display(test_data.describe())


def eda_function(datasets_list: list, base_fig_size: tuple,  palette: str, kde: bool=True, target_var: str='y'):
    """
    This function 'just like its name suggests' performs EDA; creates histograms and boxplots side-by-side
    for each individual numerical variable

    datasets_list: Takes a list of named tuples of available DataFrames (the training and test datasets)
    base_fig_size: Takes a tuple for the dimensions of the plots (also makes sure the diagram sizes are proportional
    to the number of columns in each canvas)
    palette: A string of the desired colour(not colorğŸ˜…) palette
    kde: Boolean value for whether kde lines should appear on histograms
    """
    for name, df in datasets_list:
        df = df.copy() # Use duplicates to avoid modification of the original datasets
        df = df.drop(['id'], axis=1) # Remove the useless id variableğŸ˜’

        # Identify numerical and categorical columns
        num_list = df.select_dtypes(include=np.number).columns.tolist() # returns something like ['RhythmScore', 'AudioLoudness'...]

        # NUMERICAL VARIABLES
        n_rows_num = len(num_list)
        fig_height_num = base_fig_size[1] * n_rows_num
        fig, axes = plt.subplots(n_rows_num, 2, figsize=(base_fig_size[0], fig_height_num))

        if n_rows_num == 1:
            axes = np.expand_dims(axes, axis=0)  # Ensure 2D

        for i, var_name in tqdm(enumerate(num_list), total=n_rows_num, desc=f"Plotting {name}'s numerical variables"):
            if target_var in df.columns:
                sns.histplot(data=df, x=var_name, ax=axes[i, 0], kde=kde, palette=palette)
                sns.boxplot(data=df, y=var_name, ax=axes[i, 1], palette=palette)
            else:
                sns.histplot(data=df, x=var_name, ax=axes[i, 0], kde=kde, palette=palette)
                sns.boxplot(data=df, x=var_name, ax=axes[i, 1], palette=palette)

            axes[i, 0].set_title(f'Distribution of {var_name}', fontsize=16, weight='bold')
            axes[i, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
            axes[i, 1].set_title(f'Boxplot of {var_name}', fontsize=16, weight='bold')
            axes[i, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.subplots_adjust(hspace=0.4)
        plt.show()

        
eda_function(
    datasets_list=datasets,
    base_fig_size=(18, 7),
    palette='Spectral'
)


# CORRELATION MATRIX
def corr_charts(dataset_list):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    axes = axes.flatten()
    for i, (name, dataset) in enumerate(dataset_list):  
        dataset = dataset.drop("id", axis=1)
        sns.heatmap(
            dataset.corr(),
            annot=True,
            fmt=".2f",
            cmap='viridis',
            ax=axes[i]
        )
        axes[i].set_title(f'Correlation Matrix of {name}', fontsize=16, weight='bold')
    plt.tight_layout()
        
        
corr_charts(datasets)


# Import libraries for this section
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


train_data = datasets[0][1].drop('id', axis=1)
test_data = datasets[1][1]

X, y = train_data.iloc[:, :9], train_data.iloc[:, 9:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Transform X_train and X_test
X_train_v2 = StandardScaler().fit_transform(X_train)
X_test_v2 = StandardScaler().fit_transform(X_test)

# Transform the test dataset (for model inference)
test_data_v2 = StandardScaler().fit_transform(test_data.drop('id', axis=1))


# Importing libraries
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold


# CATBOOSTğŸ�ˆ
def catboost_objective(trial):
    # Define the parameter space for catboost
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
    }
    # Define The Cross-Validation Loop
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores= []

    # For each fold, slice X_train_v2 & y_train by the indices
    for train_index, val_index in cv.split(X_train_v2, y_train):
        x_train_fold = X_train_v2[train_index]
        x_val_fold = X_train_v2[val_index]

        y_train_fold = y_train.iloc[train_index].values
        y_val_fold = y_train.iloc[val_index].values

        model = CatBoostRegressor(
            **params,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            task_type="GPU",
            devices="0"
        )

        # Fit the model
        model.fit(x_train_fold, y_train_fold,
                 eval_set=[(x_val_fold, y_val_fold)],
                 early_stopping_rounds=150,
                 use_best_model=True,
                 verbose=False
                 )
        preds = model.predict(x_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)



study = optuna.create_study(direction="minimize", sampler=TPESampler(n_startup_trials=30, seed=42))
study.optimize(catboost_objective, n_trials=100)

# Best Parameters
catboost_best_params = study.best_params
display(catboost_best_params)


# Fit the model with the best parameters

# ğŸ§ª Best parameters obtained from the previous optuna job
catboost_best_params = {
'learning_rate': 0.010078651950113877,
'depth': 7,
'l2_leaf_reg': 1.677314381456462,
'border_count': 192,
'random_strength': 1.364060608089018,
'bagging_temperature': 0.27771449415939964}

catboost_regressor = CatBoostRegressor(
    **catboost_best_params,
    random_seed=42,
    task_type="GPU",
    verbose=False
)

catboost_regressor.fit(X_train_v2, y_train)
preds = catboost_regressor.predict(X_test_v2)

print(f"RMSE Score: {np.sqrt(mean_squared_error(y_test, preds))}")


submission_preds_cb = catboost_regressor.predict(test_data_v2) # Preds on the test dataset

submission_cb = pd.DataFrame({
    'id': test_data['id'].values,
    'BeatsPerMinute': submission_preds_cb
})

display(submission_cb)
submission_cb.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submitted")




