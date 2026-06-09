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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.sample(5)


test.sample(5)


print('The shape of train.csv: ',df.shape)
print('The shape of test.csv: ',test.shape)


def mis_val_table(df):
    mis_val = df.isnull().sum()
    mis_val_per = 100*mis_val/len(df)
    table = pd.concat([mis_val,mis_val_per],axis=1)
    table.columns = ['missing values', '% of total']
    table
    return table.style.background_gradient(cmap='winter')


mis_val_table(df)


mis_val_table(test)


df.info()


# Dropping the id column
df.drop('id', axis=1, inplace=True)


numerical_cols = df.select_dtypes(np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()


def univariate_categorical_graphs(df,col_list):
    for i in col_list:
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        sns.countplot(x=df[i], palette='Pastel1')
        plt.subplot(1,2,2)
        df[i].value_counts().plot(kind= 'pie', autopct= '%.2f')

univariate_categorical_graphs(df,categorical_cols)


def univariate_numerical_graphs(df,col_list):
    for i in col_list:
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        sns.histplot(x=df[i], palette='Pastel2', kde=True, bins=10)
        plt.subplot(1,2,2)
        sns.boxplot(x=df[i], palette='Pastel2')

univariate_numerical_graphs(df,numerical_cols)


most_expensive_brand = df.groupby('Brand')['Price'].agg('mean').reset_index().sort_values(by='Price', ascending=False)
most_expensive_brand.style.background_gradient(cmap='spring')


sns.barplot(x=most_expensive_brand['Price'], y=most_expensive_brand['Brand'], palette='Pastel1')


most_expensive_material = df.groupby('Material')['Price'].agg('mean').reset_index().sort_values(by='Price', ascending=False)
most_expensive_material.style.background_gradient(cmap='spring')


sns.barplot(x=most_expensive_material['Price'], y=most_expensive_material['Material'], palette='spring')


pd.crosstab(df['Material'],df['Waterproof']).style.background_gradient(cmap='spring')


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


def impute_values(df):
    # imputing missing numerical values
    numerical_imputer = SimpleImputer(strategy='median')
    df['Weight Capacity (kg)'] = numerical_imputer.fit_transform(df['Weight Capacity (kg)'].values.reshape(-1,1))

    #imputing missing categorical values
    df['Brand'].fillna(df['Brand'].mode()[0], inplace=True)
    df['Material'].fillna(df['Material'].mode()[0], inplace=True)
    df['Size'].fillna(df['Size'].mode()[0], inplace=True)
    df['Laptop Compartment'].fillna(df['Size'].mode()[0], inplace=True)
    df['Waterproof'].fillna(df['Size'].mode()[0], inplace=True)
    df['Style'].fillna(df['Size'].mode()[0], inplace=True)
    df['Color'].fillna(df['Size'].mode()[0], inplace=True)

# applying our imputing function
impute_values(df)
impute_values(test)



mis_val_table(df)


mis_val_table(test)


def encoding_categories(df):
    # Frequency Encoding for Style and Color
    style_counts = df['Style'].value_counts()
    df['Style'] = df['Style'].map(style_counts)

    color_counts = df['Color'].value_counts()
    df['Color'] = df['Color'].map(color_counts)

    # One-Hot Encoding for Brand, Material, Waterproof, Laptop Compartment
    df = pd.get_dummies(df, columns=['Brand', 'Material', 'Waterproof', 'Laptop Compartment'], drop_first=True, dtype=int)

    # Ordinal Encoding for Size
    valid_sizes = ['Small', 'Medium', 'Large']
    ordinal_encoder = OrdinalEncoder(categories=[valid_sizes])
    df['Size'] = ordinal_encoder.fit_transform(df[['Size']])

    return df

# applying our encoding function
df = encoding_categories(df)
test = encoding_categories(test)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Split the data into X and y
X = df.drop('Price', axis= 1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Bagging": BaggingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

# Train and evaluate models
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    results.append([name, mae, rmse, r2])

# Create a DataFrame to store results
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²"])

styled_df = results_df.style.background_gradient(cmap='cool')

# Display the color-encoded DataFrame
print("Model Performance:")
display(styled_df)

# Plot the results
plt.figure(figsize=(12, 8))

# Bar plot for MAE
plt.subplot(3, 1, 1)
sns.barplot(x="Model", y="MAE", data=results_df, palette='spring')
plt.title("Mean Absolute Error (MAE)")
plt.xticks(rotation=45)

# Bar plot for RMSE
plt.subplot(3, 1, 2)
sns.barplot(x="Model", y="RMSE", data=results_df, palette='spring')
plt.title("Root Mean Squared Error (RMSE)")
plt.xticks(rotation=45)

# Bar plot for R²
plt.subplot(3, 1, 3)
sns.barplot(x="Model", y="R²", data=results_df, palette='spring')
plt.title("R² Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


!pip install optuna


import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

subset_size = 50000
subset_df = df.sample(n=subset_size, random_state=42)

subset_samples = subset_df.drop('Price', axis=1)
subset_labels = subset_df['Price']


# Define the objective function
def objective(trial):
    # Algorithms to be tuned
    classifier_name = trial.suggest_categorical('classifier', ['GB', 'AB', 'LGBM'])
    
    # Create hyperparameter search spaces
    if classifier_name == 'GB':
        # Gradient Boosting hyperparameters
        loss = trial.suggest_categorical('GB_loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
        criterion = trial.suggest_categorical('GB_criterion', ['friedman_mse', 'squared_error'])
        learning_rate = trial.suggest_float('GB_learning_rate', 0.01, 0.5)  
        n_estimators = trial.suggest_int('GB_n_estimators', 100, 300)
        max_depth = trial.suggest_int('GB_max_depth', 3, 10)
        min_samples_split = trial.suggest_int('GB_min_samples_split', 2, 5)
        min_samples_leaf = trial.suggest_int('GB_min_samples_leaf', 1, 3)
        max_features = trial.suggest_int('GB_max_features', 1,10)
        random_state = 42

        model = GradientBoostingRegressor(
            loss=loss,
            criterion=criterion,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
    elif classifier_name == 'AB':
        # AdaBoost hyperparameters
        learning_rate = trial.suggest_float('AB_learning_rate', 0.01, 0.5)  
        n_estimators = trial.suggest_int('AB_n_estimators', 50, 500)
        loss = trial.suggest_categorical('AB_loss', ['linear', 'square', 'exponential'])
        random_state = 42

        model = AdaBoostRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            loss=loss,
            random_state=random_state
        )
    elif classifier_name == 'LGBM':
        # LightGBM hyperparameters
        boosting_type = trial.suggest_categorical('LGBM_boosting_type', ['gbdt', 'dart'])
        num_leaves = trial.suggest_int('LGBM_num_leaves', 31, 127)
        max_depth = trial.suggest_int('LGBM_max_depth', -1, 12)
        learning_rate = trial.suggest_float('LGBM_learning_rate', 0.01, 0.5)
        min_split_gain = trial.suggest_float('LGBM_min_split_gain', 0, 1) 
        min_child_samples = trial.suggest_int('LGBM_min_child_samples', 20, 100)
        subsample = trial.suggest_float('LGBM_subsample', 0.7, 1.0) 
        reg_alpha = trial.suggest_float('LGBM_reg_alpha', 0, 10)  
        random_state = 42

        model = LGBMRegressor(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_split_gain=min_split_gain,
            min_child_samples=min_child_samples,
            subsample=subsample,
            reg_alpha=reg_alpha,
            random_state=random_state
        )

    # Evaluate the model using cross-validation and RMSE
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))  # RMSE scorer
    scores = cross_val_score(model, subset_samples, subset_labels, cv=3, scoring=rmse_scorer)  # 3-fold cross-validation
    rmse = np.mean(scores)  # Average RMSE across folds

    return rmse  # Optuna minimizes this value



study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)


best_params = study.best_params
best_score = study.best_value
print(best_params)
print(best_score)


from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_contour, plot_param_importances

# Optimization history
plot_optimization_history(study).show()


# Slice Plot
plot_slice(study).show()


model = AdaBoostRegressor(
    learning_rate=best_params['AB_learning_rate'],
    n_estimators=best_params['AB_n_estimators'],
    loss=best_params['AB_loss'],
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


test_predictions = model.predict(test.drop(columns=['id']))

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],  # Use the 'id' column from the test data
    'Price': test_predictions  # Predicted prices
})

# Save the submission file to a CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file saved as 'submission.csv'")

submission

