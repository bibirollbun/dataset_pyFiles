# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

playground_series_s5e2_path = kagglehub.competition_download('playground-series-s5e2')

print('Data source import complete.')



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import time
import numpy as np
import pandas as pd
from scipy.stats import skew, chisquare, kruskal, ks_2samp, chi2_contingency

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv')
train_extra_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv')


perform_adversarial_val = False

if perform_adversarial_val:
    # Concatenate the datasets and drop the 'Price' column.
    # data = pd.concat([train_df, train_extra_df], ignore_index=True).drop('Price', axis=1)
    data = pd.concat([train_df, train_extra_df], ignore_index=True)

    cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    num_cols = ['Weight Capacity (kg)', 'Price']

    # Create the target variable: 0 for train_df, 1 for train_df_additional.
    y = np.concatenate([np.zeros(len(train_df)), np.ones(len(train_extra_df))])
    X = data[cat_cols + num_cols]

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('to_string', FunctionTransformer(lambda x: x.astype(str))),
        ('encoder', OrdinalEncoder())
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=0))
    ])

    cv_preds = cross_val_predict(pipeline, X, y, cv=5, n_jobs=-1, method='predict_proba')
    roc_auc = roc_auc_score(y, cv_preds[:, 1])
    print(f"ROC-AUC score: {roc_auc:.3f}")


train_df = pd.concat([train_df, train_extra_df], ignore_index=True)


train_df.info()


train_extra_df.info()


test_df.info()


price_skewness = skew(train_df['Price'])
print(f"Skewness of Price: {price_skewness:.4f}\n")

mean_price = train_df['Price'].mean()
median_price = train_df['Price'].median()
print(f"Mean Price: {mean_price:.2f}")
print(f"Median Price: {median_price}\n")

# Create bins for uniform comparison
price_counts, bin_edges = np.histogram(train_df['Price'], bins=10)
expected_counts = [len(train_df) / 10] * 10  # Expected counts for uniform distribution
chi_stat, p_value = chisquare(price_counts, f_exp=expected_counts)
print(f"Chi-Square Statistic: {chi_stat}, p-value: {p_value:.6f}\n")


fig, axes = plt.subplots(1, 2, figsize=(14, 4))  # 1 row, 2 columns

# Plot the first histogram with 20 bins
sns.histplot(train_df['Price'], bins=20, ax=axes[0])
axes[0].set_title("Distribution of Price (20 Bins)")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")

# Plot the second histogram with 100 bins
sns.histplot(train_df['Price'], bins=100, ax=axes[1])
axes[1].set_title("Distribution of Price (100 Bins)")
axes[1].set_xlabel("Price")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# Define the mapping for Size conversion
size_mapping = {"Small": 0, "Medium": 1, "Large": 2}

# Convert the Size column using the mapping
train_df["Size"] = train_df["Size"].map(size_mapping)


numeric_features = ['Price', 'Size', 'Weight Capacity (kg)', 'Compartments']

num_features = len(numeric_features)
cols = 2
rows = (num_features // cols) + (num_features % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    bins = 20 if train_df[feature].nunique() > 20 else train_df[feature].nunique()
    sns.histplot(train_df[feature], bins=bins, palette="viridis", ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}", fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel("Frequency", fontsize=12)

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Plot correlation matrix
correlation_matrix = train_df[numeric_features].corr()
plt.figure(figsize=(4, 2))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation with Price")
plt.show()


train_df.groupby('Size').mean('Price')


categorical_features = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Define the number of rows and columns for subplots
num_features = len(categorical_features)
cols = 3
rows = (num_features // cols) + (num_features % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    value_counts = train_df[feature].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f"{feature} Distribution", fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel("Count", fontsize=12)
    if feature == 'Brand':
        axes[i].tick_params(axis='x', rotation=15)

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


print("\n#########################################")
print("Checking correlations with Price")
print("#########################################\n")
fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.boxplot(x=train_df[feature], y=train_df['Price'], ax=axes[i])
    axes[i].set_title(f"Price Distribution by {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Price")
    if feature == 'Brand':
        axes[i].tick_params(axis='x', rotation=15)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


subset_cols = [
    'Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
   'Waterproof', 'Style', 'Color',
]
train_df[train_df.duplicated(subset=subset_cols)].sort_values(by=subset_cols).head(20)


train_df.columns = [
    'id', 'brand', 'material', 'size', 'compartments',
    'laptop_compartment', 'is_waterproof', 'style', 'color',
    'weight_capacity', 'price'
]
test_df.columns = [
    'id', 'brand', 'material', 'size', 'compartments',
    'laptop_compartment', 'is_waterproof', 'style', 'color',
    'weight_capacity',
]

train_df['size'] = train_df['size'].astype('category')
train_df['compartments'] = train_df['compartments'].astype('category')

# Identify numerical and categorical columns
# num_cols = train_df.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
num_cols = ['weight_capacity']
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert all to be the same type
train_df[cat_cols] = train_df[cat_cols].astype('category')
test_df[cat_cols] = test_df[cat_cols].astype('category')
# train_df[cat_cols] = train_df[cat_cols].astype(str)
# test_df[cat_cols] = test_df[cat_cols].astype(str)


results = {}

# Kolmogorov-Smirnov Test for numerical features
for col in num_cols:
    stat, p_value = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
    results[col] = {"test": "Kolmogorov-Smirnov", "statistic": stat, "p_value": p_value}

# Chi-Square Test for categorical features
for col in cat_cols:
    train_counts = train_df[col].value_counts()
    test_counts = test_df[col].value_counts()
    common_categories = list(set(train_counts.index) & set(test_counts.index))

    if common_categories:
        train_freqs = train_counts.loc[common_categories].values
        test_freqs = test_counts.loc[common_categories].values

        # Ensure both arrays have the same shape
        contingency_table = np.array([train_freqs, test_freqs])
        stat, p_value, _, _ = chi2_contingency(contingency_table)
        results[col] = {"test": "Chi-Square", "statistic": stat, "p_value": p_value}

results_df = pd.DataFrame.from_dict(results, orient="index")
results_df


def cross_validate_models(models, X, y, kf, verbose=True):
    """
    Perform cross-validation on multiple models and compute RMSE.

    Parameters:
        models (dict): Dictionary of models to evaluate.
        X (DataFrame): Feature dataset.
        y (Series): Target variable.
        kf (KFold): KFold cross-validation splitter.
        verbose (bool): If True, prints timing information.

    Returns:
        DataFrame: RMSE scores for each model across folds.
    """
    model_scores = {name: [] for name in models.keys()}

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        if verbose:
            print(f"Starting Fold {fold}...")
        fold_start_time = time.time()

        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        for name, model in models.items():
            model_start_time = time.time()

            if name == "Median":
                median_prediction = np.median(y_train)
                y_pred = np.full_like(y_valid, fill_value=median_prediction, dtype=np.float64)
            elif name == "LightGBM":
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
                fit_model = lgb.train(model, train_data, num_boost_round=100, valid_sets=[valid_data])
                y_pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)

            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            model_scores[name].append(rmse)

            if verbose:
                print(f"{name} Model - Fold {fold} - Training & Prediction time: {time.time() - model_start_time:.2f} seconds")

        if verbose:
            print(f"Total time for Fold {fold}: {time.time() - fold_start_time:.2f} seconds")
            print("-" * 50)

    return pd.DataFrame(model_scores)


# Set up cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
target = 'price'
X = train_df.drop(columns=[target])
y = train_df[target]

# Define the models to test
models = {
    "Median": None,  # Baseline: predict the median of the training target
    "LightGBM": {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'force_row_wise': True  # Avoids auto-detection warnings
    }
    # "RandomForest": RandomForestRegressor(random_state=42),
}


run_cv = False

if run_cv:
    result_df = cross_validate_models(models, X, y, kf, verbose=False)
    summary_df = pd.DataFrame({
        "Mean RMSE": result_df.mean(),
        "Std RMSE": result_df.std()
    })

    # Perform Kruskal-Wallis H-test (non-parametric test) to compare the means of the models
    stat, p_value = kruskal(*result_df.T.values)

    # Display the test results
    test_results = pd.DataFrame({
        "Test Statistic": [stat],
        "P-Value": [p_value]
    })
    display(test_results)


# 39.10
# params_1 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 716, 'max_depth': 8, 'learning_rate': 0.026600305790036135, 'n_estimators': 1311, 'min_child_samples': 92, 'subsample': 0.2860954620455601, 'colsample_bytree': 0.3307114800766788, 'max_bin': 1000, 'reg_alpha': 0.017324668742953995, 'reg_lambda': 0.00034707410201845966, 'min_split_gain': 0.36342947391894276}

# 38.98515
# params_1 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 1170, 'max_depth': 10, 'learning_rate': 0.053781528942660674, 'n_estimators': 9016, 'min_child_samples': 40, 'subsample': 0.24461471851870792, 'colsample_bytree': 0.30891508938487233, 'max_bin': 120769, 'reg_alpha': 0.9635222561390919, 'reg_lambda': 0.007095148677467161, 'min_split_gain': 0.5429373194558597}

params_1 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 1170, 'max_depth': 10, 'learning_rate': 0.04, 'n_estimators': 11500, 'min_child_samples': 40, 'subsample': 0.22, 'colsample_bytree': 0.3, 'max_bin': 120000, 'reg_alpha': 1.4, 'reg_lambda': 0.0, 'min_split_gain': 0.5}

# params_2 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 1370, 'max_depth': 8, 'learning_rate': 0.0025, 'n_estimators': 10441, 'min_child_samples': 30, 'subsample': 0.2553813742814906, 'colsample_bytree': 0.3323979004842727, 'max_bin': 100769, 'reg_alpha': 0.0723607053959506, 'reg_lambda': 0.0456838041490431, 'min_split_gain': 0.50489645757472}

# params_1 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 705, 'max_depth': 5, 'learning_rate': 0.010603219695720537, 'n_estimators': 2565, 'min_child_samples': 37, 'subsample': 0.9063025615857419, 'colsample_bytree': 0.3322992215031262, 'max_bin': 819, 'reg_alpha': 1.2510345446729882, 'reg_lambda': 1.1649121214242006, 'min_split_gain': 0.6450731559789553}
# params_3 = {'seed': 42, 'verbose': -1, 'boosting_type': 'gbdt', 'num_leaves': 297, 'max_depth': 7, 'learning_rate': 0.1161752134912203, 'min_child_samples': 75, 'subsample': 0.6997400487564183, 'colsample_bytree': 0.6203560982160392, 'max_bin': 460, 'reg_alpha': 0.00713742020731775, 'reg_lambda': 1.437057542430506e-06, 'min_split_gain': 0.34226761546860285}

# params_1 = {'boosting_type': 'gbdt', 'num_leaves': 300, 'max_depth': 4, 'learning_rate': 0.010893807998866557, 'n_estimators': 1256, 'min_child_samples': 144, 'subsample': 0.6116373487174114, 'colsample_bytree': 0.5220182279562005, 'max_bin': 100, 'reg_alpha': 1.928625989364237, 'reg_lambda': 2.9294864365688814e-05, 'min_split_gain': 0.4741215513650401}
# params_2 = {'boosting_type': 'gbdt', 'num_leaves': 256, 'max_depth': 3, 'learning_rate': 0.004509104775809521, 'n_estimators': 1312, 'min_child_samples': 85, 'subsample': 0.6546559213149051, 'colsample_bytree': 0.5008446769889494, 'max_bin': 495, 'reg_alpha': 0.02582766894019649, 'reg_lambda': 3.421542329015424e-07, 'min_split_gain': 0.3198124843507867}
# params_3 = {'boosting_type': 'gbdt', 'num_leaves': 232, 'max_depth': 4, 'learning_rate': 0.002435037313492472, 'n_estimators': 2171, 'min_child_samples': 85, 'subsample': 0.855588319033977, 'colsample_bytree': 0.5260729208527882, 'max_bin': 122, 'reg_alpha': 0.4714309053965253, 'reg_lambda': 1.5053458450304663e-06, 'min_split_gain': 0.6253173302203185}

params_list = [params_1]
# params_list = [params_1, params_2]
# params_list = [params_1, params_2, params_3]
# params_list = [params_2, params_3]


model_features = [
    'weight_capacity', 'color', 'compartments', 'brand', 'material', 'is_waterproof'
]
# Store predictions from multiple models
predictions = []

# Train models with different parameter sets
for i, params in enumerate(params_list, start=1):
    print(f"Training model {i} of {len(params_list)}")
    train_data = lgb.Dataset(X[model_features], label=y)
    fit_model = lgb.train(params, train_data, num_boost_round=300)
    y_pred = fit_model.predict(test_df[model_features], num_iteration=fit_model.best_iteration)
    predictions.append(y_pred)

submit_df = test_df[['id']].copy()
submit_df['Price'] = np.mean(predictions, axis=0) # Average the predictions
submit_df.to_csv('submission.csv', index=False)
print(f"Submission file saved as submission.csv\n")
submit_df.head(5)

