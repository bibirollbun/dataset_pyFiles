!pip install rdkit-pypi


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


#import libraries:

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
from scipy.stats import linregress

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split,  KFold
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor


from sklearn.model_selection import RandomizedSearchCV
import optuna

from imblearn.over_sampling import SMOTE

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

# Import regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



import warnings
warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  

pd.set_option('display.max_columns', None)


train = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
test = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')


train.head()


test.head()


test.drop('T80', axis=1, inplace=True)


train.shape


test.shape


print(list(train.columns))


train.columns[train.isnull().any()]


test.columns[test.isnull().any()]


train.head()


#Set index
train.set_index('Batch_ID', inplace=True)
test.set_index('Batch_ID', inplace=True)


# Convert a SMILES string into an RDKit molecule

train['mol'] = train['Smiles'].apply(lambda s: Chem.MolFromSmiles(s))
test['mol'] = test['Smiles'].apply(lambda s: Chem.MolFromSmiles(s))
train.head()


train['Fsp3'] = train['mol'].apply(lambda m: rdMolDescriptors.CalcFractionCSP3(m))
train['BertzCT'] = train['mol'].apply(lambda m: Descriptors.BertzCT(m))
train['Chi0'] = train['mol'].apply(lambda m: Descriptors.Chi0(m))
train['Chi1'] = train['mol'].apply(lambda m: Descriptors.Chi1(m))
train['NumAromaticRings'] = train['mol'].apply(lambda m: Descriptors.NumAromaticRings(m))
train['NumAliphaticRings'] = train['mol'].apply(lambda m: Descriptors.NumAliphaticRings(m))
train['FractionCSP3'] = train['mol'].apply(lambda m: Descriptors.FractionCSP3(m)) 
train['HeavyAtomCount'] = train['mol'].apply(lambda m: Descriptors.HeavyAtomCount(m)) 
            
           


test['Fsp3'] = test['mol'].apply(lambda m: rdMolDescriptors.CalcFractionCSP3(m))
test['BertzCT'] = test['mol'].apply(lambda m: Descriptors.BertzCT(m))
test['Chi0'] = test['mol'].apply(lambda m: Descriptors.Chi0(m))
test['Chi1'] = test['mol'].apply(lambda m: Descriptors.Chi1(m))
test['NumAromaticRings'] = test['mol'].apply(lambda m: Descriptors.NumAromaticRings(m))
test['NumAliphaticRings'] = test['mol'].apply(lambda m: Descriptors.NumAliphaticRings(m))
test['FractionCSP3'] = test['mol'].apply(lambda m: Descriptors.FractionCSP3(m)) 
test['HeavyAtomCount'] = test['mol'].apply(lambda m: Descriptors.HeavyAtomCount(m)) 


train.head()




def plot_numerical_vs_T80(df, target='T80'):
    """
    Plots scatter plots of each numerical feature against the target 'T80', 
    with a regression line and annotation of key regression diagnostics 
    (slope, intercept, R², and p-value) to help assess if the regression makes sense.
    """
    # Get numerical features (excluding the target)
    numerical_features = df.select_dtypes(include='number').columns.drop(target, errors='ignore')

    for feature in numerical_features:
        plt.figure(figsize=(6, 4))
        
        # Drop any missing values in the feature and target before regression calculation
        data = df[[feature, target]].dropna()
        
        # Compute regression diagnostics using SciPy linregress
        regression_stats = linregress(data[feature], data[target])
        slope = regression_stats.slope
        intercept = regression_stats.intercept
        r_squared = regression_stats.rvalue ** 2
        p_value = regression_stats.pvalue
        
        # Create the regplot
        sns.regplot(x=feature, y=target, data=data, scatter=True, scatter_kws={'alpha': 0.5})
        
        # Annotate plot with regression statistics
        annotation = (f'Slope: {slope:.2f}\n'
                      f'Intercept: {intercept:.2f}\n'
                      f'$R^2$: {r_squared:.2f}\n'
                      f'$p$-value: {p_value:.2e}')
        
        plt.text(0.05, 0.95, annotation, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        
        plt.title(f'{feature} vs {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()



plot_numerical_vs_T80(train, target='T80')


sns.histplot(train['T80'], kde = True, bins=30)
plt.title('Distribution of T80 in train Data')
plt.show()


train['T80'].describe()


categorical_columns = train.select_dtypes(include=['object', 'category']).columns

# Print categorical columns
print(categorical_columns)


train.columns


# Drop Columns:

train.drop(['Smiles', 'mol'], axis =1, inplace =True)
test.drop(['Smiles', 'mol'], axis =1, inplace =True)


target = "T80"
X = train.drop(columns= target)


y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train.shape


X_train.shape




# Define a dictionary of candidate regressors
regressors = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.1, max_iter=10000, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}

results = {}

print("Evaluating regressors:")

# Iterate through each regressor
for name, reg in regressors.items():
    # Create a pipeline to scale features and fit the regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', reg)
    ])
    
    # Perform 5-fold cross validation on training data (scoring returns negative MSE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.mean(np.sqrt(-cv_scores))
    
    # Fit the pipeline on the entire training data and predict on the test set
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    results[name] = {"CV_RMSE": cv_rmse, "Test_RMSE": test_rmse}
    print(f"{name}: CV RMSE = {cv_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

# Select the best regressor based on test RMSE
best_regressor_name = min(results, key=lambda x: results[x]["Test_RMSE"])
best_metrics = results[best_regressor_name]
print("\nBest regressor based on test RMSE:")
print(f"{best_regressor_name}: Test RMSE = {best_metrics['Test_RMSE']:.4f}, CV RMSE = {best_metrics['CV_RMSE']:.4f}")



# Define the final XGBoost model with the optimal hyperparameters
final_xgb = xgb.XGBRegressor(
    n_estimators=100,       # Use the best number of trees found
    max_depth=6,            # Use the best depth from your hyperparameter tuning
    learning_rate=0.1,      # Adjust based on your best validation performance
    random_state=42
)

# Re-fit the model on the entire training dataset
final_xgb.fit(X_train, y_train)

# Use the re-fitted model to predict on the test data
preds = final_xgb.predict(X_test)

# Calculate the RMSE
final_rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Final XGBoost RMSE on test set:", final_rmse)


# # # Assuming X and y are your features and numerical target
# # kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # fold_best_iterations = []
# # oof_train_rmse = []
# # oof_val_rmse = []

# # # Loop through folds and use early stopping to capture the optimal boosting rounds
# # for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
# #     X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
# #     y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
# #     # Define the XGBoost regressor with early stopping enabled
# #     model = XGBRegressor(
# #         n_estimators=1000,
# #         learning_rate=0.03,
# #         max_depth=10,
# #         min_child_weight=4,
# #         colsample_bytree=0.66,
# #         subsample=0.9,
# #         gamma=1.6,
# #         reg_alpha=5.5,
# #         reg_lambda=8,
# #         eval_metric="rmse",
# #         early_stopping_rounds=100,
# #         random_state=42,
# #         tree_method="hist",
# #         verbosity=0
#     )
    
#     # Fit the model using the validation fold for early stopping
#     model.fit(
#         X_train_fold, y_train_fold,
#         eval_set=[(X_val_fold, y_val_fold)],
#         verbose=False
#     )
    
#     # Record the best iteration from early stopping
#     best_iter = model.best_iteration
#     fold_best_iterations.append(best_iter)
    
#     # Compute RMSE on training and validation folds
#     train_rmse = np.sqrt(mean_squared_error(y_train_fold, model.predict(X_train_fold)))
#     val_rmse = np.sqrt(mean_squared_error(y_val_fold, model.predict(X_val_fold)))
#     oof_train_rmse.append(train_rmse)
#     oof_val_rmse.append(val_rmse)
    
#     print(f"Fold {fold}: Best Iteration = {best_iter}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

# # Calculate the mean best iteration from all folds
# mean_best_iter = int(np.mean(fold_best_iterations))
# print(f"\nMean Best Iteration from CV: {mean_best_iter}")

# # Optionally, you can also review average RMSEs across folds:
# print(f"Average Train RMSE: {np.mean(oof_train_rmse):.4f}")
# print(f"Average Validation RMSE: {np.mean(oof_val_rmse):.4f}")

# # Retrain final model on the entire dataset using the average best iteration
# final_model = XGBRegressor(
#     n_estimators=mean_best_iter,
#     learning_rate=0.03,
#     max_depth=10,
#     min_child_weight=4,
#     colsample_bytree=0.66,
#     subsample=0.9,
#     gamma=1.6,
#     reg_alpha=5.5,
#     reg_lambda=8,
#     eval_metric="rmse",
#     random_state=42,
#     tree_method="hist",
#     verbosity=0
# )

# final_model.fit(X_train, y_train)




test.shape


# Initialize the SHAP explainer

explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer(X_test) 

#  Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")


#test_filtered = test[selected_features]


test_pred = final_xgb.predict(test)


submission = pd.DataFrame({'Batch_ID': test.index, 'T80': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)




