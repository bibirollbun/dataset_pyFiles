# ============================== INSTALL & IMPORT LIBRARIES ==============================

!pip install itables shap lime scikit-learn xgboost --quiet

# =========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.inspection import PartialDependenceDisplay
import shap
import lime
import lime.lime_tabular

from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

# =========================================================================

# Sets the seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# =========================================================================
# Set plot style
sns.set_style('whitegrid')

# # Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# Load dataset (using a sample for efficiency)
reduced_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop('id', axis=1).sample(250_000, random_state = SEED)

# Separate features and target
X = reduced_df.copy()
y = X.pop("y")

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, random_state=SEED, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

# Check splits
for name, dataset in dict(zip(
    ['X_train', 'X_valid', 'X_test'],
    [X_train.shape, X_valid.shape, X_test.shape])).items():
    
    print(name, dataset)


# ============================== PRE-PROCESSING FUNCTION ==============================

def preprocessing(df, target, map_month=True):
    df = df.copy()

    # Map month
    if map_month:
        month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        df['month'] = df['month'].map(month_map) 
    
    # Loans
    df[['housing','loan']] = (df[['housing','loan']] == 'yes').astype(int)

    # Mapping educatin
    education_map = {'unknown'  : 0, 
                     'primary'  : 1, 
                     'secondary': 2, 
                     'tertiary' : 3}
    df['education'] = df['education'].map(education_map)
    
    # Dummify default
    df['default'] = (df['default'] == 'yes').astype(int)
    
    # Make target numeric
    if target in df.columns:
        df[target] = (df[target] == 1).astype(int)
    
    return df


# Process dfs
X_train_processed = preprocessing(X_train, target='y', map_month=False)
X_valid_processed = preprocessing(X_valid, target='y', map_month=False)
X_test_processed = preprocessing(X_test, target='y', map_month=False)

show(X_valid_processed)


# Fit a "Black-Box Model"
# Identify categorical features
categorical_features = X_train_processed.select_dtypes(include=["object"]).columns.tolist()

# Define model
model = CatBoostClassifier(
    loss_function='Logloss',
    cat_features=categorical_features,
    random_seed=SEED
)

# Fit the model
model.fit(X_train_processed, y_train,
          eval_set=(X_valid_processed, y_valid),
          sample_weight = compute_sample_weight(class_weight='balanced',y=y_train),
          early_stopping_rounds=200,
          verbose_eval=200,
          use_best_model=True
         )

# Evaluate
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")


import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_processed)


# Model's baseline log-odds and probability
print(f"Model's baseline log-odds: {explainer.expected_value:.4f}")


def inv_logit(p):
    """
    Computes the inverse logit (sigmoid) function.
    This function transforms a log-odds value (p) back into a probability.
    """
    return np.exp(p) / (1 + np.exp(p))

# Model's baseline log-odds and probability
print(f"Model's baseline log-odds: {explainer.expected_value:.4f} (proba: {inv_logit(explainer.expected_value):.4f})")


# Global Explanation
shap.plots.bar(shap_values)


shap.plots.bar(shap_values.abs.max(0))


# Global feature importance
shap.summary_plot(shap_values)


# Example: SHAP dependence plot
shap.partial_dependence_plot(
    "duration",
    model.predict,
    X_test_processed,
    ice=False,
    model_expected_value=True,
    feature_expected_value=True,
)



# Example: SHAP dependence plot

shap.partial_dependence_plot(
    "balance",
    model.predict,
    X_test_processed,
    ice=False,
    model_expected_value=True,
    feature_expected_value=True,
)



# Define output_df (optional)
output_df = pd.concat([
    X_test.reset_index(drop=True), 
    pd.Series(y_pred, name='label'),
    pd.Series(y_proba, name='score')
    ], axis=1
)

# Find instances for each class
class1_example = output_df.loc[output_df['label'] == 1].first_valid_index()
class0_example = output_df.loc[output_df['label'] == 0].first_valid_index()


# Local explanation: positive outcome
print(f"Explanation for row {class1_example}:\nLABEL={y_pred[class1_example]}, PROBABILITY={y_proba[class1_example]}")

shap.initjs()
display(shap.plots.waterfall(shap_values[class1_example]))
display(shap.plots.force(shap_values[class1_example]))


# Local explanation: negative outcome
print(f"Explanation for row {class0_example}:\nLABEL={y_pred[class0_example]}, PROBABILITY={y_proba[class0_example]}")

shap.initjs()
display(shap.plots.waterfall(shap_values[class0_example]))
display(shap.plots.force(shap_values[class0_example]))


PartialDependenceDisplay.from_estimator(
model,
X_test_processed,
features=[("campaign", "balance")],
kind="average",
grid_resolution=15
)

plt.figure(figsize=(10,6))
plt.show()




