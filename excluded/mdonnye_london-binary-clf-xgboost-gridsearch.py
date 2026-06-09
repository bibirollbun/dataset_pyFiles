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

# Outlier detection
from scipy.stats import zscore

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Visualization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


df_train = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv", header = None)
trainLabels = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv", header = None)
df_test = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv", header = None)


print(f"Number of rows: {df_train.shape[0]}")
print(f"Number of columns: {df_train.shape[1]}")
print("\nColumn data types:")
print(df_train.dtypes.value_counts())


missing_values = df_train.isnull().sum()
print("Missing values per column:\n", missing_values[missing_values > 0])


numeric_cols_list = df_train.select_dtypes(include='number').columns.tolist()
categorical_cols_list = df_train.select_dtypes(exclude='number').columns.tolist()

print("ğŸ”¢ Numeric Columns:", numeric_cols_list)
print("ğŸ”¤ Categorical Columns:", categorical_cols_list)


# Detect skewness in numerical features
numerical_cols = df_train.select_dtypes(include='number')
skewness = numerical_cols.skew().sort_values(ascending=False)

# Show features with high skewness (absolute skew > 1)
high_skew = skewness[skewness.abs() > 1]
print("Highly skewed features (|skew| > 1):\n", high_skew)


# A z-score tells you how many standard deviations a value is from the mean.
z_scores = df_train.apply(zscore)
# a z-score greater than 3 means the value is extremely rare (in the outer 0.3% of a normal distribution).
# If |z| > 3, it means the value is far from the average, potentially an outlier.
# This threshold is a conventionâ€”some use 2.5 or 4 based on domain knowledge or data sensitivity.
outliers = (z_scores.abs() > 3).sum().sort_values(ascending=False)
print("Outlier counts (|z-score| > 3):\n", outliers[outliers > 3])


threshold_low_variance = 0.01  # <1% variance

# Numerical: Low variance
low_variance = {}
for col in numerical_cols:
    top_freq_ratio = df_train[col].value_counts(normalize=True).values[0]
    if top_freq_ratio > (1 - threshold_low_variance):
        low_variance[col] = top_freq_ratio

# Print results
print("\nğŸ”� Numerical features with low variance:")
print(low_variance)


stats = df_train.describe().T[['min', 'max']]
stats['range'] = stats['max'] - stats['min']
print("Features with value range from highest into lowest:\n", stats.sort_values('range', ascending=False).head())


# Compute correlation matrix
corr_matrix = df_train.corr().abs()  # abs() to catch negative correlations too

# Remove self-correlations
upper = corr_matrix.where(
  np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Set threshold
correlation_threshold = 0.9

# Find feature pairs with high correlation
high_corr_pairs = [
    (col, row, corr_matrix.loc[row, col])
    for row in upper.index
    for col in upper.columns
    if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > correlation_threshold
]

# Print results
print(f"ğŸ”� Highly correlated feature pairs (|corr| > {correlation_threshold}):")
for f1, f2, corr_val in sorted(high_corr_pairs, key=lambda x: -x[2]):
    print(f"{f1} & {f2} â†’ Corr: {corr_val:.3f}")


constant_cols = df_train.loc[:, df_train.nunique() == 1].columns.tolist()
print(f"Constant columns (no variability): {constant_cols}")


# Treat outliers by capping them at Â±3 z-score
# Make a copy to avoid modifying original
df_processed = df_train.copy()

# Identify the outlier columns with more than 3 outliers
outlier_cols = outliers[outliers > 3].index

for col in outlier_cols:
    col_mean = df_train[col].mean()
    col_std = df_train[col].std()
    upper_limit = col_mean + 3 * col_std
    lower_limit = col_mean - 3 * col_std

    # Cap the values outside Â±3 std
    df_processed[col] = np.where(df_train[col] > upper_limit, upper_limit,
                          np.where(df_train[col] < lower_limit, lower_limit, df_train[col]))

print("âœ… Outliers capped at Â±3 z-scores in columns:", list(outlier_cols))


from sklearn.preprocessing import StandardScaler

# Z-score
zscore_scaler = StandardScaler()
df_train = pd.DataFrame(zscore_scaler.fit_transform(df_train), columns=df_train.columns)


train_data = df_train
train_label = trainLabels.to_numpy()
test_data = df_test


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_data, train_label, test_size = 0.30, random_state = 101)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


def get_models():
    return {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=22),
        "Logistic Regression": LogisticRegression(random_state=0, solver='sag'),
        "SVM": SVC(gamma='auto'),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier()
    }


def train_models(models, x_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train.ravel())
        trained_models[name] = model
    return trained_models


def evaluate_models(trained_models, x_test, y_test):
    accuracies = {}
    for name, model in trained_models.items():
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"{name}: {acc:.4f}")
    return accuracies


models = get_models()
trained_original = train_models(models, x_train, y_train)
acc_original = evaluate_models(trained_original, x_test, y_test)


def plot_model_accuracies(accuracies: dict, title: str = "Accuracy of Each Model"):
    """
    Plots a bar chart of model accuracies.

    Parameters:
    - accuracies (dict): A dictionary where keys are model names and values are accuracy scores.
    - title (str): Title for the plot (default: "Accuracy of Each Model").
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
    )

    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Accuracy always between 0 and 1
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_model_accuracies(acc_original, "Original Data Model Accuracies")


from sklearn.decomposition import PCA

pca = PCA(0.85, whiten=True)
pca_train_data = pca.fit_transform(x_train)
pca_test_data = pca.transform(x_test)
print(pca_train_data.shape,'\n')
print(pca_test_data.shape,'\n')

explained_variance = pca.explained_variance_ratio_
print(explained_variance)


trained_pca = train_models(models, pca_train_data, y_train)
acc_pca = evaluate_models(trained_pca, pca_test_data, y_test)


plot_model_accuracies(acc_pca, "PCA Data Model Accuracies")


# Combine both accuracy dictionaries
combined_acc = {
    f"{name} (Original)": acc
    for name, acc in acc_original.items()
}
combined_acc.update({
    f"{name} (PCA)": acc
    for name, acc in acc_pca.items()
})

# Sort the combined results by accuracy in descending order
sorted_combined = sorted(combined_acc.items(), key=lambda x: x[1], reverse=True)

# Get the top 3 models
top_3 = sorted_combined[:3]

# Print the top 3
print("Top 3 Models with Highest Accuracy:")
for name, acc in top_3:
    print(f"{name}: {acc:.4f}")


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create the XGBoost model
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit to the training data
grid_search.fit(x_train, y_train.ravel())

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(x_test, y_test)
print("Test Accuracy with Best Model:", test_accuracy)


# Fitting our model
pred  = grid_search.predict(test_data)
grid_search_pred = pd.DataFrame(pred)
grid_search_pred.index += 1

# FRAMING OUR SOLUTION
grid_search_pred.columns = ['Solution']
grid_search_pred['Id'] = np.arange(1,grid_search_pred.shape[0]+1)
grid_search_pred = grid_search_pred[['Id', 'Solution']]

grid_search_pred.to_csv('Submission.csv',index=False)

