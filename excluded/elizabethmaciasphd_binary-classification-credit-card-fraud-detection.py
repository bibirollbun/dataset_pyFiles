# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s3e4/train.csv')


df.head()


df.isna().sum()


df.corr()


corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidth=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()


sns.pairplot(df[['V1', 'V2']])


for col in df.columns:
    print("column:", col)
    #df[col].hist(bins=50)
    #plt.show()


plt.plot(df["Time"], df["Amount"])
#df['Velocity'] = df['Amount'] / df['Time']
df.head()


# Set number of plots per row
cols = 4
num_cols = len(df.columns)
rows = (num_cols + cols - 1) // cols  # Calculate required rows

# Create subplots with smaller figure size
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

# Plot each boxplot
for i, col in enumerate(df.columns):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}', fontsize=10)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


for col in df.loc[:, 'V1':'V28']:
    lower_limit = df[col].quantile(0.05)
    upper_limit = df[col].quantile(0.95)
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)


# Set number of plots per row
cols = 4
num_cols = len(df.columns)
rows = (num_cols + cols - 1) // cols  # Calculate required rows

# Create subplots with smaller figure size
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

# Plot each boxplot
for i, col in enumerate(df.columns):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}', fontsize=10)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


df['Time'] = (df['Time'] - df['Time'].mean()) / (df['Time'].max() - df['Time'].min()) 
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].max() - df['Amount'].min())
#df['Velocity'] = (df['Velocity'] - df['Velocity'].mean()) / (df['Velocity'].max() - df['Velocity'].min())


X = df.iloc[:, 2:30] 
#X['Velocity'] = df['Velocity']


X.head()


y = df['Class']


y


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(y_test, y_pred)


accuracy


log_model = LogisticRegression(max_iter=1000)


log_model.fit(X_train, y_train)


log_y_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_y_pred)
log_accuracy


from sklearn.metrics import classification_report


print(classification_report(y_test, y_pred))


print(classification_report(y_test, log_y_pred))


df_combined = pd.concat([X, y], axis=1)

majority_class = df_combined[df_combined['Class']==0]
minority_class = df_combined[df_combined['Class']==1]

majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

df_balanced = pd.concat([majority_downsampled, minority_class]).sample(frac=1, random_state=42)


df_balanced.head()


X_balanced = df_balanced.drop('Class', axis=1)
y_balanced = df_balanced['Class']


X_balanced.describe()


X_balanced_train, X_balanced_test, y_balanced_train, y_balanced_test = train_test_split(X_balanced, y_balanced,  stratify=y_balanced, test_size=0.2, random_state=42)


balanced_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
balanced_model.fit(X_balanced_train, y_balanced_train)


y_balanced_pred = balanced_model.predict(X_balanced_test)


balanced_accuracy = accuracy_score(y_balanced_test, y_balanced_pred)
print(balanced_accuracy)
print(classification_report(y_balanced_test, y_balanced_pred))


log_balanced_model = LogisticRegression(max_iter=1000)


log_balanced_model.fit(X_balanced_train, y_balanced_train)


log_y_balanced_pred = log_balanced_model.predict(X_balanced_test)


log_balanced_accuracy = accuracy_score(y_balanced_test, log_y_balanced_pred)
print(log_balanced_accuracy)
print(classification_report(y_balanced_test, log_y_balanced_pred))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


rf.fit(X_balanced_train, y_balanced_train)
rf_balanced_y_pred = rf.predict(X_balanced_test)


rf_balanced_accuracy = accuracy_score(y_balanced_test, rf_balanced_y_pred)
print(rf_balanced_accuracy)
print(classification_report(y_balanced_test, rf_balanced_y_pred))


!pip install shap
import shap
import matplotlib.pyplot as plt


xgb_explainer = shap.TreeExplainer(balanced_model)
xgb_shap_values = xgb_explainer.shap_values(X_balanced_train)


shap.summary_plot(xgb_shap_values, X_balanced_train)


shap.summary_plot(xgb_shap_values, X_balanced_train, plot_type="bar")


X_balanced_train


# X_balanced_train, X_balanced_test, y_balanced_train, y_balanced_test

X_top_shap_balanced_train = X_balanced_train.loc[:,['V3', 'V4', 'V7', 'V19', 'V18']]
X_top_shap_balanced_test = X_balanced_test.loc[:,['V3', 'V4', 'V7', 'V19', 'V18']]


xgb_top_shap_balanced_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_top_shap_balanced_model.fit(X_top_shap_balanced_train, y_balanced_train)
xgb_top_shap_balanced_y_pred = xgb_top_shap_balanced_model.predict(X_top_shap_balanced_test)


xgb_top_shap_balanced_accuracy = accuracy_score(y_balanced_test, xgb_top_shap_balanced_y_pred)
print(xgb_top_shap_balanced_accuracy)
print(classification_report(y_balanced_test, xgb_top_shap_balanced_y_pred))


from sklearn.decomposition import PCA


pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_balanced_train)
X_test_pca = pca.transform(X_balanced_test)


xgb_pca_balanced_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_pca_balanced_model.fit(X_train_pca, y_balanced_train)
xgb_pca_balanced_y_pred = xgb_pca_balanced_model.predict(X_test_pca)


xgb_pca_balanced_accuracy = accuracy_score(y_balanced_test, xgb_pca_balanced_y_pred)
print(xgb_pca_balanced_accuracy)
print(classification_report(y_balanced_test, xgb_pca_balanced_y_pred))


from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
from tqdm import tqdm


# K-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1],
    'n_estimators': [200, 400, 800],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3]
}

# Scorer
scorer = make_scorer(f1_score)

# Manual grid search with progress bar
best_score = -np.inf
best_params = None

for params in tqdm(list(ParameterGrid(param_grid)), desc="Grid Search Progress"):
    model = xgb.XGBClassifier(**params,
                              objective="binary:logistic",
                              eval_metric="logloss",
                              tree_method="hist",
                              random_state=42,
                              n_jobs=-1)
    
    scores = cross_val_score(model, X_balanced_train, y_balanced_train, cv=cv, scoring=scorer, n_jobs=-1)
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        best_model = model


print("Best params:", best_params)
print("Best CV score:", best_score)
#best_model = grid.best_estimator_


best_model.fit(X_balanced_train, y_balanced_train)
y_gs_pred = best_model.predict(X_balanced_test)


balanced_gs_accuracy = accuracy_score(y_balanced_test, y_gs_pred)
print(balanced_gs_accuracy)
print(classification_report(y_balanced_test, y_gs_pred))


import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X_minority: pd.DataFrame, N=1, k=5, random_state=42) -> pd.DataFrame:
    """
    Manual SMOTE implementation.
    X_minority: Minority class samples (numpy array, shape [n_samples, n_features])
    N: Number of synthetic samples per original sample
    k: Number of neighbors
    """
    #np.random.seed(random_state)

    rng = np.random.default_rng(random_state)
    n_minority, n_features = X_minority.shape
    synthetic_samples = []
    
    # Fit nearest neighbors on minority data
    nn = NearestNeighbors(n_neighbors=k).fit(X_minority)
    neighbors = nn.kneighbors(X_minority, return_distance=False)
    
    for i in tqdm(range(n_minority)):
        for _ in range(N):
            # Pick one of the k nearest neighbors (excluding itself)
            neighbor_index = rng.choice(neighbors[i][1:])
            neighbor = X_minority.iloc[neighbor_index].to_numpy()
            sample = X_minority.iloc[i].to_numpy()
            
            # Random step between 0 and 1
            lam = np.random.rand()
            
            # Interpolate between sample and neighbor
            synthetic = sample + lam * (neighbor - sample)
            synthetic_samples.append(synthetic)
    
    return pd.DataFrame(synthetic_samples, columns=X_minority.columns)


df_joined = pd.concat([X, y], axis=1)


df_joined.head(1)


minority_df = df_joined[df_joined['Class']==1]
majority_df = df_joined[df_joined['Class']==0]


minority_df.shape, majority_df.shape


X_minority = minority_df.drop('Class', axis=1)
y_minority = minority_df['Class']


synthetic = smote(X_minority, N=4, k=3)  # 2 synthetic per original
for i in range(3):
    synthetic = smote(synthetic, N=4, k=3)
print("Generated samples:\n", synthetic)


synthetic.head()


synthetic.shape


synthetic['Class'] = 1


synthetic.head()


concatenated_df = pd.concat([synthetic, majority_df], axis=0)


concatenated_df.shape


X_concatenated = concatenated_df.drop('Class', axis=1)
y_concatenated = concatenated_df['Class']


X_concatenated.shape, y_concatenated.shape


X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_concatenated, y_concatenated,  test_size=0.2, random_state=42)


smote_xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
smote_xgb_model.fit(X_smote_train, y_smote_train)
smote_xgb_y_pred = smote_xgb_model.predict(X_smote_test)


smote_accuracy = accuracy_score(y_smote_test, smote_xgb_y_pred)
print(smote_accuracy)
print(classification_report(y_smote_test, smote_xgb_y_pred))


test_df = pd.read_csv('/kaggle/input/playground-series-s3e4/test.csv')


test_df.head()


test_data = test_df.drop(['Time', 'Amount', 'id'], axis=1)
submission_probabilities = smote_xgb_model.predict_proba(test_data)[:,1]


submission_probabilities


# Create a DataFrame with the Person_ID and the predicted probabilities
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Class': submission_probabilities
})

# Save the file
submission_df.to_csv('submission.csv', index=False)




