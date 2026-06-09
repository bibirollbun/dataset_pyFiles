import pandas as pd
import numpy as np

import re
import sys
import math

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
import scipy.stats as stats
from patsy import dmatrices
import statsmodels.api as sm 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# Setting up the environment
plt.style.use('seaborn-v0_8-darkgrid')

if sys.platform == 'win32':
    print('Win')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif sys.platform == 'darwin':
    print('Mac')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus']=False
# setting pandas display options
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# pd.set_option('display.max_columns', 20)  # or 1000
# pd.set_option('display.max_rows', 80)  # or 1000
# pd.set_option('display.max_colwidth', 20)  # or 199


# train_df = pd.read_csv('./train.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
display(train_df.head())
display(train_df.info())


# test_df= pd.read_csv('./test.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
display(test_df.head())
display(test_df.info())



# Transform str to float
def compare_str_to_float(x,one_str='Yes',zero_str='No'):
    if isinstance(x, str):
        try:
            if x == one_str:
                return 1.0
            elif x == zero_str:
                return 0.0
        except ValueError:
            return np.nan  # Return NaN if conversion fails
train_df['Personality'] = train_df['Personality'].apply(lambda x: compare_str_to_float(x, 'Introvert', 'Extrovert'))
train_df['Stage_fear'] = train_df['Stage_fear'].apply(lambda x: compare_str_to_float(x, 'Yes', 'No'))
test_df['Stage_fear'] = test_df['Stage_fear'].apply(lambda x: compare_str_to_float(x, 'Yes', 'No'))

train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].apply(lambda x: compare_str_to_float(x, 'Yes', 'No'))
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].apply(lambda x:  compare_str_to_float(x, 'Yes', 'No'))

# Feature Engineering
# Composite features based on the provided data
# train_df['Alone_social_gap'] = train_df['Time_spent_Alone'] - train_df['Social_event_attendance']
# test_df['Alone_social_gap'] = test_df['Time_spent_Alone'] - test_df['Social_event_attendance']


# train_df['Drain_gap'] = train_df['Drained_after_socializing'] - train_df['Friends_circle_size']
# test_df['Drain_gap'] = test_df['Drained_after_socializing'] - test_df['Friends_circle_size']



data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
display(data.head())
display(data.info())
data_with_na = data.drop('Personality', axis=1)  # Exclude the target variable
data_with_na = data_with_na[data_with_na.isnull().any(axis=1)]
display(data_with_na.head())
display(data_with_na.info())



data_na_count = data_with_na.isnull().sum()
# add a column for the percentage of missing values
data_na_percent = data_na_count / len(data) * 100
data_na_summary = pd.DataFrame({
    'Missing Values': data_na_count,
    'Percentage': data_na_percent
}).sort_values(by='Missing Values', ascending=False)
display(data_na_summary)

# Visualizing missing values
msno.matrix(data, figsize=(12, 6))
display(plt.show())


desc = data.describe()

# Calculate skewness and kurtosis for the data with missing values
skewness = data_with_na.skew(numeric_only=True)
kurtosis = data_with_na.kurt(numeric_only=True)  

desc.loc['skew'] = skewness
desc.loc['kurtosis'] = kurtosis

display(desc.T)


# Data visualization distribution

# Select numerical features for distribution plots
# Set up the subplot grid

numerical_features = data.columns.drop(['id', 'Personality']) 
n_features = len(numerical_features)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols  # auto row count

plt.figure(figsize=(16, 12))

for idx, feature in enumerate(numerical_features, 1):
    ax = plt.subplot(n_rows, n_cols, idx)
    sns.histplot(
        train_df[feature].dropna(),
        kde=True,
        bins=40,
        color='skyblue',
        alpha=0.3,
        label='Train',
        ax=ax
    )

    sns.histplot(
        test_df[feature].dropna(),
        kde=True,
        bins=40,
        color='salmon',
        alpha=0.3,
        label='Test',
        ax=ax
    )
    
    ax.set_title(f'Distribution of {feature}', fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend()
plt.tight_layout()
plt.show()




# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Data Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#  Missing Data Analysis


print("=== Miss Values Basic Statistics ===")
display(f"Train set missing values:")
print(train_df.isnull().sum())
display(f"Train Set Missing Values {train_df.isnull().any(axis=1).sum()}")

display(f"Test set missing values:")
print(test_df.isnull().sum())
display(f"Test Set Missing Values {test_df.isnull().any(axis=1).sum()}")



print("=== MissValue Visualization ===")

# MissValue Count, Matrix, Heatmap, Dendrogram
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MissValue Count 
msno.bar(data, ax=axes[0,0])
# plt.xticks(rotation=-45, ha='left')
axes[0,0].set_title('MissValue Count', fontsize=14)

# MissValue Matrix
msno.matrix(data, ax=axes[0,1])
axes[0,1].set_title('MissValue Matrix', fontsize=14)

# MissValue Correlation Heatmap
msno.heatmap(data, ax=axes[1,0])
axes[1,0].set_title('MissValue Correlation Heatmap', fontsize=14)

# MissValue Tree Dendrogram
msno.dendrogram(data, ax=axes[1,1])
axes[1,1].set_title('MissValue Tree Dendrogram', fontsize=14)

plt.tight_layout()
plt.show()

# MissValue Correlation Analysis
display("=== MissValue Correlation Analysis ===")
missing_matrix = data.isnull().astype(int)
missing_corr = missing_matrix.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(missing_corr, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, fmt='.2f')
plt.xticks(rotation=-45, ha='left')
plt.title('Miss Value Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# High correlation pairs
high_corr_pairs = []
for i in range(len(missing_corr.columns)):
    for j in range(i+1, len(missing_corr.columns)):
        corr_val = missing_corr.iloc[i, j]
        if abs(corr_val) > 0.5: 
            high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))

if high_corr_pairs:
    print("High Correlation Pairs (|correlation| > 0.5):")
    for var1, var2, corr in high_corr_pairs:
        print(f"  {var1} - {var2}: {corr:.3f}")
else:
    print("No high correlation pairs found (|correlation| > 0.5).")



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm

def assess_missing_mechanism(data, target='Personality', n_permutations=1000):
    missing_vars = data.columns[data.isnull().any()].tolist()
    if target in missing_vars:
        missing_vars.remove(target)
    if 'id' in missing_vars:
        missing_vars.remove('id')

    results = []

    for var in missing_vars:
        print(f"Analyzing Missing Mechanism for: {var}")
        temp = data.copy()
        temp[f"{var}_missing"] = temp[var].isnull().astype(int)

        features = [col for col in data.columns 
                    if col not in [var, f"{var}_missing", target, 'id']]

        complete = temp.dropna(subset=features)
        if len(complete) < 50:
            print(f"  ⚠ Too few complete cases: {len(complete)}")
            continue

        X = complete[features]
        y = complete[f"{var}_missing"]

        if len(np.unique(y)) < 2:
            print(f"  ⚠ Only one class found in y, skipping")
            continue

        X_scaled = StandardScaler().fit_transform(X)

        # Fit Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        y_pred = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred)

        # Permutation Test
        perm_auc = []
        for _ in tqdm(range(n_permutations), desc=f" Permutation AUC for {var}", leave=False):
            y_perm = shuffle(y, random_state=None)
            auc_perm = roc_auc_score(y_perm, y_pred)
            perm_auc.append(auc_perm)

        perm_auc = np.array(perm_auc)
        p_value = np.mean(perm_auc >= auc)
        ci_lower, ci_upper = np.percentile(perm_auc, [2.5, 97.5])

        # Determine mechanism
        if p_value > 0.05:
            mechanism = "Possibly MCAR"
        elif auc < 0.8:
            mechanism = "Possibly MAR"
        else:
            mechanism = "Possibly MNAR"

        # Most important features
        coef_df = pd.DataFrame({
            'feature': features,
            'coef': model.coef_[0],
            'abs_coef': np.abs(model.coef_[0])
        }).sort_values('abs_coef', ascending=False).head(5)


        results.append({
            'Variable': var,
            'AUC': auc,
            'p-value': p_value,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Mechanism': mechanism,
            'Top_Features': coef_df['feature'].tolist()
        })

    return pd.DataFrame(results)

miss_mech_df = assess_missing_mechanism(data)
display(miss_mech_df)


# MICE 
# MICE (Multiple Imputation by Chained Equations) is a method for handling missing data.
# It creates multiple complete datasets by imputing missing values based on other variables in the dataset.
# This method is particularly useful when the data is MAR (Missing At Random) or MNAR (Missing Not At Random).


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def apply_mice_imputation(data, max_iter=10, random_state=42):  
    """
    Apply MICE imputation to the dataset.
    
    Parameters:
    - data: DataFrame with missing values
    - max_iter: Maximum number of iterations for the imputer
    - random_state: Random seed for reproducibility
    
    Returns:
    - DataFrame with imputed values
    """
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    display(f"Applying MICE imputation with max_iter={max_iter} and random_state={random_state}")
    display(f"Imputing numeric columns: {data.select_dtypes(include=[np.number]).columns.tolist()}")
    imputed_data = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=data.select_dtypes(include=[np.number]).columns)
    
    # Preserve non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number])
    
    return pd.concat([imputed_df, non_numeric_cols.reset_index(drop=True)], axis=1)

# Apply MICE imputation
imputed_train_df = train_df.copy()
imputed_train_df['id'] = imputed_train_df['id'].astype(str)  # Ensure 'id' is string type for consistency
imputed_test_df = test_df.copy()
imputed_test_df['id'] = imputed_test_df['id'].astype(str)  # Ensure 'id' is string type for consistency

imputed_train_df = apply_mice_imputation(imputed_train_df.drop(columns=['Personality']))
imputed_test_df = apply_mice_imputation(imputed_test_df)

# Check if there are still missing values
print("Missing values after MICE imputation:")
print("Train set:", imputed_train_df.isnull().sum().sum())
display(imputed_train_df.head())
print("Test set:", imputed_test_df.isnull().sum().sum())
display(imputed_test_df.head())

# Check if the imputation worked By Linear Regression
def check_imputation_effectiveness(original_df, imputed_df, target_col='Personality'):
    """
    Check the effectiveness of imputation by comparing distributions and correlations.
    
    Parameters:
    - original_df: Original DataFrame with missing values
    - imputed_df: DataFrame after imputation
    - target_col: Target column to analyze
    
    Returns:
    - None
    """
    # Compare correlations
    original_corr = original_df.corr()
    imputed_corr = imputed_df.corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Changed to plt.subplots() for 1 row, 2 columns

    sns.heatmap(original_corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=axes[0])
    axes[0].set_title('Original Correlation Matrix', fontsize=16)
    plt.setp(axes[0].get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    axes[0].tick_params(axis='y', rotation=0)
    plt.tight_layout()
    
    sns.heatmap(imputed_corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=axes[1])
    axes[1].set_title('Imputed Correlation Matrix', fontsize=16)
    plt.setp(axes[1].get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    axes[1].tick_params(axis='y', rotation=0) 
    plt.tight_layout()
    plt.show()


# Check the effectiveness of imputation
check_imputation_effectiveness(train_df.drop(columns=['id','Personality'],axis=1), imputed_train_df.drop(columns=['id']), target_col='Personality')


display(imputed_train_df.head())
display(imputed_train_df.info())


imputed_data = pd.concat([imputed_train_df, imputed_test_df], axis=0)
imputed_desc = imputed_data.describe()

# Calculate skewness and kurtosis for the imputed data
imputed_skewness = imputed_data.skew(numeric_only=True)
imputed_kurtosis = imputed_data.kurt(numeric_only=True)  # Fisher 定義：Normal 為 0


imputed_desc.loc['skew'] = imputed_skewness
imputed_desc.loc['kurtosis'] = imputed_kurtosis

# Compare the original and imputed descriptions
display(desc.T)
display(imputed_desc.T)


# Visualizing missing imputed_train_df 
plt.figure(figsize=(16, 12))
for idx, feature in enumerate(imputed_train_df.columns, 1):
    display(f"Feature: {feature}")
    if feature in ['id', 'Personality']:
        continue
    ax = plt.subplot(5, 4, idx)
    sns.histplot(
        imputed_train_df[feature],
        kde=True,
        bins=40,
        color='skyblue',
        alpha=0.3,
        label='Imputed Train',
        ax=ax
    )
    
    ax.set_title(f'Distribution of {feature}', fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend()
plt.tight_layout()
plt.show()  


imputed_train_df['id'] = imputed_train_df['id'].apply(lambda x : int(x))
imputed_train_df = pd.merge(imputed_train_df, train_df[['id', 'Personality']], on='id', how='left')
display( imputed_train_df.head())
display(imputed_train_df.info())



from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import parallel_backend

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import make_scorer, accuracy_score


# Define the target variable and features
df = imputed_train_df.copy()
X = df.drop(columns=['id', 'Personality'], errors='ignore')
y = df['Personality']

# Define cross-validation and scoring
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(accuracy_score)

# To store the grid search results
grid_dict = {} 

# Create pipelines for each model
pipelines = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'HistGB': Pipeline([
        ('clf', HistGradientBoostingClassifier(random_state=42))
    ]),
    'LightGBM': Pipeline([
        ('clf', LGBMClassifier(random_state=42))
    ]),
    'CatBoost': Pipeline([
        ('clf', CatBoostClassifier(verbose=0, random_state=42))
    ]),
    'GaussianNB': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ]),
}

# Define parameter grids for each model
param_grids = {
    'LogisticRegression': {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10]
    },
    'RandomForest': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    },
    'HistGB': {
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_iter': [100, 200],
        'clf__max_depth': [3, 5]
    },
    'LightGBM': {
        'clf__num_leaves': [15, 31],
        'clf__learning_rate': [0.01, 0.1],
        'clf__n_estimators': [100, 200]
    },
    'CatBoost': {
        'clf__iterations': [100, 200],
        'clf__learning_rate': [0.01, 0.1],
        'clf__depth': [3, 5]
    },
    'GaussianNB': {
        'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

# Perform GridSearchCV for each model
results = []

for name, pipeline in pipelines.items():
    print(f" Running GridSearch for {name}…")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1
    )
    # Using parallel backend to speed up the grid search
    with parallel_backend('loky'):
        grid.fit(X, y)

    grid_dict[name] = grid 
    results.append({
        'Model': name,
        'Best CV Accuracy': grid.best_score_,
        'Best Params': grid.best_params_
    })

results_df = pd.DataFrame(results).sort_values(
    by='Best CV Accuracy', ascending=False
).reset_index(drop=True)

print("GridSearch Finished, best results are as follows:")
display(results_df)



from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

# Create a voting classifier with the best estimators from grid search
best_estimators = {}
weights = []

for res in results:
    name = res['Model']
    grid = grid_dict[name]  
    best_estimators[name] = grid.best_estimator_
    weights.append(res['Best CV Accuracy'])

# Normalize the weights
total = sum(weights)
norm_weights = [w/total for w in weights]

# Create the voting classifier with soft voting
voting_clf = VotingClassifier(
    estimators=[(name, est) for name, est in best_estimators.items()],
    voting='soft',
    weights=norm_weights,
    n_jobs=-1
)

cv_scores = cross_val_score(voting_clf, X, y, cv=5, scoring='accuracy')
display(f"Voting classifier CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit the voting classifier on the entire training set
voting_clf.fit(X, y)

# Prepare the test set for prediction
X_test = imputed_test_df.drop(columns=['id'], errors='ignore')
ids = imputed_test_df['id']

# Predict on the test set
preds = voting_clf.predict(X_test)
y_preds_proba = voting_clf.predict_proba(X_test)[:, 1]

# Convert predictions to binary labels
plt.figure(figsize=(10, 6))
plt.hist(y_preds_proba, bins=50, color='skyblue', alpha=0.7)
plt.axvline(x=0.8, color='red', linestyle='--', label='Threshold = 0.8')
plt.title('Predicted Probabilities Distribution', fontsize=16)
plt.xlabel('Predicted Probability of Introvert')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

plt.show() 

submission = pd.DataFrame({
    'id': ids,
    'Personality': preds
})
submission_checkori =  pd.DataFrame({
    'id': ids,
    'Personality_p': y_preds_proba
})

submission['Personality'] = submission['Personality'].apply(lambda x: 'Introvert' if x == 1 else 'Extrovert')
# To catch those who are close to the threshold,the Introvert Hide in the Extrovert
submission_checkori['Personality'] = submission_checkori['Personality_p'].apply(lambda x: 'Introvert' if  x > 0.8 else 'Extrovert')


display(submission.head())

# The threshold for introvert is set to 0.8, Have Score: 0.975708
submission_checkori[['id','Personality']].to_csv('submission.csv', index=False)



upper_bound = 0.9
lower_bound = 0.1
In_confusion_internal = submission_checkori[(submission_checkori['Personality_p'] > lower_bound) & (submission_checkori['Personality_p'] < upper_bound)]
# In_confusion_internal['id'] = In_confusion_internal['id'].astype(int)  # Ensure 'id' is integer type for consistency
In_confusion_internal = In_confusion_internal.sort_values(by='Personality_p', ascending=False)

display(In_confusion_internal.shape)
display(In_confusion_internal.head(10))
display(In_confusion_internal.head(60).tail(20))
display(imputed_test_df[imputed_test_df['id'].isin(In_confusion_internal['id'])].head(10))


In_confusion_internal_corr = imputed_test_df[imputed_test_df['id'].isin(In_confusion_internal['id'])].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(In_confusion_internal_corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('In Confusion Internal Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


confusion_samples = imputed_test_df[imputed_test_df['id'].isin(In_confusion_internal['id'])].copy()
confusion_samples = pd.merge(confusion_samples, In_confusion_internal, on='id', how='left')
display(confusion_samples.head(10))
confusion_samples['Is_Introvert'] = confusion_samples['Personality_p'].apply(lambda x: 1 if x > 0.8 else 0)



# Confusion Samples Visualization
# feture selection
all_features = [col for col in confusion_samples.columns 
                if col not in ['id', 'Personality', 'Is_Introvert']]
numeric_feats = confusion_samples[all_features].select_dtypes(include=['number']).columns
categorical_feats = [f for f in all_features if f not in numeric_feats]


palette = sns.color_palette("Set2", n_colors=2)
n_feats = len(all_features)
n_cols = 4
n_rows = (n_feats + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), constrained_layout=True)
axes = axes.flat


for ax, feat in zip(axes, all_features):
    if feat in numeric_feats:
        sns.histplot(
            data=confusion_samples,
            x=feat,
            hue='Personality',
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.5,
            ax=ax,
            palette=palette,
        )
    else:
        sns.countplot(
            data=confusion_samples,
            x=feat,
            hue='Personality',
            ax=ax,
            palette=palette,
            dodge=True,
            alpha=0.7,
        )
        ax.tick_params(axis='x', rotation=45)
    ax.set_title(f"{feat}", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, ['Extrovert', 'Introvert'], loc='center right', title='Personality')


for ax in axes[n_feats:]:
    ax.axis('off')

plt.show()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# perpare the features for t-SNE
X_confusion = confusion_samples.drop(columns=['id','Personality'])
display(X_confusion.head())

scaler = StandardScaler()
X_confusion_scaled = scaler.fit_transform(X_confusion)

tsne = TSNE(n_components=2, random_state=42, perplexity=min(8, len(X_confusion)-1))
X_confusion_tsne = tsne.fit_transform(X_confusion_scaled)

# Extract the predicted probabilities for the introvert class
confusion_probas = In_confusion_internal['Personality_p'].values

# Visualize the t-SNE results
plt.figure(figsize=(12, 8))

# Using scatter plot to visualize the t-SNE results
# with color representing the predicted probabilities
scatter = plt.scatter(
    X_confusion_tsne[:, 0], 
    X_confusion_tsne[:, 1], 
    c=confusion_probas, 
    cmap='RdYlBu_r', 
    s=60, 
    alpha=0.7
)

plt.colorbar(scatter, label='Predicted Probability (Introvert)')
plt.title(f't-SNE Visualization of Confused Test Samples\n(Probability between {lower_bound} and {upper_bound})', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display statistics of the predicted probabilities
print("Predicted Probabilities Statistics:")
print(f"mini: {confusion_probas.min():.4f}")
print(f"max: {confusion_probas.max():.4f}")
print(f"mean: {confusion_probas.mean():.4f}")
print(f"median: {np.median(confusion_probas):.4f}")
print(f"std: {confusion_probas.std():.4f}")




train_set_abnormal = imputed_train_df.copy()
X_train = train_set_abnormal.drop(columns=['id', 'Personality'], errors='ignore')
y_train = train_set_abnormal['Personality']


train_preds = voting_clf.predict(X_train)
train_pred_proba = voting_clf.predict_proba(X_train)[:, 1]  # Introvert 的機率


train_set_abnormal['True_Label'] = y_train
train_set_abnormal['Pred_Label'] = train_preds
train_set_abnormal['Pred_Prob'] = train_pred_proba
train_set_abnormal['Correct'] = (train_set_abnormal['True_Label'] == train_set_abnormal['Pred_Label']).astype(int)
train_set_abnormal['Error'] = 1 - train_set_abnormal['Correct']

misclassified = train_set_abnormal[train_set_abnormal['Correct'] == 0]

low_confidence = train_set_abnormal[(train_set_abnormal['Pred_Prob'] > 0.2) & (train_set_abnormal['Pred_Prob'] < 0.8)]

conflicted = train_set_abnormal[(train_set_abnormal['Correct'] == 0) & ((train_set_abnormal['Pred_Prob'] > 0.8) | (train_set_abnormal['Pred_Prob'] < 0.2))]

plt.figure(figsize=(10,6))
sns.histplot(train_set_abnormal[train_set_abnormal['Error'] == 1]['Pred_Prob'], bins=40, kde=True, color='orange')
plt.axvline(0.5, color='red', linestyle='--', label='Decision Boundary')
plt.title('Distribution of Predicted Probabilities for Misclassified Samples')
plt.xlabel('Predicted Probability (Introvert)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Error in train: {misclassified.shape[0]}")
print(f"Low Confidence: {low_confidence.shape[0]}")
print(f"Low Confidence And Error: {low_confidence[low_confidence['Correct'] == 0].shape[0]}")
print(f"Error With High Confidence: {conflicted.shape[0]}")

display(conflicted[['id', 'True_Label', 'Pred_Label', 'Pred_Prob','Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
        'Post_frequency' ]].head())




display(low_confidence.describe().T)
display(low_confidence[low_confidence['Time_spent_Alone']<0].head())
# display


# Error samples correlation matrix
conflicted = conflicted.rename(columns={'True_Label':'Is_Introvert'})

corr = conflicted[['Is_Introvert', 'Pred_Label', 'Pred_Prob','Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size','Post_frequency']].corr().round(2)
datalen = conflicted.shape[0]
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title(f'Correlation Matrix of Misclassified Samples In train data(data shape :{datalen})', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





# Check is fill NA make this error 
display(train_df[train_df['Personality'] == 0].describe().T)
# display(imputed_train_df[imputed_train_df['Drained_after_socializing']<0].describe().T)


conflicted_predIS_E = conflicted[conflicted['Pred_Label'] == 0][['Is_Introvert', 'Pred_Label', 'Pred_Prob','Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size','Post_frequency']]
display(conflicted_predIS_E.describe().T)

corr = conflicted_predIS_E.corr().round(2)
datalen = conflicted_predIS_E.shape[0]
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title(f'Correlation Matrix of Misclassified Samples In train data(data shape :{datalen})', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



conflicted_predIS_I = conflicted[conflicted['Pred_Label'] == 1][['Is_Introvert', 'Pred_Label', 'Pred_Prob','Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size','Post_frequency']]
corr = conflicted_predIS_I.corr().round(2)
display(conflicted_predIS_I.describe().T)

datalen = conflicted_predIS_I.shape[0]
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title(f'Correlation Matrix of Misclassified Samples In train data(data shape :{datalen})', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



display(conflicted_predIS_I[conflicted_predIS_I ['Drained_after_socializing']<0].describe().T)
display(conflicted_predIS_E[conflicted_predIS_E ['Drained_after_socializing']<0].describe().T)



# # =============================================
# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, make_scorer
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from lightgbm import LGBMClassifier                  # pip install lightgbm
# import optuna
# from optuna.integration import OptunaSearchCV
# from joblib import parallel_backend

# df = imputed_train_df.copy()
# X  = df.drop(columns=['id', 'Personality'])
# y  = df['Personality']

# cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scorer  = make_scorer(accuracy_score)
# sampler = optuna.samplers.TPESampler(seed=42)  

# # =============================================
# #  Define the pipelines and search spaces
# # =============================================
# # RandomForest
# rf_pipeline = Pipeline([
#     ('clf', RandomForestClassifier(random_state=42))
# ])
# rf_space = {
#     'clf__n_estimators'    : optuna.distributions.IntDistribution(200, 700),
#     'clf__max_depth'       : optuna.distributions.IntDistribution(3, 20),
#     'clf__min_samples_split': optuna.distributions.IntDistribution(2, 20),
#     'clf__min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
#     'clf__max_features'    : optuna.distributions.CategoricalDistribution(
#                                 ['sqrt', 'log2', 0.6, 0.8, 1.0])
# }

# # HistGradientBoosting
# hgb_pipeline = Pipeline([
#     ('clf', HistGradientBoostingClassifier(random_state=42))
# ])
# hgb_space = {
#     'clf__learning_rate'    : optuna.distributions.FloatDistribution(0.005, 0.2, log=True),
#     'clf__max_depth'        : optuna.distributions.IntDistribution(3, 10),
#     'clf__max_iter'         : optuna.distributions.IntDistribution(200, 1200, step=200),
#     'clf__l2_regularization': optuna.distributions.FloatDistribution(0.0, 2.0),
#     'clf__max_leaf_nodes'   : optuna.distributions.IntDistribution(15, 127, step=14),
# }

# # LightGBM
# lgbm_pipeline = Pipeline([
#     ('clf', LGBMClassifier(random_state=42, verbosity=-1))
# ])
# lgbm_space = {
#     'clf__num_leaves'        : optuna.distributions.IntDistribution(15, 255, step=8),
#     'clf__min_child_samples' : optuna.distributions.IntDistribution(5, 50),
#     'clf__subsample'         : optuna.distributions.FloatDistribution(0.6, 1.0),
#     'clf__colsample_bytree'  : optuna.distributions.FloatDistribution(0.6, 1.0),
#     'clf__reg_lambda'        : optuna.distributions.FloatDistribution(0.0, 5.0),
#     'clf__reg_alpha'         : optuna.distributions.FloatDistribution(0.0, 5.0),
#     'clf__learning_rate'     : optuna.distributions.FloatDistribution(0.003, 0.2, log=True),
#     'clf__n_estimators'      : optuna.distributions.IntDistribution(300, 1500, step=100)
# }

# # =============================================
# def build_search(pipeline, space, n_trials, study_name):
#     study = optuna.create_study(              
#         direction='maximize',
#         sampler=sampler,
#         study_name=study_name
#     )
#     return OptunaSearchCV(
#         estimator=pipeline,
#         param_distributions=space,
#         n_trials=n_trials,
#         study=study,                          
#         cv=cv,
#         scoring=scorer,
#         n_jobs=-1,
#         verbose=0,
#         random_state=42                       
#     )

# searchers = {
#     'RandomForest' : build_search(rf_pipeline,  rf_space,  75, 'rf'),
#     'HistGB'       : build_search(hgb_pipeline, hgb_space, 75, 'hgb'),
#     'LightGBM'     : build_search(lgbm_pipeline,lgbm_space,100, 'lgbm'),
# }

# best_models = {}
# for name, search in searchers.items():
#     print(f"OptunaSearchCV for {name} …")
#     with parallel_backend('loky'):
#         search.fit(X, y)

#     print(f"  ↳ Best ACC : {search.best_score_:.4f}")
#     print(f"  ↳ Best Para: {search.best_params_}")
#     best_models[name] = search.best_estimator_





# RandomForest
# Best Para: {'clf__n_estimators': 692, 'clf__max_depth': 9, 'clf__min_samples_split': 6, 'clf__min_samples_leaf': 6, 'clf__max_features': 1.0}
# HistGB
# Best Para: {'clf__learning_rate': 0.03665755365311515, 'clf__max_depth': 8, 'clf__max_iter': 800, 'clf__l2_regularization': 1.3377630212724627, 'clf__max_leaf_nodes': 99}
# LightGBM
# Best Para: {'clf__num_leaves': 191, 'clf__min_child_samples': 22, 'clf__subsample': 0.8138703399693022, 'clf__colsample_bytree': 0.7953301849377324, 'clf__reg_lambda': 4.1119089789283105, 'clf__reg_alpha': 4.308461593646719, 'clf__learning_rate': 0.0476535231031666, 'clf__n_estimators': 1100}

# best_models = {
#     'RandomForest': RandomForestClassifier(
#         n_estimators=692, max_depth=9, min_samples_split=6,
#         min_samples_leaf=6, max_features=1.0, random_state=42
#     ),
#     'HistGB': HistGradientBoostingClassifier(
#         learning_rate=0.0367, max_depth=8, max_iter=800,
#         l2_regularization=1.3378, max_leaf_nodes=99, random_state=42
#     ),
#     'LightGBM': LGBMClassifier(
#         num_leaves=191, min_child_samples=22, subsample=0.8139,
#         colsample_bytree=0.7953, reg_lambda=4.1119, reg_alpha=4.3085,
#         learning_rate=0.0477, n_estimators=1100, random_state=42, verbosity=-1
#     )
# }


# from sklearn.base import clone


# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # =============================================================
# # 1. modify oof_bagging function
# # =============================================================
# def oof_bagging(estimator, X, y, X_test, cv):
#     """
    
#         oof_pred_class : array(len(X),)      0/1
#         oof_pred_proba : array(len(X),)      P(y=1)
#         test_pred_proba: array(len(test),)   
#     """
#     oof_class  = np.zeros(len(X))
#     oof_proba  = np.zeros(len(X))
#     test_proba_folds = []

#     for fold, (tr, val) in enumerate(cv.split(X, y), 1):
#         model = clone(estimator)
#         model.fit(X.iloc[tr], y.iloc[tr])

#         # Validation
#         oof_class[val] = model.predict(X.iloc[val])
#         if hasattr(model, "predict_proba"):
#             oof_proba[val] = model.predict_proba(X.iloc[val])[:, 1]
#             test_proba_folds.append(model.predict_proba(X_test)[:, 1])
#         else:                    
#             oof_proba[val] = model.predict(X.iloc[val])
#             test_proba_folds.append(model.predict(X_test))

#         acc = accuracy_score(y.iloc[val], oof_class[val])
#         print(f"Fold {fold} ACC = {acc:.4f}")

#     cv_acc = accuracy_score(y, oof_class)
#     print(f"OOF ACC  = {cv_acc:.4f}")

#     test_pred_proba = np.mean(test_proba_folds, axis=0)
#     return oof_class, oof_proba, test_pred_proba

# # =============================================================
# # 2.  OOF Bagging
# # =============================================================
# X_train = X                      
# y_train = y
# X_test  = imputed_test_df.drop(columns=['id'])

# oof_preds_dict   = {}            # OOF proba
# test_preds_dict  = {}            # Test proba

# for name, model in best_models.items():
#     print(f"OOF Bagging for {name} …")
#     _, oof_p, test_p = oof_bagging(model, X_train, y_train, X_test, cv)
#     oof_preds_dict[name]  = oof_p
#     test_preds_dict[name] = test_p

# # =============================================================
# # 3. submission
# # =============================================================
# for name, test_p in test_preds_dict.items():
#     sub = pd.DataFrame({
#         'id': imputed_test_df['id'],
#         'Personality': np.where(test_p > 0.5, 'Introvert', 'Extrovert')
#     })
#     file_name = f'submission_{name.lower()}.csv'
#     sub.to_csv(file_name, index=False)

# # =============================================================
# # 4. Soft Voting 
# # =============================================================
# voting_test_proba = np.mean(list(test_preds_dict.values()), axis=0)
# voting_sub = pd.DataFrame({
#     'id': imputed_test_df['id'],
#     'Personality': np.where(voting_test_proba > 0.5, 'Introvert', 'Extrovert')
# })
# # voting_sub.to_csv('submission_soft_vote.csv', index=False)



# # Compare submission_soft_vote and submission 
# submission = pd.read_csv('submission.csv')
# submission_soft_vote = pd.read_csv('submission_soft_vote.csv')

# # Compare the two DataFrames
# comparison = submission.merge(submission_soft_vote, on='id', suffixes=('_original', '_soft_vote'))
# # Check if there are any differences
# differences = comparison[comparison['Personality_original'] != comparison['Personality_soft_vote']]
# if not differences.empty:
#     print("Differences found between the two submissions:")
#     print(differences)
# else:
#     print("No differences found between the two submissions.")

# # !! Not Better than submission.csv


# Best parm Model	Best CV Accuracy	Best Params
# 0	HistGB	0.969391	{'clf__learning_rate': 0.01, 'clf__max_depth': 5, 'clf__max_iter': 200}
# 1	CatBoost	0.969283	{'clf__depth': 3, 'clf__iterations': 100, 'clf__learning_rate': 0.1}
# 2	RandomForest	0.969229	{'clf__max_depth': 10, 'clf__min_samples_split': 2, 'clf__n_estimators': 200}
# 3	LightGBM	0.969175	{'clf__learning_rate': 0.1, 'clf__n_estimators': 100, 'clf__num_leaves': 15}
# 4	LogisticRegression	0.969121	{'clf__C': 0.1, 'clf__penalty': 'l1'}
# 5	GaussianNB	0.968905	{'clf__var_smoothing': 1e-09}

# RandomForest
# Best Para: {'clf__n_estimators': 692, 'clf__max_depth': 9, 'clf__min_samples_split': 6, 'clf__min_samples_leaf': 6, 'clf__max_features': 1.0}
# HistGB
# Best Para: {'clf__learning_rate': 0.03665755365311515, 'clf__max_depth': 8, 'clf__max_iter': 800, 'clf__l2_regularization': 1.3377630212724627, 'clf__max_leaf_nodes': 99}
# LightGBM
# Best Para: {'clf__num_leaves': 191, 'clf__min_child_samples': 22, 'clf__subsample': 0.8138703399693022, 'clf__colsample_bytree': 0.7953301849377324, 'clf__reg_lambda': 4.1119089789283105, 'clf__reg_alpha': 4.308461593646719, 'clf__learning_rate': 0.0476535231031666, 'clf__n_estimators': 1100}



