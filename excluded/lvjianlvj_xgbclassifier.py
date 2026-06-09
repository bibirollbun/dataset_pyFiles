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


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Statistical functions
from scipy.stats import skew

# Display utilities for Jupyter notebooks
from IPython.display import display

# Machine learning preprocessing and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")



test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(f"Dataset Shape: {train_df.shape}")

print("\nData Info:")
train_df.info()

print("\nNumerical Features Summary:")
display(train_df.describe())

print("\nFirst 10 Rows of the Dataset:")
display(train_df.head(10))


# Identify numerical and categorical columns
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
label_feature = 'Personality'
label_name = label_feature
# Remove the label feature from the categorical features
if label_feature in categorical_features:
    categorical_features.remove(label_feature)
# Print the numerical and categorical features
print('Numerical Features:', numerical_features)
print('Categorical Features:', categorical_features)



for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train_df[feature].skew():.2f}")
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")


for feature in categorical_features:
    counts = train_df[feature].value_counts()

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {feature}")
    plt.axis("equal")
    plt.show()

    # Print unique and missing values
    print(f"Number of Unique {feature}: {train_df[feature].nunique()}")
    print(f"Missing Values in {feature}: {train_df[feature].isnull().sum()}")


colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numeric_df = train_df[numerical_features]

sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()


for feature in numerical_features[:-1]:  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=train_df[feature], y=train_df[label_name], alpha=0.5
    )
    plt.title(f"{feature} vs. {label_name}")
    plt.xlabel(feature)
    plt.ylabel(label_name)
    plt.show()

correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


for cat_name in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=cat_name, hue=label_name, data=train_df)
    plt.title(f"Distribution of {label_name} across {cat_name}")
    plt.xlabel(cat_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title=label_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


import seaborn as sns
import pandas as pd

for cat_name in categorical_features:
    cross_tab = pd.crosstab(train_df[cat_name], train_df[label_name])

    plt.figure(figsize=(12, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{cat_name} vs. {label_name} (Counts)")
    plt.ylabel(cat_name)
    plt.xlabel(label_name)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=train_df, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# drop id and personality columns
X_train = train_df.drop(['id', 'Personality'], axis=1)
y_train = train_df['Personality']

X_test = test_df.drop('id', axis=1)


# remove id from numerical_features
if 'id' in numerical_features:
    numerical_features.remove('id')





# process the missing value in X_train and X_test
from sklearn.impute import SimpleImputer

# Impute missing values with the mean for numerical columns
imputer_num = SimpleImputer(strategy='mean')
X_train[numerical_features] = imputer_num.fit_transform(X_train[numerical_features])
X_test[numerical_features] = imputer_num.transform(X_test[numerical_features])

# Impute missing values with the most frequent value for categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])

# Check if there are any missing values left
print('Missing values in X_train:', X_train.isnull().sum().sum())
print('Missing values in X_test:', X_test.isnull().sum().sum())


# use scale to standardize the numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# use labelencoder to encode the categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[categorical_features] = X_train[categorical_features].apply(le.fit_transform)
X_test[categorical_features] = X_test[categorical_features].apply(le.fit_transform)


# there are 2 class in the y_train, in order to use the train model, we need to convert 2 classes into number
# use labelencoder to encode the y_train
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
# check the y_train
print(y_train)


X = X_train
y = pd.Series( y_train)
test_X = X_test


# Best XGBoost parameters from previous Optuna run
best_params = {
    'max_depth': 7,
    'learning_rate': 0.05635134330984224,
    'subsample': 0.5605235929333594,
    'colsample_bytree': 0.5594578346445631,
    'min_child_weight': 6,
    'gamma': 0.35819323772520817,
    'reg_alpha': 0.9747714669120731,
    'reg_lambda': 0.7061465594372847,
    'objective': 'multi:softprob',
    'num_class': 2,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',
    'verbosity': 0,
    'n_estimators': 1000
}


# MAP@3 Metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Training and predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(np.unique(y))))
test_preds = np.zeros((test_X.shape[0], len(np.unique(y))))
oof_true = []
oof_top3_preds = []

fold_loglosses = []
fold_map3s = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # print the shape of X_train, X_valid, y_train, y_valid
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)   

    model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=10,
        verbose=False
    )

    oof_preds[valid_idx] = model.predict_proba(X_valid)


    fold_logloss = log_loss(y_valid, oof_preds[valid_idx])
    fold_loglosses.append(fold_logloss)

    top3_preds = np.argsort(oof_preds[valid_idx], axis=1)[:, -3:][:, ::-1]
    fold_map3 = mapk(y_valid.tolist(), [list(p) for p in top3_preds], k=3)
    fold_map3s.append(fold_map3)

    print(f"  Fold {fold+1} Log Loss: {fold_logloss:.5f}")
    print(f"  Fold {fold+1} MAP@3: {fold_map3:.5f}")

    oof_true.extend(y_valid)
    oof_top3_preds.extend(top3_preds)
    test_preds += model.predict_proba(test_X) / skf.n_splits

# Overall MAP@3
map3_score = mapk(oof_true, [list(p) for p in oof_top3_preds], k=3)
print(f"\nOverall OOF MAP@3 Score: {map3_score:.5f}")


# use model to predict test_X
test_preds = model.predict(test_X)
print(test_preds)
test_preds= pd.DataFrame(test_preds)
# give test_preds 2 columns name 'Extrovert', 'Introvert'
test_preds.columns = ['Extrovert', 'Introvert']


print(test_preds)
# convert test_preds to 1 column, called Personality, this columns value follow this rule, if Extrovert value == 1, then set Extrovert else set Introvert
test_preds['Personality'] = test_preds.apply(lambda x: 'Extrovert' if x['Extrovert'] == 1 else 'Introvert', axis=1)



submission_df['Personality'] = test_preds['Personality']
submission_df.to_csv('submission5.csv', index=False)

