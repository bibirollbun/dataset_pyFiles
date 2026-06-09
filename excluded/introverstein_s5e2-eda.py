import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, zscore

import warnings
warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e2/train.csv'
    test_path = '/kaggle/input/playground-series-s5e2/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e2/sample_submission.csv'
    target = 'Price'
    seed = 3482494
    n_splits = 5


# Read the train and test sets
train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)

print(f'Shape of train set: {train.shape}')
print(f'Shape of test set: {test.shape}')


# The one column differing between train and test set must be the target variable 'Price'
set(train.columns) - set(test.columns)


# Let's plot the distribution of the target variable 'Price'
sns.set_style("whitegrid")
palette = sns.color_palette('muted')

plt.figure(figsize=(12, 8))
sns.histplot(train[CFG.target], bins=50, kde=True, color=palette[0])

plt.xlabel("Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Target Variable: Price", fontsize=14)
plt.show()


# Now let's have a look at how many null values are there in each column 
# Credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    mis_val = df.isnull().sum()
    mis_val_percent = (mis_val / len(df)) * 100

    mis_val_table = pd.DataFrame({
        'Missing Values': mis_val,
        '% of Total Values': mis_val_percent
    })

    mis_val_table = mis_val_table[mis_val_table['Missing Values'] > 0]\
        .sort_values('% of Total Values', ascending=False)\
        .round(2)

    if mis_val_table.empty:
        return pd.DataFrame({'Message': ['No missing values detected.']})

    return mis_val_table.style.background_gradient(cmap='Reds')


display(missing_values_table(train))
display(missing_values_table(test))


# Let's have a look at the datatypes of the columns
train.dtypes.reset_index().rename(
    columns={'index': 'column',
              0: 'dtype'}
)


# This makes me think that maybe all columns with object type have are categorical
# And that all columns (except id) with non-object data type are numerical features
# To support this proposition, let's print the value counts of each feature
for col in test.columns[1:]: # not using 'id'
    print(f'Feature `{col}` with dtype {test[col].dtype}, has:')
    print(f'{train[col].nunique()} unique values in train set, ')
    print(f'{test[col].nunique()} unique values in test set.\n')


# Let's store the features in lists

# First cast Compartments columns into object data type
feat = 'Compartments'
train[feat] = train[feat].astype(str)
test[feat] = test[feat].astype(str)

features = [f for f in train.columns if f not in ['id', CFG.target]]
cat_features = [f for f in features if train[f].dtype == 'object']
num_features = [f for f in features if f not in cat_features]


def plot_distributions(train, test, num_features, kde=True):
    sns.set_palette("muted")
    
    for i, f in enumerate(num_features):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        
        sns.histplot(train[f], kde=kde, ax=axes[0], color='skyblue', bins=50, label='Train')
        axes[0].set_title(f'Train distribution of {f}', fontsize=14)
        axes[0].set_xlabel(f'{f}', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        sns.histplot(test[f], kde=kde, ax=axes[1], color='salmon', bins=50, label='Test')
        axes[1].set_title(f'Test distribution of {f}', fontsize=14)
        axes[1].set_xlabel(f'{f}', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


plot_distributions(train, test, num_features)


def plot_countplots(train, test, cat_features):
    sns.set_palette('muted')
    
    for i, f in enumerate(cat_features):
        train_order = train[f].value_counts().index
        test_order = test[f].value_counts().index
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        
        sns.countplot(y=train[f], ax=axes[0], color='skyblue', label='Train', orient='h', order=train_order)
        axes[0].set_title(f'Value Counts of {f} (Train)', fontsize=14)
        axes[0].set_xlabel('Count', fontsize=12)
        axes[0].set_ylabel(f'{f}', fontsize=12)
        
        axes[0].tick_params(axis='x', labelsize=12)
        
        for p in axes[0].patches:
            axes[0].annotate(f'{p.get_width():,.0f}', 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha='left', va='center', fontsize=12, color='black', 
                             xytext=(5, 0), textcoords='offset points')

        sns.countplot(y=test[f], ax=axes[1], color='salmon', label='Test', orient='h', order=test_order)
        axes[1].set_title(f'Value Counts of {f} (Test)', fontsize=14)
        axes[1].set_xlabel('Count', fontsize=12)
        axes[1].set_ylabel(f'{f}', fontsize=12)
        
        axes[1].tick_params(axis='x', labelsize=12)
        
        for p in axes[1].patches:
            axes[1].annotate(f'{p.get_width():,.0f}', 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha='left', va='center', fontsize=12, color='black', 
                             xytext=(5, 0), textcoords='offset points')

        axes[0].grid(True, axis='x', linestyle='--', alpha=0.7)
        axes[1].grid(True, axis='x', linestyle='--', alpha=0.7)

        axes[0].legend(title='Dataset', loc='upper right', fontsize=10)
        axes[1].legend(title='Dataset', loc='upper right', fontsize=10)

        plt.tight_layout()

    plt.show()


plot_countplots(train, test, cat_features)


def plot_normalized_countplots(train, test, cat_features):
    sns.set_palette('muted')
    
    for i, f in enumerate(cat_features):
        train_counts = train[f].value_counts(normalize=True)
        test_counts = test[f].value_counts(normalize=True)
        
        train_order = train_counts.index
        test_order = test_counts.index
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        
        sns.barplot(y=train_counts.index, x=train_counts.values, ax=axes[0], color='skyblue', label='Train')
        axes[0].set_title(f'Normalized Value Counts of {f} (Train)', fontsize=14)
        axes[0].set_xlabel('Normalized Count', fontsize=12)
        axes[0].set_ylabel(f'{f}', fontsize=12)
        
        for p in axes[0].patches:
            axes[0].annotate(f'{p.get_width():.2f}', 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha='left', va='center', fontsize=12, color='black', 
                             xytext=(5, 0), textcoords='offset points')

        sns.barplot(y=test_counts.index, x=test_counts.values, ax=axes[1], color='salmon', label='Test')
        axes[1].set_title(f'Normalized Value Counts of {f} (Test)', fontsize=14)
        axes[1].set_xlabel('Normalized Count', fontsize=12)
        axes[1].set_ylabel(f'{f}', fontsize=12)
        
        for p in axes[1].patches:
            axes[1].annotate(f'{p.get_width():.2f}', 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha='left', va='center', fontsize=12, color='black', 
                             xytext=(5, 0), textcoords='offset points')

        axes[0].grid(True, axis='x', linestyle='--', alpha=0.7)
        axes[1].grid(True, axis='x', linestyle='--', alpha=0.7)

        axes[0].legend(title='Dataset', loc='upper right', fontsize=10)
        axes[1].legend(title='Dataset', loc='upper right', fontsize=10)

        plt.tight_layout()

    plt.show()


plot_normalized_countplots(train, test, cat_features)


# Let's perform adversarial validation with only feature in the train and test sets at a time to gauge the similarity of distributions 
class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, target, params=None, paradigm='train_v_test', seed=55, verbose=False):
        self.train = train.copy()
        self.test = test.copy()
        if paradigm != 'train_v_test':
            self.original = original.copy()
        self.features = features
        self.cat_features = cat_features
        self.target = target
        self.seed = seed
        self.verbose = verbose
        
        self.params = params or {
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'n_estimators': 100, 
            'gamma': 1, 
            'min_child_weight': 4,
            'verbosity': 0, 
            'enable_categorical': True,
            'eval_metric': 'logloss', 
            'early_stopping_rounds': 10,
            'random_state': seed 
        }

        self.paradigm = 0 if paradigm == 'train_v_test' else 1
        if self.paradigm == 0:
            self.df1, self.df2 = self.train.copy(), self.test.copy()
        else:
            self.df1 = pd.concat([self.train, self.test], axis=0).sample(frac=1.0, random_state=self.seed)
            self.df2 = self.original.copy().drop(target, axis=1)

    def run(self):
        self.df1 = self.df1.drop(columns=[self.target], errors='ignore') 
        self.df1['cat_'] = 0
        self.df2['cat_'] = 1
        
        df = pd.concat([self.df1, self.df2], axis=0).sample(frac=1.0, random_state=self.seed)
        
        num_features = [f for f in self.features if f not in self.cat_features]
        df_num = df[num_features+['cat_']]
        
        if len(self.cat_features) != 0:
            df_cat = df[self.cat_features].apply(lambda x: pd.factorize(x)[0])
            df = pd.concat([df_cat, df_num], axis=1)
        else:
            df = df_num
        

        X = df.drop(columns=['cat_'], errors='ignore')
        y = df['cat_']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**self.params)
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=False
            )
            
            oof_preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, oof_preds)
            scores.append(score)
            if self.verbose:
                print(f'Fold {fold + 1}: ROC-AUC score = {score:.5f}')

        print(f'Average ROC-AUC score: {np.mean(scores):.5f} Â± {np.std(scores):.5f}\n')
        return scores


for f in features:
    print(f'Feature: {f}')
    
    cat_feats = [f] if f in cat_features else []
        
    av = AdversarialValidation(
        train[[f]], test[[f]], None,
        [f], cat_feats, CFG.target, 
        paradigm='train_v_test'
    )
    _ = av.run()


# Let's gauge the similarity of distributions between the numerical columns in the train and test set

for f in num_features:
    train_data = train[f].dropna()
    test_data = test[f].dropna()
    
    ks_stat, p_value = ks_2samp(train_data, test_data)
    
    print(f"Feature: {f}")
    print(f"KS Statistic: {ks_stat}, P-value: {p_value}\n")
    if p_value > 0.05:
        print(f"Train and test distributions for '{f}' are similar\n")
    else:
        print(f"Train and test distributions for '{f}' are significantly different\n")



# Let's check the similarity of the distributions between the categorical columns in the train and test sets

for f in cat_features:
    train_counts = train[f].value_counts().sort_index()
    test_counts = test[f].value_counts().sort_index()

    all_categories = set(train_counts.index).union(set(test_counts.index))
    train_counts = train_counts.reindex(all_categories, fill_value=0)
    test_counts = test_counts.reindex(all_categories, fill_value=0)

    contingency_table = np.array([train_counts.values, test_counts.values])

    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    print(f"Feature: {f}")
    print(f"Chi-Square Statistic: {chi2_stat}, P-value: {p_value}")
    if p_value > 0.05:
        print(f"Train and test distributions for '{f}' are similar\n")
    else:
        print(f"Train and test distributions for '{f}' are significantly different\nn")


def plot_boxplots_outliers(train, test, num_features, method='zscore', threshold=3):
    sns.set_palette('muted')

    for f in num_features:
        if method == 'zscore':
            train_outliers = train[np.abs(zscore(train[f])) > threshold]
            test_outliers = test[np.abs(zscore(test[f])) > threshold]
        elif method == 'iqr':
            Q1_train, Q3_train = train[f].quantile(0.25), train[f].quantile(0.75)
            IQR_train = Q3_train - Q1_train
            lower_bound_train, upper_bound_train = Q1_train - 1.5 * IQR_train, Q3_train + 1.5 * IQR_train
            train_outliers = train[(train[f] < lower_bound_train) | (train[f] > upper_bound_train)]

            Q1_test, Q3_test = test[f].quantile(0.25), test[f].quantile(0.75)
            IQR_test = Q3_test - Q1_test
            lower_bound_test, upper_bound_test = Q1_test - 1.5 * IQR_test, Q3_test + 1.5 * IQR_test
            test_outliers = test[(test[f] < lower_bound_test) | (test[f] > upper_bound_test)]
        else:
            train_outliers, test_outliers = pd.DataFrame(), pd.DataFrame()  # No outlier detection

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        sns.boxplot(y=train[f], ax=axes[0], color='skyblue')
        if not train_outliers.empty:
            axes[0].scatter(y=train_outliers[f], x=[0] * len(train_outliers), color='red', label='Outliers', zorder=3)
        axes[0].set_title(f'Box Plot of {f} (Train)', fontsize=14)
        axes[0].set_ylabel(f'{f}', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.7)

        sns.boxplot(y=test[f], ax=axes[1], color='salmon')
        if not test_outliers.empty:
            axes[1].scatter(y=test_outliers[f], x=[0] * len(test_outliers), color='red', label='Outliers', zorder=3)
        axes[1].set_title(f'Box Plot of {f} (Test)', fontsize=14)
        axes[1].set_ylabel(f'{f}', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        for ax in axes:
            ax.legend()

        plt.tight_layout()
        plt.show()


def plot_violinplots_outliers(train, test, num_features, method='zscore', threshold=3):
    sns.set_palette('muted')

    for f in num_features:
        if method == 'zscore':
            train_outliers = train[np.abs(zscore(train[f])) > threshold]
            test_outliers = test[np.abs(zscore(test[f])) > threshold]
        elif method == 'iqr':
            Q1_train, Q3_train = train[f].quantile(0.25), train[f].quantile(0.75)
            IQR_train = Q3_train - Q1_train
            lower_bound_train, upper_bound_train = Q1_train - 1.5 * IQR_train, Q3_train + 1.5 * IQR_train
            train_outliers = train[(train[f] < lower_bound_train) | (train[f] > upper_bound_train)]

            Q1_test, Q3_test = test[f].quantile(0.25), test[f].quantile(0.75)
            IQR_test = Q3_test - Q1_test
            lower_bound_test, upper_bound_test = Q1_test - 1.5 * IQR_test, Q3_test + 1.5 * IQR_test
            test_outliers = test[(test[f] < lower_bound_test) | (test[f] > upper_bound_test)]
        else:
            train_outliers, test_outliers = pd.DataFrame(), pd.DataFrame()  # No outlier detection

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        sns.violinplot(y=train[f], ax=axes[0], color='skyblue', inner='quartile')
        if not train_outliers.empty:
            axes[0].scatter(y=train_outliers[f], x=[0] * len(train_outliers), color='red', label='Outliers', zorder=3)
        axes[0].set_title(f'Violin Plot of {f} (Train)', fontsize=14)
        axes[0].set_ylabel(f'{f}', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.7)

        sns.violinplot(y=test[f], ax=axes[1], color='salmon', inner='quartile')
        if not test_outliers.empty:
            axes[1].scatter(y=test_outliers[f], x=[0] * len(test_outliers), color='red', label='Outliers', zorder=3)
        axes[1].set_title(f'Violin Plot of {f} (Test)', fontsize=14)
        axes[1].set_ylabel(f'{f}', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        for ax in axes:
            ax.legend()

        plt.tight_layout()
        plt.show()


plot_boxplots_outliers(train, test, num_features)


plot_violinplots_outliers(train, test, num_features)


def detect_num_outliers(df, num_features, threshold=3, iqr_multiplier=1.5, method='zscore'):
    outliers = {}

    for col in num_features:
        if method == 'zscore':
            z_scores = np.abs(zscore(df[col]))  
            outlier_values = df[col][z_scores > threshold].tolist()
    
            if outlier_values:
                outliers[col] = outlier_values
        elif method == 'iqr':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
            outlier_values = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].tolist()

            if outlier_values:
                outliers[col] = outlier_values

    return outliers


def detect_cat_outliers(df, cat_features, threshold=0.01):
    outliers = {}

    for col in cat_features:
        value_counts = df[col].value_counts(normalize=True)  # Get normalized frequency
        rare_categories = value_counts[value_counts < threshold].index.tolist()

        if rare_categories:
            outliers[col] = rare_categories

    return outliers


print('Outliers in numerical columns of train dataset (according to Z-score method): ', detect_num_outliers(train, num_features, method='zscore'))
print('Outliers in numerical columns of train dataset (according to IQR method): ', detect_num_outliers(train, num_features, method='iqr'))
print('Outliers in categorical columns of train dataset: ', detect_cat_outliers(train, cat_features))

print('\nOutliers in numerical columns of test dataset (according to Z-score method): ', detect_num_outliers(test, num_features, method='zscore'))
print('Outliers in numerical columns of test dataset (according to IQR method): ', detect_num_outliers(test, num_features, method='iqr'))
print('Outliers in categorical columns of test dataset: ', detect_cat_outliers(test, cat_features))




