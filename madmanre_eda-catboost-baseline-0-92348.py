!pip install -U ydata-profiling > /dev/null
!pip install catboost > /dev/null


import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay

from ydata_profiling import ProfileReport, compare

from catboost import CatBoostClassifier
from catboost import Pool

from scipy.stats import chi2_contingency

import shap


try:
    train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

except:
    from google.colab import drive
    drive.mount('/content/gdrive')

    ROOT_DIR = '/content/gdrive/MyDrive/DeepMind/Kaggle/Predicting Loan Payback/'

    DATASET_DIR = './data/'
    TRAIN_DATASET_NAME = 'train.csv'
    TEST_DATASET_NAME = 'test.csv'

    TRAIN_DATASET_PATH = os.path.join(ROOT_DIR, DATASET_DIR, TRAIN_DATASET_NAME)
    TEST_DATASET_PATH = os.path.join(ROOT_DIR, DATASET_DIR, TEST_DATASET_NAME)

    train = pd.read_csv(TRAIN_DATASET_PATH)
    test = pd.read_csv(TEST_DATASET_PATH)


TARGET = 'loan_paid_back'
DROP = 'id'

SEED = 42


train_report = ProfileReport(train.drop([TARGET, DROP], axis=1), title="Train data report")
test_report = ProfileReport(test.drop([DROP], axis=1), title="Test data report")

comparison_report = train_report.compare(test_report)

comparison_report.to_notebook_iframe()


ht_data_train = train.drop([DROP, TARGET], axis=1)
ht_data_train['CLS'] = 0

ht_data_test = test.drop([DROP], axis=1)
ht_data_test['CLS'] = 1

ht__data_full = pd.concat([ht_data_train, ht_data_test])
ht__data_full = ht__data_full.sample(frac=1.)

print(f"Train dataset: {ht_data_train.shape}")
print(f"Test dataset: {ht_data_test.shape}")
print(f"Full dataset: {ht__data_full.shape}")


ht_X = ht__data_full.drop('CLS', axis=1)
ht_y = ht__data_full['CLS']

categorical_features = ht_X.select_dtypes(exclude=np.number).columns.tolist()
numerical_features = ht_X.select_dtypes(include=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    ht_X,
    ht_y,
    test_size=0.3,
    stratify=ht_y,
    random_state=SEED
    )


estimator = CatBoostClassifier(
    n_estimators=100,
    random_state=SEED,
    cat_features=categorical_features,
    verbose=0
)

estimator.fit(X_train, y_train)

RocCurveDisplay.from_estimator(estimator, X_test, y_test)


train_no_id = train.drop(DROP, axis=1)

n_features = len(numerical_features)
n_cols = 2
n_rows = n_features

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 4*n_rows))

if n_features == 1:
    axes = axes.reshape(1, -1)

palette = sns.color_palette("husl", len(train_no_id[TARGET].unique()))

for i, feature in enumerate(numerical_features):
    for j, target_value in enumerate(sorted(train_no_id[TARGET].unique())):
        subset = train_no_id[train_no_id[TARGET] == target_value]
        sns.kdeplot(data=subset, x=feature, label=f'{TARGET}={target_value}',
                   ax=axes[i, 0], fill=True, alpha=0.6, color=palette[j])

    axes[i, 0].set_title(f"KDE: {feature}", fontsize=12)
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].grid(True, alpha=0.3)

    sns.boxplot(data=train_no_id, x=TARGET, y=feature, ax=axes[i, 1], palette=palette)
    axes[i, 1].set_title(f"Boxplot: {feature}", fontsize=12)
    axes[i, 1].set_xlabel(TARGET)
    axes[i, 1].tick_params(axis='x', rotation=45)
    axes[i, 1].grid(True, alpha=0.3)

    stats_text = f"Î¼={train_no_id[feature].mean():.2f}, Ïƒ={train_no_id[feature].std():.2f}"
    axes[i, 0].set_title(f"KDE: {feature}\n({stats_text})", fontsize=10)

plt.tight_layout()
plt.show()


n_features = len(categorical_features)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    cross_tab = pd.crosstab(train_no_id[feature], train_no_id[TARGET], normalize='index') * 100

    cross_tab.plot(kind='bar', stacked=True, ax=axes[i],
                   color=['lightcoral', 'lightblue'])
    axes[i].set_title(f'{TARGET} -- {feature}')
    axes[i].set_ylabel('(%)')
    axes[i].legend(title=TARGET)
    axes[i].tick_params(axis='x', rotation=45)

    for container in axes[i].containers:
        axes[i].bar_label(container, fmt='%.1f%%', label_type='center')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    cross_tab = pd.crosstab(train_no_id[feature], train_no_id[TARGET], normalize='index') * 100

    sns.heatmap(cross_tab, annot=True, cmap='RdYlBu_r',
                ax=axes[i], cbar_kws={'label': '(%)'})
    axes[i].set_title(f'Heatmap: {feature} vs {TARGET}')
    axes[i].set_xlabel(TARGET)
    axes[i].set_ylabel(feature)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


def analyze_categorical_vs_binary(data, categorical_features, target):
    results = []

    for feature in categorical_features:
        cross_tab = pd.crosstab(data[feature], data[target])
        cross_tab_pct = pd.crosstab(data[feature], data[target], normalize='index') * 100

        chi2, p_value, dof, expected = chi2_contingency(cross_tab)

        n = cross_tab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))

        target_rate = data.groupby(feature)[target].mean().sort_values(ascending=False)

        results.append({
            'Feature': feature,
            'Chi2_p_value': p_value,
            'Cramers_V': cramers_v,
            'Categories': len(data[feature].unique()),
            'Most_risky_category': target_rate.index[0],
            'Highest_target_rate': target_rate.iloc[0]
        })

        print(f"\n=== {feature} ===")
        print(f"Chi-square p-value: {p_value:.4f}")
        print(f"CramÃ©r's V: {cramers_v:.4f}")
        print("\nDistribution:")
        display(pd.concat([cross_tab, cross_tab_pct],
                         keys=['Count', 'Percentage'], axis=1))

    summary_df = pd.DataFrame(results)
    print("\n=== REPORT ===")
    display(summary_df.sort_values('Cramers_V', ascending=False))

analyze_categorical_vs_binary(train_no_id, categorical_features, TARGET)


X = train.drop([DROP, TARGET], axis=1)
y = train[TARGET]

estimator = CatBoostClassifier(
    random_state=SEED,
    cat_features=categorical_features,
    verbose=0
)

estimator.fit(X, y)

features = X.columns
importance = estimator.get_feature_importance()

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['Feature'][:20][::-1], feature_importance_df['Importance'][:20][::-1])
plt.title('CatBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


sample = X.sample(2000, random_state=42)

pool = Pool(sample, label=None, cat_features=categorical_features)

shap_values = estimator.get_feature_importance(
    type='ShapValues',
    data=pool
)

shap_values_only = shap_values[:, :-1]
base_values = shap_values[:, -1]

explainer = shap.Explanation(
    values=shap_values_only,
    base_values=base_values,
    data=sample,
    feature_names=sample.columns
)

shap.summary_plot(
    explainer.values,
    sample,
    feature_names=sample.columns
)


pred = estimator.predict_proba(test.drop(DROP, axis=1))[:, 1]

baseline_submission = test[['id']].copy()
baseline_submission['loan_paid_back'] = pred

baseline_submission.to_csv('baseline_submission_v1.csv', index=False)

