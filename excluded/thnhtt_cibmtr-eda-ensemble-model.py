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


feature_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
feature_dict.head(5)


feature_dict


df_train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
df_test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


df_train.shape, df_test.shape


target_cols = [x for x in df_train.columns if x not in df_test.columns]
target_cols


df_train[['efs', 'efs_time']].describe()


df_train.select_dtypes(include = ['number']).corr()


import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt


missing_train = df_train.isnull()
missing_test = df_test.isnull()

fig, axes = plt.subplots(1, 2, figsize = (18, 6))

sns.heatmap(missing_train, cmap='viridis', cbar=True, yticklabels=False, ax=axes[0])
axes[0].set_title('Missing Values Heatmap - Training Dataset', fontsize=14)
axes[0].set_xlabel('Features', fontsize=12)
axes[0].set_ylabel('Entries', fontsize=12)

sns.heatmap(missing_test, cmap='viridis', cbar=True, yticklabels=False, ax=axes[1])
axes[1].set_title('Missing Values Heatmap - Test Dataset', fontsize=14)
axes[1].set_xlabel('Features', fontsize=12)
axes[1].set_ylabel('Entries', fontsize=12)

plt.tight_layout()
plt.show()


def missing_values_table(df):
    missing_cnt = df.isnull().sum()
    missing_percent = 100 * missing_cnt / len(df)
    datatypes = df.dtypes
    return pd.DataFrame({
        'Missing Values': missing_cnt,
        'Missing Percent': missing_percent,
        'Data Type': datatypes
    })


train_missing_table = missing_values_table(df_train)
test_missing_table = missing_values_table(df_test)


print("Missing Values Table - Training Dataset:\n")
display(train_missing_table[train_missing_table['Missing Values'] > 0].sort_values(by = ['Missing Percent'], ascending=False))
print("\n")

print("Missing Values Table - Test Dataset:\n")
display(test_missing_table[test_missing_table['Missing Values'] > 0].sort_values(by = ['Missing Percent'], ascending=False))


high_percent_missing_features = ['tce_match', 'mrd_hct', 'cyto_score_detail', 'tce_div_match', 'tce_imm_match', 'cyto_score']

feature_dict.query('@feature_dict.variable in @high_percent_missing_features')


train_missing = train_missing_table[train_missing_table['Missing Values'] > 0].sort_values(by = ['Missing Percent'], ascending=False)
test_missing = test_missing_table[test_missing_table['Missing Values'] > 0].sort_values(by = ['Missing Percent'], ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Bar plot for train dataset
train_colors = cm.get_cmap('viridis', len(train_missing))(range(len(train_missing)))
axes[0].barh(train_missing.index, train_missing['Missing Percent'], color=train_colors)
axes[0].set_title('Percentage of Missing Values (Train Data)', fontsize=12)
axes[0].set_xlabel('Percentage (%)', fontsize=10)
axes[0].set_ylabel('Features', fontsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.6)
axes[0].invert_yaxis()

# Bar plot for test dataset
test_colors = cm.get_cmap('viridis', len(test_missing))(range(len(test_missing)))
axes[1].barh(test_missing.index, test_missing['Missing Percent'], color=test_colors)
axes[1].set_title('Percentage of Missing Values (Test Data)', fontsize=12)
axes[1].set_xlabel('Percentage (%)', fontsize=10)
axes[1].grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


train_data = df_train.copy()
test_data = df_test.copy()


numeric_columns = train_data.select_dtypes(include=['number']).columns
for col in numeric_columns:
    if col in test_data.columns:
        median_value = train_data[col].median()  # Calculate the median
        train_data[col].fillna(median_value, inplace=True)
        test_data[col].fillna(median_value, inplace=True)


object_columns = train_data.select_dtypes(include=['object']).columns
for col in object_columns:
    if col in test_data.columns:
        train_data[col].fillna("<UNK>", inplace=True)
        test_data[col].fillna("<UNK>", inplace=True)


# Verify Missing Values

print("Missing Values After Imputation - Training Dataset:")
print(train_data.isnull().sum())

print("\nMissing Values After Imputation - Test Dataset:")
print(test_data.isnull().sum())


# Check for duplicate rows in the training dataset
train_duplicates = train_data.duplicated().sum() # ==============================================> New way to get number of duplicated rows
print(f"\nNumber of duplicate rows in the training dataset: {train_duplicates}")

# Check for duplicate rows in the test dataset
test_duplicates = test_data.duplicated().sum()
print(f"Number of duplicate rows in the test dataset: {test_duplicates}")


from scipy.signal import find_peaks


# Custom colormap using viridis
viridis_cmap = cm.get_cmap("viridis")

def visualize_premium_amount_with_peaks(data, feature='Premium Amount'):
    plt.figure(figsize=(9, 4))

    # Histogram with KDE
    plt.subplot(1, 2, 1)
    ax = sns.histplot(data[feature], bins=30, kde=True, color=viridis_cmap(0.5))
    plt.title(f'Histogram of {feature} with KDE', fontsize=11)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Extract KDE values to find peaks
    kde = sns.kdeplot(data[feature], ax=ax, color=viridis_cmap(0.7)).lines[0].get_data()
    kde_x, kde_y = kde[0], kde[1]
    peaks, _ = find_peaks(kde_y)

    # Highlight peaks
    for peak_idx in peaks:
        plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  # Red dots on peaks

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[feature], color=viridis_cmap(0.5))
    plt.title(f'Box Plot of {feature}', fontsize=11)
    plt.xlabel(feature, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

visualize_premium_amount_with_peaks(train_data, feature='efs_time')


columns_to_analyze = train_data.select_dtypes(include=['number']).columns.drop(target_cols)

viridis_cmap = cm.get_cmap("viridis")
# Extract three colors from the colormap
viridis_colors = [viridis_cmap(0.3), viridis_cmap(0.5), viridis_cmap(0.8)]

fig, axes = plt.subplots(len(columns_to_analyze), 3, figsize=(25, len(columns_to_analyze) * 5))

for i, column in enumerate(columns_to_analyze):
    # Histogram for train_data
    sns.histplot(train_data[column], bins=30, kde=True, color=viridis_colors[0], ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {column} (Train)', fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=10)
    axes[i, 0].set_ylabel('Frequency', fontsize=10)
    axes[i, 0].grid(visible=True, linestyle='--', alpha=0.6)

    # Boxplot for train_data
    sns.boxplot(x=train_data[column], color=viridis_colors[1], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {column} (Train)', fontsize=14)
    axes[i, 1].set_xlabel(column, fontsize=10)
    axes[i, 1].grid(visible=True, linestyle='--', alpha=0.6)

    # Boxplot for test_data
    sns.boxplot(x=test_data[column], color=viridis_colors[2], ax=axes[i, 2])
    axes[i, 2].set_title(f'Boxplot of {column} (Test)', fontsize=12)
    axes[i, 2].set_xlabel(column, fontsize=10)
    axes[i, 2].grid(visible=True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


cat_numeric_columns = []
for col in numeric_columns:
    if(col not in target_cols and len(train_data[col].unique()) < 20):
        cat_numeric_columns.append(col)

cat_numeric_columns


numeric_data = train_data.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(15, 12))

# Create the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    cbar=True,
    square=True,
    mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)),
    linewidths=0.5
)

plt.title('Correlation Heatmap of Numerical Features (Excluding Target)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


def plot_categorical_distribution(data, column_name):
    plt.figure(figsize=(18, 4))

    # Bar plot for categorical distribution
    plt.subplot(1, 2, 1)
    sns.countplot(y=column_name, data=data, palette='Set2')
    plt.title(f'Distribution of {column_name}', fontsize=12)
    plt.xlabel('Count', fontsize=10)
    plt.ylabel(column_name, fontsize=10)

    ax = plt.gca()
    for p in ax.patches:
        count = int(p.get_width())
        ax.annotate(f'{count}',
                    (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, color='black')

    sns.despine(left=True, bottom=True)

    # Pie chart for percentage distribution
    plt.subplot(1, 2, 2)
    data[column_name].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=sns.color_palette('Set2', data[column_name].nunique()),
        startangle=90,
        explode=[0.05] * data[column_name].nunique(),
        shadow=True
    )
    plt.title(f'Percentage Distribution of {column_name}', fontsize=12)
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

categorical_columns = train_data.select_dtypes(include=['object'])

for column in categorical_columns:
    plot_categorical_distribution(train_data, column)


from sklearn.preprocessing import StandardScaler


train_data = df_train.copy()
test_data = df_test.copy()


train_data.set_index('ID')
test_data.set_index('ID')


numeric_columns = train_data.select_dtypes(include=['number']).columns
for col in numeric_columns:
    if col in test_data.columns:
        median_value = train_data[col].median()  # Calculate the median
        train_data[col].fillna(median_value, inplace=True)
        test_data[col].fillna(median_value, inplace=True)

object_columns = train_data.select_dtypes(include=['object']).columns
for col in object_columns:
    if col in test_data.columns:
        train_data[col].fillna("<UNK>", inplace=True)
        test_data[col].fillna("<UNK>", inplace=True)


cat_columns = list(train_data.select_dtypes(include=['object']).columns)

# train_data = pd.get_dummies(train_data, columns=cat_columns, drop_first=True)
# test_data = pd.get_dummies(test_data, columns=cat_columns, drop_first=True)


numerical_columns = train_data.drop(target_cols, axis = 1).select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])


from pathlib import Path


class CFG:
    train_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
    test_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
    subm_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
    colorscale = 'Sunset'
    color = '#2EEFCA'
    batch_size = 32768
    early_stop = 300
    penalizer = 0.01
    n_splits = 5

    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'task_type': 'GPU',
        'num_trees': 6000,
        'bootstrap_type': 'MVS',
        'subsample': 0.8,
        'reg_lambda': 8.0,
        'depth': 8,
        'random_state': 86,
    }

    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 20,
        'num_iterations': 6000,
        'learning_rate': 0.01,
        'extra_trees': True,
        'reg_lambda': 3.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 10,
        'device': 'gpu',
        'max_bin': 255,
        'verbose': -1,
        'seed': 86
    }

    cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'num_trees': 6000,
        'bootstrap_type': 'MVS',
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'depth': 8,
        'random_state': 86,
    }

    cox2_params = {
        'grow_policy': 'Lossguide',
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'num_trees': 6500,
        'bootstrap_type': 'MVS',
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8,
        'random_state': 86,
    }

    cox3_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 16,
        'loss_function': 'Cox',
        'learning_rate': 0.02,
        'task_type': 'CPU',
        'num_trees': 7000,
        'bootstrap_type': 'MVS',
        'subsample': 0.5,  
        'reg_lambda': 6.0,
        'depth': 10,
        'random_state': 86,
    }


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter


from lifelines.utils import concordance_index
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


class MD:
    def __init__(self, early_stop, penalizer, n_splits, color):
        self.early_stop = early_stop
        self.penalizer = penalizer
        self.n_splits = n_splits
        self.color = color

    def create_target1(self, data, cat_cols):
        cph_data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(cph_data, duration_col='efs_time', event_col='efs')
        data['target1'] = cph.predict_partial_hazard(cph_data)
        return data

    def create_target2(self, data):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target2'] = kmf.survival_function_at_times(data['efs_time']).values
        return data

    def create_target3(self, data):
        naf = NelsonAalenFitter()
        naf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target3'] = naf.cumulative_hazard_at_times(data['efs_time']).values
        data['target3'] = data['target3'] * -1
        return data

    def create_target4(self, data):
        data['target4'] = data.efs_time.copy()
        data.loc[data.efs == 0, 'target4'] *= -1
        return data

    def train_model(self, data, cat_cols, params, target, title):
        for col in cat_cols:
            data[col] = data[col].astype('category')
            
        X = data.drop(['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'], axis=1)
        y = data[target]
        
        models, fold_scores = [], []
        
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=86)
        
        oof_preds = np.zeros(len(X))
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X)):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
            
            if title.startswith('LightGBM'):
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(self.early_stop, verbose=0),
                        lgb.log_evaluation(0)
                    ]
                )
                
            elif title.startswith('CatBoost'):
                model = CatBoostRegressor(**params, verbose=0, cat_features=cat_cols)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self.early_stop,
                    verbose=0
                )
                
            models.append(model)
            oof_preds[valid_index] = model.predict(X_valid)
            
            y_true_fold = data.iloc[valid_index][['ID', 'efs', 'efs_time', 'race_group']].copy()
            y_pred_fold = data.iloc[valid_index][['ID']].copy()
            y_pred_fold['prediction'] = oof_preds[valid_index]
            
            fold_score = score(y_true_fold, y_pred_fold, 'ID')
            fold_scores.append(fold_score)
        
        y_true = data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = data[['ID']].copy()
        y_pred['prediction'] = oof_preds
        
        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
        if target == 'target1':
            t = 'Cox Target'
        elif target == 'target2':
            t = 'Kaplan-Meier Target'
        elif target == 'target3':
            t = 'Nelson-Aalen Target'
        else:
            t = 'Cox Loss'
        print(f'\nOverall C-Index for {title} {t}: {c_index_score:.3f}\n')
        
        return models, oof_preds

    def infer_model(self, data, cat_cols, models):
        data = data.drop(['ID'], axis=1)
        for col in cat_cols:
            data[col] = data[col].astype('category')
        return np.mean([model.predict(data) for model in models], axis=0)


md = MD(CFG.early_stop, CFG.penalizer, CFG.n_splits, CFG.color)

train_data = md.create_target1(train_data, cat_columns)
train_data = md.create_target2(train_data)
train_data = md.create_target3(train_data)
train_data = md.create_target4(train_data)


print("Training CatBoost models...")
ctb1_models, _ = md.train_model(train_data, cat_columns, CFG.ctb_params, target='target1', title='CatBoost1')
ctb2_models, _ = md.train_model(train_data, cat_columns, CFG.ctb_params, target='target2', title='CatBoost2')
ctb3_models, _ = md.train_model(train_data, cat_columns, CFG.ctb_params, target='target3', title='CatBoost3')

print("\nTraining LightGBM models...")
lgb1_models, _ = md.train_model(train_data, cat_columns, CFG.lgb_params, target='target1', title='LightGBM1')
lgb2_models, _ = md.train_model(train_data, cat_columns, CFG.lgb_params, target='target2', title='LightGBM2')
lgb3_models, _ = md.train_model(train_data, cat_columns, CFG.lgb_params, target='target3', title='LightGBM3')

print("\nTraining Cox models...")
cox1_models, _ = md.train_model(train_data, cat_columns, CFG.cox1_params, target='target4', title='CatBoost')
cox2_models, _ = md.train_model(train_data, cat_columns, CFG.cox2_params, target='target4', title='CatBoost')
cox3_models, _ = md.train_model(train_data, cat_columns, CFG.cox3_params, target='target4', title='CatBoost')

ctb1_preds = md.infer_model(test_data, cat_columns, ctb1_models)
ctb2_preds = md.infer_model(test_data, cat_columns, ctb2_models)
ctb3_preds = md.infer_model(test_data, cat_columns, ctb3_models)

lgb1_preds = md.infer_model(test_data, cat_columns, lgb1_models)
lgb2_preds = md.infer_model(test_data, cat_columns, lgb2_models)
lgb3_preds = md.infer_model(test_data, cat_columns, lgb3_models)

cox1_preds = md.infer_model(test_data, cat_columns, cox1_models)
cox2_preds = md.infer_model(test_data, cat_columns, cox2_models)
cox3_preds = md.infer_model(test_data, cat_columns, cox3_models)


from scipy.stats import rankdata 


preds = [
    ctb1_preds, ctb2_preds, ctb3_preds,
    lgb1_preds, lgb2_preds, lgb3_preds,
    cox1_preds, cox2_preds, cox3_preds,
]

weights = [
    0.55, 3.0, 3.0,  # CatBoost weights
    0.55, 2.0, 2.0,  # LightGBM weights
    2.5, 2.5, 2.5,   # Cox weights 
]

# Create ranked predictions
ranked_preds = np.array([rankdata(p) for p in preds])
ensemble_preds = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)


# Create submission
subm_data = pd.read_csv(CFG.subm_path)
subm_data['prediction'] = ensemble_preds
subm_data.to_csv('submission.csv', index=False)
display(subm_data.head())

