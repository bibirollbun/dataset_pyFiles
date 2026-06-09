import os
import yaml
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import kruskal

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold

from typing import Sequence, Hashable

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

sns.set_palette("bright")
warnings.filterwarnings('ignore')


def apk(
    y_true: Hashable,
    y_pred: Sequence[Hashable],
    k: int = 3
) -> float:
    """
    Average Precision at k for a *single* ground-truth label.

    Parameters
    ----------
    y_true : Hashable
        The correct label.
    y_pred : Sequence[Hashable]
        Ordered model predictions (highest-ranked first).
    k : int, default 3
        Cut-off rank.

    Returns
    -------
    float
        AP@k.  For a single label this is either 0.0 (miss) or
        1 / (rank of first correct prediction), with rank starting at 1.
    """
    # keep only the top-k guesses
    y_pred = list(y_pred[:k])

    # drop later duplicates so that each label is evaluated once
    seen = set()
    deduped = []
    for label in y_pred:
        if label not in seen:
            deduped.append(label)
            seen.add(label)

    # find first hit, if any
    for rank, label in enumerate(deduped, start=1):
        if label == y_true:
            return 1.0 / rank

    return 0.0

def mapk(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Sequence[Hashable]],
    k: int = 3
) -> float:
    """
    Mean Average Precision at k (MAP@k) for a batch of samples that each
    have a single ground-truth label.

    Parameters
    ----------
    y_true : Sequence[Hashable]
        Ground-truth labels (length = n_samples).
    y_pred : Sequence[Sequence[Hashable]]
        Ranked prediction lists for every sample (same length as y_true).
    k : int, default 3
        Cut-off rank.

    Returns
    -------
    float
        MAP@k across the batch.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)=}, {len(y_pred)=}"
        )

    return float(
        np.mean([apk(t, p, k) for t, p in zip(y_true, y_pred)])
    )


def plot_feature_importance(models, features, max_features=None, figsize=(14, 4)):
    n_models = len(models)
    ncols = 2
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten()

    model_names = [f'Model {i+1}' for i in range(n_models)]

    for idx, (model, name) in enumerate(zip(models, model_names)):
        # Importance extraction -----------------------------------------
        if isinstance(model, CatBoostClassifier):
            importances = model.get_feature_importance()
            feature_names = features
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = features
        else:
            importances = np.zeros(len(model_names))
            feature_names = features

        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)
        if max_features is not None:
            fi_df = fi_df.head(max_features)

        ax = axes[idx]
        sns.barplot(
            data=fi_df,
            x="Importance",
            y="Feature",
            ax=ax,
            orient="h"
        )
        ax.set_title(name)
        ax.set_xlabel("Importance")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(
    oof_preds,
    train_df,
    target_col,
    validation_indices,
    *,
    categories,
    ncols=2,
    figsize=(6, 5),
):
    n_models = len(oof_preds)
    if n_models != len(validation_indices):
        raise ValueError("oof_preds and validation_indices must be the same length")

    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False,
    )

    for idx, (fold_preds, val_idx) in enumerate(zip(oof_preds, validation_indices)):
        y_pred = np.argmax(fold_preds, axis=1)
        y_true = train_df.iloc[val_idx][target_col].values

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

        r, c = divmod(idx, ncols)
        disp.plot(ax=axes[r][c], colorbar=False)
        axes[r][c].set_title(f"Model {idx+1}", fontweight="bold")
        axes[r][c].set_xticklabels(axes[r][c].get_xticklabels(), rotation=90)
        
    for extra in range(idx + 1, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()


config = {
    # Paths
    'train_path': '/kaggle/input/playground-series-s5e6/train.csv',
    'test_path': '/kaggle/input/playground-series-s5e6/test.csv',
    'output_folder': '../pred_outputs',
    # Base Features
    'cat_ftrs': ['Soil Type', 'Crop Type'],
    'num_ftrs': ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'],
    # Modeling
    'target': 'Fertilizer Name',
    'seed': 42,
    'n_splits': 5
}


train = pd.read_csv(config['train_path'], index_col='id')
test = pd.read_csv(config['test_path'], index_col='id')

print('Train Size:', train.shape)
print('Test Size:', test.shape)

train.head()


train.info()


# Checking for missing values
print('-Train Set-')
print('\tTotal count of missing rows:', train.isnull().sum().sum())

print('\n-Test Set-')
print('\tTotal count of missing rows:', test.isnull().sum().sum())


plt.figure(figsize=(12, 5))

for idx, ftr in enumerate(config['cat_ftrs']):
    plt.subplot(1, 2, idx+1)
    plt.title(f'{ftr}')
    ax = sns.countplot(y=ftr, data=train, order=train[ftr].value_counts().index)
    # plt.xticks(rotation=90)
    plt.ylabel('')
    
    # Add percentages
    total = len(train)
    for p in ax.patches:
        count = p.get_width()
        percentage = f'{100 * count / total:.1f}%'
        ax.text(count + total*0.0005, p.get_y() + p.get_height() / 2, percentage, va='center')

plt.tight_layout(pad=1.0)
plt.show()


plt.figure(figsize=(8, 5))

ax = sns.countplot(y='Fertilizer Name', data=train, order=train['Fertilizer Name'].value_counts().index)
plt.xticks(rotation=90)
plt.ylabel('')
    
# Add percentages
total = len(train)
for p in ax.patches:
    count = p.get_width()
    percentage = f'{100 * count / total:.1f}%'
    ax.text(count + total*0.0005, p.get_y() + p.get_height() / 2, percentage, va='center')

plt.tight_layout(pad=1.0)
plt.show()


plt.figure(figsize=(15, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.histplot(x=ftr, data=train, bins=30)
    plt.ylabel('')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.boxplot(x=ftr, data=train)
    plt.ylabel('')

plt.tight_layout()
plt.show()


temp_df = train[config['num_ftrs']].copy()
temp_df = pd.concat([temp_df, test[config['num_ftrs']]], ignore_index=True)

temp_df['Train/Test'] = ''
temp_df.loc[train.index, 'Train/Test'] = 'Train'
temp_df.loc[test.index, 'Train/Test'] = 'Test'


plt.figure(figsize=(15, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.histplot(x=ftr, data=temp_df, hue='Train/Test', bins=30, multiple='stack')
    plt.ylabel('')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.boxplot(y=ftr, x='Train/Test', data=temp_df)
    plt.ylabel('')

plt.tight_layout()
plt.show()


del temp_df


train.groupby('Soil Type')[config['num_ftrs']].agg(['mean', 'median', 'std']).T


plt.figure(figsize=(15, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.boxplot(y=ftr, x='Soil Type', data=train)
    plt.ylabel('')

plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.boxplot(y=ftr, x='Crop Type', data=train)
    plt.ylabel('')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


result = pd.crosstab(train['Crop Type'], train['Soil Type'], normalize='index') * 100
result = result.round(3)
result


ax = result.plot(kind='bar', stacked=False, figsize=(12, 4))
ax.set_ylabel('Percentage of Soil Type (%)')
ax.set_title('Soil Type Distribution within each Crop Type')
ax.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left');


result = pd.crosstab(train['Soil Type'], train['Fertilizer Name'], normalize='index') * 100
result = result.round(3)
result


ax = result.plot(kind='bar', stacked=False, figsize=(12, 4))
ax.set_ylabel('Percentage of Fertilizer (%)')
ax.set_title('Fertilizer Distribution within each Soil Type')
ax.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left');


result = pd.crosstab(train['Crop Type'], train['Fertilizer Name'], normalize='index') * 100
result = result.round(3)
result


ax = result.plot(kind='bar', stacked=False, figsize=(12, 4))
ax.set_ylabel('Percentage of Fertilizer (%)')
ax.set_title('Fertilizer Distribution within each Crop Type')
ax.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left');


result = pd.crosstab([train['Crop Type'], train['Soil Type']], train['Fertilizer Name'], normalize='index') * 100
result = result.round(3)
result.index = [f'{crop} | {soil}' for crop, soil in result.index]

plt.figure(figsize=(20, 12))
sns.heatmap(result, annot=True, cbar=False, fmt='.1f', cmap='Blues')
plt.title('Target Variable % by Crop Type and Soil Type')
plt.xlabel('Target Variable')
plt.ylabel('Crop Type | Soil Type')
plt.show()


plt.figure(figsize=(18, 10))

for idx, ftr in enumerate(config['num_ftrs']):
    plt.subplot(2, 3, idx+1)
    plt.title(f'{ftr}')
    sns.violinplot(y=ftr, x='Fertilizer Name', data=train)
    plt.ylabel('')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


temp = train.copy()
temp = temp.rename(columns={'Fertilizer Name': 'Fertilizer'})

for idx, ftr in enumerate(config['num_ftrs']):
    
    model = smf.ols(f'{ftr} ~ C(Fertilizer)', data=temp).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    anova_table['eta_sq'] = anova_table['sum_sq'] / sum(anova_table['sum_sq'])

    if idx != 0:
        print()
    
    print(f'Feature: {ftr}')
    print(anova_table)


for idx, ftr in enumerate(config['num_ftrs']):
    groups = [train.loc[train['Fertilizer Name'] == cls, ftr]
              for cls in train['Fertilizer Name'].unique()]
    
    stat, p_value = kruskal(*groups)

    H = stat
    k = len(groups)
    n = sum([len(g) for g in groups])
    
    epsilon_sq = (H - k + 1) / (n - k)

    if idx != 0:
        print()
    print(f'Feature: {ftr}')
    print(f"\tKruskal-Wallis H-statistic: {stat:.3f}")
    print(f"\tP-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\t\tResult: Reject null hypothesis: means differ between classes.")
    else:
        print("\t\tResult: Fail to reject null: no significant difference in means.")

    print(f"\tEffect size (epsilon squared): {epsilon_sq:.6f}")


features = config['num_ftrs'] + config['cat_ftrs'] 
X = train[features]
y = train['Fertilizer Name'].astype('category').cat.codes

for col in config['cat_ftrs']:
    X[col] = X[col].astype('category').cat.codes

mi_scores = mutual_info_classif(X, y, discrete_features=True)

mi_df = pd.DataFrame({'Feature': features, 'MI': mi_scores})
mi_df = mi_df.sort_values('MI', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(mi_df['Feature'], mi_df['MI'], color='skyblue')
plt.xlabel('Mutual Information')
plt.title('Mutual Information of Features with Fertilizer')
plt.gca().invert_yaxis()
plt.show()


new_num_ftrs = []
new_cat_ftrs = []


train['Soil Code'] = train['Soil Type'].astype('category').cat.codes
train['Crop Code'] = train['Crop Type'].astype('category').cat.codes

for cat in ['Soil Code', 'Crop Code']:
    for num in config['num_ftrs']:
        train[f'{cat[:-5]}_{num}'] = train[cat] * train[num]
        new_num_ftrs.append(f'{cat[:-5]}_{num}')


train['Crop_x_Soil'] = train['Soil Type'].astype('str') + '_' + train['Crop Type'].astype('str')
train['Crop_x_Soil Code'] = train['Crop_x_Soil'].astype('category').cat.codes

for num in config['num_ftrs']:
    train[f'{num}_x_Crop_x_Soil'] = train[num] * train['Crop_x_Soil Code']
    new_num_ftrs.append(f'{num}_x_Crop_x_Soil')

new_cat_ftrs.append('Crop_x_Soil')


train.drop(['Soil Code', 'Crop Code', 'Crop_x_Soil Code'], axis=1, inplace=True)


features = config['num_ftrs'] + config['cat_ftrs'] + new_num_ftrs + new_cat_ftrs
X = train[features]
y = train['Fertilizer Name'].astype('category').cat.codes

for col in config['cat_ftrs'] + new_cat_ftrs:
    X[col] = X[col].astype('category').cat.codes

mi_scores = mutual_info_classif(X, y, discrete_features=True)

mi_df = pd.DataFrame({'Feature': features, 'MI': mi_scores})
mi_df = mi_df.sort_values('MI', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(mi_df['Feature'], mi_df['MI'], color='skyblue')
plt.xlabel('Mutual Information')
plt.title('Mutual Information of Features with Fertilizer')
plt.gca().invert_yaxis()
plt.show()


for col in config['cat_ftrs'] + ['Fertilizer Name']:
    train[col] = train[col].astype('category')
    if col != 'Fertilizer Name':
        test[col] = test[col].astype('category')


train.columns


def xgb_trainer(model_to_use,
                train,
                test,
                features,
                target,
                config,
                params,
                use_stratified_kfold=True):
    
    params = params or {}

    models = []
    validation_indices = []
    # Scores
    val_acc_scores = []
    train_acc_scores = []

    val_log_scores = []
    train_log_scores = []

    # Predictions
    oof_preds = []
    test_preds = []

    cv = (
        StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
        if use_stratified_kfold
        else KFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
    )

    train[target] = train[target].astype('category').cat.codes

    for idx, (train_idx, val_idx) in enumerate(cv.split(train, train[target])):
        X_train = train.iloc[train_idx][features]
        y_train = train.iloc[train_idx][target]
        X_val = train.iloc[val_idx][features]
        y_val = train.iloc[val_idx][target]

        if model_to_use == 'xgb':
            model = XGBClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=250
            )
        elif model_to_use == 'cat':
            model = CatBoostClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=250
            )
        else:
            pass
        models.append(model)
        validation_indices.append(val_idx)
        # Predictions -----------------------------------------------------
        ## Predicted Probabilities
        pred_proba_training = model.predict_proba(X_train)
        pred_proba_valid = model.predict_proba(X_val)
        pred_proba_test = model.predict_proba(test[features])

        oof_preds.append(pred_proba_valid)
        test_preds.append(pred_proba_test)

        # Post-processing -----------------------------------------------------
        pred_training_top3 = np.argsort(pred_proba_training)[:, -3:][:, ::-1]
        pred_valid_top3 = np.argsort(pred_proba_valid)[:, -3:][:, ::-1]

        # Scores -----------------------------------------------------

        ## Accuracy
        train_acc = accuracy_score(y_train, np.argmax(pred_proba_training, axis=1))
        valid_acc = accuracy_score(y_val, np.argmax(pred_proba_valid, axis=1))

        train_acc_scores.append(train_acc)
        val_acc_scores.append(valid_acc)

        ## LOG LOSS
        train_log_loss = log_loss(y_train, pred_proba_training)
        valid_log_loss = log_loss(y_val, pred_proba_valid)

        train_log_scores.append(train_log_loss)
        val_log_scores.append(valid_log_loss)

        ## MAP@3
        train_map3 = mapk(y_train.values.tolist(), pred_training_top3.tolist(), k=3)
        valid_map3 = mapk(y_val.values.tolist(), pred_valid_top3.tolist(), k=3)

        scores = {
            'train_acc': round(train_acc, 3),
            'val_acc': round(valid_acc, 3),
            'train_log_loss': round(train_log_loss, 4),
            'val_log_loss': round(valid_log_loss, 4),
            'train_MAP3': round(train_map3, 4),
            'val_MAP3': round(valid_map3, 4)
        }

        print(f"\nFOLD: {idx + 1}\n{'-' * 10}\nTraining Acc.: {train_acc:.4f}\nValidation Acc.: {valid_acc:.4f}\nTraining Log-Loss: {train_log_loss:.4f}\nValidation Log-Loss: {valid_log_loss:.4f}\nTraining MAP@3 Score: {train_map3:.4f}\nValidation MAP@3 Score: {valid_map3:.4f}\n\n{'-' * 50}\n")

    
    print(f"\n{'=' * 50}")
    print(f"TRAINING Accuracy SCORE: {np.mean(train_acc_scores):.4f} ± {np.std(train_acc_scores):.4f}")
    print(f"VALIDATION Accuracy SCORE: {np.mean(val_acc_scores):.4f} ± {np.std(val_acc_scores):.4f}")
    print('-'*25)
    print(f"TRAINING LOG-LOSS SCORE: {np.mean(train_log_scores):.4f} ± {np.std(train_log_scores):.4f}")
    print(f"VALIDATION LOG-LOSS SCORE: {np.mean(val_log_scores):.4f} ± {np.std(val_log_scores):.4f}")
    print(f"{'=' * 50}\n")

    agg_scores = {
        'accuracy': {'mean': round(np.mean(train_acc_scores), 4), 'std': round(np.std(train_acc_scores), 4)},
        'log_loss': {'mean': round(np.mean(train_log_scores), 4), 'std': round(np.std(val_log_scores), 4)}
    }

    return models, oof_preds, test_preds, agg_scores, validation_indices


categories = train[config['target']].astype('category').cat.categories.tolist()
features = config['cat_ftrs'] + config['num_ftrs']


categories


XGB_PARAMS = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'colsample_bytree': 0.7,
    'device': 'cuda',
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 7,
    'early_stopping_rounds': 250,
    'enable_categorical': True,
    'random_state': config['seed']
}


features = config['cat_ftrs'] + config['num_ftrs']

models, oof_preds, test_preds, agg_scores, validation_indices = xgb_trainer('xgb',
                                                                          train,
                                                                          test,
                                                                          features,
                                                                          config['target'],
                                                                          config,
                                                                          XGB_PARAMS,
                                                                          use_stratified_kfold=True)


plot_feature_importance(models, features)


plot_confusion_matrices(
    oof_preds=oof_preds,                 
    train_df=train,                     
    target_col="Fertilizer Name",       
    validation_indices=validation_indices,
    categories=categories,              
    ncols=2
)


final_test_preds = np.mean(test_preds, axis=0)
final_test_top3 = np.argsort(final_test_preds)[:, -3:][:, ::-1]
final_test_top3_labels = np.array(categories)[final_test_top3]


submission = pd.DataFrame({
    'id': test.index.tolist(),
    'Fertilizer Name': [' '.join(row) for row in final_test_top3_labels]})

submission.to_csv(f'./submission.csv', index=False)


submission




