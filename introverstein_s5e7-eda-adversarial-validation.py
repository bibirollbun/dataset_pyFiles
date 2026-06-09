!pip install umap-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import umap.umap_ as umap
from xgboost import XGBClassifier
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    original_paths = ['/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv', '/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv']
    seed = 42
    target = 'Personality'
    original_available = False


train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')
if CFG.original_available:
    orig1 = pd.read_csv(CFG.original_paths[0])
    orig2 = pd.read_csv(CFG.original_paths[1])
    original = pd.concat([orig1, orig2])


display(train.duplicated().sum(), test.duplicated().sum())
if CFG.original_available:
    display(original.duplicated().sum())


# # The original dataframe has duplicate rows, need to remove them before further analysis
# original = original.drop_duplicates()


print(f'Shape of train data: {train.shape}')
print(f'Shape of test data: {test.shape}')
if CFG.original_available:
    print(f'Shape of original data: {original.shape}')


def missing_values_table(df):
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


print('Training data:\n')
display(missing_values_table(train))

print('Testing data:\n')
display(missing_values_table(test))

if CFG.original_available:
    print('Original data:\n')
    display(missing_values_table(original))


print('Training data:\n')
display(train.dtypes.reset_index().rename(columns={'index': '', 0: ''}))

print('Testing data:\n')
display(test.dtypes.reset_index().rename(columns={'index': '', 0: ''}))

if CFG.original_available:
    print('Original data:\n')
    display(original.dtypes.reset_index().rename(columns={'index': '', 0: ''}))


# Let's have a look at the number of unique values in each feature 
for col in test.columns: 
    print(f'Feature `{col}` with dtype {test[col].dtype}, has:')
    print(f'{train[col].nunique()} unique values in train set, ')
    if CFG.original_available:
        print(f'{original[col].nunique()} unique values in original data,')
    print(f'{test[col].nunique()} unique values in test set.\n')


# We have a categorical target which takes two possible values, so let's plot a pie chart
category_counts = train[CFG.target].value_counts()
colors = sns.color_palette('Set2', len(category_counts))

plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    category_counts,
    labels=category_counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=120,
    wedgeprops=dict(width=0.4, edgecolor='w'),
    textprops=dict(color='black', fontsize=13)
)

plt.setp(autotexts, weight='bold')
plt.setp(texts, weight='semibold')

plt.title('Target Distribution (Train Data)', fontsize=18, weight='bold')
plt.tight_layout()
plt.show()


if CFG.original_available:
    category_counts = original[CFG.target].value_counts()
    colors = sns.color_palette('Set2', len(category_counts))
    
    plt.figure(figsize=(4, 4))
    wedges, texts, autotexts = plt.pie(
        category_counts,
        labels=category_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=120,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops=dict(color='black', fontsize=13)
    )
    
    plt.setp(autotexts, weight='bold')
    plt.setp(texts, weight='semibold')
    
    plt.title('Target Distribution (Original Data)', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.show()


feature = 'Time_spent_Alone'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Time_spent_Alone'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Time_spent_Alone'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# The original dataset has 63 rows where `Time_spent_Alone` has the value 4.505816002819881, which happens to be the mean of the column `Time_spent_Alone`. 
# So, the original data seems to have undergone imputation. Yet, null values exist. That is an interesting quirk of the original data.
# A question which comes to mind-- Why does the `Time_spent_Alone` column have null values even when they has been clearly imputed with mean of the column?
# Interestingly enough, there are 63 null values in the original dataset.
if CFG.original_available:
    original[feature].mean(), original[feature].isnull().sum()


if CFG.original_available:
    original[feature].value_counts().reset_index()['Time_spent_Alone'].tolist()[-1]


# Mean of `Time_spent_Alone` column without considering imputed rows
if CFG.original_available:
    original[~(original[feature] == 4.505816002819881)]['Time_spent_Alone'].mean()


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Stage_fear'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Stage_fear'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Stage_fear'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Social_event_attendance'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Social_event_attendance'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Social_event_attendance'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if CFG.original_available:
    original[feature].value_counts().reset_index()[feature].tolist()[-2]


if CFG.original_available:
    original[~(original[feature] == 3.963354474982382)][feature].mean()


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Going_outside'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Going_outside'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Going_outside'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Drained_after_socializing'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Drained_after_socializing'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Drained_after_socializing'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Friends_circle_size'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Friends_circle_size'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Friends_circle_size'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if CFG.original_available:
    display(original[feature].value_counts().reset_index()[feature].tolist()[-1])


if CFG.original_available:
    display(original[~(original[feature] == 6.268862911795962)][feature].mean())


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


feature = 'Post_frequency'
category_counts = train[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Train Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature = 'Post_frequency'
category_counts = test[feature].value_counts()
colors = sns.color_palette('Set2')

plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

plt.title(f'Distribution of {feature} (Test Data)', fontsize=18, weight='bold')
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if CFG.original_available:
    feature = 'Post_frequency'
    category_counts = original[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} (Original Data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if CFG.original_available:
    display(original[feature].unique()[-2])


if CFG.original_available:
    display(original[~(original[feature] == 3.564726631393298)][feature].mean())


ct = pd.crosstab(train[CFG.target], train[feature], normalize='index') * 100
plt.figure(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
plt.xlabel(feature)
plt.ylabel(CFG.target)
plt.show()


if CFG.original_available:
    ct = pd.crosstab(original[CFG.target], original[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, num_features, target, params=None, paradigm='train_v_test', seed=99):
        self.train = train.copy()
        self.test = test.copy()
        if original is not None:
            self.original = original.copy()
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.target = target
        self.seed = seed
        self.params = params or {
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'n_estimators': 10000, 
            'gamma': 1, 
            'min_child_weight': 4,
            'verbosity': 0, 
            'enable_categorical': True,
            'eval_metric': 'logloss', 
            'early_stopping_rounds': 100,
            'random_state': seed 
        }
        self.paradigm = 0  if paradigm == 'train_v_test' else 1

        if self.paradigm == 0:
            self.df1, self.df2 = self.train.copy(), self.test.copy()
        else:
            self.df1 = pd.concat([self.train, self.test], axis=0).sample(frac=1.0, random_state=self.seed)
            self.df2 = self.original.copy().drop(target, axis=1)

    def run(self):
        self.df1 = self.df1.drop(self.target, axis=1, errors='ignore')
        self.df1['cat_'] = 0
        self.df2['cat_'] = 1

        df = pd.concat([self.df1, self.df2], axis=0).sample(frac=1.0, random_state=self.seed)
        df_num = df[self.num_features+['cat_']]
        df_cat = df[self.cat_features].apply(lambda x: pd.factorize(x)[0])
        df = pd.concat([df_cat, df_num], axis=1)

        X = df.drop(columns=['cat_'])
        y = df['cat_']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        fold_scores = []

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            oof_preds = model.predict_proba(X_val)[:, 1]

            fold_score = roc_auc_score(y_val, oof_preds)
            fold_scores.append(fold_score)
            print(f'Fold {fold}: ROC-AUC score = {fold_score:4f}')

        print(f'\nAverage ROC-AUC score: {np.mean(fold_scores):.4f}Â±{np.std(fold_scores):.4f}')

        return fold_scores


features = test.columns.tolist()

av = AdversarialValidation(
    train=train, 
    test=test,
    original=None,
    features=features,
    cat_features=features,
    num_features=[],
    target=CFG.target,
    paradigm='train_v_test',
    seed=CFG.seed
)
_ = av.run()


if CFG.original_available:
    av = AdversarialValidation(
        train=train, 
        test=test,
        original=original,
        features=features,
        cat_features=features,
        num_features=[],
        target=CFG.target,
        paradigm='comp_v_original',
        seed=CFG.seed
    )
    _ = av.run()


features = train.drop(CFG.target, axis=1)
target = train[CFG.target]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(features)

pca = PCA(n_components=2, random_state=CFG.seed)
reduced = pca.fit_transform(encoded_features)

pca_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
pca_df['target'] = target.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
plt.title('2D PCA Projection (Train Data)', fontsize=16, weight='bold')
plt.legend(title='Class', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()


if CFG.original_available:
    features = original.drop(CFG.target, axis=1)
    target = original[CFG.target]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(features)
    
    pca = PCA(n_components=2, random_state=CFG.seed)
    reduced = pca.fit_transform(encoded_features)
    
    pca_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    pca_df['target'] = target.values
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
    plt.title('2D PCA Projection (Original Data)', fontsize=16, weight='bold')
    plt.legend(title='Class', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()


features = train.drop(CFG.target, axis=1)
target = train[CFG.target]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, random_state=CFG.seed, n_iter=1000)
reduced = tsne.fit_transform(encoded_features)

tsne_df = pd.DataFrame(reduced, columns=['TSNE1', 'TSNE2'])
tsne_df['target'] = target.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
plt.title('2D t-SNE Projection (Train Data)', fontsize=16, weight='bold')
plt.legend(title='Class', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()


if CFG.original_available:
    features = original.drop(CFG.target, axis=1)
    target = original[CFG.target]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(features)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=CFG.seed, n_iter=1000)
    reduced = tsne.fit_transform(encoded_features)
    
    tsne_df = pd.DataFrame(reduced, columns=['TSNE1', 'TSNE2'])
    tsne_df['target'] = target.values
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
    plt.title('2D t-SNE Projection (Original Data)', fontsize=16, weight='bold')
    plt.legend(title='Class', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()


features = train.drop(CFG.target, axis=1)
target = train[CFG.target]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(features)

reducer = umap.UMAP(n_components=2, random_state=CFG.seed)
reduced = reducer.fit_transform(encoded_features)

umap_df = pd.DataFrame(reduced, columns=['UMAP1', 'UMAP2'])
umap_df['target'] = target.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
plt.title('2D UMAP Projection (Train Data)', fontsize=16, weight='bold')
plt.legend(title='Class', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.show()


if CFG.original_available:
    features = original.drop(CFG.target, axis=1)
    target = original[CFG.target]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(features)
    
    reducer = umap.UMAP(n_components=2, random_state=CFG.seed)
    reduced = reducer.fit_transform(encoded_features)
    
    umap_df = pd.DataFrame(reduced, columns=['UMAP1', 'UMAP2'])
    umap_df['target'] = target.values
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='target', palette='Set2', s=60, edgecolor='w', alpha=0.6)
    plt.title('2D UMAP Projection (Original Data)', fontsize=16, weight='bold')
    plt.legend(title='Class', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()




