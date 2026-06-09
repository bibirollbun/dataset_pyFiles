import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e8/train.csv'
    test_path = '/kaggle/input/playground-series-s5e8/test.csv'
    original_path = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'
    target = 'y'
    seed = 50


train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')
original = pd.read_csv(CFG.original_path, delimiter=';')


display(train.head())
display(test.head())
display(original.head())


def preprocess(df):
    _df = df.copy()
    month_mapping = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    _df['month'] = _df['month'].map(month_mapping)
    return _df

train = preprocess(train)
test = preprocess(test)
original = preprocess(original)


train.duplicated().sum(), test.duplicated().sum(), original.duplicated().sum()


print(f'Shape of training data: {train.shape}')
print(f'Shape of test data: {test.shape}')
print(f'Shape of original data: {original.shape}')


# I see many 'unknown' values in the dataframes. Maybe that is the value used for imputation in this data.
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
print(original.isnull().sum().sum())


train


features = test.columns.tolist()
num_features = ['age', 'balance', 'duration', 'campaign', 'pdays']
cat_features = [f for f in features if f not in num_features]


disp_vc = lambda df, f: display((df[f].value_counts(normalize=True) * 100).reset_index())

for f in cat_features:
    disp_vc(train, f)
    disp_vc(test, f)
    disp_vc(original, f)
    print()


class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, num_features, target, params=None, paradigm='train_v_test', seed=99):
        self.train = train.copy()
        self.test = test.copy()
        self.original = original.copy()
        self.features = features
        self.cat_features = cat_features
        self.target = target
        self.seed = seed
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
        df_num = df[num_features+['cat_']]
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


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='original_v_synthetic',
    seed=CFG.seed
)
_ = av.run()


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='train_v_test',
    seed=CFG.seed
)
_ = av.run()


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

plt.title('Target Distribution (Train Data)', fontsize=18, weight='bold')
plt.tight_layout()
plt.show()


def barplot(df, feature, label='Train'):
    category_counts = df[feature].value_counts()
    colors = sns.color_palette('Set2')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
    
    plt.title(f'Distribution of {feature} ({label} data)', fontsize=18, weight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for f in cat_features:
    barplot(train, f, label='Train')
    barplot(test, f, label='Test')
    barplot(original, f, label='Original')


def histogram(df, feature, bins=30, label='Train'):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=bins, kde=True, color='skyblue', edgecolor='black')
    
    plt.title(f'Histogram of {feature} ({label} Data)', fontsize=18)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    sns.despine()
    plt.show()


bins_list = [30, 100, 100, 50, 30]
for f, bins in zip(num_features, bins_list):
    histogram(train, f, label='Train')
    histogram(test, f, label='Test')
    histogram(original, f, label='Original')


def crosstab(df, feature, target):
    ct = pd.crosstab(df[target], df[feature], normalize='index') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel(CFG.target)
    plt.show()


for f in cat_features:
    crosstab(train, f, CFG.target)
    crosstab(original, f, CFG.target)


train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')
original = pd.read_csv(CFG.original_path, delimiter=';')

original[CFG.target] = original[CFG.target].map({'no': 0, 'yes': 1})

train_size = len(train)
test_size = len(test)
cols = [f for f in train.select_dtypes(include='object')]
combined = pd.concat([train, test, original], axis=0)
encoder = OrdinalEncoder()
combined[cols] = encoder.fit_transform(combined[cols])

train = combined.iloc[:train_size].reset_index(drop=True)
test = combined.iloc[train_size:train_size+test_size].drop(CFG.target, axis=1).reset_index(drop=True)
original = combined.iloc[train_size+test_size:].reset_index(drop=True)


X = train.drop(CFG.target, axis=1)
y = train[CFG.target]

m = RandomForestClassifier(n_jobs=-1)
m.fit(X, y)

importances = m.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feat_imp,
    palette='crest'
)

plt.title('Random Forest Feature Importance', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# Let's check feature importances again after dropping some of the least important columns
X = train.drop(columns=[CFG.target, 'default', 'loan', 'previous', 'marital', 'education'])
y = train[CFG.target]

m = RandomForestClassifier(n_jobs=-1)
m.fit(X, y)

importances = m.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feat_imp,
    palette='crest'
)

plt.title('Random Forest Feature Importance', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# Let's try this with original data used for training as well, and check if the feature importances are any different than the aforementioned
_train = pd.concat([train, original], axis=0)
X = _train.drop(CFG.target, axis=1)
y = _train[CFG.target]

m = RandomForestClassifier(n_jobs=-1)
m.fit(X, y)

importances = m.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feat_imp,
    palette='crest'
)

plt.title('Random Forest Feature Importance', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# Let's remove the least important features again
_train = pd.concat([train, original], axis=0)
X = _train.drop(columns=['default', 'loan', 'previous', 'marital', 'education', CFG.target]) 
y = _train[CFG.target]

m = RandomForestClassifier(n_jobs=-1)
m.fit(X, y)

importances = m.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feat_imp,
    palette='crest'
)

plt.title('Random Forest Feature Importance', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()




