!pip install seaborn  --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
assert int(sns.__version__.split('.')[1]) >= 13     # This notebook needs seaborn version >= 0.13


df = pd.read_csv('../input/playground-series-s5e3/train.csv')

df_test = pd.read_csv('../input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('../input/playground-series-s5e3/sample_submission.csv')

print(f'Train shape: {df.shape}')
print(f'Test shape: {df_test.shape}')


def colorize(styler):
    styler.format(precision=2)
    styler.background_gradient(cmap='Blues')
    return styler


def info(df):
    summary = pd.DataFrame()
    summary['dtype'] = df.dtypes
    summary['null'] = df.isna().sum()
    summary['unique'] = df.nunique()

    summary = pd.concat([summary, df.describe().T], axis=1)
    return summary.style.pipe(colorize)


info(df)


info(df_test)


# Configure matplotlib + seaborn
rc = {
    "axes.facecolor": "#FFFFFF",  
    "figure.facecolor": "#F8F8F8",  
    "axes.labelcolor": "#000000",
    "grid.color": "#EBEBE7" + "30",  
    "axes.edgecolor": "#F8F8F8",  
    "axes.labelcolor": "#000000",
}


sns.set(rc=rc)

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, 5))
palette = sns.color_palette(colors) 


num_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
               'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

target_column = 'rainfall'


rainfall = df[target_column].value_counts()

fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

ax[0].pie(rainfall, labels=rainfall.index, autopct='%1.1f%%', colors=colors)
ax[0].set_title(f'{target_column} distribution (Pie-chart)')

ax[1].bar(rainfall.index, rainfall.values, color=colors)
ax[1].set_title(f'{target_column} distribution (Bar-chart)')
ax[1].set_ylabel('Count')
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['no', 'yes'])

plt.tight_layout()
plt.show()


bins = 50
ncols = 4
nrows = 5

def enumerate_step(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step

fig = plt.figure(figsize=(16, 16))
for i, col in enumerate_step(num_columns, start=1, step=2):
    ax = fig.add_subplot(nrows, ncols, i)
    sns.boxplot(df, x=col, hue=target_column, ax=ax, palette=palette, gap=.1)
    ax.set_title(f'{col} (boxplot)')
    ax.legend(loc='upper left', title='rainfall')
    
    ax = fig.add_subplot(nrows, ncols, i+1)
    sns.histplot(df, x=col, hue=target_column, ax=ax, palette=palette, kde=True, bins=bins)
    ax.set_title(f'{col} (kde)')

plt.tight_layout()
plt.show()


window = 5

fig, ax = plt.subplots(len(num_columns) +1, 1, figsize=(16, 28))

ax[0].plot(df[target_column].rolling(window=10).mean(), color=colors[1])
ax[0].set_title(f'{target_column} - mean')

for i, feature in enumerate(num_columns, start=1):
    rolling_min = df[feature].rolling(window=window).min()
    rolling_mean = df[feature].rolling(window=window).mean()
    rolling_max = df[feature].rolling(window=window).max()
    
    ax[i].plot(rolling_min, color=colors[0], label='min')
    ax[i].plot(rolling_mean, color=colors[1], label='mean')
    ax[i].plot(rolling_max, color=colors[2], label='max')
    ax[i].set_title(f'{feature} (window size={window})')
    ax[i].legend()

plt.tight_layout()
plt.show()


import itertools
import math

combinations = list(itertools.combinations(num_columns, 2))
corr = df.corr()

ncols = 4
nrows = math.ceil(len(combinations) / ncols)

fig, ax = plt.subplots(nrows, ncols, figsize=(20, 40))
ax = ax.flatten()

for i, combination in enumerate(combinations):
    target0 = df[df[target_column] == 0]
    target1 = df[df[target_column] == 1]
    
    ax[i].scatter(target0.loc[:, combination[0]], target0.loc[:, combination[1]], 
                  s=4, label='no', color=colors[0])
    
    ax[i].scatter(target1.loc[:, combination[0]], target1.loc[:, combination[1]], 
                  s=4, label='yes', color=colors[2])

    feat_corr = corr.at[combination[0], combination[1]]
    ax[i].set_title(f'{combination[0]} vs. {combination[1]}\n(corr: {feat_corr:.4f})')
    ax[i].legend(title=target_column)


for i in range(len(combinations), nrows * ncols):   # Disable remaining charts...
    ax[i].axis('off')

plt.tight_layout()
plt.show()


def extract_season(month):
    if month in [12, 1, 2]: 
        return 'winter'
    elif month in [3, 4, 5]: 
        return 'spring'
    elif month in [6, 7, 8]: 
        return 'summer'
    else: 
        return 'autumn'


def feature_eng(df, inplace=False):
    if not inplace:
        df = df.copy()

    # Convert to datetime
    df['date'] = pd.to_datetime(df['day'], format='%j')
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week

    # Extract season and quarter
    df['quarter'] = df['month'].apply(lambda x: (x - 1) // 3 + 1)
    df['season'] = df['month'].apply(extract_season)

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create temparature columns
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_var'] = df[['maxtemp', 'mintemp']].std(axis=1)
    df['temp_avg'] = df[['maxtemp', 'temparature', 'mintemp']].mean(axis=1)

    # Create other columns...
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    df['temp_humidity'] = df['temparature'] * df['humidity']

    return df.drop('date', axis=1)


def preprocessing(df, inplace=False):
    if not inplace:
        df = df.copy()
        
    # Fill null-values
    numeric_cols = df.select_dtypes(exclude=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # One-Hot-encode categoric columns
    df = pd.concat([
        pd.get_dummies(df['season'], prefix='season', drop_first=True).astype('int32'),
        df.drop('season', axis=1)
    ], axis=1)
    
    return df
    

def cleanup(df, inplace=True):
    if not inplace:
        df = df.copy()

    df = df.drop(['id'], axis=1)
    return df


# Feature eng. train data
df = feature_eng(df)

# Feature eng. test data
df_test = feature_eng(df_test)


time_columns = ['week', 'month', 'season']

fig, ax = plt.subplots(1, len(time_columns), figsize=(18, 3))

for i, feature in enumerate(time_columns):
    unique = df[feature].unique()
    target0 = df[df[target_column] == 0][feature].value_counts().sort_index()
    target1 = df[df[target_column] == 1][feature].value_counts().sort_index()
    
    target0 = target0.reindex(unique, fill_value=0)
    target1 = target1.reindex(unique, fill_value=0)
    
    ax[i].bar(target0.index, target0.values, label='no', color=colors[0])
    ax[i].bar(target1.index, target1.values, bottom=target0.values, label='yes', color=colors[1])

    ax[i].set_title(f'{target_column} by {feature}')
    legend = ax[i].legend(title=target_column, fontsize=10)
    plt.setp(legend.get_title(), fontsize=10)


# Preprocess train data
df = preprocessing(df)
df = cleanup(df)

# Preprocess test data
df_test = preprocessing(df_test)
df_test = cleanup(df_test)


X = df.drop('rainfall', axis=1)
y = np.array(df['rainfall'])

# Scale data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(df_test)


from sklearn.feature_selection import mutual_info_regression

ir = mutual_info_regression(X, y)
mutual_info = pd.DataFrame({'Column': X.columns, 'Score': ir})
mutual_info = mutual_info.sort_values(by='Score', ascending=False)



plt.figure(figsize=(12, 5))
plt.bar(mutual_info['Column'], mutual_info['Score'], color=colors[1])
plt.title('Mutual Info')
plt.xlabel('Column'); plt.ylabel('Score')
plt.xticks(rotation=75)
plt.show()


random_forest_params = {
    'criterion': 'entropy',
    'max_depth': 10,
    'n_estimators': 100,
    'min_samples_split': 2
}

logistic_reg_params = {
    'C': 0.1,
    'solver': 'liblinear'
}

extra_trees_params = {
    'max_depth': 50,
    'max_leaf_nodes': 25,
    'min_samples_split': 10,
    'n_estimators': 250,
    'criterion': 'gini'
}

xgboost_params = {
    'booster': 'gbtree',
    'max_depth': 3,
    'gamma': 0,
    'alpha': 1,
    'learning_rate': 0.05
}

lgbm_params = {
    'num_leaves': 15,
   'n_estimators': 100,
   'max_depth': 100,
   'learning_rate': 0.1
}


estimators = [
    ('rf', RandomForestClassifier(**random_forest_params)),
    ('lr', LogisticRegression(**logistic_reg_params)),
    ('et', ExtraTreesClassifier(**extra_trees_params)),
    ('xgb', XGBClassifier(**xgboost_params)),
    ('lgbm', LGBMClassifier(**lgbm_params, verbose=-1))
]


def evaluate_models(models, X, y, cv=5):
    scores, roc_auc = [], []
    for (name, model) in models:
        cv_scores = cross_validate(model, 
                                   X, y, 
                                   cv=cv, 
                                   scoring=['accuracy', 'roc_auc'])
        scores.append(cv_scores['test_accuracy'])
        roc_auc.append(cv_scores['test_roc_auc'])
        
    return scores, roc_auc


scores, roc_auc = evaluate_models(estimators, X_scaled, y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

x = ['cv1', 'cv2', 'cv3', 'cv4', 'cv5']
for i in range(len(scores)):
    ax[0].plot(x, scores[i], label=estimators[i][0], color=colors[i])
    ax[1].plot(x, roc_auc[i], label=estimators[i][0], color=colors[i])

ax[0].set_title('Scores')
ax[0].legend(loc='upper right')

ax[1].set_title('ROC AUC Scores')
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()


# Build Stacking Classifier
final_estimator = LogisticRegression()

stack_clf = StackingClassifier(estimators=estimators, 
                               final_estimator=final_estimator,
                               n_jobs=-1)

stack_clf


def accuracy(y_pred, y, threshold=0.5):
    preds = (y_pred >= threshold).astype(np.uint8)
    acc = np.mean(preds == y)
    return acc
    

def fit_estimator(model, X, y, cv=5):
    skf = StratifiedKFold(cv)
    
    oof_scores = np.zeros(cv)
    oof_roc_scores = np.zeros(cv)
    oof_preds = np.zeros(len(y))
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]

        oof_score = accuracy(preds, y_val)
        oof_roc_auc = roc_auc_score(y_val, preds)

        oof_scores[i] = oof_score
        oof_roc_scores[i] = oof_roc_auc
        oof_preds[val_idx] = preds

        print(f'[{i+1}/{cv}] | AUC: {oof_roc_auc:.4f} | Score: {oof_score}\n---')  
    
    return oof_scores, oof_roc_scores, oof_preds 


_, _, oof_preds = fit_estimator(stack_clf, X_scaled, y, cv=5)


preds = stack_clf.predict_proba(X_scaled)[:, 1]
auc = roc_auc_score(y, preds)
fpr, tpr, _ = roc_curve(y, preds)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'StackingClassifier (AUC={auc:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


test_preds = stack_clf.predict_proba(X_test_scaled)[:, 1]

sample_submission['rainfall'] = test_preds
sample_submission.to_csv('submission.csv', index=False)

