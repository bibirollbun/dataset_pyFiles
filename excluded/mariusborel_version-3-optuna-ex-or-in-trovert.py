import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer, roc_curve
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Accent_r", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222',     # Dark gray plot background
    'figure.facecolor': '#222222',   # Dark gray around the figure
    'text.color': 'white',           # White text everywhere
    'axes.labelcolor': 'gold',      # White axis labels
    'xtick.color': '#82e0aa',          # White x-axis tick labels
    'ytick.color': '#82e0aa',          # White y-axis tick labels
    'grid.color': '#444444',         # Slightly lighter grid
    'axes.edgecolor': 'white'        # White border of the plot
})


include_external = True


# Competition datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# External datasets
ext_01 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')
ext_02 = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')

# Combine competition and external train data
train_ext = pd.concat([train, ext_01, ext_02], ignore_index=True)

# Define the target
target = 'Personality'

train.head()


# The shapes

print('shape of train: {}\nshape of external: {}\nshape of test: {}'.format(train.shape, ext_01.shape, test.shape))


# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1 = df_1[num_features].copy()
    adv_df_2 = df_2[num_features].copy()


    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1 = adv_df_1.assign(adv=1)
    adv_df_2 = adv_df_2.assign(adv=0)


    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=64)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv


    cv = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                        random_state = 64)
    model = LGBMClassifier(verbose=-1)

    # Get the cross validation scores
    adv_scores = []
    for i, _ in enumerate(cv.split(X_combined, y_combined)):
        X_train, X_valid, y_train, y_valid = train_test_split(X_combined, 
                                                              y_combined, 
                                                              test_size=0.3)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_valid)[:,1]
        score = roc_auc_score(y_valid, y_pred)
        adv_scores.append(score)

    #Plot the roc_curve
    mean_auc = np.mean(adv_scores)
    fpr, tpr, _ = roc_curve(y_valid, y_pred)
    plt.plot(fpr, tpr, label = 'roc_curve (AUC = %0.4f)' % mean_auc)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'gray', label = 'Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'roc_curve {name_1} vs {name_2}', weight='bold', fontsize=12)
    plt.legend()


cat_cols = test.select_dtypes(exclude='number').columns.tolist()

num_cols = test.select_dtypes(include='number').columns.tolist()


num_features = list(num_cols)
plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
adversarial_validation(train, test, 'train', 'test')
plt.subplot(1,4,2)
adversarial_validation(test, ext_01, 'train', 'ext_01')
plt.subplot(1,4,3)
adversarial_validation(train_ext, test, 'train_ext', 'test')
plt.subplot(1,4,4)
adversarial_validation(ext_01, ext_02, 'ext_01', 'ext_02')
plt.tight_layout()


if include_external:
    train = train_ext.copy()
else:
    train = train.copy()


train.info()


train.shape


# Count the missing values in the datasets
null_count = pd.DataFrame({'NaN in train': train.isna().sum(), 
                           'NaN in test': test.isna().sum()}).drop(index=[target]).astype('int') 

null_count['% NaN in train'] = train.isna().mean()*100 
null_count['% NaN in test'] =  test.isna().mean()*100

# pickup only the features with missing values
null_count.sort_values(by='NaN in train', ascending=False).head(11).style.background_gradient(cmap='Reds')


# Function that will be used to fill NaN cat_feat 
def cat_nan_filler(df):
    df['Stage_fear'] = df['Stage_fear'].fillna(df['Drained_after_socializing'])
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Stage_fear'])
    return df


train_ = cat_nan_filler(train)
test_ = cat_nan_filler(test)


print(f'\033[93m{train_.isna().sum()}\033[0m')

print(f'\n\033[92m{test_.isna().sum()}\033[0m')


# Define the function to cross count categories
def cat_cross_counting(df, feat_1, feat_2, a, b):
    plt.figure(figsize=(a, b))
    ctab_value = pd.crosstab(df[feat_1], df[feat_2])
    mask = ctab_value==0
    sns.heatmap(ctab_value, annot=True, fmt='d', cbar=False, mask=mask)
    plt.title(f'Count: {feat_1} and {feat_2}', fontsize=10, color='#82e0aa')
    # plt.show()


cat_cross_counting(train, 'Stage_fear', 'Drained_after_socializing', 4, 4)


cat_cross_counting(train, target, 'Drained_after_socializing', 4, 4)


cat_cross_counting(train, target, 'Stage_fear', 4, 4)


for num_feat in num_cols:
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(2, 3, height_ratios=[1, 6], width_ratios=[2, 2, 2])

    ax0 = fig.add_subplot(gs[0, :2])
    # Add custom text in the center
    ax0.text(0.5, 0.5, f'Distribution of {num_feat} by {target}', fontsize=12, 
             ha='center', va='center', color='#82e0aa')
    ax0.axis('off')
    
    # First plot: the global view
    ax1 = fig.add_subplot(gs[1, 0])
    ax1 = sns.boxplot(train_, x=num_feat, y=target)
    
    # Second plot: by brand
    ax2 = fig.add_subplot(gs[1, 1])
    ax2 = sns.kdeplot(train_, x=num_feat, hue=target, fill=True)

    ax0 = fig.add_subplot(gs[0, 2:])
    # Add custom text in the center
    ax0.text(0.5, 0.5, f'by Cat_feature', fontsize=12, 
             ha='center', va='center', color='red')
    ax0.axis('off')

    # Second plot: by brand
    ax4 = fig.add_subplot(gs[1, -1:])
    ax4 = sns.kdeplot(train_, x=num_feat, hue='Stage_fear', 
                      fill=True, palette='YlOrRd')

        
    plt.tight_layout()
    plt.show()


sns.pairplot(train_, hue=target, height=2, dropna=True)
plt.show()


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 3, height_ratios=[1, 1])

ax0 = fig.add_subplot(gs[0, 0])
ax0 = train_['Stage_fear'].value_counts().plot.pie(autopct='%0.2f%%', cmap='YlOrRd', radius=1.25)
ax0.set_ylabel('')
ax0.set_title('Stage_fear', fontsize=12, color='gold')

ax1 = fig.add_subplot(gs[0, 1])
ax1 = train_['Drained_after_socializing'].value_counts().plot.pie(autopct='%0.2f%%', cmap='YlOrRd', radius=1.25)
ax1.set_ylabel('')
ax1.set_title('Drained_after_socializing', fontsize=12, color='gold')

ax2 = fig.add_subplot(gs[0, 2])
ax2 = train_[target].value_counts().plot.pie(autopct='%0.2f%%', radius=1.25)
ax2.set_ylabel('')
ax2.set_title(f'{target}', fontsize=12, color='gold')

ax3 = fig.add_subplot(gs[1, 0])
ax3 = train_['Stage_fear'].value_counts().plot.bar(cmap='YlOrRd')

ax4 = fig.add_subplot(gs[1, 1])
ax4 = train_['Drained_after_socializing'].value_counts().plot.bar(cmap='YlOrRd')

ax5 = fig.add_subplot(gs[1, 2])
ax5 = train_[target].value_counts().plot.bar()

plt.tight_layout()


# binarize the cat_features
def binarize_cat_feat(df):
    df.copy()
    df['Stage_fear'] = df['Stage_fear']=='Yes'
    df['Drained_after_socializing'] = df['Drained_after_socializing']=='Yes'

    return df


corr = train_.corr(numeric_only=True)

sns.heatmap(corr, annot=True, fmt='.2f', cbar=False);


train_data = binarize_cat_feat(train_)
train_target = train_data.pop(target)

test_data = binarize_cat_feat(test_)


train_.corrwith(train_target=='Introvert').plot.barh(title='Tendency to be introvert')


X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.2, random_state=12)

[d.shape for d in [X_train, X_valid, y_train, y_valid]]


optuna_cv = KFold(n_splits=4, shuffle=True, random_state=33)

# Define the objective function
def objective(trial):
    param_grid = {
#         'metric': 'l2',
        'verbosity': -1,
#         'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'num_leaves': trial.suggest_int('num_leaves', 5, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
    }
    
    # Define the model by unpacking the chosen parameters
    model = LGBMClassifier(**param_grid, verbose=0)
    # Get and return the score
#     model.fit(X_train, y_train)
    scores = cross_val_score(model, train_data, train_target, cv=optuna_cv, n_jobs=-1, 
                        scoring = 'roc_auc')
    
    return scores.mean()


def run_optuna(runs=1):
    if runs > 1:
        # Define the sampler
        sampler = TPESampler(seed=815)
        # Create and optimize the optuna study
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name='lgbm_loan_approval_001')
        study.optimize(lambda trial: objective(trial), n_trials=runs, show_progress_bar=True)

        best_study_params = study.best_params
    else:     
        best_study_params = {'n_estimators': 970, 
                             'learning_rate': 0.017347611444144168, 
                             'lambda_l1': 0.006156465597283681, 
                             'lambda_l2': 0.07832781174510908, 
                             'max_depth': 4, 
                             'num_leaves': 31,
                             'feature_fraction': 0.473662690442311, 
                             'bagging_fraction': 0.624883821536666, 
                             'bagging_freq': 1, 
                             'min_child_samples': 6}
    
    print(f'\nThe best lgbm hyperparameters: \n{best_study_params}')
    return best_study_params


%%time
best_study_params = run_optuna(runs=100) 


clf = LGBMClassifier(**best_study_params, verbose=-1)

clf.fit(X_train, y_train)

clf.score(X_valid, y_valid)


my_spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

for f, (tr_ind, va_ind) in enumerate(my_spliter.split(train_data, train_target), start=1):
    X_tr, X_va = train_data.iloc[tr_ind], train_data.iloc[va_ind]
    y_tr, y_va = train_target.iloc[tr_ind], train_target.iloc[va_ind]

    model = LGBMClassifier(**best_study_params, verbose=-1)

    model.fit(X_tr, y_tr)

    score = model.score(X_va, y_va)
    c=90+f # choice of color
    print('\033[{}mFold_{} => accuracy: {}\n\033[0m'.format(c, f, score))


preds = clf.predict(X_valid)

class_report = classification_report(y_valid, preds)
conf_matrix = confusion_matrix(y_valid, preds)

print(f"\033[95m{class_report}\033[0m")
target_labels = ['Introvert', 'Extrovert']
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, square=True, 
            xticklabels=target_labels, yticklabels=target_labels)
plt.show()


conf_matrix_norm = confusion_matrix(y_valid, preds, normalize='pred')

sns.heatmap(conf_matrix_norm, annot=True, cbar=False, square=True, 
            xticklabels=target_labels, yticklabels=target_labels)
plt.show()


test_preds = clf.predict(test_data)


subm[target] = test_preds

subm.head()


fig = plt.figure(figsize=(6, 5))
gs = GridSpec(2, 2, height_ratios=[2, 2], width_ratios=[2, 2])

ax0 = fig.add_subplot(gs[:, :])
ax1 = subm[target].value_counts().plot.bar(color=['#d35400', '#a2006d'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center')
ax1 = fig.add_subplot(gs[:-1, -1:])
ax1 = subm[target].value_counts().plot.pie(autopct='%.2f%%', radius=1.1)
ax1.set_ylabel('')
plt.tight_layout()


subm.to_csv('submission.csv', index=False)

print('The file is ready for submission!')

