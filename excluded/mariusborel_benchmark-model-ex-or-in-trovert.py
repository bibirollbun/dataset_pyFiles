import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

include_external_data = True


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


# Define the function to cross count categories
def cat_cross_counting(df, feat_1, feat_2, a, b):
    plt.figure(figsize=(a, b))
    ctab_value = pd.crosstab(df[feat_1], df[feat_2])
    mask = ctab_value==0
    sns.heatmap(ctab_value, annot=True, fmt='d', cbar=False, mask=mask)
    plt.title(f'Count: {feat_1} and {feat_2}', fontsize=10, color='#82e0aa')
    plt.show()


# Competition files
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
subm = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# External file
ext = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')

# Def the target
target = 'Personality'

train.head()


train.info()


if include_external_data:
    train = pd.concat([train, ext], ignore_index=True)
else:
    train = train


train.shape


cat_cols = test.select_dtypes(exclude='number').columns.tolist()

num_cols = test.select_dtypes(include='number').columns.tolist()
num_cols


# Count the missing values in the datasets
null_count = pd.DataFrame({'NaN in train': train.isna().sum(), 
                           'NaN in test': test.isna().sum()}).drop(index=[target]).astype('int') 

null_count['% NaN in train'] = train.isna().mean()*100 
null_count['% NaN in test'] =  test.isna().mean()*100

# pickup only the features with missing values
null_count.sort_values(by='NaN in train', ascending=False).head(11).style.background_gradient(cmap='Reds')


train_no_nan = train.copy().dropna()


# cat_cols similarity with NaN
Similarity_with_NaN = (train['Stage_fear'] == train['Drained_after_socializing']).mean()
# cat_cols similarity without NaN
Similarity_without_NaN = (train_no_nan['Stage_fear'].dropna() == train_no_nan['Drained_after_socializing'].dropna()).mean()

print('When the NaN are dropped the similarity/correlation between the two cat_features increases from {:.2f}% to {:.2f}%.'.format(Similarity_with_NaN*100, Similarity_without_NaN*100))
print('We can therefore fill the NaN in one feature by considering the value in the other feature.')


# Let's find out if there are rows missing values in both cat_features.

count_missing_in_both_cat = (train['Stage_fear'].isna() & train['Drained_after_socializing'].isna()).sum()
percent_missing_in_both_cat = (train['Stage_fear'].isna() & train['Drained_after_socializing'].isna()).mean()*100

print('There are only {} rows where both cat_features are missing values.\nwhich represent about {:.2f}% of the dataset.'.format(count_missing_in_both_cat, percent_missing_in_both_cat))


# Function that will be used to fill NaN cat_feat 
def cat_nan_filler(df):
    df['Stage_fear'] = df['Stage_fear'].fillna(df['Drained_after_socializing'])
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Stage_fear'])
    return df


train_ = cat_nan_filler(train)
test_ = cat_nan_filler(test)


print(f'\033[93m{train_.isna().sum()}\033[0m')

print(f'\n\033[92m{test_.isna().sum()}\033[0m')


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


train_data = binarize_cat_feat(train_)
train_target = train_data.pop(target)

test_data = binarize_cat_feat(test_)


X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.2, random_state=12)

[d.shape for d in [X_train, X_valid, y_train, y_valid]]


best_study_params = {'n_estimators': 1329, 
                     'learning_rate': 0.04699817975460125, 
                     'lambda_l1': 2.7276187215306384, 
                     'lambda_l2': 7.982411262710067, 
                     'max_depth': 31, 
                     'num_leaves': 236, 
                     'feature_fraction': 0.48416757327480875, 
                     'bagging_fraction': 0.9796474839083582, 
                     'bagging_freq': 3, 
                     'min_child_samples': 58}


clf = LGBMClassifier(**best_study_params, verbose=-1)

clf.fit(X_train, y_train)

clf.score(X_valid, y_valid)


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


clf = LGBMClassifier(**best_study_params, verbose=-1)
clf.fit(train_data, train_target) # Fit on the entire train dataset


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

