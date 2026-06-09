import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns

sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 6)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df.head()


train_df.shape


test_df.shape


train_df.info()


print("Null values in dataset:")
print(train_df.isnull().sum())


for column in train_df.columns:
    unique_values = train_df[column].unique()
    print(f"Unique values for {column}: {unique_values}")


train_df['Time_spent_Alone'].fillna(train_df['Time_spent_Alone'].median(), inplace=True)
train_df['Social_event_attendance'].fillna(train_df['Social_event_attendance'].mode()[0], inplace=True)
train_df['Going_outside'].fillna(train_df['Going_outside'].mode()[0], inplace=True)
train_df['Friends_circle_size'].fillna(train_df['Friends_circle_size'].median(), inplace=True)
train_df['Post_frequency'].fillna(train_df['Post_frequency'].mode()[0], inplace=True)

train_df['Stage_fear'].fillna(train_df['Stage_fear'].mode()[0], inplace=True)
train_df['Drained_after_socializing'].fillna(train_df['Drained_after_socializing'].mode()[0], inplace=True)


print("Null values in dataset:")
print(train_df.isnull().sum())


print("Null values in dataset:")
print(test_df.isnull().sum())


test_df['Time_spent_Alone'].fillna(test_df['Time_spent_Alone'].median(), inplace=True)
test_df['Social_event_attendance'].fillna(test_df['Social_event_attendance'].mode()[0], inplace=True)
test_df['Going_outside'].fillna(test_df['Going_outside'].mode()[0], inplace=True)
test_df['Friends_circle_size'].fillna(test_df['Friends_circle_size'].median(), inplace=True)
test_df['Post_frequency'].fillna(test_df['Post_frequency'].mode()[0], inplace=True)

test_df['Stage_fear'].fillna(test_df['Stage_fear'].mode()[0], inplace=True)
test_df['Drained_after_socializing'].fillna(test_df['Drained_after_socializing'].mode()[0], inplace=True)


print("Null values in dataset:")
print(test_df.isnull().sum())


print("\nDuplicate entries in dataset:")
print(train_df.duplicated().sum())
duplicate_rows = train_df[train_df.duplicated()]


train_df = train_df.drop_duplicates()


train_df.shape


print("\nDuplicate entries in dataset:")
print(test_df.duplicated().sum())
duplicate_rows = test_df[test_df.duplicated()]


# Map 'Yes'/'No' to binary for relevant features
train_df['Drained_after_socializing_bin'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train_df['Stage_fear_bin'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})

#Social Activeness Score
train_df['Social_Activeness_Score'] = (
    train_df['Social_event_attendance'] +
    train_df['Going_outside'] +
    train_df['Friends_circle_size'] +
    train_df['Post_frequency']
)

#Introversion Index
train_df['Introversion_Index'] = (
    train_df['Time_spent_Alone'] * train_df['Drained_after_socializing_bin']
)

#Social Fatigue Risk
train_df['Social_Fatigue_Risk'] = (
    (train_df['Social_event_attendance'] + train_df['Going_outside']) *
    train_df['Drained_after_socializing_bin']
)

#Alone vs Social Ratio
train_df['Alone_to_Social_Ratio'] = train_df['Time_spent_Alone'] / (
    train_df['Social_event_attendance'] + 1  # avoid division by zero
)

#Contradiction Flag: social but has stage fear
train_df['Contradiction_Flag'] = (
    (train_df['Social_event_attendance'] > 5) &
    (train_df['Stage_fear_bin'] == 1)
).astype(int)



train_df.info()


# Map 'Yes'/'No' to binary for relevant features
test_df['Drained_after_socializing_bin'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
test_df['Stage_fear_bin'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})

# Social Activeness Score
test_df['Social_Activeness_Score'] = (
    test_df['Social_event_attendance'] +
    test_df['Going_outside'] +
    test_df['Friends_circle_size'] +
    test_df['Post_frequency']
)

# Introversion Index
test_df['Introversion_Index'] = (
    test_df['Time_spent_Alone'] * test_df['Drained_after_socializing_bin']
)

# Social Fatigue Risk
test_df['Social_Fatigue_Risk'] = (
    (test_df['Social_event_attendance'] + test_df['Going_outside']) *
    test_df['Drained_after_socializing_bin']
)

# Alone vs Social Ratio
test_df['Alone_to_Social_Ratio'] = test_df['Time_spent_Alone'] / (
    test_df['Social_event_attendance'] + 1  # avoid division by zero
)

# Contradiction Flag: social but has stage fear
test_df['Contradiction_Flag'] = (
    (test_df['Social_event_attendance'] > 5) &
    (test_df['Stage_fear_bin'] == 1)
).astype(int)


test_df.info()


categorical_cols = ['Stage_fear', 'Drained_after_socializing']
n_cols = len(categorical_cols)

fig, axes = plt.subplots(1, n_cols, figsize=(15, 5))

for i, col in enumerate(categorical_cols):
    ax = sns.countplot(
        x=col, 
        data=test_df, 
        order=test_df[col].value_counts().index,
        ax=axes[i]
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()



binary_cols = ['Drained_after_socializing_bin', 'Stage_fear_bin', 'Contradiction_Flag']
n_cols = len(binary_cols)

plt.style.use('seaborn')
fig, axes = plt.subplots(1, n_cols, figsize=(18, 6))

for i, col in enumerate(binary_cols):
    ax = sns.countplot(
        x=col, 
        data=test_df,
        ax=axes[i],
        palette='pastel'
    )
    
    axes[i].set_title(f'Distribution: {col}', fontsize=14, pad=15)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].set_xticklabels(['No', 'Yes'])
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(x=p.get_x() + p.get_width()/2, 
                y=height + 0.01*test_df[col].count(),
                s=f'{height:.0f}',
                ha='center',
                va='bottom',
                fontsize=11)

fig.suptitle('Binary Feature Distributions', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()



numerical_cols = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency', 'Social_Activeness_Score',
    'Introversion_Index', 'Social_Fatigue_Risk', 'Alone_to_Social_Ratio'
]
n_cols = 2
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

for i, col in enumerate(numerical_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.histplot(train_df[col], kde=True, bins=30, ax=ax)
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Continuous Features", fontsize=16)
plt.tight_layout()
plt.show()



numerical_cols = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency', 'Social_Activeness_Score',
    'Introversion_Index', 'Social_Fatigue_Risk', 'Alone_to_Social_Ratio'
]

n_cols = 3
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

plt.style.use('seaborn')
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
fig.suptitle('Numerical Features by Personality Type', y=1.02, fontsize=16, fontweight='bold')

for i, col in enumerate(numerical_cols):
    row = i // n_cols
    col_num = i % n_cols
    ax = axes[row, col_num] if n_rows > 1 else axes[col_num]
    
    sns.boxplot(
        x='Personality',
        y=col,
        data=train_df,
        ax=ax,
        palette='Set2',
        showmeans=True,
        meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}
    )
    
    ax.set_title(col, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Value' if col_num == 0 else '')
    ax.tick_params(axis='x', rotation=45)

# Remove empty subplots if any
for i in range(len(numerical_cols), n_rows*n_cols):
    row = i // n_cols
    col_num = i % n_cols
    fig.delaxes(axes[row, col_num] if n_rows > 1 else axes[col_num])

plt.tight_layout()
plt.show()



cat_combinations = [
    ('Stage_fear', 'Drained_after_socializing'),
    ('Personality', 'Stage_fear')
]

n_cols = 2  # 2 plots per row
n_rows = (len(cat_combinations) + n_cols - 1) // n_cols

plt.style.use('seaborn')
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
fig.suptitle('Categorical Feature Relationships', y=1.02, fontsize=16, fontweight='bold')

if n_rows == 1:
    axes = axes.reshape(1, -1)  # Ensure axes is 2D array

for i, (x, y) in enumerate(cat_combinations):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    ctab = pd.crosstab(train_df[x], train_df[y])
    sns.heatmap(
        ctab, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar=False,
        linewidths=0.5,
        annot_kws={'size': 12}
    )
    
    ax.set_title(f'{x} vs {y}', pad=12, fontsize=14)
    ax.set_xlabel(y, labelpad=10)
    ax.set_ylabel(x, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Remove empty subplots if any
for i in range(len(cat_combinations), n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()



categorical_features = [
    'Stage_fear', 'Drained_after_socializing', 
    'Drained_after_socializing_bin', 'Stage_fear_bin', 
    'Contradiction_Flag'
]

n_cols = 2  # 2 plots per row
n_rows = (len(categorical_features) + n_cols - 1) // n_cols

plt.style.use('seaborn')
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
fig.suptitle('Categorical Features Distribution by Personality', y=1.02, fontsize=16, fontweight='bold')

if n_rows == 1:
    axes = axes.reshape(1, -1)  # Ensure axes is 2D array

for i, col in enumerate(categorical_features):
    row = i // n_cols
    col_num = i % n_cols
    ax = axes[row, col_num]
    
    sns.countplot(
        x=col, 
        hue='Personality', 
        data=train_df, 
        palette='Set2', 
        ax=ax
    )
    
    ax.set_title(f'{col} Distribution', fontsize=14)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.legend(title='Personality', loc='upper right')

# Remove empty subplots if any
for i in range(len(categorical_features), n_rows * n_cols):
    row = i // n_cols
    col_num = i % n_cols
    fig.delaxes(axes[row, col_num])

plt.tight_layout()
plt.show()



selected_features = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Friends_circle_size', 'Introversion_Index', 'Personality']

plt.style.use('seaborn')
g = sns.pairplot(
    train_df[selected_features], 
    hue='Personality',
    palette='Set2',
    diag_kind='kde',
    corner=False,
    plot_kws={'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 0.3},
    height=3,
    aspect=1.1
)

g.fig.suptitle(
    'Pairwise Relationships by Personality Type', 
    y=1.05, 
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout(pad=1)

# Adjust legend position and appearance
handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',
    ncol=len(labels),
    bbox_to_anchor=(0.5, 1.02),
    frameon=True,
    title='Personality Types',
    title_fontsize='13'
)

plt.subplots_adjust(top=0.88)  # Adjust spacing for legend
plt.show()



df = train_df.copy()

binary_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0}) if col != 'Personality' else df[col].map({'Introvert': 1, 'Extrovert': 0})



df.head()


target = 'Personality'
drop_cols = [target]
X = df.drop(columns=drop_cols)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# lgbm_params = {
#     'n_estimators': [100, 500, 1000],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [5, 10, 15],
#     'num_leaves': [31, 63, 127, 255],
#     'min_child_samples': [10, 20, 30, 50],
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'reg_alpha': [0, 0.1, 0.5, 1.0],
#     'reg_lambda': [0, 0.1, 0.5, 1.0],
#     'random_state': [42],
#     'verbose': [-1]
# }

lgbm_params = {
    'n_estimators': [100]
}




skf = StratifiedKFold(n_splits=3)

lgbm = GridSearchCV(
    estimator=LGBMClassifier(random_state=42, verbosity=-1),
    param_grid=lgbm_params,
    cv=skf,
    n_jobs=3,
    verbose=-1
)

lgbm_model = lgbm.fit(X_train, y_train)

train_pred = lgbm_model.predict(X_train)
test_pred = lgbm_model.predict(X_test)

def print_scores(y_true, y_pred, dataset='Test'):
    print(f"\nðŸ“Š {dataset} Set Scores:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))

print_scores(y_train, train_pred, "Training")
print_scores(y_test, test_pred, "Testing")

