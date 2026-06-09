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


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


print("Shape of Train Data", train_df.shape)


display(train_df.head(10))


train_df.info()


train_df.nunique()


train_df.describe().transpose()


import seaborn as sns
import matplotlib.pyplot as plt

# Total missing values per column
missing = train_df.isnull().sum()
missing_percent = (missing / len(train_df)) * 100
print(missing_percent)

# Heatmap for visual check
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()


numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                  'Friends_circle_size', 'Post_frequency']

for col in numerical_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)


categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in categorical_cols:
    mode_val = train_df[col].mode()[0]
    train_df[col].fillna(mode_val, inplace=True)


train_df.isnull().sum()


plt.figure(figsize=(12, 6))
stage_fear_counts = train_df['Stage_fear'].value_counts()

# Plot barplot
ax = sns.barplot(x=stage_fear_counts.index, y=stage_fear_counts.values, palette="viridis")

plt.title("Distribution of Stage fear", fontsize=15)
plt.xlabel("Stage fear")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(12, 6))
drained_after_socializing_counts = train_df['Drained_after_socializing'].value_counts()

# Plot barplot
ax = sns.barplot(x=drained_after_socializing_counts.index, y=drained_after_socializing_counts.values, palette="viridis")

plt.title("Drained after socializing", fontsize=15)
plt.xlabel("Drained after socializing")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(12, 6))
personality_counts = train_df['Personality'].value_counts()

# Plot barplot
ax = sns.barplot(x=personality_counts.index, y=personality_counts.values, palette="viridis")

plt.title("Distribution of Personality", fontsize=15)
plt.xlabel("Personality")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Stage_fear'], train_df['Personality']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Personality vs Stage fear', fontsize=16)
plt.xlabel('Stage_fear', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Personality', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Drained_after_socializing'], train_df['Personality']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Personality vs Drained_after_socializing', fontsize=16)
plt.xlabel('Drained_after_socializing', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Personality', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


numerical_df = train_df.select_dtypes(include=['int64', 'float64'])


numerical_df.columns



from scipy import stats
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Get all pairs of numerical columns
column_pairs = combinations(['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'], 2)

# Set style
sns.set(style="whitegrid")

# Loop through each pair and plot
for col1, col2 in column_pairs:
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line
    sns.regplot(x=col1, y=col2, data=numerical_df, scatter_kws={'alpha':0.6})
    
    # Calculate statistics
    corr_coef, p_value = stats.pearsonr(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    slope, intercept, _, _, _ = stats.linregress(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    
    # Add statistics to plot
    stats_text = (f"Pearson r = {corr_coef:.2f}\n"
                  f"p-value = {p_value:.4f}\n"
                  f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    
    plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Titles and labels
    plt.title(f'{col1} vs {col2}', fontsize=14)
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Automated interpretation
    abs_r = abs(corr_coef)
    
    # Interpret Pearson r
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak or no"
    
    direction = "positive" if corr_coef > 0 else "negative" if corr_coef < 0 else "no"
    
    # Interpret p-value
    if p_value < 0.001:
        sig_text = "highly statistically significant (p < 0.001)"
    elif p_value < 0.05:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = "not statistically significant (p â‰¥ 0.05)"
    
    # Print interpretation
    print(f"\nInterpretation for {col1} vs {col2}:")
    print(f"- {strength} {direction} linear relationship")
    print(f"- The correlation is {sig_text}\n")
    print("-" * 60)  # Separator line


corr = abs(numerical_df.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,8))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
plt.xticks(rotation = 50)   # Aesthetic purposes
plt.yticks(rotation = 20)   # Aesthetic purposes
plt.show()


from scipy.stats import skew  # For skewness calculation

# Set up subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numerical_df.columns) // n_cols) + 1

# Create a figure with subplots
plt.figure(figsize=(15, 5 * n_rows))  # Adjust size as needed

# Loop through numerical columns and plot KDE + skewness
for i, column in enumerate(numerical_df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.kdeplot(data=numerical_df, x=column, fill=True)
    
    # Calculate skewness
    skewness = skew(numerical_df[column].dropna())  # Handle NaN if needed
    skew_text = f'Skewness: {skewness:.2f}'
    
    # Add skewness as text in the plot
    plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'KDE of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot box plots
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_df.columns, 1):
    plt.subplot(2, 4, i)  # Adjust subplot grid as needed
    sns.boxplot(data=train_df, y=feature, color='skyblue')
    plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import optuna


def preprocess_and_engineer_features(df, is_train=True, scaler=None):
    df = df.copy()

    ids = df['id'] if 'id' in df.columns else None

    if 'Personality' in df.columns:
        target = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
        df.drop('Personality', axis=1, inplace=True)
    else:
        target = None

    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df['Time_spent_Alone_log'] = np.log1p(df['Time_spent_Alone'])
    df['Post_frequency_log'] = np.log1p(df['Post_frequency'])

    df['Social_activity_score'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Friends_circle_size'] +
        df['Post_frequency']
    )
    df['Solitude_ratio'] = df['Time_spent_Alone'] / (df['Going_outside'] + 1e-3)
    df['Friends_x_Posts'] = df['Friends_circle_size'] * df['Post_frequency']
    df['Introvert_likelihood'] = (
        df['Time_spent_Alone'] * 1.5 -
        df['Friends_circle_size'] -
        df['Post_frequency']
    )
    df['Alone_per_activity'] = df['Time_spent_Alone'] / (df['Social_activity_score'] + 1e-3)
    df['Alone_squared'] = df['Time_spent_Alone'] ** 2
    df['Friends_to_Posts'] = df['Friends_circle_size'] / (df['Post_frequency'] + 1e-3)

    df['Alone_bin'] = pd.qcut(df['Time_spent_Alone'], q=3, labels=[0, 1, 2])
    df['Alone_bin'] = df['Alone_bin'].astype(int)

    df.drop(['id'], axis=1, errors='ignore', inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    if is_train:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when is_train=False")
        df_scaled = scaler.transform(df)

    return df_scaled, target, ids, scaler


def xgb_objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    model = XGBClassifier(**param)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()


def train_and_evaluate(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: xgb_objective(trial, X_train, y_train), n_trials=100)

    print("\nBest Optuna Parameters:", study.best_params)

    best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_model.fit(X_train, y_train)
    val_preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nTuned XGBoost Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, val_preds))

    plt.figure(figsize=(12, 6))
    plot_importance(best_model, max_num_features=15)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

    return best_model


def generate_submission(model, test_df, scaler, filename='submission.csv'):
    X_test, _, ids, _ = preprocess_and_engineer_features(test_df, is_train=False, scaler=scaler)
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        'id': ids,
        'Personality': np.where(preds == 0, 'Introvert', 'Extrovert')
    })
    submission.to_csv(filename, index=False)
    print(f"ğŸ“� Submission file saved as {filename}")


# TRAIN
X_train, y_train, _, scaler = preprocess_and_engineer_features(train_df, is_train=True)

# Train model
best_model = train_and_evaluate(X_train, y_train)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


print(test_df.columns)


# TEST
generate_submission(best_model, test_df, scaler)

