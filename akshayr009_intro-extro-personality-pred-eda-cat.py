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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

import warnings
warnings.filterwarnings('ignore')



#load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train.head(5)


def inspect_features(df, name):
    print(f"ğŸ”� {name} Set Feature Overview:")
    df_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum()
    })
    display(df_info.sort_values(by='Missing Values', ascending=False))


inspect_features(train, 'Train')

inspect_features(test_df, 'Test')


features = train.drop('id', axis = 1).columns.tolist()

sb.set_palette('Set2')
plt.figure(figsize=(15, 10))

# Loop through and plot each histogram
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    ax = sb.countplot(x=col, data=train)
    
    plt.xlabel(col)
    plt.ylabel(' ')
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Plot 1: Stage Fear
sb.boxplot(
    x='Stage_fear',
    y='Time_spent_Alone',
    hue='Personality',
    data=train,
    ax=axes[0]
)
axes[0].set_title('Time Alone by Stage Fear and Personality')
axes[0].set_xlabel('Stage Fear')
axes[0].set_ylabel('Time Spent Alone')

# Plot 2: Drained After Socializing
sb.boxplot(
    x='Drained_after_socializing',
    y='Time_spent_Alone',
    hue='Personality',
    data=train,
    ax=axes[1]
)
axes[1].set_title('Time Alone by Drained After Socializing')
axes[1].set_xlabel('Drained After Socializing')
axes[1].set_ylabel('')

# Handle legends (only once)
axes[0].legend(title='Personality', loc='upper right')
axes[1].legend_.remove()  # remove duplicate legend

plt.tight_layout()
plt.show()


# ğŸ“Š COMPARATIVE PLOTS: Friends Circle Size vs Time Spent Alone

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

sb.boxplot(
    x='Friends_circle_size',
    y='Time_spent_Alone',
    data=train,
    ax=axes[0]
)
axes[0].set_title('Time Alone vs Friends Circle Size')
axes[0].set_ylabel('Time Spent Alone')
axes[0].tick_params(axis='x', rotation=45)

sb.regplot(
    x='Friends_circle_size',
    y='Time_spent_Alone',
    data=train,
    lowess=True,
    scatter_kws={'alpha': 0.5},
    ax=axes[1]
)
axes[1].set_title('Trend: Friends Circle Size vs Time Alone')
axes[1].set_ylabel('')


for label, color in zip(['Introvert', 'Extrovert'], ['orange', 'green']):
    subset = train[train['Personality'] == label]
    sb.regplot(
        x='Friends_circle_size',
        y='Time_spent_Alone',
        data=subset,
        lowess=True,
        scatter_kws={'alpha': 0.4},
        label=label,
        ax=axes[2]
    )
axes[2].set_title('Trend by Personality')
axes[2].set_ylabel('')
axes[2].legend(title='Personality')

plt.tight_layout()
plt.show()



# ğŸ“Š TIME SPENT ALONE vs POST FREQUENCY 

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

sb.boxplot(
    x='Post_frequency',
    y='Time_spent_Alone',
    data=train,
    ax=axes[0]
)
axes[0].set_title('Time Spent Alone vs Post Frequency')
axes[0].set_xlabel('Post Frequency')
axes[0].set_ylabel('Time Spent Alone')
axes[0].tick_params(axis='x', rotation=45)


for label, color in zip(['Introvert', 'Extrovert'], ['orange', 'green']):
    subset = train[train['Personality'] == label]
    sb.regplot(
        x='Post_frequency',
        y='Time_spent_Alone',
        data=subset,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        label=label,
        ax=axes[1]
    )

axes[1].set_title('Trend: Post Frequency vs Time Alone by Personality')
axes[1].set_xlabel('Post Frequency')
axes[1].set_ylabel('')
axes[1].legend(title='Personality')

plt.tight_layout()
plt.show()



# ğŸ“Š 2x2 Visualization Grid: Social Behavior Insights

fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=False)

# Regression: Friends Circle Size vs Going Outside
for label, color in zip(['Introvert', 'Extrovert'], ['orange', 'green']):
    subset = train[train['Personality'] == label]
    sb.regplot(
        x='Friends_circle_size',
        y='Going_outside',
        data=subset,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        label=label,
        ax=axes[0, 0]
    )
axes[0, 0].set_title('Friends Circle vs Going Outside')
axes[0, 0].legend(title='Personality')

# Regression: Going Outside vs Social Event Attendance
for label, color in zip(['Introvert', 'Extrovert'], ['orange', 'green']):
    subset = train[train['Personality'] == label]
    sb.regplot(
        x='Going_outside',
        y='Social_event_attendance',
        data=subset,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        label=label,
        ax=axes[0, 1]
    )
axes[0, 1].set_title('Going Outside vs Social Event Attendance')
axes[0, 1].legend(title='Personality')

# Boxplot: Post Frequency vs Going Outside
sb.boxplot(
    x='Post_frequency',
    y='Going_outside',
    data=train,
    ax=axes[1, 0]
)
axes[1, 0].set_title('Post Frequency vs Going Outside')
axes[1, 0].set_xlabel('Post Frequency')
axes[1, 0].set_ylabel('Going Outside Frequency')
axes[1, 0].tick_params(axis='x', rotation=45)

# Boxplot: Post Frequency vs Social Event Attendance
sb.boxplot(
    x='Post_frequency',
    y='Social_event_attendance',
    data=train,
    ax=axes[1, 1]
)
axes[1, 1].set_title('Post Frequency vs Social Event Attendance')
axes[1, 1].set_xlabel('Post Frequency')
axes[1, 1].set_ylabel('Social Event Attendance')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



def process_binary_and_impute(df, binary_map={}, exclude=[]):
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

  
    for col in df.columns:
        if col not in exclude and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    

    return df
binary_columns = {
    "Stage_fear": {"No": 0, "Yes": 1},
    "Drained_after_socializing": {"No": 0, "Yes": 1},
    "Personality": {"Introvert" : 0, 'Extrovert' : 1}
}

exclude_cols = ["id", "Personality"]

train = process_binary_and_impute(train, binary_map=binary_columns, exclude=exclude_cols)
test_df = process_binary_and_impute(test_df, binary_map=binary_columns, exclude=exclude_cols)


def add_interaction_features(df):
    # Basic interaction terms
    df["Alone_x_Drained"] = df["Time_spent_Alone"] * df["Drained_after_socializing"]
    df["Friends_x_Events"] = df["Friends_circle_size"] * df["Social_event_attendance"]
    df["Posts_x_Friends"] = df["Post_frequency"] * df["Friends_circle_size"]
    df["Outside_x_Events"] = df["Going_outside"] * df["Social_event_attendance"]

    df["Alone_per_Friend"] = df["Time_spent_Alone"] / (df["Friends_circle_size"] + 1)
    df["Posts_per_Event"] = df["Post_frequency"] / (df["Social_event_attendance"] + 1)
    df["Friends_per_GoingOut"] = df["Friends_circle_size"] / (df["Going_outside"] + 1)

    df["Alone_plus_Friends"] = df["Time_spent_Alone"] + df["Friends_circle_size"]
    df["Events_plus_Posts"] = df["Social_event_attendance"] + df["Post_frequency"]

    # Psychological composite features
    df['Social_score'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Friends_circle_size']
    )

    df['Introvert_score'] = df['Time_spent_Alone'] - df['Social_score']
    df['Introvert_vs_Posting'] = df['Introvert_score'] - df['Post_frequency']
    df['Social_vs_Alone'] = df['Social_event_attendance'] - df['Time_spent_Alone']
    df['Social_Anxiety_Index'] = df['Stage_fear'] + df['Drained_after_socializing']
    df['Withdrawal_Tendency'] = df['Time_spent_Alone'] + df['Stage_fear'] + df['Drained_after_socializing']

    return df


train = add_interaction_features(train)
test_df = add_interaction_features(test_df)


x = train.drop(['Personality', 'id'], axis=1)
test  = test_df.drop('id', axis =1)
y = train['Personality']



catboost = {'iterations': 1467, 
            'learning_rate': 0.06852669420904771, 
            'depth': 2, 
            'l2_leaf_reg': 31.236169478676036, 
            'border_count': 39,
            'bagging_temperature': 0.6744458762996971, 
            'random_strength': 0.8517786189616939,
            "colsample_bylevel": 0.19459088572914465,
            "scale_pos_weight": 1.1691394390533685,
            "subsample": 0.3192330024411618,
            "min_child_samples": 160,
            'loss_function': 'Logloss', 
            'eval_metric': 'Accuracy', 
            'verbose': 0, 
            'random_state': 42
           }



oof_preds = np.zeros((x.shape[0],))   
test_preds = np.zeros((test.shape[0],))  
proba_oof = np.zeros((x.shape[0], 2))    
proba_test = np.zeros((test.shape[0], 2))


fold_accuracies = []  


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\nTraining fold {fold + 1}...")

    X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**catboost)
    model.fit(X_train, y_train, 
              eval_set=(X_val, y_val),
              early_stopping_rounds=100, 
              verbose=0)

    
    proba_oof[val_idx] = model.predict_proba(X_val)
    proba_test += model.predict_proba(test) / skf.n_splits
    val_preds = np.argmax(proba_oof[val_idx], axis=1)

   
    acc = accuracy_score(y_val, val_preds)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)  

# Print average accuracy
print(f"\nAverage CV Accuracy: {np.mean(fold_accuracies):.4f}")


#submission file for CatBoost predictions
final_preds = np.argmax(proba_test, axis=1)
label_map = {0: "Introvert", 1: "Extrovert"}
final_labels = [label_map[pred] for pred in final_preds]

submission_cat = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_labels
})

submission_cat.to_csv("submission.csv", index=False)




