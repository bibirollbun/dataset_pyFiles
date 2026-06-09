import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use("seaborn-v0_8-darkgrid")
warnings.filterwarnings("ignore")
plt.rc("font",family="SimHei",size="15")  
# import csv
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
datasert_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")




datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(datasert_df, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols)

train_df.info()


train_df.info()
train_df.describe()


train_df.head()


numeric_df = train_df.select_dtypes(include='number').drop(columns=['id'])
numeric_df.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()


train_df.info()


train_ID = train_df['id']
test_ID = test_df['id']

#Now drop the  'id' colum since it's unnecessary for  the prediction process.
train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

ntrain = train_df.shape[0] 
ntest = test_df.shape[0] 
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


all_data.info()


# 1. Create a new grouping column based on the quartiles of Social_event_attendance
all_data['social_attend_bin'] = pd.qcut(
    all_data['Social_event_attendance'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3.Perform group-wise filling of missing values
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='social_attend_bin', target_col='Time_spent_Alone'
)


all_data.drop(columns=['social_attend_bin'], inplace=True)

all_data.info()


# 1. Create a new grouping column based on the quartiles of Social_event_attendance
all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Time_spent_Alone'
)


all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


# 1. Create a new grouping column based on the quartiles of Social_event_attendance
all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


# 1. Create a new grouping column based on the quartiles of Social_event_attendance
all_data['Friends_circle_bin'] = pd.qcut(
    all_data['Friends_circle_size'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Friends_circle_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Friends_circle_bin'], inplace=True)

all_data.info()


# 1. Create a new grouping column based on the quartiles of Social_event_attendance
all_data['Post_frequency_bin'] = pd.qcut(
    all_data['Post_frequency'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Post_frequency_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Post_frequency_bin'], inplace=True)

all_data.info()


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """
        Fill missing values in `target_col` by grouping based on the quantiles of `group_source_col`, and using the median of each group to impute missing values.
        
        **Parameters:**
        - `df` (`pd.DataFrame`): The original dataset
        - `group_source_col` (`str`): The numeric column used for grouping
        - `target_col` (`str`): The target column with missing values to be filled
        - `quantiles` (`list`): Quantile breakpoints for grouping (default is quartiles)
        - `labels` (`list`): Labels for each group (default is auto-generated as Q1/Q2/...)
        
        **Returns:**
        - `pd.DataFrame`: The DataFrame with missing values filled (modifies in place)

    """
    # Automatically Generate Group Labels
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    # Step 1: Create Grouping Column
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    # Step 2: Fill Missing Values Within Groups Using the Median
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    # Step 3: Delete Temporary Columns
    df.drop(columns=[temp_bin_col], inplace=True)

    return df

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Going_outside'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Post_frequency'
)
all_data.info()


all_data.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)
all_data.info()


numeric_all_data = all_data.select_dtypes(include='number')
numeric_all_data.plot(kind='box', title='Boxplot', figsize=(12, 5))


plt.figure(figsize=(8, 5))
sns.histplot(train_df['Time_spent_Alone'], bins=30, kde=True)
plt.title("Time_spent_Alone")
plt.xlabel("Time_spent_Alone")
plt.show()


all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])
all_data.info()


all_data.head()


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')



X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
X=X_train
y=y_train


class_0 = y_train.sum()
class_1 = len(y_train) - class_0
scale_pos_weight = class_1 / class_0


xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42
)


# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft'
)


# Train with validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
ensemble.fit(X_train, y_train)


# Optimize threshold
val_probs = ensemble.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)



test_probs = ensemble.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': test_preds
})
print(submission.head())
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)
print("Submitted successfully with XGBoost")


# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# X_train = all_data[:ntrain]
# X_test = all_data[ntrain:]

# xgb_params = {
#     'max_depth': [4],
#     'learning_rate': [0.01],
#     'n_estimators': [1000],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8]
# }


# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# xgb_cv = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=1)
# xgb_cv.fit(X_train, y_train)


# xgb_pred = xgb_cv.predict(X_test)


# kaggle = pd.DataFrame({'id': test_ID, 'Personality': xgb_pred})
# kaggle['Personality'] = kaggle['Personality'].map({1: 'Extrovert', 0: 'Introvert'})


# kaggle.to_csv('submission.csv', index=False)
# print("Submitted successfully with XGBoost")


