import warnings
warnings.filterwarnings('ignore') 


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", palette="Set2")
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
df_datasert = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_datasert = (
    df_datasert
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']


def make_mi_score(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores


X = df_train.copy()
X = X.drop(columns=["id"])
X = X.dropna()
y = X.pop("Personality")

sns.set_theme(style='whitegrid')

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

discrete_features = X.dtypes == int

mi_scores = make_mi_score(X, y, discrete_features=discrete_features)

mi_scores = mi_scores.reset_index()

mi_scores.columns = ["feature", "mi_score"]

plt.figure(figsize=(14, 6))
plt.title("MI Scores for classif")
sns.barplot(data=mi_scores, y="mi_score", x="feature")
plt.tight_layout()
plt.show()


df_test = df_test.merge(df_datasert, how='left', on=merge_cols)
df_train = df_train.merge(df_datasert, how='left', on=merge_cols)

df_train.info()
df_train.describe()


train_ID = df_train['id']
test_ID = df_test['id']

df_train.drop("id", axis = 1, inplace=True)
df_test.drop("id", axis = 1, inplace=True)

ntrain = df_train.shape[0] 
ntest = df_test.shape[0] 
y_train = df_train['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)

all_data.info()


all_data['social_attend_bin'] = pd.qcut(
    all_data['Social_event_attendance'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='social_attend_bin', target_col='Time_spent_Alone'
)


all_data.drop(columns=['social_attend_bin'], inplace=True)

all_data.info()


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Time_spent_Alone'
)

all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


all_data['Friends_circle_bin'] = pd.qcut(
    all_data['Friends_circle_size'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Friends_circle_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Friends_circle_bin'], inplace=True)

all_data.info()


all_data['Post_frequency_bin'] = pd.qcut(
    all_data['Post_frequency'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Post_frequency_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Post_frequency_bin'], inplace=True)

all_data.info()


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

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
    'Stage_fear': 'unknown',
    'Drained_after_socializing': 'unknown'
}, inplace=True)
all_data.info()


all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])
all_data.info()


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


ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft'
)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
ensemble.fit(X_train, y_train)


val_probs = ensemble.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)
    acc = (preds == y_val).mean()
    
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best validation accuracy: {best_acc:.4f}")


test_probs = ensemble.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id': test_ID,
    'Personality': test_preds
})

submission.info()

print(submission.head())
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)

