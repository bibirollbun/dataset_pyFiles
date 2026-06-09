import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')


columns_names =['id', 'name', 'gender', 'age', 'city', 'occupation_status', 'profession', 
                'academic_pressure', 'work_pressure', 'cgpa', 'study_satisfaction', 'job_satisfaction', 
                'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts', 'work_study_hours', 
                'financial_stress', 'family_history_mental_illness', 'depression']


columns_test_names =['id', 'name', 'gender', 'age', 'city', 'occupation_status', 'profession', 
                     'academic_pressure', 'work_pressure', 'cgpa', 'study_satisfaction', 'job_satisfaction', 
                     'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts', 'work_study_hours', 
                     'financial_stress', 'family_history_mental_illness']


train_df = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv',names=columns_names, header=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv',names=columns_test_names, header=0)
submission = pd.read_csv('/kaggle/input/playground-series-s4e11/sample_submission.csv')


train_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


missing_percent = (train_df.isnull().sum() / len(train_df)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

sns.barplot(x=missing_percent.index, y=missing_percent.values, palette='pastel')
plt.xticks(rotation=90)
plt.title('Missin Value Percentages')
plt.show()


train_df.describe()


test_df.head()


test_df.shape


test_df.info()


test_df.isnull().sum()


missing_percent = (test_df.isnull().sum() / len(test_df)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

sns.barplot(x=missing_percent.index, y=missing_percent.values, palette='pastel')
plt.xticks(rotation=90)
plt.title('Missin Value Percentages')
plt.show()


train_df['depression'].value_counts(normalize=True) * 100


sns.countplot(x=train_df['depression'], palette='pastel')
plt.title('Depression risk (0: No, 1: Yes)');


train_df = train_df.drop(['id', 'name'], axis=1, errors='ignore') 
test_df = test_df.drop(['id', 'name'], axis=1, errors='ignore') 


#for students work_pressure and job_satisfaction = 0
#for employees academic_pressure and study_satisfaction = 0
train_df['academic_pressure'] = train_df['academic_pressure'].fillna(0)
train_df['work_pressure'] = train_df['work_pressure'].fillna(0)
train_df['cgpa'] = train_df['cgpa'].fillna(0)
train_df['study_satisfaction'] = train_df['study_satisfaction'].fillna(0)
train_df['job_satisfaction'] = train_df['job_satisfaction'].fillna(0)

test_df['academic_pressure'] = test_df['academic_pressure'].fillna(0)
test_df['work_pressure'] = test_df['work_pressure'].fillna(0)
test_df['cgpa'] = test_df['cgpa'].fillna(0)
test_df['study_satisfaction'] = test_df['study_satisfaction'].fillna(0)
test_df['job_satisfaction'] = test_df['job_satisfaction'].fillna(0)

# if profession is NaN = unemployed or student
train_df['profession'] = train_df['profession'].fillna('Other')
test_df['profession'] = test_df['profession'].fillna('Other')

train_df['dietary_habits'] = train_df['dietary_habits'].fillna(train_df['dietary_habits'].mode()[0])
train_df['degree'] = train_df['degree'].fillna(train_df['degree'].mode()[0])
train_df['financial_stress'] = train_df['financial_stress'].fillna(train_df['financial_stress'].mode()[0])

test_df['dietary_habits'] = test_df['dietary_habits'].fillna(test_df['dietary_habits'].mode()[0])
test_df['degree'] = test_df['degree'].fillna(test_df['degree'].mode()[0])
test_df['financial_stress'] = test_df['financial_stress'].fillna(test_df['financial_stress'].mode()[0])


train_df.isnull().sum()


test_df.isnull().sum()


train_df['sleep_duration'].unique()


sleep_train= {'More than 8 hours':9, 'Less than 5 hours':4, '5-6 hours':5.5, '7-8 hours':7.5,'1-2 hours':1.5, 
              '6-8 hours':7, '4-6 hours':5,'6-7 hours':6.5, '10-11 hours':10.5, '8-9 hours':8, '40-45 hours':4.5,
              '9-11 hours':10, '2-3 hours':2.5, '3-4 hours':3.5, '55-66 hours':5.5,'4-5 hours':4.5, '9-6 hours':9, 
              '1-3 hours':2, '45':4.5, '1-6 hours':3,'35-36 hours':5.5, '8 hours':8, 'No':0, '10-6 hours':8, 
              'than 5 hours':4.5,'49 hours':7, '3-6 hours':4.5,'45-48 hours':5, '9-5':7, '9-5 hours':8}


train_df['sleep_duration'] = train_df['sleep_duration'].map(sleep_train)


test_df['sleep_duration'].unique()


sleep_test= {'Less than 5 hours':4, '7-8 hours':7.5, 'More than 8 hours':9, '5-6 hours':5.5,'0':0, '9-5 hours':8, 
             '6-7 hours':6.5, '60-65 hours':6.5, '3-4 hours':3.5, '1-6 hours':3, '9-5':8, '8-9 hours':8.5,
             '4-5 hours':4.5, 'than 5 hours':4.5, '9-6 hours':9, '1-2 hours':1.5,'8-89 hours':8.5, '20-21 hours':7,
             '10-6 hours':8, '1-3 hours':2, '6 hours':6, '50-75 hours':6, '4-6 hours':5,'2-3 hours':2.5, 
             '9-11 hours':10, '9-10 hours':9.5, '3-6 hours':4.5}


test_df['sleep_duration'] = test_df['sleep_duration'].map(sleep_test)


train_df['sleep_duration'] = train_df['sleep_duration'].fillna(train_df['sleep_duration'].median())
test_df['sleep_duration'] = test_df['sleep_duration'].fillna(test_df['sleep_duration'].median())


train_df.isnull().sum()


test_df.isnull().sum()


train_len = len(train_df)
combined = pd.concat([train_df, test_df], axis=0)
cat_cols = train_df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    combined[col] = le.fit_transform(combined[col].astype(str))

train_df = combined.iloc[:train_len, :]
test_df = combined.iloc[train_len:, :]

test_df = test_df.drop(columns=['depression'], errors='ignore')


train_df.info()


test_df.info()


x = train_df.drop('depression', axis=1)
y = train_df['depression']


models = {"XGBoost": xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,random_state=42, 
                                       n_jobs=-1, eval_metric='logloss', verbosity=0),
          "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42,
                                         n_jobs=-1, verbose=-1),
          "CatBoost": CatBoostClassifier(n_estimators=500, learning_rate=0.05, depth=6, random_state=42, 
                                         verbose=0, allow_writing_files=False),
          "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,random_state=42, n_jobs=-1)}

results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    start_time = time.time()
    cv_results = cross_val_score(model, x, y, cv=skf, scoring='accuracy', n_jobs=-1)
    elapsed_time = time.time() - start_time
    
    results.append({"Model": name,"Mean Accuracy": cv_results.mean(),"Std Dev": cv_results.std(),
                    "Min Score": cv_results.min(),"Max Score": cv_results.max(),"Time (s)": elapsed_time})
    
    print(f"{name}: {cv_results.mean():.5f}")

results_df = pd.DataFrame(results).sort_values(by="Mean Accuracy", ascending=False)


results_df


from sklearn.ensemble import VotingClassifier

clf1 = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, 
                         eval_metric='logloss', verbosity=0)
clf2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
clf3 = CatBoostClassifier(n_estimators=500, learning_rate=0.05, depth=6, random_state=42, verbose=0, 
                          allow_writing_files=False)

voting_clf = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('cat', clf3)],voting='soft',weights=[1, 1, 2])

voting_clf.fit(x, y)


test_preds_proba = voting_clf.predict_proba(test_df)[:, 1] 
final_preds = (test_preds_proba > 0.5).astype(int)


submission = pd.DataFrame({'id': pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')['id'], 
                           'depression': final_preds})
submission.to_csv('submission.csv', index=False)


import joblib
columns_names =['id', 'name', 'gender', 'age', 'city', 'occupation_status', 'profession', 
                'academic_pressure', 'work_pressure', 'cgpa', 'study_satisfaction', 'job_satisfaction', 
                'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts', 'work_study_hours', 
                'financial_stress', 'family_history_mental_illness', 'depression']
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv',names=columns_names, header=0)
train.columns = train.columns.str.strip()
train = train.drop(['id', 'name'], axis=1, errors='ignore')

columns_to_encode = ['gender','city','occupation_status','profession','dietary_habits','degree',
                     'suicidal_thoughts','family_history_mental_illness']
clean_sleep_mapping = {'More than 8 hours':9, 'Less than 5 hours':4, '5-6 hours':5.5, '7-8 hours':7.5,'1-2 hours':1.5, 
                 '6-8 hours':7, '4-6 hours':5,'6-7 hours':6.5, '10-11 hours':10.5, '8-9 hours':8, 
                 '40-45 hours':4.5,'9-11 hours':10, '2-3 hours':2.5, '3-4 hours':3.5, '55-66 hours':5.5,
                 '4-5 hours':4.5, '9-6 hours':9, '1-3 hours':2, '45':4.5, '1-6 hours':3,'35-36 hours':5.5,
                 '8 hours':8, 'No':0, '10-6 hours':8, 'than 5 hours':4.5,'49 hours':7, '3-6 hours':4.5,
                 '45-48 hours':5, '9-5':7, '9-5 hours':8,'0':0,'9-5 hours':8, '60-65 hours':6.5,'4-5 hours':4.5, 
                 '8-89 hours':8.5, '20-21 hours':7,'6 hours':6, '50-75 hours':6, '9-10 hours':9.5}
encoders = {}

for col in columns_to_encode:
    le = LabelEncoder()
    train[col] = train[col].fillna('Other').astype(str)
    le.fit(train[col])
    encoders[col] = le

model = {"model": voting_clf,"encoders": encoders,"sleep_mapping": clean_sleep_mapping}

joblib.dump(model, 'depression_model.pkl')

