import pickle
from datetime import timedelta

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


path_to_train='/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv'
path_to_test='/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv'
path_to_submission = '/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/sample_submission.csv'
path_to_pickle = '/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl'


data = pd.read_csv(path_to_train, index_col='session_id')
data.head(10)


with open(path_to_pickle, 'rb') as fpkl:
    site_dict = pickle.load(fpkl)

site_dict


times = [f'time{i}' for i in range(1, 11)]
data[times] = data[times].apply(pd.to_datetime)


data.info()


data.describe()


data.isnull().sum()


data['target'].value_counts()


data.dropna()['target'].value_counts()


data = data.dropna()


data.columns


group0 = data[data['target'] == 0]
group1 = data[data['target'] == 1]



sites0 = pd.concat([group0[f'site{i}'] for i in range(1, 11)])
sites1 = pd.concat([group1[f'site{i}'] for i in range(1, 11)])


fig, ax = plt.subplots()
edgecolors = ['blue', 'orange']
ax.set_title('Alice vs Internet Robber sites')
ax.hist(sites0, label='sites of 0', bins=50, alpha=0.6)
ax.hist(sites1, label='sites of 1', bins=50, alpha=0.6)
# ax.set_yscale('log')
ax.set_xlim(0, 2000)
ax.legend()


time_data = data.sort_values(by='time1')
time_data.head(10)


time_group0 = time_data[time_data['target'] == 0]
time_group1 = time_data[time_data['target'] == 1]


# Answer for 4 question
days_alise = time_group1[times].apply(lambda x: x.dt.weekday)
days_robber = time_group0[times].apply(lambda x: x.dt.weekday)


days_alise.describe()


days_robber.describe()


# answer on 1 and 3 question
start_end_times_alice = time_group1[['time1', 'time10']].apply(lambda x: x.dt.hour * 60 * 60 + x.dt.minute * 60 + x.dt.second)
time_visit_alice = start_end_times_alice.describe()
time_visit_alice.iloc[1:].apply(lambda x: pd.to_timedelta(x, unit='s'))


start_end_times_robbers = time_group0[['time1', 'time10']].apply(lambda x: x.dt.hour * 60 * 60 + x.dt.minute * 60 + x.dt.second)
time_visit_robbers = start_end_times_robbers.describe()
time_visit_robbers.iloc[1:].apply(lambda x: pd.to_timedelta(x, unit='s'))


# answer on 5 questions
result = start_end_times_alice['time10'] - start_end_times_alice['time1']
result.describe()


result = start_end_times_robbers['time10'] - start_end_times_robbers['time1']
result.describe()


# answer on 2 question
pd.DataFrame([(time_group1[f'time{i}'] - time_group1[f'time{i-1}']).dt.seconds for i in range(2, 11)]).T.describe()


pd.DataFrame([(time_group0[f'time{i}'] - time_group0[f'time{i-1}']).dt.seconds for i in range(2, 11)]).T.describe()


def transforming(df):
    additinal_features = df
    times = [f'time{i}' for i in range(1, 11)]
    sites = [f'site{i}' for i in range(1, 11)]

    if 'session_id' in df.columns:
        additinal_features = additinal_features.drop(columns=['session_id'])
        
    additinal_features[times] = additinal_features[times].apply(pd.to_datetime)
    
    additinal_features[sites] = additinal_features[sites].fillna(0).astype(int)
    
    for i in range(1, len(times)):
        current_time = times[i]
        prev_time = times[i-1]
        additinal_features[current_time] = additinal_features[current_time].fillna(additinal_features[prev_time])
    
    additinal_features['start_hour'] = additinal_features['time1'].dt.hour
    additinal_features['start_minut'] = additinal_features['time1'].dt.minute
    additinal_features['start_sec'] = additinal_features['time1'].dt.second
    additinal_features['day_of_week'] = additinal_features['time1'].dt.weekday
    additinal_features[times] = additinal_features[times].apply(lambda x: x.dt.hour * 60 * 60 + x.dt.minute * 60 + x.dt.second)
    
    delta_names = [f'time{i}-{i-1}' for i in range(2, 11)]
    delta_times = pd.DataFrame()
    for i, col in zip(range(2, 11), delta_names):
        delta_times[col] = additinal_features[f'time{i}'] - additinal_features[f'time{i-1}']
    additinal_features = pd.concat([additinal_features, delta_times], axis=1)
    
    return additinal_features

additinal_features = transforming(time_data)
target = additinal_features['target']
del additinal_features['target']
additinal_features['target'] = target


corr_mat = additinal_features.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(additinal_features.drop(columns=['target']))

pca = PCA(n_components=6)
X_pca = pca.fit(X_scaled)


y = additinal_features['target']
# X = additinal_features.drop(columns=['target'])
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred_proba


from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC AUC Score: {roc_auc}")


test_data = pd.read_csv(path_to_test)
test_data.isnull().sum()


def transforming(df):
    additinal_features = df
    times = [f'time{i}' for i in range(1, 11)]
    sites = [f'site{i}' for i in range(1, 11)]
    additinal_features = additinal_features.drop(columns=['session_id'])
    additinal_features[times] = additinal_features[times].apply(pd.to_datetime)
    
    additinal_features[sites] = additinal_features[sites].fillna(0).astype(int)
    
    for i in range(1, len(times)):
        current_time = times[i]
        prev_time = times[i-1]
        additinal_features[current_time] = additinal_features[current_time].fillna(additinal_features[prev_time])
    
    additinal_features['start_hour'] = additinal_features['time1'].dt.hour
    additinal_features['start_minut'] = additinal_features['time1'].dt.minute
    additinal_features['start_sec'] = additinal_features['time1'].dt.second
    additinal_features['day_of_week'] = additinal_features['time1'].dt.weekday
    additinal_features[times] = additinal_features[times].apply(lambda x: x.dt.hour * 60 * 60 + x.dt.minute * 60 + x.dt.second)
    
    delta_names = [f'time{i}-{i-1}' for i in range(2, 11)]
    delta_times = pd.DataFrame()
    for i, col in zip(range(2, 11), delta_names):
        delta_times[col] = additinal_features[f'time{i}'] - additinal_features[f'time{i-1}']
    additinal_features = pd.concat([additinal_features, delta_times], axis=1)
    
    return additinal_features


transformed_test_data = transforming(test_data)
transformed_test_data = scaler.transform(transformed_test_data)
transformed_test_data = pca.transform(transformed_test_data)


transformed_test_data


pred_val = model.predict_proba(transformed_test_data)[:, 1]
pred_val


submission = pd.read_csv(path_to_submission)
submission['target'] = pred_val
submission.to_csv('submission.csv', index=False)

