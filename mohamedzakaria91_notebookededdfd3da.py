import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Available files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


try:
    users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
    kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv') 
    competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
    
    print("\nSuccessfully loaded:")
    print(f"- Users: {users.shape}")
    print(f"- Kernels: {kernels.shape}")
    print(f"- Competitions: {competitions.shape}")
    
except Exception as e:
    print(f"\nError loading data: {str(e)}")
    users, kernels, competitions = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def inspect_data(df, name):
    print(f"\n{name} Data:")
    print("Columns:", df.columns.tolist())
    print("First 2 rows:")
    display(df.head(2))
    print("Missing values:", df.isnull().sum().sum())

inspect_data(users, "Users")
inspect_data(kernels, "Kernels") 
inspect_data(competitions, "Competitions")


users.fillna({'PerformanceTier': 0}, inplace=True)

for df in [users, kernels, competitions]:
    if 'CreationDate' in df.columns:
        df['CreationDate'] = pd.to_datetime(df['CreationDate'])
    if 'RegisterDate' in df.columns:
        df['RegisterDate'] = pd.to_datetime(df['RegisterDate'])

users['TenureDays'] = (pd.to_datetime('today') - users['RegisterDate']).dt.days


df = kernels.merge(users, left_on='AuthorUserId', right_on='Id', how='left')
activity_metrics = 0
for col in ['TotalSubmissions', 'TotalDatasetVersions', 'TotalForumPosts', 'TotalKernels']:
    if col in users.columns:
        activity_metrics += users[col]
df['TotalActivities'] = activity_metrics
if 'HostUserId' in competitions.columns:
    comp_agg = competitions.groupby('HostUserId').size().reset_index(name='HostedCompetitions')
    df = df.merge(comp_agg, left_on='AuthorUserId', right_on='HostUserId', how='left')
else:
    df['HostedCompetitions'] = 0


if 'TotalVotes' in df.columns:
    y = df['TotalVotes']
else:
    print("Warning: Using dummy target variable")
    y = np.random.randint(0, 10, size=len(df))


from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
available_features = ['TenureDays', 'PerformanceTier', 'TotalActivities', 'HostedCompetitions']
X = df[[col for col in available_features if col in df.columns]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}")


try:
    test_data = pd.read_csv('/kaggle/input/meta-kaggle/test.csv')
    test_preds = model.predict(test_data[X.columns])
    submission = pd.DataFrame({'Id': test_data['Id'], 'Prediction': test_preds})
    submission.to_csv('submission.csv', index=False)
    print("Submission file created!")
except:
    print("No test.csv found - skipping submission")

