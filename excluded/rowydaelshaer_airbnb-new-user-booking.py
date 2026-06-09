import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip")


df.head()


df.shape


df.dtypes


df.describe()


df.isnull().sum()


df.columns


plt.figure(figsize=(10,5))
sns.countplot(y=df['country_destination'], order=df['country_destination'].value_counts().index)
plt.title("Distribution of Destination Countries")
plt.show()

df['country_destination'].value_counts(normalize=True) * 100


df['age'].describe()

plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=50, kde=True)
plt.title("Age Distribution")
plt.show()



sns.countplot(x=df['gender'])
plt.title("Gender Distribution")
plt.show()

df['gender'].value_counts(normalize=True) * 100


sns.countplot(x='signup_method', data=df)
sns.countplot(x='affiliate_channel', data=df)
sns.countplot(x='signup_flow', data=df)


df['age'].sort_values().head(10)
df['age'].sort_values(ascending=False).head(10)


plt.figure(figsize=(8,5))
sns.countplot(x='signup_method', data=df, order=df['signup_method'].value_counts().index)
plt.title("Signup Method Distribution")
plt.show()

df['signup_method'].value_counts(normalize=True) * 100



sns.countplot(x='signup_flow', data=df)
plt.title("Signup Flow Distribution")
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x='affiliate_channel', data=df, order=df['affiliate_channel'].value_counts().index)
plt.title("Affiliate Channel Distribution")
plt.show()


pd.crosstab(df['signup_method'], df['country_destination'], normalize='index') * 100



df['first_affiliate_tracked'].value_counts()


sns.countplot(y='first_device_type', data=df)
plt.title("First Device Type Distribution")
plt.show()

df['first_browser'].value_counts().head(10)
sns.countplot(y='first_browser', data=df, order=df['first_browser'].value_counts().index[:10])
plt.title("Top 10 First Browsers")
plt.show()


categorical_cols = ['gender', 'signup_method', 'signup_flow', 'language', 
                    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 
                    'signup_app', 'first_device_type', 'first_browser']

missing_cats = df[categorical_cols].isnull().sum()
missing_cats_percent = (missing_cats / len(df)) * 100
missing_cats_percent.sort_values(ascending=False)


for col in categorical_cols:
    df[col] = df[col].fillna('unknown')

df['gender'] = df['gender'].replace('-unknown-', 'unknown')


df[categorical_cols].isnull().sum()


df['age'].describe()
print("Number of users with age < 15:", (df['age'] < 15).sum())
print("Number of users with age > 90:", (df['age'] > 90).sum())


df.loc[(df['age'] < 15) | (df['age'] > 90), 'age'] = np.nan
df['age'].fillna(df['age'].median(), inplace=True)


plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=50, kde=True, color='green')
plt.title("Age Distribution (After Cleaning)")
plt.show()

df['age'].describe()


df['date_account_created'] = pd.to_datetime(df['date_account_created'], errors='coerce')
df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S', errors='coerce')
df['date_first_booking'] = pd.to_datetime(df['date_first_booking'], errors='coerce')


df['year_account_created'] = df['date_account_created'].dt.year
df['month_account_created'] = df['date_account_created'].dt.month
df['day_account_created'] = df['date_account_created'].dt.day


plt.figure(figsize=(8,4))
sns.countplot(x='year_account_created', data=df)
plt.title("Accounts Created per Year")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='month_account_created', data=df)
plt.title("Accounts Created per Month")
plt.show()


df['signup_delay'] = (df['timestamp_first_active'] - df['date_account_created']).dt.days



df['signup_delay'].describe()


plt.figure(figsize=(8,5))
sns.histplot(df['signup_delay'], bins=50, kde=True)
plt.title("Distribution of Signup Delay (days)")
plt.xlabel("Days between account creation and first activity")
plt.show()



categorical_cols = ['gender', 'signup_method', 'language', 'affiliate_channel',
                    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
                    'first_device_type', 'first_browser']

for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(normalize=True)[:10])



# Check missing values in date_first_booking
print("Missing date_first_booking:", df['date_first_booking'].isnull().mean() * 100, "%")


cols_to_drop = ['days_between_signup_and_active']
cols_to_drop.append('date_first_booking')
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print(" Deleted columns:", cols_to_drop)
print(" Remaining columns:", df.columns.tolist())


df.head()


plt.figure(figsize=(10,5))
sns.boxplot(x='country_destination', y='age', data=df)
plt.title("Age distribution per country_destination")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(x='country_destination', hue='gender', data=df)
plt.title("Gender distribution per country_destination")
plt.show()


df[df['gender']=='unknown']


unknown_gender = df[df['gender'] == 'unknown']
booked_unknown = unknown_gender[unknown_gender['country_destination'] != 'NDF']
num_booked = len(booked_unknown)
percentage_booked = num_booked / len(unknown_gender) * 100

print("Number of users with gender unknown who booked:", num_booked)
print("Percentage booked:", percentage_booked, "%")



plt.figure(figsize=(10,5))
sns.histplot(booked_unknown['age'], bins=30, kde=True)
plt.title("Age distribution of 'unknown' gender users who booked")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
print(booked_unknown['age'].describe())


df[df.country_destination!='NDF'].country_destination.count()


plt.figure(figsize=(8,5))
sns.countplot(x='signup_method', hue='country_destination', data=df)
plt.title("Signup Method vs Country Destination")
plt.xlabel("Signup Method")
plt.ylabel("Count")
plt.show()

signup_ct = pd.crosstab(df['signup_method'], df['country_destination'], normalize='index') * 100
print(signup_ct)



plt.figure(figsize=(12,5))
sns.countplot(x='signup_flow', hue='country_destination', data=df)
plt.title("Signup Flow vs Country Destination")
plt.xlabel("Signup Flow")
plt.ylabel("Count")
plt.show()

signup_flow_ct = pd.crosstab(df['signup_flow'], df['country_destination'], normalize='index') * 100
print(signup_flow_ct)



sessions = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip")



sessions.head()


sessions.shape


sessions.dtypes


sessions.isnull().sum()


sessions.columns


sessions['action'].value_counts().head(10)
sessions['action_type'].value_counts().head(10)
sessions['action_detail'].value_counts().head(10)
sessions['device_type'].value_counts().head(10)


user_session_count = sessions.groupby('user_id').size().reset_index(name='session_count')



user_total_secs = sessions.groupby('user_id')['secs_elapsed'].sum().reset_index(name='total_secs')


user_sessions_summary = pd.merge(user_session_count, user_total_secs, on='user_id')
user_sessions_summary.head()


user_action_counts = sessions.pivot_table(index='user_id', 
                                          columns='action', 
                                          aggfunc='size', 
                                          fill_value=0).reset_index()

user_device_counts = sessions.pivot_table(index='user_id',
                                          columns='device_type',
                                          aggfunc='size',
                                          fill_value=0).reset_index()


from functools import reduce

dfs_to_merge = [user_session_count, user_total_secs, user_action_counts, user_device_counts]
user_features = reduce(lambda left, right: pd.merge(left, right, on='user_id', how='outer'), dfs_to_merge)

user_features.head()


train_merged = df.merge(user_features, left_on='id', right_on='user_id', how='left')
train_merged.fillna(0, inplace=True)


train_merged.fillna(0, inplace=True)
train_merged.head()


print(train_merged[['session_count', 'total_secs', 'country_destination']].describe())
print(train_merged['country_destination'].value_counts(normalize=True) * 100)


plt.figure(figsize=(10,5))
sns.boxplot(x='country_destination', y='session_count', data=train_merged)
plt.title("Session Count per Country Destination")
plt.yscale('log')  
plt.show()



plt.figure(figsize=(10,5))
sns.boxplot(x='country_destination', y='total_secs', data=train_merged)
plt.title("Total Seconds Spent per Country Destination")
plt.yscale('log')
plt.show()


actions_cols = user_action_counts.columns.tolist()
actions_cols.remove('user_id')


actions_cols_in_merged = [col for col in user_action_counts.columns if col != 'user_id']

action_means = train_merged.groupby('country_destination')[actions_cols_in_merged].mean().transpose()
top_actions = action_means.sort_values('US', ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.heatmap(top_actions, annot=True, fmt=".1f", cmap='YlGnBu')
plt.title("Top 10 Actions Average per Country Destination")
plt.show()



train_merged.head()



unknown_gender_users = train_merged[train_merged['gender'] == 'unknown']
action_cols = user_action_counts.columns.tolist()
action_cols.remove('user_id') 

unknown_action_counts = unknown_gender_users[action_cols]

unknown_action_counts.describe()


unknown_gender_users = train_merged[train_merged['gender']=='unknown']

actions_cols = user_action_counts.columns.tolist()
actions_cols.remove('user_id') 

action_means_unknown = unknown_gender_users.groupby('country_destination')[actions_cols].mean().transpose()
top_actions = action_means_unknown.sort_values('US', ascending=False).head(10)  # مثال حسب US

plt.figure(figsize=(12,6))
sns.heatmap(top_actions, annot=True, fmt=".1f", cmap='Reds')
plt.title("Top 10 Actions Average per Country Destination (Unknown Gender)")
plt.show()



num_unknown = train_merged[train_merged['gender'] == 'unknown'].shape[0]
print("Unknown Gender:", num_unknown)


unknown_gender = train_merged[train_merged['gender']=='unknown']
actions_cols = [col for col in user_action_counts.columns if col != 'user_id']
unknown_gender['total_actions'] = unknown_gender[actions_cols].sum(axis=1)

active_unknown = unknown_gender[unknown_gender['total_actions'] > 0]
inactive_unknown = unknown_gender[unknown_gender['total_actions'] == 0]

print("Number of active unknown gender users:", len(active_unknown))
print("Number of inactive unknown gender users:", len(inactive_unknown))


active_unknown = train_merged[(train_merged['gender'] == 'unknown') & 
                              ((train_merged[user_action_counts.columns[1:]] > 0).any(axis=1))]

train_merged = train_merged.drop(train_merged[(train_merged['gender'] == 'unknown') & 
                                              ((train_merged[user_action_counts.columns[1:]] == 0).all(axis=1))].index)

train_merged['gender'].value_counts()



train_merged['avg_secs_per_session'] = train_merged['total_secs'] / (train_merged['session_count'] + 1e-5)
train_merged[['id', 'session_count', 'total_secs', 'avg_secs_per_session']].head()


train_merged_active = train_merged[train_merged['session_count'] > 0]
train_merged_active[['id', 'session_count', 'total_secs', 'avg_secs_per_session']].head()



action_cols = [col for col in train_merged.columns if col in sessions['action'].unique()]
train_merged['total_actions'] = train_merged[action_cols].sum(axis=1)
ratios = pd.DataFrame({col + '_ratio': train_merged[col] / (train_merged['total_actions'] + 1e-5)
                       for col in action_cols})
train_merged = pd.concat([train_merged, ratios], axis=1)


device_cols = ['Mac Desktop', 'iPhone', 'Tablet', 'Windows Desktop', 'Linux Desktop'] 
train_merged['total_devices'] = train_merged[device_cols].sum(axis=1)
for col in device_cols:
    train_merged[col + '_ratio'] = train_merged[col] / (train_merged['total_devices'] + 1e-5)



categorical_cols = ['gender', 'signup_method', 'affiliate_channel', 'first_device_type', 'first_browser']
train_merged = pd.get_dummies(train_merged, columns=categorical_cols, drop_first=True)


from sklearn.preprocessing import StandardScaler
num_cols = ['age', 'signup_delay', 'session_count', 'total_secs', 'avg_secs_per_session']
scaler = StandardScaler()
train_merged[num_cols] = scaler.fit_transform(train_merged[num_cols])



target = 'country_destination' 
X = train_merged.drop(columns=[target, 'id', 'date_account_created', 'timestamp_first_active'])  
y = train_merged[target]



cat_cols = X.select_dtypes(include='object').columns
print("Categorical columns:", cat_cols)


print(X.dtypes.value_counts())


X = pd.get_dummies(X, columns=cat_cols, drop_first=True)


print(X.dtypes.value_counts())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sample = train_merged.sample(frac=0.3, random_state=42)

target = 'country_destination'

X = sample.drop(columns=[target, 'id', 'date_account_created', 'timestamp_first_active'])
y = sample[target]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



!pip install xgboost


from sklearn.preprocessing import LabelEncoder
target = 'country_destination'

X = train_merged.drop(columns=[target, 'id', 'date_account_created', 'timestamp_first_active'])
y = train_merged[target]

cat_cols = X.select_dtypes(include='object').columns
le_X = LabelEncoder()

for col in cat_cols:
    X[col] = le_X.fit_transform(X[col].astype(str))

le_y = LabelEncoder()
y = le_y.fit_transform(y)

sample = X.copy()
sample['target'] = y
sample = sample.sample(frac=0.3, random_state=42)

X = sample.drop(columns=['target'])
y = sample['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_y.classes_))




