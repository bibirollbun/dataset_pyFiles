import numpy as np 
import pandas as pd 

df_age=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip')
df_age


df_countries=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/countries.csv.zip')
df_countries


df_submission=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sample_submission_NDF.csv.zip')
df_submission


df_sessions=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')
df_sessions


import warnings
warnings.filterwarnings("ignore")
df_train=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
df_train


df_test=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')
df_test


df_sessions.info()


df_sessions.isnull().sum()


df_sessions['action'].fillna('unknown', inplace=True)
df_sessions['action_type'].fillna('unknown', inplace=True)
df_sessions['action_detail'].fillna('unknown', inplace=True)
df_sessions['device_type'] = df_sessions['device_type'].replace('-unknown-', 'unknown')
df_sessions['action_type'] = df_sessions['action_type'].replace('-unknown-', 'unknown')
df_sessions['action_detail'] = df_sessions['action_detail'].replace('-unknown-', 'unknown')


secs_elapsed_median=df_sessions['secs_elapsed'].median()
secs_elapsed_median


df_sessions.describe().T


def impute_secs_elapsed(row):
    if row['action'] == 'unknown':
        return 0  
    elif pd.isnull(row['secs_elapsed']):
        return secs_elapsed_median  
    else:
        return row['secs_elapsed']  

df_sessions['secs_elapsed'] = df_sessions.apply(impute_secs_elapsed, axis=1)


df_sessions.isnull().sum()


df_train.info()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train['date_first_booking'] = pd.to_datetime(df_train['date_first_booking'])


df_train['date_first_booking'].median()


df_train['date_first_booking'] = df_train['date_first_booking'].fillna(df_train['date_first_booking'].median())



df_train.describe().T


Q1 = df_train['age'].quantile(0.25)
Q3 = df_train['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


lower_bound



df_train['age'] = np.where((df_train['age'] >= lower_bound) & (df_train['age'] <= upper_bound), df_train['age'], df_train['age'].median())


bins = [15, 25, 35, 45, 55, 65, 75, 85, 100]
labels = ['15-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-100']

df_train['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels, right=False)



df_train.drop(columns=['age'], inplace=True)


modevalue=df_train['first_affiliate_tracked'].mode()[0]
modevalue



df_train['first_affiliate_tracked'].fillna(modevalue, inplace=True)



df_train.isnull().sum()


df_train


oldest_date_1 = df_train['date_first_booking'].min()
latest_date_2 = df_train['date_first_booking'].max()



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import ndcg_score, accuracy_score



def preprocess_data(df):
    
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S')
    
    df['account_created_year'] = df['date_account_created'].dt.year
    df['account_created_month'] = df['date_account_created'].dt.month
    df['account_created_day'] = df['date_account_created'].dt.day
    df['first_active_year'] = df['timestamp_first_active'].dt.year
    df['first_active_month'] = df['timestamp_first_active'].dt.month
    df['first_active_day'] = df['timestamp_first_active'].dt.day
    df['first_booking_year'] = df['date_first_booking'].dt.year
    df['first_booking_month'] = df['date_first_booking'].dt.month
    df['first_booking_day'] = df['date_first_booking'].dt.day
    df.drop(['date_account_created', 'date_first_booking', 'timestamp_first_active'], axis=1, inplace=True)
    
    
    
    return df

train_users = preprocess_data(df_train)




train_users.describe().T


df_sessions.rename(columns={'user_id': 'id'}, inplace=True)


session_agg = df_sessions.groupby('id').agg({
    'secs_elapsed': ['sum', 'mean', 'count'],
    'action': ['count'],
    'action_type': ['nunique'],
    'action_detail': ['nunique'],
    'device_type': ['nunique']
}).reset_index()
session_agg.columns = ['id','total_secs', 'avg_secs', 'session_count', 
                       'count_actions', 'unique_action_types', 'unique_action_details', 
                       'unique_device_types']

train_users = train_users.merge(session_agg, on='id', how='left')


train_users.isnull().sum()


train_users.describe().T


train_users['total_secs'].fillna(train_users['total_secs'].median(), inplace=True)
train_users['avg_secs'].fillna(train_users['avg_secs'].median(), inplace=True)
train_users['session_count'].fillna(train_users['session_count'].median(), inplace=True)
train_users['count_actions'].fillna(train_users['count_actions'].median(), inplace=True)
train_users['unique_action_types'].fillna(train_users['unique_action_types'].median(), inplace=True)
train_users['unique_action_details'].fillna(train_users['unique_action_details'].median(), inplace=True)
train_users['unique_device_types'].fillna(train_users['unique_device_types'].median(), inplace=True)


df_train.drop(columns=['id'], inplace=True)



df_age.isnull().sum()


df_countries.isnull().sum()


df_test


df_test.isnull().sum()


df_test.describe().T


df_test['date_first_booking'] = pd.to_datetime(df_test['date_first_booking'], errors='coerce')


start_date = oldest_date_1
end_date = latest_date_2

date_range = pd.date_range(start=start_date, end=end_date, freq='D')

missing_dates_count = df_test['date_first_booking'].isna().sum()

random_dates = np.random.choice(date_range, size=missing_dates_count)

df_test.loc[df_test['date_first_booking'].isna(), 'date_first_booking'] = random_dates


df_test['date_first_booking'] = pd.to_datetime(df_test['date_first_booking'])


test_users = preprocess_data(df_test)



test_users.isnull().sum()


test_users['first_affiliate_tracked'] = test_users['first_affiliate_tracked'].fillna(test_users['first_affiliate_tracked'].mode()[0])



test_users['gender'] = test_users['gender'].replace('-unknown-', 'Unknown')
test_users['language'] = test_users['language'].replace('-unknown-', 'Unknown')
mode_signup_method = test_users['signup_method'].mode()[0]

# Replace 'weibo' with the mode in the 'signup_method' column bec is not present in the df_train
test_users['signup_method'] = test_users['signup_method'].replace('weibo', mode_signup_method)


Q1 = test_users['age'].quantile(0.25)
Q3 = test_users['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


test_users['age'] = np.where((test_users['age'] >= lower_bound) & (test_users['age'] <= upper_bound), test_users['age'], test_users['age'].median())


bins = [15, 25, 35, 45, 55, 65, 75, 85, 100]
labels = ['15-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-100']

test_users['age_group'] = pd.cut(test_users['age'], bins=bins, labels=labels, right=False)



test_users.isnull().sum()



session_agg = df_sessions.groupby('id').agg({
    'secs_elapsed': ['sum', 'mean', 'count'],
    'action': ['count'],
    'action_type': ['nunique'],
    'action_detail': ['nunique'],
    'device_type': ['nunique']
}).reset_index()
session_agg.columns = ['id','total_secs', 'avg_secs', 'session_count', 
                       'count_actions', 'unique_action_types', 'unique_action_details', 
                       'unique_device_types']

test_users = test_users.merge(session_agg, on='id', how='left')


test_users['total_secs'].fillna(test_users['total_secs'].median(), inplace=True)
test_users['avg_secs'].fillna(test_users['avg_secs'].median(), inplace=True)
test_users['session_count'].fillna(test_users['session_count'].median(), inplace=True)
test_users['count_actions'].fillna(test_users['count_actions'].median(), inplace=True)
test_users['unique_action_types'].fillna(test_users['unique_action_types'].median(), inplace=True)
test_users['unique_action_details'].fillna(test_users['unique_action_details'].median(), inplace=True)
test_users['unique_device_types'].fillna(test_users['unique_device_types'].median(), inplace=True)


test_users.isnull().sum()


train_users.describe()


train_users.columns


train_users.info()


import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, numerical_cols, categorical_cols):
    print("\nUnivariate Analysis for Categorical Columns:")
    if categorical_cols:
        num_categorical = len(categorical_cols)
        nrows = (num_categorical // 2) + (1 if num_categorical % 2 != 0 else 0)
        ncols = 2

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 5))
        fig.suptitle("Univariate Analysis for Categorical Columns", fontsize=16, y=1.02)

        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].tick_params(axis='x', rotation=90)
            else:
                axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    print("Univariate Analysis for Numerical Columns:")
    if numerical_cols:
        num_numerical = len(numerical_cols)
        nrows = (num_numerical // 2) + (1 if num_numerical % 2 != 0 else 0)
        ncols = 2  # Fixed to 3 columns

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 5))
        fig.suptitle("Univariate Analysis for Numerical Columns", fontsize=16, y=1.02)

        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            if col in df.columns:
                sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
            else:
                axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    


numerical_cols = [ 'account_created_year', 'account_created_month', 
                  'account_created_day', 'first_active_year', 'first_active_month', 
                  'first_active_day', 'first_booking_year', 'first_booking_month', 
                  'first_booking_day']
categorical_cols = ['gender', 'signup_method', 'language', 'affiliate_channel', 
                    'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 
                    'first_device_type', 'first_browser', 'age_group']

univariate_analysis(train_users, numerical_cols, categorical_cols)



plt.figure(figsize=(12, 6))

ax = sns.barplot(x='age_group', y='total_secs', data=train_users, estimator='mean', palette='viridis')

plt.title("Average Time Spent by each Age Group", fontsize=16)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Average Total Seconds Spent", fontsize=12)

plt.xticks(rotation=45)

ax.grid(True, linestyle='--', alpha=0.6)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

ax.set_facecolor('#f7f7f7')

plt.show()


plt.figure(figsize=(6, 6))
sns.countplot(data=df_sessions, x='action_type')
plt.title('Distribution of Action Types')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(6,6))
ax=df_sessions['device_type'].value_counts().plot(kind='bar', color='grey')
plt.title('Device Type Distribution')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.subplot(2, 2, 2)
sns.histplot(df_countries['distance_km'], bins=10, kde=True)
plt.title('Distribution of Distances')


def bivariate_analysis(df, target_col, categorical_cols):
    print("\nBivariate Analysis for Categorical Columns vs Target:")
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=col, hue=target_col, order=df[col].value_counts().index)
            plt.title(f"{col} vs {target_col}")
            plt.xticks(rotation=90)
            plt.show()

target_col = 'country_destination'
categorical_cols = ['gender', 'signup_method','signup_app','age_group']

# Perform Bivariate Analysis on Train Dataset
bivariate_analysis(train_users, target_col, categorical_cols)


plt.figure(figsize=(10,10))
ax=train_users.groupby('country_destination')['signup_app'].value_counts().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Signup App by Country Destination')
plt.xlabel('Country Destination')
plt.ylabel('Count')
ax.bar_label(ax.containers[0])
plt.show()


plt.figure(figsize=(10,10))
ax=train_users.groupby('country_destination')['signup_method'].value_counts().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Signup Method by Country Destination')
plt.xlabel('Country Destination')
plt.ylabel('Count')
ax.bar_label(ax.containers[0])
plt.show()


print("\nMultivariate Analysis: Correlation Matrix")
plt.figure(figsize=(12, 8))
corr = train_users.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.savefig(r'C:\Users\Dell\Desktop\action_type_distribution.png')  
plt.show()


train_users.info()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

columns_to_encode = ['gender','signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 
                     'first_affiliate_tracked', 'signup_app', 'first_device_type', 
                     'first_browser', 'age_group']

for column in columns_to_encode:
    train_users[column] = label_encoder.fit_transform(train_users[column].astype(str))



train_users['country_destination'] = label_encoder.fit_transform(train_users['country_destination'])#i transform it alone bec it isnot in df_test



train_users.info()


X = train_users.drop(['country_destination','id'], axis=1)
y = train_users['country_destination']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



import xgboost as xgb

model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=12,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True 
)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy on validation set: {accuracy:.4f}')

y_pred_proba = model.predict_proba(X_val)

ndcg = ndcg_score(np.array(pd.get_dummies(y_val)), y_pred_proba, k=5)
print(f'NDCG@5 on validation set: {ndcg:.4f}')



for column in columns_to_encode:
    test_users[column] = label_encoder.fit_transform(test_users[column].astype(str))


test_users= test_users.drop('id', axis=1)



test_users.info()


y_pred_proba = model.predict_proba(X_val)


