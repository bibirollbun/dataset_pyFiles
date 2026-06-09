#importing neccessary libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import tensorflow as tf 
import plotly.express as px
import matplotlib.pyplot as plt 
from sklearn.metrics import ndcg_score
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#Loading the Data
df_train = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
df_test = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')
df_train.shape, df_test.shape


train_users = df_train.shape[0]
test_users = df_test.shape[0]
total_users = train_users + test_users

print(f"We have {train_users} users in the training set and {test_users} in the test set.")
print(f"In total, we have {total_users} users.")


#Concatenating train and test data for EDA
users = pd.concat((df_train, df_test), axis=0, ignore_index=True)
users.head()


# Remove ID's since now we are not interested in making predictions
id_test = df_test.id
users.drop('id',axis=1, inplace=True)


users.info()


users.gender.replace('-unknown-', np.nan, inplace=True)


users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')


users.describe()


users[users.age < 18]['age'].describe()


users.loc[users.age > 95, 'age'] = np.nan
users.loc[users.age < 13, 'age'] = np.nan


categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')


users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')


fig = px.pie(users, names='gender', title='Gender Distribution',color_discrete_sequence=['skyblue', 'lightcoral'])
fig.show(renderer='iframe')


import plotly.graph_objects as go
destination_counts = users['country_destination'].value_counts()
destination_percentage = (destination_counts / destination_counts.sum()) * 100

text_labels = [f'{val:.1f}%' for val in destination_percentage.values]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=destination_percentage.index,
    y=destination_percentage.values,
    marker_color='lightcoral', 
    text=text_labels,  
    textposition='outside' 
))
fig.update_layout(
    xaxis_title='Destination Country',
    yaxis_title='Percentage (%)',
    title='Destinations Distribution',
)
fig.show(renderer='iframe')


mean_age = users['age'].mean()
median_age = users['age'].median()
fig_age = px.histogram(users, x='age', nbins=50, marginal='violin', color_discrete_sequence=['#FD5C64'], opacity=0.7)
fig_age.add_vline(x=mean_age, line_dash='dash', line_color='red', annotation_text=f"Mean: {mean_age:.1f}", annotation_position="top right")
fig_age.add_vline(x=median_age, line_dash='dot', line_color='green', annotation_text=f"Median: {median_age:.1f}", annotation_position="top left")
fig_age.update_layout(
    xaxis_title='Age',
    title='Age Distribution',
    showlegend=False
)
fig_age.show(renderer='iframe')


age = 45
younger = users.loc[users['age'] < age, 'country_destination'].value_counts().sum()
older = users.loc[users['age'] > age, 'country_destination'].value_counts().sum()
younger_destinations = (users.loc[users['age'] < age, 'country_destination'].value_counts() / younger) * 100
older_destinations = (users.loc[users['age'] > age, 'country_destination'].value_counts() / older) * 100
younger_text = [f"{val:.1f}%" for val in younger_destinations.values]
older_text = [f"{val:.1f}%" for val in older_destinations.values]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=younger_destinations.index,
    y=younger_destinations.values,
    name='Younger (< 45)',
    marker_color='lightcoral',  
    text=younger_text, 
    textposition='outside'
))
fig.add_trace(go.Bar(
    x=older_destinations.index,
    y=older_destinations.values,
    name='Older (> 45)',
    marker_color='skyblue',  
    text=older_text,  
    textposition='outside'
))
fig.update_layout(
    barmode='group',  
    xaxis_title='Destination Country',
    yaxis_title='Percentage (%)',
    title='Destinations Distribution by Age Group',
    legend_title='Age Group'
)
fig.show(renderer='iframe')


print((sum(users.language == 'en') / users.shape[0])*100)


device_counts = users['first_device_type'].value_counts().sort_values(ascending=True) 
fig = px.bar(
    x=device_counts.values,
    y=device_counts.index,
    orientation='h',  
    labels={'x': 'Count', 'y': 'Device Type'},
    title='Device Type Distribution',
    text=device_counts.values
)
fig.update_traces(textposition='outside', marker_color='lightcoral')
fig.show(renderer='iframe')


users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
weekly_created = users.groupby(users['date_account_created'].dt.to_period('W')).size().reset_index(name='created_count')
weekly_booked = users.groupby(users['date_first_booking'].dropna().dt.to_period('W')).size().reset_index(name='booked_count')
weekly_created['date'] = weekly_created['date_account_created'].dt.start_time
weekly_booked['date'] = weekly_booked['date_first_booking'].dt.start_time
weekly_created.drop(columns=['date_account_created'], inplace=True)
weekly_booked.drop(columns=['date_first_booking'], inplace=True)
weekly_stats = pd.merge(weekly_created, weekly_booked, on='date', how='outer').fillna(0)
weekly_stats = weekly_stats.melt(id_vars='date', var_name='Metric', value_name='Count')
fig = px.line(
    weekly_stats,
    x='date',
    y='Count',
    color='Metric',
    labels={'date': 'Date', 'Count': 'Number of Users'},
    title='Weekly Trends: Account Creation vs Bookings'
)
fig.update_layout(
    xaxis=dict(tickangle=-45, tickformat='%Y'), 
    yaxis_title='User Count',
    legend_title='Metric',
    template='simple_white' 
)
fig.show(renderer='iframe')


# Calculate the percentage of users based on affiliate providers
affiliate_provider_percentage = (users['affiliate_provider'].value_counts() / users.shape[0]) * 100
fig = px.bar(
    x=affiliate_provider_percentage.index, 
    y=affiliate_provider_percentage.values,
    text=affiliate_provider_percentage.values.round(2),
    labels={'x': 'Affiliate Providers', 'y': 'Percentage of Users'},
    title="User Distribution by Affiliate Provider",
)
fig.update_traces(textposition='outside', marker_color='lightcoral')
fig.update_layout(
    xaxis_title="Affiliate Providers",
    yaxis_title="Percentage of Users",
    template="plotly_white"
)
fig.show(renderer='iframe')


sessions = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')
sessions.head(5) 


sessions.shape


sessions['secs_elapsed'].describe()


len(sessions[sessions['secs_elapsed'].isnull()])


sns.set_style("whitegrid")
sns.kdeplot(sessions['secs_elapsed'].dropna(), shade=True)
plt.show()


median_secs = sessions['secs_elapsed'].median()
sessions['secs_elapsed'] = sessions['secs_elapsed'].fillna(median_secs)


sns.histplot(sessions['secs_elapsed'])
plt.xlim(0, 5000)  
plt.show()


def fill_age_nulls(df):
    print(f"Operation affected {df.isna().sum().age:,} records in the users dataset.")
    df.age.fillna(MEDIAN_AGE, inplace=True)


# Creating a copy to keep the original dataframe without filling the nulls
user_filled = users.copy()


# The median could be better since the age distribution is skewed
MEDIAN_AGE = users.age.median()
fill_age_nulls(user_filled)


def split_dates(df, drop_original=False, verbose=True):
    '''Splits any column with a datetime64[ns] data type into three separate columns: _year, _month, and _day.
       To improve model performance'''
    date_cols = df.select_dtypes('datetime64[ns]').columns
    if len(date_cols) == 0:
        print("No columns with dtype of datetime[ns]!")
        return None
    for date_col in date_cols:
        if verbose:
            print(f"Splitting {date_col} Column...")
        df[date_col+'_year'] = df[date_col].dt.year
        df[date_col+'_month'] = df[date_col].dt.month
        df[date_col+'_day'] = df[date_col].dt.day
    if drop_original:
        df.drop(date_cols, axis=1, inplace=True)


split_dates(user_filled)


cat_cols = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

# Initialize One Hot Encoder (Since the features are nominal)
ohe_encoder = OneHotEncoder(handle_unknown='ignore')
ohe_encoded_train = ohe_encoder.fit_transform(user_filled[cat_cols])


# get the features names
feat_ohe_names = ohe_encoder.get_feature_names_out()
# Construct Dataframe
ohe_encoded_train_df = pd.DataFrame(ohe_encoded_train.toarray(), columns=feat_ohe_names)
# Preview Dataframe
ohe_encoded_train_df.head()


y_train = user_filled['country_destination']
# Initialize LabelEncoder
lbl_encoder = LabelEncoder()
# Encode y
y_train = lbl_encoder.fit_transform(y_train)
y_train


# date columns we already split into year, month, day
columns_to_remove = ['date_account_created', 'timestamp_first_active', 'date_first_booking','date_first_active']
# Categorical columns already encoded
columns_to_remove.extend(cat_cols)
target_column = ['country_destination']
# Drop unnecessary columns
user_filled.drop(columns_to_remove+target_column, axis=1, inplace=True)


user_filled.head()


# Concatenate merged_train with ohe_encoded_train_df
x_train = pd.concat([user_filled, ohe_encoded_train_df], axis=1)
# Preview current dimensions
x_train.shape


print(f"Shape of X: {x_train.shape}")
print(f"Shape of y: {y_train.shape}")



X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
model = XGBClassifier(
    objective='multi:softmax',  
    num_class=13,  
    max_depth=6,
    learning_rate=0.3,  
    n_estimators=100,  
    # multi-class log loss measuring how well the predicted probability distribution matches the actual classes.
    eval_metric='mlogloss', 
    use_label_encoder=False  
)
model.fit(X_train, y_train)
# Make predictions
y_pred_proba = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy * 100:.2f}%")



# Ensure predictions are probabilities, not labels
y_pred_proba = model.predict_proba(X_test)  
# Get the number of classes from prediction output
num_classes = y_pred_proba.shape[1] 

# Convert y_test to one-hot encoding
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

# Compute NDCG@5 score
ndcg = ndcg_score(y_test_bin, y_pred_proba, k=5)
print(f'NDCG@5 Score: {ndcg}')

# Extract top-5 predictions for each test sample
ids = []  
cts = []  
for i in range(len(y_pred_proba)): 
    idx = id_test[i] 
    top_5_preds = np.argsort(y_pred_proba[i])[::-1][:5] 
    top_5_countries = lbl_encoder.inverse_transform(top_5_preds) # Convert to original labels
    ids += [idx] * 5  # Repeat ID 5 times
    cts += top_5_countries.tolist()


#Generate submission
#sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id_test', 'country'])
#sub.to_csv('sample_submission.csv',index=False)

