import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


df_age = pd.read_csv('age_gender_bkts.csv')
df_age.head()


df_age.info()


df_age.describe()


df_age.nunique()


df_age['country_destination'].value_counts()


df_age.isnull().sum()


df_age['age_bucket'] = df_age['age_bucket'].apply(lambda x: '100-106' if x == '100+' else x)
df_age['mean_age'] = df_age['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)
df_age = df_age.drop('age_bucket', axis=1)
df_age.head()


df_age['gender'] = df_age['gender'].apply(lambda x: 1 if x == 'FEMALE' else 0)
df_age.head()


df_age = df_age.drop('year', axis=1)
df_age.head()


plt.figure(figsize=(10,5))
sns.barplot(x='mean_age', y='population_in_thousands', hue='gender', data=df_age, ci=None)


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
pop_stats = df_age.groupby('country_destination')['population_in_thousands'].sum()
sns.barplot(x=pop_stats.index, y=pop_stats)


df_con = pd.read_csv('countries.csv')
df_con.head()


df_con.info()


df_con['country_destination'].value_counts()


df_con.isnull().sum()


sns.set_style('whitegrid')
plt.figure(figsize=(8,5))
sns.barplot(x='country_destination', y='distance_km', data=df_con)


df_son = pd.read_csv('sessions.csv')
df_son.head()


df_son.info()


df_son.isnull().sum()


df_son['action'] = df_son['action'].replace('-unknown-', np.nan)
df_son['action_type'] = df_son['action_type'].replace('-unknown-', np.nan)
df_son['action_detail'] = df_son['action_detail'].replace('-unknown-', np.nan)


sns.boxplot(data=df_son['secs_elapsed'], orient='h', palette='Set2')
plt.title('secs_elapsed')
plt.show()


df_son['secs_elapsed'].describe()


df_son['secs_elapsed'].isnull().sum()


df_son['secs_elapsed'] = df_son['secs_elapsed'].fillna(df_son['secs_elapsed'].median())


train_data = pd.read_csv('train_users_2.csv')
test_x = pd.read_csv('test_users.csv')
df_all = pd.concat((train_data, test_x), axis=0, ignore_index=True)
train_data.shape[0]  ,  test_x.shape[0]


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
destination_percentage = df_all.country_destination.value_counts() / df_all.shape[0] * 100
destination_percentage.plot(kind='bar',color='#3498DB')
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
plt.title('Destination Country Percentage')
sns.despine()


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
df_all['age']=df_all['age'].apply(lambda x : 36 if x>100 else x)
sns.distplot(df_all.age.dropna(), color='#16A085')
plt.xlabel('Age')
plt.title('age density')
sns.despine()


plt.figure(figsize=(9,6))
count = df_all['gender'].fillna('NaN').value_counts(dropna=False)
c_order = count.index
sns.countplot(x=df_all['gender'].fillna('NaN'), order=c_order , color = '#D35400')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('gender count')
for i in range(count.shape[0]):
    plt.text(i, count[i]+1200, f"{count[i]/df_all.shape[0]*100:0.1f}%")
sns.despine()


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
device_percentage = df_all.first_device_type.value_counts() / df_all.shape[0] * 100
device_percentage.plot(kind='bar',color='#196F3D')
plt.xlabel('Device used by user')
plt.ylabel('Percentage')
plt.title('device user percentage')
sns.despine()


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
affiliate_provider_percentage = df_all.affiliate_provider.value_counts() / df_all.shape[0] * 100
affiliate_provider_percentage.plot(kind='bar',color='#CB4335')
plt.xlabel('affiliate providers')
plt.ylabel('Percentage')
plt.title('Percentage of users based on affiliate providers')
sns.despine()


df_all['first_browser'].value_counts().head(10).plot(kind='bar')


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 6))
sns.boxplot(x='country_destination', y='age', data=df_train, palette="muted", ax =ax)
ax.set_ylim([10, 75])


sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=0.5)
plt.figure(figsize=(12,6))

df_all[df_all['country_destination'] != 'NDF']['date_account_created'].value_counts().sort_index().plot(kind='line', linewidth=1, color='green')
df_all[df_all['country_destination'] == 'NDF']['date_account_created'].value_counts().sort_index().plot(kind='line', linewidth=1, color='red')


#converting from timestamp to date:
train_data['date_account_created'] = pd.to_datetime(train_data['date_account_created'])
train_data['timestamp_first_active'] = pd.to_datetime(train_data['timestamp_first_active'], format='%Y%m%d%H%M%S')

test_x['date_account_created'] = pd.to_datetime(test_x['date_account_created'])
test_x['timestamp_first_active'] = pd.to_datetime(test_x['timestamp_first_active'], format='%Y%m%d%H%M%S')


#Finding the number of null entries in each column.
for col in train_data.columns:
    num_null_values = train_data[col].isnull().sum()
    if num_null_values != 0:
        print(col + " has {} null values.".format(num_null_values))
        print()


#Finding the number of null entries in each column.
for col in test_x.columns:
    num_null_values = test_x[col].isnull().sum()
    if num_null_values != 0:
        print(col + " has {} null values.".format(num_null_values))
        print()


train_data['month_first_book'] = pd.DatetimeIndex(train_data['date_first_booking']).month
train_data['quarter_first_book']= pd.DatetimeIndex(train_data['date_first_booking']).quarter
train_data['DayOfWeek_first_book']= pd.DatetimeIndex(train_data['date_first_booking']).weekday

train_data['month_account_created'] = pd.DatetimeIndex(train_data['date_account_created']).month
train_data['quarter_account_created']= pd.DatetimeIndex(train_data['date_account_created']).quarter
train_data['DayOfWeek_account_created']= pd.DatetimeIndex(train_data['date_account_created']).weekday


test_x['month_first_book'] = pd.DatetimeIndex(test_x['date_first_booking']).month
test_x['quarter_first_book']= pd.DatetimeIndex(test_x['date_first_booking']).quarter
test_x['DayOfWeek_first_book']= pd.DatetimeIndex(test_x['date_first_booking']).weekday

test_x['month_account_created'] = pd.DatetimeIndex(test_x['date_account_created']).month
test_x['quarter_account_created']= pd.DatetimeIndex(test_x['date_account_created']).quarter
test_x['DayOfWeek_account_created']= pd.DatetimeIndex(test_x['date_account_created']).weekday


def weekend(weekday):
    if weekday == 5 or weekday == 6:
        return True
    else:
        return False


from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays()

train_data['Holiday_first_book'] = train_data['date_first_booking'].isin(holidays)
train_data['Holiday_account_created'] = train_data['date_account_created'].isin(holidays)

# the column Weekend will contain true for (saturdays and sundays), and will return false otherwise
#because we think that the electricity consumption may be affected by the weekends
train_data['Weekend_first_book'] = train_data['DayOfWeek_first_book'].map(weekend)
train_data['Weekend_account_created'] = train_data['DayOfWeek_account_created'].map(weekend)

test_x['Holiday_first_book'] = test_x['date_first_booking'].isin(holidays)
test_x['Holiday_account_created'] = test_x['date_account_created'].isin(holidays)

test_x['Weekend_first_book'] = test_x['DayOfWeek_first_book'].map(weekend)
test_x['Weekend_account_created'] = test_x['DayOfWeek_account_created'].map(weekend)


train_data['Holiday_first_book'].sum()
#since there is no variance over this column,we can drop it


train_data.drop(columns=['Holiday_first_book'], inplace=True)
test_x.drop(columns=['Holiday_first_book'], inplace=True)


train_data.drop(columns=['date_account_created', 'timestamp_first_active', 'date_first_booking'], inplace= True)

test_x.drop(columns=['date_account_created', 'timestamp_first_active', 'date_first_booking'], inplace= True)
test_x


def calculate_category_counts(df, column, threshold=99.1):
    counts_df = df[[column, 'id']].groupby(column).count().sort_values(by='id', ascending=False)
    counts_df = counts_df.rename(columns={'id': f'{column}_counts'})
    counts_df['Percentage_Contribution'] = counts_df[f'{column}_counts'] / len(df)
    counts_df['accumulative_perc'] = 100 * (counts_df[f'{column}_counts'].cumsum() / counts_df[f'{column}_counts'].sum())
    counts_df = counts_df.reset_index()
    frequent_categories = counts_df[counts_df['accumulative_perc'] <= threshold][column].tolist()
    return counts_df, frequent_categories

signup_flow_cats_Counts, signup_flow_freq_cats = calculate_category_counts(train_data, 'signup_flow')
language_cats_Counts, language_freq_cats = calculate_category_counts(train_data, 'language')
affiliate_provider_cats_Counts, affiliate_provider_freq_cats = calculate_category_counts(train_data, 'affiliate_provider')
first_browser_cats_Counts, first_browser_freq_cats = calculate_category_counts(train_data, 'first_browser')



def category_merger(category, frequent_categories):
    return category if category in frequent_categories else 'other'

signup_flow_Cats_merger = lambda category: category_merger(category, signup_flow_freq_cats)
language_Cats_merger = lambda category: category_merger(category, language_freq_cats)
affiliate_provider_Cats_merger = lambda category: category_merger(category, affiliate_provider_freq_cats)
first_browser_Cats_merger = lambda category: category_merger(category, first_browser_freq_cats)


columns_to_map = ['signup_flow', 'language', 'affiliate_provider', 'first_browser']
mappers = [signup_flow_Cats_merger, language_Cats_merger, affiliate_provider_Cats_merger, first_browser_Cats_merger]

for col, mapper in zip(columns_to_map, mappers):
    train_data[col] = train_data[col].map(mapper)
    test_x[col] = test_x[col].map(mapper)


cat_list= ['gender','signup_method','signup_flow','language','affiliate_channel',
           'affiliate_provider','first_affiliate_tracked','signup_app','first_device_type',
           'first_browser','month_first_book','quarter_first_book',
           'DayOfWeek_first_book','month_account_created','quarter_account_created',
           'DayOfWeek_account_created','Holiday_account_created','Weekend_first_book','Weekend_account_created', 'country_destination']
len(cat_list)


#transofrming the age on a logarithmic scale
train_data.age=np.log(train_data.age+1)
test_x.age=np.log(test_x.age+1)
test_x


train_data.drop(columns= ['id'], inplace=True)
test_x_id= test_x.id
test_x.drop(columns= ['id'], inplace=True)


train_data.isna().sum()


train_data['age']= train_data['age'].fillna(train_data['age'].mean())
test_x['age']= test_x['age'].fillna(train_data['age'].mean())


train_data['first_affiliate_tracked']= train_data['first_affiliate_tracked'].fillna('unk')
test_x['first_affiliate_tracked']= test_x['first_affiliate_tracked'].fillna('unk')


# filling the nan values by the mode:
train_data['month_first_book']= train_data['month_first_book'].fillna(train_data['month_first_book'].mode()[0])
train_data['quarter_first_book']= train_data['quarter_first_book'].fillna(train_data['quarter_first_book'].mode()[0])
train_data['DayOfWeek_first_book']= train_data['DayOfWeek_first_book'].fillna(train_data['DayOfWeek_first_book'].mode()[0])

test_x['month_first_book']= test_x['month_first_book'].fillna(train_data['month_first_book'].mode()[0])
test_x['quarter_first_book']= test_x['quarter_first_book'].fillna(train_data['quarter_first_book'].mode()[0])
test_x['DayOfWeek_first_book']= test_x['DayOfWeek_first_book'].fillna(train_data['DayOfWeek_first_book'].mode()[0])


train_data.isna().sum()
test_x.isna().sum()


cat_train_data= train_data[['gender', 'signup_method', 'signup_flow', 'language','affiliate_channel',
            'affiliate_provider', 'first_affiliate_tracked',
            'signup_app', 'first_device_type', 'first_browser',
            'Holiday_account_created', 'Weekend_first_book', 'Weekend_account_created']]
cat_test_x= test_x[['gender', 'signup_method', 'signup_flow', 'language','affiliate_channel',
            'affiliate_provider', 'first_affiliate_tracked',
            'signup_app', 'first_device_type', 'first_browser',
            'Holiday_account_created', 'Weekend_first_book', 'Weekend_account_created']]


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# ... (your existing code to create cat_train_data and cat_test_x) ...

# Convert all columns in cat_train_data and cat_test_x to string type
cat_train_data = cat_train_data.astype(str)
cat_test_x = cat_test_x.astype(str)

# Now create and apply the OneHotEncoder
cat_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')  # handle_unknown='ignore' for safety

cat_train_data_encoded = cat_encoder.fit_transform(cat_train_data)
cat_one_hot_df = pd.DataFrame(cat_train_data_encoded.toarray())  # Convert to dense array for DataFrame

cat_test_x_encoded = cat_encoder.transform(cat_test_x)  # Use transform, not fit_transform for test data
cat_test_x_one_hot_df = pd.DataFrame(cat_test_x_encoded.toarray())  # Convert to dense array for DataFrame



cyclic_cols_df= train_data[['month_first_book', 'quarter_first_book', 'DayOfWeek_first_book',
                        'month_account_created', 'quarter_account_created', 'DayOfWeek_account_created']]
T_cyclic_cols_df= test_x[['month_first_book', 'quarter_first_book', 'DayOfWeek_first_book',
                        'month_account_created', 'quarter_account_created', 'DayOfWeek_account_created']]


import math
import numpy as np

def transform_to_cyclic_cosine(df, column):
    df[column] = 2 * math.pi * df[column] / df[column].max()
    df[column] = np.cos(df[column])

columns_to_transform = [
    'month_first_book', 'quarter_first_book', 'DayOfWeek_first_book',
    'month_account_created', 'quarter_account_created', 'DayOfWeek_account_created'
]

for col in columns_to_transform:
    transform_to_cyclic_cosine(cyclic_cols_df, col)


import numpy as np
import math

def cyclic_transform(column):
    return np.cos(2 * math.pi * column / column.max())

cols_to_transform = [
    'month_first_book', 'quarter_first_book', 'DayOfWeek_first_book',
    'month_account_created', 'quarter_account_created', 'DayOfWeek_account_created'
]

T_cyclic_cols_df[cols_to_transform] = T_cyclic_cols_df[cols_to_transform].apply(cyclic_transform)



#Standardizing the age column:
from sklearn.preprocessing import StandardScaler
st_cols= train_data[['age']]
st_cols= pd.DataFrame(StandardScaler().fit_transform(st_cols), columns=['age'] )

T_st_cols= test_x[['age']]
T_st_cols= pd.DataFrame(StandardScaler().fit_transform(T_st_cols), columns=['age'] )
T_st_cols


# st_cols, cyclic_cols_df, cat_one_hot_df
num_df= pd.merge(st_cols,cyclic_cols_df, right_index= True, left_index=True )
preprocessed_data= pd.merge(num_df, cat_one_hot_df,right_index= True, left_index=True )

T_num_df= pd.merge(T_st_cols,T_cyclic_cols_df, right_index= True, left_index=True)
T_preprocessed_data= pd.merge(T_num_df,cat_test_x_one_hot_df,right_index= True, left_index=True)
T_preprocessed_data


y= train_data['country_destination']


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y= le.transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, y, test_size=0.25, random_state=42)


from xgboost import XGBClassifier, plot_importance
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_train, y_train)


y_pred = xgb.predict_proba(X_test)


plot_importance(xgb, max_num_features=10)
f_importances= xgb.feature_importances_


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(range(12))


y_test_enc = lb.transform(y_test)


from sklearn import metrics
model_1_score= metrics.ndcg_score(y_true= y_test_enc,
                           y_score= y_pred,
                           k=5, sample_weight=None, ignore_ties=False)
model_1_score


test_predictions= xgb.predict_proba(T_preprocessed_data)


#xgb.get_score(fmap='', importance_type='weight')
#xgb.get_fscore()
feature_important = xgb.get_booster().get_score(importance_type='weight')
f_importance= pd.DataFrame.from_dict(feature_important, orient='index')
f_importance= f_importance.reset_index()
f_importance= f_importance.rename(columns={'index': 'feature', 0: 'f_score'})
f_importance.sort_values(by= 'f_score', ascending=False).head(40)


imp_feature_ls= f_importance.sort_values(by= 'f_score', ascending=False).head(50)['feature'].tolist()


X_train= X_train.rename(columns= {     'age':'age', 'month_first_book':'month_first_book',
             'quarter_first_book': 'quarter_first_book',  'DayOfWeek_first_book':'DayOfWeek_first_book',
           'month_account_created': 'month_account_created',   'quarter_account_created':'quarter_account_created', 'DayOfWeek_account_created':
       'DayOfWeek_account_created', 0 : '0',1 : '1', 2 : '2',3 : '3',4 : '4',5 : '5',6 : '6',7 : '7',8 : '8',9 : '9', 10 :'10',11 :'11',
                                  12 :'12',13 :'13',14 :'14',15 :'15',16 :'16',17 :'17',18 :'18',19 :'19',20 :'20',21 :'21',
                                  22 :'22',23 :'23',24 :'24',25 :'25',26 :'26',27 :'27', 28 :'28',29 :'29',30 :'30',31 :'31',32 :'32',33 :'33',
                                  34 :'34',35 :'35',36 :'36',37 :'37',38 :'38',39 :'39',40 :'40',41 :'41', 42 :'42',43 :'43',44 :'44',45 :'45',
                                  46 :'46',47 :'47',48 :'48',49 :'49',50 :'50',51 :'51',52 :'52',53 :'53',54 :'54',55 :'55',56 :'56',57 :'57',
                                  58 :'58',59 :'59',60 :'60',61 :'61',62 :'62',63 :'63',  64 :'64',65 :'65',66 :'66',67 :'67',68 :'68',69 :'69',70 :'70'})


preprocessed_data= preprocessed_data.rename(columns= { 'age':'age', 'month_first_book':'month_first_book',
             'quarter_first_book': 'quarter_first_book',  'DayOfWeek_first_book':'DayOfWeek_first_book',
           'month_account_created': 'month_account_created',   'quarter_account_created':'quarter_account_created', 'DayOfWeek_account_created':
       'DayOfWeek_account_created', 0 : '0',1 : '1', 2 : '2',3 : '3',4 : '4',5 : '5',6 : '6',7 : '7',8 : '8',9 : '9', 10 :'10',11 :'11',
                                  12 :'12',13 :'13',14 :'14',15 :'15',16 :'16',17 :'17',18 :'18',19 :'19',20 :'20',21 :'21',
                                  22 :'22',23 :'23',24 :'24',25 :'25',26 :'26',27 :'27', 28 :'28',29 :'29',30 :'30',31 :'31',32 :'32',33 :'33',
                                  34 :'34',35 :'35',36 :'36',37 :'37',38 :'38',39 :'39',40 :'40',41 :'41', 42 :'42',43 :'43',44 :'44',45 :'45',
                                  46 :'46',47 :'47',48 :'48',49 :'49',50 :'50',51 :'51',52 :'52',53 :'53',54 :'54',55 :'55',56 :'56',57 :'57',
                                  58 :'58',59 :'59',60 :'60',61 :'61',62 :'62',63 :'63',  64 :'64',65 :'65',66 :'66',67 :'67',68 :'68',69 :'69',70 :'70'})


Selected_features_data= preprocessed_data[imp_feature_ls]


X_train_, X_test_, y_train_, y_test_ = train_test_split(Selected_features_data, y, test_size=0.25, random_state=42)
xgb.fit(X_train_, y_train_)


y_pred_ = xgb.predict_proba(X_test_)


y_test_enc_ = lb.transform(y_test_)


model_2_score= metrics.ndcg_score(y_true= y_test_enc_,
                           y_score= y_pred_,
                           k=5, sample_weight=None, ignore_ties=False)
model_2_score

