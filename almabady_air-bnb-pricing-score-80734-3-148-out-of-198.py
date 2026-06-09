import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/spring-2025-regression-competition/dataSP25.csv')
test = pd.read_csv('/kaggle/input/spring-2025-regression-competition/compSP25.csv')


train.head(5)


train['price'].max()


train.info()


train.isna().sum().sort_values(ascending = False).head(5)


train.describe()


train.duplicated().sum()


train['host_id'].value_counts()


vis_num = ['price','minimum_nights', 'number_of_reviews','reviews_per_month'
           , 'calculated_host_listings_count','availability_365']


sns.histplot(train['price'],kde=True)
plt.xlim(0, 1000)
plt.show()


(train['price'] > 1000 ).sum()


train[train['price'] > 4000]


train['minimum_nights'].max()


sns.histplot(train['minimum_nights'], bins=50)
plt.xlim(0, 365)
plt.title('Distribution of Minimum Nights')
plt.show()


(train['minimum_nights'] > 365).sum()


train[train['minimum_nights'] > 365]['price'].max()


condition = (train['minimum_nights'] > 356)


train = train.drop(train[condition].index)


sns.histplot(train['number_of_reviews'],kde=True)
plt.xlim(0, 610)
plt.title('Distribution of Number Of Reviews')
plt.show()


sns.histplot(train['reviews_per_month'],kde=True)
plt.xlim(0, 60)
plt.title('Distribution of Reviews Per Month')
plt.show()


sns.histplot(train['calculated_host_listings_count'],kde=True)
plt.xlim(0, 350)
plt.title('Distribution of Calculated Host Listings Count')
plt.show()


sns.histplot(train['availability_365'],kde=True)
plt.xlim(0, 370)
plt.title('Distribution of Availability 365')
plt.show()


sns.histplot(train['calculated_host_listings_count'],kde=True)
plt.xlim(0, 350)
plt.title('Distribution of Calculated Host listings count')
plt.show()



sns.countplot(x = train['neighbourhood_group'])
plt.title('Distribution of Neighbourhood Group')
plt.show()


sns.boxplot(data = train , x = 'neighbourhood_group', y = 'price')


for city in train['neighbourhood_group'].unique() :
    dt = train[train['neighbourhood_group'] == city]
    plt.figure(figsize=(64,32))
    sns.countplot(x = dt['neighbourhood'])
    plt.title(f'Distribution of {city}')
    plt.show()


sns.countplot(x = train['room_type'])
plt.title('Distribution of Room Type')
plt.show()


sns.boxplot(data = train , x = 'room_type', y  = 'price' )
plt.title('Distribution of Room Type')
plt.show()


train.isna().sum().sort_values(ascending = False).head(5)


filtered_df = train[(train['number_of_reviews'] == 0) & (train['last_review'].isnull())]
len(filtered_df)


train['last_review'] = pd.to_datetime(train['last_review'])
train['last_review']


train['last_review'].min() , train['last_review'].max()


train['last_review'] = train['last_review'].fillna(pd.Timestamp('2009-01-01'))


reference_date = pd.Timestamp('2020-01-01')


train['days_since_review'] = (reference_date - train['last_review']).dt.days


train['reviews_per_month'] = train['reviews_per_month'].fillna(0)


train['name'] = train['name'].fillna('N')


test.isnull().sum().sort_values(ascending = False).head(5)


test['last_review'] = pd.to_datetime(test['last_review'])
test['last_review']


test['last_review'].min() , test['last_review'].max()


test['last_review'] = test['last_review'].fillna(pd.Timestamp('2009-01-01'))


reference_date = pd.Timestamp('2020-01-01')


test['days_since_review'] = (reference_date - test['last_review']).dt.days


test['reviews_per_month'] = test['reviews_per_month'].fillna(0)


#creating a dict of each neighbourhood_group and each unique neighbourhood within it
mapo = {}
for city in train['neighbourhood_group'].unique() :
    n = train[train['neighbourhood_group'] == city]['neighbourhood'].unique()
    mapo[city] = n


# printing it so we can search and find longitude and latitude for each
for city in train['neighbourhood_group'].unique() :
    print(f'group : {city}')
    print(f'neighbourhoods : {mapo[city]}')
    print('-'*70)


import sys
sys.path.append('/kaggle/input/lacoations-dict')


from coordinates import nyc_coords


import pandas as pd
import numpy as np

# Haversine formula to calculate the distance between two points on the Earth's surface
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # Corrected to np.arctan2
    
    distance = R * c  # Resulting distance in kilometers
    return distance

# Define a function to fetch the neighborhood's coordinates from nyc_coords dictionary
def get_neighbourhood_coords(neighbourhood_group, neighbourhood):
    # Check if the neighbourhood_group exists in the nyc_coords dictionary
    if neighbourhood_group in nyc_coords:
        # If the group exists, check if the neighbourhood exists within that group
        if neighbourhood in nyc_coords[neighbourhood_group]:
            return nyc_coords[neighbourhood_group][neighbourhood]
    return None

# Create a new column 'distance_to_neighbourhood_center' for the distance
def calculate_distance(row):
    neighbourhood_group = row['neighbourhood_group']
    neighbourhood = row['neighbourhood']
    lat, lon = row['latitude'], row['longitude']
    
    # Get the coordinates for the specific neighbourhood
    neighbourhood_coords = get_neighbourhood_coords(neighbourhood_group, neighbourhood)
    if neighbourhood_coords:
        neighbourhood_lat, neighbourhood_lon = neighbourhood_coords
        return haversine(lat, lon, neighbourhood_lat, neighbourhood_lon)
    else:
        return None  # If no coordinates are found, return None



# Apply the distance calculation to each row and create the new 'distance_to_neighbourhood_center' column
train['distance_to_center'] = train.apply(calculate_distance, axis=1)


plt.figure(figsize=(10,8))
sns.scatterplot(data = train, x = 'distance_to_center', y = 'price' , hue = 'room_type',alpha=0.4)
plt.show()


train['name_len'] = train['name'].str.len()


plt.figure(figsize=(10,8))
sns.scatterplot(data = train, x = 'name_len', y = 'price' , hue = 'room_type',alpha=0.4)
plt.ylim(0,4000)
plt.show()


# Apply the distance calculation to each row and create the new 'distance_to_neighbourhood_center' column
test['distance_to_center'] = test.apply(calculate_distance, axis=1)


test['name_len'] = test['name'].str.len()


test.isna().sum().sort_values(ascending=False).head(4)


test['distance_to_center'] = test['distance_to_center'].fillna(test['distance_to_center'].mean())


train['price'].median() , train['price'].mean()


len(train[(train['price'] > 500) & (train['number_of_reviews'] < 1) & (train['minimum_nights'] > 20)])


condition = (train['price'] > 500) & (train['number_of_reviews'] < 1) & (train['minimum_nights'] > 20)


train = train.drop(train[condition].index)


condition_2 = ( train['price'] == 0 ) 


train = train.drop(train[condition_2].index)


train['price'].median() , train['price'].mean()


train.columns.to_list()


col_to_drop = ['id','name','host_id','host_name','calculated_host_listings_count',
             'last_review','latitude','longitude',]


train.drop(columns=col_to_drop, axis = 1, inplace = True)


test.columns.to_list()


test_id = test['id']


test.drop(columns=col_to_drop, axis = 1, inplace = True)


train['neighbourhood_group'] = train['neighbourhood_group'].str.replace(' ', '_')
train['neighbourhood'] = train['neighbourhood'].str.replace(' ', '_')


test['neighbourhood_group'] = test['neighbourhood_group'].str.replace(' ', '_')
test['neighbourhood'] = test['neighbourhood'].str.replace(' ', '_')


from sklearn.model_selection import train_test_split


y = train['price']
X = train.drop('price',axis = 1)


X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.01,random_state=23)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


num_col = X_train.select_dtypes(include=['number']).columns.to_list()
one_col = ['neighbourhood_group','neighbourhood','room_type']
num_col


#Define a preprocessor without 'passthrough'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_col),
        ('onehot', OneHotEncoder(sparse=False), one_col)
    ]
)


def transformer(df, preprocessor):
    # Apply the preprocessor (fit and transform)
    temp = preprocessor.fit_transform(df)
    
    # Extracting the feature names generated by OneHotEncoder
    onehot_transformer = preprocessor.named_transformers_['onehot']
    one_col_2 = onehot_transformer.get_feature_names_out(one_col).tolist()
    
    # Return the dataframe with correct column names
    col = num_col + one_col_2
    return pd.DataFrame(temp, columns=col)


X_train, X_valid = transformer(X_train,preprocessor), transformer(X_valid,preprocessor)


print(f'Number of X_train features : {len(X_train.columns)}')
print(f'Number of X_valid features : {len(X_valid.columns)}')


y_train, y_valid = np.log1p(y_train), np.log1p(y_valid)


#solving the problem of missing columns and features
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor()


# parameter for the model
param_dist = {
    'n_estimators': [128 ,256 ,512 ,1024],
    'max_depth': [8 ,16 ,28 ,32 ,64],
    'min_samples_split': [4 ,8 ,16 ,22 ,32],
    'min_samples_leaf': [2 ,4 ,8 ,16],
    'bootstrap': [True ,False]
}

#creating a random search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=75, cv=5, n_jobs=-1, verbose=2, random_state=23)


#Fit the model
#random_search.fit(X_train, y_train)


#Get the best parameters
#print(f"Best parameters found: {random_search.best_params_}")


params = {
    'n_estimators': 1024,
    'min_samples_split': 16,
    'min_samples_leaf': 8,
    'max_depth': 64,
    'bootstrap': True
}

# Create the model
rf_model = RandomForestRegressor(**params, random_state=23)


rf_model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error


y_pred = rf_model.predict(X_valid)


# Calculate RMSE
rmse_log = np.sqrt(mean_squared_error(y_valid, y_pred))
print('rmse_log : ' ,rmse_log)


X = transformer(X,preprocessor)
y = np.log1p(y)


test = transformer(test,preprocessor)


#aligning to match columns
test_aligned = test.reindex(columns=X.columns, fill_value=0)


rf_model_2 = RandomForestRegressor(**params, random_state=23)


rf_model_2.fit(X, y)


y_test_rf = rf_model_2.predict(test_aligned)


y_test_rf = np.expm1(y_test_rf)


# submission from Random forest model
submission_1 = pd.DataFrame({
    'id' : test_id,
    'price' : y_test_rf
})


submission_1.to_csv('/kaggle/working/submission_1.csv',index=False)


from xgboost import XGBRegressor


model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=1024,         # higher with early stopping
    learning_rate=0.005,        # smaller LR for better generalization
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    random_state=42,
    n_jobs=-1,
    tree_method= 'gpu_hist'
)


model.fit(
    X_train, y_train,
    eval_set= [(X_valid, y_valid)],
    early_stopping_rounds= 350,
    verbose= 250
)


y_pred = model.predict(X_valid)
rmse_log = np.sqrt(mean_squared_error(y_valid, y_pred))
print('rmse_log : ' ,rmse_log)


y_test = model.predict(test_aligned)
y_test = np.expm1(y_test)


submission_2 = pd.DataFrame({
    'id' : test_id,
    'price' : y_test
})


submission_2.to_csv('/kaggle/working/submission_26.csv',index=False)




