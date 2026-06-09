import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



sub=pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
train=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


train.info()


test.info()


train.head()


test.head()


train.shape , test.shape


plt.figure(figsize=(10, 4))
sns.barplot(x='accident', y='price', data=train)
plt.title('Average Price by Accident History')
plt.xlabel('Accident History')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x=train['transmission'])
plt.title("Transmission Type Distribution")
plt.xlabel("Transmission Type")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x=train['accident'])
plt.title("Transmission Type Distribution")
plt.xlabel("Transmission Type")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(16,5))
sns.countplot(x=train['fuel_type'])
plt.title("count of Fuel Type")
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(train['milage'], bins=30)
plt.title("Distribution of Milage ")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(train['model_year'], bins=30)
plt.title("Distribution of Car Model Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(20,12))
sns.countplot(y=train['brand'])
plt.title("Car Brand Distribution")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


plt.figure(figsize=(15, 6))
top_models = train['model'].value_counts().nlargest(30)
sns.barplot(y=top_models.values, x=top_models.index, palette="viridis")
plt.title("Top 30 Car Models Distribution")
plt.xlabel("Count")
plt.ylabel("Model")
plt.xticks(rotation=90)
plt.show()



plt.figure(figsize=(12, 6))
top_engines = train['engine'].value_counts().nlargest(30)
sns.barplot(x=top_engines.index, y=top_engines.values)
plt.title("Top 30 Most Common Engine Types")
plt.xlabel("Engine")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()



def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR  
    out_data = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    df = df.drop(out_data.index).reset_index(drop=True)
    return df



plt.figure(figsize=(8, 5)) 
sns.boxplot(x=train['milage'])  
plt.title("Box Plot of Milage before remove outliers")
plt.xlabel("Milage")
plt.show()


plt.figure(figsize=(8, 5)) 
sns.boxplot(x=train['price'])  
plt.title("Box Plot of Price before remove outliers")
plt.xlabel("price")
plt.show()


plt.figure(figsize=(16, 6))
sns.boxplot(x='fuel_type', y='price', data=train, palette="bright")  
plt.title('Box Plot of Price by Fuel Type before remove outliers')
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()



train_with_no_outliers=remove_outliers(train,'milage')
train_with_no_outliers=remove_outliers(train_with_no_outliers , 'price')

train_with_no_outliers


plt.figure(figsize=(8, 5)) 
sns.boxplot(x=train_with_no_outliers['price'])  
plt.title("Box Plot of Milage")
plt.xlabel("price")
plt.show()


plt.figure(figsize=(16, 6))
sns.boxplot(x='fuel_type', y='price', data=train_with_no_outliers, palette="bright")  
plt.title('Box Plot of Price by Fuel Type')
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()



plt.figure(figsize=(8, 5)) 
sns.boxplot(x=train_with_no_outliers['milage'])  
plt.title("Box Plot of Milage")
plt.xlabel("Milage")
plt.show()


# Define luxury brands and clean fuels
luxury_brands = [
    "Mercedes-Benz", "Audi", "BMW", "Tesla", "Cadillac", "Genesis", 
    "Jaguar", "Lexus", "Porsche", "INFINITI", "Acura", "Maserati", 
    "Rolls-Royce", "Bentley", "Lamborghini", "Ferrari", "Lucid", 
    "Lotus", "Maybach", "Bugatti"
]

clean_fuels = ["Plug-In Hybrid", "Hybrid"]
df_train = train_with_no_outliers[['id']].copy()
df_test = test[['id']].copy()
df_train['is_luxury'] = train_with_no_outliers['brand'].isin(luxury_brands).astype(int)
df_test['is_luxury'] = test['brand'].isin(luxury_brands).astype(int)
df_train['is_clean_fuel'] = train_with_no_outliers['fuel_type'].isin(clean_fuels).astype(int)
df_test['is_clean_fuel'] = test['fuel_type'].isin(clean_fuels).astype(int)
df_train['is_automatic'] = (train_with_no_outliers['transmission'] == 'Automatic').astype(int)
df_test['is_automatic'] = (test['transmission'] == 'Automatic').astype(int)
df_train['milage_per_year'] = train_with_no_outliers['milage'] / (2024 - train_with_no_outliers['model_year'])
df_test['milage_per_year'] = test['milage'] / (2024 - test['model_year'])
df_train['car_age'] = 2024 - train_with_no_outliers['model_year']
df_test['car_age'] = 2024 - test['model_year']
df_train['is_color_matched'] = (train_with_no_outliers['ext_col'] == train_with_no_outliers['int_col']).astype(int)
df_test['is_color_matched'] = (test['ext_col'] == test['int_col']).astype(int)
df_train['rare_fuel_type'] = train_with_no_outliers['fuel_type'].apply(lambda x: 0 if x in ['Petrol', 'Diesel'] else 1)
df_test['rare_fuel_type'] = test['fuel_type'].apply(lambda x: 0 if x in ['Petrol', 'Diesel'] else 1)
#train_with_no_outliers['car_age'] = np.maximum(2024 - df['model_year'], 1)
#df['milage_per_year'] = df['milage'] / df['car_age']


df_train


df_test


train_with_no_outliers.drop('clean_title',axis=1,inplace=True)
test.drop('clean_title',axis=1,inplace=True)


# Identify categorical columns
cat_col = train_with_no_outliers.select_dtypes(include=['object']).columns
cat_col_test=test.select_dtypes(include=['object']).columns
# Apply Label Encoding on train data
for col in cat_col:
    le = LabelEncoder()
    train_with_no_outliers[col] = le.fit_transform(train_with_no_outliers[col])

#Apply Label Encoding on test data
for col in cat_col_test:
    le = LabelEncoder()
    test[col]=le.fit_transform(test[col])





def knn_impute(df, num_cols, n_neighbors=7):
    df_copy = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])
    return df_copy

df_train_imputed = knn_impute(train_with_no_outliers, ['fuel_type', 'accident'])
df_test_imputed = knn_impute(test, ['fuel_type', 'accident']) 


df_train_imputed.isna().sum()


df_test_imputed.isna().sum()


df_train_imputed = df_train_imputed.merge(df_train, on='id', how='inner')
df_test_imputed = df_test_imputed.merge(df_test, on='id', how='inner')
df_train_imputed


df_test_imputed


df_train_imputed.isna().sum()


df_test_imputed.isna().sum()


X = df_train_imputed.drop(columns=['price'])  
y = df_train_imputed['price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_params={
                'num_leaves': 426,
                 'max_depth': 15,
                 'learning_rate': 0.01,
                 'n_estimators': 7000,
                 'metric': 'rmse',
                 'reg_alpha': 1.48699088003429e-06,
                 'reg_lambda': 0.41539458543414265,
                 'verbose' : -1,
                 'early_stopping_rounds': 200,
}

lgbm_model = LGBMRegressor(**lgb_params)

lgbm_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   eval_metric='rmse'             
                  )

pred=lgbm_model.predict(df_test_imputed)
pred


sub['price']=pred


sub.to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')










