import os
os.chdir('/content')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import pandas as pd


df = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')


df.head()


df.describe(include='all')


df['brand'].nunique() #Holycow that's a lot


cat_columns = df.select_dtypes(include=['object']).columns
num_columns = df.select_dtypes(include=['int64', 'float64']).columns


for col in cat_columns:
  print(f'{col}: {df[col].nunique()} unique values')


df[num_columns].corr()


sns.heatmap(df[num_columns].corr(),cmap='coolwarm',annot=True)


for col in num_columns:
  plt.figure()
  sns.histplot(df[col])


df['price'].describe()


df.columns


sns.pairplot(df)


sns.scatterplot(x='milage',y='price',data=df)


df[df['price']>400000]


df.columns


X = df[['brand','model_year','milage','accident','clean_title']]


X.head()


X = pd.get_dummies(X,columns=['brand','model_year','accident','clean_title'],drop_first=True)


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# X['brand'] = le.fit_transform(X['brand'])


# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# X = ohe.fit_transform(X)


X.head()


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, df['price'], test_size=0.3, random_state=42)


from sklearn.linear_model import LinearRegression


lr = LinearRegression()


lr.fit(X_train,y_train)


pred = lr.predict(X_test)


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score


print(np.sqrt(mean_squared_error(y_test,pred)))
print(mean_absolute_error(y_test,pred))
print(r2_score(y_test,pred))


df.isna().sum()


#For cat columns, fill median. For cont columns, fill mean
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['accident'] = df['accident'].fillna(df['accident'].mode()[0])
df['clean_title'] = df['clean_title'].fillna(df['clean_title'].mode()[0])


from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


X = df[['brand','model_year','milage','accident','clean_title']]
y = df['price']


X.isna().sum()


# num_columns = X.select_dtypes(include=['int64', 'float64']).columns #For now taking year as continuous
# cat_columns = X.select_dtypes(include=['object']).columns


numerical_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')


# #Creating a custom function for pd.get_dummies
# def get_dummies(df,columns):
#   for col in columns:
#     df = pd.get_dummies(df,columns=[col],drop_first=True)
#   return df


# custom_cat_transformer = FunctionTransformer(get_dummies,validate=False)


#Doing this manually is better as brand will taken as continuous
cat_columns=['brand','model_year','accident','clean_title']
num_columns=['milage']



# preprocessor = ColumnTransformer(
#     transformers = [
#         ("num",numerical_transformer,num_columns),
#         ("cat",custom_cat_transformer,cat_columns)
#     ]
# )


# pipeline = Pipeline(
#     steps = [('transform',preprocessor),
#              ('model',LinearRegression())]) #Steps should be a LIST of TUPLES


encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)


X_encoded = pd.get_dummies(X,columns=cat_columns,drop_first=True) #Use columns option. Makes it very easier


scaler = StandardScaler()


X_encoded['milage'] = scaler.fit_transform(X_encoded[['milage']])


X_encoded.head()


y = np.log(df['price'])


plt.hist(y)


plt.hist(df['price'])



X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)


model = LinearRegression()


model.fit(X_train,y_train)


pred = model.predict(X_test)


r2_score(y_test,pred)


model


test_df = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')


# Fill missing values in the test set using the same strategy as the training set
test_df['fuel_type'] = test_df['fuel_type'].fillna(test_df['fuel_type'].mode()[0])
test_df['accident'] = test_df['accident'].fillna(test_df['accident'].mode()[0])
test_df['clean_title'] = test_df['clean_title'].fillna(test_df['clean_title'].mode()[0])


test_df.head()


# Prepare the test data using the same transformations
X_test_prep = test_df[['brand','model_year','milage','accident','clean_title']]



X_test_encoded = pd.get_dummies(X_test_prep, columns=cat_columns, drop_first=True)


X_test_encoded['milage'] = scaler.transform(X_test_encoded[['milage']])


X_test_encoded.columns == X_encoded.columns


X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)



y_pred = model.predict(X_test_encoded)


# Inverse transform the log-transformed predictions
y_pred = np.exp(y_pred)


# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': test_df['id'], 'price': y_pred})
os.chdir('/kaggle/working/')
# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)




