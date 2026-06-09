import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')  
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')  
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv') 


train_df 


train_extra_df


test_df


new_train_df = pd.concat([train_df, train_extra_df], ignore_index=True)

new_train_df


new_train_df.info()


test_df.info()


new_train_df.isnull().mean()*100


test_df.isnull().mean()*100


num_cols = test_df.select_dtypes(include=['number']).columns

imputation_value = new_train_df[num_cols].median()

new_train_df[num_cols] = new_train_df[num_cols].fillna(imputation_value)
test_df[num_cols] = test_df[num_cols].fillna(imputation_value)


print(new_train_df.isnull().sum())
print(test_df.isnull().sum())


obj_cols = new_train_df.select_dtypes(include=['object']).columns

new_train_df[obj_cols] = new_train_df[obj_cols].fillna('None')
test_df[obj_cols] = test_df[obj_cols].fillna('None')


print("Missing Values and Data Types for Train Dataset")

display(new_train_df.dtypes, new_train_df.isnull().sum())


test_df.isnull().sum()


new_train_df.shape


test_df.shape


new_train_df.duplicated().sum()


test_df.duplicated().sum()


new_train_df.describe()


new_train_df['Brand'].value_counts().plot(kind='bar')


plt.subplot(1, 3, 2)
sns.boxplot(x=new_train_df['Price'])
plt.title('Price Boxplot')


# KDE Plot for Price
plt.figure(figsize=(10, 6))
sns.kdeplot(new_train_df['Price'], shade=True, color='orange')
plt.title('Price KDE')

plt.tight_layout()
plt.show()


categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

plt.figure(figsize=(14, 10))

for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 4, i)
    sns.countplot(data=new_train_df, x=col, palette='Set2')
    plt.title(f'{col} Countplot')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 10))

for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(data=new_train_df, x=col, y='Price', palette='Set2')
    plt.title(f'Price vs {col}')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


X = new_train_df.drop(columns=['id','Price'])  
y = new_train_df['Price'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand', 'Material','Laptop Compartment', 'Waterproof', 'Style', 'Color']
ordinal_features = ['Size']


transformer = ColumnTransformer(transformers=[
    ('num',StandardScaler(), numerical_features),
    ('cat',OneHotEncoder(sparse_output=False,drop='first',handle_unknown='ignore'),categorical_features),
    ('ord', OrdinalEncoder(categories=[['Small', 'Medium', 'Large']],handle_unknown="use_encoded_value", unknown_value=-1), ordinal_features),
],remainder='passthrough')


model = Pipeline([
    ('transformer', transformer),
    ('regressor', LinearRegression())
])


X_train.replace("None", np.nan, inplace=True)
X_test.replace("None", np.nan, inplace=True)


X_train = X_train.fillna(X_train.mode().iloc[0])
X_test = X_test.fillna(X_train.mode().iloc[0])  # Fill with training mode


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error


rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression RMSE: {rmse}")


X_test_kaggle = test_df.drop(columns=['id'])


test_predictions = model.predict(X_test_kaggle)


submission_df = pd.DataFrame({'id': test_df['id'], 'Price': test_predictions})
submission_df.to_csv('submission.csv', index=False)


submission_df = pd.read_csv("submission.csv")
print(submission_df)

