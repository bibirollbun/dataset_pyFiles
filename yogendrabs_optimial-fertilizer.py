import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


train="Data-Set/train.csv"
test="Data-Set/test.csv"


df_train=pd.read_csv(train)
df_train.head()


df_train['Fertilizer Name'].unique()


# We have to convert Fertilizer Name into numeric digits to predict, so we use label encoder
le=LabelEncoder()
df_train['Fertilizer Name']=le.fit_transform(df_train['Fertilizer Name'])


# print(le.classes_)
df_train['Fertilizer Name'].unique()


df_train['Fertilizer Name'].value_counts(normalize=False)


df_train.info()


df_train.isna().sum()


df_train.duplicated().sum()


df_train.head()


plt.scatter(df_train['Temparature'],df_train['Fertilizer Name'])
plt.show()


plt.scatter(df_train['Fertilizer Name'],df_train['Crop Type'])
plt.show()


plt.scatter(df_train['Fertilizer Name'],df_train['Potassium'])
plt.show()


print(df['Soil Type'].unique())
print(df['Crop Type'].unique())


def encoding(df):
    le=LabelEncoder()
    df['Soil Type']=le.fit_transform(df['Soil Type'])
    df['Crop Type']=le.fit_transform(df['Crop Type'])
    return df


df_train=encoding(df_train)


X=df_train.drop(['id','Fertilizer Name'],axis=1)
y=df_train['Fertilizer Name']


x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.20)


rf=RandomForestClassifier()
rf.fit(x_train,y_train)


y_pred_rf=rf.predict(x_test)
print(accuracy_score(y_test,y_pred_rf))


importances = rf.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()


df=pd.read_csv(train)


df = df.copy()
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for col in num_cols:
    df[col] = df[col].astype('float32')


cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for col in cat_cols:
    df[col] = df[col].astype('category')


df.drop(columns=['id'], inplace=True)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


df['N/P'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
df['N+K'] = df['Nitrogen'] + df['Potassium']


X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import lightgbm as lgb


model = lgb.LGBMClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


test_df=pd.read_csv(test)
test_df.head()


# Save the ID column for final submission
ids = test_df['id']

test_df = test_df.drop(columns=['id'])

num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in num_cols:
    test_df[col] = test_df[col].astype('float32')

# Convert categoricals to category (same as training)
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    test_df[col] = test_df[col].astype('category')

# Feature Engineering (must match training)
test_df['N/P'] = test_df['Nitrogen'] / (test_df['Phosphorous'] + 1)
test_df['N+K'] = test_df['Nitrogen'] + test_df['Potassium']


# Use the trained model to predict
preds = model.predict(test_df)


# Create submission DataFrame
submission = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


