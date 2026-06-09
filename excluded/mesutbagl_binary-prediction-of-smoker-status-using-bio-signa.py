


import numpy as np # linear algebra
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_df=pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')


train_df.head()


test_df.head()


train_df.info()


train_df.shape


test_df.shape


train_df.describe()


train_df.corr()


train_df=train_df.drop('id', axis=1)
test_id=test_df['id']
test_df=test_df.drop('id', axis=1)


train_df.isnull().sum()


train_df['BMI'] = train_df['weight(kg)'] / (train_df['height(cm)'] / 100) ** 2


test_df['BMI'] = test_df['weight(kg)'] / (test_df['height(cm)'] / 100) ** 2


train_df['high_blood_pressure'] = (train_df['systolic'] > 130).astype(int)


test_df['high_blood_pressure'] = (test_df['systolic'] > 130).astype(int)


train_df['cholesterol_ratio'] = train_df['HDL'] / (train_df['LDL'] + 1e-5) 


test_df['cholesterol_ratio'] = test_df['HDL'] / (test_df['LDL'] + 1e-5)


train_df['poor_eyesight'] = ((train_df['eyesight(left)'] < 0.5) | (train_df['eyesight(right)'] < 0.5)).astype(int)
train_df['poor_hearing'] = ((train_df['hearing(left)'] < 30) | (train_df['hearing(right)'] < 30)).astype(int)


test_df['poor_eyesight'] = ((test_df['eyesight(left)'] < 0.5) | (test_df['eyesight(right)'] < 0.5)).astype(int)
test_df['poor_hearing'] = ((test_df['hearing(left)'] < 30) | (test_df['hearing(right)'] < 30)).astype(int)


train_df['log_triglyceride'] = np.log1p(train_df['triglyceride'])
train_df['log_creatinine'] = np.log1p(train_df['serum creatinine'])


test_df['log_triglyceride'] = np.log1p(test_df['triglyceride'])
test_df['log_creatinine'] = np.log1p(test_df['serum creatinine'])


X= train_df.drop('smoking', axis=1)
y=train_df['smoking']


scaler=StandardScaler()
X=scaler.fit_transform(X)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


model=RandomForestClassifier()
model.fit(X,y)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Save the models
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models saved successfully!")


final_pred=model.predict(test_df)


final_pred


# Prepare submission file
submission_df = pd.DataFrame({
    'id': test_id,  # 'id' from the test set index
    'num_sold': final_pred  # Predictions
})


submission_df


# Save the submission file
submission_df.to_csv('submission.csv', index=False)




