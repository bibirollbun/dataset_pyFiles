import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df_sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_sample.head()


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


df.shape


df_work = df.copy()


df_work.shape


df_work.head()


# df_test = pd.read_csv("")


df.head()


df.shape


df.describe()


df.columns


df["Fertilizer Name"].unique()


df['Soil Type'].unique()


df['Crop Type'].unique()


df_analysis = df.groupby('Fertilizer Name').size().reset_index()
df_analysis


df_urea  = df[df['Fertilizer Name'] == 'Urea']
df_urea


df_urea['Soil Type'].unique()


df_urea['Crop Type'].unique()


df_urea.describe()


df_dap  = df[df['Fertilizer Name'] == 'DAP']
df_dap


df_dap.describe()


df_dap['Soil Type'].unique()


df['Crop Type'].unique()


df_work.head()


encoder = OneHotEncoder(sparse=False)


encoded = encoder.fit_transform(df_work[['Crop Type', 'Soil Type']])


encoded_cols = encoder.get_feature_names_out(['Crop Type', 'Soil Type'])


df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_work.index)


df_work.shape


df_encoded.shape


df_final = pd.concat([df_work.drop(columns=['Crop Type', 'Soil Type']), df_encoded], axis=1)


df_final.shape


df_final.select_dtypes(include='number').columns


df_final.head()


df_final.shape


numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])


# x = df_final[['id', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
#        'Phosphorous', 'Crop Type_Barley', 'Crop Type_Cotton',
#        'Crop Type_Ground Nuts', 'Crop Type_Maize', 'Crop Type_Millets',
#        'Crop Type_Oil seeds', 'Crop Type_Paddy', 'Crop Type_Pulses',
#        'Crop Type_Sugarcane', 'Crop Type_Tobacco', 'Crop Type_Wheat','Soil Type_Black', 'Soil Type_Clayey', 'Soil Type_Loamy',
#        'Soil Type_Red', 'Soil Type_Sandy']]
# x


x = df_final.drop(columns=['Fertilizer Name'])
x


# df_work['Fertilizer Name'] = df_work['Fertilizer Name'].astype('category')


df_work.shape


label_encoder = LabelEncoder()


# y = df_work['Fertilizer Name'].values
# y = df_work['Fertilizer Name'].astype('category')
y = label_encoder.fit_transform(df_work['Fertilizer Name'])
y.shape


X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=99)


from sklearn.tree import DecisionTreeClassifier


clf = DecisionTreeClassifier(max_depth=25, min_samples_split=10)


clf.fit(X_train, y_train)


train_preds_clf = clf.predict(X_train)


y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy_train_clf = accuracy_score(y_train, train_preds_clf)
print(f'Accuracy: {accuracy}')
print('Train Accuracy', accuracy_train_clf)


# Accuracy: 0.15340444444444445
# Train Accuracy 0.4696190476190476


df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_test.head()


encoded_test = encoder.fit_transform(df_test[['Crop Type', 'Soil Type']])


encoded_cols_test = encoder.get_feature_names_out(['Crop Type', 'Soil Type'])


df_encoded_test = pd.DataFrame(encoded_test, columns=encoded_cols_test, index=df_test.index)


df_final_test = pd.concat([df_test.drop(columns=['Crop Type', 'Soil Type']), df_encoded_test], axis=1)


df_final_test.head()


y_pred_test_DTC = clf.predict(df_final_test)
y_pred_test_DTC


df_preds = pd.DataFrame({
    'id': df_final_test['id'],                     
    'Fertilizer Name': y_pred_test_DTC        
})

print(df_preds.head())


# submission_file = "submission.csv"
# df_preds.to_csv(submission_file, index=False)
# print(f"\nSubmission file '{submission_file}' created successfully.")


from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=10)


rf_classifier.fit(X_train, y_train)


y_pred_rf = rf_classifier.predict(x_test)


train_preds = rf_classifier.predict(X_train)


print(y_test)
print(y_pred_rf)


train_acc_rf = accuracy_score(y_train, train_preds)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy: {accuracy_rf}')
print('Train acc', train_acc_rf)


# Accuracy: 0.1697511111111111
# Train acc 0.2399904761904762


y_pred_test_RF = rf_classifier.predict(df_final_test)
y_pred_test_RF


real_preds_RF = label_encoder.inverse_transform(y_pred_test_RF)
real_preds_RF


df_preds_rf = pd.DataFrame({
    'id': df_final_test['id'],                     
    'Fertilizer Name': real_preds_RF        
})

print(df_preds_rf.head())


submission_file_rf = "submission.csv"
df_preds_rf.to_csv(submission_file_rf, index=False)
print(f"\nSubmission file '{submission_file_rf}' created successfully.")


import xgboost as xgb


y_train = y_train.astype('category')
y_test = y_test.astype('category')

y_train_encoded = y_train.cat.codes
y_test_encoded = y_test.cat.codes


y_train_encoded


xgb_train = xgb.DMatrix(X_train, y_train_encoded, enable_categorical=True)
xgb_test = xgb.DMatrix(x_test, y_test_encoded, enable_categorical=True)


params = {
    'objective': 'multi:softprob',  
    'num_class': 7,                
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'     
}
n=50
model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)


xbg_preds = model.predict(xgb_test)
xbg_preds = np.round(xbg_preds)


xgb_preds = model.predict(xgb_test) 
xgb_preds = np.argmax(xgb_preds, axis=1)


train_preds_xbg = model.predict(xgb_train)
train_xgb_preds = np.argmax(train_preds_xbg, axis=1)


train_acc_xbg = accuracy_score(y_train_encoded, train_xgb_preds)
accuracy_xbg = accuracy_score(y_test_encoded, xgb_preds)
print(f'Accuracy: {accuracy_xbg}')
print('Train acc', train_acc_xbg)







