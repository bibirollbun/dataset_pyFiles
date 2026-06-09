import numpy as np
import pandas as pd


test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


train.head(3)


test.head(3)


print('Size of Train :',train.shape)
print('Size of Test :',test.shape)


train.info()


test.info()


print("Calulating Null values from test and train")
print("Total Null Values on Train",train.isna().sum())
print("--------------------------")
print("Total Null Values on Test",test.isna().sum())


from sklearn.preprocessing import LabelEncoder,StandardScaler


train.columns


le=LabelEncoder()
cat_col=['Soil Type','Crop Type']
for i in cat_col:
    train[i]=le.fit_transform(train[i])

for i in cat_col:
    test[i]=le.fit_transform(test[i])


test['Soil Type']


from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
model = XGBClassifier(random_state=42)


features=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
       'Nitrogen', 'Potassium', 'Phosphorous']
X=train.drop(['Fertilizer Name'],axis=1)
y=train['Fertilizer Name']
sc=StandardScaler()
X_test = test[features]
X_scaled=sc.fit_transform(train[features])
X_test_scaled=sc.transform(test[features])
le_tar = LabelEncoder() 
y_enc = le_tar.fit_transform(y)


model.fit(X_scaled,y_enc)


y_pred = model.predict(X_test_scaled)


y_pred


predicted_probabilities = model.predict_proba(X_test_scaled)


y_true_encoded = y_enc
predicted_probabilities_train = model.predict_proba(X_scaled)


def ap_at_k_multiclass(true_label_encoded, predicted_probs_for_sample, k):
    top_k_indices = np.argsort(predicted_probs_for_sample)[::-1][:k]

    if true_label_encoded in top_k_indices:
        return 1.0
    else:
        return 0.0
ap_scores_at_3 = []
for i in range(len(y_true_encoded)):
    ap_scores_at_3.append(
        ap_at_k_multiclass(y_true_encoded[i], predicted_probabilities_train[i], k=3)
    )

map_at_3_score = np.mean(ap_scores_at_3)

print(f"\nCalculated MAP@3 on the training data: {map_at_3_score:.4f}")


y_pred_fertilizer_names = le_tar.inverse_transform(y_pred)


submission = pd.DataFrame({
    'id':test['id'] , # Use the stored test IDs
    'Fertilizer Name': y_pred_fertilizer_names # Your predicted fertilizer names
})
submission.to_csv('/kaggle/working/submission.csv',index=False)


submission

