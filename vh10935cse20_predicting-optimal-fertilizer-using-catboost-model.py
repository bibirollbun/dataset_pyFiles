import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.shape


test.shape


train.columns


test.columns


train.isna().sum()


test.isna().sum()


train.info()


test.info()


train.dtypes


test.dtypes


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


cat_col=['Soil Type','Crop Type']
for i in cat_col:
    train[i]=le.fit_transform(train[i])

for i in cat_col:
    test[i]=le.fit_transform(test[i])


from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
model=CatBoostClassifier()


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


model.fit(X_scaled,y_enc.ravel())


y_pred = model.predict(X_test_scaled)


predicted_probabilities = model.predict(X_test_scaled, prediction_type='Probability')


y_true_encoded = y_enc
predicted_probabilities_train = model.predict(X_scaled, prediction_type='Probability')


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

