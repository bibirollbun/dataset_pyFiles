import pandas as pd
import sys


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.head()


df.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()
numImputer = SimpleImputer(strategy='mean')
catImputer = SimpleImputer(strategy='most_frequent')
X = df.drop(columns=['id', 'Personality'])
y = df['Personality']
X.head()


num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print("Numerical Features:", num_features)
print("Categorical Features:", cat_features)


imputed_num_features = numImputer.fit_transform(X[num_features])
imputed_cat_features = catImputer.fit_transform(X[cat_features])
X[num_features] = imputed_num_features
X[cat_features] = imputed_cat_features
print("Imputation complete.")
encoded_cat_features = one_hot_encoder.fit_transform(X[cat_features])
scaled_num_features = scaler.fit_transform(X[num_features])
X_transformed = pd.DataFrame(
    data=encoded_cat_features,
    columns=one_hot_encoder.get_feature_names_out(cat_features)
).join(pd.DataFrame(scaled_num_features, columns=num_features))
X_transformed.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


import os

os.makedirs("Data/Train Data", exist_ok=True)
os.makedirs("Data/Test Data", exist_ok=True)

X_train.to_csv("Data/Train Data/X_train.csv", index=False)
y_train.to_csv("Data/Train Data/y_train.csv", index=False)
X_test.to_csv("Data/Test Data/X_test.csv", index=False)
y_test.to_csv("Data/Test Data/y_test.csv", index=False)


train_X = pd.read_csv("Data/Train Data/X_train.csv")
train_y = pd.read_csv("Data/Train Data/y_train.csv")


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=300, random_state=42)
rf.fit(train_X, train_y.values.ravel())


# on test set eval
test_X = pd.read_csv("Data/Test Data/X_test.csv")
test_y = pd.read_csv("Data/Test Data/y_test.csv")
from sklearn.metrics import classification_report, accuracy_score
y_pred = rf.predict(test_X)
print("Accuracy:", accuracy_score(test_y, y_pred))
print(classification_report(test_y, y_pred))



sub_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub_df.info()


ids = sub_df['id']
sub_df.drop(columns=['id'], inplace=True)
sub_imputed_num_features = numImputer.transform(sub_df[num_features])
sub_imputed_cat_features = catImputer.transform(sub_df[cat_features])
sub_df[num_features] = sub_imputed_num_features
sub_df[cat_features] = sub_imputed_cat_features
print("Imputation on submission data complete.")
sub_encoded_cat_features = one_hot_encoder.transform(sub_df[cat_features])
sub_scaled_num_features = scaler.transform(sub_df[num_features])
final_df = pd.DataFrame(
    data=sub_encoded_cat_features,
    columns=one_hot_encoder.get_feature_names_out(cat_features)
).join(pd.DataFrame(sub_scaled_num_features, columns=num_features))
sub_predictions =rf.predict(final_df)
submission_df = pd.DataFrame({
    'id': ids,
    'Personality': sub_predictions
})
submission_df.to_csv("submission.csv", index=False)

