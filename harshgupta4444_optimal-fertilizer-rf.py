import pandas as pd


df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


df.head()


df.info()


df.describe(include='all')


df['Fertilizer Name'].value_counts()


df['Fertilizer Name'].value_counts()


df['Soil Type'].value_counts()


df['Crop Type'].value_counts()


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


object_cols=df.select_dtypes(include='object').columns


object_cols


for col in object_cols:
  df[col]=le.fit_transform(df[col])


df.info()


df.head()


df=df.drop(columns=['id'])


df.head()


from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()


numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


df[numerical_features]=scaler.fit_transform(df[numerical_features])


df.head()


df['NPK_Ratio_N'] = df['Nitrogen'] / (df['Potassium'] + df['Phosphorous'] + 1e-6)
df['NPK_Ratio_P'] = df['Phosphorous'] / (df['Nitrogen'] + df['Potassium'] + 1e-6)
df['NPK_Ratio_K'] = df['Potassium'] / (df['Nitrogen'] + df['Phosphorous'] + 1e-6)
df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']

df.head()


df['Temp_Humidity_Index'] = df['Temparature'] * df['Humidity']

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

# Select features to create polynomial features from
features_for_poly = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]

# Fit and transform the selected features
poly_features = poly.fit_transform(features_for_poly)

# Get the names of the new features (optional but helpful)
poly_feature_names = poly.get_feature_names_out(features_for_poly.columns)

# Create a new DataFrame with the polynomial features
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

# Concatenate the original DataFrame with the new polynomial features DataFrame
df = pd.concat([df, df_poly], axis=1)

df.head()
df.info()


from sklearn.model_selection import train_test_split


X=df.drop(columns='Fertilizer Name')
y=df['Fertilizer Name']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


from sklearn.ensemble import RandomForestClassifier


model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores
model_rf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model_rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


dft=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


dft.head()


id=dft['id']


id


object_cols=dft.select_dtypes(include='object').columns


for col in object_cols:
  dft[col]=le.fit_transform(dft[col])


dft[numerical_features]=scaler.fit_transform(dft[numerical_features])


dft['NPK_Ratio_N'] = dft['Nitrogen'] / (dft['Potassium'] + dft['Phosphorous'] + 1e-6)
dft['NPK_Ratio_P'] = dft['Phosphorous'] / (dft['Nitrogen'] + dft['Potassium'] + 1e-6)
dft['NPK_Ratio_K'] = dft['Potassium'] / (dft['Nitrogen'] + dft['Phosphorous'] + 1e-6)
dft['Total_Nutrients'] = dft['Nitrogen'] + dft['Potassium'] + dft['Phosphorous']

dft.head()


dft['Temp_Humidity_Index'] = dft['Temparature'] * dft['Humidity']

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

# Select features to create polynomial features from
features_for_polyt = dft[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]

# Fit and transform the selected features
poly_featurest = poly.fit_transform(features_for_polyt)

# Get the names of the new features (optional but helpful)
poly_feature_namest = poly.get_feature_names_out(features_for_polyt.columns)

# Create a new DataFrame with the polynomial features
dft_poly = pd.DataFrame(poly_featurest, columns=poly_feature_namest, index=dft.index)

# Concatenate the original DataFrame with the new polynomial features DataFrame
df = pd.concat([dft, dft_poly], axis=1)

dft.head()
dft.info()


dft_features = dft.drop(columns=['id'])


train_cols = X_train.columns


dft_aligned = dft_features.reindex(columns=train_cols, fill_value=0) 


y_pred_proba = model_rf.predict_proba(dft_aligned)


y_pred_proba


manual_fertilizer_mapping = {
    0: '10-26-26',
    1: '14-35-14',
    2: '17-17-17',
    3: '28-28',
    4: '20-20',
    5: 'DAP',
    6: 'Urea'
}


# Function to get top N predicted fertilizer names using the manual mapping
def get_top_n_fertilizers_with_mapping(probabilities, mapping, n=3):
    # Get the indices that would sort the probabilities in descending order
    sorted_indices = probabilities.argsort()[::-1]
    # Get the top N indices
    top_n_indices = sorted_indices[:n]
    # Get the corresponding fertilizer names from the mapping
    top_n_labels = [mapping[i] for i in top_n_indices]
    # Join the labels with space
    return " ".join(top_n_labels)


predicted_fertilizers = [get_top_n_fertilizers_with_mapping(row, manual_fertilizer_mapping, n=3) for row in y_pred_proba]

submission_df = pd.DataFrame({'id': id, 'Fertilizer Name': predicted_fertilizers})

submission_df.to_csv('submissiono.csv', index=False)

print("Submission file created successfully: submission.csv")




