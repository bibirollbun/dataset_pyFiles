# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train: pd.DataFrame = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test: pd.DataFrame = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


classes: list[str] = df_train["Fertilizer Name"].unique()


import matplotlib.pyplot as plt
import seaborn as sns


sns.histplot(df_train, x="Fertilizer Name")
plt.show()


df_train.info()


df_train = df_train.drop("id", axis=1)
df_train["Soil Type"] = df_train["Soil Type"].astype("category")
df_train["Crop Type"] = df_train["Crop Type"].astype("category")
df_train["Fertilizer Name"] = df_train["Fertilizer Name"].astype("category")


numeric_df = df_train.select_dtypes(exclude=['category'])
# Melt the DataFrame to long format
melted_df = numeric_df.melt(var_name='Columns', value_name='Values')

# Create a single violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(y='Columns', x='Values', data=melted_df)

# Set the title of the plot
plt.title('Violin Plot for Each Numeric Column')

# Show the plot
plt.show()


from sklearn.preprocessing import LabelEncoder
# Nous isolons la target :
y = df_train["Fertilizer Name"]
# Encoder les labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


le.inverse_transform(y_encoded)[:10]


f1_scores = []
map3_score = []

all_y_true = []
all_y_pred = []


cat_features = df_train.drop(columns=["Fertilizer Name"]).select_dtypes(include='category').columns
cat_features


numerical_cols = df_train.drop(columns=["Fertilizer Name"]).select_dtypes(include=['int64', 'float64']).columns
numerical_cols


X = df_train.drop(columns=["Fertilizer Name"]).copy()
y = df_train["Fertilizer Name"].copy()
y_encoded = le.fit_transform(y)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


from sklearn.pipeline import Pipeline



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('nominal', nominal_transformer, ['Soil Type', 'Crop Type'])], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])


model.fit(X_train, y_train)


y_pred_proba = model.predict_proba(X_test)


y_pred_proba[0]


def map3(y_true, y_pred_proba):
    # Assuming y_pred_proba has shape (n_samples, n_classes)
    n_samples, n_classes = y_pred_proba.shape
    map3_scores = np.zeros(n_samples)
    
    for i, (true_label, probs) in enumerate(zip(y_true, y_pred_proba)):
        # Get the indices of the top 3 predicted classes
        top_3_indices = np.argsort(probs)[::-1][:3]
        
        # Check if the true label is among the top 3
        if true_label in top_3_indices:
            # Calculate the precision at the position of the true label
            precision_at_position = np.sum([1 if top_3_indices[j] == true_label else 0 for j in range(top_3_indices.tolist().index(true_label)+1)]) / (top_3_indices.tolist().index(true_label) + 1)
            map3_scores[i] = precision_at_position
        else:
            map3_scores[i] = 0.0
            
    return np.mean(map3_scores)


# Assuming le is your LabelEncoder instance
y_test_labels = le.inverse_transform(y_test)
y_test_encoded_for_map = np.array([list(classes).index(label) for label in y_test_labels]) # classes is your list of unique class labels

map3_score = map3(y_test_encoded_for_map, y_pred_proba)
print("MAP@3 Score:", map3_score)


df_test["Soil Type"] = df_test["Soil Type"].astype("category")
df_test["Crop Type"] = df_test["Crop Type"].astype("category")


validation_probas = model.predict_proba(df_test.drop(columns=['id']))


# Get the indices of the top 3 probabilities in descending order
top3_indices = np.argsort(validation_probas, axis=-1)[:, ::-1][:, :3]

# Assuming le is your LabelEncoder instance
label_names = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)

print(label_names)


df_labels = pd.DataFrame({
    "id": df_test["id"].values,
    "label_names": label_names.tolist()
})


df_labels


df_labels = pd.DataFrame({
    "id": df_test["id"].values,
    "Fertilizer Name": [" ".join(row) for row in label_names]
})


df_labels


df_labels.to_csv("submission.csv", index=False)




