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


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


test.head()


train['Fertilizer Name'].value_counts()


train.info()


test_ids = test['id']


def engineer_features(df):
    df = df.copy()
    # Basic ratios and totals
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    df['Nutrient_Total'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)
    # Simple interactions
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df



train = engineer_features(train)
test = engineer_features(test)


le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])
X = train.drop(columns=['Fertilizer Name'])
# Save encoder for later
joblib.dump(le, 'label_encoder.joblib')


# Define preprocessing
numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 
                   'Potassium', 'Phosphorous', 'N_P_ratio', 'N_K_ratio', 
                   'P_K_ratio', 'Nutrient_Total', 'Temp_Humidity']
categorical_features = ['Soil Type', 'Crop Type']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    eval_metric='mlogloss'
)



pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])



pipeline.fit(X, y)


probabilities = pipeline.predict_proba(test)


le = joblib.load('label_encoder.joblib')


top3_fertilizers = []
for prob_row in probabilities:
    top3_indices = np.argsort(prob_row)[-3:][::-1]
    top3_labels = le.inverse_transform(top3_indices)
    top3_fertilizers.append(" ".join(top3_labels))


submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': top3_fertilizers
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")




