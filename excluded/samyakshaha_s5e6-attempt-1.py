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


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print(train_df.dtypes)


for col in train_df.columns:
    print(f"{col}: {train_df[col].nunique()} unique values")
    print(train_df[col].unique())
    print("-" * 40)


y = train_df['Fertilizer Name']
X = train_df.drop(columns=['Fertilizer Name', 'id'])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
low_card_cat_cols = ['Soil Type']
high_card_cat_cols = ['Crop Type']


numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

low_card_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

high_card_pipeline = Pipeline([
    ('target', TargetEncoder())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols),
    ('low_cat', low_card_pipeline, low_card_cat_cols),
    ('high_cat', high_card_pipeline, high_card_cat_cols)
])


model = XGBClassifier(
    objective='multi:softprob',     
    num_class=7,                    
    use_label_encoder=False,
    eval_metric='mlogloss',
    learning_rate=0.1,          
    n_estimators=600,          
    max_depth=6,                
    min_child_weight=1,       
    gamma=0,                    
    subsample=0.8,              
    colsample_bytree=0.8,      
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

model_pipeline.fit(X_train, y_train)


y_proba = model_pipeline.predict_proba(X_test)
y_test_binarized = label_binarize(y_test, classes=range(7))

map_score = average_precision_score(y_test_binarized, y_proba, average='macro')
print(f"Mean Average Precision (MAP): {map_score:.4f}")

X_test_final = test_df.drop(columns=['id'])
test_predictions = model_pipeline.predict(X_test_final)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': test_predictions_labels
})

print("Submission created!")
print(submission.head())

submission.to_csv('/kaggle/working/submission.csv', index=False)

