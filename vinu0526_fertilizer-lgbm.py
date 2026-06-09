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


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


# Rename the column 
train_df = train_df.rename(columns={'Temparature': 'Temperature'})
print(train_df.columns) 


train_df.head()


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class FertilizerLGBMPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.cat_features = ['Soil Type', 'Crop Type']
        self.num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        self.pipeline = None

    def feature_engineering(self, df):
        df = df.copy()

        # Feature interactions
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1)
        df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
        df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']

        # Frequency encoding
        df['Soil_Freq'] = df['Soil Type'].map(df['Soil Type'].value_counts(normalize=True))
        df['Crop_Freq'] = df['Crop Type'].map(df['Crop Type'].value_counts(normalize=True))

        return df

    def fit(self, df):
        df = df.copy()
        df = self.feature_engineering(df)

        X = df.drop(columns=['Fertilizer Name', 'id'])
        y = df['Fertilizer Name']

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.num_features + ['N_K_ratio', 'P_K_ratio', 'Temp_Moisture', 'Humidity_Moisture', 'Soil_Freq', 'Crop_Freq']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
        ])

        self.pipeline = Pipeline([
            ('pre', preprocessor),
            ('clf', LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42))
        ])

        self.pipeline.fit(X, y_encoded)

    def predict_top3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)
        X = df.drop(columns=['id'], errors='ignore')
        probas = self.pipeline.predict_proba(X)

        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
        top3_labels = self.label_encoder.inverse_transform(top3.ravel()).reshape(top3.shape)
        return top3_labels

    def evaluate_map3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)

        X = df.drop(columns=['Fertilizer Name', 'id'])
        y_true = df['Fertilizer Name']
        y_encoded = self.label_encoder.transform(y_true)

        probas = self.pipeline.predict_proba(X)
        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]

        def apk(actual, predicted, k=3):
            if actual in predicted[:k]:
                return 1.0 / (predicted[:k].tolist().index(actual) + 1)
            return 0.0

        scores = [apk(a, p) for a, p in zip(y_encoded, top3)]
        return np.mean(scores)

    def save(self, path_prefix='fertilizer_lgbm_model'):
        joblib.dump(self.pipeline, f'{path_prefix}_pipeline.pkl')
        joblib.dump(self.label_encoder, f'{path_prefix}_label_encoder.pkl')

    def load(self, path_prefix='fertilizer_lgbm_model'):
        self.pipeline = joblib.load(f'{path_prefix}_pipeline.pkl')
        self.label_encoder = joblib.load(f'{path_prefix}_label_encoder.pkl')


from sklearn.model_selection import train_test_split

# Train/validation split
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Fertilizer Name'])

# Train model
predictor = FertilizerLGBMPredictor()
predictor.fit(train_split)


predictor.save()  # Saves model, scaler, and label encoder with default prefix


# Evaluate
val_map3 = predictor.evaluate_map3(val_split)
print(f"Validation MAP@3: {val_map3:.5f}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


test_df = test_df.rename(columns={'Temparature': 'Temperature'})
print(test_df.columns) 


predictor = FertilizerLGBMPredictor()
predictor.load()


top3_preds = predictor.predict_top3(test_df)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_preds]
})


# Save submission file
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


submission_df = pd.read_csv("/kaggle/working/submission.csv")


submission_df.head()

