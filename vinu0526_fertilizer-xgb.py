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


train_df.columns


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class FertilizerXGBPredictor:
    def __init__(self):
        self.models = []
        self.label_encoder = None
        self.cat_features = ['Soil Type', 'Crop Type']
        self.num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        self.preprocessor = None

    def feature_engineering(self, df):
        df = df.copy()
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1)
        df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
        df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
        df['Soil_Freq'] = df['Soil Type'].map(df['Soil Type'].value_counts(normalize=True))
        df['Crop_Freq'] = df['Crop Type'].map(df['Crop Type'].value_counts(normalize=True))
        return df

    def fit_cv(self, df, n_splits=5):
        df = df.copy()
        df = self.feature_engineering(df)

        X = df.drop(columns=['Fertilizer Name', 'id'])
        y = df['Fertilizer Name']

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.num_features + ['N_K_ratio', 'P_K_ratio', 'Temp_Moisture', 'Humidity_Moisture', 'Soil_Freq', 'Crop_Freq']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
        ])

        X_preprocessed = self.preprocessor.fit_transform(X)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold = 1
        for train_idx, val_idx in skf.split(X_preprocessed, y_encoded):
            print(f"\nğŸ”� Training fold {fold}...")
            fold += 1

            X_train_fold, y_train_fold = X_preprocessed[train_idx], y_encoded[train_idx]
            X_val_fold, y_val_fold = X_preprocessed[val_idx], y_encoded[val_idx]

            # Split train_fold into train/validation for early stopping
            X_train_split, X_es_val, y_train_split, y_es_val = train_test_split(
                X_train_fold, y_train_fold, test_size=0.15, random_state=42, stratify=y_train_fold
            )

            model = XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                n_estimators=3200,
                learning_rate=0.045,
                max_depth=7,
                colsample_bytree=0.6,
                colsample_bylevel=0.8,
                subsample=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )

            model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_es_val, y_es_val)],
                early_stopping_rounds=50,
                verbose=True
            )

            self.models.append(model)

            # Evaluate fold MAP@3
            probas = model.predict_proba(X_val_fold)
            top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]

            def apk(actual, predicted, k=3):
                if actual in predicted[:k]:
                    return 1.0 / (predicted[:k].tolist().index(actual) + 1)
                return 0.0

            scores = [apk(a, p) for a, p in zip(y_val_fold, top3)]
            map3 = np.mean(scores)
            print(f"ğŸ“Š Fold MAP@3: {map3:.5f}")

    def predict_top3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)
        X = df.drop(columns=['id'], errors='ignore')
        X_preprocessed = self.preprocessor.transform(X)

        probas = np.mean([model.predict_proba(X_preprocessed) for model in self.models], axis=0)
        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
        top3_labels = self.label_encoder.inverse_transform(top3.ravel()).reshape(top3.shape)
        return top3_labels

    def evaluate_map3(self, df):
        df = df.copy()
        df = self.feature_engineering(df)
        X = df.drop(columns=['Fertilizer Name', 'id'])
        y_true = df['Fertilizer Name']
        y_encoded = self.label_encoder.transform(y_true)
        X_preprocessed = self.preprocessor.transform(X)

        probas = np.mean([model.predict_proba(X_preprocessed) for model in self.models], axis=0)
        top3 = np.argsort(probas, axis=1)[:, -3:][:, ::-1]

        def apk(actual, predicted, k=3):
            if actual in predicted[:k]:
                return 1.0 / (predicted[:k].tolist().index(actual) + 1)
            return 0.0

        scores = [apk(a, p) for a, p in zip(y_encoded, top3)]
        return np.mean(scores)

    def save(self, path_prefix='fertilizer_xgb_model_cv'):
        for i, model in enumerate(self.models):
            joblib.dump(model, f'{path_prefix}_fold{i}.pkl')
        joblib.dump(self.label_encoder, f'{path_prefix}_label_encoder.pkl')
        joblib.dump(self.preprocessor, f'{path_prefix}_preprocessor.pkl')

    def load(self, path_prefix='fertilizer_xgb_model_cv', n_models=5):
        self.models = [joblib.load(f'{path_prefix}_fold{i}.pkl') for i in range(n_models)]
        self.label_encoder = joblib.load(f'{path_prefix}_label_encoder.pkl')
        self.preprocessor = joblib.load(f'{path_prefix}_preprocessor.pkl')



predictor = FertilizerXGBPredictor()
predictor.fit_cv(train_df, n_splits=5)


predictor.save()  # Saves model, scaler, and label encoder with default prefix


# Evaluate
# val_map3 = predictor.evaluate_map3(val_split)
# print(f"Validation MAP@3: {val_map3:.5f}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


test_df = test_df.rename(columns={'Temparature': 'Temperature'})
print(test_df.columns) 


predictor = FertilizerXGBPredictor()
predictor.load()


top3_preds = predictor.predict_top3(test_df)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_preds]
})


# Save submission file
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission_df = pd.read_csv("/kaggle/working/submission.csv")


submission_df.head()

