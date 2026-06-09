!pip install -q imbalanced-learn==0.11.0
from imblearn.over_sampling import SMOTE

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train.head()


train.info()


print(train["Personality"].value_counts())  # Distribusi label


#split data
X = train.drop(columns=['Personality', 'id'])
y = train['Personality']
y


#Encode untuk jenis kategorikal
le = LabelEncoder()
y = le.fit_transform(y)

y #extrovert = 0, introvert = 1


#menentukan kolom numerik dan kategorikal
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns


#pipeline prepocessing 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), #mengisi NaN untuk numerikal dengan median
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), #mengisi NaN untuk data kategorial dengan data paling sering muncul
            ('encoder', OneHotEncoder(handle_unknown='ignore')) #Encode kategorikal
        ]), cat_cols)
    ]
)


#split data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#melakukan preprocessing pada data
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)


model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)


y_pred = model.predict(X_val_prep)
print(classification_report(y_val, y_pred))


model.fit(X_train_smote, y_train_smote)


X_test = test_df.drop(columns=['id'])
X_test_prep = preprocessor.transform(X_test)


test_pred = model.predict(X_test_prep)
test_pred_label = np.where(test_pred == 0,  'Extrovert','Introvert')

test_pred_label


submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_pred_label
})

submission.to_csv('submission.csv', index=False)


submission.head()

