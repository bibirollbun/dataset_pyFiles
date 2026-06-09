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


train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv', index_col=0)
test = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv", index_col=0)
submit = pd.read_csv("/kaggle/input/multiclassificationtask/sample_submission.csv")

status = train['Status'].copy()
train.drop('Status', inplace=True, axis= 1)
train = pd.concat([train, test])


train.head()


train.info()


train.drop(13789, inplace=True) #Bitta Y status uchun
status.drop(13789, inplace=True)


train.isnull().sum()/len(train)*100


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import lightgbm as lgb


# *** 1. ModeImputer - Kategorik ustunlar uchun ***
class ModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes = X.mode().iloc[0]
        return self

    def transform(self, X):
        return X.fillna(self.modes)

# *** 2. Kategorik va sonli ustunlarni ajratamiz ***
categorical_cols = ['Drug', 'Ascites', 'Hepatomegaly', 'Spiders']
numerical_cols = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

# *** 3. NaN foizi yuqori bo'lgan ustunlarni aniqlash ***
high_nan_categorical = [col for col in categorical_cols if train[col].isna().mean() > 0.4]
high_nan_numerical = [col for col in numerical_cols if train[col].isna().mean() > 0.5]

# *** 4. Kategorik ustunlarni to'ldirish (Mode va One-Hot Encoding) ***
categorical_imputer = ColumnTransformer(
    transformers=[
        ('cat_mode', ModeImputer(), [col for col in categorical_cols if col not in high_nan_categorical]),
        ('cat_missing', SimpleImputer(strategy='constant', fill_value='Missing'), high_nan_categorical)
    ]
)

# *** 5. Sonli ustunlarni to'ldirish (IterativeImputer va LightGBM) ***
def lightgbm_impute(train, feature):
    """
    LightGBM yordamida NaN qiymatlarni bashorat qilib to'ldirish.
    """
    temp_train = train.copy()

    # NaN bo'lmagan qiymatlar va NaN qiymatlari uchun maska
    train_data = temp_train.dropna(subset=[feature])
    missing_data = temp_train[temp_train[feature].isna()]

    if missing_data.empty:
        return train  # Agar NaN bo'lmasa, o'zgartirish kiritilmaydi

    # Modelni o'rgatish uchun ma'lumotlar
    X_train = train_data.drop(columns=[feature])
    y_train = train_data[feature]

    # LightGBM modelini yaratish va o'rgatish
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    # NaN qiymatlarni bashorat qilish va to'ldirish
    X_missing = missing_data.drop(columns=[feature])
    temp_train.loc[temp_train[feature].isna(), feature] = model.predict(X_missing)

    return temp_train

# Iterative Imputer (MICE) va LightGBM o'rtasida tanlash
def advanced_numerical_imputer(train):
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

    # Past NaN foizli ustunlarni MICE bilan to'ldirish
    train[numerical_cols] = iterative_imputer.fit_transform(train[numerical_cols])

    # Yuqori NaN foizli ustunlarni LightGBM bilan to'ldirish
    for col in high_nan_numerical:
        train = lightgbm_impute(train, col)

    return train

# *** 6. Pipeline yaratish ***
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_impute', categorical_imputer, categorical_cols)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# *** 7. Ma'lumotlarni to'ldirish ***
train_filled = pipeline.fit_transform(train)

# DataFrame shakliga keltirish
train_filled = pd.DataFrame(train_filled, columns=categorical_cols)

# Kategorik ustunlar yangilandi, endi sonli ustunlar uchun ilg‘or imputer
train = pd.concat([train.drop(columns=categorical_cols), train_filled], axis=1)
train = advanced_numerical_imputer(train)


train.isnull().sum()/len(train)*100


#Scalar
stan_scal = StandardScaler()
scalar_col = ['N_Days', 'Age',  'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',  'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage' ]

#OneHot
one_hot = OneHotEncoder(drop = 'first', dtype=np.int8, sparse_output=False )
one_hot_col = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

col_trans = ColumnTransformer([
    ('stan_scal', stan_scal, scalar_col),
    ('one_hot', one_hot, one_hot_col)
], remainder='passthrough').set_output(transform='pandas')


train = col_trans.fit_transform(train)
train


train.fillna(train.median(), inplace=True) #qolganlari uchun


from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier


model_RF = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_GB = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model_KNN = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=10, random_state=42)  # Bagging bilan yaxshilangan KNN
model_SVC = CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, probability=True))  # CalibratedClassifierCV bilan SVC
model_XGB = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model_LR = LogisticRegression()


#Train va Test birga
X_train = train.iloc[:14999, :].copy()  # 0 dan 14999-gacha bo'lgan barcha qatorlar

status = pd.get_dummies(status, dtype=np.int8)
status.columns = ['Status_C', 'Status_LC', 'Status_D']
y_train = status

X_test = train.iloc[14999:, :].copy()  # 15000 dan boshlab qolgan qatorlar


base_models = [
    ('svc', model_SVC),
    ('rf', model_RF),
    ('gb', model_GB),
    ('knn', model_KNN),
    ('xgb', model_XGB)
]

# ⚡ 3. Meta-model (LightGBM yoki XGBoost predict_proba ni qo‘llab-quvvatlaydi)
meta_model =  model_LR

# ⚡ 4. StackingClassifier yaratish (cv=5, predict_proba ishlaydi)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, stack_method='predict_proba')


%%time
# Stacking bilan 27 daqiqa vaqt oladi! GPU da
models = {}  # Har bir modelni saqlash uchun
predictions = {}  # Test bashoratlarini saqlash uchun
preds = {}

for column in y_train.columns:
    stacking_model.fit(X_train, y_train[column])  # Har bir ustun uchun alohida model
    models[column] = stacking_model
    predictions[column] = stacking_model.predict_proba(X_test)[:, 1]
    preds[column] = models[column].predict(X_test)


# Test bashoratlarini DataFrame shakliga keltirish
y_train_pred = pd.DataFrame(predictions, index=X_test.index)
y_train_pred.drop(13789, inplace=True) #nimagadur qo'shib qo'yibdi
y_train_pred.insert(0, "id", np.arange(15000, 15000 + len(y_train_pred)))
y_train_pred.index = np.arange(0, 10000)

# Faylni CSV formatda saqlash
y_train_pred.to_csv("sirroz_stack_update_2.csv", index=False)

# Natijani ko'rsatish
y_train_pred

