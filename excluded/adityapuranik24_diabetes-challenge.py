import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from ydata_profiling import ProfileReport
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import warnings
import pickle
warnings.filterwarnings("ignore")
%matplotlib inline


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


# ProfileReport(df_train).to_file("diabetes_train_data_profile.html")



corr = df_train.select_dtypes(include='number').corr(method='spearman')
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True,cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Matrix")
plt.show()


x = df_train.drop(columns=['id', 'employment_status', 'diagnosed_diabetes'], axis=1)
y = df_train['diagnosed_diabetes']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)


log_and_scale = Pipeline(steps=[
    ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
    ('scale', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('LogScale', log_and_scale,
         ['physical_activity_minutes_per_week', 'diet_score', 'triglycerides']),

        ('Scaling', StandardScaler(), [
            'alcohol_consumption_per_week', 'sleep_hours_per_day',
            'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio',
            'systolic_bp', 'diastolic_bp', 'heart_rate',
            'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol'
        ]),

        ('Nominal', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         ['gender', 'ethnicity']),

        ('Ordinal', OrdinalEncoder(categories=[
            ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
            ['Never', 'Former', 'Current'], ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
        ]), ['income_level', 'smoking_status', 'education_level'])
    ],
    remainder='passthrough'
)


pipeline = Pipeline(steps=[
    ('Applying transformation & preprossing', preprocessor),
    ('Model Tarining', LogisticRegression())]
)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


y_pred_test = pipeline.predict(df_test)


submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': y_pred_test
})


submission.to_csv("submission.csv", index=False)




