import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import xgboost as xgb
import catboost as cat
import xgboost as xgb
from sklearn.model_selection import  train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel



df = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')



print(df.info())
print(df.isnull().sum())


df['BMI'] = df['Weight'] / (df['Height']/100)**2
df_test['BMI'] = df_test['Weight'] / (df_test['Height']/100)**2


def high_classify(df):
    if df['Sex'] == 'male':
        if df['Height'] < 160.0:
            return 'Short'
        elif 160.0 <= df['Height'] < 180.0:
            return 'Average'
        elif df['Height'] >= 180.0:
            return 'High'
        else:
            return 'Others'
    else:
        if df['Height'] < 145.0:
            return 'Short'
        elif 145.0 <= df['Height'] < 170.0:
            return 'Average'
        elif df['Height'] >= 170:
            return 'High'
        else:
            return 'Others'


df['Height_classify'] = df.apply(high_classify, axis = 1)
df_test['Height_classify'] = df_test.apply(high_classify, axis = 1)


df['Duration_HeartRate'] = df['Duration'] * df['Heart_Rate']
df_test['Duration_HeartRate'] = df_test['Duration'] * df_test['Heart_Rate']


df['Body_Temp_sub'] = df['Body_Temp'] - np.mean(df['Body_Temp'])
df_test['Body_Temp_sub'] = df_test['Body_Temp'] - np.mean(df_test['Body_Temp'])


df['Height_classify'].value_counts()


def bmi_classify(df):
    bmi = df['BMI']
    if bmi < 18.5 or bmi >= 30:
        return 'Abnormal'
    elif bmi < 25:
        return 'Normal'
    else:
        return 'Overweight'



df['BMI_classify'] = df.apply(bmi_classify, axis = 1)
df_test['BMI_classify'] = df_test.apply(bmi_classify, axis = 1)



df['BMI_classify'].value_counts()


def age_classify(df):
    if df['Age'] < 35:
        return 'adult'
    elif 35 <= df['Age'] < 60:
        return 'middle_age'
    elif 60 <= df['Age'] < 80:
        return 'elderly'
    else:
        return 'Others'


df['Age_classify'] = df.apply(age_classify, axis = 1)
df_test['Age_classify'] = df_test.apply(age_classify, axis = 1)



high = ['Short','Average','High','Others']
bmi = ['Abnormal','Normal', 'Overweight']
age = ['adult','middle_age','elderly','Others']

columns_to_encode = ['Height_classify', 'BMI_classify', 'Age_classify']
df[columns_to_encode] = df[columns_to_encode].astype(str)
df_test[columns_to_encode] = df_test[columns_to_encode].astype(str)

ord_encoded = OrdinalEncoder(categories=[high, bmi, age], handle_unknown='use_encoded_value', unknown_value=-1)

df[columns_to_encode] = ord_encoded.fit_transform(df[columns_to_encode]).astype(int)
df_test[columns_to_encode] = ord_encoded.transform(df_test[columns_to_encode]).astype(int)



df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
}).astype(int)
df_test['Sex'] = df_test['Sex'].map({
    'male': 0,
    'female': 1
}).astype(int)


def combine_cat_feature(df, cols):
    combined = df[cols[0]].astype(str)
    for i in range(1, len(cols)):
        combined = combined.str.cat(df[cols[i]].astype(str), sep='')
    return combined


df['Age_Sex'] = combine_cat_feature(df,['Age_classify','Sex'])
df['Age_BMI'] = combine_cat_feature(df, ['Age_classify','BMI_classify'])
df['Age_Heigh'] = combine_cat_feature(df, ['Age_classify','Height_classify'])
df['BMI_Heigh'] = combine_cat_feature(df, ['BMI_classify','Height_classify'])
df['BMI_Sex'] = combine_cat_feature(df, ['BMI_classify','Sex'])
df['Heigh_Sex'] = combine_cat_feature(df, ['Height_classify','Sex'])

# combine 3 feature
df['Age_Sex_BMI'] = combine_cat_feature(df,['Age_classify','Sex','BMI_classify'])
df['Age_Sex_Heigh'] = combine_cat_feature(df,['Age_classify','Sex','Height_classify'])
df['Age_BMI_Heigh'] = combine_cat_feature(df, ['Age_classify','BMI_classify','Height_classify'])
df['BMI_Sex_Heigh'] = combine_cat_feature(df, ['BMI_classify','Sex','Height_classify'])



df_test['Age_Sex'] = combine_cat_feature(df_test,['Age_classify','Sex'])
df_test['Age_BMI'] = combine_cat_feature(df_test, ['Age_classify','BMI_classify'])
df_test['Age_Heigh'] = combine_cat_feature(df_test, ['Age_classify','Height_classify'])
df_test['BMI_Heigh'] = combine_cat_feature(df_test, ['BMI_classify','Height_classify'])
df_test['BMI_Sex'] = combine_cat_feature(df_test, ['BMI_classify','Sex'])
df_test['Heigh_Sex'] = combine_cat_feature(df_test, ['Height_classify','Sex'])

df_test['Age_Sex_BMI'] = combine_cat_feature(df_test,['Age_classify','Sex','BMI_classify'])
df_test['Age_Sex_Heigh'] = combine_cat_feature(df_test,['Age_classify','Sex','Height_classify'])
df_test['Age_BMI_Heigh'] = combine_cat_feature(df_test, ['Age_classify','BMI_classify','Height_classify'])
df_test['BMI_Sex_Heigh'] = combine_cat_feature(df_test, ['BMI_classify','Sex','Height_classify'])


cols = df.select_dtypes('object')
for col in cols:
    df[col] = df[col].astype(int)

cols_test = df_test.select_dtypes('object')
for col_test in cols_test:
    df_test[col_test] = df_test[col_test].astype(int)


X_sel = df[['Duration', 'Heart_Rate', 'Age', 'Sex', 'Body_Temp', 'Weight', 'Age_Sex_BMI', 'Age_Sex_Heigh', 'Age_Sex', 'Age_Heigh', 'Age_classify']]
Y = df['Calories']
pipe_cat = Pipeline([
    ('scale', StandardScaler()),
    ('model', cat.CatBoostRegressor(verbose = 100,
                                    max_depth= 10, 
                                    learning_rate= 0.18116379483606515, 
                                    l2_leaf_reg= 2.3177203762114944, 
                                    iterations= 2649,
                                    loss_function= 'RMSE',
                                    random_seed= 42,
                                    eval_metric= 'RMSE'
                                   ))
])


pipe_cat.fit(X_sel, Y)


y_pred_1 = pipe_cat.predict(X_sel)
rmsle = np.sqrt(np.mean((np.log1p(Y) - np.log1p(y_pred_1))**2))
print(f'RMSLE = {rmsle:.6f}')


df_test['predicted_cat'] = pipe_cat.predict(df_test[['Duration', 'Heart_Rate', 'Age', 'Sex', 'Body_Temp', 'Weight', 'Age_Sex_BMI', 'Age_Sex_Heigh', 'Age_Sex', 'Age_Heigh', 'Age_classify']])
df_submission = pd.DataFrame(
        {
            "id": df_test['id'],
            "Calories": df_test['predicted_cat']
        }
    )
df_submission.to_csv('submission.csv', index=False)



print(f"Number of negative values in Submission: {sum(df_submission['Calories'] < 0)}")

