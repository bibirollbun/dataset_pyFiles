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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


dt_train = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
dt_test = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/test.csv')
dt_train.head()


cols = ['id','Occupation']
dt_train.drop_duplicates(inplace=True)
dt_test.drop_duplicates(inplace=True)
X = dt_train.drop(columns=cols)            
X = X.drop(columns='Premium Amount')
Xt = dt_test.drop(columns=cols)
y = dt_train['Premium Amount']


X.isnull().sum() / X.shape[0]*100


def data_Filling(data) :
    # Fillling NaN values of Customer Feedback
    for occ in data['Smoking Status'].unique():
        for edu in data['Exercise Frequency'].unique():

            mask = ((data['Smoking Status'] == occ) &
                    (data['Exercise Frequency'] == edu))

            group = data.loc[mask, 'Customer Feedback']

            if group.isna().sum() == 0:
                continue

            value_counts = group.value_counts(normalize=True)
    
            if value_counts.empty:
                probs = [0.333,0.333,0.334]
                choices = ['Poor','Good','Average']
            else:
                probs = []
                choices = []
                for cat in ['Poor','Good','Average']:
                    choices.append(cat)
                    probs.append(value_counts.get(cat, 0))

                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = [0.333,0.333,0.334]

            n_missing = group.isna().sum()
            fill_values = np.random.choice(choices, size=n_missing, p=probs)

            data.loc[mask & data['Customer Feedback'].isna(), 'Customer Feedback'] = fill_values


    # Fillling NaN values of Marital Status
    for occ in data['Location'].unique():
        for edu in data['Education Level'].unique():

            mask = ((data['Location'] == occ) &
                    (data['Education Level'] == edu))

            group = data.loc[mask, 'Marital Status']

            if group.isna().sum() == 0:
                continue

            value_counts = group.value_counts(normalize=True)
    
            if value_counts.empty:
                probs = [0.333,0.333,0.334] 
                choices = ['Single','Married','Divorced']
            else:
                probs = []
                choices = []
                for cat in ['Single','Married','Divorced']:
                    choices.append(cat)
                    probs.append(value_counts.get(cat, 0))

                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = [0.333,0.333,0.334]

            n_missing = group.isna().sum()
            fill_values = np.random.choice(choices, size=n_missing, p=probs)

            data.loc[mask & data['Marital Status'].isna(), 'Marital Status'] = fill_values

    # Fillling NaN values of No. of Children
    for occ in data['Location'].unique():
        for edu in data['Education Level'].unique():

            mask = ((data['Location'] == occ) &
                    (data['Education Level'] == edu))

            group = data.loc[mask, 'Number of Children']

            if group.isna().sum() == 0:
                continue

            value_counts = group.value_counts(normalize=True)
    
            if value_counts.empty:
                probs = [0.2, 0.2, 0.2, 0.2, 0.2] 
                choices = [0.0, 1.0, 2.0, 3.0, 4.0]
            else:
                probs = []
                choices = []
                for cat in [0.0, 1.0, 2.0, 3.0, 4.0]:
                    choices.append(cat)
                    probs.append(value_counts.get(cat, 0))

                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = [0.2, 0.2, 0.2, 0.2, 0.2]

            n_missing = group.isna().sum()
            fill_values = np.random.choice(choices, size=n_missing, p=probs)

            data.loc[mask & data['Number of Children'].isna(), 'Number of Children'] = fill_values

    # Filling the NaN values of Age : 
    for child in data['Number of Children'].unique():
        for marital in data['Policy Type'].unique():

            mask = ((data['Number of Children'] == child) &
                    (data['Policy Type'] == marital))

            group = data.loc[mask, 'Age']

            
            if group.isna().sum() == 0:
                continue

        
            if group.dropna().shape[0] > 0:
                mean_age = group.mean()
                std_age = group.std()

                
                if np.isnan(std_age) or std_age == 0:
                    std_age = 5  

                n_missing = group.isna().sum()
                fill_values = np.random.normal(mean_age, std_age, n_missing)

                
                fill_values = np.clip(fill_values, 18, 64)

            else:
            
                n_missing = group.isna().sum()
                fill_values = np.random.randint(18, 65, n_missing)

            data.loc[mask & data['Age'].isna(), 'Age'] = fill_values
    
    # Filling NaN values of Insurance Duration and Vehicle Age
    imputer = SimpleImputer(strategy='mean')
    data[['Vehicle Age','Insurance Duration','Previous Claims']] = imputer.fit_transform(data[['Vehicle Age','Insurance Duration','Previous Claims']])

    # Filling NaN values of Annual Income 
    original_income = data['Annual Income'].dropna().copy()
    segment_count = 0
    imputed_count = 0
    fallback_count = 0
    for marital in data['Policy Type'].unique():
            for edu in data['Education Level'].unique():
                for children in data['Number of Children'].unique():
                    
                    mask = ((data['Policy Type'] == marital) &
                        (data['Education Level'] == edu) &
                        (data['Number of Children'] == children))
                    
                    group = data.loc[mask, 'Annual Income']
                    
                    if group.isna().sum() == 0:
                        continue
                    
                    segment_count += 1
                    n_missing = group.isna().sum()
                    
                    existing_values = group.dropna()
                    
                    if len(existing_values) >= 5:  
                        fill_values = np.random.choice(existing_values, size=n_missing, replace=True)
                        imputed_count += n_missing
                    else:
                        fallback_mask = ((data['Policy Type'] == marital) &
                                        (data['Education Level'] == edu))
                        fallback_group = data.loc[fallback_mask, 'Annual Income'].dropna()
                        
                        if len(fallback_group) >= 5:
                            fill_values = np.random.choice(fallback_group, size=n_missing, replace=True)
                            fallback_count += n_missing
                        else:
                            fill_values = np.random.choice(original_income, size=n_missing, replace=True)
                            fallback_count += n_missing
                    data.loc[mask & data['Annual Income'].isna(), 'Annual Income'] = fill_values

    # # Filling NaN values of Credit Score
    original_income = data['Credit Score'].dropna().copy()
    segment_count = 0
    imputed_count = 0
    fallback_count = 0
    for marital in data['Location'].unique():
            for edu in data['Policy Type'].unique():
                for children in data['Property Type'].unique():
                    
                    mask = ((data['Location'] == marital) &
                        (data['Policy Type'] == edu) &
                        (data['Property Type'] == children))
                    
                    group = data.loc[mask, 'Credit Score']
                    
                    if group.isna().sum() == 0:
                        continue
                    
                    segment_count += 1
                    n_missing = group.isna().sum()
                    
                    existing_values = group.dropna()
                    
                    if len(existing_values) >= 5:  
                        fill_values = np.random.choice(existing_values, size=n_missing, replace=True)
                        imputed_count += n_missing
                    else:
                        fallback_mask = ((data['Location'] == marital) &
                                        (data['Policy Type'] == edu))
                        fallback_group = data.loc[fallback_mask, 'Credit Score'].dropna()
                        
                        if len(fallback_group) >= 5:
                            fill_values = np.random.choice(fallback_group, size=n_missing, replace=True)
                            fallback_count += n_missing
                        else:
                            fill_values = np.random.choice(original_income, size=n_missing, replace=True)
                            fallback_count += n_missing
                    data.loc[mask & data['Credit Score'].isna(), 'Credit Score'] = fill_values

    # Filling NaN values of health Score    
    original_income = data['Health Score'].dropna().copy()
    segment_count = 0
    imputed_count = 0
    fallback_count = 0
    for marital in data['Location'].unique():
            for edu in data['Exercise Frequency'].unique():
                for children in data['Smoking Status'].unique():
                    
                    mask = ((data['Location'] == marital) &
                        (data['Exercise Frequency'] == edu) &
                        (data['Smoking Status'] == children))
                    
                    group = data.loc[mask, 'Health Score']
                    
                    if group.isna().sum() == 0:
                        continue
                    
                    segment_count += 1
                    n_missing = group.isna().sum()
                    
                    existing_values = group.dropna()
                    
                    if len(existing_values) >= 5:  
                        fill_values = np.random.choice(existing_values, size=n_missing, replace=True)
                        imputed_count += n_missing
                    else:
                        fallback_mask = ((data['Location'] == marital) &
                                        (data['Exercise Frequency'] == edu))
                        fallback_group = data.loc[fallback_mask, 'Health Score'].dropna()
                        
                        if len(fallback_group) >= 5:
                            fill_values = np.random.choice(fallback_group, size=n_missing, replace=True)
                            fallback_count += n_missing
                        else:
                            fill_values = np.random.choice(original_income, size=n_missing, replace=True)
                            fallback_count += n_missing
                    data.loc[mask & data['Health Score'].isna(), 'Health Score'] = fill_values




data_Filling(X)
data_Filling(Xt)


X.isnull().sum()


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

ordinal_cols = [
    'Education Level', 'Location', 'Policy Type', 'Smoking Status',
    'Exercise Frequency', 'Property Type', 'Customer Feedback'
]

ordinal_categories = [
    ['High School', "Bachelor's", 'Master\'s', 'PhD'],
    ['Rural', 'Suburban', 'Urban'],
    ['Basic', 'Comprehensive', 'Premium'],
    ['No', 'Yes'],
    ['Rarely', 'Monthly', 'Weekly', 'Daily'],
    ['House', 'Condo', 'Apartment'],
    ['Poor', 'Good', 'Average'],]

onehot_cols = ['Marital Status']

transformer = ColumnTransformer(transformers=[
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), onehot_cols)
], remainder='passthrough')

X_enc = transformer.fit_transform(X)
Xt_enc = transformer.transform(Xt)

encoded_feature_names = transformer.get_feature_names_out()

X_enc = pd.DataFrame(X_enc, columns=encoded_feature_names, index=X.index)
Xt_enc = pd.DataFrame(Xt_enc, columns=encoded_feature_names, index=Xt.index)


cols = ['Education Level', 'Location', 'Policy Type',
       'Smoking Status', 'Exercise Frequency', 'Property Type',
       'Customer Feedback',
       'Status_Married', 'Status_Single',
       'Age', 'Annual Income',
       'Number of Children', 'Health Score','Previous Claims',
       'Vehicle Age', 'Credit Score',
       'Insurance Duration','Policy Start Date']

X_enc.columns = cols
Xt_enc.columns = cols


X_enc.head()


X_enc['Children_Insurance_Vehicle'] = X_enc['Number of Children'] + X_enc['Insurance Duration'] + X_enc['Vehicle Age'] 
Xt_enc['Children_Insurance_Vehicle'] = Xt_enc['Number of Children'] + Xt_enc['Insurance Duration'] + Xt_enc['Vehicle Age'] 


sns.kdeplot(X_enc['Number of Children'])
plt.show()
sns.kdeplot(X_enc['Children_Insurance_Vehicle'])
plt.show()


X_enc['Edu_Loc_Policy_Smoking'] = X_enc['Education Level'] + X_enc['Location'] + X_enc['Policy Type'] + X_enc['Smoking Status']
Xt_enc['Edu_Loc_Policy_Smoking'] = Xt_enc['Education Level'] + Xt_enc['Location'] + Xt_enc['Policy Type'] + Xt_enc['Smoking Status']


X_enc['Exer_Pro_Cust_Status'] = X_enc['Exercise Frequency'] + X_enc['Property Type'] + X_enc['Customer Feedback'] + X_enc['Status_Married'] + X_enc['Status_Single']
Xt_enc['Exer_Pro_Cust_Status'] = Xt_enc['Exercise Frequency'] + Xt_enc['Property Type'] + Xt_enc['Customer Feedback'] + Xt_enc['Status_Married'] + Xt_enc['Status_Single']


def extract_date_features(data, date_col='Policy Start Date'):
    data = data.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data['Policy_Year'] = data[date_col].dt.year
        data['Policy_Month'] = data[date_col].dt.month
        data['Policy_Quarter'] = data[date_col].dt.quarter
        data['Policy_DayOfYear'] = data[date_col].dt.dayofyear
        data['Policy_DayOfWeek'] = data[date_col].dt.dayofweek
        ref_date = pd.Timestamp('2025-10-05')
        data['Policy_Age_Days'] = (ref_date - data[date_col]).dt.days
        data.drop(columns=[date_col], inplace=True)
    return data


def create_features(data):
    data = data.copy()
    if 'Age' in data.columns and 'Smoking Status' in data.columns:
        data['Age_Smoking_Interaction'] = data['Age'] * (data['Smoking Status'] == 1.0).astype(int)
    if 'Health Score' in data.columns and 'Exercise Frequency' in data.columns:
        freq_map = {'Daily': 4, 'Weekly': 3, 'Monthly': 2, 'Rarely': 1, 'Never': 0}
        data['Health_Exercise_Interaction'] = data['Health Score'] * data['Exercise Frequency'].map(freq_map).fillna(0)
    if 'Annual Income' in data.columns and 'Number of Children' in data.columns:
        data['Income_per_Child'] = data['Annual Income'] / (data['Number of Children'] + 1)
    if 'Credit Score' in data.columns and 'Annual Income' in data.columns:
        data['Credit_to_Income_Ratio'] = data['Credit Score'] / (data['Annual Income'] + 1)
    if 'Previous Claims' in data.columns and 'Insurance Duration' in data.columns:
        data['Claims_per_Year'] = data['Previous Claims'] / (data['Insurance Duration'] + 0.1)
    if 'Vehicle Age' in data.columns:
        data['Vehicle_Age_Squared'] = data['Vehicle Age'] ** 2
    if 'Age' in data.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = [1, 2, 3, 4, 5, 6]
        data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
        data['Age_Group'] = data['Age_Group'].astype(float).fillna(0).astype(int)
    return data



X_enc = extract_date_features(X_enc)
Xt_enc = extract_date_features(Xt_enc)
X_enc = create_features(X_enc)
Xt_enc = create_features(Xt_enc)


def winsorize_outliers(df, lower=0.01, upper=0.99):
    df_win = df.copy()
    for col in df.columns:
        if not col.endswith('_missing'):  
            lower_bound = df[col].quantile(lower)
            upper_bound = df[col].quantile(upper)
            df_win[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df_win

X_enc = winsorize_outliers(X_enc, lower=0.01, upper=0.99)
Xt_enc = winsorize_outliers(Xt_enc, lower=0.01, upper=0.99)


X_enc.drop(columns=['Number of Children','Insurance Duration','Vehicle Age'], inplace=True)
Xt_enc.drop(columns=['Number of Children','Insurance Duration','Vehicle Age'], inplace=True)
X_enc.drop(columns=['Exercise Frequency','Property Type','Customer Feedback','Status_Married','Status_Single'], inplace=True)
Xt_enc.drop(columns=['Exercise Frequency','Property Type','Customer Feedback','Status_Married','Status_Single'], inplace=True)
X_enc.drop(columns=['Education Level','Location','Policy Type','Smoking Status'], inplace=True)
Xt_enc.drop(columns=['Education Level','Location','Policy Type','Smoking Status'], inplace=True)


temp = X_enc.copy()
temp_test = Xt_enc.copy()


import pandas as pd

X_enc_converted = X_enc.copy()
for col in X_enc_converted.select_dtypes(include='object').columns:
   X_enc_converted[col] = pd.to_numeric(X_enc_converted[col], errors='coerce')

X_enc_converted = X_enc_converted.fillna(0).astype(int)

print(X_enc_converted.dtypes)



import pandas as pd
Xt_enc_converted = Xt_enc.copy()
for col in Xt_enc_converted.select_dtypes(include='object').columns:
   Xt_enc_converted[col] = pd.to_numeric(Xt_enc_converted[col], errors='coerce')

Xt_enc_converted = Xt_enc_converted.fillna(0).astype(int)

print(Xt_enc_converted.dtypes)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_enc_converted,y,test_size=0.4,random_state=42)


from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Universal model evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, X_final, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    final_pred = model.predict(X_final)
    
    # Calculate metrics
    train_rmse = calculate_rmse(y_train, y_train_pred)
    test_rmse = calculate_rmse(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"\nRMSE: {train_rmse:.4f}")
    # print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"MAE:  {train_mae:.4f}")
    # print(f"Test MAE:   {test_mae:.4f}")
    print(f"R² Score:   {train_r2:.4f}")
    # print(f"Test R²:    {test_r2:.4f}")
    
    # Target variable statistics
    print("\nTarget Variable Statistics:")
    print(f"Min:   {y_train.min():.4f}")
    print(f"Max:   {y_train.max():.4f}")
    print(f"Range: {(y_train.max() - y_train.min()):.4f}")
    print(f"Mean:  {y_train.mean():.4f}")
    print(f"Std:   {y_train.std():.4f}")

    # RMSE percentage errors
    range_y = y_train.max() - y_train.min()
    rmse_pct_range = (test_rmse / range_y) * 100
    rmse_pct_mean = (test_rmse / y_train.mean()) * 100

    # print(f"\nRMSE as % of range: {rmse_pct_range:.2f}%")
    # print(f"RMSE as % of mean:  {rmse_pct_mean:.2f}%")
    
    # Return test RMSE and final predictions
    return {
        "model_name": model_name,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "predictions": final_pred
    }



dt = DecisionTreeRegressor(random_state=42, max_depth=10)
pred_dt = evaluate_model(dt, X_train, y_train, X_test, y_test,Xt_enc, "Decision Tree")


from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.01, max_depth=10, n_jobs=-1)
pred_xg = evaluate_model(xgb, X_train, y_train, X_test, y_test,Xt_enc_converted, "XGBoost")


from lightgbm import LGBMRegressor

# Initialize LightGBM model
lgb_model = LGBMRegressor(
    n_estimators=100,
    random_state=42,
    learning_rate=0.01,
    max_depth=13
)

# Evaluate using your evaluate_model() function
pred_lgb = evaluate_model(lgb_model, X_train, y_train, X_test, y_test, Xt_enc_converted, "LightGBM")


from catboost import CatBoostRegressor
cat = CatBoostRegressor(n_estimators=100, random_state=42, learning_rate=0.01, max_depth=16)
pred_cat = evaluate_model(cat, X_train, y_train, X_test, y_test,Xt_enc_converted, "CatBoost")


final_pred = (pred_cat['predictions'] + pred_dt['predictions'] + pred_lgb['predictions'] + pred_xg['predictions'])/4


test_ids = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/test.csv')['id']


submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': final_pred
})
submission.to_csv('submission.csv', index=False)


submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': pred_cat['predictions']
})
submission.to_csv('submission1.csv', index=False)


X_sampled1 = X_train.sample(n=260000, random_state=64)
y_sample1 = y_train.loc[X_sampled1.index]
cat1 = CatBoostRegressor(n_estimators=100, random_state=64, learning_rate=0.01, max_depth=16)
pred_cat1 = evaluate_model(cat1, X_sampled1, y_sample1, X_test, y_test,Xt_enc_converted, "CatBoost")


X_sampled2 = X_train.sample(n=260000, random_state=46)
y_sample2 = y_train.loc[X_sampled2.index]
cat2 = CatBoostRegressor(n_estimators=100, random_state=46, learning_rate=0.01, max_depth=16)
pred_cat2 = evaluate_model(cat2, X_sampled2, y_sample2, X_test, y_test,Xt_enc_converted, "CatBoost")


X_sampled3 = X_train.sample(n=260000, random_state=123)
y_sample3 = y_train.loc[X_sampled2.index]
cat3 = CatBoostRegressor(n_estimators=100, random_state=123, learning_rate=0.02, max_depth=16)
pred_cat3 = evaluate_model(cat3, X_sampled3, y_sample3, X_test, y_test,Xt_enc_converted, "CatBoost")


X_sampled4 = X_train.sample(n=260000, random_state=321)
y_sample4 = y_train.loc[X_sampled2.index]
cat4 = CatBoostRegressor(n_estimators=100, random_state=321, learning_rate=0.02, max_depth=16)
pred_cat4 = evaluate_model(cat4, X_sampled4, y_sample4, X_test, y_test,Xt_enc_converted, "CatBoost")


final_pred = (pred_cat1['predictions'] + pred_cat2['predictions'] + pred_cat3['predictions'] + pred_cat4['predictions'])/4


submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': final_pred
})
submission.to_csv('submission2.csv', index=False)




