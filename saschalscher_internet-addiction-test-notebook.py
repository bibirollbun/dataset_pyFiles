import numpy as np
import pandas as pd
import os
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pickle
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

#A lot of it is copied from and changed from Lennart https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368


# from lennarthaupts
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368
import pandas as pd


def calculate_weights(series):
    # Create bins for the target variable and assign weights based on frequency
    bins = pd.cut(series, bins=10, labels=False)
    weights = bins.value_counts().reset_index()
    weights.columns = ["target_bins", "count"]
    weights["count"] = 1 / weights["count"]
    weight_map = weights.set_index("target_bins")["count"].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()



def detect_outliers(df, column_name):
    # drop NaN
    df_clean = df[column_name].dropna()

    Q1 = df_clean.quantile(0.15)
    Q3 = df_clean.quantile(0.85)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # find outliers
    outliers = df_clean[(df_clean < lower_bound) | (df_clean > upper_bound)]

    return outliers



# Code for finding optimal thresholds copied from: Michael Semenoff
# https://www.kaggle.com/code/michaelsemenoff/cmi-actigraphy-feature-engineering-selection

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


def round_with_thresholds(raw_preds, thresholds):
    return np.where(
        raw_preds < thresholds[0],
        int(0),
        np.where(
            raw_preds < thresholds[1],
            int(1),
            np.where(raw_preds < thresholds[2], int(2), int(3)),
        ),
    )


def optimize_thresholds(y_true, raw_preds, start_vals=[0.5, 1.5, 2.5]):
    def fun(thresholds, y_true, raw_preds):
        rounded_preds = round_with_thresholds(raw_preds, thresholds)
        return -cohen_kappa_score(y_true, rounded_preds, weights="quadratic")

    res = minimize(fun, x0=start_vals, args=(y_true, raw_preds), method="Powell")
    assert res.success
    return res.x



# Code for finding optimal thresholds copied from: Michael Semenoff
# https://www.kaggle.com/code/michaelsemenoff/cmi-actigraphy-feature-engineering-selection

import numpy as np
from sklearn.metrics import cohen_kappa_score

base_thresholds = [30, 50, 80]


def cross_validate(
    model_,
    data,
    features,
    score_col,
    index_col,
    cv,
    sample_weights=False,
    verbose=False,
):
    kappa_scores = []
    oof_score_predictions = np.zeros(len(data))

    #score_to_index_thresholds = base_thresholds
    thresholds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        y_train_score = data[score_col].iloc[train_idx]
        y_train_index = data[index_col].iloc[train_idx]
        # y_val_score = data[score_col].iloc[val_idx]
        y_val_index = data[index_col].iloc[val_idx]

        # Train model with sample weights if provided
        if sample_weights:
            weights = calculate_weights(y_train_score)
            model_.fit(X_train, y_train_score, sample_weight=weights)
        else:
            model_.fit(X_train, y_train_score)

        y_pred_train_score = model_.predict(X_train)
        y_pred_val_score = model_.predict(X_val)

        oof_score_predictions[val_idx] = y_pred_val_score

        # Find optimal threshold in sample
        t_1 = optimize_thresholds(
            y_train_index, y_pred_train_score, start_vals=base_thresholds
        )
        thresholds.append(t_1)

        y_pred_val_index = round_with_thresholds(y_pred_val_score, t_1)

        kappa_score = cohen_kappa_score(
            y_val_index, y_pred_val_index, weights="quadratic"
        )
        kappa_scores.append(kappa_score)

        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa Score = {kappa_score}")

    if verbose:
        print(f"## Mean CV Kappa Score: {np.mean(kappa_scores)} ##")
        print(f"## Std CV: {np.std(kappa_scores)}")

    return np.mean(kappa_scores), oof_score_predictions, thresholds


def n_cross_validate(
    model_,
    data,
    features,
    score_col,
    index_col,
    cv,
    seeds,
    sample_weights=False,
    verbose=False,
):

    scores = []
    for seed in seeds:
        cv.random_state = seed
        score, oof, _ = cross_validate(
            model_,
            data,
            features,
            score_col,
            index_col,
            cv,
            sample_weights=True,
            verbose=False,
        )
        scores.append(score)
    return score, oof



df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
df_test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')


# List of BIA-columns to check for outliers
# without BIA-BIA_FMI 
columns_BIA = [
    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 
    'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 
    'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW'
]


 filtered_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight'] + columns_BIA]

# Initialize a dictionary for storing outlier data with one row per 'id'
outlier_data = {}

# Loop through columns to find outliers
for column_name in columns_BIA:
    outliers = detect_outliers(filtered_df, column_name)
    
    # Loop through outliers and add them to the dictionary
    for index, value in outliers.items():
        if filtered_df.loc[index, 'id'] not in outlier_data:
            outlier_data[filtered_df.loc[index, 'id']] = {
                'id': filtered_df.loc[index, 'id'],
                'age': filtered_df.loc[index, 'Basic_Demos-Age'],
                'sex': filtered_df.loc[index, 'Basic_Demos-Sex'],
                'height': filtered_df.loc[index, 'Physical-Height'],
                'weight': filtered_df.loc[index, 'Physical-Weight']
            }
        
        # Add the outlier value for this column
        outlier_data[filtered_df.loc[index, 'id']][column_name] = value

# Convert the dictionary to a DataFrame
outlier_df = pd.DataFrame.from_dict(outlier_data, orient='index')

outlier_df = outlier_df.sort_values(by="id")
outlier_df.to_csv('outliers_BIA.csv', index=False)


filtered_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight'] + columns_BIA]

# Initialize a dictionary for storing outlier data with one row per 'id'
outlier_data = {}

# Loop through columns to find outliers
for column_name in columns_BIA:
    outliers = detect_outliers(filtered_df, column_name)
    
    # Loop through outliers and add them to the dictionary
    for index, value in outliers.items():
        if filtered_df.loc[index, 'id'] not in outlier_data:
            outlier_data[filtered_df.loc[index, 'id']] = {
                'id': filtered_df.loc[index, 'id'],
                #'age': filtered_df.loc[index, 'Basic_Demos-Age'],
                #'sex': filtered_df.loc[index, 'Basic_Demos-Sex'],
                'height': filtered_df.loc[index, 'Physical-Height'],
                'weight': filtered_df.loc[index, 'Physical-Weight']
            }
        
        # Add the outlier value for this column
        outlier_data[filtered_df.loc[index, 'id']][column_name] = value

# Convert the dictionary to a DataFrame
outlier_df = pd.DataFrame.from_dict(outlier_data, orient='index')

outlier_df = outlier_df.sort_values(by="id")
outlier_df.to_csv('outliers_BIA_general.csv', index=False)


height_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height']]
weight_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Weight']]
# bmi_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI']] without BMI
waist_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Waist_Circumference']]
diastolic_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Diastolic_BP']]
heartrate_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-HeartRate']]
systolic_df = df[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Systolic_BP']]


 # replace 0 with NaN
height_df.loc[height_df['Physical-Height'] == 0, 'Physical-Height'] = np.nan
weight_df.loc[weight_df['Physical-Weight'] == 0, 'Physical-Weight'] = np.nan
# bmi_df.loc[bmi_df['Physical-BMI'] == 0, 'Physical-BMI'] = np.nan
waist_df.loc[waist_df['Physical-Waist_Circumference'] == 0, 'Physical-Waist_Circumference'] = np.nan
diastolic_df.loc[diastolic_df['Physical-Diastolic_BP'] == 0, 'Physical-Diastolic_BP'] = np.nan
heartrate_df.loc[heartrate_df['Physical-HeartRate'] == 0, 'Physical-HeartRate'] = np.nan
systolic_df.loc[systolic_df['Physical-Systolic_BP'] == 0, 'Physical-Systolic_BP'] = np.nan

# sort → age & sex
height_df = height_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
weight_df = weight_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
# bmi_df = bmi_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
waist_df = waist_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
diastolic_df = diastolic_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
heartrate_df = heartrate_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])
systolic_df = systolic_df.sort_values(by=['Basic_Demos-Age', 'Basic_Demos-Sex'])

# Dictionary for Outliers 
height_outliers = []
weight_outliers = []
# bmi_outliers = []
waist_outliers = []
diastolic_outliers = []
heartrate_outliers = []
systolic_outliers = []

# Group data by age and sex and applie the detect_outliers() function separately.
# outliers for height
for (age, sex), group in height_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-Height')
    for index, value in outliers.items():
        height_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'height_outlier': value
        })

# outliers for weight
for (age, sex), group in weight_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-Weight')
    for index, value in outliers.items():
        weight_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'weight_outlier': value
        })

# outliers for bmi
# for (age, sex), group in bmi_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    # outliers = detect_outliers(group, 'Physical-BMI')
    # for index, value in outliers.items():
        # bmi_outliers.append({
            #'id': group.loc[index, 'id'],
            #'age': age,
            #'sex': sex,
            #'bmi_outlier': value
        #})

# outliers for waist circumference
for (age, sex), group in waist_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-Waist_Circumference')
    for index, value in outliers.items():
        waist_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'waist_outlier': value
        })

# outliers for diastolic BP
for (age, sex), group in diastolic_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-Diastolic_BP')
    for index, value in outliers.items():
        diastolic_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'diastolic_outlier': value
        })

# outliers for heart rate
for (age, sex), group in heartrate_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-HeartRate')
    for index, value in outliers.items():
        heartrate_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'heartrate_outlier': value
        })

# outliers for systolic BP
for (age, sex), group in systolic_df.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex']):
    outliers = detect_outliers(group, 'Physical-Systolic_BP')
    for index, value in outliers.items():
        systolic_outliers.append({
            'id': group.loc[index, 'id'],
            'age': age,
            'sex': sex,
            'systolic_outlier': value
        })

# create outlier-dataframe
#height_outliers_df = pd.DataFrame(height_outliers)
#weight_outliers_df = pd.DataFrame(weight_outliers)
height_outliers_df = pd.DataFrame(height_outliers, columns=['id', 'age', 'sex', 'height_outlier'])
weight_outliers_df = pd.DataFrame(weight_outliers, columns=['id', 'age', 'sex', 'weight_outlier'])

# bmi_outliers_df = pd.DataFrame(bmi_outliers)
#waist_outliers_df = pd.DataFrame(waist_outliers)
#diastolic_outliers_df = pd.DataFrame(diastolic_outliers)
#heartrate_outliers_df = pd.DataFrame(heartrate_outliers)
#systolic_outliers_df = pd.DataFrame(systolic_outliers)

waist_outliers_df = pd.DataFrame(waist_outliers, columns=['id', 'age', 'sex', 'waist_outlier'])
diastolic_outliers_df = pd.DataFrame(diastolic_outliers, columns=['id', 'age', 'sex', 'diastolic_outlier'])
heartrate_outliers_df = pd.DataFrame(heartrate_outliers, columns=['id', 'age', 'sex', 'heartrate_outlier'])
systolic_outliers_df = pd.DataFrame(systolic_outliers, columns=['id', 'age', 'sex', 'systolic_outlier'])

# Merge outlier dataframes
outlier_df = pd.merge(height_outliers_df, weight_outliers_df, on=['id', 'age', 'sex'], how='outer')
# outlier_df = pd.merge(outlier_df, bmi_outliers_df, on=['id', 'age', 'sex'], how='outer')
outlier_df = pd.merge(outlier_df, waist_outliers_df, on=['id', 'age', 'sex'], how='outer')
outlier_df = pd.merge(outlier_df, diastolic_outliers_df, on=['id', 'age', 'sex'], how='outer')
outlier_df = pd.merge(outlier_df, heartrate_outliers_df, on=['id', 'age', 'sex'], how='outer')
outlier_df = pd.merge(outlier_df, systolic_outliers_df, on=['id', 'age', 'sex'], how='outer')

# id sort
outlier_df = outlier_df.sort_values(by='id')
outlier_df.to_csv('outliers_physicals_without_BMI.csv', index=False)

# manually check if the outliers are correct :) 


import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# copy of the dataset for modifications :)
ds_copy = df.copy()

ds = df

test_ds = df_test.copy()

# load outliers_BIA.csv
bia = pd.read_csv('outliers_BIA.csv')

# load outliers_physicals_wihtout_BMI.csv
physicals = pd.read_csv('outliers_physicals_without_BMI.csv')    


# List of columns to check and replace 0 values with NaN
columns_physical_outlier = [
    'Physical-Height', 'Physical-Weight', 'Physical-BMI', 
    'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
    'Physical-HeartRate', 'Physical-Systolic_BP'
]


# replace 0 with NaN
ds_copy[columns_physical_outlier] = ds_copy[columns_physical_outlier].replace(0, np.nan)


# Check for any remaining 0 values
zero_counts = ds_copy[columns_physical_outlier].eq(0).sum()

# Output the count of remaining zeros
print("Count of remaining 0 values in each column:")
print(zero_counts)

# check NaN values
print("\nSample rows with NaN values after replacement:")
print(ds_copy[columns_physical_outlier].isna().sum())


# List of columns to check for outliers in physicals csv
outlier_columns = ['height_outlier', 'weight_outlier', 
                   'waist_outlier', 'diastolic_outlier', 'heartrate_outlier', 
                   'systolic_outlier']


 # copy of ds_copy and set the outliers to NaN to calculate the mean
ds_copy_copy = ds_copy.copy()

# Loop over each outlier column and check the outliers for each id
for outlier_col, physical_col in zip(outlier_columns, columns_physical_outlier):
    # Get the rows where outliers are marked (i.e., not NaN in the outlier column)
    outlier_ids = physicals[physicals[outlier_col].notna()]['id'].values

    # Replace the corresponding physical measurement values with NaN in ds_copy_copy
    ds_copy_copy.loc[ds_copy_copy['id'].isin(outlier_ids), physical_col] = np.nan


# Filter the outliers_physicals dataset to check outliers for a specific ID
ds_copy_copy_filtered = ds_copy_copy[ds_copy_copy['id'] == '00e6167c']
physicals_filtered = physicals[physicals['id'] == '00e6167c']

# Check if the outliers for this ID are NaN
print("outliers replaced with NaN")
print(ds_copy_copy_filtered[['id', 'Physical-Height', 'Physical-Weight', 
                              'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
                              'Physical-HeartRate', 'Physical-Systolic_BP']])

print("outliers from outliers_phyiscals")
print(physicals_filtered)


# Calculate mean for each column grouped by age & sex
mean_values = ds_copy_copy.groupby(['Basic_Demos-Age', 'Basic_Demos-Sex'])[columns_physical_outlier].mean()


# Loop over the outlier columns and corresponding physical columns
for outlier_col, physical_col in zip(outlier_columns, columns_physical_outlier):
    # Get the rows where outliers are marked (i.e., not NaN in the outlier column)
    outlier_ids = physicals[physicals[outlier_col].notna()]['id'].values
    
    # Iterate through each id and replace the outlier values in ds_copy at the cell level
    for id_value in outlier_ids:
        # Find the corresponding mean value for this id's age and sex
        age = ds_copy.loc[ds_copy['id'] == id_value, 'Basic_Demos-Age'].values[0]
        sex = ds_copy.loc[ds_copy['id'] == id_value, 'Basic_Demos-Sex'].values[0]
        
        # Get the mean value for this age and sex group
        mean_value = mean_values.loc[(age, sex), physical_col]
        
        # Replace the outlier value in the specific cell of ds_copy with the mean value
        ds_copy.loc[(ds_copy['id'] == id_value), physical_col] = ds_copy.loc[
            (ds_copy['id'] == id_value) & 
            pd.notna(ds_copy[physical_col]), physical_col
        ].apply(lambda x: mean_value if pd.notna(x) else x)


# Filter ds_copy to check the cell value for ID 00e6167c in Physical-Height
ds_copy_filtered = ds_copy[ds_copy['id'] == '00e6167c']

# Print the filtered cell to check the updated data
print(ds_copy_filtered[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight',  
                              'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
                              'Physical-HeartRate', 'Physical-Systolic_BP']])


# calculate BMI for the original dataset

# Caclulate BMI (Umrechnung der Einheiten)
ds['BMI_Calculated'] = (ds['Physical-Weight'] * 0.453592) / ((ds['Physical-Height'] * 0.0254) ** 2)

# compare with existing values
ds['BMI_Correct'] = ds['BMI_Calculated'].round(2) == ds['Physical-BMI'].round(2)

# Round both columns to 2 decimal places and compare
incorrect_bmi = ds[ds['Physical-BMI'].notna() & ds['BMI_Calculated'].notna() &
                   (round(ds['BMI_Calculated'], 0) != round(ds['Physical-BMI'], 0))]

# Display the rows where the BMI is incorrect
incorrect_bmi[['Physical-Weight', 'Physical-Height', 'Physical-BMI', 'BMI_Calculated']]

# seems to be ok :) 


# calculate BMI for ds_copy

# Caclulate BMI (Umrechnung der Einheiten)
ds_copy['BMI_Calculated'] = (ds_copy['Physical-Weight'] * 0.453592) / ((ds_copy['Physical-Height'] * 0.0254) ** 2)

# compare with existing values
ds_copy['BMI_Correct'] = ds_copy['BMI_Calculated'].round(4) == ds_copy['Physical-BMI'].round(4)

# Round both columns to 2 decimal places and compare
incorrect_bmi = ds_copy[ds_copy['Physical-BMI'].notna() & ds_copy['BMI_Calculated'].notna() &
                   (round(ds_copy['BMI_Calculated'], 0) != round(ds['Physical-BMI'], 0))]

# Display the rows where the BMI is incorrect
incorrect_bmi[['id', 'Physical-Weight', 'Physical-Height', 'Physical-BMI', 'BMI_Calculated']]



 # For these rows, overwrite Physical-BMi with BMI_Calculated

# Iterate through the rows in ds_copy to replace incorrect Physical-BMI values with BMI_Calculated
for index, row in incorrect_bmi.iterrows():
    id_value = row['id']
    calculated_bmi = row['BMI_Calculated']
    
    # Replace the incorrect Physical-BMI value with the calculated BMI value
    ds_copy.loc[ds_copy['id'] == id_value, 'Physical-BMI'] = calculated_bmi
    

# Recheck the replaced values and filter again for any remaining incorrect BMI values
ds_copy['BMI_Correct'] = ds_copy['BMI_Calculated'].round(2) == ds_copy['Physical-BMI'].round(2)

# Filter rows where the BMI is still incorrect
incorrect_bmi_after = ds_copy[ds_copy['Physical-BMI'].notna() & ds_copy['BMI_Calculated'].notna() & 
                              (round(ds_copy['BMI_Calculated'], 0) != round(ds_copy['Physical-BMI'], 0))]

# Display the rows where the BMI is still incorrect after replacement
incorrect_bmi_after[['id', 'Physical-Weight', 'Physical-Height', 'Physical-BMI', 'BMI_Calculated']]


for index, row in incorrect_bmi_after.iterrows():
    id_value = row['id']
    calculated_bmi = row['BMI_Calculated']
    
    # Replace the incorrect Physical-BMI value with the calculated BMI value
    ds_copy.loc[ds_copy['id'] == id_value, 'Physical-BMI'] = calculated_bmi
    

# Recheck the replaced values and filter again for any remaining incorrect BMI values
ds_copy['BMI_Correct'] = ds_copy['BMI_Calculated'].round(2) == ds_copy['Physical-BMI'].round(2)

# Filter rows where the BMI is still incorrect
incorrect_bmi_after2= ds_copy[ds_copy['Physical-BMI'].notna() & ds_copy['BMI_Calculated'].notna() & 
                              (round(ds_copy['BMI_Calculated'], 0) != round(ds_copy['Physical-BMI'], 0))]

# Display the rows where the BMI is still incorrect after replacement
incorrect_bmi_after2[['id', 'Physical-Weight', 'Physical-Height', 'Physical-BMI', 'BMI_Calculated']]


# The file outliers_BIA.csv was not accurate enough, so we are manually identifying and handling the outliers in the BIA columns. :)


 # remove rows with id cedf96c5 and e252dcb6
ds_copy = ds_copy[~ds_copy['id'].isin(['cedf96c5', 'e252dcb6'])]

# BIA-BIA_BMC: If greater than 25 or less than 0.1, set NaN
ds_copy['BIA-BIA_BMC'] = ds_copy['BIA-BIA_BMC'].apply(lambda x: np.nan if x > 25 or x < 0.1 else x)

# BIA-BIA_BMI: If greater than 45 or less than 10, set NaN
ds_copy['BIA-BIA_BMI'] = ds_copy['BIA-BIA_BMI'].apply(lambda x: np.nan if x > 45 or x < 10 else x)

# BIA-BIA_BMR: If greater than 4000, set NaN
ds_copy['BIA-BIA_BMR'] = ds_copy['BIA-BIA_BMR'].apply(lambda x: np.nan if x > 4000 else x)

# BIA-BIA_DEE: If greater than 5000, set NaN
ds_copy['BIA-BIA_DEE'] = ds_copy['BIA-BIA_DEE'].apply(lambda x: np.nan if x > 5000 else x)

# BIA-BIA_ECW: If greater than 60 or less than 3, set NaN
ds_copy['BIA-BIA_ECW'] = ds_copy['BIA-BIA_ECW'].apply(lambda x: np.nan if x > 60 or x < 3 else x)

# BIA-BIA_FFMI: If greater than 40 or less than 10, set NaN
ds_copy['BIA-BIA_FFMI'] = ds_copy['BIA-BIA_FFMI'].apply(lambda x: np.nan if x > 40 or x < 10 else x)

# BIA-BIA_FMI: If less than 0, set NaN, if greater than weight, replace with weight
ds_copy['BIA-BIA_FMI'] = ds_copy.apply(lambda row: np.nan if row['BIA-BIA_FMI'] < 0 else (row['Physical-Weight'] if row['BIA-BIA_FMI'] > row['Physical-Weight'] else row['BIA-BIA_FMI']), axis=1)

# BIA-BIA_Fat: If less than 0, set NaN
ds_copy['BIA-BIA_Fat'] = ds_copy['BIA-BIA_Fat'].apply(lambda x: np.nan if x < 0 else x)

# BIA-BIA_ICW: If greater than 100, set NaN
ds_copy['BIA-BIA_ICW'] = ds_copy['BIA-BIA_ICW'].apply(lambda x: np.nan if x > 100 else x)

# BIA-BIA_LDM: If greater than 100, set NaN
ds_copy['BIA-BIA_LDM'] = ds_copy['BIA-BIA_LDM'].apply(lambda x: np.nan if x > 100 else x)

# BIA-BIA_LST: If greater than weight, replace with weight
ds_copy['BIA-BIA_LST'] = ds_copy.apply(lambda row: row['Physical-Weight'] if row['BIA-BIA_LST'] > row['Physical-Weight'] else row['BIA-BIA_LST'], axis=1)

# BIA-BIA_SMM: If less than 10 or greater than weight, set NaN
ds_copy['BIA-BIA_SMM'] = ds_copy.apply(lambda row: np.nan if row['BIA-BIA_SMM'] < 10 or row['BIA-BIA_SMM'] > row['Physical-Weight'] else row['BIA-BIA_SMM'], axis=1)

# BIA-BIA_TBW: If greater than weight, set NaN
ds_copy['BIA-BIA_TBW'] = ds_copy.apply(lambda row: np.nan if row['BIA-BIA_TBW'] > row['Physical-Weight'] else row['BIA-BIA_TBW'], axis=1)

# BIA-BIA_FFM: If greater than weight, replace with weight
ds_copy['BIA-BIA_FFM'] = ds_copy.apply(lambda row: row['Physical-Weight'] if row['BIA-BIA_FFM'] > row['Physical-Weight'] else row['BIA-BIA_FFM'], axis=1)


import numpy as np
import pandas as pd

def clean_tbw_ffm(df):

    df = df.copy()  # Work on a copy to avoid modifying the original
    
    # Check for required columns
    required_cols = ['BIA-BIA_TBW', 'BIA-BIA_FFM', 'Physical-Weight']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Warning: Missing required columns: {missing}")
        return df
    
    # First pass: mark any impossible values
    tbw_impossible = (df['BIA-BIA_TBW'] > df['Physical-Weight']) & df['BIA-BIA_TBW'].notna() & df['Physical-Weight'].notna()
    ffm_impossible = (df['BIA-BIA_FFM'] > df['Physical-Weight']) & df['BIA-BIA_FFM'].notna() & df['Physical-Weight'].notna()
    
    # Case 1: TBW is possible but FFM is impossible - recalculate FFM from TBW
    recalc_ffm = (~tbw_impossible) & ffm_impossible & df['BIA-BIA_TBW'].notna()
    if recalc_ffm.sum() > 0:
        print(f"Case 1: Recalculating {recalc_ffm.sum()} FFM values from valid TBW values")
        # Print details of each recalculation
        for idx in df[recalc_ffm].index:
            original_ffm = df.loc[idx, 'BIA-BIA_FFM']
            weight = df.loc[idx, 'Physical-Weight']
            tbw = df.loc[idx, 'BIA-BIA_TBW']
            new_ffm_raw = tbw / 0.73
            
            # Check if new FFM also exceeds weight
            if new_ffm_raw > weight:
                print(f"  Row {idx}: FFM {original_ffm:.2f} > Weight {weight:.2f}, recalculated FFM {new_ffm_raw:.2f} still > Weight, capping at {weight:.2f}")
                new_ffm = weight
            else:
                print(f"  Row {idx}: FFM {original_ffm:.2f} > Weight {weight:.2f}, recalculating from TBW {tbw:.2f} → New FFM: {new_ffm_raw:.2f}")
                new_ffm = new_ffm_raw
                
            df.loc[idx, 'BIA-BIA_FFM'] = new_ffm
    
    # Case 2: FFM is possible but TBW is impossible - recalculate TBW from FFM
    recalc_tbw = tbw_impossible & (~ffm_impossible) & df['BIA-BIA_FFM'].notna()
    if recalc_tbw.sum() > 0:
        print(f"Case 2: Recalculating {recalc_tbw.sum()} TBW values from valid FFM values")
        # Print details of each recalculation
        for idx in df[recalc_tbw].index:
            original_tbw = df.loc[idx, 'BIA-BIA_TBW']
            weight = df.loc[idx, 'Physical-Weight']
            ffm = df.loc[idx, 'BIA-BIA_FFM']
            new_tbw_raw = ffm * 0.73
            
            # Check if new TBW also exceeds weight
            if new_tbw_raw > weight:
                print(f"  Row {idx}: TBW {original_tbw:.2f} > Weight {weight:.2f}, recalculated TBW {new_tbw_raw:.2f} still > Weight, setting to NaN")
                new_tbw = np.nan
            else:
                print(f"  Row {idx}: TBW {original_tbw:.2f} > Weight {weight:.2f}, recalculating from FFM {ffm:.2f} → New TBW: {new_tbw_raw:.2f}")
                new_tbw = new_tbw_raw
                
            df.loc[idx, 'BIA-BIA_TBW'] = new_tbw
    
    # Case 3: Both are impossible - set TBW to NaN and FFM to weight
    both_impossible = tbw_impossible & ffm_impossible
    if both_impossible.sum() > 0:
        print(f"Case 3: Found {both_impossible.sum()} rows where both TBW and FFM exceed weight")
        # Print details of each case
        for idx in df[both_impossible].index:
            tbw = df.loc[idx, 'BIA-BIA_TBW']
            ffm = df.loc[idx, 'BIA-BIA_FFM']
            weight = df.loc[idx, 'Physical-Weight']
            print(f"  Row {idx}: TBW {tbw:.2f} > Weight {weight:.2f}, FFM {ffm:.2f} > Weight {weight:.2f}")
            print(f"      Setting TBW to NaN and FFM to {weight:.2f}")
        df.loc[both_impossible, 'BIA-BIA_TBW'] = np.nan
        df.loc[both_impossible, 'BIA-BIA_FFM'] = df.loc[both_impossible, 'Physical-Weight']
    
    # Final pass: ensure any missing checks are caught
    remaining_tbw_impossible = (df['BIA-BIA_TBW'] > df['Physical-Weight']) & df['BIA-BIA_TBW'].notna() & df['Physical-Weight'].notna()
    remaining_ffm_impossible = (df['BIA-BIA_FFM'] > df['Physical-Weight']) & df['BIA-BIA_FFM'].notna() & df['Physical-Weight'].notna()
    
    if remaining_tbw_impossible.sum() > 0:
        print(f"Final check: Found {remaining_tbw_impossible.sum()} remaining rows with TBW > Weight, setting to NaN")
        df.loc[remaining_tbw_impossible, 'BIA-BIA_TBW'] = np.nan
        
    if remaining_ffm_impossible.sum() > 0:
        print(f"Final check: Found {remaining_ffm_impossible.sum()} remaining rows with FFM > Weight, capping at Weight")
        df.loc[remaining_ffm_impossible, 'BIA-BIA_FFM'] = df.loc[remaining_ffm_impossible, 'Physical-Weight']
    
    return df

# Example usage
#df_cleaned = clean_tbw_ffm(df)
#ds_copy = clean_tbw_ffm(ds_copy)


# check one column
sorted_bmc = ds_copy.sort_values(by='BIA-BIA_BMC', ascending=True)

# show 10 first rows
print(sorted_bmc[['id', 'BIA-BIA_BMC']].head(10))


train_columns = set(ds_copy.columns)
test_columns = set(test_ds.columns)

only_in_train = sorted(train_columns - test_columns)
only_in_test = sorted(test_columns - train_columns)

max_len = max(len(only_in_train), len(only_in_test))
only_in_train += [''] * (max_len - len(only_in_train))
only_in_test += [''] * (max_len - len(only_in_test))


df_diff = pd.DataFrame({'Only in Train': only_in_train, 
                        ' ': [' ']*max_len,  # empty column for space
                        'Only in Test': only_in_test})

print(df_diff.to_string(index=False))


def check_sii_range(row):
    if pd.notna(row['PCIAT-PCIAT_Total']) and pd.notna(row['sii']):
        if row['sii'] == 0 and (0 <= row['PCIAT-PCIAT_Total'] < 31):
            return True
        elif row['sii'] == 1 and (31 <= row['PCIAT-PCIAT_Total'] < 50):
            return True
        elif row['sii'] == 2 and (50 <= row['PCIAT-PCIAT_Total'] < 80):
            return True
        elif row['sii'] == 3 and (80 <= row['PCIAT-PCIAT_Total'] <= 100):
            return True
    return False

ds_copy['Correct_sii'] = ds_copy.apply(check_sii_range, axis=1)

# Zeige die Zeilen, bei denen die Übereinstimmung nicht korrekt ist und keine NaN-Werte vorliegen
incorrect_rows = ds_copy[~ds_copy['Correct_sii'] & pd.notna(ds_copy['PCIAT-PCIAT_Total']) & pd.notna(ds_copy['sii'])]

# Gib die falschen Zeilen aus
print(incorrect_rows[['id', 'PCIAT-PCIAT_Total', 'sii']])

# Optional: Ausgabe der Anzahl der falschen Übereinstimmungen
incorrect_count = len(incorrect_rows)
print(f"Falsche Übereinstimmungen: {incorrect_count}")


# CGAs if over 100 to Nan
ds_copy['CGAS-CGAS_Score'] = ds_copy['CGAS-CGAS_Score'].mask(ds_copy['CGAS-CGAS_Score'] > 100)


 # from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368
def perform_pca(train, test, n_components=None, random_state=42):
    
    pca = PCA(n_components=n_components, random_state=random_state)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio of the components:\n {explained_variance_ratio}")
    print(np.sum(explained_variance_ratio))
    
    train_pca_df = pd.DataFrame(train_pca, columns=[f'PC_{i+1}' for i in range(train_pca.shape[1])])
    test_pca_df = pd.DataFrame(test_pca, columns=[f'PC_{i+1}' for i in range(test_pca.shape[1])])
    
    return train_pca_df, test_pca_df, pca


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368
def time_features(df):
    # Convert time_of_day to hours
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    # Basic features 
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum(),
    ]
    
    # Define conditions for night, day, and no mask (full data)
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    no_mask = np.ones(len(df), dtype=bool)
    
    # List of columns of interest and masks
    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [no_mask, night, day]
    
    # Helper function for feature extraction
    def extract_stats(data):
        return [
            data.mean(), 
            data.std(), 
            data.max(), 
            data.min(), 
            data.diff().mean(), 
            data.diff().std()
        ]
    
    # Iterate over keys and masks to generate the statistics
    for key in keys:
        for mask in masks:
            filtered_data = df.loc[mask, key]
            features.extend(extract_stats(filtered_data))

    return features

# Code for parallelized computation of time series data from: Sheikh Muhammad Abdullah 
# https://www.kaggle.com/code/abdmental01/cmi-best-single-model
def process_file(filename, dirname):
    # Process file and extract time features
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return time_features(df), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    # Load time series from directory in parallel
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df


# load time series
train_ts = load_time_series('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet')

test_ts = load_time_series('/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet')


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368

# drop id
df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)

# scale with standardscaler
scaler = StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

# replace missing values with mean
for c in df_train.columns:
    m = np.mean(df_train[c])
    df_train[c] = df_train[c].fillna(m)
    df_test[c] = df_test[c].fillna(m)

print(df_train.shape)

SEED = 42
df_train_pca, df_test_pca, pca = perform_pca(df_train, df_test, n_components=15, random_state=SEED)

df_train_pca['id'] = train_ts['id']
df_test_pca['id'] = test_ts['id']

train = pd.merge(ds_copy, df_train_pca, how="left", on='id')
test = pd.merge(test_ds, df_test_pca, how="left", on='id')
train.shape


import numpy as np
import pandas as pd

# --- Gender Code Definitions (Confirmed: Male=0, Female=1) ---
MALE_CODE = 0
FEMALE_CODE = 1

# --- Helper function for Safe Normalization (Unchanged) ---
def safe_normalize(value, group, sex, map_male, map_female,
                   male_code=MALE_CODE, female_code=FEMALE_CODE):
    """
    Safely normalizes a value using age group and sex-specific maps.
    Handles NaN groups, NaN values, unexpected sex codes, missing map keys, and division by zero.
    """
    if pd.isna(group) or pd.isna(value) or pd.isna(sex):
        return np.nan
    try:
        group_int = int(group) # Ensure group is int for lookup
        if sex == male_code:
            norm_value = map_male[group_int]
        elif sex == female_code:
            norm_value = map_female[group_int]
        else:
            return np.nan
        if norm_value == 0 or pd.isna(norm_value):
            return np.nan if value != 0 else 0
        return value / norm_value
    except KeyError:
        # print(f"Warning: Group {group_int} not found in normalization maps for sex code {sex}.") # Optional
        return np.nan
    except Exception as e:
        # print(f"An error occurred during normalization (Group: {group}, Sex: {sex}): {e}") # Optional
        return np.nan

def feature_engineering(df):
    """
    Performs feature engineering on the input DataFrame.
    - Creates age groups.
    - Normalizes various metrics (BMI, Grip, Fitness Tests, BIA, Physical) by age group and sex.
    - Uses updated, more plausible normalization values where available.
    - Creates aggregate features.
    - Drops original and intermediate columns.
    """
    df = df.copy() # Work on a copy

    # Drop season columns if they exist
    season_cols = [col for col in df.columns if 'Season' in col]
    if season_cols:
        df = df.drop(columns=season_cols, axis=1)

    # --- Age Grouping (Unchanged) ---
    def assign_group(age):
        if pd.isna(age) or not isinstance(age, (int, float)):
             return np.nan
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan
    df["group"] = df['Basic_Demos-Age'].apply(assign_group)

    # --- Normalization Maps (Using plausible suggestions - VALIDATE WITH SOURCE DATA & UNITS!) ---

    # Physical Measurements - ** ADDED - VALIDATE UNITS/VALUES **
    # Height (cm?)
    Height_map_male =   {0: 110, 1: 116, 2: 122, 3: 128, 4: 138, 5: 150, 6: 163, 7: 175, 8: 177} # Suggestion
    Height_map_female = {0: 109, 1: 115, 2: 121, 3: 127, 4: 137, 5: 151, 6: 160, 7: 163, 8: 164} # Suggestion
    # Weight (kg?)
    Weight_map_male =   {0: 18.5, 1: 20.5, 2: 23.0, 3: 26.0, 4: 32.0, 5: 40.0, 6: 52.0, 7: 65.0, 8: 70.0} # Suggestion
    Weight_map_female = {0: 18.0, 1: 20.0, 2: 22.5, 3: 25.0, 4: 31.0, 5: 41.0, 6: 50.0, 7: 55.0, 8: 58.0} # Suggestion
    # Waist Circumference (cm?)
    Waist_map_male =   {0: 52, 1: 54, 2: 56, 3: 58, 4: 62, 5: 67, 6: 73, 7: 78, 8: 82} # Suggestion
    Waist_map_female = {0: 51, 1: 53, 2: 55, 3: 57, 4: 61, 5: 66, 6: 70, 7: 74, 8: 76} # Suggestion

    # --- Other Maps (Existing - still need validation) ---
    # BMI (kg/m^2 ?)
    BMI_map_male =   {0: 16.3, 1: 15.9, 2: 16.1, 3: 16.8, 4: 17.3, 5: 19.2, 6: 20.2, 7: 22.3, 8: 23.6}
    BMI_map_female = {0: 15.8, 1: 15.5, 2: 15.8, 3: 16.4, 4: 17.1, 5: 18.8, 6: 19.8, 7: 21.5, 8: 22.9}
    # Grip Strength (GSD) (kg?)
    GSD_max_map_male =   {0: 7.0,  1: 9.0,  2: 11.0, 3: 13.0, 4: 18.0, 5: 24.0, 6: 32.0, 7: 38.0, 8: 42.0}
    GSD_min_map_male =   {0: 6.0,  1: 8.0,  2: 10.0, 3: 12.0, 4: 16.0, 5: 22.0, 6: 29.0, 7: 34.0, 8: 37.0}
    GSD_max_map_female = {0: 6.0,  1: 8.0,  2: 10.0, 3: 12.0, 4: 16.0, 5: 20.0, 6: 25.0, 7: 28.0, 8: 30.0}
    GSD_min_map_female = {0: 5.0,  1: 7.0,  2: 9.0,  3: 11.0, 4: 14.0, 5: 18.0, 6: 22.0, 7: 25.0, 8: 27.0}
    # Curl-ups (CU) (Reps/min?)
    cu_map_male =   {0: 2.0, 1: 4.0, 2: 7.0, 3: 10.0, 4: 14.0, 5: 18.0, 6: 22.0, 7: 24.0, 8: 25.0}
    cu_map_female = {0: 2.0, 1: 3.0, 2: 6.0, 3: 9.0,  4: 13.0, 5: 16.0, 6: 19.0, 7: 21.0, 8: 22.0}
    # Push-ups (PU) (Standard Reps?)
    pu_map_male =   {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 7.0,  5: 10.0, 6: 15.0, 7: 20.0, 8: 25.0}
    pu_map_female = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0,  5: 7.0,  6: 8.0,  7: 9.0,  8: 10.0}
    # Trunk-lifts (TL) (Inches/cm?)
    tl_map_male =   {0: 7.0, 1: 8.0, 2: 8.0, 3: 9.0, 4: 9.0,  5: 10.0, 6: 11.0, 7: 12.0, 8: 12.0}
    tl_map_female = {0: 7.0, 1: 7.0, 2: 8.0, 3: 8.0, 4: 9.0,  5: 9.0,  6: 10.0, 7: 11.0, 8: 11.0}
    # BMR (kcal/day?)
    bmr_map_male =   {0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0, 4: 1283.0, 5: 1350.0, 6: 1481.0, 7: 1519.0, 8: 1650.0}
    bmr_map_female = {0: 865.0, 1: 875.0, 2: 924.0, 3: 972.0,  4: 1203.0, 5: 1250.0, 6: 1385.0, 7: 1430.0, 8: 1560.0}
    # DEE (kcal/day?)
    dee_map_male =   {0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0, 4: 2132.0, 5: 2250.0, 6: 2528.0, 7: 2566.0, 8: 2793.0}
    dee_map_female = {0: 1400.0, 1: 1450.0, 2: 1570.0, 3: 1650.0, 4: 2000.0, 5: 2100.0, 6: 2300.0, 7: 2400.0, 8: 2600.0}
    # FFM (kg or lbs?) - ** VERIFY UNITS/VALUES **
    ffm_map_male   = {0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0, 4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0}
    ffm_map_female = {0: 40.0, 1: 41.0, 2: 46.0, 3: 51.0, 4: 57.0, 5: 72.0, 6: 88.0, 7: 98.0, 8: 105.0}
    # ECW (L?)
    ecw_map_male   = {0: 9.0, 1: 9.5, 2: 10.0, 3: 10.5, 4: 12.0, 5: 15.0, 6: 18.0, 7: 20.0, 8: 22.0}
    ecw_map_female = {0: 8.5, 1: 9.0, 2: 9.5, 3: 10.0, 4: 11.5, 5: 14.0, 6: 16.5, 7: 18.5, 8: 20.0}
    # ICW (L?)
    icw_map_male   = {0: 15.0, 1: 16.0, 2: 17.5, 3: 18.5, 4: 20.0, 5: 22.5, 6: 25.0, 7: 27.0, 8: 29.0}
    icw_map_female = {0: 14.0, 1: 15.0, 2: 16.5, 3: 17.5, 4: 19.0, 5: 21.0, 6: 23.0, 7: 25.0, 8: 27.0}
    # BMC (kg?)
    BMC_map_male =   {0: 0.8, 1: 0.9, 2: 1.1, 3: 1.3, 4: 1.6, 5: 2.0, 6: 2.5, 7: 3.0, 8: 3.2}
    BMC_map_female = {0: 0.7, 1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 1.8, 6: 2.2, 7: 2.5, 8: 2.6}
    # Fat (%)
    Fat_map_male =   {0: 16.0, 1: 15.0, 2: 15.5, 3: 16.5, 4: 17.0, 5: 16.0, 6: 15.0, 7: 14.5, 8: 15.0}
    Fat_map_female = {0: 18.0, 1: 17.5, 2: 18.0, 3: 19.0, 4: 21.0, 5: 23.0, 6: 25.0, 7: 26.0, 8: 27.0}
    # SMM (kg?)
    SMM_map_male =   {0: 10.0, 1: 11.0, 2: 12.5, 3: 14.5, 4: 17.0, 5: 21.0, 6: 26.0, 7: 30.0, 8: 31.0}
    SMM_map_female = {0: 9.5,  1: 10.0, 2: 11.5, 3: 12.5, 4: 14.5, 5: 17.0, 6: 19.5, 7: 21.5, 8: 22.5}
    # TBW (L? or kg?)
    TBW_map_male =   {0: 15.0, 1: 16.5, 2: 18.5, 3: 21.0, 4: 24.0, 5: 29.5, 6: 35.0, 7: 40.0, 8: 42.0}
    TBW_map_female = {0: 14.0, 1: 15.0, 2: 17.0, 3: 19.0, 4: 21.5, 5: 25.5, 6: 30.0, 7: 33.5, 8: 35.0}
    # FFMI (kg/m^2?)
    FFMI_map_male =   {0: 12.5, 1: 12.8, 2: 13.2, 3: 13.8, 4: 14.5, 5: 16.0, 6: 18.0, 7: 20.0, 8: 21.0}
    FFMI_map_female = {0: 12.0, 1: 12.3, 2: 12.7, 3: 13.3, 4: 14.0, 5: 15.0, 6: 16.5, 7: 17.5, 8: 18.0}
    # FMI (kg/m^2?)
    FMI_map_male =   {0: 2.5, 1: 2.2, 2: 2.4, 3: 2.8, 4: 3.0, 5: 3.2, 6: 3.5, 7: 3.8, 8: 4.0}
    FMI_map_female = {0: 2.8, 1: 2.6, 2: 2.8, 3: 3.2, 4: 3.8, 5: 4.5, 6: 5.5, 7: 6.0, 8: 6.5}


    # --- Feature Creation / Normalization ---

    # --- ADDED: Physical Measurement Normalization ---
    df["Height_norm"] = df.apply(lambda x: safe_normalize(x.get("Physical-Height"), x['group'], x['Basic_Demos-Sex'], Height_map_male, Height_map_female), axis=1)
    df["Weight_norm"] = df.apply(lambda x: safe_normalize(x.get("Physical-Weight"), x['group'], x['Basic_Demos-Sex'], Weight_map_male, Weight_map_female), axis=1)
    df["Waist_norm"] = df.apply(lambda x: safe_normalize(x.get("Physical-Waist_Circumference"), x['group'], x['Basic_Demos-Sex'], Waist_map_male, Waist_map_female), axis=1)

    # --- Existing Normalizations ---
    # BMI
    bmi_cols = ['Physical-BMI', 'BIA-BIA_BMI']
    existing_bmi_cols = [col for col in bmi_cols if col in df.columns]
    if len(existing_bmi_cols) > 0:
        df['BMI_mean'] = df[existing_bmi_cols].mean(axis=1, skipna=True)
        df['BMI_norm'] = df.apply(lambda x: safe_normalize(x.get('BMI_mean'), x['group'], x['Basic_Demos-Sex'], BMI_map_male, BMI_map_female), axis=1)
    else:
        df['BMI_norm'] = np.nan

    # FGC Zones Aggregate
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
             'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
             'FGC-FGC_TL_Zone']
    existing_zones = [zone for zone in zones if zone in df.columns]
    if existing_zones:
        df['FGC_Zones_mean'] = df[existing_zones].mean(axis=1, skipna=True)
        df['FGC_Zones_min'] = df[existing_zones].min(axis=1, skipna=True)
        df['FGC_Zones_max'] = df[existing_zones].max(axis=1, skipna=True)
    else:
        df['FGC_Zones_mean'], df['FGC_Zones_min'], df['FGC_Zones_max'] = np.nan, np.nan, np.nan

    # Grip Strength (Gender Specific)
    grip_cols = ['FGC-FGC_GSND', 'FGC-FGC_GSD']
    existing_grip_cols = [col for col in grip_cols if col in df.columns]
    if len(existing_grip_cols) > 0:
        df['GS_raw_max'] = df[existing_grip_cols].max(axis=1, skipna=True)
        df['GS_raw_min'] = df[existing_grip_cols].min(axis=1, skipna=True)
        df['GS_max_norm'] = df.apply(lambda x: safe_normalize(x.get('GS_raw_max'), x['group'], x['Basic_Demos-Sex'], GSD_max_map_male, GSD_max_map_female), axis=1)
        df['GS_min_norm'] = df.apply(lambda x: safe_normalize(x.get('GS_raw_min'), x['group'], x['Basic_Demos-Sex'], GSD_min_map_male, GSD_min_map_female), axis=1)
    else:
        df['GS_max_norm'], df['GS_min_norm'] = np.nan, np.nan

    # Fitness Tests (CU, PU, TL)
    df['CU_norm'] = df.apply(lambda x: safe_normalize(x.get('FGC-FGC_CU'), x['group'], x['Basic_Demos-Sex'], cu_map_male, cu_map_female), axis=1)
    df['PU_norm'] = df.apply(lambda x: safe_normalize(x.get('FGC-FGC_PU'), x['group'], x['Basic_Demos-Sex'], pu_map_male, pu_map_female), axis=1)
    df['TL_norm'] = df.apply(lambda x: safe_normalize(x.get('FGC-FGC_TL'), x['group'], x['Basic_Demos-Sex'], tl_map_male, tl_map_female), axis=1)

    # Reach (Min/Max)
    reach_cols = ['FGC-FGC_SRL', 'FGC-FGC_SRR']
    existing_reach_cols = [col for col in reach_cols if col in df.columns]
    if len(existing_reach_cols) > 0:
        df["SR_min"] = df[existing_reach_cols].min(axis=1, skipna=True)
        df["SR_max"] = df[existing_reach_cols].max(axis=1, skipna=True)
    else:
        df["SR_min"], df["SR_max"] = np.nan, np.nan

    # BIA Metrics Normalization
    df["BMR_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_BMR"), x['group'], x['Basic_Demos-Sex'], bmr_map_male, bmr_map_female), axis=1)
    df["DEE_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_DEE"), x['group'], x['Basic_Demos-Sex'], dee_map_male, dee_map_female), axis=1)
    if "BIA-BIA_DEE" in df.columns and "BIA-BIA_BMR" in df.columns:
        df["DEE_BMR_diff"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]
    else:
        df["DEE_BMR_diff"] = np.nan
    df["FFM_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_FFM"), x['group'], x['Basic_Demos-Sex'], ffm_map_male, ffm_map_female), axis=1)
    df["ECW_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_ECW"), x['group'], x['Basic_Demos-Sex'], ecw_map_male, ecw_map_female), axis=1)
    df["ICW_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_ICW"), x['group'], x['Basic_Demos-Sex'], icw_map_male, icw_map_female), axis=1)
    df["ECW_ICW_norm_ratio"] = np.where(
        (df["ICW_norm"].isna()) | (df["ICW_norm"] == 0) | (df["ECW_norm"].isna()),
        np.nan, df["ECW_norm"] / df["ICW_norm"]
    )
    df["BMC_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_BMC"), x['group'], x['Basic_Demos-Sex'], BMC_map_male, BMC_map_female), axis=1)
    df["Fat_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_Fat"), x['group'], x['Basic_Demos-Sex'], Fat_map_male, Fat_map_female), axis=1)
    df["SMM_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_SMM"), x['group'], x['Basic_Demos-Sex'], SMM_map_male, SMM_map_female), axis=1)
    df["TBW_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_TBW"), x['group'], x['Basic_Demos-Sex'], TBW_map_male, TBW_map_female), axis=1)
    df["FFMI_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_FFMI"), x['group'], x['Basic_Demos-Sex'], FFMI_map_male, FFMI_map_female), axis=1)
    df["FMI_norm"] = df.apply(lambda x: safe_normalize(x.get("BIA-BIA_FMI"), x['group'], x['Basic_Demos-Sex'], FMI_map_male, FMI_map_female), axis=1)


    # --- Feature Dropping ---
    # Added new raw Physical columns to the drop list
    drop_feats_candidates = [
        # Original Physical Measures + BMI mean
        'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumenference', # Added
        'Physical-BMI', 'BIA-BIA_BMI', 'BMI_mean',
        # Original Grip + raw min/max
        'FGC-FGC_GSND', 'FGC-FGC_GSD', 'GS_raw_max', 'GS_raw_min',
        # Original Fitness tests
        'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL',
        # Original Reach tests
        'FGC-FGC_SRL', 'FGC-FGC_SRR',
        # Original BIA - Consider keeping BMR/DEE if DEE_BMR_diff is used
        #'BIA-BIA_BMR', 'BIA-BIA_DEE',
        'BIA-BIA_FFM', 'BIA-BIA_ECW', 'BIA-BIA_ICW',
        'BIA-BIA_BMC', 'BIA-BIA_Fat', 'BIA-BIA_SMM', 'BIA-BIA_TBW',
        'BIA-BIA_FFMI', 'BIA-BIA_FMI', # Add these if they exist in input
        # Original Zone scores
        'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
        'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
        'FGC-FGC_TL_Zone',
        # Intermediate normalized water values (if only ratio is needed)
        'ECW_norm', 'ICW_norm'
    ]

    cols_to_drop = [col for col in drop_feats_candidates if col in df.columns]
    df = df.drop(columns=cols_to_drop, axis=1)

    return df


train = feature_engineering(train)
test = feature_engineering(test)


# remove columns 
columns_to_remove = [
    "BMI_Calculated", "BMI_Correct", "Correct_sii"
]

train = train.drop(columns=columns_to_remove, errors="ignore")
test = test.drop(columns=columns_to_remove, errors="ignore")


train.to_csv("train_after_FE.csv", index=False)
test.to_csv("test_after_FE.csv", index=False)


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368

def bin_data(train, test, columns, n_bins=10):
    import pandas as pd
    import numpy as np
    
    # Combine train and test for consistent bin edges
    combined = pd.concat([train, test], axis=0)

    bin_edges = {}

    for col in columns:
        # Compute quantile bin edges
        try:
            edges = pd.qcut(combined[col], n_bins, retbins=True, duplicates="drop")[1]
            bin_edges[col] = edges
        except ValueError:
            print(f"Skipping column {col}: not enough unique values to create {n_bins} bins.")

    # Apply the same bin edges to both train and test
    for col, edges in bin_edges.items():
        num_bins = len(edges) - 1  
        train[col] = pd.cut(
            train[col], bins=edges, labels=range(num_bins), include_lowest=True
        ).astype(float)
        test[col] = pd.cut(
            test[col], bins=edges, labels=range(num_bins), include_lowest=True
        ).astype(float)

    return train, test


columns_to_bin_updated = [
    "PAQ_A-PAQ_A_Total", # Assuming this raw feature is kept/available
    "DEE_BMR_diff",      # Correct name for the difference feature
    "ECW_ICW_norm_ratio" # Correct name for the ratio feature
    # Potentially add other non-normalized features if they exist and need binning
]

# Filter the list to only include columns that actually exist in the dataframe
# (Good practice in case some features weren't generated)
final_columns_to_bin = [col for col in columns_to_bin_updated if col in train.columns]


# Apply binning only to the selected columns
if final_columns_to_bin: # Proceed only if there are columns to bin
    train, test = bin_data(train, test, final_columns_to_bin, n_bins=10)
else:
    print("No valid columns found for binning. Skipping binning step.")
    train, test = train, test # Use the unbinned featured data


# Features to exclude, because they're not in test
exclude = ['PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
           'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
           'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
           'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
           'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
           'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id']

y_model = "PCIAT-PCIAT_Total" # Score, target for the model
y_comp = "sii" # Index, target of the competition
features = [f for f in train.columns if f not in exclude]

# Categorical features
cat_c = []

# Mapping of categorical features (already dropped in feature engineering)
for col in cat_c:
    a_map = {}
    all_unique = set(train[col].unique()) | set(test[col].unique())
    for i, value in enumerate(all_unique):
        a_map[value] = i

    train[col] = train[col].map(a_map)
    test[col] = test[col].map(a_map)
    
train = train[train["sii"].notna()] # Keep rows where target is available
train.shape


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_bins(train, columns, n_bins=10):
    # Berechne die Anzahl der Reihen, basierend auf der Anzahl der Spalten
    num_cols = len(columns)
    rows = (num_cols // 3) + (num_cols % 3 > 0)  # 3 Plots pro Reihe
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))  # 3 Spalten pro Reihe
    axes = axes.flatten()  

    for i, col in enumerate(columns):
        if col in train:
            # Bin-Grenzen berechnen (direkt aus den Daten)
            min_val, max_val = train[col].min(), train[col].max()
            bins = np.linspace(min_val, max_val, n_bins + 1)  # n_bins gleichmäßige Bins

            counts, bins, patches = axes[i].hist(train[col], bins=bins, color='#003F5C', edgecolor='white')

            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Bin-Verteilung für {col}')

            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[i].set_xticks(bin_centers)

            axes[i].set_xticklabels([f"{round(b, 2)}" for b in bin_centers], rotation=0, ha='center')

            axes[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            axes[i].grid(axis='y', linestyle='-', alpha=0.7)
            axes[i].grid(axis='x', linestyle='')

    # Falls weniger als 3*rows Plots da sind, leere Subplots ausblenden
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

# Aufrufen mit den gebinnten Spalten
plot_bins(train, final_columns_to_bin, n_bins=10)


train.to_csv("train_after_bin.csv", index=False)
test.to_csv("test_after_bin.csv", index=False)


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LassoCV


# load dataset after binning 
train = pd.read_csv('train_after_bin.csv')
test = pd.read_csv('test_after_bin.csv')


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368

# Features to exclude, because they're not in test
exclude = ['PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
           'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
           'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
           'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
           'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
           'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id']

y_model = "PCIAT-PCIAT_Total" # Score, target for the model
y_comp = "sii" # Index, target of the competition
features = [f for f in train.columns if f not in exclude]

cat_c = []

for col in cat_c:
    a_map = {}
    all_unique = set(train[col].unique()) | set(test[col].unique())
    for i, value in enumerate(all_unique):
        a_map[value] = i

    train[col] = train[col].map(a_map)
    test[col] = test[col].map(a_map)
    
train = train[train["sii"].notna()] # Keep rows where target is available
train.shape


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368

# Plot distribution of total scores which determine the sii
# Note the excess zeros -> consider other objective functions
sns.set_theme(style="whitegrid")
plt.hist(train['PCIAT-PCIAT_Total'], bins=50, color="darkorange")
plt.title('Score Distribution')
plt.show()


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368


class Impute_With_Model:
    
    def __init__(self, na_frac=0.5, min_samples=0):
        self.model_dict = {}
        self.mean_dict = {}
        self.features = None
        self.na_frac = na_frac
        self.min_samples = min_samples
        
    def find_features(self, data, feature, tmp_features):
        missing_rows = data[feature].isna()
        na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0)
        valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]
        return valid_features

    def fit_models(self, model, data, features):
        self.features = features
        n_data = data.shape[0]
        for feature in features:
            self.mean_dict[feature] = np.mean(data[feature])
        for feature in tqdm(features):
            if data[feature].isna().sum() > 0:
                model_clone = clone(model)
                X = data[data[feature].notna()].copy()
                tmp_features = [f for f in features if f != feature]
                tmp_features = self.find_features(data, feature, tmp_features)
                if len(tmp_features) >= 1 and X.shape[0] > self.min_samples:
                    for f in tmp_features:
                        X[f] = X[f].fillna(self.mean_dict[f])
                    model_clone.fit(X[tmp_features], X[feature])
                    self.model_dict[feature] = (model_clone, tmp_features.copy())
                else:
                    self.model_dict[feature] = ("mean", np.mean(data[feature]))
            
    def impute(self, data):
        imputed_data = data.copy()
        for feature, model in self.model_dict.items():
            missing_rows = imputed_data[feature].isna()
            if missing_rows.any():
                if model[0] == "mean":
                    imputed_data[feature].fillna(model[1], inplace=True)
                else:
                    tmp_features = [f for f in self.features if f != feature]
                    X_missing = data.loc[missing_rows, tmp_features].copy()
                    for f in tmp_features:
                        X_missing[f] = X_missing[f].fillna(self.mean_dict[f])
                    imputed_data.loc[missing_rows, feature] = model[0].predict(X_missing[model[1]])
        return imputed_data


# from lennarthaupts 
# https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368

# values with more the 30% missing

missing = pd.DataFrame(train.isna().sum() / len(train))
missing[missing[0] > 0.3][:60]


# Predict Missing Values with Lasso
SEED = 9365
model = LassoCV(cv=5, random_state=SEED)
imputer = Impute_With_Model(na_frac=0.4) 
# na_frac is the maximum fraction of missing values until which a feature is imputed with the model
# if there are more missing values than for example 40% then we revert to mean imputation
imputer.fit_models(model, train, features)
train = imputer.impute(train)
test = imputer.impute(test)


train.to_csv("train_cleaned.csv", index=False)
test.to_csv("test_cleaned.csv", index=False)


# load dataset
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')


SEED = 643
n_splits = 10
optimize_params = False
n_trials = 25 # n_trials for optuna 
voting = True
base_thresholds = [30, 50, 80]


def objective(trial, X, features, score_col, index_col, cv, sample_weights=False):
    params = {
        'loss_function': trial.suggest_categorical('loss_function', ['Tweedie:variance_power=1.5', 'Poisson', 'RMSE']),
        'random_state': SEED,
        'iterations': trial.suggest_int('iterations', 100, 300),
        'depth': trial.suggest_int('depth', 2, 4),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True), 
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 1e-1, log=True),  
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
    }
    model = CatBoostRegressor(**params, verbose=0)
    
    seeds = [random.randint(1, 10000) for _ in range(20)]
    score, _ = n_cross_validate(model, X, features, score_col, index_col, cv, seeds, sample_weights=True, verbose=True)
    
    return score
    
def run_optimization(X, features, score_col, index_col, n_trials=30, cv=None, sample_weights=False):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, features, score_col, index_col, cv, sample_weights), 
                   n_trials=n_trials)
    
    print("Best params for CatBoost:", study.best_params)
    print("Best score:", study.best_value)
    return study.best_params


# delete rows without "sii"
train = train[train["sii"].notna()]  


kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


# Features to exclude, because they're not in test

exclude = [
    'PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
    'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
    'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
    'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
    'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
    'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id'
]
features = [f for f in train.columns if f not in exclude]


if optimize_params:
    cat_params = run_optimization(train, features, 'PCIAT-PCIAT_Total', 'sii', n_trials=n_trials, cv=kf, sample_weights=True)


from sklearn.neural_network import MLPRegressor


cat_params = {
    'objective': 'RMSE', 
    'iterations': 238, 
    'depth': 4, 
    'learning_rate': 0.044523361750173816, 
    'l2_leaf_reg': 0.09301285673435761, 
    'subsample': 0.6902492783438681, 
    'bagging_temperature': 0.3007304771330199, 
    'random_strength': 3.562201626987314, 
    'min_data_in_leaf': 60
}

# Parameters for LGBM, XGB and CatBoost
lgb_params = {
    'objective': 'poisson', 
    'n_estimators': 295, 
    'max_depth': 4, 
    'learning_rate': 0.04505693066482616, 
    'subsample': 0.6042489155604022, 
    'colsample_bytree': 0.5021876720502726, 
    'min_data_in_leaf': 100
}

xgb_params = {'objective': 'reg:tweedie', 'num_parallel_tree': 12, 'n_estimators': 236, 'max_depth': 3, 'learning_rate': 0.04223740904479563, 'subsample': 0.7157264603586825, 'colsample_bytree': 0.7897918901977528, 'reg_alpha': 0.005335705058190553, 'reg_lambda': 0.0001897435318347022, 'tweedie_variance_power': 1.1393958601390142}

xgb_params_2 = {
    'objective': 'reg:tweedie', 
    'num_parallel_tree': 18, 
    'n_estimators': 175, 
    'max_depth': 3, 
    'learning_rate': 0.032620453423049305, 
    'subsample': 0.6155579670568023, 
    'colsample_bytree': 0.5988773292417443, 
    'reg_alpha': 0.0028895066837627205, 
    'reg_lambda': 0.002232531512636924, 
    'tweedie_variance_power': 1.1708678482038286
}

xtrees_params = {
    'n_estimators': 500, 
    'max_depth': 15, 
    'min_samples_leaf': 20, 
    'bootstrap': False
}

mlp_params = {
    'hidden_layer_sizes': (200, 150, 100, 50),  # Wider and deeper network
    'activation': 'relu',  # 'tanh' or 'logistic' could be alternatives
    'solver': 'adam',
    'alpha': 0.001,  # L2 regularization strength
    'learning_rate': 'adaptive',
    'max_iter': 1000,  # Increase iterations
    'early_stopping': True,
    'validation_fraction': 0.2,  # For early stopping
    'n_iter_no_change': 20,  # Patience for early stopping
    'batch_size': 32  # Try mini-batch training
}



# 1. Add polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
#train = poly.fit_transform(train[features])
#test = poly.transform(test[features])

# 2. Feature selection - keep only the most important features
#from sklearn.feature_selection import SelectKBest, f_regression
#selector = SelectKBest(f_regression, k=min(20, len(features)))
#train_selected = selector.fit_transform(train_scaled[features], train['PCIAT-PCIAT_Total'])
#test_selected = selector.transform(test_scaled[features])

# 3. Add more advanced scaling - try normalization or robust scaling
#from sklearn.preprocessing import RobustScaler
#robust_scaler = RobustScaler()
#train_robust = robust_scaler.fit_transform(train[features])
#test_robust = robust_scaler.transform(test[features])


class MLPRegressorWrapper(MLPRegressor):
    def fit(self, X, y, sample_weight=None, **kwargs):
        # Ignore sample_weight and call parent's fit
        return super().fit(X, y, **kwargs)


# Create new DataFrames with polynomial features
X_train_poly = poly.fit_transform(train[features])
poly_feature_names = poly.get_feature_names_out(features)

# Convert back to pandas DataFrames
train_poly = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=train.index)
train_poly['PCIAT-PCIAT_Total'] = train['PCIAT-PCIAT_Total']  # Add target variable
train_poly['sii'] = train['sii']  # Add the 'sii' column

X_test_poly = poly.transform(test[features])
test_poly = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=test.index)

# Now scale
scaler = StandardScaler()
train_poly_scaled = train_poly.copy()
test_poly_scaled = test_poly.copy()
train_poly_scaled[poly_feature_names] = scaler.fit_transform(train_poly[poly_feature_names])
test_poly_scaled[poly_feature_names] = scaler.transform(test_poly[poly_feature_names])

# Create and train MLP model
mlp_params = {
    'hidden_layer_sizes': (200, 150, 100, 50),  # Wider and deeper network
    'activation': 'relu',  # 'tanh' or 'logistic' could be alternatives
    'solver': 'adam',
    'alpha': 0.001,  # L2 regularization strength
    'learning_rate': 'adaptive',
    'max_iter': 1000,  # Increase iterations
    'early_stopping': True,
    'validation_fraction': 0.2,  # For early stopping
    'n_iter_no_change': 20,  # Patience for early stopping
    'batch_size': 32  # Try mini-batch training
}

mlp_model = MLPRegressorWrapper(**mlp_params, random_state=SEED)

# Cross-validate with the new features
score_mlp, oof_mlp, mlp_thresholds = cross_validate(
    mlp_model, train_poly_scaled, poly_feature_names, 'PCIAT-PCIAT_Total', 'sii', kf, 
    verbose=True, sample_weights=True
)

# For the final fit - use the polynomial features consistently
mlp_model.fit(train_poly_scaled[poly_feature_names], train_poly_scaled['PCIAT-PCIAT_Total'])
test_mlp = mlp_model.predict(test_poly_scaled[poly_feature_names])

# Calculate ensemble thresholds
mlp_thresholds_ens = mlp_thresholds[0]  # Or use whatever logic you have for ensemble thresholds

# Apply thresholds
test_mlp_rounded = round_with_thresholds(test_mlp, mlp_thresholds_ens)



# Create submission
submission = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")
submission['sii'] = test_mlp_rounded
submission.to_csv("submission.csv", index=False)

