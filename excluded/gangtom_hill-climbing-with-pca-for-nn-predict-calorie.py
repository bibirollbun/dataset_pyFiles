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


#%pip install optuna
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

SEED = 45
warnings.simplefilter('ignore')
DEVICE = 'gpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(SEED)



usePCA = False
useFeatureEngineering = False
#minimalFeatures = False
useFeatureCombinations = False
useTargetEncoderFeatures = False
useCategoryFeatures = False
useCategoryCombTransformations = False


def RMSLE_calc(y_actual, y_pred):
    
    try:
        if y_actual.shape[1] !=1:
            y_actual = y_actual.reshape(-1,)
        if y_pred.shape[1] != 1:
           y_pred = y_pred.reshape(-1,)
    except:
        a=1
    
    return np.sqrt(np.mean(np.power((np.log1p(y_pred) - np.log1p(y_actual)), 2)))
    #mean_sqr = np.mean(sqr)
    #return np.sqrt(mean_sqr)



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dirname = '/kaggle/input/playground-series-s5e5/'
               
train_data = pd.read_csv(dirname + '/train.csv')
calories_data = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')
test_data = pd.read_csv(dirname + '/test.csv')





print(train_data.shape)
test_data['id'] = "test_" + test_data['id'].astype(str)
test_data['Calories'] = -1
print(calories_data[['User_ID', 'Gender']].describe(include='all'))
calories_data['id'] = calories_data['User_ID']
calories_data['Sex'] = calories_data['Gender']
calories_data.drop(columns=['User_ID', 'Gender'], inplace=True)
train_data = pd.concat([train_data, calories_data, test_data], axis=0)
train_data.reset_index(inplace=True, drop=True)
print(train_data.shape)


# Display the first few rows of the dataframe
print("First 5 rows of the training data:")
print(train_data.head())

# Get information about the dataframe
print("\nInformation about the training data:")
print(train_data.info())

# Describe the numerical columns
print("\nDescriptive statistics of numerical columns:")
print(train_data.describe())

# Check for missing values
print("\nMissing values in the training data:")
print(train_data.isnull().sum())



#Split Train, Valid & Test sets

valid_prop = 0.15
tr_index = [random.random() > valid_prop for idx in train_data.loc[train_data.Calories !=-1, :].index] + [True for idx in train_data.loc[train_data.Calories ==-1, :].index]
len(tr_index)
print(pd.Series(tr_index).value_counts())
print("Only Validation Set:")
print(pd.Series(np.invert(tr_index)).value_counts())
print("Only Test Set:")
print(pd.Series(((tr_index)) & (train_data.Calories ==-1)).value_counts())
print("Only Train Set:")
print(pd.Series(((tr_index)) & (train_data.Calories !=-1)).value_counts())
print("Train + Validation Set:")
print(pd.Series((train_data.Calories !=-1)).value_counts())

val_index = pd.Series(np.invert(tr_index))
test_index = pd.Series(((tr_index)) & (train_data.Calories ==-1))
train_index = pd.Series(((tr_index)) & (train_data.Calories !=-1))

train_val_index = pd.Series((train_data.Calories !=-1))





call_num = 0

def Age_brackets(df):
    if df['Age'] < 18:
        return ["Under 18", 1]
    elif (df['Age'] >= 18) & (df['Age'] <=25):
        return ["18-25", 2]
    elif (df['Age'] > 25) & (df['Age'] <= 35):
        return ["25-35", 3]
    elif (df['Age'] > 35) & (df['Age'] <= 45):
        return ["35-45", 4]
    elif (df['Age'] > 45) & (df['Age'] <= 55):
        return ["45-55", 5]
    elif (df['Age'] > 55) & (df['Age'] <= 65):
        return ["55-65", 6]
    elif (df['Age'] > 65) & (df['Age'] <= 75):
        return ["65-75", 7]
    else:
        return ["75 and over", 8]
    

def Get_decile_brackets(val, max, min):
    quantile_width = ((max - min)/10.0)
    quantile_borders = [min + quantile_width*i for i in range(0, 11, 1)]
    #global call_num
    #call_num+=1
    #print(call_num)


    for i in range(1, 11, 1):
        if (val > quantile_borders[i-1]) & (val <= quantile_borders[i]):
            return (format(f"{int(quantile_borders[i-1])}-{int(quantile_borders[i])}"), i)
        elif val == quantile_borders[0]:
            return (format(f"{int(quantile_borders[0])}-{int(quantile_borders[1])}"), 1)
    
    return (None, None)
    
#train_data[["Height_Bracket", "Height_Category"]] = train_data[["Height"]].apply(Get_quantile_brackets, axis=1, result_type='expand')

#Height_brackets(train_data)




def target_encode(df, train_row_index, val_row_index, test_row_index, cols_to_encode, target_col):
    encoder = ce.TargetEncoder()
    for col in cols_to_encode:
        # print(df.loc[train_row_index, col])
        df.loc[train_row_index, col + "_te"] = encoder.fit_transform(df.loc[train_row_index, [col]], df.loc[train_row_index, target_col])
        df.loc[val_row_index, col + "_te"] = encoder.transform(df.loc[val_row_index, [col]])
        df.loc[test_row_index, col + "_te"] = encoder.transform(df.loc[test_row_index, [col]])
    return df



#Basic Feature mapping & some obvious derived features
gender_map = {"male": 0, "female": 1}
train_data["Sex"] = train_data["Sex"].astype(str).map(gender_map)
train_data["Sex_Reversed"] = 1-train_data["Sex"]
train_data["BMI"] = train_data["Weight"]*1.0/np.power(train_data["Height"]/100.0, 2)
train_data['Intensity'] = train_data["Heart_Rate"]/train_data["Duration"]
train_data['Heart_rate_pct_of_max'] = train_data["Heart_Rate"]*100.0/(220.0 - train_data['Age'])

#Derived features using 
train_data[["Age_Bracket", "Age_Category"]] = train_data[["Age"]].apply(Age_brackets, axis=1, result_type='expand')
bracket_cols = ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for col in list(set(bracket_cols).difference(['Age'])):
    min_val = train_data[col].min()
    max_val = train_data[col].max()        
    train_data[col + "_Bracket"], train_data[col + "_Category"] = zip(*[Get_decile_brackets(val, max_val, min_val) for val in train_data[col].to_numpy()])

if useFeatureEngineering:
    train_data["Prod_Age_Height"] = train_data["Age"] * train_data["Height"]
    train_data["Prod_Age_Weight"] = train_data["Age"] * train_data["Weight"]
    train_data["Prod_Height_Duration"] = train_data["Duration"] * train_data["Height"]
    train_data["Prod_Weight_Duration"] = train_data["Duration"] * train_data["Weight"]
    train_data["Prod_Duration_HeartRate"] = train_data["Duration"] * train_data["Heart_Rate"]
    train_data["Prod_Duration_BodyTemp"] = train_data["Duration"] * train_data["Body_Temp"]
    train_data["Prod_HeartRate_BodyTemp"] = train_data["Heart_Rate"] * train_data["Body_Temp"]
    train_data["Sex_Duration"] = train_data["Duration"] * train_data["Sex"]
    train_data["Sex_BodyTemp"] = train_data["Body_Temp"] * train_data["Sex"]
    train_data["Sex_HeartRate"] = train_data["Heart_Rate"] * train_data["Sex"]

import itertools
if useCategoryCombTransformations:
    AllCombs = []
    for i in range(2, len(bracket_cols)+1, 1):
        AllCombs.extend(itertools.combinations(bracket_cols, i))
    
    print(f"Number of combinations are: {len(AllCombs)}")

    for combs in AllCombs:
        print(combs)
        col_n = "_".join(list(combs)) + "_Brackets"
        train_data[col_n] = ""
        for i in range(len(combs)):
            train_data[col_n] = train_data[col_n] + "_" + train_data[combs[i] + "_Bracket"]

        
numeric_cols = train_data.select_dtypes(include=np.number).columns.tolist()
object_cols = train_data.select_dtypes(include=np.object_).columns.tolist()



# Visualize the distribution of the target variable (Calories)
plt.figure(figsize=(8, 6))
sns.histplot(train_data.loc[train_data['Calories'] != -1,'Calories'], kde=True)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()
plt.close()


# Visualize the correlation matrix
correlation_matrix = train_data.loc[(train_data.id.astype(str).str.contains("test") == False), numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()



# Scatter plots of features vs. Calories
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
fig, axes = plt.subplots(nrows = 4, ncols=2, figsize = (12, 12))
axes_flat = axes.flatten()
for feature, ax in zip(features, axes_flat):
    #plt.figure(figsize=(16, 12))
    if feature == 'Gender':
        sns.boxplot(x=feature, y='Calories', data=train_data.loc[train_data.Calories !=-1,:], ax=ax)
    else:
        sns.scatterplot(x=feature, y='Calories', data=train_data.loc[train_data.Calories !=-1,:], ax=ax)
    ax.set_title(f'{feature} vs. Calories')
#    ax.xlabel(feature)
plt.tight_layout()
plt.show()
plt.close()


# Visualize the distributions of all numeric columns
numeric_cols.remove('Calories')  # Remove the target variable
#numeric_cols.remove('id') #Remove id
numeric_cols.remove('Sex') #Remove Sex

fig, axes = plt.subplots(nrows = 3, ncols=2, figsize = (12, 8))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, numeric_cols):
    #plt.figure(figsize=(10, 5))
    sns.histplot(train_data[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
    
plt.show()
plt.tight_layout()
plt.close()


# Apply transformations to make distributions more Gaussian
train_data_transform = train_data.copy()
train_data_transform['Age_lt'], _ = stats.boxcox(train_data['Age'].astype(float))
train_data_transform['Height_lt'], _ = stats.boxcox(train_data['Height'])
train_data_transform['Weight_lt'], _ = stats.boxcox(train_data['Weight'])
train_data_transform['Duration_lt'],_ = stats.boxcox(train_data['Duration'])
train_data_transform['Heart_Rate_lt'], _ = stats.boxcox(train_data['Heart_Rate'])
train_data_transform['Body_Temp_lt'] = train_data['Body_Temp']
train_data_transform.loc[train_data.Calories != -1, 'Calories_lt'] = np.log1p(train_data.loc[train_data.Calories != -1, 'Calories'])
#train_data_transform.loc[train_data.Calories !=-1, 'Calories_lt'], calories_boxcox_lambda = stats.boxcox(train_data.loc[train_data.Calories !=-1, 'Calories'])


#Scale the data
stdscaler = StandardScaler()
mmscaler = MinMaxScaler()

mmscaler_cols = []#'Age_lt', 'Body_Temp_lt', 'BMI']

#stdscaler_cols = ['Height_lt', 'Weight_lt', 'Duration_lt', 'Heart_Rate_lt', 'Prod_Age_Height', 'Prod_Age_Weight', 'Prod_Height_Duration', 'Prod_Weight_Duration', 'Prod_Duration_HeartRate', 'Prod_Duration_BodyTemp', 'Prod_HeartRate_BodyTemp']

#Don't scale Calories column
numeric_cols = list(set(train_data_transform.select_dtypes(include=np.number).columns.tolist()).difference(['Calories_lt', 'Calories']))
#numeric_cols = train_data_transform.select_dtypes(include=np.number).columns.tolist()
stdscaler_cols = list(set(numeric_cols).difference(set(mmscaler_cols)))

print("Columns scaled using MinMaxScaler: ", mmscaler_cols)
print("Columns scaled using MinMaxScaler: ", stdscaler_cols)

for col in mmscaler_cols:
    train_data_transform[col] = mmscaler.fit_transform(train_data_transform[col].to_numpy().reshape(-1, 1))

for col in stdscaler_cols:
    train_data_transform[col] = stdscaler.fit_transform(train_data_transform[col].to_numpy().reshape(-1, 1))


object_cols = train_data.select_dtypes(include=np.object_).columns.tolist()
train_data_transform = target_encode(train_data_transform, train_row_index=train_index, val_row_index=val_index, test_row_index=test_index, cols_to_encode= list(set(object_cols).difference(['id'])), target_col='Calories_lt')

"""
train_data_transform['Age_lt'] = mmscaler.fit_transform(train_data_transform['Age_lt'].to_numpy().reshape(-1, 1))
train_data_transform['Height_lt'] = stdscaler.fit_transform(train_data_transform['Height_lt'].to_numpy().reshape(-1, 1))
train_data_transform['Weight_lt'] = stdscaler.fit_transform(train_data_transform['Weight_lt'].to_numpy().reshape(-1, 1))
train_data_transform['Duration_lt'] = stdscaler.fit_transform(train_data_transform['Duration_lt'].to_numpy().reshape(-1, 1))
train_data_transform['Heart_Rate_lt'] = stdscaler.fit_transform(train_data_transform['Heart_Rate_lt'].to_numpy().reshape(-1, 1))
train_data_transform['Body_Temp_lt'] = mmscaler.fit_transform(train_data_transform['Body_Temp_lt'].to_numpy().reshape(-1, 1))
train_data_transform['BMI'] = mmscaler.fit_transform(train_data_transform['BMI'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Age_Height'] = stdscaler.fit_transform(train_data_transform['Prod_Age_Height'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Age_Weight'] = stdscaler.fit_transform(train_data_transform['Prod_Age_Weight'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Height_Duration'] = stdscaler.fit_transform(train_data_transform['Prod_Height_Duration'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Weight_Duration'] = stdscaler.fit_transform(train_data_transform['Prod_Weight_Duration'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Duration_HeartRate'] = stdscaler.fit_transform(train_data_transform['Prod_Duration_HeartRate'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_Duration_BodyTemp'] = stdscaler.fit_transform(train_data_transform['Prod_Duration_BodyTemp'].to_numpy().reshape(-1, 1))
train_data_transform['Prod_HeartRate_BodyTemp'] = stdscaler.fit_transform(train_data_transform['Prod_HeartRate_BodyTemp'].to_numpy().reshape(-1, 1))
"""


compare_cols = {"Age":"Age_lt", "Height":"Height_lt", "Duration":"Duration_lt", "Heart_Rate":"Heart_Rate_lt", "Body_Temp":"Body_Temp_lt", "Calories":"Calories_lt"}

flg, axes = plt.subplots(nrows=len(compare_cols), ncols=2, figsize=(12, 18))
axes_flat = axes.flatten()

# Visualize distributions after transformation
i = 0
for col, tr_col in compare_cols.items():
    #plt.figure(figsize=(10, 5))
    if col == 'Calories':
        sns.histplot(train_data.loc[train_data.Calories !=-1, col], kde=True, ax = axes_flat[i])
        sns.histplot(train_data_transform.loc[train_data.Calories !=-1, tr_col], kde=True, ax = axes_flat[i+1]) 
    else:
        sns.histplot(train_data[col], kde=True, ax = axes_flat[i])
        sns.histplot(train_data_transform[tr_col], kde=True, ax = axes_flat[i+1])
    axes_flat[i].set_title(f'Distribution of {col} (Actual)')
    axes_flat[i+1].set_title(f'Distribution of {col} (Transformed)')
    i+=2

plt.tight_layout()       
plt.show()
plt.close()



from sklearn.linear_model import LinearRegression
def get_blend_weights(valids, actual_valids):
    print("Getting blended weights using LinearRegression...")
    #actual_valids = actual_valids.reset_index().iloc[:, 1:]
    #print(actual_valids.head())

    #valids is a list of lists of dimensions a x 1
    colms = list(range(len(valids))) + ['actual_score']
    num_items = -1

    if not isinstance(valids, list):
        num_items = 1
        valids = [valids]
    else:
        num_items = len(valids)
    
    if num_items == -1:
        raise ValueError("No validation scores provided")
    
    dfv = pd.DataFrame(valids[0], columns=[str(colms[0])])

    i=1
    for col in colms[1:-1]:
        dfv[str(col)] = valids[i]
        i+=1
    
    dfv['actual_score'] = actual_valids
    
    #inp = pd.DataFrame(np.transpose(np.array(list(valids)+ [actual_valids])), columns = columns)

    lr2 = LinearRegression(fit_intercept=False)
    #inp = pd.DataFrame(np.transpose([xgb_valids, lgb_valids, cat_valids, actual_valids.to_numpy()]), columns=['rgb_predict', 'lgb_predict', 'cat_predict', 'actual_values'])
    #select_rand_1 = np.array([random.uniform(0, 1) >0.35 for _ in range(len(dfv.index))])
    print(dfv.head())
    #print(dfv.iloc[:, -1].head())

    X = dfv.iloc[:, :-1]
    y = dfv.iloc[:, -1]

    #lr2_trainX = dfv.iloc[dfv.index[select_rand_1], :-1]
    #lr2_trainy = dfv.iloc[dfv.index[select_rand_1], -1:]

    #lr2_validX = dfv.iloc[dfv.index[~select_rand_1], :-1]
    #lr2_validy = dfv.iloc[dfv.index[~select_rand_1], -1:]

    print(X.head())
    print(y.head())

    print("Nulls in X ", X[X.isnull()].shape[0])
    print("Nulls in y ", y[y.isnull()].shape[0])
    lr2.fit(X, y)
    y_pred = lr2.predict(X)
    #y_pred_lr2 = lr2.predict(lr2_validX)
    
    #print("Coeffs: ", lr2.coef_)

    print("Weights: ", " ; ".join([str(it) for it in list(lr2.coef_)]))

    print("Predicted RMSLE: ", RMSLE_calc(actual_valids, np.array(y_pred).reshape(-1, 1)))
    return tuple(lr2.coef_)


# ======================= Blending ======================= #
def blend_predictions(preds_list, weights):
    print("Weights given: ", weights)
    result = pd.DataFrame()
    if weights is not None:
        if len(preds_list) == len(weights):
            result = preds_list[0]*weights[0]
            for i in range(1, len(weights)):
                result += preds_list[i] * weights[i]
            return result
        raise ValueError("No weights or length of weights and predictions lists are not equal")


# ======================= Save Submission ======================= #
def save_submission(preds, filename="submission.csv"):    
    sample_submission = pd.read_csv(dirname + "/sample_submission.csv")
    sample_submission['Calories'] = preds
    sample_submission.to_csv(filename, index=False)
    print(f"ðŸš€ Submission file saved as: {filename}")



print(train_data_transform.columns)

x_cols_masterset = ['Sex', 'Age_lt', 'Height_lt', 'Weight_lt', 'Duration_lt',
              'Heart_Rate_lt', 'Body_Temp_lt', 'BMI', 'Prod_Age_Height', 'Prod_Age_Weight',
              'Prod_Height_Duration', 'Prod_Weight_Duration',
              'Prod_Duration_HeartRate', 'Prod_Duration_BodyTemp',
              'Prod_HeartRate_BodyTemp', 'Weight_Bracket_te',
       'Duration_Bracket_te', 'Height_Bracket_te', 'Age_Bracket_te', 'Age_Category', 'Height_Category', 'Weight_Category', 'Duration_Category', 'Heart_Rate_Bracket_te', 'Body_Temp_Bracket_te', 'Heart_Rate_Category', 'Body_Temp_Category']

basic_feature_set = ['Sex', 'Age_lt', 'Height_lt', 'Weight_lt', 'Duration_lt',
              'Heart_Rate_lt', 'Body_Temp_lt', 'Sex_Reversed', 'Intensity', 'BMI', 'Heart_rate_pct_of_max']#, 'BMI']
feature_combinations_set = ['Prod_Age_Height', 'Prod_Age_Weight',
              'Prod_Height_Duration', 'Prod_Weight_Duration',
              'Prod_Duration_HeartRate', 'Prod_Duration_BodyTemp',
              'Prod_HeartRate_BodyTemp', 'Sex_Duration', 'Sex_BodyTemp', 'Sex_HeartRate']
encoder_feature_set = ['Weight_Bracket_te',
       'Duration_Bracket_te', 'Height_Bracket_te', 'Age_Bracket_te', 'Heart_Rate_Bracket_te', 'Body_Temp_Bracket_te']
category_feature_set = ['Age_Category', 'Height_Category', 'Weight_Category', 'Duration_Category',  'Heart_Rate_Category', 'Body_Temp_Category']
category_combinations_feature_set = [col for col in train_data_transform.columns if col.__contains__("_Brackets_te")]

x_cols_inclset = basic_feature_set


if useFeatureCombinations:
       x_cols_inclset += feature_combinations_set

if useTargetEncoderFeatures:
       x_cols_inclset += encoder_feature_set

if useCategoryFeatures:
       x_cols_inclset += category_feature_set

if useCategoryCombTransformations:
       x_cols_inclset += category_combinations_feature_set

# if useFeatureEngineering:
#        x_cols_inclset = ['Sex', 'Age_lt', 'Height_lt', 'Weight_lt', 'Duration_lt',
#               'Heart_Rate_lt', 'Body_Temp_lt', 'BMI', 'Prod_Age_Height', 'Prod_Age_Weight',
#               'Prod_Height_Duration', 'Prod_Weight_Duration',
#               'Prod_Duration_HeartRate', 'Prod_Duration_BodyTemp',
#               'Prod_HeartRate_BodyTemp', 'Weight_Bracket_te',
#        'Duration_Bracket_te', 'Height_Bracket_te', 'Age_Bracket_te'] #, 'Age_Category', 'Height_Category', 'Weight_Category', 'Duration_Category']
#        x_cols_reducedset = [x for x in x_cols_inclset if 'Prod_' not in x]
# else:
#        x_cols_inclset = ['Sex', 'Age_lt', 'Height_lt', 'Weight_lt', 'Duration_lt',
#               'Heart_Rate_lt', 'Body_Temp_lt', 'BMI', 'Weight_Bracket_te',
#        'Duration_Bracket_te', 'Height_Bracket_te', 'Age_Bracket_te'] #, 'Age_Category', 'Height_Category', 'Weight_Category', 'Duration_Category']
#        x_cols_reducedset = [x for x in x_cols_inclset if 'Prod_' not in x]

# if minimalFeatures:
#        x_cols_inclset = ['Sex', 'Age_lt', 'Height_lt', 'Weight_lt', 'Duration_lt',
#               'Heart_Rate_lt', 'Body_Temp_lt', 'Weight_Bracket_te',
#        'Duration_Bracket_te', 'Height_Bracket_te', 'Age_Bracket_te']

x_cols = x_cols_inclset
#train_data_transform['Calories_lt_pred'] = -1000
#train_data_transform.loc[train_data['Calories'] != -1, 'Calories_lt_pred'] = np.log(1+train_data_transform.loc[train_data['Calories'] != -1, 'Calories_lt'])
y_cols = ['Calories_lt']

print(x_cols)
print(y_cols)


len(x_cols)


# if useFeatureEngineering:
#     n_components = 10
# else:
#     n_components = 5

n_components = len(x_cols)

if len(x_cols) > 30:
    n_components = 30

princomp = PCA(n_components=n_components)
train_pcaed = pd.DataFrame(princomp.fit_transform(train_data_transform.loc[:, x_cols].dropna(inplace=False).to_numpy()), index=train_data.index)
print(pd.DataFrame(train_pcaed).head(30))


print((princomp.explained_variance_ratio_*100))


#rfr_result_list.append((select_cols, "RFR", val_rfr_predict_run, test_rfr_predict_run, loss_val))

def Save_Predictions(test_df, val_df = None, filename=None):
    
    if filename is None:
        filename = "ensemble_preds" + "_" + datetime.now().strftime("%m%d%Y_%H%M%S_IST") + ".csv"
    if val_df is not None:
        val_filename = "val_" + filename
    test_filename = "test_" + filename
    #sample_submission = pd.read_csv(dir_path + "/sample_submission.csv")
    #sample_submission['Calories'] = preds
    test_df.to_csv(dir_path + "/" + test_filename, index=True)
    if val_df is not None:
        val_df.to_csv(dir_path + "/" + val_filename, index=True)
    print(f"ðŸš€ Submissions files saved as: test_{filename}. If val_ is provided, val_{filename} is also saved.")

def Save_Preds_To_File(result_tuple):
    if result_tuple is None:
        raise ValueError("The results list / tuple is empty.")
    i = 0
    val_res_df = None
    test_res_df = None
    for tple in result_tuple:
        col_name = tple[1] + "_" + str(i).zfill(3)+"_" + str(round(tple[4], 7))
        if val_res_df is None:
            val_res_df = pd.DataFrame(data=pd.Series(np.exp(tple[2])-1), columns=[col_name])
        else:
            val_res_df[col_name] = np.exp(tple[2])-1
        
        if test_res_df is None:
            test_res_df = pd.DataFrame(data=pd.Series(np.exp(tple[3])-1), columns=[col_name])
        else:
            test_res_df[col_name] = np.exp(tple[3])-1
        
        i+=1


    val_res_df.index = y_valid.index
    val_res_df['actual_values'] = np.exp(y_valid)-1


    test_res_df.index = X_test.index
    #ExecuteHillClimbing(val_res_df, model_pred_col_names=list(val_res_df.columns.difference(['actual_values'])), actual_col_name="actual_values")

    Save_Predictions(test_df = test_res_df, val_df = val_res_df)


def train_LR_model(X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")

    lr = LinearRegression()
    #Train on train set only
    lr.fit(X_train_data, y_train_data)
    if X_valid_data is not None:
        val_lr_predict = lr.predict(X_valid_data)
    if X_test_data is not None:
        test_lr_predict = lr.predict(X_test_data)
    
    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"LR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(val_lr_predict.reshape(-1, 1))-1))
    
    if (X_valid_data is not None) & (X_test_data is not None):
        return val_lr_predict, test_lr_predict
    elif (X_test_data is not None):
        return None, test_lr_predict
    elif (X_valid_data is not None):
        return val_lr_predict, None
    else:
        return None, None
    


if usePCA:
    X_train = train_pcaed[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed[val_index, :]
    X_test = train_pcaed[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]

X_train_all_data = train_data_transform.loc[train_val_index, x_cols]
y_train_all_data = train_data_transform.loc[train_val_index, y_cols]

X_train_all_data_pca = train_pcaed.loc[train_val_index, :]

val_lr_predict, test_lr_predict = train_LR_model(X_train_data=X_train, y_train_data=y_train, X_valid_data=X_valid, y_valid_data=y_valid, X_test_data=X_test)



def train_RFR_model(X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")
        
    # rfr = RandomForestRegressor(criterion='squared_error', max_depth=30, min_samples_leaf = 2, min_samples_split= 10, n_estimators=700, n_jobs=-1, verbose=1)
    rfr = RandomForestRegressor(criterion='squared_error', max_depth=20, min_samples_leaf = 4, min_samples_split= 20, n_estimators=700, n_jobs=-1, verbose=1)
    rfr.fit(X_train_data, y_train_data.to_numpy().reshape(-1,))

    #rfr.fit(train_data_transform.loc[train_index, x_cols], train_data_transform.loc[train_index, y_cols].to_numpy().reshape(-1,))
    #RFR Validation RMSLE:  0.06080167133579928

    if (X_valid_data is not None):
        val_rfr_predict = rfr.predict(X_valid_data)
    
    if (X_test_data is not None):
        test_rfr_predict = rfr.predict(X_test_data)

    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"RFR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(val_rfr_predict.reshape(-1, 1))-1))
    
    if (X_valid_data is not None) & (X_test_data is not None):
        return val_rfr_predict, test_rfr_predict
    elif (X_test_data is not None):
        return None, test_rfr_predict
    elif (X_valid_data is not None):
        return val_rfr_predict, None
    else:
        return None, None




if usePCA:
    X_train = train_pcaed[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed[val_index, :]
    X_test = train_pcaed[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]

val_rfr_predict, test_rfr_predict = train_RFR_model(X_train_data=X_train, y_train_data=y_train, X_valid_data=X_valid, y_valid_data=y_valid, X_test_data=X_test)




# def scheduler(epoch, lr):
#      if epoch < 400:
#          return 0.005
#      elif epoch == 1000:
#          return 0.002
#      else: 
#          return lr*0.999


# if usePCA:
#     X_train = train_pcaed[train_index, :]
#     y_train = train_data_transform.loc[train_index, y_cols]
#     X_valid = train_pcaed[val_index, :]
#     X_test = train_pcaed[test_index, :]
#     y_valid = train_data_transform.loc[val_index, y_cols]
# else:
#     X_train = train_data_transform.loc[train_index, x_cols]
#     y_train = train_data_transform.loc[train_index, y_cols]
#     X_valid = train_data_transform.loc[val_index, x_cols]
#     X_test = train_data_transform.loc[test_index, x_cols]
#     y_valid = train_data_transform.loc[val_index, y_cols]

# def create_ann_model(learning_rate, hidden_units1, hidden_units2, hidden_units3, patience):

# # Build the neural network model
#     model = keras.Sequential([
#         layers.Dense(hidden_units1, activation='relu', input_shape=[train_data_transform.loc[train_index, x_cols].shape[1]],
#                     kernel_regularizer=regularizers.l2(0.00001)),
#         layers.Dense(hidden_units2, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
#         layers.Dense(hidden_units3, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
#         layers.Dense(1)
#     ])

#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor='val_rmse',  # Monitor validation loss
#         patience=patience,        # Stop after 10 epochs of no improvement
#         restore_best_weights=True  # Keep the weights of the best epoch
#     )

#     # Compile the model
#     model.compile(loss='mse',  # Use MSE for optimization, RMSE is just the square root
#                 optimizer=tf.keras.optimizers.Adam(learning_rate),
#                 metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    
#     return model, early_stopping


# def train_model(model, early_stopping):

#     lr_callback = keras.callbacks.LearningRateScheduler(scheduler)

    
#     # Train the model
#     history = model.fit(X_train, y_train.to_numpy().reshape(-1,), epochs=500, batch_size=64000, validation_split=0.15, callbacks=[early_stopping, lr_callback])

#     # Evaluate on the validation set (after early stopping)
#     val_loss, val_mae, val_mse, val_rmse = model.evaluate(X_valid, y_valid.to_numpy().reshape(-1,), verbose=0)

#     return val_rmse


# # Define the objective function
# def objective(trial):
#     # Suggest hyperparameters (e.g., learning rate, number of hidden layers)
#     learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
#     hidden_units1 = trial.suggest_int("hidden_units1", 4, 24, step = 4)
#     hidden_units2 = trial.suggest_int("hidden_units2", 4, 24, step = 4)
#     hidden_units3 = trial.suggest_int("hidden_units3", 4, 24, step = 4)
#     patience = trial.suggest_int("patience", 10, 40, step = 10)
#     # ... other hyperparameters

#     # Create the ANN model (using PyTorch or TensorFlow)
#     model, es = create_ann_model(learning_rate, hidden_units1, hidden_units2, hidden_units3, patience)

#     # Train the model
#     train_loss = train_model(model, es)
    
#     # Return the objective value (e.g., validation loss)
#     return train_loss

# # Create a study
# study = optuna.create_study(direction="minimize")

# # Run optimization
# study.optimize(objective, n_trials=10)

# # Get the best hyperparameters
# best_params = study.best_params

# # Get the best objective value
# best_objective_value = study.best_value


def scheduler(epoch, lr):
     if epoch < 400:
         return 0.005
     elif epoch == 1000:
         return 0.002
     else: 
         return lr*0.999

def train_ann_model(X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")

    lr_callback = keras.callbacks.LearningRateScheduler(scheduler)

    keras.utils.set_random_seed(SEED)

    # Build the neural network model
    model_act = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=[X_train_data.shape[1]],
                    kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dense(1)
    ])

    # loss_fn = keras.losses.MeanSquaredError(
    #     reduction="sum_over_batch_size", name="mean_squared_error", dtype=None
    # )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_rmse',  # Monitor validation loss
        patience=100,        # Stop after k epochs of no improvement
        restore_best_weights=True  # Keep the weights of the best epoch
    )



    # Compile the model
    model_act.compile(loss='mse',  # Use MSE for optimization, RMSE is just the square root
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Train the model
    history = model_act.fit(X_train_data, y_train_data.to_numpy().reshape(-1,), epochs=2000, batch_size=64000, validation_split=0.15, callbacks=[early_stopping, lr_callback])

    # Evaluate on the validation set (after early stopping)
    #val_loss, val_mae, val_mse, val_rmse = model_act.evaluate(X_valid_data, y_valid_data.to_numpy().reshape(-1,), verbose=0)

    #print(val_rmse)

    if (X_valid_data is not None):
        val_nn_predict = model_act.predict(X_valid_data)
    
    if (X_test_data is not None):
        test_nn_predict = model_act.predict(X_test_data)

    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"ANN Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(val_nn_predict.reshape(-1, 1))-1))
    
    if (X_valid_data is not None) & (X_test_data is not None):
        return val_nn_predict, test_nn_predict
    elif (X_test_data is not None):
        return None, test_nn_predict
    elif (X_valid_data is not None):
        return val_nn_predict, None
    else:
        return None, None

    
    

    #print(f"ANN Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(val_nn_predict.reshape(-1, 1))-1))




print(usePCA)
old_usePCA = usePCA
usePCA = True

if usePCA:
    X_train = train_pcaed.loc[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed.loc[val_index, :]
    X_test = train_pcaed.loc[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]

val_nn_predict, test_nn_predict = train_ann_model(X_train_data=X_train, y_train_data=y_train, X_valid_data=X_valid, y_valid_data=y_valid, X_test_data=X_test)

usePCA = old_usePCA

if usePCA:
    X_train = train_pcaed.loc[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed.loc[val_index, :]
    X_test = train_pcaed.loc[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]

print(usePCA)


N_FOLDS = 10


#XGBoost Model
def lr_decay(epoch):
    if epoch < 300:
        return 0.03 #0.045
    elif epoch <= 500:
        return 0.02 #0.015
    else:
        return 0.012 #0.01

def train_xgb_model(folds, X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'max_depth': 18,
        'learning_rate': 0.035,
        'min_child_weight': 60,
        'reg_alpha': 4,
        'reg_lambda': 2,
        'subsample': 0.85,
        'colsample_bytree': 0.7,
        'colsample_bynode': 0.5,
        'device': 'gpu'
    }

    callbacks = [xgb.callback.LearningRateScheduler(lr_decay)]
    
    if X_valid_data is not None:    
        test_valids = np.zeros(len(X_valid_data))
    else:
        test_valids = 0
    
    if X_test_data is not None:
        test_preds = np.zeros(len(X_test_data))
    else:
        test_preds = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_train_data), 1):
        print(f"\nðŸ”µ Fold {fold_idx} XGBoost Training...")
        X_train_local, y_train_local = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        X_valid_local, y_valid_local = X_train_data.iloc[valid_idx], y_train_data.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train_local, label=y_train_local)
        dvalid = xgb.DMatrix(X_valid_local, label=y_valid_local)

        if X_test_data is not None:
            X_test_fold = X_test_data[X_train_local.columns].copy()
            dtest = xgb.DMatrix(X_test_fold)
        
        if X_valid_data is not None:
            X_valid_ext_fold = X_valid_data[X_train_local.columns].copy()
            dvalid_ext = xgb.DMatrix(X_valid_ext_fold)

        
        model = xgb.train(
                params,
                dtrain,
                num_boost_round=100000,
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=30,
                verbose_eval=500,
                callbacks=callbacks
            )

        if X_test_data is not None:
            preds =  model.predict(dtest)
            test_preds += preds
        
        if X_valid_data is not None:
            valids = model.predict(dvalid_ext)
            test_valids += valids
        
    test_preds /= N_FOLDS
    test_valids /= N_FOLDS

    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"XGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(test_valids.reshape(-1, 1))-1))
    
    return test_valids, test_preds


def train_lgb_model(folds, X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 256, #256
        'min_child_samples': 40, #40
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 4,
        'reg_lambda': 2,
        'random_state': SEED,
        'verbose': -1
#       'min_gain_to_split': 0.001, #parameter not present
#        'max_depth': 20 #parameter not present
#        ,'device': 'gpu'
    }

    if X_valid_data is not None:    
        test_valids = np.zeros(len(X_valid_data))
    else:
        test_valids = 0
    
    if X_test_data is not None:
        test_preds = np.zeros(len(X_test_data))
    else:
        test_preds = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_train_data), 1):
        print(f"\nðŸŸ¢ Fold {fold_idx} LightGBM Training...")
        X_train_local, y_train_local = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        X_valid_local, y_valid_local = X_train_data.iloc[valid_idx], y_train_data.iloc[valid_idx]
        if X_test_data is not None:
            X_test_fold = X_test_data[X_train_local.columns].copy()

        if X_valid_data is not None:    
            X_valid_ext_fold = X_valid_data[X_train_local.columns].copy()

        #X_train, X_valid, X_test_fold, X_valid_ext_fold = target_encode(X_train, X_valid, X_test_fold, X_valid_ext_fold, y_train)

        train_dataset = lgb.Dataset(X_train_local, y_train_local)
        valid_dataset = lgb.Dataset(X_valid_local, y_valid_local, reference=train_dataset)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=100000,
            valid_sets=[train_dataset, valid_dataset],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(500)
            ]        )

        if X_test_data is not None:
            preds = model.predict(X_test_fold)
            test_preds += preds

        if X_valid_data is not None:
            valids = model.predict(X_valid_ext_fold)
            test_valids += valids       
        

    test_preds /= N_FOLDS
    test_valids /= N_FOLDS

    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"LGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(test_valids.reshape(-1, 1))-1))
    
    return test_valids, test_preds

def train_cat_model(folds, X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.") 


    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': SEED,
        'depth': 8,
        'learning_rate': 0.03, #0.035
        'l2_leaf_reg': 3,
        'border_count': 32,
        'verbose': 500,
        'task_type': 'GPU' if DEVICE == 'cuda' else 'CPU'
    }

    if X_valid_data is not None:    
        test_valids = np.zeros(len(X_valid_data))
    else:
        test_valids = 0
    
    if X_test_data is not None:
        test_preds = np.zeros(len(X_test_data))
    else:
        test_preds = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_train_data), 1):
        print(f"\nðŸ’œ Fold {fold_idx} CatBoost Training...")
        X_train_local, y_train_local = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        X_valid_local, y_valid_local = X_train_data.iloc[valid_idx], y_train_data.iloc[valid_idx]
        if X_test_data is not None:
            X_test_fold = X_test_data[X_train_local.columns].copy()
        if X_valid_data is not None:
            X_valid_ext_fold = X_valid_data[X_train_local.columns].copy()

        #X_train, X_valid, X_test_fold, X_valid_ext_fold = target_encode(X_train, X_valid, X_test_fold, X_valid_ext_fold, y_train)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train_local, y_train_local,
            eval_set=[(X_valid_local, y_valid_local)],
            early_stopping_rounds=50,
            verbose=500
        )

        if X_test_data is not None:
            preds = model.predict(X_test_fold)
            test_preds += preds
        
        if X_valid_data is not None:
            valids = model.predict(X_valid_ext_fold)
            test_valids += valids    
        
    test_preds /= N_FOLDS
    test_valids /= N_FOLDS

    if (X_valid_data is not None) & (y_valid_data is not None):
        print(f"CAT Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(test_valids.reshape(-1, 1))-1))

    return test_valids, test_preds




if usePCA:
    X_train = train_pcaed.loc[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed.loc[val_index, :]
    X_test = train_pcaed.loc[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_xgb_valids, test_xgb_preds = train_xgb_model(X_train_data= X_train, y_train_data= y_train, X_valid_data= X_valid, y_valid_data=y_valid, X_test_data=X_test, folds= kf)
test_lgb_valids, test_lgb_preds = train_lgb_model(X_train_data= X_train, y_train_data= y_train, X_valid_data= X_valid, y_valid_data=y_valid, X_test_data=X_test, folds= kf)
test_cat_valids, test_cat_preds = train_cat_model(X_train_data= X_train, y_train_data= y_train, X_valid_data= X_valid, y_valid_data=y_valid, X_test_data=X_test, folds= kf)

print(f"XGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_xgb_valids.reshape(-1, 1))-1))
print(f"LGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_lgb_valids.reshape(-1, 1))-1))
print(f"CAT Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_cat_valids.reshape(-1, 1))-1))


# # Define a callable function to dynamically adjust the learning rate
def lgb_learning_rate(current_round):
    # Example: Decay the learning rate after every 20 rounds
    if current_round > 2000:
        return 0.01  # Reduce the learning rate
    elif (current_round <= 2000) & (current_round > 1000):
        return 0.015
    else:
        return 0.02  # Keep the initial learning rate


def train_lgb_model_2(folds, X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        #'learning_rate': 0.045,
        'num_leaves': 90, #256
        #'min_child_samples': 35, #40
        'subsample': 0.85,
        'colsample_bytree': 0.65,
        'reg_alpha': 1.5,
        'reg_lambda': 0.005,
        'random_state': SEED,
        'min_gain_to_split': 0.000005,
        'verbose': -1,
        #'max_depth': 40 #parameter not present
#        ,'device': 'gpu'
    }

    if X_valid_data is not None:    
        test_valids = np.zeros(len(X_valid_data))
    else:
        test_valids = 0
    
    if X_test_data is not None:
        test_preds = np.zeros(len(X_test_data))
    else:
        test_preds = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_train_data), 1):
        print(f"\nðŸŸ¢ Fold {fold_idx} LightGBM Training...")
        X_train_local, y_train_local = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        X_valid_local, y_valid_local = X_train_data.iloc[valid_idx], y_train_data.iloc[valid_idx]

        if X_test_data is not None:
            X_test_fold = X_test_data[X_train_local.columns].copy()

        if X_valid_data is not None:
            X_valid_ext_fold = X_valid_data[X_train_local.columns].copy()

        #X_train, X_valid, X_test_fold, X_valid_ext_fold = target_encode(X_train, X_valid, X_test_fold, X_valid_ext_fold, y_train)

        train_dataset = lgb.Dataset(X_train_local, y_train_local)
        valid_dataset = lgb.Dataset(X_valid_local, y_valid_local, reference=train_dataset)

        # Create the reset_parameter callback
        learning_rate_callback = lgb.reset_parameter(learning_rate=lgb_learning_rate)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=100000,
            valid_sets=[train_dataset, valid_dataset],
            callbacks=[
                lgb.early_stopping(600),
                lgb.log_evaluation(200),
                learning_rate_callback
            ]        )

        if X_test_data is not None:
            preds = model.predict(X_test_fold)
            test_preds += preds

        if X_valid_data is not None:
            valids = model.predict(X_valid_ext_fold)        
            test_valids += valids

    test_preds /= N_FOLDS
    test_valids /= N_FOLDS

    if (y_valid_data is not None) & (X_valid_data is not None):
        print(f"LGB 2 Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(test_valids.reshape(-1, 1))-1))
    
    return test_valids, test_preds





if usePCA:
    X_train = train_pcaed.loc[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed.loc[val_index, :]
    X_test = train_pcaed.loc[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]
    
test_lgb_valids_2, test_lgb_preds_2 = train_lgb_model_2(X_train_data= X_train, y_train_data= y_train, X_valid_data= X_valid, y_valid_data=y_valid, X_test_data=X_test, folds= kf)

print(f"LGB 2 Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_lgb_valids_2.reshape(-1, 1))-1))


#from sklearn.metrics import DistanceMetric
#dist = DistanceMetric.get_metric('seuclidean')

def train_knnr_model(X_train_data, y_train_data, X_valid_data = None, y_valid_data = None, X_test_data = None):

    print(f"Training data (X) has {len(X_train_data)} rows and {len(X_train_data.columns)} columns")
    print(f"Training data (y) has {len(y_train_data)} rows and {len(y_train_data.columns)} columns")
    if X_valid_data is not None:
        print(f"Validation data (X) has {len(X_valid_data)} rows and {len(X_valid_data.columns)} columns")
        if y_valid_data is not None:
            print(f"Validation data (y) has {len(y_valid_data)} rows and {len(y_valid_data.columns)} columns")
    else:
        print("Validation data not passed.")
    
    if X_test_data is not None:
        print(f"Test data (X) has {len(X_test_data)} rows and {len(X_test_data.columns)} columns")
    else:
        print("Test data not passed.")

    knnr = KNeighborsRegressor(n_neighbors = 25, weights='distance', algorithm='auto', leaf_size=20, p=1.0, n_jobs=-1, metric='euclidean') #, algorithm='kd_tree', leaf_size=20, p=1.0, 
    knnr.fit(X_train_data, y_train_data)

    if X_valid_data is not None:
        val_knnr_predict = knnr.predict(X_valid_data)
        if y_valid_data is not None:
            print(f"KNNR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid_data)-1, np.exp(val_knnr_predict.reshape(-1, 1))-1))

    if X_test_data is not None:
        test_knnr_predict = knnr.predict(X_test_data)
    

    if (X_valid_data is not None) & (X_test_data is not None):
        return val_knnr_predict, test_knnr_predict
    elif (X_test_data is not None):
        return None, test_knnr_predict
    elif (X_valid_data is not None):
        return val_knnr_predict, None
    else:
        return None, None
    



if usePCA:
    X_train = train_pcaed.loc[train_index, :]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_pcaed.loc[val_index, :]
    X_test = train_pcaed.loc[test_index, :]
    y_valid = train_data_transform.loc[val_index, y_cols]
else:
    X_train = train_data_transform.loc[train_index, x_cols]
    y_train = train_data_transform.loc[train_index, y_cols]
    X_valid = train_data_transform.loc[val_index, x_cols]
    X_test = train_data_transform.loc[test_index, x_cols]
    y_valid = train_data_transform.loc[val_index, y_cols]
    
val_knnr_predict, test_knnr_predict = train_knnr_model(X_train_data= X_train, y_train_data= y_train, X_valid_data= X_valid, y_valid_data=y_valid, X_test_data=X_test)



print(f"LR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(val_lr_predict.reshape(-1, 1))-1))
print(f"RFR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(val_rfr_predict.reshape(-1, 1))-1))
print(f"XGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_xgb_valids.reshape(-1, 1))-1))
print(f"LGB Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_lgb_valids.reshape(-1, 1))-1))
print(f"LGB 2 Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_lgb_valids_2.reshape(-1, 1))-1))
print(f"CAT Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(test_cat_valids.reshape(-1, 1))-1))
print(f"ANN with PCA components - Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(val_nn_predict.reshape(-1, 1))-1))
print(f"KNNR Validation RMSLE: ", RMSLE_calc(np.exp(y_valid)-1, np.exp(val_knnr_predict.reshape(-1, 1))-1))



data_preds = {
    'xgb_predictions': np.exp(test_xgb_preds.reshape(-1, ))-1, 
 'lgb_predictions': np.exp(test_lgb_preds.reshape(-1, ))-1, 
 'cat_predictions': np.exp(test_cat_preds.reshape(-1, ))-1, 
 'rfr_predictions': np.exp(test_rfr_predict.reshape(-1, ))-1, 
 'nn_w_pcv_predictions': np.exp(test_nn_predict.reshape(-1, ))-1, 
 'knnr_predictions': np.exp(test_knnr_predict.reshape(-1, ))-1
 }

df = pd.DataFrame(data_preds)
#df.to_csv(dirname + "/best_performing_predictions.csv")


#Placeholder for hill-climbing algorithm

# import pandas as pd
# import numpy as np
from sklearn.metrics import mean_squared_error # We'll use this for MSE then take sqrt for RMSE

# --- 1. Setup: Create a Sample DataFrame (Replace with your actual data) ---
# (Same as before, ensure you have your DataFrame 'df' ready)
# data = {
#     'model1_pred': np.random.rand(100) * 10,
#     'model2_pred': np.random.rand(100) * 10,
#     'model3_pred': np.random.rand(100) * 10,
#     'model4_pred': np.random.rand(100) * 10,
#     'model5_pred': np.random.rand(100) * 10,
#     'actual_y': np.random.rand(100) * 10 + np.random.randn(100)
# }

# [np.exp(test_xgb_valids.reshape(-1, ))-1, 
#  np.exp(test_lgb_valids.reshape(-1, ))-1, 
#  np.exp(test_cat_valids.reshape(-1, ))-1, 
#  np.exp(val_rfr_predict.reshape(-1, ))-1, 
#  np.exp(val_nn_predict.reshape(-1, ))-1, 
#  np.exp(val_knnr_predict.reshape(-1, ))-1], 
# actual_valids = np.exp(y_valid.to_numpy().reshape(-1, ))-1

data = {
    'xgb_valids': np.exp(test_xgb_valids.reshape(-1, ))-1, 
 'lgb_valids': np.exp(test_lgb_valids.reshape(-1, ))-1, 
 'cat_valids': np.exp(test_cat_valids.reshape(-1, ))-1, 
 'rfr_valids': np.exp(val_rfr_predict.reshape(-1, ))-1, 
 'nn_valids': np.exp(val_nn_predict.reshape(-1, ))-1, 
 'knnr_valids': np.exp(val_knnr_predict.reshape(-1, ))-1, 
'actual_valids': np.exp(y_valid.to_numpy().reshape(-1, ))-1
}

df = pd.DataFrame(data)

model_pred_cols = ['xgb_valids', 
 'lgb_valids', 
 'cat_valids', 
 'rfr_valids', 
 'nn_valids', 
 'knnr_valids']
actual_col = 'actual_valids'

# --- 2. Objective Function ---
def calculate_ensemble_performance(weights, df, model_cols, actual_col, metric='rmse'): # Default to rmse
    """
    Calculates the performance of the ensemble given a set of weights.

    Args:
        weights (np.array): Array of weights for each model.
        df (pd.DataFrame): DataFrame containing model predictions and actual values.
        model_cols (list): List of column names for model predictions.
        actual_col (str): Column name for actual target values.
        metric (str): 'rmse', 'mse', 'mae', etc.

    Returns:
        float: The calculated performance metric. Lower is better for rmse/mse/mae.
    """
    if len(weights) != len(model_cols):
        raise ValueError("Length of weights must match the number of model prediction columns.")

    ensemble_prediction = np.zeros(len(df))
    for i, col in enumerate(model_cols):
        ensemble_prediction += weights[i] * df[col]

    if metric == 'mse':
        return mean_squared_error(df[actual_col], ensemble_prediction)
    elif metric == 'rmse': # Root Mean Squared Error
        mse = mean_squared_error(df[actual_col], ensemble_prediction)
        return np.sqrt(mse)
    elif metric == 'mae':
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(df[actual_col], ensemble_prediction)
    elif metric == 'rmsle':
        return RMSLE_calc(df[actual_col], ensemble_prediction)
    # Add other metrics as needed
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose 'rmse', 'mse', 'mae', etc.")

# --- 3. Neighborhood Function (Same as before) ---
def get_neighbors(current_weights, step_size=0.01, num_models=5):
    """
    Generates neighboring weight combinations.
    (Code from the previous response - no changes needed here for RMSE)
    """
    neighbors = []
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                new_weights_inc = current_weights.copy()
                new_weights_inc[i] += step_size

                new_weights_dec = current_weights.copy()
                if new_weights_dec[i] - step_size >= 0:
                    new_weights_dec[i] -= step_size
                    neighbors.append(new_weights_dec / np.sum(new_weights_dec))
                neighbors.append(new_weights_inc / np.sum(new_weights_inc))
            else:
                new_weights = current_weights.copy()
                if new_weights[i] - step_size >= 0:
                    new_weights[i] -= step_size
                    new_weights[j] += step_size
                    neighbors.append(new_weights / np.sum(new_weights))

    valid_neighbors = []
    for nw in neighbors:
        nw[nw < 0] = 0
        if np.sum(nw) > 0:
             valid_neighbors.append(nw / np.sum(nw))
        else:
            valid_neighbors.append(np.ones(num_models) / num_models)

    unique_neighbors = []
    for neighbor in valid_neighbors:
        is_unique = True
        for un in unique_neighbors:
            if np.allclose(neighbor, un):
                is_unique = False
                break
        if is_unique:
            unique_neighbors.append(neighbor)
    return unique_neighbors


# --- 4. Hill Climbing Algorithm (Logic for minimization remains the same) ---
def hill_climbing_ensemble_weights(df, model_cols, actual_col,
                                   num_models=5,
                                   max_iterations=1000,
                                   step_size=0.01,
                                   initial_weights=None,
                                   metric='rmsle', # Default to rmse
                                   verbose=True):
    """
    Optimizes ensemble weights using the Hill Climbing algorithm.
    (Code from the previous response - minor change in default metric and metric check)
    """
    if initial_weights is None:
        current_weights = np.ones(num_models) / num_models
    else:
        current_weights = np.array(initial_weights)
        if not np.isclose(np.sum(current_weights), 1.0):
            print("Warning: Initial weights do not sum to 1. Normalizing.")
            current_weights = current_weights / np.sum(current_weights)

    current_performance = calculate_ensemble_performance(current_weights, df, model_cols, actual_col, metric)

    if verbose:
        print(f"Initial Weights: {current_weights}, Initial {metric.upper()}: {current_performance:.6f}")

    for iteration in range(max_iterations):
        neighbors = get_neighbors(current_weights, step_size, num_models)

        if not neighbors:
            if verbose:
                print("No valid neighbors generated. Stopping.")
            break

        best_neighbor_weights = None
        best_neighbor_performance = current_performance

        # Determine if we are minimizing (rmse, mse, mae) or maximizing
        minimize_metric = metric.lower() in ['rmse', 'mse', 'mae', 'rmsle'] # Added 'rmse' here explicitly

        for neighbor_weights in neighbors:
            performance = calculate_ensemble_performance(neighbor_weights, df, model_cols, actual_col, metric)

            if minimize_metric:
                if performance < best_neighbor_performance:
                    best_neighbor_performance = performance
                    best_neighbor_weights = neighbor_weights
            else: # Maximizing metric (e.g., accuracy)
                if performance > best_neighbor_performance:
                    best_neighbor_performance = performance
                    best_neighbor_weights = neighbor_weights

        if best_neighbor_weights is not None:
            improved = False
            if minimize_metric and best_neighbor_performance < current_performance:
                improved = True
            elif not minimize_metric and best_neighbor_performance > current_performance:
                improved = True
            
            if improved:
                current_weights = best_neighbor_weights
                current_performance = best_neighbor_performance
                if verbose:
                    print(f"Iteration {iteration + 1}: New Best Weights: {current_weights}, New Best {metric.upper()}: {current_performance:.6f}")
            else:
                if verbose:
                    print(f"Iteration {iteration + 1}: No improvement found. Stopping.")
                break
        else:
            if verbose:
                print(f"Iteration {iteration + 1}: No better neighbor found. Stopping.")
            break

    if verbose:
        print("\nOptimization Finished.")
    return current_weights, current_performance

# --- 5. Putting it Together ---
if __name__ == "__main__":
    num_models_in_ensemble = len(model_pred_cols)
    max_iters = 400
    perturb_step_size = 0.005
    evaluation_metric = 'rmsle' # <<<< CHANGED TO RMSE

    initial_w = [0.34347241701964903, 0.140644667695536, 0.04421217431274808, 0.20121428322298082, 0.23030827111552635, 0.04118926412595423]

    print(f"Optimizing ensemble weights for {num_models_in_ensemble} models to minimize {evaluation_metric.upper()}.\n")

    best_weights, best_performance = hill_climbing_ensemble_weights(
        df,
        model_pred_cols,
        actual_col,
        num_models=num_models_in_ensemble,
        max_iterations=max_iters,
        step_size=perturb_step_size,
        initial_weights=initial_w,
        metric=evaluation_metric,
        verbose=True
    )

    print(f"\n--- Results ---")
    print(f"Optimal Weights: {np.round(best_weights, 4)}")
    print(f"Best Ensemble {evaluation_metric.upper()}: {best_performance:.6f}")

    equal_weights = np.ones(num_models_in_ensemble) / num_models_in_ensemble
    equal_weights_performance = calculate_ensemble_performance(equal_weights, df, model_pred_cols, actual_col, evaluation_metric)
    print(f"Equal Weights Performance ({evaluation_metric.upper()}): {equal_weights_performance:.6f}")

    print(f"\nIndividual Model Performances (Lower {evaluation_metric.upper()} is better):")
    for model_col in model_pred_cols:
        # Calculate RMSE for individual models
        individual_rmsle = RMSLE_calc(df[actual_col], df[model_col])
        #individual_rmse = np.sqrt(individual_mse)
        print(f"  {model_col}: {individual_rmsle:.4f}")



#Train models with entire training data

# _, final_test_rfr_preds = train_RFR_model(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test)
# _, final_test_xgb_preds = train_xgb_model(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test, folds=kf)
# _, final_test_nn_preds = train_ann_model(X_train_data=X_train_all_data_pca, y_train_data=y_train_all_data, X_test_data=X_test)
# _, final_test_lgb_preds = train_lgb_model(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test, folds=kf)
# _, final_test_lgb_2_preds = train_lgb_model_2(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test, folds=kf)
# _, final_test_cat_preds = train_cat_model(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test, folds=kf)
# _, final_test_knnr_preds = train_knnr_model(X_train_data=X_train_all_data, y_train_data=y_train_all_data, X_test_data=X_test)


from datetime import datetime

weights = None
print("LR Weights: ")
weights = get_blend_weights(valids = [np.exp(test_xgb_valids.reshape(-1, 1))-1, np.exp(test_lgb_valids.reshape(-1, 1))-1, np.exp(test_cat_valids.reshape(-1, 1))-1, np.exp(val_rfr_predict.reshape(-1, 1))-1, np.exp(val_nn_predict.reshape(-1, 1))-1, np.exp(val_knnr_predict.reshape(-1, 1))-1], actual_valids = np.exp(train_data_transform.loc[pd.Series(np.invert(tr_index)), y_cols].to_numpy().reshape(-1, 1))-1)
print("Considering hill climbing weights: ")
print(f"Optimal Weights: {np.round(best_weights, 4)}")

weights = best_weights

#weights = None
#LR Weights:  0.34347241701964903 ; 0.140644667695536 ; 0.04421217431274808 ; 0.20121428322298082 ; 0.23030827111552635 ; 0.04118926412595423
# Predicted RMSLE:  0.057849308044189085
if weights is None:
    #weights = (0.42431909597667744, 0.24135094057899834, 0.07626447599988127, 0.18056283833886275, 0.07867085852836331)
    #weights = (0.39666667, 0.04666667, 0.13666667, 0.26666667, 0.13666667, 0.01666667)
    weights = (1.0, 0, 0, 0, 0, 0)
    #Iteration 33: New Best Weights: [0.39666667 0.04666667 0.13666667 0.26666667 0.13666667 0.01666667], New Best RMSLE: 0.057786
    # Iteration 22: New Best Weights: [0.40058893 0.04182879 0.13689031 0.26630448 0.14301475 0.01137275], New Best RMSLE: 0.057786

#final_preds = blend_predictions([np.exp(final_test_xgb_preds.reshape(-1, 1))-1, np.exp(final_test_lgb_preds.reshape(-1, 1))-1, np.exp(final_test_cat_preds.reshape(-1, 1))-1, np.exp(final_test_rfr_preds.reshape(-1, 1))-1, np.exp(final_test_nn_preds.reshape(-1, 1))-1, np.exp(final_test_knnr_preds.reshape(-1, 1))-1], weights=weights)
final_preds = blend_predictions([np.exp(test_xgb_preds.reshape(-1, 1))-1, np.exp(test_lgb_preds.reshape(-1, 1))-1, np.exp(test_cat_preds.reshape(-1, 1))-1, np.exp(test_rfr_predict.reshape(-1, 1))-1, np.exp(test_nn_predict.reshape(-1, 1))-1, np.exp(test_knnr_predict.reshape(-1, 1))-1], weights=weights)

#final_preds = (weights[0] * (np.exp(test_preds.reshape(-1, 1))-1) + weights[1] * (np.exp(test_rfr_predict.reshape(-1, 1))-1) + weights[2] * (np.exp(test_lr_predict.reshape(-1, 1))-1))
save_submission(np.clip(final_preds, 1, 314), filename="Submission" + datetime.now().strftime("%m%d%Y_%H%M%S_IST") + ".csv")


# correlation_matrix = train_data_transform.loc[train_data.id.astype(str).str.contains("test"), train_data_transform.columns.difference(['id'])].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()


# from autogluon.tabular import TabularDataset, TabularPredictor

# # Load training data (assuming it's in a CSV file)
# train_data = TabularDataset(dir_path + "/train.csv")

# # Create a TabularPredictor instance
# predictor = TabularPredictor(label="Calories", path=dir_path + "/AutoGluonModels/")

# # Train the model (with a time limit)
# predictor.fit(train_data, time_limit=3600)

