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


train = pd.read_csv("/kaggle/input/predict-podcast-listening-time/train.csv")
test = pd.read_csv("/kaggle/input/predict-podcast-listening-time/test.csv")


train.head()


train.info()


test.info()


y = train['Listening_Time_minutes']
y.head()


import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(train)
# plt.show()




train.columns


train['Episode_Number'] =  train['Episode_Title'].str.extract(r'Episode (\d+)',expand=False).astype('Int64')
test['Episode_Number'] =  test['Episode_Title'].str.extract(r'Episode (\d+)',expand=False).astype('Int64')



train.head()


test.head()


col = (train.dtypes == 'object')
object_cols = list(col[col].index)

print(f"The columns with categorical data are : {object_cols}")



genre = list(train['Genre'].unique())
print(f"The {train['Genre'].nunique()} categories in Genre are : {genre}")



train['Podcast_Name'].unique()


from category_encoders import TargetEncoder

# Initialize the encoder
te_genre = TargetEncoder()

# Fit on training data only (avoid data leakage)
train['Genre_TE'] = te_genre.fit_transform(train['Genre'], train['Listening_Time_minutes'])

# Transform test data using the same encoder
test['Genre_TE'] = te_genre.transform(test['Genre'])

# (Optional) Drop the original 'Genre' column if not needed
train.drop('Genre', axis=1, inplace=True)
test.drop('Genre', axis=1, inplace=True)

train.head()



from category_encoders import TargetEncoder

te = TargetEncoder()
train['Podcast_Name_TE'] = te.fit_transform(train['Podcast_Name'],train['Listening_Time_minutes'])
test['Podcast_Name_TE'] = te.transform(test['Podcast_Name'])

train.head()


# Target encode Publication_Day and Publication_Time

te_pub = TargetEncoder()
train[['Publication_Day_TE', 'Publication_Time_TE']] = te_pub.fit_transform(
    train[['Publication_Day', 'Publication_Time']], train['Listening_Time_minutes']
)
test[['Publication_Day_TE', 'Publication_Time_TE']] = te_pub.transform(
    test[['Publication_Day', 'Publication_Time']]
)

# Drop original columns
train.drop(['Publication_Day', 'Publication_Time'], axis=1, inplace=True)
test.drop(['Publication_Day', 'Publication_Time'], axis=1, inplace=True)



from sklearn.preprocessing import OrdinalEncoder

onc = OrdinalEncoder()

data_to_transform = ['Episode_Sentiment']
train[data_to_transform] = onc.fit_transform(train[data_to_transform])
test[data_to_transform] = onc.transform(test[data_to_transform])
train.head()



missing_cols = [x for x in train.columns if train[x].isnull().any()]
print(f"The columns with missing data are : {missing_cols}")


train.columns


x_train = train[[ 'Podcast_Name_TE','Episode_Length_minutes',
       'Genre_TE', 'Host_Popularity_percentage', 'Publication_Day_TE',
       'Publication_Time_TE', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment', 'Episode_Number']]


x_test = test[[ 'Podcast_Name_TE','Episode_Length_minutes',
       'Genre_TE', 'Host_Popularity_percentage', 'Publication_Day_TE',
       'Publication_Time_TE', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment', 'Episode_Number']]


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy = 'median')
imputed_train_data = pd.DataFrame(imputer.fit_transform(x_train))
imputed_test_data = pd.DataFrame(imputer.transform(x_test))

imputed_train_data.columns = x_train.columns
imputed_test_data.columns = x_test.columns

imputed_train_data.info()


col = (imputed_train_data.dtypes == 'float64')
numerical_cols = list(col[col].index)

print(f"The columns with numerical data are : {numerical_cols}")


# # Add statistical features
# imputed_train_data['row_mean'] = imputed_train_data[numerical_cols].mean(axis=1)
# imputed_train_data['row_std'] = imputed_train_data[numerical_cols].std(axis=1)
# imputed_train_data['row_max'] = imputed_train_data[numerical_cols].max(axis=1)
# imputed_train_data['row_min'] = imputed_train_data[numerical_cols].min(axis=1)

# imputed_test_data['row_mean'] = imputed_test_data[numerical_cols].mean(axis=1)
# imputed_test_data['row_std'] = imputed_test_data[numerical_cols].std(axis=1)
# imputed_test_data['row_max'] = imputed_test_data[numerical_cols].max(axis=1)
# imputed_test_data['row_min'] = imputed_test_data[numerical_cols].min(axis=1)


# imputed_train_data.info()


for col in imputed_train_data.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=imputed_train_data, x=col)
    plt.title(f"Boxplot for {col}")
    plt.show()


def remove_outliers_iqr(df, columns):
    original_shape = df.shape
    removed_info = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        after = df.shape[0]
        removed = before - after

        removed_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'removed_rows': removed
        }

    final_shape = df.shape

    print(f"Original shape: {original_shape}")
    print(f"Final shape after outlier removal: {final_shape}")
    print("\nOutlier removal details per column:")
    for col, info in removed_info.items():
        print(f"{col}: Removed {info['removed_rows']} rows")
    
    return df



# Combine X and y
data = pd.concat([imputed_train_data, y], axis=1)

columns_to_check = data.columns

# Remove outliers on combined data
data_cleaned = remove_outliers_iqr(data, columns_to_check)







# numerical_cols = ['Podcast_Name_TE', 'Episode_Length_minutes', 'Genre_TE', 'Host_Popularity_percentage', 'Publication_Day_TE', 'Publication_Time_TE','Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Episode_Number','Listening_Time_minutes']

# print(f"The columns with numerical data are : {numerical_cols}")


data_cleaned.head()






corr_matrix = data_cleaned.corr()


plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',square=True)
plt.title("Correlation")
plt.show()



numerical_cols = ['Podcast_Name_TE','Episode_Length_minutes',
       'Genre_TE', 'Host_Popularity_percentage', 'Publication_Day_TE',
       'Publication_Time_TE', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment', 'Episode_Number']

print(f"The columns with numerical data are : {numerical_cols}")


# Add statistical features
data_cleaned['row_mean'] = data_cleaned[numerical_cols].mean(axis=1)
data_cleaned['row_std'] = data_cleaned[numerical_cols].std(axis=1)
data_cleaned['row_max'] = data_cleaned[numerical_cols].max(axis=1)
data_cleaned['row_min'] = data_cleaned[numerical_cols].min(axis=1)


imputed_test_data['row_mean'] = imputed_test_data[numerical_cols].mean(axis=1)
imputed_test_data['row_std'] = imputed_test_data[numerical_cols].std(axis=1)
imputed_test_data['row_max'] = imputed_test_data[numerical_cols].max(axis=1)
imputed_test_data['row_min'] = imputed_test_data[numerical_cols].min(axis=1)



data_cleaned['Episode_Length_per_Ad'] = data_cleaned['Episode_Length_minutes'] / (data_cleaned['Number_of_Ads'] + 1)
imputed_test_data['Episode_Length_per_Ad'] = imputed_test_data['Episode_Length_minutes'] / (imputed_test_data['Number_of_Ads'] + 1)

data_cleaned['HostGuest_Interaction'] = data_cleaned['Host_Popularity_percentage'] * data_cleaned['Guest_Popularity_percentage']
imputed_test_data['HostGuest_Interaction'] = imputed_test_data['Host_Popularity_percentage'] * imputed_test_data['Guest_Popularity_percentage']

data_cleaned['Row_Mean_to_Std'] = data_cleaned['row_mean'] / (data_cleaned['row_std'] + 1e-6)
imputed_test_data['Row_Mean_to_Std'] = imputed_test_data['row_mean'] / (imputed_test_data['row_std'] + 1e-6)


data_cleaned.info()


# Combine X and y
data = data_cleaned

columns_to_check = data.columns

# Remove outliers on combined data
data_cleaned_1 = remove_outliers_iqr(data, columns_to_check)



X = data_cleaned_1[['Podcast_Name_TE','Episode_Length_minutes',
       'Genre_TE', 'Host_Popularity_percentage',
       'Publication_Time_TE', 'Number_of_Ads',
       'Episode_Sentiment','row_max','row_mean','row_std','Listening_Time_minutes',
        'Episode_Length_per_Ad', 'HostGuest_Interaction', 'Row_Mean_to_Std']]

y = X.pop('Listening_Time_minutes')


print(X.shape)
print(y.shape)

X.head()


from sklearn.model_selection import train_test_split
trainX,valX,trainY,valY = train_test_split(X,y,train_size=0.8,random_state=2)

print(trainX.shape)
print(valX.shape)


missing_cols = [x for x in trainX.columns if trainX[x].isnull().any()]
print(f"The columns with missing data are : {missing_cols}")


X_testing = imputed_test_data[['Podcast_Name_TE','Episode_Length_minutes',
       'Genre_TE', 'Host_Popularity_percentage',
       'Publication_Time_TE', 'Number_of_Ads',
       'Episode_Sentiment','row_max','row_mean','row_std',"Episode_Length_per_Ad","HostGuest_Interaction","Row_Mean_to_Std" ]]


# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor



# # All models including stacking
# models = {
#     # 'LinearRegression': LinearRegression(),
#     # 'Ridge': Ridge(alpha=1.0, random_state=2),
#     # 'Lasso': Lasso(alpha=0.001, random_state=2),
#     # 'GradientBoosting': GradientBoostingRegressor(
#     #     n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=2
#     # ),
#     # 'XGBoost': XGBRegressor(
#     #     n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
#     #     colsample_bytree=0.8, random_state=2, n_jobs=-1
#     # ),
#     # 'LightGBM': LGBMRegressor(
#     #     n_estimators=500, learning_rate=0.05, max_depth=-1, subsample=0.8,
#     #     colsample_bytree=0.8, random_state=2, n_jobs=-1
#     # ),
#     # 'CatBoost': CatBoostRegressor(
#     #     iterations=500, learning_rate=0.05, depth=6, random_state=2, verbose=0
#     # ),
    
#     'RandomForest': RandomForestRegressor(
#         n_estimators=800, max_depth=30, min_samples_split=5, min_samples_leaf=2,
#         max_features='sqrt', bootstrap=True, random_state=2, n_jobs=-1
#     )
    
# }

# # --- Evaluation Loop ---
# results = {}
# for name, model in models.items():
#     model.fit(trainX, trainY)
#     pred = model.predict(valX)

#     rmse = mean_squared_error(valY, pred, squared=False)
#     mae = mean_absolute_error(valY, pred)
#     r2 = r2_score(valY, pred)

#     results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
#     print(f"\n{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    


# # class BlendedModel:
# #     def __init__(self, models, weights):
# #         self.models = models
# #         self.weights = np.array(weights) / np.sum(weights)  # normalize to sum=1

# #     def fit(self, X, y):
# #         # models are already trained, so no need to refit
# #         pass

# #     def predict(self, X):
# #         preds = [w * model.predict(X) for model, w in zip(self.models, self.weights)]
# #         return np.sum(preds, axis=0)


# # # --- Build blended model using best 4 models ---
# # blend = BlendedModel(
# #     models=[
# #         models['RandomForest'],
# #         models['XGBoost'],
# #         models['LightGBM'],
# #         models['CatBoost']
# #     ],
# #     weights=[0.45, 0.25, 0.20, 0.10]  # you can tune these
# # )

# # # --- Evaluate blended model ---
# # pred_blend = blend.predict(valX)
# # rmse = mean_squared_error(valY, pred_blend, squared=False)
# # mae = mean_absolute_error(valY, pred_blend)
# # r2 = r2_score(valY, pred_blend)

# # results['Blended'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
# # print(f"\nBlended - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")




# best_model_name = min(results, key=lambda k: results[k]['RMSE'])
# print(f"\nBest model: {best_model_name}")



# import joblib

# # Save models
# for name, model in models.items():
#     filename = f"/kaggle/working/{name}_model.pkl"
#     joblib.dump(model, filename)
#     print(f"✅ Saved {name} to {filename}")




# import joblib

# # List of models to reload
# model_names = ["RandomForest"]

# loaded_models = {}
# for name in model_names:
#     filename = f"/kaggle/working/{name}_model.pkl"
#     loaded_models[name] = joblib.load(filename)
#     print(f"✅ Loaded {name} from {filename}")

# # Example usage
# model = loaded_models["RandomForest"]




from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define models
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=400, max_depth=30, random_state=2, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=2, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, random_state=2, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=2
    ),
    "CatBoost": CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=6,
        random_state=2, verbose=0
    )
}



results = {}
val_preds = {}

for name, model in models.items():
    model.fit(trainX, trainY)
    preds = model.predict(valX)

    rmse = mean_squared_error(valY, preds, squared=False)
    mae = mean_absolute_error(valY, preds)
    r2 = r2_score(valY, preds)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    val_preds[name] = preds

    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")



# Get weights proportional to R² scores
r2_scores = np.array([results[m]["R2"] for m in models.keys()])
weights = r2_scores / r2_scores.sum()

# Weighted average prediction
weighted_pred = np.zeros_like(valY, dtype=float)
for w, (name, pred) in zip(weights, val_preds.items()):
    weighted_pred += w * pred

rmse = mean_squared_error(valY, weighted_pred, squared=False)
mae = mean_absolute_error(valY, weighted_pred)
r2 = r2_score(valY, weighted_pred)

print(f"\nWeighted Ensemble - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")



avg_pred = np.mean(list(val_preds.values()), axis=0)

rmse = mean_squared_error(valY, avg_pred, squared=False)
mae = mean_absolute_error(valY, avg_pred)
r2 = r2_score(valY, avg_pred)

print(f"Simple Average Ensemble - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# model = RandomForestRegressor(
#         n_estimators=300,random_state=2, n_jobs=-1
#     )

# results = {}

# model.fit(trainX, trainY)
# pred = model.predict(valX)

# rmse = mean_squared_error(valY, pred, squared=False)
# mae = mean_absolute_error(valY, pred)
# r2 = r2_score(valY, pred)

# results["RandomForest"] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
# print(f"\nRandomForest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")



# model.fit(trainX, trainY)
# pred = model.predict(valX)

# rmse = mean_squared_error(valY, pred, squared=False)
# mae = mean_absolute_error(valY, pred)
# r2 = r2_score(valY, pred)

# # results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
# print(f"\n{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# filename = f"/kaggle/working/RandomForest_model.pkl"
# joblib.dump(model, filename)
# print(f"✅ Saved {name} to {filename}")




# # model = models[best_model_name]

# print(model)

# model.fit(X, y)

# prediction = model.predict(X_testing)


# Retrain all models on full data
for model in models.values():
    model.fit(X, y)

# Collect test predictions
test_preds = [model.predict(X_testing) for model in models.values()]

# Weighted ensemble on test
final_test_pred = np.average(test_preds, axis=0, weights=weights)

# Save submission
output = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": final_test_pred
})
output.to_csv("submission.csv", index=False)
print("✅ Submission saved!")



# output = pd.DataFrame({'id': test['id'],
#                        'Listening_Time_minutes': (prediction)})
# output.to_csv('submission.csv', index=False)


df = pd.read_csv("submission.csv")
df.head()





