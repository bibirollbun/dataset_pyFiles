import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

from datetime import datetime

from itertools import combinations

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_log_error

from xgboost import XGBRegressor


df = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')

df


df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


df.describe()


df["Premium Amount"].plot(kind = "hist")


for col in df:
    print(col)


all_df = df.columns.to_list()
int_df = list(df.select_dtypes(include=['float', 'int' ]).columns)
char_df = list(df.select_dtypes(include=['object' ]).columns)

int_df.remove('id')


df[df["Annual Income"].isna()]


df.isna().sum()




#Noted that all categorical columns could have thier missing values preserved so we will include a "Blank" value in all. 
#We also have one NA value in "Insurance Duration" which we will give to the 1 category

for col in char_df:
    df[col] = df[col].fillna("Blank")
    print(col, "has this many NA values", df[col].isna().sum())

for col in char_df:
    df_test[col] = df_test[col].fillna("Blank")
    print(col, "has this many NA values", df_test[col].isna().sum())

df["Insurance Duration"] = df["Insurance Duration"].fillna(1.0)


#For the numerical data, we will impute the missing data with the Median

for col in int_df:
    median = df[col].median()
    if df[col].isna().sum() > 1:
        df[col] = df[col].fillna(median)
        print(col, "has this many NA values:", df[col].isna().sum())


df.isna().sum()


#Annual Income is skewed - we can fix this with taking the log instead
df["Annual Income"].plot(kind="hist")


df["Annual Income"] = np.log1p(df["Annual Income"])
df_test["Annual Income"] = np.log1p(df["Annual Income"])


df["Annual Income"].plot(kind="hist")


temp_df = df[["Age", "Annual Income", "Number of Dependents", "Health Score", "Vehicle Age","Credit Score", "Insurance Duration"]].copy()

temp_df.dropna(axis = 0, inplace = True)

temp_df.corrwith(df["Premium Amount"]).sort_values().plot(kind= "barh", figsize = (6,6))


for val in char_df:
    if (len(df[val].value_counts()) < 40):
        print(df[val].value_counts())
    else:
        print(val, " Is quite large...")



#The Policy Start Date column is tricky as there are so many differnt values. There are a couple of routes we can take:

## 1. We can work with the dates as is and put stress on our models
## 2. We can bin the dates into Year or Month_Year, losing some of the specificity to gain computation efficiency and model accuracy

df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"]).dt.strftime('%Y')
df_test["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"]).dt.strftime('%Y')

print(df.head())

print(df["Policy Start Date"].value_counts())


temp_char = char_df.copy()

dummy_df = pd.get_dummies(df[temp_char],drop_first = True)
dummy_test = pd.get_dummies(df_test[temp_char], drop_first = True)

dummy_df.corrwith(df["Premium Amount"]).sort_values().plot(kind= "barh", figsize = (10,25))


dummy_df


dummy_test


def create_pairwise_columns(df):
    # Check if all columns are of type bool
    if not all(df.dtypes.apply(lambda x: x == 'bool')):
        raise ValueError("All columns in the DataFrame must be of type bool.")
    
    # Create pairwise combinations of columns
    column_pairs = list(combinations(df.columns, 2))
    
    # Use a dictionary to store new columns
    pairwise_data = {
        f"{col1}_AND_{col2}": np.logical_and(df[col1], df[col2]) 
        for col1, col2 in column_pairs
    }
    
    # Add all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(pairwise_data, index = df.index)], axis=1)
    
    return df



new = create_pairwise_columns(dummy_df)
new_test = create_pairwise_columns(dummy_test)

new
    


corrnew = new.corrwith(df["Premium Amount"])

corrnew.dropna(inplace= True)


corrnew1 = corrnew.sort_values(ascending = False, key = abs)[:80]

corrnew1 = corrnew1.sort_values()

ax = corrnew1.plot(kind= "barh", figsize = (10,35))

plt.axvline(0.005, c='red', linestyle='--')
plt.axvline(-0.005, c='red', linestyle='--')


def find_relevant(series, threshold):
    relevant_columns = []
    for col in series.index:
        if abs(series[col]) > threshold and "_AND_" in col:
            relevant_columns.append(col) 
    return relevant_columns

newtest = find_relevant(corrnew, 0.005) 
print("\n\nRelevant columns:", len(newtest))


df = pd.concat([df, new[newtest]], axis = 1)
df_test = pd.concat([df_test, new_test[newtest]], axis = 1)

char_df.extend(newtest)
df


ohencode_df = np.array(char_df)
num_df = ('Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration')

cat_imputer = SimpleImputer(strategy= "constant", fill_value = "Blank")
num_imputer = SimpleImputer(strategy = "median")

num_scaler = StandardScaler()

num_transformer = Pipeline(steps = [('impute', num_imputer)])
num_transformer_s = Pipeline(steps = [('impute', num_imputer), ('scale', num_scaler)])
ohcat_transformer = Pipeline( steps = [('impute', cat_imputer), ('encode', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers = [("oh_cat", ohcat_transformer, ohencode_df), ("num", num_transformer_s, num_df)])

feature_selection = SelectKBest(score_func=f_classif, k=15)


X = df.drop(['Premium Amount'], axis = 1)
y = df['Premium Amount']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)


model = XGBRegressor(learning_rate = np.float64(0.02028633586934382), max_depth = 9, n_estimators = 161, subsample = np.float64(0.6770233310419052), random_state = 0)

#We need to take the log of our y variable as there are issues with negative values in the output
y_train_log = np.log1p(y_train)

xgb_pipeline = Pipeline(steps = [('preprocessor', preprocessor),
                                 ('feature_selection', feature_selection),
                                 ('model', model)]
                        )

xgb_pipeline.fit(X_train, y_train_log)


y_pred_log = xgb_pipeline.predict(X_valid)

#We then unpack the log to get the true pred
y_pred = np.expm1(y_pred_log)


print(f'Root Mean Squared Log Error: {math.sqrt(mean_squared_log_error(y_valid, y_pred))}')

# Without F_stat feature selection: Root Mean Squared Log Error: 1.0503561746556593
# With F_stat feature selection: Root Mean Squared Log Error: 1.049335138846521
# With F_stat feature selection and best params: Root Mean Squared Log Error: 1.0486716732014243





# X_train_prepped = pd.DataFrame(preprocessor.fit_transform(X_train), columns= preprocessor.get_feature_names_out())

# X_train_fet = pd.DataFrame(feature_selection.fit_transform(X_train_prepped, y_train), columns=feature_selection.get_feature_names_out())

# X_train_sample = X_train_fet[:100000]
# y_train_sample = y_train[:100000]


# param_grid = {
#     'max_depth': stats.randint(3, 10),
#     'learning_rate': stats.uniform(0.01, 0.1),
#     'subsample': stats.uniform(0.5, 0.5),
#     'n_estimators':stats.randint(50, 1000)
#}


# param_search = RandomizedSearchCV(
#     estimator = XGBRegressor(random_state = 0),
#     param_distributions = param_grid,
#     n_iter=50,
#     cv=5,
#     scoring= 'neg_root_mean_squared_log_error',
#     n_jobs =-1, #use all cores
#     verbose=1,
#     return_train_score=True,
#     random_state=0
#)


# param_search.fit(X_train_sample, y_train_sample)

# best_params=param_search.best_params_
# best_score=param_search.best_score_

# print("Best Params", best_params)
# print("Best Score", best_score)

###Best Params {'learning_rate': np.float64(0.02028633586934382), 'max_depth': 9, 'n_estimators': 161, 'subsample': np.float64(0.6770233310419052)}
###Best Score -1.1382992555441165


# model = RandomForestRegressor(n_estimators = 250, n_jobs = -1)

# my_pipeline = Pipeline(steps = [('preprocessor', preprocessor), 
#                                 ('feature_selection', feature_selection), 
#                                 ('model', model)]
#                                 )

# my_pipeline.fit(X_train, y_train)





# y_pred = my_pipeline.predict(X_valid)

# print(f'Root Mean Squared Log Error: {math.sqrt(mean_squared_log_error(y_valid, y_pred))}')

# "Root Mean Squared Log Error: 1.1447432853335768"


final_pred_log = xgb_pipeline.predict(df_test)
final_pred = np.expm1(final_pred_log)
df_test["Premium Amount"] = final_pred.astype(float).round(3)
df_test["Premium Amount"]


pd.DataFrame(final_pred).plot(kind='hist')


submission = df_test[["id", "Premium Amount"]]
submission.set_index("id")
submission.to_csv('submission.csv', index = False)
submission

