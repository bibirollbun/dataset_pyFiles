# Approach
# 1. How to tackle the missing data problem? What technique to use for imputation?
# 2. Apply SARIMA, ARIMA, Prophet (basically compare results of time series wrt other models)
# 3. Apply XGBoost LGBM 
# 4. Compare results using ensemble model
# Apply MAE on log transformed target


# load libraries 
import pandas as pd
import numpy as np
import holidays
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train.drop(columns = ["id"], inplace = True)


print("The numer of rows for the train data are :",len(train)) 


for c in train.columns : 
    print(train[c].unique())


# Let's just take an based on the prediction and create a baseline
baseline = train.groupby(["country","store", "product"])["num_sold"].mean().fillna(0).reset_index()
baseline_test_merge = pd.merge(test, baseline, how = "left", on = ["country","store", "product"])


# submission = baseline_test_merge[["id","num_sold"]]
# submission.to_csv("submission.csv", index=False)
# This has a score of 0.17853 ~ 40%ile


train[train["country"] == "Kenya"].groupby(["country","product", "store"])["num_sold"].apply(lambda x: x.isna().sum())


train[train["country"] == "Kenya"].groupby(["country","product", "store"])["num_sold"].mean()


# Well there seems to be a lot of discussion on how to impute the tagret varibale
# Some people have suggested to drop the rows or impute them with constants like mean or absolute values 


# impute_data = pd.merge(train[train["num_sold"].isna()], 
#                         baseline,
#                         how = "left",
#                         on = ["country","store", "product"])

# # Updating up the index for better merge 
# impute_data.set_index("id", inplace = True)

# # Imputing the data 
# train.loc[train["num_sold"].isna(), "num_sold"] = impute_data["num_sold_y"]


train = train[~train["num_sold"].isna()]


def create_date_features(df, date_column='date'):    
    # Convert date column to datetime if it's not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Basic date components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter
    df['day'] = df[date_column].dt.weekday
    
    # Cyclical features
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Weekend indicator
    df['weekend'] = df['day'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # Year-month combination
    df['year_month'] = df['year'].astype(str) + "_" + df['month'].astype(str)
    
    return df

train = create_date_features(train)


# Loading a package to import holidays observed in different countries
# Lifted this code and idea shamelessly from a discussion thread started by Cata Danna
ca_holidays = holidays.country_holidays('CA') # Canada
fi_holidays = holidays.country_holidays('FI') # Finland
it_holidays = holidays.country_holidays('IT') # Italy
ke_holidays = holidays.country_holidays('KE') # Kenya
no_holidays = holidays.country_holidays('NO') # Norway
sg_holidays = holidays.country_holidays('SG') # Singapore


def set_holiday(row):
    VAL_HOLIDAY = 1
    if row["country"] == "Canada" and row["date"] in ca_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Finland" and row["date"] in fi_holidays:
        row["holiday"] = VAL_HOLIDAY
    
    elif row["country"] == "Italy" and row["date"] in it_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Kenya" and row["date"] in ke_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Norway" and row["date"] in no_holidays:
        row["holiday"] = VAL_HOLIDAY

    elif row["country"] == "Singapore" and row["date"] in sg_holidays:
        row["holiday"] = VAL_HOLIDAY

    return row

train = train.apply(set_holiday, axis=1)
train["holiday"].fillna(0, inplace = True)


# Adding the GDP data
gdp_per_capita = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")
gdp_ppp_per_capita = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_ppp_per_capita.csv")

country_df = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']

gdp_per_capita_dict = {}
gdp_ppp_per_capita_dict = {}

for country in country_df :
    gdp_per_capita.drop(columns = ['Unnamed: 65', '2020', '2019'])
    mean_val = gdp_per_capita[gdp_per_capita["Country Name"] == country].iloc[:,-5:].transpose().mean().round(2).squeeze()
    gdp_per_capita_dict.update({country : mean_val})

for country in country_df :
    gdp_ppp_per_capita.drop(columns = ['Unnamed: 65', '2020', '2019'])
    mean_val = gdp_ppp_per_capita[gdp_ppp_per_capita["Country Name"] == country].iloc[:,-5:].transpose().mean().round(2).squeeze()
    gdp_ppp_per_capita_dict.update({country : mean_val})

wealth_df = pd.merge(pd.DataFrame(list(gdp_per_capita_dict.items()), columns=['country', 'GDP per Capita']),
                     pd.DataFrame(list(gdp_ppp_per_capita_dict.items()), columns=['country', 'PPP per Capita']),
                     how = 'left', on = 'country')

train = pd.merge(train, wealth_df, how = "left", on = "country")


# Viz the data 
fig = px.line(train,
              x = "date",
              y = "num_sold",
              color = "country",
              facet_row="product",
              facet_col="store")
fig.show()


# Before building a model it is necessary that we do the variable encoding 
# label encoding variables
le = LabelEncoder()

for col in train.select_dtypes(include = "object").columns :
    train[col] = le.fit_transform(train[col])


# Applying transformations to the test data
# Creating datetime features for test dataset
test = create_date_features(test)

# Creating a column for holiday
test = test.apply(set_holiday, axis=1)
test["holiday"].fillna(0, inplace = True)

# Creating a column for country
test = pd.merge(test, wealth_df, how = "left", on = "country")

# test.drop(columns = ["date", "id"], inplace = True)


# Applying encoding
for col in test.select_dtypes(include = "object").columns :
    test[col] = le.fit_transform(test[col])


# Doing the train test split on the train data
X = train.drop(columns = ["num_sold", "date"]) # Dropping the date column as we already have date related columns and keeping it may result in multicollinearity
# Did some feature engineering where I divided the GDP related columns by 10000 so that the splits in LGBM are better
X[['GDP per Capita', 'PPP per Capita']] = X[['GDP per Capita', 'PPP per Capita']]/10000
y = np.log1p(train["num_sold"])

# kfold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Applying LightGBM since it is something that's been in vogue recently. Hence 
def objective(trial):
    # Suggest hyperparameters
    params = {
        "objective": "regression",
        "metric": "mae",  # Evaluation metric
        "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "verbose" : -1,
        "feature_pre_filter": False, 
        "device" : "gpu"
    }

    # Cross-validation loop
    fold_scores = []  # To store scores for each fold

    # Initialize lists to store all predictions and actuals
    all_preds = []
    all_actuals = []
    
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold, 
                             # categorical_feature = ["country","product", "store", "year_month"],
                             # categorical_feature = categorical_indices,
                             free_raw_data = True)
        test_data = lgb.Dataset(X_val_fold, label=y_val_fold, 
                            free_raw_data = True)
        
        # Train the model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, test_data],
            # verbose = False,  # Suppress output for brevity
            # early_stopping_rounds=10
        )
    
        # Predict on validation fold and calculate MAE
        preds = model.predict(X_val_fold)
        fold_score = mean_absolute_error(y_val_fold, preds)
        fold_scores.append(fold_score)
        
        # Append predictions and actuals
        all_preds.extend(preds)
        all_actuals.extend(y_val_fold)

    # Average MAE across folds
    mean_score = np.mean(fold_scores)
    return mean_score

# Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# # Best trial and parameters
# print("Best trial:")
# print(study.best_trial)
# print("Best hyperparameters:")
# print(study.best_params)


# # Get the best parameters from Optuna
# best_params = study.best_params

# These params have been generated already using Optuna  
best_params = {'boosting': 'gbdt',
               'num_leaves': 100, 
               'learning_rate': 0.2527335171237368, 
               'feature_fraction': 0.84182560868287, 
               'bagging_fraction': 0.831223440171023, 
               'max_depth': 9, 
               'min_data_in_leaf': 85}

# Add the fixed parameters that were in your objective function
best_params.update({
    "objective": "regression",
    "metric": "mae",
    "boosting": "gbdt",
    "verbose": -1,
    "feature_pre_filter": False
})


final_train = lgb.Dataset(train.drop(columns = ["date", "num_sold"]),
                         label=np.log1p(train["num_sold"]))

final_test = test.drop(columns = ["date", "id"])


# # Train the final model with the best parameters
final_model = lgb.train(
    best_params,
    final_train,
    num_boost_round=100
)


# Variable importance 
feature_importance = final_model.feature_importance(importance_type="gain")  # or "split"
feature_names = train.drop(columns = ["date", "num_sold"]).columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Visualization using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.show()


# This to me seems a little unusual that day related variables literally have no affect on the final outcome


# Visualizing Actual vs Pred 
# Convert to arrays for visualization
# all_preds = np.array(all_preds)
# all_actuals = np.array(all_actuals)

# # Plot predictions vs actuals
# plt.figure(figsize=(8, 8))
# plt.scatter(all_actuals, all_preds, alpha=0.6)
# plt.plot([all_actuals.min(), all_actuals.max()], [all_actuals.min(), all_actuals.max()], "r--", lw=2)
# plt.title("Predictions vs Actuals (All Folds)")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()


# # Now you can make predictions
y_pred = final_model.predict(final_test)


# Making this submission
submission = pd.concat([test["id"], pd.Series(np.expm1(y_pred), name="num_sold")], axis=1)
submission.to_csv("submission.csv", index=False)

