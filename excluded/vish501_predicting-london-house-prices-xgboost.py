import os # Files in the dataset

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

from matplotlib.pylab import rcParams
from sklearn.impute import SimpleImputer # To fill NaN using strategies
from sklearn.metrics import mean_absolute_error # Error calculation
from sklearn.model_selection import train_test_split # Splitting the dataset into train and valid
from xgboost import XGBRegressor, plot_tree # XGBoost


# Setting enviorment
pd.plotting.register_matplotlib_converters()
pd.options.display.float_format = '{:20.4f}'.format

warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline


# Reading all the filepaths in the dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_dataset_url = '/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv'
test_dataset_url = '/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv'


housing_data_train = pd.read_csv(train_dataset_url, index_col='ID')
housing_data_train.head()


housing_data_train.info(verbose=True)


housing_data_train.describe(include='all')


# 'fullAddress', 'postcode', and 'outcode': Do not seem to be encodeable
# 'country': There is only one unique country in the dataset
# As such dropping these from the dataset before further analysis

cols_to_drop = ['fullAddress', 'postcode', 'outcode', 'country']
housing_data_train.drop(cols_to_drop, axis=1, inplace=True)
housing_data_train


housing_data_train.hist(bins=50, figsize=(15,8))
plt.suptitle('Distribution of Housing Data')
plt.show()


housing_data_train.plot(kind='box', subplots=True, sharey=False, figsize=(18, 6))
plt.subplots_adjust(wspace=0.5) 
plt.suptitle('Quartile Distribution of Housing Data')
plt.show()


print(f'Unique Number of tenures in the Dataset: {housing_data_train.tenure.unique().size}')
print(f'Unique Number of propertyTypes in the Dataset: {housing_data_train.propertyType.unique().size}')
print(f'Unique Number of currentEnergyRatings in the Dataset: {housing_data_train.currentEnergyRating.unique().size}')


# Strip-ploting categorical columns NAN data
cols = ['tenure', 'propertyType', 'currentEnergyRating']

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    sns.stripplot(data=housing_data_train[housing_data_train[cols[i]].isna()], x=cols[i], y='price', ax=ax)

plt.tight_layout()
plt.show()


# Checking missing data in training set
missing_data = housing_data_train.isna().sum()
missing_percentage = missing_data / len(housing_data_train) * 100

print(missing_data, '\n')
print(missing_percentage)


# Checking missing data in test dataset
housing_data_test = pd.read_csv(test_dataset_url, index_col='ID')
housing_data_test.drop(cols_to_drop, axis=1, inplace=True)

missing_data = housing_data_test.isna().sum()
missing_percentage = missing_data / len(housing_data_test) * 100

print(missing_data, '\n')
print(missing_percentage)


# Seperating target from predictors
target = housing_data_train.price.copy()
housing_data_train.drop(['price'], axis=1, inplace=True)

# Select categorical columns with relatively low cardinality and numeric values
low_cardinality_cols = [cname for cname in housing_data_train.columns if housing_data_train[cname].nunique() < 10 and housing_data_train[cname].dtype == "object"]
numeric_cols = [cname for cname in housing_data_train.columns if housing_data_train[cname].dtype in ['int64', 'float64']]

# Keeping selected columns only
my_cols = low_cardinality_cols + numeric_cols
housing_train_trimmed = housing_data_train[my_cols].copy()
housing_test_trimmed = housing_data_test[my_cols].copy()

# One-hot encode the data
X_train_full = pd.get_dummies(housing_train_trimmed)
X_test = pd.get_dummies(housing_test_trimmed)

assert X_train_full.shape[0] == housing_train_trimmed.shape[0]
assert X_test.shape[0] == housing_test_trimmed.shape[0]

# Making sure test has same columns as train after one-hot encode
X_train_full, X_test = X_train_full.align(X_test, join='left', axis=1)

assert X_train_full.shape[1] >= X_test.shape[1]

# Breaking train into train and test
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, target, train_size=0.8, test_size=0.2, random_state=1234)

assert X_train.shape[0] + X_valid.shape[0] == X_train_full.shape[0]


# XGBRegressor model without early stopping hyper-parameter
def XGB_Modeler(X_train, X_valid, y_train, y_valid, n_estimators=1000, max_depth=7, learning_rate=0.01, random_state=1234):
    
    # Inititalizing the model
    XGB_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    
    # Fitting the model
    XGB_model.fit(X_train, y_train, verbose=False)

    # Validating the model using MAE 
    predication = XGB_model.predict(X_valid)
    mae = mean_absolute_error(predication, y_valid)
    
    return XGB_model, mae


# XGBRegressor model with early stopping hyper-parameter
def XGB_Modeler_Early(X_train, X_valid, y_train, y_valid, n_estimators=1000,
                min_split_loss=2, n_jobs=4, early_stopping_rounds=10,
                max_depth=7, learning_rate=0.01, random_state=1234):
    
    # Inititalizing the model
    XGB_model = XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             n_jobs=n_jobs,
                             early_stopping_rounds=early_stopping_rounds,
                             min_split_loss=min_split_loss,
                             random_state=random_state,)
    
    # Fitting the model
    XGB_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # Validating the model using MAE 
    predication = XGB_model.predict(X_valid)
    mae = mean_absolute_error(predication, y_valid)
    
    return XGB_model, mae


# Testing using XGB_Modeler_Early
def XGB_Modeler_Early_Tester(X_train, X_valid, y_train, y_valid):
    
    # Hyper-parameters to test with
    max_depths = [5,7,9]
    learning_rate = [0.05,0.01,0.001]

    # Storing optimal model and rates for other models
    model_best = None
    mea_best = None
    
    model_rates = {}
    count = 0

    # Testing hyper-parameters
    for i in max_depths:
        for j in learning_rate:
            model, mae = XGB_Modeler_Early(X_train, X_valid, y_train, y_valid, max_depth=i, learning_rate=j)
            if not mea_best or mae < mea_best:
                model_best = model
                mea_best = mae
                
            count += 1
            model_rates[count] = {'Max Depth': i, 'Alpha': j, 'MAE': mae}

    # Converting model rates data to a dataset for easier reading
    models_df = pd.DataFrame.from_dict(model_rates, orient='index').sort_values(by='MAE').reset_index(drop=True)

    return model_best, mea_best, models_df


#model_m1, mea_m1, models_m1 = XGB_Modeler_Early_Tester(X_train, X_valid, y_train, y_valid)
#models_m1


# Test 2 - NaNs converted to 0s
X_train_m2 = X_train.fillna(0)
X_valid_m2 = X_valid.fillna(0)
X_test_m2 = X_test.fillna(0)

#model_m2, mea_m2, models_m2 = XGB_Modeler_Early_Tester(X_train_m2, X_valid_m2, y_train, y_valid)
#models_m2


# Test 3 - Filling with closest value interpolate(method='nearest')
X_train_m3 = X_train.interpolate(method='nearest')
X_valid_m3 = X_valid.interpolate(method='nearest')
X_test_m3 = X_test.interpolate(method='nearest')

#model_m3, mea_m3, models_m3 = XGB_Modeler_Early_Tester(X_train_m3, X_valid_m3, y_train, y_valid)
#models_m3


# Test 4 - Filling with mean SimpleImputer(missing_values=nan, strategy='mean')
imputer_mean = SimpleImputer(strategy='mean')

X_train_m4 = pd.DataFrame(imputer_mean.fit_transform(X_train), columns = X_train.columns)
X_valid_m4 = pd.DataFrame(imputer_mean.transform(X_valid), columns = X_train.columns)
X_test_m4 = pd.DataFrame(imputer_mean.transform(X_test), columns = X_train.columns)

#model_m4, mea_m4, models_m4 = XGB_Modeler_Early_Tester(X_train_m4, X_valid_m4, y_train, y_valid)
#models_m4


# Test 5 - Filling with mean SimpleImputer(missing_values=nan, strategy='median')
imputer_mean = SimpleImputer(strategy='median')

X_train_m5 = pd.DataFrame(imputer_mean.fit_transform(X_train), columns = X_train.columns)
X_valid_m5 = pd.DataFrame(imputer_mean.transform(X_valid), columns = X_train.columns)
X_test_m5 = pd.DataFrame(imputer_mean.transform(X_test), columns = X_train.columns)

#model_m5, mea_m5, models_m5 = XGB_Modeler_Early_Tester(X_train_m5, X_valid_m5, y_train, y_valid)
#models_m5


# It seems Method 3 performed the best on the leaderboards for the competition
X_train_final = X_train.interpolate(method='nearest')
X_valid_final = X_valid.interpolate(method='nearest')
X_test_final = X_test.interpolate(method='nearest')

model_final, _ = XGB_Modeler_Early(X_train_final, X_valid_final, y_train, y_valid, max_depth=7, learning_rate=0.01)
preds_test_final = model_final.predict(X_test_final)


# Plotting the decisions tree
rcParams['figure.figsize'] = 50,50
plot_tree(model_final, rankdir='LR')


# Saving test predictions to file
output = pd.DataFrame({'ID': X_test.index,
                       'price': preds_test_final})
output.to_csv('submission.csv', index=False)

