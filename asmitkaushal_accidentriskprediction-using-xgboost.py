import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.core.reshape.encoding import get_dummies
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from math import sqrt


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')



train.head()


test.head()


train.info()


train.describe()


print("Missing values in train:", sum(train.isnull().sum()))
print("Missing values in test:", sum(test.isnull().sum()))



print("Duplicate values in train:", train.duplicated().sum())
print("Duplicate values in test:", test.duplicated().sum())




#check columns in test and train are same or not
train.columns[:-1] == test.columns


#numeric and categorical feature list removing the target variable and id 
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
numeric_features = numeric_features.drop(['id', 'accident_risk'])

categorical_features = train.select_dtypes(include=['object']).columns
boolean_features = train.select_dtypes(include=['bool']).columns

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Boolean features:", boolean_features)



# Distribution of Target Variable
fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.2) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(train['accident_risk'], bins=30, kde=True, color='saddlebrown', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of accident_risk-Target Variable")
ax_hist.set_xlabel("")

sns.boxplot(x=train['accident_risk'], ax=ax_box, color='saddlebrown')
ax_box.set_xlabel("accident_risk")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()


# Distribution of numeric features with histplot
train[numeric_features].hist(bins=25, figsize=(15,10), color='saddlebrown')
plt.suptitle('Distribution of numeric features', fontsize=20)
plt.show()



#show the unique values of categorical features
categorical_features = train.select_dtypes(include=['object']).columns
print("The Unique values of categorical features are:\n")
for i in categorical_features:
    print(i,train[i].unique())
#plot a pieplot for each categorical feature make it in a single figure with two above and two down
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].pie(train[categorical_features[0]].value_counts(), labels=train[categorical_features[0]].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['teal', 'yellow', 'blue'])
axes[0, 0].set_title(categorical_features[0], fontsize=12)

# Pie chart 2 (top-right)
axes[0, 1].pie(train[categorical_features[1]].value_counts(), labels=train[categorical_features[1]].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['teal', 'yellow', 'blue'])
axes[0, 1].set_title(categorical_features[1], fontsize=12)

# Pie chart 3 (bottom-left)
axes[1, 0].pie(train[categorical_features[2]].value_counts(), labels=train[categorical_features[2]].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['teal', 'yellow', 'blue'])
axes[1, 0].set_title(categorical_features[2], fontsize=12)

# Pie chart 4 (bottom-right)
axes[1, 1].pie(train[categorical_features[3]].value_counts(), labels=train[categorical_features[3]].value_counts().index, autopct='%1.1f%%', startangle=90, colors=['teal', 'yellow', 'blue'])
axes[1, 1].set_title(categorical_features[3], fontsize=12)
plt.show()



# Create a boxplot for each feature
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes = axes.flatten()

for i, feature in enumerate(train[numeric_features]):
    sns.boxplot(data=train, y=feature, ax=axes[i], color='teal')
    axes[i].set_title(f'Boxplot of {feature}', fontsize=12)
    axes[i].set_ylabel(feature)

plt.tight_layout()
plt.show()


#Heatmap 
#include all the features excludong id and category features
plt.figure(figsize=(15, 10))
plt.title('Correlation Heatmap', fontsize=20)
sns.heatmap(train[['num_lanes', 'curvature', 'speed_limit', 'road_signs_present', 'public_road', 'holiday', 'school_season', 'num_reported_accidents', 'accident_risk']].corr(), annot=True, cmap='seismic', fmt='.2f', linewidths=0.5)
plt.show()




train_engineered = train.copy()
test_engineered = test.copy()


#Drop id column
train_engineered.drop(['id'], axis=1, inplace=True)
test_engineered.drop(['id'], axis=1, inplace=True)


#label encoding the categorical features
for col in categorical_features:
    le = LabelEncoder()
    train_engineered[col] = le.fit_transform(train_engineered[col])
    test_engineered[col] = le.transform(test_engineered[col])



#Boolean features encoding
train_engineered[boolean_features] = train_engineered[boolean_features].astype(int)
test_engineered[boolean_features] = test_engineered[boolean_features].astype(int)


#  Curvature × Multiple Lanes
train_engineered['curve_lane_risk'] = train_engineered['curvature'] * train_engineered['num_lanes']
#  Extreme Risk Flag (triple threat)
train_engineered['extreme_risk'] = (
    (train_engineered['weather'] != 'clear') & 
    (train_engineered['lighting'] != 'daylight') & 
    (train_engineered['speed_limit'] > 60)
).astype(int)
# Speed × Lanes - High likelihood of success
train_engineered['speed_lanes_risk'] = train_engineered['speed_limit'] * train_engineered['num_lanes']

#sharp curve
train_engineered['sharp_curve_high_speed'] = (
    (train_engineered['curvature'] > train_engineered['curvature'].quantile(0.75)) & 
    (train_engineered['speed_limit'] > 60)
).astype(int)

print("Added 4 new features")
print(f"Total features: {train_engineered.shape[1]}")

# Display new features
new_features = ['curve_lane_risk', 
                 'extreme_risk','speed_lanes_risk',
                 'sharp_curve_high_speed',]
                 
print("\nNew features created:")
for feat in new_features:
    print(f"  • {feat}")


#for test dataset

#  Curvature × Multiple Lanes
test_engineered['curve_lane_risk'] = test_engineered['curvature'] * test_engineered['num_lanes']
#  Extreme Risk Flag (triple threat)
test_engineered['extreme_risk'] = (
    (test_engineered['weather'] != 'clear') & 
    (test_engineered['lighting'] != 'daylight') & 
    (test_engineered['speed_limit'] > 60)
).astype(int)
# Speed × Lanes - High likelihood of success
test_engineered['speed_lanes_risk'] = test_engineered['speed_limit'] * test_engineered['num_lanes']

#sharp curve
test_engineered['sharp_curve_high_speed'] = (
    (test_engineered['curvature'] > test_engineered['curvature'].quantile(0.75)) & 
    (test_engineered['speed_limit'] > 60)
).astype(int)


#remove features that are not required
train_engineered = train_engineered.drop(columns=['road_signs_present','time_of_day','school_season'])



test_engineered = test_engineered.drop(columns=['road_signs_present','time_of_day','school_season'])


train_engineered.head()


#heatmap
plt.figure(figsize=(12,10))
sns.heatmap(train_engineered.corr(), annot=True, cmap='seismic', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()



#prepare the data
X = train_engineered.drop(columns=['accident_risk'])
y = train_engineered['accident_risk']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#model training
from tabnanny import verbose

def train_models(X_train, y_train,X_test,y_test):
    models = {
        'RandomForest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor(verbose=-1),
        'CatBoost': CatBoostRegressor(verbose=0)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        print(f'{name}: RMSE = {rmse}')

train_models(X_train, y_train,X_test,y_test)        


from sklearn.model_selection import RandomizedSearchCV

# # Define the parameter grid
# param_dist = {
#     'iterations': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'depth': [4, 6, 8],
#     'l2_leaf_reg': [1, 3, 5, 7, 9],
#     'random_strength': [0.5, 1.0, 1.5],
#     'bagging_temperature': [0.5, 1.0, 2.0]
# }

# # Initialize the base model
# cat_model = CatBoostRegressor(verbose=0)

# # Create the RandomizedSearchCV object
# random_search = RandomizedSearchCV(
#     estimator=cat_model,
#     param_distributions=param_dist,
#     n_iter=10,             
#     cv=5,         
#     scoring='neg_root_mean_squared_error', 
#     n_jobs=-1,    
#     verbose=1              
# )

# # Fit the model
# random_search.fit(X_train, y_train)

# # Get the best model
# best_model = random_search.best_estimator_
# print("Best parameters:", random_search.best_params_)
# print("Best score:", random_search.best_score_)


# Define the parameter grid for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize XGBoost model
xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the model
random_search.fit(X_train, y_train)
# Get the best model
best_xgb_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


#predict on the best_model for test.csv and create a submisson file with id, accident_risk columns

id = test["id"]
ypred = best_xgb_model.predict(test_engineered)
submission = pd.DataFrame({"id": id, "accident_risk": ypred})
submission.to_csv("submission.csv", index=False)

