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


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")



# Function to display missing values and data types
def check_missing_and_dtypes(df, name="Dataset"):
    print(f"\n{name} Info:")
    print("-" * 50)
    nulls = df.isnull().sum()
    dtypes = df.dtypes
    summary = pd.DataFrame({
        "Data Type": dtypes,
        "Missing Values": nulls,
        "Missing (%)": (nulls / len(df)) * 100
    })
    print(summary)
    return summary


train_info = check_missing_and_dtypes(train, "Train Set")
test_info = check_missing_and_dtypes(test, "Test Set")


#univariate distribution of the features.
col = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
colors = sb.color_palette("husl", len(col))

fig, axes = plt.subplots(3, 3, figsize=(12, 8))  
axes = axes.flatten()


for i in range(len(col)):
    sb.histplot(train[col[i]], kde=True, ax=axes[i], color=colors[i])
    axes[i].set_ylabel("")


sb.countplot(x='Sex', data=train, ax=axes[len(col)])
axes[len(col)].set_ylabel("")


for j in range(len(col) + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#relationship between numeric features and calories burned
col = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]


fig, axes = plt.subplots(2,3, figsize = (8, 6) )

axes = axes.flatten()

for i in range(len(axes)):
    if i < len(col):
        sb.scatterplot(x = col[i], y = 'Calories', data = train, ax = axes[i])
        axes[i].set_ylabel("")
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()


"""
From the charts above, there is no apparent linear relationship between age, height, or weight and calories burned. 
However, a linear relationship appears to exist between calories and duration, heart rate, and body temperature.   

"""
#create categorical grouping for age
bins = [20, 35, 55, 80]
labels = ['Young Adults', 'Middle Age', 'Senior']

train['Age_Grp'] = pd.cut(train['Age'], bins = bins, labels = labels, right = True, include_lowest = True)
#test['Age_Grp'] = pd.cut(test['Age'], bins = bins, labels = labels, right = True, include_lowest = True)


#bivariate relationship between age grp and calories
sb.boxplot(x='Age_Grp', y='Calories', data=train)
plt.title('Calories distribution by Age Group')
plt.show()


""""
From the boxplot, the amount of calories burned tends to increase from young adults to seniors. 
Next, let's explore how calorie expenditure varies across sex and workout duration.
"""
g = sb.FacetGrid(train, col="Age_Grp", row="Sex", hue="Heart_Rate", palette="viridis", height=4)
g.map(sb.scatterplot, "Duration", "Calories")

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Calories vs Duration by Age Group and Sex (Heart Rate as color)')

plt.tight_layout()
plt.show()


# Encode 'Sex' to numeric values
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype(int)


# Note: These polynomial features might cause overfitting. 
# I've tried them and plan to focus on PCA and feature selection to mitigate this.

def feature_engineering(df):
    # Intensity index
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']

    # Log transformations 
    df['Age'] = np.log1p(df['Age'])
    df['Body_Temp'] = np.log1p(df['Body_Temp'])

    # Basal Metabolic Rate
    df['BMR'] = (
        10 * df['Weight'] + 
        6.25 * df['Height'] - 
        5 * df['Age'] + 
        np.where(df['Sex'] == 1, 5, -161)
    )

    # Core interactions
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['Age_Body_Temp'] = df['Age'] * df['Body_Temp']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']

    # Height & Weight interactions 
    df['Height_Weight'] = df['Height'] * df['Weight']
    df['Height_Duration'] = df['Height'] * df['Duration']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['Weight_HeartRate'] = df['Weight'] * df['Heart_Rate']
    df['Weight_BodyTemp'] = df['Weight'] * df['Body_Temp']
    df['Height_Temp_Interaction'] = df['Height'] * df['Body_Temp']
    df['Weight_Duration_Temp'] = df['Weight'] * df['Duration'] * df['Body_Temp']
    df['Height_Duration_Temp'] = df['Height'] * df['Duration'] * df['Body_Temp']
    df['Weight_HR_Duration'] = df['Weight'] * df['Heart_Rate'] * df['Duration']
    df['Height_HR_Duration'] = df['Height'] * df['Heart_Rate'] * df['Duration']

    # Advanced exertion interactions
    df['Weight_Intensity_Index'] = df['Weight'] * df['Intensity_Index']
    df['Height_Intensity_Index'] = df['Height'] * df['Intensity_Index']
    df['Weight_HR_Temp_Interaction'] = df['Weight'] * df['HR_Temp_Interaction']
    df['Height_HR_Temp_Interaction'] = df['Height'] * df['HR_Temp_Interaction']

    # Ratio and Normalized Features
    df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Temp_per_kg'] = df['Body_Temp'] / df['Weight']
    df['HR_per_cm'] = df['Heart_Rate'] / df['Height']
    df['Duration_per_cm'] = df['Duration'] / df['Height']
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Energy & exertion approximations
    df['Energy_Exerted'] = df['Weight'] * df['Heart_Rate'] * df['Duration'] / 10000
    df['Weighted_Intensity'] = df['Intensity_Index'] * df['Weight']

    # BMR interactions
    df['BMR_HR'] = df['BMR'] * df['Heart_Rate']
    df['BMR_Duration'] = df['BMR'] * df['Duration']
    df['BMR_Temp'] = df['BMR'] * df['Body_Temp']
    df['BMR_Intensity'] = df['BMR'] * df['Intensity_Index']

    # Polynomial and log features
    df['HR_Squared'] = df['Heart_Rate'] ** 2
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Temp_Squared'] = df['Body_Temp'] ** 2
    df['Log_HR'] = np.log1p(df['Heart_Rate'])

    # Sex-based interaction features
    df['Sex_male_HR'] = df['Heart_Rate'] * (df['Sex'] == 1)
    df['Sex_female_HR'] = df['Heart_Rate'] * (df['Sex'] == 0)

    df['Sex_male_Weight'] = df['Weight'] * (df['Sex'] == 1)
    df['Sex_female_Weight'] = df['Weight'] * (df['Sex'] == 0)


    # Log transform target (Calories) 
    if 'Calories' in df.columns:
        df['Calories'] = np.log1p(df['Calories'])

    return df


# Apply to both datasets
train = feature_engineering(train)
test = feature_engineering(test)


test_df = test.drop(['id'], axis = 1)

x = train.drop(['Calories', 'id', 'Age_Grp'], axis = 1)
y = train['Calories']


#split x and y into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 4)


## Define custom RMSLE metric
def rmsle(y_val_log, y_pred_log):
    # Reverse log1p transformation
    y_val = np.expm1(y_val_log)
    y_pred = np.expm1(y_pred_log)
    preds = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_val, preds))

# Define the Optuna objective function
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 7000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 16),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log = True),
        'grow_policy': 'SymmetricTree',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 34,
        'early_stopping_rounds': 100,
        'use_best_model': True,
        'verbose': 0
    }


    train_pool = Pool(data=x_train, label=y_train)
    valid_pool = Pool(data=x_val, label=y_val)


    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool)


    preds = model.predict(valid_pool)
    score = rmsle(y_val, preds)
    return score


# Train CatBoost with the best parameters obtained from the Optuna hyperparameter tuning above.
cat_model = CatBoostRegressor(
    iterations = 4788,
    learning_rate = 0.021048601257810478,
    l2_leaf_reg = 2.211703712973945,
    grow_policy = 'SymmetricTree',
    loss_function = 'RMSE', 
    eval_metric='RMSE',
    random_seed = 34,
    
    depth = 14,
    verbose = 0
)

cat_model.fit(x_train, y_train,
             eval_set = (x_val, y_val),
             use_best_model = True)


cat_pred = cat_model.predict(x_val)

cat_pred = np.expm1(cat_pred)
y_val = np.expm1(y_val)

rmsle = np.sqrt(mean_squared_log_error(cat_pred, y_val))

rmsle


rmsle


# Extract and Plot RMSE per Iteration

evals_result = cat_model.evals_result_

# Plot RMSE for training and validation
plt.figure(figsize=(10, 6))
plt.plot(evals_result['learn']['RMSE'], label='Train RMSE')
plt.plot(evals_result['validation']['RMSE'], label='Validation RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('CatBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.show()





# Get feature importance scores
feature_importances = cat_model.get_feature_importance()
feature_names = x_train.columns


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'][:25][::-1], importance_df['Importance'][:25][::-1])
plt.xlabel("Importance")
plt.title("Top 25 Feature Importances")
plt.tight_layout()
plt.show()


top_features_25 = importance_df.sort_values(by='Importance', ascending=False).head(25)['Feature'].tolist()

x_top25 = x[top_features_25]
x_val_25 = test_df[top_features_25]


# Define model
cat_model_top25 = CatBoostRegressor(
    iterations=4788,
    learning_rate=0.021048601257810478,
    l2_leaf_reg=2.211703712973945,
    grow_policy='SymmetricTree',
    loss_function='RMSE',
    eval_metric='RMSE',
    depth=14,
    verbose=0,
    random_seed=34
)

# RMSLE function
def rmsle(y_val_log, y_pred_log):
    y_val = np.expm1(y_val_log)
    y_pred = np.expm1(y_pred_log)
    preds = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_val, preds))

# KFold setup
cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = []
test_preds = np.zeros(len(x_val_25))


for fold, (train_idx, val_idx) in enumerate(cv.split(x_top25, y)):
    print(f"Training fold {fold + 1}...")

    x_tr, x_val = x_top25.iloc[train_idx], x_top25.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    
    model = cat_model_top25.fit(x_tr, y_tr)

    
    y_val_pred_log = model.predict(x_val)
    fold_score = rmsle(y_val, y_val_pred_log)
    cv_scores.append(fold_score)

    # Predict on test set
    fold_preds = np.expm1(model.predict(x_val_25))
    fold_preds = np.maximum(0, fold_preds)
    test_preds += fold_preds
    

# Average the predictions
test_preds /= cv.get_n_splits()

# Print CV scores
print("RMSLE CV scores:", cv_scores)
print("Mean RMSLE CV score:", np.mean(cv_scores))





# Save for submission
submission = pd.DataFrame({
    "id": test["id"],  # replace 'id' with your ID column
    "Calories": test_preds
})
submission.to_csv("submission.csv", index=False)




