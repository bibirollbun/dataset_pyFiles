# This cell might look complicated
# Code takes 2 inputs as train_df and train_org
# train_df is dataset that comes from Predict Calorie Expenditure Competition
# train_org is a dataset that I added, it is an extra dataset
# In order to increase accuracy of the trainer as train_org and train_df are getting concatted as train


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')



train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_org = pd.read_csv('/kaggle/input/calories-extra/calories.csv')
train = (
    pd.concat([train_df, train_org], ignore_index=True)
      .drop_duplicates()
      .reset_index(drop=True)
)

test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
print("cell 2 done")








# This cell defines numerical_features and creates new features to increase accuracy of the trainer as following:
# Creating BMI and Age_x_Dur columns
# BMI = weight / (height in meters)^2
# "Age_x_Dur" = Age × Duration
# "HR_x_Dur"  = Heart_Rate × Duration
# "Temp_x_Dur"= Body_Temp × Duration



test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


for df in [train, test]:
    df['BMI']        = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Age_x_Dur']  = df['Age'] * df['Duration']
    df['HR_x_Dur']   = df['Heart_Rate'] * df['Duration']
    df['Temp_x_Dur'] = df['Body_Temp'] * df['Duration']




numerical_features = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'BMI', 'Age_x_Dur', 'HR_x_Dur', 'Temp_x_Dur'
]
print("done")


# The code starts by creating lots of new features from the existing numeric columns—things like multiplying each pair of features together and
# adding-subtracting them, taking their ratios,
# and also computes mean, std, max, min, median
# Then, it turns “Sex” into a 0/1 category.
# Then stick them back onto the original tables. Finally, it drops “id” and “Calories” to make the feature matrix X,
# Takes the log of “Calories” for y, and keep X_test ready for the model comes next.


import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[numerical_features])
poly_test = poly.transform(test[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

train = pd.concat([train.reset_index(drop=True), poly_train_df], axis=1)
test = pd.concat([test.reset_index(drop=True), poly_test_df], axis=1)

X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])


FEATURES = X.columns.tolist()



# Here it sets up a 7‐fold cross‐validation and define three  regressors as CatBoost, XGBoost and LightGBM
# For each model, it loops over the folds, split our feature matrix X and target y into training and validation slices,
# and train the model on train split while using the validation split for early stopping(if it is available).
# After fitting, iit predicts on the validation fold  and
# Then predict on the full test set, accumulating an average over folds.
# After that computes RMSLE for each fold to keep track of how well the model is doing, print out each fold’s score and timing,
# And finally prinst the mean and standard deviation of RMSLE across all folds for each model so we can compare their performance.



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
models = {
    'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000, learning_rate=0.02,
                            gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
                            enable_categorical=True, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
                              subsample=0.9, random_state=42, verbose=-1)
}

results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
        
        x_train = x_train.loc[:, ~x_train.columns.duplicated()]
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
        x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

        start = time.time()
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)

        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        
        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)
        
        print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")


print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")




# Firstly the code takes each model’s out‐of‐fold predictions and test predictions,
# Then defines a  RMSLE loss that uses CatBoost, XGBoost, and LightGBM outputs with weights summing to one weight.
# By using SciPy’s minimize function, it finds the best of the best three weights that minimize validation RMSLE.
# Prints those weights,
# After that combines the test‐set predictions accordingly, clip them to a reasonable range,
# Finally writes them into the submission file reports the mean and median of the calorie estimates.



from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error

oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
test_preds = {name: np.expm1(results[name]['pred']) for name in results}
y_true = np.expm1(y)

def rmsle_loss(weights):
    blended = (
        weights[0] * oof_preds['CatBoost'] +
        weights[1] * oof_preds['XGBoost'] +
        weights[2] * oof_preds['LightGBM']
    )
    return np.sqrt(mean_squared_log_error(y_true, blended))

initial_weights = [1/3, 1/3, 1/3]
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * 3

res = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"\n Optimized Weights:")
print(f"CatBoost = {best_weights[0]:.4f}")
print(f"XGBoost  = {best_weights[1]:.4f}")
print(f"LightGBM = {best_weights[2]:.4f}")

blended_preds = (
    best_weights[0] * test_preds['CatBoost'] +
    best_weights[1] * test_preds['XGBoost'] +
    best_weights[2] * test_preds['LightGBM']
)

blended_preds = np.clip(blended_preds, 1, 314)

submission['Calories'] = blended_preds
submission.to_csv('submission.csv', index=False)

print("\nSubmission Head:")
print(submission.head())

print(f"\nPredicted Mean: {blended_preds.mean():.2f}")
print(f"Predicted Median: {np.median(blended_preds):.2f}")



# df1, df2, and df3 are all similar inputs to our competition that I imported by dowloading them as dataset

# While we create the final submission, the code uses df1, df2, and df3 by weights of 40%, 30%, and 30%.



import pandas as pd
import numpy as np

df1 = pd.read_csv("/kaggle/input/caloriecast-adaptive-ensemble-engine-for-s5e5/submission.csv")
df2 = pd.read_csv("/kaggle/input/ensemble-of-solutions/submission.csv")
df3 = pd.read_csv("/kaggle/input/ps-s5e5-log-blended-cat-xgboost-with-50-fold-cv/ensemble_submission.csv")


final_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")  

final_submission['Calories'] = (0.4 * df1['Calories']) + (0.3 * df2['Calories'])+(.3 * df3['Calories'])
final_submission.to_csv('submission.csv', index=False)


# This cell is the plot part and I will explain it by putting titles


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# New features
for df in (train, test):
    df['Sex']        = df['Sex'].astype(str).str.lower().map({'male': 1, 'female': 0})
    df['BMI']        = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Age_x_Dur']  = df['Age'] * df['Duration']
    df['HR_x_Dur']   = df['Heart_Rate'] * df['Duration']
    df['Temp_x_Dur'] = df['Body_Temp'] * df['Duration']

feature_cols = [
    "Sex","Age","Height","Weight","Duration",
    "Heart_Rate","Body_Temp","BMI","Age_x_Dur",
    "HR_x_Dur","Temp_x_Dur"
]
target_col = "Calories"

# 1) .describe() part (Summary)
print(train[["Age","Duration","Heart_Rate","Body_Temp","Calories"]].describe().T)

# 2) Sex Distribution (pie chart)
sex_counts = train['Sex'].value_counts().sort_index()
plt.figure(figsize=(6,6))
plt.pie(sex_counts, labels=['Female','Male'], autopct="%1.1f%%", startangle=90)
plt.title("Sex Distribution")
plt.axis('equal')
plt.show()

# 3) Calories Distribution (as a violin plot)
plt.figure(figsize=(8,4))
sns.violinplot(x=train['Calories'], inner="box")
plt.title("Calories Distribution")
plt.xlabel("Calories")
plt.show()

# 4) Age, Duration and Calories scatter matrix (no color)
cm = scatter_matrix(
    train[["Age","Duration","Calories"]],
    figsize=(8,8),
    diagonal='hist'
)
for ax in cm.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=8)
    ax.set_ylabel(ax.get_ylabel(), fontsize=8)
plt.suptitle("Age–Duration–Calories Relation", y=0.93)
plt.show()

# 5) Feature Correlation Heat Map (default colormap)
corr = train[["Age","Duration","Heart_Rate","Body_Temp","BMI","Calories"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(6,5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0)
plt.title("Feature Correlation")
plt.show()

# 6) Average Calories Burnt by Duration (barplot)
avg_dur = train.groupby("Duration")["Calories"].mean().reset_index()
plt.figure(figsize=(8,4))
sns.barplot(x="Duration", y="Calories", data=avg_dur, edgecolor="k")
plt.xticks(rotation=-45)
plt.title("Average Calories Burned by Duration")
plt.xlabel("Duration (min)")
plt.ylabel("Mean Calories")
plt.show()

# 7) Heart Rate vs Calories
plt.figure(figsize=(8,4))
sns.scatterplot(x="Heart_Rate", y="Calories", data=train)
plt.title("Heart Rate vs Calories")
plt.xlabel("Heart Rate")
plt.ylabel("Calories")
plt.show()

# 8) Body Temp vs Calories
plt.figure(figsize=(8,4))
sns.scatterplot(x="Body_Temp", y="Calories", data=train)
plt.title("Body Temp vs Calories")
plt.xlabel("Body Temp")
plt.ylabel("Calories")
plt.show()

# 9) Calories vs New Added features
for feat in ['BMI','Age_x_Dur','HR_x_Dur','Temp_x_Dur']:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=feat, y="Calories", data=train)
    plt.title(f"{feat} vs Calories")
    plt.xlabel(feat)
    plt.ylabel("Calories")
    plt.show()





