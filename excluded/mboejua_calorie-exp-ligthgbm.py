import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error,make_scorer

warnings.filterwarnings('ignore')


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
train.head()


train.info()  # No Missing value


train['Sex'].value_counts()



#Feature Engineering
def features(df):
    df = df.copy()
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Cardio'] = df['Heart_Rate'] * df['Duration']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['BP'] = df['Weight'] * df['Heart_Rate']
    
    return df



train = features(train)
test = features(test)



#Explore distribution spread
train.describe()


num_cols = train.select_dtypes(['int','float64']).columns
num_cols = [x for x in num_cols if x not in ['Calories','Age']]
num_cols


fig, axes = plt.subplots(4, 3, figsize=(20,15), constrained_layout=True)
axes = axes.flatten()

#for x,  y in zip(train[num_cols].columns, axes):
#    sns.histplot(data=train, x=x, ax=y)
#plt.show()

for x,  y in zip(train[num_cols].columns, axes):
    sns.histplot(data=train,ax=y , x=x, #y='Calories',
                 hue='Sex', kde=True, bins=20)
    y.set_title(f"Histogram plot of {x}", fontsize=10)
    y.set_xlabel(x, fontsize=10)
    y.set_ylabel("count" ,fontsize=10)
    y.tick_params(axis='both', labelsize=10)
plt.show()


fig, axes = plt.subplots(4, 3, figsize=(20,15), constrained_layout=True)
axes = axes.flatten()

for x,  y in zip(train[num_cols].columns, axes):
    sns.scatterplot(data=train,ax=y , x=x, y='Calories',
                 hue='Sex', palette='coolwarm')
    y.set_title(f"Scatter plot of {x} vs Calories", fontsize=10)
    y.set_xlabel(x, fontsize=10)
    y.set_ylabel("count" ,fontsize=10)
    y.tick_params(axis='both', labelsize=10)
plt.show()


train_y = train.pop('Calories')


X_train, X_test, y_train, y_test = train_test_split(train, train_y, random_state=42, test_size=0.2)


rmsle_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)),
    greater_is_better=False
)

model = LGBMRegressor(random_state=42,verbose=-1)


grid = {
    'num_leaves': [128, 256],
    'learning_rate': [ 0.05],
    'n_estimators': [1000],
    'max_depth':[8,10]
}

# Grid search
grid_search = GridSearchCV(estimator=model, param_grid=grid, 
                           cv=5, scoring=rmsle_scorer, verbose=-1, n_jobs=-1
                          )

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)



results_df = pd.DataFrame(grid_search.cv_results_)
results_df.sort_values('rank_test_score', inplace=True)
results_df.head()




bm = grid_search.best_estimator_
y_pred = np.clip(bm.predict(X_test), 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"Test RMSLE: {rmsle:.4f}")


importance = bm.feature_importances_
features = X_train.columns  
sorted_idx = np.argsort(importance)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importance[sorted_idx], color="greenyellow")
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('LightGBM Regression Feature Importance')
plt.gca().invert_yaxis()
plt.show()


y_pred = np.clip(bm.predict(test), 0, None)
res = pd.DataFrame({"id":test.index,
                    "Calories": y_pred})

#res = res.reset_index()
res = res.sort_values("id")
res.to_csv('submission.csv', index=False)
res.head(10)

