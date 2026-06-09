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


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer


import optuna


import joblib


!pip install fancyimpute


from fancyimpute import IterativeImputer


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df_train.head()


df_train.isna().sum()


train = df_train.copy()


print("Longueur du dataset avant supression: ", len(df_train))
train.fillna(0)
train.dropna(inplace = True)
print("Longueur du dataset après supression: ", len(train))


train.drop(["id","Podcast_Name","Episode_Title"], axis = 1, inplace = True)


train["Publication-Time"] = train["Publication_Day"] + "_" + train["Publication_Time"]
train["Number_ads_by_min"] = round(train["Number_of_Ads"] / train["Episode_Length_minutes"], 8)
train.head()


train["Week-end"] = np.where(train["Publication_Day"].isin(["Saturday", "Sunday"]), "Oui", "Non")
count_week_end = train["Week-end"].value_counts()

colors = ['#66b3ff', '#99ff99']


fig, ax = plt.subplots()
ax.pie(count_week_end, labels=count_week_end.index, autopct='%1.1f%%', colors=colors, radius=1,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=False)

ax.set_title('Répartition Week-end / Semaine')

plt.show()



custom_order = [
    "Monday_Morning", "Monday_Afternoon", "Monday_Night",
    "Tuesday_Morning", "Tuesday_Afternoon", "Tuesday_Night",
    "Wednesday_Morning", "Wednesday_Afternoon", "Wednesday_Night",
    "Thursday_Morning", "Thursday_Afternoon", "Thursday_Night",
    "Friday_Morning", "Friday_Afternoon", "Friday_Night",
    "Saturday_Morning", "Saturday_Afternoon", "Saturday_Night",
    "Sunday_Morning", "Sunday_Afternoon", "Sunday_Night"
]

train["Publication-Time"] = pd.Categorical(train["Publication-Time"], categories=custom_order, ordered=True)



counts = train["Publication-Time"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(counts.index, counts.values, marker='o')
ax.set_title("Distribution des heures de publication")
ax.set_xlabel("Heure de publication")
ax.set_ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(train.drop("Listening_Time_minutes", axis = 1),train["Listening_Time_minutes"], shuffle = True, random_state = 42)


object_columns = X_train.select_dtypes(include=['object', "string"]).columns.tolist()
nb_columns = X_train.select_dtypes(include=['int64', "float64"]).columns.tolist()
print("Object columns names -> ", object_columns )
print("Number columns names -> ", nb_columns)


preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(), object_columns),  
        ('scaler', RobustScaler(), nb_columns)  
    ]
)


pipe = Pipeline([
    ('Preprocessor', preprocessor), 
    ('Lasso', Lasso())
])

predict = pipe.fit(X_train,y_train).predict(X_test)
print("RMSE de la pipeline: " , np.sqrt(mean_squared_error(y_test, predict)))


def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-6, 1e+6, log=True)  
    tol = trial.suggest_float('tol',  0.00001,0.001, log=True)  
    model = Lasso(alpha=alpha, max_iter = 10000, tol = tol)
    pipe = Pipeline([
        ('Preprocessor', preprocessor),  
        ('Lasso', model)  
    ])
    score = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return np.sqrt(np.mean(-score))

study = optuna.create_study(direction='minimize')  

study.optimize(objective, n_trials=100)

print(f"Meilleur alpha trouvé : {study.best_params['alpha']}")
print(f"Meilleur tol trouvé : {study.best_params['tol']}")


best_alpha = study.best_params['alpha']
best_tol = study.best_params['tol']



best_lasso_model = Lasso(alpha=study.best_params['alpha'], tol = study.best_params['tol'] , max_iter = 10000)

final_pipe = Pipeline([
    ('Preprocessor', preprocessor),  
    ('Lasso', best_lasso_model)  
])

final_pipe.fit(X_train, y_train)

y_pred = final_pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE sur les données de test : {rmse}")


fitted_preprocessor = pipe.named_steps['Preprocessor']

num_features = fitted_preprocessor.transformers_[1][2]  # or nb_columns directly

ohe = fitted_preprocessor.named_transformers_['ohe']
cat_features = ohe.get_feature_names_out(object_columns)

feature_names = np.concatenate([cat_features, num_features])


coefficients = final_pipe.named_steps['Lasso'].coef_


feature_names


coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Importance': np.abs(coefficients)
})

coef_df = coef_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
plt.barh(coef_df['Feature'], coef_df['Importance'], color='steelblue')
plt.gca().invert_yaxis() 
plt.xlabel('Absolute Coefficient (Importance)')
plt.title('Lasso Feature Importance')
plt.tight_layout()
plt.show()


joblib.dump(final_pipe, 'model_pipeline.pkl')


train_mice = df_train.copy()
train_mice.drop(["id","Podcast_Name","Episode_Title"], axis = 1, inplace = True)
train_mice["Publication-Time"] = train_mice["Publication_Day"] + "_" + train_mice["Publication_Time"]
train_mice["Number_ads_by_min"] = round(train["Number_of_Ads"] / train_mice["Episode_Length_minutes"], 8)
train_mice["Week-end"] = np.where(train_mice["Publication_Day"].isin(["Saturday", "Sunday"]), "Oui", "Non")


X_train, X_test, y_train, y_test = train_test_split(train_mice.drop("Listening_Time_minutes", axis = 1),train_mice["Listening_Time_minutes"], shuffle = True, random_state = 42)


mice_imputer = IterativeImputer(max_iter=10, random_state=42)

pipe = Pipeline([
    ('Preprocessor', preprocessor), 
    ('MICE', mice_imputer),  
    ('Lasso', best_lasso_model)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE sur les données de test : {rmse}")


test = df_test.copy()
id_test = df_test["id"]


test.drop(["id","Podcast_Name","Episode_Title"], axis = 1, inplace = True)
test["Publication-Time"] = test["Publication_Day"] + "_" + test["Publication_Time"]
test["Number_ads_by_min"] = round(train["Number_of_Ads"] / test["Episode_Length_minutes"], 8)
test["Week-end"] = np.where(test["Publication_Day"].isin(["Saturday", "Sunday"]), "Oui", "Non")


y_sub = pipe.predict(test)


results = pd.DataFrame({
    'id': id_test,
    'Listening_Time_minutes': y_sub
})

results.to_csv("pipeline_mice", index=False)

