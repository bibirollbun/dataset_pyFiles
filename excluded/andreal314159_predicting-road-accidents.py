# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import root_mean_squared_error

from sklearn.preprocessing import OneHotEncoder 


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score




accidents = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv").drop(columns = ["id"])
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
display(accidents.head())

# numeric features
display(accidents.describe(include= np.number))

# non numeric features
 
accidents["time_of_day"] = pd.Categorical(accidents.time_of_day, categories = ["morning", "afternoon", "evening"], ordered = True)
accidents["weather"] = pd.Categorical(accidents.weather, categories = ["clear", "rainy", "foggy"], ordered = True)
accidents["lighting"] = pd.Categorical(accidents.lighting, categories = ["daylight", "dim", "night"], ordered = True)
accidents["road_type"] = pd.Categorical(accidents.road_type, categories= ["rural","urban", "highway"], ordered = True)


test["time_of_day"] = pd.Categorical(test.time_of_day, categories = ["morning", "afternoon", "evening"], ordered = True)
test["weather"] = pd.Categorical(test.weather, categories = ["clear", "rainy", "foggy"], ordered = True)
test["lighting"] = pd.Categorical(test.lighting, categories = ["daylight", "dim", "night"], ordered = True)
test["road_type"] = pd.Categorical(test.road_type, categories= ["rural","urban", "highway"], ordered = True)




display(accidents.describe(exclude = np.number))
display(accidents.isna().sum())
display(accidents.info())


num_cols = accidents.drop(columns="accident_risk").select_dtypes(["int64", "float64"]).columns
train = accidents[num_cols]

#test = test[num_cols]
X = train
y = accidents[["accident_risk"]]
model = LinearRegression()
model = Lasso()
model = DecisionTreeRegressor()
model = Ridge()
def get_score(model, X, y):
    """ 
    Calculate the estimated score for a model with negative root mean squared error. 
    To have a balanced estimate it uses cross validation and then calculates the average adjusted with standard deviation. 
    """
    scores =  cross_val_score(
        estimator=model,
        X=X,
        y = y,
        scoring = 'neg_root_mean_squared_error' # the algorithm wants to maximize the score, therefore we need the negative in order to motivate the algorithm for the smallest error in absolut value 
        )
    estimated_score = abs(np.mean(scores)) + np.std(scores)
    print(f"Scores: {-scores} with standard deviation of {np.std(scores)}, estimate: {estimated_score}")
    return estimated_score
get_score(model, X,y)



# Making bins for the output: 
accidents["risk_category"] = pd.cut(accidents.accident_risk, 3, labels = ["low", "medium", "high"])
accidents.head()


g = sns.histplot(accidents.accident_risk).set(title="Distribution of accident risk")
sns.despine()
plt.show()





# Taking a look at the numerical features

import seaborn as sns
import matplotlib.pyplot as plt

# First for integers let's look at the counts
for col in accidents.drop(columns=["accident_risk", "risk_category"]).select_dtypes([int, "object", bool]).columns: 

    plt.figure(figsize = (6,4))
    g = sns.countplot(
        data = accidents,
        x = col,
        hue = "risk_category",
        palette = "icefire"
        
    ).set(title=f"Distribution of {col} for risk categories")
    sns.despine()
    plt.show()


for col in accidents.drop(columns = ["risk_category", "accident_risk"]).select_dtypes(float).columns: 
    plt.figure(figsize = (6,4))
    sns.histplot(
        data = accidents,
        x = col,
        hue="risk_category", 
        palette = "icefire",
    )
    sns.despine()
    plt.show()



#accidents["weather"] = accidents.time_of_day.replace({"":0, "afternoon": 1, "evening": 2})

accidents["time_of_day_num"] = accidents.time_of_day.cat.codes
accidents["weather_num"] = accidents.weather.cat.codes
accidents["lighting_num"] = accidents.lighting.cat.codes
accidents["road_type"] = accidents.road_type.cat.codes
accidents.head()





# visualizing correlation on a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    data = accidents.select_dtypes(exclude="category").corr(), 
    cmap = "Purples",
    cbar = True,
    fmt = ".2f",
    annot = True).set(title="Correlation of features ")
sns.despine()
plt.show()


features_to_keep = ["curvature", "speed_limit", "lighting", "weather", "num_reported_accidents"]

accidents_sel = accidents[features_to_keep]
test_sel = test[features_to_keep]
accidents_sel.head()



#onehot_encoder = OneHotEncoder()
#accidents_sel_enc = onehot_encoder.fit_transform(accidents_sel)
accidents_sel_enc = pd.get_dummies(accidents_sel, drop_first = True, dtype =int)
test_sel_enc = pd.get_dummies(test_sel, drop_first = True, dtype =int)

accidents_sel_enc


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
accidents_sel_enc_scaled = scaler.fit_transform(accidents_sel_enc)
test_sel_enc_scaled = scaler.transform(test_sel_enc)


display(accidents.head())


%%time 

from sklearn.ensemble import ExtraTreesRegressor
model = LinearRegression()
#model = Lasso() # .1631439679825607
#model = DecisionTreeRegressor()
#model = XGBRegressor()
#model = GradientBoostingRegressor()
model = RandomForestRegressor(n_estimators = 50, max_depth = 10)
model = ExtraTreesRegressor()
model = Ridge()
X = accidents_sel_enc_scaled 

y = accidents["accident_risk"]
score = get_score(model, X,y)
display(f"{model}: {score}")



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
if "risk_category" in accidents.columns: 
    accidents = accidents.drop(columns =["risk_category"])


numeric_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
numeric_cols = ["curvature", "speed_limit", "num_reported_accidents"]

cat_cols = ["road_type", "lighting","weather", "road_signs_present", "public_road", "time_of_day", "holiday", "school_season"]
cat_cols = ["lighting","weather"]


preprocessor = ColumnTransformer( # this will preprocess the numerical and categorical features separately.
    transformers = [
        ("numerics", MinMaxScaler(), numeric_cols), # numerics are scaled only
        ("categoricals_and_bools", 
         Pipeline(steps = [     # categoricals are both encoded and scaled, so they need a minipipeline
            ('onehot encoding', OneHotEncoder(drop="first", sparse_output = False)),
            ('scaling', MinMaxScaler())
             ]), 
        cat_cols
        ),
        
    ], 
    remainder = "drop",
    verbose = False,
    verbose_feature_names_out = False
)


pipeline = Pipeline(steps = [("preprocessor", preprocessor),
                            ("model", model)])
#pipeline.fit(accidents.drop(columns = ["accident_risk"]), accidents.accident_risk)
#predictions = pipeline.predict(accidents.drop(columns="accident_risk"))
#predictions[:5]


preprocessor.fit_transform(accidents[numeric_cols + cat_cols])
preprocessor.get_feature_names_out()


get_score(pipeline, accidents[numeric_cols + cat_cols], accidents.accident_risk)


model# 


model =RandomForestRegressor()
model = ExtraTreesRegressor()
model = Ridge()
model = XGBRegressor(random_state = 0, max_depth = 10)
pipeline = Pipeline(steps = [("preprocessor", preprocessor),
                            ("model", model)])



%%time
from sklearn.model_selection import GridSearchCV
X = accidents.drop(columns = "accident_risk")
y = accidents.accident_risk
param_grid = {"model__n_estimators": [50,100,150]} # RandomForest
param_grid = {"model__min_samples_leaf": [5, 10], # Decision Tree
             "model__max_depth": [6,8,10]}

param_grid = {"model__n_estimators": [50, 100, 150], # GradientBoosting
             "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [5, 8]}

param_grid = {"model__n_estimators": [50, 100, 200], # Random Forest, ExtraTrees
             "model__min_samples_leaf": [4, 6,8]}

param_grid =  {"model__min_child_weight": [5, 7], # XGBRegressor
               "model__learning_rate": [0.05, 0.1],
               "model__subsample": [0.5, 0.7, 1]}

#param_grid =  {"model__alpha": [0.5, 1, 2, 2.5, 3, 5]} # Ridge
if True: 
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring = 'neg_root_mean_squared_error', verbose = 3)
    grid_search.fit(X,y) 
    display(grid_search.best_estimator_)
    display(grid_search.best_params_)
    display(grid_search.best_score_)


#Let's run the best estimator on and 
final_model = grid_search.best_estimator_
#final_model = ExtraTreesRegressor("min_samples_leaf"=8, n_estimators = 200)
final_pipeline = Pipeline(steps = [("preprocessor", preprocessor),
                            ("model", final_model)])


X.head()


pipeline.fit(X, y)
predictions = pipeline.predict(test) 
predictions = pd.DataFrame({"id": test.id, "accident_risk":predictions.ravel()})
predictions.to_csv("submission.csv", index=False)


sub = pd.read_csv("/kaggle/working/submission.csv")
sub




