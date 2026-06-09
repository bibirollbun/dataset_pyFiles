import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,VotingRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler


from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    import numpy as np

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
        
import warnings
warnings.filterwarnings('ignore')



!pip install catboost --quiet



# df=pd.read_csv(r'/content/train.csv')
# train_df=df=pd.read_csv(r'/content/train.csv') 

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv") #to vis
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")




pd.concat([df.head(5), df.sample(5), df.tail(5)])


df.info()


df.isna().sum()


df.duplicated().sum()


df.describe()


df.describe(include='object')



pd.DataFrame([ df.nunique(), df.dtypes ], index=['Unique Values', 'Data Types'])


df['road_type'].value_counts()
#balanced


plt.figure(figsize=(10,6))
sns.histplot(df['accident_risk'], bins=30, kde=False, color='teal')
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Count')
plt.show()




INPUT_FEATURES = df.drop('accident_risk',axis=1).select_dtypes(include='number').columns.tolist()

N_COLS = 3
N_ROWS = math.ceil(len(INPUT_FEATURES) / N_COLS)

plt.figure(figsize=(15, 5 * N_ROWS))

for i, feature in enumerate(INPUT_FEATURES):
    plt.subplot(N_ROWS, N_COLS, i + 1)
    plt.title(f"Box plot: {feature}")
    sns.boxplot(data=df, y=feature)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()



def IQR(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    min_v = Q1 - 1.5 * IQR
    max_v = Q3 + 1.5 * IQR

    return series.clip(lower=min_v, upper=max_v)



df.columns


def simple_risk_score(X):
    """
    Calculate weighted risk score based on multiple factors
    Weights are calibrated based on feature importance
    """
    # Normalize curvature to 0-1 range
    curvature_norm = X["curvature"]

    # Speed risk (exponential increase after 60)
    speed_risk = np.where(X["speed_limit"] >= 60,
                          (X["speed_limit"] - 60) / 40,  # normalize to 0-1
                          0)

    # Lighting risk
    lighting_risk = (X["lighting"] == "night").astype(int) * 0.8

    # Weather risk
    weather_risk = np.where(X["weather"] == "rainy", 0.8,
                           np.where(X["weather"] == "foggy", 0.9, 0))

    # Historical accidents (higher weight)
    accident_risk = np.clip(X["num_reported_accidents"] / 7, 0, 1)  # normalize

    # Weighted combination
    mu = (
        0.25 * curvature_norm +      # Road geometry
        0.20 * speed_risk +           # Speed factor
        0.15 * lighting_risk +        # Visibility
        0.20 * weather_risk +         # Weather conditions
        0.20 * accident_risk          # Historical data (most important)
    )

    # Sigmoid transformation for smooth 0-1 output
    y = 1 / (1 + np.exp(- (mu - 0.5) * 10))

    return y

df["risk_score"] = simple_risk_score(df)

df[["risk_score"]].head(10)


df["curve_speed_interaction"] = df["curvature"] * df["speed_limit"]



df['bad_weather_night'] = ((df['weather'] != 'clear').astype(int)) * ((df['lighting'] == 'night').astype(int))





df['holiday_traffic'] = df['holiday'].astype(int) * df['num_lanes']


# df['curvature_sq'] = df['curvature'] ** 2


df['road_width_proxy'] = df['num_lanes'] / (1 + df['curvature'])


df['time_of_day'].value_counts()


df.columns


df['road_type'].value_counts()


plt.figure(figsize=(15, 15))
INPUT_FEATURES = df.drop('accident_risk',axis=1).select_dtypes(include='number').columns.tolist()

plt.title("Input Features Correlation")

sns.heatmap(
    df[INPUT_FEATURES].corr(),
    annot=True,
    cmap='coolwarm',
    )

plt.show()


#draw corr with target
plt.figure(figsize=(8, 10))

corr_with_target = df.corr(numeric_only=True)['accident_risk'].sort_values(ascending=False)

sns.heatmap(corr_with_target.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title("Correlation with Target: accident_risk")
plt.show()




from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(df[INPUT_FEATURES], df['accident_risk'])

for i in range(len(INPUT_FEATURES)):
    print(f"{INPUT_FEATURES[i]:15s}: F-value = {f_values[i]:6.3f}, p-value = {p_values[i]:3.3f}")


# df = df.drop([id]) is droped down here


df.describe()


# aplly same preprocessing on train and test
def preprocessing(df):

    lighting_map = {"dim": 0, "daylight": 1, "night": 2}
    df["lighting"] = df["lighting"].map(lighting_map)
    df["weather"] = df["weather"].map({"foggy": 0, "clear": 1, "rainy": 2})
    df["holiday"] = df["holiday"].astype(int)
    df["school_season"] = df["school_season"].astype(int)
    df["public_road"] = df["public_road"].astype(int)
    df["road_signs_present"] = df["road_signs_present"].astype(int)
    df["time_of_day"] = df["time_of_day"].map({"morning": 0, "afternoon": 1, "evening": 2, "night": 3})
    df['road_type'] = df['road_type'].map({'highway': 0, 'rural': 1, 'urban': 2})

    # Feature engineering
    df["risk_score"] = simple_risk_score(df)
    df["curve_speed_interaction"] = df["curvature"] * df["speed_limit"]
    df['bad_weather_night'] = ((df['weather'] != 1).astype(int)) * ((df['lighting'] == 2).astype(int))
    df['holiday_traffic'] = df['holiday'].astype(int) * df['num_lanes']
    df['road_width_proxy'] = df['num_lanes'] / (1 + df['curvature'])


    # not useful
    # min_max_scaler = MinMaxScaler()
    # cols_to_scale=['speed_limit','curve_speed_interaction']
    # df[cols_to_scale] = min_max_scaler.fit_transform(df[cols_to_scale])

    return df


# #to work on a sample
# dfsample, _ = train_test_split(
#     train,
#     test_size=0.6,   # Work on 40% and leave 60%
#     random_state=42
# )
# # I'll comment this after finishing


# go for the whole data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv") # to train

X_train,X_val, y_train, y_val = train_test_split(
    train_df.drop(columns=['accident_risk','id']),
    df['accident_risk'],

    test_size=0.25,
    random_state=124,
  )
#


X_train=preprocessing(X_train)
X_val = preprocessing(X_val)



X_train.shape, X_val.shape, y_train.shape, y_val.shape


RANDOM_STATE = 123

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lgbm_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    subsample=0.8,
    random_seed=RANDOM_STATE,
    verbose=False,
    loss_function='RMSE'
)

gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=RANDOM_STATE
)

ensemble = VotingRegressor([
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('cat', cat_model),
    ('gb', gb_model)
])

stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', cat_model),
        ('gb', gb_model)
    ],
    final_estimator=XGBRegressor(
        learning_rate=0.03,
        max_depth=3,
        n_estimators=300,
        random_state=RANDOM_STATE
    ),
    n_jobs=-1
)

models_list = {
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    # "CatBoost": cat_model,
    # "Gradient Boosting": gb_model,
    # "Ensemble (Voting)": ensemble,
    # "Stacking Regressor": stack_model
}

print(" Models initialized successfully â€” ready for training loop.")



best_model_name = None
best_rmse = float('inf')
best_model = None  # storing the best model

for name, model in models_list.items():
    print("=" * 70)
    print(f"Model: {name}")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)

    print(f"RMSE for train: {rmse_train:.4f}")
    print(f"RMSE for val:  {rmse_val:.4f}")

    if rmse_val < best_rmse:
        best_rmse = rmse_val
        best_model_name = name
        best_model = model

print("\n" + "=" * 70)
print(f"âœ… Best Model so far: {best_model_name} with RMSE = {best_rmse:.4f}")



test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
# test=pd.read_csv(r'/content/test.csv')

test['accident_risk'] = 0.5
print("Test shape:", test.shape )
test.head()


test_pre=preprocessing(test)
test_pre.columns.value_counts().sum()



df.columns.value_counts().sum()


#predict with best model
test_pre['accident_risk']=best_model.predict(test_pre.drop(columns=['id','accident_risk'],axis=1))




test_pre.isna().sum()



submission =test_pre[['id', 'accident_risk']]
submission.to_csv('submission_final.csv', index=False)
print("Submission file created!")


# # make preds for all models uncomment to see  
# xgb_pred = models_list["XGBoost"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# lgbm_pred = models_list["LightGBM"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# gb_pred = models_list["Gradient Boosting"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# ens_pred = models_list["Ensemble (Voting)"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# stack_pred = models_list["Stacking Regressor"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))

# y_avr = (
#     0.3 * lgbm_pred +
#     0.25 * xgb_pred +
#     0.2 * gb_pred +
#     0.15 * ens_pred +
#     0.1 * stack_pred
# )

# test_pre['accident_risk'] = np.clip(y_avr, 0, 1)

# submission = test_pre[['id', 'accident_risk']]
# submission.to_csv('submission_blend.csv', index=False)

# print("Blended submission file created successfully!")




# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(stack_pred, 0, 1)
# }).to_csv('submission_stack.csv', index=False)




# # âœ… save every model and also save thier avg
# xgb_pred = models_list["XGBoost"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# lgbm_pred = models_list["LightGBM"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# # gb_pred = models_list["Gradient Boosting"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# # ens_pred = models_list["Ensemble (Voting)"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# # stack_pred = models_list["Stacking Regressor"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))
# cat_pred  = models_list["CatBoost"].predict(test_pre.drop(columns=['id', 'accident_risk'], axis=1))

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(xgb_pred, 0, 1)
# }).to_csv('submission_xgb.csv', index=False)

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(lgbm_pred, 0, 1)
# }).to_csv('submission_lgbm.csv', index=False)

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(gb_pred, 0, 1)
# }).to_csv('submission_gb.csv', index=False)

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(ens_pred, 0, 1)
# }).to_csv('submission_ensemble.csv', index=False)

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(stack_pred, 0, 1)
# }).to_csv('submission_stacking.csv', index=False)

# pd.DataFrame({
#     'id': test_pre['id'],
#     'accident_risk': np.clip(cat_pred, 0, 1)
# }).to_csv('submission_catboost.csv', index=False)

# y_avr = (
#     0.25 * lgbm_pred +
#     0.25 * xgb_pred +
#     0.15 * gb_pred +
#     0.15 * ens_pred +
#     0.10 * stack_pred +
#     0.10 * cat_pred
# )

# test_pre['accident_risk'] = np.clip(y_avr, 0, 1)

# submission = test_pre[['id', 'accident_risk']]
# submission.to_csv('submission_blend.csv', index=False)

# print("âœ… Blended submission file created successfully: submission_blend.csv")


