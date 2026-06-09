import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
xtest = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
ytest = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
print(train.shape)
print(xtest.shape)
print(ytest.shape)## Load Data


test = pd.concat([xtest, ytest], axis=1)
test.drop(columns="id" , axis=1 , inplace=True)

df = pd.concat([train, test], axis=0, ignore_index=True)


df.drop(columns="id" , axis=1 , inplace=True)


print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())


print("shape:", df.shape)  
print("\ndtypes:\n", df.dtypes)
print("\nnull:\n", df.isna().sum().sum())  #
print("\ndesc:\n", df.describe())


num_feature =  df.select_dtypes(include='number').columns.tolist()
cat_featuer = df.select_dtypes(exclude='number').columns.tolist()
num_feature.remove("accident_risk")


for col in cat_featuer:
    print(f"\n unique {col}:", df[col].unique())
    print(f"\n nunique {col}:", df[col].nunique())


df.hist(figsize=(12, 10))  
plt.show()

sns.boxplot(x='road_type', y='accident_risk', data=df)  # 
plt.show()

num_cols_tafget = num_feature + ['accident_risk']
corr = df[num_cols_tafget].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop("accident_risk")

# Loop through each numerical column
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")



cols_per_row = 4
total_plots = len(numeric_cols)
rows = int(np.ceil(total_plots / cols_per_row))

# ساخت شکل و محورها
fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 4))
axes = axes.flatten()  # تبدیل به لیست برای دسترسی راحت‌تر

# رسم نمودارها
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)

# حذف محورها اضافی (اگر تعداد نمودارها کمتر از تعداد محورها باشد)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']  
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']  
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']  


X = df.drop('accident_risk', axis=1)
y = df['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),  
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols + bool_cols)])

X_train_preprocessed = preprocessor.fit_transform(X_train)
print("type preprosseing:", X_train_preprocessed.shape)  


baseline_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])
baseline_model.fit(X_train, y_train)


baseline_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])
baseline_model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred = baseline_model.predict(X_test)
y_pred = np.clip(y_pred, 0, 1)  # محدود به [0,1] برای جلوگیری از مقادیر غیرمنطقی
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE base LR:", rmse)
print("R2 base LR:", r2 )


xgb1_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))])
xgb1_model.fit(X_train, y_train)


# Evaluation xgboost
y_pred_xgb = xgb1_model.predict(X_test)
y_pred_xgb = np.clip(y_pred_xgb, 0, 1)

rmse_xgb1 = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test , y_pred_xgb )
print("RMSE XGBoost ver:1:", rmse_xgb1)
print("R2   XGBoost ver:1:", r2_xgb)


# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

#Preprocessing pipelines
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_cols),('cat', categorical_transformer, categorical_cols)])


models = {
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        objective='reg:squarederror'
    )}

# Extract model from dictionary
xgb_model = models["XGBoost"]

pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', xgb_model)])


# Define RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipeline,
    X, y,
    cv=cv,
    scoring={'RMSE': rmse_scorer, 'R2': 'r2'},
    return_train_score=False
)

# Results
rmse_scores = np.sqrt(-cv_scores['test_RMSE'])  # Convert negative MSE to RMSE
r2_scores = cv_scores['test_R2']

print("XGBoost Model Evaluation")
print(f" Avg RMSE: {rmse_scores.mean():.4f}")
print(f" Std Dev: {rmse_scores.std():.4f}")
print(f" Avg R²: {r2_scores.mean():.4f}")

# Fit the pipeline on full training data
pipeline.fit(X, y)

# Predict on test data
final_preds = pipeline.predict(xtest)



submission = pd.DataFrame({'id': xtest['id'],'y': final_preds})

submission.to_csv('submission.csv', index=False)
print("\n✅ Submission file 'submission.csv' created using XGBoost pipeline!")

