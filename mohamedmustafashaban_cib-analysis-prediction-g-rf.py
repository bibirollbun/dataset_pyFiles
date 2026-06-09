import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import shap
from lime.lime_tabular import LimeTabularExplainer
from warnings import filterwarnings
filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train_data.isnull().sum()


imputer = SimpleImputer(strategy='median')
train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data.select_dtypes(include=['float64', 'int64'])))
train_data_imputed.columns = train_data.select_dtypes(include=['float64', 'int64']).columns


# Check for missing values after handling
print("\nMissing Values After Handling:")
print(train_data.isnull().sum())


train_data_processed = pd.concat([train_data_imputed, train_data.select_dtypes(include=['object'])], axis=1)


train_data_processed = pd.get_dummies(train_data_processed)


# 1.1 Histogram for 'efs_time'
plt.figure(figsize=(8, 6))
sns.histplot(train_data['efs_time'], bins=30, kde=True)
plt.title("Distribution of efs_time")
plt.show()



# 1.2 KDE Plot for 'age_at_hct'
plt.figure(figsize=(8, 6))
sns.kdeplot(train_data['age_at_hct'], shade=True)
plt.title("KDE Plot of Age at HCT")
plt.show()



# 2.1 Scatter Plot between 'age_at_hct' and 'efs_time'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age_at_hct', y='efs_time', data=train_data)
plt.title("Scatter Plot: Age at HCT vs efs_time")
plt.show()



# 4.1 Box Plot of 'efs_time' by 'race_group'
plt.figure(figsize=(10, 6))
sns.boxplot(x='race_group', y='efs_time', data=train_data)
plt.title("Box Plot: efs_time by Race Group")
plt.xticks(rotation=45)
plt.show()


# 4.2 Violin Plot of 'age_at_hct' by 'graft_type'
plt.figure(figsize=(10, 6))
sns.violinplot(x='graft_type', y='age_at_hct', data=train_data)
plt.title("Violin Plot: Age at HCT by Graft Type")
plt.show()


# 5. Cumulative Distribution Function (CDF) for 'efs_time'
plt.figure(figsize=(8, 6))
sns.ecdfplot(train_data['efs_time'])
plt.title("CDF of efs_time")
plt.show()


# 6.1 Line Plot of 'efs_time' over 'year_hct'
plt.figure(figsize=(10, 6))
sns.lineplot(x='year_hct', y='efs_time', data=train_data)
plt.title("Line Plot: efs_time over Year of HCT")
plt.show()


# 6.2 Bar Plot of Mean 'efs_time' by 'race_group'
plt.figure(figsize=(10, 6))
sns.barplot(x='race_group', y='efs_time', data=train_data, estimator=np.mean)
plt.title("Bar Plot: Mean efs_time by Race Group")
plt.xticks(rotation=45)
plt.show()


# 6.6 Interactive Violin Plot of 'age_at_hct' by 'graft_type'
fig = px.violin(train_data, x='graft_type', y='age_at_hct', title="Interactive Violin Plot: Age at HCT by Graft Type")
fig.show()



# 6.7 Interactive Histogram of 'efs_time' Distribution
fig = px.histogram(train_data, x='efs_time', nbins=30, title="Interactive Histogram: Distribution of efs_time")
fig.show()


# 6.8 Interactive KDE Plot of 'age_at_hct' Distribution
fig = px.density_contour(train_data, x='age_at_hct', title="Interactive KDE Plot: Age at HCT")
fig.show()


# 6.10 Interactive 3D Scatter Plot between 'age_at_hct', 'efs_time', and 'race_group'
fig = px.scatter_3d(train_data, x='age_at_hct', y='efs_time', z='comorbidity_score', color='race_group', title="Interactive 3D Scatter Plot")
fig.show()


# 10. Interactive Line Plot of 'efs_time' over 'year_hct'
fig = px.line(train_data, x='year_hct', y='efs_time', color='race_group', title="Interactive Line Plot: efs_time over Year of HCT")
fig.show()



# 11. Interactive Area Plot of 'efs_time' by 'race_group'
fig = px.area(train_data, x='year_hct', y='efs_time', color='race_group', title="Interactive Area Plot: efs_time by Race Group")
fig.show()


# 12. Interactive Sunburst Chart of 'race_group' and 'graft_type'
fig = px.sunburst(train_data, path=['race_group', 'graft_type'], title="Interactive Sunburst Chart: Race Group and Graft Type")
fig.show()


# 14. Interactive Treemap of 'race_group' and 'graft_type'
fig = px.treemap(train_data, path=['race_group', 'graft_type'], title="Interactive Treemap: Race Group and Graft Type")
fig.show()


fig = px.scatter_3d(train_data, x='age_at_hct', y='efs_time', z='comorbidity_score', color='race_group', title="Interactive 3D Scatter Plot")
fig.show()


# Exploratory Data Analysis (EDA)
# Distribution of the target variable 'efs'
sns.countplot(x='efs', data=train_data)
plt.title("Distribution of Event-Free Survival (efs)")
plt.show()



# Distribution of 'efs_time'
sns.histplot(train_data['efs_time'], bins=30, kde=True)
plt.title("Distribution of Time to Event-Free Survival (efs_time)")
plt.show()


# Preprocessing
# Separate features and target
X = train_data_processed.drop('efs', axis=1) 
y = train_data_processed['efs']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def clean_feature_names(df):
    df.columns = [col.replace('[', '_').replace(']', '_').replace('<', '').replace('>', '').replace('/', '_') for col in df.columns]
    return df

X_train = clean_feature_names(X_train)
X_val = clean_feature_names(X_val)


model_xgb = xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse')
model_xgb.fit(X_train, y_train)


y_pred_xgb = model_xgb.predict(X_val)
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_xgb)))
print("XGBoost R^2:", r2_score(y_val, y_pred_xgb))



from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)


y_pred_rf = model_rf.predict(X_val)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_rf)))
print("Random Forest R^2:", r2_score(y_val, y_pred_rf))


explainer = shap.Explainer(model_xgb)
shap_values = explainer(X_val)


shap.summary_plot(shap_values, X_val)


explainer_lime = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, mode='regression')
i = 0  
exp = explainer_lime.explain_instance(X_val.values[i], model_xgb.predict)
exp.show_in_notebook()



test_data_processed = pd.get_dummies(test_data, drop_first=True)


missing_cols = set(X_train.columns) - set(test_data_processed.columns)
for col in missing_cols:
    test_data_processed[col] = 0


test_data_processed = test_data_processed[X_train.columns]


predictions = model_xgb.predict(test_data_processed)


submission= pd.DataFrame({
    'ID': test_data['ID'],  
    'Prediction': predictions
})



submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')

print(submission.dtypes)


submission= pd.DataFrame({
    'ID':submission['ID'],  
    'Prediction': predictions
})


# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")


submission.to_csv('submission_corrected.csv', index=False)


submission




