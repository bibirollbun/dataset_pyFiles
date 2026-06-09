import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')



df.info()


df.describe()


df.head()


df['Cover_Type'].value_counts().sort_index()


wilderness_cols = [col for col in df.columns if 'Wilderness_Area' in col]
soil_cols = [col for col in df.columns if 'Soil_Type' in col]



#Histogram
df.hist(figsize=(16,12), bins=30)
plt.tight_layout()


#HeatMap
sns.heatmap(df.corr() , cmap='coolwarm')
plt.show()


#Boxplot of elevation VS Cover Type

sns.boxplot(df,x='Cover_Type', y='Elevation')


df.drop(columns='Id' , inplace=True)


# Combine binary wilderness columns into a single numeric categorical column
wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]

# Find the active column (which one has 1)
df['Wilderness_Area'] = df[wilderness_cols].idxmax(axis=1)
df['Wilderness_Area'] = df['Wilderness_Area'].str.extract('(\d)').astype(int)

# Drop original one-hot encoded columns
df.drop(columns=wilderness_cols, inplace=True)



soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]

df['Soil_Type'] = df[soil_cols].idxmax(axis=1)
df['Soil_Type'] = df['Soil_Type'].str.extract('(\d+)').astype(int)

df.drop(columns=soil_cols, inplace=True)



# Example: Manhattan and Euclidean distance to hydrology
df['Hydro_Manhattan'] = df['Horizontal_Distance_To_Hydrology'] + df['Vertical_Distance_To_Hydrology']
df['Hydro_Euclidean'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)



from sklearn.preprocessing import MinMaxScaler

continuous_cols = ['Elevation', 'Aspect', 'Slope',
                   'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                   'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                   'Hillshade_Noon', 'Hillshade_3pm',
                   'Horizontal_Distance_To_Fire_Points',
                   'Hydro_Manhattan', 'Hydro_Euclidean']

# Standardize using sklearn
scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])



df.head()


from sklearn.model_selection import train_test_split

X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Train-test split (stratify to preserve class balance)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


lr=LogisticRegression()
lr.fit(X_train , y_train)
y_pred_lr=lr.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))





from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))



# Subtract 1 to match XGBoost format
y_train_xgb = y_train - 1
y_val_xgb = y_val - 1


from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train, y_train_xgb)
y_pred_xgb = xgb.predict(X_val)

# Add 1 back to predictions for consistency with original labels
y_pred_xgb += 1

# Evaluation
print("XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
print(classification_report(y_val, y_pred_xgb))



from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}


from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train_xgb)

print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)



best_xgb = grid_search.best_estimator_

# Predict on validation set
y_pred_best = best_xgb.predict(X_val)

# Shift labels back
y_pred_best += 1

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
print("Validation Accuracy:", accuracy_score(y_val, y_pred_best))
print(classification_report(y_val, y_pred_best))



# Load test data
test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
test_ids = test_df['Id']
test_df.drop(columns=['Id'], inplace=True)



# Combine Wilderness_Area
wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
test_df['Wilderness_Area'] = test_df[wilderness_cols].idxmax(axis=1).str.extract('(\d)').astype(int)
test_df.drop(columns=wilderness_cols, inplace=True)

# Combine Soil_Type
soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
test_df['Soil_Type'] = test_df[soil_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
test_df.drop(columns=soil_cols, inplace=True)



# Distance features
test_df['Hydro_Manhattan'] = test_df['Horizontal_Distance_To_Hydrology'] + test_df['Vertical_Distance_To_Hydrology']
test_df['Hydro_Euclidean'] = np.sqrt(test_df['Horizontal_Distance_To_Hydrology']**2 + test_df['Vertical_Distance_To_Hydrology']**2)



# Apply the same Minmax scaler
test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])



# Predict on test data
y_test_pred = best_xgb.predict(test_df)

# Add 1 back (we subtracted 1 earlier for training)
y_test_pred += 1



# Build submission DataFrame
submission = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': y_test_pred
})

# Save as CSV
submission.to_csv('submission.csv', index=False)





