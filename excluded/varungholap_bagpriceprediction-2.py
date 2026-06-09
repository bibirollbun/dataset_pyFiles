import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head()


df.drop(columns=["id"], inplace=True)
df.head()


df.isnull().sum()


df.count()


df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].median(), inplace=True)


df.fillna(df.mode().iloc[0], inplace=True)


df.isnull().sum()


print(df.duplicated().sum())
df.drop_duplicates(inplace=True)


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder


from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df['Brand_Material'] = df['Brand'].astype(str) + "_" + df['Material'].astype(str)



le= LabelEncoder()
df['Size']=le.fit_transform(df['Size'])


poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Size', 'Compartments']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Size', 'Compartments']))

# Rename polynomial columns to avoid conflicts
poly_df.columns = [f"poly_{col}" for col in poly_df.columns]

# Concatenate polynomial features with the dataset
df = pd.concat([df, poly_df], axis=1)


df['Weight_Bins'] = pd.cut(df['Weight Capacity (kg)'], bins=3, labels=['Low', 'Medium', 'High'])


df.head()


df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})


categorical_features = ['Brand', 'Material', 'Color', 'Style', 'Brand_Material']
ordinal_features = ['Weight_Bins']
binary_features = ['Laptop Compartment', 'Waterproof']
numerical_features = ['Size', 'Compartments', 'Weight Capacity (kg)']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # No encoding for numerical features
        ('cat', OneHotEncoder(drop='first'), categorical_features),  # One-hot encode nominal data
        ('ord', OrdinalEncoder(), ordinal_features),  # Label encode ordinal data
        ('bin', 'passthrough', binary_features)  # No encoding for binary features
    ])


from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0))
])


X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__subsample': [0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.8, 0.9, 1.0]
}


grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)



grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


from sklearn.metrics import mean_squared_error, r2_score
y_pred = best_model.predict(X_test)


print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


df_test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

df_test.head()


df_test["Weight Capacity (kg)"].fillna(df_test["Weight Capacity (kg)"].median(), inplace=True)
df_test.fillna(df_test.mode().iloc[0], inplace=True)


df_test.drop_duplicates(inplace=True)


df_test['Brand_Material'] = df_test['Brand'].astype(str) + "_" + df_test['Material'].astype(str)


le= LabelEncoder()
df_test['Size']=le.fit_transform(df_test['Size'])


poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_test[['Size', 'Compartments']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Size', 'Compartments']))

# Rename polynomial columns to avoid conflicts
poly_df.columns = [f"poly_{col}" for col in poly_df.columns]

# Concatenate polynomial features with the dataset
df = pd.concat([df_test, poly_df], axis=1)


df_test['Weight_Bins'] = pd.cut(df_test['Weight Capacity (kg)'], bins=3, labels=['Low', 'Medium', 'High'])


df_test['Laptop Compartment'] = df_test['Laptop Compartment'].map({'Yes': 1, 'No': 0})
df_test['Waterproof'] = df_test['Waterproof'].map({'Yes': 1, 'No': 0})


df_test.head()
df_test_id=df_test['id']
df_test = df_test.drop(columns=['id'])


y_test_pred=best_model.predict(df_test)


y_test_pred


output_df = pd.DataFrame({'id': df_test_id, 'Price': y_test_pred})

# Save to CSV
output_df.to_csv('submission.csv', index=False)

print(output_df)




