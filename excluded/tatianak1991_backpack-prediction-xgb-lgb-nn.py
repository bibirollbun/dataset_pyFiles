import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

sns.set()


pd.set_option('display.float_format', '{:.2f}'.format)


import phik


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_ex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df_train.info()


df_train.describe()


plt.hist(df_train.Price, bins=20)
plt.show()


df_train_ex.info()


df_train_ex.describe()


plt.hist(df_train_ex.Price, bins=20)
plt.show()


df = pd.concat([df_train, df_train_ex], ignore_index=True)


df.drop('id', axis=1, inplace=True)


df


# Check for duplicates
df.duplicated().sum()


df.info()


df_test.info()


df.describe(include='all')


plt.figure(figsize=(12, 4))
plt.hist(df.Price, bins=50)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


sns.scatterplot(data = df, x = 'Weight Capacity (kg)', y='Price', hue='Size', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.show()


df.columns


fig, axes = plt.subplots(4, 2, figsize=(15, 30), sharey=True)  
cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
        'Waterproof', 'Style', 'Color']

for i, column in enumerate(cols):
    ax = axes[i // 2, i % 2]  
    sns.boxplot(data=df, x=column, y='Price', hue=column, palette="mako", ax=ax)
    ax.set_title(f'{column}', fontsize=14)
    ax.set_xlabel("")  
    ax.set_ylim(0, 170)
    
    legend = ax.get_legend()
    if legend is not None:
        legend.set_visible(False)  

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
corr = df.phik_matrix(interval_cols=['Weight Capacity (kg)', 'Price'])
sns.heatmap(corr.round(2), annot=True, cmap='coolwarm')
plt.show()


## Missing data 
missing_percentage = df.isna().sum()/len(df) * 100
missing_count = df.isna().sum()

missing_df = pd.DataFrame({'missing_data_count': missing_count, 
    'missing_data_%': missing_percentage.round(3),
})
missing_df.loc['total_for_all_columns'] = {'missing_data_count': missing_df['missing_data_count'].sum(),'missing_data_%': missing_df['missing_data_%'].sum()}

missing_df


# Calculate number of rows with MANY missing value
rows_with_multiple_missing = df.isna().sum(axis=1) > 1
num_rows_with_multiple_missing = rows_with_multiple_missing.sum()

# Calculate number of rows with 1 missing value
one_missing = df.isna().sum(axis=1) == 1
num_rows_one_missing = one_missing.sum()

# Calculate number of rows with NO missing values
no_missing = df.isna().sum(axis=1) == 0
num_rows_no_missing = no_missing.sum()

print("Percent of Rows with more than one missing values:", (num_rows_with_multiple_missing/len(df))*100)
print("Percent of Rows with ONE missing value:", (num_rows_one_missing/len(df))*100)
print("Percent of Rows with NO missing values:", (num_rows_no_missing/len(df))*100)


# Mean Price for each group
mean_price_no_missing = df.loc[no_missing, 'Price'].mean()
mean_price_one_missing = df.loc[one_missing, 'Price'].mean()
mean_price_multiple_missing = df.loc[rows_with_multiple_missing, 'Price'].mean()

# Group labels
labels = [f'No Missing Values \n(Mean Price: {mean_price_no_missing:.2f})', 
          f'One Missing Value in Row \n(Mean Price: {mean_price_one_missing:.2f}) ', 
          f'More than One Missing Value in Row \n(Mean Price: {mean_price_multiple_missing:.2f})']

# Percentages for Pie Chart
sizes = [(num_rows_no_missing/len(df))*100, 
         (num_rows_one_missing/len(df))*100, 
         (num_rows_with_multiple_missing/len(df))*100]

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=0, colors=['grey','lightgrey','#EEEDEB'], explode = (0, 0, 0.1))
plt.title('Percentage of Rows with Missing Values')

plt.show()


X = df.drop("Price", axis=1)
y = df["Price"]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


X_valid.shape


y_valid.shape


X_train.shape


y_train.shape


numerical_cols = ['Weight Capacity (kg)', 'Compartments']
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


# Preprocess data 

preprocessor = ColumnTransformer(
    transformers=[
        # Numerical data transformations: Nan -> median, scale
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),   
            ('scaler', StandardScaler())                   
        ]), numerical_cols),
        # Categorical data transformations: Nan -> most frequent, dummies
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))                 
        ]), categorical_cols)
    ])


preprocessor.fit(X_train)


X_train_transformed = preprocessor.transform(X_train)


new_df = pd.DataFrame(X_train_transformed, columns = numerical_cols + list(preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols)))
new_df


import random
from sklearn.cluster import KMeans

seed = 42
np.random.seed(seed)
random.seed(seed)


score = []

for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=seed)
    kmeans.fit(X_train_transformed)
    score.append(kmeans.inertia_)


plt.plot(range(1, 20), score, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method')
for i, txt in enumerate(range(1, 20)):
    plt.annotate(txt, (range(1, 20)[i], score[i]))
    plt.tight_layout()

plt.show()


kmeans = KMeans(n_clusters=4, n_init='auto', random_state=seed)
kmeans.fit(X_train_transformed)
labels = kmeans.labels_


class AddClusterLabels(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto')

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        # Add the cluster labels as an additional column to the data
        labels = self.kmeans.predict(X).reshape(-1, 1)
        return np.hstack([X, labels])


from sklearn.decomposition import PCA


# Obtain the principal components
pca_2 = PCA(2)
pca_2.fit(X_train_transformed)
principal_comp_2 = pca_2.fit_transform(X_train_transformed)


# Create a dataframe with the two components
pca_df = pd.DataFrame(data=principal_comp_2, columns= ['pca1', 'pca2'])
pca_clusters = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_clusters, palette="tab10")
plt.title('Principal Component Analysis')
plt.show()


pca_full = PCA().fit(X_train_transformed)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

# Plot Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Explained Variance by Number of Principal Components')
plt.grid(True)
plt.show()


# Fit PCA 
pca = PCA(n_components=10)  
pca_results = pca.fit_transform(X_train_transformed)  

# Convert PCA results into a DataFrame
df_pca = pd.DataFrame(pca_results, columns=[f"PCA{x}" for x in range(1, 11)])

df_pca


class AddPrincipalComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        # Apply PCA and append the components as new columns to the data (in array form)
        pca_components = self.pca.transform(X)
        return np.hstack([X, pca_components])


# Define RMSE scoring function

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def rmse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred))


import xgboost as xgb


model_xgboost = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                 n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.6, colsample_bytree=1.0, random_state=42)


pipeline_XSBoost = Pipeline(steps=[
                           ('preprocessor', preprocessor),
                           ('clustering', AddClusterLabels(n_clusters=4)),
                           ('pca', AddPrincipalComponents(n_components=10)),
                           ('xsboost', model_xgboost) 
])


# Fit the pipeline
pipeline_XSBoost.fit(X_train, y_train)

# Make predictions
y_pred = pipeline_XSBoost.predict(X_valid)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE: {rmse}")


# cross-validation with RMSE
cv_scores = cross_val_score(pipeline_XSBoost, X, y, cv=5, scoring=rmse_scorer)

# Print average RMSE from cross-validation
print(f"Average RMSE from cross-validation: {cv_scores.mean()}")


import lightgbm as lgb


lightgbm_model = lgb.LGBMRegressor(objective='regression',
                                   metric='rmse', 
                                   seed=seed, 
                                   n_estimators=500, 
                                   early_stopping_rounds=60,
                                   learning_rate=0.01)


pipeline_LGB = Pipeline(steps=[
                              ('preprocessor', preprocessor),
                              ('clustering', AddClusterLabels(n_clusters=4)),
                              ('pca', AddPrincipalComponents(n_components=10)),
                              ('lgb', lightgbm_model) 
])


# Fit the entire pipeline on the training data
pipeline_LGB[:-1].fit(X_train, y_train)

# transform the validation data using all steps except the last one
X_valid_transformed = pipeline_LGB[:-1].transform(X_valid)


# Fit the pipeline
pipeline_LGB.fit(X_train, y_train, lgb__eval_set=[(X_valid_transformed, y_valid)])

# Make predictions
y_pred = pipeline_LGB.predict(X_valid)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE: {rmse}")


best_iteration = pipeline_LGB.named_steps['lgb'].best_iteration_
print(f"Best iteration from cross-validation: {best_iteration}")


lgb.plot_metric(lightgbm_model)
plt.show()


# Predict test data
predictions = pipeline_LGB.predict(df_test)


# create submission file
# my_predictions = pd.DataFrame({'id': df_test['id'], 'Price': predictions})
# my_predictions.to_csv('/kaggle/working/submission.csv', index=False)


from catboost import CatBoostRegressor


catboost_model = CatBoostRegressor(iterations=6000, 
                                   learning_rate=0.01,
                                   depth=8,
                                   loss_function='RMSE',
                                   random_seed=seed,
                                   verbose=100,
                                   early_stopping_rounds=100)


pipeline_CB = Pipeline(steps=[
                              ('preprocessor', preprocessor),
                              ('clustering', AddClusterLabels(n_clusters=4)),
                              ('pca', AddPrincipalComponents(n_components=10)),
                              ('cb', catboost_model) 
])


# Fit the entire pipeline on the training data
pipeline_CB[:-1].fit(X_train, y_train)

# transform the validation data using all steps except the last one
X_valid_transformed = pipeline_CB[:-1].transform(X_valid)


# Fit the pipeline
pipeline_CB.fit(X_train, y_train, cb__eval_set=(X_valid_transformed, y_valid))

# Make predictions
y_pred = pipeline_CB.predict(X_valid)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE: {rmse}")


# create submission file
predictions_cat = pipeline_CB.predict(df_test)
cat_predictions = pd.DataFrame({'id': df_test['id'], 'Price': predictions_cat})
cat_predictions.to_csv('/kaggle/working/submission2.csv', index=False)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


pipeline_without_model = Pipeline(steps=[
                                        ('preprocessor', preprocessor),
                                        ('clustering', AddClusterLabels(n_clusters=4)),
                                        ('pca', AddPrincipalComponents(n_components=10)),
])


pipeline_without_model.fit(X_train, y_train)
X_train_nn = pipeline_without_model.transform(X_train)
X_valid_nn = pipeline_without_model.transform(X_valid)


model_nn = Sequential([
    Input(shape=(X_train_nn.shape[1],)),  # Explicit Input layer
    BatchNormalization(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),    
  
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model_nn.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Set early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=10, 
                           restore_best_weights=True,
                           min_delta=0.001) # minimium amount of change to count as an improvement

# Train the model
history = model_nn.fit(X_train_nn, y_train, 
                        validation_data=(X_valid_nn, y_valid), 
                        epochs=50, 
                        batch_size=500, 
                        callbacks=[early_stop],
                        verbose=1)

# Predict on test set
y_pred = model_nn.predict(X_valid_nn)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Test RMSE:", rmse)


history_df = pd.DataFrame(history.history)
history_df.loc[3:, ['loss', 'val_loss']].plot()


test = pipeline_without_model.transform(df_test)


# create submission file
predictions_nn = model_nn.predict(test)
predictions_nn


nn_predictions = pd.DataFrame({'id': df_test['id'], 'Price': np.ravel(predictions_nn)})
nn_predictions.to_csv('/kaggle/working/submission3.csv', index=False)




