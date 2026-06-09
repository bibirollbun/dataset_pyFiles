import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


## ğŸ‘€ Preview of Training Data


train_df.info()



print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)



features   = train_df.drop(columns=['accident_risk','id']).columns
target = train_df['accident_risk']
num_features = train_df[features].select_dtypes(include=['int64', 'float64']).columns
cat_features = train_df[features].select_dtypes(include=['object', 'category']).columns
bin_features = train_df[features].select_dtypes(include=['bool']).columns
cat_features = cat_features.append(bin_features)
print("Numerical features:", num_features)
print("Categorical features:", cat_features)
print("Binary features:", bin_features)
print("Number of numerical features:", len(num_features))
print("Number of categorical features:", len(cat_features))
print("Number of binary features:", len(bin_features))


train_df.duplicated().sum()


# plt the distrubution of the numerical data
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(num_features):
    ax = axs[i // 2, i % 2]
    sns.histplot(train_df[col], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()


# Plot distribution of categorical data
fig, axs = plt.subplots(len(cat_features), 2, figsize=(10, 5 * len(cat_features)))
for i, col in enumerate(cat_features):
    if col in train_df.columns:
        sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, palette='viridis', ax=axs[i, 0])
        axs[i, 0].set_title(f'{col} Counts')
        axs[i, 0].set_xlabel(col)
        axs[i, 0].set_ylabel('Count')
        axs[i, 0].tick_params(axis='x', rotation=45)

        # Plot the percentage on the right subplot
        axs[i, 1].pie(train_df[col].value_counts(), labels=train_df[col].value_counts().index, autopct='%1.1f%%', startangle=90)
        axs[i, 1].set_title(f'{col} Distribution')
    else:
        axs[i, 0].text(0.5, 0.5, f'Column {col} not found', ha='center', va='center')
        axs[i, 1].axis('off')

plt.tight_layout()
plt.show()


print("The information of the target variable is:")
print(train_df["accident_risk"].info())
print("The description of the target variable is:")
print(train_df["accident_risk"].describe().T)
print("The number of unique values in the target variable is:", train_df["accident_risk"].nunique())

# plot the distrubution of target variable
plt.figure(figsize=(8,5))
sns.histplot(train_df['accident_risk'], bins=30, kde=True, color='blue')
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()


train_df.head()


# Categorical features vs target

fig , axes = plt.subplots(2,4, figsize=(16,8))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9,0.66,0.33])
target = 'accident_risk'
for i,col in enumerate(cat_features) :

    grouped = train_df.groupby(col)[target].mean()

    axes[i].bar(grouped.index.astype(str), grouped.values , color=colors)  # .astype(str) to handle non-string indices
    
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



sns.pairplot(train_df.select_dtypes(include=['float64', 'int64']), diag_kind='kde')
plt.show()



# plt the correlation heatmap
plt.figure(figsize=(12, 8))
corr = train_df[num_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

X = train_df.drop(columns=['id', 'accident_risk'])
y = train_df['accident_risk']

num_feats = X.select_dtypes(include=['int64', 'float64']).columns
cat_feats = X.select_dtypes(include=['object', 'category','bool']).columns

print("Numerical features:", num_feats)
print("Categorical features:", cat_feats)




numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_feats),
        ('cat', categorical_transformer, cat_feats)
    ])
X_processed = preprocessor.fit_transform(X)
print("Processed feature shape:", X_processed.shape)

x_train, x_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("Training set shape:", x_train.shape, y_train.shape)


from xgboost import XGBRegressor
param = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'gamma': 0.0,
        'reg_alpha': 0.0,      # L1 regularization
        'reg_lambda': 1.0,    # L2 regularization
        'min_child_weight': 1,
        'max_delta_step': 0,
        'scale_pos_weight': 1.0,
        'tree_method': 'hist',
        'booster': 'gbtree',
        'sampling_method': 'uniform',
        'random_state': 42,
        'verbosity': 0

}
model = XGBRegressor(**param)
model.fit(x_train, y_train)


# mean_square,r2,amse,
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
y_pred = model.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Mean Absolute Error:", mae)


# plot the regrossr plot
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Accident Risk')
plt.ylabel('Predicted Accident Risk')
plt.title('Actual vs Predicted Accident Risk')
plt.show()


import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# âœ… Make sure your model is already fitted
# model.fit(x_train, y_train)

# Extract feature importances
booster = model.get_booster()
importance = booster.get_score(importance_type='weight')  # or 'gain', 'cover'


# Plot top features
plt.figure(figsize=(10, 6))
# Build human-readable feature names from the fitted preprocessor
num_cols = list(preprocessor.transformers_[0][2])  # numeric original names
cat_cols = list(preprocessor.transformers_[1][2])  # categorical original names

# get the fitted OneHotEncoder inside the pipeline
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))

feature_names = num_cols + cat_ohe_names  # final feature names in the same order as X_processed

# Map XGBoost importance keys ('f0', 'f1', ...) to these feature names
items = []
for k, v in importance.items():
    idx = int(k[1:])  # drop leading 'f' and convert to int
    name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
    items.append((idx, name, v))

importance_df = pd.DataFrame(items, columns=['Index', 'Feature', 'Importance']).sort_values('Importance', ascending=False).reset_index(drop=True)

# show the top 10 with original feature names
print(importance_df[['Feature', 'Importance']].head(10))

# Plot top 10 features (use original names)
plt.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
plt.xlabel('F Score')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()





cat_features = test_df.select_dtypes(include=['object', 'category','bool']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
test_cat_encoded = encoder.fit_transform(test_df[cat_features])
test_df[num_feats] = StandardScaler().fit_transform(test_df[num_feats])
test_df_encoded = np.hstack((test_df[num_feats].values, test_cat_encoded))
y_pred_test = model.predict(test_df_encoded)
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission_df['accident_risk'] = y_pred_test
submission_df.to_csv('submission.csv', index=False)


