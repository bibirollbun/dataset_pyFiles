import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_data = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print(f'Train shape: {train_data.shape}')
print(f'Test shape: {test_data.shape}')


train_data = train_data.drop('id', axis=1)
train_data.head()


train_data.info()


train_data.describe()


missing_data = train_data.isnull().sum()
print(missing_data[missing_data > 0])


# Unique values for categorical variables
categorical_cols = train_data.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

for col in categorical_cols:
    print(f"{col}: {train_data[col].nunique()} unique values")
    print(train_data[col].value_counts().head())
    print()


# Numeric columns
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {list(numeric_cols)}")


if len(numeric_cols)>1:
    corr_matrix = train_data[numeric_cols].corr()
    print("Correlation matrix:")
    print(corr_matrix)

    #Plot correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


# Visualize distribution

n_cols = len(numeric_cols)
n_rows = (n_cols+2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(10, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]

for i, col in enumerate(numeric_cols):
    axes[i].hist(train_data[col], bins=30, alpha=0.7, color='skyblue', edgecolor= 'black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide empty subplots
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()


def boxplot_categoric_cols_with_target(df, target):
    categoric_cols = train_data.select_dtypes(include=['object']).columns
    n_cols = len(categoric_cols)

    if n_cols == 0:
        print("No numeric columns to visualize")
        return

    n_rows = (n_cols+2)//3 # 3 columns per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    # Create box plots
    for i, col in enumerate(categoric_cols):
        sns.boxplot(data=df, x=col, y=target, ax=axes[i])
        axes[i].set_title(f'{target} by {col}')

    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


boxplot_categoric_cols_with_target(train_data, 'accident_risk')


# Encoding Features

# Encoding boolean columns
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
train_data[bool_cols] = train_data[bool_cols].astype(int)
test_data[bool_cols] = test_data[bool_cols].astype(int)

# Encoding categorical variables, since there is no ordering, we will simply use OneHotEncoding
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
train_data = pd.get_dummies(train_data, columns=cat_cols, drop_first=True)
test_data = pd.get_dummies(test_data, columns=cat_cols, drop_first=True)


X_train = train_data.drop('accident_risk', axis=1)
y_train = train_data['accident_risk']
X_test = test_data


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# Split train/validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size = 0.2,
    random_state=42
)


xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)


# Train model
xgb_model.fit(X_tr, y_tr)


pred_val_xgb = xgb_model.predict(X_val)


rmse = mean_squared_error(y_val, pred_val_xgb, squared=False)
print(f"\n Validation RMSE: {rmse:.5f}")


xgb_model.fit(X_train, y_train)


pred_final = xgb_model.predict(X_test)


pred_final = np.clip(pred_final, 0, 1)


sample_data["accident_risk"] = pred_final
sample_data.to_csv("submission.csv", index=False)
print("\n submission.csv created successfully and ready for Kaggle upload!")

