import numpy as np
from matplotlib import  pyplot as plt
import pandas as pd


train_path = '/kaggle/input/playground-series-s5e5/train.csv'
train = pd.read_csv(train_path, index_col=0)


print("Shape:", train.shape)
train.head()


train.describe(include='all')


train.info()


train.isnull().sum()


train.duplicated().sum()


train[train.duplicated()]


print(f"Percentage of duplicates is {(train.duplicated().sum()/ len(train))* 100}%")


# Remove duplicates

train = train.drop_duplicates()
print(f"Shape of clear train set is {train.shape}")


train.to_csv('/kaggle/working/clear_train.csv')


# Numerical
train.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


## CategoricaL

for col in train.select_dtypes(include = 'object'). columns:
    ax = train[col].value_counts().plot(kind = 'bar', title= col)
    ax.bar_label(ax.containers[0], label_type='edge')
    plt.show()


import seaborn as sns


# Numerical vs Numerical
sns.pairplot(train.select_dtypes(include=[np.number]))
plt.show()


# sns.pairplot(train, hue="Sex")
# plt.show()


# Categorical vs Numerical
for cat in train.select_dtypes(include='object').columns:
    for num in train.select_dtypes(include=[np.number]).columns:
        sns.boxplot(x=cat, y=num, data=train)
        plt.title(f'{cat} vs {num}')
        plt.show()


corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


sns.histplot(train['Calories'], kde=True)



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


X = train.drop('Calories', axis = 1)
y = train['Calories']


# Preprocessing
numeric_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
categorical_features = ["Sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first'),categorical_features )
    ]
)


model = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])


model.fit(X, y)


# Predict
y_pred = model.predict(X)


# Ensure no negative predictions for RMSLE
y_pred = np.maximum(0, y_pred)

# RMSLE
rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.4f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.6)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")
plt.grid(True)
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col=0)

test.shape


test.head()


test_predict = model.predict(test)


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
print(sample_sub.shape)
sample_sub.head()


sample_sub['Calories'] = test_predict 


sample_sub.head()


sample_sub.to_csv("submission.csv", index=False)

