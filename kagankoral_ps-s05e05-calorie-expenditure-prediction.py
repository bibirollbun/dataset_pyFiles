import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_df.head()


print("-----Top 5 Rows-----")
print(train_df.head(),"\n")
print("-----Train Data Info-----")
print(train_df.info(),"\n")
print("-----Train Data Description-----")
print(train_df.describe().T)


numeric_features = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

# Histograms
train_df[numeric_features].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()


# Correlation
corr = train_df[numeric_features].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()


plt.pie(train_df['Sex'].value_counts().values, labels = train_df['Sex'].value_counts().index, autopct='%1.1f%%')
plt.title('Ratios of Man & Woman Data')
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(data=train_df, x='Sex', y='Calories')
plt.title("Calories Distribution by Sex")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df[numeric_features])
plt.xticks(rotation=45)
plt.title("Outlier Check")
plt.show()


# BMI : Body Mass Index
# train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)

# Intensity : Heart Rate relative to Age
# train_df['Intensity'] = train_df['Heart_Rate'] / train_df['Age']

train_df.head()


y = train_df['Calories']
y


# Encoding & Scaling
categorical_features = ['Sex']
numerical_features = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

preprocessor = ColumnTransformer(
    transformers = [
        ('cat',OneHotEncoder(handle_unknown = 'ignore'), categorical_features),
        ('num',StandardScaler(), numerical_features)
    ]
)

pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor)
])


X_data = pipeline.fit_transform(train_df)

columns_cat = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
columns_all = list(columns_cat) + numerical_features
X_train_new = pd.DataFrame(X_data, columns=columns_all)
X_train_new.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

fold = 1
oof_preds = np.zeros(len(X_train_new))
fold_scores = []
models = []

for train_idx, val_idx in skf.split(X_train_new, y):
    print(f"Fold {fold}")

    X_train, X_val = X_train_new.iloc[train_idx], X_train_new.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)

    preds = model.predict(X_val)
    oof_preds[val_idx] = preds
    rmse = mean_squared_error(y_val, preds, squared=False)
    fold_scores.append(rmse)
    models.append(model)

    print(f"Fold {fold} RMSE: {rmse:.4f}")
    fold += 1

print(f"Average RMSE: {np.mean(fold_scores):.4f}")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# df_test['BMI'] = df_test['Weight'] / ((df_test['Height'] / 100) ** 2)

# df_test['Intensity'] = df_test['Heart_Rate'] / df_test['Age']

df_test_new = pipeline.transform(df_test)
test_df = pd.DataFrame(df_test_new, columns=columns_all)

test_df.head()


test_preds = np.zeros(len(test_df))

for model in models:
    test_preds += model.predict(test_df)

test_preds /= len(models)


submission = pd.DataFrame({
    'id': df_test['id'],
    'Calories': test_preds
})

submission.to_csv("submission.csv", index=False)




