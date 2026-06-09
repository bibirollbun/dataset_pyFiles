%%time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


%%time
train = pd.read_csv('../input/playground-series-s5e2/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s5e2/test.csv', index_col=0)

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


cols = train.columns


%%time
train.info()


%%time
train.describe()


%%time
test.describe()


%%time
train.describe(include=[object])


%%time
test.describe(include=[object])


train.isnull().mean() * 100


%%time
train.isnull().corr()


%%time
combs_list = list(combinations(cols, 3))

combs_nulls = {}

for t in combs_list:
    count = train[list(t)].isnull().all(axis=1).sum()
    combs_nulls[t] = count

df_combs = pd.DataFrame(list(combs_nulls.items()), columns=['Comb', 'Count_Null'])
df_combs.sort_values(by='Count_Null', ascending=False)


categorical_features = train.select_dtypes(include=["object"]).columns.tolist()
continuous_features = train.select_dtypes(include=["number"]).columns.tolist()

continuous_features.remove("Price")


for col in train.columns:
    plt.figure(figsize=(12, 5))

    if col in categorical_features:
        ax1 = plt.subplot(1, 2, 1)
        sns.countplot(y=train[col], order=train[col].value_counts().index[:10], palette="viridis", ax=ax1)  # Prime 10 categorie
        ax1.set_title(f"Distribution {col}")

        ax2 = plt.subplot(1, 2, 2)
        sns.boxplot(data=train, x="Price", y=col, palette="viridis", ax=ax2)
        ax2.set_title(f"Price Distribution {col}")

    elif col in continuous_features:
        ax1 = plt.subplot(1, 2, 1)
        sns.histplot(train[col], bins=30, color="blue", ax=ax1)
        ax1.set_title(f"Distribution {col}")

        ax2 = plt.subplot(1, 2, 2)
        sns.histplot(x=train[col], y=train["Price"], bins=30, cmap="viridis", cbar=True, ax=ax2)
        ax2.set_title(f"Price Distribution {col}")

    plt.tight_layout()
    plt.show()



from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_val, testing_set = train_test_split(train, test_size=0.1, random_state=42)
training_set, val_set = train_test_split(train_val, test_size=0.125, random_state=42)  
X_train = training_set.drop(columns='Price')
y_train = training_set['Price']

X_val = val_set.drop(columns='Price')
y_val = val_set['Price']

X_test = testing_set.drop(columns='Price')
y_test = testing_set['Price']

X_train[categorical_features] = X_train[categorical_features].fillna('NaN')
X_val[categorical_features] = X_val[categorical_features].fillna('NaN')
X_test[categorical_features] = X_test[categorical_features].fillna('NaN')

model = CatBoostRegressor(iterations=800, 
                          depth=8, 
                          learning_rate=0.01,
                          cat_features=categorical_features,
                          loss_function='RMSE')

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


import shap

explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer(X_test)


shap.plots.beeswarm(shap_values)


shap.plots.bar(shap_values)


errors = np.abs(y_pred - y_test)
df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred, 'Error': errors})
df_sorted = df.sort_values(by='Error', ascending=False)
df_sorted[:20]


sns.histplot(y_test, color='blue', kde=True, label='y_test', stat='density', bins=30)
sns.histplot(y_pred, color='red', kde=True, label='y_pred', stat='density', bins=30)

plt.legend()
plt.title('Comparison y_pred vs y_test')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()



sns.scatterplot(x=y_test, y=y_pred)
plt.title("Predictions vs Actual value")
plt.xlabel("Actual Value")
plt.ylabel("Predictions")
plt.show()


residui = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residui)
plt.axhline(0, color='red', linestyle='--')  # Linea orizzontale a zero
plt.title("Residual Plot")
plt.xlabel("Prediction")
plt.ylabel("Residual value")
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(residui, kde=True, color='blue')  # kde=True aggiunge la curva di densità
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


test.head()


test[categorical_features] = test[categorical_features].fillna('NaN')


y_sub = model.predict(test)


sns.histplot(y_sub, color='blue', kde=True, label='y_sub', stat='density', bins=30)

plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()



y_sub


submission = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission["Price"] = y_sub

display(submission.head())

submission.to_csv("submission.csv", index=False)




