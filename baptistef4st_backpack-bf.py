import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")



X_train = train_df.drop(["id", "Price"], axis=1)
y_train = train_df["Price"]
X_test = test_df.drop("id", axis=1)
X_train["Laptop Compartment"] = X_train["Laptop Compartment"].map({"Yes":1, "No":0})
X_train["Waterproof"] = X_train["Waterproof"].map({"Yes":1, "No":0})
X_test["Laptop Compartment"] = X_test["Laptop Compartment"].map({"Yes":1, "No":0})
X_test["Waterproof"] = X_test["Waterproof"].map({"Yes":1, "No":0})
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_val)
print(mean_absolute_error(y_val, preds))



model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
test_preds = model.predict(X_test)
submission = pd.DataFrame({"id": test_df["id"], "Price": test_preds})
submission.to_csv("submission.csv", index=False)


importances = model.feature_importances_
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(5)
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Top 5 Most Important Features')
plt.show()



df_val = pd.DataFrame({'Actual': y_val, 'Predicted': preds})
df_val_top5 = df_val.sort_values(by='Predicted', ascending=False).head(5)
df_val_top5.reset_index(inplace=True)
df_val_top5.rename(columns={'index': 'Sample'}, inplace=True)
df_melt = df_val_top5.melt(id_vars='Sample', value_vars=['Actual', 'Predicted'], var_name='Type', value_name='Price')
plt.figure(figsize=(10,6))
sns.barplot(x='Sample', y='Price', hue='Type', data=df_melt)
plt.title('Top 5 Predicted vs Actual Prices')
plt.xlabel('Validation Sample Index')
plt.ylabel('Price')
plt.show()


