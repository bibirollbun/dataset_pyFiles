!pip install pytorch-tabnet -q


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

display(train.head(3))
display(test.head(3))


train["Weight Capacity (kg)_nan"] = train["Weight Capacity (kg)"].isna().astype(int)
train["Weight Capacity (kg)"].fillna(0, inplace=True)

categorical_columns = train.select_dtypes(include=["object"]).columns 
train[categorical_columns] = train[categorical_columns].fillna("missing_category")

display(train.head())


import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor

categorical_columns = train.select_dtypes(include=["object"]).columns
cat_idxs = [train.columns.get_loc(col) for col in categorical_columns]
cat_dims = [train[col].nunique() for col in categorical_columns]

for col in categorical_columns:
    train[col] = train[col].astype('category').cat.codes

y = train["Price"].values  
X = train.drop(columns=["Price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TabNet Regressor
reg = TabNetRegressor(
    device_name=device,  
    optimizer_params=dict(lr=1e-2),  
    n_d=64,  
    n_a=64,  
    gamma=1.5,  
    lambda_sparse=1e-3,  
    verbose=10,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
)

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.reshape(-1, 1) 
y_test_np = y_test.reshape(-1, 1)

reg.fit(
    X_train_np, y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_metric=['rmse'],
    max_epochs=100,  
    patience=20,  
    batch_size=1024,  
    virtual_batch_size=128,  
    num_workers=4,
)

y_pred = reg.predict(X_test.to_numpy())
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")


explain_matrix, masks = reg.explain(X_test.to_numpy())
feature_importance = np.mean(explain_matrix, axis=0)

plt.figure(figsize=(10, 5))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("TabNet Feature Importance")
plt.show()


y_test_1d = y_test.ravel()
y_pred_1d = y_pred.ravel()

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test_1d, y=y_pred_1d)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


test["Weight Capacity (kg)_nan"] = test["Weight Capacity (kg)"].isna().astype(int)
test["Weight Capacity (kg)"].fillna(0, inplace=True)

categorical_columns_test = test.select_dtypes(include=["object"]).columns
test[categorical_columns_test] = test[categorical_columns_test].fillna("missing_category")

for col in categorical_columns_test:
    if col in categorical_columns:
        test[col] = test[col].astype('category').cat.codes
    else:
        print(f"Warning: {col} is not in training data, filling with -1")
        test[col] = -1

X_test_final = test.to_numpy()
y_test_pred = reg.predict(X_test_final).ravel()

submission = pd.DataFrame({"id": test.index, "Price": y_test_pred})
submission.to_csv("submission.csv", index=False)

display(submission.head())




