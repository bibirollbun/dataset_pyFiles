import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder 
%load_ext cudf.pandas
import seaborn as sns
import cudf
import matplotlib.pyplot as plt
from cuml import Lasso
from sklearn.model_selection import KFold


train = cudf.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = cudf.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
cols = list(train.columns[1:-1])
print(cols)
print(f'len - {len(cols)}')


train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)
train['Brand'].fillna(train['Brand'].mode()[0], inplace=True)
train['Material'].fillna(train['Material'].mode()[0], inplace=True)
train['Size'].fillna(train['Size'].mode()[0], inplace=True)
train['Laptop Compartment'].fillna(train['Laptop Compartment'].mode()[0], inplace=True)
train['Waterproof'].fillna(train['Waterproof'].mode()[0], inplace=True)
train['Style'].fillna(train['Style'].mode()[0], inplace=True)
train['Color'].fillna(train['Color'].mode()[0], inplace=True)

test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)
test['Brand'].fillna(test['Brand'].mode()[0], inplace=True)
test['Material'].fillna(test['Material'].mode()[0], inplace=True)
test['Size'].fillna(test['Size'].mode()[0], inplace=True)
test['Laptop Compartment'].fillna(test['Laptop Compartment'].mode()[0], inplace=True)
test['Waterproof'].fillna(test['Waterproof'].mode()[0], inplace=True)
test['Style'].fillna(test['Style'].mode()[0], inplace=True)
test['Color'].fillna(test['Color'].mode()[0], inplace=True)


new_columns = {}
new_columns2 = {}
COLS2 = []
for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols[i+1:]):
        name = f"{c1}-{c2}"
        new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str")
        new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str")
        COLS2.append(name)
        print(f"{i}-{i+j+1}, ", end='')
train = cudf.concat([train, cudf.DataFrame(new_columns)], axis=1)
test = cudf.concat([test, cudf.DataFrame(new_columns2)], axis=1)
print()
print(len(COLS2),"bi-grams generated")


new_columns = {}
new_columns2 = {}
COLS3 = []
for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols[i+1:]):
        for k, c3 in enumerate(cols[i+j+2:]):
            name = f"{c1}-{c2}-{c3}"
            new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str") + "_" + train[c3].astype("str")
            new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str") + "_" + test[c3].astype("str")
            COLS3.append(name)
            print(f"{i}-{i+j+1}-{i+j+k+2}, ", end='')
train = cudf.concat([train, cudf.DataFrame(new_columns)], axis=1)
test = cudf.concat([test, cudf.DataFrame(new_columns2)], axis=1)
print()
print(len(COLS3),"tri-grams generated")


del test['id']
del train['id']
Price = train.Price
del train['Price']


meow = [f"{c}-TE" for c in cols+COLS2+COLS3]
print(f'we have a {len(meow)} feautures')


TE = TargetEncoder(n_folds=25, smooth=20,  stat='mean')
for col in train.columns:
    TE.fit(train[col], Price)
    train[f"TE_{col}"] = TE.transform(train[col])
    test[f"TE_{col}"] = TE.transform(test[col])


for i in train.columns:
    if train[i].dtype == 'object':
        train[i], _ = train[i].factorize()
        train[i] = train[i] - train[i].min()
        test[i], _ = test[i].factorize()
        test[i] = test[i] - test[i].min()
train.head(1)


test.head(1)


total_missing = test.isna().sum().sum()
print("Общее количество пропущенных значений:", total_missing)


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"Fold {fold + 1}")

    X_train = train.loc[train_idx].copy().to_numpy() 
    y_train = Price.iloc[train_idx].values  
    X_valid = train.loc[valid_idx].copy().to_numpy() 
    y_valid = Price.iloc[valid_idx].values    
    X_test = test.to_numpy()  
    
    model = Lasso(alpha=1e2)
    model.fit(X_train, y_train)
    
    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / kf.n_splits 
    print("--" * 25)

y_true = Price.to_numpy() 

rmse = np.sqrt(mean_squared_error(y_true, oof_preds))
print(f"Validation RMSE: {rmse}")


data = {'Names': train.columns,
        'Numbers': model.coef_}


df = pd.DataFrame(data)
df = df.sort_values("Numbers",ascending=True)
df = df.loc[df.Numbers != 0]

df.plot(x='Names', y='Numbers', kind='barh', legend=False, figsize=(10, 20))

plt.show()


sub = cudf.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = test_preds
print("Submission shape:",sub.shape)
sub.to_csv(f"submission.csv",index=False)
sub.head()


print(min(test_preds))
print(max(test_preds))

