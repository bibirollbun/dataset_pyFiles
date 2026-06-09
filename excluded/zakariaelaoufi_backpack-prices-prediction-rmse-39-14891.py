import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


train_backpack = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
extra_backpack = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_backpack = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


backpack = pd.concat([train_backpack, extra_backpack])
backpack.drop('id', axis=1, inplace=True)
test_backpack.drop('id', axis=1, inplace=True)


backpack.columns = backpack.columns.str.replace(" ", "_")
test_backpack.columns = test_backpack.columns.str.replace(" ", "_")


backpack.head()


backpack.info()


threshold = 3
mask = backpack.isna().sum(axis=1) >= threshold
filtered_backpack = backpack[mask]


clean_backpack1 = backpack[~mask]


nominal_col = ['Brand', 'Material', 'Style', 'Color']
clean_backpack2 = clean_backpack1.dropna(subset=nominal_col)


clean_backpack2.isna().sum()


def get_labels_dict(data, col):
    labels = data[col].sort_values(ascending=False).unique().tolist()
    label_dic = {k:i for i, k in enumerate(labels)}
    if np.nan in label_dic.keys():
        label_dic[np.nan] = np.nan
    return label_dic


def reverse_dic(dic):
    return {v: k for k, v in dic.items()}


from sklearn.impute import KNNImputer

def column_imputation(data, to_impute_obj_col, to_impute_num_col):
    before_backpack = data.copy()
    
    # Create dictionaries dynamically for obj col
    label_dicts = {col: get_labels_dict(before_backpack, col) for col in to_impute_obj_col}
    
    # Map ordinal categories using generated dictionaries
    for col in to_impute_obj_col:
        before_backpack[col] = before_backpack[col].map(label_dicts[col])
            
    # Perform KNN Imputation
    imputer = KNNImputer(n_neighbors=3, weights='distance')
    imputed_backpack = imputer.fit_transform(before_backpack[to_impute_obj_col + to_impute_num_col])
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_backpack, columns=to_impute_obj_col + to_impute_num_col, index=before_backpack.index)

    # Map back to original categorical values
    for col in to_impute_obj_col:
        imputed_df[col] = imputed_df[col].round().astype(int)
        imputed_df[col] = imputed_df[col].map(reverse_dic(label_dicts[col]))

    imputed_backpack = data.copy()
    imputed_backpack[to_impute_obj_col + to_impute_num_col] = round(imputed_df)
    
    return imputed_backpack


to_impute_obj_col = ['Size', 'Laptop_Compartment', 'Waterproof']
to_impute_num_col = ['Weight_Capacity_(kg)']


for col in nominal_col:
    mod_value = test_backpack[col].mode()[0]
    test_backpack[col] = test_backpack[col].fillna(mod_value)


print(test_backpack.isnull().sum())


imputed_test_backpack = column_imputation(test_backpack, to_impute_obj_col, to_impute_num_col)


imputed_test_backpack.isnull().sum()


total_revenu = {
  "Adidas": 23.19,
  "Nike": 51.54,
  "Puma": 8.88,
  "Under Armour": 5.9,
  "Jansport": 10.5
}


final_backpack = clean_backpack2.dropna().copy()
final_backpack['total_revenue_2023'] = final_backpack['Brand'].map(total_revenu)
imputed_test_backpack['total_revenue_2023'] = imputed_test_backpack['Brand'].map(total_revenu)


final_backpack.head()


X = final_backpack.drop(['Price','Color'], axis=1)
y = final_backpack['Price']


ordinal_categories = {
    'Waterproof': ['Yes', 'No'],
    'Size': ['Small', 'Medium', 'Large'],
    'Laptop_Compartment': ['Yes', 'No']
}


num_col = X.select_dtypes(include='number').columns.tolist()


to_onehot_col = X.drop(list(to_impute_obj_col + num_col), axis=1).columns.tolist()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('onehot encoding', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), to_onehot_col),
    ('ordinal', OrdinalEncoder(
            categories=[ordinal_categories[col] for col in ordinal_categories],
        ), list(ordinal_categories.keys())),
    ('minmaxscaler', MinMaxScaler(), num_col)]
    , remainder="passthrough", verbose_feature_names_out=False
).set_output(transform="pandas")

X_transformed = transformer.fit_transform(X)


X_transformed.head()


from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import datetime as dt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
lr = LinearRegression()


dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)


lgb_param = {
    'application':'regression',
    'learning_rate':0.1, 
}


lr.fit(X_train, y_train)


start = dt.datetime.now()
clf = lgb.train(lgb_param, dtrain, 50)
end = dt.datetime.now()
elapsed = end - start


lr.score(X_train, y_train)


lr.score(X_test, y_test)


y_pred = lr.predict(X_test)
y_pred2 = clf.predict(X_test) 


from sklearn.metrics import mean_squared_error

rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))


print('Root Mean Squared Error Linear Regression: ', rmse1)
print('Root Mean Squared Error LightGBM: ', rmse2)


X_test_transformed = transformer.transform(imputed_test_backpack)


sub_pred = lr.predict(X_test_transformed)


sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sub['Price'] = sub_pred
sub.to_csv('submission.csv', index=False)

