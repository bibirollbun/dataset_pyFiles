import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import duckdb


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")


df.info()


df.head(10)


df["Compartments"] = df["Compartments"].astype("int")
column_attributes = df.select_dtypes(include="object").columns
df[column_attributes] = df[column_attributes].astype('category')
df = df.set_index("id")


df[["Compartments", "Weight Capacity (kg)", "Price"]].hist(bins=50)
plt.show()


df.describe()


df["Material"].unique()


backpack_df = df.copy()


column_attributes = backpack_df.select_dtypes(include="object").columns
for col in column_attributes:
    ds = backpack_df[[col]].value_counts()
    plt.figure(figsize=(5,3))
    plt.bar(height = ds, x=[col[0] for col in ds.index])
    plt.title(col)
plt.show()    


backpack_df.select_dtypes(include="number").corr()["Price"]*100


backpack_df.head(5)


backpack_df, backpack_prices = backpack_df.drop(columns=["Price"]), backpack_df["Price"].copy()


weight_capacity_categorizer = lambda x : 0 if x <=12 else (1 if 12<x>=24 else 2)
backpack_df["wt_cat"] = backpack_df["Weight Capacity (kg)"].apply(weight_capacity_categorizer)


backpack_df


backpack_df_numeric = backpack_df.select_dtypes(include="number")


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer


backpack_df["Color"].unique()


num_pipeline = make_pipeline(SimpleImputer(strategy='mean'))

cat_imputer = Pipeline([("imputation",SimpleImputer(strategy="most_frequent"))])

cat_1hot_pipeline = make_pipeline(cat_imputer, OneHotEncoder(handle_unknown="ignore"))

cat_ord_pipeline_color = make_pipeline(cat_imputer, 
                                        OrdinalEncoder(categories=[['Red', 'Pink', 'Green', 'Blue', 'Gray', 'Black']]))
cat_ord_pipeline_size = make_pipeline(cat_imputer, 
                                        OrdinalEncoder(categories=[["Small", "Medium", "Large"]]))
cat_ord_rest = make_pipeline(cat_imputer, OrdinalEncoder())


cat_1hot_attribs = ['Brand', 'Material', 'Style']

preprocessing = ColumnTransformer([
    
    ('num', num_pipeline, backpack_df_numeric.columns),
    ('cat_1hot', cat_1hot_pipeline, cat_1hot_attribs),
    ('cat_ord_size', cat_ord_pipeline_size, ["Size"]),
    ('cat_ord_color', cat_ord_pipeline_color,['Color']),
    ('cat_ord_others',cat_ord_rest, ['Laptop Compartment', 'Waterproof'])
    
] )

preprocessing


backpack_prepared = preprocessing.fit_transform(backpack_df)


preprocessed_df = pd.DataFrame(backpack_prepared,columns= preprocessing.get_feature_names_out())
preprocessed_df


from sklearn.metrics import mean_squared_error


preprocessing


from sklearn.linear_model import LinearRegression


modified_final = make_pipeline(preprocessing, LinearRegression())


extra_train_set = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
targets = extra_train_set[["Price"]].copy()
extra_train_set = extra_train_set.drop(columns=["Price","id"])


extra_train_set["wt_cat"] = extra_train_set["Weight Capacity (kg)"].apply(weight_capacity_categorizer)


extra_train_set.shape


modified_final.fit(extra_train_set[:20369], targets[:20369])


realtest = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
#realtest1 = realtest.drop(columns="Price")
realtest["wt_cat"] = realtest["Weight Capacity (kg)"].apply(weight_capacity_categorizer)


final_preds = modified_final.predict(realtest)


output = pd.DataFrame(final_preds.round(3)).reset_index().rename(columns={0:"Price", 'index':'id'})#.to_csv('subs.csv')
output['id'] += 300000
output.to_csv("submission.csv", index=False)

