# importing required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


# reading the datasets - train and test
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# remove id columns
train_df = train.drop(["id","y"], axis=1)
test_df = test.drop("id", axis=1)


print("train df\n",train_df.isnull().sum())
print("\ntest_df\n",test_df.isnull().sum())


# columns as their data types
def segregate_categorical_numerical(df):
    columns = list(df.columns)
    numeric_columns = []
    categorical_columns = []
    for column in columns:
        dtype = train_df[column].dtype
        # code for separting numeric and object dtype columns
        if dtype in ['int64','float64']:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns

train_num_cols, train_cat_cols = segregate_categorical_numerical(train_df)
test_num_cols, test_cat_cols = segregate_categorical_numerical(test_df)

print("Numeric Features in...")
print("train data:",train_num_cols)
print("test data:",test_num_cols)
print("\nCategorical Features in...")
print("train data:",train_cat_cols)
print("test data:",test_cat_cols)


# encoding categorical features using label encoding
le = LabelEncoder()
train_categorical_features = dict()
test_categorical_features = dict()
for cat_cols in train_cat_cols:
    if cat_cols == "month":
        continue
    train_categorical_features[cat_cols] = le.fit_transform(train_df[cat_cols])
    test_categorical_features[cat_cols] = le.transform(test_df[cat_cols])
train_categorical_features = pd.DataFrame(train_categorical_features)
test_categorical_features = pd.DataFrame(test_categorical_features)


# encoding numerical features using RobustScaler
rs = RobustScaler()
# removing the day column for customized changing
del_cols = train_cat_cols + ["day"]
train_df_rs = train_df.drop(del_cols, axis=1)
test_df_rs = test_df.drop(del_cols, axis=1)
# fit and transform the data
train_numerical_features_rs = rs.fit_transform(train_df_rs)
test_numerical_features_rs = rs.transform(test_df_rs)
# turning back to dataframe
train_numerical_features = pd.DataFrame(train_numerical_features_rs, columns=train_df_rs.columns)
test_numerical_features = pd.DataFrame(test_numerical_features_rs, columns=test_df_rs.columns)


# preprocessing the day and month feature
# combining day and month column
def new_day_month(df):
    day_month = []
    day = df["day"]
    month = df["month"]
    for x in range(len(day)):
        day_month.append(f"{day[x]}_{month[x]}")
    return day_month

# counting how many zeros and ones for unique day_month
def count_day_month_y(dm,y):
    day_month_y = dict()
    for x in range(len(y)):
        if dm[x] not in day_month_y.keys():
            day_month_y[dm[x]] = [0,0]
        if y[x] == 0:
            day_month_y[dm[x]][0] += 1
        else:
            day_month_y[dm[x]][1] += 1
    return day_month_y

# making the final feature column
def final_day_month(dm,cdm):
    fdm = []
    for x in dm:
        zeros = cdm[x][0]
        ones = cdm[x][1]
        # probability of 1 for that date of the year
        p = round(ones/(zeros+ones),5)
        fdm.append(p)
    return fdm

train_day_month = new_day_month(train_df)
train_count_day_month = count_day_month_y(train_day_month,list(train["y"]))
final_train_day_month = final_day_month(train_day_month,train_count_day_month)


# there's no "y" feature in test data because it is the target feature
test_day_month = new_day_month(test_df)
final_test_day_month = []
for x in test_day_month:
    if x not in train_day_month:
        # if a day is not present in training set, we instantiate it with 0.00001
        final_test_day_month.append(0.00001)
        continue
    final_test_day_month.append(final_train_day_month[train_day_month.index(x)])


# combining all the processed features for train and test dataset
# for train data
final_train_day_month = pd.DataFrame({
    "day_month": final_train_day_month
})
train_y = pd.DataFrame({
    "y": list(train["y"])
})
processed_train_df = pd.concat([train_categorical_features,train_numerical_features,final_train_day_month,train_y], axis=1)
# for test data
final_test_day_month = pd.DataFrame({
    "day_month": final_test_day_month
})
test_id = pd.DataFrame({"id": list(test["id"])})
processed_test_df = pd.concat([test_id,test_categorical_features,test_numerical_features,final_test_day_month], axis=1)


# save the processed dataframes as csv files
# train data
processed_train_df.to_csv("/kaggle/working/train.csv", index=False)
# test data
processed_test_df.to_csv("/kaggle/working/test.csv", index=False)

