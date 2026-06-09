import pandas as pd
df = pd.read_csv('../data/introvert/train.csv')
print(df.columns)


import numpy as np
from sklearn.impute import SimpleImputer
# 空值
isna_df = df.isna()
cols = df.columns
for col in cols:
    if(len(isna_df[col].unique()) > 1):
        print(col)

# Time_spent_Alone的缺失值用平均值代替
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Time_spent_Alone"] = imp.fit_transform(df["Time_spent_Alone"].to_numpy().reshape(-1,1))

# Stage_fear 可以用IterativeImputer处理，但是其他列可能存在空值
# 如果是外向，那么Stage_fear是no。
for idx in df.index:
    # df.loc[idx, ["Stage_fear"]] 返回的是一个Series，不会直接等于None
    # 空值在pandas中通常是np.nan，不能用 is None 判断
    if pd.isna(df.loc[idx, "Stage_fear"]):
        if df.loc[idx, "Personality"] == "Extrovert":
            df.loc[idx, "Stage_fear"] = "No"
        elif df.loc[idx, "Personality"] == "Introvert":
            df.loc[idx, "Stage_fear"] = "Yes"


# Social_event_attendance
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Social_event_attendance"] = imp.fit_transform(df["Social_event_attendance"].to_numpy().reshape(-1,1))

# Going_outside
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Going_outside"] = imp.fit_transform(df["Going_outside"].to_numpy().reshape(-1,1))

# Drained_after_socializing
for idx in df.index:
    # df.loc[idx, ["Stage_fear"]] 返回的是一个Series，不会直接等于None
    # 空值在pandas中通常是np.nan，不能用 is None 判断
    if pd.isna(df.loc[idx, "Drained_after_socializing"]):
        if df.loc[idx, "Personality"] == "Extrovert":
            df.loc[idx, "Drained_after_socializing"] = "No"
        elif df.loc[idx, "Personality"] == "Introvert":
            df.loc[idx, "Drained_after_socializing"] = "Yes"

# Friends_circle_size
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Friends_circle_size"] = imp.fit_transform(df["Friends_circle_size"].to_numpy().reshape(-1,1))

# Post_frequency
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Post_frequency"] = imp.fit_transform(df["Post_frequency"].to_numpy().reshape(-1,1))


print(df["Time_spent_Alone"].unique())
print(df["Stage_fear"].unique())
print(df["Social_event_attendance"].unique())
print(df["Going_outside"].unique())
print(df["Drained_after_socializing"].unique())
print(df["Friends_circle_size"].unique())
print(df["Post_frequency"].unique())


print(df.duplicated().unique()) # 没有重复值


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report 


# Only get dummies for the categorical columns, then concatenate with numerical features
dummies = pd.get_dummies(df[["Stage_fear", "Drained_after_socializing"]])
X = pd.concat([
	df[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	dummies
], axis=1)
y = LabelEncoder().fit_transform(df["Personality"])
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


test_df = pd.read_csv('../data/introvert/test.csv')


isna_df = test_df.isna()
cols = test_df.columns
for col in cols:
    if(len(isna_df[col].unique()) > 1):
        print(col)

# Time_spent_Alone的缺失值用平均值代替
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df["Time_spent_Alone"] = imp.fit_transform(test_df["Time_spent_Alone"].to_numpy().reshape(-1,1))

# Social_event_attendance的缺失值用平均值代替
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df["Social_event_attendance"] = imp.fit_transform(test_df["Social_event_attendance"].to_numpy().reshape(-1,1))

# Going_outside的缺失值用平均值代替
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df["Going_outside"] = imp.fit_transform(test_df["Going_outside"].to_numpy().reshape(-1,1))

# Friends_circle_size
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df["Friends_circle_size"] = imp.fit_transform(test_df["Friends_circle_size"].to_numpy().reshape(-1,1))

# Post_frequency
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_df["Post_frequency"] = imp.fit_transform(test_df["Post_frequency"].to_numpy().reshape(-1,1))




# Stage_fear
# Stage_fear与Personality关联很大，因此，可以利用Stage_fear以外的其它特征建立模型，即利用Stage_fear不为空的数据训练模型（Drained_after_socializing字段为空的数据不参与模型训练），Stage_fear作为y，其它特征作为X，利用该模型预测Stage_fear为空的数据。

# 只用Stage_fear不为空且Drained_after_socializing不为空的数据训练模型
train_sf = test_df[(~test_df["Stage_fear"].isna()) & (~test_df["Drained_after_socializing"].isna())]
dummies = pd.get_dummies(train_sf[["Drained_after_socializing"]])
X_sf = pd.concat([
	 train_sf[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	dummies
], axis=1)

y_sf = train_sf["Stage_fear"]


sf_model = RandomForestClassifier(n_estimators=100, random_state=0)
sf_model.fit(X_sf, y_sf)

# 对Stage_fear为空的数据进行预测
test_sf_na = test_df[(test_df["Stage_fear"].isna()) & (~test_df["Drained_after_socializing"].isna())]
X_sf_na = pd.concat([
    test_sf_na[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	pd.get_dummies(test_sf_na[["Drained_after_socializing"]])
], axis=1)

sf_pred = sf_model.predict(X_sf_na)

# 填充预测结果
test_df.loc[X_sf_na.index, "Stage_fear"] = sf_pred

# 随机为test_df["Stage_fear"]为空的数据的Stage_fear字段填写Yes或者No

mask = test_df['Stage_fear'].isna()

# 生成与 NaN 数量相同数量的随机 'Yes' 或 'No'
random_choices = np.random.choice(['Yes', 'No'], size=mask.sum())

# 填入原 DataFrame
test_df.loc[mask, 'Stage_fear'] = random_choices


# 处理Drained_after_socializing：和处理Stage_fear同样的方式
train_das = test_df[~test_df["Drained_after_socializing"].isna()]
dummies = pd.get_dummies(train_das[["Stage_fear"]])
X_das = pd.concat([
	 train_das[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	dummies
], axis=1)

y_das = train_das["Drained_after_socializing"]


das_model = RandomForestClassifier(n_estimators=100, random_state=0)
das_model.fit(X_das, y_das)

# 对Drained_after_socializing为空的数据进行预测
test_das_na = test_df[(test_df["Drained_after_socializing"].isna())]
X_das_na = pd.concat([
    test_das_na[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	pd.get_dummies(test_das_na[["Stage_fear"]])
], axis=1)

das_pred = das_model.predict(X_das_na)

# 填充预测结果
test_df.loc[X_das_na.index, "Drained_after_socializing"] = das_pred

# 随机为test_df["Drained_after_socializing"]为空的数据的Drained_after_socializing字段填写Yes或者No

mask = test_df['Drained_after_socializing'].isna()

# 生成与 NaN 数量相同数量的随机 'Yes' 或 'No'
random_choices = np.random.choice(['Yes', 'No'], size=mask.sum())

# 填入原 DataFrame
test_df.loc[mask, 'Drained_after_socializing'] = random_choices


result_pred = model.predict(pd.concat([
    test_df[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]],
	pd.get_dummies(test_df[["Stage_fear", "Drained_after_socializing"]])
], axis=1))

# 0对应 Extrovert，1对应Introvert
# 将result_pred转换为Personality标签
personality_map = {0: "Extrovert", 1: "Introvert"}
personality_pred = [personality_map[x] for x in result_pred]

# 生成提交结果
submit_df = pd.DataFrame({
    "id": test_df["id"],
    "Personality": personality_pred
})

# 保存为csv文件
submit_df.to_csv("submission.csv", index=False)

