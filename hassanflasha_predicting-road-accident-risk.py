import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("D:\\DATA\\Books\\Data Science\\Projects\\Predicting Road Accident Risk\\train.csv")


df


df.info()


df.describe()


df['road_type'].unique()


df["lighting"].unique()


df["weather"].unique()


df["time_of_day"].unique()


for col in df.select_dtypes(include=['number']).columns:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()

#We will keep the outliers since they're Important to our study and prediction while keeping in mind to choose a model robust to outliers.
# Outliers percentage is too low to effect on our study so the best option is to keep them.


df.loc[(df["num_reported_accidents"] > 3),'num_reported_accidents'].count(), df.loc[(df["accident_risk"] > 0.8),'accident_risk'].count()


pd.pivot_table(df,values= 'accident_risk',index ='road_type',aggfunc='mean')


pd.pivot_table(df,values = 'accident_risk',index='weather',aggfunc='mean')


pd.pivot_table(df,values = 'accident_risk',index='time_of_day',aggfunc='mean')


df.cov(numeric_only= True)


df.corr(method='spearman',numeric_only = True)


df_test = pd.read_csv("D:\\DATA\\Books\\Data Science\\Projects\\Predicting Road Accident Risk\\test.csv")


df = df.drop(columns= ['id',"school_season","road_signs_present"])
df_test = df_test.drop(columns= ['id',"school_season","road_signs_present"])


road_type_encode = {'urban':0,'rural':1,'highway':2}
lighting_encode = {'daylight':0,'dim':1,'night':2}
weather_encode = {'clear':0,'rainy':1,'foggy':2}
time_of_day_encode= {'morning':0,'afternoon':1,'evening':2}
df["road_type"] = df["road_type"].replace(road_type_encode)
df["lighting"] = df["lighting"].replace(lighting_encode)
df["weather"] = df["weather"].replace(weather_encode)
df["time_of_day"] = df["time_of_day"].replace(time_of_day_encode)
df["public_road"] = df["public_road"].astype(int)
df["holiday"] = df["holiday"].astype(int)
df_test["road_type"] = df_test["road_type"].replace(road_type_encode)
df_test["lighting"] = df_test["lighting"].replace(lighting_encode)
df_test["weather"] = df_test["weather"].replace(weather_encode)
df_test["time_of_day"] = df_test["time_of_day"].replace(time_of_day_encode)
df_test["public_road"] = df_test["public_road"].astype(int)
df_test["holiday"] = df_test["holiday"].astype(int)


df


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


x = df.drop(columns = 'accident_risk')
y = df['accident_risk']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)


model1 = RandomForestRegressor(random_state = 42)
model1.fit(x_train,y_train)


model1.score(x_test,y_test)


x_final = df_test
y_predict = model1.predict(x_final)


submission_file = pd.read_csv("D:\\DATA\\Books\\Data Science\\Projects\\Predicting Road Accident Risk\\test.csv")
ids = submission_file["id"]


submission_file = pd.DataFrame({'id': ids, 'accident_risk': y_predict})


submission_file


submission_file.to_csv('submission_file.csv',index = False)

