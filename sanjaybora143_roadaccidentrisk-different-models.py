# Importing Libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder


# Importing Train dataset

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

train.head()


# Importing test dataset

test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

test.head()


# Importing sammple_submission dataset

sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

sample_submission.head()


# Information about dataset
print("Information : ")
print(train.info())

# Missing Values
print("\nMissing Values : ")
print(train.isnull().sum())

# Duplicate Entries
print("\nDuplcate Entries : ")
print(train.duplicated().sum())


road_type_types = train['road_type'].value_counts()

plot_axes = road_type_types.plot(kind = 'bar')

for p in plot_axes.patches:
    height = p.get_height()
    plot_axes.annotate(f'{height:,.0f}',
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center',
                textcoords='offset points',
                xytext=(0, 8),
                fontsize=9,
                fontweight='bold')

plt.show()


num_lanes_counts = train['num_lanes'].value_counts()

num_lanes_counts


train['curvature'].plot(kind = 'kde')


train['speed_limit'].plot(kind = 'hist')


lighting_types = train['lighting'].value_counts()
lighting_types.plot(kind = 'barh')



weather_types = train['weather'].value_counts()

weather_types.plot(kind = 'bar')


time_of_day_types = train['time_of_day'].value_counts()

time_of_day_types.plot(kind = 'barh')


train['num_reported_accidents'].plot(kind = 'hist')


plt.figure(figsize = (12,5))
n = 0

obj_columns = ['road_type','lighting','weather','time_of_day']
for col in obj_columns : 
    n = n+1
    plt.subplot(2,2,n)
    sns.barplot(data = train, x = col , y = 'accident_risk')
    plt.title(f'accident_risk vs {col}')

plt.tight_layout()
plt.show()


plt.figure(figsize = (12,5))
n = 0

bool_columns = ['road_signs_present','public_road','holiday','school_season']
for col in bool_columns : 
    n = n+1
    plt.subplot(2,2,n)
    sns.barplot(data = train, x = col , y = 'accident_risk')
    plt.title(f'accident_risk vs {col}')

plt.tight_layout()
plt.show()


plt.figure(figsize = (12,7))
corr = train.corr(numeric_only = True)

sns.heatmap(corr, annot = True, linewidth = 0.2)


# Deleting 'id' column in training and test

# Training dataset

train = train.drop('id',axis = 1)

# testing data

ids = test['id']
test = test.drop('id',axis = 1)


# Splitting training dataset to test models

x = train.drop('accident_risk',axis = 1)
y = train['accident_risk']


x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.2,
                                                random_state = 20)


minmax_scaler = ('minmax',MinMaxScaler(),['num_lanes','curvature','speed_limit','num_reported_accidents'])
ordinal_encode = ('ordinal',OrdinalEncoder(categories = [[False,True],[False,True],[False,True],[False,True]]),['road_signs_present','public_road','holiday','school_season'])
onehot_encode = ('onehot',OneHotEncoder(),['road_type','lighting','weather','time_of_day'])


trf1 = ColumnTransformer(transformers = [
    minmax_scaler,
    ordinal_encode,
    onehot_encode
],remainder = 'passthrough')


from sklearn.linear_model import LinearRegression


model_1 = LinearRegression()


pipe_1 = make_pipeline(trf1,model_1)


pipe_1.fit(x_train,y_train)


from sklearn.metrics import mean_absolute_error
y_pred = pipe_1.predict(x_test)

print("MAE : ",mean_absolute_error(y_pred,y_test))


from sklearn.linear_model import Ridge


model_2 = Ridge(alpha = 100)


pipe_2 = make_pipeline(trf1,model_2)


pipe_2.fit(x_train,y_train)


y_pred_2 = pipe_2.predict(x_test)
print("MAE : ",mean_absolute_error(y_pred_2,y_test))


overall_pred = pipe_2.predict(test)


submission_df = pd.DataFrame({
    'id' : ids,
    'accident_risk' : overall_pred
    
})


submission_df.to_csv('submission.csv',index = False)

