import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test =pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
data=train
y=train[['Listening_Time_minutes']]


train.isnull().sum()


test.head()


data.head()


for i in ['Podcast_Name','Episode_Length_minutes','Genre','Host_Popularity_percentage','Publication_Day','Guest_Popularity_percentage','Number_of_Ads','Publication_Time','Episode_Sentiment']:

        plt.figure(figsize=(20, 5))
        sns.histplot(data[i], kde=True)
        plt.xticks(rotation=90)
        plt.title(f'{i}')
        plt.tight_layout()
        plt.show()


for i in ['Podcast_Name', 'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage',
          'Publication_Day', 'Guest_Popularity_percentage', 'Number_of_Ads',
          'Publication_Time', 'Episode_Sentiment']:

    plt.figure(figsize=(12, 5))
    
    if data[i].dtype == 'object' or data[i].nunique() < 10:  # categorical
        sns.boxplot(x=data[i], y=data['Listening_Time_minutes'])
        plt.xticks(rotation=90)
    else: 
        sns.scatterplot(x=data[i], y=data['Listening_Time_minutes'])
    
    plt.title(f'{i} vs Listening_Time')
    plt.tight_layout()
    plt.show()




custom_palette = ['#9b59b6', '#f39c12', '#1abc9c']
sns.set(style='darkgrid')


data['Source'] = 'All Data'  


numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


def generate_numerical_feature_visualizations(feature_name):
    plt.figure(figsize=(14, 5))

  
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x=feature_name, y="Source", palette=custom_palette)
    plt.xlabel(feature_name)
    plt.title(f"Box Plot for {feature_name} Across Datasets")

 
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=feature_name, color=custom_palette[0], kde=True, bins=30, label="All Data", alpha=0.6)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {feature_name}")
    plt.legend(title="Dataset")

    plt.tight_layout()
    plt.show()


def generate_categorical_feature_visualizations(feature_name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x=feature_name, y="Listening_Time_minutes", palette=custom_palette)
    plt.xlabel(feature_name)
    plt.ylabel("Listening Time (Minutes)")
    plt.title(f"Box Plot for {feature_name} vs Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for feature in numerical_features:
    generate_numerical_feature_visualizations(feature)


for feature in categorical_features:
    generate_categorical_feature_visualizations(feature)



data.drop('Source', axis=1, inplace=True)




import seaborn as sns
import matplotlib.pyplot as plt

data_numeric = data.select_dtypes(include=['number'])

correlation_matrix = data_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()



numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

numerical_data = data[numerical_features]
data_with_target = numerical_data.copy()
data_with_target['Listening_Time_minutes'] = data['Listening_Time_minutes']

corr_data = data_with_target.corr()
mask_data = np.triu(np.ones_like(corr_data, dtype=bool))

annot_kws = {"size": 16, "rotation": 45}

plt.figure(figsize=(10, 10))

sns.heatmap(corr_data, mask=mask_data, cmap='viridis', annot=True,
            square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Entire Dataset', fontsize=24)

plt.tight_layout()
plt.show()




data.Genre.unique()


data.isnull().sum()


(data['Guest_Popularity_percentage']==0).sum()


def preprocessing1(data):
    data = data.copy()
    data['Night_Show'] = (data['Publication_Time'] == 'Night').astype(int)
    data['Morning_Show'] = (data['Publication_Time'] == 'Morning').astype(int)
    data['Offday'] = data['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    data[['Episode_Length_minutes']]=data[['Episode_Length_minutes']].fillna(train[['Episode_Length_minutes']].mean())
    data['Guest_Popularity_percentage']=data['Guest_Popularity_percentage'].fillna(data['Guest_Popularity_percentage'].mean())
    data['Difference']=(data['Guest_Popularity_percentage']-data['Host_Popularity_percentage'])

    data['Total']=(data['Guest_Popularity_percentage']+data['Host_Popularity_percentage'])
    data['Density']=data['Number_of_Ads']/data['Episode_Length_minutes']
    data['Adfree']=(data['Number_of_Ads']==0).astype(int)
    data['Squared_Guest_Popularity_percentage']=(data['Guest_Popularity_percentage']**2)
    data['Squared_Host_Popularity_percentage']=(data['Host_Popularity_percentage']**2)
    data['Average']=(data['Guest_Popularity_percentage']+data['Host_Popularity_percentage'])/2
    data['Episode_Title'] = data['Episode_Title'].str.extract(r'(\d+)').astype(float)
    data['Engagement_Score'] = data['Host_Popularity_percentage'] * 0.6 + data['Guest_Popularity_percentage'] * 0.4
    data['Is_Prime_Time'] = data['Publication_Time'].isin(['Afternoon', 'Evening']).astype(int)
    genre_popularity = data.groupby('Genre')['Host_Popularity_percentage'].mean()
    data['Genre_Popularity'] = data['Genre'].map(genre_popularity)
    data['Host_Guest_Interaction'] = data['Host_Popularity_percentage'] * data['Guest_Popularity_percentage']
    data['Length_Popularity_Interaction'] = data['Episode_Length_minutes'] * data['Host_Popularity_percentage']
    return data



data=preprocessing1(data)
test=preprocessing1(test)
data['Number_of_Ads'] = data['Number_of_Ads'].fillna(data['Number_of_Ads'].mean().round())
test['Number_of_Ads'] = test['Number_of_Ads'].fillna(test['Number_of_Ads'].mean().round())
test['Density'] = test['Density'].fillna(1)
data['Density'] = data['Density'].fillna(1)


def categorize_popularity(value):
    if value > 70:
        return 2
    elif value >= 40:
        return 1
    else:
        return 0

def categorize_Episode(value):
    if 1<=value <=25:
        return 1
    elif 26<=value <=50:
        return 2
    elif 51<=value <=75:
        return 3
    elif 76<=value <=100:
        return 4

def preprocessing2(data):
    data['Host_Popularity_Level'] = data['Host_Popularity_percentage'].apply(categorize_popularity)
    data['Guest_Popularity_Level'] = data['Guest_Popularity_percentage'].apply(categorize_popularity)
    data['Episode']=data['Episode_Title'].apply(categorize_Episode)
    return data



data=preprocessing2(data)
test=preprocessing2(test)


data.head()


le = LabelEncoder()
for i in ['Publication_Day','Genre','Publication_Time','Podcast_Name'] :
    data[i] = le.fit_transform(data[i])
    test[i]=le.transform(test[i])


for col in ['Podcast_Name',	'Episode_Title',	'Episode_Length_minutes',	'Genre',	'Host_Popularity_percentage',	'Publication_Day',
         'Publication_Time',	'Guest_Popularity_percentage']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower) | (data[col] > upper)]

    print(f'{col}: Found {len(outliers)} outliers')

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


Q1 = data['Episode_Length_minutes'].quantile(0.25)
Q3 = data['Episode_Length_minutes'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#data = data[(data['Episode_Length_minutes'] >= lower_bound) & (data['Episode_Length_minutes'] <= upper_bound)]


encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
def preprocessing3(data):
    encoded =encoder.fit_transform(data[['Episode_Sentiment']])
    columns=encoder.get_feature_names_out(['Episode_Sentiment'])
    encoded_df=pd.DataFrame(encoded,columns=columns,index=data.index)
    data=pd.concat([data,encoded_df],axis=1)
    data=data.drop(columns=['Episode_Sentiment'])
    return data




data=preprocessing3(data)
test=preprocessing3(test)


data.Podcast_Name.nunique()


data.reset_index(drop=True, inplace=True)
data.tail()


data.head()


data=data.drop(columns='id')
test=test.drop(columns='id')


columns_to_scale = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                    'Guest_Popularity_percentage', 'Number_of_Ads']

scaler = StandardScaler()

data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
test[columns_to_scale] = scaler.transform(test[columns_to_scale])


data.head()


y=data.Listening_Time_minutes
data=data.drop(columns=['Listening_Time_minutes'])


test.head()


data.head()


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.2)


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


X_test.isnull().sum()


X_train.isnull().sum()


y_test.isnull().sum()


y_train.isnull().sum()


dt_model = DecisionTreeRegressor(max_depth= 10, max_features= None, min_samples_leaf= 2, min_samples_split= 2)
dt_model.fit(X_train, y_train)
y_pred2 = dt_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
print(f'Decision Tree Regressor RMSE: {rmse}')


rf_model = RandomForestRegressor(
    n_estimators=120,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    min_samples_leaf= 1,
    min_samples_split= 2
)

rf_model.fit(X_train, y_train)
y_pred3 = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred3))
print(f'Random Forest Regressor RMSE: {rmse}')


gb_model = GradientBoostingRegressor(
    n_estimators=400,
    max_depth=15,
    max_features='sqrt',
    learning_rate=0.1,
    random_state=42,
    min_samples_leaf=1,
    min_samples_split=2
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print(f'Gradient Boosting: {rmse_gb}')


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=550,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost Regressor RMSE: {rmse_xgb}')



from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=30,
    min_child_samples=50,
    min_child_weight=0.001,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    boosting_type='gbdt',
    importance_type='gain'
)
lgb_model.fit(X_train,y_train)
y_pred_lgb=lgb_model.predict(X_test)
rmse_lgb=np.sqrt(mean_squared_error(y_test,y_pred_lgb))
print(rmse_lgb)
#12.842


RMSE=float('inf')
z=0
for j in [0.0001,0.005,0.001,0.01,0.1,0.3,0.5,0.7,0.8]:
        for k in [0.0001,0.005,0.001,0.01,0.1,0.3,0.5,0.7,0.8]:
            for l in [0.0001,0.005,0.001,0.01,0.1,0.3,0.5,0.7,0.8]:
                final_preds_weighted = (
                                        np.array(y_pred3) * j +
                                        np.array(y_pred_gb) * k+
                                        np.array(y_pred_xgb) * l
                                       )
                rmse = np.sqrt(mean_squared_error(y_test, final_preds_weighted))
                z=z+1
                if rmse<RMSE :
                    RMSE=rmse
                    m=j
                    n=k
                    o=l

                if z%100==0:
                   print(f'  {j}  {k} {l} \n RMSE {rmse}')
print(f'\n lowest RMSE = {RMSE} \n {m} {n} {o}')


y_pred_xgb = xgb_model.predict(test)
y_pred3 = rf_model.predict(test)
y_pred_gb = gb_model.predict(test)
final_preds= (np.array(y_pred_xgb) * o +np.array(y_pred3) * m +np.array(y_pred_gb) * n)


final_preds


test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': final_preds
})


submission.to_csv('submissioncomp.csv', index=False)


submission

