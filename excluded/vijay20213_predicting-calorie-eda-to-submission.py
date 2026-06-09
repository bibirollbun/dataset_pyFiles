import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sbn
import plotly.graph_objects as go


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train_df.head()


test_df.head()


train_df.info()


test_df.info()


class Analysis:
    def __init__(self, train_df:pd.DataFrame, test_df:pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df
        
        
    def plotPie(self, df:pd.DataFrame, col_name:str, title:str=None):
        col_data = dict(df[col_name].value_counts())
        labels, values = list(col_data.keys()), list(col_data.values())
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0)]  
        )
        fig.update_layout(
            title_text=title
        )    
        fig.show()
    
    
    def plotPieBasic(self, df:pd.DataFrame, labels:list, values:list, title:str=None):
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0)]  
        )
        fig.update_layout(
            title_text=title
        )    
        fig.show()
        

    def menVsWomen(self, col):
        labels = (f"Men {col}", f"Women Average {col}")
        men_bmi, women_bmi = self.train_df[self.train_df['Sex'].isin(['male'])][col].mean(), self.train_df[~self.train_df['Sex'].isin(['male'])][col].mean()
        avg_bmi = (men_bmi, women_bmi)
        self.plotPieBasic(self.train_df, labels, avg_bmi, title=f"Men vs Women average {col} (Train Data)")
        
        men_bmi, women_bmi = self.test_df[ self.test_df['Sex'].isin(['male'])][col].mean(),  self.test_df[~ self.test_df['Sex'].isin(['male'])][col].mean()
        avg_bmi = (men_bmi, women_bmi)
        self.plotPieBasic( self.test_df, labels, avg_bmi, title=f"Men vs Women average {col} (Test Data)")


    
    def plot_distribution(self, df, col:str, title:str,  bin_size=10):
        heights = df[col].dropna()

        min_height = int(heights.min())
        max_height = int(heights.max())
        bins = list(range(min_height, max_height + bin_size, bin_size))
        labels = [f"{b}-{b+bin_size-1}" for b in bins[:-1]]
    
        df[f'{col} Range'] = pd.cut(heights, bins=bins, labels=labels, right=False)
    
        height_counts = df[f'{col} Range'].value_counts().sort_index()

        fig = go.Figure(
            data=[go.Bar(x=height_counts.index.astype(str), y=height_counts.values)]
        )
    
        fig.update_layout(
            title=title,
            xaxis_title=f'{col} Range (cm)',
            yaxis_title='Population Count',
            xaxis_tickangle=-45
        )
    
        fig.show()

        
analyzer = Analysis(train_df = train_df, test_df = test_df)


analyzer.plotPie(train_df, 'Sex', 'Male vs Female ratio in training data')
analyzer.plotPie(test_df, 'Sex', 'Male vs Female ratio in testing data')


train_df['Age'].describe()

age_slabs = ('20-30', '30-40', '40-50', '50-60', '60-70', '70-80')
train_age_count = {}
test_age_count = {}
for key in age_slabs:
    min_age, max_age = map(int, list(key.split('-')))
    train_count = train_df[(train_df['Age']>=min_age) & (train_df['Age']<max_age)].shape[0]
    test_count = test_df[(test_df['Age']>=min_age) & (test_df['Age']<max_age)].shape[0]
    train_age_count[key] = train_count
    test_age_count[key] = test_count


analyzer.plotPieBasic(train_df, labels=list(train_age_count.keys()), values=list(train_age_count.values()), title='Training data Age percentage slabwise')
analyzer.plotPieBasic(test_df, labels=list(test_age_count.keys()), values=list(test_age_count.values()), title='Testing data Age percentage slabwise')


analyzer.plot_distribution(train_df, 'Height', title="Population Distribution by Height Range (Train data)")
analyzer.plot_distribution(test_df, 'Height', title="Population Distribution by Height Range (Test Data)")


analyzer.plot_distribution(train_df, 'Weight', title="Population Distribution by Weight Range (Train Data)")
analyzer.plot_distribution(test_df, 'Weight', title="Population Distribution by Weight Range (Test Data)")


train_df['BMI'] = train_df['Weight'].astype(float) / (train_df['Height'].astype(float) * 0.0254)
test_df['BMI'] = train_df['Weight'].astype(float) / (train_df['Height'].astype(float) * 0.0254)


analyzer.menVsWomen('BMI')


analyzer.menVsWomen('Duration')


analyzer.menVsWomen('Heart_Rate')


analyzer.menVsWomen('Body_Temp')


from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


# Lable Encoding

cat_cols = ['Sex', 'Height Range', 'Weight Range']
num_cols = [col for col in test_df.columns if col not in cat_cols]

encoder = LabelEncoder()

train_df['Age_enc'] = encoder.fit_transform(train_df['Sex'])
train_df['Height_Range_en'] = encoder.fit_transform(train_df['Height Range'])
train_df['Weight_Range_en'] = encoder.fit_transform(train_df['Weight Range'])

test_df['Age_enc'] = encoder.fit_transform(test_df['Sex'])
test_df['Height_Range_en'] = encoder.fit_transform(test_df['Height Range'])
test_df['Weight_Range_en'] = encoder.fit_transform(test_df['Weight Range'])
train_df.head()


train_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                  'Body_Temp', 'BMI', 'Age_enc', 'Height_Range_en', 
                  'Weight_Range_en']
target = 'Calories'
drop_features = ['id', 'Sex', 'Height Range', 'Weight Range']

print(len(train_df.columns) == len(train_features + [target] + drop_features))

# TRAIN TEST SPLIT 
X, y = train_df[train_features], train_df[target].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=train_df[target])
X_train.shape, y_test.shape



regression_models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    KNeighborsRegressor(),
    GaussianProcessRegressor(),
    MLPRegressor()
]



## EXPERIMETATION ON DIFFERENT REGRESSION MODELS

# metrices = {}
# for algo in regression_models:
#     print('training initiated ...')
#     model = algo.fit(X_train, y_train)
#     print('training done ...')
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     y_pred = model.predict(X_test)
#     y_pred = list(map(int, y_pred))
#     y_pred = list(map(float, y_pred))
#     precision = precision_score(y_test, y_pred, average='macro')
#     recall = recall_score(y_test, y_pred, average='macro')
#     f1 = f1_score(y_test, y_pred, average='macro')
#     results = {'train_score':train_score, 'test_score':test_score, 
#                            'precision':precision, 'recall':recall, 'f1':f1}
#     metrices[str(algo)] = results
#     print(str(algo), results)


final_model = RandomForestRegressor()
final_model.fit(X, y)


result = final_model.predict(test_df[train_features])
submission = test_df[['id']]
submission = submission.copy()
submission['Calories'] = result

submission.to_csv("Submission.csv", index=False)


submission







