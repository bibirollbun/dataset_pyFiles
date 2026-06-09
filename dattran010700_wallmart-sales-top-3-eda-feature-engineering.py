# Data handling
import pandas as pd
import numpy as np

# Viz
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Sklearn
from sklearn import model_selection, metrics

# Feature selection
import eli5
from eli5.sklearn import PermutationImportance

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn import linear_model, ensemble
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# Remove warnings
import warnings
warnings.filterwarnings('ignore') 


features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')


features.head()


train.head()


stores.head()


feature_store = features.merge(stores, how='inner', on = "Store")


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)


train_df.head()


test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


feature_store = features.merge(stores, how='inner', on = "Store")

# Converting date column to datetime 
feature_store['Date'] = pd.to_datetime(feature_store['Date'])
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Adding some basic datetime features
feature_store['Day'] = feature_store['Date'].dt.day
feature_store['Week'] = feature_store['Date'].dt.week
feature_store['Month'] = feature_store['Date'].dt.month
feature_store['Year'] = feature_store['Date'].dt.year


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)


test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


train_df.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .set_caption("Stats Summary of Numeric Variables")\
                            .background_gradient(subset=['min'], cmap='Reds')\
                            .background_gradient(subset=['max'], cmap='Greens')\
                            .background_gradient(subset=['std'], cmap='GnBu')\
                            .background_gradient(subset=['50%'], cmap='GnBu')


palletes = {
   'continuos':{'blues': ['#03045E', '#023E8A', '#0077B6', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4', '#90E0EF', '#ADE8F4', '#CAF0F8'],
                'green_n_blues': ['#D9ED92', '#B5E48C', '#99D98C', '#76C893', '#52B69A', '#34A0A4', '#168AAD', '#1A759F', '#1E6091', '#184E77']
               }
            }


template = dict(layout=go.Layout(font=dict(family="Enriqueta", size=12))) # Cabin | Franklin Bold


df_weeks = train_df.groupby('Week').sum()

fig = px.line(data_frame=df_weeks, x=df_weeks.index, y='Weekly_Sales', 
              template='simple_white', 
              labels={'Weekly_Sales' : 'Total Sales', 'x' : 'Weeks'})

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales over the year across every week</b>', 'x': 0.075},
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))



legend_names = {'MarkDown1': "MD 1",
                'MarkDown2': 'MD 2',
                'MarkDown3': 'MD 3',
                'MarkDown4': 'MD 4',
                'MarkDown5': 'MD 5',
                'Weekly_Sales': 'Sales'}


fig = px.line(df_weeks, x=df_weeks.index, y=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales'], 
              color_discrete_sequence=palletes['continuos']['green_n_blues'],
              template='simple_white', 
              labels={'value' : 'Total Sales', 'x' : 'Weeks'})

for trace_index, trace in enumerate(fig.data):
    trace.name = legend_names[trace.name]

fig.update_layout(
    template=template, 
    title={'text':'<b>Markdowns (MD) vs Sales</b>', 'x': 0.075},
    legend_title_text='<b>MDs & Sales</b>',
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))


weekly_sales = train_df.groupby(['Year','Week'], as_index = False).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2010 = train_df.loc[train_df['Year']==2010].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2011 = train_df.loc[train_df['Year']==2011].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2012 = train_df.loc[train_df['Year']==2012].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})

weekly_sales_data = {
    '2010': weekly_sales2010['Weekly_Sales']['mean'].to_dict(),
    '2011': weekly_sales2011['Weekly_Sales']['mean'].to_dict(),
    '2012': weekly_sales2012['Weekly_Sales']['mean'].to_dict()
}

weekly_sales_df = pd.DataFrame(weekly_sales_data)


line_columns = ['2010', '2011', '2012']

weekly_sales_df_sorted = weekly_sales_df.sort_index()

fig = px.line(weekly_sales_df_sorted, x=weekly_sales_df_sorted.index, y=line_columns, 
              labels={'x': 'Week', 'value': 'Total Sales'},
              color_discrete_sequence=palletes['continuos']['blues'])


fig.update_layout(
    template=template, 
    margin=dict(b=95),
    title={'text':'<b>Sales across the years by weeks<b>', 'x': 0.075}, xaxis_title='Week',
    legend_title_text='<b>Year</b>',
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))

fig.add_annotation(
    x=47, y=25000,
    text="<b>Thanksgiving</b>", 
    bordercolor="#585858",
    showarrow=False, 
    borderpad=2.5, 
    bgcolor='white')

fig.add_annotation(
    x=51, y=29000,
    text="<b>Christmas</b>",  
    bordercolor="#585858", 
    showarrow=False, 
    borderpad=2.5,
    bgcolor='white')


# Converting the temperature to celsius for a better interpretation
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)


train_plt = train_df.sample(frac=.1, random_state=42)


fig = px.histogram(train_plt, x='Temperature', y='Weekly_Sales', color='IsHoliday', marginal='box', opacity=0.55,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Behaviour of Temperature and Sales by Holiday<br>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title=' ',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))




fig=px.histogram(train_plt, x='Fuel_Price', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.55,
                 facet_col='IsHoliday', facet_col_spacing=0.05,
                 color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(template=template, 
                  title={'text':'<b>Fuel Price behaviour and Sales by Holiday</b>', 'x': 0.075}, 
                  yaxis_title='Total Sales', xaxis_title='',
                  legend_title_text='<b>Holidays</b>',
                  legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Fuel Price")


fig = px.histogram(train_plt, x='CPI', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.55,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   title='CPI and sales by holiday',color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Inflation (CPI) impact in Sales by Holiday</b><br>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title='',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Consumer Price Index")


fig = px.histogram(train_plt, x='Unemployment', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Unemployment Rate and Sales by Holiday</b>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title='',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Unemployment Rate")


sizes = train_plt.groupby('Size').mean()
fig = px.line(sizes, x = sizes.index, y = sizes.Weekly_Sales, template='simple_white',
              labels={'Weekly_Sales' : 'Total Sales', 'Size' : 'Store Size'})

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales across different Store sizes</b>', 'x': 0.075},
    yaxis=dict(showline=True))


store_type = pd.concat([stores['Type'], stores['Size']], axis=1)

fig = px.box(store_type, x='Type', y='Size', color='Type', 
             title='Store size and Store type',
             color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Store Size and Store Type</b>', 'x': 0.075},
    yaxis_title='Size', 
    xaxis_title='Type',
    yaxis=dict(showline=True))


store_sale = pd.concat([stores['Type'], train_df['Weekly_Sales']], axis=1)

fig = px.box(store_sale.dropna(), x='Type', y='Weekly_Sales', color='Type', 
             color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Store Type and Sales</b>', 'x': 0.075},
    yaxis_title='Total Sales', 
    xaxis_title='Type',
    yaxis=dict(showline=True))


depts = train_plt.groupby('Dept').mean().sort_values(by='Weekly_Sales', ascending=False)

fig = px.bar(depts, x=depts.index, y=depts.Weekly_Sales, color=depts.Weekly_Sales, 
             color_continuous_scale=palletes['continuos']['green_n_blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales across Departaments</b>', 'x': 0.075},
    legend_title_text='<b>Sales</b>',
    yaxis=dict(showline=True))


corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
df_mask = corr.mask(mask).round(2)

fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                  x=df_mask.columns.tolist(),
                                  y=df_mask.columns.tolist(),
                                  colorscale=palletes['continuos']['green_n_blues'],
                                  hoverinfo="none", 
                                  showscale=True, ygap=1, xgap=1)

fig.update_xaxes(side="bottom")

fig.update_layout(
    template=template, 
    width=900, 
    height=700,
    margin=dict(l=100),
    title={'text':'<b>Feature correlation (Heatmap)</b>', 'x': 0.075},
    title_x=0.5, 
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange='reversed',
)

for i in range(len(fig.layout.annotations)):
    if fig.layout.annotations[i].text == 'nan':
        fig.layout.annotations[i].text = ""

fig.show()


weekly_sales_corr = train_df.corr().iloc[2,:]
corr_df = pd.DataFrame(data = weekly_sales_corr, index = weekly_sales_corr.index ).sort_values (by='Weekly_Sales', ascending=False)
corr_df = corr_df.iloc[1:]

fig = px.bar(corr_df, x=corr_df.index, y='Weekly_Sales', color=corr_df.index, labels={'index':'Features'},
             color_discrete_sequence=palletes['continuos']['green_n_blues'])

fig.update_traces(showlegend=False)

fig.update_layout(
    template=template, 
    title={'text':'<b>Features and his correlation with Sales</b>', 'x': 0.075},
    yaxis_title='Sales Increase',
    yaxis=dict(showline=True))


data_train = train_df.copy()
data_test = test_df.copy()


data_train['Days_to_Thansksgiving'] = (pd.to_datetime(train_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_train['Days_to_Christmas'] = (pd.to_datetime(train_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


data_test['Days_to_Thansksgiving'] = (pd.to_datetime(test_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_test['Days_to_Christmas'] = (pd.to_datetime(test_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


data_train['SuperBowlWeek'] = train_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_train['LaborDay'] = train_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_train['Tranksgiving'] = train_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_train['Christmas'] = train_df['Week'].apply(lambda x: 1 if x == 52 else 0)


data_test['SuperBowlWeek'] = test_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_test['LaborDay'] = test_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_test['Tranksgiving'] = test_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_test['Christmas'] = test_df['Week'].apply(lambda x: 1 if x == 52 else 0)


data_train['MarkdownsSum'] = train_df['MarkDown1'] + train_df['MarkDown2'] + train_df['MarkDown3'] + train_df['MarkDown4'] + train_df['MarkDown5'] 


data_test['MarkdownsSum'] = test_df['MarkDown1'] + test_df['MarkDown2'] + test_df['MarkDown3'] + test_df['MarkDown4'] + test_df['MarkDown5']


data_train.isna().sum()[data_train.isna().sum() > 0].sort_values(ascending=False)


data_test.isna().sum()[data_test.isna().sum() > 0].sort_values(ascending=False)


data_train.fillna(0, inplace = True)


data_test['CPI'].fillna(data_test['CPI'].mean(), inplace = True)
data_test['Unemployment'].fillna(data_test['Unemployment'].mean(), inplace = True)


data_test.fillna(0, inplace = True)


data_train['IsHoliday'] = data_train['IsHoliday'].apply(lambda x: 1 if x == True else 0)
data_test['IsHoliday'] = data_test['IsHoliday'].apply(lambda x: 1 if x == True else 0)


data_train['Type'] = data_train['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))
data_test['Type'] = data_test['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))


features = [feature for feature in data_train.columns if feature not in ('Date','Weekly_Sales')]


X = data_train[features].copy()
y = data_train.Weekly_Sales.copy()


data_sample = data_train.copy().sample(frac=.25)
X_sample = data_sample[features].copy()
y_sample = data_sample.Weekly_Sales.copy()


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_sample, y_sample, random_state=0, test_size=0.15)


feat_model = xgb.XGBRegressor(random_state=0).fit(X_train, y_train)


perm = PermutationImportance(feat_model, random_state=1).fit(X_valid, y_valid)
features = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())


features_weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())
features_weights


f_importances = pd.Series(dict(zip(X_valid.columns.tolist(), perm.feature_importances_))).sort_values(ascending=False)


weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names=X_valid.columns.tolist())
result = pd.read_html(weights.data)[0]
result


# Eval metric for the competition
def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)


models = {
          '    LGBM': lgb.LGBMRegressor(random_state = 0),
          ' XGBoost': xgb.XGBRegressor(random_state = 0, objective = 'reg:squarederror'),
          'Catboost': cb.CatBoostRegressor(random_state = 0, verbose=False),          
          '    HGBR': HistGradientBoostingRegressor(random_state = 0),
          ' ExtraTr': ensemble.ExtraTreesRegressor(bootstrap = True, random_state = 0),
          ' RandomF': ensemble.RandomForestRegressor(random_state = 0),
         }


def model_evaluation (name, model, models, X_train, y_train, X_valid, y_valid):
   
    rmses = []
    
    for i in range(len(models)):
    
        # Model fit
        model.fit(X_train, y_train)
        
        # Model predict
        y_preds = model.predict(X_valid)

        # RMSE
        rmse = np.sqrt(np.mean((y_valid - y_preds)**2))
        rmses.append(rmse)
        
    return np.mean(rmses)


for name, model in models.items():
    print(name + ' Valid RMSE {:.4f}'.format(model_evaluation(name, model, models, X_train, y_train, X_valid, y_valid)) )


X_baseline = X[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_baseline, y, random_state=0, test_size=0.1)


RF = ensemble.RandomForestRegressor(n_estimators=60, max_depth=25, min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, y_train)


test = data_test[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()
predict_rf = RF.predict(test)


sample_submission['Weekly_Sales'] = np.round(predict_rf, 2)
sample_submission.to_csv('RDF.csv',index=False)


import joblib
file_name = 'random_forest_final_model.joblib'

joblib.dump(RF, file_name)


ETR = ensemble.ExtraTreesRegressor(n_estimators=50, bootstrap = True, random_state = 0)
ETR.fit(X_train, y_train)


predict_etr = ETR.predict(test)


sample_submission['Weekly_Sales'] = np.round(predict_etr, 2)
sample_submission.to_csv('ETR.csv',index=False)


avg_preds = (predict_rf + predict_etr) / 2


sample_submission['Weekly_Sales'] = np.round(avg_preds, 2)
sample_submission.to_csv('ETR_RDF.csv',index=False)


import joblib
file_name = 'extra_tree_final_model.joblib'

joblib.dump(ETR, file_name)

