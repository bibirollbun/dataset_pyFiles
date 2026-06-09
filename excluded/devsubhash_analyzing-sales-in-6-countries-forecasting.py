import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
import optuna
import warnings
warnings.filterwarnings('ignore')

from IPython.display import clear_output
from scipy.stats import mode
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_subm = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


custom_colors = ["#CC3749","#FF7372","#01C4ED","#00D97C",'#E9DE34']
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors),size=0.8)
plt.tick_params(axis='both', labelsize=0, length = 0)
custom_cmap = colors.LinearSegmentedColormap.from_list("custom", custom_colors)


df_train_row_count, df_train_column_count = df_train.shape
print('Total number of rows:', df_train_row_count)
print('Total number of columns:', df_train_column_count)


df_test_row_count, df_test_column_count = df_test.shape
print('Total number of rows:', df_test_row_count)
print('Total number of columns:', df_test_column_count)


df_train.head()


df_train.info()


feat_float = df_train.select_dtypes(float).columns
feat_int = df_train.select_dtypes(int).columns
feat_object = df_train.select_dtypes(object).columns
print("Float Features:",feat_float)
print("Integer Features:",feat_int)
print("Object Features:",feat_object)


labels=['Integer Features', 'Object Features']
values= [len(feat_int), len(feat_object)]


fig = go.Figure()
fig.add_trace(go.Pie(values = values,labels = labels,hole = 0.6, 
                     hoverinfo ='label+percent'))
fig.update_traces(textfont_size = 12, hoverinfo ='label+percent',textinfo ='label', 
                  showlegend = False,marker = dict(colors =["#FF7372","#01C4ED"]),
                  title = dict(text = 'Features Distribution'))  
fig.update_layout(height = 500, width = 700, bargap = 0.1, xaxis = dict(tickmode ='linear'),
                  title_text ="<b>Features Distribution</b>",paper_bgcolor ="#FDFCF6",
                  plot_bgcolor ="#FDFCF6",
                  title_font = dict(size = 20, family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line", xref ='paper', yref ='paper',
                                 x0 = -0.08, y0 = 1.09, x1 = 0.405, y1 = 1.09)])
fig.show()


df_train['date'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['dayOfMonth'] = df_train['date'].dt.day
df_train['dayOfYear'] = df_train['date'].dt.dayofyear
df_train['weekday'] = df_train['date'].dt.weekday
df_train['year_sin'] = np.sin(2 * np.pi * df_train['year'])
df_train['year_cos'] = np.cos(2 * np.pi * df_train['year'])
df_train['month_sin'] = np.sin(2 * np.pi * df_train['month'] / 12)
df_train['month_cos'] = np.cos(2 * np.pi * df_train['month'] / 12)


df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['dayOfMonth'] = df_test['date'].dt.day
df_test['dayOfYear'] = df_test['date'].dt.dayofyear
df_test['weekday'] = df_test['date'].dt.weekday
df_test['year_sin'] = np.sin(2 * np.pi * df_test['year'])
df_test['year_cos'] = np.cos(2 * np.pi * df_test['year'])
df_test['month_sin'] = np.sin(2 * np.pi * df_test['month'] / 12)
df_test['month_cos'] = np.cos(2 * np.pi * df_test['month'] / 12)


print('Duration (Train Date): ', df_train['date'].min(), df_train['date'].max())
print('Duration (Test Date): ', df_test['date'].min(), df_test['date'].max())
print('(Train) Days: ',(df_train['date'].max() - df_train['date'].min()))
print('(Test) Days: ',(df_test['date'].max() - df_test['date'].min()))


df_train.describe().T


df_train.isna().sum()


df_test.isna().sum()


df_train = df_train.dropna()


print ("Unique values are:\n",df_train.nunique())


fig = go.Figure()
fig.add_trace(go.Scatter(x =[0, 1, 2, 3, 4], y =[1.6, 1.6, 1.6, 1.6, 1.6], mode="text", 
    text=["<span style='font-size:28px;color:#ffbe0b;'><b>3</b></span>", 
          "<span style='font-size:28px;color:#fb5607;'><b>5</b></span>",
          "<span style='font-size:28px;color:#ff006e;'><b>6</b></span>",
          "<span style='font-size:28px;color:#8338ec;'><b>2556</b></span>",
          "<span style='font-size:28px;color:#3a86ff;'><b>1094</b></span>"], textposition ="bottom center"))

fig.add_trace(go.Scatter(x=[0, 1, 2, 3, 4],y=[0.7, 0.7, 0.7, 0.7, 0.7], mode="text", 
    text=["Stores", "Products", "Countries", "(Train) Days","(Test) Days"], textposition ="bottom center"))

fig.add_hline(y = 2.2, line_width = 5, line_color ='gray')
fig.add_hline(y = 0.1, line_width = 3, line_color ='gray')
fig.update_yaxes(visible = False)
fig.update_xaxes(visible = False)
fig.update_layout(showlegend = False, height = 300, width = 970, title ='<b>Kaggle Merchandise Summary</b>', 
                  title_x = 0.5, title_y = 0.8, xaxis_range = [-0.5,5], yaxis_range = [-0.2,2.2],
                  plot_bgcolor ='#FDFCF6', paper_bgcolor ='#FDFCF6',
                  font = dict(size = 18, color ='#323232'),
                  title_font = dict(size = 25, family ='Verdana',color ='#03045e'))


kr = df_train[df_train.store == 'Premium Sticker Mart'].groupby(['date','store']).agg(num_sold =('num_sold','sum')).reset_index()
km = df_train[df_train.store == 'Stickers for Less'].groupby(['date','store']).agg(num_sold = ('num_sold','sum')).reset_index()
kt = df_train[df_train.store == 'Discount Stickers'].groupby(['date','store']).agg(num_sold = ('num_sold','sum')).reset_index()
sold_product = df_train.groupby('product').agg(num_sold =('num_sold','sum')).reset_index().sort_values(by ='num_sold', ascending=False)
sold_country = df_train.groupby('country').agg(num_sold =('num_sold','sum')).reset_index().sort_values(by ='num_sold', ascending=False)


fig = make_subplots(rows = 2, cols = 2, specs =[[{"type": "bar"}, {"type": "pie"}], [{"colspan": 2}, None]],
                    column_widths = [0.75, 0.25], vertical_spacing = 0.05, horizontal_spacing = 0.22,
                    subplot_titles =("Total Sales per Country", "Total Product Sales", "Sales Trend"))

fig.add_trace(go.Bar(x = sold_country['num_sold'], y = sold_country['country'], 
                     marker = dict(color=["#CC3749","#FF7372","#01C4ED","#00D97C",'#E9DE34',"#bdeaee"]),
                     name ='Country', orientation ='h'), row = 1, col = 1)

fig.add_trace(go.Pie(values = sold_product['num_sold'], labels = sold_product['product'], name ='Product',
                     marker = dict(colors=["#CC3749","#FF7372","#01C4ED","#00D97C",'#E9DE34']), hole = 0.5,
                     hoverinfo ='label+percent+value', textinfo ='label'), row = 1, col = 2)

fig.update_traces(row = 1, col = 2)
fig.add_trace(go.Scatter(x = kr['date'],y = kr.num_sold, mode ='lines', name ='Premium Sticker Mart',
                         marker = dict(color ="#FF7372")),row = 2, col = 1)
fig.add_trace(go.Scatter(x = km['date'],y = km.num_sold, mode ='lines', name ='Stickers for Less', 
                         marker = dict(color="#01C4ED")),row = 2, col = 1)
fig.add_trace(go.Scatter(x = kt['date'],y = kt.num_sold, mode ='lines', name ='Discount Stickers', 
                         marker = dict(color="#E9DE34")),row = 2, col = 1)
fig.update_xaxes(showgrid = False, row = 1, col = 1)
fig.update_yaxes(showgrid = False, categoryorder ='total ascending', row = 1, col = 1)

fig.update_yaxes(showgrid = True,categoryorder ='total ascending', linewidth = 2, row = 2, col = 1)
fig.update_xaxes(showgrid = False,categoryorder ='total ascending', linewidth = 2, row = 2, col = 1)

fig.update_xaxes(visible = False, row = 1, col = 1)
fig.update_layout(height = 800, font_color ='#28221D', bargap = 0.2, 
                  title_text ="<b>Kaggle Merchandise Analysis</b>",paper_bgcolor ="#FDFCF6", 
                  plot_bgcolor = "#FDFCF6", title_font = dict(size = 20, family ='Verdana',color ='#003566'),
                  font = dict(color ='black'), hoverlabel = dict(bgcolor ="black"), showlegend = False)

fig.update_layout(legend = dict(orientation ="h", yanchor ="top", y = 1.133, xanchor ="right", x = 1))
fig.update_layout(shapes = [dict(type ="line", xref='paper',yref ='paper',
                                x0 = -0.045, y0 = 1.055, x1 = 0.509, y1 = 1.055)])
fig.show()


df_train.store.unique()


fig = go.Figure()
kr = df_train[df_train.store =='Discount Stickers'].groupby(['store','product']).agg(num_sold =('num_sold','sum')).reset_index()
km = df_train[df_train.store =='Stickers for Less'].groupby(['store','product']).agg(num_sold =('num_sold','sum')).reset_index()
kt = df_train[df_train.store =='Premium Sticker Mart'].groupby(['store','product']).agg(num_sold =('num_sold','sum')).reset_index()
fig.add_trace(go.Bar(x = kr['product'], y = kr["num_sold"], name ='Discount Stickers',
    marker = dict(color=["#FF7372","#FF7372","#FF7372","#FF7372","#FF7372"]))) 
fig.add_trace(go.Bar(x = km['product'], y = km["num_sold"], name = 'Stickers for Less',
    marker = dict(color =["#01C4ED","#01C4ED","#01C4ED","#01C4ED","#01C4ED"])))
fig.add_trace(go.Bar(x = kt['product'], y = kt["num_sold"], name = 'Premium Sticker Mart',
    marker = dict(color =['#E9DE34','#E9DE34','#E9DE34','#E9DE34','#E9DE34'])))
fig.update_layout(height = 600, width = 900, bargap = 0.1, xaxis = dict(tickmode ='linear'),
                  title_text ="<b>Product Sales Analysis</b>",paper_bgcolor ="#FDFCF6",plot_bgcolor ="#FDFCF6",
                  title_font = dict(size = 20, family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line", xref ='paper', yref ='paper',
                                 x0 = -0.06, y0 = 1.09, x1 = 0.405, y1 = 1.09)])
fig.show()


df_train.country.value_counts()


country1 = df_train[df_train['country'] == 'Canada']
country2 = df_train[df_train['country'] == 'Finland']
country3 = df_train[df_train['country'] == 'Italy']
country4 = df_train[df_train['country'] == 'Kenya']
country5 = df_train[df_train['country'] == 'Norway']
country6 = df_train[df_train['country'] == 'Singapore']

fig = make_subplots(rows = 2, cols = 3, specs = [[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],
         [{'type':'domain'},{'type':'domain'},{'type':'domain'}]],  
                    vertical_spacing = 0.04, horizontal_spacing = 0.04,
                    subplot_titles =("Canada","Finland","Italy","Kenya","Norway","Singapore"))

fig.add_trace(go.Pie(labels=country1['product'],values=country1['num_sold']), 1,1)
fig.add_trace(go.Pie(labels=country2['product'],values=country2['num_sold']), 1,2)
fig.add_trace(go.Pie(labels=country3['product'],values=country3['num_sold']), 1,3)
fig.add_trace(go.Pie(labels=country4['product'],values=country4['num_sold']), 2,1)
fig.add_trace(go.Pie(labels=country5['product'],values=country5['num_sold']), 2,2)
fig.add_trace(go.Pie(labels=country6['product'],values=country6['num_sold']), 2,3)

fig.update_traces(marker = dict(colors =["#FF7372","#01C4ED","#00D97C",'#E9DE34',"#CC3749"]),hole=0.5)
fig.update_layout(height = 600, width = 900, bargap = 0.1, xaxis = dict(tickmode ='linear'),
                  title_text ="<b>Product Sales by Country</b>",paper_bgcolor ="#FDFCF6",
                  plot_bgcolor ="#FDFCF6",
                  title_font = dict(size = 20, family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line", xref ='paper', yref ='paper',
                                 x0 = -0.07, y0 = 1.09, x1 = 0.435, y1 = 1.09)])

fig.show()


fig = go.Figure()
kr_2 = df_train[df_train.store =='Discount Stickers'].groupby(['date','store','product']).agg(num_sold =('num_sold','sum')).reset_index()
kr_bk1 = kr_2[kr_2['product'] == 'Holographic Goose']
kr_bk2 = kr_2[kr_2['product'] == 'Kaggle']
kr_bk3 = kr_2[kr_2['product'] == 'Kaggle Tiers']
kr_bk4 = kr_2[kr_2['product'] == 'Kerneler']
kr_bk5 = kr_2[kr_2['product'] == 'Kerneler Dark Mode']

fig.add_trace(go.Scatter(x = kr_bk1['date'],y = kr_bk1.num_sold, mode='lines', name ='Holographic Goose',
                         marker=dict(color="#CC3749")))              
fig.add_trace(go.Scatter(x = kr_bk2['date'],y = kr_bk2.num_sold, mode='lines', name ='Kaggle',
                         marker=dict(color="#FF7372")))              
fig.add_trace(go.Scatter(x = kr_bk3['date'],y = kr_bk3.num_sold, mode='lines', name ='Kaggle Tiers',
                         marker=dict(color="#01C4ED")))  
fig.add_trace(go.Scatter(x = kr_bk4['date'],y = kr_bk4.num_sold, mode='lines', name ='Kerneler',
                         marker=dict(color="#00D97C"))) 
fig.add_trace(go.Scatter(x = kr_bk5['date'],y = kr_bk5.num_sold, mode='lines', name ='Kerneler Dark Mode',
                         marker=dict(color="#E9DE34"))) 

fig.update_yaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_xaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_layout(height = 450, width = 900, title_text ="<b>Discount Stickers Sales</b>",
                  paper_bgcolor ="#FDFCF6",plot_bgcolor = "#FDFCF6", 
                  title_font = dict(size=20,family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line",xref ='paper',yref ='paper',
                                 x0 = -0.06, y0 = 1.14, x1 = 0.385, y1 = 1.14)])
fig.show()


fig = go.Figure()
kr_2 = df_train[df_train.store =='Stickers for Less'].groupby(['date','store','product']).agg(num_sold =('num_sold','sum')).reset_index()
kr_bk1 = kr_2[kr_2['product'] == 'Holographic Goose']
kr_bk2 = kr_2[kr_2['product'] == 'Kaggle']
kr_bk3 = kr_2[kr_2['product'] == 'Kaggle Tiers']
kr_bk4 = kr_2[kr_2['product'] == 'Kerneler']
kr_bk5 = kr_2[kr_2['product'] == 'Kerneler Dark Mode']

fig.add_trace(go.Scatter(x = kr_bk1['date'],y = kr_bk1.num_sold, mode='lines', name ='Holographic Goose',
                         marker=dict(color="#CC3749")))              
fig.add_trace(go.Scatter(x = kr_bk2['date'],y = kr_bk2.num_sold, mode='lines', name ='Kaggle',
                         marker=dict(color="#FF7372")))              
fig.add_trace(go.Scatter(x = kr_bk3['date'],y = kr_bk3.num_sold, mode='lines', name ='Kaggle Tiers',
                         marker=dict(color="#01C4ED")))  
fig.add_trace(go.Scatter(x = kr_bk4['date'],y = kr_bk4.num_sold, mode='lines', name ='Kerneler',
                         marker=dict(color="#00D97C"))) 
fig.add_trace(go.Scatter(x = kr_bk5['date'],y = kr_bk5.num_sold, mode='lines', name ='Kerneler Dark Mode',
                         marker=dict(color="#E9DE34"))) 

fig.update_yaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_xaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_layout(height = 450, width = 900, title_text ="<b>Stickers for Less Sales</b>",
                  paper_bgcolor ="#FDFCF6",plot_bgcolor = "#FDFCF6", 
                  title_font = dict(size=20,family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line",xref ='paper',yref ='paper',
                                 x0 = -0.06, y0 = 1.14, x1 = 0.365, y1 = 1.14)])
fig.show()


fig = go.Figure()
kr_2 = df_train[df_train.store =='Premium Sticker Mart'].groupby(['date','store','product']).agg(num_sold =('num_sold','sum')).reset_index()
kr_bk1 = kr_2[kr_2['product'] == 'Holographic Goose']
kr_bk2 = kr_2[kr_2['product'] == 'Kaggle']
kr_bk3 = kr_2[kr_2['product'] == 'Kaggle Tiers']
kr_bk4 = kr_2[kr_2['product'] == 'Kerneler']
kr_bk5 = kr_2[kr_2['product'] == 'Kerneler Dark Mode']

fig.add_trace(go.Scatter(x = kr_bk1['date'],y = kr_bk1.num_sold, mode='lines', name ='Holographic Goose',
                         marker=dict(color="#CC3749")))              
fig.add_trace(go.Scatter(x = kr_bk2['date'],y = kr_bk2.num_sold, mode='lines', name ='Kaggle',
                         marker=dict(color="#FF7372")))              
fig.add_trace(go.Scatter(x = kr_bk3['date'],y = kr_bk3.num_sold, mode='lines', name ='Kaggle Tiers',
                         marker=dict(color="#01C4ED")))  
fig.add_trace(go.Scatter(x = kr_bk4['date'],y = kr_bk4.num_sold, mode='lines', name ='Kerneler',
                         marker=dict(color="#00D97C"))) 
fig.add_trace(go.Scatter(x = kr_bk5['date'],y = kr_bk5.num_sold, mode='lines', name ='Kerneler Dark Mode',
                         marker=dict(color="#E9DE34"))) 

fig.update_yaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_xaxes(showgrid = True, gridwidth = 0, categoryorder ='total ascending')
fig.update_layout(height = 450, width = 900, title_text ="<b>Premium Sticker Mart Sales</b>",
                  paper_bgcolor ="#FDFCF6",plot_bgcolor = "#FDFCF6", 
                  title_font = dict(size=20,family ='Verdana', color ='#003566'),
                  hoverlabel = dict(font_size = 13))
fig.update_layout(shapes = [dict(type ="line",xref ='paper',yref ='paper',
                                 x0 = -0.06, y0 = 1.14, x1 = 0.465, y1 = 1.14)])
fig.show()


df_train['num_sold'] = np.log1p(df_train['num_sold'])
df_train = df_train.drop(labels=['date', 'id'], axis=1)
df_test = df_test.drop(labels=['date', 'id'], axis=1)

num_features = list(set(df_train.select_dtypes(exclude=['object']).columns) - {'num_sold'})
cat_features = list(df_train.select_dtypes(include=['object']).columns)

test_numeric_cols = list(set(df_test.select_dtypes(exclude=['object']).columns) - {'id'})
test_categorical_cols = list(df_test.select_dtypes(include=['object']).columns)


combined_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)


for feature in cat_features:
    combined_data[feature], _ = pd.factorize(combined_data[feature])
    combined_data[feature] -= combined_data[feature].min()
    combined_data[feature] = combined_data[feature].astype('int32').astype('category')

for feature in num_features:
    if combined_data[feature].dtype == 'float64':
        combined_data[feature] = combined_data[feature].astype('float32')
    elif combined_data[feature].dtype == 'int64':
        combined_data[feature] = combined_data[feature].astype('int32')


df_train = combined_data.iloc[:len(df_train)].reset_index(drop=True)
df_test = combined_data.iloc[len(df_train):].reset_index(drop=True).drop(columns='num_sold', axis=1)


X= df_train.drop(columns=['num_sold'])
y = df_train['num_sold']


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 250, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 80),
        'subsample': trial.suggest_float('subsample', 0.35, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0.005, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.005, 0.8),
        'enable_categorical': True,
        'n_jobs': -1,
        'random_state': 42,
        'device': 'gpu'
    }
    
    model = XGBRegressor(**params)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=0
    )
    
    y_predicted = model.predict(X_valid)
    mape_score = mean_absolute_percentage_error(y_valid, y_predicted)
    
    return mape_score


study_xgb = optuna.create_study(study_name = "Forecasting_Sales_XGBoost", direction='minimize')
study_xgb.optimize(objective, n_trials=50, show_progress_bar=True)


best_params = study_xgb.best_params
best_params.update({'enable_categorical': True,'device': 'gpu'})


kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=123)  
cross_val_scores, test_predictions = [], []

for fold_num, (train_idx, valid_idx) in enumerate(kfold_splitter.split(df_train)):
    print(f'Processing Fold {fold_num + 1}')
    train_X, valid_X = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]

    regressor = XGBRegressor(**best_params)
    regressor.fit(
        train_X, train_y,
        eval_set=[(valid_X, valid_y)],
        verbose=200
    )

    predictions = regressor.predict(valid_X)
    fold_mape = mean_absolute_percentage_error(valid_y, predictions)
    print(f'MAPE for Fold {fold_num + 1}: {fold_mape}')
    cross_val_scores.append(fold_mape)
    test_predictions.append(regressor.predict(df_test))
    average_mape = np.mean(cross_val_scores)
    print(f"Average MAPE: {average_mape:.4f}")


df_subm['num_sold'] = np.expm1(regressor.predict(df_test))
df_subm.to_csv('submission.csv', index=False)
df_subm

