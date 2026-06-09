import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold , TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score,accuracy_score, roc_curve, auc
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from itertools import cycle
import warnings, gc
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))

#Custom Color Palette ðŸŽ¨
custom_colors = ["#70d6ff","#ff4d6d","#8338ec","#90cf8e","#ffd670"]
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors),size=1.2)
plt.tick_params(axis='both', labelsize=0, length = 0)


df_train = pd.read_feather('../input/amexfeather/train_data.ftr')
df_test = pd.read_feather('../input/amexfeather/test_data.ftr')


num_rows = df_train.shape[0]
num_features = df_train.shape[1] 

numerical_cols = df_train.select_dtypes(include=["number"]).columns
numerical_cols = numerical_cols.tolist()
numerical_cols.remove("target")
categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns
categorical_cols = categorical_cols.tolist()
categorical_cols.remove("customer_ID")
datetime_cols = df_train.select_dtypes(include=['datetime64']).columns


print(f"Number of rows: {num_rows}")
print(f"Number of features: {num_features}, including the customer_ID")
print(f"Number of numerical columns: {len(numerical_cols)}")
print(f"Number of date columns: {len(datetime_cols)}")
print(f"Number of categorical columns: {len(categorical_cols)}")

print(f"\nNumerical columns: {numerical_cols}")
print(f"Date columns: {datetime_cols}")
print(f"Categorical columns: {categorical_cols}")


df_train.shape


df_test.shape


num_rows = df_test.shape[0]
num_features = df_test.shape[1]

numerical_cols = df_test.select_dtypes(include=["number"]).columns
numerical_cols = numerical_cols.tolist()


categorical_cols = df_test.select_dtypes(include=['object', 'category']).columns
categorical_cols = categorical_cols.tolist()
categorical_cols.remove("customer_ID")
datetime_cols = df_train.select_dtypes(include=['datetime64']).columns


print(f"Number of rows: {num_rows}")
print(f"Number of features: {num_features}, including the customer_ID")
print(f"Number of numerical columns: {len(numerical_cols)}")
print(f"Number of date columns: {len(datetime_cols)}")
print(f"Number of categorical columns: {len(categorical_cols)}")

print(f"\nNumerical columns: {numerical_cols}")
print(f"Date columns: {datetime_cols}")
print(f"Categorical columns: {categorical_cols}")


df_train['S_2'] = pd.to_datetime(df_train['S_2'])
df_test['S_2'] = pd.to_datetime(df_test['S_2'])

# Find the earliest and latest dates
start_date_train = df_train['S_2'].min()
end_date_train = df_train['S_2'].max()
start_date_test = df_test['S_2'].min()
end_date_test = df_test['S_2'].max()


# # Print the result
print(f"The train data exists from {start_date_train.strftime('%Y-%m-%d')} to {end_date_train.strftime('%Y-%m-%d')}.")
print(f"The test data exists from {start_date_test.strftime('%Y-%m-%d')} to {end_date_test.strftime('%Y-%m-%d')}.")


plt.figure(figsize=(16, 5))
sns.histplot(data=df_train, x="S_2", bins=100)
plt.title("Distribution of statements by time for train data", fontsize=16)
plt.xlabel("count", fontsize=14)
plt.ylabel("n_records", fontsize=14);


df_test['S_2'] = pd.to_datetime(df_test['S_2'])
plt.figure(figsize=(16, 5))
sns.histplot(data=df_test, x="S_2", bins = 100)
plt.title("Distribution of statements by time for test data", fontsize=16)
plt.xlabel("count", fontsize=14)
plt.ylabel("n_records", fontsize=14);


customer_presence = df_train.groupby(['customer_ID','target']).size().reset_index().rename(columns={0:'presence'})
customer_presence["presence"].value_counts()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
train_sc = df_train.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Train statements per customer')
ax1.pie(train_sc, labels=train_sc.index)
ax1.set_title(train_sc.name)
test_sc = df_test.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Test statements per customer')
ax2.pie(test_sc, labels=test_sc.index)
ax2.set_title(test_sc.name)
plt.show()


# convert S_2 column into datetime and sort the dataframe by customer_ID and date (S_2); selecting latest statement
df_train['S_2'] = pd.to_datetime(df_train['S_2'])
df_train = df_train.sort_values(['customer_ID', 'S_2'])
df_train = df_train.sort_values('S_2').groupby('customer_ID').tail(1)

# Group by customer_ID and apply custom logic of slecting earliest default row per customer or otherwise latest row/statement
# def select_row(group):
#     # Check if there's any row where target == 1
#     first_target_1 = group[group['target'] == 1]
#     if not first_target_1.empty:
#         return first_target_1.iloc[0]  # Select the first row with target == 1
#     else:
#         return group.iloc[-1]  # Otherwise, select the last row in the group

# df_train = df_train.groupby('customer_ID').apply(select_row).reset_index(drop=True)




df_test = df_test.sort_values(['customer_ID', 'S_2'])
df_test['S_2'] = pd.to_datetime(df_test['S_2'])
df_test = df_test.sort_values('S_2').groupby('customer_ID').tail(1)


df_train.replace([np.inf, -np.inf], np.nan, inplace=True)


df_train['D_66'] = df_train['D_66'].fillna(0).astype('category') # converting nan values to 0 since there are only two value counts (1 and nan)


tmp = df_train.isna().sum().div(len(df_train)).mul(100).sort_values(ascending=False)


fig, ax = plt.subplots(2,1, figsize=(25,10))
sns.barplot(x=tmp[:100].index, y=tmp[:100].values, ax=ax[0])
sns.barplot(x=tmp[100:].index, y=tmp[100:].values, ax=ax[1])
ax[0].set_ylabel("Percentage [%]"), ax[1].set_ylabel("Percentage [%]")
ax[0].tick_params(axis='x', rotation=90); ax[1].tick_params(axis='x', rotation=90)
plt.suptitle("Amount of missing data")
plt.tight_layout()
plt.show()


null_columns_to_drop = tmp[tmp>80].index.tolist()
df_train_adc = df_train.drop(null_columns_to_drop, axis =1)
df_test_adc = df_test.drop(null_columns_to_drop, axis =1)



D_columns = df_train_adc.filter(like='D_', axis=1).columns.tolist()
B_columns = df_train_adc.filter(like='B_', axis=1).columns.tolist()
S_columns = df_train_adc.filter(like='S_', axis=1).columns.tolist()
P_columns = df_train_adc.filter(like='P_', axis=1).columns.tolist()
R_columns = df_train_adc.filter(like='R_', axis=1).columns.tolist()


labels=['Delinquency', 'Spend','Payment','Balance','Risk']
values= [len(D_columns), len(S_columns),len(P_columns), len(B_columns),len(R_columns)]
fig_1 = go.Figure()
fig_1.add_trace(go.Pie(values = values,labels = labels,hole = 0.6, 
                     hoverinfo ='label+percent'))
fig_1.update_traces(textfont_size = 12, hoverinfo ='label+percent',textinfo ='label', 
                  showlegend = False,marker = dict(colors =["#70d6ff","#ff9770"]),
                  title = dict(text = 'Feature Distribution'))  
fig_1.show()


target_class = pd.DataFrame({'count': df_train_adc.target.value_counts(),
                             'percentage': df_train_adc['target'].value_counts() / df_train_adc.shape[0] * 100
})


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Pie(values = target_class['count'],labels = target_class.index,hole = 0.6, 
                     hoverinfo ='label+percent'))
fig.update_traces(textfont_size = 12, hoverinfo ='label+percent',textinfo ='label', 
                  showlegend = False,marker = dict(colors =["#90cf8e","#ff70a6"]),
                  title = dict(text = 'Target Distribution'))  
fig.show()


stat_plot = df_train_adc.reset_index().groupby('S_2')['customer_ID'].nunique().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x = stat_plot['S_2'], y = stat_plot['customer_ID']))
fig.update_layout(title="Customer Statements", width = 800, height = 600,xaxis_title ='Statement Date',
                  paper_bgcolor='rgb(0,0,0,0)',plot_bgcolor='rgb(0,0,0,0)') 
fig['data'][0]['line']['color']="#ff9770"
fig.show()


del_cols = [c for c in df_train_adc.columns if (c.startswith(('D','t'))) & (c not in categorical_cols)]
df_del = df_train_adc[del_cols]
spd_cols = [c for c in df_train_adc.columns if (c.startswith(('S','t'))) & (c not in categorical_cols)]
df_spd = df_train_adc[spd_cols]
pay_cols = [c for c in df_train_adc.columns if (c.startswith(('P','t'))) & (c not in categorical_cols)]
df_pay = df_train_adc[pay_cols]
bal_cols = [c for c in df_train_adc.columns if (c.startswith(('B','t'))) & (c not in categorical_cols)]
df_bal = df_train_adc[bal_cols]
ris_cols = [c for c in df_train_adc.columns if (c.startswith(('R','t'))) & (c not in categorical_cols)]
df_ris = df_train_adc[ris_cols]


fig, axes = plt.subplots(29, 3, figsize = (35,150))
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(del_cols) - 1:
        sns.kdeplot(x = del_cols[i], hue='target', data = df_del, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
fig.suptitle('Distribution of Delinquency Variables', fontsize = 35, x = 0.5, y = 1)
plt.tight_layout()
plt.show()


plt.figure(figsize =(11,11))
corr = df_del.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, mask = mask, robust = True, center = 0,square = True, linewidths =.6, cmap = custom_colors)
plt.title('Correlation of Delinquency Variables')
plt.show()


high_pos_corr = corr[corr > 0.9]

# Stack the correlation matrix and reset the index
stacked_corr = (
    high_pos_corr.stack()
    .reset_index()
    .rename(columns={0: 'correlation'})
)

# Ensure unique pairs by keeping only where level_0 < level_1
stacked_corr = stacked_corr[stacked_corr['level_0'] < stacked_corr['level_1']]

# Sort by correlation value
stacked_corr = stacked_corr.sort_values(by='correlation', ascending=False)

print(stacked_corr)


# high_pos_corr = corr[corr < -0.9]

# # Stack the correlation matrix and reset the index
# stacked_corr = (
#     high_pos_corr.stack()
#     .reset_index()
#     .rename(columns={0: 'correlation'})
# )

# # Ensure unique pairs by keeping only where level_0 < level_1
# stacked_corr = stacked_corr[stacked_corr['level_0'] < stacked_corr['level_1']]

# # Sort by correlation value
# stacked_corr = stacked_corr.sort_values(by='correlation', ascending=False)

# print(stacked_corr)


#columns to drop to avoid multicollinearity
D_drop_columns = ["D_62", "D_143", "D_141", "D_103", "D_118", "D_74", "D_58"]


fig, axes = plt.subplots(8, 3, figsize = (16,18))
fig.suptitle('Distribution of Spend Variables', fontsize = 15, x = 0.5, y = 1)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(spd_cols) - 1:
        sns.kdeplot(x = spd_cols[i], hue ='target', data = df_spd, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize = (11,11))
corr = df_spd.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask = mask, robust = True, center = 0,square = True, linewidths = .6, cmap = custom_colors)
plt.title('Correlation of Spend Variables')
plt.show()


fig, axes = plt.subplots(1, 3, figsize = (12,4))
fig.suptitle('Distribution of Payment Variables',fontsize = 15)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(pay_cols) - 1:
        sns.kdeplot(x = pay_cols[i], hue ='target', data = df_pay, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize = (6,6))
corr = df_pay.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, mask = mask, robust = True, center = 0,square = True, linewidths = .6, cmap = custom_colors)
plt.title('Correlation of Payment Variables')
plt.show()


fig, axes = plt.subplots(10, 4, figsize = (15,24))
fig.suptitle('Distribution of Balance Variables',fontsize = 15, x = 0.5, y = 1)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(bal_cols) - 1:
        sns.kdeplot(x = bal_cols[i], hue ='target', data = df_bal, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize = (11,11))
corr = df_bal.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, mask = mask, robust=True, center = 0,square = True, linewidths =.6, cmap = custom_colors)
plt.title('Correlation of Balance Variables')
plt.show()


high_pos_corr = corr[corr > 0.9]

# Stack the correlation matrix and reset the index
stacked_corr = (
    high_pos_corr.stack()
    .reset_index()
    .rename(columns={0: 'correlation'})
)

# Ensure unique pairs by keeping only where level_0 < level_1
stacked_corr = stacked_corr[stacked_corr['level_0'] < stacked_corr['level_1']]

# Sort by correlation value
stacked_corr = stacked_corr.sort_values(by='correlation', ascending=False)

print(stacked_corr)


B_drop_columns = ["B_11", "B_13", 'B_23', 'B_1' ,'B_2', "B_14"]


df_train_adc


fig = make_subplots(rows=4, cols=3, 
                    subplot_titles=categorical_cols[:-1], 
                    vertical_spacing=0.1)
pal=['#016CC9','#DEB078']
row=0
c=[1,2,3]*5
plot_df= df_train_adc[['D_63', 'D_64', 'D_68', 'B_30','D_66', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'target']]
for i,col in enumerate(categorical_cols[:-1]):
    if i%3==0:
        row+=1
    plot_df[col]=plot_df[col].astype(object)
    df=plot_df.groupby(col)['target'].value_counts().rename('count').reset_index().replace('',np.nan)
    
    fig.add_trace(go.Bar(x=df[df.target==1][col], y=df[df.target==1]['count'],
                          marker_line=dict(color=pal[1],width=2), 
                         hovertemplate='Value %{x} Frequency = %{y}',
                         name='Default', showlegend=(True if i==0 else False)),
                  row=row, col=c[i])
    fig.add_trace(go.Bar(x=df[df.target==0][col], y=df[df.target==0]['count'],
                          marker_line=dict(color=pal[0],width=2),
                         hovertemplate='Value %{x} Frequency = %{y}',
                         name='Paid', showlegend=(True if i==0 else False)),
                  row=row, col=c[i])
    if i%3==0:
        fig.update_yaxes(title='Frequency',row=row,col=c[i])
fig.update_layout(template=temp,title="Distribution of Categorical Variables",
                  legend=dict(orientation="h",yanchor="bottom",y=1.03,xanchor="right",x=0.2),
                  barmode='group',height=1500,width=900)
fig.show()


palette = cycle(["#ffd670","#70d6ff","#ff4d6d","#8338ec","#90cf8e"])
targ = df_train_adc.drop(columns = ["customer_ID", 'D_63','D_64']).corrwith(df_train_adc['target'], axis=0)
val = [str(round(v ,1) *100) + '%' for v in targ.values]
fig = go.Figure()
fig.add_trace(go.Bar(y=targ.index, x= targ.values, orientation='h',text = val, marker_color = next(palette)))
fig.update_layout(title = "Correlation of variables with Target",width = 750, height = 3500,
                  paper_bgcolor='rgb(0,0,0,0)',plot_bgcolor='rgb(0,0,0,0)')


tmp_corr = df_train_adc.drop(columns = ["customer_ID", 'D_63','D_64']).corrwith(df_train_adc['target'], axis=0)


indexes = tmp_corr[abs(tmp_corr) > 0.5].index
indexes


df_train_adc = df_train_adc.drop(D_drop_columns, axis =1)
df_test_adc = df_test_adc.drop(D_drop_columns, axis =1)

df_train_adc = df_train_adc.drop(B_drop_columns, axis =1)
df_test_adc = df_test_adc.drop(B_drop_columns, axis =1)


df_train_adc.set_index("customer_ID", inplace=True)
df_test_adc.set_index("customer_ID", inplace=True)


numerical_cols = df_train_adc.select_dtypes(include=["number"]).columns
numerical_cols = numerical_cols.tolist()
numerical_cols.remove("target")



num_imputer = SimpleImputer(strategy='mean') #can also use median to better handle outliers
cat_imputer = SimpleImputer(strategy='most_frequent')

df_train_adc[numerical_cols] = num_imputer.fit_transform(df_train_adc[numerical_cols])
df_train_adc[categorical_cols] = cat_imputer.fit_transform(df_train_adc[categorical_cols])

df_test_adc[numerical_cols] = num_imputer.transform(df_test_adc[numerical_cols])
df_test_adc[categorical_cols] = cat_imputer.transform(df_test_adc[categorical_cols])


label_encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df_train_adc[col] = le.fit_transform(df_train_adc[col])
    label_encoders[col] = le
for col in categorical_cols:
    le = label_encoders[col]  # Retrieve the trained LabelEncoder
    df_test_adc[col] = le.transform(df_test_adc[col])  # Transform the test column


df_train_adc.reset_index(inplace = True)
df_test_adc.reset_index(inplace = True)


train_data = df_train_adc.drop(columns=['customer_ID','target',"S_2"])
test_data = df_test_adc.drop(columns=['customer_ID','S_2'])


# Function to cap/floor outliers using IQR
def cap_outliers_iqr(df):
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)  # First quartile
        Q3 = df[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1                # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)  # Capping outliers
    return df

# Apply the function
train_data_cleaned = cap_outliers_iqr(train_data)
print(train_data_cleaned)


# Step 1: standardize the data (importanct for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(train_data)

# Step 2: perform default PCA without n_components parameter
pca = PCA()
pca.fit(data_scaled)

# Step 3: Compute cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Step 4: Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')  # Optional
plt.legend()
plt.show()


optimal_components = np.argmax(cumulative_variance >= 0.9) + 1
print(f"Optimal number of components: {optimal_components}")


pca = PCA(n_components=optimal_components)
pca_result_train = pca.fit_transform(train_data)
pca_result_test = pca.transform(test_data)

df_pca_train = pd.DataFrame(pca_result_train)
df_pca_test = pd.DataFrame(pca_result_test)


train_data['target']  = df_train_adc['target'].values
df_pca_train['target'] = df_train_adc['target'].values


# Prepare features and target
X = train_data.drop('target', axis=1)
y = train_data['target']

print('Feature count:', len(X.columns))
print('X shape:', X.shape)
print('y shape:', y.shape)


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def plot_roc(y_val,y_prob):
    colors=px.colors.qualitative.Prism
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11),
                             name='Random Chance',mode='lines', showlegend=False,
                             line=dict(color="Black", width=1, dash="dot")))
    for i in range(len(y_val)):
        y=y_val[i]
        prob=y_prob[i]
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr,tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=colors[::-1][i+1], width=3),
                                 hovertemplate = 'True positive rate = %{y:.3f}<br>False positive rate = %{x:.3f}',
                                 name='Fold {}:  Gini = {:.3f}, AUC = {:.3f}'.format(i+1, gini[i],roc_auc)))
    fig.update_layout( title="Cross-Validation ROC Curves",
                      hovermode="x unified", width=700,height=600,
                      xaxis_title='False Positive Rate (1 - Specificity)',
                      yaxis_title='True Positive Rate (Sensitivity)',
                      legend=dict(orientation='v', y=.07, x=1, xanchor="right",
                                  bordercolor="black", borderwidth=.5))
    fig.show()


# # Define LightGBM model
# model = lgb.LGBMClassifier(
#     objective='binary',
#     early_stopping=50,
#     verbose=-1
# )

# # Parameter grid
# param_grid = {
#     'n_estimators': [250,500,1000],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'num_leaves': [15, 31, 63],
#     'min_child_samples': [1000, 2000, 500]
# }

# # Scoring metric
# scorer = make_scorer(accuracy_score)

# # Define GridSearchCV with PredefinedSplit
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring=scorer,
#     cv=predefined_split,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model
# grid_search.fit(X_train_full, y_train_full)

# # Results
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)


tscv = TimeSeriesSplit(n_splits=5)

params = {'boosting_type': 'gbdt',
              'n_estimators': 1000,
              'num_leaves': 50,
              'learning_rate': 0.05,
              'colsample_bytree': 0.9,
              'min_child_samples': 2000,
              'max_bins': 500,
              'reg_alpha': 2,
              'objective': 'binary',
              "early_stopping_rounds": 200,
              'verbose': -1, 
              'random_state': 21}

accuracy_scores = []

# Iterate through Time Series splits
for split, (train_index, test_index) in enumerate(tscv.split(X)):
    # Split data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"Split {split + 1}: Train size = {len(X_train)}, Test size = {len(X_test)}")

    clf = LGBMClassifier(**params).fit(X_train, y_train,
                                       eval_set=[(X_train, y_train), (X_test, y_test)],
                                                                             eval_metric=['auc','binary_logloss'])
    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    y_pred_prob = clf.predict_proba(X_test)[:,1]
    y_pred=pd.DataFrame(data={'prediction': y_pred_prob})
    y_true_pred = pd.DataFrame(data={'target':y_test.values})
    gini_score=amex_metric(y_true = y_true_pred, y_pred = y_pred)
    print(f"Split {split + 1}: Accuracy = {accuracy:.2f}, Gini = {gini_score:.2f}")


# Print overall results
print("\nAccuracy scores for each split:", accuracy_scores)
print("Mean Accuracy:", np.mean(accuracy_scores))


# submission = pd.DataFrame()
# submission['customer_ID'] = df_test_adc["customer_ID"]
# submission["prediction"] = clf.predict_proba(test_data)[:,1]
# submission.to_csv('submission_tsplit.csv', index=False)


y_valid, gbm_val_probs, gbm_test_preds, gini=[],[],[],[]
ft_importance=pd.DataFrame(index=X.columns)

sk_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

for fold, (train_idx, val_idx) in enumerate(sk_fold.split(X, y)):

    print("\nFold {}".format(fold+1))
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_val, y_val = X.iloc[val_idx,:], y[val_idx]
    print("Train shape: {}, {}, Valid shape: {}, {}\n".format(
        X_train.shape, y_train.shape, X_val.shape, y_val.shape))

    params = {'boosting_type': 'gbdt',
              'n_estimators': 1000,
              'num_leaves': 50,
              'learning_rate': 0.05,
              'colsample_bytree': 0.9,
              'min_child_samples': 2000,
              'max_bins': 500,
              'reg_alpha': 2,
              'objective': 'binary',
              "early_stopping_rounds": 200,
              'verbose': -1,
              'random_state': 21}

    gbm = LGBMClassifier(**params).fit(X_train, y_train,
                                       eval_set=[(X_train, y_train), (X_val, y_val)],
                                                                             eval_metric=['auc','binary_logloss'])
    gbm_prob = gbm.predict_proba(X_val)[:,1]
    gbm_val_probs.append(gbm_prob)
    y_valid.append(y_val)

    y_pred=pd.DataFrame(data={'prediction':gbm_prob})
    y_true=pd.DataFrame(data={'target':y_val.reset_index(drop=True)})
    gini_score=amex_metric(y_true = y_true, y_pred = y_pred)
    gini.append(gini_score)

    auc_score=roc_auc_score(y_val, gbm_prob)
    gbm_test_preds.append(gbm.predict_proba(test_data)[:,1])
    ft_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
    print("Validation Gini: {:.5f}, AUC: {:.4f}".format(gini_score,auc_score))

    del X_train, y_train, X_val, y_val
    _ = gc.collect()

del X, y



submission = pd.DataFrame()
submission['customer_ID'] = df_test_adc["customer_ID"]
submission["prediction"] = gbm.predict_proba(test_data)[:,1]
submission.to_csv('submission_skfold.csv', index=False)


plot_roc(y_valid, gbm_val_probs)


ft_importance['avg'] = ft_importance.mean(axis=1)
ft_importance = ft_importance.avg.nlargest(50).sort_values(ascending=True)

pal=sns.color_palette("YlGnBu", 65).as_hex()
fig=go.Figure()
for i in range(len(ft_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=ft_importance[i],
                       line_color=pal[::-1][i],opacity=0.8,line_width=4))
fig.add_trace(go.Scatter(x=ft_importance, y=ft_importance.index, mode='markers',
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
fig.update_layout(template=temp,title='LGBM Feature Importance<br>Top 50',
                  margin=dict(l=150,t=80),
                  xaxis=dict(title='Importance', zeroline=False),
                  yaxis_showgrid=False, height=1000, width=800)
fig.show()


correlation_matrix = train_data[ft_importance.index.tolist()].corr()


plt.figure(figsize = (11,11))
mask = np.triu(np.ones_like(correlation_matrix, dtype = bool))
sns.heatmap(correlation_matrix, mask = mask, robust=True, center = 0,square = True, linewidths =.6, cmap = custom_colors)
plt.title('Correlation of top 50 features')
plt.show()


import shap
shap.initjs()
from lime.lime_tabular import LimeTabularExplainer
import lime
# Assuming the following variables are defined:
# model: trained machine learning model (e.g., LightGBM)
# X_train: training data (features)
# X_test: test data (features)
# y_test: test data (target)

# -------------------------------
# SHAP (SHapley Additive exPlanations)
# -------------------------------

# 1. Initialize the SHAP explainer
explainer_shap = shap.TreeExplainer(gbm)  # For tree-based models like LightGBM/XGBoost

# 2. Compute SHAP values (using a subset of the data for performance)
X_sample = test_data.sample(100)  # Use a smaller subset for SHAP computation
shap_values = explainer_shap.shap_values(X_sample)

# 3. Global feature importance
shap.summary_plot(shap_values, X_sample)

# 4. Individual prediction explanation
instance_index = 0  # Specify the row index to explain
shap.force_plot(
    explainer_shap.expected_value[1],  # For binary classification (index 1 = class 1)
    shap_values[1][instance_index],   # SHAP values for the selected instance
    test_data.iloc[instance_index]
)

# 5. SHAP dependence plot for a specific feature
shap.dependence_plot('P_2', shap_values[1], X_sample)  # Replace 'P_2' with the desired feature





shap.plots.force(explainer_shap.expected_value[0], shap_values[0])


shap.force_plot(explainer_shap.expected_value[0], shap_values[0][73,:], X_test.iloc[0,:], link="logit")


shap.force_plot(explainer_shap.expected_value[0], shap_values[0][42,:], X_test.iloc[0,:], link="logit")


shap.force_plot(explainer_shap.expected_value[0], shap_values[0], test_data, link="logit")


# -------------------------------
# LIME (Local Interpretable Model-agnostic Explanations)
# -------------------------------

# 1. Initialize the LIME explainer
explainer_lime = LimeTabularExplainer(
    training_data=train_data.drop(columns=["target"]).values,  # Training data (convert to numpy array)
    mode='classification',         # 'classification' for classification tasks
    feature_names=train_data.columns.tolist(), # Feature names
    class_names=['target'],  
    discretize_continuous=True ,   # Discretize continuous variables for interpretability
    verbose=True, 
)

# 2. Explain a single prediction
instance_index = 42  # Specify the row index to explain
instance = test_data.iloc[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=gbm.predict_proba,
    num_features = 10
)

exp.show_in_notebook(show_table=True)


instance_index = 73  # Specify the row index to explain
instance = test_data.iloc[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=gbm.predict_proba,
    num_features=10
)

exp.show_in_notebook(show_table=True)


instance_index = 42  # Specify the row index to explain
instance = test_data.iloc[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=gbm.predict_proba,
    num_features=10
)

exp.show_in_notebook(show_table=True)


instance_index = 2  # Specify the row index to explain
instance = test_data.iloc[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=gbm.predict_proba,
    num_features=10
)

exp.show_in_notebook(show_table=True)




