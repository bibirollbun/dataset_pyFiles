#Import the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from IPython.core.display import display, HTML
from warnings import filterwarnings
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
from tqdm.auto import tqdm
from xgboost import XGBClassifier, callback as xgb_callback
warnings.filterwarnings("ignore", category=FutureWarning)
filterwarnings('ignore')
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df.head(5)


num_records = len(df)

print(f"number of records : {num_records}")


num_columns = len(df.columns)

print(f"number of features : {num_columns-1}")


def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['Duplicate'] = df.duplicated().sum()
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['avg'] = desc['mean'].values
    summ['std dev'] = desc['std'].values
    summ['top value'] = desc['top'].values
    summ['Freq'] = desc['freq'].values
    return summ
summary(df).style.background_gradient()


unique_counts = df.nunique()

html_output = "<br>".join([f"<b style='color:green;'>{col}  :</b> <b style='color:black;'>{count}</b>" for col, count in unique_counts.items()])

display(HTML(html_output))


def univariateAnalysis_category(cols):
    print("Distribution of", cols)
    print("_" * 60)
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
        '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
        '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'  
    ]
    value_counts = cat_columns[cols].value_counts()
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        title=f'Distribution of {cols}',
        labels={'x': 'Categories', 'y': 'Count'},
        color_discrete_sequence=[colors]
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12), 
        title_font=dict(size=30),
        legend_font=dict(color='white', size=12),
        width=500,
        height=400
    )
    fig.show()
    percentage = (value_counts / value_counts.sum()) * 100
    fig = px.pie(
        values=percentage,
        names=value_counts.index,
        labels={'names': 'Categories', 'values': 'Percentage'},
        hole=0.5,
        color_discrete_sequence=colors
    )
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f'{cols}',
        font=dict(size=18, color='white'),
        showarrow=False
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        legend=dict(x=0.9, y=0.5),
        legend_font=dict(color='white', size=12),
        width=500,
        height=400
    )
    fig.show()
    print("       ")
cat_columns = df[['Temparature', 'Crop Type', 'Soil Type', 'Fertilizer Name']]
for x in cat_columns:
    univariateAnalysis_category(x)





colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4',  '#4682B4', 'red',  
        '#7CFC00', '#98FB98', '#9370DB', 
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'  
    ]




numerical_features = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous", 
]

# Groupby function for scatter plot
def groupby(data, x):
    result = data.groupby(x).size().rename('count').reset_index()
    return result

# Histogram plot function
def create_histplot(df, x, color_index=0):
    fig = px.histogram(df, x, nbins=50)
    fig.update_traces(marker_color=colors[color_index % len(colors)])
    fig.update_layout(
        title=f'Histogram of {x}',
        width=560, height=370,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        font=dict(color='white', size=15), 
        title_font=dict(size=25)
    )
    fig.show()

# Scatter plot function
def create_scatter_plot(data, x, y, color, width=550, height=350):  
    fig = px.scatter(data, x=x, y=y, size=y, color_discrete_sequence=[color])
    fig.update_traces(marker=dict(opacity=1))
    fig.update_layout(
        title=f'Scatter of {x}',
        xaxis_title=x,
        yaxis_title=y,
        plot_bgcolor='black',
        width=width,
        paper_bgcolor='black',
        font=dict(color='white', size=15),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=height,
        title_font=dict(size=25)
    )
    fig.show()


for i, feature in enumerate(numerical_features):
    print(f"\n\n==== Plots for {feature} ====\n")
    
    # Histogram
    create_histplot(df, feature, color_index=i)
    
    # Scatter Plot 
    grouped_data = groupby(df, feature)
    create_scatter_plot(grouped_data, feature, 'count', colors[i % len(colors)])



def show_treemap(col):
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
        '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
        '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'  
    ]
    print("\n")
    df_type_series = df.groupby(col)['Potassium'].count().sort_values(ascending=False).head(10)
    treemap_df = df_type_series.reset_index()
    treemap_df.columns = [col, 'count']
    fig = px.treemap(
        treemap_df, 
        path=[col], 
        values='count',
        title=f'Treemap of {col}',
        color_discrete_sequence=colors
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        width=500,
        height=400
    )
    fig.update_traces(
        textinfo="label+value",
        textfont=dict(color='white')
    )
    fig.show()

columns = ['Temparature', 'Soil Type', 'Crop Type', 'Fertilizer Name']
for col in columns:
    show_treemap(col)




def hist(df, x, color):
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#4682B4', 'red',
        '#7CFC00', '#98FB98', '#9370DB',
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
    ]
    fig = px.histogram(
        df,
        x=x,
        color=color,
        barmode='group',
        color_discrete_sequence=colors
    )
    fig.update_layout(
        xaxis_title=x,
        yaxis_title='Count',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=15),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white', showline=False),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='white', showline=False),
        legend_title_text=color,
        legend_font=dict(color='white', size=12),
        width=520,
        height=350,
        title=f"{x} vs {color}",
        title_font=dict(size=21, weight="bold")
    )
    fig.show()

def crosstab_heatmap(df, feature, color):
    cross_tab = pd.crosstab(df[feature], df[color])
    fig = px.imshow(
        cross_tab,
        text_auto=True,
        color_continuous_scale='Tealrose',
        labels=dict(x=color, y=feature, color="Count"),
        aspect="auto"
    )
    fig.update_layout(
        title=f"{feature} vs. {color}",
        xaxis_title=color,
        yaxis_title=feature,
        width=600,
        height=400,
        font=dict(size=14),
    )
    fig.update_xaxes(tickangle=45)
    fig.show()

def plot_all_for_features(df, features, color):
    for feature in features:
        print(f"\nPlotting for feature: {feature}")
        hist(df, feature, color)
        crosstab_heatmap(df, feature, color)


features = ['Soil Type', 'Crop Type', 'Temparature']
color = 'Fertilizer Name'


plot_all_for_features(df, features, color)



import plotly.graph_objects as go

numeric_df = df.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr().round(2)  # round to 2 decimals for readability

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    text=correlation_matrix.values,
    texttemplate="%{text}",
    colorscale='Tealrose',
    zmin=-1, zmax=1,
    colorbar=dict(title="Correlation")
))

fig.update_layout(
    title='Correlation Heatmap of numeric features',
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

fig.show()



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


cat_columns = [i for i in train.columns if train[i].dtype == np.object_]
label_enc = LabelEncoder()
for i in cat_columns[:-1]:
    train[i] = label_enc.fit_transform(train[i])
    test[i] = label_enc.transform(test[i])


train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])


train.head()


test.head()


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


X = train.drop('Fertilizer Name',axis = 1)
y = train["Fertilizer Name"]




FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape=(len(train), y.nunique()))
pred_prob = np.zeros(shape=(len(test), y.nunique()))

class TQDMCallback(xgb_callback.TrainingCallback):
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc, leave=False, ncols=100, dynamic_ncols=True, position=1)
    
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

folds_pbar = tqdm(total=FOLDS, desc="Folds", position=0, ncols=100)

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    folds_pbar.set_description_str(f"Folds (Fold {fold_idx})")

    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    xgb_model = XGBClassifier(
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=500,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        early_stopping_rounds=100,
        objective='multi:softprob',
        random_state=13,
        enable_categorical=True,
        device='cuda'
    )

    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
        callbacks=[TQDMCallback(total=xgb_model.get_params()['n_estimators'], desc=f"Fold {fold_idx}")]
    )

    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob += xgb_model.predict_proba(test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)

    folds_pbar.set_postfix_str(f"Fold {fold_idx} MAP@3: {map3_score:.5f}")
    folds_pbar.update(1)

folds_pbar.close()



top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")

