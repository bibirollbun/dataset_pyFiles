import pandas as pd
import seaborn as sbn

raw_train_df = pd.read_excel("/kaggle/input/retencion-en-telefonia-movil-202503/traintelco.xlsx")
raw_train_df = raw_train_df.drop(columns=['id'])


from datetime import datetime
from dateutil.relativedelta import relativedelta

format_string = "%Y-%m-%d %H:%M:%S.%f"
current_date = datetime.strptime("2019.01.01", "%Y.%m.%d")

def months_between(end_date, start_date):
    rd = relativedelta(end_date, start_date)
    return rd.years * 12 + rd.months

def prepare_data(raw_train_df):
    columns = raw_train_df.columns.to_list()

    columns[0] = 'DOB'
    columns[1] = 'Client category'
    columns[2] = 'Online'
    columns[3] = 'Phone age'
    columns[4] = 'Premium'
    columns[5] = 'Income'
    columns[6] = 'CountDelay'
    columns[7] = 'Experience'
    columns[8] = 'TimeUse'
    
    try:
        columns[9] = 'Target'
    except:
        print("test")
    raw_train_df.columns = columns
    print(raw_train_df["DOB"][20])
    raw_train_df["Age"] = raw_train_df["DOB"].apply(lambda x: 2019 - x.year)

    #raw_train_df["TimeRegistry"] = raw_train_df["DOB"].apply(lambda x: x.hour)

    raw_train_df["AgeRegistry"] = raw_train_df["Experience"].apply(lambda x: months_between(current_date, x))
    
    raw_train_df["MidlMonth"] = raw_train_df["TimeUse"] / raw_train_df["AgeRegistry"]
    raw_train_df["MidCost"] = raw_train_df["Income"] / raw_train_df["TimeUse"]
    train_df = raw_train_df.drop(columns=["DOB", "Experience"])
    
    train_df['CountDelay_sq'] = train_df['CountDelay']**2
    train_df['Income_sq'] = train_df['Income']**2

    return train_df

train_df = prepare_data(raw_train_df)
train_df.head(5)



import plotly.express as px

fig = px.histogram(
    train_df,
    x='TimeUse',
    color='Premium',
    title='Distribution of Addiction Time by Premium Status',
    labels={'TimeUse': 'Addiction Time, minutes', 'count': 'Count'}
)
fig.show()

import plotly.express as px




df_agg = train_df.groupby(['Age', 'Premium'], as_index=False)['TimeUse'].mean()
df_agg.head(5)

fig = px.bar(
    df_agg,
    x="Age",
    y="TimeUse",
    color="Premium",
    title="Average Addiction Time vs Age by Premium Status",
    labels={
        'TimeUse': 'Average Addiction Time (minutes)',
        'Age': 'Age',
        'Premium': 'Premium Status'
    },
    barmode='group'
)
fig.show()

df_agg = train_df.groupby(['Age', 'Online'], as_index=False)['TimeUse'].mean()
fig = px.bar(
    df_agg,
    x="Age",
    y="TimeUse",
    color="Online",
    title="Average Addiction Time vs Age by Online Status",
    labels={
        'TimeUse': 'Average Addiction Time (minutes)',
        'Age': 'Age',
        'Online': 'Online use/not'
    },
    barmode='group'
)
fig.show()



fig = px.density_heatmap(
    train_df,
    x="Age",
    y="TimeUse",
    nbinsx=30,      
    nbinsy=30, 
    title="Heatmap: Age vs Addiction Time",
    labels={
        'Age': 'Age',
        'TimeUse': 'Addiction Time (minutes)'
    },
    color_continuous_scale='Blues'
)
fig.show()


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

features = ["Age", "AgeRegistry", "TimeUse", "MidlMonth","MidCost",'CountDelay_sq','Income_sq', "CountDelay", "Premium", "Online", "Target"]
df_corr = train_df[features].dropna()

corr_matrix = df_corr.corr()#pirson

fig = px.imshow(
    corr_matrix,
    text_auto=True,      
    color_continuous_scale='Blues', 
    aspect="auto",
    title="Correlation Heatmap of Features"
)

fig.update_layout(
    width=800,
    height=600
)

fig.show()


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
X = train_df.drop(columns=["Target"])
y = train_df["Target"]

categorical_features = ["Premium", "Online", "Client category"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=66)

counts = y_train.value_counts()
scale_pos_weight_value = counts[0] / counts[1]

model = CatBoostClassifier(
    iterations = 700,
    learning_rate=0.01,
    depth=6,
    scale_pos_weight=scale_pos_weight_value,
    loss_function='Logloss',
    eval_metric="AUC",
    random_seed=666,
    verbose=100,
    early_stopping_rounds=50
)

search_params = model.grid_search(
    {
        'depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.08],
        'l2_leaf_reg': [1, 2, 3, 5]
    },
    X=X_train,
    y=y_train,
    cv = 3,
    plot=True
)
best_params = search_params['params']
print(best_params)
model = CatBoostClassifier(
    **best_params,
    iterations = 1500,
    scale_pos_weight=scale_pos_weight_value,
    loss_function='Logloss',
    eval_metric="AUC",
    random_seed=666,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(X_train, y_train,
          cat_features=categorical_features,
          eval_set=(X_val, y_val),
          plot=True)

y_pred = model.predict(X_val)
y_pred_one = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_one)
print(auc)

feature_importance = model.get_feature_importance(prettified=True)
print(feature_importance)

model_filename = "model_without_sqr.cbm"
model.save_model(model_filename)


from catboost import Pool
explainer = model.get_feature_importance(
    data= Pool(X_val, y_val, cat_features=categorical_features),
    type='ShapValues'
)


object_index = 5
shap_values_for_one_object = explainer[object_index][:-1] 

feature_names = X_val.columns.tolist()


shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_values_for_one_object
})

shap_df['abs_shap'] = shap_df['shap_value'].abs()
shap_df = shap_df.sort_values(by='abs_shap', ascending=True)

colors = ['firebrick' if val > 0 else 'royalblue' for val in shap_df['shap_value']]



fig = go.Figure()

fig.add_trace(go.Bar(
    x=shap_df['shap_value'],
    y=shap_df['feature'],
    orientation='h',
    marker_color=colors,
    text=shap_df['shap_value'].round(3),
    textposition='outside'
))

fig.update_layout(
    title_text=f'<b>Importance of Features for #{object_index} (SHAP Values)</b>',
    title_x=0.5,
    xaxis_title='Influence of predict (SHAP Value)',
    yaxis_title='Feature',
    template='plotly_white',
    margin=dict(l=200) 
)

fig.show()


test_df = pd.read_excel("/kaggle/input/retencion-en-telefonia-movil-202503/testelco.xlsx")
ids = pd.DataFrame({"id": test_df["id"]})
test_df = test_df.drop(columns=["id"])
test_df['Fecha de nacimiento'] = pd.to_datetime(test_df['Fecha de nacimiento'], format=format_string)
test_df['Fecha inicio contrato'] = pd.to_datetime(test_df['Fecha inicio contrato'], format=format_string)
X_test = prepare_data(test_df)
y_pred_res = model.predict(X_test)

res_df = pd.DataFrame({"id": ids["id"], "resultado": y_pred_res})
res_df.to_csv('submission.csv', index=False)



train_df['CountDelay_bin'] = pd.cut(train_df['CountDelay'], bins=15)


binned_data = train_df.groupby('CountDelay_bin')['Target'].mean().reset_index()


binned_data['CountDelay_mid'] = binned_data['CountDelay_bin'].apply(lambda x: x.mid)


fig = px.bar(
    binned_data,
    x='CountDelay_mid',
    y='Target',
    title='Probality Target=1 about CountDelay',
    labels={'CountDelay_mid': 'CountDelay', 'Target': '% Target=1'}
)
fig.show()

