import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


alt.data_transformers.disable_max_rows()



df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e6/test.csv')




df_train


df_train.describe()


df_train.isna().sum()


df_heat_map = df_train.drop(columns=['Fertilizer Name','Soil Type','Crop Type'])

corr = df_heat_map.corr()
sns.heatmap(corr, annot=True, linewidths=0.5, cmap='coolwarm')
plt.show()


grouped = df_train.groupby(['Crop Type','Soil Type'])['Fertilizer Name'].value_counts().reset_index(name='count')
grouped.to_csv('grouped.csv')


grouped = df_train.groupby(['Crop Type'])['Fertilizer Name'].value_counts().reset_index(name='count')
fertilizers = df_train['Fertilizer Name'].unique()
charts = []

for fertilize in fertilizers:
    fertilizers_data = grouped[grouped['Fertilizer Name']==fertilize]
    source = fertilizers_data
    chart = alt.Chart(source).mark_arc().encode(
        theta = "Crop Type",
        color = alt.Color("count",scale=alt.Scale(scheme='viridis')),
        tooltip=["Crop Type","count",'Fertilizer Name']
    ).properties(
        width=350,
        height=450,
        title = f' fertiize: {fertilize} distrbution by Crop'
    )
    charts.append(chart)

n_cols = 5 
rows = [alt.hconcat(*charts[i:i + n_cols]) for i in range(0, len(charts), n_cols)]
mosaic = alt.vconcat(*rows)

mosaic


grouped_portion = df_train.groupby(['Crop Type','Soil Type','Fertilizer Name']).size().reset_index(name='count')
grouped_fertilize = grouped_portion.groupby('Fertilizer Name')['count'].sum().reset_index(name='total')
grouped_merge = grouped_fertilize.merge(grouped_portion,on='Fertilizer Name')
grouped_merge['proportions'] = grouped_merge['count'] / grouped_merge['total']
source = grouped_merge
alt.Chart(source).mark_bar().encode(
    x=alt.X('Crop Type:N', sort='-y', title='Crop Type'),
    y =alt.Y('proportions:Q',title='Proportion'),
    color='Soil Type',
    tooltip=['Crop Type:N', 'Soil Type:N', 'Fertilizer Name:N', 'count:Q']

).properties(
width=350,
height=450,
title = f'fertilizer per proportion'
).facet(
    column= 'Fertilizer Name:N'
)



chart1 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Temparature):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

chart2 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Humidity):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

chart3 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Moisture):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

chart3 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Nitrogen):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

chart4 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Potassium):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

chart5 = alt.Chart(df_train).mark_bar().encode(
    x='Soil Type:N',
    y='sum(Phosphorous):Q',
    color='Soil Type:N',
    tooltip=['Soil Type:N', 'Temparature:Q', 'Crop Type:N']
).properties(
    width=150,
    height=150
).facet(
    column='Crop Type:N'
)

charts = [chart1, chart2, chart3, chart4, chart5]
n_cols=  11
#rows = [alt.hconcat(*charts[i:i + n_cols]) for i in range (0, len(charts),n_cols)]
#mosaic = alt.vconcat(*rows)
mosaic = alt.vconcat(*charts)
mosaic




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score # metricas
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix





le = LabelEncoder()
df_model = df_train.copy()
df_model = pd.get_dummies(df_model,columns=["Soil Type", "Crop Type"])
x = df_model.drop(columns='Fertilizer Name')
y = le.fit_transform(df_model['Fertilizer Name'])

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state=42)
pipe = make_pipeline(StandardScaler(),XGBClassifier())
pipe.fit(x_train,y_train)
score = pipe.score(x_test,y_test)
print(f"training set: {x_train.shape}, test set shape: {x_test.shape}")
print(score)





models = {
    "RandomForestClassifier" : RandomForestClassifier(),
    "Xgboost" : XGBClassifier()
}

metrics = {
    'Model': [],
    'accuracy_score' : [],
    'precision_score' : [],
    'recall_score' : [],
    'f1_score' : [],
    'classification_report'  : [],

}




for name, model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    metrics['Model'].append(name)
    metrics['accuracy_score'].append(accuracy)
    metrics['precision_score'].append(precision)
    metrics['recall_score'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['classification_report'].append(report)    
    

results = pd.DataFrame(metrics)
results


xgb = XGBClassifier()
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(x_train, y_train)

print(f'Mejor accuracy: {random_search.best_score_:.4f}')
print(f'Mejores parámetros: {random_search.best_params_}')


best_xgb = XGBClassifier(
    colsample_bytree=0.8123620356542087,
    learning_rate=0.2952142919229748,
    max_depth=5,
    n_estimators=121,
    subsample=0.8795975452591109,
    use_label_encoder=False,
    eval_metric='mlogloss'  # Para evitar warnings
)

# Entrenalo con el set de entrenamiento
best_xgb.fit(x_train, y_train)

# Predecí con el test set
y_pred = best_xgb.predict(x_test)

# Evaluá el resultado
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))



y_pred = best_xgb.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
plt.Figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=le.classes_,yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


df_test_ml = df_test.copy()
df_test_ml=pd.get_dummies(df_test_ml,columns=["Soil Type", "Crop Type"])
df_test_ml


df_test_ml.info



y_pred = model.predict(df_test_ml)
submission = pd.DataFrame({
    'id':df_test['id'],
    'Fertilizer Name' : le.inverse_transform(y_pred)
})



submission.to_csv('submission.csv',index=False)

