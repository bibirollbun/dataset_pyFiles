import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import plotly.graph_objects as go
import plotly.io as pio  
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from lightgbm import LGBMRegressor
import optuna 
from plotly.subplots import make_subplots
import plotly.subplots as sp
from IPython.display import display, HTML
import warnings
warnings.filterwarnings("ignore")
pio.renderers.default = 'iframe_connected'


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


train.describe()


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())
print('-'*20)
print("Number of Rows:",train.shape[0])
print('-'*20)
print("Number of Columns:",train.shape[1])


train.nunique()


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for i in cat_cols:
    print(f"Unique categories in '{i}' column: {train[i].unique()}")
    print("<--- --- --- --- --- --- --- --- --- --->\n")


for i in cat_cols:
    print(f"Distribution of '{i}' column: {train[i].value_counts()}")
    print("<--- --- --- --- --- --- --- --- --- --->\n")


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean())
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())


train.head()


brand_counts = train['Brand'].value_counts().reset_index()
brand_counts.columns = ['Brand', 'Count']

fig = px.bar(
    brand_counts, 
    x='Brand', 
    y='Count', 
    title='Brand Distribution',
    color='Count',
    color_continuous_scale='GnBu'
)

fig.update_layout(
    xaxis_title='Brand',
    yaxis_title='Count',
    template='plotly_white',
    width=600,
    height=500
)

fig.show()


mat_counts = train['Material'].value_counts().reset_index()
mat_counts.columns = ['Material', 'Count']

fig = px.bar(
    mat_counts, 
    x='Material', 
    y='Count', 
    title='Material Distribution',
    color='Count',
    color_continuous_scale='GnBu'
)

fig.update_layout(
    xaxis_title='Material',
    yaxis_title='Count',
    template='plotly_white',
    width=600,
    height=500
)

fig.show()


style_counts = train['Style'].value_counts().reset_index()
style_counts.columns = ['Style', 'Count']

fig = px.bar(
    style_counts, 
    x='Style', 
    y='Count', 
    title='Style Distribution',
    color='Count',
    color_continuous_scale='GnBu'
)

fig.update_layout(
    xaxis_title='Style',
    yaxis_title='Count',
    template='plotly_white',
    width=600,
    height=500
)

fig.show()


color_counts = train['Color'].value_counts().reset_index()
color_counts.columns = ['Color', 'Count']

fig = px.bar(
    color_counts, 
    x='Color', 
    y='Count', 
    title='Color Distribution',
    color='Count',
    color_continuous_scale='GnBu'
)

fig.update_layout(
    xaxis_title='Color',
    yaxis_title='Count',
    template='plotly_white',
    width=600,
    height=500
)

fig.show()


lap_counts = train['Laptop Compartment'].value_counts().reset_index()
lap_counts.columns = ['Laptop Compartment', 'Count']

fig = px.pie(
    lap_counts, 
    names='Laptop Compartment', 
    values='Count', 
    title='Laptop Compartment Distribution',
    color='Count', 
    color_discrete_sequence=px.colors.sequential.GnBu  
)
fig.update_layout(width=600, height=500)
fig.show()


waterproof_counts = train['Waterproof'].value_counts().reset_index()
waterproof_counts.columns = ['Waterproof', 'Count']

fig2 = px.pie(
    waterproof_counts, 
    names='Waterproof', 
    values='Count', 
    title='Waterproof Distribution',
    color='Count', 
    color_discrete_sequence=px.colors.sequential.GnBu  
)
fig2.update_layout(width=600, height=500)
fig2.show()


size_counts = train['Size'].value_counts().reset_index()
size_counts.columns = ['Size', 'Count']

fig3 = px.pie(
    size_counts, 
    names='Size', 
    values='Count', 
    title='Size Distribution',
    color='Count', 
    color_discrete_sequence=px.colors.sequential.GnBu  
)
fig3.update_layout(width=600, height=500)
fig3.show()


plt.figure(figsize=(8, 12))  

# Compartments
plt.subplot(3, 1, 1)
sns.histplot(train['Compartments'], kde=True, color='skyblue', bins=10)
plt.title('Compartments Distribution')
plt.xlabel('Compartments')
plt.ylabel('Frequency')

# Weight Capacity
plt.subplot(3, 1, 2)
sns.histplot(train['Weight Capacity (kg)'], kde=True, color='lightcoral', bins=10)
plt.title('Weight Capacity Distribution')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Frequency')

# Price
plt.subplot(3, 1, 3)
sns.histplot(train['Price'], kde=True, color='lightgreen', bins=10)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


brand_material_counts = train.groupby(['Brand', 'Material']).size().reset_index(name='Count')

fig = px.bar(
    brand_material_counts, 
    x='Brand', 
    y='Count', 
    color='Material', 
    title='Brand vs Material Distribution', 
    barmode='group', 
    color_continuous_scale='Cividis'
)

fig.update_layout(
    xaxis_title='Brand',
    yaxis_title='Count',
    template='plotly_white',
    width=750,
    height=500
)

fig.show()


brand_size_counts = train.groupby(['Brand', 'Size']).size().reset_index(name='Count')

fig = px.bar(
    brand_size_counts, 
    x='Brand', 
    y='Count', 
    color='Size', 
    title='Brand vs Size Distribution', 
    barmode='group', 
    color_continuous_scale='Cividis'
)

fig.update_layout(
    xaxis_title='Brand',
    yaxis_title='Count',
    template='plotly_white',
    width=750,
    height=500
)

fig.show()


print("Average Distribution of Price:")
for cat in cat_cols:
    print(train.groupby(cat)['Price'].agg(avg= 'mean', count= 'count', std = 'std').round(2))
    print('-'*30)
    print()


print("Average Distribution of Weight:")
for col in cat_cols:
    print(f"Distribution of '{col}':")
    print(train.groupby(cat)['Weight Capacity (kg)'].agg(avg= 'mean', count= 'count', std = 'std').round(2))
    print("-" * 30)  


categorical_pairs = [
    ('Brand', 'Waterproof'),
    ('Brand', 'Laptop Compartment'),
    ('Waterproof', 'Laptop Compartment')
]


def chi_square_test(train, col1, col2):
    contingency_table = pd.crosstab(train[col1], train[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-Square Test: {col1} vs {col2}")
    print(f"Chi-Square Statistic: {chi2:.4f}, p-value: {p:.4f}, Degrees of Freedom: {dof}")
    
    if p < 0.05:
        print(f"ğŸ”¹ Significant Relationship: {col1} and {col2} are dependent.")
    else:
        print(f"âšª No Significant Relationship: {col1} and {col2} are independent.")

for col1, col2 in categorical_pairs:
    chi_square_test(train, col1, col2)


le = LabelEncoder()

for col in cat_cols:
    train[col] = le.fit_transform(train[col])


le = LabelEncoder()

for col in cat_cols:
    test[col] = le.fit_transform(test[col])


train.head()


corr_matrix = train.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


X = train.drop(columns=['Price'])
y = train['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', -1, 10),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e2),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e2),
        'device': 'gpu'
    }

    model = LGBMRegressor(**params, verbose=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)


#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=30)

#lgb_param = study.best_params
#print("Best parameters found: ", lgb_param)


best_params = {
    'n_estimators': 1985,
    'max_depth': 7,
    'learning_rate': 0.001295661934160046,
    'num_leaves': 30,
    'min_child_samples': 24,
    'subsample': 0.6487334653300381,
    'colsample_bytree': 0.7102224739453876,
    'reg_alpha': 7.171182774502767,
    'reg_lambda': 0.06413641328286093,
    'device': 'gpu'  
}


model = lgb.LGBMRegressor(**best_params)
model.fit(X, y)


test.head()


submission_ids = test['id']
predictions = model.predict(test)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

