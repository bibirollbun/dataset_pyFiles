from IPython.display import Image
Image('/kaggle/input/formula-map3/Formula_MAP3.png')


# Base
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

#Preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_regression

#Models ML 
from xgboost import XGBClassifier
from xgboost import plot_importance
import lightgbm as lgb
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

#Model evaluation
import shap
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns = ['id'])
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv").drop(columns = ['id'])
df_orig  = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
sub      = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


df_train.head()


df_train.info()


df_train['Fertilizer Name'].unique(), df_orig['Fertilizer Name'].unique()


df_train['Crop Type'].unique()


df_train['Soil Type'].unique()


# All orig. dataset is a train data. In the validation we have X_train {from df_train} + Orig.data {from all df_orig} /// X_valid {from df_train}
#df_train = pd.concat([df_train, df_orig], axis = 0)
#df_train.info()


df_original_copy = df_orig.copy()
for i in range(6):
    df_orig = pd.concat([df_orig, df_original_copy], axis = 0)


df_train.describe(exclude = 'object').style.background_gradient(axis = 1, cmap = 'rainbow', low = 0.1, high = 1.0)


df_train.describe(include = 'object').T


'''
result = Description(df_train,
                     stats=["nobs", "missing", "mean", "std_err", "ci", "ci", 
                            "std", "iqr", "mad", "coef_var", "range", "max", 
                            "min", "skew", "kurtosis", "mode","median",
                            "percentiles", "distinct", "top", "freq"],
                     alpha = 0.05,
                     use_t = True)
num_stat = (pd.DataFrame(result.summary(), columns = ['STAT', 'Temparature', 'Humidity', 
                                          'Moisture', 'Nitrogen', 'Potassium', 
                                          'Phosphorous'], index = ["nobs", "missing", "mean", "std_err", "ci", "ci", 
                                                                   "std", "iqr", "mad", "coef_var", "range", "max", 
                                                                   "min", "skew", "kurtosis", "mode","mode_freq","median",
                                                                   "1","5","10","25","50","75","90","95","99"]).drop(columns = ['STAT']))
num_stat
'''


plt.figure(figsize=(8, 4))
ax = sns.countplot(data = df_train, x = 'Fertilizer Name', order = df_train['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Type Distribution', fontsize = 14)
plt.xlabel('Fertilizer Type')
plt.ylabel('Count')
plt.xticks(rotation = 45)
total = len(df_train)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), ha = 'center', va= 'center', xytext = (0, 5), textcoords='offset points')
plt.show()


fig = px.pie(values = df_train['Fertilizer Name'].value_counts(), 
             names = df_train['Fertilizer Name'].unique(), 
             title = 'Fertilizer distribution',
             color_discrete_sequence = px.colors.qualitative.Pastel)
init_notebook_mode(connected=True)
py.iplot(fig)


plt.figure(figsize=(8, 4))
ax = sns.countplot(data = df_train, x = 'Soil Type', order = df_train['Soil Type'].value_counts().index)
plt.title('Soil Type Distribution', fontsize = 14)
plt.xlabel('Soil Type')
plt.ylabel('Count')
total = len(df_train)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), ha ='center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(data = df_train, y = 'Crop Type', order = df_train['Crop Type'].value_counts().index)
plt.title('Crop Type Distribution', fontsize = 14)
plt.xlabel('Count')
plt.ylabel('Crop Type')
plt.show()


num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Histograms (Fertilizer Name)
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, feature in enumerate(num_features):
    sns.histplot(data = df_train, x = df_train[feature], hue = 'Fertilizer Name', kde = True, ax = axes[i//3, i%3], bins = 30, color = 'skyblue')
    axes[i//3, i%3].set_title(f'{feature} Distribution', fontsize=12)
plt.tight_layout()
plt.show()


# Histograms (Soil Type)
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, feature in enumerate(num_features):
    sns.histplot(data = df_train, x = df_train[feature], hue = 'Soil Type', kde = True, ax = axes[i//3, i%3], bins = 30, color = 'skyblue')
    axes[i//3, i%3].set_title(f'{feature} Distribution', fontsize = 12)
plt.tight_layout()
plt.show()


# Histograms (Crop Type)
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, feature in enumerate(num_features):
    sns.histplot(data = df_train, x = df_train[feature], hue = 'Crop Type', kde = True, ax = axes[i//3, i%3], bins = 30, color = 'skyblue')
    axes[i//3, i%3].set_title(f'{feature} Distribution', fontsize = 12)
plt.tight_layout()
plt.show()


# Boxplots
plt.figure(figsize = (10, 5))
sns.boxplot(data = df_train[num_features], palette = 'rainbow')
plt.title('Numerical Feature Distributions', fontsize = 14)
plt.xticks(rotation=45)
plt.show()


# Let's add an encoded target to the correlation matrix.
Fertilizer_Name_train = df_train['Fertilizer Name']
Fertilizer_Name_orig  = df_orig['Fertilizer Name']

le = LabelEncoder()
df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])
df_orig['Fertilizer Name']  = le.fit_transform(df_orig['Fertilizer Name'])


num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']
sns.pairplot(df_train[num_features], corner = True)


plt.figure(figsize = (15, 5))
sns.countplot(data = df_train, x = 'Soil Type', hue = 'Fertilizer Name', 
              hue_order = df_train['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Distribution by Soil Type', fontsize = 14)
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.legend(title = 'Fertilizer', loc = 'best')
plt.show()


plt.figure(figsize=(12, 8))
sns.countplot(data = df_train, y = 'Crop Type', hue = 'Fertilizer Name', 
              hue_order = df_train['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Distribution by Crop Type', fontsize = 14)
plt.xlabel('Count')
plt.ylabel('Crop Type')
plt.legend(title = 'Fertilizer', loc = 'lower right')
plt.show()


plt.figure(figsize = (10, 5))
corr_matrix = df_train[num_features + ['Fertilizer Name']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = 'seismic', mask = mask)
plt.title('Numerical Feature Correlations', fontsize = 14)
plt.show()


sample_df = df_train.sample(frac=0.5, random_state=42)
fig = px.scatter_3d(sample_df,
                    x='Nitrogen',
                    y='Phosphorous',
                    z='Potassium',
                    color='Temparature',
                    size_max=14,
                    hover_name='Fertilizer Name',
                    hover_data=['Crop Type', 'Soil Type'],
                    color_continuous_scale='rainbow',
                    title='<b>NPK Relationship with Temperature and Moisture</b><br>''<i>Color: Temperature | Size: Moisture</i>',
                    labels={'Nitrogen': 'Nitrogen (N)',
                            'Phosphorous': 'Phosphorous (P)',
                            'Potassium': 'Potassium (K)',
                            'Temparature': 'Temperature (Â°C)',
                            'Moisture': 'Moisture Level'})

fig.update_layout(scene=dict(xaxis_title='<b>NITROGEN</b>',
                             yaxis_title='<b>PHOSPHOROUS</b>',
                             zaxis_title='<b>POTASSIUM</b>',
                             camera=dict(eye=dict(x=1.5, y=1.5, z=0.1))),
                  coloraxis_colorbar=dict(title='Temperature',
                                          thickness=20,
                                          len=0.75),
                  margin=dict(l=0, r=0, b=0, t=40),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

fig.update_traces(hovertemplate=("<b>Fertilizer: %{hovertext}</b><br>"
                                 "N: %{x} | P: %{y} | K: %{z}<br>"
                                 "Temp: %{marker.color}Â°C<br>"
                                 "Moisture: %{marker.size:.1f}<br>"
                                 "Crop: %{customdata[0]}<br>"
                                 "Soil: %{customdata[1]}"), 
                  marker_size = 4)
init_notebook_mode(connected=True)
py.iplot(fig)


'''
        ---Useless features--- (+3-5% error rate)
unique_soil_train =  df_train['Soil Type'].unique()
unique_soil_test  =  df_test['Soil Type'].unique()

for soil in unique_soil_train:
    temparature_col = f'Temparature_SoilType_{soil}'
    humidity_col    = f'Humidity_SoilType_{soil}'
    moisture_col    = f'Moisture_SoilType_{soil}'
    nitrogen_col    = f'Nitrogen_SoilType_{soil}'
    potassium_col   = f'Potassium_SoilType_{soil}'
    phosphorous_col = f'Phosphorous_SoilType_{soil}'
    df_train[temparature_col] = np.where(df_train['Soil Type']  == soil, df_train['Temparature'], 0)
    df_train[humidity_col]    =  np.where(df_train['Soil Type'] == soil, df_train['Humidity'],    0)
    df_train[moisture_col]    =  np.where(df_train['Soil Type'] == soil, df_train['Moisture'],    0)
    df_train[nitrogen_col]    =  np.where(df_train['Soil Type'] == soil, df_train['Nitrogen'],    0)
    df_train[potassium_col]   =  np.where(df_train['Soil Type'] == soil, df_train['Potassium'],   0)
    df_train[phosphorous_col] =  np.where(df_train['Soil Type'] == soil, df_train['Phosphorous'], 0)

for soil in unique_soil_test:
    temparature_col = f'Temparature_SoilType_{soil}'
    humidity_col    = f'Humidity_SoilType_{soil}'
    moisture_col    = f'Moisture_SoilType_{soil}'
    nitrogen_col    = f'Nitrogen_SoilType_{soil}'
    potassium_col   = f'Potassium_SoilType_{soil}'
    phosphorous_col = f'Phosphorous_SoilType_{soil}'
    df_test[temparature_col] = np.where(df_test['Soil Type']  == soil, df_test['Temparature'], 0)
    df_test[humidity_col]    =  np.where(df_test['Soil Type'] == soil, df_test['Humidity'],    0)
    df_test[moisture_col]    =  np.where(df_test['Soil Type'] == soil, df_test['Moisture'],    0)
    df_test[nitrogen_col]    =  np.where(df_test['Soil Type'] == soil, df_test['Nitrogen'],    0)
    df_test[potassium_col]   =  np.where(df_test['Soil Type'] == soil, df_test['Potassium'],   0)
    df_test[phosphorous_col] =  np.where(df_test['Soil Type'] == soil, df_test['Phosphorous'], 0)
'''


'''
        ---Useless features--- (+3-5% error rate)
unique_crop_train =  df_train['Crop Type'].unique()
unique_crop_test  =  df_test['Crop Type'].unique()

for crop in unique_crop_train:
    temparature_col = f'Temparature_SoilType_{crop}'
    humidity_col    = f'Humidity_SoilType_{crop}'
    moisture_col    = f'Moisture_SoilType_{crop}'
    nitrogen_col    = f'Nitrogen_SoilType_{crop}'
    potassium_col   = f'Potassium_SoilType_{crop}'
    phosphorous_col = f'Phosphorous_SoilType_{crop}'
    df_train[temparature_col] = np.where(df_train['Crop Type']  == crop, df_train['Temparature'], 0)
    df_train[humidity_col]    =  np.where(df_train['Crop Type'] == crop, df_train['Humidity'],    0)
    df_train[moisture_col]    =  np.where(df_train['Crop Type'] == crop, df_train['Moisture'],    0)
    df_train[nitrogen_col]    =  np.where(df_train['Crop Type'] == crop, df_train['Nitrogen'],    0)
    df_train[potassium_col]   =  np.where(df_train['Crop Type'] == crop, df_train['Potassium'],   0)
    df_train[phosphorous_col] =  np.where(df_train['Crop Type'] == crop, df_train['Phosphorous'], 0)

for crop in unique_crop_test:
    temparature_col = f'Temparature_SoilType_{crop}'
    humidity_col    = f'Humidity_SoilType_{crop}'
    moisture_col    = f'Moisture_SoilType_{crop}'
    nitrogen_col    = f'Nitrogen_SoilType_{crop}'
    potassium_col   = f'Potassium_SoilType_{crop}'
    phosphorous_col = f'Phosphorous_SoilType_{crop}'
    df_test[temparature_col] = np.where(df_test['Crop Type']  == crop, df_test['Temparature'], 0)
    df_test[humidity_col]    =  np.where(df_test['Crop Type'] == crop, df_test['Humidity'],    0)
    df_test[moisture_col]    =  np.where(df_test['Crop Type'] == crop, df_test['Moisture'],    0)
    df_test[nitrogen_col]    =  np.where(df_test['Crop Type'] == crop, df_test['Nitrogen'],    0)
    df_test[potassium_col]   =  np.where(df_test['Crop Type'] == crop, df_test['Potassium'],   0)
    df_test[phosphorous_col] =  np.where(df_test['Crop Type'] == crop, df_test['Phosphorous'], 0)
'''


# + empirical features [N to P, N to K, etc.]       and xgb_params_1 + without aggregate features - CV=0.34586%
# Without empirical features [N to P, N to K, etc.] and xgb_params_1 + without aggregate features - **CV=0.36185%**
# Without empirical features [N to P, N to K, etc.] and xgb_params_2 + Without aggregate features - CV=0.34941%
# + empirical features [N to P, N to K, etc.]       and xgb_params_1 + aggregate features (mean) - CV=0.33148%
# Without empirical features [N to P, N to K, etc.] and xgb_params_1 + aggregate features (mean) - CV=0.34964%
# Without empirical features [N to P, N to K, etc.] and xgb_params_1 + without aggregate features + unique_soil_train - CV=0.34416%

def categorical_aggregations(df):
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

    for i in range(1, len(categorical_cols) + 1):
        if i == 1:
            for cat_col in categorical_cols:
                aggs = df.groupby(cat_col).agg({num_col: ['mean'] for num_col in numerical_cols})
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                df = df.merge(aggs, on = cat_col, how = 'left')
        elif i == 2:
            for j in range(len(categorical_cols)):
                for k in range(j + 1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    aggs = df.groupby([cat_col1, cat_col2]).agg({num_col: ['mean'] for num_col in numerical_cols})
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    df = df.merge(aggs, on = [cat_col1, cat_col2], how = 'left')
    return df

#df_train = categorical_aggregations(df_train)
#df_test = categorical_aggregations(df_test)


'''
        ---Useless features--- (+5-7% error rate)
new_le = LabelEncoder()

df_train['Coder_Soil_Type'] = new_le.fit_transform(df_train['Soil Type'])
df_test['Coder_Soil_Type']  = new_le.transform(df_test['Soil Type'])

df_train['Coder_Crop_Type'] = new_le.fit_transform(df_train['Crop Type'])
df_test['Coder_Crop_Type']  = new_le.transform(df_test['Crop Type'])


def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

list_1 = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
list_2 = ['Coder_Soil_Type', 'Coder_Crop_Type']

df_train = add_feature_cross_terms(df_train, list_1, list_2)
df_test = add_feature_cross_terms(df_test, list_1, list_2)
'''


'''
        ---Useless features--- (+3-5% error rate)
def fe(df):
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['Climate_Index'] = (df['Temparature'] + df['Humidity']) / 2
    df['Water_Stress'] = df['Humidity'] - df['Moisture']
    return df

df_train = fe(df_train)
df_test =  fe(df_test)
'''


df_train = df_train.reset_index(drop = True)


df_orig = df_orig.reset_index(drop = True)


df_test = df_test.reset_index(drop = True)


def binarization(df):
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    for col in num_cols:
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
    return df

df_train = binarization(df_train)
df_orig = binarization(df_orig)
df_test  = binarization(df_test)


df_train.info()


label_encoder_category = LabelEncoder()
cat_cols = ['Crop Type', 'Soil Type']
for col in cat_cols:
    df_train[col] = label_encoder_category.fit_transform(df_train[col])
    df_orig[col]  = label_encoder_category.fit_transform(df_orig[col])
    df_test[col]  = label_encoder_category.fit_transform(df_test[col])


for col in cat_cols:
    df_train[col] = df_train[col].astype("category")
    df_orig[col]  = df_orig[col].astype("category")
    df_test[col]  = df_test[col].astype("category")


df_train.info()


df_orig.info()


# Let's check if the column order is the same
columns_match = df_train.columns.equals(df_test.columns.append(pd.Index(['Fertilizer Name'])))
print(f"Is the column order the same: {columns_match}")

if not columns_match:
    df_train_without_FN = df_train.drop(columns=['Fertilizer Name'])
    common_columns = [col for col in df_test.columns if col in df_train_without_FN.columns]
    df_train_without_FN = df_train_without_FN[common_columns]
    df_test = df_test[common_columns]
    df_train = pd.concat([df_train_without_FN, df_train['Fertilizer Name']], axis = 1)
    print("The column order has been corrected")

# Let's check again
df_train_without_FN = df_train.drop(columns=['Fertilizer Name'])
columns_match_after_drop = df_train_without_FN.columns.equals(df_test.columns)
print(f"Is the column order the same after calories fall: {columns_match_after_drop}")


df_train.info()


df_test.info()


num_cols = list(df_train.drop(columns = ['Fertilizer Name']).select_dtypes(exclude=['object','category']).columns)
cat_cols = list(df_train.select_dtypes(include=['object','category']).columns.difference(['Fertilizer Name']))
np.size(num_cols + cat_cols) == df_train.drop(columns = ['Fertilizer Name']).shape[1]


# Evalueation functions (eval_metric)
def map3(true, pred, k = 3):
    final_score = []
    label = [[i] for i in true] # numpy to list in elements (max_prob, medium_prob, min_prob)
    for a, b in zip(label, pred):
        b = b[:k] # We take only 3 elements, axis = 1
        score = 0.0 # Our MAP@3 score the n line of b
        hit = 0.0 # if an element from the pred array is in the same true row and is not contained in rep
        rep = set() # repeated matching pred[i]=label[i] elements, we are not considering them.
        for i, j in enumerate(b): # i = [0,1,2] since there are only 3 elements in pred lines, j = [j1 j2 j3] for i = 1, j = [j4 j5 j6] for i = 2, etc.
            if j in a and j not in rep:
                hit+=1.0
                score+=hit/(i+1.0)
                rep.add(j)
        final_score.append(score/(min(len(a), k)))
    return np.mean(final_score)


'''
y_full = df_train['Fertilizer Name'].reset_index(drop = True)
X_full = df_train.drop(columns = ['Fertilizer Name']).reset_index(drop = True)
X_original = df_orig.drop(columns = ['Fertilizer Name'])
y_original = df_orig['Fertilizer Name']
X_test = df_test

scaler = StandardScaler().set_output(transform = "pandas")

default_param = {'objective': 'multi:softmax',  # For multi-class classification (or softprob)
                 'num_class': len(le.classes_),  # Number of fertilizer classes
                 'eval_metric': 'mlogloss',     # Multi-class log loss
                 'tree_method': 'gpu_hist',
                 'device': 'cuda',
                 'seed': 42,
                 'enable_categorical': "True"
                }

#---------------------------------------------------------------------------------
trained_model_XGB = XGBClassifier(**default_param).fit(X_full, y_full, verbose = 0)
feature_names = X_full.columns
importances = trained_model_XGB.feature_importances_

feature_imp = pd.DataFrame({'Feature': feature_names,
                            'Value': importances})

feature_imp = feature_imp.sort_values('Value', ascending = False).head(105)
#---------------------------------------------------------------------------------
y_pred = trained_model_XGB.predict_proba(X_full)
top_3_preds = np.argsort(y_pred, axis = 1)[:, -3:][:, ::-1]
map3_score = map3(np.array(y_full), top_3_preds)
print(f"MAP@3 score on full data = {round(map3_score, 5)}%")
#---------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
ax = plt.subplot()
sns.barplot(x = 'Value', 
            y = 'Feature', 
            data = feature_imp,
            palette = 'seismic',
            edgecolor = 'black',
            linewidth = 1,
            ax = ax)

plt.title('XGB Feature Importance', fontsize = 16, pad = 20, fontweight = 'bold')
plt.xlabel('Feature Importance', fontsize = 12, labelpad = 10)
plt.ylabel('Features', fontsize = 12, labelpad = 10)

ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
sns.despine(left = True, bottom = True)

for i, v in enumerate(feature_imp['Value']):
    ax.text(v + 0.001, i, f'{v:.4f}', color = 'black', ha = 'left', va = 'center', fontsize = 9)

plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
'''


'''
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_full, y_full, random_state = 42, discrete_features = True)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_full.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending = False), columns = ['Mutual Information'])
mutual_info.style.bar(subset = ['Mutual Information'], cmap = 'seismic')
'''


'''
plt.figure(figsize = (25, 15))
corr_matrix = pd.concat([X_full, y_full], axis = 1).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = 'seismic', mask = mask, vmin = -1, vmax = 1)
plt.title('Numerical Feature Correlations - Final', fontsize = 14)
plt.show()
'''


'''
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 3000, 6000, step=100),
        'max_depth': trial.suggest_int("max_depth", 8, 15, step=1),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-2, 10, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 0.99),
        'gamma': trial.suggest_float("gamma", 0, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.99),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'enable_categorical': 'False',
        'seed': 42
    }

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    map3_scores_all = []
    y_valid_proba = np.zeros(shape = (len(X_full), y_full.nunique()))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_full, y_full)):
        X_train, X_valid = X_full.iloc[train_idx].copy(), X_full.iloc[valid_idx].copy()
        y_train, y_valid = y_full[train_idx].copy(), y_full[valid_idx].copy()
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train,
                  eval_set = [(X_valid, y_valid)],
                  early_stopping_rounds = 500,
                  verbose = 500)
        
        y_valid_proba[valid_idx] = model.predict_proba(X_valid)
        
        top_3_preds = np.argsort(y_valid_proba[valid_idx], axis = 1)[:, -3:][:, ::-1] 
        
        map3_score = map3(np.array(y_full), top_3_preds)
        map3_scores_all.append(map3_score)
        print(f"  Fold {fold+1} MAP@3: {map3_score:.5f}")

    mean_map3_scores_all = np.mean(map3_scores_all)
    print(f"ğŸš€ Trial {trial.number} finished with mean MAP@3: {mean_map3_scores_all:.5f}\n")
    return mean_map3_scores_all
'''


#sampler = TPESampler(seed = 42)
#study_2 = optuna.create_study(direction = "maximize", sampler = sampler)
#study_2.optimize(objective, n_trials = 100)


'''
# Optuna v1
xgb_params_1 = {'max_depth': 12,
                'colsample_bytree': 0.467,
                'subsample': 0.86,
                'n_estimators': 4000,
                'learning_rate': 0.03,
                'gamma': 0.26,
                'max_delta_step': 4,
                'reg_alpha': 2.7,
                'reg_lambda': 1.4,
                'tree_method': 'gpu_hist',
                'objective': 'multi:softprob',
                'random_state': 13,
                'enable_categorical': True,
                'device': 'gpu',
                'early_stopping_rounds': 50}

# Optuna v2 + interactive
xgb_params_2 = {'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': len(np.unique(y_full)), 
                'max_depth': 7,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'max_bin': 128,
                'colsample_bytree': 0.3, 
                'colsample_bylevel': 1,  
                'colsample_bynode': 1,  
                'tree_method': 'gpu_hist',  
                'random_state': 42,
                'device': "gpu",
                'enable_categorical': True,
                'n_estimators': 10000,
                'early_stopping_rounds': 50
               }

model_1 = XGBClassifier(**xgb_params_2)
'''


cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)


'''
map3_scores_valid_xgb = []
y_pred_val_xgb   = np.zeros((len(X_full),  len(le.classes_)))
y_pred_test_xgb  = np.zeros((len(df_test), len(le.classes_)))
'''


'''
for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, y_full)):
    print(f"\n Fold XGBBoost {fold + 1}")
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    y_train = y_full.iloc[idx_train].copy()
    y_valid = y_full.iloc[idx_valid].copy()

    X_train = pd.concat([X_train, X_original], axis = 0, ignore_index = True)
    y_train = pd.concat([y_train, y_original], axis = 0, ignore_index = True)

    model_1.fit(X_train, y_train,
                eval_set = [(X_train, y_train), (X_valid, y_valid)],
                verbose = 500)
    
    y_pred_val_xgb[idx_valid]   = model_1.predict_proba(X_valid)
    #y_pred_train_xgb[idx_train] = model_1.predict_proba(X_train) #idx_train = 150 000 lines, while X_train = 150 000 lines + 100 000 lines from orig.dataset = 250 000 lines
    y_pred_test_xgb            += model_1.predict_proba(X_test)
    
    y_true_val   = [[label] for label in y_valid]
    #y_true_train = [[label] for label in y_train]
    top_3_xgb_val_pred   = np.argsort(y_pred_val_xgb[idx_valid],   axis = 1)[:, -3:][:, ::-1]
    #top_3_xgb_train_pred = np.argsort(y_pred_train_xgb[idx_train], axis = 1)[:, -3:][:, ::-1]

    fold_map3_valid_xgb = map3(y_true_val,   top_3_xgb_val_pred)
    #fold_map3_train_xgb = map3(y_true_train, top_3_xgb_train_pred)
    map3_scores_valid_xgb.append(fold_map3_valid_xgb)
    #map3_scores_train_xgb.append(fold_map3_train_xgb)
    print(f"Fold XGBBoost {fold + 1} MAP@3 on valid data: {fold_map3_valid_xgb:.5f}")
    #print(f"Fold XGBBoost {fold + 1} MAP@3 on train data: {fold_map3_train_xgb:.5f}")
'''


'''
Overall_top_3_xgb_val_pred   = np.argsort(y_pred_val_xgb,   axis = 1)[:, -3:][:, ::-1]
#Overall_top_3_xgb_train_pred = np.argsort(y_pred_train_xgb, axis = 1)[:, -3:][:, ::-1]

overall_map3_valid_xgb = map3(np.array(y_full),  Overall_top_3_xgb_val_pred)
#overall_map3_train_xgb = map3(np.array(y_full),  Overall_top_3_xgb_train_pred)

print(f"\nğŸ�¯ Overall CV XGBBoost MAP@3 on valid data: {overall_map3_valid_xgb:.5f}")
#print(f"\nğŸ�¯ Overall CV XGBBoost MAP@3 on train data: {overall_map3_train_xgb:.5f}")

y_pred_test_xgb /= 5
'''


'''
score_fold_5_jupyter = [0.36229, 0.36117, 0.36293, 0.36169, 0.36120]
# The results were very strange in Jupyter Lab and Kaggle, since the random number generator was the same in both cases.
score_fold_5 = [0.36225, 0.36160, 0.36283, 0.36165, 0.36088]
folds = [1,2,3,4,5]

fig = go.Figure()

fig.add_trace(go.Bar(x = folds,
                     y = score_fold_5,
                     marker = dict(color = score_fold_5,
                                   colorscale = 'Rainbow',
                                   line = dict(color = 'black', width = 1))))

for i, val in enumerate(score_fold_5):
    fig.add_annotation(x = folds[i],
                       y = val + 0.0003,
                       text = f"{val:.5f}",
                       showarrow = False,
                       font = dict(size = 22))

fig.update_layout(
    title = dict(text = "<b>Validation Scores per Fold for XGBoost model</b>",
                 font = dict(size=24, family="Arial", color="#333"),
                 xanchor = 'center',
                 yanchor = 'top',
                 x = 0.5),
    xaxis = dict(title = "<b>Fold Number</b>",
                 tickmode = 'array',
                 tickvals = folds,
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    yaxis = dict(title = "<b>Score</b>",
                 range = [0.3605, 0.3635],
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    plot_bgcolor = 'white',
    height = 600,
    width = 900,
    hoverlabel = dict(bgcolor = "white",
                      font_size = 16,
                      font_family = "Arial"))
init_notebook_mode(connected=True)
py.iplot(fig)
'''


'''
results_xgb = model_1.evals_result()
plt.figure(figsize=(10,5))
plt.plot(results_xgb["validation_0"]["mlogloss"], label="Validation loss")
plt.plot(results_xgb["validation_1"]["mlogloss"], label="Training loss")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Graphics of Loss function (mlogloss) for XGBoostModels')
plt.legend();
'''


'''
feature_names = X_full.columns
importances = model_1.feature_importances_

feature_imp = pd.DataFrame({'Feature': feature_names,
                            'Value': importances})

feature_imp = feature_imp.sort_values('Value', ascending = False).head(105)
#---------------------------------------------------------------------------------
y_pred = model_1.predict_proba(X_full)
top_3_preds = np.argsort(y_pred, axis = 1)[:, -3:][:, ::-1]
map3_score = map3(np.array(y_full), top_3_preds)
print(f"MAP@3 score on full data = {round(map3_score, 5)}%")
#---------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
ax = plt.subplot()
sns.barplot(x = 'Value', 
            y = 'Feature', 
            data = feature_imp,
            palette = 'seismic',
            edgecolor = 'black',
            linewidth = 1,
            ax = ax)

plt.title('XGB Feature Importance with Optuna Params', fontsize = 16, pad = 20, fontweight = 'bold')
plt.xlabel('Feature Importance', fontsize = 12, labelpad = 10)
plt.ylabel('Features', fontsize = 12, labelpad = 10)

ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
sns.despine(left = True, bottom = True)

for i, v in enumerate(feature_imp['Value']):
    ax.text(v + 0.001, i, f'{v:.4f}', color = 'black', ha = 'left', va = 'center', fontsize = 9)

plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------
'''


'''
df_train = pd.read_csv("train.csv").drop(columns = ['id'])
df_test  = pd.read_csv("test.csv").drop(columns = ['id'])
df_orig  =  pd.read_csv("Fertilizer Prediction.csv")
sub      = pd.read_csv("sample_submission.csv")

df_original_copy = df_orig.copy()
for i in range(6):
    df_orig = pd.concat([df_orig, df_original_copy], axis = 0)

# Let's add an encoded target to the correlation matrix.
Fertilizer_Name_train = df_train['Fertilizer Name']
Fertilizer_Name_orig  = df_orig['Fertilizer Name']

le = LabelEncoder()
df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])
df_orig['Fertilizer Name']  = le.fit_transform(df_orig['Fertilizer Name'])

df_train = df_train.reset_index(drop = True)
df_orig  = df_orig.reset_index(drop = True)
df_test  = df_test.reset_index(drop = True)

print(f"df_train shape = {df_train.shape}")
print(f"df_orig shape  = {df_orig.shape}")
print(f"df_test shape  = {df_test.shape}")
print("==============================================================")

def binarization(df):
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns]
    for col in num_cols:
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
        df[f'{col}_Binned']  = df[col].astype(str).astype('category')
    return df

df_train = binarization(df_train)
df_orig  = binarization(df_orig)
df_test  = binarization(df_test)

print(f"df_train shape after binarization  = {df_train.shape}")
print(f"df_orig shape after binarization   = {df_orig.shape}")
print(f"df_test shape after binarization   = {df_test.shape}")
print("==============================================================")

label_encoder_category = LabelEncoder()
cat_cols = ['Crop Type', 'Soil Type']
for col in cat_cols:
    df_train[col] = label_encoder_category.fit_transform(df_train[col])
    df_orig[col]  = label_encoder_category.fit_transform(df_orig[col])
    df_test[col]  = label_encoder_category.fit_transform(df_test[col])
for col in cat_cols:
    df_train[col] = df_train[col].astype("category")
    df_orig[col]  = df_orig[col].astype("category")
    df_test[col]  = df_test[col].astype("category")

print(f"df_train shape after transform Crop and Soil  = {df_train.shape}")
print(f"df_orig shape after transform Crop and Soil   = {df_orig.shape}")
print(f"df_test shape after transform Crop and Soil   = {df_test.shape}")
print("==============================================================")

# OneHotEncoding
# Train Data
crop_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
soil_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)

crop_encoded = crop_encoder.fit_transform(df_train[['Crop Type']])
soil_encoded = soil_encoder.fit_transform(df_train[['Soil Type']])
crop_df = pd.DataFrame(crop_encoded, columns = crop_encoder.get_feature_names_out(['Crop Type']))
soil_df = pd.DataFrame(soil_encoded, columns = soil_encoder.get_feature_names_out(['Soil Type']))
df_train = pd.concat([df_train, crop_df, soil_df], axis = 1)
print(f"New shape df_train after OHE: {df_train.shape}")

# OneHotEncoding (0.33000 only binarization and transform category -> 0.35000)
# Test Data
crop_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
soil_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
crop_encoded = crop_encoder.fit_transform(df_test[['Crop Type']])
soil_encoded = soil_encoder.fit_transform(df_test[['Soil Type']])
crop_df = pd.DataFrame(crop_encoded, columns = crop_encoder.get_feature_names_out(['Crop Type']))
soil_df = pd.DataFrame(soil_encoded, columns = soil_encoder.get_feature_names_out(['Soil Type']))
df_test = pd.concat([df_test, crop_df, soil_df], axis = 1)
print(f"New shape df_test after OHE: {df_test.shape}")

# OneHotEncoding
# Orig Data
crop_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
soil_encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
crop_encoded = crop_encoder.fit_transform(df_orig[['Crop Type']])
soil_encoded = soil_encoder.fit_transform(df_orig[['Soil Type']])
crop_df = pd.DataFrame(crop_encoded, columns = crop_encoder.get_feature_names_out(['Crop Type']))
soil_df = pd.DataFrame(soil_encoded, columns = soil_encoder.get_feature_names_out(['Soil Type']))
df_orig = pd.concat([df_orig, crop_df, soil_df], axis = 1)
print(f"New shape df_orig after OHE: {df_orig.shape}")
print("==============================================================")

# Let's check if the column order is the same
columns_match = df_train.columns.equals(df_test.columns.append(pd.Index(['Fertilizer Name'])))
print(f"Is the column order the same: {columns_match}")
if not columns_match:
    df_train_without_FN = df_train.drop(columns=['Fertilizer Name'])
    common_columns = [col for col in df_test.columns if col in df_train_without_FN.columns]
    df_train_without_FN = df_train_without_FN[common_columns]
    df_test = df_test[common_columns]
    df_train = pd.concat([df_train_without_FN, df_train['Fertilizer Name']], axis = 1)
    print("The column order has been corrected")

# Let's check again
df_train_without_FN = df_train.drop(columns=['Fertilizer Name'])
columns_match_after_drop = df_train_without_FN.columns.equals(df_test.columns)
print(f"Is the column order the same after calories fall: {columns_match_after_drop}")
'''


'''
y_full = df_train['Fertilizer Name'].reset_index(drop = True)
X_full = df_train.drop(columns = ['Fertilizer Name']).reset_index(drop = True)
X_original = df_orig.drop(columns = ['Fertilizer Name'])
y_original = df_orig['Fertilizer Name']
X_test = df_test
'''


'''
# Optuna - only OneHotEncodeing - 100 trials - CV=0.35377 LB=0.35332
def objective(trial):
    catboost_params = {
        'iterations': trial.suggest_int("iterations", 5000, 10000, step = 100),
        'depth': trial.suggest_int("depth", 4, 12, step = 1),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log = True),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-3, 10, log = True),
        'random_strength': trial.suggest_float("random_strength", 1e-3, 10, log = True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 10.0),
        'border_count': trial.suggest_int("border_count", 32, 255),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 100, step = 1),
        'grow_policy': trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
        'loss_function': 'MultiClass',
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False,
    }

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    map3_scores_all = []
    y_valid_proba = np.zeros((len(X_full), len(le.classes_)))
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_full, y_full)):
        X_train, X_valid = X_full.iloc[train_idx].copy(), X_full.iloc[valid_idx].copy()
        y_train, y_valid = y_full.iloc[train_idx].copy(), y_full.iloc[valid_idx].copy()
        X_train = pd.concat([X_train, X_original], axis = 0, ignore_index = True)
        y_train = pd.concat([y_train, y_original], axis = 0, ignore_index = True)

        categorical_columns = ['Soil Type', 'Crop Type', 'Temparature_Binned', 'Humidity_Binned', 'Moisture_Binned', 'Nitrogen_Binned', 'Potassium_Binned', 'Phosphorous_Binned']
        
        model = CatBoostClassifier(**catboost_params, early_stopping_rounds=50)
        
        model.fit(X_train, y_train,
                  eval_set=(X_valid, y_valid),
                  cat_features = categorical_columns,
                  verbose=0)

        y_valid_proba[valid_idx] = model.predict_proba(X_valid)
        top_3_preds = np.argsort(y_valid_proba[valid_idx], axis = 1)[:, -3:][:, ::-1] 
        y_true = np.array(y_valid)
        
        map3_score = map3(y_true, top_3_preds)
        map3_scores_all.append(map3_score)
        print(f"  Fold {fold+1} MAP@3: {map3_score:.5f}")

    mean_map3_scores_all = np.mean(map3_scores_all)
    print(f"ğŸš€ Trial {trial.number} finished with mean MAP@3: {mean_map3_scores_all:.5f}\n")
    return mean_map3_scores_all
'''


#sampler = TPESampler(seed = 42)
#study_1 = optuna.create_study(direction = "maximize", sampler = sampler)
#study_1.optimize(objective, n_trials = 100)


'''
cat_params = {'iterations': 4200, 
              'depth': 9, 
              'learning_rate': 0.012311151478845978, 
              'l2_leaf_reg': 0.01182407665426616, 
              'random_strength': 0.2351783662718401, 
              'bagging_temperature': 0.8517467698288006, 
              'border_count': 129, 
              'min_data_in_leaf': 75, 
              'grow_policy': 'Depthwise',
              'loss_function': 'MultiClass',
              'task_type': 'GPU',
              'devices': '0',
              'random_seed': 42,
              'verbose': True,
              'allow_writing_files': False}

model_2 = CatBoostClassifier(**cat_params, early_stopping_rounds = 50)
'''


'''
map3_scores_valid_cat = []
y_pred_val_cat   = np.zeros((len(X_full), len(le.classes_)))
y_pred_test_cat  = np.zeros((len(df_test), len(le.classes_)))
train_metrics, valid_metrics = [], []
'''


'''
for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, y_full)):
    print(f"\n Fold CatBoost {fold + 1}")
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    y_train = y_full.iloc[idx_train].copy()
    y_valid = y_full.iloc[idx_valid].copy()

    X_train = pd.concat([X_train, X_original], axis = 0, ignore_index = True)
    y_train = pd.concat([y_train, y_original], axis = 0, ignore_index = True)

    categorical_columns = ['Soil Type', 'Crop Type', 'Temparature_Binned', 'Humidity_Binned', 'Moisture_Binned', 'Nitrogen_Binned', 'Potassium_Binned', 'Phosphorous_Binned']
    model_2.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features = categorical_columns,
                verbose = 500)
    
    results = model_2.get_evals_result()
    
    y_pred_val_cat[idx_valid] = model_2.predict_proba(X_valid)
    y_pred_test_cat += model_2.predict_proba(X_test)
    
    y_true_val   = np.array(y_valid)
    top_3_cat_val_pred   = np.argsort(y_pred_val_cat[idx_valid],   axis = 1)[:, -3:][:, ::-1]

    fold_map3_valid_cat = map3(y_true_val,   top_3_cat_val_pred)
    map3_scores_valid_cat.append(fold_map3_valid_cat)
    print(f"Fold CatBoost {fold + 1} MAP@3 on valid data: {fold_map3_valid_cat:.5f}")
'''


'''
Overall_top_3_cat_val_pred   = np.argsort(y_pred_val_cat, axis = 1)[:, -3:][:, ::-1]
#Overall_top_3_cat_train_pred = np.argsort(y_pred_train_cat, axis = 1)[:, -3:][:, ::-1]

overall_map3_valid_cat = map3(np.array(y_full),  Overall_top_3_cat_val_pred)
#overall_map3_train_cat = map3(np.array(y_full),  Overall_top_3_cat_train_pred)

print(f"\nğŸ�¯ Overall CV CatBoost MAP@3 on valid data: {overall_map3_valid_cat:.5f}")
#print(f"\nğŸ�¯ Overall CV CatBoost MAP@3 on train data: {overall_map3_train_cat:.5f}")

y_pred_test_cat /= 5
'''


'''
score_fold_5_jupyter = [0.35442, 0.35279, 0.35481, 0.35409, 0.35270]
score_fold_5 = [0.35105, 0.35018, 0.35155, 0.35073, 0.35019]
folds = [1,2,3,4,5]

fig = go.Figure()

fig.add_trace(go.Bar(x = folds,
                     y = score_fold_5,
                     marker = dict(color = score_fold_5,
                                   colorscale = 'Rainbow',
                                   line = dict(color = 'black', width = 1))))

for i, val in enumerate(score_fold_5):
    fig.add_annotation(x = folds[i],
                       y = val + 0.0003,
                       text = f"{val:.5f}",
                       showarrow = False,
                       font = dict(size = 22))

fig.update_layout(
    title = dict(text = "<b>Validation Scores per Fold for CatBoost model</b>",
                 font = dict(size=24, family="Arial", color="#333"),
                 xanchor = 'center',
                 yanchor = 'top',
                 x = 0.5),
    xaxis = dict(title = "<b>Fold Number</b>",
                 tickmode = 'array',
                 tickvals = folds,
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    yaxis = dict(title = "<b>Score</b>",
                 range = [0.35000, 0.35200],
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    plot_bgcolor = 'white',
    height = 600,
    width = 900,
    hoverlabel = dict(bgcolor = "white",
                      font_size = 16,
                      font_family = "Arial"))
init_notebook_mode(connected=True)
py.iplot(fig)
'''


'''
results_cat = model_2.get_evals_result()
plt.figure(figsize=(10,5))
plt.plot(results_cat["learn"]["MultiClass"], label="Learn - MultiClass")
plt.plot(results_cat["validation"]["MultiClass"], label = "validation - MultiClass")
plt.axvline(4200, color="gray", label="Optimal iteration")
plt.xlabel("Iterations")
plt.ylabel("MultiClass Loss")
plt.title('Graphics of Loss function (MultiClass) for CatBoost model')
plt.legend();
'''


'''
def objective(trial):
    lgbm_params = {
                    'n_estimators': trial.suggest_int("n_estimators", 3000, 6000, step = 100),
                    'max_depth': trial.suggest_int("max_depth", 6, 13, step = 1),
                    'num_leaves': trial.suggest_int("num_leaves", 10, 300, step = 10),
                    'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log = True),
                    #'reg_alpha': trial.suggest_float("reg_alpha", 1e-2, 10, log=True),
                    #'reg_lambda': trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
                    'subsample': trial.suggest_float("subsample", 0.6, 0.99),
                    'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.99),
                    #'min_child_weight': trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
                    #'min_child_samples': trial.suggest_int("min_child_samples", 1, 100),
                    'objective': 'multiclass',
                    'num_class': len(le.classes_),
                    'device': 'gpu',
                    #'gpu_platform_id': 0,
                    #'gpu_device_id': 0,
                    'random_state': 42,
                    'verbose': -1,
                    #'force_col_wise': True
                }

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    map3_scores_all = []
    y_valid_proba = np.zeros(shape = (len(X_full), y_full.nunique()))
    
    categorical_columns = ['Soil Type', 'Crop Type', 'Temparature_Binned', 
                           'Humidity_Binned', 'Moisture_Binned', 'Nitrogen_Binned', 
                           'Potassium_Binned', 'Phosphorous_Binned']

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_full, y_full)):
        X_train, X_valid = X_full.iloc[train_idx].copy(), X_full.iloc[valid_idx].copy()
        y_train, y_valid = y_full.iloc[train_idx].copy(), y_full.iloc[valid_idx].copy()

        model = LGBMClassifier(**lgbm_params)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  categorical_feature=categorical_columns,
                  callbacks=[lgb.early_stopping(50, verbose = 500)]
                )
        
        y_valid_proba[valid_idx] = model.predict_proba(X_valid)
        top_3_preds = np.argsort(y_valid_proba[valid_idx], axis=1)[:, -3:][:, ::-1] 
        
        map3_score = map3(np.array(y_valid), top_3_preds)
        map3_scores_all.append(map3_score)
        print(f"  Fold {fold+1} MAP@3: {map3_score:.5f}")

    mean_map3_scores_all = np.mean(map3_scores_all)
    print(f"ğŸš€ Trial {trial.number} finished with mean MAP@3: {mean_map3_scores_all:.5f}\n")
    return mean_map3_scores_all
'''


#sampler = TPESampler(seed = 42)
#study_10 = optuna.create_study(direction = "maximize", sampler = sampler)
#study_10.optimize(objective, n_trials = 25)


'''
lgbm_params = {'n_estimators': 4400, 
               'max_depth': 12, 
               'num_leaves': 60, 
               'learning_rate': 0.018785426399210624, 
               'subsample': 0.8310416818561965, 
               'colsample_bytree': 0.6181156609607991,
               'objective': 'multiclass',
               'num_class': len(le.classes_),
               'device': 'gpu', 
               'random_state': 42,
               'verbose': -1}

model_3 = LGBMClassifier(**lgbm_params)
'''


'''
map3_scores_valid_lgbm = []
y_pred_val_lgbm   = np.zeros((len(X_full), len(le.classes_)))
y_pred_test_lgbm  = np.zeros((len(df_test), len(le.classes_)))
'''


'''
for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, y_full)):
    
    print(f"\n Fold LightGBM {fold + 1}")
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    y_train = y_full.iloc[idx_train].copy()
    y_valid = y_full.iloc[idx_valid].copy()

    X_train = pd.concat([X_train, X_original], axis = 0, ignore_index = True)
    y_train = pd.concat([y_train, y_original], axis = 0, ignore_index = True)

    categorical_columns = ['Soil Type', 'Crop Type', 'Temparature_Binned', 
                           'Humidity_Binned', 'Moisture_Binned', 'Nitrogen_Binned', 
                           'Potassium_Binned', 'Phosphorous_Binned']
    
    model_3.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                categorical_feature=categorical_columns,
                callbacks=[lgb.early_stopping(50, verbose = 500)])
    
    y_pred_val_lgbm[idx_valid] = model_3.predict_proba(X_valid)
    y_pred_test_lgbm += model_3.predict_proba(X_test)
    
    y_true_val = np.array(y_valid)
    top_3_lgbm_val_pred   = np.argsort(y_pred_val_lgbm[idx_valid],   axis = 1)[:, -3:][:, ::-1]

    fold_map3_valid_lgbm = map3(y_true_val,   top_3_lgbm_val_pred)
    map3_scores_valid_lgbm.append(fold_map3_valid_lgbm)
    print(f"Fold LightGBM {fold + 1} MAP@3 on valid data: {fold_map3_valid_lgbm:.5f}")
'''


'''
Overall_top_3_lgbm_val_pred   = np.argsort(y_pred_val_lgbm, axis = 1)[:, -3:][:, ::-1]

overall_map3_valid_lgbm = map3(np.array(y_full),  Overall_top_3_lgbm_val_pred)

print(f"\nğŸ�¯ Overall CV LightGBM MAP@3 on valid data: {overall_map3_valid_lgbm:.5f}")

y_pred_test_lgbm /= 5
'''


'''
score_fold_5 = [0.37243, 0.37324, 0.37333, 0.37114, 0.37261]
folds = [1,2,3,4,5]

fig = go.Figure()

fig.add_trace(go.Bar(x = folds,
                     y = score_fold_5,
                     marker = dict(color = score_fold_5,
                                   colorscale = 'Rainbow',
                                   line = dict(color = 'black', width = 1))))

for i, val in enumerate(score_fold_5):
    fig.add_annotation(x = folds[i],
                       y = val + 0.0003,
                       text = f"{val:.5f}",
                       showarrow = False,
                       font = dict(size = 22))

fig.update_layout(
    title = dict(text = "<b>Validation Scores per Fold for LightGBM model</b>",
                 font = dict(size=24, family="Arial", color="#333"),
                 xanchor = 'center',
                 yanchor = 'top',
                 x = 0.5),
    xaxis = dict(title = "<b>Fold Number</b>",
                 tickmode = 'array',
                 tickvals = folds,
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    yaxis = dict(title = "<b>Score</b>",
                 range = [0.37000, 0.37500],
                 title_font = dict(size=18),
                 gridcolor = 'lightgray'),
    plot_bgcolor = 'white',
    height = 600,
    width = 900,
    hoverlabel = dict(bgcolor = "white",
                      font_size = 16,
                      font_family = "Arial"))
init_notebook_mode(connected=True)
py.iplot(fig)
'''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from keras import initializers
from keras import regularizers


def build_model_optuna(hyperparams, input_shape, num_classes = 7):
    
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    n_layers = hyperparams.suggest_int("n_layers", 3, 8, 1)

    for i in range(n_layers):
        n_units    = hyperparams.suggest_int(f"units_{i}", 16, 512, step = 10)
        activation = hyperparams.suggest_categorical(f"activation_{i}", ["relu", "tanh"])
        '''
        if i % 2 == 1:
            l1_reg = hyperparams.suggest_float(f"l1_reg_{i}", 1e-2, 50, log = True)
            l2_reg = hyperparams.suggest_float(f"l2_reg_{i}", 1e-2, 50, log = True)
            kernel_reg = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        else:
            kernel_reg = None
        '''
        model.add(layers.Dense(units=n_units,
                               activation=activation))
        '''
        if (i == 2 and n_layers == 5) or \
           (i == 3 and n_layers == 6) or \
           ((i == 2 or i == 4) and n_layers == 7) or \
           ((i == 3 or i == 5) and n_layers == 8):
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        '''
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optim = hyperparams.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = hyperparams.suggest_float("learning_rate", 1e-4, 0.5, log = True)
    
    if optim == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    
    return model


#strategy = tf.distribute.MirroredStrategy()
#print('DEVICES AVAILABLE: {}'.format(strategy.num_replicas_in_sync))


'''
def objective(trial):
    map3_scores_all = []
    y_valid_proba = np.zeros((len(X_full), len(le.classes_)))
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_full, y_full)):
        X_train, X_valid = X_full.iloc[train_idx].copy(), X_full.iloc[valid_idx].copy()
        y_train, y_valid = y_full.iloc[train_idx].copy(), y_full.iloc[valid_idx].copy()

        # Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes = 7)
        y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes = 7)

        model = build_model_optuna(trial, X_full.shape[1])
        
        model.fit(X_train, y_train_onehot,
                  epochs = 50, 
                  validation_data = [X_valid, y_valid_onehot], 
                  batch_size = 1024,
                  verbose = 0)
        
        y_valid_proba[valid_idx] = model.predict(X_valid)
        top_3_preds = np.argsort(y_valid_proba[valid_idx], axis = 1)[:, -3:][:, ::-1] 
        y_true = np.array(y_valid)
        
        map3_score = map3(y_true, top_3_preds)
        map3_scores_all.append(map3_score)
        print(f"  Fold {fold+1} MAP@3: {map3_score:.5f}")

    mean_map3_scores_all = np.mean(map3_scores_all)
    print(f"ğŸš€ Trial {trial.number} finished with mean MAP@3: {mean_map3_scores_all:.5f}\n")
    return mean_map3_scores_all
'''


#sampler = TPESampler(seed = 42)
#study_3 = optuna.create_study(direction = "maximize", sampler = sampler)
#study_3.optimize(objective, n_trials = 100)


'''
# CV = 0.29600 (without - reg, dropout)
def Evaluate_Optuna_Model(hyperparams, input_shape, num_classes = 7):
    
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    n_layers = hyperparams.get("n_layers", 4)
    
    for i in range(n_layers):
        n_units = hyperparams.get(f"units_{i}", 64)
        activation = hyperparams.get(f"activation_{i}", "relu")
        model.add(layers.Dense(units=n_units, activation=activation))

    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optim = hyperparams.get("optimizer", "adam")
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    
    optimizer = tf.keras.optimizers.get(optim)
    optimizer.learning_rate = learning_rate
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics = ["accuracy"])
    return model
'''


'''
def fit_history(X, y, folds = 5):
    cv_GSF = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    history = []
    for i, (train, val) in enumerate(cv_GSF.split(X_full, y_full)):
        
        print(f"Fold NN FIT = {i + 1}")
        
        best_hyperparams = {'n_layers': 4, 
                            'units_0': 460, 'activation_0': 'tanh', 
                            'units_1': 462, 'activation_1': 'relu', 
                            'units_2': 128, 'activation_2': 'tanh', 
                            'units_3': 444, 'activation_3': 'tanh', 
                            'optimizer': 'adam', 
                            'learning_rate': 0.0002288738114460098}
        
        model = Evaluate_Optuna_Model(best_hyperparams, X_full.shape[1])
        history.append(model.fit(X_full.iloc[train].copy(), tf.keras.utils.to_categorical(y_full[train].copy(), num_classes = 7), 
                                 epochs = 50, 
                                 validation_data = [X_full.iloc[val].copy(), tf.keras.utils.to_categorical(y_full[val].copy(), num_classes = 7)], 
                                 batch_size = 1024, 
                                 verbose = 1))
    return history
'''


'''
best_hyperparams = {'n_layers': 4, 
                    'units_0': 460, 'activation_0': 'tanh', 
                    'units_1': 462, 'activation_1': 'relu', 
                    'units_2': 128, 'activation_2': 'tanh', 
                    'units_3': 444, 'activation_3': 'tanh', 
                    'optimizer': 'adam', 
                    'learning_rate': 0.0002288738114460098}

model4 = Evaluate_Optuna_Model(best_hyperparams, X_full.shape[1])
model4.summary()
history = fit_history(X_full, y_full, folds = 5)
'''


#history = history[0]
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(loss) + 1)


'''
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label = 'Training loss - categorical_crossentropy')
plt.plot(epochs, val_loss, label = 'Validation loss - categorical_crossentropy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
'''


'''
def fit_pred(X, y, folds = 5):
    
    map3_scores_valid_nn = []
    map3_scores_train_nn = []
    y_pred_val_nn   = np.zeros((len(X_full), len(le.classes_)))
    y_pred_train_nn = np.zeros((len(X_full), len(le.classes_)))
    y_pred_test_nn  = np.zeros((len(df_test), len(le.classes_)))
    
    cv_GSF = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    # ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ¾Ğ±Ğ½Ğ¾Ğ²Ğ»Ñ�ĞµÑ‚ Ğ²ĞµÑ�Ğ° Ğ½Ğ° ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¼ Ñ„Ğ¾Ğ»Ğ´Ğµ, Ğ° Ğ½Ğµ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ¸Ñ…
    best_hyperparams = {'n_layers': 4, 
                        'units_0': 460, 'activation_0': 'tanh', 
                        'units_1': 462, 'activation_1': 'relu', 
                        'units_2': 128, 'activation_2': 'tanh', 
                        'units_3': 444, 'activation_3': 'tanh', 
                        'optimizer': 'adam', 
                        'learning_rate': 0.0002288738114460098}
    
    for i, (idx_train, idx_valid) in enumerate(cv_GSF.split(X_full, y_full)):

        print(f"Fold NN PREDICT = {i + 1}")
        
        model = Evaluate_Optuna_Model(best_hyperparams, X_full.shape[1])
        model.fit(X_full.iloc[idx_train].copy(), tf.keras.utils.to_categorical(y_full[idx_train].copy(), num_classes = 7), 
                  epochs = 50, 
                  validation_data = [X_full.iloc[idx_valid].copy(), tf.keras.utils.to_categorical(y_full[idx_valid].copy(), num_classes = 7)], 
                  batch_size = 1024, 
                  verbose = 1)

        y_pred_val_nn[idx_valid] = model.predict(X_full.iloc[idx_valid].copy())
        y_pred_train_nn[idx_train] = model.predict(X_full.iloc[idx_train].copy())
        y_pred_test_nn += model.predict(X_test)
    
        y_true_val   = np.array(y_full[idx_valid].copy())
        y_true_train = np.array(y_full[idx_train].copy())
        top_3_nn_val_pred   = np.argsort(y_pred_val_nn[idx_valid],   axis = 1)[:, -3:][:, ::-1]
        top_3_nn_train_pred = np.argsort(y_pred_train_nn[idx_train], axis = 1)[:, -3:][:, ::-1]

        fold_map3_valid_nn = map3(y_true_val,   top_3_nn_val_pred)
        fold_map3_train_nn = map3(y_true_train, top_3_nn_train_pred)
        map3_scores_valid_nn.append(fold_map3_valid_nn)
        map3_scores_train_nn.append(fold_map3_train_nn)
        print(f"Fold NN {fold + 1} MAP@3 on valid data: {fold_map3_valid_nn:.5f}")
        print(f"Fold NN {fold + 1} MAP@3 on train data: {fold_map3_train_nn:.5f}")  

    y_pred_test_nn /= 5
    
    return (y_pred_test_nn, y_pred_val_nn, y_pred_train_nn)
'''


#y_pred_test_nn, y_pred_val_nn, y_pred_train_nn = fit_pred(X_full, y_full, folds = 5)


from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


'''
result_Y_valid = y_full

lgbm_arg_val = np.argsort(y_pred_val_lgbm, axis=1)[:, -3:][:, ::-1]
lgbm_inverse_val = le.inverse_transform(lgbm_arg_val.ravel()).reshape(lgbm_arg_val.shape)

cat_arg_val = np.argsort(y_pred_val_cat, axis=1)[:, -3:][:, ::-1]
cat_inverse_val = le.inverse_transform(cat_arg_val.ravel()).reshape(cat_arg_val.shape)

xgb_arg_val = np.argsort(y_pred_val_xgb, axis=1)[:, -3:][:, ::-1]
xgb_inverse_val = le.inverse_transform(xgb_arg_val.ravel()).reshape(xgb_arg_val.shape)

lgbm_df = pd.DataFrame({'LGBM':  [' '.join(row) for row in lgbm_inverse_val]})
cat_df  = pd.DataFrame({'Cat' :  [' '.join(row) for row in cat_inverse_val ]})
xgb_df  = pd.DataFrame({'XGB' :  [' '.join(row) for row in xgb_inverse_val ]})

result_X_valid = pd.concat([lgbm_df, cat_df, xgb_df], axis = 1)

lgbm_arg_test = np.argsort(y_pred_test_lgbm, axis=1)[:, -3:][:, ::-1]
lgbm_inverse_test = le.inverse_transform(lgbm_arg_test.ravel()).reshape(lgbm_arg_test.shape)

cat_arg_test = np.argsort(y_pred_test_cat, axis=1)[:, -3:][:, ::-1]
cat_inverse_test = le.inverse_transform(cat_arg_test.ravel()).reshape(cat_arg_test.shape)

xgb_arg_test = np.argsort(y_pred_test_xgb, axis=1)[:, -3:][:, ::-1]
xgb_inverse_test = le.inverse_transform(xgb_arg_test.ravel()).reshape(xgb_arg_test.shape)

lgbm_df_test = pd.DataFrame({'LGBM':  [' '.join(row) for row in lgbm_inverse_test]})
cat_df_test  = pd.DataFrame({'Cat' :  [' '.join(row) for row in cat_inverse_test ]})
xgb_df_test  = pd.DataFrame({'XGB' :  [' '.join(row) for row in xgb_inverse_test ]})

result_X_test = pd.concat([lgbm_df_test, cat_df_test, xgb_df_test], axis = 1)
'''


#result_X_test


'''
def objective(trial):

    lr_params = {'penalty': trial.suggest_categorical('penalty', ['l1']),
                 'C': trial.suggest_float('C', 1e-4, 50, log = True),
                 'solver': 'saga',
                 'max_iter': trial.suggest_int('max_iter', 100, 1000, step = 50),
                 'l1_ratio': trial.suggest_float('l1_ratio', 0.001, 1), #if trial.params['penalty'] == 'elasticnet' else None,
                 'random_state': 42,
                 'n_jobs': -1,
                 'multi_class': 'multinomial'}
    
    # Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ l1_ratio ĞµÑ�Ğ»Ğ¸ Ğ½Ğµ elasticnet
    #if lr_params['penalty'] != 'elasticnet':
    #    lr_params.pop('l1_ratio', None)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    map3_scores_all = []
    y_valid_proba = np.zeros(shape=(len(y_pred_val_xgb), y_full.nunique()))

    # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), result_X_valid.columns)],
                                     remainder='passthrough')

    for fold, (train_idx, valid_idx) in enumerate(cv.split(result_X_valid, result_Y_valid)):
        X_train, X_valid = result_X_valid.iloc[train_idx], result_X_valid.iloc[valid_idx]
        y_train, y_valid = result_Y_valid.iloc[train_idx], result_Y_valid.iloc[valid_idx]
        
        X_train_encoded = preprocessor.fit_transform(X_train)
        X_valid_encoded = preprocessor.transform(X_valid)
        
        model = LogisticRegression(**lr_params)
        model.fit(X_train_encoded, y_train)
        
        y_valid_proba[valid_idx] = model.predict_proba(X_valid_encoded)
        top_3_preds = np.argsort(y_valid_proba[valid_idx], axis=1)[:, -3:][:, ::-1]
        
        map3_score = map3(np.array(y_valid), top_3_preds)
        map3_scores_all.append(map3_score)
        print(f"  Fold {fold+1} for LogisticRegression - MAP@3: {map3_score:.5f}")

    mean_map3_scores_all = np.mean(map3_scores_all)
    print(f"ğŸš€ Trial {trial.number} finished with mean MAP@3: {mean_map3_scores_all:.5f}\n")
    return mean_map3_scores_all
'''


#sampler = TPESampler(seed = 42)
#study_20 = optuna.create_study(direction = "maximize", sampler = sampler)
#study_20.optimize(objective, n_trials = 50)


'''
map3_scores_valid_LR = []
y_pred_val_LR   = np.zeros((len(result_X_valid), len(le.classes_)))
y_pred_test_LR  = np.zeros((len(df_test), len(le.classes_)))
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

lr_params = {'penalty': 'l1', 
             'C': 0.2580695379659472, 
             'max_iter': 200, 
             'l1_ratio': 0.15683852581586644,
             'solver': 'saga',
             'random_state': 42,
             'n_jobs': -1,
             'multi_class': 'multinomial'}

preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), result_X_valid.columns)], remainder='passthrough')
X_full_encoded = preprocessor.fit_transform(result_X_valid)
X_test_encoded = preprocessor.transform(result_X_test)

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full_encoded, result_Y_valid)):
    
    print(f"\n Fold LogRegression {fold + 1}")
    X_train = X_full_encoded[idx_train].copy()
    X_valid = X_full_encoded[idx_valid].copy()
    y_train = result_Y_valid[idx_train].copy()
    y_valid = result_Y_valid[idx_valid].copy()

    final_model = LogisticRegression(**lr_params)
    final_model.fit(X_train, y_train)
    
    y_pred_val_LR[idx_valid] = final_model.predict_proba(X_valid)
    y_pred_test_LR          += final_model.predict_proba(X_test_encoded)
    
    y_true_val = np.array(y_valid)
    top_3_lgbm_val_pred   = np.argsort(y_pred_val_LR[idx_valid],   axis = 1)[:, -3:][:, ::-1]

    fold_map3_valid_LR = map3(y_true_val,   top_3_lgbm_val_pred)
    map3_scores_valid_LR.append(fold_map3_valid_LR)
    print(f"Fold LogRegression {fold + 1} MAP@3 on valid data: {fold_map3_valid_LR:.5f}")
'''


#y_pred_test_LR /= 5


#top_3_preds_test_data = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
#top_3_labels_test_data = le.inverse_transform(top_3_preds_test_data.ravel()).reshape(top_3_preds_test_data.shape)
#submission = pd.DataFrame({'id': sub['id'],
#                           'Fertilizer Name': [' '.join(row) for row in top_3_labels_test_data]})
final_sub = pd.read_csv('/kaggle/input/formula-map3/submission_LR_2.csv') #!!!!in the last hours of the competition, I don't have time to fully debug the Kaggle code.
final_sub.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")

