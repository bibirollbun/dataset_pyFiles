import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error, confusion_matrix, accuracy_score, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import kde
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Title
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from math import pi
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


import warnings
warnings.filterwarnings('ignore')



# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
OD = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)
print("Original data shape:", OD.shape)


train_df.head().style.background_gradient(cmap='gist_rainbow_r')


OD.head().style.background_gradient(cmap='gist_rainbow')


train_df = pd.concat([train_df, OD], ignore_index=True)


test_df.head().style.background_gradient(cmap='gist_rainbow_r')


sample_submission.head().style.background_gradient(cmap= 'Spectral_r')


train_df.describe().style.background_gradient(cmap='YlOrBr_r')


test_df.describe().style.background_gradient(cmap='Purples_r')


train_df.info()


train_df.isnull().sum()


# Helper function for consistent plot styling
def set_plot_style(ax, title, xlabel, ylabel, title_fontsize=16, label_fontsize=14, grid=True):
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_facecolor('#f0f0f0')  # Set a light background color

# Helper function to print results with styling
def print_styled(text, color='blue', bold=True):
    style = f"color: {color};"
    if bold:
        style += "font-weight: bold;"
    print(f"\n\033[{style}m{text}\033[0m") # ANSI escape codes for styling



# 1. Density Plot
plt.figure(figsize=(12, 6))
print("Nitrogen Summary Stats:\n", train_df['Nitrogen'].describe())  # Print result
sns.kdeplot(train_df['Nitrogen'], shade=True, color='skyblue')
set_plot_style(plt.gca(), 'Nitrogen (N) Density', 'Nitrogen Level', 'Density')
plt.show()

# 2. Burtin-Pie Chart
fertilizer_counts = train_df['Fertilizer Name'].value_counts()
print("Fertilizer Counts:\n", fertilizer_counts)  # Print result
plt.figure(figsize=(8, 8))
plt.pie(fertilizer_counts, labels=fertilizer_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Fertilizer Distribution', fontsize=16, fontweight='bold')
plt.show()

# 3. Stacked Area Chart
soil_crop = train_df.groupby(['Soil Type', 'Crop Type']).size().unstack()
print("Soil-Crop Group Counts:\n", soil_crop.fillna(0).astype(int))  # Print result
soil_crop.plot(kind='area', figsize=(12, 6))
set_plot_style(plt.gca(), 'Crop Distribution by Soil Type', 'Soil Type', 'Count')
plt.show()

# 4. Enhanced Scatter Plot (N vs P colored by Fertilizer)
print("Sample of N vs P vs Fertilizer:\n", train_df[['Nitrogen', 'Phosphorous', 'Fertilizer Name']].head())  # Print result
plt.figure(figsize=(12, 7))
palette = sns.color_palette("husl", n_colors=train_df['Fertilizer Name'].nunique())
sns.scatterplot(
    x='Nitrogen',
    y='Phosphorous',
    hue='Fertilizer Name',
    palette=palette,
    data=train_df,
    s=100,
    edgecolor='black',
    alpha=0.7
)
plt.title('Nitrogen vs Phosphorous Colored by Fertilizer Type', fontsize=18, fontweight='bold')
plt.xlabel('Nitrogen Level', fontsize=14, fontweight='bold')
plt.ylabel('Phosphorous Level', fontsize=14, fontweight='bold')
plt.legend(title='Fertilizer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_facecolor('#f7f7f7')
plt.tight_layout()
plt.show()

# 5. Heatmap (Correlation Matrix)
correlation_matrix = train_df[['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity', 'Moisture']].corr()
print("Correlation Matrix:\n", correlation_matrix)  # Print result
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.show()

# 6. Stacked Split Bar Chart
soil_crop = train_df.groupby(['Soil Type', 'Crop Type']).size().unstack()
print("Stacked Bar Data:\n", soil_crop.fillna(0).astype(int))  # Print result

# Plotting using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(soil_crop))
for crop in soil_crop.columns:
    ax.bar(soil_crop.index, soil_crop[crop], bottom=bottom, label=crop)
    bottom += soil_crop[crop]

set_plot_style(ax, 'Stacked Split Bar Chart: Crop Distribution by Soil Type', 'Soil Type', 'Count')
ax.legend(title='Crop Type')
plt.show()



# Feature Encoding
cat_cols = train_df.select_dtypes(include=['object','category']).columns.drop('Fertilizer Name')
oe = OrdinalEncoder()
train_df[cat_cols] = oe.fit_transform(train_df[cat_cols])
test_df[cat_cols] = oe.transform(test_df[cat_cols])

# Encoding
le = LabelEncoder()
train_df['Fertilizer Name'] = le.fit_transform(train_df['Fertilizer Name'])

# Add 'Fertilizer Name' column to test_df and fill with a placeholder value
test_df['Fertilizer Name'] = -1  # Or any other suitable placeholder
test_df['Fertilizer Name'] = test_df['Fertilizer Name'].astype(int)

for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif df[col].dtype == 'float64':
            df[col]= df[col].astype('float32') # Using float32 instead of float16

# Crucially: Check for NaN and inf
print("NaN values in train_df:", train_df.isnull().sum().sum())
print("inf values in train_df:", np.isinf(train_df).values.sum())
print("NaN values in test_df:", test_df.isnull().sum().sum())
print("inf values in test_df:", np.isinf(test_df).values.sum())



# Replace inf with a large value BEFORE imputing NaN
max_finite_value = np.nanmax(train_df[train_df != np.inf])
train_df.replace([np.inf, -np.inf], max_finite_value, inplace=True)
test_df.replace([np.inf, -np.inf], max_finite_value, inplace=True)

# Imputation (Replace NaN with mean)
imputer = SimpleImputer(strategy='mean')
train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
test_df = pd.DataFrame(imputer.transform(test_df), columns=train_df.columns) # Use train_df.columns here


# Verify again:
print("NaN values in train_df after imputation:", train_df.isnull().sum().sum())
print("inf values in train_df after replacement:", np.isinf(train_df).values.sum())
print("NaN values in test_df after imputation:", test_df.isnull().sum().sum())
print("inf values in test_df after replacement:", np.isinf(test_df).values.sum())


X = train_df.drop('Fertilizer Name', axis=1)
y = train_df['Fertilizer Name']


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


xgb_model = XGBClassifier(
    max_depth=17,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=10000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',
    device='cuda',
    missing= np.nan # Very important!  Set 'missing'
)

FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train_df) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test_df),y.nunique()))

feature_importances = pd.DataFrame(index=X.columns)  # Initialize DataFrame for feature importances

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    xgb_model.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 0)
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob +=xgb_model.predict_proba(test_df.drop('Fertilizer Name', axis = 1)) / FOLDS # Average the predictions.  Very important!

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f" FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")
    
    # Feature Importance Analysis for EACH fold
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    fold_importance = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
    fold_importance = fold_importance.set_index('Feature')
    feature_importances[f'Fold_{i+1}'] = fold_importance['Importance']


# Aggregate Feature Importances
feature_importances['Mean'] = feature_importances.mean(axis=1)
feature_importances = feature_importances.sort_values(by='Mean', ascending=False)

# Plotting Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Mean', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances', fontsize=16, fontweight='bold')
plt.xlabel('Mean Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.show()




top_indices = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1] # Get top 3 indices
top_fertilizers = le.inverse_transform(top_indices.ravel()).reshape(top_indices.shape)  

sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")  

# create the submission
submission = pd.DataFrame({'id': sample_sub['id'],
                           'Fertilizer Name': [' '.join(row) for row in top_fertilizers]})


submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
submission.head()





