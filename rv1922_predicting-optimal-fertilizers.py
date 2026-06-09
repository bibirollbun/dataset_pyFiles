import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import log_loss
import plotly.io as pio
import plotly.subplots as sp
import time
import plotly.figure_factory as ff  
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from category_encoders import TargetEncoder
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import warnings
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.head()


train.info()


train.describe().T


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())

print("Number of Rows:",train.shape[0])

print("Number of Columns:",train.shape[1])


num_col = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", num_col.tolist())


cat_col = train.select_dtypes(include=['object']).columns
print("Categorical Column Names:", cat_col.tolist())


cat_col =  ['Soil Type', 'Crop Type', 'Fertilizer Name']
num_col = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'


train['Soil Type'].value_counts()


soil_counts = train['Soil Type'].value_counts().reset_index()
soil_counts.columns = ['Soil Type', 'count']

fig = px.bar(
    soil_counts,
    x='Soil Type',
    y='count',
    color='Soil Type',
    title='Distribution of Soil Types',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(
    xaxis_title='Soil Type',
    yaxis_title='Count',
    showlegend=False,
    template='simple_white',
    width=600,
    height=400
)

fig.show()


train['Crop Type'].value_counts()


crop_counts = train['Crop Type'].value_counts().reset_index()
crop_counts.columns = ['Crop Type', 'count']

fig = px.bar(
    crop_counts,
    x='Crop Type',
    y='count',
    color='Crop Type',
    title='Distribution of Crop Types',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(
    xaxis_title='Crop Type',
    yaxis_title='Count',
    showlegend=False,
    template='simple_white',
    width=700,
    height=500
)

fig.show()


train['Fertilizer Name'].value_counts()


ferti_counts = train['Fertilizer Name'].value_counts().reset_index()
ferti_counts.columns = ['Fertilizer Name', 'count']

fig = px.bar(
    ferti_counts,
    x='Fertilizer Name',
    y='count',
    color= 'Fertilizer Name',
    title='Distribution of Fertilizer Name',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(
    xaxis_title='Fertilizer Name',
    yaxis_title='Count',
    showlegend=False,
    template='simple_white',
    width=600,
    height=400
)

fig.show()


for col in num_col:
    stats = train[col].describe().round(2)  
    print(f"--- {col} ---")
    print(stats.T) 
    print()  


fig = sp.make_subplots(rows=3, cols=2, subplot_titles=num_col)

colors = px.colors.qualitative.Bold

for i, col in enumerate(num_col):
    row = i // 2 + 1
    col_pos = i % 2 + 1
    fig.add_trace(
        go.Histogram(
            x=train[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=800,
    width=800,
    title_text='Distribution of Numerical Features',
    showlegend=False,
    template='simple_white'
)

fig.show()


fig = sp.make_subplots(rows=3, cols=2, subplot_titles=num_col)
colors = px.colors.qualitative.Bold

for i, col in enumerate(num_col):
    row = i // 2 + 1
    col_pos = i % 2 + 1

    fig.add_trace(
        go.Box(
            y=train[num_col].dropna(),
            name=col,
            marker_color=colors[i % len(colors)],
            boxpoints='outliers'  
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=800,
    width=800,
    title_text='Distribution of Numerical Features (Box Plots)',
    showlegend=False,
    template='simple_white'
)

fig.show()


fig_soil = px.histogram(
    train,
    x='Soil Type',
    color=target_col,
    barmode='group',
    title='Fertilizer Distribution by Soil Type',
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_soil.update_layout(
    xaxis_title='Soil Type',
    yaxis_title='Count',
    template='simple_white',
    width=800,
    height=500
)

fig_soil.show()


df_norm = (
    train.groupby(['Crop Type', target_col])
    .size()
    .reset_index(name='Count')
)

df_norm['Proportion'] = df_norm.groupby('Crop Type')['Count'].transform(lambda x: x / x.sum())

fig_crop = px.bar(
    df_norm,
    x='Crop Type',
    y='Proportion',
    color=target_col,
    title='Normalized Fertilizer Distribution by Crop Type',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig_crop.update_layout(
    barmode='stack',
    xaxis_title='Crop Type',
    yaxis_title='Proportion',
    template='simple_white',
    width=800,
    height=500
)

fig_crop.show()


soil_crop_counts = pd.crosstab(
    train["Soil Type"],
    train["Crop Type"]
)
print("=== Raw Counts: Soil Type vs. Crop Type ===")
print(soil_crop_counts)


crop_fert_counts = pd.crosstab(
    train["Crop Type"],
    train["Fertilizer Name"]
)

print("\n=== Raw Counts: Crop Type vs. Fertilizer Name ===")
print(crop_fert_counts)


corr_matrix = train[num_col].corr(method='pearson')

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",              
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,              
    aspect="auto",
    labels={'color': 'Pearson r'},
    title="Numeric Feature Correlation Heatmap"
)

fig.update_layout(
    width=600,
    height=600
)

fig.show()


for df in [train, test]:
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif df[col].dtype == 'float64':
            df[col]= df[col].astype('float16')


train.head()


target_col = "Fertilizer Name"
X = train.drop(columns=[target_col])
y = train[target_col]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


def mapk(actual, predicted, k=3):
    """
    Calculate Mean Average Precision at K (MAP@K)
    
    This metric evaluates how well our model ranks the correct fertilizer
    recommendations in the top K predictions.
    """
    def apk(a, p, k):
        """Average Precision at K for a single sample"""
        p = p[:k]  # Take only top k predictions
        score = 0.0
        hits = 0
        seen = set()
        
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)  # Precision at position i+1
                seen.add(pred)
        return score / min(len(a), k)
    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


print("ğŸ”„ Preparing target variable...")
le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])  
X = train.drop('Fertilizer Name', axis=1)


test_copy = test.copy()


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
    enable_categorical=False,
    tree_method='hist',
    device='cuda'
)


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(train), y.max() + 1))
pred_prob = np.zeros((len(test), y.max() + 1))

target_enc_cols = ['Soil Type', 'Crop Type']

print("ğŸš€ Starting cross-validation training...")
for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f'########## Fold {i+1} ##########')
    
    x_train, x_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    te = TargetEncoder(cols=target_enc_cols, smoothing=10, min_samples_leaf=100)
    x_train[target_enc_cols] = te.fit_transform(x_train[target_enc_cols], y_train)
    x_valid[target_enc_cols] = te.transform(x_valid[target_enc_cols])
    
    test_enc = test_copy.copy()
    test_enc[target_enc_cols] = te.transform(test_copy[target_enc_cols])
    
    print(f"Training model for fold {i+1}...")
    xgb_model.fit(
        x_train, y_train, 
        eval_set=[(x_valid, y_valid)], 
        verbose=0
    )
    
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob += xgb_model.predict_proba(test_enc)
    
    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")

print("ğŸ“ˆ Averaging predictions across folds...")
pred_prob /= FOLDS

print("ğŸ�¯ Calculating overall CV performance...")
oof_top_3_preds = np.argsort(oof, axis=1)[:, -3:][:, ::-1]
oof_actual = [[label] for label in y]
overall_map3 = mapk(oof_actual, oof_top_3_preds)
print(f"Overall CV MAP@3 Score: {overall_map3:.5f}")


print("ğŸ“� Preparing submission file...")

top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]

top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission_df = pd.DataFrame({
    'id': test['id'] if 'id' in test.columns else range(len(test)),
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


print("\nğŸ“‹ First 10 rows of submission:")
print(submission_df.head(10))

