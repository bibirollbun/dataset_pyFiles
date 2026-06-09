# Core data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import colorcet as cc  # Advanced color gradients

# Jupyter display utilities
from IPython.display import display, Markdown, HTML

# Statistical functions
from scipy.stats import skew

# Plotly for interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

# Machine learning and preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")



train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
original_data = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


# Stylish section headers
display(Markdown("<h2 style='color:#4facfe;'>ğŸ“¦ <b>Dataset Shapes</b></h2>"))
print(f"Train Data Shape: {train_data.shape}")
print(f"Original Data Shape: {original_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

display(Markdown("<h2 style='color:#43e97b;'>ğŸ”� <b>Data Previews</b></h2>"))
display(Markdown("**Train Data (last 5 rows):**"))
display(train_data.tail())

display(Markdown("**Original Data (first 5 rows):**"))
display(original_data.head())

display(Markdown("**Test Data (first 5 rows):**"))
display(test_data.head())

display(Markdown("<h2 style='color:#38f9d7;'>ğŸ“� <b>DataFrame Info</b></h2>"))
print("Train Data Info:")
train_data.info()
print('----------------------------------------------------')
print("Original Data Info:")
original_data.info()
print('----------------------------------------------------')
print("Test Data Info:")
test_data.info()


import seaborn as sns

import matplotlib.pyplot as plt

# Define the columns to plot (numerical features)
plot_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# 3ï¸�âƒ£ Feature Impact by Fertilizer Type (Seaborn version)
for col in plot_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=original_data, x='Fertilizer Name', y=col, palette='Set2')
    plt.title(f'Boxplot of {col} by Fertilizer Name', fontsize=16, fontweight='bold')
    plt.xlabel('Fertilizer Name')
    plt.ylabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=original_data, x='Fertilizer Name', y=col, palette='Set2', inner='box')
    plt.title(f'Violin Plot of {col} by Fertilizer Name', fontsize=16, fontweight='bold')
    plt.xlabel('Fertilizer Name')
    plt.ylabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()



def print_unique_values(df, categorical_columns, dataset_name="Dataset"):
    print("\n" + "=" * 50)
    print(f"Unique values in {dataset_name} categorical features")
    print("=" * 50)
    for col in categorical_columns:
        unique_vals = sorted(df[col].unique())
        value_counts = df[col].value_counts()
        top_value = value_counts.index[0]
        top_freq = value_counts.iloc[0]
        print(f"{col} - Number of unique values: {len(unique_vals)}")
        print(f"Unique values: {unique_vals}")
        print(f"Top value: '{top_value}' (Frequency: {top_freq})\n")

train_cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
original_cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
test_cat_cols = ['Soil Type', 'Crop Type']

print_unique_values(train_data, train_cat_cols, "Train Data")
print_unique_values(original_data, original_cat_cols, "Original Data")
print_unique_values(test_data, test_cat_cols, "Test Data")


import pandas as pd
import plotly.graph_objects as go

def data_quality_report(df_dict):
    summary_data = []

    for name, df in df_dict.items():
        missing = df.isnull().sum().sum()
        dups = df.duplicated().sum()
        total = len(df)

        summary_data.append({
            "Dataset": name,
            "Total Rows": f"{total:,}",
            "Missing Values": f"{missing:,}",
            "Duplicates": f"{dups:,}",
            "Duplicate %": f"{(dups / total):.2%}" if total > 0 else "0.00%"
        })

    summary_df = pd.DataFrame(summary_data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color='gray',
            font=dict(color='white', size=14),
            align='center'
        ),
        cells=dict(
            values=[summary_df[col] for col in summary_df.columns],
            fill_color='lightgray',
            align='center',
            font=dict(size=13)
        )
    )])

    fig.update_layout(
        title="Data Quality Report Summary",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )

    fig.show()

    return summary_df



# Drop 'id' column from train_data to align columns
train_data_aligned = train_data.drop(columns=['id'])

# Concatenate train_data_aligned and original_data
combined_data = pd.concat([train_data_aligned, original_data], ignore_index=True)

print(f"Combined dataset shape: {combined_data.shape}")



# Sample data (replace with your actual DataFrame)
fertilizers = ['Urea', 'DAP', 'Potash', 'NPK', 'Compost']
np.random.seed(42)
train_data = pd.DataFrame({
    'Fertilizer Name': np.random.choice(fertilizers, 100),
})
original_data = train_data.copy()

# ğŸŒˆ Color-rich, crystal-clear dashboard function
def create_fertilizer_distribution(data, title):
    counts = data['Fertilizer Name'].value_counts().reset_index()
    counts.columns = ['Fertilizer', 'Count']
    
    turbo_colors = pc.sequential.Turbo[:len(counts)]
    bold_colors = pc.qualitative.Bold[:len(counts)+1]  # +1 for root node

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{title} - Count", f"{title} - Sunburst"),
        specs=[[{"type": "bar"}, {"type": "domain"}]]
    )

    # ğŸ“Š Bar Chart (Horizontal)
    fig.add_trace(go.Bar(
        y=counts['Fertilizer'],
        x=counts['Count'],
        orientation='h',
        marker=dict(color=counts['Count'], colorscale='Turbo'),
        text=counts['Count'],
        textposition='outside',
        insidetextfont=dict(color='white'),
        outsidetextfont=dict(color='white'),
        name='Fertilizer Count'
    ), row=1, col=1)

    # ğŸŒ� Sunburst Chart
    fig.add_trace(go.Sunburst(
        labels=["Fertilizers"] + counts['Fertilizer'].tolist(),
        parents=[""] + ["Fertilizers"] * len(counts),
        values=[counts['Count'].sum()] + counts['Count'].tolist(),
        branchvalues='total',
        marker=dict(colors=bold_colors),
        textinfo="label+percent entry",
        insidetextfont=dict(size=14, color='white')
    ), row=1, col=2)

    # ğŸ§­ Layout and styling
    fig.update_layout(
        title={
            'text': f"ğŸŒ± Fertilizer Distribution Dashboard â€“ {title}",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=24, color='white', family='Arial Black')
        },
        template='plotly_dark',
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='white'),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig

# ğŸš€ Run & Display
fig_train = create_fertilizer_distribution(train_data, "Train Data")
fig_original = create_fertilizer_distribution(original_data, "Original Data")
fig_train.show()
fig_original.show()



# âš™ï¸� Setup
target_variable = 'Fertilizer Name'
categorical_features = ['Soil Type', 'Crop Type']
label_encoders = {}

# ğŸ�¯ Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    combined_data[col + '_Encoded'] = le.fit_transform(combined_data[col])
    label_encoders[col] = le
    
    # Encode test set with fallback
    test_data[col + '_Encoded'] = test_data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# ğŸ�¯ Encode target variable
target_le = LabelEncoder()
combined_data[target_variable + '_Encoded'] = target_le.fit_transform(combined_data[target_variable])

# ğŸ§¹ Drop original categorical columns
combined_data.drop(columns=categorical_features, inplace=True)
test_data.drop(columns=categorical_features, inplace=True)

# ğŸ§¹ Drop original target column
combined_data.drop(columns=[target_variable], inplace=True)

# Print the encoded DataFrame
print("Encoded Combined DataFrame:")
print(combined_data.head())


# ğŸ§¹ Remove whitespace in column names
combined_data.columns = combined_data.columns.str.replace(' ', '_', regex=True)
test_data.columns = test_data.columns.str.replace(' ', '_', regex=True)


# Select only numerical features for correlation
numerical_features = combined_data.select_dtypes(include=['int64', 'int32']).columns.tolist()

corr_matrix = combined_data[numerical_features].corr().round(2)

plt.figure(figsize=(14, 10))

# Use a different colorcet colormap, e.g., cc.m_rainbow or cc.m_bmy for a new look
cmap = cc.m_rainbow  # Try cc.m_bmy, cc.m_coolwarm, etc.

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap=cmap,
    fmt='.2f',
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    annot_kws={"size": 11, "weight": "bold", "color": "#222"}
)

plt.title('Correlation Heatmap of Numerical Features', fontsize=18, fontweight='bold', color='#1b1b1b', pad=22)
plt.xticks(rotation=45, ha='right', fontsize=12, color='#1b1b1b')
plt.yticks(rotation=0, fontsize=12, color='#1b1b1b')
plt.gca().set_facecolor('#f7fafd')
plt.tight_layout()
plt.show()



# Select features and target
feature_columns = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
    'Soil_Type_Encoded', 'Crop_Type_Encoded'
]
target_column = 'Fertilizer_Name_Encoded'

# Set Training and Test Dataset (no scaling)
X = combined_data[feature_columns]
y = combined_data[target_column]
test_X = test_data[feature_columns]



import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

# Split your data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class_labels = np.unique(y_train)

# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                score += 1.0 / (i + 1.0)
                break
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Optional: Sklearn-compatible XGBoost (not used but kept for flexibility)
class SklearnCompatibleXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define models with GPU usage and limited complexity
models = [
    ('XGBoost', xgb.XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        n_estimators=100,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=42
    ))
]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models:
    print(f"\nğŸ”� Evaluating {name}")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('model', model)
    ])

    fold_log_losses = []
    fold_map3_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)

        y_proba = pipeline.predict_proba(X_val)
        y_pred_top3 = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
        y_pred_top3_labels = class_labels[y_pred_top3]

        # Log Loss
        ll = log_loss(y_val, y_proba, labels=class_labels)
        fold_log_losses.append(ll)

        # MAP@3
        m3 = mapk(y_val, y_pred_top3_labels, k=3)
        fold_map3_scores.append(m3)

        print(f"  Fold {fold_idx} - Log Loss: {ll:.4f}, MAP@3: {m3:.4f}")

    print(f"\nâœ… Average Log Loss: {np.mean(fold_log_losses):.4f}")
    print(f"âœ… Average MAP@3: {np.mean(fold_map3_scores):.4f}")



# Extract feature importances from the trained model (last fold)
feature_importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

plt.figure(figsize=(12, 6))
import plotly.express as px

fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Viridis',
    title='Feature Importance (Last Fold XGBoost Model)',
    height=500
)
fig.update_layout(
    xaxis_title='Importance Score',
    yaxis_title='Feature',
    coloraxis_colorbar=dict(title='Importance'),
    plot_bgcolor='#f7f7f7'
)
fig.show()




# Fertilizer decoder
fertilizer_decoder = dict(zip(target_le.transform(target_le.classes_), target_le.classes_))

# Generate test_preds using the trained pipeline and test_X
test_preds = pipeline.predict_proba(test_X)

test_top3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]

submission_labels = []
for row in test_top3_preds:
    labels = [fertilizer_decoder[i] for i in row]
    submission_labels.append(' '.join(labels))

submission_df = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': submission_labels
})
submission_df.to_csv('my_submission.csv', index=False)
print("\nSubmission file saved!")


# Print Submission preview
print(submission_df.head(10))

