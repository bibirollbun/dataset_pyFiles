# Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import plotly.io as pio
from IPython.core.display import display, HTML
from plotly.subplots import make_subplots
import random
# Set the default renderer for both Plotly Express and Graph Objects
pio.renderers.default = 'iframe_connected'
# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Enhanced styled table function
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "#ffffff"), ("background-color", "#2e7d32")]}
    ]).set_properties(**{
        "text-align": "center",
        "font-size": "14px",
        "color": "#013305",
        "border": "1px solid #c8e6c9"
    }).hide(axis="index")
    return styled_df.to_html()

# Fancy animated gradient header
def styled_heading(text, source_url=None):
    dataset_link = f'<p style="margin-top: 10px;"><a href="{source_url}" target="_blank" style="color:#013305; font-size: 15px; text-decoration: none;"><b>ğŸ”— View Original Dataset</b></a></p>' if source_url else ""
    return f"""
    <div style="
        background: linear-gradient(270deg, #d4ff52, #85ff8e, #d4ff52);
        background-size: 600% 600%;
        animation: gradientBG 6s ease infinite;
        color: #013305;
        font-family: 'Segoe UI', sans-serif;
        font-size: 30px;
        text-align: center;
        padding: 24px 16px;
        border-radius: 16px;
        width: 85%;
        margin: 20px auto;
        font-weight: bold;
        text-shadow: 1px 1px 2px #A5D6A7;
        border: 4px solid #013305;
        box-shadow: 0 0 12px #A5D6A7;
    ">
        {text}
        {dataset_link}
    </div>

    <style>
    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    """

# Fancy sub-section card
def subheader(text):
    return f"""
    <div style="
        font-size: 18px;
        color: #013305;
        background: #f1f8e9;
        border-left: 6px solid #81c784;
        padding: 12px;
        margin-top: 20px;
        font-weight: bold;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    ">{text}</div>
    """

# Function to print styled dataset analysis
def print_dataset_analysis(dataset, dataset_name, dataset_url=None, n_top=5):
    display(HTML(styled_heading(f"ğŸ“Š {dataset_name} Overview", dataset_url)))

    display(HTML(subheader("ğŸ“� Dataset Shape")))
    display(HTML(f"<p style='color:#013305; font-size:16px;'>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>"))

    display(HTML(subheader("ğŸ”� First Few Rows")))
    display(HTML(style_table(dataset.head(n_top))))

    display(HTML(subheader("ğŸ“Š Summary Statistics")))
    display(HTML(style_table(dataset.describe())))

    display(HTML(subheader("ğŸ”§ Null Values")))
    null_counts = dataset.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if null_columns.sum() == 0:
        display(HTML("<p style='font-size: 16px; color: #388e3c;'>âœ… No null values found.</p>"))
    else:
        null_df = pd.DataFrame({'Column': null_columns.index, 'Missing Values': null_columns.values})
        display(HTML(style_table(null_df)))

    display(HTML(subheader("â™»ï¸� Duplicate Rows")))
    duplicate_count = dataset.duplicated().sum()
    color = "#e65100" if duplicate_count > 0 else "#388e3c"
    msg = f"{duplicate_count} duplicate rows found." if duplicate_count > 0 else "âœ… No duplicate rows."
    display(HTML(f"<p style='font-size: 16px; color: {color};'>{msg}</p>"))

    display(HTML(subheader("ğŸ—‚ï¸� Data Types")))
    dtype_df = pd.DataFrame({'Column Name': dataset.columns, 'Data Type': dataset.dtypes.values})
    display(HTML(style_table(dtype_df)))

    display(HTML(subheader("ğŸ“‹ Column Names")))
    display(HTML(f"<p style='font-size: 16px; color: #013305;'>{', '.join(dataset.columns)}</p>"))

    display(HTML(subheader("ğŸ”¢ Unique Values Per Column")))
    unique_df = pd.DataFrame({
        'Column Name': dataset.columns,
        'Unique Sampled Values': [
            ', '.join(map(str, dataset[col].unique()[:7])) + ('...' if dataset[col].nunique() > 7 else '')
            for col in dataset.columns
        ]
    })
    display(HTML(style_table(unique_df)))

# Load datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv") 
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
original_data = pd.read_csv("/kaggle/input/fertilizers-original-dataset/Fertilizer Prediction.csv")

print_dataset_analysis(train_data, "Training Data", "/kaggle/input/playground-series-s5e6/train.csv")
print_dataset_analysis(test_data, "Test Data", "/kaggle/input/playground-series-s5e6/test.csv")
print_dataset_analysis(sample_sub, "Sample Submission", "/kaggle/input/playground-series-s5e6/sample_submission.csv")
print_dataset_analysis(original_data, "Original Data", "/kaggle/input/fertilizers-original-dataset/Fertilizer Prediction.csv")



# Set style
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.titleweight': 'bold',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.figsize': (22, 18)
})

# Custom color palettes
soil_colors = ['#d4a373', '#b08968', '#7f5539', '#9c6644', '#ddb892']
crop_colors = ['#588157', '#3a5a40', '#a3b18a', '#dad7cd', '#344e41']
fertilizer_colors = ['#ffb703', '#fb8500', '#8ecae6', '#219ebc', '#023047', '#ff006e', '#8338ec', '#3a86ff']

# Subplot layout for bar plots only
fig, axes = plt.subplots(2, 2, figsize=(22, 14))
fig.suptitle("Fertilizer Dataset â€“ Categorical Feature Distributions", fontsize=24, weight='bold', color='#1a1a1a')

# 1. Crop Type Distribution
crop_order = train_data['Crop Type'].value_counts().index
sns.countplot(x='Crop Type', data=train_data, order=crop_order, palette=crop_colors, ax=axes[0, 0])
axes[0, 0].set_title("Popular Crop Types", fontsize=18, weight='bold', color='#3a5a40')
axes[0, 0].tick_params(axis='x', rotation=30, width=2)

for p in axes[0, 0].patches:
    height = p.get_height()
    axes[0, 0].text(p.get_x() + p.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', fontsize=12, fontweight='bold')

# 2. Soil Type Distribution
soil_order = train_data['Soil Type'].value_counts().index
sns.countplot(x='Soil Type', data=train_data, order=soil_order, palette=soil_colors, ax=axes[0, 1])
axes[0, 1].set_title("Soil Type Frequency", fontsize=18, weight='bold', color='#9c6644')
axes[0, 1].tick_params(axis='x', rotation=30, width=2)

for p in axes[0, 1].patches:
    height = p.get_height()
    axes[0, 1].text(p.get_x() + p.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', fontsize=12, fontweight='bold')

# 3. Fertilizer Usage
fert_order = train_data['Fertilizer Name'].value_counts().index
sns.countplot(x='Fertilizer Name', data=train_data, order=fert_order, palette=fertilizer_colors, ax=axes[1, 0])
axes[1, 0].set_title("Most Used Fertilizers", fontsize=18, weight='bold', color='#8338ec')
axes[1, 0].tick_params(axis='x', rotation=45, width=2)

for p in axes[1, 0].patches:
    height = p.get_height()
    axes[1, 0].text(p.get_x() + p.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', fontsize=12, fontweight='bold')

axes[1, 1].axis('off')

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Define features and colors
features = ['Temparature', 'Humidity', 'Moisture']
color_palettes = [
    ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22'],
    ['#16a085', '#c0392b', '#2980b9', '#f39c12', '#8e44ad', '#27ae60', '#d35400'],
    ['#1abc9c', '#e84393', '#00cec9', '#fdcb6e', '#6c5ce7', '#00b894', '#e17055']
]

fertilizers = list(train_data['Fertilizer Name'].unique())
fert_to_x = {fert: i for i, fert in enumerate(fertilizers)}

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=[f"{feature} Distribution by Fertilizer" for feature in features],
    vertical_spacing=0.15
)

# Plotting
for i, feature in enumerate(features):
    row_index = i + 1
    palette = color_palettes[i]

    for j, fert in enumerate(fertilizers):
        fert_data = train_data[train_data['Fertilizer Name'] == fert][feature].dropna()
        mean_val = fert_data.mean()
        median_val = fert_data.median()
        color = palette[j % len(palette)]

        x_box = fert_to_x[fert]
        x_scatter = x_box - 0.35  # <-- SHIFT LEFT

        # Box Plot
        fig.add_trace(
            go.Box(
                y=fert_data,
                x=[x_box] * len(fert_data),
                name=fert,
                boxpoints=False,
                marker_color=color,
                showlegend=False
            ),
            row=row_index, col=1
        )

        # Mean Annotation
        fig.add_annotation(
            x=x_box, y=mean_val,
            text=f"Mean: {mean_val:.2f}",
            showarrow=False, yshift=10,
            font=dict(size=11, color='black'),
            xanchor='center',
            row=row_index, col=1
        )

        # Median Annotation
        fig.add_annotation(
            x=x_box, y=median_val,
            text=f"Median: {median_val:.2f}",
            showarrow=False, yshift=-15,
            font=dict(size=11, color='darkblue'),
            xanchor='center',
            row=row_index, col=1
        )

        # Sample for scatter
        fert_sample = fert_data.sample(100, random_state=0) if len(fert_data) > 100 else fert_data

        # Scatter Dots to Left
        fig.add_trace(
            go.Scatter(
                x=np.random.normal(loc=x_scatter, scale=0.05, size=len(fert_sample)),
                y=fert_sample,
                mode='markers',
                marker=dict(size=4, color='rgba(0,0,0,0.4)', opacity=0.5),
                hovertext=[f"{feature}: {y:.3f}" for y in fert_sample],
                hoverinfo='text',
                showlegend=False
            ),
            row=row_index, col=1
        )

    # Custom x-axis labels
    fig.update_xaxes(
        tickvals=list(range(len(fertilizers))),
        ticktext=fertilizers,
        title_text="Fertilizer Name",
        row=row_index, col=1
    )

    fig.update_yaxes(title_text=feature, row=row_index, col=1)

# Layout
fig.update_layout(
    height=1200,
    width=950,
    title_text="ğŸŒ± Temperature, Humidity & Moisture Impact Across Fertilizers",
    title_font_size=24,
    plot_bgcolor='white',
    font=dict(family="Arial", size=12),
    showlegend=False
)

fig.show()



# Features and grouping categories
features = ['Temparature', 'Humidity', 'Moisture']
groupings = ['Soil Type', 'Crop Type']

# Create a subplot grid
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        f"{feat} by {group}" for group in groupings for feat in features
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.07
)

# Color palette
colors = ['#1abc9c', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#e67e22', '#16a085', '#0b8720', '#87810b', '#41b32b', '#b35d2b']

# Loop for each subplot
for row_idx, group_col in enumerate(groupings, start=1):
    for col_idx, feature in enumerate(features, start=1):
        grouped = train_data.groupby(group_col)[feature].mean().reset_index()
        grouped = grouped.sort_values(by=feature, ascending=False)

        hover_texts = [
            f"{group_col}: {grouped[group_col].iloc[i]}<br>{feature}: {grouped[feature].iloc[i]:.2f}"
            for i in range(len(grouped))
        ]

        fig.add_trace(
            go.Bar(
                x=grouped[group_col],
                y=grouped[feature],
                marker_color=colors[:len(grouped)],
                text=[f"{v:.2f}" for v in grouped[feature]],
                textposition='auto',
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False
            ),
            row=row_idx, col=col_idx
        )

# Layout styling
fig.update_layout(
    height=800,
    width=1100,
    title_text="ğŸŒ¿Temperature, Humidity & Moisture impact on Soil & Crops",
    title_font_size=24,
    title_font_color='#2c3e50',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial", size=12)
)

# Axis labels
for i in range(3):
    fig.update_xaxes(title_text=groupings[0], row=1, col=i+1)
    fig.update_xaxes(title_text=groupings[1], row=2, col=i+1)
    fig.update_yaxes(title_text=features[i], row=1, col=i+1)
    fig.update_yaxes(title_text=features[i], row=2, col=i+1)

fig.show()



# Features to compare
nutrients = ['Nitrogen', 'Phosphorous', 'Potassium']

# Create subplot
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Top Crop Types by Avg {nutrient}" for nutrient in nutrients],
    horizontal_spacing=0.08
)

# Color palette (3 distinct colors)
colors = ['#16a085', '#f39c12', '#8e44ad']

# Add bar charts for each nutrient
for i, nutrient in enumerate(nutrients):
    # Group by Crop Type, compute mean, sort descending, take top 10
    top_crops = train_data.groupby('Crop Type')[nutrient].mean().sort_values(ascending=False).head(10)
    
    fig.add_trace(
        go.Bar(
            x=top_crops.index,
            y=top_crops.values,
            marker_color=colors[i],
            text=[f"{v:.2f}" for v in top_crops.values],
            textposition='auto',
            hovertext=[
                f"Crop Type: {crop}<br>{nutrient}: {val:.2f}" 
                for crop, val in zip(top_crops.index, top_crops.values)
            ],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=i+1
    )

# Final layout adjustments
fig.update_layout(
    height=450,
    width=1200,
    title_text="ğŸŒ¾ Nutrient Levels by Top Crop Types Visualization",
    title_font_size=22,
    title_font_color='#2c3e50',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial", size=12),
    margin=dict(t=60, b=40, l=40, r=40)
)

# Update labels
for i in range(3):
    fig.update_xaxes(title_text="Crop Type", tickangle=45, row=1, col=i+1)
    fig.update_yaxes(title_text=nutrients[i], row=1, col=i+1)

fig.show()



# Nutrients and environmental factors
nutrients = ['Nitrogen', 'Phosphorous', 'Potassium']
env_features = ['Temparature', 'Humidity', 'Moisture']

# Copy data and bin environmental features
binned_data = train_data.copy()
for col in env_features:
    binned_data[f"{col}_bin"] = pd.cut(binned_data[col], bins=15)

# Color mapping for nutrients
color_map = {
    'Nitrogen': '#1abc9c',
    'Phosphorous': '#f39c12',
    'Potassium': '#8e44ad'
}

# Create subplot grid
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        f"{nutrient} vs {env}" for nutrient in nutrients for env in env_features
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.07
)

# Add traces for each (nutrient Ã— environmental feature)
for i, nutrient in enumerate(nutrients):
    for j, env in enumerate(env_features):
        # Group and average
        grouped = binned_data.groupby(f"{env}_bin")[nutrient].mean().reset_index()
        bin_midpoints = grouped[f"{env}_bin"].apply(lambda x: x.mid)

        fig.add_trace(
            go.Scatter(
                x=bin_midpoints,
                y=grouped[nutrient],
                mode='lines+markers',
                line=dict(color=color_map[nutrient], width=2),
                marker=dict(size=6),
                name=f"{nutrient} vs {env}" if (i == 0 and j == 0) else None,
                hovertemplate=f"{env}: %{{x:.1f}}<br>{nutrient}: %{{y:.2f}}"
            ),
            row=i+1, col=j+1
        )

        # Axis labels
        fig.update_xaxes(title_text=f"{env}", row=i+1, col=j+1)
        fig.update_yaxes(title_text=f"{nutrient}", row=i+1, col=j+1)

# Layout update
fig.update_layout(
    height=1000,
    width=1200,
    title_text="ğŸ“ˆ Nutrient Trends by Environmental Factors Visualization",
    title_font_size=22,
    title_font_color='#2c3e50',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial", size=12),
    showlegend=False,
    margin=dict(t=60, b=40, l=40, r=40)
)

fig.show()



# Data Augmentation 
def augment_original_data(data, multiplier=7):
    """Augment external data by replicating it multiple times"""
    augmented_data = data.copy()
    original_copy = data.copy()
    
    for _ in range(multiplier):
        augmented_data = pd.concat([augmented_data, original_copy], axis=0, ignore_index=True)
    
    return augmented_data

external_augmented = augment_original_data(original_data)
print("Augmented external data shape:", external_augmented.shape)


# Feature Engineering
def create_advanced_features(df):
    """Create sophisticated features for better model performance"""
    df_enhanced = df.copy()
    
    # Original numerical columns
    numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
    
    # Create binned features (categorical representation)
    for feature in numeric_features:
        df_enhanced[f'{feature}_binned'] = df_enhanced[feature].astype(str)
    
    # Create interaction features
    df_enhanced['NPK_ratio'] = df_enhanced['Nitrogen'] / (df_enhanced['Phosphorous'] + df_enhanced['Potassium'] + 1e-5)
    df_enhanced['PK_ratio'] = df_enhanced['Phosphorous'] / (df_enhanced['Potassium'] + 1e-5)
    df_enhanced['temp_humidity_interaction'] = df_enhanced['Temparature'] * df_enhanced['Humidity']
    df_enhanced['moisture_npk_sum'] = df_enhanced['Moisture'] + df_enhanced['Nitrogen'] + df_enhanced['Phosphorous'] + df_enhanced['Potassium']
    
    # Polynomial features for key nutrients
    df_enhanced['nitrogen_squared'] = df_enhanced['Nitrogen'] ** 2
    df_enhanced['phosphorous_squared'] = df_enhanced['Phosphorous'] ** 2
    df_enhanced['potassium_squared'] = df_enhanced['Potassium'] ** 2
    
    return df_enhanced

# Apply feature engineering
train_enhanced = create_advanced_features(train_data)
test_enhanced = create_advanced_features(test_data)
external_enhanced = create_advanced_features(external_augmented)

print("Enhanced train shape:", train_enhanced.shape)


# Encoding Categorical Features
categorical_columns = [col for col in train_enhanced.select_dtypes(include=["object",'category']).columns if col != "Fertilizer Name"] + \
                     [col for col in train_enhanced.columns if col.endswith("_binned")]

print("Categorical columns:", categorical_columns)

# Encode categorical features
for column in categorical_columns:
    combined_values = pd.concat([train_enhanced[column], test_enhanced[column], external_enhanced[column]]).unique()
    encoder = LabelEncoder().fit(combined_values)
    
    train_enhanced[column] = encoder.transform(train_enhanced[column])
    test_enhanced[column] = encoder.transform(test_enhanced[column])
    external_enhanced[column] = encoder.transform(external_enhanced[column])


# Target Encoding
target_encoder = LabelEncoder()
combined_targets = pd.concat([train_enhanced["Fertilizer Name"], external_enhanced["Fertilizer Name"]])
target_encoder.fit(combined_targets)

train_enhanced["Fertilizer Name"] = target_encoder.transform(train_enhanced["Fertilizer Name"])
external_enhanced["Fertilizer Name"] = target_encoder.transform(external_enhanced["Fertilizer Name"])

print("Number of unique fertilizers:", len(target_encoder.classes_))


# Prepare Training Data
for column in categorical_columns:
    train_enhanced[column] = train_enhanced[column].astype("category")
    test_enhanced[column] = test_enhanced[column].astype("category")
    external_enhanced[column] = external_enhanced[column].astype("category")

# Define features and targets
feature_columns = train_enhanced.drop(["id", "Fertilizer Name"], axis=1)
target_column = train_enhanced["Fertilizer Name"]
test_features = test_enhanced.drop("id", axis=1)
external_features = external_enhanced.drop(["Fertilizer Name"], axis=1)
external_targets = external_enhanced["Fertilizer Name"]
test_identifiers = test_enhanced["id"]

print("Feature columns shape:", feature_columns.shape)
print("Test features shape:", test_features.shape)


# Define MAP@3 Evaluation Metric
def calculate_map3(actual_labels, predicted_probs, k=3):
    """Calculate Mean Average Precision at k=3"""
    def average_precision_k(actual, predicted, k):
        predicted = predicted[:k]
        score = 0.0
        num_hits = 0
        seen_predictions = set()
        
        for i, prediction in enumerate(predicted):
            if prediction in actual and prediction not in seen_predictions:
                num_hits += 1
                score += num_hits / (i + 1.0)
                seen_predictions.add(prediction)
        
        return score / min(len(actual), k)
    
    return np.mean([average_precision_k(actual, predicted, k) for actual, predicted in zip(actual_labels, predicted_probs)])



# LightGBM Model Configuration
lgb_parameters = {
    'objective': 'multiclass',
    'num_class': len(np.unique(target_column)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 8000,
    'early_stopping_rounds': 100,
    'categorical_feature': categorical_columns
}

print("LightGBM parameters configured")


# Cross-Validation
NUM_FOLDS = 7
stratified_kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize prediction arrays
out_of_fold_predictions = np.zeros(shape=(len(train_enhanced), target_column.nunique()))
test_predictions = np.zeros(shape=(len(test_enhanced), target_column.nunique()))
fold_map3_scores = []

print("Starting cross-validation training...")

for fold_idx, (train_indices, valid_indices) in enumerate(stratified_kfold.split(feature_columns, target_column)):
    print(f"{'='*20} FOLD {fold_idx+1} {'='*20}")
    
    # Split data
    X_train_fold, X_valid_fold = feature_columns.iloc[train_indices], feature_columns.iloc[valid_indices]
    y_train_fold, y_valid_fold = target_column.iloc[train_indices], target_column.iloc[valid_indices]
    
    # Augment with external data
    X_train_augmented = pd.concat([X_train_fold, external_features], axis=0, ignore_index=True)
    y_train_augmented = pd.concat([y_train_fold, external_targets], axis=0, ignore_index=True)
    
    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train_augmented, y_train_augmented, categorical_feature=categorical_columns)
    valid_dataset = lgb.Dataset(X_valid_fold, y_valid_fold, categorical_feature=categorical_columns, reference=train_dataset)
    
    # Train model
    model = lgb.train(
        lgb_parameters,
        train_dataset,
        valid_sets=[train_dataset, valid_dataset],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(100)]
    )
    
    # Generate predictions
    out_of_fold_predictions[valid_indices] = model.predict(X_valid_fold, num_iteration=model.best_iteration)
    test_predictions += model.predict(test_features, num_iteration=model.best_iteration)
    
    # Calculate MAP@3 for this fold
    top3_predictions = np.argsort(out_of_fold_predictions[valid_indices], axis=1)[:, -3:][:, ::-1]
    actual_labels = [[label] for label in y_valid_fold]
    fold_map3 = calculate_map3(actual_labels, top3_predictions)
    fold_map3_scores.append(fold_map3)
    
    print(f"âœ… FOLD {fold_idx+1} MAP@3 Score: {fold_map3:.5f}")



# Final Results
# Average test predictions across folds
test_predictions /= NUM_FOLDS

# Calculate overall MAP@3
overall_map3 = np.mean(fold_map3_scores)
print(f"\nğŸ�¯ Overall MAP@3 Score: {overall_map3:.5f}")

# Generate top-3 predictions for submission
top3_indices = np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1]
top3_fertilizer_names = target_encoder.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)



# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_identifiers,
    'Fertilizer Name': [' '.join(row) for row in top3_fertilizer_names]
})

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission saved as 'submission.csv'")


# Save prediction arrays for future use
np.save('lgb_oof_predictions.npy', out_of_fold_predictions)
np.save('lgb_test_predictions.npy', test_predictions)
print("âœ… Prediction arrays saved")

# Display submission preview
print("\nğŸ“„ Submission Preview:")
display(submission_df.head(10))
print("=====================================================")
print(f"\nSubmission shape: {submission_df.shape}")

