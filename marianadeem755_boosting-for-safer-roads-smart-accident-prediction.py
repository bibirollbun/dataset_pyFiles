# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import random, io
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# ignore warnings
import warnings
import warnings
warnings.filterwarnings("ignore")  


# Ignore All Deprecation / Future / User Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
# Fix Missing Emoji / Glyph Warnings
plt.rcParams['axes.unicode_minus'] = False
# fallback font with broad Unicode support
plt.rcParams['font.family'] = 'DejaVu Sans'  
 # preload fonts to reduce glyph issues
fm.findSystemFonts(fontpaths=None, fontext='ttf') 



# TABLE STYLING
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("color", "#EAB308"),  # Lime yellow for header text
            ("background-color", "#0B1023"),
            ("font-weight", "bold"),
            ("font-size", "14px"),
            ("padding", "10px"),
            ("border", "1px solid #06B6D4"),
            ("text-shadow", "0 0 6px #EAB308")  # Glow effect for header
        ]},
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("color", "#E2E8F0"),
            ("background-color", "#111827"),
            ("border", "1px solid #1E3A8A"),
            ("padding", "8px")
        ]}
    ]).hide(axis="index")
    return styled_df.to_html()

# RANDOM COLOR
def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(120, 255),
        random.randint(100, 255),
        random.randint(120, 255)
    )

# STYLED HEADINGS
def styled_heading(text, gradient, text_color='white', border_color='#EAB308', font_size='25px'):
    return f"""
    <div style="
        background: {gradient};
        background-size: 300% 300%;
        animation: gradientMove 6s ease infinite;
        text-align: center;
        color: {text_color};
        padding: 22px 40px;
        border-radius: 25px;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        text-transform: uppercase;
        font-size: {font_size};
        letter-spacing: 3px;
        border: 3px solid {border_color};
        box-shadow: 0px 0px 20px rgba(234, 179, 8, 0.6);
        margin: 25px 0;
    ">
        {text}
    </div>
    <style>
        @keyframes gradientMove {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    </style>
    """

# SECTION HEADER
def section_header(title, gradient):
    return f"""
    <div style="
        background: {gradient};
        color: white;
        text-align: center;
        padding: 10px 25px;
        border-radius: 12px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 22px;
        margin-top: 15px;
        box-shadow: 0px 4px 12px rgba(147, 51, 234, 0.6);
    ">
        {title}
    </div>
    """

# MAIN FUNCTION
def print_dataset_analysis(dataset, dataset_name, n_top=5, palette_index=0):
    main_gradients = [
        "linear-gradient(90deg, #0B1023, #1E3A8A, #9333EA, #F97316)",
        "linear-gradient(90deg, #1E3A8A, #06B6D4, #EAB308)",
        "linear-gradient(90deg, #9333EA, #F97316, #EAB308)"
    ]
    sub_gradients = [
        "linear-gradient(90deg, #1E3A8A, #9333EA)",
        "linear-gradient(90deg, #F97316, #EAB308)",
        "linear-gradient(90deg, #06B6D4, #1E3A8A)"
    ]

    heading_gradient = main_gradients[palette_index % len(main_gradients)]
    sub_gradient = sub_gradients[palette_index % len(sub_gradients)]

    # MAIN HEADING
    display(HTML(styled_heading(f"🚀 {dataset_name} Overview", heading_gradient)))

    # SHAPE
    display(HTML(section_header("📏 Shape of the Dataset", sub_gradient)))
    display(HTML(f"""
    <div style='
        color:#FACC15;
        font-family:Poppins;
        font-size:17px;
        font-weight:600;
        text-align:center;
        background:#0B1023;
        border:1px solid #1E3A8A;
        padding:10px;
        border-radius:10px;
        margin:10px 0;
        box-shadow:0 0 10px rgba(250,204,21,0.4);
    '>
        {dataset.shape[0]} <span style='color:#06B6D4;'>rows</span> × {dataset.shape[1]} <span style='color:#9333EA;'>columns</span>
    </div>
    """))

    # FIRST 5 ROWS
    display(HTML(section_header("👀 First 5 Rows", sub_gradient)))
    display(HTML(style_table(dataset.head(n_top))))

    # SUMMARY STATISTICS
    display(HTML(section_header("📊 Summary Statistics", sub_gradient)))
    display(HTML(style_table(dataset.describe())))

    # NULL VALUES
    display(HTML(section_header("🚨 Null Values", sub_gradient)))
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML("<p style='color:#10B981;font-weight:600;'>✅ No null values found.</p>"))
    else:
        null_df = null_counts[null_counts > 0].to_frame(name='Null Values')
        null_df['Column'] = null_df.index
        display(HTML(style_table(null_df)))

    # DUPLICATE ROWS
    display(HTML(section_header("🔁 Duplicate Rows", sub_gradient)))
    duplicate_count = dataset.duplicated().sum()
    display(HTML(f"<p style='color:#FACC15;font-weight:600;'>{duplicate_count} duplicate rows found.</p>"))

    # DATA TYPES
    display(HTML(section_header("🔢 Data Types", sub_gradient)))
    dtypes_df = pd.DataFrame({
        'Column': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    })
    display(HTML(style_table(dtypes_df)))

    # COLUMN NAMES
    display(HTML(section_header("📋 Column Names", sub_gradient)))
    column_html = ", ".join([f"<span style='color:#EAB308; font-weight:600; text-shadow:0 0 6px #EAB308;'>{col}</span>" for col in dataset.columns])
    display(HTML(f"<div style='font-family:Poppins;font-size:15px;margin:10px 0;'>{column_html}</div>"))

    # UNIQUE VALUES
    display(HTML(section_header("🔍 Unique Values", sub_gradient)))
    unique_values_df = pd.DataFrame({
        'Column': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Unique Values': [dataset[col].nunique() for col in dataset.columns]
    })
    display(HTML(style_table(unique_values_df)))

    # DATASET INFO
    display(HTML(section_header("ℹ️ Dataset Info", sub_gradient)))
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    info = buffer.getvalue()
    display(HTML(f"<pre style='background:#0B1023;color:#E2E8F0;padding:15px;border-radius:12px;font-size:13px;'>{info}</pre>"))


# LOAD DATA
data_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission_sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# DISPLAY ALL
print_dataset_analysis(data_train, "Training Data", palette_index=0)
print_dataset_analysis(data_test, "Testing Data", palette_index=1)
print_dataset_analysis(submission_sample, "Sample Submission Data", palette_index=2)



# Define target column
target_col = "accident_risk"

# Summary statistics for numerical features
numerical_cols = data_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("\nNumerical Features Summary:")
display(data_train[numerical_cols].describe())
print("============================================================================================")
# Summary for categorical features
categorical_cols = data_train.select_dtypes(include=["object", "category"]).columns.tolist()
print("\nCategorical Features Unique Values:")
for col in categorical_cols:
    display(f"{col}: {data_train[col].nunique()} unique values")



# Custom Neon-Dark Palette
neon_theme = {
    "background": "#0B1023",   
    "grid": "#1E3A8A",         
    "accent_blue": "#06B6D4", 
    "accent_orange": "#F97316",
    "accent_purple": "#9333EA",
    "accent_yellow": "#EAB308",
    "text": "#E2E8F0"          
}

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": neon_theme["background"],
    "figure.facecolor": neon_theme["background"],
    "axes.edgecolor": neon_theme["accent_purple"],
    "grid.color": neon_theme["grid"],
    "text.color": neon_theme["text"],
    "axes.labelcolor": neon_theme["text"],
    "xtick.color": neon_theme["accent_yellow"],
    "ytick.color": neon_theme["accent_yellow"],
    "font.family": "Poppins",
    "axes.titleweight": "bold"
})



plt.figure(figsize=(8,5))
barplot = sns.barplot(
    x="road_type", y="accident_risk", data=data_train,
    palette=[neon_theme["accent_purple"], neon_theme["accent_orange"], neon_theme["accent_blue"]],
    edgecolor=neon_theme["accent_yellow"], linewidth=1.5
)

# Add values above bars
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(
        f'{height:.2f}',  # Format to 2 decimal places
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom',
        fontsize=10, color=neon_theme["accent_yellow"],
        weight='bold'
    )

plt.title("Average Accident Risk by Road Type", fontsize=16, color=neon_theme["accent_yellow"])
plt.xlabel("Road Type", fontsize=13, color=neon_theme["accent_blue"])
plt.ylabel("Mean Accident Risk", fontsize=13, color=neon_theme["accent_blue"])
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(
    x="weather", y="accident_risk", data=data_train,
    palette=[neon_theme["accent_blue"], neon_theme["accent_purple"], neon_theme["accent_orange"]],
    linewidth=1.2
)
plt.title("Accident Risk Distribution by Weather Condition", fontsize=16, color=neon_theme["accent_yellow"])
plt.xlabel("Weather", fontsize=13)
plt.ylabel("Accident Risk", fontsize=13)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
barplot = sns.barplot(
    x="time_of_day", y="accident_risk", data=data_train,
    palette=[neon_theme["accent_orange"], neon_theme["accent_purple"], neon_theme["accent_blue"]],
    edgecolor=neon_theme["accent_yellow"]
)

# Add values above bars
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(
        f'{height:.2f}', 
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom',
        fontsize=10, color=neon_theme["accent_yellow"],
        weight='bold'
    )

plt.title("Accident Risk by Time of Day", fontsize=16, color=neon_theme["accent_yellow"])
plt.xlabel("Time of Day", fontsize=13)
plt.ylabel("Mean Risk", fontsize=13)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
sns.lineplot(
    x="speed_limit", y="accident_risk", data=data_train,
    color=neon_theme["accent_cyan"] if "accent_cyan" in neon_theme else neon_theme["accent_blue"],
    marker="o", linewidth=2.5
)
plt.title("Relationship between Speed Limit and Accident Risk", fontsize=16, color=neon_theme["accent_yellow"])
plt.xlabel("Speed Limit (km/h)", fontsize=13)
plt.ylabel("Average Accident Risk", fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
sns.scatterplot(
    x="curvature", y="accident_risk", data=data_train.sample(5000, random_state=42),
    alpha=0.6, color=neon_theme["accent_purple"], edgecolor=neon_theme["accent_yellow"]
)
plt.title("Accident Risk vs Road Curvature", fontsize=16, color=neon_theme["accent_yellow"])
plt.xlabel("Curvature", fontsize=13)
plt.ylabel("Accident Risk", fontsize=13)
plt.tight_layout()
plt.show()



# Visualize Target Distribution
plt.figure(figsize=(8, 6))
sns.histplot(
    data_train[target_col],
    bins=30,
    kde=True,
    color=neon_theme["accent_orange"],
    edgecolor=neon_theme["accent_yellow"],
    linewidth=1.2
)
plt.title("Distribution of Accident Risk", fontsize=18, color=neon_theme["accent_yellow"])
plt.xlabel("Accident Risk", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(alpha=0.25, linestyle='--')
plt.show()

# Correlation Heatmap for Numerical Features
plt.figure(figsize=(10, 8))
numerical_data = data_train[numerical_cols]
corr_matrix = numerical_data.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=sns.color_palette(
        [neon_theme["accent_blue"], neon_theme["accent_purple"], neon_theme["accent_orange"]],
        as_cmap=True
    ),
    cbar_kws={"label": "Correlation Strength"},
    linewidths=0.5,
    annot_kws={"color": neon_theme["text"], "size": 10, "weight": "bold"}
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=18, color=neon_theme["accent_yellow"])
plt.xlabel("")
plt.ylabel("")
plt.show()



def generate_freq_bin_features(train_data, test_data, feature_list, num_features, cat_features):
    """
    Generate frequency and binning features for the dataset.
    
    Parameters:
    - train_data: Training dataframe
    - test_data: Test dataframe
    - feature_list: List of feature columns
    - num_features: List of numerical feature columns
    - cat_features: List of categorical feature columns
    
    Returns:
    - Modified train and test dataframes with new features
    - Updated list of numerical features
    """
    train_mod, test_mod = train_data.copy(), test_data.copy()
    
    for feat in feature_list:
        # Frequency encoding
        freq_map = train_mod[feat].value_counts(normalize=True)
        train_mod[f"{feat}_frequency"] = train_mod[feat].map(freq_map)
        test_mod[f"{feat}_frequency"] = test_mod[feat].map(freq_map).fillna(train_mod[f"{feat}_frequency"].mean())
        
        # Binning for numerical features
        if feat in num_features:
            for quant in [5, 10, 15]:
                try:
                    train_mod[f"{feat}_quantile_{quant}"], bins = pd.qcut(
                        train_mod[feat], q=quant, labels=False, retbins=True, duplicates="drop"
                    )
                    test_mod[f"{feat}_quantile_{quant}"] = pd.cut(
                        test_mod[feat], bins=bins, labels=False, include_lowest=True
                    ).fillna(0)
                except Exception as e:
                    print(f"Error binning {feat} for {quant} quantiles: {e}")
                    train_mod[f"{feat}_quantile_{quant}"] = 0
                    test_mod[f"{feat}_quantile_{quant}"] = 0
    
    # Update numerical features list
    updated_num_features = train_mod.drop(columns=cat_features + [target_col]).columns.tolist()
    return train_mod, test_mod, updated_num_features

# Identify features
all_features = data_train.drop(columns=target_col).columns.tolist()
cat_features = [col for col in all_features if data_train[col].dtype in ["object", "category"]]
num_features = [col for col in all_features if data_train[col].dtype not in ["object", "category", "bool"] and col != "id"]

# Apply feature engineering
data_train_proc, data_test_proc, updated_num_features = generate_freq_bin_features(
    data_train, data_test, all_features, num_features, cat_features
)

# Convert categorical features to category type
data_train_proc[cat_features] = data_train_proc[cat_features].astype("category")
data_test_proc[cat_features] = data_test_proc[cat_features].astype("category")

# Additional feature: Interaction between curvature and speed_limit
data_train_proc["curvature_speed_interaction"] = data_train_proc["curvature"] * data_train_proc["speed_limit"]
data_test_proc["curvature_speed_interaction"] = data_test_proc["curvature"] * data_test_proc["speed_limit"]
updated_num_features.append("curvature_speed_interaction")

print("\nNew Features Created:")
print(data_train_proc.columns.tolist())


# Map num_reported_accidents
accident_col = "num_reported_accidents"
accident_mapping = {0: 0, 1: 0, 2: 0, 3: 2, 4: 4, 5: 3, 6: 1, 7: 0}
data_train_proc[accident_col] = data_train_proc[accident_col].map(accident_mapping)
data_test_proc[accident_col] = data_test_proc[accident_col].map(accident_mapping)

# Drop unnecessary columns
columns_to_drop = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_frequency"]
existing_drops = [col for col in columns_to_drop if col in data_train_proc.columns]
data_train_proc = data_train_proc.drop(columns=existing_drops)
data_test_proc = data_test_proc.drop(columns=existing_drops)

# Drop ID column and duplicates
data_train_proc = data_train_proc.drop(columns="id")
data_test_proc_ids = data_test_proc["id"]  # Save IDs for submission
data_test_proc = data_test_proc.drop(columns="id")
data_train_proc = data_train_proc.drop_duplicates()

# Verify final columns
print("\nFinal Training Columns:")
print(data_train_proc.columns.tolist())




# Feature Preparation
cat_features = [col for col in data_train_proc.columns 
                if data_train_proc[col].dtype.name == "category" and col != target_col]

updated_num_features = [col for col in data_train_proc.columns 
                        if data_train_proc[col].dtype in ["int64", "float64"] and col != target_col]
# Model Definitions
models = {
    "XGBoost": XGBRegressor(
        max_depth=11,
        learning_rate=0.011,
        subsample=0.82,
        colsample_bytree=0.81,
        min_child_weight=3,
        gamma=0.011,
        reg_alpha=0.12,
        reg_lambda=0.4,
        max_delta_step=1,
        colsample_bylevel=0.86,
        colsample_bynode=0.88,
        scale_pos_weight=0.36,
        max_bin=512,
        tree_method="hist",
        device="cuda",
        random_state=42,
        enable_categorical=True
    ),
    "RandomForest": Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
                ("num", "passthrough", updated_num_features)
            ]
        )),
        ("regressor", RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        force_row_wise=True,
        extra_trees=True
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500,
        learning_rate=0.01,
        depth=10,
        cat_features=cat_features,
        random_state=42,
        verbose=0
    )
}

# Model Evaluation Function
def evaluate_models(models, X, y, n_splits=5):
    """
    Evaluate models using k-fold cross-validation and return RMSE scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_scores = {}
    
    for model_name, model in models.items():
        rmse_scores = []
        print(f"\n🚀 Evaluating {model_name}...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict and calculate RMSE
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            rmse_scores.append(rmse)
            print(f"Fold {fold + 1} RMSE: {rmse:.7f}")
        
        mean_rmse = np.mean(rmse_scores)
        model_scores[model_name] = mean_rmse
        print(f"{model_name} Mean RMSE: {mean_rmse:.7f}")
    
    return model_scores

# Prepare Data
X_train_cv = data_train_proc.drop(columns=target_col)
y_train_cv = data_train_proc[target_col]

# Evaluate Models
model_scores = evaluate_models(models, X_train_cv, y_train_cv)




# neon theme color list for Seaborn
neon_palette = [
    neon_theme["accent_orange"],
    neon_theme["accent_purple"],
    neon_theme["accent_blue"],
    neon_theme["accent_yellow"]
]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=list(model_scores.values()), 
    y=list(model_scores.keys()), 
    palette=neon_palette, 
    edgecolor="#EAB308",
    linewidth=1.5
)
plt.title("Model Comparison: Mean RMSE from Cross-Validation", fontsize=16, fontweight="bold", color="#F97316")
plt.xlabel("Mean RMSE", fontsize=13, color="#06B6D4")
plt.ylabel("Model", fontsize=13, color="#06B6D4")
plt.grid(True, color="#1E3A8A", alpha=0.4)
plt.tight_layout()
plt.show()


# Select the best model
best_model_name = min(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Mean RMSE: {model_scores[best_model_name]:.7f}")

# Train on full data
best_model.fit(X_train_cv, y_train_cv)

# Feature Importance Visualization
def plot_feature_importance(model_name, model, X, neon_theme):
    plt.figure(figsize=(10, 6))
    
    # Get feature importance safely per model
    if model_name == "XGBoost":
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    elif model_name == "RandomForest":
        feature_importance = pd.Series(
            model.named_steps["regressor"].feature_importances_,
            index=model.named_steps["preprocessor"].get_feature_names_out()
        )
    elif model_name == "LightGBM":
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    elif model_name == "CatBoost":
        feature_importance = pd.Series(model.get_feature_importance(), index=X.columns)
    else:
        print("Feature importance not supported for this model.")
        return

    top_features = feature_importance.nlargest(10)

    # Use reversed list of colors from neon_theme for gradient feel
    color_list = list(neon_theme.values())[::-1]
    
    sns.barplot(
        x=top_features.values, 
        y=top_features.index, 
        palette=color_list[:len(top_features)],  # ensure matching length
        edgecolor="#EAB308",
        linewidth=1.5
    )
    
    plt.title(f"Top 10 Feature Importance ({model_name})", fontsize=16, fontweight="bold", color="#F97316")
    plt.xlabel("Importance", fontsize=13, color="#06B6D4")
    plt.ylabel("Features", fontsize=13, color="#06B6D4")
    plt.grid(True, color="#1E3A8A", alpha=0.4)
    
    # Show values above bars
    for p in plt.gca().patches:
        width = p.get_width()
        plt.gca().annotate(
            f'{width:.2f}',
            (width, p.get_y() + p.get_height() / 2),
            ha='left', va='center',
            fontsize=10, color="#FBBF24",
            weight='bold'
        )

    plt.tight_layout()
    plt.show()

# Plot for the best model
plot_feature_importance(best_model_name, best_model, X_train_cv, neon_theme)



# Predict on test set
test_predictions = best_model.predict(data_test_proc)

# Ensure predictions are between 0 and 1
test_predictions = np.clip(test_predictions, 0, 1)

# Create submission dataframe
submission_df = pd.DataFrame({
    "id": data_test_proc_ids,
    target_col: test_predictions
})

# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("\nSubmission File Created:")
display(submission_df.head())

