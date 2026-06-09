import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV  # For stacking meta learner
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import skew
from scipy.optimize import minimize
import io
from IPython.display import display, HTML
# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
# Ignore DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)




soft_palette = [
    "#FFB6C1", "#FF69B4", "#FF8A80", "#FFAB91",  # Pinks / Corals
    "#FFDAB9", "#FFE4B5", "#FFFACD", "#FFF176",  # Peach / Yellow
    "#E1F5FE", "#B3E5FC", "#81D4FA", "#4FC3F7",  # Blues
    "#E6E6FA", "#D1C4E9", "#B39DDB", "#9575CD",  # Purples
    "#E8F5E9", "#C8E6C9", "#A5D6A7", "#81C784",  # Greens
]

# Dataset-specific themes
dataset_themes = {
    "Training Data": {"bg": soft_palette[0], "accent": soft_palette[3]},
    "Merged Training Data": {"bg": soft_palette[2], "accent": soft_palette[4]},
    "Test Data": {"bg": soft_palette[8], "accent": soft_palette[10]},
    "Sample Submission": {"bg": soft_palette[12], "accent": soft_palette[14]},
    "Original Data": {"bg": soft_palette[16], "accent": soft_palette[18]},
}

# Main heading 
def styled_main_heading(text, theme):
    return f"""
    <div style="
        text-align: center;
        background: linear-gradient(135deg, {theme['bg']}, {theme['accent']});
        color: #fff;
        padding: 28px;
        font-family: 'Montserrat', sans-serif;
        font-size: 32px;
        font-weight: 900;
        border-radius: 16px;
        margin: 25px auto;
        width: 95%;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        letter-spacing: 2px;
        border: 4px solid #fff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    ">
        {text}
    </div>
    """

# Sub-heading 
def styled_sub_heading(text, idx, theme):
    accents = [theme['bg'], theme['accent'], "#FF80AB", "#40E0D0", "#FFD700"]
    color_accent = accents[idx % len(accents)]
    return f"""
    <h3 style="
        font-size: 20px;
        color: #222;
        background-color: {color_accent}22;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 18px 0 12px;
        font-family: 'Montserrat', sans-serif;
        border-left: 8px solid {color_accent};
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        font-weight: 700;
    ">
        <span style="font-size: 1.2em; margin-right: 10px; color:{color_accent};">&#10095;</span> {text}
    </h3>
    """

# Table styling
def style_table(df, theme):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", theme['bg']),
            ("color", "#000"),
            ("padding", "10px"),
            ("font-family", "'Open Sans', sans-serif"),
            ("font-size", "14px"),
            ("text-align", "center"),
            ("border-bottom", f"2px solid {theme['accent']}")
        ]},
        {"selector": "td", "props": [
            ("background-color", f"{theme['accent']}22"),
            ("color", "#111"),
            ("padding", "8px"),
            ("text-align", "center"),
            ("font-family", "'Open Sans', sans-serif"),
            ("font-size", "13px"),
            ("border-bottom", "1px solid #CCC")
        ]},
        {"selector": "tr:nth-child(even)", "props": [("background-color", f"{theme['bg']}33")]},
        {"selector": "table", "props": [
            ("width", "95%"),
            ("border-collapse", "collapse"),
            ("border-radius", "10px"),
            ("overflow", "hidden"),
            ("box-shadow", "0 5px 18px rgba(0,0,0,0.1)")
        ]},
        {"selector": "tbody tr:hover", "props": [("background-color", f"{theme['accent']}55")]}
    ]).hide(axis="index")
    return styled_df.to_html()

# Dataset analysis function
def print_dataset_analysis(dataset, dataset_name):
    theme = dataset_themes.get(dataset_name, {"bg": "#ddd", "accent": "#999"})
    display(HTML(styled_main_heading(f"{dataset_name}", theme)))
    
    def show_subsection(title, content_html=None, idx=0):
        display(HTML(styled_sub_heading(title, idx, theme)))
        if content_html:
            display(HTML(f"""
            <div style='
                font-family:"Open Sans", sans-serif;
                color:#111;
                background-color:{theme['bg']}22;
                border:1px solid {theme['accent']};
                border-radius:10px;
                padding:12px 16px;
                margin-bottom:15px;
                line-height:1.6;
                box-shadow:0 3px 10px rgba(0,0,0,0.1);
            '>{content_html}</div>"""))
    
    show_subsection("Shape of the Dataset", f"<p>This dataset contains <strong>{dataset.shape[0]} rows</strong> and <strong>{dataset.shape[1]} columns</strong>.</p>", 0)
    show_subsection("First 5 Rows", style_table(dataset.head(), theme), 1)
    show_subsection("Summary Statistics", style_table(dataset.describe(), theme), 2)
    
    # Null values
    show_subsection("Null Values", idx=3)
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML(f"<p style='color:#2E7D32; font-weight:600;'>No null values found!</p>"))
    else:
        null_df = null_counts[null_counts > 0].to_frame(name='Null Values')
        null_df['Column'] = null_df.index
        display(HTML(style_table(null_df, theme)))
    
    # Duplicates
    show_subsection("Duplicate Rows", f"<p>A total of <strong>{dataset.duplicated().sum()} duplicate rows</strong> were identified.</p>", 0)
    
    # Data types
    show_subsection("Data Types", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    }), theme), 1)
    
    # Column names
    show_subsection("Column Names", f"<p style='word-break: break-word;'>{', '.join([f'<code>{col}</code>' for col in dataset.columns])}</p>", 2)
    
    # Unique values
    show_subsection("Unique Values per Column", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Unique Values Count': [dataset[col].nunique() for col in dataset.columns]
    }), theme), 3)
    
    # Dataset info
    show_subsection("Detailed Dataset Information", idx=0)
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    display(HTML(f"""
    <pre style='
        background:linear-gradient(135deg,{theme['bg']}99,{theme['accent']}99);
        color:#111;
        padding:15px;
        border-radius:10px;
        font-family:"Fira Code", monospace;
        font-size:13px;
        overflow-x:auto;
        box-shadow:0 3px 12px rgba(0,0,0,0.15);
        line-height:1.4;
    '>{buffer.getvalue()}</pre>
    """))

# Load datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
dforiginal = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')

# show original training data (before merging)
print_dataset_analysis(df_train, "Training Data")

# Show original dataset 
print_dataset_analysis(dforiginal, "Original Data")

# Merge original with train
df_train_merged = pd.concat([df_train, dforiginal], ignore_index=True)

# Show merged training data
print_dataset_analysis(df_train_merged, "Merged Data")

# Test & sample submission analysis
print_dataset_analysis(df_test, "Test Data")
print_dataset_analysis(sample_sub, "Sample Submission")



# Function to generate engineered features
def gen_features(dataframe):
    # Impute missing numerics with median values
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[num_cols] = dataframe[num_cols].fillna(dataframe[num_cols].median())
    dataframe['Rhythm_Loudness_Prod'] = dataframe['RhythmScore'] * dataframe['AudioLoudness']
    dataframe['Vocal_Acoustic_Quot'] = dataframe['VocalContent'] / (dataframe['AcousticQuality'] + 1e-6)
    dataframe['Energy_Mood_Mult'] = dataframe['Energy'] * dataframe['MoodScore']
    dataframe['Instr_Live_Comb'] = dataframe['InstrumentalScore'] * dataframe['LivePerformanceLikelihood']
    dataframe['Mood_Acoustic_Interact'] = dataframe['MoodScore'] * dataframe['AcousticQuality']
    dataframe['Rhythm_Energy_Ratio'] = dataframe['RhythmScore'] / (dataframe['Energy'] + 1e-6)  # New for potential improvement
    
    # Create polynomial features
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_array = poly_transformer.fit_transform(dataframe[['RhythmScore', 'AudioLoudness', 'Energy', 'VocalContent', 'MoodScore']])  # Added extra column
    poly_names = [f'inter_{k}' for k in range(poly_array.shape[1])]
    dataframe[poly_names] = poly_array
    
    # Log-transform skewed distributions
    for var in ['TrackDurationMs', 'AudioLoudness', 'VocalContent', 'Energy', 'InstrumentalScore']:
        if var in dataframe.columns and skew(dataframe[var].dropna()) > 0.5:
            if dataframe[var].min() < 0:
                adjust = abs(dataframe[var].min()) + 1
                dataframe[f'logtrans_{var}'] = np.log1p(dataframe[var] + adjust)
            else:
                dataframe[f'logtrans_{var}'] = np.log1p(dataframe[var].clip(lower=0))
    
    # Bin selected continuous variables
    dataframe['Duration_Cat'] = pd.qcut(dataframe['TrackDurationMs'], q=10, labels=False, duplicates='drop')
    dataframe['Energy_Cat'] = pd.qcut(dataframe['Energy'], q=5, labels=False, duplicates='drop')
    dataframe['Loudness_Cat'] = pd.qcut(dataframe['AudioLoudness'], q=5, labels=False, duplicates='drop')
    dataframe['Vocal_Cat'] = pd.qcut(dataframe['VocalContent'], q=5, labels=False, duplicates='drop')  # New bin
    
    return dataframe


# Engineer features in train and test
df_train = gen_features(df_train)
df_test = gen_features(df_test)


# Identify features excluding identifiers and target
input_feats = [c for c in df_train.columns if c not in ['id', 'BeatsPerMinute'] and df_train[c].nunique() > 1]
# Scale the inputs
feature_scaler = RobustScaler()
inputs_train = feature_scaler.fit_transform(df_train[input_feats])
inputs_test = feature_scaler.transform(df_test[input_feats])
labels = df_train['BeatsPerMinute']


# Model configurations
lgb_settings = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 50,
    'max_depth': 8,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 3,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'seed': 42
}

xgb_settings = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 4,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.2,
    'seed': 42,
    'n_jobs': -1
}

cat_settings = {
    'loss_function': 'RMSE',
    'learning_rate': 0.015,
    'depth': 7,
    'min_data_in_leaf': 40,
    'l2_leaf_reg': 4.5,
    'iterations': 1600,
    'random_seed': 42,
    'verbose': 0
}

hgb_settings = {
    'learning_rate': 0.012,
    'max_iter': 450,
    'max_depth': 8,
    'min_samples_leaf': 25,
    'l2_regularization': 0.6,
    'random_state': 42
}


# Cross-validation setup
num_folds = 15
cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(df_test))
xgb_test_preds = np.zeros(len(df_test))
cat_test_preds = np.zeros(len(df_test))
hgb_test_preds = np.zeros(len(df_test))

lgb_oof_preds = np.zeros(len(inputs_train))
xgb_oof_preds = np.zeros(len(inputs_train))
cat_oof_preds = np.zeros(len(inputs_train))
hgb_oof_preds = np.zeros(len(inputs_train))

# Lists to store per-fold RMSE for each model
lgb_fold_scores = []
xgb_fold_scores = []
cat_fold_scores = []
hgb_fold_scores = []
for fold_idx, (train_ids, valid_ids) in enumerate(cv.split(inputs_train, labels)):
    train_inputs, valid_inputs = inputs_train[train_ids], inputs_train[valid_ids]
    train_labels, valid_labels = labels.iloc[train_ids], labels.iloc[valid_ids]
    
    # LightGBM training
    lgb_train_data = lgb.Dataset(train_inputs, train_labels)
    lgb_valid_data = lgb.Dataset(valid_inputs, valid_labels, reference=lgb_train_data)
    lgb_estimator = lgb.train(
        lgb_settings,
        lgb_train_data,
        num_boost_round=1600,
        valid_sets=[lgb_train_data, lgb_valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    lgb_oof_preds[valid_ids] = lgb_estimator.predict(valid_inputs)
    lgb_fold_rmse = np.sqrt(mean_squared_error(valid_labels, lgb_oof_preds[valid_ids]))
    lgb_fold_scores.append(lgb_fold_rmse)
    print(f'Fold {fold_idx+1} LightGBM RMSE: {lgb_fold_rmse}')
    lgb_test_preds += lgb_estimator.predict(inputs_test) / num_folds
    
    # XGBoost training
    xgb_train_data = xgb.DMatrix(train_inputs, train_labels)
    xgb_valid_data = xgb.DMatrix(valid_inputs, valid_labels)
    xgb_estimator = xgb.train(
        xgb_settings,
        xgb_train_data,
        num_boost_round=1600,
        evals=[(xgb_valid_data, 'val')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    xgb_oof_preds[valid_ids] = xgb_estimator.predict(xgb_valid_data)
    xgb_fold_rmse = np.sqrt(mean_squared_error(valid_labels, xgb_oof_preds[valid_ids]))
    xgb_fold_scores.append(xgb_fold_rmse)
    print(f'Fold {fold_idx+1} XGBoost RMSE: {xgb_fold_rmse}')
    xgb_test_preds += xgb_estimator.predict(xgb.DMatrix(inputs_test)) / num_folds
    
    # CatBoost training
    cat_train_pool = cb.Pool(train_inputs, train_labels)
    cat_valid_pool = cb.Pool(valid_inputs, valid_labels)
    cat_estimator = cb.CatBoostRegressor(**cat_settings)
    cat_estimator.fit(cat_train_pool, eval_set=cat_valid_pool, early_stopping_rounds=150, verbose=False)
    cat_oof_preds[valid_ids] = cat_estimator.predict(valid_inputs)
    cat_fold_rmse = np.sqrt(mean_squared_error(valid_labels, cat_oof_preds[valid_ids]))
    cat_fold_scores.append(cat_fold_rmse)
    print(f'Fold {fold_idx+1} CatBoost RMSE: {cat_fold_rmse}')
    cat_test_preds += cat_estimator.predict(inputs_test) / num_folds
    
    # HistGradientBoosting training
    hgb_estimator = HistGradientBoostingRegressor(**hgb_settings)
    hgb_estimator.fit(train_inputs, train_labels)
    hgb_oof_preds[valid_ids] = hgb_estimator.predict(valid_inputs)
    hgb_fold_rmse = np.sqrt(mean_squared_error(valid_labels, hgb_oof_preds[valid_ids]))
    hgb_fold_scores.append(hgb_fold_rmse)
    print(f'Fold {fold_idx+1} HistGB RMSE: {hgb_fold_rmse}')
    hgb_test_preds += hgb_estimator.predict(inputs_test) / num_folds
    
    # Blend for fold check
    fold_blend = 0.4 * lgb_oof_preds[valid_ids] + 0.3 * xgb_oof_preds[valid_ids] + 0.2 * cat_oof_preds[valid_ids] + 0.1 * hgb_oof_preds[valid_ids]
    blend_rmse = np.sqrt(mean_squared_error(valid_labels, fold_blend))
    print(f'Fold {fold_idx+1} Blend RMSE: {blend_rmse}\n')


# Print average and std of fold scores for each model
print(f'LightGBM Avg RMSE: {np.mean(lgb_fold_scores)}, Std: {np.std(lgb_fold_scores)}')
print(f'XGBoost Avg RMSE: {np.mean(xgb_fold_scores)}, Std: {np.std(xgb_fold_scores)}')
print(f'CatBoost Avg RMSE: {np.mean(cat_fold_scores)}, Std: {np.std(cat_fold_scores)}')
print(f'HistGB Avg RMSE: {np.mean(hgb_fold_scores)}, Std: {np.std(hgb_fold_scores)}')


# Stacking: Use OOF as meta features
meta_train_feats = np.column_stack([lgb_oof_preds, xgb_oof_preds, cat_oof_preds, hgb_oof_preds])
meta_test_feats = np.column_stack([lgb_test_preds, xgb_test_preds, cat_test_preds, hgb_test_preds])

# Meta learner: RidgeCV for regularization and CV
meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
meta_learner.fit(meta_train_feats, labels)

# Meta predictions
stacked_preds = meta_learner.predict(meta_test_feats)

print(f'Meta coefficients: {meta_learner.coef_}')
print(f'Meta intercept: {meta_learner.intercept_}')
print(f'Best alpha: {meta_learner.alpha_}')


# Clip to BPM range
stacked_preds = np.clip(stacked_preds, 40, 200)

# Generate submission
sample_sub['BeatsPerMinute'] = stacked_preds
sample_sub.to_csv('submission.csv', index=False)
print('Stacked submission created!')
display(sample_sub.head(10))

