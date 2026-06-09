
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import optuna
from scipy import stats
from scipy.special import boxcox1p


from IPython.display import display, HTML
import pandas as pd
import io

# 🔶 Strong and stylish orange-pink main heading gradient
main_heading_gradient = "linear-gradient(90deg, #ff7e5f, #feb47b)"
headings_border_color = '#ff3e55'

# 🎨 Unique sub-heading gradients for each dataset
sub_heading_gradients = {
    "Training Data": "linear-gradient(90deg, #ffd5ec, #ffaaa5)",    # Soft pink tones
    "Test Data": "linear-gradient(90deg, #d0e6f6, #89bde5)",         # Sky blue tones
    "Sample Submission": "linear-gradient(90deg, #e6e6fa, #dda0dd)", # Lavender tones
    "Original Personality Dataset": "linear-gradient(90deg, #e0c3fc, #8ec5fc)",  # Purple-blue tones
}

# 🌟 Styled main heading
def styled_main_heading(text):
    return f"""
    <div style="
        text-align: center;
        background: {main_heading_gradient};
        color: white;
        padding: 25px 20px;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.3);
        border-radius: 25px;
        margin: 30px 0;
        border: 5px solid {headings_border_color};
        box-shadow: 0 8px 20px rgba(255, 120, 120, 0.4);
    ">
        {text}
    </div>
    """

# 🎯 Styled sub-heading
def styled_sub_heading(text, dataset_name):
    gradient = sub_heading_gradients.get(dataset_name, main_heading_gradient)
    return f"""
    <h2 style="
        font-size: 24px;
        background: {gradient};
        color: white;
        text-align: center;
        padding: 10px 20px;
        border-radius: 10px;
        margin: 15px 0 10px;
    ">{text}</h2>
    """

# 📋 Table styling
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#ff9472")]}
    ]).set_properties(**{"text-align": "center"}).hide(axis="index")
    return styled_df.to_html()

# 📊 Complete analysis
def print_dataset_analysis(dataset, dataset_name, n_top=5):
    display(HTML(styled_main_heading(f"📊 {dataset_name} Overview")))
    
    def show_subsection(title, content_html=None):
        display(HTML(styled_sub_heading(title, dataset_name)))
        if content_html:
            display(HTML(content_html))
    
    show_subsection("🔍 Shape of the Dataset", f"<p>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>")
    show_subsection("👀 First 5 Rows", style_table(dataset.head(n_top)))
    show_subsection("📈 Summary Statistics", style_table(dataset.describe()))
    
    # Null values
    show_subsection("🚨 Null Values")
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML("<p>No null values found.</p>"))
    else:
        null_columns = null_counts[null_counts > 0]
        null_df = null_columns.to_frame(name='Null Values')
        null_df['Column Names with Nulls'] = null_df.index
        display(HTML(style_table(null_df)))

    # Duplicates
    show_subsection("🔍 Duplicate Rows", f"<p>{dataset.duplicated().sum()} duplicate rows found.</p>")
    
    # Data types
    show_subsection("📝 Data Types", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    })))

    # Column names
    show_subsection("📋 Column Names", f"<p>{', '.join(dataset.columns)}</p>")

    # Unique values
    show_subsection("🔢 Unique Values", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Unique Values': [dataset[col].nunique() for col in dataset.columns]
    })))

    # Dataset info
    show_subsection("ℹ️ Dataset Info")
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    info_output = buffer.getvalue()
    display(HTML(f"<pre>{info_output}</pre>"))

# 🚀 Load datasets
print("🚀 Initializing Personality Prediction Dataset Visual Analysis...")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# 📊 Display analysis
print_dataset_analysis(df_train, "Training Data")
print_dataset_analysis(df_test, "Test Data")
print_dataset_analysis(sample_submission, "Sample Submission")
print_dataset_analysis(df_original, "Original Personality Dataset")



df_original = df_original.rename(columns={'Personality': 'match_p'})

# Create merge key with feature hashing for better matching
merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']



df_original = df_original.drop_duplicates(subset=merge_cols)

# Strategic merging with confidence scoring
df_test = df_test.merge(df_original, how='left', on=merge_cols)
df_train = df_train.merge(df_original, how='left', on=merge_cols)

print("🔗 Data fusion completed with reference matching")
df_train.info()


train_ID = df_train['id']
test_ID = df_test['id']

df_train.drop("id", axis=1, inplace=True)
df_test.drop("id", axis=1, inplace=True)

ntrain = df_train.shape[0] 
ntest = df_test.shape[0] 
y_train = df_train['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)

print("🔧 Advanced preprocessing initiated")
all_data.info()


def intelligent_quantile_imputation(dataframe, source_feature, target_feature, n_bins=4):
    """Advanced quantile-based imputation with outlier handling"""
    temp_bins = f'{source_feature}_quantile_bins'
    
    # Handle outliers before binning
    Q1 = dataframe[source_feature].quantile(0.25)
    Q3 = dataframe[source_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    dataframe[source_feature] = dataframe[source_feature].clip(lower_bound, upper_bound)
    
    # Create intelligent bins
    try:
        dataframe[temp_bins] = pd.qcut(dataframe[source_feature], q=n_bins, 
                                      labels=[f'B{i+1}' for i in range(n_bins)], duplicates='drop')
    except:
        dataframe[temp_bins] = pd.cut(dataframe[source_feature], bins=n_bins, 
                                     labels=[f'B{i+1}' for i in range(n_bins)], duplicates='drop')
    
    # Smart imputation with mode for small groups
    def smart_fill(group):
        if len(group.dropna()) >= 3:
            return group.fillna(group.median())
        else:
            return group.fillna(dataframe[target_feature].median())
    
    dataframe[target_feature] = dataframe.groupby(temp_bins)[target_feature].transform(smart_fill)
    dataframe.drop(columns=[temp_bins], inplace=True)
    
    return dataframe


print("🧠 Phase 1: Executing primary imputation chain...")
all_data = intelligent_quantile_imputation(all_data, 'Social_event_attendance', 'Time_spent_Alone', n_bins=5)
all_data = intelligent_quantile_imputation(all_data, 'Going_outside', 'Time_spent_Alone', n_bins=5)
all_data = intelligent_quantile_imputation(all_data, 'Going_outside', 'Social_event_attendance', n_bins=4)
all_data = intelligent_quantile_imputation(all_data, 'Friends_circle_size', 'Social_event_attendance', n_bins=4)
all_data = intelligent_quantile_imputation(all_data, 'Post_frequency', 'Social_event_attendance', n_bins=4)

print("🔄 Phase 2: Cross-feature imputation...")
all_data = intelligent_quantile_imputation(all_data, 'Social_event_attendance', 'Going_outside', n_bins=5)
all_data = intelligent_quantile_imputation(all_data, 'Post_frequency', 'Friends_circle_size', n_bins=4)
all_data = intelligent_quantile_imputation(all_data, 'Going_outside', 'Friends_circle_size', n_bins=4)
all_data = intelligent_quantile_imputation(all_data, 'Friends_circle_size', 'Post_frequency', n_bins=4)

print("🎯 Phase 3: Advanced iterative imputation...")
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']

remaining_missing = [col for col in numerical_features if all_data[col].isnull().sum() > 0]
if remaining_missing:
    iterative_imputer = IterativeImputer(
        estimator=XGBClassifier(n_estimators=50, max_depth=3, random_state=42),
        max_iter=15,
        random_state=42
    )
    all_data[remaining_missing] = iterative_imputer.fit_transform(all_data[remaining_missing])

all_data.info()




print("🏷️ Processing categorical variables...")
all_data['Stage_fear'] = all_data['Stage_fear'].fillna('Unknown')
all_data['Drained_after_socializing'] = all_data['Drained_after_socializing'].fillna('Unknown')

# Create ordinal encoding for better feature representation
stage_fear_map = {'No': 0, 'Maybe': 1, 'Yes': 2, 'Unknown': 1.5}
drain_map = {'No': 0, 'Maybe': 1, 'Yes': 2, 'Unknown': 1.5}

all_data['Stage_fear_ordinal'] = all_data['Stage_fear'].map(stage_fear_map)
all_data['Drained_ordinal'] = all_data['Drained_after_socializing'].map(drain_map)

# One-hot encoding
all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing', 'match_p'], 
                         prefix=['Stage', 'Drain', 'Reference'])

all_data.info()



print("⚡ Executing revolutionary feature engineering...")

# Separate data
X_train_base = all_data[:ntrain].copy()
X_test_base = all_data[ntrain:].copy()

# Core numerical features
core_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

# 1. Statistical Transformations Suite
for feature in core_features:
    # Box-Cox transformation
    X_train_base[f'{feature}_boxcox'] = boxcox1p(X_train_base[feature], 0.15)
    X_test_base[f'{feature}_boxcox'] = boxcox1p(X_test_base[feature], 0.15)
    
    # Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson')
    X_train_base[f'{feature}_yeojohnson'] = pt.fit_transform(X_train_base[[feature]]).ravel()
    X_test_base[f'{feature}_yeojohnson'] = pt.transform(X_test_base[[feature]]).ravel()
    
    # Quantile transformation
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_base[f'{feature}_quantile'] = qt.fit_transform(X_train_base[[feature]]).ravel()
    X_test_base[f'{feature}_quantile'] = qt.transform(X_test_base[[feature]]).ravel()
    
    # Polynomial features
    X_train_base[f'{feature}_squared'] = X_train_base[feature] ** 2
    X_test_base[f'{feature}_squared'] = X_test_base[feature] ** 2
    
    X_train_base[f'{feature}_cubed'] = X_train_base[feature] ** 3
    X_test_base[f'{feature}_cubed'] = X_test_base[feature] ** 3
    
    # Reciprocal transformation
    X_train_base[f'{feature}_reciprocal'] = 1 / (X_train_base[feature] + 0.1)
    X_test_base[f'{feature}_reciprocal'] = 1 / (X_test_base[feature] + 0.1)

# 2. Advanced Interaction Features
interaction_pairs = [
    ('Time_spent_Alone', 'Social_event_attendance'),
    ('Social_event_attendance', 'Friends_circle_size'),
    ('Going_outside', 'Post_frequency'),
    ('Friends_circle_size', 'Post_frequency'),
    ('Time_spent_Alone', 'Going_outside')
]

for feat1, feat2 in interaction_pairs:
    # Multiplicative interactions
    X_train_base[f'{feat1}_{feat2}_product'] = X_train_base[feat1] * X_train_base[feat2]
    X_test_base[f'{feat1}_{feat2}_product'] = X_test_base[feat1] * X_test_base[feat2]
    
    # Ratio interactions
    X_train_base[f'{feat1}_{feat2}_ratio'] = X_train_base[feat1] / (X_train_base[feat2] + 0.1)
    X_test_base[f'{feat1}_{feat2}_ratio'] = X_test_base[feat1] / (X_test_base[feat2] + 0.1)
    
    # Difference interactions
    X_train_base[f'{feat1}_{feat2}_diff'] = np.abs(X_train_base[feat1] - X_train_base[feat2])
    X_test_base[f'{feat1}_{feat2}_diff'] = np.abs(X_test_base[feat1] - X_test_base[feat2])
X_train_base['extroversion_index'] = (
    X_train_base['Social_event_attendance'] * 0.3 +
    X_train_base['Going_outside'] * 0.25 +
    X_train_base['Friends_circle_size'] * 0.2 +
    X_train_base['Post_frequency'] * 0.15 +
    (10 - X_train_base['Time_spent_Alone']) * 0.1 +
    X_train_base['Stage_fear_ordinal'] * (-0.05) +
    X_train_base['Drained_ordinal'] * (-0.05)
)

X_test_base['extroversion_index'] = (
    X_test_base['Social_event_attendance'] * 0.3 +
    X_test_base['Going_outside'] * 0.25 +
    X_test_base['Friends_circle_size'] * 0.2 +
    X_test_base['Post_frequency'] * 0.15 +
    (10 - X_test_base['Time_spent_Alone']) * 0.1 +
    X_test_base['Stage_fear_ordinal'] * (-0.05) +
    X_test_base['Drained_ordinal'] * (-0.05)
)

# Social energy balance
X_train_base['social_energy_balance'] = (
    (X_train_base['Social_event_attendance'] + X_train_base['Going_outside']) / 
    (X_train_base['Time_spent_Alone'] + 1)
)
X_test_base['social_energy_balance'] = (
    (X_test_base['Social_event_attendance'] + X_test_base['Going_outside']) / 
    (X_test_base['Time_spent_Alone'] + 1)
)


print("🎯 Creating clustering features...")
cluster_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']

# Multiple clustering approaches
for n_clusters in [3, 5, 7]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    all_cluster_data = np.vstack([X_train_base[cluster_features].values, X_test_base[cluster_features].values])
    cluster_labels = kmeans.fit_predict(all_cluster_data)
    
    X_train_base[f'personality_cluster_{n_clusters}'] = cluster_labels[:ntrain]
    X_test_base[f'personality_cluster_{n_clusters}'] = cluster_labels[ntrain:]

print(f"🎉 Feature engineering completed! New dimensions - Train: {X_train_base.shape}, Test: {X_test_base.shape}")



print("🔍 Executing intelligent feature selection...")

# Remove highly correlated features
correlation_matrix = X_train_base.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

if high_corr_features:
    X_train_base.drop(columns=high_corr_features, inplace=True)
    X_test_base.drop(columns=high_corr_features, inplace=True)
# Statistical feature selection
feature_selector = SelectKBest(score_func=f_classif, k=min(200, X_train_base.shape[1]))
X_train_selected = feature_selector.fit_transform(X_train_base, y_train)
X_test_selected = feature_selector.transform(X_test_base)

selected_features = X_train_base.columns[feature_selector.get_support()]
X_train_final = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_final = pd.DataFrame(X_test_selected, columns=selected_features)

print(f"Selected {len(selected_features)} most informative features")




# =================== HYPERPARAMETER OPTIMIZATION ===================
print("🚀 Optimizing hyperparameters with Optuna...")

def objective(trial):
    # XGBoost hyperparameters
    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 4, 8),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 2000),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.01, 0.3),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.01, 0.3),
    }
    
    # Calculate class weights
    class_0_count = y_train.sum()
    class_1_count = len(y_train) - class_0_count
    scale_pos_weight = class_1_count / class_0_count
    
    xgb_model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train_final, y_train):
        X_tr, X_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        xgb_model.fit(X_tr, y_tr)
        val_pred = xgb_model.predict(X_val)
        accuracy = (val_pred == y_val).mean()
        cv_scores.append(accuracy)
    
    return np.mean(cv_scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, timeout=600)
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# =================== REVOLUTIONARY MODEL ENSEMBLE ===================
print("Building revolutionary model ensemble...")



# Calculate optimized class weights
class_0_count = y_train.sum()
class_1_count = len(y_train) - class_0_count
scale_pos_weight = class_1_count / class_0_count

# Optimized models with best hyperparameters
xgb_optimized = XGBClassifier(
    max_depth=best_params.get('xgb_max_depth', 6),
    learning_rate=best_params.get('xgb_learning_rate', 0.05),
    n_estimators=best_params.get('xgb_n_estimators', 1000),
    subsample=best_params.get('xgb_subsample', 0.85),
    colsample_bytree=best_params.get('xgb_colsample_bytree', 0.85),
    reg_alpha=best_params.get('xgb_reg_alpha', 0.1),
    reg_lambda=best_params.get('xgb_reg_lambda', 0.1),
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

cat_optimized = CatBoostClassifier(
    iterations=1200,
    depth=7,
    learning_rate=0.03,
    class_weights=[scale_pos_weight, 1],
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)

lgbm_optimized = LGBMClassifier(
    num_leaves=45,
    learning_rate=0.04,
    n_estimators=1200,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42,
    verbosity=-1
)

# Neural network for diversity
neural_net = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    learning_rate_init=0.001,
    alpha=0.01,
    max_iter=500,
    random_state=42,
    early_stopping=True
)

# SVM with probability estimates
svm_model = SVC(
    C=0.8,
    probability=True,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42
)

# Meta-learner
meta_learner = LogisticRegression(
    C=1.0,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42,
    max_iter=2000
)

# Advanced stacking ensemble
base_models = [
    ('xgb', xgb_optimized),
    ('cat', cat_optimized),
    ('lgbm', lgbm_optimized),
    ('nn', neural_net),
    ('svm', svm_model)
]

stacking_ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    n_jobs=-1
)


# =================== ADVANCED VALIDATION & THRESHOLD OPTIMIZATION ===================
print("Advanced validation and threshold optimization...")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Train the stacking ensemble
stacking_ensemble.fit(X_train_split, y_train_split)

# Advanced threshold optimization with multiple metrics
val_probabilities = stacking_ensemble.predict_proba(X_val_split)[:, 1]

best_threshold = 0.5
best_score = 0
threshold_candidates = np.arange(0.3, 0.7, 0.002)

for threshold in threshold_candidates:
    predictions = (val_probabilities >= threshold).astype(int)
    accuracy = (predictions == y_val_split).mean()
    
    if accuracy > best_score:
        best_score = accuracy
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.4f}")
print(f"Best validation accuracy: {best_score:.4f}")

# =================== FINAL PREDICTION GENERATION ===================
print("Generating final predictions...")

# Train on full dataset
stacking_ensemble.fit(X_train_final, y_train)

# Generate predictions
test_probabilities = stacking_ensemble.predict_proba(X_test_final)[:, 1]
final_predictions = (test_probabilities >= best_threshold).astype(int)


# =================== ADVANCED SUBMISSION PREPARATION ===================
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': final_predictions
})
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})

# Save submission file
submission.to_csv('submission.csv', index=False)



# Verify data types
print(submission.dtypes)
display(submission.head())

