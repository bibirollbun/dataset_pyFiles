# Install and import essentials
!pip install autogluon==1.2 -q
!pip install plotly -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AutoGluon
from autogluon.tabular import TabularPredictor

print("ğŸ“¦ All libraries imported successfully!")

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
test_ids = test_df['id'].copy()

# Try to load original dataset for augmentation
try:
    original_df = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=';')
    print("âœ… Original dataset loaded - will use for data augmentation")
    USE_ORIGINAL = True
    
    # Convert original dataset target to match synthetic (yes/no -> 1/0)
    if 'y' in original_df.columns:
        original_df['y'] = original_df['y'].map({'yes': 1, 'no': 0})
        print("   â€¢ Converted original target: yes/no -> 1/0")
    
except:
    print("âš ï¸� Original dataset not found - using synthetic data only")
    USE_ORIGINAL = False

print(f"ğŸ“Š Train: {train_df.shape}, Test: {test_df.shape}")
print(f"ğŸ“Š Features: {train_df.shape[1]-1} (excluding ID)")
print(f"ğŸ“Š Target distribution: {train_df['y'].value_counts().to_dict()}")


train_df.head()


train_df.info()


# Remove ID for analysis
train_clean = train_df.drop('id', axis=1)

# Create comprehensive target analysis
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Target Distribution', 'Original vs Synthetic', 
                   'Feature Importance Preview', 'Data Quality Check'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# Target distribution (sort by index to ensure 0,1 order)
target_counts = train_clean['y'].value_counts().sort_index()
fig.add_trace(go.Bar(x=['No (0)', 'Yes (1)'], y=target_counts.values, 
                     marker_color=['#FF6B6B', '#4ECDC4'], name='Synthetic'), row=1, col=1)

# Compare with original if available
if USE_ORIGINAL:
    orig_counts = original_df['y'].value_counts().sort_index()
    fig.add_trace(go.Bar(x=['No (0)', 'Yes (1)'], y=orig_counts.values, 
                         marker_color=['#FFB6C1', '#87CEEB'], name='Original'), row=1, col=2)

# Quick correlation preview (top features)
corr_data = train_clean.select_dtypes(include=[np.number]).corr()['y'].abs().sort_values(ascending=False)[1:6]
fig.add_trace(go.Bar(x=corr_data.index, y=corr_data.values, 
                     marker_color='gold', name='Correlation'), row=2, col=1)

# Data quality - missing values
missing_data = train_clean.isnull().sum()
if missing_data.sum() == 0:
    fig.add_trace(go.Bar(x=['Complete Data'], y=[100], 
                         marker_color='green', name='Quality'), row=2, col=2)
else:
    top_missing = missing_data[missing_data > 0].head(5)
    fig.add_trace(go.Bar(x=top_missing.index, y=top_missing.values, 
                         marker_color='orange', name='Missing'), row=2, col=2)

fig.update_layout(height=600, showlegend=False, 
                  title_text="ğŸ�¯ Quick Data Overview Dashboard")
fig.show()

print(f"ğŸ�¯ Class Distribution: {target_counts[0]:,} (No) vs {target_counts[1]:,} (Yes)")
print(f"ğŸ“Š Imbalance Ratio: {target_counts[0]/target_counts[1]:.1f}:1")
print(f"ğŸ“ˆ Positive Class Rate: {target_counts[1]/target_counts.sum()*100:.2f}%")


# Visualize key features that typically matter in bank marketing
key_features = ['age', 'duration', 'campaign', 'balance', 'pdays', 'previous']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    if feature in train_clean.columns:
        # Create separate data for each target class
        data_0 = train_clean[train_clean['y'] == 0][feature]
        data_1 = train_clean[train_clean['y'] == 1][feature]
        
        # Create box plot
        bp = axes[i].boxplot([data_0, data_1], labels=['No (0)', 'Yes (1)'], patch_artist=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('#FF6B6B')
        bp['boxes'][1].set_facecolor('#4ECDC4')
        
        axes[i].set_title(f'{feature.title()} by Target')
        axes[i].grid(True, alpha=0.3)
        
        # Add summary stats
        mean_0, mean_1 = data_0.mean(), data_1.mean()
        axes[i].text(0.02, 0.98, f'No: Î¼={mean_0:.1f}\nYes: Î¼={mean_1:.1f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('ğŸ“ˆ Key Numerical Features by Target', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Analyze key categorical features - focus on most impactful ones
categorical_features = ['job', 'education', 'contact', 'poutcome', 'marital', 'housing']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    if feature in train_clean.columns and i < 6:
        # Calculate subscription rate by category
        feature_target = pd.crosstab(train_clean[feature], train_clean['y'], normalize='index')
        
        # Sort by subscription rate (descending)
        if 1 in feature_target.columns:
            feature_target = feature_target.sort_values(1, ascending=False)
            
            # Plot subscription rate
            bars = axes[i].bar(range(len(feature_target)), feature_target[1], 
                              color='skyblue', alpha=0.8)
            axes[i].set_xticks(range(len(feature_target)))
            axes[i].set_xticklabels(feature_target.index, rotation=45, ha='right')
            axes[i].set_title(f'Subscription Rate by {feature.title()}')
            axes[i].set_ylabel('Subscription Rate')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, feature_target[1]):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('ğŸ“Š Categorical Features Impact Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print insights for top categorical features
print("ğŸ”� Key Insights:")
for feature in ['job', 'contact', 'poutcome']:
    if feature in train_clean.columns:
        rates = pd.crosstab(train_clean[feature], train_clean['y'], normalize='index')
        if 1 in rates.columns:
            best_category = rates[1].idxmax()
            best_rate = rates[1].max()
            print(f"   â€¢ {feature.title()}: '{best_category}' has highest rate ({best_rate:.1%})")


# Smart data augmentation (if original dataset available)
def prepare_training_data(train_df, original_df=None):
    """Prepare training data with optional augmentation"""
    
    # Start with synthetic data
    train_final = train_df.drop('id', axis=1).copy()
    
    if original_df is not None:
        print("ğŸ”„ Augmenting with original data...")
        
        # Ensure same columns
        common_cols = set(train_final.columns) & set(original_df.columns)
        train_synthetic = train_final[list(common_cols)]
        original_subset = original_df[list(common_cols)]
        
        # Add a reasonable amount of original data (don't overwhelm synthetic)
        sample_size = min(len(original_subset), len(train_synthetic) // 3)
        original_sample = original_subset.sample(n=sample_size, random_state=42)
        
        # Combine
        train_final = pd.concat([train_synthetic, original_sample], ignore_index=True)
        print(f"   â€¢ Added {len(original_sample):,} original samples")
        print(f"   â€¢ Total training samples: {len(train_final):,}")
    
    return train_final

# Prepare training data
if USE_ORIGINAL:
    train_data = prepare_training_data(train_df, original_df)
else:
    train_data = train_df.drop('id', axis=1).copy()

test_data = test_df.drop('id', axis=1).copy()

print(f"âœ… Final training shape: {train_data.shape}")
print(f"ğŸ’¡ Features: {', '.join([col for col in train_data.columns if col != 'y'][:5])}...")


# Configure and train AutoGluon for maximum performance
print("ğŸš€ Starting AutoGluon Training with Optimal Configuration...")

# Create predictor with best settings
predictor = TabularPredictor(
    label='y',
    eval_metric='roc_auc',
    problem_type='binary',
    path='./autogluon_best'
).fit(
    train_data=train_data,
    presets='best_quality',      # Maximum quality preset
    time_limit=3600 * 4,         # 4 hours training
    num_bag_folds=10,            # More folds = better ensemble
    num_stack_levels=3,          # Deep stacking
    refit_full=True,             # Refit on full data
    set_best_to_refit_full=True, # Use refit as best model
    verbosity=2
)

print("âœ… Training Complete!")


# Get model leaderboard and create performance visualization
leaderboard = predictor.leaderboard(silent=True)

# Create comprehensive performance dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Top 10 Models Performance', 'Training Time vs Accuracy', 
                   'Model Type Distribution', 'Stack Level Performance'),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "pie"}, {"type": "bar"}]]
)

# Top 10 models
top_10 = leaderboard.head(10)
fig.add_trace(go.Bar(
    x=top_10['score_val'], 
    y=top_10['model'].str[:25],  # Truncate long names
    orientation='h',
    marker_color='skyblue'
), row=1, col=1)

# Training time vs performance
fig.add_trace(go.Scatter(
    x=top_10['fit_time'], 
    y=top_10['score_val'],
    mode='markers',
    marker=dict(size=10, color='coral'),
    text=top_10['model'].str[:20]
), row=1, col=2)

# Model type distribution
model_types = top_10['model'].str.extract(r'([A-Za-z]+)')[0].value_counts()
fig.add_trace(go.Pie(
    labels=model_types.index,
    values=model_types.values,
    textinfo='label+percent'
), row=2, col=1)

# Stack level performance
stack_perf = top_10.groupby('stack_level')['score_val'].mean()
fig.add_trace(go.Bar(
    x=stack_perf.index.astype(str),
    y=stack_perf.values,
    marker_color='gold'
), row=2, col=2)

fig.update_layout(height=600, showlegend=False,
                  title_text="ğŸ�† AutoGluon Model Performance Dashboard")
fig.show()

# Print key results
best_score = leaderboard.iloc[0]['score_val']
print(f"\nğŸ�‰ RESULTS:")
print(f"ğŸ�† Best Model: {leaderboard.iloc[0]['model']}")
print(f"ğŸ“Š Best ROC-AUC: {best_score:.6f}")
print(f"ğŸ�¯ Target to Beat: 0.970300")
print(f"ğŸ“ˆ Improvement: {'+' if best_score > 0.970300 else ''}{(best_score - 0.970300)*100:.4f}%")

if best_score > 0.970300:
    print("\nğŸ�† SUCCESS! Benchmark Exceeded! ğŸ�‰")
else:
    print(f"\nğŸ“ˆ Close! Gap: {(0.970300 - best_score)*100:.4f}%")


# Generate predictions
print("ğŸ�¯ Generating Final Predictions...")
predictions = predictor.predict_proba(test_data, as_pandas=True)

# Extract probabilities for class 1 (subscription = yes)
if isinstance(predictions, pd.DataFrame):
    if 1 in predictions.columns:
        pred_probs = predictions[1].values
    else:
        # If columns are named differently
        pred_probs = predictions.iloc[:, 1].values
elif isinstance(predictions, np.ndarray):
    pred_probs = predictions[:, 1] if predictions.ndim > 1 else predictions
else:
    pred_probs = predictions

# Ensure pred_probs is a numpy array
pred_probs = np.array(pred_probs)

# Quick prediction analysis
print(f"ğŸ“Š Prediction Stats:")
print(f"   â€¢ Mean probability: {pred_probs.mean():.4f}")
print(f"   â€¢ Std probability: {pred_probs.std():.4f}")
print(f"   â€¢ Min probability: {pred_probs.min():.6f}")
print(f"   â€¢ Max probability: {pred_probs.max():.6f}")
print(f"   â€¢ Predicted positives: {(pred_probs > 0.5).sum():,} ({(pred_probs > 0.5).mean()*100:.1f}%)")

# Visualize predictions
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(pred_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('ğŸ�¯ Prediction Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
confidence = np.maximum(pred_probs, 1 - pred_probs)
plt.hist(confidence, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Prediction Confidence')
plt.ylabel('Count')
plt.title('ğŸ�¯ Prediction Confidence')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'y': pred_probs
})

submission.to_csv("submission.csv", index=False)
print(f"âœ… Submission saved! Shape: {submission.shape}")
print("\nğŸ“‹ Sample predictions:")
print(submission.head())

