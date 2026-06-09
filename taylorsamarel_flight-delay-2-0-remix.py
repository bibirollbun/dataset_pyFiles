# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("="*80)
print(" "*20 + "âœˆï¸�  ADVANCED FLIGHT DELAY PREDICTION SYSTEM  âœˆï¸�")
print("="*80)

# ========== DATA GENERATION ==========
def generate_realistic_flight_data(n_samples=100000, test_mode=False):
    """Generate realistic flight data with complex patterns"""
    
    # Expanded lists for more realistic data
    carriers = ['AA', 'UA', 'DL', 'WN', 'B6', 'AS', 'NK', 'F9', 'G4', 'HA', 
                'SY', 'VX', 'EV', 'OO', 'YX', 'OH', 'ZW', 'MQ', '9E', 'YV', 'FL', 'US']
    
    major_hubs = ['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'DEN', 'SFO', 'LAS', 'PHX', 'IAH']
    secondary = ['SEA', 'MCO', 'EWR', 'MSP', 'BOS', 'DTW', 'PHL', 'LGA', 'FLL', 'BWI']
    regional = ['DCA', 'SLC', 'MDW', 'HNL', 'SAN', 'TPA', 'PDX', 'DAL', 'STL', 'HOU']
    all_airports = major_hubs + secondary + regional
    
    data = pd.DataFrame({
        'Month': np.random.choice([f'c-{i}' for i in range(1, 13)], n_samples),
        'DayofMonth': np.random.choice([f'c-{i}' for i in range(1, 32)], n_samples),
        'DayOfWeek': np.random.choice([f'c-{i}' for i in range(1, 8)], n_samples),
        'DepTime': np.random.normal(1330, 450, n_samples).clip(1, 2400).astype(int),
        'UniqueCarrier': np.random.choice(carriers, n_samples, p=np.random.dirichlet(np.ones(len(carriers)))),
        'Origin': np.random.choice(all_airports, n_samples),
        'Dest': np.random.choice(all_airports, n_samples),
        'Distance': np.random.gamma(2, 350, n_samples).clip(100, 3500).astype(int)
    })
    
    if not test_mode:
        # Complex delay patterns
        delays = []
        for idx, row in data.iterrows():
            base_prob = 0.18
            
            # Time factors with more granularity
            hour = row['DepTime'] // 100
            if hour in [6, 7, 8]:  # Morning rush
                base_prob += 0.12
            elif hour in [16, 17, 18, 19]:  # Evening rush
                base_prob += 0.15
            elif hour in [22, 23, 0, 1, 2]:  # Late night
                base_prob += 0.08
            
            # Day patterns
            day = int(row['DayOfWeek'].split('-')[1])
            if day == 5:  # Friday
                base_prob += 0.08
            elif day == 1:  # Monday
                base_prob += 0.06
            elif day in [6, 7]:  # Weekend
                base_prob -= 0.03
            
            # Seasonal patterns
            month = int(row['Month'].split('-')[1])
            if month in [6, 7, 8]:  # Summer
                base_prob += 0.07
            elif month in [11, 12]:  # Holidays
                base_prob += 0.10
            elif month in [1, 2]:  # Winter weather
                base_prob += 0.09
            
            # Carrier reliability
            if row['UniqueCarrier'] in ['NK', 'F9', 'G4']:
                base_prob += 0.10
            elif row['UniqueCarrier'] in ['DL', 'AS']:
                base_prob -= 0.05
            
            # Hub congestion
            if row['Origin'] in major_hubs or row['Dest'] in major_hubs:
                base_prob += 0.04
            
            # Distance effects
            if row['Distance'] > 2000:
                base_prob += 0.05
            elif row['Distance'] < 300:
                base_prob -= 0.03
            
            # Add some randomness
            base_prob *= np.random.uniform(0.8, 1.2)
            delays.append('Y' if np.random.random() < min(base_prob, 0.65) else 'N')
        
        data['dep_delayed_15min'] = delays
    
    return data

# Generate data
print("\nğŸ“Š Generating Enhanced Training Data...")
train = generate_realistic_flight_data(100000, test_mode=False)
test = generate_realistic_flight_data(100000, test_mode=True)

print(f"âœ“ Training samples: {len(train):,}")
print(f"âœ“ Test samples: {len(test):,}")
print(f"âœ“ Features: {train.shape[1] - 1}")
print(f"âœ“ Delay rate: {(train['dep_delayed_15min'] == 'Y').mean():.2%}")

# ========== DATA PREPROCESSING ==========
print("\nğŸ”§ Advanced Feature Engineering...")

all_data = pd.concat([train, test], ignore_index=True, sort=False)

# Convert target
train['delayed'] = (train['dep_delayed_15min'] == 'Y').astype(int)
all_data['delayed'] = 0
all_data.loc[all_data['dep_delayed_15min'] == 'Y', 'delayed'] = 1

# Convert categorical columns
for col in ['Month', 'DayofMonth', 'DayOfWeek']:
    all_data[col] = all_data[col].str.replace('c-', '').astype(int)
    train[col] = train[col].str.replace('c-', '').astype(int)

# Advanced feature engineering
all_data['Route'] = all_data['Origin'] + '_' + all_data['Dest']
all_data['UniqueCarrier_Origin'] = all_data['UniqueCarrier'] + "_" + all_data['Origin']
all_data['UniqueCarrier_Dest'] = all_data['UniqueCarrier'] + "_" + all_data['Dest']

# Time features
all_data['hour'] = all_data['DepTime'] // 100
all_data['minute'] = all_data['DepTime'] % 100
all_data['hour_sin'] = np.sin(2 * np.pi * all_data['hour'] / 24)
all_data['hour_cos'] = np.cos(2 * np.pi * all_data['hour'] / 24)
all_data['day_sin'] = np.sin(2 * np.pi * all_data['DayOfWeek'] / 7)
all_data['day_cos'] = np.cos(2 * np.pi * all_data['DayOfWeek'] / 7)
all_data['month_sin'] = np.sin(2 * np.pi * all_data['Month'] / 12)
all_data['month_cos'] = np.cos(2 * np.pi * all_data['Month'] / 12)

# Boolean features
all_data['is_weekend'] = all_data['DayOfWeek'].isin([6, 7]).astype(int)
all_data['is_rush_hour'] = all_data['hour'].isin([6,7,8,16,17,18,19]).astype(int)
all_data['is_holiday_season'] = all_data['Month'].isin([11, 12, 7, 8]).astype(int)

# Seasonal features
all_data['summer'] = all_data['Month'].isin([6, 7, 8]).astype(int)
all_data['autumn'] = all_data['Month'].isin([9, 10, 11]).astype(int)
all_data['winter'] = all_data['Month'].isin([12, 1, 2]).astype(int)
all_data['spring'] = all_data['Month'].isin([3, 4, 5]).astype(int)

# Time binning
time_bins = [0, 600, 900, 1200, 1500, 1800, 2100, 2400]
time_labels = ['night', 'early_morning', 'morning', 'afternoon', 'late_afternoon', 'evening', 'late_evening']
all_data['time_bin'] = pd.cut(all_data['DepTime'], bins=time_bins, labels=time_labels, include_lowest=True)

# Distance binning
dist_bins = [0, 300, 600, 1000, 1500, 2000, 5000]
dist_labels = ['very_short', 'short', 'medium', 'medium_long', 'long', 'very_long']
all_data['dist_bin'] = pd.cut(all_data['Distance'], bins=dist_bins, labels=dist_labels, include_lowest=True)

# Create interaction features
all_data['rush_hour_friday'] = (all_data['is_rush_hour'] * (all_data['DayOfWeek'] == 5)).astype(int)
all_data['long_holiday_flight'] = ((all_data['Distance'] > 1500) * all_data['is_holiday_season']).astype(int)

# Drop columns
all_data = all_data.drop(['DepTime', 'Distance', 'dep_delayed_15min'], axis=1, errors='ignore')

print(f"âœ“ Created {len(all_data.columns)} features")

# ========== COMPREHENSIVE VISUALIZATIONS ==========
print("\nğŸ“Š Creating Advanced Visualizations...")

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Overall delay distribution
ax1 = fig.add_subplot(gs[0, 0])
delay_dist = train['delayed'].value_counts()
ax1.pie(delay_dist.values, labels=['On Time', 'Delayed'], autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90)
ax1.set_title('Overall Delay Distribution', fontsize=12, fontweight='bold')

# 2. Delays by hour (heatmap style)
ax2 = fig.add_subplot(gs[0, 1:3])
hourly_delays = train.groupby('hour')['delayed'].agg(['mean', 'count'])
bars = ax2.bar(hourly_delays.index, hourly_delays['mean'], 
               color=plt.cm.RdYlGn_r(hourly_delays['mean']))
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Delay Rate')
ax2.set_title('Delay Patterns Throughout the Day', fontsize=12, fontweight='bold')
ax2.set_xticks(range(0, 24, 2))

# 3. Day of week pattern
ax3 = fig.add_subplot(gs[0, 3])
day_delays = train.groupby('DayOfWeek')['delayed'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax3.plot(day_delays.index, day_delays.values, 'o-', linewidth=2, markersize=8, color='#3498db')
ax3.set_xticks(range(1, 8))
ax3.set_xticklabels(days, rotation=45)
ax3.set_ylabel('Delay Rate')
ax3.set_title('Weekly Delay Pattern', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Monthly patterns
ax4 = fig.add_subplot(gs[1, 0:2])
monthly_delays = train.groupby('Month')['delayed'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['#3498db' if m not in [6,7,8,11,12] else '#e74c3c' for m in range(1, 13)]
ax4.bar(monthly_delays.index, monthly_delays.values, color=colors)
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(months, rotation=45)
ax4.set_ylabel('Delay Rate')
ax4.set_title('Monthly Delay Patterns (Red = Peak Seasons)', fontsize=12, fontweight='bold')

# 5. Carrier performance
ax5 = fig.add_subplot(gs[1, 2:4])
carrier_stats = train.groupby('UniqueCarrier')['delayed'].agg(['mean', 'count'])
carrier_stats = carrier_stats[carrier_stats['count'] > 500].sort_values('mean', ascending=True)
top_carriers = carrier_stats.tail(10)
colors_carrier = ['#2ecc71' if x < 0.2 else '#f39c12' if x < 0.25 else '#e74c3c' 
                  for x in top_carriers['mean']]
ax5.barh(range(len(top_carriers)), top_carriers['mean'], color=colors_carrier)
ax5.set_yticks(range(len(top_carriers)))
ax5.set_yticklabels(top_carriers.index)
ax5.set_xlabel('Delay Rate')
ax5.set_title('Carrier Performance (Top 10 by Volume)', fontsize=12, fontweight='bold')

# 6. Distance vs Delays
ax6 = fig.add_subplot(gs[2, 0:2])
dist_data = train.copy()
dist_data['dist_group'] = pd.cut(dist_data['Distance'], bins=10)
dist_delays = dist_data.groupby('dist_group')['delayed'].mean()
ax6.plot(range(len(dist_delays)), dist_delays.values, 'o-', linewidth=2, color='#9b59b6')
ax6.set_xlabel('Distance Group (Short â†’ Long)')
ax6.set_ylabel('Delay Rate')
ax6.set_title('Delay Rate by Flight Distance', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Time of day heatmap
ax7 = fig.add_subplot(gs[2, 2:4])
pivot_data = train.pivot_table(values='delayed', index='DayOfWeek', columns='hour', aggfunc='mean')
sns.heatmap(pivot_data, cmap='RdYlGn_r', ax=ax7, cbar_kws={'label': 'Delay Rate'})
ax7.set_yticklabels(days, rotation=0)
ax7.set_xlabel('Hour of Day')
ax7.set_ylabel('Day of Week')
ax7.set_title('Delay Heatmap: Day vs Hour', fontsize=12, fontweight='bold')

# 8. Feature correlation matrix (subset)
ax8 = fig.add_subplot(gs[3, :2])
numeric_cols = ['Month', 'DayOfWeek', 'hour', 'minute', 'is_weekend', 
                'is_rush_hour', 'is_holiday_season', 'delayed']
corr_data = train[numeric_cols].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax8,
            cbar_kws={'label': 'Correlation'})
ax8.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# 9. Delay rate distribution
ax9 = fig.add_subplot(gs[3, 2:])
# Calculate delay rates by various groupings
group_delays = []
for carrier in train['UniqueCarrier'].unique()[:20]:
    carrier_delay_rate = train[train['UniqueCarrier'] == carrier]['delayed'].mean()
    group_delays.append(carrier_delay_rate)

ax9.hist(group_delays, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
ax9.axvline(x=np.mean(group_delays), color='red', linestyle='--', label=f'Mean: {np.mean(group_delays):.3f}')
ax9.set_xlabel('Delay Rate')
ax9.set_ylabel('Frequency')
ax9.set_title('Distribution of Delay Rates Across Carriers', fontsize=12, fontweight='bold')
ax9.legend()

plt.suptitle('Flight Delay Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ========== AUTOML WITH MULTIPLE MODELS ==========
print("\nğŸ¤– AutoML Model Training & Optimization...")

# Prepare data
new_train = all_data.iloc[:100000]
new_test = all_data.iloc[100000:]

feature_columns = [col for col in new_train.columns if col != 'delayed']
X = new_train[feature_columns]
y = new_train['delayed'].fillna(0).astype(int)

# Encode categorical features
print("\nğŸ”„ Encoding categorical features...")
X_encoded = X.copy()
X_test_encoded = new_test[feature_columns].copy()

label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        all_values = pd.concat([X[col], X_test_encoded[col]]).fillna('missing')
        le.fit(all_values)
        X_encoded[col] = le.transform(X[col].fillna('missing'))
        X_test_encoded[col] = le.transform(X_test_encoded[col].fillna('missing'))
        label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
X_test_encoded[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ========== AUTOML MODEL COMPARISON ==========
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Hyperparameter grids for tuning
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    },
    'Extra Trees': {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [10, 20]
    },
    'AdaBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.5, 1.0]
    },
    'Logistic Regression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'lbfgs']
    }
}

print("\nğŸ�� Training and tuning multiple models...")
results = {}

for name, model in models.items():
    print(f"\nğŸ“ˆ {name}:")
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=3,  # 3-fold cross-validation
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    results[name] = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ========== MODEL COMPARISON VISUALIZATION ==========
print("\nğŸ“Š Model Performance Comparison...")

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. AUC Comparison
ax = axes[0, 0]
model_names = list(results.keys())
auc_scores = [results[m]['auc'] for m in model_names]
bars = ax.bar(range(len(model_names)), auc_scores, color='skyblue')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('AUC Score')
ax.set_title('Model Comparison - AUC Score', fontweight='bold')
ax.set_ylim([min(auc_scores) * 0.95, max(auc_scores) * 1.02])

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom')

# 2. Multiple Metrics Comparison
ax = axes[0, 1]
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(model_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in model_names]
    ax.bar(x + i * width, values, width, label=metric.capitalize())

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('Score')
ax.set_title('Multiple Metrics Comparison', fontweight='bold')
ax.legend()

# 3. Cross-validation scores
ax = axes[0, 2]
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]
ax.errorbar(range(len(model_names)), cv_means, yerr=cv_stds, 
            fmt='o', markersize=8, capsize=5, capthick=2)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('CV AUC Score')
ax.set_title('Cross-Validation Performance', fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. ROC Curves for all models
ax = axes[1, 0]
for name, result in results.items():
    model = result['model']
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 5. Feature Importance (Best Model)
ax = axes[1, 1]
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    ax.barh(range(len(feature_imp)), feature_imp['importance'], color='coral')
    ax.set_yticks(range(len(feature_imp)))
    ax.set_yticklabels(feature_imp['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 15 Features ({best_model_name})', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Feature importance\nnot available', 
            ha='center', va='center', fontsize=12)
    ax.set_title('Feature Importance', fontweight='bold')

# 6. Confusion Matrix (Best Model)
ax = axes[1, 2]
y_pred = best_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Delay', 'Delay'],
            yticklabels=['No Delay', 'Delay'])
ax.set_title(f'Confusion Matrix ({best_model_name})', fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

plt.suptitle('AutoML Model Performance Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== FEATURE SELECTION ==========
print("\nğŸ”� Advanced Feature Selection...")

# Select K Best features
selector = SelectKBest(f_classif, k=30)
X_train_selected = selector.fit_transform(X_train, y_train)
selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]

print(f"âœ“ Selected top {len(selected_features)} features")
print(f"  Top 10: {selected_features[:10]}")

# ========== ENSEMBLE MODEL ==========
print("\nğŸ�¯ Creating Ensemble Model...")

# Get top 3 models
top_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]

# Create ensemble predictions
ensemble_preds = []
weights = []

for name, result in top_models:
    model = result['model']
    pred = model.predict_proba(X_val)[:, 1]
    ensemble_preds.append(pred)
    weights.append(result['auc'])  # Weight by AUC score

# Weighted average
weights = np.array(weights) / sum(weights)
ensemble_pred = np.average(ensemble_preds, weights=weights, axis=0)

# Calculate ensemble metrics
ensemble_auc = roc_auc_score(y_val, ensemble_pred)
print(f"\nâœ¨ Ensemble Model Performance:")
print(f"  AUC: {ensemble_auc:.4f}")
print(f"  Models used: {[name for name, _ in top_models]}")
print(f"  Weights: {weights}")

# ========== FINAL PREDICTIONS ==========
print("\nğŸ�² Generating Final Predictions...")

# Use the best single model or ensemble
final_model = results[best_model_name]['model']
final_model.fit(X_encoded, y)

# Predict on test set
test_predictions = final_model.predict_proba(X_test_encoded)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': range(len(test_predictions)),
    'dep_delayed_15min': test_predictions
})

# ========== FINAL SUMMARY ==========
print("\n" + "="*80)
print(" "*30 + "ğŸ“Š FINAL SUMMARY")
print("="*80)

print(f"\nğŸ�† Best Model: {best_model_name}")
print(f"   Parameters: {results[best_model_name]['best_params']}")
print(f"   AUC Score: {results[best_model_name]['auc']:.4f}")
print(f"   CV Score: {results[best_model_name]['cv_mean']:.4f} Â± {results[best_model_name]['cv_std']:.4f}")

print("\nğŸ“ˆ Prediction Statistics:")
print(f"   Min probability: {test_predictions.min():.4f}")
print(f"   Max probability: {test_predictions.max():.4f}")
print(f"   Mean probability: {test_predictions.mean():.4f}")
print(f"   Std deviation: {test_predictions.std():.4f}")

print("\nğŸ“Š Model Rankings by AUC:")
for i, (name, result) in enumerate(sorted(results.items(), 
                                          key=lambda x: x[1]['auc'], 
                                          reverse=True), 1):
    print(f"   {i}. {name}: {result['auc']:.4f}")

# Save submission
submission.to_csv('advanced_flight_delays.csv', index=False)
print(f"\nâœ… Predictions saved to 'advanced_flight_delays.csv'")
print("="*80)

