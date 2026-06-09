# imports data reading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from itertools import combinations
import os
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
palette = ["yellowgreen", "palegreen", "forestgreen", "olive", "darkkhaki"]


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train.head()


print(train.head())


test.head()


train.info()
print("=" * 25)
test.info()


train_unq = train.nunique()
test_unq = test.nunique()
print("Number of unique values for Train:","\n",
      train_unq,"\n", "=" * 25,"\n", "Number of unique values for Test:","\n",
      test_unq)


ferts = train["Fertilizer Name"].unique()
crops = train["Crop Type"].unique()
print(f"Ferilizers Uniques: {ferts}","\n" *2, f"Crops Uniques: {crops}")


train.describe()


train = pd.concat([train, original], ignore_index=True)


categorical_cols = train.select_dtypes(include=['object', 'category']).columns

# Set up subplot grid
n_cols = 3
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()  # Convert to 1D array for easy iteration

max_bars = 15

for i, col in enumerate(categorical_cols):
    
    value_counts = train[col].value_counts()
    
    if len(value_counts) > max_bars:
        value_counts = value_counts.head(max_bars)
    

    value_counts.plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(f'Distribution of {col}', fontsize=13)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for p in axes[i].patches:
        axes[i].annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=9)

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=3.0)
plt.suptitle('Categorical Features Distribution', fontsize=18, y=1.02)
plt.show()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
if 'id' in numeric_cols:
    numeric_cols = numeric_cols.drop('id')

n_cols = 2
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
axes = axes.flatten() 

grid_style = dict(alpha=0.3, linestyle='--', linewidth=0.8)

for i, col in enumerate(numeric_cols):
    # Get data
    data = train[col].dropna()
    
    # Create histogram with custom palette color
    hist_color = palette[i % len(palette)]
    n, bins, patches = axes[i].hist(data, 
                                   bins=30, 
                                   color=hist_color,
                                   alpha=0.7,
                                   edgecolor='white',
                                   density=True)  # Use density for KDE
    
    # Add KDE line with complementary color
    kde_color = palette[(i + 2) % len(palette)]  # Skip adjacent color
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 300)
    axes[i].plot(x_vals, kde(x_vals), color=kde_color, linewidth=2.5, label='KDE')
    
    # Add statistical annotations
    mean_val = data.mean()
    median_val = data.median()
    axes[i].axvline(mean_val, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    axes[i].axvline(median_val, color='#2ca02c', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    # Titles and labels
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=11)
    axes[i].set_ylabel('Density', fontsize=11)
    axes[i].grid(**grid_style)
    axes[i].legend(fontsize=9, framealpha=0.7)

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=3.0)
plt.suptitle('Numeric Features Distribution with KDE', fontsize=20, y=1.02)
plt.show()


#Correlation heatmap (Numeric features)
corr = train[numeric_cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='YlGn', square=True,
            xticklabels=numeric_cols, yticklabels=numeric_cols, cbar=True)

plt.title('Correlation Matrix')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


mean_nutrients = (train.groupby("Fertilizer Name")[["Nitrogen", "Potassium", "Phosphorous"]]
                   .mean().reset_index())
mean_nutrients


melted = mean_nutrients.melt(
    id_vars='Fertilizer Name', 
    value_vars=['Nitrogen', 'Potassium', 'Phosphorous'], 
    var_name='Nutrient', 
    value_name='Average Amount'
)


plt.figure(figsize=(12, 6))
sns.barplot(
    data=melted, 
    x='Fertilizer Name', 
    y='Average Amount', 
    hue='Nutrient',
    palette=palette
)
plt.title("Average Nutrient Composition per Fertilizer")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


nutrient_means_by_soil = train.groupby("Soil Type")[["Nitrogen", "Phosphorous", "Potassium"]].mean().reset_index()

# Transform to long format for plotting
df_melted = nutrient_means_by_soil.melt(
    id_vars="Soil Type",
    value_vars=["Nitrogen", "Phosphorous", "Potassium"],
    var_name="Nutrient",
    value_name="Average Level"
)


plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_melted,
    x="Soil Type",
    y="Average Level",
    hue="Nutrient",
    palette = palette
)
plt.title("Average Existing Nutrient Levels by Soil Type")
plt.xticks(rotation=45)
plt.ylabel("Mean Value")
plt.xlabel("Soil Type")
plt.tight_layout()
plt.show()


nutrient_means_by_soil = train.groupby("Soil Type")[["Nitrogen", "Phosphorous", "Potassium"]].mean().reset_index()
nutrient_means_by_soil


train.groupby(["Soil Type","Crop Type","Fertilizer Name"])[["Nitrogen","Phosphorous","Potassium"]].mean()


ct = train.groupby(["Soil Type", "Crop Type", "Fertilizer Name"]).size().unstack(fill_value=0)
ct


for col in ["Temparature", "Humidity", "Moisture"]:
    plt.figure(figsize=(6,3))
    sns.boxplot(data=train, x="Fertilizer Name", y=col, palette=palette)
    plt.xticks(rotation=45)
    plt.title(f"{col} by Fertilizer")
    plt.tight_layout()
    plt.show()


fert_c = pd.crosstab(train["Crop Type"], train["Fertilizer Name"], normalize="index")
fert_c


#Custom MAP@3 implementation
def mapk(y_true, y_pred, k=3):
    """
    Compute MAP@k for single-label ground truths.
    y_true: list of lists, each inner list contains the single true label index.
    y_pred: list of lists, each inner list contains k predicted label indices.
    """
    N = len(y_true)
    scores = []
    for true, preds in zip(y_true, y_pred):
        score = 0.0
        found = False
        for i, p in enumerate(preds[:k], start=1):
            if p in true and not found:
                score = 1.0 / i
                found = True
                break
        scores.append(score)
    return np.mean(scores)


def create_features(df):
    
    # Calc mean for each numerical feature
    #means = df[numeric_cols].mean()
    
    #for feature in numeric_cols:
        # Squared difference from mean
        #df[f'{feature}_mean_sq_diff'] = (df[feature] - means[feature]) ** 2
        
        # Absolute difference from mean
        #df[f'{feature}_mean_abs_diff'] = (df[feature] - means[feature]).abs()
    
    #square and square root features
    #for feature in numeric_cols:
        #df[f'{feature}_sq'] = df[feature] ** 2
        #df[f'{feature}_sqrt'] = np.sqrt(df[feature])
    
    # Calculate Euclidean distance from mean vector
    #df['euclidean_dist'] = np.sqrt(
        #df['Temparature_mean_sq_diff'] +
        #df['Humidity_mean_sq_diff'] +
        #df['Moisture_mean_sq_diff'] +
        #df['Nitrogen_mean_sq_diff'] +
        #df['Potassium_mean_sq_diff'] +
        #df['Phosphorous_mean_sq_diff']
    #)
    
    return df

# Apply feature engineering to train and test datasets
#train = create_features(train)
#test = create_features(test)


cat_cols= ['Soil Type', 'Crop Type']
feature_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    feature_encoders[col] = le

# encode the target column
target_encoder = LabelEncoder()
train['target_encoded'] = target_encoder.fit_transform(train['Fertilizer Name'])


"""
columns_to_encode = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
       'Nitrogen', 'Potassium', 'Phosphorous']



pair_size = [2, 3, 4]

for r in pair_size: 
    combinations_list = list(combinations(columns_to_encode,r))
    batch_size = 20
    
    for i in range(0, len(combinations_list), batch_size):
        batch = combinations_list[i:i+batch_size]
        for cols in tqdm(batch):
            new_col_name = '_'.join(cols)

            train[new_col_name] = train[list(cols)].astype(str).agg('_'.join, axis=1) 
            train[new_col_name] = train[new_col_name].astype('category')

            test[new_col_name] = test[list(cols)].astype(str).agg('_'.join, axis=1) 
            test[new_col_name] = test[new_col_name].astype('category')
        gc.collect()
        print(f"Memory usage: {train.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print(f"Total number of columns: {len(train.columns)}")
"""


features = [
    'Temparature',
    'Humidity',
    'Moisture',
    'Nitrogen',
    'Potassium',
    'Phosphorous',
    'Soil Type',
    'Crop Type'
]
X = train.drop(columns=['id', 'Fertilizer Name', 'target_encoded'])                
y = train['target_encoded'].values       

X_test  = test.drop(columns=["id"]).copy()    


#cross-validation with XGBoost

def map3_scorer(y_true, y_pred_probs):
    """
    y_true: arrayâ€�like of shape (n_samples,)
    y_pred_probs: array of shape (n_samples, n_classes)
    Returns MAP@3.
    """
    y_true_wrapped = [[int(lbl)] for lbl in y_true]
    top3 = np.argsort(-y_pred_probs, axis=1)[:, :3].tolist()
    return mapk(y_true_wrapped, top3, 3)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
map3_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
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
    device = "cuda"
    )
    
    model.fit(
        X_train, 
        y_train,  
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    val_probs = model.predict_proba(X_val)
    score = map3_scorer(y_val, val_probs)
    map3_scores.append(score)
    print(f"Fold {fold} MAP@3: {score:.4f}")

print(f"\nMean MAP@3 across 5 folds: {np.mean(map3_scores):.4f}")


# Train final model & predict on test

model_xgb = xgb.XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=13,
    tree_method='hist',
    device = "cuda"
)
model_xgb.fit(X, y)
xgb_probs = model_xgb.predict_proba(X_test)


top3_idx = np.argsort(xgb_probs, axis=1)[:, -3:][:, ::-1]
flat = top3_idx.flatten()
names_flat = target_encoder.inverse_transform(flat)
top3_names = names_flat.reshape(top3_idx.shape) 

predictions = [" ".join(row) for row in top3_names]


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': predictions
})
submission.to_csv('submission.csv', index=False)
print("\n'submission.csv' has been created.")

