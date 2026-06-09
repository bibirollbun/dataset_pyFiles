import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# ======================
# Basic Dataset Overview
# ======================
print("📦 Dataset Shapes:")
print(f"Train: {train_df.shape}")
print(f"Test : {test_df.shape}")

# Data types and nulls
print("\n🧾 Data Types:")
print(train_df.dtypes)

print("\n🧼 Missing Values:")
print(train_df.isnull().sum())

# ======================
# Target Variable Analysis
# ======================
print("\n🎯 Target Variable - 'Fertilizer Name'")
print(f"Number of classes: {train_df['Fertilizer Name'].nunique()}")

print("\n📊 Class Distribution:")
print(train_df['Fertilizer Name'].value_counts())

# ======================
# Target Distribution Plot
# ======================
plt.figure(figsize=(10, 6))
sns.countplot(y='Fertilizer Name', data=train_df, 
              order=train_df['Fertilizer Name'].value_counts().index,
              palette="Set2")
plt.title('Target Class Distribution: Fertilizer Name', fontsize=14)
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print(f"📥 Loaded: {len(train_df)} training samples, {len(test_df)} test samples")

# ============================
# 1) Split Features (X) and Target (y)
# ============================
X = train_df.drop(['Fertilizer Name', 'id'], axis=1)
y = train_df['Fertilizer Name']

# ============================
# 2) Encode Categorical Features
# ============================
categorical_cols = ['Soil Type', 'Crop Type']

le_soil = LabelEncoder()
le_crop = LabelEncoder()

X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])

# Safe transform for unseen values
def safe_transform(encoder, series):
    known_classes = set(encoder.classes_)
    fallback_value = encoder.transform([encoder.classes_[0]])[0]
    
    transformed = []
    unknown_count = 0
    for val in series:
        if val in known_classes:
            transformed.append(encoder.transform([val])[0])
        else:
            transformed.append(fallback_value)
            unknown_count += 1
    
    if unknown_count > 0:
        print(f"⚠️ {unknown_count} unknown values found in {encoder.__class__.__name__}")
    
    return transformed

# Apply safe transform to test set
X_test = test_df.drop(['id'], axis=1).copy()
X_test['Soil Type'] = safe_transform(le_soil, X_test['Soil Type'])
X_test['Crop Type'] = safe_transform(le_crop, X_test['Crop Type'])

# ============================
# 3) Encode the Target
# ============================
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ============================
# 4) Train/Validation Split
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# ============================
# 5) Save Artifacts
# ============================

# 5.1 Encoders
with open('artifacts/le_soil.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

with open('artifacts/le_crop.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

with open('artifacts/le_target.pkl', 'wb') as f:
    pickle.dump(le_target, f)

# 5.2 Save as numpy arrays
np.save('artifacts/X_train.npy', X_train.values)
np.save('artifacts/X_val.npy', X_val.values)
np.save('artifacts/y_train.npy', y_train)
np.save('artifacts/y_val.npy', y_val)
np.save('artifacts/X_test.npy', X_test.values)

# 5.3 Save also as CSVs for debugging
pd.DataFrame(X_train, columns=X.columns).to_csv('artifacts/X_train.csv', index=False)
pd.DataFrame(X_val, columns=X.columns).to_csv('artifacts/X_val.csv', index=False)
pd.DataFrame(X_test, columns=X_test.columns).to_csv('artifacts/X_test.csv', index=False)
pd.Series(y_train).to_csv('artifacts/y_train.csv', index=False, header=['target'])
pd.Series(y_val).to_csv('artifacts/y_val.csv', index=False, header=['target'])

# 5.4 Save metadata
metadata = {
    'n_train_samples': len(X_train),
    'n_val_samples': len(X_val),
    'n_test_samples': len(X_test),
    'n_features': X_train.shape[1],
    'feature_names': X.columns.tolist(),
    'target_classes': le_target.classes_.tolist(),
    'n_target_classes': len(le_target.classes_),
    'categorical_encoders': {
        'soil_classes': le_soil.classes_.tolist(),
        'crop_classes': le_crop.classes_.tolist()
    },
    'split_params': {
        'test_size': 0.2,
        'random_state': 42,
        'stratified': True
    }
}

with open('artifacts/preprocessing_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# ============================
# 6) Final Checks
# ============================
print("\n" + "="*50)
print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*50)

print(f"\n📊 SHAPES:")
print(f"   X_train: {X_train.shape}")
print(f"   X_val  : {X_val.shape}")
print(f"   X_test : {X_test.shape}")

print(f"\n🎯 TARGET DISTRIBUTION:")
unique, counts = np.unique(y_train, return_counts=True)
for class_idx, count in zip(unique, counts):
    class_name = le_target.classes_[class_idx]
    pct = (count / len(y_train)) * 100
    print(f"   {class_name}: {count} ({pct:.1f}%)")

print(f"\n🔤 CATEGORICAL FEATURES:")
print(f"   Soil Type: {len(le_soil.classes_)} unique classes")
print(f"   Crop Type: {len(le_crop.classes_)} unique classes")

print(f"\n💾 SAVED ARTIFACTS:")
artifacts = [
    'le_soil.pkl', 'le_crop.pkl', 'le_target.pkl',
    'X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy', 'X_test.npy',
    'X_train.csv', 'X_val.csv', 'X_test.csv', 'y_train.csv', 'y_val.csv',
    'preprocessing_metadata.pkl'
]
for artifact in artifacts:
    path = f'artifacts/{artifact}'
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024  # KB
        print(f"   ✓ {artifact} ({size:.1f} KB)")

print(f"\n🚀 Ready for Part 3: LightGBM Baseline")



# Core packages
import os
import pickle
import pandas as pd
import numpy as np

# Model
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



# Load preprocessed datasets
X_train = pd.read_csv('artifacts/X_train.csv')
X_val   = pd.read_csv('artifacts/X_val.csv')
X_test  = pd.read_csv('artifacts/X_test.csv')
y_train = np.load('artifacts/y_train.npy')
y_val   = np.load('artifacts/y_val.npy')

# Load label encoder
with open('artifacts/le_target.pkl', 'rb') as f:
    le_target = pickle.load(f)

print("✅ Data loaded successfully")



from lightgbm import LGBMClassifier, early_stopping, log_evaluation

clf = LGBMClassifier(
    objective='multiclass',
    num_class=len(le_target.classes_),
    learning_rate=0.1,
    n_estimators=300,
    random_state=42
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(20),
        log_evaluation(50)
    ]
)



from sklearn.metrics import accuracy_score

y_val_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
print(f"✅ Validation Accuracy: {acc:.4f}")



y_test_proba = clf.predict_proba(X_test)
top_3_indices = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]

final_predictions = []
for row in top_3_indices:
    preds = le_target.inverse_transform(row)
    final_predictions.append(" ".join(preds))



test_ids = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')['id']
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': final_predictions
})
submission.to_csv('submission.csv', index=False)

print("✅ 'submission.csv' is ready for Kaggle submission!")
submission.head()



from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le_target.classes_),
    random_state=42
)

param_grid = {
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63],
    'n_estimators': [100, 300]
}

grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("✅ Best Parameters:")
print(grid_search.best_params_)



import matplotlib.pyplot as plt
import pandas as pd

feature_importance = best_model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance - LightGBM")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()



def mapk(actual, predicted, k=3):
    score = 0.0
    for i in range(len(actual)):
        pred = predicted[i]
        target = actual[i]
        try:
            index = pred.index(target)
            if index < k:
                score += 1.0 / (index + 1)
        except ValueError:
            continue
    return score / len(actual)

# Predictions
y_val_proba = best_model.predict_proba(X_val)
top_3_preds = np.argsort(y_val_proba, axis=1)[:, -3:][:, ::-1]
top_3_list = [list(row) for row in top_3_preds]

map3 = mapk(y_val.tolist(), top_3_list)
print(f"✅ MAP@3 on Validation Set: {map3:.4f}")



import pickle
import os

# Define the artifacts directory (use 'artifacts' as in the notebook)
base_dir = 'artifacts'
os.makedirs(base_dir, exist_ok=True)

# Save the best model
with open(os.path.join(base_dir, 'best_model_part4.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Optimized model saved as 'best_model_part4.pkl'")

# Save enhanced test set for Part 5
X_test.to_csv(os.path.join(base_dir, 'X_test_enhanced.csv'), index=False)
print("✅ X_test saved as 'X_test_enhanced.csv'")



import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Paths
base_dir = 'artifacts'

# Load enhanced test features (or fallback to original test set)
try:
    X_test_enhanced = pd.read_csv(os.path.join(base_dir, 'X_test_enhanced.csv'))
    print(f"✅ Enhanced test data loaded: {X_test_enhanced.shape}")
except FileNotFoundError:
    print("⚠️ Enhanced test data not found. Using original test data instead.")
    X_test_enhanced = pd.read_csv(os.path.join(base_dir, 'X_test.csv'))

# Load best model
with open(os.path.join(base_dir, 'best_model_part4.pkl'), 'rb') as f:
    best_model = pickle.load(f)

# Load target label encoder
with open(os.path.join(base_dir, 'le_target.pkl'), 'rb') as f:
    le_target = pickle.load(f)

print(f"🎯 Number of classes: {len(le_target.classes_)}")
print(f"📝 Classes: {list(le_target.classes_)}")



# Predict probabilities
y_test_proba = best_model.predict_proba(X_test_enhanced)

# Confidence scores
max_probabilities = np.max(y_test_proba, axis=1)
mean_confidence = np.mean(max_probabilities)
std_confidence = np.std(max_probabilities)

print(f"✅ Test predictions shape: {y_test_proba.shape}")
print(f"📊 Mean confidence: {mean_confidence:.4f} | Std: {std_confidence:.4f}")

# Apply Top-k confidence strategy
top_3_indices = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
top_3_probabilities = np.sort(y_test_proba, axis=1)[:, -3:][:, ::-1]

final_predictions = []
prediction_details = []

for i, (indices, probs) in enumerate(zip(top_3_indices, top_3_probabilities)):
    fertilizer_names = le_target.inverse_transform(indices)
    if probs[0] > 0.7:
        prediction = fertilizer_names[0]
    elif probs[0] > 0.4:
        prediction = " ".join(fertilizer_names[:2])
    else:
        prediction = " ".join(fertilizer_names)
    
    final_predictions.append(prediction)
    prediction_details.append({
        'id': i,
        'top1': fertilizer_names[0],
        'top2': fertilizer_names[1],
        'top3': fertilizer_names[2],
        'prob1': probs[0],
        'prob2': probs[1],
        'prob3': probs[2],
        'strategy': 'top1' if probs[0] > 0.7 else ('top2' if probs[0] > 0.4 else 'top3')
    })



strategy_counts = pd.DataFrame(prediction_details)['strategy'].value_counts()
primary_predictions = [pred.split()[0] for pred in final_predictions]
prediction_distribution = pd.Series(primary_predictions).value_counts()

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Histogram of confidence
ax1.hist(max_probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(mean_confidence, color='red', linestyle='--', label=f'Mean: {mean_confidence:.3f}')
ax1.set_title('Confidence Distribution')
ax1.set_xlabel('Max Probability')
ax1.set_ylabel('Frequency')
ax1.legend()

# Main prediction distribution
prediction_distribution.plot(kind='bar', ax=ax2, color='lightcoral')
ax2.set_title('Main Fertilizer Prediction Distribution')
ax2.set_xlabel('Fertilizer')
ax2.set_ylabel('Count')

# Strategy usage
strategy_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', startangle=90)
ax3.set_title('Prediction Strategy Usage')
ax3.set_ylabel('')

# Top-3 probability boxplot
prob_data = pd.DataFrame(prediction_details)[['prob1', 'prob2', 'prob3']]
ax4.boxplot([prob_data['prob1'], prob_data['prob2'], prob_data['prob3']], labels=['Top-1', 'Top-2', 'Top-3'])
ax4.set_title('Top-3 Probability Distribution')
ax4.set_ylabel('Probability')

plt.tight_layout()
plt.show()



# Load test IDs
test_ids = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')['id']


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': final_predictions
})

# Save files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f'submission_final_{timestamp}.csv'

submission.to_csv(submission_filename, index=False)
submission.to_csv('submission.csv', index=False)  # default for Kaggle

print(f"✅ Submission saved as: {submission_filename}")



# Check basic structure
print("Submission rows:", len(submission))
print("Columns:", submission.columns.tolist())
print("Null values:", submission.isnull().sum().sum())

# Preview predictions
print("Sample predictions:")
print(submission.head(10))

# Validate predicted classes
all_predicted_classes = set()
for pred in final_predictions:
    all_predicted_classes.update(pred.split())

invalid_classes = all_predicted_classes - set(le_target.classes_)
if invalid_classes:
    print("⚠️ Invalid classes found:", invalid_classes)
else:
    print("✅ All predicted classes are valid.")

# Save prediction details
details_df = pd.DataFrame(prediction_details)
details_df['id'] = test_ids
details_df.to_csv(f'prediction_details_{timestamp}.csv', index=False)
print("📄 Prediction details saved.")


