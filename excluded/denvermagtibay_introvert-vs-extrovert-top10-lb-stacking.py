import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("talk")
plt.rc("font", family="SimHei", size=15)


train_path = "/kaggle/input/playground-series-s5e7/train.csv"
test_path = "/kaggle/input/playground-series-s5e7/test.csv"
datasert_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
ds = pd.read_csv(datasert_path)

# Define merge columns first
merge_cols = [
    'Time_spent_Alone', 
    'Stage_fear', 
    'Social_event_attendance',
    'Going_outside', 
    'Drained_after_socializing', 
    'Friends_circle_size', 
    'Post_frequency'
]

# Clean and dedupe datasert DataFrame
ds = (
    ds.rename(columns={'Personality': 'match_p'})
      .drop_duplicates(merge_cols)
)

# Merge
train = train.merge(ds, how='left', on=merge_cols)
test = test.merge(ds, how='left', on=merge_cols)


import matplotlib.pyplot as plt
import seaborn as sns

# Custom Theme
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
custom_palette = sns.color_palette("Set2")

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "axes.edgecolor": "black",
    "grid.color": ".8"
})

# Extract numeric features from the train set
num = train.select_dtypes(include='number').drop(columns=['id'])

# Now draw the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1, linewidths=0.5, linecolor='white')
plt.title("ğŸ”¥ Feature Correlation Heatmap", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


train_ID = train['id']
test_ID = test['id']
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

y = train['Personality'].map({'Extrovert':1, 'Introvert':0}).values


all_df = pd.concat([train, test], ignore_index=True).drop(columns=['Personality'])


def fill_by_quantile(df, src, tgt, quantiles=[0, .25, .5, .75, 1.0]):
    labels = [f"Q{i}" for i in range(1, len(quantiles))]
    bin_col = f"{src}_bin"
    df[bin_col] = pd.qcut(df[src], q=quantiles, labels=labels)
    df[tgt] = df[tgt].fillna(df.groupby(bin_col)[tgt].transform('median'))
    df.drop(columns=[bin_col], inplace=True)
    return df


all_df = fill_by_quantile(all_df, 'Social_event_attendance', 'Time_spent_Alone')
all_df = fill_by_quantile(all_df, 'Social_event_attendance', 'Going_outside')
all_df = fill_by_quantile(all_df, 'Post_frequency', 'Friends_circle_size')
all_df = fill_by_quantile(all_df, 'Going_outside', 'Friends_circle_size')
all_df = fill_by_quantile(all_df, 'Friends_circle_size', 'Post_frequency')

all_df[['Stage_fear', 'Drained_after_socializing']] = all_df[['Stage_fear','Drained_after_socializing']].fillna('UnKnow')


# Boxplots for numeric distributions:

plt.figure(figsize=(14, 6))
sns.boxplot(data=all_df.select_dtypes(include='number'), palette="pastel")
plt.title("ğŸ“¦ Boxplot of All Numeric Features", fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Histograms + KDE:

plt.figure(figsize=(10, 6))
sns.histplot(train['Time_spent_Alone'], bins=30, kde=True, color=custom_palette[0])
plt.title("ğŸ•’ Time Spent Alone Distribution", fontsize=16, weight='bold')
plt.xlabel("Time_spent_Alone")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# Pairplot among key features:

sns.pairplot(
    train[merge_cols + ['Personality']], 
    hue='Personality', 
    palette={"Introvert": custom_palette[1], "Extrovert": custom_palette[2]},
    diag_kind='kde'
)
plt.suptitle("ğŸ”„ Pairwise Feature Relationships by Personality", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Countplots:

plt.figure(figsize=(8, 5))
sns.countplot(data=all_df, x='Stage_fear', hue='match_p', palette=custom_palette)
plt.title("ğŸ�¤ Stage Fear Levels by Personality Type", fontsize=16, weight='bold')
plt.xlabel("Stage Fear Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


all_ohe = pd.get_dummies(
    all_df,
    columns=['Stage_fear','Drained_after_socializing','match_p'],
    prefix=['Stage','Drained','match']
)
X = all_ohe.iloc[:len(train)]
X_test = all_ohe.iloc[len(train):]


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

# Compute class imbalance weight
pos = y.sum()
neg = len(y) - pos
spw = neg / pos

xgb = XGBClassifier(max_depth=4, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, random_state=42)
cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, class_weights=[spw,1], random_seed=42, verbose=0)
lgbm = LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=300, subsample=0.8, colsample_bytree=0.8, class_weight={0:spw,1:1}, random_state=42)

ensemble = VotingClassifier([('xgb', xgb), ('cat', cat), ('lgbm', lgbm)], voting='soft')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ensemble.fit(X_tr, y_tr)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ğŸŒˆ Set consistent style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
custom_palette = sns.color_palette("Set2")

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "axes.edgecolor": "black",
    "grid.color": ".8"
})

# ğŸ§ª Threshold tuning (optional but recommended)
val_proba = ensemble.predict_proba(X_val)[:, 1]
best_threshold, best_acc = 0.5, 0

for t in np.arange(0.4, 0.6, 0.01):
    preds = (val_proba >= t).astype(int)
    acc = (preds == y_val).mean()
    if acc > best_acc:
        best_threshold, best_acc = t, acc

print(f"âœ… Best Threshold: {best_threshold:.2f} | Validation Accuracy: {best_acc:.4f}")

# ğŸ”� Predict with best threshold
val_preds = (val_proba >= best_threshold).astype(int)

# ğŸ“‹ Classification Report
print("\nğŸ“‹ Classification Report (Validation):")
print(classification_report(y_val, val_preds, target_names=["Introvert", "Extrovert"]))

# ğŸ”² Confusion Matrix
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=["Introvert", "Extrovert"],
            yticklabels=["Introvert", "Extrovert"])
plt.title("ğŸ§® Confusion Matrix", fontsize=16, weight='bold')
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# ğŸ“ˆ AUC-ROC Curve
roc_auc = roc_auc_score(y_val, val_proba)
fpr, tpr, _ = roc_curve(y_val, val_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color=custom_palette[3], linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ğŸš€ ROC Curve (Validation Set)", fontsize=16, weight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


val_proba = ensemble.predict_proba(X_val)[:,1]
best_threshold, best_acc = 0.5, 0

# Threshold tuning loop
for t in np.arange(0.4, 0.6, 0.01):
    acc = ((val_proba >= t).astype(int) == y_val).mean()
    if acc > best_acc:
        best_threshold, best_acc = t, acc

# Predict on test set
test_proba = ensemble.predict_proba(X_test)[:,1]
test_preds = (test_proba >= best_threshold).astype(int)

# Create submission file
submission = pd.DataFrame({'id': test_ID, 'Personality': test_preds})
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)

# Show result
print(f"âœ… Submitted successfully with threshold={best_threshold:.2f}, val_acc={best_acc:.4f}")
print("\nğŸ”� Sample Predictions:")
print(submission.head())

