import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


# ## 1ï¸�âƒ£ ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
print("ğŸ“¥ Loading datasets...")
train_df = pd.read_csv("../input/birdclef-2025/train.csv")
taxonomy_df = pd.read_csv("../input/birdclef-2025/taxonomy.csv")
sample_submission_df = pd.read_csv("../input/birdclef-2025/sample_submission.csv")


# ## 2ï¸�âƒ£ Ø§Ø³ØªÙƒØ´Ø§Ù� Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
print("\nğŸ”� First few rows of train data:")
print(train_df.head())
print("\nğŸ“Š Train Data Info:")
train_df.info()
print("\nğŸ“Š Taxonomy Data Info:")
taxonomy_df.info()




train_df = train_df.assign(
    latitude=train_df["latitude"].fillna(train_df["latitude"].mean()),
    longitude=train_df["longitude"].fillna(train_df["longitude"].mean())
)
print(train_df)


# Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø°Ø§Øª Ø§Ù„Ø£Ù‡Ù…ÙŠØ©
train_cleaned = train_df[["primary_label", "latitude", "longitude"]]

# ## 4ï¸�âƒ£ ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
print("\nğŸ“Œ Splitting the data...")
X = train_cleaned[["latitude", "longitude"]]
y = train_cleaned["primary_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# âœ… ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ø£Ø³Ø§Ø³ÙŠ
print("\nğŸ¤– Training RandomForest Model...")
model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)


# âœ… Ø§Ø®ØªÙŠØ§Ø± Ø£ÙƒØ«Ø± 100 ØªØµÙ†ÙŠÙ� Ø´ÙŠÙˆØ¹Ù‹Ø§
print("\nğŸ”� Selecting top 100 most common labels...")
top_labels = train_cleaned["primary_label"].value_counts().index[:100]
filtered_df = train_cleaned[train_cleaned["primary_label"].isin(top_labels)]

X_filtered = filtered_df[["latitude", "longitude"]]
y_filtered = filtered_df["primary_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# âœ… Ø¥Ø¹Ø§Ø¯Ø© ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
print("\nâš¡ Retraining the model with top 100 labels...")
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)

# âœ… ØªØ­ÙˆÙŠÙ„ y_test Ø¥Ù„Ù‰ One-Hot Encoding
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# âœ… Ø¶Ù…Ø§Ù† ØªØ·Ø§Ø¨Ù‚ Ø¹Ø¯Ø¯ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©
min_cols = min(y_pred_proba.shape[1], y_test_bin.shape[1])
y_test_bin, y_pred_proba = y_test_bin[:, :min_cols], y_pred_proba[:, :min_cols]

# âœ… Ø­Ø³Ø§Ø¨ ROC-AUC Ù�Ù‚Ø· Ø¹Ù†Ø¯ ÙˆØ¬ÙˆØ¯ Ø£ÙƒØ«Ø± Ù…Ù† Ù�Ø¦Ø©
if len(np.unique(y_test)) > 1:
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average="macro", multi_class="ovr")
    print(f"âœ… ROC-AUC Score: {roc_auc:.7f}")
else:
    print("âš ï¸� Skipping ROC-AUC calculation: Only one class present in y_test.")


# ## 7ï¸�âƒ£ ØªØ¬Ù‡ÙŠØ² Ù…Ù„Ù� Ø§Ù„Ø¥Ø±Ø³Ø§Ù„
print("\nğŸ“¤ Preparing submission file...")
submission = sample_submission_df.copy()
required_labels = submission.columns[1:]  # Ø§Ø³ØªØ¨Ø¹Ø§Ø¯ 'row_id'

y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=lb.classes_)

# Ø§Ù„ØªØ£ÙƒØ¯ Ù…Ù† Ø£Ù† Ø¹Ø¯Ø¯ Ø§Ù„Ø¹ÙŠÙ†Ø§Øª Ù�ÙŠ y_pred_proba_df ÙŠØ·Ø§Ø¨Ù‚ submission
if y_pred_proba_df.shape[0] > submission.shape[0]:
    y_pred_proba_df = y_pred_proba_df.iloc[:submission.shape[0], :]
elif y_pred_proba_df.shape[0] < submission.shape[0]:
    missing_rows = submission.shape[0] - y_pred_proba_df.shape[0]
    padding = np.zeros((missing_rows, y_pred_proba_df.shape[1]))
    y_pred_proba_df = pd.concat([y_pred_proba_df, pd.DataFrame(padding, columns=y_pred_proba_df.columns)], ignore_index=True)

# Ø§Ù„ØªØ£ÙƒØ¯ Ù…Ù† Ø£Ù† Ø¬Ù…ÙŠØ¹ Ø§Ù„Ù�Ø¦Ø§Øª Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø© Ù…ÙˆØ¬ÙˆØ¯Ø©
for label in required_labels:
    if label not in y_pred_proba_df:
        y_pred_proba_df[label] = 0  # ØªØ¹ÙŠÙŠÙ† Ø§Ù„Ù‚ÙŠÙ… ØºÙŠØ± Ø§Ù„Ù…ØªÙˆÙ‚Ø¹Ø© Ø¥Ù„Ù‰ 0

# ØªØ±ØªÙŠØ¨ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© ÙƒÙ…Ø§ Ù‡Ùˆ Ù…Ø·Ù„ÙˆØ¨ Ù�ÙŠ Ù…Ù„Ù� Ø§Ù„Ø¥Ø±Ø³Ø§Ù„
y_pred_proba_df = y_pred_proba_df[required_labels]

# ØªØ¹ÙŠÙŠÙ† Ø§Ù„Ù‚ÙŠÙ… Ù�ÙŠ Ù…Ù„Ù� Ø§Ù„Ø¥Ø±Ø³Ø§Ù„
submission.iloc[:, 1:] = y_pred_proba_df.values

# Ø¶Ø¨Ø· Ø§Ù„Ù‚ÙŠÙ… Ø¯Ø§Ø®Ù„ Ø§Ù„Ù†Ø·Ø§Ù‚ [0, 1]
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0, 1)

# Ø­Ù�Ø¸ Ù…Ù„Ù� Ø§Ù„Ø¥Ø±Ø³Ø§Ù„
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved successfully!")





