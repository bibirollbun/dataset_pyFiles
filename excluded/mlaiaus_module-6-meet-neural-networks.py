import pandas as pd
import numpy as np
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to our message-level data
DATA_PATH = "/kaggle/input/needle-in-the-hashtag/example_train.csv"

# Load the message-level data
df = pd.read_csv(DATA_PATH)

print("Message-level data loaded âœ…")
print(df.head())

# If user_features already exists from a previous module, keep it.
# Otherwise, we quickly rebuild a simple user_features table here.
if 'user_features' in globals():
    print("\nFound existing user_features from earlier modules. Using that.")
else:
    print("\nuser_features not found. Rebuilding user-level features from example_train.csv ...")
    
    # Make a working copy
    df = df.copy()
    
    # Make sure text is a string
    df['text'] = df['text'].fillna("").astype(str)
    
    # Helper: parse the category_labels string into a Python list
    def parse_labels(x):
        if isinstance(x, list):
            return x
        if not isinstance(x, str):
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    
    df['labels_list'] = df['category_labels'].apply(parse_labels)
    
    # Basic text length features
    df['num_words'] = df['text'].str.split().str.len()
    df['num_chars'] = df['text'].str.len()
    
    # For each message, decide if it's mainly recovery_support, benign, or risky
    def classify_message(labels):
        labels = set(labels)
        if "recovery_support" in labels:
            return "recovery_support"
        elif "benign" in labels:
            return "benign"
        else:
            return "risky"
    
    df['message_group'] = df['labels_list'].apply(classify_message)
    
    # Group by user_id to build user-level features
    user_grouped = df.groupby('user_id')
    
    # Aggregate basic stats
    user_features = user_grouped.agg(
        num_messages = ('text', 'size'),
        avg_words    = ('num_words', 'mean'),
        avg_chars    = ('num_chars', 'mean')
    )
    
    # Fractions of each message type per user
    def fraction_stats(group):
        total = len(group)
        frac_recovery = (group['message_group'] == 'recovery_support').mean()
        frac_benign   = (group['message_group'] == 'benign').mean()
        frac_risky    = (group['message_group'] == 'risky').mean()
        return pd.Series({
            'frac_recovery_support': frac_recovery,
            'frac_benign': frac_benign,
            'frac_risky': frac_risky
        })
    
    frac_df = user_grouped.apply(fraction_stats)
    
    user_features = user_features.join(frac_df)
    
    # Assign each user to benign_user / recovery_user / risky_user based on their highest fraction
    def user_group_from_row(row):
        scores = [
            row['frac_benign'],
            row['frac_recovery_support'],
            row['frac_risky']
        ]
        idx = int(np.argmax(scores))
        # 0 â†’ benign_user, 1 â†’ recovery_user, 2 â†’ risky_user
        mapping = ['benign_user', 'recovery_user', 'risky_user']
        return mapping[idx]
    
    user_features['user_group'] = user_features.apply(user_group_from_row, axis=1)
    
    print("user_features built from message-level data âœ…")

print("\nuser_features preview:")
display(user_features.head())
print("\nNumber of users:", user_features.shape[0])



# Define a mapping from text labels to numeric IDs
group_to_id = {
    'benign_user': 0,
    'recovery_user': 1,
    'risky_user': 2
}

id_to_group = {v: k for k, v in group_to_id.items()}

# Create group_id column
user_features['group_id'] = user_features['user_group'].map(group_to_id)

print("Unique user_group values and their IDs:")
display(user_features[['user_group', 'group_id']].drop_duplicates())



# Choose the feature columns weâ€™ll feed into the neural net
feature_cols = [
    'num_messages',
    'avg_words',
    'avg_chars',
    'frac_recovery_support',
    'frac_benign',
    'frac_risky'
]

# Build X and y
X = user_features[feature_cols].values
y = user_features['group_id'].values

print("X shape (num_users, num_features):", X.shape)
print("y shape (num_users,):", y.shape)



# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape, "Labels:", y_train.shape)
print("Validation set shape:", X_val.shape, "Labels:", y_val.shape)



# Create a StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data only
scaler.fit(X_train)

# Transform both training and validation data
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Feature scaling complete âœ…")
print("First row of X_train_scaled:\n", X_train_scaled[0])



# Overall class counts
overall_counts = user_features['group_id'].value_counts().sort_index()

print("Overall class counts (by group_id):")
print(overall_counts)

# Helper: pretty-print counts with text labels
def pretty_counts(name, counts):
    print(f"\n{name} class counts:")
    total = counts.sum()
    for gid, cnt in counts.items():
        label = id_to_group.get(gid, f"id_{gid}")
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  {gid} ({label}): {cnt} users ({pct:.1f}%)")

# Training set class counts
train_counts = pd.Series(y_train).value_counts().sort_index()
pretty_counts("Training", train_counts)

# Validation set class counts
val_counts = pd.Series(y_val).value_counts().sort_index()
pretty_counts("Validation", val_counts)


