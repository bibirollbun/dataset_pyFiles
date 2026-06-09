# ðŸ“¦ Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ðŸ“‚ Load your dataset (adjust path as needed)
df = pd.read_csv("/kaggle/input/machine-learning/train_dataset.csv")

# ðŸ§¹ Basic Cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # Normalize column names
df['satisfaction'] = df['satisfaction'].str.strip().str.lower()        # Normalize values

# ðŸŸ¢ Create Binary Satisfaction Column
df['satisfaction_binary'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:38:38.429324Z","iopub.execute_input":"2025-08-07T18:38:38.430723Z","iopub.status.idle":"2025-08-07T18:38:39.087027Z","shell.execute_reply.started":"2025-08-07T18:38:38.430683Z","shell.execute_reply":"2025-08-07T18:38:39.085072Z"},"jupyter":{"outputs_hidden":false}}



# ðŸŽ¯ 1. Overall Satisfaction Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='satisfaction_binary', data=df, palette='viridis')
plt.xticks([0, 1], ['Neutral/Dissatisfied', 'Satisfied'])
plt.title("Overall Satisfaction Distribution")
plt.xlabel("Passenger Satisfaction")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ðŸ“Š 2. Satisfaction by Class
plt.figure(figsize=(6,4))
sns.barplot(x='class', y='satisfaction_binary', data=df, errorbar=None
, palette='Set2')
plt.title("Satisfaction Rate by Travel Class")
plt.ylabel("Satisfaction Rate")
plt.xlabel("Class")
plt.tight_layout()
plt.show()

# ðŸ“Š 3. Satisfaction by Type of Travel
plt.figure(figsize=(6,4))
sns.barplot(x='type_of_travel', y='satisfaction_binary', data=df, errorbar=None
, palette='Set3')
plt.title("Satisfaction Rate by Travel Type")
plt.ylabel("Satisfaction Rate")
plt.xlabel("Type of Travel")
plt.tight_layout()
plt.show()


# ðŸ“Š 4. Age Grouping (Optional - Create buckets)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior'])

plt.figure(figsize=(6,4))
sns.barplot(x='age_group', y='satisfaction_binary', data=df, errorbar=None, palette='coolwarm')
plt.title("Satisfaction by Age Group")
plt.ylabel("Satisfaction Rate")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()

# ðŸ“ˆ 5. Correlation Heatmap for Service Scores
service_cols = [
    'inflight_wifi_service', 'departure/arrival_time_convenient', 'ease_of_online_booking',
    'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
    'inflight_entertainment', 'on-board_service', 'leg_room_service',
    'baggage_handling', 'checkin_service', 'inflight_service', 'cleanliness',
    'satisfaction_binary'
]

plt.figure(figsize=(12,10))
corr = df[service_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Services and Satisfaction")
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:39:23.999900Z","iopub.execute_input":"2025-08-07T18:39:24.000324Z","iopub.status.idle":"2025-08-07T18:39:24.286794Z","shell.execute_reply.started":"2025-08-07T18:39:24.000293Z","shell.execute_reply":"2025-08-07T18:39:24.285496Z"},"jupyter":{"outputs_hidden":false}}
# ðŸ“‰ 6. Satisfaction vs. Delay
plt.figure(figsize=(6,4))
sns.boxplot(x='satisfaction_binary', y='arrival_delay_in_minutes', data=df)
plt.xticks([0, 1], ['Neutral/Dissatisfied', 'Satisfied'])
plt.title("Arrival Delay vs Satisfaction")
plt.tight_layout()
plt.show()


# ðŸ“‹ 7. Summary Statistics
print("\nðŸ§® Satisfaction Breakdown:")
print(df['satisfaction'].value_counts(normalize=True) * 100)

print("\nðŸ“Š Average Scores for Satisfied vs Dissatisfied:")
print(df.groupby('satisfaction')[[
    'inflight_wifi_service', 'seat_comfort', 'inflight_entertainment',
    'cleanliness', 'on-board_service'
]].mean().round(2))

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:39:46.036380Z","iopub.execute_input":"2025-08-07T18:39:46.036882Z","iopub.status.idle":"2025-08-07T18:39:46.234297Z","shell.execute_reply.started":"2025-08-07T18:39:46.036856Z","shell.execute_reply":"2025-08-07T18:39:46.233213Z"},"jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(6,4))
sns.barplot(x='gender', y='satisfaction_binary', data=df, errorbar=None, palette='pastel')
plt.title("Satisfaction Rate by Gender")
plt.ylabel("Satisfaction Rate")
plt.xlabel("Gender")
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:39:54.966098Z","iopub.execute_input":"2025-08-07T18:39:54.966524Z","iopub.status.idle":"2025-08-07T18:39:55.492422Z","shell.execute_reply.started":"2025-08-07T18:39:54.966495Z","shell.execute_reply":"2025-08-07T18:39:55.491181Z"},"jupyter":{"outputs_hidden":false}}
corr = df.corr(numeric_only=True)
satisfaction_corr = corr['satisfaction_binary'].drop('satisfaction_binary').sort_values(ascending=False)

plt.figure(figsize=(8,6))
satisfaction_corr.plot(kind='barh', color='teal')
plt.title("Correlation of Features with Satisfaction")
plt.xlabel("Correlation with Satisfaction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:40:03.539222Z","iopub.execute_input":"2025-08-07T18:40:03.539768Z","iopub.status.idle":"2025-08-07T18:40:04.358945Z","shell.execute_reply.started":"2025-08-07T18:40:03.539736Z","shell.execute_reply":"2025-08-07T18:40:04.357191Z"},"jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='arrival_delay_in_minutes', hue='satisfaction', bins=30, kde=True, element='step')
plt.title("Arrival Delays vs. Passenger Satisfaction")
plt.xlabel("Arrival Delay (Minutes)")
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:40:14.383234Z","iopub.execute_input":"2025-08-07T18:40:14.383642Z","iopub.status.idle":"2025-08-07T18:40:14.576306Z","shell.execute_reply.started":"2025-08-07T18:40:14.383615Z","shell.execute_reply":"2025-08-07T18:40:14.574934Z"},"jupyter":{"outputs_hidden":false}}
plt.figure(figsize=(6,4))
sns.barplot(x='customer_type', y='satisfaction_binary', data=df, errorbar=None, palette='muted')
plt.title("Satisfaction Rate by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Satisfaction Rate")
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:40:24.277758Z","iopub.execute_input":"2025-08-07T18:40:24.278114Z","iopub.status.idle":"2025-08-07T18:40:24.752051Z","shell.execute_reply.started":"2025-08-07T18:40:24.278091Z","shell.execute_reply":"2025-08-07T18:40:24.750808Z"},"jupyter":{"outputs_hidden":false}}
pivot = df.pivot_table(index='class', columns='type_of_travel', values='satisfaction_binary', aggfunc='mean')

plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Satisfaction Rate by Class and Travel Type")
plt.tight_layout()
plt.show()



# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:55:35.423298Z","iopub.execute_input":"2025-08-07T18:55:35.423691Z","iopub.status.idle":"2025-08-07T18:55:36.548046Z","shell.execute_reply.started":"2025-08-07T18:55:35.423667Z","shell.execute_reply":"2025-08-07T18:55:36.546910Z"},"jupyter":{"outputs_hidden":false}}
# Import necessary libs for model loading or training
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load train and test datasets again (or pass from previous step if possible)
train_df = pd.read_csv("/kaggle/input/machine-learning/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning/test_dataset_exam.csv")

# Preprocessing steps same as in prediction notebook
train_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
test_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# Label encoding categorical features
combined = pd.concat([train_df.drop(columns=['satisfaction']), test_df], axis=0)
categorical_cols = combined.select_dtypes(include='object').columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le

train_df[categorical_cols] = combined.iloc[:len(train_df)][categorical_cols]
test_df[categorical_cols] = combined.iloc[len(train_df):][categorical_cols]

# Encode target variable
le_target = LabelEncoder()
train_df['satisfaction'] = le_target.fit_transform(train_df['satisfaction'])

# Train model on full train data (skip train/test split for quick submission)
X = train_df.drop(columns=['satisfaction', 'id'], errors='ignore')
y = train_df['satisfaction']

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X, y)

# Predict on test data
X_test = test_df.drop(columns=['id'], errors='ignore')
test_preds = model.predict(X_test)

# Prepare submission dataframe
submission_df = pd.DataFrame({
    'ID': test_df['id'],
    'satisfaction': le_target.inverse_transform(test_preds)
})

submission_df.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")
print(submission_df.head())


# %% [code]


