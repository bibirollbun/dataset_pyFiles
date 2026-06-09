# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modeling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



df=pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
df


print('First 5 rows:')
df.head()


print('Shape of the dataset:')
df.shape


df.info()


print('Summary statistics:')
df.describe()


print('Missing values in each column:')
print(df.isnull().sum())



# Count of unique values per column
print("Unique values per column:")
print(df.nunique())




# Count duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")



# Check data types of each column
print(df.dtypes)

#OR show only columns with 'object' (i.e., string/categorical) data type
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical columns in the dataset:")
print(categorical_columns)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset load 
df = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')

# Numeric columns (based on your dataset)
numeric_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                'no_of_week_nights', 'lead_time', 'arrival_year',
                'arrival_month', 'arrival_date', 'no_of_previous_cancellations',
                'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                'no_of_special_requests']

# Plotting histogram + boxplot for each feature
fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=2, figsize=(14, 4 * len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    # Histogram
    sns.histplot(df[col], kde=True, ax=axes[i, 0], bins=30, color='skyblue')
    axes[i, 0].set_title(f'Histogram of {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')

    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1], color='lightcoral')
    axes[i, 1].set_title(f'Boxplot of {col}')
    axes[i, 1].set_xlabel(col)

plt.tight_layout()
plt.show()





# Categorical columns (numerical IDs used to represent categories)
categorical_cols = ['type_of_meal_plan', 'required_car_parking_space',
                    'room_type_reserved', 'market_segment_type',
                    'repeated_guest', 'booking_status']

# Countplots
n = len(categorical_cols)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 5 * n))

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i], palette='pastel')
    axes[i].set_title(f'Count Plot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].bar_label(axes[i].containers[0])  # To show value labels on bars

plt.tight_layout()
plt.show()



| Column Name                  | Meaning / Description                                            |
| ---------------------------- | ---------------------------------------------------------------- |
| `type_of_meal_plan`          | 0,1,2,3 â†’ Meal Plan A, B, C, or Not Selected                     |
| `required_car_parking_space` | 1 â†’ Car parking is required, 0 â†’ Car parking not required        |
| `room_type_reserved`         | Values from 0â€“6 indicating different room categories             |
| `market_segment_type`        | 0,1,2... â†’ Booking channel like Online, Offline, Corporate, etc. |
| `repeated_guest`             | 1 â†’ Returning customer, 0 â†’ New customer                         |
| `booking_status`             | 1 â†’ Booking confirmed, 0 â†’ Booking cancelled                     |



import seaborn as sns
import matplotlib.pyplot as plt

# Boxplots
fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(10, 5 * len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='booking_status', y=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Booking Status')
    axes[i].set_xlabel('Booking Status (0 = Cancelled, 1 = Confirmed)')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()




# Categorical columns
categorical_cols = ['type_of_meal_plan',
                    'required_car_parking_space',
                    'room_type_reserved',
                    'market_segment_type',
                    'repeated_guest']

# Plot setup
fig, axes = plt.subplots(nrows=len(categorical_cols), ncols=1, figsize=(10, 5 * len(categorical_cols)))

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='booking_status', data=df, ax=axes[i], palette='pastel')
    axes[i].set_title(f'{col} vs Booking Status')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Booking Status', labels=['Cancelled (0)', 'Confirmed (1)'])
    axes[i].bar_label(axes[i].containers[0])
    axes[i].bar_label(axes[i].containers[1])

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Only numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()



selected_cols = ['lead_time', 'avg_price_per_room', 'no_of_special_requests', 'booking_status']
sns.pairplot(df[selected_cols], hue='booking_status', palette='husl')
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()



import pandas as pd

# Group by meal plan and room type and get booking status rate
group_data = df.groupby(['type_of_meal_plan', 'room_type_reserved'])['booking_status'].mean().reset_index()

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=group_data, x='type_of_meal_plan', y='booking_status', hue='room_type_reserved')
plt.title("Booking Confirmation Rate by Meal Plan and Room Type")
plt.ylabel("Average Booking Status (1=Confirmed)")
plt.xlabel("Meal Plan")
plt.legend(title="Room Type")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='room_type_reserved', y='lead_time', hue='booking_status')
plt.title("Lead Time by Room Type and Booking Status")
plt.xlabel("Room Type")
plt.ylabel("Lead Time (days)")
plt.show()



# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Step 2: Read data
df = pd.read_csv("/kaggle/input/playground-series-s3e7/train.csv")

# Step 3: Split features and target
X = df.drop(['booking_status', 'id'], axis=1)
y = df['booking_status']

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("ðŸ”¹ Logistic Regression")
print(confusion_matrix(y_test, y_pred_log))
# print('classification_report')
print(classification_report(y_test, y_pred_log))




tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("ðŸ”¹ Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))



rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("ðŸ”¹ Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("ðŸ”¹ XGBoost Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))





