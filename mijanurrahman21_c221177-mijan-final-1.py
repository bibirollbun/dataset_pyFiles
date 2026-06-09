import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier  # Changed model

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Data preprocessing for training data
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Define features and target variable
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM Classifier (better than Random Forest)
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Preprocess the test dataset
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Select features for prediction
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Make predictions
solution['satisfaction'] = model.predict(X_test)

# Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Optionally display the first few results
print(solution[['satisfaction']].head())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")


# Step 3: Set style for plots
sns.set(style="whitegrid")

# Step 4: Analysis

# 1. Which customer type takes the most flights?
customer_counts = train_data['Customer Type'].value_counts()

# 2. Comfortable flight time (average convenience score)
avg_time_convenience = train_data.groupby('Type of Travel')['Departure/Arrival time convenient'].mean()

# 3. Age vs. Flight Distance correlation
age_distance_corr = train_data[['Age', 'Flight Distance']].corr().iloc[0, 1]

# 4. Food and drink consumption by gender
food_by_gender = train_data.groupby('Gender')['Food and drink'].mean()

# 5. Most departure delays
top_delayed = train_data.sort_values(by='Departure Delay in Minutes', ascending=False).head(5)

# 6. Online bookings (score > 3 implies booked online successfully)
online_bookings_count = (train_data['Ease of Online booking'] > 3).sum()

# 7. Arrival delay count (> 0 mins)
arrival_delayed_count = (train_data['Arrival Delay in Minutes'] > 0).sum()

# Step 5: Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Customer Type Count
sns.barplot(x=customer_counts.index, y=customer_counts.values, ax=axs[0, 0])
axs[0, 0].set_title("Flight Count by Customer Type")
axs[0, 0].set_ylabel("Number of Flights")

# Plot 2: Avg. Time Convenience
sns.barplot(x=avg_time_convenience.index, y=avg_time_convenience.values, ax=axs[0, 1])
axs[0, 1].set_title("Average Time Convenience by Travel Type")
axs[0, 1].set_ylabel("Average Score")

# Plot 3: Age vs Flight Distance Scatter
sns.scatterplot(data=train_data, x='Age', y='Flight Distance', ax=axs[1, 0], alpha=0.5)
axs[1, 0].set_title(f"Age vs. Flight Distance (Corr: {age_distance_corr:.2f})")

# Plot 4: Food and Drink by Gender
sns.barplot(x=food_by_gender.index, y=food_by_gender.values, ax=axs[1, 1])
axs[1, 1].set_title("Average Food & Drink Score by Gender")
axs[1, 1].set_ylabel("Score")

plt.tight_layout()
plt.show()

# Step 6: Print Results
print("\n===== SUMMARY REPORT =====")
print(f"1. Flights by Customer Type:\n{customer_counts}")
print("\n2. Avg. Time Convenience by Travel Type:\n", avg_time_convenience)
print(f"\n3. Correlation between Age and Flight Distance: {age_distance_corr:.2f}")
print("\n4. Avg. Food & Drink Score by Gender:\n", food_by_gender)
print("\n5. Top 5 Departure Delays:\n", top_delayed[['id', 'Departure Delay in Minutes']])
print(f"\n6. Online Bookings (Ease of Online Booking > 3): {online_bookings_count}")
print(f"7. Arrival Delayed Passengers (Arrival Delay > 0 min): {arrival_delayed_count}")


