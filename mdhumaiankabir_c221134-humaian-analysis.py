#1. What are the most important features influencing satisfaction?
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load your training data
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Step 1: Create satisfaction column
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Step 2: Prepare features and labels
X = df.drop(columns=['Unnamed: 0', 'id', 'avg_satisfaction', 'satisfaction'])
y = df['satisfaction']

# Step 3: Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Step 4: Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Step 5: Plot Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance Score")
plt.title("Top Features Influencing Satisfaction (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Print top 10 features
print(importance_df.head(10))




#2: How does flight class (Eco, Business, etc.) affect satisfaction levels?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Recreate satisfaction target
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Plot average satisfaction by Class
sns.barplot(data=df, x='Class', y='satisfaction')
plt.title('Satisfaction Rate by Flight Class')
plt.ylabel('Average Satisfaction (1 = Satisfied)')
plt.xlabel('Flight Class')
plt.show()

# Print average satisfaction values
print(df.groupby('Class')['satisfaction'].mean().sort_values(ascending=False))



#3: Do loyal customers report higher satisfaction than disloyal ones?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Recreate satisfaction target
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Visualization
sns.barplot(data=df, x='Customer Type', y='satisfaction')
plt.title('Satisfaction by Customer Type')
plt.ylabel('Average Satisfaction (1 = Satisfied)')
plt.xlabel('Customer Type')
plt.show()

# Print average satisfaction values
print(df.groupby('Customer Type')['satisfaction'].mean().sort_values(ascending=False))



#4: Does gender play any role in satisfaction?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Recreate satisfaction column
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Barplot: Gender vs Satisfaction
sns.barplot(data=df, x='Gender', y='satisfaction')
plt.title('Satisfaction by Gender')
plt.ylabel('Average Satisfaction (1 = Satisfied)')
plt.xlabel('Gender')
plt.show()

# Print average satisfaction by gender
print(df.groupby('Gender')['satisfaction'].mean().sort_values(ascending=False))



#5: What is the correlation between flight distance and satisfaction?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Recreate satisfaction column
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Scatter plot: Flight Distance vs Satisfaction
sns.boxplot(x='satisfaction', y='Flight Distance', data=df)
plt.title('Flight Distance Distribution by Satisfaction')
plt.xlabel('Satisfaction (0 = Dissatisfied, 1 = Satisfied)')
plt.ylabel('Flight Distance (km)')
plt.show()

# Calculate average distance for each group
avg_distances = df.groupby('satisfaction')['Flight Distance'].mean()
print("\nAverage Flight Distance by Satisfaction:\n")
print(avg_distances)



#6: How does departure delay affect satisfaction?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Recreate satisfaction column
service_features = [
    'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Online boarding',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
df['avg_satisfaction'] = df[service_features].mean(axis=1)
df['satisfaction'] = (df['avg_satisfaction'] >= 3.5).astype(int)

# Boxplot: Departure Delay vs Satisfaction
sns.boxplot(x='satisfaction', y='Departure Delay in Minutes', data=df)
plt.title('Departure Delay Distribution by Satisfaction')
plt.xlabel('Satisfaction (0 = Dissatisfied, 1 = Satisfied)')
plt.ylabel('Departure Delay (Minutes)')
plt.show()

# Summary statistics
delay_summary = df.groupby('satisfaction')['Departure Delay in Minutes'].describe()
print("Departure Delay Summary by Satisfaction:\n")
print(delay_summary)



#7. Does inflight entertainment correlate with satisfaction?
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load training data
#train_data = pd.read_csv("train_dataset.csv")

plt.figure(figsize=(8, 6))
sns.scatterplot(data= df, x='Inflight entertainment', y='satisfaction')
plt.title('Scatter Plot of Inflight Entertainment vs. Satisfaction')
plt.xlabel('Inflight Entertainment (Encoded)')
plt.ylabel('Satisfaction (Encoded)')
plt.show()


#8. Are older passengers more or less satisfied?
# Calculate the correlation between Age and satisfaction
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Create a LabelEncoder instance
le = LabelEncoder()

# Encode the 'satisfaction' column
df['satisfaction_encoded'] = le.fit_transform(df['satisfaction'])


correlation_age_satisfaction = df['Age'].corr(df['satisfaction_encoded'])
print(f"Correlation between Age and satisfaction: {correlation_age_satisfaction:.2f}")

# Visualize the relationship between Age and satisfaction
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='satisfaction_encoded')
plt.title('Scatter Plot of Age vs. Satisfaction')
plt.xlabel('Age')
plt.ylabel('Satisfaction (Encoded)')
plt.show()

