import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Preprocessing: Handle categorical features (Sex)
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])  # Use the same encoder fitted on the training data


# Feature Scaling
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()

train[numerical_features] = scaler.fit_transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features]) # Use the same scaler fitted on training data

# Separate features (X) and target (y) from the training data
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']



# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Modeling (Linear Regression) 
model = LinearRegression()
model.fit(X_train, y_train)


# Prediction on validation set
y_pred_val = model.predict(X_val)
y_pred_val[y_pred_val < 0] = 0  # Ensure no negative predictions
y_pred_train = model.predict(X_train)
y_pred_train[y_pred_train < 0] = 0


# RMSLE Function
def rmsle(y_true, y_pred):
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return rmsle

# Calculate RMSLE on validation set
rmsle_val = rmsle(y_val, y_pred_val)
rmsle_train = rmsle(y_train, y_pred_train)

print(f"RMSLE on Validation Set: {rmsle_val}")
print(f"RMSLE on Train Set: {rmsle_train}")




# Prediction on the test set
X_test = test.drop('id', axis=1)
predictions = model.predict(X_test)
predictions[predictions < 0] = 0 # Ensure no negative predictions


# Create the submission file
submission['Calories'] = predictions
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


submission.head()


# --- EDA and Visualization ---

# 1. Calories Distribution
plt.figure(figsize=(12, 6))
sns.histplot(train['Calories'], kde=True, color='skyblue')
plt.title('Distribution of Calories', fontsize=16, fontweight='bold',color='midnightblue')
plt.xlabel('Calories', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_facecolor('#f0f0f0')  # Light grey background
plt.show()
print("Observation: The distribution shows the frequency of different calorie values, providing an overview of calorie consumption patterns.")

# 2. Joint Distribution Plot (Calories vs. Duration)
plt.figure(figsize=(10, 8))
sns.jointplot(x='Duration', y='Calories', data=train, kind='kde', fill=True, cmap='viridis')
plt.suptitle('Joint Distribution of Calories and Duration', fontsize=16, fontweight='bold',color='midnightblue', y=1.02)
plt.show()
print("Observation:  Visualizes the joint distribution of Calories and Duration using kernel density estimation.  The contour plot shows areas of higher density, and the marginal distributions are displayed on the sides.")


# 3. Calories Boxplot by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Calories', data=train, palette='pastel')
plt.title('Calories Burned by Gender', fontsize=16, fontweight='bold',color='midnightblue')
plt.xlabel('Gender', fontsize=14,color='forestgreen')
plt.ylabel('Calories', fontsize=14,color='forestgreen')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_facecolor('#f5f5dc')  # Beige background
plt.show()
print("Observation: The boxplot compares calorie consumption between genders, displaying median, quartiles, and outliers.")

# 4. Correlation Heatmap
correlation_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold',color='midnightblue')
plt.xticks(fontsize=12,color='forestgreen')
plt.yticks(fontsize=12,color='forestgreen')
plt.show()
print("Observation: The heatmap visualizes the correlation coefficients between variables, with color intensity indicating the strength and direction of relationships.")

# 5. Pairplot (Subset of Numerical Features)
subset_features = ['Age', 'Height', 'Weight', 'Calories']
sns.pairplot(train[subset_features], diag_kind='kde', corner=True)
plt.suptitle('Pairwise Relationships between Age, Height, Weight, and Calories', fontsize=16, fontweight='bold',color='midnightblue' ,y=1.02)
plt.show()
print("Observation:  Displays pairwise relationships between selected features. The diagonal shows the kernel density estimation, and the off-diagonal plots show scatter plots of each pair of variables.")


# 6. Violin Plot (Calories by Sex)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sex', y='Calories', data=train, palette='muted')
plt.title('Calories Distribution by Gender (Violin Plot)', fontsize=16, fontweight='bold',color='midnightblue')
plt.xlabel('Gender', fontsize=14,color='forestgreen')
plt.ylabel('Calories', fontsize=14,color='forestgreen')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], fontsize=12)  
plt.yticks(fontsize=12)
plt.gca().set_facecolor('#f8f8ff')
plt.show()
print("Observation:  Displays the distribution of Calories for each gender, combining the features of a box plot and a kernel density plot.  It provides a more detailed view of the distribution shape.")

# 7.  Stacked Bar Chart (Calories Range by Age Group)
age_bins = pd.cut(train['Age'], bins=[0, 25, 40, 60, 100], labels=['0-25', '26-40', '41-60', '60+'])
calories_bins = pd.cut(train['Calories'], bins=[0, 50, 100, 150, 200, 350], labels=['0-50', '51-100', '101-150', '151-200', '201+'])
ct = pd.crosstab(age_bins, calories_bins)
ct.plot.bar(stacked=True, figsize=(12, 7), colormap='viridis')
plt.title('Calories Range Distribution by Age Group (Stacked Bar Chart)', fontsize=16, fontweight='bold',color='red')
plt.xlabel('Age Group', fontsize=14,color='forestgreen')
plt.ylabel('Number of Individuals', fontsize=14,color='forestgreen')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_facecolor('#faebd7')
plt.legend(title='Calories Range', fontsize=10)
plt.show()
print("Observation:  Shows the distribution of calorie ranges within different age groups, revealing how calorie consumption varies with age.")

# 8.  Ridgeline Plot (Distribution of Calories by Age) 
try:
    import joypy
    plt.figure(figsize=(12, 8))
    joypy.joyplot(train[['Age', 'Calories']], by="Age", ylim='own', figsize=(12, 8), colormap=plt.cm.autumn_r)
    plt.title('Distribution of Calories by Age (Ridgeline Plot)', fontsize=16, fontweight='bold',color='red')
    plt.xlabel('Calories', fontsize=14,color='forestgreen')
    plt.ylabel('Age', fontsize=14,color='forestgreen')
    plt.show()
    print("Observation:  Displays the distribution of Calories for each age group, allowing for a visual comparison of the distributions across different age categories.  Install joypy:  `pip install joypy`")
except ImportError:
    print("joypy is not installed. Install it using `pip install joypy` to use the Ridgeline Plot.")


# 9. Scatter Plot with Marginal Histograms
sns.displot(train, x="Duration", y="Calories", kind="kde", rug=True)
plt.title('Scatter Plot with Marginal Histograms', fontsize=16, fontweight='bold',color='green')
plt.show()
print("Observation:  A scatter plot of Duration vs. Calories with histograms on the margins, showing the distribution of each variable separately.")


# 10.  Andrews Curves 
from pandas.plotting import andrews_curves
plt.figure(figsize=(12, 8))
andrews_curves(train.sample(200), 'Sex', colormap='rainbow')
plt.title('Andrews Curves Plot (Sample)', fontsize=16, fontweight='bold',color='forestgreen')
plt.show()
print("Observation:  Represents each observation as a curve, allowing you to visualize the relationships between variables. The curves are based on Fourier series and can help identify clusters and outliers.")




