from IPython.core.display import display, HTML

display(HTML("""
<a id="2"></a>
<h1 style="
    background-color: #FFD700; 
    font-family: 'Times New Roman', serif; 
    color: #1A1A1A; 
    font-size: 150%; 
    text-align: center; 
    border-radius: 10px; 
    padding: 10px;">
  ML-Based Classification of Introverts and Extroverts
</h1>
"""))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



# Loading the dataset to a Pandas DataFrame
Train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


# Print First 5 rows of our DataFrame
Train_dataset.head()


# checking the number of rows and Columns in the data frame
Train_dataset.shape


# Drop ID (not useful for prediction)
Train_dataset = Train_dataset.drop(columns=["id"])


Train_dataset.isnull().sum()


# Define column groups
numeric_cols = ["Time_spent_Alone", "Social_event_attendance", 
                "Going_outside", "Friends_circle_size", "Post_frequency"]
categorical_cols = ["Stage_fear", "Drained_after_socializing"]

# Fill NaN in categorical columns with mode
for col in categorical_cols:
    mode_val = Train_dataset[col].mode()[0]
    Train_dataset[col] = Train_dataset[col].fillna(mode_val)

# Fill NaN in numeric columns with median
numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
# Replace inf values with NaN, then fill NaN with median
Train_dataset[numeric_cols] = Train_dataset[numeric_cols].replace([np.inf, -np.inf], np.nan)
Train_dataset[numeric_cols] = Train_dataset[numeric_cols].fillna(Train_dataset[numeric_cols].median())



#check missing value
Train_dataset.isnull().sum()


#Check descriptive statistics
Train_dataset.describe(include='all')



#Target distribution plot
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Personality', hue='Personality', data=Train_dataset, palette='husl')
ax.set_title('Target Distribution', pad=15, color='#EDEDED')
ax.set_xlabel('Personality', labelpad=10, color='#EDEDED')
ax.set_ylabel('Count', labelpad=10, color='#EDEDED')
if ax.get_legend():
    ax.legend_.remove()
plt.gca().set_facecolor('#1A1A1A')  # Axes background
plt.gcf().patch.set_facecolor('#1A1A1A')  # Figure background
ax.grid(color='#6D6D6D', linestyle='--')
ax.tick_params(colors='#EDEDED')
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()


# Histograms of numeric features

fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(9, 20))
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05)

for ax, col in zip(axes, numeric_cols):
    ax.hist(Train_dataset[col].dropna(), bins=30, color='#1B47A7', edgecolor='#EDEDED')
    ax.set_title(f'{col} Distribution', pad=15, color='#EDEDED')
    ax.set_xlabel(col, labelpad=10, color='#EDEDED')
    ax.set_ylabel('Count', labelpad=10, color='#EDEDED')
    ax.set_facecolor('#1A1A1A')
    ax.grid(color='#6D6D6D', linestyle='--')
    ax.tick_params(colors='#EDEDED')
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.patch.set_facecolor('#1A1A1A')
plt.suptitle('Feature Distributions', color='#EDEDED', fontsize=14)
plt.tight_layout()
plt.show()



# Violin plots comparing by Personality
groups = len(numeric_cols)
rows = (groups + 1)//2
fig, axes = plt.subplots(rows, 2, figsize=(12, 18))
axes = axes.flatten()
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05)
for ax, col in zip(axes, numeric_cols):
    sns.violinplot(x='Personality', y=col, hue='Personality', data=Train_dataset,
               palette=['#1B47A7','#1BA75F'], legend=False, ax=ax)
    ax.set_title(col + ' by Personality', pad=15, color='#EDEDED')
    ax.set_xlabel('Personality', labelpad=10, color='#EDEDED')
    ax.set_ylabel(col, labelpad=10, color='#EDEDED')
    ax.set_facecolor('#1A1A1A')
    ax.grid(color='#6D6D6D', linestyle='--')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#EDEDED')
# Remove unused axes
for unused in axes[len(numeric_cols):]:
    unused.remove()
fig.patch.set_facecolor('#1A1A1A')
plt.show()


# Correlation heatmap
corr_mat = Train_dataset[numeric_cols].corr()
plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Correlation between numeric features', pad=15, color='#EDEDED')
plt.xticks(color='#EDEDED')
plt.yticks(color='#EDEDED')
plt.gca().set_facecolor('#1A1A1A')
plt.gcf().patch.set_facecolor('#1A1A1A')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()



# Create boxplot
groups = len(numeric_cols)
rows = (groups + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(12, 18))
axes = axes.flatten()
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)

# Draw box plots
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(x='Personality', y=col, hue='Personality', data=Train_dataset,
                palette=['#1B47A7', '#1BA75F'], ax=ax, dodge=False)
    ax.set_title(f'{col} by Personality', pad=15)
    ax.set_xlabel('Personality', labelpad=10)
    ax.set_ylabel(col, labelpad=10)
    ax.grid(color='gray', linestyle='--')
    for spine in ax.spines.values():
        spine.set_visible(False)

# Remove any unused subplot axes
for unused in axes[len(numeric_cols):]:
    unused.remove()

# Set background color 
fig.patch.set_facecolor('#1A1A1A')
plt.show()


# Remove outliers using IQR for each numeric column
def remove_outliers_iqr(Train_dataset, columns):
    for col in columns:
        Q1 = Train_dataset[col].quantile(0.25)
        Q3 = Train_dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        Train_dataset = Train_dataset[(Train_dataset[col] >= lower_bound) & (Train_dataset[col] <= upper_bound)]
    return Train_dataset

# Remove outliers
Train_dataset_clean = remove_outliers_iqr(Train_dataset, numeric_cols)

# Boxplots of filtered data by Personality
plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)

sns.boxplot(data=Train_dataset_clean[numeric_cols], palette=['#1B47A7']*len(numeric_cols))
plt.title('Boxplots of Numeric Features (Outliers Removed)', pad=15, color='#EDEDED')
plt.xticks(rotation=45, color='#EDEDED')
plt.yticks(color='#EDEDED')
plt.gca().set_facecolor('#1A1A1A')
plt.gcf().patch.set_facecolor('#1A1A1A')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.grid(color='#6D6D6D', linestyle='--', axis='y')
plt.xlabel('Features', labelpad=10, color='#EDEDED')
plt.ylabel('Value', labelpad=10, color='#EDEDED')
plt.show()



# Features and target
X = Train_dataset.drop(columns=["Personality"])
y = Train_dataset["Personality"]


# Preprocess pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)


# Define model and pipeline
model = RandomForestClassifier(n_estimators=300, random_state=42)
clf = Pipeline(steps=[('prep', preprocessor), ('model', model)])

# Split and train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)


# Validation predictions
val_pred = clf.predict(X_val)
acc = accuracy_score(y_val, val_pred)
print('Validation Accuracy:', acc)


# Classification report
report = classification_report(y_val, val_pred)
print(report)


# Confusion matrix plot
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix', pad=15, color='#EDEDED')
plt.xlabel('Predicted', labelpad=10, color='#EDEDED')
plt.ylabel('Actual', labelpad=10, color='#EDEDED')
plt.xticks(color='#EDEDED')
plt.yticks(color='#EDEDED')
plt.gca().set_facecolor('#1A1A1A')
plt.gcf().patch.set_facecolor('#1A1A1A')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.show()


# Generate report file
with open('model_report.txt','w') as f:
    f.write('RandomForestClassifier Model Report\
')
    f.write('Validation Accuracy: ' + str(acc) + '\
')
    f.write(report)
print('Report saved to model_report.txt')


# Save predictions for test.csv
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Preprocess test
X_test = test_dataset[categorical_cols + numeric_cols].copy()
X_test.update(X_test[categorical_cols].fillna('No'))
X_test.update(X_test[numeric_cols].fillna(test_dataset[numeric_cols].median()))


test_pred = clf.predict(X_test)


submission = pd.DataFrame({'id': test_dataset['id'], 'Personality': test_pred})
submission 



submission.to_csv('submission.csv', index=False)
print('submission.csv generated')




