LEARNING_RATE = 0.001
N_OF_ESTIMATES = 500
RANDOM_SEED = 42


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline


from sklearn.metrics import  classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


train = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")
# submission = pd.read_csv("/kaggle/input/playground-series-s4e2/sample_submission.csv")


train_df = train.copy()
test_df = test.copy()


train.head()


train.shape


train.rename(columns={"family_history_with_overweight": "family_history"}, inplace=True)


train.info()


train.describe().T.style.background_gradient(cmap="Greens")


train.isnull().sum()/train.shape[0] # percentage of the null values


train.duplicated().sum()


numerical = train.select_dtypes(include=["number"])
numerical.columns


categorical = train.select_dtypes(include=["object"])
categorical.columns


for col in list(train.describe(include="object")):
    print(f"Column: {col}'s count values:\n")

    # Create a dictionary to store value counts
    value_count_dict = {
        'Value': train[col].value_counts().index.tolist(),
        'Count': train[col].value_counts().values.tolist()
    }

    # Convert dictionary to DataFrame
    value_count_df = pd.DataFrame(value_count_dict)
    display(value_count_df)
    
    print("\n" + "-"*40 + "\n")


# Loop over numerical columns
for col in list(train.select_dtypes(include='number')):
    print(f"Column: {col} - Value Counts:\n")
    
    # Create a dictionary to store value counts
    value_count_dict = {
        'Value': train[col].value_counts().index.tolist(),
        'Count': train[col].value_counts().values.tolist()
    }

    # Convert to DataFrame
    value_count_df = pd.DataFrame(value_count_dict)
    display(value_count_df)

    print("\n" + "-"*40 + "\n")



# Initialize lists to store results
number_of_outliers = [None] * len(train.select_dtypes(include=["number"]).columns)
q99 = [None] * len(train.select_dtypes(include=["number"]).columns)
q1 = [None] * len(train.select_dtypes(include=["number"]).columns)
outlier_percentage = [None] * len(train.select_dtypes(include=["number"]).columns)
total_rows = len(train)

# Loop over numerical columns
for i, p in enumerate(train.select_dtypes(include=["number"]).columns):
    q99[i], q1[i] = np.percentile(train[p], [99, 1])
    
    # Identify outliers (values beyond 99th and 1st percentiles)
    outliers = (train[p] > q99[i]) | (train[p] < q1[i])
    number_of_outliers[i] = outliers.sum()
    
    # Calculate percentage of outliers
    outlier_percentage[i] = (number_of_outliers[i] / total_rows) * 100
    
    # Print the results
    print(f'Outliers in {p}: {number_of_outliers[i]} ({outlier_percentage[i]:.2f}% of total rows)')
    print("*" * 40)


# Convert results into a DataFrame for better readability
outlier_summary = pd.DataFrame({
    'Column': train.select_dtypes(include=["number"]).columns,
    'Number of Outliers': number_of_outliers,
    'Outlier Percentage': outlier_percentage,
    '1st Percentile (q1)': q1,
    '99th Percentile (q99)': q99
})

# Display the outlier summary
display(outlier_summary)











numerical.hist(figsize=(12, 10), bins=20, color='#4caba4', grid=False)


def categorical_analysis(col, categorical):
    custom_colors = ["#4caba4", "#6782a8", "#a3c4dc", "#96d1c7", "#d0e1f9", "#7da6bf", "#b0d6d5"]
    
    value_counts = categorical[col].value_counts()
    counts = categorical[col].value_counts()
    labels = counts.index.tolist()
    values = counts.values
    colors = custom_colors[:len(labels)]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(col, fontsize=16, fontweight='bold')

    # Donut chart
    wedges, texts, autotexts = ax[0].pie(
        values, labels=labels, autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.5), colors=colors,
        pctdistance=0.75, textprops={'fontsize': 10, 'color': 'white', 'weight': 'bold'}
    )
    ax[0].set_title('Donut Chart', fontsize=12, fontweight='bold')

    # Bar plot
    sns.countplot(data=categorical, y=col, ax=ax[1], order=labels, palette=colors)
    for i, v in enumerate(value_counts):
        ax[1].text(v + 1, i, str(v), color='black',fontsize=10, va='center')
    sns.despine(ax=ax[1])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].set_title('Count Plot', fontsize=12, fontweight='bold')
    ax[1].tick_params(axis='y', labelsize=9)
    

    plt.tight_layout()
    plt.show()



for column in categorical.columns:
    print(f"Plotting: {column}")
    try:
        categorical_analysis(column, categorical=categorical)
    except Exception as e:
        print(f"Error plotting {column}: {e}")








def plot_categorical_data(data , column):
    
    fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
    
    sns.countplot(data=data,x=column,palette="Blues_d",ax=axes[0])
    sns.countplot(data=data,x=column,palette="Blues_d",hue="NObeyesdad",ax=axes[1])
    sns.countplot(data=data,x="NObeyesdad",palette="Blues_d",hue=column,ax=axes[2])
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=90)



plot_categorical_data(train, "Gender")


plot_categorical_data(train, "family_history")


plot_categorical_data(train, "FAVC")


plot_categorical_data(train, "CAEC")


plot_categorical_data(train, "SMOKE")


plot_categorical_data(train, "SCC")


plot_categorical_data(train, "CALC")


plot_categorical_data(train, "MTRANS")





# def plot_numerical_data(data: pd.DataFrame, column: str):
#     # Clean data by replacing inf values with NaN
#     data = data.replace([np.inf, -np.inf], np.nan)
#     # pd.option_context('mode.use_inf_as_na', True)
    
#     # Create the plot with 1 row and 4 columns
#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
    
#     # Plot 1: Histogram without hue
#     sns.histplot(data=data, x=column, ax=axes[0])
    
#     # Plot 2: Histogram with hue
#     sns.histplot(data=data, x=column, hue="NObeyesdad", ax=axes[1], palette="Blues_d")
    
#     # Plot 3: Boxplot
#     sns.boxplot(data=data, x=column, ax=axes[2])
    
#     # Plot 4: Barplot with the mean value of column grouped by 'NObeyesdad'
#     mean = data[[column, "NObeyesdad"]].groupby(("NObeyesdad")).mean().reset_index()
#     sns.barplot(data=mean, x="NObeyesdad", y=column, ax=axes[3])
    
#     # Rotate x-axis ticks for all plots
#     for ax in axes:
#         ax.tick_params(axis='x', rotation=90)

#     # Show the plot
#     plt.show()



# plot_numerical_data(train, "Age")


# plot_numerical_data(train, "Height")


# plot_numerical_data(train, "Weight")


# plot_numerical_data(train, "FCVC")


# plot_numerical_data(train, "NCP")


# plot_numerical_data(train, "CH2O")


# plot_numerical_data(train, "FAF")


# plot_numerical_data(train, "TUE")








# print the heat map of numerical columns
plt.figure(figsize=(12, 10))
sns.heatmap(train[numerical.columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()





gender_mapping = {'Male': 0, 'Female': 1}
yes_no_mapping = {'no': 0, 'yes': 1}
freq_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
transportion_mapping = {'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Automobile': 4}
target_mapping = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6}
mapping_dict = {
    'Gender' : gender_mapping,
    'family_history_with_overweight' : yes_no_mapping,
    'FAVC' : yes_no_mapping,
    'CAEC': freq_mapping,
    'SMOKE': yes_no_mapping,
    'SCC': yes_no_mapping,
    'CALC': freq_mapping,
    'MTRANS': transportion_mapping,
    'NObeyesdad': target_mapping
}

def preprocess(df):
    df = df.copy()
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    df = df.drop(columns=['id'])
    return df


train = preprocess(train)


plt.figure(figsize=(20, 12))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='black', cbar=True)








train2 = train_df.copy()


train2 = preprocess(train2)


train , test = train_test_split(train2, test_size=0.2, random_state=RANDOM_SEED)


X_train = train.drop(columns=['NObeyesdad'])
y_train = train['NObeyesdad']
X_test = test.drop(columns=['NObeyesdad'])
y_test = test['NObeyesdad']


lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

# Create a DataFrame for plotting
lda_df = pd.DataFrame({'LDA1': X_lda[:, 0], 'LDA2': X_lda[:, 1], 'Target': y_train})
sns.scatterplot(data=lda_df, x='LDA1', y='LDA2', hue='Target')
plt.title('LDA 2D projection')
plt.show()


# Scale the data after splitting to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





# Create the model
rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Predict on the testing data
y_pred = rf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))





# Create the model
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.01, subsample=0.8, random_state=42)

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Predict on the testing data
y_pred_xgb = xgb_model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred_xgb))





# Create the model
lgb_model = LGBMClassifier(random_state=42)

# Fit the model on the training data
lgb_model.fit(X_train_scaled, y_train)

# Predict on the testing data
y_pred_lgb = lgb_model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred_lgb))


# print the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_lgb)
print(f"Accuracy: {accuracy:5f}")





importances = lgb_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]


# Plot the feature importances of the forest
plt.figure(figsize=(10, 12))
plt.title("Feature Importances")
colors = sns.color_palette("viridis", len(importances))
plt.barh(range(X_train.shape[1]), importances[indices], color=colors)
plt.yticks(range(X_train.shape[1]), feature_names[indices])
plt.gca().invert_yaxis()  # Invert y-axis to match ranking order
plt.xlabel("Importance Score")
plt.show()






# Define the map_back function
def map_back(predictions, true_labels):
    """
    Maps predictions back to their original labels.
    Assumes predictions and true_labels are pandas Series or numpy arrays.
    """
    target_map = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Overweight_Level_I": 2, "Overweight_Level_II": 3,
                "Obesity_Type_I": 4, "Obesity_Type_II": 5, "Obesity_Type_III": 6}
    
    # Create a reverse mapping
    reverse_map = {v: k for k, v in target_map.items()}
    
    # Map the predictions and true labels
    predictions = [reverse_map[p] for p in predictions]
    true_labels = [reverse_map[t] for t in true_labels]

    return pd.DataFrame({'Predicted': predictions, 'True': true_labels}).reset_index(drop=True)


y_pred_mapped = map_back(y_pred_lgb, y_test)
y_pred_mapped





def map_predictions_to_labels(predictions):
    reverse_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }
    return [reverse_map[pred] for pred in predictions]


test = test_df.copy()


test = preprocess(test)


test = scaler.transform(test)


y_prediction = lgb_model.predict(test)
y_prediction


y_prediction_back = map_predictions_to_labels(y_prediction)


submission = pd.DataFrame({'id': test_df['id'], 'NObeyesdad': y_prediction_back})


submission.to_csv('submission.csv', index=False)
submission.head()







