import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

sns.set_style("whitegrid")


polycystic_df  = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
polycystic_df.drop('ID', axis=1, inplace=True)


test_data = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')
test_data.drop('ID', axis=1, inplace=True)


polycystic_df.head()


polycystic_df.info()


polycystic_df.isnull().sum()


polycystic_df.dropna(axis=0, how='any', inplace=True)


polycystic_df.isna().sum()


def polycystic_age(data):
    # Modify the data directly without using data.copy
    # This function performs 3 demographic segmentations
    data.loc[data['Age'].isin([
        'Less than 20-25', '20-25', '15-20', 'Less than 20', 'Less than 20)', '22-25', '20'
    ]), 'Age'] = '15-25'  # 15-24
    
    data.loc[data['Age'].isin([
        '25-30', '30-35', '30-25', '30-30', '25-25'
    ]), 'Age'] = '25-35'  # 25-34
    
    data.loc[data['Age'].isin([
        '35-44', '45 and above', '30-40', '45-49', '50-60'
    ]), 'Age'] = '+35'  # +35


polycystic_age(polycystic_df)


polycystic_df['Age'].value_counts(normalize=True)


def polycystic_weight_range(data):    
    bins = [0, 50, 68, 81, 100]  # Define weight ranges
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    data['weight_range'] = pd.cut(data['Weight_kg'], bins=bins, labels=labels)


polycystic_weight_range(polycystic_df)


polycystic_df['weight_range'].value_counts(normalize=True)


def polycystic_no_yes_adjustment(data, col):
    data[col] = data[col].replace({
        'Yes, diagnosed by a doctor': 'Yes',
        'No, Yes, not diagnosed by a doctor':'No'
    })


def polycystic_no_yes_not_adjustment(data, col):
    data[col] = data[col].replace({
        'Yes Significantly': 'Yes',
        'No, Yes, not diagnosed by a doctor':'No'
    })


def polycystic_somewhat_adjustment(data, col):
    data[col] = data[col].replace({
        'Somewhat': 'Yes',
        'Yes Significantly': 'Yes',
        'Not at All':'No',
        'Not Much': 'No'
    })


polycystic_no_yes_adjustment(polycystic_df, "Hirsutism")


polycystic_df['Hirsutism'].value_counts(normalize=True)


polycystic_no_yes_adjustment(polycystic_df, "Conception_Difficulty")


polycystic_df['Conception_Difficulty'].value_counts(normalize=True)


polycystic_no_yes_not_adjustment(polycystic_df, "Hormonal_Imbalance")


polycystic_df['Hormonal_Imbalance'].value_counts(normalize=True)


polycystic_somewhat_adjustment(polycystic_df, 'Exercise_Benefit')


polycystic_df['Exercise_Benefit'].value_counts(normalize=True)


def exercise_type_generalisation(data):
    data.loc[data['Exercise_Type'].isin([
        "Cardio (e.g., running, cycling, swimming)",
        "Cardio (e.g.",
        "Cardio (e.g., running, cycling, swimming), None",
        "Cardio"
    ]), 'Exercise_Type'] = 'Cardio'

    data.loc[data['Exercise_Type'].isin([
        "Flexibility and balance (e.g., yoga, pilates)",
        "Flexibility and balance (e.g., yoga, pilates), None",
        "Flexibility and balance (e.g."
    ]), 'Exercise_Type'] = 'Flexibility and Balance'
    
    data.loc[data['Exercise_Type'].isin([
        "Strength training (e.g., weightlifting, resistance exercises)",
        "Strength training (e.g.",
        "Strength training",
        "Strength (e.g."
    ]), 'Exercise_Type'] = 'Strength Training'
    
    data.loc[data['Exercise_Type'].isin([
        "High-intensity interval training (HIIT)"
    ]), 'Exercise_Type'] = 'HIIT'
    
    data.loc[data['Exercise_Type'].isin([
        "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)",
        "Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)",
        "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)",
        "Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)",
        "Strength training (e.g.", "Flexibility and balance (e.g."
    ]), 'Exercise_Type'] = 'Mixed Exercise'
    
    data.loc[data['Exercise_Type'].isin([
        "No Exercise",
        "Not Applicable",
        "No"
    ]), 'Exercise_Type'] = 'No Exercise'
    
    data.loc[data['Exercise_Type'].isin([
        "Somewhat",
        "Yes Significantly",
        "Sleep_Benefit",
    ]), 'Exercise_Type'] = 'Other/Somewhat'


exercise_type_generalisation(polycystic_df)


polycystic_df['Exercise_Type'].value_counts(normalize=True)


exercise_freq_map = {
    "Rarely": "Rarely",
    "1-2 Times a Week": "1-2 a week",
    "Never": "Never",
    "3-4 Times a Week": "3-4 a week",
    "6-8 Times a Week": "6-8 a week",
    "Less than usual": "Rarely",
    "Less than 6 hours": "< 6h",
    "Less than 6-8 Times a Week": "Rarely",
    "30-35": "Rarely",
    "Somewhat": "Rarely",
    "6-8 hours": "6-8 a week",
    "1/2 Times a Week": "1-2 a week"
}

def polycystic_exercise_freq_map(data):
    data["Exercise_Frequency"] = data["Exercise_Frequency"].map(exercise_freq_map)

polycystic_exercise_freq_map(polycystic_df)


exercise_duration_map = {
    "Not Applicable": "not applicable",
    "Less than 30 minutes": "< 30min",
    "30 minutes": "30min - 1 hour",
    "30 minutes to 1 hour": "30min - 1 hour",
    "45 minutes": "30min - 1 hour",
    "More than 30 minutes": "30min - 1 hour",
    "Less than 6 hours": "30min - 1 hour",
    "6-8 hours": "not applicable",
    "Less than 20 minutes": "< 30min",
    "20 minutes": "< 30min",
    "40 minutes": "30min - 1 hour",
    "3-4 Times a Week": "not applicable",
    "1-2 Times a Week": "not applicable",
    "Not Much": "not applicable"
}

def polycystic_exercise_duration_map(data):
    data["Exercise_Duration"] = data["Exercise_Duration"].map(exercise_duration_map)

polycystic_exercise_duration_map(polycystic_df)


sleep_hours_map = {
    "6-8 hours": "6-8 hours",
    "Less than 6 hours": "< 6h",
    "9-12 hours": "9-12 hours",
    "More than 12 hours": "outlier",
    "3-4 hours": "< 6h",
    "6-8 Times a Week": "outlier",
    "6-12 hours": "6-8 hours",
    "20 minutes": "outlier"
}

def polycystic_sleep_hours_map(data):
    data["Sleep_Hours"] = data["Sleep_Hours"].map(sleep_hours_map)
    # Remove outliers
    # data = data[data["Sleep_Hours"] != "outlier"]

polycystic_sleep_hours_map(polycystic_df)


polycystic_df.boxplot('Weight_kg')
plt.show()


def remove_outliers_iqr(df):
    data = df.copy()
    columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        f1 = q3 - 1.5 * iqr
        f2 = q3 + 1.5 * iqr
        data = data[(data[col] >= f1) & (data[col] <= f2)]
        
    return data


clean_polycystic_df = remove_outliers_iqr(polycystic_df)


clean_polycystic_df.boxplot('Weight_kg')
plt.show()


def polycystic_ovary_syndrome_by_col_plot(data,col,rotation):
    # Perform grouping and aggregation
    data_col_pcos = data.groupby([col])['PCOS'].value_counts().reset_index()
    # Create a pivot table to help us perform the desired viz
    pivot_data_age_pcos = data_col_pcos.pivot(index=col, columns='PCOS', values='count').fillna(0).sort_values('Yes',ascending=False)
    
    # stacked bar
    pivot_data_age_pcos.plot(kind='bar', stacked=True, figsize=(14, 6))
    plt.xlabel(col)
    plt.ylabel('count')
    plt.title(f'Polycystic ovary syndrome by {col}')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Age',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Exercise_Duration',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Exercise_Type',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Exercise_Frequency',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Sleep_Hours',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'weight_range',0)


polycystic_ovary_syndrome_by_col_plot(clean_polycystic_df,'Insulin_Resistance',0)


X = clean_polycystic_df.drop(['PCOS', 'weight_range'], axis=1)
y = clean_polycystic_df['PCOS']


# Define ordinal categories
ordinal_categories = { 
    "Age": ["15-25", "25-35", "+35"],
    "Exercise_Frequency": ["Never", "Rarely", "< 6h", "1-2 a week", "3-4 a week", "6-8 a week"],
    "Exercise_Duration": ["not applicable", "< 30min", "30min - 1 hour"],
    "Sleep_Hours": ["< 6h", "6-8 hours", "9-12 hours", "outlier"],
    # "weight_range": ["Underweight", "Normal", "Overweight", "Obese"]
}

# Define one-hot encoded columns
col_one_hot = ["Hormonal_Imbalance", "Hyperandrogenism", "Hirsutism", "Conception_Difficulty", "Insulin_Resistance", "Exercise_Type", "Exercise_Benefit"]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Define column transformer
transformer = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(
            categories=[ordinal_categories[col] for col in ordinal_categories],
        ), list(ordinal_categories.keys())),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'), col_one_hot)
    ],
    remainder="passthrough", verbose_feature_names_out=False
).set_output(transform="pandas")

# Transform the dataset
X_transformed = transformer.fit_transform(X)


X_transformed = (X_transformed-X_transformed.mean())/X_transformed.std()


X_transformed.head()


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

rf_classifier = RandomForestClassifier(n_estimators=40, random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)
logistic_rg = LogisticRegression()
svm_classifier = SVC(kernel='rbf', random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.2, random_state=42)


len(X_train), len(X_test), len(y_train), len(y_test)


rf_classifier.fit(X_train, y_train)
knn.fit(X_train, y_train)
logistic_rg.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)


print(f'Random forest accuracy: {rf_classifier.score(X_train, y_train)}')
print(f'KNN accuracy: {knn.score(X_train, y_train)}')
print(f'Logistic accuracy: {logistic_rg.score(X_train, y_train)}')
print(f'SVM accuracy: {svm_classifier.score(X_train, y_train)}')


print(f'Random forest accuracy: {rf_classifier.score(X_test, y_test)}')
print(f'KNN accuracy: {knn.score(X_test, y_test)}')
print(f'Logistic accuracy: {logistic_rg.score(X_test, y_test)}')
print(f'SVM accuracy: {svm_classifier.score(X_test, y_test)}')


y_pred = svm_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)


cm_display = ConfusionMatrixDisplay(cm, display_labels=['No','Yes'])
cm_display.plot(cmap=plt.cm.Blues)
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, pos_label='Yes')
print("Precision:", precision)

# Calculate recall (sensitivity)
recall = recall_score(y_test, y_pred, pos_label='Yes')
print("Recall (Sensitivity):", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, pos_label='Yes')
print("F1-Score:", f1)


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Convert y_test and y_pred to binary labels
y_test_bin = label_binarize(y_test, classes=['No', 'Yes']).ravel()
y_pred_bin = label_binarize(y_pred, classes=['No', 'Yes']).ravel()

# Compute ROC AUC Score
roc_auc = roc_auc_score(y_test_bin, y_pred_bin)
print("ROC AUC Score:", roc_auc)


polycystic_age(test_data)


polycystic_somewhat_adjustment(test_data, 'Conception_Difficulty')


polycystic_somewhat_adjustment(test_data, 'Insulin_Resistance')


polycystic_exercise_freq_map(test_data)


exercise_type_generalisation(test_data)


polycystic_exercise_duration_map(test_data)


polycystic_sleep_hours_map(test_data)


polycystic_somewhat_adjustment(test_data, 'Exercise_Benefit')


for column in test_data.columns:
    mode_value = polycystic_df[column].mode()[0]
    test_data[column] = test_data[column].fillna(mode_value)


X_test_transformed = transformer.fit_transform(test_data)


# Add missing columns to the test data
X_test_transformed['Exercise_Type_HIIT'] = 0
X_test_transformed['Exercise_Type_Mixed Exercise'] = 0
# Reorder columns in the test data to match the training data
X_test_transformed = X_test_transformed[X_transformed.columns]


y_pred_test_set = svm_classifier.predict(X_test_transformed)


sub = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/sample_submission.csv')
sub['PCOS'] = y_pred_test_set
sub['PCOS'] = sub['PCOS'].map({
    'No': 0,
    'Yes': 1
})
sub.to_csv('submission.csv', index=False)

