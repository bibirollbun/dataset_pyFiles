import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')




import os
train_path = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
test_path = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'


if os.path.exists(train_path) and os.path.exists(test_path):
	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)
else:
	print("One or both files do not exist.")
    



train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
columns_not_in_test = sorted(list(train_cols - test_cols))

columns_to_exclude = ['PCIAT-PCIAT_Total', 'PCIAT-Season', 'sii']
question_columns = [
    col for col in columns_not_in_test if col not in columns_to_exclude
]

question_columns


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='PCIAT-PCIAT_Total', y='sii', alpha=0.6)

plt.title('Biểu đồ phân tán giữa PCIAT-PCIAT_Total và sii', fontsize=14)
plt.xlabel('PCIAT-PCIAT_Total', fontsize=12)
plt.ylabel('sii', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


def recalculate_sii(row):
    if pd.isna(row['PCIAT-PCIAT_Total']):
        return np.nan
    max_possible = row['PCIAT-PCIAT_Total'] + row[question_columns].isna().sum() * 5
    if row['PCIAT-PCIAT_Total'] <= 30 and max_possible <= 30:
        return 0
    elif 31 <= row['PCIAT-PCIAT_Total'] <= 49 and max_possible <= 49:
        return 1
    elif 50 <= row['PCIAT-PCIAT_Total'] <= 79 and max_possible <= 79:
        return 2
    elif row['PCIAT-PCIAT_Total'] >= 80 and max_possible >= 80:
        return 3
    return np.nan

train_df['recalc_sii'] = train_df.apply(recalculate_sii, axis=1)


train_df['recalc_sii'].isna().sum()


mismatch_rows = train_df[
    (train_df['recalc_sii'] != train_df['sii']) & train_df['sii'].notna()
]

mismatch_rows[question_columns + ['recalc_sii']].style.map(
    lambda x: 'background-color: #FFC0CB' if pd.isna(x) else ''
)


train_df['sii'] = train_df['recalc_sii']
train_df = train_df.drop(mismatch_rows.index)

train_df[columns_not_in_test + ['recalc_sii']]


na_total_rows = train_df[train_df['sii'].isna()]
na_total_rows


train_df = train_df.dropna(subset=['PCIAT-PCIAT_Total'])
train_df


for column in question_columns:
    if train_df[column].isna().any():
        mode_value = train_df[column].mode()[0]
        train_df[column] = train_df[column].fillna(mode_value)

train_df[columns_not_in_test + ['recalc_sii']]


train_df.drop(columns='recalc_sii', inplace=True)
train_df[columns_not_in_test]


train_df = train_df.drop(columns=question_columns, errors='ignore')
train_df = train_df.drop(columns='PCIAT-PCIAT_Total', errors='ignore')


train_df.shape


train_df.info()


train_df['Basic_Demos-Age'].describe()


# Đếm số lượng học sinh 
studytime_counts = train_df['Basic_Demos-Age'].value_counts().sort_index()

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
bars = plt.bar(studytime_counts.index.astype(str), studytime_counts.values, color='skyblue', edgecolor='black')

# Thêm tiêu đề và nhãn
plt.title('Biểu đồ phân phối nhóm tuổi người tham gia', fontsize=14)
plt.xlabel('Nhóm tuổi', fontsize=12)
plt.ylabel('Số lượng người tham gia', fontsize=12)

# Ghi số lượng lên đầu cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()


def apply_age_group(df):
    df['Age Group'] = pd.cut(
        df['Basic_Demos-Age'],
        bins=[4, 12, 18, 22],
        labels=['Children', 'Adolescents', 'Adults'],
    )
    return df

train_df = apply_age_group(train_df)
test_df = apply_age_group(test_df)


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
train_df['Age_Group_Label'] = le.fit_transform(train_df['Age Group'])
test_df['Age_Group_Label'] = le.fit_transform(test_df['Age Group'])
train_df[['Age Group', 'Age_Group_Label']]


# Lấy các cột cần xem tương quan
selected_cols = ['CGAS-CGAS_Score', 'Basic_Demos-Age', 'Basic_Demos-Sex']

# Tính ma trận tương quan
corr = train_df[selected_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Basic_Demos and SII')
plt.tight_layout()
plt.show()


season_counts = train_df['CGAS-Season'].value_counts(normalize=True)
season_counts





sns.scatterplot(data=train_df, x='sii', y='CGAS-CGAS_Score', palette='viridis')
plt.title("Relationship between Age and CGAS Score")
plt.show()


missing_rows = train_df['CGAS-CGAS_Score'].isna()
missing_rows 


physical_columns = [
 'Physical-BMI',
 'Physical-Height',
 'Physical-Weight',
 'Physical-Waist_Circumference',
 'Physical-Diastolic_BP',
 'Physical-HeartRate',
 'Physical-Systolic_BP'
]

wh_cols = [
    'Physical-BMI', 'Physical-Height',
    'Physical-Weight', 'Physical-Waist_Circumference'
]

heart_cols = [
 'Physical-Diastolic_BP',
 'Physical-HeartRate',
 'Physical-Systolic_BP'
]


train_df[wh_cols].describe()


train_df[wh_cols] = train_df[wh_cols].replace(0, np.nan)
test_df[wh_cols] = test_df[wh_cols].replace(0, np.nan)
train_df[wh_cols].describe()


train_df[wh_cols].isna().sum()


# Lấy các cột cần xem tương quan
selected_cols = ['Physical-Height', 'Physical-Weight', 'Basic_Demos-Age', 'Basic_Demos-Sex']

# Tính ma trận tương quan
corr = train_df[selected_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Basic_Demos and SII')
plt.tight_layout()
plt.show()


sns.boxplot(data=train_df, x='Basic_Demos-Age', y='Physical-Height')
plt.title("CGAS Score Distribution by Sex")
plt.show()


sns.boxplot(data=train_df, x='Basic_Demos-Age', y='Physical-Weight')
plt.title("CGAS Score Distribution by Sex")
plt.show()


sns.boxplot(data=train_df, x='Basic_Demos-Sex', y='Physical-Weight')
plt.title("CGAS Score Distribution by Sex")
plt.show()


lbs_to_kg = 0.453592
inches_to_cm = 2.54

def process_physical_BMI(df):
    df['Physical-Weight'] = df['Physical-Weight'] * lbs_to_kg
    df['Physical-Height'] = df['Physical-Height'] * inches_to_cm
    df['Physical-Waist_Circumference'] = df['Physical-Waist_Circumference'] * inches_to_cm
    
    df['Physical-BMI'] = np.where(
        df['Physical-Weight'].notna() & df['Physical-Height'].notna(),
        df['Physical-Weight'] / ((df['Physical-Height'] / 100) ** 2),
        np.nan
    )
    
    return df

train_df = process_physical_BMI(train_df)
test_df = process_physical_BMI(test_df)


def convert_season_to_numeric(df, season_columns):
    # Định nghĩa mapping thứ tự cho các mùa
    season_mapping = {
        'Spring': 0,
        'Summer': 1,
        'Fall': 2,
        'Winter': 3
    }
    
    # Kiểm tra từng cột trong danh sách
    for col in season_columns:
        if col in df.columns:
            # In ra các giá trị trước khi ánh xạ
            print(f"Giá trị trước khi ánh xạ trong cột {col}:")
            print(df[col].unique())
            
            # Áp dụng mapping
            df[col] = df[col].map(season_mapping)
            
            # In ra các giá trị sau khi ánh xạ
            print(f"Giá trị sau khi ánh xạ trong cột {col}:")
            print(df[col].unique())
    
    return df


season_columns = [
    'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
    'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
    'PAQ_A-Season', 'PAQ_C-Season',  'SDS-Season', 
    'PreInt_EduHx-Season'
]

# Áp dụng hàm cho tập train và test
test_df = convert_season_to_numeric(test_df, season_columns)

# Kết quả
print("Test DataFrame sau khi chuyển đổi:")
print(test_df[season_columns].head())

train_df = convert_season_to_numeric(train_df, season_columns)






train_df['Physical-Weight'].isna().sum()


# Lấy các cột cần xem tương quan
selected_cols = ['BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season', 'sii']

# Tính ma trận tương quan
corr = train_df[selected_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Basic_Demos and SII')
plt.tight_layout()
plt.show()


train_df = train_df.drop(columns=question_columns, errors='ignore')



seasonCols = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season', 'PCIAT-Season']

train_df = train_df.drop(columns=question_columns, errors='ignore')
train_df = train_df.drop(columns=seasonCols, errors='ignore')
train_df = train_df.drop(columns=['PCIAT-PCIAT_Total', 'Age Group', 'Age_Group_Label'], errors='ignore')


filtered_features = train_df.drop(columns=['id', 'sii'])  

# Loại bỏ các hàng có giá trị NaN trong y
train_df = train_df.dropna(subset=['sii'])
X = filtered_features
y = train_df['sii']

# Define the preprocessing pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, filtered_features.columns.tolist())
])

# Fit and transform X
preprocessor.fit(X)
X = pd.DataFrame(preprocessor.transform(X), columns=filtered_features.columns)



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Random seed
seed = 2023

# List of models
models = [
    #LinearSVC(max_iter=10000, random_state=seed),
    #SVC(random_state=seed),
    #KNeighborsClassifier(metric='minkowski', p=2),
    #LogisticRegression(solver='liblinear', max_iter=1000),
    #DecisionTreeClassifier(random_state=seed),
    #RandomForestClassifier(random_state=seed),
    #ExtraTreesClassifier(random_state=seed),
    #AdaBoostClassifier(random_state=seed),
    XGBClassifier(eval_metric='logloss', random_state=seed)    
]

# Function to generate baseline results
def generate_baseline_results(models, X, y, metrics='accuracy', cv=5, plot_results=False):
    # Define k-fold
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    entries = []
    
    # Loop through each model
    for model in models:
        model_name = model.__class__.__name__
        print(f"Training: {model_name}")
        scores = cross_val_score(model, X, y, scoring=metrics, cv=kfold)
        # Lưu kết quả của tất cả các mô hình vào entries
        entries.extend([(model_name, fold_idx, score) for fold_idx, score in enumerate(scores)])
    
    # Create DataFrame
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_id', 'accuracy_score'])
    
    # Optional: Plot results if specified
    if plot_results:
        sns.boxplot(x='model_name', y='accuracy_score', data=cv_df, color='lightblue', showmeans=True)
        plt.title("Boxplot of baseline Model Accuracy using 5-fold cross-validation")
        plt.xticks(rotation=45)
        plt.show()
    
    # Summary result
    mean = cv_df.groupby('model_name')['accuracy_score'].mean()
    std = cv_df.groupby('model_name')['accuracy_score'].std()

    baseline_results = pd.concat([mean, std], axis=1)
    baseline_results.columns = ['Mean', 'Standard Deviation']

    # Sort results
    baseline_results.sort_values(by='Mean', ascending=False, inplace=True)

    return baseline_results

# Chạy hàm và hiển thị kết quả
cv_results = generate_baseline_results(models, X, y, metrics='accuracy', cv=5, plot_results=False)

# In toàn bộ kết quả
print(cv_results)



# Preprocess the test data
X_test = test_df.drop(columns=['id'])
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=filtered_features.columns)



# Use the trained model to make predictions
best_model =  XGBClassifier(solver='liblinear', max_iter=1000)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'sii': y_test_pred
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.") 
print(submission)







