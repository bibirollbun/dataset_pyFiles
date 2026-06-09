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



import os

train_path = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
test_path = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'

if os.path.exists(train_path) and os.path.exists(test_path):
	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)
else:
	print("One or both files do not exist.")
    



feature_cols = [
    'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 
    'Physical-BMI', 'Physical-Height', 'Physical-Weight', 
    'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
    'Physical-HeartRate', 'Physical-Systolic_BP',  
    'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 
    'Fitness_Endurance-Time_Sec', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 
    'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 
    'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 
    'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 
    'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 
    'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI', 
    'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 
    'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 
    'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 
    'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total', 
    'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
    'PreInt_EduHx-computerinternet_hoursday'
]
# Loại bỏ các hàng có giá trị NaN trong y
train_df = train_df.dropna(subset=['sii'])
X = train_df[feature_cols]
y = train_df['sii']

# Define the preprocessing pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, feature_cols)
])

# Fit and transform X
preprocessor.fit(X)
X = pd.DataFrame(preprocessor.transform(X), columns=feature_cols)



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
    LinearSVC(max_iter=10000, random_state=seed),
    SVC(random_state=seed),
    KNeighborsClassifier(metric='minkowski', p=2),
    LogisticRegression(solver='liblinear', max_iter=1000),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    ExtraTreesClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
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
X_test = test_df[feature_cols]
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=feature_cols)

# Use the trained model to make predictions
best_model = XGBClassifier(eval_metric='logloss', random_state=2023)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'sii': y_test_pred
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")  # In ra thông báo khi hoàn thành




