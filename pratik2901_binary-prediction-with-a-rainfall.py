#! pip install dataprep -q


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from pathlib import Path
# from dataprep.eda import create_report
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Import from imbalanced learn
from sklearn.base import BaseEstimator, TransformerMixin # needed for custom transformer
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Add advanced models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


INPUT_PATH = Path('/kaggle/input/playground-series-s5e3/')
OUTPUT_PATH = Path('/kaggle/working/')


train_data = pd.read_csv(INPUT_PATH/'train.csv', index_col = 'id')
#test_data = pd.read_csv(INPUT_PATH/'test.csv', index_col = 'id')
submission_data = pd.read_csv(INPUT_PATH/'sample_submission.csv')


train_data.head()


def create_features(dataframe: pd.DataFrame)-> pd.DataFrame :
    dataframe['date'] = pd.to_datetime(dataframe['day'], format='%j', errors='coerce')
    # Extract day, month and quarter
    dataframe['day_of_month'] = dataframe['date'].dt.day
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['quarter'] = dataframe['date'].dt.quarter
    dataframe['day_of_week'] = dataframe['date'].dt.dayofweek # The day of the week with Monday=0, Sunday=6.
    dataframe['ind_month_start'] = dataframe['date'].dt.is_month_start
    dataframe['ind_month_end'] = dataframe['date'].dt.is_month_end
    dataframe['week_of_year'] = dataframe['date'].dt.isocalendar().week
    
    # Drop intermediate 'date' column
    dataframe.drop(columns=['date'], inplace=True)

    # Add cyclical encoding
    dataframe['day_sin'] = np.sin((2 * np.pi * dataframe['day']) / 365)
    dataframe['day_cos'] = np.cos((2 * np.pi * dataframe['day']) / 365)

    #Wind Direction Components
    dataframe['wind_x'] = np.sin(np.pi * dataframe['winddirection'] / 180)
    dataframe['wind_y'] = np.cos(np.pi * dataframe['winddirection'] / 180)
    
    #Temperature Range
    dataframe['temp_range'] = dataframe['maxtemp'] - dataframe['mintemp']
    #Wet Bulb Temperature
    dataframe['wetbulb_temp'] = dataframe['temparature'] - ((100 - dataframe['humidity'])/5)
    #Temperature Anomaly (Real)
    dataframe['temp_anomaly_real'] = dataframe['temparature'] - dataframe['mintemp']
    #Temperature Anomaly (Synthetic)
    dataframe['avg_vs_synth'] = dataframe['temparature'] - (dataframe['temp_range']/2)
    #Humidity to Temperature Ratio
    dataframe['humidity_temp_ratio'] = dataframe['humidity'] / (dataframe['temparature']+1)
    #Cloud Cover Percentage
    dataframe['cloud_cover_idx'] = 100 - (dataframe['sunshine']*10)
    #Clear Sky Probability
    dataframe['clear_sky_prob'] = dataframe['sunshine']/24
    #Wind Power
    dataframe['wind_power'] = 0.5* 1.225 * np.power((dataframe['windspeed']/3.6),3)
    #Heat Index
    dataframe['heat_index'] = (-8.784 + 1.611 * dataframe['temparature']) + (2.339 * dataframe['humidity']) - (0.146 * dataframe['temparature'] * dataframe['humidity'])
    #Dewpoint Depression
    dataframe['dewpoint_depression'] = dataframe['temparature'] - dataframe['dewpoint']
    #Apparent Temperature
    dataframe['apparent_temp'] = dataframe['temparature'] - ((dataframe['windspeed']*0.7)/3.6)   
    del dataframe['temp_range']
    return dataframe


train_data['rainfall'].value_counts(normalize=True).plot.bar(color = ["g","r"]) 
plt.title('Total number of Rain vs No Rain samples in the dataset') 
plt.show()


# --- 3. Split Data ---
X = train_data.drop(columns=['rainfall'])
y = train_data[['rainfall']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)


# --- 4. Define Preprocessing Steps ---
feature_creation = FunctionTransformer(create_features)
scaler = StandardScaler()
lda = LinearDiscriminantAnalysis(n_components=1)
feature_selector_rfe = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=10) # RFE
feature_selector_kbest = SelectKBest(f_classif, k=10) # KBest


ensemble = [('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
            ('lgbm', lgb.LGBMClassifier(random_state=42, is_unbalance=True)),
            ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc')),
            #('lr',LogisticRegression(random_state=42, class_weight='balanced')),
            ('svm',SVC(probability=True, random_state=42, class_weight='balanced'))]


# --- 5. Define Classifiers ---

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42), # Gradient Boosting is robust to imbalance.
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, is_unbalance=True,verbose = -1), # LightGBM handles imbalance well.
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc'), # XGBoost handles imbalance well.
    'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0), # CatBoost handles imbalance well.
    'Voting Classifier': VotingClassifier(estimators=ensemble, n_jobs=-1,voting='soft'), # soft voting is better for AUC
    'Stacking Classifier':StackingClassifier(estimators=ensemble,final_estimator=LogisticRegression(),n_jobs=-1,stack_method='predict_proba')
}


# --- 6. Define Evaluation Metric and Cross-Validation Strategy ---
auc_scorer = make_scorer(roc_auc_score)
cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42) # Stratified for imbalanced data


# --- 7. Pipeline and Model Evaluation ---
results = {}
for name, clf in classifiers.items():
    # Using imbalanced-learn pipeline to handle SMOTE *after* train-test split, *before* cross validation
    pipeline = ImbPipeline([
        ('features', feature_creation),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler),
        ('feature_selector', feature_selector_kbest), # or feature_selector_rfe
        ('smote', SMOTE(random_state=42)),
        ('classifier', clf)
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=auc_scorer)
    results[name] = scores
    print(f'{name} - Mean AUC-ROC: {scores.mean():.4f}, Std: {scores.std():.4f}')


# --- 8. Best Model Selection and Test Set Evaluation ---
best_model_name = max(results, key=lambda k: results[k].mean())
best_model = classifiers[best_model_name]

final_pipeline = ImbPipeline([
        ('features', feature_creation),
        ('scaler', scaler),
        ('feature_selector', feature_selector_kbest), # or feature_selector_rfe
        ('smote', SMOTE(random_state=42,sampling_strategy='not majority')),
        ('classifier', best_model)
    ])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f'\nBest Model ({best_model_name}) - Test AUC-ROC: {test_auc:.4f}')


y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Find optimal threshold
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)


# --- 9. Feature Importance (Example with Random Forest) ---
if best_model_name == 'Random Forest':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances (Random Forest):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

elif best_model_name == 'Gradient Boosting':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances (Gradient Boosting):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

elif best_model_name == 'Logistic Regression':
    selected_features = final_pipeline.named_steps['feature_selector'].get_support(indices=True)
    feature_names = final_pipeline.named_steps['features'].transform(X_train).columns[selected_features]
    coefficients = final_pipeline.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    print("\nFeature Coefficients (Logistic Regression):")
    print(feature_importance.sort_values(by='Coefficient', ascending=False))

elif best_model_name == 'Decision Tree':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances (Decision Tree):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

elif best_model_name == 'LightGBM':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances (LightGBM):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

elif best_model_name == 'XGBoost':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("\nFeature Importances (XGBoost):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

elif best_model_name == 'CatBoost':
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': final_pipeline.named_steps['classifier'].feature_importances_})
    print("\nFeature Importances (CatBoost):")
    print(feature_importance.sort_values(by='Importance', ascending=False))
else:
    print(f"Feature importance not available for {best_model_name}")


# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Rain', 'Rain'])
disp.plot()
plt.title(f'Confusion Matrix - {best_model_name} \n[FPR: {round(cm[1][0]/cm[1].sum(),2)}] - [FNR: {round(cm[0][1]/cm[0].sum(),2)}] ')
plt.show()


# Evaluate the model
cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Rain', 'Rain'])
disp.plot()
plt.title(f'Confusion Matrix(Optimal Threshold) - {best_model_name} \n[FPR: {round(cm[1][0]/cm[1].sum(),2)}] - [FNR: {round(cm[0][1]/cm[0].sum(),2)}] ')
plt.show()


test_data = pd.read_csv(INPUT_PATH/'test.csv', index_col = 'id')


if 'rainfall' in test_data.columns:
    del test_data['rainfall']


test_data.isnull().sum()


print(test_data['winddirection'].mode())
test_data['winddirection'] = test_data['winddirection'].fillna(70)


test_data.isnull().sum()


# Make predictions on the test set
y_pred = final_pipeline.predict(test_data)
y_pred_proba = final_pipeline.predict_proba(test_data)[:, 1]
test_data['rainfall'], test_data['rainfall_proba'] = y_pred,y_pred_proba
test_data['rainfall_proba_optimal'] = (y_pred_proba >= optimal_threshold).astype(int)


submission_data=test_data['rainfall_proba_optimal'].reset_index()
submission_data.columns =  ['id','rainfall']


submission_data.head()


submission_data.to_csv(OUTPUT_PATH/"submissoin-V5-OptimalProbTh.csv",index=False)


# create_report(train_data)

