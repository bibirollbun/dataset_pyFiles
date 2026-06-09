import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import os


train_dataset = pd.read_csv('/kaggle/input/playground-series-s3e5/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s3e5/test.csv')


train_dataset


test_dataset


train_dataset_copy = train_dataset.copy()
test_dataset_copy = test_dataset.copy()


train_dataset_copy.info()


test_dataset_copy.info()


train_data_description = train_dataset_copy.describe()
train_data_description


train_dataset_copy = train_dataset_copy.drop(['Id'], axis=1)
test_dataset_copy = test_dataset_copy.drop(['Id'], axis=1)


quality_counts = train_dataset_copy['quality'].value_counts()
quality_counts


train_dataset_copy['quality'].unique()


labels = quality_counts.index
values = quality_counts.values

plt.figure(figsize=(8,6))
plt.bar(labels, values)
plt.xlabel('Quality Levels')
plt.ylabel('Number of Wines')
plt.title("Distribution of Wines by Quality Levels")
plt.show()


correlations = train_dataset_copy.corr()['quality'].sort_values(ascending=False)


sns.set_style("whitegrid")

plt.figure(figsize=(10,6))
sns.barplot(x=correlations.index, y=correlations.values, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.title('Feature Correlations with Target Variable (quality)')
plt.ylabel('Correlation Value')
plt.xlabel('Features')
plt.show()


numerical_features = train_dataset_copy.drop(['quality'], axis=1).columns

for feature in numerical_features:
  plt.figure(figsize=(14,6))

  plt.subplot(1,2,1)
  plt.hist(train_dataset_copy[feature], bins=30, edgecolor='k', alpha=0.7)
  plt.title(f'Histogram of: {feature}')
  plt.xlabel(feature)
  plt.ylabel('Frequency')


  plt.subplot(1,2,2)
  plt.boxplot(train_dataset_copy[feature], vert=False, patch_artist=True,
              boxprops=dict(facecolor='lightblue'))
  plt.title(f'Boxplot of: {feature}')
  plt.xlabel(feature)
  plt.tight_layout()
  plt.show


feature_analysis = {}

for feature in numerical_features:
  q1 = train_dataset_copy[feature].quantile(0.25)
  q3 = train_dataset_copy[feature].quantile(0.75)

  IQR = q3 - q1
  lower_bound = q1 - 1.5 * IQR
  upper_bound = q3 + 1.5 * IQR
  outliers = train_dataset_copy[(train_dataset_copy[feature] < lower_bound) | (train_dataset_copy[feature] > upper_bound)]
  number_outliers = len(outliers)

  feature_analysis[feature] = {
      'IQR': IQR,
      'Lower Bound': lower_bound,
      'Upper Bound': upper_bound,
      'Number of Outliers': number_outliers,
  }

  outlier_summary = pd.DataFrame(feature_analysis).T


outlier_summary



mean_values_by_quality = train_dataset_copy.groupby('quality').mean()

plt.figure(figsize=(12,8))
sns.heatmap(mean_values_by_quality, annot=True, fmt='.2f', cmap='YlGnBu',
            cbar=True)
plt.title('Heatmap of Feature Average Grouped by Quality')
plt.xlabel('Features')
plt.ylabel('Quality')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features = scaler.fit_transform(train_dataset_copy.drop(['quality'], axis=1))

scaled_features_df = pd.DataFrame(scaled_features, columns=train_dataset_copy.drop(['quality'], axis=1).columns)

corr_matrix = scaled_features_df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap (Standardized)')
plt.show()


scaled_features_df


from sklearn.model_selection import train_test_split

X = train_dataset_copy.iloc[:, :-1]
y = train_dataset_copy.iloc[:, -1]


X


y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, n_estimators=500, class_weight='balanced')

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy Score Random Forest: {accuracy_rf}')


print(classification_report(y_test, y_pred_rf))


cm_rf = confusion_matrix(y_test, y_pred_rf)

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
disp_rf.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix Random Forest')
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10]
}


grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                              param_grid_rf, cv=5, scoring='f1_weighted')

grid_search_rf.fit(X_train, y_train)
print(grid_search_rf.best_params_)


best_model_rf = grid_search_rf.best_estimator_
best_model_rf


y_pred_rf_grid = best_model_rf.predict(X_test)

print("Test Accuracy:", best_model_rf.score(X_test, y_test))

print(f'Accuracy Score Random Forest with Grid Search:', 
      accuracy_score(y_test, y_pred_rf_grid))


print(classification_report(y_test, y_pred_rf_grid))


cm_rf_grid = confusion_matrix(y_test, y_pred_rf_grid)

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf_grid, display_labels=rf.classes_)
disp_rf.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix Random Forest')
plt.show()


feature_importances_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances_rf)


importances = rf.feature_importances_
feature_importances = feature_importances_rf.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances Random Forest')
plt.gca().invert_yaxis()
plt.show()


X_train_selected = X_train.drop(['fixed acidity', 'pH', 'residual sugar', 'citric acid', 'free sulfur dioxide', 
                                 ], axis=1)
X_test_selected = X_test.drop(['fixed acidity', 'pH', 'residual sugar', 'citric acid', 'free sulfur dioxide',
                               ], axis=1)


rf.fit(X_train_selected, y_train)


y_pred_rf_selected = rf.predict(X_test_selected)


accuracy_rf_selected = accuracy_score(y_test, y_pred_rf_selected)
print(f'Accuracy Score Random Forest with selected Features: {accuracy_rf_selected}')


print(classification_report(y_test, y_pred_rf_selected))


cm_rf_selected = confusion_matrix(y_test, y_pred_rf_selected)

disp_rf_selected = ConfusionMatrixDisplay(confusion_matrix=cm_rf_selected, display_labels=rf.classes_)
disp_rf_selected.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix Random Forest with selected Features')
plt.show()


grid_search_rf.fit(X_train_selected, y_train)
print(grid_search_rf.best_params_)


best_model_rf_selected = grid_search_rf.best_estimator_
best_model_rf_selected


best_model_rf_selected.fit(X_train_selected, y_train)

y_pred_rf_selected_grid = best_model_rf_selected.predict(X_test_selected)

print('Test Accuracy: ', best_model_rf_selected.score(X_test_selected, y_test))

print('Accuracy Score for Random Forest with Feature Importance: ', 
      accuracy_score(y_test, y_pred_rf_selected_grid))


print(classification_report(y_test, y_pred_rf_selected_grid))


cm_rf_selected_grid = confusion_matrix(y_test, y_pred_rf_selected_grid)

disp_rf_selected_grid = ConfusionMatrixDisplay(confusion_matrix=cm_rf_selected_grid, 
                                               display_labels=rf.classes_)
disp_rf_selected_grid.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix Random Forest with selected Features')
plt.show()


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


np.unique(y_train_encoded)


xgb = XGBClassifier(random_state=42)

xgb.fit(X_train, y_train_encoded)


y_pred_xgb = xgb.predict(X_test)

print('Accuracy Score for XGBoost: ', accuracy_score(y_test_encoded, y_pred_xgb))


print(classification_report(y_test_encoded, y_pred_xgb))


cm_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)

disp_xgb  = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb.classes_)
disp_xgb.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix XGBoost')
plt.show()


param_grid_xgb = {
    'n_estimators': [200, 300],
    'max_depth': [5, 7],
    'eta': [0.05, 0.1],
    'subsample': [0.8, 1.],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [1, 3],
    'scale_pos_weight': [1, 3, 5],
}


grid_search_xgb = GridSearchCV(xgb, param_grid=param_grid_xgb, cv=5,
                               scoring='f1_weighted')

grid_search_xgb.fit(X_train, y_train_encoded)


print(grid_search_xgb.best_params_)


best_model_xgb = grid_search_xgb.best_estimator_
best_model_xgb


y_pred_xgb_grid = best_model_xgb.predict(X_test)

print("Test Accuracy:", best_model_xgb.score(X_test, y_test_encoded))

print('Accuracy Score for XGBoost: ', accuracy_score(y_test_encoded, y_pred_xgb_grid))


print(classification_report(y_test_encoded, y_pred_xgb_grid))


y_pred_xgb_grid_original = label_encoder.inverse_transform(y_pred_xgb_grid)


cm_xgb_grid = confusion_matrix(y_test, y_pred_xgb_grid_original)

disp_xgb_grid  = ConfusionMatrixDisplay(confusion_matrix=cm_xgb_grid, display_labels=label_encoder.classes_)
disp_xgb_grid.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix XGBoost GridSearch')
plt.show()


feature_importances_xgb = pd.DataFrame(
{    'Feature': X_train.columns,
    'Importances': xgb.feature_importances_}
).sort_values(by='Importances', ascending= False)

print(feature_importances)


plt.figure(figsize=(10,6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances XGBoost')
plt.gca().invert_yaxis()
plt.show()


X_train_selected_xgb = X_train.drop(['free sulfur dioxide', 'pH', 'fixed acidity', 'residual sugar', 'citric acid',
                                     'total sulfur dioxide', 'volatile acidity', 'density',], axis=1)
X_test_selected_xgb = X_test.drop(['free sulfur dioxide', 'pH', 'fixed acidity', 'residual sugar', 'citric acid',
                                   'total sulfur dioxide', 'volatile acidity', 'density',], axis=1)


grid_search_xgb.fit(X_train_selected_xgb, y_train_encoded)
print(grid_search_xgb.best_params_)


best_model_xgb_selected = grid_search_xgb.best_estimator_
best_model_xgb_selected


best_model_xgb_selected.fit(X_train_selected_xgb, y_train_encoded)

y_pred_xgb_selected = best_model_xgb_selected.predict(X_test_selected_xgb)

print("Test Accuracy:", best_model_xgb_selected.score(X_test_selected_xgb, 
                                                      y_test_encoded))

print('Accuracy Score for XGBoost with Feature Importance: ', 
      accuracy_score(y_test_encoded, y_pred_xgb_selected))


print(classification_report(y_test_encoded, y_pred_xgb_selected))


y_pred_xgb_original = label_encoder.inverse_transform(y_pred_xgb_selected)


cm_xgb_selected = confusion_matrix(y_test, y_pred_xgb_original)

disp_xgb_selected = ConfusionMatrixDisplay(confusion_matrix=cm_xgb_selected, display_labels=label_encoder.classes_)
disp_xgb_selected.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix XGBoost with selected Features')
plt.show()


predictions_xgb_grid = best_model_xgb.predict(test_dataset_copy)
predictions = label_encoder.inverse_transform(predictions_xgb_grid)


submission = pd.DataFrame({"Id": test_dataset["Id"], "Quality": predictions})
print(submission)


submission.to_csv('/kaggle/working/submission.csv')

