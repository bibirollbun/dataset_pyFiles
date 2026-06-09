import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/kaggle/input/datascience-4-competition/train.csv")
test_df = pd.read_csv("/kaggle/input/datascience-4-competition/test.csv")


df.head()


df.shape


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df, palette='viridis')
plt.title("Class Distribution (Labels 0–10)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


feature_cols = [f"feature{i}" for i in range(64)]

# Get frequency with normalization (mean)
feature_freq = df[feature_cols].mean()

# Plot
plt.figure(figsize=(14, 5))
sns.barplot(x=feature_freq.index, y=feature_freq.values, palette="Blues_d")
plt.xticks(rotation=90)
plt.ylabel("Proportion of 1s")
plt.title("Frequency of 1s in Each Feature (Sorted)")
plt.tight_layout()
plt.show()


from sklearn.feature_selection import mutual_info_classif

X = df.drop(columns=['ID', 'label'])  # Features 
y = df['label']  # Target

# Compute mutual information
mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)

mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Plot top features
plt.figure(figsize=(10, 10))
sns.barplot(x=mi_series.values, y=mi_series.index, palette='viridis')
plt.title('Mutual Information Scores of Features')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 2D 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="tab10", s=60)
plt.title("PCA: 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Class")
plt.grid(True)
plt.show()



import umap.umap_ as umap

# UMAP 2D 
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="tab10", s=60)
plt.title("UMAP: 2D Projection")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Class")
plt.grid(True)
plt.show()



# UMAP 3D
reducer_3d = umap.UMAP(n_components=3, random_state=42)
X_umap_3d = reducer_3d.fit_transform(X_scaled)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y, cmap='tab10', s=50)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
plt.title("UMAP 3D Projection")
plt.show()



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_umap_3d)

df_umap = pd.DataFrame({'cluster': clusters, 'label': y})
class_counts_per_cluster = df_umap.groupby(['cluster', 'label']).size().unstack(fill_value=0)

print(class_counts_per_cluster)



from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

k_values = range(5, 65, 5)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

def mutual_info_fixed(X, y):
        return mutual_info_classif(X, y, random_state=42)
    
for k in k_values:
    accuracies = []
    
    selector = SelectKBest(score_func=mutual_info_fixed, k=k)
    X_selected = selector.fit_transform(X, y)


    for train_idx, val_idx in cv.split(X_selected, y):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    results[k] = mean_acc
    print(f"k = {k}: Mean CV Accuracy = {mean_acc:.4f}")

best_k = max(results, key=results.get)
print(f"\nBest k = {best_k} with accuracy = {results[best_k]:.4f}")



from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Define a custom scoring function with a fixed random_state
def mutual_info_fixed(X, y):
    return mutual_info_classif(X, y, random_state=42)

selector = SelectKBest(score_func=mutual_info_fixed, k=best_k)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]


len(selected_features)


X = df[selected_features]
y = df['label']

X_test = test_df[selected_features]  


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def KFoldValidate(model, X, y):
    # Set up Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # metrics
    acc_scores, f1_scores, precision_scores, recall_scores = [], [], [], []
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # model 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
        precision_scores.append(precision_score(y_val, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_val, y_pred, average='weighted'))
    

    cv_results = pd.DataFrame({
        'Fold': list(range(1, 6)),
        'Accuracy': acc_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores
    })

    cv_results.loc['Average'] = cv_results.mean(numeric_only=True)
    
    return cv_results



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr.fit(X_train, y_train)

y_pred_val = lr.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))


result = KFoldValidate(lr, X, y)
print(result.round(3))


model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)

model.fit(X, y)
y_pred_test = model.predict(X_test)

submission = pd.read_csv('/kaggle/input/datascience-4-competition/sample_submission.csv')  
submission['label'] = y_pred_test
submission.to_csv('submission_logistic.csv', index=False)


from sklearn.svm import SVC

svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train, y_train)

# Predict on validation set
y_pred_val = svm.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(svm, X, y)
print(result.round(3))


model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')

model.fit(X, y)
y_pred_test = model.predict(X_test)
  
submission['label'] = y_pred_test
submission.to_csv('submission_svm(rbf).csv', index=False)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


param_grid = {
    'C': [0.1, 1, 10],  
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  
    'class_weight': [None, 'balanced'],  
    'probability': [True, False],  
    'shrinking': [True, False],  
    'decision_function_shape': ['ovr', 'ovo'], 
}


svc = SVC(kernel='rbf', random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    svc,
    param_grid,
    cv=5,
    scoring='accuracy',  
    n_jobs=-1,
    verbose=2
)


grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)



from sklearn.svm import SVC

svm = SVC(C= 0.1, coef0=5, degree=2, gamma='scale', kernel='poly', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train, y_train)

# Predict on validation set
y_pred_val = svm.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(svm, X, y)
print(result.round(3))


model = SVC(C= 0.1, coef0=5, degree=2, gamma='scale', kernel='poly', probability=True, random_state=42, class_weight='balanced')

model.fit(X, y)
y_pred_test = model.predict(X_test)

submission['label'] = y_pred_test
submission.to_csv('submission_svm(poly).csv', index=False)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'kernel': ['poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 0.01, 0.1],
    'coef0': [0, 1, 5],
    'C': [0.1, 1, 10]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_weighted')
grid.fit(X, y)

print("Best parameters:", grid.best_params_)



from sklearn.svm import SVC

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predict on validation set
y_pred_val = svm.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(svm, X, y)
print(result.round(3))


model = SVC(kernel='linear', probability=True, random_state=42)

model.fit(X, y)
y_pred_test = model.predict(X_test)

submission['label'] = y_pred_test
submission.to_csv('submission_svm(linear).csv', index=False)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}


rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Run grid search
grid_search.fit(X, y)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", round(grid_search.best_score_, 4))

# Best model
best_rf = grid_search.best_estimator_


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', max_depth=10, max_features='log2', 
                            min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict on validation set
y_pred_val = rf.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(rf, X, y)
print(result.round(3))


model = RandomForestClassifier(class_weight='balanced', max_depth=10, max_features='log2', 
                            min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42)
model.fit(X, y)
y_pred_test = model.predict(X_test)

submission['label'] = y_pred_test
submission.to_csv('submission_rf.csv', index=False)



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'criterion' : ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    estimator=ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)



et = ExtraTreesClassifier(criterion='entropy', random_state=42, class_weight='balanced', max_depth=10, max_features='log2',
                         min_samples_leaf=2, min_samples_split=5)
et.fit(X_train, y_train)

y_pred_val = et.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(et, X, y)
print(result.round(3))


model = ExtraTreesClassifier(criterion='entropy', random_state=42, class_weight='balanced', max_depth=10, max_features='log2',
                         min_samples_leaf=2, min_samples_split=5)
model.fit(X, y)
y_pred_test = model.predict(X_test)

submission['label'] = y_pred_test
submission.to_csv('submission_extra_trees.csv', index=False)



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Model
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=11,
    eval_metric='mlogloss',
    use_label_encoder=False,
    verbosity=1,
    tree_method='hist', 
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 5],
    'gamma': [0, 1]
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X, y)

# Results
print("Best Parameters:", grid_search.best_params_)
print("Best Score (Accuracy):", grid_search.best_score_)



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

xgb = XGBClassifier(objective='multi:softprob', gamma=1, learning_rate=0.05,  subsample=0.8, min_child_weight=5, max_depth=6, 
                    num_class=11, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Compute balanced sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb.fit(X_train, y_train)


# Predict on validation set
y_pred_val = xgb.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(xgb, X, y)
print(result.round(3))



model = XGBClassifier(objective='multi:softprob', gamma=1, learning_rate=0.05,  subsample=0.8, min_child_weight=5, max_depth=6, 
                      num_class=11, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Compute balanced sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

model.fit(X, y)
y_pred_test = model.predict(X_test)

# save submission
submission['label'] = y_pred_test
submission.to_csv('submission_xgb.csv', index=False)



from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

catb = CatBoostClassifier(
    iterations=300,                
    learning_rate=0.01,            
    depth=5,                      
    l2_leaf_reg=3,                 
    random_strength=1.5,          
    loss_function='MultiClass',
    eval_metric='Accuracy',      
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0
)


catb.fit(X_train, y_train)


# Predict on validation set
y_pred_val = catb.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))



result = KFoldValidate(catb, X, y)
print(result.round(3))


model = CatBoostClassifier(
    iterations=300,                
    learning_rate=0.05,            
    depth=5,                      
    l2_leaf_reg=3,                 
    random_strength=1.5,          
    loss_function='MultiClass',
    eval_metric='Accuracy',      
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0
)


model.fit(X, y)
y_pred_test = model.predict(X_test)

submission['label'] = y_pred_test
submission.to_csv('submission_catboost.csv', index=False)


