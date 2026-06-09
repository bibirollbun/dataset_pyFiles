import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import math

# ML models
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


train.info()


test.info()


train['ratio_going_out_vs_spend_alone'] = train['Going_outside']/(train['Time_spent_Alone'])
test['ratio_going_out_vs_spend_alone'] = test['Going_outside']/(test['Time_spent_Alone'])


X_train = train.drop(columns = ['id','Personality'])
Y_train = train['Personality']

X_test = test.drop(columns=['id'])


cat_cols = []
cont_cols = []
for cols in X_train.columns:
    if X_train[cols].dtype == 'object':
        cat_cols.append(cols)
    else:
        cont_cols.append(cols)

print(f"Categorical columns = {cat_cols}")
print(f"Continuous columns = {cont_cols}")


X_train.isnull().sum().sort_values(ascending=False)/len(X_train)


X_test.isnull().sum().sort_values(ascending=False)/len(X_test)


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.heatmap(X_train.isnull(), cbar=False)
plt.subplot(1,2,2)
sns.heatmap(X_test.isnull(),cbar=False)
plt.show()


train['Personality'].value_counts(normalize=False).plot(kind='bar', title='Target Distribution')



print("For Train Dataset : Numerical columns distribution")
cols = cont_cols  # your continuous columns
n_cols = 3         # plots per row
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(X_train[col], kde=True, ax=axes[i])
    axes[i].set_title(f'{col} Distribution')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


print("For Test Dataset : Numerical columns distribution")
cols = cont_cols  # your continuous columns
n_cols = 3         # plots per row
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(X_test[col], kde=True, ax=axes[i])
    axes[i].set_title(f'{col} Distribution')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



cols = cont_cols  # your continuous columns
n_cols = 3
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(x='Personality', y=col, data=train, ax=axes[i])
    axes[i].set_title(f'{col} vs Personality')

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


sns.heatmap(train[cont_cols].corr(), annot=True, cmap='coolwarm')


n_cols = 2
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, hue='Personality', data=train, ax=axes[i])
    axes[i].set_title(f'{col} vs Personality')
    axes[i].tick_params(axis='x', rotation=45)

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


train.drop(columns=['id']).groupby('Personality')[cont_cols].describe().T



n_cols = 2
n_rows = math.ceil(len(cont_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
axes = axes.flatten()

for i, feature in enumerate(cont_cols):
    sns.kdeplot(data=train, x=feature, hue='Personality', ax=axes[i], fill=True, common_norm=False, alpha=0.5)
    axes[i].set_title(f'PDF of {feature} by Personality')

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]

for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna('nan')

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Define CatBoost pool (optional but recommended for speed with categorical features)
train_pool = Pool(x_train, y_train, cat_features=cat_feature_indices)
val_pool = Pool(x_val, y_val, cat_features=cat_feature_indices)



# Suppose y_train is your binary label array (e.g., 'Introvert'=0, 'Extrovert'=1)
from collections import Counter
counter = Counter(y_train)
total = sum(counter.values())
class_weights = [total / counter[c] for c in sorted(counter.keys())]
print(f"Class weights :{class_weights}")

# Initialize model with GPU
model = CatBoostClassifier(
    loss_function='Logloss',
    class_weights=class_weights,
    eval_metric='Accuracy',
    task_type='GPU',  # if you're using GPU
    verbose=100
)


# Train the model
model.fit(x_train, y_train, eval_set=(x_val, y_val), cat_features=cat_feature_indices)

# Predict and evaluate
y_pred = model.predict(x_val)
print(f"Accuracy Score = {accuracy_score(y_val,y_pred)}")
print(classification_report(y_val, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")



importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns

# Create DataFrame for better readability
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.title('CatBoost Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


new_X_train = X_train.drop(columns = ['Stage_fear','Going_outside','Drained_after_socializing'])


cat_features = new_X_train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_feature_indices = [new_X_train.columns.get_loc(col) for col in cat_features]

for col in cat_features:
    new_X_train[col] = new_X_train[col].astype(str).fillna('nan')

x_train, x_val, y_train, y_val = train_test_split(new_X_train, Y_train, test_size=0.2, random_state=42)

# Define CatBoost pool (optional but recommended for speed with categorical features)
train_pool = Pool(x_train, y_train, cat_features=cat_feature_indices)
val_pool = Pool(x_val, y_val, cat_features=cat_feature_indices)



# Suppose y_train is your binary label array (e.g., 'Introvert'=0, 'Extrovert'=1)
from collections import Counter
counter = Counter(y_train)
total = sum(counter.values())
class_weights = [total / counter[c] for c in sorted(counter.keys())]
print(f"Class weights :{class_weights}")

# Initialize model with GPU
model2 = CatBoostClassifier(
    loss_function='Logloss',
    class_weights=class_weights,
    eval_metric='Accuracy',
    task_type='GPU',  # if you're using GPU
    verbose=100
)


# Train the model
model2.fit(x_train, y_train, eval_set=(x_val, y_val), cat_features=cat_feature_indices)

# Predict and evaluate
y_pred = model2.predict(x_val)
print(f"Accuracy Score = {accuracy_score(y_val,y_pred)}")
print(classification_report(y_val, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")



importances = model2.get_feature_importance(train_pool)
feature_names = new_X_train.columns

# Create DataFrame for better readability
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.title('CatBoost Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


X_train_xgb = X_train.copy()

X_train_xgb['ratio_going_out_vs_spend_alone'].replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_xgb[cat_cols] = X_train_xgb[cat_cols].fillna('missing')

ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_encoded = ord_enc.fit_transform(X_train_xgb[cat_cols])

X_cont = X_train_xgb[cont_cols] 
X_all = np.hstack([X_cont.values, X_cat_encoded])

le = LabelEncoder()
y = le.fit_transform(train['Personality'])

x_train, x_val, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

model_xgb = XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    objective='binary:logistic',     # log loss
    eval_metric=['logloss', 'error'],             # accuracy
    use_label_encoder=False,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42
)

model_xgb.fit(x_train, y_train)


y_pred = model_xgb.predict(x_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print(classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred, labels=model_xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_xgb.classes_)
disp.plot(cmap="Blues")


# plot_importance(model_xgb, importance_type='gain', title='Top 20 Important Features')
# plt.tight_layout()
# plt.show()


from xgboost import plot_importance

feature_names = list(X_cont.columns) + list(ord_enc.get_feature_names_out(cat_cols))

plot_importance(
    model_xgb,
    importance_type='gain',
    title='Top 20 Important Features',
    max_num_features=20,
    xlabel='Gain',
    height=0.5,
    show_values=False
)

ax = plt.gca()
keys = list(model_xgb.get_booster().get_score(importance_type='gain').keys())
ax.set_yticklabels([feature_names[int(k[1:])] for k in keys])  # 'f0' â†’ 0, 'f12' â†’ 12
plt.tight_layout()
plt.show()



X_train_xgb_new = X_train[['Stage_fear','Drained_after_socializing']]
X_train_xgb_new


X_train_xgb_new[cat_cols] = X_train_xgb[cat_cols].fillna('missing')

ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_encode_new = ord_enc.fit_transform(X_train_xgb_new[cat_cols])

le = LabelEncoder()
y = le.fit_transform(train['Personality'])

x_train, x_val, y_train, y_val = train_test_split(
    X_cat_encode_new, y, test_size=0.2, stratify=y, random_state=42
)

model_xgb = XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    objective='binary:logistic',     # log loss
    eval_metric=['logloss', 'error'],             # accuracy
    use_label_encoder=False,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42
)

model_xgb.fit(x_train, y_train)


y_pred = model_xgb.predict(x_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print(classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred, labels=model_xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_xgb.classes_)
disp.plot(cmap="Blues")


x_train.shape


# import optuna
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
#         "tree_method": "gpu_hist",  # or "hist" if no GPU
#         "predictor": "gpu_predictor",
#         "objective": "binary:logistic",
#         "eval_metric": "logloss",
#         "use_label_encoder": False,
#         "random_state": 42,
#     }

#     model_hyp = XGBClassifier(**params)
#     scores = cross_val_score(model_hyp, X_cat_encode_new, y, cv=3, scoring="accuracy")
#     return scores.mean()



import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        "tree_method": "gpu_hist",  # or "hist" if no GPU
        "predictor": "gpu_predictor",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
    }

    model_hyp = XGBClassifier(**params)
    scores = cross_val_score(model_hyp, X_all, y, cv=3, scoring="accuracy")
    return scores.mean()



study = optuna.create_study(direction="maximize", study_name="xgb_tuning")
study.optimize(objective, n_trials=50)


print("Best trial:")
print(f"  Accuracy: {study.best_value:.4f}")
print("  Params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")


final_model = XGBClassifier(
    **study.best_params,
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

final_model.fit(x_train, y_train)

y_pred = final_model.predict(x_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print(classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred, labels=final_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
disp.plot(cmap="Blues")


X_test = test[X_train_xgb.columns]
X_test['ratio_going_out_vs_spend_alone'].replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_encoded = ord_enc.fit_transform(X_test[cat_cols])
X_test_cont = X_test[cont_cols] 
X_all_test = np.hstack([X_test_cont.values, X_test_encoded])
predictions = final_model.predict(X_all_test)
test_preds = le.inverse_transform(predictions)



# X_test = test[X_train_xgb_new.columns]
# X_test_encoded = ord_enc.fit_transform(X_test[cat_cols])
# predictionsx = model_xgb.predict(X_test_encoded)
# test_preds = le.inverse_transform(predictionsx)


# X_test = test[X_train_xgb_new.columns]
# X_test_encoded = ord_enc.fit_transform(X_test[cat_cols])
# predictions = final_model.predict(X_test_encoded)
# test_preds = le.inverse_transform(predictions)


submission = pd.DataFrame({
    'id': test.id,  # Adjust if your test IDs start from 18524
    'Personality': test_preds
})
submission.head()


submission.to_csv('submission.csv', index=False)




