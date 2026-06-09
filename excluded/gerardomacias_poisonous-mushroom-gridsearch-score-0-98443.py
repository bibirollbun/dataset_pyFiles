from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
#from sklearn.tree import plot_tree

from sklearn.metrics import matthews_corrcoef, accuracy_score, classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
train_df.head(10)


train_df.shape, test_df.shape


train_df.info()


test_df.info()


train_df.isna().sum()


test_df.isna().sum()


train_df.describe().T


sample_train_df = train_df.sample(200000) # just a few sample to improve some tols


corr_matrix = sample_train_df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación lineal (Pearson)")
plt.show()


colors = ['b','r','g']
head = ['cap-diameter', 'stem-height', 'stem-width']

fig, axes = plt.subplots(1,3, figsize=(15,4))

if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]


for i, h in enumerate(head):
    sns.histplot(sample_train_df[h], kde=True, ax=axes[i], color=colors[i])
    axes[i].set_title(h)
    
plt.tight_layout()
plt.show()


stem_color = sample_train_df['stem-color'].value_counts()
plt.figure(figsize=(14,8))
plt.pie(stem_color, labels=stem_color.index, autopct='%1.1f%%')
plt.title(" Distribution of stem_color")
plt.show()


stem_classes = sample_train_df['class'].value_counts()
plt.figure(figsize=(14,8))
plt.pie(stem_classes, labels=stem_classes.index, autopct='%1.1f%%')
plt.title('Classes Distribution')
plt.show()


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


le = LabelEncoder()
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


missing_values_intrain = sample_train_df.isnull().mean() * 100
missing_values_intrain


print("Columns in train_df with more tha 10% missing values")
missing_values_intrain[missing_values_intrain > 10]


columns_drop = missing_values_intrain[missing_values_intrain > 95].index
sample_train_df = sample_train_df.drop(columns=columns_drop)
sample_train_df.head()


columns_train = sample_train_df.select_dtypes(include=['object']).columns
columns_train = columns_train[columns_train != 'class']
#ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
sample_train_df[columns_train] = oe.fit_transform(sample_train_df[columns_train].astype(str))


columns_train


sample_train_df.head()


sample_train_df = sample_train_df.drop(columns=['id'])
sample_train_df.head()


sample_train_df['class'] = le.fit_transform(sample_train_df['class'])
sample_train_df.head()


X = sample_train_df.drop(columns=['class'])
y = sample_train_df['class']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("model", None)
])

param_grid = [{
    "model":[DecisionTreeClassifier()],
    "model__criterion":["gini", "entropy"],
    "model__max_depth":[10,15,20,25,None],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf":[1,2,4,6,8],
    "model__max_features":["sqrt", "log2", None],
    "model__class_weight": [None, "balanced"]
},
              {
    "model":[XGBClassifier(eval_metric="logloss")],
    "model__learning_rate":[0.001,0.01,0.1],
    "model__n_estimators":[100,200],
    "model__max_depth":[5,10,15,20,25],
    "model__subsample":[0.8, 1.0],
    "model__colsample_bytree":[0.8, 1.0]
}
]

grid_search = GridSearchCV(
    estimator = pipe, 
    param_grid = param_grid, 
    cv=kf, 
    scoring="matthews_corrcoef",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)


print(f"Best Model: {grid_search.best_estimator_.named_steps['model']}")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best MCC: {grid_search.best_score_}")


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mcc_test = matthews_corrcoef(y_test, y_pred)
print(f"MCC score test: {mcc_test:.4f}")


best_model = grid_search.best_estimator_.named_steps["model"]
importances = best_model.feature_importances_


from xgboost import plot_tree

plt.figure(figsize=(20,20))
plot_tree(best_model, num_trees=0)
plt.show()


!pip install graphviz


import graphviz

booster = best_model.get_booster()
booster.feature_names = X_train.columns.tolist()
dot_data = booster[0].get_dump(with_stats=True, dump_format="dot")[0]
graph = graphviz.Source(dot_data)
graph.render("xgb_tree", format="png")

# show tree
from IPython.display import Image
Image("xgb_tree.png")


train_df_copy = train_df.copy()
test_df_copy = test_df.copy()


columns_train_full = train_df_copy.select_dtypes(include=['object']).columns
columns_train_full = columns_train_full[columns_train_full != 'class']
columns_test_full = test_df_copy.select_dtypes(include=['object']).columns


ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).set_output(transform='pandas')

train_df_copy[columns_train_full] = ordinal_encoder.fit_transform(train_df_copy[columns_train_full].astype(str))
test_df_copy[columns_test_full] = ordinal_encoder.transform(test_df_copy[columns_test_full].astype(str))


train_df_copy = train_df_copy.drop(columns=['id'])
test_df_copy = test_df_copy.drop(columns=['id'])


train_df_copy.head()


test_df_copy.head()


train_df_copy['class'] = LabelEncoder().fit_transform(train_df_copy['class'])
train_df_copy.head()


X_full = train_df_copy.drop(columns=['class'])
y_full = train_df_copy['class']


final_model = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=25,
    n_estimators=100,
    subsample=0.8
)


final_model.fit(X_full, y_full)


y_pred = final_model.predict(test_df_copy)


y_pred_class = ['e' if val == 0 else 'p' for val in y_pred]


df_submission = pd.DataFrame({
    'id':test_df['id'],
    'class': y_pred_class
})


df_submission


df_submission.to_csv("submission_1", index=False)

