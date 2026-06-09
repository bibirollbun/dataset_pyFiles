import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import plotly.graph_objects as go

TEST_PATH = "/kaggle/input/playground-series-s5e3/test.csv"
TRAIN_PATH = "/kaggle/input/playground-series-s5e3/train.csv"
SAMPLE_PATH = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_df = pd.read_csv(TRAIN_PATH, index_col='id')
test_df = pd.read_csv(TEST_PATH, index_col='id')

train_df.head()


test_df.head()


train_df.info()


test_df.info()


# This is simulator package which help to compare two dataframes or two csv files.
# simulator is able to return a summary of comparion of each column
!pip install git+https://github.com/vijaytakbhate2002/simulator.git


from simulator import CompareData

 # def compareCols(self, primary_key:str, first_df:pd.DataFrame=None, second_df:pd.DataFrame=None):
test_df['id'] = test_df.index

comparator = CompareData()
comparator.compareCols(primary_key="id", first_df=test_df, second_df=test_df.fillna(test_df.mean()))
test_df = test_df.drop(['id'], axis='columns')  
test_df = test_df.fillna(test_df.mean())


comparator.summary()


train_df.describe()


test_df.describe()


compare_params = ('count', 'mean', 'std', 'min', '50%', 'max')


def visualize_comparison(tuple1, tuple2, labels, comp_params):
    if len(tuple1) != len(tuple2):
        raise ValueError("Tuples must be of the same length")
    
    if labels is None:
        labels = [f'Item {i+1}' for i in range(len(tuple1))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=labels, y=tuple1, name='Train', marker_color='blue'))
    fig.add_trace(go.Bar(x=labels, y=tuple2, name='Test', marker_color='red'))
    
    fig.update_layout(
        title=f'Pairwise Comparison of ** {comp_params} ** Values',
        xaxis_title='Rain Parameters',
        yaxis_title='Values',
        barmode='group'
    )
    
    fig.show()

train_summary = train_df.describe().iloc[[0,1, 2, 3, 5, 7]].drop(['rainfall'], axis='columns').iterrows()
test_summary = test_df.describe().iloc[[0, 1, 2, 3, 5, 7]].iterrows()

for i, (train_data, test_data) in enumerate(zip(train_summary, test_summary)):
    train_dict, test_dict = dict(train_data[1]), dict(test_data[1])
    labels = tuple(test_dict.keys())
    train_vals, test_vals = tuple(train_dict.values()), tuple(test_dict.values())
    visualize_comparison(train_vals, test_vals, labels, compare_params[i])


def plot_pie_chart(values, labels):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title_text="Pie Chart")
    fig.show()

values = (train_df[train_df['rainfall'] == 0].shape[0], train_df[train_df['rainfall'] == 1].shape[0])
labels = ("No rainfall", "Rainfall")
plot_pie_chart(values, labels)


from imblearn.combine import SMOTETomek
X = train_df.drop(['rainfall'], axis='columns')
y = train_df['rainfall']

smote_tomek = SMOTETomek(random_state=42)
X, y = smote_tomek.fit_resample(X, y)

values = (y[y==0].shape[0], y[y==1].shape[0])
labels = ("No rainfall", "Rainfall")
plot_pie_chart(values, labels)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train.shape, y_test.shape


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    auc = roc_auc_score(y_test, y_pred_proba)
    TP = np.sum((y_pred == 1) & (y_test == 1))  
    FP = np.sum((y_pred == 1) & (y_test == 0))  
    FN = np.sum((y_pred == 0) & (y_test == 1))  
    TN = np.sum((y_pred == 0) & (y_test == 0))  

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    fpr = FP/(FP +TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"Model: {model_name}")
    print(f"FPR (False Positive Rate): {fpr}")
    print(f"TPR (True Positive Rate): {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("-" * 40)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)  
}

for model_name, model in models.items():
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name)


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
}

param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5]
    }
}

results = {}

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], 
                               scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'best_params': best_params,
        'best_cv_auc': best_score,
        'test_auc': test_auc
    }
    
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best CV AUC for {model_name}: {best_score}")
    print(f"Test AUC for {model_name}: {test_auc}")
    print("-" * 40)

for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best CV AUC: {result['best_cv_auc']}")
    print(f"Test AUC: {result['test_auc']}")
    print("-" * 40)



model = LGBMClassifier(learning_rate= 0.2, max_depth= 10, n_estimators= 300, num_leaves= 31)
model.fit(X_train, y_train)


from sklearn.metrics import roc_curve, auc
y_pred = model.predict(X_test)
y_true = y_test
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

# Create the ROC curve plot using Plotly
fig = go.Figure()

# Add ROC curve
fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f'ROC curve (AUC = {roc_auc:.2f})',
    line=dict(color='blue', width=2)
))

# Add diagonal line (random guess)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random Guess',
    line=dict(color='gray', dash='dash', width=1)
))

# Customize layout
fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate (FPR)',
    yaxis_title='True Positive Rate (TPR)',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    legend=dict(x=0.6, y=0.1),
    template='plotly_white'
)

fig.show()


rf = RandomForestClassifier(max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 300)
rf.fit(X, y)


lgbmc = LGBMClassifier(learning_rate= 0.2, max_depth= 10, n_estimators= 300, num_leaves= 31)
lgbmc.fit(X, y)


cb = CatBoostClassifier(learning_rate= 0.2, depth= 12, iterations= 300, l2_leaf_reg= 5)
cb.fit(X, y)


rainfall = cb.predict(test_df)
submissions = pd.DataFrame()
submissions['id'] = test_df.index
submissions['rainfall'] = rainfall 
submissions.head()


submissions.to_csv("Submissions_catboost_02_03_2025(with imbalance hadling).csv", index=False)




