import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix


%%time 
train_transactions=pd.read_csv('../input/train_transaction.csv')
train_identity=pd.read_csv('../input/train_identity.csv')
print('Train data set is loaded !')


train_transactions.head()


train_transactions.info()


train_identity.info()


x=train_transactions['isFraud'].value_counts().values
sns.barplot([0,1],x)
plt.title('Target variable count')



train=train_transactions.merge(train_identity,how='left',left_index=True,right_index=True)
y_train=train['isFraud'].astype('uint8')
print('Train shape',train.shape)



del train_transactions,train_identity

print("Data set merged ")






%%time
# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage2(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df






%%time
train = reduce_mem_usage2(train)





X_train,X_test,y_train,y_test=train_test_split(train.drop('isFraud',axis=1),y_train,test_size=.2,random_state=1)


X=pd.concat([X_train,y_train],axis=1)


not_fraud=X[X.isFraud==0]
fraud=X[X.isFraud==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
upsampled.isFraud.value_counts()




y=upsampled.isFraud.value_counts()
sns.barplot(y=y,x=[0,1])
plt.title('upsampled data class count')
plt.ylabel('count')


not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

# checking counts
downsampled.isFraud.value_counts()


y=downsampled.isFraud.value_counts()
sns.barplot(y=y,x=[0,1])
plt.title('downsampled data class count')
plt.ylabel('count')


from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=1000, random_state=10
)

df = pd.DataFrame(X)
df['target'] = y
df.target.value_counts().plot(kind='bar', title='Count (target)')


def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=1)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    prob=lr.predict_proba(X_test)
    return (prob[:,1],y_test)


probs,y_test=logistic(X,y)


def plot_pre_curve(y_test,probs):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("precision recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the plot
    plt.show()
    
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc(y_test, probs):
    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {auc:.4f}")
    
    # Plot no skill line (diagonal line)
    plt.plot([0, 1], [0, 1], linestyle='--')
    
    # Plot the ROC curve for the model
    plt.plot(fpr, tpr, marker='.')
    
    # Title and labels
    plt.title("ROC curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # Show the plot
    plt.show()


plot_pre_curve(y_test,probs)


plot_roc(y_test,probs)


def plot_2d_space(X_train, y_train,X=X,y=y ,label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(8,4))
   
    for l, c, m in zip(np.unique(y), colors, markers):
        ax1.scatter(
            X_train[y_train==l, 0],
            X_train[y_train==l, 1],
            c=c, label=l, marker=m
        )
    for l, c, m in zip(np.unique(y), colors, markers):
        ax2.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
   
    ax1.set_title(label)
    ax2.set_title('original data')
    plt.legend(loc='upper right')
    plt.show()




# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])

# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()



import imblearn


from imblearn.under_sampling import RandomUnderSampler

ran=RandomUnderSampler(return_indices=True) ##intialize to return indices of dropped rows
X_rs,y_rs,dropped = ran.fit_sample(X,y)

print("The number of removed indices are ",len(dropped))
plot_2d_space(X_rs,y_rs,X,y,'Random under sampling')



probs,y_test=logistic(X_rs,y_rs)
plot_pre_curve(y_test,probs)


plot_roc(y_test,probs)


from imblearn.over_sampling import RandomOverSampler

ran=RandomOverSampler()
X_ran,y_ran= ran.fit_resample(X,y)

print('The new data contains {} rows '.format(X_ran.shape[0]))

plot_2d_space(X_ran,y_ran,X,y,'over-sampled')



probs,y_test=logistic(X_ran,y_ran)
plot_pre_curve(y_test,probs)


plot_roc(y_test,probs)


from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

#print('Removed indexes:', id_tl)

plot_2d_space(X_tl, y_tl,X,y, 'Tomek links under-sampling')


probs,y_test=logistic(X_tl,y_tl)
plot_pre_curve(y_test,probs)


plot_roc(y_test,probs)




from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm,X,y, 'SMOTE over-sampling')




X_sm.shape


probs,y_test=logistic(X_sm,y_sm)
plot_pre_curve(y_test,probs)


plot_roc(y_test,probs)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, precision_recall_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier


# Function to train a model and return prediction probabilities and true labels
def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)
    return probs, preds, y_test


# Dictionary of models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'SVM (probabilistic)': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}


# # Function to evaluate and collect all results
# def evaluate_models(X, y):
#     results = {}
#     for name, model in models.items():
#         print(f"Training {name}...")
#         probs, preds, y_test = train_model(model, X, y)
#         results[name] = {
#             'probs': probs,
#             'preds': preds,
#             'y_test': y_test,
#             'roc_auc': roc_auc_score(y_test, probs),
#             'report': classification_report(y_test, preds, output_dict=True),
#             'confusion_matrix': confusion_matrix(y_test, preds)
#         }
#     return results


# # Function to plot ROC curves
# def plot_roc_curves(results):
#     plt.figure(figsize=(10,8))
#     for name, res in results.items():
#         fpr, tpr, _ = roc_curve(res['y_test'], res['probs'])
#         plt.plot(fpr, tpr, label=f'{name} (AUC = {res["roc_auc"]:.2f})')
#     plt.plot([0,1],[0,1],'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves')
#     plt.legend()
#     plt.grid()
#     plt.show()


# # Function to plot confusion matrices
# def plot_confusion_matrices(results):
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))
#     axes = axes.flatten()
#     for idx, (name, res) in enumerate(results.items()):
#         sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
#         axes[idx].set_title(name)
#         axes[idx].set_xlabel('Predicted')
#         axes[idx].set_ylabel('True')
#     plt.tight_layout()
#     plt.show()


import os


# # Function to train and evaluate an ensemble model
# def train_ensemble(X, y):
#     estimators = [
#         ('lr', LogisticRegression(max_iter=1000)),
#         ('rf', RandomForestClassifier(n_estimators=100)),
#         ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
#         ('lgbm', LGBMClassifier())
#     ]
#     ensemble = VotingClassifier(estimators=estimators, voting='soft')
#     probs, preds, y_test = train_model(ensemble, X, y)
#     roc_auc = roc_auc_score(y_test, probs)
#     print(f"Ensemble Model ROC AUC: {roc_auc:.2f}")
#     return ensemble, probs, preds, y_test


from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score, f1_score


def evaluate_and_plot_model(model_name, probs, preds, y_test, save_dir):
    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Specificity
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    # Store metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Specificity': specificity
    }
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{model_name}_ROC_Curve.png"))
    plt.close()
    
    # Plot PR Curve
    plt.figure(figsize=(6,5))
    plt.plot(recall_curve, precision_curve, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{model_name}_PR_Curve.png"))
    plt.close()
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(save_dir, f"{model_name}_Confusion_Matrix.png"))
    plt.close()
    
    return metrics, fpr, tpr

def train_and_evaluate_all_models(X, y, save_dir='model_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_metrics = {}
    roc_data = {}
    model_outputs = {}  # <-- NEW: to collect (probs, y_test)

    for name, model in models.items():
        print("="*70)
        print(f"Training and Evaluating: {name}")
        print("="*70)
        
        probs, preds, y_test = train_model(model, X, y)
        
        # Save outputs
        model_outputs[name] = (probs, y_test)

        metrics, fpr, tpr = evaluate_and_plot_model(name, probs, preds, y_test, save_dir)
        model_metrics[name] = metrics
        roc_data[name] = (fpr, tpr, metrics['ROC AUC'])
        
        print(f"Metrics for {name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\n")
    
    # After all models, plot combined ROC
    plt.figure(figsize=(10,8))
    for name, (fpr, tpr, roc_auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_val:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "Combined_ROC_Curves.png"))
    plt.close()
    
    # Make metrics into a DataFrame
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, "model_metrics_summary.csv"))
    
    return metrics_df, model_outputs


# Assuming you have already defined models

metrics_df, model_outputs = train_and_evaluate_all_models(X, y)



def custom_soft_ensemble(model_outputs, save_dir='model_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_tests = [y for _, y in model_outputs.values()]
    for y in y_tests[1:]:
        assert all(y == y_tests[0]), "Mismatch in y_test across models"
    y_test = y_tests[0]
    
    all_probs = np.stack([probs for probs, _ in model_outputs.values()], axis=0)
    avg_probs = np.mean(all_probs, axis=0)
    
    final_preds = (avg_probs > 0.5).astype(int)
    
    model_name = "Custom_Soft_Ensemble"
    metrics, fpr, tpr = evaluate_and_plot_model(model_name, avg_probs, final_preds, y_test, save_dir)
    
    print(f"Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    metrics_df = pd.DataFrame(metrics, index=[model_name])
    metrics_df.to_csv(os.path.join(save_dir, "custom_soft_ensemble_metrics.csv"))
    
    return metrics_df



ensemble_metrics_df = custom_soft_ensemble(model_outputs)










