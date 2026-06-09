import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)

import warnings
warnings.filterwarnings('ignore')

seed = 42

# Set Seaborn theme with dark grid
sns.set_theme(style="darkgrid", palette="Dark2", font_scale=0.8)

# Update matplotlib parameters for dark background and white labels
plt.rcParams.update({
    'axes.facecolor': '#222222',     # Dark gray plot background
    'figure.facecolor': '#222222',   # Dark gray around the figure
    'text.color': '#00FFFF',           # White text everywhere
    'axes.labelcolor': '#00FFFF',      # White axis labels
    'xtick.color': '#00FFFF',          # White x-axis tick labels
    'ytick.color': '#00FFFF',          # White y-axis tick labels
    'grid.color': '#444444',         # Slightly lighter grid
    'axes.edgecolor': 'white'        # White border of the plot
})


h2o.init()


# choice of dataset
dataset_dico = {
    'introvert-extrovert_external_1': 1,
    'introvert-extrovert_external_2': 2
    
}

pd.DataFrame(dataset_dico.keys(), dataset_dico.values(),  columns=['Problem'])


dataset = 546

include_ext = True
use_imputed = False
feat_eng = False


if dataset == 1:
    problem = 'extrovert vs introvertbehavior first external data'
    target = 'Personality'
    train_ = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')
    # train_[target]= train_[target].map({0: 'No', 1: 'Yes'})
    train_df = train_.sample(frac=0.8, random_state=seed)
    test_df = train_.iloc[[f for f in train_.index.tolist() if f not in train_df.index.tolist()], :]
    test_target = test_df.pop(target)
    subm_df = pd.DataFrame(test_df.index, columns=['id'])


elif dataset == 2:
    problem = 'extrovert vs introvertbehavior second external Data'
    target = 'Personality'
    train_ = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
    # train_[target]= train_[target].map({0: 'No', 1: 'Yes'})
    train_df = train_.sample(frac=0.8, random_state=seed)
    test_df = train_.iloc[[f for f in train_.index.tolist() if f not in train_df.index.tolist()], :]
    test_target = test_df.pop(target)
    subm_df = pd.DataFrame(test_df.index, columns=['id'])

else : # Work on competition datasets
    if use_imputed:
        problem = 'Competition: Introvert vs Extrovert using imputed data from a previous notebook'
        target = 'Personality'
        train_df = pd.read_csv('/kaggle/input/nan-imputed-datasets/imputed_train_data.csv')
        if include_ext:
            train_df = pd.concat([train_df, ext_1_df, ext_2_df], ignore_index=True)
        else:
            train_df = train_df
        test_df = pd.read_csv('/kaggle/input/nan-imputed-datasets/imputed_test_data.csv')
        subm_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
    
    if feat_eng:
        problem = 'Competition: Introvert vs Extrovert using imputed data from a previous notebook'
        target = 'Personality'
        train_df = pd.read_csv('/kaggle/input/features-eng-intro-vs-extrovert/train_eng.csv')
        if include_ext:
            train_df = pd.concat([train_df, ext_1_df, ext_2_df], ignore_index=True)
        else:
            train_df = train_df
        test_df = pd.read_csv('/kaggle/input/features-eng-intro-vs-extrovert/train_eng.csv')
        subm_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
        
    else:
        problem = 'Competition: Introvert vs Extrovert'
        target = 'Personality'
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv').drop(columns=['id'])
        ext_1_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
        ext_2_df = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')
        if include_ext:
            train_df = pd.concat([train_df, ext_1_df, ext_2_df], ignore_index=True)
        else:
            train_df = train_df
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv').drop(columns=['id'])
        subm_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(f'\nWe are working on {problem} classification'.upper())


X = train_df.copy()

le = LabelEncoder()
if X[target].dtype!='int':
    y = le.fit_transform(X.pop(target))
else:
    y = X.pop(target)

X.head()


# num_feats = test_df.select_dtypes(include='number').columns.tolist()
# cat_feats = test_df.select_dtypes(exclude='number').columns.tolist()


# # Define function for data preparation
# def df_preparator(df):
#     df = df.copy()
#     df[cat_feats] = df[cat_feats].fillna('missing')
#     df[num_feats] = df[num_feats].fillna(df[num_feats].mean())

#     df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1, 'missing': -1})
#     df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1, 'missing':-1})
    
#     df['Stage_vs_Drained'] = (df['Drained_after_socializing'] == df['Stage_fear'])*1
#     df['Stage_+_Drained'] = df['Drained_after_socializing'] + df['Stage_fear']
#     # df['Stage_or_Drained'] = df['Drained_after_socializing']== | df['Stage_fear']==1
#     # df['Stage_and_Drained'] = df['Drained_after_socializing']==1 & df['Stage_fear']==1
#     for feat_1 in num_feats:
#         # df[f'sin({feat_1})'] = np.sin(df[feat_1]*np.pi/4)
#         # df[f'cos({feat_1})'] = np.cos(df[feat_1]*np.pi/4)
#         for feat_2 in num_feats:
#             if feat_1 != feat_2:
#                 df[f'{feat_1}*{feat_2}'] = df[feat_1]*df[feat_2]
#                 df[f'{feat_1}/{feat_2}'] = np.clip(np.divide(df[feat_1], (df[feat_1] + df[feat_2])), 0, 10)
#     return df


num_feats = test_df.select_dtypes(include='number').columns.tolist()
encoder = LabelEncoder()


# Define function for data preparation
def df_preparator(df):
    # create a copy of the dataset
    df = df.copy()
    # fillna in cat_features
    df['Stage_fear'] = df['Stage_fear'].fillna('missing')
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('missing')
    df['Stage_Drained'] = df['Stage_fear']+ '_' + df['Drained_after_socializing']
    # # Binarize the cat_features
    # df['Stage_fear'] = df['Stage_fear']=='Yes'
    # df['Drained_after_socializing'] = df['Drained_after_socializing']=='Yes'
    # Encode the cat_features
    df['Stage_fear'] = encoder.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = encoder.fit_transform(df['Drained_after_socializing'])
    
    
    # for feat_1 in num_feats:
    #     # df[f'sin({feat_1})'] = np.sin(df[feat_1]*np.pi/4)
    #     # df[f'cos({feat_1})'] = np.cos(df[feat_1]*np.pi/4)
    #     for feat_2 in num_feats:
    #         if feat_1 != feat_2:
    #             df[f'{feat_1}*{feat_2}'] = df[feat_1]*df[feat_2]
    #             df[f'{feat_1}/{feat_2}'] = np.clip(np.divide(df[feat_1], (df[feat_1] + df[feat_2])), 0, 10)
    return df


train_df = df_preparator(train_df)
test_df = df_preparator(test_df)


train_df.shape


test_df.shape


tr_data = h2o.H2OFrame(train_df)
ts_data = h2o.H2OFrame(test_df)
tr_data.head(5)


eg_dg = train_df.copy()

eg_tr, eg_ts, eg_va = tr_data.split_frame(ratios = [0.7, 0.15])

[d.shape for d in [eg_tr, eg_ts, eg_va]]


tr_data.describe()


from h2o.frame import H2OFrame
with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
    pd_df = tr_data.as_data_frame()


aml = H2OAutoML(
    max_runtime_secs=600,
    max_models=30, 
    seed=seed, 
    nfolds=8,  
    stopping_rounds=6,
    # exclude_algos=['StackedEnsemble', 'DeepLearning'],
    verbosity='None',
)


%%time

aml.train(y=target, training_frame=tr_data)

# aml.train(X = X, y = y, training_frame= eg_tr, validation_frame = eg_va)


from h2o.explanation import explain

explain(aml.leader, tr_data, figsize=(8, 6), top_n_features=10)


lb = aml.leaderboard
print(f'\033[91m{lb}\n\033[0m')


best_model = aml.leader

print(f'\033[96m{best_model}\n\033[0m')


# Model performance metrics
try:
    perf = best_model.model_performance(tr_data)
    print(f"Model Accuracy: {perf.accuracy()[0][1]:.4f}")
except:
    pass


try:
    cm = perf.confusion_matrix()
    # 2. Print the confusion matrix
    print("Confusion Matrix on Training Data:")
    print(cm)
except:
    pass


h2o.save_model(model=best_model, path="best_model_path", force=True)


predictions = best_model.predict(ts_data)

prediction_df = predictions.as_data_frame()

prediction_df.head()


prediction_df.info()


sub_file = subm_df.copy()
sub_file[target] = prediction_df['predict'].values

sub_file.head()


from sklearn.metrics import accuracy_score, r2_score

try: # For classification problems
    test_score = accuracy_score(test_target, sub_file[target])
    # Set up the bar chart
    plt.figure(figsize=(8, .3))
    plt.barh(['Score'], [test_score], color='#00FFFF')
    plt.xlim(0, 1)
    plt.xticks([n/10 for n in range(0, 11)])
    plt.title('Accuracy', fontsize=14)
    plt.text(test_score + 0.01, 0, f'{test_score:.4f}', va='center', 
              weight='bold', fontsize=14)
    
    # Set background color
    plt.gca().set_facecolor('k')
    plt.gcf().patch.set_facecolor('darkgrey')  # Also set the figure background
    
    # Remove spines for cleaner look
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()

except Exception as e: # If it is not a classification problem
    # print('The test true values are not available')
    try:
        test_score = r2_score(test_target, sub_file[target])
        print(f'r2_score on {target.upper()} for test data: {test_score:.8f}')
    except Exception as e:
        print('We need to submit to the competion to get the score')



fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 3])

# Define the explode values for pie chart
n_classes = sub_file[target].value_counts()
explode = [0.05 for n in n_classes]

target_count = sub_file[target].value_counts()

ax0 = fig.add_subplot(gs[:, :-1])
ax0 = target_count.plot.bar(color=['#00ac74', '#8b4513', 'darkviolet', 'violet', 'lightgreen'])
for count in ax0.containers:
    ax0.bar_label(count, label_type='center', fmt='%d')
ax1 = fig.add_subplot(gs[:, 1:])
ax1 = target_count.plot.pie(autopct='%.2f%%',
                            shadow = True,
                            radius=1.28,
                            explode=explode,
                            startangle=270)
ax1 = pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.4, ax=ax1)
ax1.set_ylabel('')
plt.tight_layout()


sub_file.to_csv('submission.csv', index=False)
print('The submission file is ready!')


# >>> import h2o
# >>> from h2o.automl import H2OAutoML
# >>> h2o.init()
# >>> # Import a sample binary outcome train/test set into H2O
# >>> train = h2o.import_file("/kaggle/input/playground-series-s5e7/train.csv")
# >>> test = h2o.import_file("/kaggle/input/playground-series-s5e7/test.csv")
# >>> # Identify the response and set of predictors
# >>> y = "Personality"
# >>> x = list(train.columns)  #if x is defined as all columns except the response, then x is not required
# >>> x.remove(y)
# >>> # For binary classification, response should be a factor
# >>> train[y] = train[y].asfactor()
# eg_dg = train_df.copy()
# eg_tr, eg_ts, eg_va = train.split_frame(ratios = [0.7, 0.15])
# print(f'Shapes:{[d.shape for d in [eg_tr, eg_ts, eg_va]]}')
# # >>> test[y] = test[y].asfactor()
# >>> # Run AutoML for 30 seconds
# >>> aml = H2OAutoML(max_runtime_secs = 300, nfolds=6)
# >>> aml.train(x = x, y = y, training_frame = eg_tr, validation_frame = eg_va)
# >>> # Print Leaderboard (ranked by xval metrics)
# >>> aml.leaderboard
# >>> # (Optional) Evaluate performance on a test set
# >>> perf = aml.leader.model_performance(eg_ts)
# >>> perf.auc()


# perf


# lb = aml.leaderboard
# lb


# >>> perf = aml.leader.model_performance(eg_va)
# >>> perf.auc()

