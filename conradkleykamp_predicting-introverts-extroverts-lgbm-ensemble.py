# Importing and loading necessary libraries and packages

# Fundamental libraries
import pandas as pd
import numpy as np

# Hiding warnings
import warnings
warnings.filterwarnings("ignore")

# Data viz
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'white', palette = 'Set2')
pal = sns.color_palette('Set2')

# Scipy stats
from scipy.stats import skew

# Catboost
import catboost
from catboost import Pool, CatBoostClassifier
from catboost.utils import eval_metric

# Sklearn
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

# LightGBM
import lightgbm
from lightgbm import LGBMClassifier, plot_importance

# Optuna
import optuna
from optuna.samplers import TPESampler

# XGBoost
from xgboost import XGBClassifier

# Tqdm
from tqdm import tqdm

# Shap
import shap
shap.initjs()


# Loading in Kaggle datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')

# Loading in original dataset
df_original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")


# Viewing first 5 entries of 'df_train'
df_train.head()


# Looking at the info of 'df_train'
df_train.info()


# Creating a function to show a summary of a given dataset
def show_summary(df):
    sum_df = pd.DataFrame(index = list(df))
    sum_df['Dtype'] = df.dtypes
    sum_df['Count'] = df.count()
    sum_df['#Unique'] = df.nunique()
    sum_df['%Unique'] = sum_df['#Unique'] / len(df) * 100
    sum_df['#Null'] = df.isnull().sum()
    sum_df['%Null'] = sum_df['#Null'] / len(df) * 100
    print(sum_df)


# Examining summary of 'df_train'
show_summary(df_train)


# Examining summary statistics of each numeric column in 'df_train'
df_train.describe()


# Viewing first 5 entries of 'df_test'
df_test.head()


# Looking at the info of 'df_test'
df_test.info()


# Examining summary of 'df_test'
show_summary(df_test)


# Examining summary statistics of each numeric column in 'df_test'
df_test.describe()


# Creating countplot for target variable 'Personality'
ax = sns.countplot(x='Personality', data=df_train, palette='Set2')
for label in ax.containers:
  ax.bar_label(label)
ax.set_ylabel('Count')
ax.set_xlabel('Personality')
ax.set_title('Personality Distribution')
ax.set_ylim(0, 15000)
plt.show()


# Checking the % proportion of each Personality class
print(df_train['Personality'].value_counts(normalize=True))


# Creating 'showplot' function to plot categorical features
def showplot(columnname):
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax = ax.flatten()
    value_counts = df_train[columnname].value_counts()
    labels = value_counts.index.tolist()
    colors =["#4caba4", "#d68c78",'#a3a2a2','#ab90a0', '#e6daa3', '#6782a8', '#8ea677']
    
    # Donut Chart
    wedges, texts, autotexts = ax[0].pie(
        value_counts, autopct='%1.1f%%',textprops={'size': 9, 'color': 'white','fontweight':'bold' }, colors=colors,
        wedgeprops=dict(width=0.35),  startangle=80,   pctdistance=0.85  )
    centre_circle = plt.Circle((0, 0), 0.6, fc='white')
    ax[0].add_artist(centre_circle)

    # Count Plot
    sns.countplot(data=df_train, y=columnname, ax=ax[1], palette=colors, order=labels)
    for i, v in enumerate(value_counts):
        ax[1].text(v + 1, i, str(v), color='black',fontsize=10, va='center')
    sns.despine(left=True, bottom=True)
    plt.yticks(fontsize=9,color='black')
    ax[1].set_ylabel(None)
    plt.xlabel("")
    plt.xticks([])
    fig.suptitle(columnname, fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# Using 'showplot' function to visualize 'Stage_fear'
showplot('Stage_fear')


# Using 'showplot function to visualize 'Drained_after_socializing'
showplot('Drained_after_socializing')


# Visualizing distributions of numeric features (histograms with KDE line)
fig, ax = plt.subplots(5, 1, figsize = (15, 15), dpi = 300)
ax = ax.flatten()
data_numeric = df_train.drop(['Personality', 'Stage_fear', 'Drained_after_socializing'], axis=1)
features = data_numeric.columns

for i, column in enumerate(features):

    sns.histplot(df_train[column], ax=ax[i], color=pal[0], fill=True, kde=True, bins=30)
    sns.histplot(df_test[column], ax=ax[i], color=pal[2], fill=True, kde=True, bins=30)
    ax[i].set_title(f'{column}', size = 14)
    ax[i].set_xlabel(None)

fig.suptitle('Distributions of Numeric Features', fontsize = 24, fontweight = 'bold')
fig.legend(['Train','Test'])
plt.tight_layout()


# Checking skewness of numeric features in 'df_train'
df_train.skew(numeric_only=True)


# Checking skewness of numeric features in 'df_test'
df_test.skew(numeric_only=True)


# Visualizing correlation matrix of numeric features in training data
plt.figure(figsize=(14,10))
corr=data_numeric.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm', linewidths=0.5, fmt=',.2f', vmax=1, vmin=-1, center=0)
plt.suptitle('Correlation Matrix', fontsize=16, fontweight='bold')
plt.show()


# Visualizing pairplot of 'df_train'
sns.pairplot(df_train.drop(['Stage_fear', 'Drained_after_socializing'], axis=1), hue='Personality', corner=True)
plt.suptitle('Pairplot of Training Data')
plt.show()


# Dropping null values from 'df_train' and 'df_original'
# Creating temporary dataframes for each: 'df_train_temp' & 'df_original_temp'
df_train_temp = df_train.dropna()
df_original_temp = df_original.dropna()


# Declaring categorical and numerical columns
cat_cols = (['Stage_fear', 'Drained_after_socializing', 'Personality'])
num_cols = (['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency'])

# Combining categorical and numerical columns into 'features'
features = cat_cols + num_cols


# Creating new target column 'label', where all train set samples are labeled with 0, and all orignal set samples with 1
df_train_temp['label'] = 0
df_original_temp['label'] = 1
target = 'label'

# Combinging 'features' and target 'label' into 'all_cols'
all_cols = features + [target]


# Checking out the shape of 'all_cols' in train and original datasets
df_train_temp[all_cols].shape, df_original_temp[all_cols].shape


# Defining a function to create adversarial data: combines, shuffles, and re-splits the two datasets
# The resulting datasets include a mixture of the train and orignal data
def create_adversarial_data(df_train_temp, df_original_temp, cols, N_val=10000):
    df_master = pd.concat([df_train_temp[cols], df_original_temp[cols]], axis=0)
    adversarial_test = df_master.sample(N_val, replace=False)
    adversarial_train = df_master[~df_master.index.isin(adversarial_test.index)]
    return adversarial_train, adversarial_test

# Applying function to train and orignal data, checking out the resulting shapes
adversarial_train, adversarial_test = create_adversarial_data(df_train_temp, df_original_temp, all_cols)
adversarial_train.shape, adversarial_test.shape


# Setting up the Catboost model for adversarial validation
train_data = Pool(
    data=adversarial_train[features],
    label=adversarial_train[target],
    cat_features=cat_cols
)
holdout_data = Pool(
    data=adversarial_test[features],
    label=adversarial_test[target],
    cat_features=cat_cols
)

# Establishing parameters for the Catboost classifier
params = {
    'iterations': 100,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': 42,
    'verbose': 0
}

# Fitting the model to the data
model = CatBoostClassifier(**params)
_ = model.fit(train_data, eval_set=holdout_data)


# Setting up ROC Curve plot
def plot_roc(y_trues, y_preds, labels, x_max=1.0):
    fig, ax = plt.subplots()
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax.legend()
    ax.grid()
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim([-0.01, x_max])
    _ = ax.set_ylabel('True Positive Rate')
    
# Plotting ROC Curve plot
plot_roc(
    [holdout_data.get_label()],
    [model.predict_proba(holdout_data)[:,1]],
    ['Baseline']
)


# Defining function to plot feature importance
def plot_importances(model, holdout_data, features):
    shap_values = model.get_feature_importance(holdout_data, type='ShapValues')
    expected_value = shap_values[0,-1]
    shap_values = shap_values[:,:-1]
    shap.summary_plot(shap_values, holdout_data, feature_names=features, plot_type='bar')
    
# Plotting feature importance
plot_importances(model, holdout_data, features)


# Removing 'Post_frequency' and 'Going_outside' and retraining the model
params2 = dict(params)
params2.update({"ignored_features": ['Post_frequency', 'Going_outside']})
model2 = CatBoostClassifier(**params2)
_ = model2.fit(train_data, eval_set=holdout_data, plot=False, verbose=False)


# Plotting updated ROC Curve plot
plot_roc(
    [holdout_data.get_label()]*2,
    [model.predict_proba(holdout_data)[:,1], model2.predict_proba(holdout_data)[:,1]],
    ['Baseline', "Removing Important Features"]
)


# Reloading in training and testing sets, this time including the 'id' column
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# Making fresh copies of the training and testing data
df_train_pre = df_train.copy()
df_test_pre = df_test.copy()


# Encoding 'Personality' as 1 (Extrovert) & 0 (Introvert) for training data
df_train_pre['Personality'].replace({'Extrovert': 1, 'Introvert': 0}, inplace=True)


# One-hot encoding categorical variables for training and testing data
df_train_pre = pd.get_dummies(df_train_pre, columns=['Stage_fear', 'Drained_after_socializing'])
df_test_pre = pd.get_dummies(df_test_pre, columns=['Stage_fear', 'Drained_after_socializing'])


# Creating function to impute global median values of each column
def clean_missing_values(df):

    # Listing features that need imputing
    features_to_impute = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]

    print("Cleaning data using global medians:\n")

    # Looping through each feature and imputing
    for col in features_to_impute:
        median_val = df[col].median()
        print(f"Feature: {col}") 
        print(f"Median used: {median_val}")
        df[col] = df[col].fillna(median_val)
        print(f"Nulls remaining: {df[col].isna().sum()}\n")

# Applying function to data
clean_missing_values(df_train_pre)
clean_missing_values(df_test_pre)


# Feature engineering, creating new features (columns) that represent interactions
def feature_engineering(df):

    # Interactions between extroverted activities
    df['Social_x_Outside'] = df['Social_event_attendance'] * df['Going_outside']
    df['Social_x_Friends'] = df['Social_event_attendance'] * df['Friends_circle_size']
    df['Social_x_Posts'] = df['Social_event_attendance'] * df['Post_frequency']
    df['Outside_x_Friends'] = df['Going_outside'] * df['Friends_circle_size']
    df['Outside_x_Posts'] = df['Going_outside'] * df['Post_frequency']
    df['Friends_x_Posts'] = df['Friends_circle_size'] * df['Post_frequency']

    # Interactions between alone time and extroverted activites
    df['Alone_x_Social'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['Alone_x_Outside'] = df['Time_spent_Alone'] * df['Going_outside']
    df['Alone_x_Friends'] = df['Time_spent_Alone'] * df['Friends_circle_size']
    df['Alone_x_Posts'] = df['Time_spent_Alone'] * df['Post_frequency']
    
# Applying function to training and testing sets
feature_engineering(df_train_pre)
feature_engineering(df_test_pre)


# Verifying changes made to the training data
df_train_pre.info()


# Verifying changes made to the testing data
df_test_pre.info()


# Steps taken IF including the original dataset
run = 0

if run == 1:
    # Reading in a clean version of the original dataset
    df_original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

    # Applying encoding steps to the original dataset
    df_original['Personality'].replace({'Extrovert': 1, 'Introvert': 0}, inplace=True)
    df_original = pd.get_dummies(df_original, columns=['Stage_fear', 'Drained_after_socializing'])

    # Applying feature engineering to original dataset
    feature_engineering(df_original)
    
    # Combining training data and original data
    df_combined = pd.concat([df_train_pre, df_original], axis=0, ignore_index=True)
    
    # Creating final versions of each cleaned dataset
    train = df_combined.copy()
    test = df_test_pre.copy()


# Steps taken IF NOT including the original dataset

run = 1

if run == 1:
    # Creating final versions of each cleaned dataset (NOT INCLUDING ORIGINAL)
    train = df_train_pre.copy()
    test = df_test_pre.copy()


# Calculating weight for positive/negative classes
Introvert = (train['Personality'] == 0).sum()
Extrovert = (train['Personality'] == 1).sum()
scale_pos_weight = Introvert / Extrovert
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# Setting up training and testing sets for the model
ID = test['id']
X_train = train.drop(['Personality', 'id'], axis=1)
y_train = train['Personality']
X_test = test.drop(['id'], axis=1)


# Defining the Optuna trial which will work to tune the parameters of the LGBM model

# Creating the objective for optuna
def objective(trial):

    imbalance_strategy = trial.suggest_categorical("imbalance_strategy", ["scale_pos_weight", "is_unbalance"])
    
    params = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0)
    }

    # Applying imbalance handling
    if imbalance_strategy == "scale_pos_weight":
        params['scale_pos_weight'] = scale_pos_weight
    else:
        params['is_unbalance'] = True
    
    # Fitting LGBM model with parameters from the trials
    model = LGBMClassifier(**params)
    # Stratified sampling 
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_splits = cv.split(X_train, y_train)
    
    # Creating empty scores list to hold AUC scores from each trialed model
    scores = []
    for train_idx, val_idx in cv_splits:
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred_acc = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred_acc)
        scores.append(score)
    
    # Printing and returning mean AUC scores
    mean_score = np.mean(scores)
    print(f"Mean Accuracy Score = {mean_score:.5f}")
    return mean_score

# When set to 1, optuna will create a study to find the optimal parameters
run = 0

if run == 1:
    
    # Each optuna study uses an independent sampler with a TPE algorithm
    # For each trial, the TPE essentially uses Gaussian Mixture Models to identify the optimal parameter value
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=30, multivariate=True, seed=42), direction="maximize")
    study.optimize(objective, n_trials=100)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial.params)


# Recording best parameters from trial #1
best_params = {'objective': 'binary',
               'verbosity': -1,
               'boosting_type': 'gbdt',
               'random_state': 42,
               'num_leaves': 259, 
               'learning_rate': 0.029385442161128397, 
               'n_estimators': 369, 
               'subsample_for_bin': 175904, 
               'reg_alpha': 2.2990464275110867, 
               'reg_lambda': 0.009126367790369154, 
               'max_depth': 8, 
               'colsample_bytree': 0.368023545639538, 
               'subsample': 0.7112554200243772, 
               'min_child_samples': 100, 
               'feature_fraction': 0.570042007618262, 
               'bagging_fraction': 0.7591648261818684}


# Recording best parameters from trial #2 
# Added an imbalance strategy parameter
best_params_2 = {'objective': 'binary',
                 'verbosity': -1,
                 'boosting_type': 'gbdt',
                 'random_state': 42,
                 'imbalance_strategy': 'is_unbalance', 
                 'num_leaves': 54, 
                 'learning_rate': 0.014279602588224386, 
                 'n_estimators': 225, 
                 'subsample_for_bin': 274214, 
                 'reg_alpha': 0.0032476530557493336, 
                 'reg_lambda': 1.8189460933139643e-08, 
                 'max_depth': 10, 
                 'colsample_bytree': 0.6049618875191283, 
                 'subsample': 0.9441173058325185, 
                 'min_child_samples': 60, 
                 'feature_fraction': 0.6413167361944319, 
                 'bagging_fraction': 0.7624510407122069}


# Recording best parameters from trial #3
# Includes imbalance strategy & one-hot encoded categorical variables
best_params_3 = {'objective': 'binary',
                 'verbosity': -1,
                 'boosting_type': 'gbdt',
                 'random_state': 42,
                 'imbalance_strategy': 'scale_pos_weight', 
                 'num_leaves': 38, 
                 'learning_rate': 0.018360371052560654, 
                 'n_estimators': 401, 
                 'subsample_for_bin': 98381, 
                 'reg_alpha': 0.01773808344715304, 
                 'reg_lambda': 4.447246083105632, 
                 'max_depth': 6, 
                 'colsample_bytree': 0.7984469645792099, 
                 'subsample': 0.8538489583255215, 
                 'min_child_samples': 52, 
                 'feature_fraction': 0.7047879681680635, 
                 'bagging_fraction': 0.7107751393073273}


# Recording best parameters from trial #4
# Includes all previous changes
# Added feature interaction variables
# BEST PERFORMANCE SO FAR
best_params_4 = {'objective': 'binary',
                 'verbosity': -1,
                 'boosting_type': 'gbdt',
                 'random_state': 42,
                 'imbalance_strategy': 'scale_pos_weight', 
                 'num_leaves': 50, 
                 'learning_rate': 0.01070571694557106, 
                 'n_estimators': 232, 
                 'subsample_for_bin': 228003, 
                 'reg_alpha': 6.599897876421907, 
                 'reg_lambda': 0.00026228386167264473, 
                 'max_depth': 8, 
                 'colsample_bytree': 0.339624290751257, 
                 'subsample': 0.41367275994282515, 
                 'min_child_samples': 14, 
                 'feature_fraction': 0.6181842682137473, 
                 'bagging_fraction': 0.9204242347586222}


# Fitting the model with the best parameters!
final_model = LGBMClassifier(**best_params_4)
final_model.fit(X_train, y_train)


# Obtaining final mean accuracy score, using stratified sampling again
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_splits = tqdm(cv.split(X_train, y_train), total=cv.get_n_splits(), desc='CV Progress')

scores = []
for train_idx, val_idx in cv_splits:
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    final_model.fit(X_train_fold, y_train_fold)
    y_pred = final_model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_pred)
    scores.append(score)
    
    print(f'score: {score:.5f}')

print(f"Mean Score ＝ {np.mean(scores):.5f}") 


# Visualizing accuracy scores across folds
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title("Accuracy Scores Across Folds", fontsize=14)
plt.xlabel("Fold", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.grid(True)
plt.xticks(range(1, len(scores) + 1))
plt.show()


# Viewing confusion matrix, predictions compared to validation data
y_pred = final_model.predict(X_val_fold)
cm = confusion_matrix(y_val_fold, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# Viewing feature importance for the final model
plot_importance(final_model)


# This ensemble setup has been adapted from a great work by H-Z-NING (linked in the references section)

# Splitting 'train' dataset into features (X) and target (y)
X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Setting up ensemble of models (XGBoost, CatBoost, LGBM)
xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0)

lgbm = LGBMClassifier(**best_params_4)

# Creating ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)],
    voting='soft')

# Fitting ensemble model to training data
ensemble.fit(X_train, y_train)


# Optimizing prediction threshold
val_probs = ensemble.predict_proba(X_valid)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)


# Making prediction probabilities on the test data, using the final ensemble model
probs = ensemble.predict_proba(X_test)[:, 1]

# Converting probabilities to predictions
predictions = (probs >= best_threshold).astype(int)

# Creating submission with model predictions
submission = pd.DataFrame({'id': ID, "Personality": predictions})

# Converting 1s back to Extrovert and 0s back to Introvert
submission['Personality'].replace({1: 'Extrovert', 0: 'Introvert'}, inplace=True)


# Creating .csv file for submissions and scoring
run = 1

if run == 1:
    submission.to_csv('submission.csv', index=False)

