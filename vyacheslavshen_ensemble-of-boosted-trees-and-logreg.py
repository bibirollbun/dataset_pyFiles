import numpy as np
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns



# ! ls /kaggle/input/mt5-base-checkpoint


# from transformers import MT5EncoderModel, AutoTokenizer
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MT5EncoderModel.from_pretrained("/kaggle/input/mt5-base-checkpoint").to(device)
# tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/mt5-base-checkpoint")
# article = "WSDM Human preference multilingual competition"
# input_ids = tokenizer(article, return_tensors="pt").to(device).input_ids
# with torch.no_grad():
#     outputs = model(input_ids)
# hidden_state = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
# print(hidden_state.shape)


EMBED_DIM = 2048
SEED = 42

lgb_param = {'num_leaves': 70, 
             'max_depth': 20, 
             'learning_rate': 0.04, 
             'n_estimators': 240, 
             'reg_alpha': 2.2, 
             'reg_lambda': 20,
             'random_state': SEED, 
             'verbose': -1}

xgb_param = {'n_estimators': 450, 
             'learning_rate': 0.05, 
             'max_depth': 2, 
             'reg_lambda': 27, 
             'min_data_in_leaf': 34,
             'random_state': SEED, 
             'verbose': -1}

cat_param = {'depth': 5, 
             'learning_rate': 0.015, 
             'n_estimators': 1980, 
             'reg_lambda': 0.70,
             'random_state': SEED, 
             'verbose': 0}

lr_param = {
    'C': 0.01,
    'solver': 'saga',
    'max_iter': 10000,
    'random_state': SEED
}


train = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet")
test = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet")


train.head()


# Set a threshold for grouping small languages
threshold = 50

# Separate common and rare languages
language_counts = train['language'].value_counts()
common_languages = language_counts[language_counts > threshold]
others_count = language_counts[language_counts <= threshold].sum()

# Add the "Others" category
common_languages['Others'] = others_count

print("Languages in Others category: ", list(language_counts[language_counts <= threshold].index))

# Plot
plt.figure(figsize=(12, 6))
common_languages.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Language Counts (Grouped Small Counts into "Others")', fontsize=16)
plt.xlabel('Language', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Set the number of top languages to display
top_n = 10

# Get the top N languages
top_languages = language_counts[:top_n]

# Plot
plt.figure(figsize=(12, 6))
top_languages.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title(f'Top {top_n} Languages in Train Dataset', fontsize=16)
plt.xlabel('Language', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



language_counts = train['language'].value_counts()
threshold = 100
train_df = train.copy()
# Separate common and rare languages
train_df['language_grouped'] = train_df['language'].apply(
    lambda x: x if language_counts[x] > threshold else 'Others'
)

# Count tokens (naive)
train_df['prompt_tokens'] = train_df['prompt'].apply(lambda x: len(str(x).split()))
train_df['response_a_tokens'] = train_df['response_a'].apply(lambda x: len(str(x).split()))
train_df['response_b_tokens'] = train_df['response_b'].apply(lambda x: len(str(x).split()))

language_order = train_df['language_grouped'].value_counts().index

# List of columns to plot
length_columns = ['prompt_tokens', 'response_a_tokens', 'response_b_tokens']
titles = ['Prompt Token Length', 'Response A Token Length', 'Response B Token Length']

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12), sharex=True)
for i, col in enumerate(length_columns):
    sns.boxplot(
        data=train_df,
        x='language_grouped',
        y=col,
        ax=axes[i],
        order=language_order,  # Sorted order
        palette='Set3',
        showfliers=False  # Hide outliers
    )
    axes[i].set_title(f'Distribution of {titles[i]} by Language', fontsize=14)
    axes[i].set_xlabel('Language (Grouped)' if i == 2 else '')  # X-axis only for the last plot
    axes[i].set_ylabel('Length', fontsize=12)
    axes[i].set_yscale('log')  # Apply logarithmic scale
    axes[i].tick_params(axis='x', rotation=45, labelsize=10)

# Adjust layout
plt.tight_layout()
plt.show()


preprocessor = ColumnTransformer(
    transformers=[
        ('prompt_feats', TfidfVectorizer(analyzer = 'char_wb',max_features=EMBED_DIM), 'prompt'),
        ('response_a_feats', TfidfVectorizer(analyzer = 'char_wb',max_features=EMBED_DIM), 'response_a'),
        ('response_b_feats', TfidfVectorizer(analyzer = 'char_wb',max_features=EMBED_DIM), 'response_b')
    ]
)

train_tfidf = preprocessor.fit_transform(train)
test_tfidf = preprocessor.transform(test)
train['winner'] = train['winner'].map({"model_a": 0, "model_b": 1})


# @torch.no_grad()
# def extract_feature_batch(texts):
#     """
#     Extract features for a batch of texts.
#     """
#     input_ids = tokenizer(texts, return_tensors="pt", 
#                           padding=True, 
#                           truncation=True, 
#                           max_length=4096).input_ids.to(device)
#     outputs = model(input_ids)
#     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# def extract_features_df(df, batch_size=32):
#     """
#     Extract features for the entire dataframe using batching.
#     """
#     feats = []
#     for i in tqdm(range(0, len(df), batch_size)):
#         batch = df.iloc[i:i + batch_size]
#         # Extract features for prompt, response_a, and response_b
#         feat_prompt = extract_feature_batch(batch['prompt'].tolist())
#         feat_a = extract_feature_batch(batch['response_a'].tolist())
#         feat_b = extract_feature_batch(batch['response_b'].tolist())

#         # Concatenate features for each row
#         batch_feats = np.concatenate([feat_prompt, feat_a, feat_b], axis=1)  # axis=1 because features are row vectors
#         feats.append(batch_feats)
#         torch.cuda.empty_cache()

#     return np.vstack(feats)

# torch.cuda.empty_cache()

# # Extract features for train and test datasets
# train_tfidf = extract_features_df(train, batch_size=4)
# test_tfidf = extract_features_df(test, batch_size=4)

# # Map 'winner' column
# train['winner'] = train['winner'].map({"model_a": 0, "model_b": 1})


def text_stat(df, txt_col):
    for col in tqdm(txt_col, desc="Processing text columns"):

        df[f'{col}_length'] = df[col].apply(len)
        df[f'{col}_word_count'] = df[col].apply(lambda x: len(x.split()))
        df[f'{col}_char_count'] = df[col].apply(lambda x: sum([len(word) for word in x.split()]))
        df[f'{col}_avg_word_length'] = df[f'{col}_char_count'] / df[f'{col}_word_count']
        
        df[f'{col}_punctuation_count'] = df[col].apply(lambda x: sum([1 for char in x if char in string.punctuation]))
        df[f'{col}_capitalized_count'] = df[col].apply(lambda x: sum([1 for word in x.split() if word.isupper()]))
        df[f'{col}_special_char_count'] = df[col].apply(lambda x: sum([1 for char in x if not char.isalnum() and not char.isspace()]))
        df[f'{col}_unique_word_count'] = df[col].apply(lambda x: len(set(x.split())))
        df[f'{col}_lexical_diversity'] = df[f'{col}_unique_word_count'] / df[f'{col}_word_count']

        df[f'{col}_word_length_mean'] = df[col].apply(lambda x: np.mean([len(word) for word in x.split()]))
        df[f'{col}_word_length_median'] = df[col].apply(lambda x: np.median([len(word) for word in x.split()]))
        df[f'{col}_word_length_max'] = df[col].apply(lambda x: max([len(word) for word in x.split()], default=0))
        df[f'{col}_word_length_min'] = df[col].apply(lambda x: min([len(word) for word in x.split()], default=0))

        df[f'{col}_sentence_length_mean'] = df[col].apply(lambda x: np.mean([len(sentence.split()) for sentence in x.split('.') if sentence.strip()]))
        df[f'{col}_sentence_length_median'] = df[col].apply(lambda x: np.median([len(sentence.split()) for sentence in x.split('.') if sentence.strip()]))
        df[f'{col}_sentence_length_max'] = df[col].apply(lambda x: max([len(sentence.split()) for sentence in x.split('.') if sentence.strip()], default=0))
        df[f'{col}_sentence_length_min'] = df[col].apply(lambda x: min([len(sentence.split()) for sentence in x.split('.') if sentence.strip()], default=0))
    
    df['response_length_diff_a_b'] = df['response_a_length'] - df['response_b_length']
    df['response_length_diff_b_a'] = df['response_b_length'] - df['response_a_length']
    df['response_length_ratio_a_b'] = df['response_a_length'] / (df['response_b_length'] + 1e-6)  
    df['response_length_ratio_b_a'] = df['response_b_length'] / (df['response_a_length'] + 1e-6)  
    
    return df

txt_col = ['prompt', 'response_a', 'response_b']

train = text_stat(train, txt_col)
test = text_stat(test, txt_col)


ADDITIONAL_FEATURES = list(set(train.columns) - 
                           set(pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet").columns))
# ADDITIONAL_FEATURES


num_features = train_tfidf.shape[1]
new_columns = [f"tfidf{i+1}" for i in range(num_features)]

train_tfidf_ = pd.DataFrame(train_tfidf.toarray(), columns=new_columns)
train_tfidf_ = pd.concat([train[ADDITIONAL_FEATURES], train_tfidf_], axis=1)

test_tfidf_ = pd.DataFrame(test_tfidf.toarray(), columns=new_columns)
test_tfidf_ = pd.concat([test[ADDITIONAL_FEATURES], test_tfidf_], axis=1)


X = train_tfidf_
y = train['winner']

print("Train shape: ", X.shape) # 3*EMBED_DIM + LEN(ADDITIONAL_FEATURES)


# def objective(trial):
#     # LightGBM parameters
#     lgb_param = {
#         'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('lgb_max_depth', 5, 50),
#         'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1),
#         'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
#         'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.0, 5.0),
#         'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.0, 50.0),
#         'random_state': SEED,
#         'device_type': 'cpu',
#         'verbose': -1
#     }

#     # XGBoost parameters
#     xgb_param = {
#         'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
#         'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
#         'max_depth': trial.suggest_int('xgb_max_depth', 2, 10),
#         'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1.0, 50.0),
#         'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 50),
#         'random_state': SEED,
#         'device': 'cpu',
#         'verbosity': 0
#     }

#     # CatBoost parameters
#     cat_param = {
#         'depth': trial.suggest_int('cat_depth', 4, 10),
#         'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
#         'n_estimators': trial.suggest_int('cat_n_estimators', 100, 2000),
#         'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0.1, 10.0),
#         'random_state': SEED,
#         'task_type': 'CPU',
#         'verbose': 0
#     }

#     # Create individual models
#     lgb_model = lgb.LGBMClassifier(**lgb_param)
#     xgb_model = xgb.XGBClassifier(**xgb_param)
#     cat_model = CatBoostClassifier(**cat_param)

#     # Combine models into a VotingClassifier with fixed weights
#     models = [('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model)]
#     weights = [1, 1, 1]  # Fixed weights
#     voting_clf = VotingClassifier(estimators=models, voting='soft', weights=weights)

#     # Perform Stratified K-Fold Cross-Validation
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
#     scores = cross_val_score(voting_clf, X, y, cv=skf, scoring='accuracy', n_jobs=1)
#     return scores.mean()

# # Create an Optuna study with TPE Sampler and Median Pruner
# study = optuna.create_study(
#     direction='maximize',
#     sampler=optuna.samplers.TPESampler(seed=SEED),
#     pruner=optuna.pruners.MedianPruner()
# )

# # Run optimization
# # if running locally try to increase n_trials
# study.optimize(objective, n_trials=50, gc_after_trial=True)

# # Print the best trial
# print("Best trial:")
# print(f"  Value: {study.best_trial.value}")
# print("  Params:")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")


# # Extract the best hyperparameters from the Optuna study
# best_params = study.best_trial.params

# # Create the models with the best hyperparameters
# lgb_model = lgb.LGBMClassifier(
#     num_leaves=best_params['lgb_num_leaves'],
#     max_depth=best_params['lgb_max_depth'],
#     learning_rate=best_params['lgb_learning_rate'],
#     n_estimators=best_params['lgb_n_estimators'],
#     reg_alpha=best_params['lgb_reg_alpha'],
#     reg_lambda=best_params['lgb_reg_lambda'],
#     random_state=SEED,
#     verbose=-1
# )

# xgb_model = xgb.XGBClassifier(
#     n_estimators=best_params['xgb_n_estimators'],
#     learning_rate=best_params['xgb_learning_rate'],
#     max_depth=best_params['xgb_max_depth'],
#     reg_lambda=best_params['xgb_reg_lambda'],
#     min_child_weight=best_params['xgb_min_child_weight'],
#     random_state=SEED,
#     verbosity=0
# )

# cat_model = CatBoostClassifier(
#     depth=best_params['cat_depth'],
#     learning_rate=best_params['cat_learning_rate'],
#     n_estimators=best_params['cat_n_estimators'],
#     l2_leaf_reg=best_params['cat_l2_leaf_reg'],
#     random_state=SEED,
#     verbose=0
# )

# # Combine models into the VotingClassifier
# models = [('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model)]
# weights = [1, 1, 1]
# voting_clf = VotingClassifier(estimators=models, voting='soft', weights=weights)

# # Fit the VotingClassifier on the full training dataset
# voting_clf.fit(X, y)

# # Make predictions on the test data
# y_pred = voting_clf.predict(test_tfidf_)


lgb_model = lgb.LGBMClassifier(**lgb_param)
xgb_model = xgb.XGBClassifier(**xgb_param)
cat_model = CatBoostClassifier(**cat_param)
imputer = SimpleImputer(strategy='constant', fill_value=0)
logreg_model = make_pipeline(imputer, StandardScaler(), LogisticRegression(**lr_param))

models = [('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model), ('lr', logreg_model)]
weights = [1, 1, 1, 1] 
voting_clf = VotingClassifier(estimators=models, voting='soft', weights=weights)
voting_clf.fit(X, y)
y_pred = voting_clf.predict(test_tfidf_)


y_pred_labels = ['model_a' if label == 0 else 'model_b' for label in y_pred]

submission= pd.DataFrame({'id': test['id'], 'winner': y_pred_labels})
submission.to_csv('submission.csv', index=False)

submission.head()




