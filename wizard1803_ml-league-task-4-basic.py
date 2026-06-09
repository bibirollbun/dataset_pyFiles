import numpy as np
import pandas as pd
import re
import string
from wordcloud import WordCloud


!pip install wordcloud


Base_path = "/kaggle/input/comments-classification/Dataset"
train_path = "/kaggle/input/comments-classification/Dataset/train.csv"
test_path = "/kaggle/input/comments-classification/Dataset/test.csv"


train = pd.read_csv(f"{train_path}")
test = pd.read_csv(f"{test_path}")


total_samples = len(train)
print("Total samples:", total_samples)

# Percentage of class 0
count_0 = (train['psychotic_depression'] == 0).sum()
percent_0 = (count_0 / total_samples) * 100
print(f"Percentage of 0: {percent_0:.2f}%")

# Percentage of class 1
count_1 = (train['psychotic_depression'] == 1).sum()
percent_1 = (count_1 / total_samples) * 100
print(f"Percentage of 1: {percent_1:.2f}%")


import matplotlib.pyplot as plt
text_class0 = " ".join(train[train['psychotic_depression']==0]['comment_text'])
text_class1 = " ".join(train[train['psychotic_depression']==1]['comment_text'])

# Create word clouds
wordcloud0 = WordCloud(width=1600, height=800, background_color='white').generate(text_class0)
wordcloud1 = WordCloud(width=1600, height=800, background_color='black', colormap='autumn').generate(text_class1)

# Plot
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud0, interpolation='bilinear')
plt.axis('off')
plt.title("Class 0 - Comments")

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title("Class 1 - Comments")

plt.show()


cleaned_bad_words = [
    'mothjer', 'fuck', 'fucker', 'fucking', 'shit', 'asshole', 'bitch', 'bastard', 'cunt', 'damn', 'crap', 'dick', 'cock', 'balls', 'boobs', 'buttsecks',
    'anal', 'rape', 'sex', 'sexsex', 'fucksex', 'suck', 'go fuck', 'fuck u', 'faggot', 'fag', 'super gay',
    'idiot', 'moron', 'loser', 'dickhead', 'retarded', 'stupid', 'jerk', 'scum', 'trash',
    'nigger', 'nigga', 'jew', 'racist', 'nazi', 'hitler', 'heil', 'mexicans', 'assad',
    'die', 'kill', 'banned', 'poop', 'wanker', 'fggt', 'cocksucker', 'offfuck'
]

# Convert to set for faster lookup
bad_words_set = set(cleaned_bad_words)

# Function to count bad words in text
def count_bad_words(text):
    text_lower = str(text).lower()
    count = 0
    for bad_word in bad_words_set:
        count += text_lower.count(bad_word.lower())
    return count

# Add 'bad_word_count' feature to train dataset
train['bad_word_count'] = train['comment_text'].apply(count_bad_words)

# Add 'bad_word_count' feature to text dataset
test['bad_word_count'] = test['comment_text'].apply(count_bad_words)

# Preview the results
print(train[['comment_text', 'bad_word_count']].head())
print(test[['comment_text', 'bad_word_count']].head())



correlation = train['bad_word_count'].corr(train['psychotic_depression'])
print(f"Correlation between bad_word_count and psychotic_depression: {correlation:.4f}")



pip install nltk


import re
import string
def extract_indirect_features(df, text_col="comment_text"):
    # character-level
    df['char_count'] = df[text_col].apply(len)
    df['num_digits'] = df[text_col].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_spaces'] = df[text_col].apply(lambda x: x.count(" "))
    
    # word-level
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    df['unique_word_count'] = df[text_col].apply(lambda x: len(set(x.split())))
    df['repetition_ratio'] = df['word_count'] / (df['unique_word_count'] + 1)
    df['avg_word_length'] = df[text_col].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )
    
    # punctuation
    df['num_punctuations'] = df[text_col].apply(lambda x: sum(c in string.punctuation for c in x))
    df['num_exclamations'] = df[text_col].apply(lambda x: x.count("!"))
    df['num_questions'] = df[text_col].apply(lambda x: x.count("?"))
    df['num_quotes'] = df[text_col].apply(lambda x: x.count('"') + x.count("'"))
    
    # casing
    df['num_upper_words'] = df[text_col].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['upper_ratio'] = df[text_col].apply(lambda x: sum(c.isupper() for c in x) / (len(x) + 1))
    
    # special tokens
    df['num_stopwords'] = df[text_col].apply(
        lambda x: sum(1 for w in x.lower().split() if w in STOPWORDS)
    )
    df['stopword_ratio'] = df['num_stopwords'] / (df['word_count'] + 1)
    
    # line/sentence structure
    df['num_lines'] = df[text_col].apply(lambda x: x.count("\n") + 1)
    df['num_sentences'] = df[text_col].apply(lambda x: len(re.split(r"[.!?]", x)))
    
    return df


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

train = extract_indirect_features(train, "comment_text")
test = extract_indirect_features(test, "comment_text")


import matplotlib.pyplot as plt
import seaborn as sns

# Select all numeric feature columns except ID, text, and target
feature_cols = [
    col for col in train.columns 
    if col not in ["id", "comment_text", "psychotic_depression"]
]

# Plot all features as boxplots
n_features = len(feature_cols)
n_cols = 3  # number of plots per row
n_rows = (n_features + n_cols - 1) // n_cols

plt.figure(figsize=(18, 5 * n_rows))

for i, feature in enumerate(feature_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x="psychotic_depression", y=feature, data=train)
    plt.title(f"{feature} by Target")
    plt.xlabel("psychotic_depression (0 or 1)")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


train_df = train.drop(['id', 'comment_text', 'psychotic_depression'], axis=1)

X = train_df
y = train['psychotic_depression']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1



from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)

from imblearn.over_sampling import BorderlineSMOTE

smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='threadpoolctl')


!pip install optuna


!pip install catboost
!pip install lightgbm


import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'verbose': 0,
        'random_state': 42
    }
    
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train_res, y_train_res, cv=3, scoring='f1_macro').mean()
    return score

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=15)

best_cat_params = study_cat.best_params
print("Best CatBoost params:", best_cat_params)


import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train_res, y_train_res, cv=3, scoring='f1_macro').mean()
    return score

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=15)

best_lgb_params = study_lgb.best_params
print("Best LGBM params:", best_lgb_params)


from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

best_lgb = LGBMClassifier(**best_lgb_params)
best_cat = CatBoostClassifier(**best_cat_params)

best_lgb.fit(X_train_res, y_train_res)
best_cat.fit(X_train_res, y_train_res)

ensemble = VotingClassifier(
    estimators=[
        ('lgb', best_lgb),
        ('cat', best_cat)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble.fit(X_train_res, y_train_res)

y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]  # for ROC-AUC (binary case)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))



y_pred = ensemble.predict(X_test)

# ROC-AUC handling binary vs multi-class
if len(set(y_test)) == 2:
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
else:
    y_proba = ensemble.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc:.4f}")


test_df = test.drop(['id', 'comment_text'], axis=1)

final_pred = ensemble.predict(test_df)

# Predict probabilities for class 1 (optional, if needed)
final_proba = ensemble.predict_proba(test_df)[:, 1]

# If you want to create a submission DataFrame
import pandas as pd

submission = pd.DataFrame({
    'ID': test['id'],      # Make sure test_df has the 'id' column
    'psychotic_depression': final_pred
})

submission.to_csv('submission.csv', index=False)

