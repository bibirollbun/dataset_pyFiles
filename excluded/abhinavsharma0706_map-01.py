##import the required libraries
!pip install polars
!pip install lightgbm
import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import lightgbm as lgb
from scipy.sparse import hstack
from sklearn.utils.class_weight import compute_sample_weight


# Load train/test datasets
train = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Fill missing misconceptions with 'NA'
train = train.with_columns([
    pl.col('Misconception').fill_null('NA').cast(pl.Utf8).alias('Misconception')
])

# Create target_cat (Category:Misconception string label)
train = train.with_columns([
    (pl.col('Category') + ":" + pl.col('Misconception')).alias('target_cat')
])


# Encode Category labels
category_counts = train['Category'].value_counts().sort('count', descending=True)
map_target1 = {row['Category']: idx for idx, row in enumerate(category_counts.iter_rows(named=True))}

# Encode Misconception labels
misconception_counts = train['Misconception'].value_counts().sort('count', descending=True)
map_target2 = {row['Misconception']: idx for idx, row in enumerate(misconception_counts.iter_rows(named=True))}

# Map to numeric columns
train = train.with_columns([
    pl.col('Category').map_elements(lambda x: map_target1.get(x, -1), return_dtype=pl.Int64).alias('target1'),
    pl.col('Misconception').map_elements(lambda x: map_target2.get(x, -1), return_dtype=pl.Int64).alias('target2')
])


# Define sentence construction
def create_sentence(row):
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nExplanation: {row['StudentExplanation']}"

# Apply to train/test
train = train.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(create_sentence, return_dtype=pl.Utf8).alias('sentence')
])
test = test.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(create_sentence, return_dtype=pl.Utf8).alias('sentence')
])


print("Creating TF-IDF features...")

# Combine train and test for TF-IDF fitting
all_sentences = pd.concat([
    train.select('sentence').to_pandas(),
    test.select('sentence').to_pandas()
])

# Word-level TF-IDF (1-3 grams)
tfidf1 = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word',
                         max_df=0.95, min_df=2, max_features=10000)
tfidf1.fit(all_sentences['sentence'])
train_tfidf1 = tfidf1.transform(train['sentence'].to_pandas())
test_tfidf1 = tfidf1.transform(test['sentence'].to_pandas())

# Char-level TF-IDF (4-6 grams)
tfidf2 = TfidfVectorizer(stop_words='english', ngram_range=(4, 6), analyzer='char',
                         max_df=0.95, min_df=2, max_features=5000)
tfidf2.fit(all_sentences['sentence'])
train_tfidf2 = tfidf2.transform(train['sentence'].to_pandas())
test_tfidf2 = tfidf2.transform(test['sentence'].to_pandas())

# Combine TF-IDF features
train_embeddings = hstack([train_tfidf1, train_tfidf2])
test_embeddings = hstack([test_tfidf1, test_tfidf2])
print(f'Combined train sparse shape: {train_embeddings.shape}')
print(f'Combined test sparse shape: {test_embeddings.shape}')


print("\nTraining Category models...")
ytrain1_lr = np.zeros((len(train), len(map_target1)))
ytrain1_lgb = np.zeros((len(train), len(map_target1)))
ytest1_lr = np.zeros((len(test), len(map_target1)))
ytest1_lgb = np.zeros((len(test), len(map_target1)))

train_target1 = train['target1'].to_numpy()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for i, (train_idx, valid_idx) in enumerate(skf.split(train_embeddings, train_target1)):
    print(f"Category Fold {i}, Train: {len(train_idx)}, Valid: {len(valid_idx)}")

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(train_embeddings[train_idx], train_target1[train_idx])
    ytrain1_lr[valid_idx] = lr_model.predict_proba(train_embeddings[valid_idx])
    ytest1_lr += lr_model.predict_proba(test_embeddings) / 10

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(train_embeddings[train_idx], train_target1[train_idx],
                  eval_set=[(train_embeddings[valid_idx], train_target1[valid_idx])],
                  callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    ytrain1_lgb[valid_idx] = lgb_model.predict_proba(train_embeddings[valid_idx])
    ytest1_lgb += lgb_model.predict_proba(test_embeddings) / 10

# Weighted Ensemble
ytrain1 = 0.6 * ytrain1_lr + 0.4 * ytrain1_lgb
ytest1 = 0.6 * ytest1_lr + 0.4 * ytest1_lgb

print("Category ACC:", np.mean(train_target1 == np.argmax(ytrain1, 1)))
print("Category F1:", sklearn.metrics.f1_score(train_target1, np.argmax(ytrain1, 1), average='weighted'))


print("\nTraining Misconception models...")

# Re-fit TF-IDF for Misconception task
tfidf_misc = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), analyzer='word',
                             max_df=0.90, min_df=2, max_features=15000)
tfidf_misc.fit(all_sentences['sentence'])
train_embeddings_misc = tfidf_misc.transform(train['sentence'].to_pandas())
test_embeddings_misc = tfidf_misc.transform(test['sentence'].to_pandas())

ytrain2_lr = np.zeros((len(train), len(map_target2)))
ytrain2_lgb = np.zeros((len(train), len(map_target2)))
ytest2_lr = np.zeros((len(test), len(map_target2)))
ytest2_lgb = np.zeros((len(test), len(map_target2)))

train_target2 = train['target2'].to_numpy()

for i, (train_idx, valid_idx) in enumerate(skf.split(train_embeddings_misc, train_target2)):
    print(f"Misconception Fold {i}, Train: {len(train_idx)}, Valid: {len(valid_idx)}")

    # Logistic Regression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5, random_state=42)
    lr_model.fit(train_embeddings_misc[train_idx], train_target2[train_idx])
    ytrain2_lr[valid_idx] = lr_model.predict_proba(train_embeddings_misc[valid_idx])
    ytest2_lr += lr_model.predict_proba(test_embeddings_misc) / 10

    # LightGBM with class weights
    sample_weights = compute_sample_weight('balanced', train_target2[train_idx])
    lgb_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=50,
                                   max_depth=8, min_child_samples=20, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(train_embeddings_misc[train_idx], train_target2[train_idx],
                  sample_weight=sample_weights,
                  eval_set=[(train_embeddings_misc[valid_idx], train_target2[valid_idx])],
                  callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
    ytrain2_lgb[valid_idx] = lgb_model.predict_proba(train_embeddings_misc[valid_idx])
    ytest2_lgb += lgb_model.predict_proba(test_embeddings_misc) / 10

# Weighted Ensemble
ytrain2 = 0.7 * ytrain2_lr + 0.3 * ytrain2_lgb
ytest2 = 0.7 * ytest2_lr + 0.3 * ytest2_lgb

print("Misconception ACC:", np.mean(train_target2 == np.argmax(ytrain2, 1)))
print("Misconception F1:", sklearn.metrics.f1_score(train_target2, np.argmax(ytrain2, 1), average='weighted'))


# Create reverse label mappings
map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

# Ensure NA class (0) in misconception prediction is never selected
ytrain2[:, 0] = 0

# Top-3 predictions for validation
predicted1 = np.argsort(-ytrain1, axis=1)[:, :3]
predicted2 = np.argsort(-ytrain2, axis=1)[:, :3]

# Combine Category and Misconception predictions
predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    predict.append(pred)

# Validation true labels
train_target_cat = train['target_cat'].to_list()

# Accuracy at K
print("\nValidation Results:")
print("Acc@1:", np.mean([train_target_cat[i] == predict[i][0] for i in range(len(predict))]))
print("Acc@2:", np.mean([train_target_cat[i] == predict[i][1] for i in range(len(predict))]))
print("Acc@3:", np.mean([train_target_cat[i] == predict[i][2] for i in range(len(predict))]))

# MAP@3 implementation
def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif t == p[1]:
            score += 1 / 2
        elif t == p[2]:
            score += 1 / 3
    return score / len(target_list)

print(f"MAP@3: {map3(train_target_cat, predict)}")


# Prevent NA prediction in final misconception output
ytest2[:, 0] = 0

# Top-3 predictions for test set
predicted1 = np.argsort(-ytest1, axis=1)[:, :3]
predicted2 = np.argsort(-ytest2, axis=1)[:, :3]

# Combine predicted labels
predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, j]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    predict.append(" ".join(pred))


# Load sample submission and insert predictions
sub = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub = sub.with_columns([
    pl.Series('Category:Misconception', predict)
])

# Save to CSV
sub.write_csv("submission.csv")

print("\nSubmission file created successfully!")

