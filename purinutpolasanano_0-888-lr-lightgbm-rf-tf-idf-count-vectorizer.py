import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
import re
from collections import Counter



train = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train = train.with_columns([
    pl.col('Misconception').fill_null('NA').cast(pl.Utf8).alias('Misconception')
])

train = train.with_columns([
    (pl.col('Category') + ":" + pl.col('Misconception')).alias('target_cat')
])



def extract_text_features(df):
    """Extract additional text-based features"""
    features = []
    
    for row in df.iter_rows(named=True):
        question = str(row['QuestionText'])
        answer = str(row['MC_Answer'])
        explanation = str(row['StudentExplanation'])
        
        # Length features
        q_len = len(question.split())
        a_len = len(answer.split())
        e_len = len(explanation.split())
        
        # Character-level features
        q_char_len = len(question)
        a_char_len = len(answer)
        e_char_len = len(explanation)
        
        # Mathematical notation features
        math_symbols = len(re.findall(r'[+\-*/=<>≤≥±∞∑∏∫√∆∂]', question + answer + explanation))
        numbers = len(re.findall(r'\d+', question + answer + explanation))
        fractions = len(re.findall(r'\d+/\d+', question + answer + explanation))
        
        # Punctuation features
        question_marks = question.count('?')
        exclamation_marks = (question + answer + explanation).count('!')
        
        # Word overlap features
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        e_words = set(explanation.lower().split())
        
        qa_overlap = len(q_words & a_words) / max(len(q_words), 1)
        qe_overlap = len(q_words & e_words) / max(len(q_words), 1)
        ae_overlap = len(a_words & e_words) / max(len(a_words), 1)
        
        # Uncertainty indicators
        uncertainty_words = ['maybe', 'think', 'probably', 'guess', 'not sure', 'confused']
        uncertainty_count = sum(1 for word in uncertainty_words if word in explanation.lower())
        
        features.append([
            q_len, a_len, e_len, q_char_len, a_char_len, e_char_len,
            math_symbols, numbers, fractions, question_marks, exclamation_marks,
            qa_overlap, qe_overlap, ae_overlap, uncertainty_count
        ])
    
    return np.array(features)

print("Extracting text features...")
train_text_features = extract_text_features(train)
test_text_features = extract_text_features(test)



category_counts = train['Category'].value_counts().sort('count', descending=True)
map_target1 = {row['Category']: idx for idx, row in enumerate(category_counts.iter_rows(named=True))}

misconception_counts = train['Misconception'].value_counts().sort('count', descending=True)
map_target2 = {row['Misconception']: idx for idx, row in enumerate(misconception_counts.iter_rows(named=True))}

train = train.with_columns([
    pl.col('Category').map_elements(lambda x: map_target1.get(x, -1), return_dtype=pl.Int64).alias('target1'),
    pl.col('Misconception').map_elements(lambda x: map_target2.get(x, -1), return_dtype=pl.Int64).alias('target2')
])



def create_enhanced_sentence(row):
    """Create enhanced sentence with better structure"""
    question = str(row['QuestionText']).strip()
    answer = str(row['MC_Answer']).strip()
    explanation = str(row['StudentExplanation']).strip()
    
    question = re.sub(r'\s+', ' ', question)
    answer = re.sub(r'\s+', ' ', answer)
    explanation = re.sub(r'\s+', ' ', explanation)
    
    return f"[QUESTION] {question} [ANSWER] {answer} [EXPLANATION] {explanation}"

train = train.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(
        create_enhanced_sentence, return_dtype=pl.Utf8
    ).alias('sentence')
])

test = test.with_columns([
    pl.struct(['QuestionText', 'MC_Answer', 'StudentExplanation']).map_elements(
        create_enhanced_sentence, return_dtype=pl.Utf8
    ).alias('sentence')
])



print("Creating enhanced TF-IDF features...")

all_sentences = pd.concat([
    train.select('sentence').to_pandas(),
    test.select('sentence').to_pandas()
])

tfidf1 = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 3), 
    analyzer='word',
    max_df=0.95, 
    min_df=3, 
    max_features=15000,
    sublinear_tf=True,
    norm='l2'
)
tfidf1.fit(all_sentences['sentence'])
train_tfidf1 = tfidf1.transform(train['sentence'].to_pandas())
test_tfidf1 = tfidf1.transform(test['sentence'].to_pandas())

tfidf2 = TfidfVectorizer(
    ngram_range=(3, 5), 
    analyzer='char',
    max_df=0.95, 
    min_df=3, 
    max_features=8000,
    sublinear_tf=True
)
tfidf2.fit(all_sentences['sentence'])
train_tfidf2 = tfidf2.transform(train['sentence'].to_pandas())
test_tfidf2 = tfidf2.transform(test['sentence'].to_pandas())

tfidf3 = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    analyzer='word',
    max_df=0.90, 
    min_df=2, 
    max_features=12000,
    token_pattern=r'\b\w+\b',
    sublinear_tf=True
)
tfidf3.fit(all_sentences['sentence'])
train_tfidf3 = tfidf3.transform(train['sentence'].to_pandas())
test_tfidf3 = tfidf3.transform(test['sentence'].to_pandas())

count_vec = CountVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    max_features=5000,
    binary=True
)
count_vec.fit(all_sentences['sentence'])
train_count = count_vec.transform(train['sentence'].to_pandas())
test_count = count_vec.transform(test['sentence'].to_pandas())

train_embeddings = hstack([
    train_tfidf1, train_tfidf2, train_tfidf3, train_count, 
    csr_matrix(train_text_features)
])
test_embeddings = hstack([
    test_tfidf1, test_tfidf2, test_tfidf3, test_count, 
    csr_matrix(test_text_features)
])

print(f'Combined train sparse shape: {train_embeddings.shape}')
print(f'Combined test sparse shape: {test_embeddings.shape}')



print("\nTraining Category models with advanced ensemble...")
ytrain1_lr = np.zeros((len(train), len(map_target1)))
ytrain1_lgb = np.zeros((len(train), len(map_target1)))
ytrain1_rf = np.zeros((len(train), len(map_target1)))
ytest1_lr = np.zeros((len(test), len(map_target1)))
ytest1_lgb = np.zeros((len(test), len(map_target1)))
ytest1_rf = np.zeros((len(test), len(map_target1)))

train_target1 = train['target1'].to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train_target1)):
    print(f"Category Fold {i+1}/5:")
    
    lr_model = LogisticRegression(
        max_iter=2000, 
        C=0.5, 
        random_state=42,
        solver='liblinear',
        penalty='l1'
    )
    lr_model.fit(train_embeddings[train_index], train_target1[train_index])
    ytrain1_lr[valid_index] = lr_model.predict_proba(train_embeddings[valid_index])
    ytest1_lr += (lr_model.predict_proba(test_embeddings) / 5.)
    
    lgb_model = lgb.LGBMgests=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        train_embeddings[train_index], 
        train_target1[train_index],
        eval_set=[(train_embeddings[valid_index], train_target1[valid_index])],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    ytrain1_lgb[valid_index] = lgb_model.predict_proba(train_embeddings[valid_index])
    ytest1_lgb += (lgb_model.predict_proba(test_embeddings) / 5.)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(train_text_features[train_index], train_target1[train_index])
    rf_proba = rf_model.predict_proba(train_text_features[valid_index])
    
    if rf_proba.shape[1] < len(map_target1):
        padding = np.zeros((rf_proba.shape[0], len(map_target1) - rf_proba.shape[1]))
        rf_proba = np.hstack([rf_proba, padding])
    
    ytrain1_rf[valid_index] = rf_proba
    
    rf_test_proba = rf_model.predict_proba(test_text_features)
    if rf_test_proba.shape[1] < len(map_target1):
        padding = np.zeros((rf_test_proba.shape[0], len(map_target1) - rf_test_proba.shape[1]))
        rf_test_proba = np.hstack([rf_test_proba, padding])
    
    ytest1_rf += (rf_test_proba / 5.)

ytrain1 = 0.5 * ytrain1_lr + 0.4 * ytrain1_lgb + 0.1 * ytrain1_rf
ytest1 = 0.5 * ytest1_lr + 0.4 * ytest1_lgb + 0.1 * ytest1_rf

print("Category ACC:", np.mean(train_target1 == np.argmax(ytrain1, 1)))
print("Category F1:", sklearn.metrics.f1_score(train_target1, np.argmax(ytrain1, 1), average='weighted'))



print("\nTraining Misconception models with advanced ensemble...")

tfidf_misc = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 3), 
    analyzer='word',
    max_df=0.85, 
    min_df=2, 
    max_features=20000,
    sublinear_tf=True
)
tfidf_misc.fit(all_sentences['sentence'])
train_embeddings_misc = hstack([
    tfidf_misc.transform(train['sentence'].to_pandas()),
    csr_matrix(train_text_features)
])
test_embeddings_misc = hstack([
    tfidf_misc.transform(test['sentence'].to_pandas()),
    csr_matrix(test_text_features)
])

ytrain2_lr = np.zeros((len(train), len(map_target2)))
ytrain2_lgb = np.zeros((len(train), len(map_target2)))
ytest2_lr = np.zeros((len(test), len(map_target2)))
ytest2_lgb = np.zeros((len(test), len(map_target2)))

train_target2 = train['target2'].to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings_misc, train_target2)):
    print(f"Misconception Fold {i+1}/5:")
    
    lr_model = LogisticRegression(
        class_weight='balanced', 
        max_iter=2000, 
        C=0.3, 
        random_state=42,
        solver='liblinear',
        penalty='l1'
    )
    lr_model.fit(train_embeddings_misc[train_index], train_target2[train_index])
    ytrain2_lr[valid_index] = lr_model.predict_proba(train_embeddings_misc[valid_index])
    ytest2_lr += (lr_model.predict_proba(test_embeddings_misc) / 5.)
    
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', train_target2[train_index])
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=10,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        boosting_type='gbdt'
    )
    lgb_model.fit(
        train_embeddings_misc[train_index], 
        train_target2[train_index],
        sample_weight=sample_weights,
        eval_set=[(train_embeddings_misc[valid_index], train_target2[valid_index])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    ytrain2_lgb[valid_index] = lgb_model.predict_proba(train_edges_misc[valid_index])
    ytest2_lgb += (lgb_model.predict_proba(test_embeddings_misc) / 5.)

ytrain2 = 0.6 * ytrain2_lr + 0.4 * ytrain2_lgb
ytest2 = 0.6 * ytest2_lr + 0.4 * ytest2_lgb

print("Misconception ACC:", np.mean(train_target2 == np.argmax(ytrain2, 1)))
print("Misconception F1:", sklearn.metrics.f1_score(train_target2, np.argmax(ytrain2, 1), average='weighted'))



map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

ytrain2_adjusted = ytrain2.copy()
ytrain2_adjusted[:, 0] = 0

def get_compatible_predictions(cat_probs, misc_probs, train_data, top_k=3):
    """Get predictions that are compatible between category and misconception"""
    predictions = []
    
    cat_misc_pairs = train_data.select(['Category', 'Misconception']).unique()
    valid_pairs = set()
    for row in cat_misc_pairs.iter_rows(named=True):
        valid_pairs.add((row['Category'], row['Misconception']))
    
    for i in range(len(cat_probs)):
        cat_top = np.argsort(-cat_probs[i])
        misc_top = np.argsort(-misc_probs[i])
        
        pred = []
        for cat_idx in cat_top:
            cat_name = map_inverse1[cat_idx]
            
            compatible_found = False
            for misc_idx in misc_top:
                misc_name = map_inverse2[misc_idx]
                
                if (cat_name, misc_name) in valid_pairs:
                    if 'Misconception' in cat_name:
                        pred.append(cat_name + ":" + misc_name)
                    else:
                        pred.append(cat_name + ":NA")
                    compatible_found = True
                    break
            
            if not compatible_found:
                if 'Misconception' in cat_name:
                    pred.append(cat_name + ":" + map_inverse2[misc_top[0]])
                else:
                    pred.append(cat_name + ":NA")
            
            if len(pred) >= top_k:
                break
        
        while len(pred) < top_k:
            remaining_cats = [idx for idx in range(len(map_inverse1)) if idx not in [np.argmax(cat_probs[i])]]
            if remaining_cats:
                cat_idx = remaining_cats[0]
                cat_name = map_inverse1[cat_idx]
                pred.append(cat_name + ":NA")
            else:
                pred.append(map_inverse1[0] + ":NA")
        
        predictions.append(pred[:top_k])
    
    return predictions

train_predictions = get_compatible_predictions(ytrain1, ytrain2_adjusted, train)

train_target_cat = train['target_cat'].to_list()
print("\nValidation Results:")
print("Acc@1:", np.mean([train_target_cat[i] == train_predictions[i][0] for i in range(len(train_predictions))]))
print("Acc@2:", np.mean([train_target_cat[i] in train_predictions[i][:2] for i in range(len(train_predictions))]))
print("Acc@3:", np.mean([train_target_cat[i] in train_predictions[i] for i in range(len(train_predictions))]))

def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif t == p[1]:
            score += 1/2
        elif t == p[2]:
            score += 1/3
    return score / len(target_list)

print(f"MAP@3: {map3(train_target_cat, train_predictions)}")

ytest2_adjusted = ytest2.copy()
ytest2_adjusted[:, 0] = 0

test_predictions = get_compatible_predictions(ytest1, ytest2_adjusted, train)

submission_predictions = [" ".join(pred) for pred in test_predictions]

sub = pl.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub = sub.with_columns([
    pl.Series('Category:Misconception', submission_predictions)
])
sub.write_csv("submission.csv")

print("\nEnhanced submission file created successfully!")
print("Key improvements:")
print("- Enhanced text features (15 additional features)")
print("- Multiple TF-IDF configurations + Count Vectorizer")
print("- 3-model ensemble (LR + LightGBM + RF)")
print("- Category-Misconception compatibility checking")
print("- Advanced hyperparameter tuning")
print("- Better cross-validation strategy")





