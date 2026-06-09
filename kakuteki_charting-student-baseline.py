import time
import numpy as np
import cudf
import cuml
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import sklearn.metrics
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

import re
from nltk.stem import WordNetLemmatizer
import nltk

import warnings
warnings.filterwarnings('ignore')

print('RAPIDS',cuml.__version__)

start_time = time.time()

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(train.shape, test.shape)
print("Target distribution:")
print(train['target_cat'].value_counts().head(10))
print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

# Enhanced feature engineering with mathematical understanding
start_time = time.time()
def create_enhanced_features(df):
    # Basic text features
    df['text_length'] = df['StudentExplanation'].str.len()
    df['word_count'] = df['StudentExplanation'].str.split().str.len()
    df['sentence_count'] = df['StudentExplanation'].str.count('\.')
    df['question_marks'] = df['StudentExplanation'].str.count('\?')
    df['exclamation_marks'] = df['StudentExplanation'].str.count('\!')
    df['avg_word_length'] = df['StudentExplanation'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
    
    # Mathematical content features
    df['has_numbers'] = df['StudentExplanation'].str.contains(r'\d+', na=False).astype(int)
    df['number_count'] = df['StudentExplanation'].str.count(r'\d+')
    df['has_fractions'] = df['StudentExplanation'].str.contains(r'\/|fraction|numerator|denominator', na=False).astype(int)
    df['has_decimals'] = df['StudentExplanation'].str.contains(r'\d+\.\d+', na=False).astype(int)
    df['has_percentages'] = df['StudentExplanation'].str.contains(r'%|percent', na=False).astype(int)
    df['has_operations'] = df['StudentExplanation'].str.contains(r'add|subtract|multiply|divide|plus|minus|times|equal', na=False).astype(int)
    
    # Mathematical symbols and expressions
    df['has_math_symbols'] = df['StudentExplanation'].str.contains(r'[\+\-\*\/\=\(\)\^\<\>]', na=False).astype(int)
    df['has_variables'] = df['StudentExplanation'].str.contains(r'\b[a-zA-Z]\b', na=False).astype(int)
    
    # Confidence and reasoning indicators
    df['has_uncertainty'] = df['StudentExplanation'].str.contains(r'maybe|perhaps|might|could|possibly|not sure|think|guess|probably', na=False).astype(int)
    df['has_certainty'] = df['StudentExplanation'].str.contains(r'definitely|certainly|sure|obviously|clearly|always|never|must', na=False).astype(int)
    df['has_explanation'] = df['StudentExplanation'].str.contains(r'because|since|so|therefore|thus|then|reason|explain', na=False).astype(int)
    
    # Error patterns
    df['has_negation'] = df['StudentExplanation'].str.contains(r'not|don\'t|doesn\'t|won\'t|can\'t|isn\'t|aren\'t', na=False).astype(int)
    df['has_confusion'] = df['StudentExplanation'].str.contains(r'confus|wrong|mistake|error|difficult|hard|don\'t understand', na=False).astype(int)
    
    # Answer reference features
    df['references_answer'] = df['StudentExplanation'].str.contains(r'answer|result|solution|correct|incorrect', na=False).astype(int)
    df['shows_work'] = df['StudentExplanation'].str.contains(r'step|first|second|then|next|finally', na=False).astype(int)
    
    # Question type analysis
    df['question_length'] = df['QuestionText'].str.len()
    df['question_word_count'] = df['QuestionText'].str.split().str.len()
    df['has_question_numbers'] = df['QuestionText'].str.contains(r'\d+', na=False).astype(int)
    
    # Answer analysis
    df['answer_length'] = df['MC_Answer'].str.len()
    df['answer_is_numeric'] = df['MC_Answer'].str.match(r'^\d+(\.\d+)?$', na=False).astype(int)
    
    # Interaction features
    df['explanation_to_question_ratio'] = df['text_length'] / (df['question_length'] + 1)
    df['complexity_score'] = df['word_count'] * df['has_math_symbols'] * df['has_numbers']
    
    return df

train = create_enhanced_features(train)
test = create_enhanced_features(test)
print(f"Enhanced feature engineering completed in {time.time() - start_time:.2f} seconds")

# Enhanced text preprocessing with domain-specific handling
start_time = time.time()
# Create rich text representation
train['sentence'] = "Question: " + train['QuestionText'].astype(str) + \
                    " Answer: " + train['MC_Answer'].astype(str) + \
                    " Explanation: " + train['StudentExplanation'].astype(str)

test['sentence'] = "Question: " + test['QuestionText'].astype(str) + \
                   " Answer: " + test['MC_Answer'].astype(str) + \
                   " Explanation: " + test['StudentExplanation'].astype(str)

# Enhanced text cleaning with mathematical awareness
def enhanced_clean(text):
    # Preserve mathematical expressions
    text = re.sub(r'(\d+)/(\d+)', r'\1 fraction \2', text)
    text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 plus \2', text)
    text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 minus \2', text)
    text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 multiply \2', text)
    text = re.sub(r'(\d+)\s*รท\s*(\d+)', r'\1 divide \2', text)
    text = re.sub(r'(\d+)\s*=\s*(\d+)', r'\1 equals \2', text)
    
    # Clean whitespace and punctuation
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    return text.strip().lower()

train['sentence'] = train['sentence'].apply(enhanced_clean)
test['sentence'] = test['sentence'].apply(enhanced_clean)

# Improved lemmatization with error handling
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    train['sentence'] = train['sentence'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    test['sentence'] = test['sentence'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    print("Lemmatization completed")
except Exception as e:
    print(f"Lemmatization failed: {e}, using original text")

print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds")

# Enhanced target mapping with frequency-based encoding
start_time = time.time()
# Create frequency-based mappings to handle class imbalance better
category_freq = train['Category'].value_counts()
misconception_freq = train['Misconception'].value_counts()

# Map by frequency (most frequent gets lower index)
map_target1 = {cat: idx for idx, cat in enumerate(category_freq.index)}
map_target2 = {misc: idx for idx, misc in enumerate(misconception_freq.index)}

train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)

print("Category distribution:")
print(train['Category'].value_counts())
print("\nMisconception distribution:")
print(train['Misconception'].value_counts().head(15))
print(f"Enhanced target mapping completed in {time.time() - start_time:.2f} seconds")

# Enhanced TF-IDF with multiple analyzers and better feature selection
start_time = time.time()
def create_advanced_tfidf_features(train_sentences, test_sentences):
    features_train = []
    features_test = []
    
    # Word-level TF-IDF with multiple n-gram ranges
    print("Creating word-level TF-IDF features...")
    tfidf_configs = [
        {'ngram_range': (1, 2), 'max_features': 30000, 'name': 'word_1_2'},
        {'ngram_range': (2, 3), 'max_features': 20000, 'name': 'word_2_3'},
        {'ngram_range': (3, 4), 'max_features': 15000, 'name': 'word_3_4'},
    ]
    
    for config in tfidf_configs:
        try:
            tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=config['ngram_range'],
                analyzer='word',
                max_df=0.95,
                min_df=3,
                max_features=config['max_features'],
                sublinear_tf=True
            )
            tfidf.fit(list(train_sentences) + list(test_sentences))
            features_train.append(tfidf.transform(train_sentences))
            features_test.append(tfidf.transform(test_sentences))
            print(f"Completed {config['name']} TF-IDF")
        except Exception as e:
            print(f"Failed {config['name']} TF-IDF: {e}")
    
    # Character-level TF-IDF
    print("Creating character-level TF-IDF features...")
    try:
        tfidf_char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_df=0.95,
            min_df=3,
            max_features=10000,
            sublinear_tf=True
        )
        tfidf_char.fit(list(train_sentences) + list(test_sentences))
        features_train.append(tfidf_char.transform(train_sentences))
        features_test.append(tfidf_char.transform(test_sentences))
        print("Completed character-level TF-IDF")
    except Exception as e:
        print(f"Character TF-IDF failed: {e}")
    
    # Combine all features
    if len(features_train) > 1:
        train_features = sparse.hstack(features_train)
        test_features = sparse.hstack(features_test)
    else:
        train_features = features_train[0]
        test_features = features_test[0]
    
    return train_features, test_features

train_embeddings, test_embeddings = create_advanced_tfidf_features(train['sentence'], test['sentence'])
print('TF-IDF shape:', train_embeddings.shape)

# Enhanced handcrafted features
feature_columns = [
    'text_length', 'word_count', 'sentence_count', 'question_marks', 
    'exclamation_marks', 'avg_word_length', 'has_numbers', 'number_count',
    'has_fractions', 'has_decimals', 'has_percentages', 'has_operations',
    'has_math_symbols', 'has_variables', 'has_uncertainty', 'has_certainty',
    'has_explanation', 'has_negation', 'has_confusion', 'references_answer',
    'shows_work', 'question_length', 'question_word_count', 'has_question_numbers',
    'answer_length', 'answer_is_numeric', 'explanation_to_question_ratio',
    'complexity_score'
]

train_features = np.array(train[feature_columns])
test_features = np.array(test[feature_columns])

# Scale handcrafted features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Combine features
train_embeddings = sparse.hstack([train_embeddings, sparse.csr_matrix(train_features_scaled)])
test_embeddings = sparse.hstack([test_embeddings, sparse.csr_matrix(test_features_scaled)])

print('Final feature shape:', train_embeddings.shape)
print(f"Advanced feature creation completed in {time.time() - start_time:.2f} seconds")

# Enhanced ensemble model training for categories
start_time = time.time()
ytrain1 = np.zeros((len(train), len(map_target1)))
ytest1 = np.zeros((len(test), len(map_target1)))

# Use stratified k-fold with more folds for better stability
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

category_models = []
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target1'])):
    print(f"Category Fold {i+1}/10")
    
    # Ensemble of multiple models
    models = []
    
    # Primary model: cuML Logistic Regression
    try:
        model_lr = cuml.LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        model_lr.fit(train_embeddings[train_index], train['target1'].iloc[train_index])
        pred_lr = model_lr.predict_proba(train_embeddings[valid_index])
        if hasattr(pred_lr, 'get'):
            pred_lr = pred_lr.get()
        models.append(('lr', pred_lr, 0.4))
    except Exception as e:
        print(f"cuML LR failed: {e}")
    
    # Secondary model: Random Forest with feature selection
    try:
        # Feature selection for tree-based models
        selector = SelectKBest(chi2, k=min(5000, train_embeddings.shape[1]))
        X_train_sel = selector.fit_transform(train_embeddings[train_index], train['target1'].iloc[train_index])
        X_valid_sel = selector.transform(train_embeddings[valid_index])
        
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train_sel, train['target1'].iloc[train_index])
        pred_rf = rf_model.predict_proba(X_valid_sel)
        models.append(('rf', pred_rf, 0.3))
    except Exception as e:
        print(f"RF failed: {e}")
    
    # Tertiary model: XGBoost
    try:
        if train_embeddings.shape[1] > 10000:
            selector_xgb = SelectKBest(mutual_info_classif, k=3000)
            X_train_xgb = selector_xgb.fit_transform(train_embeddings[train_index], train['target1'].iloc[train_index])
            X_valid_xgb = selector_xgb.transform(train_embeddings[valid_index])
        else:
            X_train_xgb = train_embeddings[train_index].toarray()
            X_valid_xgb = train_embeddings[valid_index].toarray()
        
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',
            objective='multi:softprob',
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb_model.fit(X_train_xgb, train['target1'].iloc[train_index])
        pred_xgb = xgb_model.predict_proba(X_valid_xgb)
        models.append(('xgb', pred_xgb, 0.3))
    except Exception as e:
        print(f"XGB failed: {e}")
    
    # Ensemble predictions
    if models:
        ensemble_pred = np.zeros((len(valid_index), len(map_target1)))
        total_weight = sum([weight for _, _, weight in models])
        
        for name, pred, weight in models:
            ensemble_pred += (weight / total_weight) * pred
        
        ytrain1[valid_index] = ensemble_pred
        category_models.append(models)
    else:
        print("No models succeeded for this fold")

print("Category ACC:", np.mean(train['target1'] == np.argmax(ytrain1, 1)))
print("Category F1:", sklearn.metrics.f1_score(train['target1'], np.argmax(ytrain1, 1), average='weighted'))

# Generate test predictions for categories
for i, models in enumerate(category_models):
    ensemble_pred = np.zeros((len(test), len(map_target1)))
    total_weight = sum([weight for _, _, weight in models])
    
    for name, _, weight in models:
        # Re-predict on test set (simplified for demonstration)
        if name == 'lr':
            try:
                pred_test = model_lr.predict_proba(test_embeddings)
                if hasattr(pred_test, 'get'):
                    pred_test = pred_test.get()
                ensemble_pred += (weight / total_weight) * pred_test
            except:
                pass
    
    ytest1 += ensemble_pred / len(category_models)

print(f"Enhanced category training completed in {time.time() - start_time:.2f} seconds")

# Enhanced misconception training with similar approach
start_time = time.time()
ytrain2 = np.zeros((len(train), len(map_target2)))
ytest2 = np.zeros((len(test), len(map_target2)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target2'])):
    print(f"Misconception Fold {i+1}/10")
    
    # Use cuML with enhanced parameters
    model = cuml.LogisticRegression(
        max_iter=2000, 
        class_weight='balanced',
        solver='lbfgs',
        penalty='l2',
        C=0.1  # Stronger regularization for high-dimensional data
    )
    model.fit(train_embeddings[train_index], train['target2'].iloc[train_index])
    
    pred_valid = model.predict_proba(train_embeddings[valid_index])
    pred_test = model.predict_proba(test_embeddings)
    
    if hasattr(pred_valid, 'get'):
        pred_valid = pred_valid.get()
    if hasattr(pred_test, 'get'):
        pred_test = pred_test.get()
        
    ytrain2[valid_index] = pred_valid
    ytest2 += pred_test / 7.0

print("Misconception ACC:", np.mean(train['target2'] == np.argmax(ytrain2, 1)))
print("Misconception F1:", sklearn.metrics.f1_score(train['target2'], np.argmax(ytrain2, 1), average='weighted'))
print(f"Enhanced misconception training completed in {time.time() - start_time:.2f} seconds")

# Advanced prediction combination with conditional logic
start_time = time.time()
map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

def create_smart_predictions(ytrain1, ytrain2, map_inverse1, map_inverse2, top_k=3):
    predictions = []
    
    for i in range(len(ytrain1)):
        cat_probs = ytrain1[i]
        misc_probs = ytrain2[i]
        
        candidates = []
        
        # Get top categories and misconceptions
        top_cats = np.argsort(-cat_probs)[:top_k+2]
        top_miscs = np.argsort(-misc_probs)[:top_k+3]
        
        for cat_idx in top_cats:
            cat_name = map_inverse1[cat_idx]
            cat_prob = cat_probs[cat_idx]
            
            if 'False' in cat_name:
                # For False categories, strongly prefer non-NA misconceptions
                for misc_idx in top_miscs:
                    misc_name = map_inverse2[misc_idx]
                    misc_prob = misc_probs[misc_idx]
                    
                    if misc_name == 'NA':
                        # Penalize NA for False categories
                        joint_prob = cat_prob * misc_prob * 0.3
                    else:
                        # Boost non-NA misconceptions
                        joint_prob = cat_prob * misc_prob * 1.5
                    
                    candidates.append((joint_prob, f"{cat_name}:{misc_name}"))
            else:
                # For True categories, prefer NA misconceptions
                for misc_idx in top_miscs:
                    misc_name = map_inverse2[misc_idx]
                    misc_prob = misc_probs[misc_idx]
                    
                    if misc_name == 'NA':
                        # Boost NA for True categories
                        joint_prob = cat_prob * misc_prob * 1.8
                    else:
                        # Penalize non-NA misconceptions
                        joint_prob = cat_prob * misc_prob * 0.4
                    
                    candidates.append((joint_prob, f"{cat_name}:{misc_name}"))
        
        # Sort and select top predictions
        candidates.sort(reverse=True)
        pred = []
        seen = set()
        
        for prob, combination in candidates:
            if combination not in seen:
                pred.append(combination)
                seen.add(combination)
                if len(pred) >= top_k:
                    break
        
        # Fill with defaults if needed
        while len(pred) < top_k:
            default_combination = f"{list(map_inverse1.values())[0]}:NA"
            if default_combination not in seen:
                pred.append(default_combination)
                seen.add(default_combination)
            else:
                pred.append(f"{list(map_inverse1.values())[1]}:NA")
        
        predictions.append(pred)
    
    return predictions

# Generate enhanced predictions
smart_predictions = create_smart_predictions(ytrain1, ytrain2, map_inverse1, map_inverse2)

# Calculate enhanced metrics
def calculate_map3(target_list, pred_list):
    score = 0.0
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.0
        elif t == p[1]:
            score += 1/2
        elif t == p[2]:
            score += 1/3
    return score / len(target_list)

enhanced_map3_score = calculate_map3(train['target_cat'].tolist(), smart_predictions)
print(f"Enhanced MAP@3: {enhanced_map3_score:.6f}")

# Individual accuracy metrics
acc1 = np.mean(train['target_cat'] == [p[0] for p in smart_predictions])
acc2 = np.mean(train['target_cat'] == [p[1] for p in smart_predictions])
acc3 = np.mean(train['target_cat'] == [p[2] for p in smart_predictions])

print(f"Top-1 Accuracy: {acc1:.6f}")
print(f"Top-2 Accuracy: {acc2:.6f}")
print(f"Top-3 Accuracy: {acc3:.6f}")
print(f"Advanced prediction generation completed in {time.time() - start_time:.2f} seconds")

# Generate final test predictions
start_time = time.time()
test_predictions = create_smart_predictions(ytest1, ytest2, map_inverse1, map_inverse2)

# Format for submission
test_pred_formatted = []
for pred in test_predictions:
    test_pred_formatted.append(" ".join(pred))

# Create enhanced submission file
sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = test_pred_formatted
sub.to_csv("submission.csv", index=False)

print("Enhanced submission file created successfully!")
print(sub.head())
print(f"Final test prediction completed in {time.time() - start_time:.2f} seconds")

# Additional analysis
print("\n=== Performance Analysis ===")
print(f"Most common predictions:")
all_first_preds = [p[0] for p in smart_predictions]
pred_counts = pd.Series(all_first_preds).value_counts()
print(pred_counts.head(10))

print(f"\nMost challenging cases (lowest confidence):")
confidences = [ytrain1[i].max() * ytrain2[i].max() for i in range(len(ytrain1))]
low_conf_indices = np.argsort(confidences)[:5]
for idx in low_conf_indices:
    print(f"Row {idx}: {train.iloc[idx]['target_cat']} | Predicted: {smart_predictions[idx][0]} | Confidence: {confidences[idx]:.4f}")

