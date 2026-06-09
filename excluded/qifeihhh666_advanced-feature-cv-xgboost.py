import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


import warnings
warnings.filterwarnings("ignore")
print("ok")


def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip().lower()

def extract_math_features(text):
    return {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*/=]', text)),
        'starts_with_number': int(bool(re.match(r'^\d+', text))),
        'has_frac_token': int('FRAC_' in text)
    }

def extract_math_features(text):
    return {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*/=]', text)),
        'starts_with_number': int(bool(re.match(r'^\d+', text))),
        'has_frac_token': int('FRAC_' in text)
    }

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def more_math_feats(text):
    return {
        'starts_with_frac': int(text.startswith('FRAC_')),
        'contains_eq': int('=' in text)
    }

def create_features(df):
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        tokens = df[col].astype(str).apply(advanced_clean).apply(fast_lemmatize).str.split()
        df[f'{col}_tok_count'] = tokens.apply(len)
        df[f'{col}_uniq_ratio'] = tokens.apply(lambda x: len(set(x)) / (len(x) + 1))

        feats = pd.DataFrame(df[col].astype(str).apply(more_math_feats).tolist())
        feats.columns = [f'{col.lower()}_{c}' for c in feats.columns]
        df = pd.concat([df, feats], axis=1)

    for col in ['QuestionText', 'MC_Answer']:
        features = df[col].astype(str).apply(advanced_clean).apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        features.columns = [f'{prefix}{c}' for c in features.columns]
        df = pd.concat([df, features], axis=1)

    df['answer_to_question_ratio'] = df['mc_answer_len'] / (df['question_len'] + 1)
    df['explanation_to_answer_ratio'] = df['explanation_len'] / (df['mc_answer_len'] + 1)
    return df

print("ok")


# 加载数据
df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")  # 原 train → df
end_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")  # 原 test → end_df

# 数据预处理
df['Misconception'] = df['Misconception'].fillna('NA').astype(str)  # 操作 df
df['target_cat'] = df['Category'] + ':' + df['Misconception']
df = df.sort_values('target_cat').reset_index(drop=True)

# 创建组合文本列
df['combined_text'] = "Question: " + df['QuestionText'] + " Answer: " + df['MC_Answer'] + " Explanation: " + df['StudentExplanation']
end_df['combined_text'] = "Question: " + end_df['QuestionText'] + " Answer: " + end_df['MC_Answer'] + " Explanation: " + end_df['StudentExplanation']

# 特征工程
df = create_features(df)  # 对 df 操作
end_df = create_features(end_df)  # 对 end_df 操作

# 文本清洗和词形还原
df['cleaned_text'] = df['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
end_df['cleaned_text'] = end_df['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

df['mc_cleaned'] = df['MC_Answer'].astype(str).apply(advanced_clean).apply(fast_lemmatize)
end_df['mc_cleaned'] = end_df['MC_Answer'].astype(str).apply(advanced_clean).apply(fast_lemmatize)

df['q_cleaned'] = df['QuestionText'].astype(str).apply(advanced_clean).apply(fast_lemmatize)
end_df['q_cleaned'] = end_df['QuestionText'].astype(str).apply(advanced_clean).apply(fast_lemmatize)

# TF-IDF 向量化
# 1. 整体文本特征
tfidf_word = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000)
tfidf_word.fit(pd.concat([df['cleaned_text'], end_df['cleaned_text']]))  # 合并 df 和 end_df 的文本
df_word = tfidf_word.transform(df['cleaned_text'])
end_df_word = tfidf_word.transform(end_df['cleaned_text'])

# 2. 解释文本特征
tfidf_expl = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=3000)
tfidf_expl.fit(pd.concat([df['StudentExplanation'], end_df['StudentExplanation']]))
df_expl = tfidf_expl.transform(df['StudentExplanation'])
end_df_expl = tfidf_expl.transform(end_df['StudentExplanation'])

# 3. 选项文本特征
tfidf_mc = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
tfidf_mc.fit(pd.concat([df['mc_cleaned'], end_df['mc_cleaned']]))
df_mc = tfidf_mc.transform(df['mc_cleaned'])
end_df_mc = tfidf_mc.transform(end_df['mc_cleaned'])

# 4. 问题和选项的余弦相似度
q_vecs = tfidf_mc.transform(df['q_cleaned'])
a_vecs = tfidf_mc.transform(df['mc_cleaned'])
df['qa_cosine'] = cosine_similarity(q_vecs, a_vecs).diagonal()  # 结果存入 df

q_vecs_test = tfidf_mc.transform(end_df['q_cleaned'])
a_vecs_test = tfidf_mc.transform(end_df['mc_cleaned'])
end_df['qa_cosine'] = cosine_similarity(q_vecs_test, a_vecs_test).diagonal()  # 结果存入 end_df

# 5. 字符级n-gram特征
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
char_tfidf.fit(pd.concat([df['cleaned_text'], end_df['cleaned_text']]))
df_char = char_tfidf.transform(df['cleaned_text'])
end_df_char = char_tfidf.transform(end_df['cleaned_text'])

print("ok")


# 数值特征
numeric_cols = features = [
    # 基础长度特征
    'mc_answer_len',  # 多选题答案（MC_Answer）的文本长度（字符数）
    'explanation_len',  # 学生解释（StudentExplanation）的文本长度（字符数）
    'question_len',  # 问题（QuestionText）的文本长度（字符数）

    # 比例特征
    'explanation_to_question_ratio',  # 解释长度 / 问题长度（衡量解释相对于问题的详细程度）
    'answer_to_question_ratio',  # 答案长度 / 问题长度（衡量答案相对于问题的详细程度）
    'explanation_to_answer_ratio',  # 解释长度 / 答案长度（衡量解释相对于答案的详细程度）

    # 文本分词统计特征
    'QuestionText_tok_count',  # 问题文本的分词数量（按空格或标点分割）
    'QuestionText_uniq_ratio',  # 问题文本中唯一词占比（唯一词数 / 总词数）
    'MC_Answer_tok_count',  # 答案文本的分词数量
    'MC_Answer_uniq_ratio',  # 答案文本中唯一词占比
    'StudentExplanation_tok_count',  # 解释文本的分词数量
    'StudentExplanation_uniq_ratio',  # 解释文本中唯一词占比

    # 文本开头特征
    'questiontext_starts_with_frac',  # 问题文本是否以分数（如"1/2"）开头（0/1）
    'mc_answer_starts_with_frac',  # 答案文本是否以分数开头（0/1）
    'studentexplanation_starts_with_frac',  # 解释文本是否以分数开头（0/1）
    'starts_with_number',  # 文本是否以数字开头（问题/答案/解释的通用特征）
    'mc_starts_with_number',  # 答案文本是否以数字开头（更具体的版本）

    # 数学符号相关特征
    'questiontext_contains_eq',  # 问题文本是否包含等号"="（0/1）
    'mc_answer_contains_eq',  # 答案文本是否包含等号"="（0/1）
    'studentexplanation_contains_eq',  # 解释文本是否包含等号"="（0/1）
    'frac_count',  # 解释文本中分数（如"1/2"）的出现次数
    'number_count',  # 解释文本中数字的出现次数
    'operator_count',  # 解释文本中数学运算符（如+,-,*,/）的出现次数
    'mc_frac_count',  # 答案文本中分数的出现次数
    'mc_number_count',  # 答案文本中数字的出现次数
    'mc_operator_count',  # 答案文本中数学运算符的出现次数
    'mc_has_frac_token',  # 答案文本是否包含至少一个分数（0/1）

    # 语义相似度特征
    'qa_cosine'  # 问题（Question）和答案（Answer）的文本嵌入（如TF-IDF/BERT）的余弦相似度
]

X_numeric = sparse.csr_matrix(df[numeric_cols].fillna(0).values)  # df 的数值特征
X_numeric_test = sparse.csr_matrix(end_df[numeric_cols].fillna(0).values)  # end_df 的数值特征

# 合并所有特征
X_df = sparse.hstack([df_word, df_expl, df_char, df_mc, X_numeric])  # 原 X_train → X_df
X_end_df = sparse.hstack([end_df_word, end_df_expl, end_df_char, end_df_mc, X_numeric_test])  # 原 X_test → X_end_df

# 目标变量编码（仅对 df）
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target_cat'])  # 仅编码 df 的标签
target_classes = le.classes_
n_classes = len(le.classes_)
y = df['target_encoded'].values  # 目标变量来自 df

# 4. 正确分割特征和目标
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

print(f"Train (df) shape: {X_df.shape}")
print(f"Test (end_df) shape: {X_end_df.shape}")
print(f"Number of classes: {n_classes}")

print("ok")


oof_preds = np.zeros((X_df.shape[0], n_classes))
test_preds = np.zeros((X_end_df.shape[0], n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': n_classes, # Crucial to tell XGBoost the total number of classes
    'eval_metric': 'mlogloss',
    'max_depth': 12, # Adjusted from example to fit current optimal range
    'learning_rate': 0.05, # Adjusted
    'subsample': 0.85, # Adjusted slightly
    'colsample_bytree': 0.85, # Adjusted slightly
    'tree_method': 'gpu_hist',
    'gpu_id': 0, # Assuming GPU is available and ID is 0
    'random_state': 42,
    'n_jobs': -1 # Use all available cores
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_df, y)):
    print(f"--- Fold {fold+1}/{skf.n_splits} ---")
    
    # Create DMatrix objects for the current fold
    dtrain = xgb.DMatrix(X_df[trn_idx], label=y[trn_idx])
    dvalid = xgb.DMatrix(X_df[val_idx], label=y[val_idx])

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1500, # Increased max rounds, rely on early stopping
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=75, # Increased early stopping rounds
        verbose_eval=100 # Print progress every 100 rounds
    )
    
    # Predict OOF probabilities
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    
    # Accumulate test predictions
    test_preds += model.predict(xgb.DMatrix(X_end_df), iteration_range=(0, model.best_iteration)) / skf.n_splits


# --- Evaluation ---
oof_logloss = log_loss(y, oof_preds)
print(f"\nOOF Log Loss: {oof_logloss:.4f}")

# Convert OOF probabilities to predicted categories
oof_pred_category = np.argmax(oof_preds, axis=1)
 
# 打印分类报告
print("\n=== OOF Classification Report ===")
print(classification_report(
    y, 
    oof_pred_category,
    target_names=[str(c) for c in np.unique(y)],
    digits=4
))




# 计算混淆矩阵
# Calculate evaluation metrics
accuracy = accuracy_score(y, oof_pred_category)
cm = confusion_matrix(y, oof_pred_category)

# Visualization setup with red color scheme and appropriate size
plt.figure(figsize=(12, 10))  
plt.style.use('seaborn')

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Reds',  
    annot_kws={"size": 8},  # font size
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
    cbar=False,
    linewidths=0.5, #  Add thin borders between cells / 单元格边框
    linecolor='gray'
)

# Improved label display / 优化标签显示
plt.xticks(fontsize=10, rotation=45, ha='right')  
plt.yticks(fontsize=10)

plt.xlabel('Predicted Labels', fontsize=12, labelpad=10)
plt.ylabel('True Labels', fontsize=12, labelpad=10)
plt.title('Confusion Matrix', fontsize=14, pad=20)

# 自动调整布局并添加边距
plt.tight_layout()
plt.show()


# the code copy from @rajathrpai notebook [CV 0.9188] XGBoost KFold
def calculate_map3_score(true_labels, predicted_prob_arrays, label_encoder):
    """
    Calculates the MAP@3 score.
    true_labels: Series/array of true encoded labels.
    predicted_prob_arrays: Numpy array of predicted probabilities for each class.
    label_encoder: Fitted LabelEncoder object to decode predicted labels.
    """
    score = 0.
    num_samples = len(true_labels)
    
    # Get top 3 predicted class indices for each sample
    top3_indices = predicted_prob_arrays.argsort(axis=1)[:, -3:][:, ::-1]
    
    # Decode true labels to original category:misconception format
    # Ensure true_labels is an array for inverse_transform
    true_decoded_labels = label_encoder.inverse_transform(np.asarray(true_labels))
    
    # Decode predicted labels for MAP@3 calculation
    predicted_decoded_labels_list = []
    for indices_row in top3_indices:
        predicted_decoded_labels_list.append(label_encoder.inverse_transform(indices_row).tolist())

    for t, p_list in zip(true_decoded_labels, predicted_decoded_labels_list):
        if t == p_list[0]: score += 1.
        elif len(p_list) > 1 and t == p_list[1]: score += 1/2
        elif len(p_list) > 2 and t == p_list[2]: score += 1/3
    return score / num_samples


oof_map3_score = calculate_map3_score(y, oof_preds, le)
print(f"OOF MAP@3 Score: {oof_map3_score:.4f}")


# --- Submission ---
# Get top 3 predictions from averaged test predictions
top3_indices = test_preds.argsort(axis=1)[:, -3:][:, ::-1] # Sort descending and take top 3

test_predictions_labels = []
for indices in top3_indices:
    pred_labels = [target_classes[i] for i in indices]
    test_predictions_labels.append(' '.join(pred_labels))

sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")  # 替换为你的数据路径
sample_submission['Category:Misconception'] = test_predictions_labels
submission_filename = "submission.csv" #submission_xgboost_dmatrix_full_features.csv
sample_submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file created: {submission_filename}")
print("XGBoost model training complete with improved features and DMatrix handling.")
sample_submission.head()




