import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("库导入完成")


# 加载数据集
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print("数据集概览:")
print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print(f"样本提交文件形状: {sample_submission.shape}")

print("\n训练集列名:")
print(list(train_df.columns))

print("\n测试集列名:")
print(list(test_df.columns))


# 探索目标变量
print("目标变量分布:")
print(train_df['rule_violation'].value_counts())
print("\n百分比:")
print(train_df['rule_violation'].value_counts(normalize=True).round(3))

# 规则分析
print("\n规则分析:")
print("\n训练集中的规则:")
for i, rule in enumerate(train_df['rule'].unique()):
    count = sum(train_df['rule'] == rule)
    violation_rate = train_df[train_df['rule'] == rule]['rule_violation'].mean()
    print(f"{i+1}. {rule}")
    print(f"   样本数: {count}, 违规率: {violation_rate:.1%}")

print("\n测试集中的规则:")
for i, rule in enumerate(test_df['rule'].unique()):
    count = sum(test_df['rule'] == rule)
    print(f"{i+1}. {rule}")
    print(f"   样本数: {count}")

# 按规则查看违规分布
rule_stats = train_df.groupby('rule')['rule_violation'].agg(['count', 'sum', 'mean'])
rule_stats.columns = ['总样本数', '违规数', '违规率']
print("\n规则统计:")
print(rule_stats)


# 显示样本例子
print("样本例子:")

print("\n违规样本:")
violation_samples = train_df[train_df['rule_violation'] == 1].sample(2, random_state=42)
for idx, row in violation_samples.iterrows():
    print(f"\n规则: {row['rule'][:50]}...")
    print(f"评论: {row['body'][:150]}...")
    print(f"子版块: {row['subreddit']}")

print("\n非违规样本:")
no_violation_samples = train_df[train_df['rule_violation'] == 0].sample(2, random_state=42)
for idx, row in no_violation_samples.iterrows():
    print(f"\n规则: {row['rule'][:50]}...")
    print(f"评论: {row['body'][:150]}...")
    print(f"子版块: {row['subreddit']}")


# 检查缺失值
print("训练集缺失值:")
print(train_df.isnull().sum())
print("\n测试集缺失值:")
print(test_df.isnull().sum())

# 填充缺失值
train_df = train_df.fillna('')
test_df = test_df.fillna('')

print("\n缺失值处理完成")


class RedditFeatureExtractor:
    """Reddit评论特征提取器 - 针对未见过规则优化"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2
        )
        self.scaler = StandardScaler()
    
    def extract_text_features(self, df):
        """提取规则无关的文本特征"""
        features = pd.DataFrame()
        
        # 基本文本统计
        features['length'] = df['body'].str.len()
        features['word_count'] = df['body'].str.split().str.len()
        features['sentence_count'] = df['body'].str.count(r'[.!?]+')
        features['avg_word_length'] = features['length'] / features['word_count'].replace(0, 1)
        
        # 字符级特征 - 通用违规模式
        features['caps_ratio'] = df['body'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        features['digit_ratio'] = df['body'].apply(
            lambda x: sum(1 for c in str(x) if c.isdigit()) / max(len(str(x)), 1)
        )
        features['punct_ratio'] = df['body'].apply(
            lambda x: sum(1 for c in str(x) if c in '!@#$%^&*()') / max(len(str(x)), 1)
        )
        
        # === 通用违规信号 (不依赖特定规则) ===
        
        # URL和链接模式
        features['has_url'] = df['body'].str.contains(r'https?://[^\s]+', case=False, na=False).astype(int)
        features['short_url_patterns'] = df['body'].str.contains(
            r'bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly', case=False, na=False
        ).astype(int)
        
        # 商业/促销语言 (通用)
        features['has_discount'] = df['body'].str.contains(
            r'\b(discount|off|save|deal|free|win|prize|offer|sale)\b', case=False, na=False
        ).astype(int)
        features['has_action_words'] = df['body'].str.contains(
            r'\b(click|buy|order|download|subscribe|visit|join|register)\b', case=False, na=False
        ).astype(int)
        features['has_urgency'] = df['body'].str.contains(
            r'\b(urgent|limited|now|today|hurry|fast|quick|immediate)\b', case=False, na=False
        ).astype(int)
        
        # 价格和金钱符号
        features['has_price'] = df['body'].str.contains(
            r'[\$€£¥%]|\bdollar|\bcent|\bprice|\bcost', case=False, na=False
        ).astype(int)
        
        # 联系信息模式
        features['has_contact'] = df['body'].str.contains(
            r'\b(email|phone|contact|call|message|dm|pm)\b', case=False, na=False
        ).astype(int)
        
        # 专业建议语言 (可能违规的建议)
        features['has_advice_language'] = df['body'].str.contains(
            r'\b(should|must|need to|have to|recommend|suggest|advise)\b', case=False, na=False
        ).astype(int)
        features['has_professional_terms'] = df['body'].str.contains(
            r'\b(professional|expert|specialist|certified|licensed)\b', case=False, na=False
        ).astype(int)
        
        # 垃圾邮件模式
        features['repeated_chars'] = df['body'].apply(self._count_repeated_chars)
        features['exclamation_count'] = df['body'].str.count('!')
        features['question_count'] = df['body'].str.count(r'\?')
        features['caps_words'] = df['body'].str.count(r'\b[A-Z]{2,}\b')
        
        # 可疑格式模式
        features['has_brackets'] = df['body'].str.contains(r'\[.*\]', na=False).astype(int)
        features['has_parentheses'] = df['body'].str.contains(r'\(.*\)', na=False).astype(int)
        features['has_asterisks'] = df['body'].str.contains(r'\*.*\*', na=False).astype(int)
        
        # 社交媒体模式
        features['has_hashtags'] = df['body'].str.contains(r'#\w+', na=False).astype(int)
        features['has_mentions'] = df['body'].str.contains(r'@\w+', na=False).astype(int)
        
        return features.fillna(0)
    
    def extract_similarity_features(self, df):
        """基于正负样本的相似度特征 - 这是处理未见规则的关键"""
        features = pd.DataFrame()
        
        # 与正例的相似度 (更重要的特征)
        for i in [1, 2]:
            pos_col = f'positive_example_{i}'
            if pos_col in df.columns:
                # 多种相似度度量
                features[f'pos_{i}_jaccard'] = df.apply(
                    lambda row: self._jaccard_similarity(row['body'], row[pos_col]), axis=1
                )
                features[f'pos_{i}_length_ratio'] = df.apply(
                    lambda row: self._length_ratio(row['body'], row[pos_col]), axis=1
                )
                features[f'pos_{i}_common_patterns'] = df.apply(
                    lambda row: self._pattern_similarity(row['body'], row[pos_col]), axis=1
                )
        
        # 与负例的相似度
        for i in [1, 2]:
            neg_col = f'negative_example_{i}'
            if neg_col in df.columns:
                features[f'neg_{i}_jaccard'] = df.apply(
                    lambda row: self._jaccard_similarity(row['body'], row[neg_col]), axis=1
                )
                features[f'neg_{i}_length_ratio'] = df.apply(
                    lambda row: self._length_ratio(row['body'], row[neg_col]), axis=1
                )
                features[f'neg_{i}_common_patterns'] = df.apply(
                    lambda row: self._pattern_similarity(row['body'], row[neg_col]), axis=1
                )
        
        # 比较特征 - 关键的区分特征
        if 'pos_1_jaccard' in features.columns and 'neg_1_jaccard' in features.columns:
            features['pos_vs_neg_diff'] = features['pos_1_jaccard'] - features['neg_1_jaccard']
            features['max_pos_similarity'] = features[[c for c in features.columns if 'pos_' in c and 'jaccard' in c]].max(axis=1)
            features['max_neg_similarity'] = features[[c for c in features.columns if 'neg_' in c and 'jaccard' in c]].max(axis=1)
            features['similarity_ratio'] = features['max_pos_similarity'] / (features['max_neg_similarity'] + 0.01)
        
        return features.fillna(0)
    
    def _count_repeated_chars(self, text):
        """计算重复字符模式"""
        if pd.isna(text):
            return 0
        text = str(text)
        pattern = r'(.)\1{2,}'  # 3个或更多重复字符
        return len(re.findall(pattern, text))
    
    def _jaccard_similarity(self, text1, text2):
        """Jaccard相似度"""
        if pd.isna(text1) or pd.isna(text2):
            return 0
        
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _length_ratio(self, text1, text2):
        """长度比例相似度"""
        if pd.isna(text1) or pd.isna(text2):
            return 0
        
        len1, len2 = len(str(text1)), len(str(text2))
        if len1 == 0 and len2 == 0:
            return 1
        if len2 == 0:
            return 0
            
        return min(len1, len2) / max(len1, len2)
    
    def _pattern_similarity(self, text1, text2):
        """模式相似度（URL、大写、标点等）"""
        if pd.isna(text1) or pd.isna(text2):
            return 0
        
        text1, text2 = str(text1), str(text2)
        
        # 检查相似的模式
        patterns = [
            r'https?://[^\s]+',  # URLs
            r'[A-Z]{2,}',        # 大写单词
            r'[!]{2,}',          # 多个感叹号
            r'[\d]{3,}',         # 数字
            r'[@#]\w+',          # 提及/标签
            r'\$\d+',            # 价格
            r'\b(free|deal|discount)\b',  # 关键词
        ]
        
        similarities = []
        for pattern in patterns:
            matches1 = len(re.findall(pattern, text1, re.IGNORECASE))
            matches2 = len(re.findall(pattern, text2, re.IGNORECASE))
            
            if matches1 == 0 and matches2 == 0:
                similarities.append(1)
            elif matches1 == 0 or matches2 == 0:
                similarities.append(0)
            else:
                similarities.append(min(matches1, matches2) / max(matches1, matches2))
        
        return np.mean(similarities) if similarities else 0


# 提取特征
print("开始特征提取...")

feature_extractor = RedditFeatureExtractor()

# 提取训练集特征
print("提取训练集特征...")
train_text_features = feature_extractor.extract_text_features(train_df)
train_similarity_features = feature_extractor.extract_similarity_features(train_df)
train_manual_features = pd.concat([train_text_features, train_similarity_features], axis=1)

# TF-IDF特征
print("生成TF-IDF特征...")
train_tfidf = feature_extractor.tfidf.fit_transform(train_df['body'])

# 提取测试集特征
print("提取测试集特征...")
test_text_features = feature_extractor.extract_text_features(test_df)
test_similarity_features = feature_extractor.extract_similarity_features(test_df)
test_manual_features = pd.concat([test_text_features, test_similarity_features], axis=1)
test_tfidf = feature_extractor.tfidf.transform(test_df['body'])

# 标准化手工特征
print("标准化特征...")
train_manual_scaled = feature_extractor.scaler.fit_transform(train_manual_features)
test_manual_scaled = feature_extractor.scaler.transform(test_manual_features)

# 合并所有特征
train_features = np.hstack([train_tfidf.toarray(), train_manual_scaled])
test_features = np.hstack([test_tfidf.toarray(), test_manual_scaled])

print(f"特征提取完成！")
print(f"训练集特征维度: {train_features.shape}")
print(f"测试集特征维度: {test_features.shape}")
print(f"手工特征数量: {train_manual_features.shape[1]}")
print(f"TF-IDF特征数量: {train_tfidf.shape[1]}")


# 准备目标变量
y = train_df['rule_violation']

print(f"目标变量形状: {y.shape}")
print(f"类别分布: {y.value_counts().to_dict()}")


# 定义个体模型
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=150, 
        random_state=42, 
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42, 
        learning_rate=0.1,
        max_depth=6
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        C=0.5
    )
}

print(f"定义了 {len(models)} 个模型")


# 训练个体模型并进行交叉验证
print("开始训练个体模型...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_scores = {}
trained_models = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    
    # 交叉验证
    cv_scores = cross_val_score(model, train_features, y, cv=cv, scoring='roc_auc')
    model_scores[name] = cv_scores
    
    print(f"CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    # 训练完整模型
    model.fit(train_features, y)
    trained_models[name] = model

print("\n模型性能总结:")
for name, scores in model_scores.items():
    print(f"{name}: {scores.mean():.4f} (±{scores.std()*2:.4f})")


# 创建集成模型
print("创建集成模型...")

# 使用投票分类器
ensemble = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('lr', models['Logistic Regression'])
    ],
    voting='soft'  # 使用概率投票
)

# 训练集成模型
ensemble.fit(train_features, y)

# 集成模型交叉验证
ensemble_cv_scores = cross_val_score(ensemble, train_features, y, cv=cv, scoring='roc_auc')
print(f"集成模型 CV AUC: {ensemble_cv_scores.mean():.4f} (±{ensemble_cv_scores.std()*2:.4f})")

print("集成模型训练完成")


# 对测试集进行预测
print("对测试集进行预测...")

# 个体模型预测
individual_predictions = {}
for name, model in trained_models.items():
    pred_proba = model.predict_proba(test_features)[:, 1]
    individual_predictions[name] = pred_proba
    print(f"{name} 预测完成")

# 集成模型预测
ensemble_predictions = ensemble.predict_proba(test_features)[:, 1]
print("集成模型预测完成")

print(f"\n预测完成！生成了 {len(test_df)} 个预测")


# 创建最终提交文件
final_submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': ensemble_predictions
})

# 保存提交文件
final_submission.to_csv('submission.csv', index=False)
print("最终提交文件已保存为 'submission.csv'")


