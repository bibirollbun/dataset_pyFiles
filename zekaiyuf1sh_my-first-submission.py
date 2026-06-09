import re
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# ====== 配置文件路径 ======
BASE_DIR = '/kaggle/input/make-data-count-finding-data-references'
TRAIN_LABELS = f'{BASE_DIR}/train_labels.csv'
TEST_PDF_DIR = f'{BASE_DIR}/test/PDF'
TEST_XML_DIR = f'{BASE_DIR}/test/XML'
TRAIN_XML_DIR = f'{BASE_DIR}/train/XML'
MODEL_PATH = 'citation_classifier.joblib'

# ====== 数据引用识别配置 ======
DOI_PATTERN = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/[-._;()/:a-zA-Z0-9]+)\b'

REPO_PATTERNS = {
    'GEO': [r'GSE\d+', r'GSM\d+'],
    'PDB': [r'[1-9][a-z0-9]{3}', r'PDB\s*[:=]?\s*([1-9][a-z0-9]{3})'],
    'ArrayExpress': [r'E-[A-Z]+-\d+', r'E-MTAB-\d+'],
    'CHEMBL': [r'CHEMBL\d+'],
    'EMBL': [r'E-MEXP-\d+'],
    'SRA': [r'SR[APRX]\d+', r'ERS\d+'],
    'UniProt': [r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]'],
    'GenBank': [r'[A-Z]{2}\d{6}']
}

# ====== 核心功能函数 ======
def normalize_doi(doi):
    """标准化DOI格式"""
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi\s*[:=]\s*', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'["\',;)\]\}\s]*$', '', doi)  # 移除尾部标点和空白
    return f'https://doi.org/{doi.lower()}'

def extract_text(file_path):
    """根据文件类型提取文本"""
    if file_path.endswith('.xml'):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            return ' '.join(text_parts)
        except Exception as e:
            print(f"XML解析错误 {file_path}: {str(e)}")
            return ""
    
    elif file_path.endswith('.pdf'):
        # 跳过PDF文件
        return ""
    
    return ""

def find_citations(text):
    """在文本中查找所有数据引用"""
    if not text:
        return []
    
    citations = set()
    
    # 查找DOIs
    for doi_match in re.finditer(DOI_PATTERN, text, re.IGNORECASE):
        doi = doi_match.group(0)
        normalized_doi = normalize_doi(doi)
        # 验证DOI格式
        if re.match(r'https://doi.org/10\.[0-9]{4,}/', normalized_doi):
            citations.add(normalized_doi)
    
    # 查找数据库ID
    for repo, patterns in REPO_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                id_str = match.group(1).upper() if match.groups() else match.group(0).upper()
                
                # 特殊处理PDB格式
                if repo == 'PDB':
                    if 'PDB' in match.group(0).upper():
                        citations.add(id_str)
                    else:
                        citations.add(f"PDB {id_str}")
                else:
                    citations.add(id_str)
    
    return list(citations)

def extract_context(text, identifier, window_size=200):
    """提取引用标识符的上下文"""
    if not text or not identifier:
        return ""
    
    # 安全转义标识符中的所有正则特殊字符
    try:
        # 对于DOI，只使用后缀部分
        if identifier.startswith('https://doi.org/'):
            search_str = re.escape(identifier[16:])
        else:
            search_str = re.escape(identifier)
    except Exception as e:
        print(f"标识符转义错误: {identifier} - {str(e)}")
        return ""
    
    contexts = []
    
    # 使用简单的字符串查找替代正则表达式
    start_idx = 0
    while True:
        # 不区分大小写的查找
        idx = text.lower().find(search_str.lower(), start_idx)
        if idx == -1:
            break
            
        start = max(0, idx - window_size)
        end = min(len(text), idx + len(search_str) + window_size)
        contexts.append(text[start:end])
        start_idx = idx + len(search_str)
    
    return ' '.join(contexts)

# ====== 分类模型训练 ======
def train_classifier():
    """训练引用类型分类器"""
    # 加载训练标签
    train_df = pd.read_csv(TRAIN_LABELS)
    print(f"训练标签加载完成，共 {len(train_df)} 条记录")
    
    # 构建训练数据
    texts, labels = [], []
    processed_articles = set()
    
    for idx, row in train_df.iterrows():
        article_id = row['article_id'].replace('/', '_')
        if article_id in processed_articles:
            continue
            
        xml_path = f'{TRAIN_XML_DIR}/{article_id}.xml'
        if not os.path.exists(xml_path):
            continue
            
        text = extract_text(xml_path)
        if not text:
            continue
            
        # 获取当前文章的所有数据引用
        article_citations = train_df[train_df['article_id'] == row['article_id']]
        
        for _, citation_row in article_citations.iterrows():
            dataset_id = citation_row['dataset_id']
            context = extract_context(text, dataset_id)
            
            if context:
                texts.append(context)
                labels.append(citation_row['type'])
        
        processed_articles.add(article_id)
        if len(processed_articles) % 100 == 0:
            print(f"已处理 {len(processed_articles)} 篇文章")
    
    print(f"训练数据准备完成，共 {len(texts)} 个样本")
    
    if not texts:
        print("错误：没有可用的训练文本")
        return None
    
    # 训练分类模型
    model = make_pipeline(
        TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english'),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    )
    model.fit(texts, labels)
    
    # 保存模型
    dump(model, MODEL_PATH)
    print(f"模型训练完成并保存至 {MODEL_PATH}")
    return model

# ====== 测试集处理 ======
def process_test_set():
    """处理测试集并生成提交文件"""
    # 加载或训练模型
    if os.path.exists(MODEL_PATH):
        print("加载预训练模型...")
        model = load(MODEL_PATH)
    else:
        print("训练新模型...")
        model = train_classifier()
    
    if model is None:
        print("无法训练模型，使用基于规则的分类")
        model = None
    
    # 收集测试文件 - 只处理XML文件
    test_articles = set()
    for file in os.listdir(TEST_XML_DIR):
        if file.endswith('.xml'):
            article_id = file.replace('.xml', '')
            test_articles.add(article_id)
    
    print(f"发现 {len(test_articles)} 篇有XML的测试文章")
    
    # 处理每篇文章
    results = []
    for article_id in test_articles:
        xml_path = f'{TEST_XML_DIR}/{article_id}.xml'
        
        # 只处理XML文件
        if os.path.exists(xml_path):
            text = extract_text(xml_path)
        else:
            continue
            
        if not text:
            continue
        
        # 识别所有引用
        citations = find_citations(text)
        if not citations:
            continue
        
        # 分类引用类型
        for dataset_id in citations:
            context = extract_context(text, dataset_id)
            if not context:
                continue
                
            if model:
                try:
                    citation_type = model.predict([context])[0]
                except:
                    citation_type = classify_by_rules(context)
            else:
                citation_type = classify_by_rules(context)
            
            results.append({
                'article_id': article_id,
                'dataset_id': dataset_id,
                'type': citation_type
            })
    
    # 生成提交文件
    if results:
        submission = pd.DataFrame(results)
        submission = submission.drop_duplicates(subset=['article_id', 'dataset_id', 'type'])
        submission['row_id'] = range(len(submission))
        submission = submission[['row_id', 'article_id', 'dataset_id', 'type']]
    else:
        submission = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
    
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件生成完成，包含 {len(submission)} 条记录")
    return submission

def classify_by_rules(context):
    """基于规则的分类备选方案"""
    if not context:
        return 'Primary'
    
    context_lower = context.lower()
    
    # 指示Primary的关键词
    primary_indicators = [
        'this study', 'our data', 'generated for', 'collected for', 
        'produced in', 'experimental data', 'newly sequenced',
        'primary data', 'raw data', 'collected in this work',
        'specifically for this research'
    ]
    
    # 指示Secondary的关键词
    secondary_indicators = [
        'obtained from', 'publicly available', 'retrieved from',
        'downloaded from', 'existing dataset', 'previous study',
        'previously published', 'reuse of', 'secondary data',
        'public database', 'archive data'
    ]
    
    # 计算关键词出现次数
    primary_count = sum(1 for word in primary_indicators if word in context_lower)
    secondary_count = sum(1 for word in secondary_indicators if word in context_lower)
    
    # 基于关键词计数决定类型
    if primary_count > secondary_count:
        return 'Primary'
    elif secondary_count > primary_count:
        return 'Secondary'
    else:
        # 默认返回Primary
        return 'Primary'

# ====== 执行入口 ======
if __name__ == '__main__':
    print("开始处理测试集...")
    submission = process_test_set()
    
    # 如果没有找到任何引用，创建空的提交文件
    if submission is None or submission.empty:
        empty_submission = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
        empty_submission.to_csv('submission.csv', index=False)
        print("创建空的提交文件")
    else:
        print(f"成功生成提交文件，包含 {len(submission)} 条记录")

