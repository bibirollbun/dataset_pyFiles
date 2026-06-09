# -*- coding: utf-8 -*-
"""
H&M个性化时尚推荐竞赛 - 优化高级解决方案
"""

# 安装必要的库
print("安装必要的库...")
!pip install implicit lightgbm tqdm -q

# 导入所需库
print("导入必要的库...")
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
import gc
import os
import warnings
import lightgbm as lgb
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# 尝试导入implicit库
try:
    from implicit.als import AlternatingLeastSquares
    print("成功导入implicit库")
    use_implicit = True
except:
    print("无法导入implicit库，将使用替代方案")
    from sklearn.decomposition import NMF
    use_implicit = False

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 记录开始时间
start_time = datetime.now()
print(f"开始执行 - 当前时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 设置数据路径
BASE_PATH = '/kaggle/input/h-and-m-personalized-fashion-recommendations'
TRANSACTIONS_PATH = f'{BASE_PATH}/transactions_train.csv'
ARTICLES_PATH = f'{BASE_PATH}/articles.csv'
CUSTOMERS_PATH = f'{BASE_PATH}/customers.csv'
SAMPLE_SUB_PATH = f'{BASE_PATH}/sample_submission.csv'

# -----------------------------------------------------------------------------
# 第1部分：数据加载与基础预处理
# -----------------------------------------------------------------------------

print("\n开始数据加载与预处理...")

# 加载数据
transactions = pd.read_csv(TRANSACTIONS_PATH, dtype={'article_id': str})
articles = pd.read_csv(ARTICLES_PATH, dtype={'article_id': str})
customers = pd.read_csv(CUSTOMERS_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

# 确保商品ID是字符串格式，并补充前导零至10位
def format_article_id(x):
    try:
        return f"{int(x):010d}"
    except:
        return x

transactions['article_id'] = transactions['article_id'].apply(format_article_id)
articles['article_id'] = articles['article_id'].apply(format_article_id)

# 展示数据基本信息
print(f"交易数据大小: {transactions.shape}")
print(f"商品数据大小: {articles.shape}")
print(f"客户数据大小: {customers.shape}")
print(f"需要预测的用户数: {len(sample_submission['customer_id'].unique())}")

# 时间处理
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
last_date = transactions['t_dat'].max()
print(f"数据集最后日期: {last_date}")

# 添加时间特征
transactions['day_diff'] = (last_date - transactions['t_dat']).dt.days
transactions['week'] = transactions['t_dat'].dt.isocalendar().week
transactions['month'] = transactions['t_dat'].dt.month
transactions['year'] = transactions['t_dat'].dt.year
transactions['dayofweek'] = transactions['t_dat'].dt.dayofweek

# -----------------------------------------------------------------------------
# 第2部分：增强特征工程
# -----------------------------------------------------------------------------

print("\n开始高级特征工程...")

# 计算时间窗口
last_week_date = last_date - timedelta(days=7)
last_2weeks_date = last_date - timedelta(days=14)
last_month_date = last_date - timedelta(days=30)
cutoff_date = last_date - timedelta(days=90)  # 使用最近90天的数据

# 应用时间衰减权重 - 越近的交易权重越高
def calculate_decay_weight(days, max_days=90, min_weight=0.1):
    if days > max_days:
        return min_weight / 2
    return min_weight + (1.0 - min_weight) * (1.0 - days / max_days)

transactions['weight'] = transactions['day_diff'].apply(calculate_decay_weight)

# 筛选最近的交易数据
recent_transactions = transactions[transactions['t_dat'] >= cutoff_date].copy()
print(f"最近90天交易记录数: {len(recent_transactions)}")

# 抽取商品元数据特征
print("\n处理商品元数据...")
# 产品层级
articles['product_type_no'] = articles['article_id'].str[-7:-5].fillna('00')
articles['product_group_name'] = articles['article_id'].str[-5:-4].fillna('0')

# 合并商品类别特征
articles['product_category'] = articles['product_type_no'] + '_' + articles['product_group_name']

# 添加商品特征到交易数据
articles_features = articles[['article_id', 'product_type_no', 'product_group_name', 
                             'index_group_name', 'section_name', 'garment_group_name']]

# 计算不同时间窗口的热门商品
print("\n计算热门商品（使用时间衰减）...")

# 加权热门商品 - 考虑时间衰减
weighted_popular = transactions.groupby('article_id')['weight'].sum().reset_index()
weighted_popular = weighted_popular.sort_values('weight', ascending=False)
weighted_popular_items = weighted_popular.head(100)['article_id'].tolist()

# 最后一周的热门商品
last_week_data = recent_transactions[recent_transactions['t_dat'] >= last_week_date]
last_week_popular = last_week_data['article_id'].value_counts().head(100).index.tolist()

# 最后两周的热门商品
last_2weeks_data = recent_transactions[recent_transactions['t_dat'] >= last_2weeks_date]
last_2weeks_popular = last_2weeks_data['article_id'].value_counts().head(100).index.tolist()

# 最后一个月的热门商品
last_month_data = recent_transactions[recent_transactions['t_dat'] >= last_month_date]
last_month_popular = last_month_data['article_id'].value_counts().head(100).index.tolist()

print(f"加权热门商品数: {len(weighted_popular_items)}")
print(f"最后一周热门商品数: {len(last_week_popular)}")
print(f"最后两周热门商品数: {len(last_2weeks_popular)}")
print(f"最后一个月热门商品数: {len(last_month_popular)}")

# 计算用户购买历史和高级用户特征
print("\n计算用户购买历史和特征...")

# 用户最近购买的商品 - 带有时间权重
user_purchases = {}
user_item_weights = defaultdict(dict)  # 用户-商品权重

# 为每个用户创建加权历史
for customer, group in tqdm(recent_transactions.groupby('customer_id')):
    # 按时间排序
    sorted_items = group.sort_values('t_dat', ascending=False)
    
    # 获取用户购买的所有商品，带权重
    items_with_weights = []
    for _, row in sorted_items.iterrows():
        article = row['article_id']
        weight = row['weight']
        items_with_weights.append((article, weight))
        user_item_weights[customer][article] = weight
    
    # 保存用户的购买历史
    user_purchases[customer] = items_with_weights

# 用户最后一周购买的商品
user_last_week_purchases = {}
for customer, group in tqdm(last_week_data.groupby('customer_id')):
    user_items = group['article_id'].unique().tolist()
    user_last_week_purchases[customer] = user_items

# 计算用户对商品类别的偏好
user_category_affinity = defaultdict(Counter)
for customer, group in tqdm(recent_transactions.merge(articles_features, on='article_id').groupby('customer_id')):
    # 计算用户对不同类别的偏好
    for _, row in group.iterrows():
        product_type = row['product_type_no']
        weight = row['weight']
        user_category_affinity[customer][product_type] += weight
        
        garment_group = row.get('garment_group_name', 'unknown')
        user_category_affinity[customer][f"garment_{garment_group}"] += weight

print(f"有购买历史的用户数: {len(user_purchases)}")
print(f"最后一周有购买的用户数: {len(user_last_week_purchases)}")

# 释放内存
del last_week_data, last_2weeks_data, last_month_data
gc.collect()

# -----------------------------------------------------------------------------
# 第3部分：协同过滤模型
# -----------------------------------------------------------------------------

print("\n准备协同过滤模型...")

# 创建用户-物品交互矩阵（使用时间衰减权重）
ui_data = []
for customer, items_weights in tqdm(user_purchases.items()):
    for article, weight in items_weights:
        ui_data.append((customer, article, weight))

ui_df = pd.DataFrame(ui_data, columns=['customer_id', 'article_id', 'weight'])

# 创建ID映射
print("创建用户和商品ID映射...")
le_customer = LabelEncoder()
le_article = LabelEncoder()

ui_df['customer_enc'] = le_customer.fit_transform(ui_df['customer_id'])
ui_df['article_enc'] = le_article.fit_transform(ui_df['article_id'])

# 获取编码后的最大值
max_customer = ui_df['customer_enc'].max() + 1
max_article = ui_df['article_enc'].max() + 1

print(f"用户数: {max_customer}, 商品数: {max_article}")

# 创建稀疏矩阵
print("创建加权用户-商品交互稀疏矩阵...")
sparse_ui_matrix = csr_matrix(
    (ui_df['weight'], (ui_df['customer_enc'], ui_df['article_enc'])),
    shape=(max_customer, max_article)
)
print(f"稀疏矩阵形状: {sparse_ui_matrix.shape}")

# 创建反向映射
customer_idx_to_id = {idx: le_customer.classes_[idx] for idx in range(len(le_customer.classes_))}
article_idx_to_id = {idx: le_article.classes_[idx] for idx in range(len(le_article.classes_))}

# 初始化推荐结果存储
cf_recs = {}

# 训练ALS模型
if use_implicit:
    print("\n训练ALS协同过滤模型...")
    model = AlternatingLeastSquares(
        factors=200,            # 因子数量
        regularization=0.01,    # 正则化参数
        iterations=20,          # 迭代次数 
        calculate_training_loss=True,
        num_threads=4,          # 多线程
        random_state=42
    )
    model.fit(sparse_ui_matrix, show_progress=True)
    
    # 准备推荐
    print("\n为用户生成ALS推荐...")
    test_customers = set(sample_submission['customer_id'].unique())
    customers_in_train = set(le_customer.classes_)
    test_customers_in_train = test_customers.intersection(customers_in_train)
    
    print(f"在训练数据中的测试用户数: {len(test_customers_in_train)}")
    
    # 为训练集中的测试用户生成推荐
    for customer_id in tqdm(test_customers_in_train):
        try:
            # 找到用户编码索引
            customer_idx = np.where(le_customer.classes_ == customer_id)[0][0]
            
            # 获取推荐
            recs = model.recommend(
                customer_idx, 
                sparse_ui_matrix[customer_idx], 
                N=100,
                filter_already_liked_items=False
            )
            
            # 转换回原始商品ID
            rec_items = [article_idx_to_id[idx] for idx, _ in recs]
            cf_recs[customer_id] = rec_items
        except Exception as e:
            # 跳过异常
            continue

print(f"成功为 {len(cf_recs)} 个用户生成了ALS推荐")

# -----------------------------------------------------------------------------
# 第4部分：高级推荐策略
# -----------------------------------------------------------------------------

print("\n构建高级推荐策略...")

# 计算商品的全局流行度得分
article_popularity = transactions['article_id'].value_counts().to_dict()

# 根据商品购买时间计算最近流行度
recent_popularity = recent_transactions['article_id'].value_counts().to_dict()

# 计算商品类别的流行度
article_type_popularity = {}
if 'product_type_no' in articles.columns:
    merged_data = recent_transactions.merge(articles[['article_id', 'product_type_no']], on='article_id', how='left')
    article_type_popularity = merged_data['product_type_no'].value_counts().to_dict()

# 计算商品相似度 - 基于用户共现
print("计算商品相似度...")
item_user_matrix = sparse_ui_matrix.T

# 计算余弦相似度的分子 (dot product)
item_similarity = item_user_matrix.dot(item_user_matrix.T).toarray()

# 计算范数
norms = np.sqrt(np.sum(item_user_matrix.power(2), axis=1)).reshape(1, -1)
norms_matrix = np.dot(norms.T, norms)

# 防止除零错误
norms_matrix[norms_matrix == 0] = 1e-10

# 计算余弦相似度
item_similarity = item_similarity / norms_matrix

# 对角线设为0（商品与自身的相似度）
np.fill_diagonal(item_similarity, 0)

# 为商品创建相似商品字典
similar_items = {}
for i in tqdm(range(min(1000, item_similarity.shape[0]))):  # 仅为前1000个热门商品计算
    article_id = article_idx_to_id[i]
    similar_indices = np.argsort(-item_similarity[i])[:20]  # 取相似度最高的20个
    similar_items[article_id] = [article_idx_to_id[idx] for idx in similar_indices]

print(f"为 {len(similar_items)} 个商品计算了相似商品")

# -----------------------------------------------------------------------------
# 第5部分：融合策略生成最终推荐
# -----------------------------------------------------------------------------

print("\n融合多种策略生成最终推荐...")

# 为新用户和冷启动用户准备推荐
cold_start_recs = last_week_popular[:12]  # 使用最近一周的热门商品

# 检查推荐是否符合要求的函数
def validate_recommendation(items, required_length=12):
    """确保推荐列表满足要求"""
    # 去重
    unique_items = list(dict.fromkeys(items))
    
    # 如果不足所需长度，使用热门商品填充
    if len(unique_items) < required_length:
        remaining = required_length - len(unique_items)
        for item in last_week_popular:
            if item not in unique_items:
                unique_items.append(item)
                remaining -= 1
                if remaining == 0:
                    break
    
    # 如果还不足，使用全局热门商品填充
    if len(unique_items) < required_length:
        for item in weighted_popular_items:
            if item not in unique_items:
                unique_items.append(item)
                if len(unique_items) >= required_length:
                    break
    
    # 截断为所需长度
    return unique_items[:required_length]

# 为用户生成最终推荐
def generate_recommendations(customer_id):
    """融合多种策略为单个用户生成推荐"""
    candidates = []
    
    # 1. 如果用户有最近一周的购买记录，优先推荐
    if customer_id in user_last_week_purchases:
        candidates.extend(user_last_week_purchases[customer_id])
    
    # 2. 使用用户购买历史
    if customer_id in user_purchases:
        # 获取用户所有购买的商品，按权重排序
        history_items = sorted(
            user_purchases[customer_id], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 添加历史购买的商品
        history_articles = [item for item, _ in history_items]
        candidates.extend(history_articles)
        
        # 3. 添加用户购买过的商品的相似商品
        for article in history_articles[:5]:  # 只考虑最近的5个商品
            if article in similar_items:
                candidates.extend(similar_items[article])
    
    # 4. 使用协同过滤推荐
    if customer_id in cf_recs:
        candidates.extend(cf_recs[customer_id])
    
    # 5. 添加用户偏好的商品类别中的热门商品
    if customer_id in user_category_affinity:
        # 获取用户最喜欢的商品类别
        top_categories = sorted(
            user_category_affinity[customer_id].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # 取前3个类别
        
        for category, _ in top_categories:
            if category.startswith('garment_'):
                # 处理garment类别
                garment = category[8:]  # 去掉"garment_"前缀
                category_items = articles[articles['garment_group_name'] == garment]['article_id'].tolist()
            else:
                # 处理产品类型
                category_items = articles[articles['product_type_no'] == category]['article_id'].tolist()
            
            # 按流行度排序该类别下的商品
            if category_items:
                category_pop = {item: article_popularity.get(item, 0) for item in category_items}
                top_category_items = sorted(category_pop.items(), key=lambda x: x[1], reverse=True)[:10]
                candidates.extend([item for item, _ in top_category_items])
    
    # 6. 添加热门商品以确保有足够的候选项
    candidates.extend(last_week_popular)
    candidates.extend(weighted_popular_items)
    
    # 验证并格式化最终推荐
    final_recommendations = validate_recommendation(candidates)
    
    # 确保商品ID格式正确（10位数字字符串）
    formatted_recommendations = []
    for item in final_recommendations:
        try:
            formatted_recommendations.append(f"{int(item):010d}")
        except:
            formatted_recommendations.append(item)
    
    return formatted_recommendations

# 生成所有用户的推荐
print("\n为所有用户生成最终推荐...")
test_customers = sample_submission['customer_id'].unique()
results = []

# 使用进度条跟踪处理
for customer in tqdm(test_customers):
    try:
        # 获取用户推荐
        recommendations = generate_recommendations(customer)
        
        # 格式化为字符串
        prediction = ' '.join(recommendations)
        results.append((customer, prediction))
    except Exception as e:
        # 如果出错，使用冷启动推荐
        formatted_cold_start = [f"{int(item):010d}" for item in cold_start_recs]
        prediction = ' '.join(formatted_cold_start)
        results.append((customer, prediction))
        print(f"用户 {customer} 推荐生成失败: {str(e)}")

# 创建提交DataFrame
submission = pd.DataFrame(results, columns=['customer_id', 'prediction'])

# 验证提交文件格式
print("\n验证提交文件格式...")
def validate_prediction_format(pred_str):
    """验证预测格式是否正确"""
    items = pred_str.split()
    return len(items) == 12

invalid_predictions = submission[~submission['prediction'].apply(validate_prediction_format)]
if len(invalid_predictions) > 0:
    print(f"警告：发现 {len(invalid_predictions)} 个格式不正确的预测")
    print("示例：")
    print(invalid_predictions.head())
    
    # 修复格式不正确的预测
    print("修复格式不正确的预测...")
    for idx in invalid_predictions.index:
        formatted_cold_start = [f"{int(item):010d}" for item in cold_start_recs]
        submission.at[idx, 'prediction'] = ' '.join(formatted_cold_start)
else:
    print("所有预测格式正确")

# 保存提交文件
submission.to_csv('submission.csv', index=False)

# 最终验证
print("\n最终验证...")
final_submission = pd.read_csv('submission.csv')
print(f"提交文件行数: {len(final_submission)}")
print("预测示例:")
print(final_submission['prediction'].head())

# 记录结束时间
end_time = datetime.now()
print(f"\n完成! 已生成提交文件 'submission.csv'")
print(f"总运行时间: {(end_time - start_time).total_seconds() / 60:.2f} 分钟")

