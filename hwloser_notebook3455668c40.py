# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import pyarrow as pa
import numpy as np
import json
import time
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Generator
import os
import gc
import pickle
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import logging
import datetime
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ---------------------------------
# 第1部分: 数据加载和预处理
# ---------------------------------

def stream_jsonl(file_path: str, batch_size: int = 5000) -> Generator[List[Dict], None, None]:
    """以批次形式流式读取JSONL文件，避免一次性加载全部内存"""
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # 不要忘记最后一个可能小于batch_size的批次
            yield batch


def stream_events(file_path: str, batch_size: int = 5000) -> Generator[List[Dict], None, None]:
    """从JSONL文件流式读取事件，将每个事件转换为扁平格式"""
    for sessions_batch in stream_jsonl(file_path, batch_size):
        events_batch = []
        for session in sessions_batch:
            session_id = session['session']
            for event in session['events']:
                events_batch.append({
                    'session': session_id,
                    'aid': event['aid'],
                    'ts': event['ts'],
                    'type': event['type']
                })
        yield events_batch


def process_in_batches(data, batch_size=1000, process_func=None, desc="处理"):
    """通用批处理函数，显示批数信息"""
    
    total_items = len(data)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    print(f"开始{desc}...")
    print(f"总计 {total_items:,} 项，将分为 {total_batches:,} 批 (每批 {batch_size:,} 项)")
    
    results = []
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        # 计算批次范围
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        
        # 获取当前批次数据
        batch_data = data[start_idx:end_idx]
        
        # 处理批次
        print(f"处理批次 {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)...")
        if process_func:
            batch_result = process_func(batch_data)
            results.append(batch_result)
        
        # 显示进度
        elapsed = time.time() - start_time
        items_per_second = (batch_idx+1) * batch_size / elapsed if elapsed > 0 else 0
        remaining_batches = total_batches - (batch_idx+1)
        remaining_time = (elapsed / (batch_idx+1)) * remaining_batches
        
        print(f"完成进度: {(batch_idx+1)/total_batches*100:.1f}% - "
              f"速度: {items_per_second:.1f} 项/秒 - "
              f"预计剩余时间: {str(datetime.timedelta(seconds=int(remaining_time)))}")
    
    # 处理结果
    if results and hasattr(results[0], 'append') and callable(getattr(results[0], 'append')):
        final_result = pd.concat(results, ignore_index=True)
    else:
        final_result = results
    
    elapsed = time.time() - start_time
    print(f"{desc}完成! 总批数: {total_batches}, 总耗时: {elapsed:.1f} 秒")
    
    return final_result


def peek_jsonl(file_path, num_samples=2):
    """查看JSONL文件的前几个样本"""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    
    print(f"=== JSONL文件结构样本 ({file_path}) ===")
    for i, sample in enumerate(samples):
        print(f"\n样本 #{i+1}:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    # 分析会话结构
    if samples:
        sample = samples[0]
        print("\n=== 会话结构分析 ===")
        print(f"会话ID: {sample.get('session')}")
        print(f"事件数量: {len(sample.get('events', []))}")
        
        if 'events' in sample and sample['events']:
            event = sample['events'][0]
            print(f"\n事件示例:")
            print(f"  aid (物品ID): {event.get('aid')}")
            print(f"  ts (时间戳): {event.get('ts')}")
            print(f"  type (类型): {event.get('type')}")


def load_and_preview_data(file_path, format='jsonl', batch_size=3, event_limit=10):
    """加载并预览数据，支持JSONL和Parquet格式"""
    if format == 'jsonl':
        # 对于JSONL，加载第一个批次
        batch = next(stream_jsonl(file_path, batch_size))
        
        # 转换为扁平事件格式
        events = []
        for session in batch:
            session_id = session['session']
            for event in session['events'][:event_limit]:  # 限制每个会话的事件数量
                events.append({
                    'session': session_id,
                    'aid': event['aid'],
                    'ts': event['ts'],
                    'type': event['type']
                })
        
        events_df = pd.DataFrame(events)
        
        # 显示原始会话数据
        print(f"=== 原始会话数据 (前{len(batch)}个会话) ===")
        for i, session in enumerate(batch):
            print(f"\n会话 #{i+1}, ID: {session['session']}, 事件数: {len(session['events'])}")
            events_sample = session['events'][:event_limit]
            if len(session['events']) > event_limit:
                print(f"(只显示前{event_limit}个事件，共{len(session['events'])}个)")
            for j, event in enumerate(events_sample):
                print(f"  事件 #{j+1}: 物品={event['aid']}, 类型={event['type']}, 时间戳={event['ts']}")
    
    elif format == 'parquet':
        # 对于Parquet，直接读取
        events_df = pd.read_parquet(file_path)
        # 只取前几行进行展示
        events_df = events_df.head(batch_size * event_limit)
    
    # 显示事件数据框
    print(f"\n=== 扁平化事件数据 ===")
    display(events_df.head(10))
    
    # 显示数据基本统计信息
    print(f"\n=== 数据统计信息 ===")
    print(f"总事件数: {len(events_df)}")
    print(f"唯一会话数: {events_df['session'].nunique()}")
    print(f"唯一物品数: {events_df['aid'].nunique()}")
    print(f"事件类型分布:")
    display(events_df['type'].value_counts())
    
    # 可视化事件类型分布
    plt.figure(figsize=(8, 5))
    sns.countplot(data=events_df, x='type')
    plt.title('事件类型分布')
    plt.xlabel('事件类型')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return events_df


def inspect_features(features_df, feature_type='session', sample_size=5):
    """检查生成的特征"""
    print(f"=== {feature_type.capitalize()}特征检查 ===")
    print(f"特征数量: {len(features_df)}")
    print(f"特征列: {features_df.columns.tolist()}")
    print(f"\n前{sample_size}个样本:")
    display(features_df.head(sample_size))
    
    # 显示数值特征的基本统计量
    numeric_cols = features_df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n数值特征统计:")
        display(features_df[numeric_cols].describe())
        
        # 可视化特征分布
        fig, axes = plt.subplots(len(numeric_cols)//3 + 1, 3, figsize=(15, 3*len(numeric_cols)//3 + 3))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(features_df[col].dropna(), ax=axes[i], kde=True)
                axes[i].set_title(f'{col} 分布')
                axes[i].set_xlabel(col)
        
        # 隐藏未使用的子图
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def inspect_covisitation_matrix(matrix, top_items=3, top_covisits=5):
    """检查共同访问矩阵"""
    print(f"=== 共同访问矩阵检查 ===")
    print(f"矩阵大小 (唯一物品数): {len(matrix)}")
    
    # 找出共同访问次数最多的几个物品
    item_weights = [(item_id, sum(counter.values())) for item_id, counter in matrix.items()]
    top_items_by_weight = sorted(item_weights, key=lambda x: x[1], reverse=True)[:top_items]
    
    for item_id, weight in top_items_by_weight:
        print(f"\n物品 {item_id} (总共同访问次数: {weight}):")
        top_covisited = matrix[item_id].most_common(top_covisits)
        for covisit_id, count in top_covisited:
            print(f"  - 物品 {covisit_id}: 共同访问 {count} 次")


def inspect_word2vec_model(model, sample_items=3, top_similar=5):
    """检查Word2Vec模型"""
    print(f"=== Word2Vec模型检查 ===")
    print(f"向量维度: {model.vector_size}")
    print(f"词汇量 (唯一物品数): {len(model.wv)}")
    
    # 随机选择几个物品查看其相似物品
    vocab = list(model.wv.index_to_key)
    import random
    sample_vocabs = random.sample(vocab, min(sample_items, len(vocab)))
    
    for item_id in sample_vocabs:
        print(f"\n物品 {item_id} 的相似物品:")
        try:
            similar_items = model.wv.most_similar(item_id, topn=top_similar)
            for similar_id, similarity in similar_items:
                print(f"  - 物品 {similar_id}: 相似度 {similarity:.4f}")
        except KeyError:
            print(f"  物品不在模型词汇表中")


def create_session_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """为每个会话创建特征"""
    # 按会话分组
    session_features = []
    
    for session_id, group in tqdm(events_df.groupby('session')):
        # 基本会话特征
        num_events = len(group)
        session_duration = group['ts'].max() - group['ts'].min() if num_events > 1 else 0
        unique_items = group['aid'].nunique()
        
        # 事件类型统计
        type_counts = group['type'].value_counts().to_dict()
        num_clicks = type_counts.get('clicks', 0)
        num_carts = type_counts.get('carts', 0)
        num_orders = type_counts.get('orders', 0)
        
        # 用户行为特征
        cart_to_click_ratio = num_carts / num_clicks if num_clicks > 0 else 0
        order_to_cart_ratio = num_orders / num_carts if num_carts > 0 else 0
        
        # 时间相关特征
        if num_events > 1:
            timestamps = sorted(group['ts'].values)
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_between_events = np.mean(time_diffs)
            std_time_between_events = np.std(time_diffs) if len(time_diffs) > 1 else 0
        else:
            avg_time_between_events = 0
            std_time_between_events = 0
        
        # 会话最后事件特征
        sorted_group = group.sort_values('ts')
        last_event_type = sorted_group.iloc[-1]['type']
        last_event_ts = sorted_group.iloc[-1]['ts']
        last_event_aid = sorted_group.iloc[-1]['aid']
        
        # 重复商品交互特征
        repeated_aids = group['aid'].value_counts()
        max_interactions_for_single_aid = repeated_aids.max() if not repeated_aids.empty else 0
        num_repeated_aids = sum(repeated_aids > 1)
        
        # 将所有特征添加到列表中
        session_features.append({
            'session': session_id,
            'num_events': num_events,
            'session_duration': session_duration,
            'unique_items': unique_items,
            'num_clicks': num_clicks,
            'num_carts': num_carts,
            'num_orders': num_orders,
            'cart_to_click_ratio': cart_to_click_ratio,
            'order_to_cart_ratio': order_to_cart_ratio,
            'avg_time_between_events': avg_time_between_events,
            'std_time_between_events': std_time_between_events,
            'last_event_type': last_event_type,
            'last_event_ts': last_event_ts,
            'last_event_aid': last_event_aid,
            'max_interactions_for_single_aid': max_interactions_for_single_aid,
            'num_repeated_aids': num_repeated_aids
        })
    
    return pd.DataFrame(session_features)


def create_item_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """为每个商品创建特征"""
    # 按商品ID分组
    item_features = []
    
    # 计算全局统计量
    total_clicks = events_df[events_df['type'] == 'clicks'].shape[0]
    total_carts = events_df[events_df['type'] == 'carts'].shape[0]
    total_orders = events_df[events_df['type'] == 'orders'].shape[0]
    
    for aid, group in tqdm(events_df.groupby('aid')):
        # 基本统计量
        num_sessions = group['session'].nunique()
        
        # 事件类型统计
        type_counts = group['type'].value_counts().to_dict()
        num_clicks = type_counts.get('clicks', 0)
        num_carts = type_counts.get('carts', 0)
        num_orders = type_counts.get('orders', 0)
        
        # 转化率特征
        cart_to_click_ratio = num_carts / num_clicks if num_clicks > 0 else 0
        order_to_cart_ratio = num_orders / num_carts if num_carts > 0 else 0
        
        # 全局流行度特征 (TF-IDF思想)
        click_popularity = num_clicks / total_clicks if total_clicks > 0 else 0
        cart_popularity = num_carts / total_carts if total_carts > 0 else 0
        order_popularity = num_orders / total_orders if total_orders > 0 else 0
        
        # 会话覆盖率
        session_coverage = num_sessions / events_df['session'].nunique()
        
        # 时间特征
        if len(group) > 0:
            first_ts = group['ts'].min()
            last_ts = group['ts'].max()
            avg_ts = group['ts'].mean()
            ts_span = last_ts - first_ts if last_ts > first_ts else 0
        else:
            first_ts = 0
            last_ts = 0
            avg_ts = 0
            ts_span = 0
        
        # 将所有特征添加到列表中
        item_features.append({
            'aid': aid,
            'num_sessions': num_sessions,
            'num_clicks': num_clicks,
            'num_carts': num_carts,
            'num_orders': num_orders,
            'cart_to_click_ratio': cart_to_click_ratio,
            'order_to_cart_ratio': order_to_cart_ratio,
            'click_popularity': click_popularity,
            'cart_popularity': cart_popularity,
            'order_popularity': order_popularity,
            'session_coverage': session_coverage,
            'first_ts': first_ts,
            'last_ts': last_ts,
            'avg_ts': avg_ts,
            'ts_span': ts_span
        })
    
    return pd.DataFrame(item_features)


def create_interaction_features(events_df: pd.DataFrame, session_features: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """创建会话-商品交互特征"""
    interaction_features = []
    
    # 将会话和商品特征转换为字典，方便查找
    session_dict = session_features.set_index('session').to_dict('index')
    item_dict = item_features.set_index('aid').to_dict('index')
    
    # 为每个交互创建特征
    for _, row in tqdm(events_df.iterrows(), total=len(events_df)):
        session_id = row['session']
        aid = row['aid']
        
        # 基本交互信息
        interaction = {
            'session': session_id,
            'aid': aid,
            'ts': row['ts'],
            'type': row['type']
        }
        
        # 添加会话特征
        if session_id in session_dict:
            for k, v in session_dict[session_id].items():
                if k != 'session':  # 跳过ID字段
                    interaction[f'session_{k}'] = v
        
        # 添加商品特征
        if aid in item_dict:
            for k, v in item_dict[aid].items():
                if k != 'aid':  # 跳过ID字段
                    interaction[f'item_{k}'] = v
        
        # 添加交互特征
        # 例如：该会话中与该商品的交互次数
        session_item_interactions = events_df[(events_df['session'] == session_id) & (events_df['aid'] == aid)]
        interaction['interaction_count'] = len(session_item_interactions)
        
        # 将特征添加到列表
        interaction_features.append(interaction)
    
    return pd.DataFrame(interaction_features)


def create_covisitation_matrices(events_df: pd.DataFrame, output_dir: str) -> Dict[str, Dict[int, Counter]]:
    """创建不同类型的共同访问矩阵并保存到磁盘"""
    os.makedirs(output_dir, exist_ok=True)
    matrices = {}
    
    # 1. 点击到点击的共同访问矩阵（12小时窗口）
    click_matrix_path = os.path.join(output_dir, 'click_to_click_matrix.pkl')
    if os.path.exists(click_matrix_path):
        print(f"加载现有点击-点击矩阵从 {click_matrix_path}")
        with open(click_matrix_path, 'rb') as f:
            matrices['click_to_click'] = pickle.load(f)
    else:
        print("创建点击-点击共同访问矩阵...")
        click_df = events_df[events_df['type'] == 'clicks'].copy()
        
        # 按会话分组并按时间戳排序
        session_aids = []
        for session_id, group in click_df.groupby('session'):
            sorted_group = group.sort_values('ts')
            session_aids.append((session_id, sorted_group['aid'].tolist(), sorted_group['ts'].tolist()))
        
        # 创建共同访问矩阵
        click_matrix = defaultdict(Counter)
        time_window = 12 * 60 * 60  # 12小时（秒）
        
        for session_id, aids, timestamps in tqdm(session_aids):
            for i in range(len(aids)):
                for j in range(len(aids)):
                    if i != j and abs(timestamps[i] - timestamps[j]) <= time_window:
                        click_matrix[aids[i]][aids[j]] += 1
        
        matrices['click_to_click'] = click_matrix
        
        # 保存矩阵
        with open(click_matrix_path, 'wb') as f:
            pickle.dump(click_matrix, f)
    
    # 2. 点击到购物车/订单的共同访问矩阵（24小时窗口）
    click_to_cart_matrix_path = os.path.join(output_dir, 'click_to_cart_matrix.pkl')
    if os.path.exists(click_to_cart_matrix_path):
        print(f"加载现有点击-购物车矩阵从 {click_to_cart_matrix_path}")
        with open(click_to_cart_matrix_path, 'rb') as f:
            matrices['click_to_cart'] = pickle.load(f)
    else:
        print("创建点击-购物车共同访问矩阵...")
        # 获取点击事件
        click_df = events_df[events_df['type'] == 'clicks'].copy()
        # 获取购物车和订单事件
        cart_order_df = events_df[events_df['type'].isin(['carts', 'orders'])].copy()
        
        # 创建共同访问矩阵
        click_to_cart_matrix = defaultdict(Counter)
        time_window = 24 * 60 * 60  # 24小时（秒）
        
        # 按会话分组
        for session_id, group in tqdm(events_df.groupby('session')):
            # 该会话的点击事件
            session_clicks = group[group['type'] == 'clicks']
            # 该会话的购物车/订单事件
            session_cart_orders = group[group['type'].isin(['carts', 'orders'])]
            
            if len(session_clicks) == 0 or len(session_cart_orders) == 0:
                continue
            
            # 对于每个点击事件，查找24小时内的购物车/订单事件
            for _, click_row in session_clicks.iterrows():
                click_aid = click_row['aid']
                click_ts = click_row['ts']
                
                for _, cart_order_row in session_cart_orders.iterrows():
                    cart_order_aid = cart_order_row['aid']
                    cart_order_ts = cart_order_row['ts']
                    
                    # 如果购物车/订单事件发生在点击事件后的24小时内
                    if 0 <= cart_order_ts - click_ts <= time_window:
                        click_to_cart_matrix[click_aid][cart_order_aid] += 1
        
        matrices['click_to_cart'] = click_to_cart_matrix
        
        # 保存矩阵
        with open(click_to_cart_matrix_path, 'wb') as f:
            pickle.dump(click_to_cart_matrix, f)
    
    return matrices


def train_word2vec_embeddings(events_df: pd.DataFrame, output_path: str, vector_size: int = 64) -> Word2Vec:
    """训练Word2Vec嵌入以查找相似商品"""
    if os.path.exists(output_path):
        print(f"加载现有Word2Vec模型从 {output_path}")
        return Word2Vec.load(output_path)
    
    print("训练Word2Vec模型...")
    # 将每个会话转化为商品ID序列
    sessions = []
    for session_id, group in tqdm(events_df.groupby('session')):
        # 按时间戳排序
        sorted_group = group.sort_values('ts')
        # 获取序列中的商品ID
        aid_sequence = sorted_group['aid'].astype(str).tolist()
        sessions.append(aid_sequence)
    
    # 训练Word2Vec模型
    model = Word2Vec(
        sentences=sessions,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1,  # 使用Skip-gram
        epochs=5
    )
    
    # 保存模型
    model.save(output_path)
    
    return model


def find_similar_items(model: Word2Vec, item_id: int, top_n: int = 20) -> List[int]:
    """使用Word2Vec模型找到相似商品"""
    try:
        similar_items = model.wv.most_similar(str(item_id), topn=top_n)
        # 转换回整数并返回
        return [int(item_id) for item_id, _ in similar_items]
    except KeyError:
        # 如果商品ID不在模型中
        return []


def generate_revisit_candidates(session_events: List[Dict]) -> List[int]:
    """根据用户之前的交互生成重访候选集"""
    # 获取会话中的所有商品ID
    aids = [event['aid'] for event in session_events]
    # 由于用户可能多次与同一商品交互，我们使用集合去重
    unique_aids = list(set(aids))
    return unique_aids


def generate_candidates_for_session(
    session_events: List[Dict],
    covisit_matrices: Dict[str, Dict[int, Counter]],
    w2v_model: Word2Vec,
    top_n: int = 20
) -> Dict[str, List[int]]:
    """为会话生成所有类型的候选项"""
    # 获取会话中的商品ID
    session_aids = [event['aid'] for event in session_events]
    
    # 1. 重访候选项
    revisit_candidates = generate_revisit_candidates(session_events)
    
    # 2. 共同访问候选项（点击到点击）
    click_covisit_candidates = []
    for aid in session_aids:
        if aid in covisit_matrices.get('click_to_click', {}):
            # 获取与该商品最常共同点击的商品
            top_covisited = [item for item, _ in covisit_matrices['click_to_click'][aid].most_common(top_n)]
            click_covisit_candidates.extend(top_covisited)
    
    # 3. 共同访问候选项（点击到购物车/订单）
    cart_covisit_candidates = []
    for aid in session_aids:
        if aid in covisit_matrices.get('click_to_cart', {}):
            # 获取用户在点击该商品后最常添加到购物车/下单的商品
            top_cart_items = [item for item, _ in covisit_matrices['click_to_cart'][aid].most_common(top_n)]
            cart_covisit_candidates.extend(top_cart_items)
    
    # 4. 相似商品候选项（使用Word2Vec）
    similar_candidates = []
    for aid in session_aids:
        similar_items = find_similar_items(w2v_model, aid, top_n=top_n)
        similar_candidates.extend(similar_items)
    
    # 合并所有候选项并按类型分组
    all_candidates = {
        'clicks': list(set(revisit_candidates + click_covisit_candidates + similar_candidates)),
        'carts': list(set(revisit_candidates + click_covisit_candidates + cart_covisit_candidates + similar_candidates)),
        'orders': list(set(revisit_candidates + cart_covisit_candidates + similar_candidates))
    }
    
    return all_candidates


# ---------------------------------
# 第5部分: 候选排序与特征生成
# ---------------------------------


def create_ranking_features(
    session_events: List[Dict],
    candidates: Dict[str, List[int]],
    covisit_matrices: Dict[str, Dict[int, Counter]],
    w2v_model: Word2Vec,
    item_features_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """为候选项创建排序特征"""
    ranking_features = {}
    
    # 获取会话中的商品ID和事件
    session_aids = [event['aid'] for event in session_events]
    session_id = session_events[0]['session'] if session_events else None
    
    # 按事件类型创建特征
    for event_type in ['clicks', 'carts', 'orders']:
        features_list = []
        
        for candidate_aid in candidates[event_type]:
            # 基本特征
            features = {
                'session': session_id,
                'aid': candidate_aid,
                'event_type': event_type,
                
                # 是否在会话中出现过
                'is_in_session': 1 if candidate_aid in session_aids else 0,
                
                # 在会话中出现的次数
                'occurrence_count': session_aids.count(candidate_aid),
                
                # 最近一次出现的位置 (从会话末尾算起)
                'last_occurrence_pos': len(session_aids) - 1 - session_aids[::-1].index(candidate_aid) if candidate_aid in session_aids else -1,
                
                # 共同访问矩阵特征
                'click_covisit_score': sum(covisit_matrices.get('click_to_click', {}).get(aid, {}).get(candidate_aid, 0) for aid in session_aids),
                'cart_covisit_score': sum(covisit_matrices.get('click_to_cart', {}).get(aid, {}).get(candidate_aid, 0) for aid in session_aids),
                
                # Word2Vec相似度特征
                'w2v_max_similarity': max([w2v_model.wv.similarity(str(aid), str(candidate_aid)) 
                                          for aid in session_aids 
                                          if str(aid) in w2v_model.wv and str(candidate_aid) in w2v_model.wv] or [0]),
                'w2v_avg_similarity': np.mean([w2v_model.wv.similarity(str(aid), str(candidate_aid)) 
                                              for aid in session_aids 
                                              if str(aid) in w2v_model.wv and str(candidate_aid) in w2v_model.wv] or [0])
            }
            
            # 添加商品特征
            if item_features_df is not None and candidate_aid in item_features_df['aid'].values:
                item_row = item_features_df[item_features_df['aid'] == candidate_aid].iloc[0]
                for col in item_features_df.columns:
                    if col != 'aid':
                        features[f'item_{col}'] = item_row[col]
            
            features_list.append(features)
        
        # 转换为DataFrame
        if features_list:
            ranking_features[event_type] = pd.DataFrame(features_list)
        else:
            ranking_features[event_type] = pd.DataFrame()
    
    return ranking_features


def train_ranking_model(train_df: pd.DataFrame, features: List[str], target: str = 'is_target'):
    """训练排序模型"""
    import lightgbm as lgb
    
    # 准备训练数据
    X = train_df[features]
    y = train_df[target]
    
    # 设置模型参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 训练模型
    print(f"训练{target}排序模型...")
    lgb_train = lgb.Dataset(X, y)
    model = lgb.train(params, lgb_train, num_boost_round=100)
    
    # 特征重要性
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    
    print("特征重要性:")
    display(feature_imp.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(10))
    plt.title('特征重要性')
    plt.tight_layout()
    plt.show()
    
    return model


def prepare_training_data(events_df: pd.DataFrame, item_features_df: pd.DataFrame, 
                         covisit_matrices: Dict[str, Dict[int, Counter]], w2v_model: Word2Vec,
                         num_sessions: int = 1000, max_candidates_per_session: int = 100):
    """准备模型训练数据"""
    training_data = {
        'clicks': [],
        'carts': [],
        'orders': []
    }
    
    # 获取唯一会话
    unique_sessions = events_df['session'].unique()
    if len(unique_sessions) > num_sessions:
        # 随机选择会话
        import random
        selected_sessions = random.sample(list(unique_sessions), num_sessions)
    else:
        selected_sessions = unique_sessions
    
    # 为每个会话生成训练数据
    for session_id in tqdm(selected_sessions, desc="准备训练数据"):
        # 获取该会话的所有事件
        session_df = events_df[events_df['session'] == session_id].sort_values('ts')
        
        # 划分为历史和目标
        # 使用时间戳的80%作为划分点
        split_ts = session_df['ts'].min() + 0.8 * (session_df['ts'].max() - session_df['ts'].min())
        
        history_df = session_df[session_df['ts'] < split_ts].copy()
        target_df = session_df[session_df['ts'] >= split_ts].copy()
        
        # 如果历史或目标为空，则跳过
        if len(history_df) == 0 or len(target_df) == 0:
            continue
        
        # 将历史转换为事件列表
        history_events = [
            {
                'session': row['session'],
                'aid': row['aid'],
                'ts': row['ts'],
                'type': row['type']
            }
            for _, row in history_df.iterrows()
        ]
        
        # 获取目标商品
        target_clicks = set(target_df[target_df['type'] == 'clicks']['aid'])
        target_carts = set(target_df[target_df['type'] == 'carts']['aid'])
        target_orders = set(target_df[target_df['type'] == 'orders']['aid'])
        
        # 生成候选项
        candidates = generate_candidates_for_session(
            history_events, covisit_matrices, w2v_model, top_n=max_candidates_per_session
        )
        
        # 为每种事件类型创建排序特征
        ranking_features = create_ranking_features(
            history_events, candidates, covisit_matrices, w2v_model, item_features_df
        )
        
        # 添加目标标签并加入训练数据
        for event_type in ['clicks', 'carts', 'orders']:
            if event_type in ranking_features and not ranking_features[event_type].empty:
                df = ranking_features[event_type].copy()
                
                # 添加目标标签
                if event_type == 'clicks':
                    df['is_target'] = df['aid'].isin(target_clicks).astype(int)
                elif event_type == 'carts':
                    df['is_target'] = df['aid'].isin(target_carts).astype(int)
                else:  # orders
                    df['is_target'] = df['aid'].isin(target_orders).astype(int)
                
                # 添加到训练数据
                training_data[event_type].append(df)
    
    # 合并所有会话的训练数据
    for event_type in training_data:
        if training_data[event_type]:
            training_data[event_type] = pd.concat(training_data[event_type], ignore_index=True)
        else:
            training_data[event_type] = pd.DataFrame()
        
        print(f"{event_type}事件训练数据大小: {len(training_data[event_type])}")
        print(f"正样本比例: {training_data[event_type]['is_target'].mean():.4f}")
    
    return training_data


def predict_for_session(
    session_events: List[Dict],
    covisit_matrices: Dict[str, Dict[int, Counter]],
    w2v_model: Word2Vec,
    item_features_df: pd.DataFrame,
    models: Dict[str, object],
    top_n: int = 20
) -> Dict[str, List[int]]:
    """为会话生成预测"""
    # 生成候选项
    candidates = generate_candidates_for_session(
        session_events, covisit_matrices, w2v_model, top_n=100  # 生成更多候选项以便排序
    )
    
    # 为候选项创建排序特征
    ranking_features = create_ranking_features(
        session_events, candidates, covisit_matrices, w2v_model, item_features_df
    )
    
    # 使用模型预测各个候选项的分数
    predictions = {}
    
    for event_type in ['clicks', 'carts', 'orders']:
        if event_type in ranking_features and not ranking_features[event_type].empty:
            df = ranking_features[event_type]
            
            # 提取特征列
            feature_cols = [col for col in df.columns if col not in ['session', 'aid', 'event_type']]
            
            # 预测分数
            if event_type in models and models[event_type] is not None:
                scores = models[event_type].predict(df[feature_cols])
                
                # 将预测分数与商品ID配对并排序
                scored_items = list(zip(df['aid'].values, scores))
                sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
                
                # 获取前top_n个商品
                predictions[event_type] = [item[0] for item in sorted_items[:top_n]]
            else:
                # 如果没有模型，则使用基于规则的候选项
                predictions[event_type] = candidates[event_type][:top_n]
        else:
            # 如果没有候选项，则返回空列表
            predictions[event_type] = []
    
    return predictions


def create_submission(
    test_path: str,
    covisit_matrices: Dict[str, Dict[int, Counter]],
    w2v_model: Word2Vec,
    item_features_df: pd.DataFrame,
    models: Dict[str, object],
    output_path: str = 'submission.csv',
    batch_size: int = 100
) -> None:
    """创建比赛提交文件"""
    # 打开输出文件
    with open(output_path, 'w') as f:
        # 写入标题行
        f.write('session_type,labels\n')
        
        # 分批处理测试数据
        batch_count = 0
        for session_batch in stream_jsonl(test_path, batch_size=batch_size):
            batch_count += 1
            print(f"处理测试批次 {batch_count}...")
            
            for session in tqdm(session_batch, desc=f"批次 {batch_count}"):
                session_id = session['session']
                
                # 预测
                predictions = predict_for_session(
                    session['events'], covisit_matrices, w2v_model, 
                    item_features_df, models
                )
                
                # 写入预测结果
                for event_type in ['clicks', 'carts', 'orders']:
                    # 确保至少有一些预测
                    preds = predictions.get(event_type, [])
                    if len(preds) < 20:
                        # 如果预测不足20个，则使用基于规则的方法补充
                        rule_based = generate_candidates_for_session(
                            session['events'], covisit_matrices, w2v_model
                        )[event_type]
                        
                        # 添加不在当前预测中的商品
                        for item in rule_based:
                            if item not in preds and len(preds) < 20:
                                preds.append(item)
                    
                    # 格式化预测结果
                    labels_str = ' '.join(map(str, preds[:20]))  # 最多20个预测
                    f.write(f"{session_id};{event_type},{labels_str}\n")
            
            # 释放内存
            gc.collect()
    
    print(f"提交文件已保存至 {output_path}")


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str, batch_size: int = 10000) -> None:
    """Convert a JSONL file to Parquet format for better I/O performance."""
    import os
    import pandas as pd
    
    print(f"Converting {jsonl_path} to Parquet format...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    # Remove the file if it exists
    if os.path.exists(parquet_path):
        print(f"Removing existing file: {parquet_path}")
        os.remove(parquet_path)
    
    # Process in batches
    all_data = []
    batch_idx = 0
    
    for events_batch in stream_events(jsonl_path, batch_size=batch_size):
        all_data.extend(events_batch)
        batch_idx += 1
        print(f"Processed batch {batch_idx}")
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(all_data)
    df.to_parquet(
        parquet_path, 
        engine='pyarrow', 
        index=False, 
        compression='snappy'
    )
    
    print(f"Conversion completed: {parquet_path}")


def evaluate_predictions(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    weight_clicks: float = 0.10,
    weight_carts: float = 0.30,
    weight_orders: float = 0.60
) -> Dict[str, float]:
    """评估预测结果"""
    results = {}
    
    # 为每种类型计算召回率
    for event_type in ['clicks', 'carts', 'orders']:
        type_truth = truth_df[truth_df['type'] == event_type]
        
        # 会话和真实标签的映射
        session_labels = {}
        for _, row in type_truth.iterrows():
            session_id = row['session']
            aid = row['aid']
            if session_id not in session_labels:
                session_labels[session_id] = []
            session_labels[session_id].append(aid)
        
        # 获取预测
        type_preds = pred_df[pred_df['type'] == event_type]
        
        # 计算召回率
        recall_numerator = 0
        recall_denominator = 0
        
        for session_id, true_aids in session_labels.items():
            # 获取该会话的预测
            session_preds = type_preds[type_preds['session'] == session_id]
            if len(session_preds) > 0:
                pred_aids = session_preds.iloc[0]['preds']
                # 计算交集大小
                hits = len(set(true_aids).intersection(set(pred_aids)))
                recall_numerator += hits
                recall_denominator += min(20, len(true_aids))
        
        # 计算最终召回率
        if recall_denominator > 0:
            recall = recall_numerator / recall_denominator
        else:
            recall = 0
        
        results[f'recall_{event_type}'] = recall
    
    # 计算加权得分
    weighted_score = (
        results.get('recall_clicks', 0) * weight_clicks +
        results.get('recall_carts', 0) * weight_carts +
        results.get('recall_orders', 0) * weight_orders
    )
    results['weighted_score'] = weighted_score
    
    return results    


"""主函数"""
# 路径配置
working_dir = '/kaggle/working/'
data_dir = '/kaggle/input/otto-recommender-system/'
processed_dir = os.path.join(working_dir, 'processed')
matrices_dir = os.path.join(processed_dir, 'matrices')
models_dir = os.path.join(processed_dir, 'models')
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(matrices_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
    
train_path = os.path.join(data_dir, 'train.jsonl')
test_path = os.path.join(data_dir, 'test.jsonl')
    
# 1. 数据预处理
# 如果需要，将JSONL转换为Parquet
train_parquet = os.path.join(processed_dir, 'train_events.parquet')
test_parquet = os.path.join(processed_dir, 'test_events.parquet')


# if not os.path.exists(train_parquet):
convert_jsonl_to_parquet(train_path, train_parquet, batch_size=5000)


if not os.path.exists(test_parquet):
    convert_jsonl_to_parquet(test_path, test_parquet, batch_size=5000)



    # # 2. 加载数据
    # print("加载训练数据...")
    # train_df = pd.read_parquet(train_parquet)
    
    # # 对于大数据集，可以取样本加速开发过程
    # # train_df = train_df.sample(frac=0.1, random_state=42)
    
    # # 3. 特征工程
    # # 创建会话特征
    # session_features_path = os.path.join(processed_dir, 'session_features.parquet')
    # if os.path.exists(session_features_path):
    #     session_features = pd.read_parquet(session_features_path)
    # else:
    #     print("创建会话特征...")
    #     session_features = create_session_features(train_df)
    #     session_features.to_parquet(session_features_path, index=False)
    
    # # 创建商品特征
    # item_features_path = os.path.join(processed_dir, 'item_features.parquet')
    # if os.path.exists(item_features_path):
    #     item_features = pd.read_parquet(item_features_path)
    # else:
    #     print("创建商品特征...")
    #     item_features = create_item_features(train_df)
    #     item_features.to_parquet(item_features_path, index=False)
    
    # # 4. 候选生成
    # # 创建共同访问矩阵
    # print("创建共同访问矩阵...")
    # covisit_matrices = create_covisitation_matrices(train_df, matrices_dir)
    
    # # 训练Word2Vec模型
    # w2v_path = os.path.join(processed_dir, 'word2vec_model.bin')
    # w2v_model = train_word2vec_embeddings(train_df, w2v_path, vector_size=64)
    
    # # 5. 准备训练数据
    # # 这一步可能会消耗大量内存，根据需要调整参数
    # training_data_path = os.path.join(processed_dir, 'training_data.pkl')
    # if os.path.exists(training_data_path):
    #     print(f"加载现有训练数据从 {training_data_path}")
    #     with open(training_data_path, 'rb') as f:
    #         training_data = pickle.load(f)
    # else:
    #     print("准备训练数据...")
    #     training_data = prepare_training_data(
    #         train_df, item_features, covisit_matrices, w2v_model,
    #         num_sessions=10000, max_candidates_per_session=100
    #     )
        
    #     # 保存训练数据
    #     with open(training_data_path, 'wb') as f:
    #         pickle.dump(training_data, f)
    
    # # 6. 训练排序模型
    # models = {}
    
    # for event_type in ['clicks', 'carts', 'orders']:
    #     # 如果有足够的训练数据，则训练模型
    #     if event_type in training_data and len(training_data[event_type]) > 0:
    #         df = training_data[event_type]
            
    #         # 选择特征列
    #         feature_cols = [col for col in df.columns 
    #                         if col not in ['session', 'aid', 'event_type', 'is_target']]
            
    #         # 训练模型
    #         model_path = os.path.join(models_dir, f'{event_type}_model.pkl')
    #         if os.path.exists(model_path):
    #             print(f"加载现有{event_type}模型从 {model_path}")
    #             with open(model_path, 'rb') as f:
    #                 models[event_type] = pickle.load(f)
    #         else:
    #             print(f"训练{event_type}模型...")
    #             models[event_type] = train_ranking_model(df, feature_cols, 'is_target')
                
    #             # 保存模型
    #             with open(model_path, 'wb') as f:
    #                 pickle.dump(models[event_type], f)
    #     else:
    #         print(f"没有足够的{event_type}训练数据，将使用基于规则的方法")
    #         models[event_type] = None
    
    # # 7. 创建提交文件
    # submission_path = 'submission.csv'
    # print("创建提交文件...")
    # create_submission(
    #     test_path, covisit_matrices, w2v_model, item_features, models,
    #     output_path=submission_path, batch_size=100
    # )
    
    # print(f"完成！提交文件已保存至 {submission_path}")

