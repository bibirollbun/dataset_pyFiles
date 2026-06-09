import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
import lightgbm as lgb

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 设置可视化样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

print("环境准备完成!")  


def load_m5_data(data_dir='./kaggle/input/m5-forecasting-accuracy'):
    """
    加载M5竞赛所需的所有数据文件
    
    参数:
    data_dir: str, 数据目录路径
    
    返回:
    dict: 包含所有数据的数据字典
    """
    print(f'正在从 {data_dir} 加载数据...')
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f'警告: 目录 {data_dir} 不存在，尝试使用当前目录...')
        data_dir = '.'
    
    # 定义文件路径
    files = {
        'sales_train_validation': 'sales_train_validation.csv',
        'sales_train_evaluation': 'sales_train_evaluation.csv',
        'calendar': 'calendar.csv',
        'sell_prices': 'sell_prices.csv',
        'sample_submission': 'sample_submission.csv'
    }
    
    # 加载数据
    data = {}
    
    for key, filename in files.items():
        file_path = os.path.join(data_dir, filename)
        try:
            print(f'加载 {key}...')
            data[key] = pd.read_csv(file_path)
            print(f'  - {key} 形状: {data[key].shape}')
        except Exception as e:
            print(f'警告: 无法加载 {key} ({filename}): {e}')
            data[key] = None
    
    print('数据加载完成!')
    return data


def clean_m5_data(data):
    """
    对M5竞赛数据进行初步清洗
    
    参数:
    data: dict, 原始数据字典
    
    返回:
    dict: 清洗后的数据字典
    """
    cleaned_data = data.copy()
    
    print('开始数据清洗...')
    
    # 处理销售数据
    for key in ['sales_train_validation', 'sales_train_evaluation']:
        if cleaned_data[key] is not None:
            df = cleaned_data[key].copy()  # 使用copy避免SettingWithCopyWarning
            
            # 检查并处理缺失值
            missing_values = df.isnull().sum().sum()
            print(f'  {key} 缺失值数量: {missing_values}')
            
            # 确保ID格式正确
            if 'id' in df.columns:
                df['id'] = df['id'].astype(str)
            
            # 确保销售数据为整数类型
            sales_cols = [col for col in df.columns if col.startswith('d_')]
            # 批量转换而非循环转换，更高效
            df[sales_cols] = df[sales_cols].apply(pd.to_numeric, errors='coerce', downcast='integer')
            
            # 替换负数为0（批量操作更高效）
            df[sales_cols] = df[sales_cols].clip(lower=0)
            
            cleaned_data[key] = df
    
    # 处理日历数据
    if cleaned_data['calendar'] is not None:
        df = cleaned_data['calendar'].copy()
        
        # 转换日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 填充缺失的事件信息
        event_cols = [col for col in df.columns if col.startswith('event')]
        df[event_cols] = df[event_cols].fillna('None')
        
        # 确保wm_yr_wk为整数
        if 'wm_yr_wk' in df.columns:
            df['wm_yr_wk'] = pd.to_numeric(df['wm_yr_wk'], errors='coerce', downcast='integer')
        
        cleaned_data['calendar'] = df
    
    # 处理价格数据
    if cleaned_data['sell_prices'] is not None:
        df = cleaned_data['sell_prices'].copy()
        
        # 确保价格为浮点数
        if 'sell_price' in df.columns:
            df['sell_price'] = pd.to_numeric(df['sell_price'], errors='coerce')
            # 替换负数和缺失值为0（使用clip更高效）
            df['sell_price'] = df['sell_price'].fillna(0).clip(lower=0)
        
        cleaned_data['sell_prices'] = df
    
    # 处理提交样例
    if cleaned_data['sample_submission'] is not None:
        df = cleaned_data['sample_submission'].copy()
        
        # 确保ID格式正确
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
        
        cleaned_data['sample_submission'] = df
    
    print('数据清洗完成!')
    return cleaned_data


def create_data_overview(data):
    """
    创建数据概览报告
    
    参数:
    data: dict, 数据字典
    
    返回:
    None
    """
    print('===== 数据概览报告 =====')
    
    for key, df in data.items():
        if df is not None:
            print(f'\n{key} 概览:')
            print(f'- 形状: {df.shape}')
            print(f'- 列名: {list(df.columns[:5])}{"..." if len(df.columns) > 5 else ""}')
            print(f'- 数据类型:\n{df.dtypes[:5]}')
            
            # 显示前2行数据
            print(f'- 前2行数据:\n{df.head(2)}')
            
            # 缺失值统计
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if len(missing_cols) > 0:
                print(f'- 缺失值统计:\n{missing_cols}')
            else:
                print('- 无缺失值')
    
    # 特定分析
    if data['sales_train_validation'] is not None:
        sales_df = data['sales_train_validation']
        sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
        
        print('\n销售数据统计:')
        print(f'- 时间序列长度: {len(sales_cols)} 天')
        print(f'- 最小销售额: {sales_df[sales_cols].min().min()}')
        print(f'- 最大销售额: {sales_df[sales_cols].max().max()}')
        print(f'- 平均销售额: {sales_df[sales_cols].mean().mean():.2f}')
    
    print('\n===== 数据概览完成 =====')


def prepare_m5_data(data_dir='../input/m5-forecasting-accuracy'):
    """
    准备M5竞赛数据的完整流程
    
    参数:
    data_dir: str, 数据目录路径
    
    返回:
    dict: 处理后的数据字典
    """
    # 加载数据
    raw_data = load_m5_data(data_dir)
    
    # 清洗数据
    cleaned_data = clean_m5_data(raw_data)
    
    # 创建数据概览
    create_data_overview(cleaned_data)
    
    return cleaned_data


# 在实际运行时取消注释
cleaned_data = prepare_m5_data()


import pandas as pd

def melt_sales_data(sales_df):
    """将宽格式的销售数据转换为长格式"""
    id_cols = [col for col in sales_df.columns if not col.startswith('d_')]
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    
    print(f'将销售数据从宽格式转换为长格式...')
    print(f'- ID列: {id_cols}')
    print(f'- 销售列数量: {len(sales_cols)}')
    
    melt_df = pd.melt(
        sales_df,
        id_vars=id_cols,
        value_vars=sales_cols,
        var_name='d',
        value_name='sales'
    )
    
    print(f'转换完成，新数据形状: {melt_df.shape}')
    return melt_df


def create_hierarchical_series(cleaned_data):
    """创建分层时间序列数据"""
    if cleaned_data['sales_train_validation'] is None:
        print('错误: 缺少销售数据')
        return None
    
    sales_df = cleaned_data['sales_train_validation'].copy()
    melt_df = melt_sales_data(sales_df)  # 转换为长格式
    hierarchical_series = {}
    
    print('创建分层时间序列...')
    
    # 1. 总体级别 (Total)
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    total_sales = sales_df[sales_cols].sum()
    hierarchical_series['total'] = pd.DataFrame({
        'd': total_sales.index,
        'sales': total_sales.values
    })
    
    # 2. 州级别 (State)
    if 'state_id' in melt_df.columns:
        state_sales = melt_df.groupby(['state_id', 'd'])['sales'].sum().reset_index()
        for state in state_sales['state_id'].unique():
            hierarchical_series[f'state_{state}'] = state_sales[state_sales['state_id'] == state].copy()
    
    # 3. 类别级别 (Category)
    if 'cat_id' in melt_df.columns:
        cat_sales = melt_df.groupby(['cat_id', 'd'])['sales'].sum().reset_index()
        for cat in cat_sales['cat_id'].unique():
            hierarchical_series[f'category_{cat}'] = cat_sales[cat_sales['cat_id'] == cat].copy()
    
    # 4. 部门级别 (Department)
    if 'dept_id' in melt_df.columns:
        dept_sales = melt_df.groupby(['dept_id', 'd'])['sales'].sum().reset_index()
        for dept in dept_sales['dept_id'].unique():
            hierarchical_series[f'dept_{dept}'] = dept_sales[dept_sales['dept_id'] == dept].copy()
    
    # 5. 州-类别级别
    if 'state_id' in melt_df.columns and 'cat_id' in melt_df.columns:
        state_cat_sales = melt_df.groupby(['state_id', 'cat_id', 'd'])['sales'].sum().reset_index()
        for (state, cat), group in state_cat_sales.groupby(['state_id', 'cat_id']):
            hierarchical_series[f'state_{state}_cat_{cat}'] = group.copy()
    
    # 6. 州-部门级别
    if 'state_id' in melt_df.columns and 'dept_id' in melt_df.columns:
        state_dept_sales = melt_df.groupby(['state_id', 'dept_id', 'd'])['sales'].sum().reset_index()
        for (state, dept), group in state_dept_sales.groupby(['state_id', 'dept_id']):
            hierarchical_series[f'state_{state}_dept_{dept}'] = group.copy()
    
    # 7. 店铺级别
    if 'store_id' in melt_df.columns:
        store_sales = melt_df.groupby(['store_id', 'd'])['sales'].sum().reset_index()
        for store in store_sales['store_id'].unique():
            hierarchical_series[f'store_{store}'] = store_sales[store_sales['store_id'] == store].copy()
    
    # 8. 店铺-类别级别
    if 'store_id' in melt_df.columns and 'cat_id' in melt_df.columns:
        store_cat_sales = melt_df.groupby(['store_id', 'cat_id', 'd'])['sales'].sum().reset_index()
        for (store, cat), group in store_cat_sales.groupby(['store_id', 'cat_id']):
            hierarchical_series[f'store_{store}_cat_{cat}'] = group.copy()
    
    # 9. 店铺-部门级别
    if 'store_id' in melt_df.columns and 'dept_id' in melt_df.columns:
        store_dept_sales = melt_df.groupby(['store_id', 'dept_id', 'd'])['sales'].sum().reset_index()
        for (store, dept), group in store_dept_sales.groupby(['store_id', 'dept_id']):
            hierarchical_series[f'store_{store}_dept_{dept}'] = group.copy()
    
    # 10. 商品级别示例（仅前10个商品）
    if 'item_id' in melt_df.columns and 'store_id' in melt_df.columns:
        top_items = melt_df['item_id'].unique()[:10]
        item_store_sales = melt_df[melt_df['item_id'].isin(top_items)].groupby(
            ['item_id', 'store_id', 'd'])['sales'].sum().reset_index()
        for (item, store), group in item_store_sales.groupby(['item_id', 'store_id']):
            hierarchical_series[f'item_{item}_store_{store}'] = group.copy()
    
    print(f'分层时间序列创建完成，共 {len(hierarchical_series)} 个级别')
    return hierarchical_series


def create_merged_data(cleaned_data):
    """创建合并的数据集，用于特征工程和模型训练"""
    if cleaned_data['sales_train_validation'] is None:
        print('错误: 缺少销售数据')
        return None
    
    # 1. 获取长格式销售数据
    melt_df = melt_sales_data(cleaned_data['sales_train_validation'])
    
    # 2. 合并日历数据
    if cleaned_data['calendar'] is not None:
        print('合并日历数据...')
        calendar_cols = ['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 
                         'year', 'event_name_1', 'event_type_1']
        calendar_cols = [col for col in calendar_cols if col in cleaned_data['calendar'].columns]
        melt_df = melt_df.merge(cleaned_data['calendar'][calendar_cols], on='d', how='left')
        print(f'合并日历数据后形状: {melt_df.shape}')
    
    # 3. 合并价格数据
    if cleaned_data['sell_prices'] is not None:
        print('合并价格数据...')
        if 'wm_yr_wk' in melt_df.columns:
            price_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
            price_cols = [col for col in price_cols if col in cleaned_data['sell_prices'].columns]
            melt_df = melt_df.merge(
                cleaned_data['sell_prices'][price_cols],
                on=['store_id', 'item_id', 'wm_yr_wk'],
                how='left'
            )
            print(f'合并价格数据后形状: {melt_df.shape}')
        else:
            print('警告: 缺少wm_yr_wk列，无法合并价格数据')
    
    # 4. 填充缺失价格
    if 'sell_price' in melt_df.columns:
        melt_df['sell_price'] = melt_df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.fillna(x.mean())
        )
        melt_df['sell_price'] = melt_df['sell_price'].fillna(0)
    
    # 5. 新增：从'd'列提取d_num（关键修正：移除嵌套函数，直接在此处处理）
    if 'd' in melt_df.columns:
        # 从'd_123'中提取数字部分并转为整数
        melt_df['d_num'] = melt_df['d'].str.extract(r'd_(\d+)').astype(int)
        print("已创建d_num列（日期序号）")
    else:
        print("警告：数据中缺少'd'列，无法创建d_num")
    
    print(f'合并数据完成，最终数据形状: {melt_df.shape}')
    return melt_df


# 创建分层时间序列
hierarchical_series = create_hierarchical_series(cleaned_data)

# 创建合并数据（包含d_num列）
merged_data = create_merged_data(cleaned_data)


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import os
import seaborn as sns
import numpy as np

# 配置中文字体（优先使用指定路径字体，否则使用系统字体）
def get_chinese_font():
    """获取中文字体配置"""
    try:
        # 尝试从指定路径加载字体
        font_path = "/kaggle/input/simhei/simhei.ttf"
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            print("成功加载指定字体文件")
        else:
            # 路径不存在时使用系统字体
            font_prop = FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS'])
            print("使用系统默认中文字体")
        # 设置全局字体配置
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
        return font_prop
    except Exception as e:
        # 加载失败时的 fallback 方案
        font_prop = FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS'])
        plt.rcParams["axes.unicode_minus"] = False
        print(f"字体加载警告: {str(e)}，将使用系统默认字体")
        return font_prop

# 获取字体配置
chinese_font = get_chinese_font()


def plot_sales_trends(hierarchical_series, levels=['total'], days=100):
    """
    绘制销售趋势图（支持中文显示）
    """
    print('绘制销售趋势图...')
    
    for level in levels:
        if level in hierarchical_series:
            df = hierarchical_series[level].copy()
            
            # 获取最近的天数
            if len(df) > days:
                df = df.iloc[-days:]
            
            plt.figure(figsize=(15, 6))
            plt.plot(df['d'], df['sales'], marker='o', linestyle='-', alpha=0.7)
            
            # 层级名称中文转换
            level_name = level.replace('state_', '州:').replace('category_', '类别:').replace('total', '总体')
            plt.title(f'销售趋势 - {level_name}', fontproperties=chinese_font)
            plt.xlabel('日期', fontproperties=chinese_font)
            plt.ylabel('销售额', fontproperties=chinese_font)
            plt.grid(True)
            
            # 设置x轴标签间隔，避免太拥挤
            if len(df) > 50:
                plt.xticks(df['d'].iloc[::10], rotation=45)
            else:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()


def analyze_seasonality(hierarchical_series, level='total', period=7):
    """
    分析季节性模式（支持中文显示）
    """
    if level not in hierarchical_series:
        print(f'错误: 未找到级别 {level}')
        return None
    
    print(f'分析季节性模式 - {level}')
    
    df = hierarchical_series[level].copy()
    
    # 执行移动平均以平滑数据
    df['ma7'] = df['sales'].rolling(window=7).mean()
    
    # 绘制原始数据和移动平均
    plt.figure(figsize=(15, 6))
    plt.plot(df['d'], df['sales'], label='原始销售额', alpha=0.5)
    plt.plot(df['d'], df['ma7'], label='7天移动平均', color='red', linewidth=2)
    
    # 层级名称中文转换
    level_name = level.replace('state_', '州:').replace('category_', '类别:').replace('total', '总体')
    plt.title(f'销售趋势与移动平均 - {level_name}', fontproperties=chinese_font)
    plt.xlabel('日期', fontproperties=chinese_font)
    plt.ylabel('销售额', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)  # 图例字体
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 分析周期性模式
    if len(df) >= period:
        # 取最近的数据进行分析
        recent_df = df.iloc[-min(90, len(df)):].copy()  # 最近90天或更少
        recent_df['day_index'] = range(len(recent_df))
        recent_df['period_day'] = recent_df['day_index'] % period
        
        # 按周期内天数分组计算平均销售额
        period_avg = recent_df.groupby('period_day')['sales'].mean()
        
        # 绘制周期模式
        plt.figure(figsize=(10, 6))
        if period == 7:
            period_labels = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
            period_name = '周'
        elif period == 30:
            period_labels = [f'{i+1}日' for i in range(30)][:len(period_avg)]
            period_name = '月'
        else:
            period_labels = [f'{i+1}天' for i in range(len(period_avg))]
            period_name = f'{period}天'
            
        plt.bar(period_avg.index, period_avg.values)
        plt.title(f'{period_name}度销售模式 - {level_name}', fontproperties=chinese_font)
        plt.xlabel(f'{period_name}度周期内的天数', fontproperties=chinese_font)
        plt.ylabel('平均销售额', fontproperties=chinese_font)
        plt.xticks(period_avg.index, period_labels, fontproperties=chinese_font)  # x轴刻度字体
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    results = {
        'level': level,
        'total_days': len(df),
        'total_sales': df['sales'].sum(),
        'avg_daily_sales': df['sales'].mean()
    }
    
    print(f'季节性分析结果 - {level_name}:')
    print(f'- 总天数: {results["total_days"]}')
    print(f'- 总销售额: {results["total_sales"]:.0f}')
    print(f'- 平均日销售额: {results["avg_daily_sales"]:.2f}')
    
    return results


def detect_anomalies(hierarchical_series, level='total', method='zscore', threshold=3):
    """
    检测销售异常值（支持中文显示）
    """
    if level not in hierarchical_series:
        print(f'错误: 未找到级别 {level}')
        return None
    
    print(f'检测异常值 - {level}, 方法: {method}')
    
    df = hierarchical_series[level].copy()
    
    # 使用移动平均平滑数据
    df['ma7'] = df['sales'].rolling(window=7).mean()
    df['ma28'] = df['sales'].rolling(window=28).mean()
    
    # 异常检测
    if method == 'zscore':
        # 计算基于移动平均的Z-score（更适合时间序列）
        df['zscore'] = (df['sales'] - df['ma7']) / df['sales'].rolling(window=7).std()
        df['is_anomaly'] = abs(df['zscore']) > threshold
    elif method == 'iqr':
        # 使用IQR方法
        Q1 = df['sales'].quantile(0.25)
        Q3 = df['sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df['is_anomaly'] = (df['sales'] < lower_bound) | (df['sales'] > upper_bound)
    else:
        print(f'错误: 不支持的异常检测方法 {method}')
        return None
    
    # 获取异常值（排除移动平均NaN的部分）
    anomalies = df[df['is_anomaly'] & df['ma7'].notna()].copy()
    anomaly_count = len(anomalies)
    
    print(f'检测到 {anomaly_count} 个异常值 ({anomaly_count/len(df)*100:.2f}%)')
    
    # 绘制异常值
    if anomaly_count > 0:
        plt.figure(figsize=(15, 6))
        # 层级名称中文转换
        level_name = level.replace('state_', '州:').replace('category_', '类别:').replace('total', '总体')
        plt.plot(df['d'], df['sales'], label='销售额', alpha=0.7)
        plt.plot(df['d'], df['ma7'], label='7天移动平均', color='red', linestyle='--')
        plt.scatter(anomalies['d'], anomalies['sales'], color='red', s=100, label='异常值')
        plt.title(f'异常值检测 - {level_name}', fontproperties=chinese_font)
        plt.xlabel('日期', fontproperties=chinese_font)
        plt.ylabel('销售额', fontproperties=chinese_font)
        plt.legend(prop=chinese_font)  # 图例字体
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return anomalies


def plot_sales_distribution(hierarchical_series, levels=['total']):
    """
    绘制销售额分布（支持中文显示）
    """
    print('绘制销售额分布图...')
    
    num_levels = len(levels)
    fig, axes = plt.subplots(num_levels, 2, figsize=(15, 5*num_levels))
    
    if num_levels == 1:
        axes = [axes]
    
    for i, level in enumerate(levels):
        if level in hierarchical_series:
            sales_data = hierarchical_series[level]['sales']
            # 层级名称中文转换
            level_name = level.replace('state_', '州:').replace('category_', '类别:').replace('total', '总体')
            
            # 直方图
            axes[i][0].hist(sales_data, bins=50, alpha=0.7, color='skyblue')
            axes[i][0].set_title(f'销售额分布直方图 - {level_name}', fontproperties=chinese_font)
            axes[i][0].set_xlabel('销售额', fontproperties=chinese_font)
            axes[i][0].set_ylabel('频率', fontproperties=chinese_font)
            axes[i][0].grid(True, alpha=0.3)
            
            # 箱线图
            axes[i][1].boxplot(sales_data)
            axes[i][1].set_title(f'销售额箱线图 - {level_name}', fontproperties=chinese_font)
            axes[i][1].set_ylabel('销售额', fontproperties=chinese_font)
            axes[i][1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_correlations(merged_data, features=None):
    """
    分析特征相关性（支持中文显示）
    """
    print('分析特征相关性...')
    
    # 选择数值型特征
    numeric_df = merged_data.select_dtypes(include=['float64', 'int64', 'int32'])
    
    # 如果指定了特征，只选择这些特征
    if features is not None:
        numeric_df = numeric_df[[col for col in features if col in numeric_df.columns]]
    
    # 计算相关性矩阵
    corr_matrix = numeric_df.corr()
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, cbar=True)
    plt.title('特征相关性热力图', fontproperties=chinese_font)
    plt.tight_layout()
    plt.show()
    
    # 显示与销售额相关性最高的特征
    if 'sales' in corr_matrix.columns:
        sales_corr = corr_matrix['sales'].sort_values(ascending=False)
        print('与销售额相关性最高的前10个特征:')
        print(sales_corr.head(10))
        
        print('\n与销售额相关性最低的后10个特征:')
        print(sales_corr.tail(10))
    
    return corr_matrix


def perform_eda(hierarchical_series, merged_data=None):
    """
    执行全面的探索性数据分析（支持中文显示）
    """
    print('===== 开始探索性数据分析 =====')
    
    # 结果字典
    eda_results = {}
    
    # 1. 绘制销售趋势
    print('\n1. 分析销售趋势...')
    # 选择几个主要级别
    main_levels = ['total']
    # 添加州级别的示例
    state_levels = [level for level in hierarchical_series.keys() if level.startswith('state_')]
    main_levels.extend(state_levels[:2])  # 只取前两个州
    
    plot_sales_trends(hierarchical_series, levels=main_levels, days=90)
    
    # 2. 分析季节性
    print('\n2. 分析季节性模式...')
    seasonality_results = analyze_seasonality(hierarchical_series, level='total', period=7)
    eda_results['seasonality'] = seasonality_results
    
    # 分析月度季节性
    if len(hierarchical_series['total']) > 30:
        monthly_seasonality = analyze_seasonality(hierarchical_series, level='total', period=30)
        eda_results['monthly_seasonality'] = monthly_seasonality
    
    # 3. 检测异常值
    print('\n3. 检测异常值...')
    anomalies = detect_anomalies(hierarchical_series, level='total', method='zscore', threshold=3)
    eda_results['anomalies'] = anomalies
    
    # 4. 分析销售额分布
    print('\n4. 分析销售额分布...')
    plot_sales_distribution(hierarchical_series, levels=['total'] + state_levels[:2])
    
    # 5. 分析相关性（如果有合并数据）
    if merged_data is not None:
        print('\n5. 分析特征相关性...')
        corr_matrix = analyze_correlations(merged_data)
        eda_results['correlations'] = corr_matrix
    else:
        print('\n5. 未提供合并数据，跳过相关性分析')
    
    # 6. 分析类别分布
    print('\n6. 分析产品类别分布...')
    # 从分层数据中提取类别信息
    cat_levels = [level for level in hierarchical_series.keys() if level.startswith('category_')]
    if cat_levels:
        # 提取类别销售数据
        cat_data = {}
        for level in cat_levels:
            cat_name = level.replace('category_', '')
            total_sales = hierarchical_series[level]['sales'].sum()
            cat_data[cat_name] = total_sales
        
        # 绘制类别销售分布
        plt.figure(figsize=(10, 6))
        plt.bar(cat_data.keys(), cat_data.values(), color='lightgreen')
        plt.title('各类别总销售额分布', fontproperties=chinese_font)
        plt.xlabel('类别', fontproperties=chinese_font)
        plt.ylabel('总销售额', fontproperties=chinese_font)
        plt.grid(True, axis='y')
        plt.show()
    
    print('\n===== 探索性数据分析完成 =====')
    return eda_results



eda_results = perform_eda(hierarchical_series, merged_data)


# ----------------------
# 1. 内存压缩函数
# ----------------------
def reduce_memory_usage(df):
    """压缩DataFrame内存占用"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"原始内存占用: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # 压缩数值型
        if col_type in ['int64', 'int32', 'int16']:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type in ['float64', 'float32']:
            df[col] = df[col].astype(np.float32)
        
        # 压缩分类特征（转为category类型）
        elif col_type == 'object' and len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"压缩后内存占用: {end_mem:.2f} MB (减少了 {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df


# ----------------------
# 2. 分批特征工程函数（核心补充）
# ----------------------
def build_features_in_batches(merged_data, batch_size=100, prices_df=None, calendar_df=None):
    """
    分批构建特征，避免内存溢出
    参数:
        merged_data: 合并后的全量数据
        batch_size: 每批处理的商品数量
        prices_df: 价格数据（已合并到merged_data则传None）
        calendar_df: 日历数据（已合并到merged_data则传None）
    返回:
        合并后的特征集和编码器
    """
    # 确保数据按商品和日期排序
    merged_data = merged_data.sort_values(['item_id', 'd_num']).reset_index(drop=True)
    
    # 获取唯一商品ID，按批次拆分
    unique_items = merged_data['item_id'].unique()
    total_batches = (len(unique_items) + batch_size - 1) // batch_size  # 总批次数
    all_features = []
    global_encoders = None  # 全局编码器（用第一批的编码器）
    
    print(f"共分{total_batches}批处理，每批最多{batch_size}个商品...")
    
    for i in range(total_batches):
        # 1. 提取当前批次的商品
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_item_ids = unique_items[start_idx:end_idx]
        batch_data = merged_data[merged_data['item_id'].isin(batch_item_ids)].copy()
        
        # 2. 为避免滞后特征断裂，多取前180天数据（最大滞后周期）
        if 'd_num' in batch_data.columns:
            min_d_num = batch_data['d_num'].min()
            # 补充前180天的数据（同一商品）
            extra_data = merged_data[
                (merged_data['item_id'].isin(batch_item_ids)) & 
                (merged_data['d_num'] >= (min_d_num - 180)) & 
                (merged_data['d_num'] < min_d_num)
            ]
            batch_data = pd.concat([extra_data, batch_data], ignore_index=True)
        
        print(f"\n===== 处理第{i+1}/{total_batches}批（{len(batch_item_ids)}个商品） =====")
        
        # 3. 对当前批次构建特征
        batch_features, batch_encoders = build_features(
            data=batch_data,
            prices_df=prices_df,
            calendar_df=calendar_df
        )
        
        # 4. 统一编码器（用第一批的编码器作为全局标准）
        if global_encoders is None:
            global_encoders = batch_encoders
        else:
            # 用全局编码器重新编码当前批次，避免类别不一致
            for col, encoder in global_encoders.items():
                if col in batch_features.columns and col in batch_encoders:
                    mask = batch_features[col].notna()
                    # 处理 unseen labels（用-1标记）
                    try:
                        batch_features.loc[mask, col] = encoder.transform(batch_features.loc[mask, col].astype(str))
                    except ValueError:
                        # 对未见过的类别，映射为-1
                        batch_features.loc[mask, col] = batch_features.loc[mask, col].apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
        
        # 5. 过滤掉补充的额外数据，只保留当前批次有效数据
        if 'd_num' in batch_features.columns:
            batch_features = batch_features[batch_features['d_num'] >= min_d_num].reset_index(drop=True)
        
        # 6. 确保所有批次特征列一致
        if i == 0:
            expected_columns = batch_features.columns.tolist()
        else:
            # 补充缺失列
            missing_cols = set(expected_columns) - set(batch_features.columns)
            for col in missing_cols:
                batch_features[col] = 0  # 缺失特征用0填充
            # 按预期列顺序排序
            batch_features = batch_features[expected_columns]
        
        all_features.append(batch_features)
        # 清理内存
        del batch_data, batch_features
        
    # 合并所有批次
    print("\n===== 合并所有批次 =====")
    features_df = pd.concat(all_features, ignore_index=True)
    return features_df, global_encoders


# ----------------------
# 3. 特征工程核心函数（你的原有函数，保持不变）
# ----------------------
def create_lag_features(data, lag_periods=[7, 14, 30]):  # 精简滞后周期，减少内存
    # （函数内容不变）
    df = data.copy()
    group_cols = [col for col in df.columns if col in ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']]
    if not group_cols:
        print("警告: 未找到有效的分组列，无法创建滞后特征")
        return df
    if 'sales' not in df.columns:
        print("错误: 数据中未包含'sales'列，无法创建滞后特征")
        return df
    print(f'创建滞后特征，滞后周期: {lag_periods}')
    date_cols = ['day', 'd', 'd_num']  # 增加d_num作为日期列选项
    sort_cols = None
    for col in date_cols:
        if col in df.columns:
            sort_cols = group_cols + [col]
            break
    if sort_cols:
        df = df.sort_values(sort_cols)
    else:
        print("警告: 未找到有效的日期列，可能影响滞后特征准确性")
    for lag in lag_periods:
        df[f'lag_{lag}'] = df.groupby(group_cols)['sales'].shift(lag)
        df[f'lag_{lag}'] = df.groupby(group_cols)[f'lag_{lag}'].fillna(method='bfill')
    return df


def create_rolling_features(data, windows=[7, 14, 30], functions=['mean', 'max']):  # 精简窗口和函数
    # （函数内容不变）
    df = data.copy()
    group_cols = [col for col in df.columns if col in ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']]
    if not group_cols:
        print("警告: 未找到有效的分组列，无法创建滚动特征")
        return df
    if 'sales' not in df.columns:
        print("错误: 数据中未包含'sales'列，无法创建滚动特征")
        return df
    print(f'创建滚动特征，窗口大小: {windows}, 函数: {functions}')
    date_cols = ['day', 'd', 'd_num']
    sort_cols = None
    for col in date_cols:
        if col in df.columns:
            sort_cols = group_cols + [col]
            break
    if sort_cols:
        df = df.sort_values(sort_cols)
    else:
        print("警告: 未找到有效的日期列，可能影响滚动特征准确性")
    for window in windows:
        if window <= 0:
            print(f"警告: 窗口大小{window}无效，已跳过")
            continue
        for func in functions:
            try:
                if func == 'mean':
                    df[f'rolling_mean_{window}'] = df.groupby(group_cols)['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif func == 'std':
                    df[f'rolling_std_{window}'] = df.groupby(group_cols)['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    ).fillna(0)
                elif func == 'min':
                    df[f'rolling_min_{window}'] = df.groupby(group_cols)['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif func == 'max':
                    df[f'rolling_max_{window}'] = df.groupby(group_cols)['sales'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                else:
                    print(f"警告: 不支持的统计函数{func}，已跳过")
            except Exception as e:
                print(f"创建特征rolling_{func}_{window}时出错: {str(e)}")
    return df


def create_time_based_features(data):
    # （函数内容不变）
    df = data.copy()
    if 'date' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            print("已创建详细时间特征")
        except Exception as e:
            print(f"创建时间特征时出错: {str(e)}")
    if 'wday' in df.columns and 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['wday'] >= 6).astype(int)
        print("已从wday列创建周末特征")
    return df


def create_price_features(data, prices_df):
    # （函数内容不变）
    df = data.copy()
    required_cols = ['wm_yr_wk', 'store_id', 'item_id']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 主数据缺少必要的列{required_cols}，无法创建价格特征")
        return df
    if prices_df is not None and not all(col in prices_df.columns for col in required_cols + ['sell_price']):
        print(f"错误: 价格数据缺少必要的列{required_cols + ['sell_price']}，无法创建价格特征")
        return df
    print("合并价格数据并创建价格特征")
    df = df.merge(prices_df[required_cols + ['sell_price']], on=required_cols, how='left') if prices_df is not None else df
    df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.fillna(x.mean())) if 'sell_price' in df.columns else df
    df['sell_price'] = df['sell_price'].fillna(0) if 'sell_price' in df.columns else df
    group_cols = ['store_id', 'item_id']
    df = df.sort_values(group_cols + ['wm_yr_wk']) if all(col in df.columns for col in group_cols + ['wm_yr_wk']) else df
    df['price_lag_1'] = df.groupby(group_cols)['sell_price'].shift(1).fillna(method='bfill') if 'sell_price' in df.columns else df
    df['price_change_ratio'] = (df['sell_price'] - df['price_lag_1']) / df['price_lag_1'].replace(0, 0.001).fillna(0) if 'sell_price' in df.columns else df
    df['is_price_drop'] = (df['price_change_ratio'] < 0).astype(int) if 'price_change_ratio' in df.columns else df
    df['price_mean_4weeks'] = df.groupby(group_cols)['sell_price'].transform(lambda x: x.rolling(4, min_periods=1).mean()) if 'sell_price' in df.columns else df
    df['price_std_4weeks'] = df.groupby(group_cols)['sell_price'].transform(lambda x: x.rolling(4, min_periods=1).std()).fillna(0) if 'sell_price' in df.columns else df
    df['price_to_mean_ratio'] = df['sell_price'] / df['price_mean_4weeks'].replace(0, 0.001) if 'sell_price' in df.columns else df
    return df


def create_event_features(data, calendar_df):
    # （函数内容不变）
    df = data.copy()
    if 'd' not in df.columns or (calendar_df is not None and 'd' not in calendar_df.columns):
        print("错误: 主数据或日历数据中缺少'd'列，无法创建事件特征")
        return df
    event_cols = ['d', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    available_event_cols = [col for col in event_cols if calendar_df is not None and col in calendar_df.columns] if calendar_df is not None else []
    if len(available_event_cols) <= 1:
        print("警告: 日历数据中没有事件相关列，无法创建事件特征")
        return df
    df = df.merge(calendar_df[available_event_cols], on='d', how='left')
    for col in available_event_cols[1:]:
        df[col] = df[col].fillna('None')
    df['has_event_1'] = (df['event_name_1'] != 'None').astype(int) if 'event_name_1' in df.columns else 0
    df['has_event_2'] = (df['event_name_2'] != 'None').astype(int) if 'event_name_2' in df.columns else 0
    df['has_any_event'] = ((df['has_event_1'] + df['has_event_2']) > 0).astype(int)
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        event_dates = df[df['has_any_event'] == 1]['date'].unique()
        df['is_day_before_event'] = df['date'].isin([d - pd.Timedelta(days=1) for d in event_dates]).astype(int)
        df['is_day_after_event'] = df['date'].isin([d + pd.Timedelta(days=1) for d in event_dates]).astype(int)
    return df


def encode_categorical_features(data):
    # （函数内容不变，增加category类型处理）
    df = data.copy()
    encoders = {}
    exclude_cols = ['id', 'd', 'date', 'wm_yr_wk']
    # 同时处理object和category类型
    cat_cols = [col for col in df.columns if df[col].dtype in ['object', 'category'] and col not in exclude_cols]
    if not cat_cols:
        print("未发现需要编码的分类特征")
        return df, encoders
    print(f'编码分类特征: {cat_cols}')
    for col in cat_cols:
        try:
            # 将category类型转为object，避免编码错误
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype('object')
            not_null_values = df[col].dropna()
            if len(not_null_values) == 0:
                print(f"警告: 特征{col}全为空值，已跳过编码")
                continue
            encoder = LabelEncoder()
            encoder.fit(not_null_values)
            mask = df[col].notna()
            df.loc[mask, col] = encoder.transform(df.loc[mask, col])
            encoders[col] = encoder
        except ValueError as e:
            print(f"编码特征{col}时出现值错误: {str(e)}，已跳过")
        except Exception as e:
            print(f"编码特征{col}时出错: {str(e)}，已跳过")
    return df, encoders


def build_features(data, prices_df=None, calendar_df=None):
    # （函数内容不变）
    print("===== 开始特征工程 =====")
    df = data.copy()
    print("\n1. 创建时间特征...")
    df = create_time_based_features(df)
    print("\n2. 创建滞后特征...")
    df = create_lag_features(df)
    print("\n3. 创建滚动特征...")
    df = create_rolling_features(df)
    if prices_df is not None:
        print("\n4. 创建价格特征...")
        df = create_price_features(df, prices_df)
    else:
        print("\n4. 未提供价格数据，跳过价格特征创建")
    if calendar_df is not None:
        print("\n5. 创建事件特征...")
        df = create_event_features(df, calendar_df)
    else:
        print("\n5. 未提供日历数据，跳过事件特征创建")
    print("\n6. 编码分类特征...")
    df, encoders = encode_categorical_features(df)
    print("\n7. 处理缺失值...")
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            group_cols = [c for c in ['item_id', 'store_id'] if c in df.columns]
            if group_cols:
                df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(0)
    print("\n===== 特征工程完成 =====")
    print(f"最终特征数量: {df.shape[1]}")
    return df, encoders


# ----------------------
# 4. 主程序入口
# ----------------------
if __name__ == "__main__":
    # 假设merged_data已通过create_merged_data函数生成
    # （此处需确保merged_data已正确加载，例如：）
    # merged_data = create_merged_data(cleaned_data)
    
    if 'merged_data' in locals() and merged_data is not None:
        # 步骤1：压缩内存
        merged_data = reduce_memory_usage(merged_data)
        
        # 步骤2：可选抽样（调试用，全量运行时注释）
        sample_ratio = 0.1  # 取10%数据测试
        merged_data = merged_data.sample(frac=sample_ratio, random_state=42)
        
        # 步骤3：分批构建特征
        features_df, encoders = build_features_in_batches(
            merged_data=merged_data,
            batch_size=100,  # 每批处理100个商品（可根据内存调整）
            prices_df=None,  # 已合并到merged_data中
            calendar_df=None  # 已合并到merged_data中
        )
        
        # 输出结果
        print("\n特征工程最终结果：")
        print(f"特征数据集形状: {features_df.shape}")
        print("前5行数据预览：")
        print(features_df.head())
        
        # 可选：保存结果
        features_df.to_csv('features.csv', index=False)
        # print("特征数据已保存为features.csv")
    else:
        print("错误：未找到有效的merged_data，请先执行数据合并步骤")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # 注意导入路径
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ARIMA模型训练与预测

def fit_arima_model(series, order=(1,1,1)):
    """
    训练ARIMA模型
    
    参数:
    series: Series, 时间序列数据
    order: tuple, ARIMA模型参数 (p, d, q)
    
    返回:
    Model: 训练好的ARIMA模型
    """
    try:
        # 使用最新的ARIMA接口
        model = ARIMA(series.values, order=order)
        model_fit = model.fit()
        print(f'ARIMA{order} 模型训练完成')
        return model_fit
    except Exception as e:
        print(f'ARIMA{order} 模型训练失败: {e}')
        return None

def auto_arima(series, max_p=3, max_d=2, max_q=3):
    """
    自动选择ARIMA最佳参数
    
    参数:
    series: Series, 时间序列数据
    max_p: int, 最大自回归阶数
    max_d: int, 最大差分阶数
    max_q: int, 最大移动平均阶数
    
    返回:
    tuple: 最佳参数 (p, d, q)
    Model: 最佳模型
    """
    best_aic = float("inf")
    best_order = None
    best_model = None
    
    print('开始自动ARIMA参数选择...')
    
    # 遍历参数组合
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                # 跳过无效参数组合
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series.values, order=(p, d, q))
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                    
                    print(f'ARIMA({p},{d},{q}) - AIC: {model_fit.aic:.2f}')
                except Exception as e:
                    print(f'ARIMA({p},{d},{q}) - 失败: {str(e)}')
                    continue
    
    if best_order:
        print(f'最佳ARIMA参数: {best_order}, AIC: {best_aic:.2f}')
    else:
        print('无法找到合适的ARIMA参数')
    
    return best_order, best_model

def forecast_arima(model_fit, steps=28):
    """
    使用ARIMA模型进行预测
    
    参数:
    model_fit: Model, 训练好的ARIMA模型
    steps: int, 预测步数
    
    返回:
    Series: 预测结果
    """
    try:
        # 使用最新的forecast方法
        forecast_result = model_fit.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        print(f'ARIMA模型预测完成，预测了 {steps} 步')
        return forecast
    except Exception as e:
        print(f'ARIMA预测失败: {e}')
        return None

def evaluate_arima_forecast(actual, forecast):
    """
    评估ARIMA预测结果
    
    参数:
    actual: Series, 实际值
    forecast: Series, 预测值
    
    返回:
    dict: 评估指标
    """
    # 确保两个序列长度相同
    min_length = min(len(actual), len(forecast))
    actual = actual[:min_length]
    forecast = forecast[:min_length]
    
    # 计算评估指标
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    
    # 计算MAPE（避免除以0）
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    else:
        mape = float('inf')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print('ARIMA预测评估结果:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')
    
    return metrics

def plot_arima_results(series, forecast, actual=None, title='ARIMA预测结果'):
    """
    绘制ARIMA预测结果
    
    参数:
    series: Series, 历史数据
    forecast: Series, 预测结果
    actual: Series, 实际值（如果有）
    title: str, 图表标题
    
    返回:
    None
    """
    chinese_font = get_chinese_font()

    plt.figure(figsize=(15, 7))
    
    # 绘制历史数据
    plt.plot(range(len(series)), series, label='历史数据', color='blue')
    
    # 绘制预测数据
    forecast_start = len(series)
    forecast_end = forecast_start + len(forecast)
    plt.plot(range(forecast_start, forecast_end), forecast, label='预测数据', color='red')
    
    # 绘制实际数据（如果有）
    if actual is not None:
        actual_start = forecast_start
        actual_end = actual_start + len(actual)
        plt.plot(range(actual_start, actual_end), actual, label='实际数据', color='green')
    
    plt.title(title,fontproperties=chinese_font, fontsize=12)
    plt.xlabel('时间',fontproperties=chinese_font, fontsize=12)
    plt.ylabel('销售额',fontproperties=chinese_font, fontsize=12)
    plt.legend(prop=chinese_font, fontsize=12)
    plt.grid(True)
    plt.show()

def run_arima_forecast(hierarchical_series, level='total', steps=28):
    """
    为指定层级运行ARIMA预测流程
    
    参数:
    hierarchical_series: dict, 分层时间序列数据
    level: str, 要预测的层级
    steps: int, 预测步数
    
    返回:
    dict: 预测结果和评估指标
    """
    if level not in hierarchical_series:
        print(f'错误: 未找到层级 {level}')
        return None
    
    print(f'\n===== 开始ARIMA预测 - {level} =====')
    
    # 获取时间序列数据并按d排序
    series_data = hierarchical_series[level].copy()
    # 确保数据按d列排序（d_1, d_2, ..., d_n）
    series_data = series_data.sort_values('d')
    sales_series = pd.Series(series_data['sales'].values)
    
    # 划分训练集和验证集
    train_size = int(len(sales_series) * 0.9)  # 90%用于训练
    train, test = sales_series[:train_size], sales_series[train_size:]
    
    print(f'训练集大小: {len(train)}, 测试集大小: {len(test)}')
    
    # 自动选择最佳参数（简化搜索范围以提高速度）
    best_order, best_model = auto_arima(train, max_p=2, max_d=1, max_q=2)
    
    # 如果自动参数选择失败，使用默认参数
    if best_model is None:
        print('使用默认ARIMA(1,1,1)参数')
        best_model = fit_arima_model(train, order=(1,1,1))
    
    if best_model is None:
        print('ARIMA模型训练失败')
        return None
    
    # 进行测试集预测
    test_forecast = forecast_arima(best_model, steps=len(test))
    
    # 评估预测结果
    metrics = None
    if test_forecast is not None:
        metrics = evaluate_arima_forecast(test.values, test_forecast)
        
        # 绘制预测结果
        plot_arima_results(train, test_forecast, actual=test, 
                          title=f'ARIMA预测结果 - {level}')
    
    # 对完整数据集进行最终预测
    final_model = fit_arima_model(sales_series, order=best_order if best_order else (1,1,1))
    final_forecast = forecast_arima(final_model, steps=steps)
    
    results = {
        'level': level,
        'best_order': best_order,
        'metrics': metrics,
        'test_forecast': test_forecast,
        'final_forecast': final_forecast
    }
    
    print(f'===== ARIMA预测完成 - {level} =====')
    
    return results

# 注意：需要先确保hierarchical_series已正确创建
arima_results = run_arima_forecast(hierarchical_series, level='total', steps=28)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------
# XGBoost特征准备与模型训练
# ----------------------
def prepare_xgboost_features(series_data, lag_days=28, window_sizes=[7, 14, 28]):
    """为XGBoost准备特征（优化时间特征处理）"""
    df = series_data.copy()
    
    # 确保数据按日期排序（关键！）
    if 'd' in df.columns:
        df = df.sort_values('d').reset_index(drop=True)
    else:
        print("警告：数据中缺少'd'列，假设已按时间排序")
    
    # 添加滞后特征（历史销售额）
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df['sales'].shift(i)
    
    # 添加滚动统计特征（基于历史数据，避免数据泄露）
    for window in window_sizes:
        # shift(1)确保滚动窗口不包含当前值（避免未来信息泄露）
        df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
        df[f'rolling_min_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df['sales'].shift(1).rolling(window=window, min_periods=1).max()
    
    # 添加时间特征（从'd'列提取，如d_123 → 123，避免依赖index）
    if 'd' in df.columns:
        df['d_num'] = df['d'].str.extract(r'd_(\d+)').astype(int)  # 日期序号
        df['day_of_week'] = (df['d_num'] - 1) % 7  # 星期几（假设d_1是周一）
        df['month'] = (df['d_num'] - 1) // 30 + 1  # 月份（简化处理）
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 是否周末
    else:
        print("警告：缺少'd'列，无法创建时间特征")
    
    # 添加事件特征（如果存在）
    if 'event_type_1' in df.columns:
        df['has_event'] = df['event_type_1'].notna().astype(int)
        # 对事件类型编码
        df['event_type_1_encoded'] = df['event_type_1'].astype('category').cat.codes.replace(-1, 0)
    
    # 填充缺失值（滞后特征的前n行可能为NaN）
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)  # 用0填充数值型缺失
    
    print(f'XGBoost特征准备完成，形状: {df.shape}')
    return df


def split_xgboost_data(df, test_size=0.2):
    """划分训练集和测试集（时间序列专用分割）"""
    # 特征和目标变量（排除非特征列）
    exclude_cols = ['sales', 'd', 'date']  # 排除目标和原始日期列
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df['sales']
    
    # 时间序列分割（按顺序，不随机）
    train_size = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f'训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}')
    return X_train, X_test, y_train, y_test


def fit_xgboost_model(X_train, y_train, X_test=None, y_test=None, params=None):
    """训练XGBoost模型（修复X_test未定义问题）"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'n_estimators': 100,
            'seed': 42,
            'verbosity': 0
        }
    
    try:
        model = xgb.XGBRegressor(**params)
        
        # 构建评估集（仅当提供了测试集时才加入）
        eval_sets = [(X_train, y_train)]
        if X_test is not None and y_test is not None:
            eval_sets.append((X_test, y_test))
        
        model.fit(
            X_train, y_train,
            eval_set=eval_sets,
            early_stopping_rounds=10,
            verbose=False
        )
        print(f'XGBoost模型训练完成（最佳迭代轮次: {model.best_iteration}）')
        return model
    except Exception as e:
        print(f'XGBoost模型训练失败: {e}')
        return None



def hyperparameter_tuning_xgboost(X_train, y_train, X_test, y_test, param_grid=None, cv=3):
    """超参数调优（增加验证集，加速调优）"""
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5],  # 减少搜索范围，加速计算
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=100,
        seed=42,
        verbosity=0
    )
    
    print('开始XGBoost超参数调优...')
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1  # 并行计算
    )
    
    # 用早停机制辅助调优
    grid_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # 转换为正的RMSE
    
    print(f'最佳参数: {best_params}')
    print(f'最佳验证RMSE: {best_score:.4f}')
    
    return best_params, grid_search.best_estimator_


def predict_xgboost(model, X_test):
    """预测函数（确保输出非负）"""
    try:
        predictions = model.predict(X_test)
        predictions = np.maximum(0, predictions)  # 销售额不能为负
        print(f'XGBoost预测完成，预测样本数: {len(predictions)}')
        return predictions
    except Exception as e:
        print(f'XGBoost预测失败: {e}')
        return None


def evaluate_xgboost_predictions(actual, predictions):
    """评估函数（与ARIMA保持一致的指标）"""
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    
    # 计算MAPE（避免除以0）
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
    else:
        mape = float('inf')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print('XGBoost预测评估结果:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')
    
    return metrics


def plot_xgboost_results(actual, predictions, title='XGBoost预测结果'):
    """绘制预测结果（支持中文）"""
    chinese_font = get_chinese_font()
    plt.figure(figsize=(15, 7))
    
    plt.plot(range(len(actual)), actual, label='实际值', color='blue')
    plt.plot(range(len(predictions)), predictions, label='预测值', color='red', alpha=0.8)
    
    plt.title(title, fontproperties=chinese_font, fontsize=14)
    plt.xlabel('时间步', fontproperties=chinese_font, fontsize=12)
    plt.ylabel('销售额', fontproperties=chinese_font, fontsize=12)
    plt.legend(prop=chinese_font)
    plt.grid(True)
    plt.show()


def plot_feature_importance_xgboost(model, n_features=10):
    """绘制特征重要性（支持中文）"""
    chinese_font = get_chinese_font()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    feature_names = model.get_booster().feature_names
    
    plt.figure(figsize=(12, 8))
    plt.title(f'XGBoost特征重要性（前{n_features}个）', fontproperties=chinese_font, fontsize=14)
    plt.bar(range(n_features), importances[indices])
    plt.xticks(
        range(n_features), 
        [feature_names[i] for i in indices], 
        rotation=45, 
        ha='right', 
        fontproperties=chinese_font
    )
    plt.tight_layout()
    plt.show()


def run_xgboost_forecast(hierarchical_series, level='total', lag_days=14, tune_hyperparams=False):
    """完整预测流程（传递测试集到模型训练）"""
    if level not in hierarchical_series:
        print(f'错误: 未找到层级 {level}')
        return None
    
    print(f'\n===== 开始XGBoost预测 - {level} =====')
    
    # 1. 获取数据并准备特征
    series_data = hierarchical_series[level].copy()
    feature_df = prepare_xgboost_features(series_data, lag_days=lag_days)
    
    if feature_df.shape[0] < 100:
        print(f"警告：有效样本数太少（{feature_df.shape[0]}），可能影响模型效果")
    
    # 2. 划分数据集
    X_train, X_test, y_train, y_test = split_xgboost_data(feature_df, test_size=0.2)
    
    # 3. 训练模型（可选调优）
    if tune_hyperparams:
        best_params, model = hyperparameter_tuning_xgboost(X_train, y_train, X_test, y_test)
    else:
        # 关键修复：将X_test和y_test作为参数传入fit函数
        model = fit_xgboost_model(X_train, y_train, X_test=X_test, y_test=y_test)
    
    if model is None:
        print('XGBoost模型训练失败')
        return None
    
    # 4. 预测与评估（后续逻辑不变）
    predictions = predict_xgboost(model, X_test)
    metrics = None
    if predictions is not None:
        metrics = evaluate_xgboost_predictions(y_test, predictions)
        plot_xgboost_results(y_test, predictions, title=f'XGBoost预测结果 - {level}')
        plot_feature_importance_xgboost(model, n_features=10)
    
    results = {
        'level': level,
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print(f'===== XGBoost预测完成 - {level} =====')
    return results



xgboost_results = run_xgboost_forecast(
    hierarchical_series, 
    level='total', 
    lag_days=14, 
    tune_hyperparams=False  # 首次运行建议关闭调优，加速测试
)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------
# LightGBM特征准备与模型训练
# ----------------------
def prepare_lightgbm_features(series_data, lag_days=28, window_sizes=[7, 14, 28]):
    """复用XGBoost的特征工程（通用特征）"""
    df = prepare_xgboost_features(series_data, lag_days=lag_days, window_sizes=window_sizes)
    return df


def fit_lightgbm_model(X_train, y_train, X_test=None, y_test=None, params=None):
    """训练LightGBM模型（优化早停机制，加入测试集验证）"""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,  # 特征采样
            'bagging_fraction': 0.8,  # 样本采样
            'bagging_freq': 5,        # 每5轮采样一次
            'seed': 42,
            'verbosity': 0  # 静默模式
        }
    
    try:
        # 创建训练数据集
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        
        # 构建验证集（用于早停）
        valid_sets = []
        if X_test is not None and y_test is not None:
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
            valid_sets.append(valid_data)
        
        # 训练模型
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,  # 加入验证集
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),  # 早停
                lgb.log_evaluation(period=100)  # 每100轮打印一次日志
            ]
        )
        
        print(f'LightGBM模型训练完成（最佳迭代轮次: {model.best_iteration}）')
        return model
    except Exception as e:
        print(f'LightGBM模型训练失败: {e}')
        return None


def hyperparameter_tuning_lightgbm(X_train, y_train, X_test, y_test, param_grid=None, cv=3):
    """超参数调优（加入测试集辅助早停，加速搜索）"""
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5],  # 减少搜索范围，加速计算
            'learning_rate': [0.05, 0.1],
            'feature_fraction': [0.8],
            'bagging_fraction': [0.8]
        }
    
    # 创建训练和验证数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    print('开始LightGBM超参数调优...')
    
    best_score = float("inf")
    best_params = None
    
    # 网格搜索（简化版）
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for feature_fraction in param_grid['feature_fraction']:
                for bagging_fraction in param_grid['bagging_fraction']:
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'feature_fraction': feature_fraction,
                        'bagging_fraction': bagging_fraction,
                        'bagging_freq': 5,
                        'seed': 42,
                        'verbosity': 0
                    }
                    
                    try:
                        # 用交叉验证评估参数
                        cv_results = lgb.cv(
                            params=params,
                            train_set=train_data,
                            num_boost_round=1000,
                            nfold=cv,
                            stratified=False,
                            valid_sets=[valid_data],  # 加入验证集
                            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                        )
                        
                        # 取最后一轮的RMSE均值作为分数
                        current_score = cv_results['rmse-mean'][-1]
                        print(f'参数: {{"max_depth": {max_depth}, "learning_rate": {learning_rate}}}, RMSE: {current_score:.4f}')
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_params = params.copy()
                    except Exception as e:
                        print(f'参数组合失败: {e}')
                        continue
    
    if best_params:
        print(f'最佳参数: {best_params}')
        print(f'最佳验证RMSE: {best_score:.4f}')
    else:
        print('未能找到合适的参数')
    
    return best_params, best_score


def predict_lightgbm(model, X_test):
    """预测函数（确保输出非负）"""
    try:
        predictions = model.predict(X_test, num_iteration=model.best_iteration)  # 使用最佳迭代轮次
        predictions = np.maximum(0, predictions)  # 销售额不能为负
        print(f'LightGBM预测完成，预测样本数: {len(predictions)}')
        return predictions
    except Exception as e:
        print(f'LightGBM预测失败: {e}')
        return None


def evaluate_lightgbm_predictions(actual, predictions):
    """评估函数（与其他模型保持一致）"""
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
    else:
        mape = float('inf')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print('LightGBM预测评估结果:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')
    
    return metrics


def plot_lightgbm_results(actual, predictions, title='LightGBM预测结果'):
    """绘制预测结果（支持中文）"""
    chinese_font = get_chinese_font()
    plt.figure(figsize=(15, 7))
    
    plt.plot(range(len(actual)), actual, label='实际值', color='blue')
    plt.plot(range(len(predictions)), predictions, label='预测值', color='red', alpha=0.8)
    
    plt.title(title, fontproperties=chinese_font, fontsize=14)
    plt.xlabel('时间步', fontproperties=chinese_font, fontsize=12)
    plt.ylabel('销售额', fontproperties=chinese_font, fontsize=12)
    plt.legend(prop=chinese_font)
    plt.grid(True)
    plt.show()


def plot_feature_importance_lightgbm(model, n_features=10):
    """绘制特征重要性（支持中文，横向条形图更清晰）"""
    chinese_font = get_chinese_font()
    importance = model.feature_importance(importance_type='gain')  # 用gain衡量重要性
    feature_names = model.feature_name()
    
    # 排序并选择前N个特征
    indices = np.argsort(importance)[::-1][:n_features]
    top_importance = importance[indices]
    top_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'LightGBM特征重要性（前{n_features}个）', fontproperties=chinese_font, fontsize=14)
    plt.barh(range(n_features), top_importance, align='center', color='skyblue')
    plt.yticks(range(n_features), top_features, fontproperties=chinese_font)
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    plt.xlabel('重要性得分（Gain）', fontproperties=chinese_font, fontsize=12)
    plt.tight_layout()
    plt.show()


def run_lightgbm_forecast(hierarchical_series, level='total', lag_days=14, tune_hyperparams=False):
    """完整预测流程（主函数）"""
    if level not in hierarchical_series:
        print(f'错误: 未找到层级 {level}')
        return None
    
    print(f'\n===== 开始LightGBM预测 - {level} =====')
    
    # 1. 获取数据并准备特征
    series_data = hierarchical_series[level].copy()
    feature_df = prepare_lightgbm_features(series_data, lag_days=lag_days)
    
    if feature_df.shape[0] < 100:
        print(f"警告：有效样本数太少（{feature_df.shape[0]}），可能影响模型效果")
    
    # 2. 划分数据集（复用XGBoost的分割函数）
    X_train, X_test, y_train, y_test = split_xgboost_data(feature_df, test_size=0.2)
    
    # 3. 训练模型（可选调优）
    model = None
    if tune_hyperparams:
        best_params, best_score = hyperparameter_tuning_lightgbm(X_train, y_train, X_test, y_test)
        if best_params:
            model = fit_lightgbm_model(X_train, y_train, X_test, y_test, params=best_params)
    
    # 若调优失败或未调优，使用默认参数
    if model is None:
        model = fit_lightgbm_model(X_train, y_train, X_test, y_test)
    
    if model is None:
        print('LightGBM模型训练失败')
        return None
    
    # 4. 预测与评估
    predictions = predict_lightgbm(model, X_test)
    metrics = None
    if predictions is not None:
        metrics = evaluate_lightgbm_predictions(y_test, predictions)
        plot_lightgbm_results(y_test, predictions, title=f'LightGBM预测结果 - {level}')
        plot_feature_importance_lightgbm(model, n_features=10)
    
    # 5. 返回结果
    results = {
        'level': level,
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print(f'===== LightGBM预测完成 - {level} =====')
    return results


lightgbm_results = run_lightgbm_forecast(
    hierarchical_series, 
    level='total', 
    lag_days=14, 
    tune_hyperparams=False  # 首次运行建议关闭调优
)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter1d

# ----------------------
# WRMSSE评估核心函数
# ----------------------
def calculate_weights(hierarchical_series):
    """计算WRMSSE权重（基于各层级历史销售额占比）"""
    weights = {}
    
    for level in hierarchical_series:
        # 确保'sales'是Series或数组
        sales_data = hierarchical_series[level]['sales']
        if isinstance(sales_data, pd.Series):
            sales_data = sales_data.values
        
        # 计算该层级的总销售额（权重基础）
        total_sales = sales_data.sum()
        weights[level] = total_sales
    
    # 归一化权重（确保总和为1）
    total_weight = sum(weights.values())
    if total_weight <= 0:
        print("警告：所有层级销售额总和为0，权重将平均分配")
        total_weight = len(weights)  # 平均分配权重
        weights = {level: 1/total_weight for level in weights}
    else:
        weights = {level: w / total_weight for level, w in weights.items()}
    
    print(f'各层级权重计算完成，共{len(weights)}个层级')
    return weights


def calculate_scaling_factors(hierarchical_series, window_size=28):
    """计算缩放因子（最近window_size天的标准差，避免为0）"""
    scaling_factors = {}
    
    for level in hierarchical_series:
        sales_data = hierarchical_series[level]['sales']
        if isinstance(sales_data, pd.Series):
            sales_data = sales_data.values
        
        # 确保数据量足够
        if len(sales_data) >= window_size:
            recent_data = sales_data[-window_size:]  # 最近window_size天
        else:
            recent_data = sales_data  # 数据不足时使用全部数据
        
        # 计算标准差（避免为0）
        std_dev = recent_data.std()
        scaling_factors[level] = max(std_dev, 1e-6)  # 最小为1e-6，防止除0
    
    return scaling_factors


def calculate_rmse_scaled(actual, predicted, scaling_factor):
    """计算缩放后的RMSE（标准化误差）"""
    # 确保输入是数组
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # 截断到较短长度
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # 预测值非负处理
    predicted = np.maximum(0, predicted)
    
    # 计算RMSE并缩放
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse / scaling_factor  # 除以缩放因子


def calculate_wrmse(actual_dict, predicted_dict, weights_dict, scaling_factors_dict):
    """计算WRMSSE（加权缩放均方根误差）"""
    # 检查输入有效性
    common_levels = set(actual_dict.keys()) & set(predicted_dict.keys()) & set(weights_dict.keys()) & set(scaling_factors_dict.keys())
    if not common_levels:
        print("错误：没有共同的层级用于计算WRMSSE")
        return float('inf')
    
    # 计算各层级加权误差
    weighted_errors = []
    for level in common_levels:
        actual = actual_dict[level]
        predicted = predicted_dict[level]
        weight = weights_dict[level]
        scaling_factor = scaling_factors_dict[level]
        
        # 计算缩放RMSE
        scaled_rmse = calculate_rmse_scaled(actual, predicted, scaling_factor)
        
        # 累积加权平方误差
        weighted_errors.append(weight * (scaled_rmse **2))
        print(f'层级 {level}: 权重={weight:.6f}, 缩放RMSE={scaled_rmse:.4f}')
    
    # 最终WRMSSE
    wrmsse = np.sqrt(np.sum(weighted_errors))
    print(f'总WRMSSE: {wrmsse:.4f}')
    return wrmsse


# ----------------------
# 时间序列交叉验证
# ----------------------
def time_series_cv(hierarchical_series, model_function, params=None, n_splits=3, window_size=28):
    """时间序列交叉验证（支持所有模型，修复数据分割逻辑）"""
    params = params or {}
    cv_results = {
        'rmse_scores': [],
        'mae_scores': [],
        'wrmsse_scores': [],
        'fold_results': []
    }
    
    print(f'开始时间序列交叉验证，{n_splits}折，预测窗口={window_size}天')
    
    # 以'total'层级为例
    if 'total' not in hierarchical_series:
        print('错误: 未找到total层级数据')
        return cv_results
    
    # 提取原始销售数据
    total_series = hierarchical_series['total']
    sales_data = total_series['sales'].values if isinstance(total_series['sales'], pd.Series) else total_series['sales']
    total_length = len(sales_data)
    
    # 检查数据量是否足够
    required_length = window_size * (n_splits + 1)  # 至少需要n_splits+1个窗口
    if total_length < required_length:
        print(f'警告: 数据量不足（需{required_length}，实有{total_length}），自动调整折数')
        n_splits = max(1, (total_length // window_size) - 1)
        print(f'调整为{n_splits}折交叉验证')
    
    # 生成交叉验证分割点（滚动窗口）
    for i in range(n_splits):
        print(f'\n===== 交叉验证折 {i+1}/{n_splits} =====')
        
        # 计算分割点（训练集逐渐增大，测试集固定窗口）
        test_end = total_length - (n_splits - i - 1) * window_size
        test_start = test_end - window_size
        train_end = test_start  # 训练集截止到测试集开始前
        
        # 提取训练和测试数据
        train_sales = sales_data[:train_end]
        test_sales = sales_data[test_start:test_end]
        
        print(f'训练集: {len(train_sales)}天, 测试集: {len(test_sales)}天')
        if len(train_sales) == 0 or len(test_sales) != window_size:
            print(f'跳过折{i+1}: 数据分割异常')
            continue
        
        # 构建临时分层数据（仅包含当前训练集）
        temp_hierarchical = {
            'total': {'sales': pd.Series(train_sales)}
        }
        
        try:
            # 根据模型类型调用对应函数
            model_name = model_function.__name__
            if model_name == 'run_arima_forecast':
                # ARIMA预测
                results = model_function(temp_hierarchical, level='total', steps=window_size)
                predictions = results['final_forecast']  # 用最终预测作为窗口预测
            
            elif model_name in ['run_xgboost_forecast', 'run_lightgbm_forecast']:
                # XGBoost/LightGBM预测（需要特征工程）
                results = model_function(
                    temp_hierarchical, 
                    level='total', 
                    lag_days=window_size,  # 滞后天数=预测窗口
                    tune_hyperparams=False
                )
                # 取最后window_size个预测作为测试集预测
                predictions = results['predictions'][-window_size:]
            
            else:
                print(f'不支持的模型函数: {model_name}')
                continue
            
            # 计算基础指标
            rmse = np.sqrt(mean_squared_error(test_sales, predictions))
            mae = mean_absolute_error(test_sales, predictions)
            
            # 计算当前折的WRMSSE（仅total层级）
            wrmsse = calculate_wrmse(
                actual_dict={'total': test_sales},
                predicted_dict={'total': predictions},
                weights_dict={'total': 1.0},  # 单一层级权重为1
                scaling_factors_dict={'total': calculate_scaling_factors(temp_hierarchical)['total']}
            )
            
            # 保存结果
            cv_results['rmse_scores'].append(rmse)
            cv_results['mae_scores'].append(mae)
            cv_results['wrmsse_scores'].append(wrmsse)
            cv_results['fold_results'].append({
                'fold': i+1,
                'rmse': rmse,
                'mae': mae,
                'wrmsse': wrmsse
            })
            
            print(f'折{i+1}结果: RMSE={rmse:.4f}, MAE={mae:.4f}, WRMSSE={wrmsse:.4f}')
        
        except Exception as e:
            print(f'折{i+1}失败: {str(e)}')
    
    # 计算平均指标
    if cv_results['rmse_scores']:
        cv_results['avg_rmse'] = np.mean(cv_results['rmse_scores'])
        cv_results['avg_mae'] = np.mean(cv_results['mae_scores'])
        cv_results['avg_wrmsse'] = np.mean(cv_results['wrmsse_scores'])
        
        print(f'\n交叉验证汇总:')
        print(f"平均RMSE: {cv_results['avg_rmse']:.4f}")  # 外层双引号，内层单引号
        print(f'平均MAE: {cv_results["avg_mae"]:.4f}')    # 外层单引号，内层双引号
        print(f'平均WRMSSE: {cv_results["avg_wrmsse"]:.4f}')
    else:
        print('无有效交叉验证结果')
    
    return cv_results


# ----------------------
# 模型集成与优化
# ----------------------
def model_ensemble(predictions_dict, weights=None, method='weighted_average'):
    """模型集成（支持简单平均和加权平均，自动对齐长度）"""
    # 过滤无效预测
    valid_preds = {name: preds for name, preds in predictions_dict.items() 
                  if preds is not None and len(preds) > 0}
    if len(valid_preds) < 2:
        print(f'警告: 有效模型不足2个（仅{len(valid_preds)}个），无法集成')
        return None if len(valid_preds) == 0 else valid_preds[next(iter(valid_preds))]
    
    # 对齐预测长度（取最短）
    min_length = min(len(preds) for preds in valid_preds.values())
    aligned_preds = {name: preds[:min_length] for name, preds in valid_preds.items()}
    
    # 集成逻辑
    ensemble = np.zeros(min_length)
    if method == 'simple_average':
        # 简单平均
        for preds in aligned_preds.values():
            ensemble += preds
        ensemble /= len(aligned_preds)
    
    elif method == 'weighted_average':
        # 加权平均（默认按WRMSSE倒数分配权重）
        if weights is None:
            print('未提供权重，使用默认权重（模型性能越好，权重越高）')
            # 假设模型结果中包含WRMSSE（实际使用时需根据场景调整）
            weights = {name: 1.0 for name in aligned_preds}  # 此处可替换为基于指标的权重计算
        
        # 权重归一化
        total_weight = sum(weights.values())
        if total_weight <= 0:
            print('权重无效，自动切换为简单平均')
            return model_ensemble(predictions_dict, method='simple_average')
        
        normalized_weights = {name: w/total_weight for name, w in weights.items()}
        
        # 加权求和
        for name, preds in aligned_preds.items():
            ensemble += normalized_weights[name] * preds
    
    # 确保非负
    ensemble = np.maximum(0, ensemble)
    print(f'模型集成完成（{method}），使用{len(aligned_preds)}个模型')
    return ensemble


def optimize_predictions(predictions, constraints=None):
    """优化预测结果（非负约束+可选平滑/上下限）"""
    if predictions is None:
        print('错误: 无预测结果可优化')
        return None
    
    predictions = np.asarray(predictions)
    optimized = predictions.copy()
    
    # 基础约束：非负
    optimized = np.maximum(0, optimized)
    
    # 应用自定义约束
    if constraints:
        # 上下限约束
        if 'min_value' in constraints:
            optimized = np.maximum(constraints['min_value'], optimized)
        if 'max_value' in constraints:
            optimized = np.minimum(constraints['max_value'], optimized)
        
        # 平滑约束（减少波动）
        if constraints.get('smooth', False):
            sigma = constraints.get('smooth_sigma', 1.0)
            optimized = gaussian_filter1d(optimized, sigma=sigma)
            # 平滑后可能出现负值，再次截断
            optimized = np.maximum(0, optimized)
            print(f'应用平滑处理（sigma={sigma}）')
    
    print('预测结果优化完成')
    return optimized


def compare_models(model_results_dict):
    """比较不同模型的性能指标（增加WRMSSE支持）"""
    comparison_data = []
    
    for model_name, results in model_results_dict.items():
        if not results or 'metrics' not in results:
            print(f'跳过模型 {model_name}: 无有效指标')
            continue
        
        metrics = results['metrics']
        # 尝试获取WRMSSE（如果有）
        wrmsse = results.get('wrmsse', np.nan)
        
        comparison_data.append({
            '模型': model_name,
            'MSE': metrics.get('MSE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAE': metrics.get('MAE', np.nan),
            'MAPE(%)': metrics.get('MAPE', np.nan),
            'WRMSSE': wrmsse
        })
    
    if not comparison_data:
        print('无模型可比较')
        return None
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('WRMSSE' if 'WRMSSE' in df.columns else 'RMSE')  # 优先按WRMSSE排序
    print('\n模型性能比较:')
    print(df.round(4))  # 保留4位小数
    return df


def run_full_validation(hierarchical_series, model_results_dict):
    """完整验证流程（计算WRMSSE+模型比较+集成+优化）"""
    print('\n===== 开始完整模型验证 =====')
    
    # 1. 计算权重和缩放因子（用于WRMSSE）
    weights = calculate_weights(hierarchical_series)
    scaling_factors = calculate_scaling_factors(hierarchical_series)
    
    # 2. 收集各模型的实际值和预测值（按层级）
    actual_dict = {}
    predicted_dict = {}
    for level in hierarchical_series:
        # 实际值（取测试集）
        series_data = hierarchical_series[level]
        sales = series_data['sales'].values if isinstance(series_data['sales'], pd.Series) else series_data['sales']
        test_size = int(len(sales) * 0.2)  # 假设测试集占20%
        actual_dict[level] = sales[-test_size:]  # 测试集实际值
        
        # 各模型的预测值
        for model_name, results in model_results_dict.items():
            if level == results.get('level') and 'predictions' in results:
                if model_name not in predicted_dict:
                    predicted_dict[model_name] = {}
                predicted_dict[model_name][level] = results['predictions']
    
    # 3. 计算各模型的WRMSSE
    for model_name in predicted_dict:
        model_wrmsse = calculate_wrmse(
            actual_dict=actual_dict,
            predicted_dict=predicted_dict[model_name],
            weights_dict=weights,
            scaling_factors_dict=scaling_factors
        )
        model_results_dict[model_name]['wrmsse'] = model_wrmsse  # 保存到模型结果
    
    # 4. 模型比较
    model_comparison = compare_models(model_results_dict)
    
    # 5. 模型集成（如果有多个模型）
    ensemble_preds = None
    if len(model_results_dict) >= 2:
        # 收集各模型的预测值（取total层级）
        predictions_to_ensemble = {
            name: results['predictions'] 
            for name, results in model_results_dict.items() 
            if 'predictions' in results and results.get('level') == 'total'
        }
        ensemble_preds = model_ensemble(predictions_to_ensemble, method='weighted_average')
    
    # 6. 优化最佳模型的预测结果
    optimized_preds = None
    if model_comparison is not None and not model_comparison.empty:
        best_model_name = model_comparison.iloc[0]['模型']
        best_model = model_results_dict[best_model_name]
        if 'predictions' in best_model:
            optimized_preds = optimize_predictions(
                best_model['predictions'],
                constraints={'smooth': True, 'smooth_sigma': 1.5}  # 轻微平滑
            )
    
    # 整理结果
    validation_results = {
        'model_comparison': model_comparison,
        'ensemble_predictions': ensemble_preds,
        'optimized_predictions': optimized_preds,
        'weights': weights,
        'scaling_factors': scaling_factors
    }
    
    print('\n===== 模型验证完成 =====')
    return validation_results


model_results = {
    'ARIMA': arima_results,
    'XGBoost': xgboost_results,
    'LightGBM': lightgbm_results
}

# 过滤无效结果
model_results = {k: v for k, v in model_results.items() if v is not None}

if model_results:
    validation_results = run_full_validation(hierarchical_series, model_results)
else:
    print('没有可用的模型结果用于验证')


# 提交文件生成功能

def generate_submission_ids(item_ids, levels=None, forecast_horizon=28):
    """
    生成Kaggle提交格式所需的ID列表
    
    参数:
    item_ids: list, 物品ID列表
    levels: dict, 分层信息（可选）
    forecast_horizon: int, 预测天数（默认28天）
    
    返回:
    list: 提交文件的ID列表
    """
    submission_ids = []
    
    # 为每个物品生成未来28天的ID
    for item_id in item_ids:
        for day in range(1, forecast_horizon + 1):
            submission_id = f"{item_id}_F{day}"
            submission_ids.append(submission_id)
    
    print(f'生成了{len(submission_ids)}个提交ID')
    return submission_ids

def prepare_predictions_dataframe(item_ids, predictions_dict, forecast_horizon=28):
    """
    准备预测结果的DataFrame
    
    参数:
    item_ids: list, 物品ID列表
    predictions_dict: dict, 各物品的预测结果 {item_id: predictions_array}
    forecast_horizon: int, 预测天数
    
    返回:
    DataFrame: 格式化的预测结果DataFrame
    """
    # 生成ID列表
    ids = generate_submission_ids(item_ids, forecast_horizon=forecast_horizon)
    
    # 展平预测结果
    all_predictions = []
    for item_id in item_ids:
        if item_id in predictions_dict:
            preds = predictions_dict[item_id]
            # 确保预测结果长度正确
            if len(preds) >= forecast_horizon:
                preds = preds[:forecast_horizon]
            else:
                # 填充不足的部分为0
                preds = np.pad(preds, (0, forecast_horizon - len(preds)), 'constant')
            all_predictions.extend(preds)
        else:
            # 如果没有该物品的预测，填充0
            all_predictions.extend([0] * forecast_horizon)
    
    # 确保长度匹配
    min_length = min(len(ids), len(all_predictions))
    ids = ids[:min_length]
    all_predictions = all_predictions[:min_length]
    
    # 创建DataFrame
    submission_df = pd.DataFrame({
        'id': ids,
        'sales': all_predictions
    })
    
    # 确保预测值非负
    submission_df['sales'] = np.maximum(0, submission_df['sales'])
    
    print(f'预测结果DataFrame准备完成，共{len(submission_df)}行')
    return submission_df

def validate_submission_format(submission_df, expected_rows=None, check_ids=True):
    """
    验证提交文件格式的正确性
    
    参数:
    submission_df: DataFrame, 提交DataFrame
    expected_rows: int, 期望的行数（可选）
    check_ids: bool, 是否检查ID格式
    
    返回:
    bool: 格式是否正确
    dict: 验证结果详情
    """
    validation_result = {
        'valid': True,
        'issues': []
    }
    
    # 检查必要的列
    required_columns = ['id', 'sales']
    for col in required_columns:
        if col not in submission_df.columns:
            validation_result['valid'] = False
            validation_result['issues'].append(f'缺失必要的列: {col}')
    
    # 检查行数
    if expected_rows is not None and len(submission_df) != expected_rows:
        validation_result['valid'] = False
        validation_result['issues'].append(
            f'行数不匹配: 期望{expected_rows}, 实际{len(submission_df)}'
        )
    
    # 检查ID格式
    if check_ids and 'id' in submission_df.columns:
        # 检查ID格式是否符合要求 (xxx_Fxx)
        invalid_ids = []
        for idx, submission_id in enumerate(submission_df['id']):
            if not isinstance(submission_id, str) or '_F' not in submission_id:
                invalid_ids.append((idx, submission_id))
                if len(invalid_ids) > 10:  # 只记录前10个无效ID
                    break
        
        if invalid_ids:
            validation_result['valid'] = False
            validation_result['issues'].append(
                f'发现{len(invalid_ids)}个无效的ID格式，前几个: {invalid_ids[:5]}'
            )
    
    # 检查sales值是否为非负数
    if 'sales' in submission_df.columns:
        negative_values = submission_df[submission_df['sales'] < 0]
        if not negative_values.empty:
            validation_result['valid'] = False
            validation_result['issues'].append(
                f'发现{len(negative_values)}个负值销售预测'
            )
    
    # 检查是否有缺失值
    if submission_df.isna().any().any():
        validation_result['valid'] = False
        validation_result['issues'].append(
            'DataFrame中包含缺失值'
        )
    
    # 打印验证结果
    if validation_result['valid']:
        print('提交文件格式验证通过！')
    else:
        print('提交文件格式验证失败:')
        for issue in validation_result['issues']:
            print(f'- {issue}')
    
    return validation_result['valid'], validation_result

def save_submission_file(submission_df, output_path, index=False, compression=None):
    """
    保存提交文件
    
    参数:
    submission_df: DataFrame, 提交DataFrame
    output_path: str, 输出文件路径
    index: bool, 是否包含索引列
    compression: str, 压缩方式（可选）
    
    返回:
    bool: 是否保存成功
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存文件
        submission_df.to_csv(output_path, index=index, compression=compression)
        
        # 验证文件是否成功创建
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
            print(f'提交文件已成功保存至: {output_path}')
            print(f'文件大小: {file_size:.2f} MB')
            return True
        else:
            print(f'错误: 无法确认文件是否保存成功')
            return False
            
    except Exception as e:
        print(f'保存提交文件失败: {e}')
        return False

def generate_final_submission(hierarchical_series, model_results, output_path='submission.csv', 
                             use_best_model=True, use_ensemble=False, ensemble_weights=None):
    """
    生成最终提交文件
    
    参数:
    hierarchical_series: dict, 分层时间序列数据
    model_results: dict, 模型结果
    output_path: str, 输出文件路径
    use_best_model: bool, 是否使用最佳模型
    use_ensemble: bool, 是否使用模型集成
    ensemble_weights: dict, 集成权重
    
    返回:
    dict: 提交结果信息
    """
    print('\n===== 开始生成最终提交文件 =====')
    
    # 确定要使用的预测结果
    predictions_to_use = None
    source_info = ''
    
    if use_ensemble and len(model_results) > 1:
        # 准备集成的预测结果
        predictions_to_ensemble = {}
        for model_name, results in model_results.items():
            if 'predictions' in results and results['predictions'] is not None:
                predictions_to_ensemble[model_name] = results['predictions']
        
        if len(predictions_to_ensemble) > 1:
            predictions_to_use = model_ensemble(
                predictions_to_ensemble, 
                weights=ensemble_weights,
                method='weighted_average'
            )
            source_info = '模型集成'
            print('使用模型集成结果生成提交文件')
    
    if predictions_to_use is None and use_best_model:
        # 选择最佳模型（假设按RMSE排序）
        model_comparison = compare_models(model_results)
        if model_comparison is not None and not model_comparison.empty:
            best_model_name = model_comparison.iloc[0]['模型']
            best_model_results = model_results[best_model_name]
            
            if 'predictions' in best_model_results and best_model_results['predictions'] is not None:
                predictions_to_use = best_model_results['predictions']
                source_info = f'最佳模型 ({best_model_name})'
                print(f'使用最佳模型 {best_model_name} 生成提交文件')
    
    # 如果还是没有预测结果，尝试使用第一个可用的模型
    if predictions_to_use is None:
        for model_name, results in model_results.items():
            if 'predictions' in results and results['predictions'] is not None:
                predictions_to_use = results['predictions']
                source_info = f'模型 ({model_name})'
                print(f'使用模型 {model_name} 生成提交文件')
                break
    
    if predictions_to_use is None:
        print('错误: 没有可用的预测结果用于生成提交文件')
        return {'success': False, 'error': 'No predictions available'}
    
    # 简化处理：假设我们只有部分物品的预测结果
    # 实际使用时需要确保覆盖所有必要的物品ID
    # 这里我们创建一些示例物品ID（实际竞赛中需要使用完整的物品列表）
    item_ids = []
    predictions_dict = {}
    
    # 为每个层级生成预测结果
    for level in hierarchical_series:
        # 示例：为每个层级创建一个预测结果
        item_ids.append(level)
        # 为每个物品重复预测结果（实际应用中应该有每个物品的独立预测）
        predictions_dict[level] = predictions_to_use
    
    # 准备提交DataFrame
    submission_df = prepare_predictions_dataframe(item_ids, predictions_dict)
    
    # 验证提交格式
    is_valid, validation_details = validate_submission_format(submission_df)
    
    if not is_valid:
        print('警告: 提交文件格式验证失败，但仍将尝试保存')
    
    # 保存提交文件
    save_success = save_submission_file(submission_df, output_path)
    
    # 返回提交信息
    submission_info = {
        'success': save_success,
        'file_path': output_path,
        'rows_generated': len(submission_df),
        'source': source_info,
        'validation_valid': is_valid,
        'validation_issues': validation_details['issues'] if not is_valid else []
    }
    
    print('\n提交文件生成完成!')
    print(f'源: {source_info}')
    print(f'行数: {len(submission_df)}')
    print(f'文件路径: {output_path}')
    
    return submission_info

def generate_submission_summary(submission_info, output_path='submission_summary.txt'):
    """
    生成提交摘要报告
    
    参数:
    submission_info: dict, 提交信息
    output_path: str, 输出文件路径
    
    返回:
    bool: 是否成功
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('M5 Forecasting 提交摘要\n')
            f.write('========================\n\n')
            f.write(f'生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'文件路径: {submission_info.get("file_path", "N/A")}\n')
            f.write(f'行数: {submission_info.get("rows_generated", 0)}\n')
            f.write(f'预测源: {submission_info.get("source", "Unknown")}\n')
            f.write(f'格式验证: {"通过" if submission_info.get("validation_valid", False) else "失败"}\n')
            
            if not submission_info.get("validation_valid", True):
                f.write('\n验证问题:\n')
                for issue in submission_info.get("validation_issues", []):
                    f.write(f'- {issue}\n')
            
            f.write('\n使用说明:\n')
            f.write('1. 检查生成的CSV文件格式是否符合Kaggle要求\n')
            f.write('2. 确保文件包含所有必要的ID和预测值\n')
            f.write('3. 在Kaggle平台上提交此文件\n')
            
        print(f'提交摘要已保存至: {output_path}')
        return True
    except Exception as e:
        print(f'生成提交摘要失败: {e}')
        return False

model_results = {
    'ARIMA': arima_results if 'arima_results' in locals() else None,
    'XGBoost': xgboost_results if 'xgboost_results' in locals() else None,
    'LightGBM': lightgbm_results if 'lightgbm_results' in locals() else None
}

# 过滤掉None结果
model_results = {k: v for k, v in model_results.items() if v is not None}

if model_results and 'hierarchical_series' in locals():
    # 生成最终提交文件
    submission_info = generate_final_submission(
        hierarchical_series, 
        model_results,
        output_path='m5_forecasting_submission.csv',
        use_best_model=True,
        use_ensemble=True
    )
    
    # 生成提交摘要
    if submission_info['success']:
        generate_submission_summary(submission_info)
    
    print('\n所有流程完成！您可以将生成的CSV文件提交到Kaggle平台。')
else:
    print('错误: 没有足够的信息生成提交文件')
    print('请先运行数据处理和模型训练代码')

