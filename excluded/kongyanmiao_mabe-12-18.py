import pandas as pd
import numpy as np
import json
import gc
import os
import glob
import joblib
import itertools
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

# ================= 配置类 (CFG) =================
class CFG:
    # 比赛数据的标准路径 (Kaggle 环境)
    train_path = '/kaggle/input/MABe-mouse-behavior-detection/train.csv'
    test_path = '/kaggle/input/MABe-mouse-behavior-detection/test.csv'
    
    # 追踪数据 (Tracking Data) 的路径
    train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking'
    test_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/test_tracking'
    
    # 标注数据 (Annotations) 的路径
    train_annotation_path = '/kaggle/input/MABe-mouse-behavior-detection/train_annotation'
    
    model_name = 'two_stage_model'  # 模型保存文件夹
    seed = 42

print("Step 1: 环境配置完成。")


# 1. 读取元数据 CSV
print("Step 2: 正在读取 CSV 文件...")

# 检查文件是否存在，防止路径错误
if not os.path.exists(CFG.train_path):
    print(f"错误: 找不到文件 {CFG.train_path}，请检查路径配置。")
else:
    train = pd.read_csv(CFG.train_path)
    test = pd.read_csv(CFG.test_path)

    # 2. 基础预处理
    # 计算每一行有多少只老鼠 (总数4 - 缺失的只数)
    train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

    # 过滤掉 'MABe22' 实验室的数据 (通常不用于训练)
    train = train.query("~lab_id.str.startswith('MABe22_')").reset_index(drop=True)

    # 3. 提取所有出现的“身体部位追踪配置”
    body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

    # 定义需要丢弃的冗余身体部位
    drop_body_parts = [
        'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
        'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
        'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
    ]

    print(f"训练集形状 (过滤后): {train.shape}")
    print(f"测试集形状: {test.shape}")
    print("\n[预览] 训练集前 3 行:")
    display(train.head(3))


import pandas as pd
import numpy as np
import json

def robustify(submission, dataset, traintest, traintest_directory=None):
    """
    清洗提交文件，处理重叠帧，并为缺失视频填充默认值
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    # 1. 确保 start/stop frame 是整数
    # 处理可能的字符串或浮点数，并将 NaN 填充为 0
    submission['start_frame'] = pd.to_numeric(submission['start_frame'], errors='coerce').fillna(0).astype(int)
    submission['stop_frame'] = pd.to_numeric(submission['stop_frame'], errors='coerce').fillna(0).astype(int)
    
    # 2. 过滤无效帧 (Start >= Stop)
    old_len = len(submission)
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != old_len:
        print(f"Robstify: Dropped {old_len - len(submission)} rows with start >= stop")

    # 3. 处理重叠帧 (同个视频同个老鼠不能同时做两个动作)
    # 按 video_id + agent_id + target_id 分组
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = -1
        
        for i, (_, row) in enumerate(group.iterrows()):
            # 如果当前动作的开始时间 小于 上一个动作的结束时间，说明重叠了
            if row['start_frame'] < last_stop_frame:
                mask[i] = False # 丢弃重叠的后一个动作
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])
        
    submission = pd.concat(group_list) if group_list else submission

    # 4. 兜底填充 (处理没有任何预测结果的视频)
    # Kaggle 评分系统要求每个视频至少有一条预测，否则会报错
    s_list = []
    
    # 获取已经有预测的视频列表
    predicted_videos = set(submission.video_id.unique())
    
    for idx, row in dataset.iterrows():
        video_id = row['video_id']
        lab_id = row['lab_id']
        
        # 如果是 MABe22 数据或已有预测，跳过
        if lab_id.startswith('MABe22') or video_id in predicted_videos:
            continue
            
        # 如果 behaviors_labeled 为空，跳过
        if not isinstance(row.behaviors_labeled, str):
            continue

        # print(f"Warning: Video {video_id} has no predictions. Filling with dummy data.")
        
        # 读取该视频的元数据，获取最大帧数
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        if not os.path.exists(path): continue
            
        try:
            vid = pd.read_parquet(path)
            start_frame = vid.video_frame.min()
            stop_frame = vid.video_frame.max() + 1
        except:
            # 如果读取失败，随便给个默认值
            start_frame = 0
            stop_frame = 100
    
        # 解析该视频需要预测哪些行为
        try:
            vid_behaviors = json.loads(row['behaviors_labeled'])
            vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
            vid_behaviors = [b.split(',') for b in vid_behaviors] # [agent, target, action]
            vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        except:
            continue
    
        # 为每对 agent-target 均匀生成一段 dummy 预测
        # 比如把整个视频时长平分给所有可能的动作
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            if len(actions) == 0: continue
            
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        print(f"Robustify: Filled {len(s_list)} empty segments for missing videos.")
        fill_df = pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        submission = pd.concat([submission, fill_df], ignore_index=True)

    submission = submission.reset_index(drop=True)
    return submission

print("Robustify 函数定义完成。")


# ================= 基础工具函数 =================

def safe_rolling(series, window, func, min_periods=None):
    """安全的滚动窗口计算，处理 NaN"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """
    FPS 自适应缩放
    [cite_start]将 '30FPS下的帧数' 转换为 '当前视频FPS下的帧数' [cite: 38]
    """
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """带符号的 FPS 缩放 (用于 shift 操作)"""
    if n_frames_at_30fps == 0: return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """从元数据中获取 FPS"""
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

# ================= 几何计算函数 =================

def _angle_between(v1, v2):
    """计算两个向量之间的夹角 (弧度)"""
    dot = v1['x'] * v2['x'] + v1['y'] * v2['y']
    norm = np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2)
    # 加上 1e-6 防止除以 0
    cos_angle = dot / (norm + 1e-6)
    # clip 防止数值误差导致超出 [-1, 1]
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def _triangle_area(p1, p2, p3):
    """计算三角形面积 (Shoelace公式)，用于衡量身体形态"""
    return 0.5 * np.abs(
        p1['x'] * (p2['y'] - p3['y']) +
        p2['x'] * (p3['y'] - p1['y']) +
        p3['x'] * (p1['y'] - p2['y'])
    )

print("Step 3: 工具函数定义完成。")


# ================= 辅助特征函数 =================

def add_arena_features(X, cx, cy, arena_dims=None):
    """添加场地相关特征 (趋触性/Wall-hugging)"""
    # 如果没有场地大小信息，返回原特征
    if arena_dims is None or np.isnan(arena_dims).any():
        return X
    
    w, h = arena_dims
    # 距离场地中心的距离
    dist_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
    X['dist_center'] = dist_center
    
    # 距离最近墙壁的距离
    dist_wall_x = np.minimum(cx, w - cx)
    dist_wall_y = np.minimum(cy, h - cy)
    X['dist_wall'] = np.minimum(dist_wall_x, dist_wall_y)
    
    return X

# 为了保持代码独立性，把之前用到的 add_* 函数也在这里快速定义一下
# (在实际完整脚本中，这些通常作为独立函数存在，这里为了方便运行整合在一起)
def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff().fillna(0); vel_y = center_y.diff().fillna(0)
    acc_x = vel_x.diff().fillna(0); acc_y = vel_y.diff().fillna(0)
    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)
    for w in [25, 50]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=1).mean()
    return X

def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * fps
    for scale in [20, 60]:
        ws = _scale(scale, fps)
        X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=1).mean()
        X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=1).std()
    return X

# ================= 单体特征提取 (Single) =================

def transform_single(single_mouse, body_parts_tracked, fps, arena_dims=None):
    """
    输入: 单只老鼠的坐标 DataFrame (index=frame, columns=bodyparts)
    输出: 特征 DataFrame
    """
    # 获取这只老鼠有哪些身体部位被追踪了 (例如 'nose', 'tail_base')
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # 1. 基础点对点距离 (形态学)
    # 计算所有部位两两之间的欧几里得距离平方
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    
    # 2. 几何特征 (面积与角度)
    # 头部面积 (Nose-EarL-EarR)
    if all(p in available_body_parts for p in ['nose', 'ear_left', 'ear_right']):
        X['head_area'] = _triangle_area(single_mouse['nose'], single_mouse['ear_left'], single_mouse['ear_right'])
    
    # 脊柱弯曲角 (Nose-Center-Tail)
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['spine_angle'] = _angle_between(v1, v2)

    # 3. 运动学特征 (基于身体中心)
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        # 场地位置 (Wall hugging)
        X = add_arena_features(X, cx, cy, arena_dims)

        for w in [5, 15, 30]:
            ws = _scale(w, fps) # 自适应窗口
            roll = dict(min_periods=1, center=True)
            
            # 位置均值 (平滑轨迹)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            
            # 速度与高频震动 (Jitter)
            vel_x = cx.diff().fillna(0); vel_y = cy.diff().fillna(0)
            speed = np.sqrt(vel_x**2 + vel_y**2) * fps
            
            X[f'speed_mean_{w}'] = speed.rolling(ws, **roll).mean()
            
            # Jitter: 加速度的大小 (Grooming时身体中心会有微小高频震动)
            acc_mag = np.sqrt(vel_x.diff()**2 + vel_y.diff()**2).fillna(0)
            X[f'jitter_{w}'] = acc_mag.rolling(ws, **roll).mean()
            
            # 静止比率 (Resting Ratio): 窗口内有多少比例的时间速度接近0
            is_resting = (speed < 1.0).astype(float)
            X[f'rest_ratio_{w}'] = is_resting.rolling(ws, **roll).mean()

        # 调用其他特征函数
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)

    elite_keywords = [
        'speed',        # 速度绝对是核心
        'jitter',       # 抓挠/理毛的关键
        'facing',       # 社交朝向变化
        'spine_angle',  # 身体弯曲（转身/蜷缩）
        'head_area'     # 头部伸缩（探测）
    ]
    
    # 只保留同时满足：1. 在当前X中存在 2. 属于精英特征 的列
    target_cols = [c for c in X.columns if any(k in c for k in elite_keywords)]
    
    # 2. 减少时间步长：如果内存还不够，只保留一个步长（比如只用 10）
    # 去掉 [5, 15]，改成 [10] 可以减少一半的新增列
    steps = [6] # 取个中间值，约 0.2秒
    
    for step in steps:
        X[[f'{c}_lag{step}' for c in target_cols]] = X[target_cols].shift(step).fillna(method='bfill')
        X[[f'{c}_lead{step}' for c in target_cols]] = X[target_cols].shift(-step).fillna(method='ffill')
    
    # 3. 差分只保留给 "speed" (加速度)
    # 仅计算速度的变化，其他特征的差分暂时舍弃以省内存
    speed_cols = [c for c in target_cols if 'speed' in c]
    for c in speed_cols:
        X[f'{c}_diff_6'] = X[c] - X[c].shift(6).fillna(0)
    return X.astype(np.float16)

# ================= 双体特征提取 (Pair) =================

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """
    输入: 两只老鼠的联合 DataFrame (columns=['A', 'B'])
    输出: 交互特征 DataFrame
    """
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # 1. 跨个体距离 (Inter-mouse distances)
    # 计算 A 的所有部位 到 B 的所有部位 的距离
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })

    # 2. 社交几何 (Social Geometry) - 关键改进点
    has_vec_parts_A = all(p in avail_A for p in ['nose', 'body_center'])
    has_vec_parts_B = all(p in avail_B for p in ['nose', 'body_center'])
    
    if has_vec_parts_A and has_vec_parts_B:
        cA = mouse_pair['A']['body_center']
        cB = mouse_pair['B']['body_center']
        noseA = mouse_pair['A']['nose']
        
        # 1. 构建 A 的身体向量 (Center -> Nose)
        vec_body_A = noseA - cA
        norm_body_A = np.sqrt(vec_body_A['x']**2 + vec_body_A['y']**2) + 1e-6
        
        # 单位方向向量 (Heading Vector)
        u_A_x = vec_body_A['x'] / norm_body_A
        u_A_y = vec_body_A['y'] / norm_body_A
        
        # 2. 构建 A -> B 的向量
        vec_AB = cB - cA
        
        # 3. 投影计算 (Egocentric Coordinates)
        # 前向距离 (Forward): A到B的向量 在 A朝向上的投影
        # 正值表示 B 在 A 前方，负值表示在后方
        X['dist_forward'] = vec_AB['x'] * u_A_x + vec_AB['y'] * u_A_y
        
        # 侧向距离 (Lateral): A到B的向量 在 A朝向的法线上的投影
        # 垂直向量 (-y, x)
        X['dist_lateral'] = vec_AB['x'] * (-u_A_y) + vec_AB['y'] * u_A_x
        
        # 4. 相对角度 (A看B的角度，-pi 到 pi)
        # 使用 arctan2 计算更精确的角度
        X['angle_A_to_B'] = np.arctan2(X['dist_lateral'], X['dist_forward'])
    # 3. 动态交互 (Dynamic Interaction)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        # 两只老鼠中心点的距离
        dist_full = np.sqrt(np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1))
        
        for w in [10, 30]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            
            # 接近速度 (Approach Speed): 距离变小的速率
            # diff < 0 表示距离变小(接近)，取负号变正值
            approach_speed = -dist_full.diff().fillna(0) * fps
            X[f'approach_speed_{w}'] = approach_speed.rolling(ws, **roll).mean()
            
            # 速度同步性 (Speed Correlation): 区分追逐(高相关)和路过(低相关)
            vA = np.sqrt(mouse_pair['A']['body_center'].diff()['x']**2 + mouse_pair['A']['body_center'].diff()['y']**2)
            vB = np.sqrt(mouse_pair['B']['body_center'].diff()['x']**2 + mouse_pair['B']['body_center'].diff()['y']**2)
            X[f'speed_corr_{w}'] = vA.rolling(ws, **roll).corr(vB)

    elite_keywords = [
        'speed',        # 速度绝对是核心
        'jitter',       # 抓挠/理毛的关键
        'facing',       # 社交朝向变化
        'spine_angle',  # 身体弯曲（转身/蜷缩）
        'head_area'     # 头部伸缩（探测）
    ]
    
    # 只保留同时满足：1. 在当前X中存在 2. 属于精英特征 的列
    target_cols = [c for c in X.columns if any(k in c for k in elite_keywords)]
    
    # 2. 减少时间步长：如果内存还不够，只保留一个步长（比如只用 10）
    # 去掉 [5, 15]，改成 [10] 可以减少一半的新增列
    steps = [6] # 取个中间值，约 0.2秒
    
    for step in steps:
        X[[f'{c}_lag{step}' for c in target_cols]] = X[target_cols].shift(step).fillna(method='bfill')
        X[[f'{c}_lead{step}' for c in target_cols]] = X[target_cols].shift(-step).fillna(method='ffill')
    
    # 3. 差分只保留给 "speed" (加速度)
    # 仅计算速度的变化，其他特征的差分暂时舍弃以省内存
    speed_cols = [c for c in target_cols if 'speed' in c]
    for c in speed_cols:
        X[f'{c}_diff_6'] = X[c] - X[c].shift(6).fillna(0)
        
    return X.astype(np.float16)

# ================= 数据生成器 =================

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    读取 Parquet 文件，生成 (特征, 元数据, 标签) 的元组流。
    
    [修复内容]: 
    1. 增加了对 behaviors_labeled 字段类型的检查，跳过非字符串 (NaN) 数据。
    2. 增加了 JSON 解析的异常捕获。
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        
    for _, row in dataset.iterrows():
        # 【修复关键点 1】: 检查标签是否为字符串。如果是 float (NaN)，直接跳过该视频。
        if not isinstance(row.behaviors_labeled, str):
            continue
            
        lab_id = row.lab_id
        video_id = row.video_id
        
        # 1. 读取追踪数据 (Tracking Data)
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        if not os.path.exists(path): 
            continue
            
        try:
            vid = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading tracking file {path}: {e}")
            continue

        # 过滤冗余身体部位 (如果存在)
        # 注意: drop_body_parts 需在外部定义，或者在这里硬编码
        if 'drop_body_parts' in globals() and len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
            
        # 转换数据格式: (Frame, Mouse, BodyPart) -> (Frame, Mouse_BodyPart_Coord)
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid; gc.collect()
        
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        
        # 空间归一化 (像素 -> 厘米)
        if hasattr(row, 'pix_per_cm_approx') and row.pix_per_cm_approx > 0:
            pvid /= row.pix_per_cm_approx

        # 2. 解析行为标签 (JSON -> DataFrame)
        try:
            vid_behaviors = json.loads(row.behaviors_labeled)
            # 清洗数据格式
            vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
            vid_behaviors = [b.split(',') for b in vid_behaviors]
            # 转换为 DataFrame: [agent, target, action]
            vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        except Exception as e:
            # 【修复关键点 2】: 如果 JSON 解析依然报错，跳过该视频
            print(f"Skipping video {video_id}: JSON parse error - {e}")
            continue
        
        # 3. 读取标注真值 (仅训练模式)
        annot = None
        if traintest == 'train':
            annot_path = path.replace(f'{traintest}_tracking', 'train_annotation')
            if os.path.exists(annot_path):
                try:
                    annot = pd.read_parquet(annot_path)
                except:
                    continue
            else:
                continue

        # 4. 生成 Single 模式数据 (Target == 'self')
        if generate_single:
            subset = vid_behaviors.query("target == 'self'")
            unique_agents = np.unique(subset.agent)
            
            for agent_str in unique_agents:
                try:
                    # 解析 Mouse ID (兼容 'mouse1' 和 '1' 两种格式)
                    if 'mouse' in agent_str:
                        mouse_id = int(agent_str.replace('mouse', ''))
                    else:
                        mouse_id = int(agent_str)
                        
                    # 提取单只老鼠的坐标数据
                    if mouse_id in pvid.columns.get_level_values(0):
                        single_mouse = pvid.loc[:, mouse_id]
                        
                        meta = pd.DataFrame({
                            'video_id': video_id,
                            'agent_id': agent_str,
                            'target_id': 'self',
                            'video_frame': single_mouse.index,
                            'frames_per_second': row.frames_per_second,
                            'arena_width': row.get('arena_width_cm', np.nan),
                            'arena_height': row.get('arena_height_cm', np.nan)
                        })
                        
                        if traintest == 'train':
                            possible_actions = np.unique(subset.query("agent == @agent_str").action)
                            # 初始化全0标签矩阵
                            y = pd.DataFrame(0, index=single_mouse.index, columns=possible_actions)
                            
                            # 填入标注数据
                            if annot is not None:
                                agent_annot = annot.query("agent_id == @mouse_id and target_id == @mouse_id")
                                for _, r in agent_annot.iterrows():
                                    if r['action'] in y.columns:
                                        y.loc[r['start_frame']:r['stop_frame'], r['action']] = 1
                            yield 'single', single_mouse, meta, y
                        else:
                            # 测试模式返回需要预测的动作列表
                            possible_actions = np.unique(subset.query("agent == @agent_str").action)
                            yield 'single', single_mouse, meta, possible_actions
                except Exception as e:
                    # print(f"Error yielding single data: {e}")
                    pass

        # 5. 生成 Pair 模式数据 (Target != 'self')
        if generate_pair:
            subset = vid_behaviors.query("target != 'self'")
            if len(subset) > 0:
                all_mice = np.unique(pvid.columns.get_level_values(0))
                # 遍历所有两两组合
                for m1, m2 in itertools.permutations(all_mice, 2):
                    agent_str = f"mouse{m1}"
                    target_str = f"mouse{m2}"
                    
                    # 检查这对老鼠是否有互动任务
                    pair_actions = subset.query("agent == @agent_str and target == @target_str")
                    if len(pair_actions) == 0: 
                        continue
                        
                    # 拼接两只老鼠的数据
                    mouse_pair = pd.concat([pvid[m1], pvid[m2]], axis=1, keys=['A', 'B'])
                    
                    meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        'frames_per_second': row.frames_per_second
                    })
                    
                    possible_actions = np.unique(pair_actions.action)
                    
                    if traintest == 'train':
                        y = pd.DataFrame(0, index=mouse_pair.index, columns=possible_actions)
                        if annot is not None:
                            # 注意 annot 中的 target_id 也是 int
                            pair_annot = annot.query("agent_id == @m1 and target_id == @m2")
                            for _, r in pair_annot.iterrows():
                                if r['action'] in y.columns:
                                    y.loc[r['start_frame']:r['stop_frame'], r['action']] = 1
                        yield 'pair', mouse_pair, meta, y
                    else:
                        yield 'pair', mouse_pair, meta, possible_actions

print("Step 4: 特征提取与生成器定义完成。")


# ================= Step 5: 特征工程测试脚本 =================

print("Step 5: 开始测试特征提取...")

# 1. 随机取一个样本视频 (这里取第一行)
sample_row = train.iloc[0:1].copy()
video_id = sample_row.video_id.iloc[0]
lab_id = sample_row.lab_id.iloc[0]
print(f"测试视频 ID: {video_id} (Lab: {lab_id})")

# 2. 获取该视频的 FPS 和 身体部位配置
# 注意：这里需要重新获取一下 body_parts_tracked，因为 generate_mouse_data 内部不处理这个
bpt_str = sample_row.body_parts_tracked.iloc[0]
bpt = json.loads(bpt_str)
# 过滤冗余部位
if len(bpt) > 5:
    bpt = [b for b in bpt if b not in drop_body_parts]
print(f"追踪的身体部位 ({len(bpt)}个): {bpt}")

# 3. 初始化生成器
# 注意：这里只生成 Single 和 Pair 各自的第一个片段来做演示
gen = generate_mouse_data(
    sample_row, 
    'train', 
    traintest_directory=CFG.train_tracking_path,
    generate_single=True, 
    generate_pair=True
)

found_single = False
found_pair = False

print("\n" + "="*40)

# 4. 遍历生成器结果
for mode, data, meta, label in gen:
    
    # --- 测试 Single 模式 ---
    if mode == 'single' and not found_single:
        print(f"【模式: Single】 (Agent: {meta.agent_id.iloc[0]})")
        
        # 获取 FPS
        fps = _fps_from_meta(meta, {}, default_fps=30.0)
        print(f"视频 FPS: {fps}")
        
        # === 核心：调用 transform_single ===
        # 假设这里没有场地大小信息，传入 None
        X_single = transform_single(data, bpt, fps, arena_dims=None)
        
        print(f"特征矩阵形状: {X_single.shape} (行=帧数, 列=特征数)")
        
        # 检查我们新加的特征是否存在
        check_feats = ['head_area', 'spine_angle', 'jitter_5', 'rest_ratio_30']
        print(f"检查新特征是否存在:")
        for f in check_feats:
            status = "✅ 存在" if f in X_single.columns else "❌ 缺失 (可能缺少对应身体部位)"
            print(f"  - {f}: {status}")
            
        # 打印前5行数据预览
        print("\n数据预览 (前5行, 随机5列):")
        # 随机选几列展示，包含新特征
        cols_to_show = [c for c in check_feats if c in X_single.columns]
        if len(cols_to_show) < 5:
            cols_to_show += list(X_single.columns[:5])
        display(X_single[cols_to_show].head())
        
        found_single = True
        print("-" * 40)

    # --- 测试 Pair 模式 ---
    elif mode == 'pair' and not found_pair:
        print(f"【模式: Pair】 (Agent: {meta.agent_id.iloc[0]}, Target: {meta.target_id.iloc[0]})")
        
        fps = _fps_from_meta(meta, {}, default_fps=30.0)
        
        # === 核心：调用 transform_pair ===
        X_pair = transform_pair(data, bpt, fps)
        
        print(f"特征矩阵形状: {X_pair.shape}")
        
        # 检查新特征
        check_feats = ['A_facing_B', 'relative_orientation', 'approach_speed_30', 'speed_corr_30']
        print(f"检查社交新特征:")
        for f in check_feats:
            status = "✅ 存在" if f in X_pair.columns else "❌ 缺失"
            print(f"  - {f}: {status}")
            
        print("\n数据预览 (前5行, 随机5列):")
        cols_to_show = [c for c in check_feats if c in X_pair.columns]
        if len(cols_to_show) < 5:
            cols_to_show += list(X_pair.columns[:5])
        display(X_pair[cols_to_show].head())
        
        found_pair = True
        print("-" * 40)
        
    # 如果两种模式都测过了，就退出循环
    if found_single and found_pair:
        break

if not found_single and not found_pair:
    print("警告: 该视频没有生成任何数据 (可能没有标注或文件缺失)。")


# ================= Step 6: 定义两阶段训练函数 =================

from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm  # 引入进度条工具

def tune_threshold(y_prob, y_true):
    """简单的阈值搜索，寻找最佳 F1 分数对应的阈值"""
    best_f1 = 0
    best_th = 0.3
    # 在 0.1 到 0.9 之间搜索
    for th in np.arange(0.1, 0.9, 0.05):
        pred = (y_prob >= th).astype(int)
        # 如果预测全是0，f1 undefined，这里处理一下
        if pred.sum() == 0 and y_true.sum() == 0:
            f1 = 1.0
        else:
            f1 = f1_score(y_true, pred, zero_division=0)
            
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th


def train_stage1_binary(X, label, meta, section_name):
    """
    Stage 1: 训练二分类 Proposal 模型 (GPU 加速)
    目标: 快速筛选出包含动作的时间片段。
    """
    print(f"  [Stage 1] Training Binary Proposal Model (GPU)...")
    
    y_binary = label.max(axis=1).values.astype(int)
    groups = meta.video_id
    
    # GPU 参数
    model_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 600,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 2.0,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        'device': 'gpu',            # 启用 GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    cv = GroupKFold(n_splits=3)
    oof_binary = np.zeros(len(y_binary))
    models = []
    
    # 手动打印 Fold 进度
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_binary, groups)):
        try:
            model = LGBMClassifier(**model_params)
            model.fit(X.iloc[train_idx], y_binary[train_idx])
        except Exception as e:
            print(f"    Fold {fold+1}: GPU failed, fallback to CPU.")
            model_params['device'] = 'cpu'
            model = LGBMClassifier(**model_params)
            model.fit(X.iloc[train_idx], y_binary[train_idx])
            
        oof_binary[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        models.append(model)
        
    best_th = tune_threshold(oof_binary, y_binary)
    score = f1_score(y_binary, (oof_binary >= best_th).astype(int), zero_division=0)
    print(f"  [Stage 1] Binary F1: {score:.4f} (Threshold: {best_th:.2f})")
    
    # 保存
    save_dir = f"{CFG.model_name}/{section_name}/stage1"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(models, f"{save_dir}/binary_models.pkl")
    joblib.dump(best_th, f"{save_dir}/binary_threshold.pkl")
    
    return oof_binary, best_th

def train_stage2_specific(X, label, meta, section_name):
    """
    Stage 2: 训练具体动作分类器 (带进度条 + 智能 GPU 切换)
    """
    print(f"  [Stage 2] Training Specific Action Classifiers...")
    
    thresholds = {}
    save_dir_base = f"{CFG.model_name}/{section_name}/stage2"
    action_columns = label.columns
    
    # === 进度条包裹循环 ===
    for action in tqdm(action_columns, desc="  Training Actions", leave=False):
        
        # 1. 准备数据
        action_mask = ~label[action].isna().values
        if action_mask.sum() == 0: continue
            
        y_action = label[action][action_mask].values.astype(int)
        
        # 样本过少跳过
        if y_action.sum() < 5:
            thresholds[action] = 1.0 
            continue
            
        X_action = X[action_mask]
        groups_action = meta.video_id[action_mask]
        
        # 2. 智能设备选择: 样本 > 1万 用 GPU，否则用 CPU (避免传输开销)
        use_gpu = len(X_action) > 10000
        
        model_params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1,
            'device': 'gpu' if use_gpu else 'cpu'
        }
        
        # 3. 交叉验证训练
        cv = GroupKFold(n_splits=3)
        oof_action = np.zeros(len(y_action))
        models = []
        
        for train_idx, valid_idx in cv.split(X_action, y_action, groups_action):
            try:
                model = LGBMClassifier(**model_params)
                model.fit(X_action.iloc[train_idx], y_action[train_idx])
            except:
                # 兜底回退 CPU
                model_params['device'] = 'cpu'
                model = LGBMClassifier(**model_params)
                model.fit(X_action.iloc[train_idx], y_action[train_idx])
                
            oof_action[valid_idx] = model.predict_proba(X_action.iloc[valid_idx])[:, 1]
            models.append(model)
            
        # 4. 保存结果
        best_th = tune_threshold(oof_action, y_action)
        thresholds[action] = best_th
        
        action_dir = f"{save_dir_base}/{action}"
        os.makedirs(action_dir, exist_ok=True)
        joblib.dump(models, f"{action_dir}/models.pkl")
        
    joblib.dump(thresholds, f"{save_dir_base}/thresholds.pkl")
    print(f"  [Stage 2] Finished. Thresholds saved.")
    
    return thresholds

print("Step 6: 训练函数定义完成。")


# ================= Step 7: 执行训练主循环  =================
from tqdm.auto import tqdm  

print("Step 7: 开始训练流程...")

# 获取总的 Section 数量
total_sections = len(body_parts_tracked_list)

# 使用 tqdm 包裹最外层循环
# desc: 进度条左边的描述文字
# position=0, leave=True: 保证进度条在 Notebook 中显示正常不闪烁
start_section = 0 
for section in tqdm(range(start_section, total_sections), desc="Overall Progress", position=0, leave=True):
    bpt_str = body_parts_tracked_list[section]
    
    # 解析配置
    try:
        bpt = json.loads(bpt_str)
        if len(bpt) > 5:
            bpt = [b for b in bpt if b not in drop_body_parts]
    except:
        continue
        
    # 打印当前正在处理的 Section 信息
    # tqdm.write 是专门配合进度条的打印函数，不会打断进度条的显示
    tqdm.write(f"\n{'='*10} Section {section+1}/{total_sections} {'='*10}")
    tqdm.write(f"Body Parts ({len(bpt)}): {bpt}")
    
    # 1. 筛选出属于该配置的训练数据
    train_subset = train[train.body_parts_tracked == bpt_str]
    if len(train_subset) == 0: continue

    # 构建 FPS 查找表
    _fps_lookup = (train_subset[['video_id', 'frames_per_second']]
                   .drop_duplicates('video_id')
                   .set_index('video_id')['frames_per_second'].to_dict())
    
    # 2. 分别处理 'single' 和 'pair' 模式
    # 这里也可以加一个小的进度条，或者直接打印
    for mode in ['single', 'pair']:
        tqdm.write(f"  >>> Mode: {mode} (Generating Data...)")
        
        # --- 数据生成 ---
        data_list, label_list, meta_list = [], [], []
        
        gen = generate_mouse_data(
            train_subset, 'train', 
            traintest_directory=CFG.train_tracking_path,
            generate_single=(mode=='single'), 
            generate_pair=(mode=='pair')
        )
        
        # 收集数据
        count = 0
        # 这里的生成器因为不知道总长度，很难加精确进度条，我们每生成100个打印一个点
        for switch, data, meta, label in gen:
            if switch != mode: continue
            data_list.append(data)
            meta_list.append(meta)
            label_list.append(label)
            count += 1
            if count % 100 == 0:
                print(".", end="") # 简单的可视化
            
        print("") # 换行
        tqdm.write(f"  Generated {count} clips.")
        if count == 0: continue
            
        # --- 特征转换 ---
        tqdm.write(f"  Extracting features...")
        X_parts = []
        # 使用 tqdm 显示特征提取进度
        for data_i, meta_i in tqdm(zip(data_list, meta_list), total=len(data_list), desc=f"  Feat Eng ({mode})", leave=False):
            fps = _fps_from_meta(meta_i, _fps_lookup)
            # 根据模式调用不同的特征函数
            if mode == 'single':
                Xi = transform_single(data_i, bpt, fps)
            else:
                Xi = transform_pair(data_i, bpt, fps)
            X_parts.append(Xi)
            
        # 拼接大矩阵
        X_all = pd.concat(X_parts, axis=0, ignore_index=True)
        label_all = pd.concat(label_list, axis=0, ignore_index=True)
        meta_all = pd.concat(meta_list, axis=0, ignore_index=True)
        
        del data_list, meta_list, label_list, X_parts
        gc.collect()
        
        tqdm.write(f"  Data shape: {X_all.shape}")
        
        # --- 开始训练 ---
        section_name = f"{section}_{mode}"
        
        # Stage 1: 二分类
        tqdm.write(f"  Training Stage 1...")
        train_stage1_binary(X_all, label_all, meta_all, section_name)
        
        # Stage 2: 多分类
        tqdm.write(f"  Training Stage 2...")
        train_stage2_specific(X_all, label_all, meta_all, section_name)
        
        del X_all, label_all, meta_all
        gc.collect()

    tqdm.write(f"Section {section} done.")
    
    # [重要] 如果你想跑完全部数据，请保留下面这行注释
    #break 

print("\nStep 7: 训练流程结束。")


# ================= Step 8: 两阶段推理 (Stage 1 + Stage 2) =================
# 包含：流式处理、模型预加载、内存保护

print("Step 8: 开始两阶段推理 (Stage 1 + Stage 2)...")

def get_proposals(binary_prob, threshold=0.25, min_duration=5, smoothing_window=15):
    """
    改进后的后处理函数：
    1. 更大的平滑窗口
    2. 二值化
    3. 闭运算 (Filling Gaps)
    """
    # 1. 概率平滑 (加大窗口，比如 15)
    probs_smooth = pd.Series(binary_prob).rolling(window=smoothing_window, center=True, min_periods=1).mean().fillna(0).values
    
    # 2. 初始 Mask
    binary_mask = (probs_smooth > threshold).astype(int)
    
    # 3. 填补微小空洞 (Gap Filling)
    # 逻辑：如果两个动作之间只有 < 10 帧的间隔，认为它是连续的
    # 这里用简单的循环实现，也可以用 scipy.ndimage.binary_closing
    max_gap = 10 
    zeros = np.where(binary_mask == 0)[0]
    # 这一步比较耗时，这里提供一个简化的 Pandas 实现思路：
    # 将 mask 转换为 Series，计算 cumsum 分组，合并短的 0 组
    
    # 简单实现：再做一次 Rolling Max 膨胀，然后腐蚀 (类似形态学闭运算)
    # 先让动作“变胖”填满缝隙，再“变瘦”回原大小
    import scipy.ndimage as ndimage
    # 结构元素大小控制填补能力
    struct = np.ones(max_gap) 
    binary_mask = ndimage.binary_closing(binary_mask, structure=struct).astype(int)

    # 4. 提取片段
    diffs = np.diff(np.concatenate(([0], binary_mask, [0])))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]
    
    proposals = []
    for s, e in zip(starts, stops):
        # 过滤过短的动作 (可能是噪声)
        if (e - s) >= min_duration:
            proposals.append((s, e))
            
    return proposals

# --- 1. 预加载所有模型 (Pre-load Models) ---
# 这样不用每次处理视频都读硬盘，速度快很多
print("Pre-loading models into memory...")
models_cache = {}

# 遍历所有可能的 Section
for section in range(len(body_parts_tracked_list)):
    models_cache[section] = {}
    for mode in ['single', 'pair']:
        section_name = f"{section}_{mode}"
        model_dir = f"{CFG.model_name}/{section_name}"
        
        # 检查该 Section 是否训练过
        if not os.path.exists(f"{model_dir}/stage1/binary_models.pkl"):
            continue
            
        models_cache[section][mode] = {}
        
        # 加载 Stage 1
        try:
            models_cache[section][mode]['s1_models'] = joblib.load(f"{model_dir}/stage1/binary_models.pkl")
            # 尝试加载阈值，如果没有就用默认 0.25
            if os.path.exists(f"{model_dir}/stage1/binary_threshold.pkl"):
                models_cache[section][mode]['s1_th'] = joblib.load(f"{model_dir}/stage1/binary_threshold.pkl")
            else:
                models_cache[section][mode]['s1_th'] = 0.25
        except Exception as e:
            print(f"Warning: Failed to load Stage 1 for {section_name}: {e}")
            continue

        # 加载 Stage 2
        try:
            if os.path.exists(f"{model_dir}/stage2/thresholds.pkl"):
                thresholds = joblib.load(f"{model_dir}/stage2/thresholds.pkl")
                models_cache[section][mode]['s2_ths'] = thresholds
                models_cache[section][mode]['s2_models'] = {}
                
                for action in thresholds.keys():
                    m_path = f"{model_dir}/stage2/{action}/models.pkl"
                    if os.path.exists(m_path):
                        models_cache[section][mode]['s2_models'][action] = joblib.load(m_path)
        except Exception as e:
            print(f"Warning: Failed to load Stage 2 for {section_name}: {e}")

print(f"Models loaded for sections: {[k for k in models_cache.keys() if models_cache[k]]}")


# --- 2. 主推理循环 (Stream Inference) ---
submission_list = []

for section in range(len(body_parts_tracked_list)):
    bpt_str = body_parts_tracked_list[section]
    
    # [关键] 筛选当前 Section 的测试视频
    # 如果你在本地只训练了 Section 9，而测试视频是 Section 0 的，这里就会过滤掉，这是正常的
    test_subset = test[test.body_parts_tracked == bpt_str]
    if len(test_subset) == 0: 
        continue
    
    # 检查模型是否加载了
    if section not in models_cache:
        # print(f"Skipping Section {section} (Videos exist but no model trained)")
        continue

    print(f"Processing Section {section} ({len(test_subset)} videos)...")
    
    try:
        bpt = json.loads(bpt_str)
        if len(bpt) > 5:
            bpt = [b for b in bpt if b not in drop_body_parts]
    except:
        continue

    # FPS 查找表
    _fps_lookup = (test_subset[['video_id', 'frames_per_second']]
                   .drop_duplicates('video_id')
                   .set_index('video_id')['frames_per_second'].to_dict())

    for mode in ['single', 'pair']:
        # 检查该模式是否有模型
        if mode not in models_cache[section] or 's1_models' not in models_cache[section][mode]:
            continue
            
        # 拿缓存的模型
        cache = models_cache[section][mode]
        s1_models = cache['s1_models']
        s1_th = cache['s1_th']
        s2_models_dict = cache.get('s2_models', {})
        s2_ths = cache.get('s2_ths', {})

        # 生成器：一次只拿一个视频的数据
        gen = generate_mouse_data(
            test_subset, 'test', 
            traintest_directory=CFG.test_tracking_path, 
            generate_single=(mode=='single'), 
            generate_pair=(mode=='pair')
        )
        
        for switch, data, meta, _ in gen:
            if switch != mode: continue
            
            try:
                # 1. 特征提取
                fps = _fps_from_meta(meta, _fps_lookup)
                if mode == 'single':
                    X = transform_single(data, bpt, fps)
                else:
                    X = transform_pair(data, bpt, fps)
                
                # 2. === Stage 1: 找候选片段 (Proposals) ===
                # Ensemble 平均预测
                s1_probs = np.mean([clf.predict_proba(X)[:, 1] for clf in s1_models], axis=0)
                
                # 获取时间段
                proposals = get_proposals(s1_probs, threshold=s1_th)
                
                if not proposals:
                    del X, data, meta
                    continue
                    
                # 3. === Stage 2: 具体分类 ===
                # 如果没有 Stage 2 模型 (比如全被过滤了)，就跳过
                if not s2_models_dict:
                    del X, data, meta
                    continue

                for start_idx, stop_idx in proposals:
                    # 只看这一个小片段
                    X_clip = X.iloc[start_idx:stop_idx]
                    
                    best_action = None
                    best_score = -1.0
                    
                    # 遍历所有动作分类器
                    for action, models in s2_models_dict.items():
                        # 预测概率
                        probs = np.mean([m.predict_proba(X_clip)[:, 1] for m in models], axis=0)
                        
                        # 取 90% 分位数作为该片段的得分
                        raw_score = np.percentile(probs, 90)
                        
                        # 归一化比较
                        th = s2_ths.get(action, 0.5)
                        norm_score = raw_score / (th + 1e-6)
                        
                        if raw_score > th and norm_score > best_score:
                            best_score = norm_score
                            best_action = action
                    
                    # 只有选出了最佳动作才保存
                    if best_action:
                        real_start = meta.video_frame.iloc[start_idx]
                        real_stop = meta.video_frame.iloc[stop_idx-1] + 1
                        
                        submission_list.append({
                            'video_id': meta.video_id.iloc[0],
                            'agent_id': meta.agent_id.iloc[0],
                            'target_id': meta.target_id.iloc[0],
                            'action': best_action,
                            'start_frame': real_start,
                            'stop_frame': real_stop
                        })

                # 及时清理内存
                del X, data, meta, s1_probs
                # gc.collect() # 不必每次都调，影响速度
                
            except Exception as e:
                print(f"Error processing video: {e}")
                continue
    
    # 跑完一个 Section 清理一次
    gc.collect()

# --- 生成最终 CSV ---
print("Inference done. Generating submission...")

if len(submission_list) > 0:
    submission_df = pd.DataFrame(submission_list)
else:
    # 兜底：如果本地测试因为没有对应的模型导致没结果，这是正常的
    # 但为了保证代码不报错，我们生成一行假数据
    print("Warning: No predictions generated (likely due to Section mismatch in local test).")
    submission_df = pd.DataFrame({
        'video_id': [test.video_id.iloc[0]],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['other'],
        'start_frame': [0],
        'stop_frame': [1]
    })

# 最后的清洗和填充
final_submission = robustify(submission_df, test, 'test')
final_submission.to_csv('submission.csv', index=True, index_label='row_id')

print(f"SUCCESS: submission.csv generated with {len(final_submission)} rows.")
print(final_submission.head())

