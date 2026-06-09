import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



savant_pitch_by_pitch=pd.read_csv("/content/savant_data_2021_2023.csv")
lahman_people_data =pd.read_csv("/content/lahman_people.csv")


savant_pitch_by_pitch.head()


savant_pitch_by_pitch.info()


savant_pitch_by_pitch.isnull().sum()


null_counts = savant_pitch_by_pitch.isnull().sum()
high_null_columns = null_counts[null_counts > 20000]
print(high_null_columns)


lahman_people_data.head()


column_to_drop = ['game_date','zone','game_type','stand','p_throws',
                  'home_team','away_team','type','on_3b','on_2b','on_1b','hc_x','hc_y',
                  'fielder_2','sv_id','sz_top','sz_bot','pitcher_1',
                  'outs_when_up', 'inning', 'inning_topbot',
                  'fielder_2_1','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7','fielder_8','fielder_9',
                  'pitch_name','home_score','away_score','bat_score','fld_score',
                  'post_away_score','post_home_score','post_bat_score','post_fld_score',
                  'if_fielding_alignment','of_fielding_alignment','spin_axis','delta_home_win_exp','delta_run_exp',
                  'role_key','pitch_number_appearance','pitcher_at_bat_number','times_faced']
df_1 =  savant_pitch_by_pitch.drop(column_to_drop, axis=1)


print(df_1.columns)


df_1.info()


df_1.isnull().sum()


null_counts = df_1.isnull().sum()
high_null_columns = null_counts[null_counts > 20000]
print(high_null_columns)


# Define the variables for pitcher and batter
pitcher_variables = [
    'pitcher', 'events', 'description', 'balls', 'strikes', 'pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az', 'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y','release_pos_x', 'release_pos_z',
    'pitch_number', 'sp_indicator', 'rp_indicator', 'game_year', 'game_pk','at_bat_number'
]

batter_variables = [
    'batter', 'events', 'bb_type', 'balls', 'strikes', 'hit_distance_sc', 'hit_location',
    'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle', 'game_year', 'game_pk','at_bat_number'
]


# have the data of pitcher and batter
pitcher_data = df_1[pitcher_variables]
batter_data = df_1[batter_variables]

# check the data after divided
print("Pitcher Data Preview:")
print(pitcher_data.info())

print("\nBatter Data Preview:")
print(batter_data.info())


# The provided code already creates pitcher_data and batter_data DataFrames.
# To download these DataFrames, you can use the to_csv() method.

pitcher_data.to_csv('pitcher_data.csv', index=False)
batter_data.to_csv('batter_data.csv', index=False)

# These CSV files will be downloaded from your Colab environment.
# You can find them in the "Files" tab on the left sidebar.

# Alternatively, you can use the files.download() method from google.colab
from google.colab import files
files.download('pitcher_data.csv')
files.download('batter_data.csv')


def create_comprehensive_pitcher_features(df):
    """整合所有投手相关的多维数据为综合特征"""

    # 1. 速度相关综合特征
    def calculate_total_velocity(vx0, vy0, vz0):
        """计算总速度大小"""
        return np.sqrt(vx0**2 + vy0**2 + vz0**2)

    def calculate_velocity_angles(vx0, vy0, vz0):
        """计算速度的方向角度"""
        horizontal_angle = np.arctan2(vx0, -vy0)
        vertical_angle = np.arctan2(vz0, -vy0)
        return horizontal_angle, vertical_angle

    # 2. 释放点位置相关特征
    def calculate_release_distance(x, y, z):
        """计算释放点到原点的三维距离"""
        return np.sqrt(x**2 + y**2 + z**2)

    def calculate_release_angles(x, y, z):
        """计算释放点的水平和垂直角度"""
        horizontal_angle = np.arctan2(x, y)
        vertical_angle = np.arctan2(z, np.sqrt(x**2 + y**2))
        return horizontal_angle, vertical_angle

    # 3. 加速度综合特征
    def calculate_total_acceleration(ax, ay, az):
        """计算总加速度大小"""
        return np.sqrt(ax**2 + ay**2 + az**2)

    # 计算综合特征
    df['total_velocity'] = calculate_total_velocity(df['vx0'], df['vy0'], df['vz0'])
    df['horizontal_velocity_angle'], df['vertical_velocity_angle'] = calculate_velocity_angles(
        df['vx0'], df['vy0'], df['vz0']
    )

    df['release_distance'] = calculate_release_distance(
        df['release_pos_x'], df['release_pos_y'], df['release_pos_z']
    )
    df['release_horizontal_angle'], df['release_vertical_angle'] = calculate_release_angles(
        df['release_pos_x'], df['release_pos_y'], df['release_pos_z']
    )

    df['total_acceleration'] = calculate_total_acceleration(df['ax'], df['ay'], df['az'])

    # 创建投手特征
    return {
        # 速度综合特征
        'avg_total_velocity': df.groupby('pitcher')['total_velocity'].mean(),
        'total_velocity_std': df.groupby('pitcher')['total_velocity'].std(),
        'avg_velocity_horizontal_angle': df.groupby('pitcher')['horizontal_velocity_angle'].mean(),
        'avg_velocity_vertical_angle': df.groupby('pitcher')['vertical_velocity_angle'].mean(),
        'velocity_angle_consistency': df.groupby('pitcher').agg({
            'horizontal_velocity_angle': 'std',
            'vertical_velocity_angle': 'std'
        }).mean(axis=1),

        # 释放点综合特征
        'avg_release_distance': df.groupby('pitcher')['release_distance'].mean(),
        'release_distance_std': df.groupby('pitcher')['release_distance'].std(),
        'avg_release_horizontal_angle': df.groupby('pitcher')['release_horizontal_angle'].mean(),
        'avg_release_vertical_angle': df.groupby('pitcher')['release_vertical_angle'].mean(),
        'release_angle_consistency': df.groupby('pitcher').agg({
            'release_horizontal_angle': 'std',
            'release_vertical_angle': 'std'
        }).mean(axis=1),

        # 加速度综合特征
        'avg_total_acceleration': df.groupby('pitcher')['total_acceleration'].mean(),
        'total_acceleration_std': df.groupby('pitcher')['total_acceleration'].std(),

        # 投球一致性综合指标
        'mechanical_consistency': df.groupby('pitcher').agg({
            'release_distance': 'std',
            'total_velocity': 'std',
            'total_acceleration': 'std'
        }).mean(axis=1)
    }


pitcher_stats = pitcher_data.groupby(['pitcher', 'game_year']).agg(
    # 控球指标
    avg_balls_per_pa=('balls', 'mean'),     # 每次打席平均坏球数
    avg_strikes_per_pa=('strikes', 'mean'),  # 每次打席平均好球数
    total_pitches=('pitch_number', 'count'),  # 总投球数
    batters_faced=('balls', 'count'),       # 面对打者数

    # 球数分布
    strike_rate=('strikes', lambda x: x.sum() / (x.sum() + pitcher_data.loc[x.index, 'balls'].sum())),  # 好球率

    # 投手角色
    games_as_sp=('sp_indicator', 'sum'),     # 先发场次
    games_as_rp=('rp_indicator', 'sum'),     # 后援场次

    # 效率指标
    pitches_per_pa=('pitch_number', 'mean')  # 每个打席的平均投球数
).reset_index()

# 添加计算字段
pitcher_stats['strike_to_ball_ratio'] = pitcher_stats['avg_strikes_per_pa'] / pitcher_stats['avg_balls_per_pa']
pitcher_stats['pitches_per_game'] = pitcher_stats['total_pitches'] / (pitcher_stats['games_as_sp'] + pitcher_stats['games_as_rp'])

print(pitcher_stats)


pitcher_data.head()


pitcher_data.info()


pitcher_data.isna().sum()


unique_events = pitcher_data['events'].unique()
print(unique_events)


def create_important_event_stats(pitcher_data):
   """计算投手重要的事件统计"""
   return pitcher_data.groupby(['pitcher', 'game_year']).agg({
       'events': [
           # 核心指标
           ('strikeout_rate', lambda x: (x == 'strikeout').mean()),  # 三振率
           ('walk_rate', lambda x: (x == 'walk').mean()),            # 保送率
           ('hit_by_pitch_rate', lambda x: (x == 'hit_by_pitch').mean()),  # 触身球率

           # 被安打指标
           ('single_rate', lambda x: (x == 'single').mean()),        # 一垒安打率
           ('double_rate', lambda x: (x == 'double').mean()),        # 二垒安打率
           ('triple_rate', lambda x: (x == 'triple').mean()),        # 三垒安打率
           ('home_run_rate', lambda x: (x == 'home_run').mean()),    # 全垒打率

           # 出局数指标
           ('field_out_rate', lambda x: (x == 'field_out').mean()),  # 外野手接杀率
           ('force_out_rate', lambda x: (x == 'force_out').mean()),  # 封杀率
           ('double_play_rate', lambda x: (x == 'grounded_into_double_play').mean()),  # 双杀率

           # 计算总数
           ('total_events', 'count')  # 总事件数
       ]
   })

# 添加一些组合指标
def add_combined_stats(stats_df):
   """添加一些组合统计指标"""
   # 展平多级索引列名
   stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

   # 添加组合指标
   stats_df['hit_rate'] = (
       stats_df['events_single_rate'] +
       stats_df['events_double_rate'] +
       stats_df['events_triple_rate'] +
       stats_df['events_home_run_rate']
   )  # 总被安打率

   stats_df['slugging_against'] = (
       stats_df['events_single_rate'] +
       2 * stats_df['events_double_rate'] +
       3 * stats_df['events_triple_rate'] +
       4 * stats_df['events_home_run_rate']
   )  # 被长打率

   stats_df['out_rate'] = (
       stats_df['events_field_out_rate'] +
       stats_df['events_force_out_rate'] +
       stats_df['events_double_play_rate']
   )  # 总出局率

   return stats_df.reset_index()

# 使用示例：
event_stats = create_important_event_stats(pitcher_data)
final_stats = add_combined_stats(event_stats)

# 查看结果
print(final_stats.head())


# 查看所有不同的description值
unique_descriptions = pitcher_data['description'].unique()
print("所有不同的description值：")
print(unique_descriptions)

# 查看每种description的出现频率
desc_counts = pitcher_data['description'].value_counts()
print("\n每种description的出现频次：")
print(desc_counts)


def create_description_stats(pitcher_data):
    """计算投手投球描述的主要统计"""
    return pitcher_data.groupby(['pitcher', 'game_year']).agg({
        'description': [
            # 最主要的投球结果
            ('ball_rate', lambda x: (x == 'ball').mean()),              # 坏球率（129769次）
            ('foul_rate', lambda x: (x == 'foul').mean()),             # 界外球率（67627次）
            ('hit_into_play_rate', lambda x: (x == 'hit_into_play').mean()),  # 击中球率（65450次）
            ('called_strike_rate', lambda x: (x == 'called_strike').mean()),  # 看strike率（64707次）
            ('swinging_strike_rate', lambda x: (x == 'swinging_strike').mean()),  # 挥空率（41807次）

            # 次要但有意义的结果
            ('blocked_ball_rate', lambda x: (x == 'blocked_ball').mean()),     # 封阻球率（9710次）
            ('foul_tip_rate', lambda x: (x == 'foul_tip').mean()),            # 触击球率（3754次）
            ('swinging_strike_blocked_rate', lambda x: (x == 'swinging_strike_blocked').mean()),  # 挥空触击率（2636次）

            # 总投球数
            ('total_pitches', 'count')
        ]
    })

# 添加组合指标
def add_description_combined_stats(stats_df):
    """添加关键组合统计指标"""
    # 展平多级索引列名
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    # 添加重要的组合指标
    stats_df['strike_rate'] = (  # 总好球率
        stats_df['description_called_strike_rate'] +
        stats_df['description_swinging_strike_rate'] +
        stats_df['description_foul_rate']
    )

    stats_df['contact_rate'] = (  # 接触率
        stats_df['description_hit_into_play_rate'] +
        stats_df['description_foul_rate']
    ) / (
        stats_df['description_hit_into_play_rate'] +
        stats_df['description_foul_rate'] +
        stats_df['description_swinging_strike_rate']
    )

    stats_df['swing_and_miss_rate'] = (  # 挥空率
        stats_df['description_swinging_strike_rate'] +
        stats_df['description_swinging_strike_blocked_rate']
    )

    return stats_df.reset_index()


# 查看所有不同的pitch_type值及其频次
pitch_type_counts = pitcher_data['pitch_type'].value_counts()
print("每种pitch_type的出现频次：")
print(pitch_type_counts)


def create_pitch_type_stats(pitcher_data):
    """计算投手投球类型的统计"""
    return pitcher_data.groupby(['pitcher', 'game_year']).agg({
        'pitch_type': [
            # 主要球种 (>20000次)
            ('FF_rate', lambda x: (x == 'FF').mean()),  # 四缝快速球率 (135875)
            ('SL_rate', lambda x: (x == 'SL').mean()),  # 滑球率 (69108)
            ('SI_rate', lambda x: (x == 'SI').mean()),  # 伸卡球率 (60007)
            ('CH_rate', lambda x: (x == 'CH').mean()),  # 变化球率 (44259)
            ('CU_rate', lambda x: (x == 'CU').mean()),  # 曲球率 (29383)
            ('FC_rate', lambda x: (x == 'FC').mean()),  # 切球率 (27568)

            # 次要球种 (>1000次)
            ('KC_rate', lambda x: (x == 'KC').mean()),  # 指关节曲球率 (9043)
            ('FS_rate', lambda x: (x == 'FS').mean()),  # 分叉球率 (5801)
            ('ST_rate', lambda x: (x == 'ST').mean()),  # (4520)
            ('SV_rate', lambda x: (x == 'SV').mean()),  # (1414)

            # 球种多样性统计
            ('pitch_types_used', lambda x: x.nunique()),  # 使用的球种数量
            ('primary_pitch', lambda x: x.mode()[0]),     # 最常用的球种
            ('primary_pitch_rate', lambda x: x.value_counts(normalize=True).max()),  # 主要球种使用率

            # 总数统计
            ('total_pitches', 'count')
        ]
    })

# 添加组合指标
def add_pitch_type_combined_stats(stats_df):
    """添加球种组合统计指标"""
    # 展平多级索引列名
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    # 添加组合指标
    stats_df['fastball_family_rate'] = (  # 快速球系使用率
        stats_df['pitch_type_FF_rate'] +
        stats_df['pitch_type_SI_rate'] +
        stats_df['pitch_type_FC_rate']
    )

    stats_df['breaking_ball_rate'] = (  # 折射球系使用率
        stats_df['pitch_type_SL_rate'] +
        stats_df['pitch_type_CU_rate'] +
        stats_df['pitch_type_KC_rate']
    )

    stats_df['offspeed_rate'] = (  # 变速球系使用率
        stats_df['pitch_type_CH_rate'] +
        stats_df['pitch_type_FS_rate']
    )

    return stats_df.reset_index()

# 使用示例：
pitch_type_stats = create_pitch_type_stats(pitcher_data)
final_pitch_stats = add_pitch_type_combined_stats(pitch_type_stats)


# 确保 game_year 和 game_pk 是整数
pitcher_data['game_year'] = pitcher_data['game_year'].astype('int64')
pitcher_data['game_pk'] = pitcher_data['game_pk'].astype('int64')



def create_career_stats(pitcher_data):
    """创建投手的生涯统计特征"""
    # 正确处理game_year
    pitcher_data['game_year'] = pitcher_data['game_year'].astype(str).str[-4:].astype(int)
    print("Years in data:", pitcher_data['game_year'].unique())
    print("Number of pitchers:", len(pitcher_data['pitcher'].unique()))

    # 2021-2022年的数据作为历史数据
    historical_data = pitcher_data[pitcher_data['game_year'].isin([2021, 2022])]
    print("Number of pitchers in historical data:", len(historical_data['pitcher'].unique()))

    career_stats = historical_data.groupby('pitcher').agg({
        # 生涯控球指标
        'balls': ['mean', 'std'],                # 坏球率及其稳定性
        'strikes': ['mean', 'std'],              # 好球率及其稳定性

        # 生涯球速指标
        'release_speed': ['mean', 'std', 'max'], # 球速特征

        # 生涯出场统计
        'pitch_number': 'sum',                   # 总投球数

        # 生涯表现指标
        'events': lambda x: (x == 'strikeout').mean()  # 生涯三振率
    })

    # 展平多级索引列名
    career_stats.columns = ['_'.join(col).strip() for col in career_stats.columns.values]
    career_stats = career_stats.reset_index()

    print("Career stats shape after creation:", career_stats.shape)
    print("Sample of career stats:")
    print(career_stats.head())

    return career_stats

# 重新运行career_stats的创建
career_stats = create_career_stats(pitcher_data)


def create_player_features(lahman_df, game_year):
    """创建球员个人特征，基于比赛年份"""
    # 确保只使用有效的player_mlb_id
    valid_lahman = lahman_df[lahman_df['player_mlb_id'].notna()].copy()

    # 打印一些调试信息
    print("Number of valid player IDs:", len(valid_lahman))
    print("Sample of valid IDs:", valid_lahman['player_mlb_id'].head())

    features_df = pd.DataFrame({
        'player_mlb_id': valid_lahman['player_mlb_id'],  # 保持ID
        'age': game_year - valid_lahman['birthYear'],
        'experience': game_year - pd.to_datetime(valid_lahman['debut']).dt.year,
        'weight': valid_lahman['weight'],
        'height': valid_lahman['height'],
        'bmi': (valid_lahman['weight'] * 703) / (valid_lahman['height'] ** 2)
    })

    # 创建类别特征
    bats_dummies = pd.get_dummies(valid_lahman['bats'], prefix='bats')
    throws_dummies = pd.get_dummies(valid_lahman['throws'], prefix='throws')

    # 合并所有特征，保持索引对应关系
    features_df = pd.concat([features_df, bats_dummies, throws_dummies], axis=1)

    print("\nFeatures DataFrame info:")
    print(features_df.info())

    return features_df

def combine_all_features(pitcher_data, lahman_data):
    """组合所有特征"""
    # 获取数据中的年份
    game_year = int(pitcher_data['game_year'].astype(str).str[-4:].iloc[0])
    print(f"Using game year: {game_year}")

    # 创建球员特征
    player_features = create_player_features(lahman_data, game_year)

    # 检查pitcher_data中的ID
    print("\nUnique pitchers in pitch data:", len(pitcher_data['pitcher'].unique()))
    print("Sample pitcher IDs:", pitcher_data['pitcher'].head())


    # 计算生涯统计
    career_stats = create_career_stats(pitcher_data)
    career_stats = career_stats.add_prefix('career_')
    career_stats = career_stats.rename(columns={'career_pitcher': 'pitcher'})

    # **只合并生涯统计和球员特征**
    all_features = pd.merge(
        career_stats,
        player_features,
        left_on='pitcher',
        right_on='player_mlb_id',
        how='left'
    )
    print("\nFinal shape:", all_features.shape)

    # 检查最终结果中的空值
    null_counts = all_features.isnull().sum()
    print("\nColumns with null values:")
    print(null_counts[null_counts > 0])

    return all_features

# 使用修改后的函数
final_features = combine_all_features(pitcher_data, lahman_people_data)



final_features.head()


# 检查ID的匹配情况
def check_id_matching(pitcher_data, lahman_data):
    pitcher_ids = set(pitcher_data['pitcher'].unique())
    lahman_ids = set(lahman_people_data['player_mlb_id'].dropna().unique())

    matches = pitcher_ids.intersection(lahman_ids)

    print(f"Pitch-by-pitch unique pitchers: {len(pitcher_ids)}")
    print(f"Lahman unique players: {len(lahman_ids)}")
    print(f"Matching IDs: {len(matches)}")

    return matches

matching_ids = check_id_matching(pitcher_data, lahman_people_data)


def combine_all_features(pitcher_data, lahman_data):
    """组合所有特征"""
    # 1. 获取匹配的ID
    matching_ids = set(pitcher_data['pitcher'].unique()).intersection(
        set(lahman_data['player_mlb_id'].dropna().unique())
    )
    print(f"Number of matching IDs: {len(matching_ids)}")

    # 2. 只使用匹配的投手数据
    valid_pitcher_data = pitcher_data[pitcher_data['pitcher'].isin(matching_ids)].copy()
    valid_lahman_data = lahman_data[lahman_data['player_mlb_id'].isin(matching_ids)]

    # 3. 获取 game_year（取数据里的最大年份）
    game_year = pitcher_data['game_year'].max()
    print(f"Using game year: {game_year}")

    # 4. 计算生涯统计
    career_stats = create_career_stats(valid_pitcher_data)
    print("Career stats shape:", career_stats.shape)

    # 5. 计算球员特征（正确传入 game_year）
    player_features = create_player_features(valid_lahman_data, game_year)
    player_features['player_mlb_id'] = valid_lahman_data['player_mlb_id']
    print("Player features shape:", player_features.shape)

    # 6. 合并 **生涯统计** 和 **球员特征**
    all_features = career_stats.merge(
        player_features,
        left_on='pitcher',
        right_on='player_mlb_id',
        how='inner'
    )

    print("\nFinal combined features shape:", all_features.shape)

    # 7. 检查空值
    null_counts = all_features.isnull().sum()
    print("\nColumns with null values:")
    print(null_counts[null_counts > 0])

    return all_features

# 执行合并
final_features = combine_all_features(pitcher_data, lahman_people_data)

# 查看结果
print("\nFinal dataset columns:", final_features.columns.tolist())
print("\nFirst few rows of the final dataset:")
print(final_features.head())



print(pitcher_data.columns)
print(lahman_people_data.columns)



print([col for col in pitcher_data.columns if 'year' in col.lower()])


print(pitcher_data.columns)


print(pitcher_data.dtypes['game_year'])


pitcher_data.head()


if 'game_year' not in pitcher_data.columns:
    print("Warning: 'game_year' is missing in pitcher_data!")

if 'game_year' not in lahman_people_data.columns:
    print("Warning: 'game_year' is missing in lahman_people_data!")



print(pitcher_data.columns)  # 确保 'game_year' 存在



print("Pitcher Data Columns:", pitcher_data.columns)
print("Lahman People Data Columns:", lahman_people_data.columns)



pitcher_data.rename(columns={'pitcher': 'player_mlb_id'}, inplace=True)



print(pitcher_data.columns)



# 使用 'player_mlb_id' 作为匹配列进行内连接
merged_data = pitcher_data.merge(lahman_people_data, on='player_mlb_id', how='inner')

# 确保 'game_year' 存在
if 'game_year' not in merged_data.columns:
    merged_data['game_year'] = pitcher_data['game_year']

print(merged_data.head())



# 检查 final_features 的列名
print(final_features.columns)



final_features.head()


final_features.info()


# 计算 playing_time
batters_faced_averages = (
    pitcher_data.groupby(['player_mlb_id', 'game_year', 'game_pk', 'at_bat_number'])  # 按投手、赛季、比赛、打席分组
    .size()  # 计算唯一组合的数量
    .reset_index(name='at_bat_count')  # 命名计数列
    .groupby(['player_mlb_id', 'game_year'])  # 按投手和赛季分组
    .size()  # 计算每个投手在每个赛季的打席数
    .reset_index(name='playing_time')  # 命名 playing_time 列
    .groupby('player_mlb_id')  # 最后按投手 ID 分组
    .playing_time.mean().round(0)  # 计算平均值并四舍五入
    .reset_index()
)

# 将 playing_time 合并到 merged_data
merged_data = merged_data.merge(batters_faced_averages, on='player_mlb_id', how='left')

# 现在 merged_data 里应该包含 playing_time
print(merged_data[['player_mlb_id', 'game_year', 'playing_time']].head())



print(final_features.columns)



final_features.head()


print(merged_data.columns)


print(final_features.columns)


# Merge final_features with batters_faced_averages to align the rows
final_features = final_features.merge(batters_faced_averages[['player_mlb_id', 'playing_time']],
                                      on='player_mlb_id', how='left')

# Now X and y will have the same number of rows
X = final_features.drop(columns=['playing_time'])
y = final_features['playing_time']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training data shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Testing data shape: X_test = {X_test.shape}, y_test = {y_test.shape}")



# 确保 'playing_time' 已经计算并存在于 merged_data
train_data = merged_data[merged_data['game_year'].between(2021, 2023)].copy()
predict_data = merged_data[merged_data['game_year'] == 2023].copy()

# 检查数据形状
print("Train data shape:", train_data.shape)
print("Predict data shape:", predict_data.shape)



# 检查非数值型列
non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_columns)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 移除不需要的列
X = final_features.drop(columns=['player_mlb_id'])

# 识别数值列和类别列
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = ['bats_B', 'bats_L', 'bats_R', 'throws_L', 'throws_R']

# 对类别变量进行独热编码
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# 只对数值特征进行标准化
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# 假设 final_features 中已经有 'playing_time' 作为目标变量
y = final_features['playing_time']

# 划分训练集和测试集（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印划分后的数据集形状
print(f"Training data shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Testing data shape: X_test = {X_test.shape}, y_test = {y_test.shape}")


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 确认我们的目标变量和特征
# 目标变量是 playing_time
y = final_features['playing_time']

# 特征是我们之前处理过的 final_features 中的其他列
# 移除 player_mlb_id 和 playing_time
X = final_features.drop(columns=['player_mlb_id', 'playing_time'])

# 2. 数据预处理
# 分别获取数值和分类特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

# 打印特征信息，确认我们的处理是否正确
print("数值特征:", numeric_features.tolist())
print("分类特征:", categorical_features.tolist())

# 3. 标准化数值特征，编码分类特征
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 标准化数值特征
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 对分类特征进行编码
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 4. 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. 获取特征重要性并展示
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n特征重要性排名:")
print(feature_importance)

# 7. 可视化特征重要性
plt.figure(figsize=(12, 8))
top_15 = feature_importance.head(15)
plt.barh(top_15['特征'], top_15['重要性'])
plt.xlabel('特征重要性')
plt.ylabel('特征')
plt.title('预测 Playing Time 的前15个最重要特征')
plt.tight_layout()
plt.show()

# 8. 评估模型性能
from sklearn.metrics import mean_squared_error, r2_score
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R2 分数: {r2:.2f}")


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 准备数据
y = final_features['playing_time']
X = final_features.drop(columns=['player_mlb_id', 'playing_time'])

# 2. 数据预处理
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

# 标准化数值特征
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 对分类特征进行编码
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. 预测和评估
y_pred = rf.predict(X_test)

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n模型评估:")
print(f"RMSE: {rmse:.2f}")

# 6. 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n特征重要性排名:")
print(feature_importance)

# 7. 可视化特征重要性
plt.figure(figsize=(12, 8))
top_15 = feature_importance.head(15)
plt.barh(top_15['特征'], top_15['重要性'])
plt.xlabel('特征重要性')
plt.ylabel('特征')
plt.title('预测 Playing Time 的前15个最重要特征')
plt.tight_layout()
plt.show()

# 8. 打印一些预测示例
print("\n预测示例:")
example_df = pd.DataFrame({
    '实际值': y_test[:5],
    '预测值': y_pred[:5],
    '差异': abs(y_test[:5] - y_pred[:5])
})
print(example_df)


batter_data.info()


# 1. Calculate average hit distance for each player
batter_stats = batter_data.groupby(['batter', 'game_year'])['hit_distance_sc'].mean().reset_index()
batter_stats = batter_stats.rename(columns={'hit_distance_sc': 'avg_hit_distance'})
batter_stats.head()


# 2. Know the hit location for each player each year
# Function to calculate the most common hit location and hit location standard deviation for each batter per year
def compute_hit_location_features(df):
    grouped = df.groupby(['batter', 'game_year'])['hit_location']

    # Calculate the most common hit location (if multiple modes, take the first one)
    most_common_hit_location = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Calculate the standard deviation of the hit location
    hit_location_std = grouped.std()

    # Combine the results into a new DataFrame
    features = pd.DataFrame({
        'most_common_hit_location': most_common_hit_location,
        'hit_location_std': hit_location_std
    }).reset_index()

    return features

# Calculate hit location features for batters
hit_features = compute_hit_location_features(batter_data)

# Merge batter_stats with hit_features on batter and game_year
batter_stats = batter_stats.merge(hit_features, on=['batter', 'game_year'], how='left')

# Display the result data
batter_stats.head()


# 3. Calculate strike rate/walk rate/BA/
# Strike Rate
batter_stats = batter_stats.merge(
    batter_data.assign(strikeout=batter_data['events'].str.contains('strikeout', na=False).astype(int))
    .groupby(['batter', 'game_year'])['strikeout'].mean()
    .reset_index(name='strikeout_rate'),
    on=['batter', 'game_year'],
    how='left'
)

# Walk Rate
batter_stats = batter_stats.merge(
    batter_data.assign(walk=batter_data['events'].str.contains('walk', na=False).astype(int))
    .groupby(['batter', 'game_year'])['walk'].mean()
    .reset_index(name='walk_rate'),
    on=['batter', 'game_year'],
    how='left'
)


# Batting Average (BA)
# Define hit events
hit_events = ['single', 'double', 'triple', 'home_run']

# Define at-bat events
at_bat_events = [
    'single', 'double', 'triple', 'home_run',  # Hits
    'strikeout', 'field_out', 'grounded_into_double_play', 'fielders_choice',  # Outs
    'fielders_choice_out', 'double_play', 'strikeout_double_play', 'triple_play',  # Outs
    'sac_fly', 'sac_bunt'  # Sacrifice plays
]

# Calculate hits
batter_data['is_hit'] = batter_data['events'].isin(hit_events).astype(int)

# Calculate at-bats
batter_data['is_at_bat'] = batter_data['events'].isin(at_bat_events).astype(int)

# Group by player and year, calculate hits and at-bats
hits_by_batter = batter_data.groupby(['batter', 'game_year'])['is_hit'].sum().reset_index()
hits_by_batter.columns = ['batter', 'game_year', 'hits']

at_bats_by_batter = batter_data.groupby(['batter', 'game_year'])['is_at_bat'].sum().reset_index()
at_bats_by_batter.columns = ['batter', 'game_year', 'at_bats']

# Merge hits and at-bats
ba_stats = pd.merge(hits_by_batter, at_bats_by_batter, on=['batter', 'game_year'], how='left')

# Calculate batting average (BA)
ba_stats['BA'] = ba_stats['hits'] / ba_stats['at_bats']
ba_stats['BA'] = ba_stats['BA'].fillna(0)  # Handle division by zero

# Keep only the BA column
ba_stats = ba_stats[['batter', 'game_year', 'BA']]

# Assume batter_stats already exists
# batter_stats = pd.read_csv('batter_stats.csv')

# Merge batter_stats and ba_stats (only merge the BA column)
batter_stats = pd.merge(batter_stats, ba_stats, on=['batter', 'game_year'], how='left')

#
# Review
batter_stats.head()


# 4. Calculate Hard-hit Rate
# Define hard-hit
batter_data['hard_hit'] = (
    (batter_data['launch_speed'] > 95) &
    (batter_data['launch_angle'] > 10) &
    (batter_data['launch_angle'] < 40)
).astype(int)

# Calculate hard-hit rate
hard_hit_rate = batter_data.groupby(['batter', 'game_year'])['hard_hit'].mean().reset_index()
hard_hit_rate.columns = ['batter', 'game_year', 'hard_hit_rate']

# Merge dataset
batter_stats = pd.merge(batter_stats, hard_hit_rate, on=['batter', 'game_year'], how='left')

#
# Review
batter_stats.head()


# 5. Calculate other averages for the columns
cols_to_avg = [
    'estimated_ba_using_speedangle',
    'estimated_woba_using_speedangle',
    'woba_value',
    'woba_denom',
    'babip_value',
    'iso_value'
]

# Average the dataset
batter_avg_stats = batter_data.groupby(['batter', 'game_year'])[cols_to_avg].mean().reset_index()

# merge to batter_stats
batter_stats = batter_stats.merge(batter_avg_stats, on=['batter', 'game_year'], how='left')

# review
batter_stats.head()


# 6. Analyze BB_type for each player
# Calculate batted ball type distribution
bb_type_distribution = (
    batter_data.groupby(['batter', 'game_year', 'bb_type'])
    .size()
    .unstack(fill_value=0)
)

# Remove the 'bb_type_nan' column if it exists
bb_type_distribution = bb_type_distribution.drop(columns=['nan'], errors='ignore')

# Get unique bb_type values (excluding NaN)
unique_bb_types = [bb_type for bb_type in batter_data['bb_type'].dropna().unique()]

# Ensure all unique batted ball types are included
bb_type_distribution = bb_type_distribution.reindex(columns=unique_bb_types, fill_value=0)

# Compute proportions, avoiding division by zero and NaN values
bb_type_distribution = bb_type_distribution.apply(
    lambda x: x / x.dropna().sum() if x.dropna().sum() > 0 else x, axis=1
)

# Reset index
bb_type_distribution = bb_type_distribution.reset_index()

# Dynamically rename columns based on unique bb_types
bb_type_distribution.columns = ['batter', 'game_year'] + [f'bb_type_{bb_type}' for bb_type in unique_bb_types]

# Merge bb_type_distribution into batter_stats
batter_stats = pd.merge(batter_stats, bb_type_distribution, on=['batter', 'game_year'], how='left')

# Fill any missing values with 0
batter_stats = batter_stats.fillna(0)

# Check if columns are now in batter_stats
batter_stats.head()


# 7. High-Quality Hit Rate
# Count the number of occurrences for each launch_speed_angle type per batter per game_year
launch_speed_counts = (
    batter_data.groupby(['batter', 'game_year', 'launch_speed_angle'])
    .size()
    .unstack(fill_value=0)  # Convert to columns
    .reset_index()
)

# Rename columns to ensure all hit types are included
launch_speed_counts = launch_speed_counts.rename(columns={
    1: 'weak_count',
    2: 'topped_count',
    3: 'under_count',
    4: 'flare_count',
    5: 'solid_count',
    6: 'barrel_count'
}).fillna(0)

# Calculate total number of hits
launch_speed_counts['total_hits'] = (
    launch_speed_counts['weak_count'] +
    launch_speed_counts['topped_count'] +
    launch_speed_counts['under_count'] +
    launch_speed_counts['flare_count'] +
    launch_speed_counts['solid_count'] +
    launch_speed_counts['barrel_count']
)

# Compute high-quality hit rate (solid + barrel) / total
launch_speed_counts['hq_hit_rate'] = (
    (launch_speed_counts['solid_count'] + launch_speed_counts['barrel_count']) /
    launch_speed_counts['total_hits']
)

# Avoid division by zero errors
launch_speed_counts['hq_hit_rate'] = launch_speed_counts['hq_hit_rate'].fillna(0)

# Merge high-quality hit rate into batter_stats
batter_stats = pd.merge(batter_stats, launch_speed_counts[['batter', 'game_year', 'hq_hit_rate']],
                         on=['batter', 'game_year'], how='left')

# Fill any missing values
batter_stats['hq_hit_rate'] = batter_stats['hq_hit_rate'].fillna(0)

# Display results
batter_stats.head()


# Merge lahman_people_data and batter_stats based on player_mlb_id = batter
merged_data = pd.merge(lahman_people_data[['player_mlb_id', 'birthYear', 'weight', 'height', 'bats', 'debut']], batter_stats,
                       left_on='player_mlb_id',
                       right_on='batter',
                       how='right'
)

# Assuming merged_data is your DataFrame
merged_data['age'] = merged_data['game_year'] - merged_data['birthYear']

# Convert the 'debut' column to datetime format and extract the year
merged_data['debut_year'] = pd.to_datetime(merged_data['debut']).dt.year

# Calculate experience
merged_data['experience'] = merged_data['game_year'] - merged_data['debut_year']

batter_stats = merged_data.drop(columns=['player_mlb_id', 'birthYear', 'debut_year', 'debut'])

batter_stats.info()


# Step 1: Group by 'batter', 'game_year', 'game_pk', and 'at_bat_number' to preserve unique observations
plate_appearances = batter_data.groupby(
    ['batter', 'game_year', 'game_pk', 'at_bat_number']
).size().reset_index(name='count')

# Step 2: Now group by 'batter' and 'game_year' to count the number of unique observations (playing time)
plate_appearances = plate_appearances.groupby(
    ['batter', 'game_year']
)['count'].sum().reset_index(name='playing_time')

# Step 3: Merge with the batter_stats
batter_stats = pd.merge(batter_stats, plate_appearances[['batter', 'game_year', 'playing_time']],
                         on=['batter', 'game_year'], how='left')


batter_stats.info()


# Build a lag data to use this year's performance to predict next year's playing time
# Extract 2023 data for training to predict playing_time for 2024
batter_predict_data = batter_stats[batter_stats['game_year'] == 2023].copy()

# Create lagged data for next year's playing_time
batter_lag = batter_stats.copy()

# Create lagged feature: Shift playing_time one year forward as the target for 2024 prediction
batter_lag['next_year_playing_time'] = batter_lag.groupby('batter')['playing_time'].shift(-1)

# Drop rows without next year's playing_time (those that can't be used for training)
batter_lag = batter_lag.dropna(subset=['next_year_playing_time'])

# Sort by year to ensure the data is in correct order
batter_lag = batter_lag.sort_values(by=['game_year'])

# drop null value in batter dataset
batter_lag = batter_lag.dropna()

# One-hot encoding for 'bats'
batter_lag = pd.get_dummies(batter_lag, columns=['bats'])

# View the result
batter_lag.head()


# Build a heatmap to analyse the variables
# Import libararies
import seaborn as sns
import matplotlib.pyplot as plt

# Build the heatmap
batter_lag_without_batter = batter_lag.drop(columns=['batter'])
correlation_matrix = batter_lag_without_batter.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming your data is in a DataFrame called df

# Step 1: Prepare the data
# Features (X) and target (y)
X = batter_lag_without_batter[['weight', 'height', 'avg_hit_distance', 'most_common_hit_location', 'hit_location_std',
        'strikeout_rate', 'walk_rate', 'BA', 'hard_hit_rate', 'estimated_ba_using_speedangle',
        'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value',
        'bb_type_line_drive', 'bb_type_fly_ball', 'bb_type_ground_ball', 'bb_type_popup', 'hq_hit_rate',
        'age', 'experience', 'bats_B', 'bats_L', 'bats_R']]  # Use your independent variables here

y = batter_lag_without_batter['next_year_playing_time']  # Target variable

# Step 2: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)  # Calculate RMSE
r2_rf = r2_score(y_test, y_pred_rf)

# Print the evaluation metrics
print("Random Forest Regressor:")
print(f"Mean Absolute Error: {mae_rf}")
print(f"Mean Squared Error: {mse_rf}")
print(f"Root Mean Squared Error: {rmse_rf}")
print(f"R-squared: {r2_rf}")



import xgboost as xgb

# Prepare data in DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # For regression
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train the model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions
y_pred_xgb = xgb_model.predict(dtest)

# Evaluate the model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)  # Calculate RMSE
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost:")
print(f"Mean Absolute Error: {mae_xgb}")
print(f"Mean Squared Error: {mse_xgb}")
print(f"Root Mean Squared Error: {rmse_xgb}")
print(f"R-squared: {r2_xgb}")


from sklearn.svm import SVR

# Initialize the model
svr_model = SVR(kernel='rbf')

# Fit the model
svr_model.fit(X_train, y_train)

# Make predictions
y_pred_svr = svr_model.predict(X_test)

# Evaluate the model
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)  # Calculate RMSE
r2_svr = r2_score(y_test, y_pred_svr)

print("Support Vector Regressor (SVR):")
print(f"Mean Absolute Error: {mae_svr}")
print(f"Mean Squared Error: {mse_svr}")
print(f"Root Mean Squared Error: {rmse_svr}")
print(f"R-squared: {r2_svr}")



from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model
gbr_model.fit(X_train, y_train)

# Make predictions
y_pred_gbr = gbr_model.predict(X_test)

# Evaluate the model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)  # Calculate RMSE
r2_gbr = r2_score(y_test, y_pred_gbr)

print("Gradient Boosting Regressor:")
print(f"Mean Absolute Error: {mae_gbr}")
print(f"Mean Squared Error: {mse_gbr}")
print(f"Root Mean Squared Error: {rmse_gbr}")
print(f"R-squared: {r2_gbr}")


from sklearn.linear_model import LinearRegression

# Initialize the model
lr_model = LinearRegression()

# Fit the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)  # Calculate RMSE
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression:")
print(f"Mean Absolute Error: {mae_lr}")
print(f"Mean Squared Error: {mse_lr}")
print(f"Root Mean Squared Error: {rmse_lr}")
print(f"R-squared: {r2_lr}")



from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Define your models
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.01, random_state=42)
model_lr = LinearRegression()

# Create the voting regressor (Averaging the predictions)
voting_model = VotingRegressor(estimators=[
    ('random_forest', model_rf),
    ('gradient_boosting', model_gbr),
    ('linear_regression', model_lr)
])

# Train the model
voting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
mae_voting = mean_absolute_error(y_test, y_pred_voting)
mse_voting = mean_squared_error(y_test, y_pred_voting)
rmse_voting = np.sqrt(mse_voting)
r2_voting = voting_model.score(X_test, y_test)

# Print performance metrics
print("Voting Regressor Performance:")
print(f"Mean Absolute Error: {mae_voting}")
print(f"Mean Squared Error: {mse_voting}")
print(f"Root Mean Squared Error: {rmse_voting}")
print(f"R-squared: {r2_voting}")


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Define your base models
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=300, learning_rate=0.01, random_state=42)),
    ('linear_regression', LinearRegression())
]

# Define the meta-model (the model that will make the final prediction)
meta_model = LinearRegression()

# Create the stacking regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_stacking = stacking_model.predict(X_test)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mse_stacking)
r2_stacking = stacking_model.score(X_test, y_test)

# Print performance metrics
print("Stacking Regressor Performance:")
print(f"Mean Absolute Error: {mae_stacking}")
print(f"Mean Squared Error: {mse_stacking}")
print(f"Root Mean Squared Error: {rmse_stacking}")
print(f"R-squared: {r2_stacking}")



'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)
'''


'''
# Instantiate the model with the best hyperparameters
gbr_model2 = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=300,
    subsample=0.8
)

# Train the model on the training data
gbr_model2.fit(X_train, y_train)
# Predictions on the test set
y_pred = gbr_model2.predict(X_test)

# Calculate the performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error
import numpy as np

# MAE, MSE, RMSE, R-squared
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = gbr_model2.score(X_test, y_test)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
'''


# Feature importance
feature_importance = gbr_model.feature_importances_

# Visualize feature importance
plt.barh(X_train.columns, feature_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Gradient Boosting Regressor")
plt.show()


# Drop the specified columns
X_train_dropped = X_train.drop(['bats_R', 'bats_L', 'bats_B'], axis=1)
X_test_dropped = X_test.drop(['bats_R', 'bats_L', 'bats_B'], axis=1)

# Initialize the model
gbr_model3 = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model
gbr_model3.fit(X_train_dropped, y_train)

# Make predictions
y_pred_gbr = gbr_model3.predict(X_test_dropped)

# Evaluate the model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)  # Calculate RMSE
r2_gbr = r2_score(y_test, y_pred_gbr)

print("Gradient Boosting Regressor:")
print(f"Mean Absolute Error: {mae_gbr}")
print(f"Mean Squared Error: {mse_gbr}")
print(f"Root Mean Squared Error: {rmse_gbr}")
print(f"R-squared: {r2_gbr}")


batter_predict_data.info()


# drop null value
batter_predict_data = batter_predict_data.dropna()


# Predict the playing time for 2024
X = batter_predict_data[['weight', 'height', 'avg_hit_distance', 'most_common_hit_location', 'hit_location_std',
        'strikeout_rate', 'walk_rate', 'BA', 'hard_hit_rate', 'estimated_ba_using_speedangle',
        'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value',
        'bb_type_line_drive', 'bb_type_fly_ball', 'bb_type_ground_ball', 'bb_type_popup', 'hq_hit_rate',
        'age', 'experience']]

y_pred_2024 = gbr_model3.predict(X)

# Add the predicted playing_time to the batter_predict_data
batter_predict_data['predicted_playing_time'] = y_pred_2024


def create_combined_predictions(batter_data, pitcher_data, lahman_people_data):
    """整合打者和投手的预测"""

    # Batter预测
    batter_features = ['weight', 'height', 'avg_hit_distance', 'most_common_hit_location',
                      'hit_location_std', 'strikeout_rate', 'walk_rate', 'BA',
                      'hard_hit_rate', 'estimated_ba_using_speedangle',
                      'estimated_woba_using_speedangle', 'woba_value', 'woba_denom',
                      'babip_value', 'iso_value', 'bb_type_line_drive', 'bb_type_fly_ball',
                      'bb_type_ground_ball', 'bb_type_popup', 'hq_hit_rate', 'age', 'experience']

    X_batter = batter_predict_data[batter_features]
    batter_predictions = gbr_model3.predict(X_batter)
    batter_results = pd.DataFrame({
        'PLAYER_ID': batter_predict_data['batter'],
        'PLAYING_TIME': batter_predictions
    })

    # Pitcher预测
    X_pitcher = final_features.drop(columns=['player_mlb_id', 'playing_time'])

    # 数据预处理
    numeric_features = X_pitcher.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_pitcher.select_dtypes(include=['object', 'category', 'bool']).columns

    scaler = StandardScaler()
    X_pitcher[numeric_features] = scaler.fit_transform(X_pitcher[numeric_features])

    for col in categorical_features:
        le = LabelEncoder()
        X_pitcher[col] = le.fit_transform(X_pitcher[col])

    pitcher_predictions = rf.predict(X_pitcher)
    pitcher_results = pd.DataFrame({
        'PLAYER_ID': final_features['pitcher'],
        'PLAYING_TIME': pitcher_predictions
    })

    # 合并预测结果
    combined_predictions = pd.concat([
        batter_results,
        pitcher_results
    ])

    # 按球员ID分组并汇总playing time
    final_predictions = combined_predictions.groupby('PLAYER_ID')['PLAYING_TIME'].sum().reset_index()

    return final_predictions

# 使用函数
final_predictions = create_combined_predictions(batter_data, pitcher_data, lahman_people_data)

# 保存结果
final_predictions.to_csv('submission.csv', index=False)

print("Final predictions shape:", final_predictions.shape)
print("\nSample of predictions:")
print(final_predictions.head())


def analyze_predictions(predictions_df):
    """分析预测结果的统计信息"""
    stats = {
        'Length': len(predictions_df),
        'Min': predictions_df['PLAYING_TIME'].min(),
        'First Quartile': predictions_df['PLAYING_TIME'].quantile(0.25),
        'Median': predictions_df['PLAYING_TIME'].median(),
        'Mean': predictions_df['PLAYING_TIME'].mean(),
        'Third Quartile': predictions_df['PLAYING_TIME'].quantile(0.75),
        'Max': predictions_df['PLAYING_TIME'].max()
    }

    print("\nPrediction Statistics:")
    print(f"Number of players: {stats['Length']}")
    print(f"Min playing time: {stats['Min']:.1f}")
    print(f"1st Qu.: {stats['First Quartile']:.1f}")
    print(f"Median: {stats['Median']:.1f}")
    print(f"Mean: {stats['Mean']:.1f}")
    print(f"3rd Qu.: {stats['Third Quartile']:.1f}")
    print(f"Max: {stats['Max']:.1f}")

    # 检查是否符合预期范围
    if stats['Length'] != 1149:
        print("\nWarning: Number of players differs from expected (1149)")
    if stats['Min'] < 1.0 or stats['Max'] > 1197.0:
        print("\nWarning: Playing time outside expected range (1.0-1197.0)")

    return stats

# 在生成final_predictions后使用
final_predictions = create_combined_predictions(batter_data, pitcher_data, lahman_people_data)
stats = analyze_predictions(final_predictions)

# 如果需要调整预测值到合理范围
final_predictions['PLAYING_TIME'] = final_predictions['PLAYING_TIME'].clip(1.0, 1197.0)

# 再次检查调整后的统计
stats_after_clip = analyze_predictions(final_predictions)

# 保存最终结果
final_predictions.to_csv('submission.csv', index=False)

