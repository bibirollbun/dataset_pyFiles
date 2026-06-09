import os
from functools import wraps

import numpy as np
import pandas as pd

DATA_DIR = '/kaggle/input/nfl-big-data-bowl-2025'
CACHE_DATA_DIR = '/kaggle/working/data-cache'


if not os.path.isdir(CACHE_DATA_DIR):
    os.makedirs(CACHE_DATA_DIR, exist_ok=True)


def _make_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        df[col] = df[col].astype('category')


def load_games() -> pd.DataFrame:
    games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'),
                        parse_dates=['gameDate'])

    _make_categoricals(df=games, cat_cols=['homeTeamAbbr', 'visitorTeamAbbr'])

    return games


def load_players() -> pd.DataFrame:
    players = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))

    def height_str_to_inches(s: str) -> int:
        ft, inches = s.split('-')
        return 12 * int(ft) + int(inches)

    players.loc[:, 'height_in'] = players.height.apply(height_str_to_inches)

    _make_categoricals(df=players, cat_cols=['collegeName', 'position'])

    return players


def pq_cache(f_pq):
    def pq_decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            f_pq_full = os.path.join(CACHE_DATA_DIR, f_pq.format(**kwargs))
            try:
                return pd.read_parquet(f_pq_full)
            except FileNotFoundError:
                df = func(*args, **kwargs)
                df.to_parquet(f_pq_full, index=False)
                return df
        return wrapped_func
    return pq_decorator


@pq_cache('plays.pq')
def load_plays() -> pd.DataFrame:
    plays = pd.read_csv(os.path.join(DATA_DIR, 'plays.csv'))
    plays['playNullifiedByPenalty'] = plays['playNullifiedByPenalty'] == 'Y'
    _make_categoricals(df=plays, cat_cols=['possessionTeam', 'defensiveTeam', 'passResult',
                                           'offenseFormation'])
    return plays


def get_ballcarrier() -> pd.DataFrame:
    plays = load_plays()
    return plays[['gameId', 'playId', 'ballCarrierId']]


@pq_cache('tackles.pq')
def load_tackles() -> pd.DataFrame:
    tackles = pd.read_csv(os.path.join(DATA_DIR, 'tackles.csv'))
    return tackles


@pq_cache('tracking_week_{week_num}.pq')
def load_tracking_week(week_num: int) -> pd.DataFrame:
    f_name = os.path.join(DATA_DIR, f'tracking_week_{week_num}.csv')
    tw = pd.read_csv(f_name)
    _make_categoricals(df=tw, cat_cols=['club', 'playDirection', 'event'])
    return tw

@pq_cache('tracking_week_{week_num_start}_{week_num_end}.pq')
def load_all_tracking(week_num_start: int, week_num_end: int) -> pd.DataFrame:
    t = None
    for week_num in range(week_num_start, week_num_end + 1):
        tw = load_tracking_week(week_num=week_num)
        if t is None:
            t = tw
        else:
            t = pd.concat([t, tw])
    return t


games = load_games()
players = load_players()
plays = load_plays()


# just dropping columns for memory reasons, larger compute can hang on to these
tracking = (load_all_tracking(week_num_start=1, week_num_end=9)
            .drop(columns=['frameId', 'time', 'jerseyNumber', 'x', 's', 'a', 'dis', 'o', 'dir']))


# let's add on an indicator for when the player is on offense -- it will be handy later on
tracking = (tracking
            .merge(plays[['gameId', 'playId', 'possessionTeam']].rename(columns={'possessionTeam': 'club'}),
                   how='left', on=['gameId', 'playId', 'club'], indicator=True))

tracking.loc[:, 'is_offense'] = tracking['_merge'] == 'both'

tracking.drop(columns=['_merge'], inplace=True)


assert tracking.shape == (59_327_373, 10)


idx_cols = ['gameId', 'playId']
meta_cols = idx_cols + [
    'qbSpike',
    'qbKneel',
    'playDescription',
]

meta = (plays
        [meta_cols]
        .copy()
        .set_index(idx_cols)
        .sort_index())

meta.loc[:, 'is_spike'] = meta.qbSpike.fillna(False)
meta.loc[:, 'is_kneel'] = meta.qbKneel == 1
meta.loc[:, 'should_ignore'] = meta.is_spike | meta.is_kneel
meta.drop(columns=['qbSpike', 'qbKneel'], inplace=True)

meta.head()


# converting all home / away features to possession team features
def get_possession_versions(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    pvh = (plays
           [['gameId', 'playId', 'possessionTeam',
             'preSnapHomeScore', 'preSnapVisitorScore',
             'preSnapHomeTeamWinProbability']]
           .copy()
           .set_index(['gameId', 'playId'])
           .sort_index())

    pvh = pvh.join(games.set_index('gameId')[['homeTeamAbbr']], how='left')
    pvh.loc[:, 'possession_team_is_home'] = pvh.possessionTeam == pvh.homeTeamAbbr
    pvh.drop(columns='homeTeamAbbr', inplace=True)

    pvh.loc[:, 'home_score_delta'] = pvh.preSnapHomeScore - pvh.preSnapVisitorScore
    pvh.loc[:, 'possession_team_score_delta'] = (2 * pvh.possession_team_is_home - 1) * pvh.home_score_delta
    pvh.drop(columns='home_score_delta', inplace=True)

    pvh.loc[:, 'presnap_possession_team_win_probability'] = np.where(
        pvh.possession_team_is_home,
        pvh.preSnapHomeTeamWinProbability,
        1 - pvh.preSnapHomeTeamWinProbability
    )

    pvh = (pvh
           [['possession_team_is_home',
             'possession_team_score_delta',
             'presnap_possession_team_win_probability']])

    pvh.loc[:, 'possession_team_score_delta_cat'] = np.where(
        pvh.possession_team_score_delta > 16, 'up 3+ scores',
        np.where(
            pvh.possession_team_score_delta > 8, 'up 2 scores',
            np.where(
                pvh.possession_team_score_delta > 0, 'up 1 score',
                np.where(
                    pvh.possession_team_score_delta == 0, 'tied',
                    np.where(
                        pvh.possession_team_score_delta > -9, 'down 1 score',
                        np.where(
                            pvh.possession_team_score_delta > -17, 'down 2 scores',
                            'down 3+ scores'
                        )
                    )
                )
            )
        )
    )

    pvh.possession_team_score_delta_cat = pvh.possession_team_score_delta_cat.astype('category')

    return pvh


possession_team_versions = get_possession_versions(plays=plays, games=games)
possession_team_versions.head()


def add_hash_info(df: pd.DataFrame) -> pd.DataFrame:
    field_width_yds = 53.3
    midfield_yds = field_width_yds / 2

    hash_width_ft = 18.5
    hash_width_yds = hash_width_ft / 3

    qtr_hash_width_yds = hash_width_yds / 4

    y_home_center_boundary = midfield_yds - qtr_hash_width_yds
    y_center_visitor_boundary = midfield_yds + qtr_hash_width_yds

    df.loc[:, 'hv_hash'] = np.where(df.y <= y_home_center_boundary, 'home',
                                    np.where(df.y <= y_center_visitor_boundary, 'center', 'visitor'))

    df.loc[:, 'ball_hash_cat'] = np.where(
        df.hv_hash == 'center',
        'center',
        np.where(
            (df.playDirection == 'right'),
            # heading right with ball on home hash: right of center
            np.where(df.hv_hash == 'home', 'right', 'left'),
            # heading left with ball on home hash: left of center
            np.where(df.hv_hash == 'home', 'left', 'right')
        )
    )
    df.ball_hash_cat = df.ball_hash_cat.astype('category')

    return df


def get_ball_hash_info(tracking: pd.DataFrame) -> pd.DataFrame:
    ball_location = (tracking
                     [(tracking.displayName == 'football')
                      & (tracking.frameType == 'SNAP')]
                     .copy())

    ball_location = add_hash_info(df=ball_location)

    return (ball_location
            [['gameId', 'playId', 'hv_hash', 'ball_hash_cat']]
            .set_index(['gameId', 'playId'])
            .sort_index())


hash_info = get_ball_hash_info(tracking=tracking)
hash_info.head()


def get_personnel_grouping(tracking: pd.DataFrame) -> pd.DataFrame:
    players_on_field = (tracking
                        [(tracking.frameType == 'SNAP')
                         & (tracking.is_offense)]
                        [['gameId', 'playId', 'nflId', 'club']]
                        .copy())

    skill_players = players[players.position.isin(['QB', 'RB', 'WR', 'TE', 'FB'])]

    offense_on_field = (players_on_field
                        .merge(skill_players[['nflId', 'position']], how='inner', on='nflId'))

    num_skill_players = (offense_on_field
                         .drop(columns=['nflId'])
                         .pivot_table(values='club',
                                      index=['gameId', 'playId'],
                                      columns='position',
                                      aggfunc='count',
                                      observed=True)
                         .fillna(0.0))

    num_skill_players.columns.name = None

    num_skill_players.loc[:, 'pg_rb'] = num_skill_players.FB + num_skill_players.RB
    num_skill_players.loc[:, 'pg_te'] = num_skill_players.TE
    num_skill_players.loc[:, 'pg_cat'] = (num_skill_players.pg_rb.astype(int).astype(str) + num_skill_players.pg_te.astype(int).astype(str)).astype('category')

    return num_skill_players


personnel_grouping = get_personnel_grouping(tracking=tracking)

personnel_grouping.head()


personnel_grouping.pg_cat.value_counts()


def get_plays_w_motion(tracking: pd.DataFrame) -> pd.DataFrame:
    plays_w_motion = (tracking
                      [tracking.event == 'man_in_motion']
                      [['gameId', 'playId']]
                      .copy()
                      .sort_values(by=['gameId', 'playId'])
                      .drop_duplicates())

    plays_w_motion.loc[:, 'had_motion'] = True
    plays_w_motion.set_index(['gameId', 'playId'], inplace=True)

    return plays_w_motion


plays_w_motion = get_plays_w_motion(tracking=tracking)
plays_w_motion.head()


def get_starting_qb_change_data(tracking: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ['club', 'gameId', 'playId']

    snap_frames = (tracking
                   [(tracking.frameType == 'SNAP')
                    & (tracking.displayName != 'football')]
                   [['gameId', 'playId', 'club', 'nflId']])

    qbs = players[players.position == 'QB']

    qbs_on_field = (snap_frames
                    .merge(qbs[['nflId', 'displayName']], how='inner', on='nflId')
                    .set_index(idx_cols)
                    .sort_index())

    starting_qb = qbs_on_field.groupby(level=[0, 1]).nflId.first()
    starting_qb.name = 'starter_nflId'
    prev_week_starter = starting_qb.groupby(level=0).shift()
    prev_week_starter.name = 'prev_wk_starter_nflId'

    # starting_qb.head(100)

    starting_qb_change_data = qbs_on_field.join(starting_qb).join(prev_week_starter)
    starting_qb_change_data.loc[:, 'starter_is_in'] = starting_qb_change_data.nflId == starting_qb_change_data.starter_nflId
    starting_qb_change_data.loc[:, 'prev_wk_starter_is_in'] = np.where(
        starting_qb_change_data.prev_wk_starter_nflId.isna(),
        starting_qb_change_data.starter_is_in,
        starting_qb_change_data.nflId == starting_qb_change_data.prev_wk_starter_nflId
    )

    # some plays have multiple QBs (e.g. every play Taysom Hill is in)
    starting_qb_change_data = (starting_qb_change_data
                         .reset_index()
                         .sort_values(by=idx_cols + ['starter_is_in', 'prev_wk_starter_is_in'],
                                      ascending=[True] * len(idx_cols) + [False, False])
                         .groupby(idx_cols)
                         .first())

    return starting_qb_change_data


starting_qb_change_data = get_starting_qb_change_data(tracking=tracking, players=players)
starting_qb_change_data.head()


starting_qb_change_data.starter_is_in.value_counts(normalize=True)


starting_qb_change_data.prev_wk_starter_is_in.value_counts(normalize=True)


def get_down_and_distance_tendencies(plays: pd.DataFrame) -> pd.DataFrame:
    dnd = plays[['gameId', 'playId', 'possessionTeam', 'down', 'yardsToGo', 'isDropback']].copy()
    dnd.loc[:, 'dist_cat'] = np.where(
        dnd.yardsToGo <= 2, 'short',
        np.where(
            dnd.yardsToGo <= 6, 'medium',
            np.where(
                dnd.yardsToGo <= 15, 'long',
                'extra long'
            )
        )
    )
    dnd.dist_cat = dnd.dist_cat.astype('category')
    dnd.sort_values(by=['possessionTeam', 'down', 'dist_cat', 'gameId', 'playId'],
                    inplace=True)

    tendencies = (dnd
                  .groupby(['possessionTeam', 'down', 'dist_cat'], observed=True)
                  .rolling(window=30, min_periods=1, closed='left')
                  .isDropback
                  .mean())
    tendencies.name = 'is_dropback_tendency_100'

    # fillna: no info, 50/50
    tendencies.fillna(0.5, inplace=True)

    return (dnd
            .join(tendencies.reset_index(level=(0, 1, 2), drop=True))
            .set_index(['gameId', 'playId'])
            [['dist_cat', 'is_dropback_tendency_100']])


dnd_tendencies = get_down_and_distance_tendencies(plays=plays)
dnd_tendencies.head()


def get_score_delta_cat(plays: pd.DataFrame,
                        possession_team_versions: pd.DataFrame) -> pd.DataFrame:
    pvc = (possession_team_versions
           [['possession_team_score_delta_cat']]
           .copy())

    id_vals = (plays
               [['gameId', 'playId', 'isDropback', 'possessionTeam']]
               .copy()
               .set_index(['gameId', 'playId'])
               .sort_index())

    pvc = (pvc
           .join(id_vals)
           .reset_index()
           .sort_values(by=['possessionTeam', 'possession_team_score_delta_cat',
                            'gameId', 'playId']))

    tendencies = (pvc
                  .groupby(['possessionTeam', 'possession_team_score_delta_cat'], observed=True)
                  .rolling(window=30, min_periods=1, closed='left')
                  .isDropback
                  .mean())
    tendencies.name = 'score_delta_tendency_100'

    # fillna: no info, 50/50
    tendencies.fillna(0.5, inplace=True)

    return (pvc
            .join(tendencies.reset_index(level=(0, 1), drop=True))
            .set_index(['gameId', 'playId'])
            [['score_delta_tendency_100']])


score_delta_cat = get_score_delta_cat(plays=plays,
                                      possession_team_versions=possession_team_versions)

score_delta_cat.head()


idx_cols = ['gameId', 'playId']
target_cols = idx_cols + [
    # playcall decision
    'isDropback', 'playAction',

    # play results
    'passResult', 'passLength', 'passLocationType', 'rushLocationType',
    'prePenaltyYardsGained',

    # qb movement
    'dropbackType', 'dropbackDistance', 'timeToThrow',

    # pff features
    'pff_runConceptPrimary', 'pff_runConceptSecondary', 'pff_runPassOption',
    'pff_passCoverage', 'pff_manZone'
]

targets = (plays
           [target_cols]
           .copy()
           .set_index(idx_cols)
           .sort_index())

targets.head()


idx_cols = ['gameId', 'playId']
presnap_feature_cols = idx_cols + [
    # game / playcalling context
    'quarter', 'down', 'yardsToGo',

    # field location
    'yardlineNumber', 'absoluteYardlineNumber',

    # formation
    'offenseFormation', 'receiverAlignment',

    # predicted input features
    'preSnapHomeTeamWinProbability', 'expectedPoints',

    # timing
    'gameClock', 'playClockAtSnap',

    # keeping this to compute another feature, then dropping
    'pff_manZone',
]

presnap_features = (plays
                    [presnap_feature_cols]
                    .copy()
                    .set_index(idx_cols)
                    .sort_index())

# objects that we want to treat as categories
presnap_features.receiverAlignment = presnap_features.receiverAlignment.astype('category')

# game clock to seconds left in game (have to handle negatives for overtime)
def get_elapsed_seconds_in_game(presnap_features: pd.DataFrame) -> pd.Series:
    seconds_in_quarter = 15 * 60
    ms = presnap_features.gameClock.str.split(':', expand=True)
    ms.columns = ['minutes', 'seconds']
    ms.minutes = ms.minutes.astype(int)
    ms.seconds = ms.seconds.astype(int)
    ms.loc[:, 'seconds_left_in_quarter'] = ms.minutes * 60 + ms.seconds
    quarters_so_far = (presnap_features.quarter - 1)
    seconds_so_far = quarters_so_far * seconds_in_quarter

    t = seconds_so_far + seconds_in_quarter - ms.seconds_left_in_quarter

    return t

presnap_features.loc[:, 'elapsed_time'] = get_elapsed_seconds_in_game(presnap_features=presnap_features)
presnap_features.loc[:, 'is_under_2_min'] = (presnap_features.elapsed_time.between(29 * 60, 30 * 60)
                                             | presnap_features.elapsed_time.between(59 * 60, 60 * 60))
presnap_features.loc[:, 'is_overtime'] = presnap_features.quarter > 4

# joining in all home / away features to possession team features
presnap_features = presnap_features.join(possession_team_versions)

# joining in other derived features
presnap_features = presnap_features.join(hash_info, how='left').drop(columns='hv_hash')
presnap_features = presnap_features.join(personnel_grouping, how='left')
presnap_features = presnap_features.join(plays_w_motion, how='left')
presnap_features.fillna({'had_motion': False}, inplace=True)
presnap_features.had_motion = presnap_features.had_motion.astype(bool)
presnap_features.loc[:, 'had_motion_d_in_zone'] = presnap_features.had_motion & (presnap_features.pff_manZone == 'Zone')
presnap_features.drop(columns='pff_manZone', inplace=True)
presnap_features = (presnap_features
                    .join(starting_qb_change_data[['starter_is_in', 'prev_wk_starter_is_in']].reset_index(0, drop=True).sort_index(),
                          how='left'))
presnap_features = presnap_features.join(dnd_tendencies, how='left')
presnap_features = presnap_features.join(score_delta_cat, how='left')

# interaction features
presnap_features.loc[:, 'down_x_yardsToGo'] = presnap_features.down * presnap_features.yardsToGo
presnap_features.loc[:, 'yards_per_remaining_down'] = presnap_features.yardsToGo / (5 - presnap_features.down)

# fixing some nans
presnap_features.fillna({'playClockAtSnap': presnap_features.playClockAtSnap.mean(),
                         'starter_is_in': False,
                         'prev_wk_starter_is_in': False},
                        inplace=True)
presnap_features.starter_is_in = presnap_features.starter_is_in.astype(bool)
presnap_features.prev_wk_starter_is_in = presnap_features.prev_wk_starter_is_in.astype(bool)


presnap_features.head()


assert presnap_features.shape[0] == plays.shape[0]


X = presnap_features.join(meta[['should_ignore']], how='left')
X = X[~X.should_ignore]
X.sort_index(inplace=True)


assert X.isna().sum().sum() == 0, 'one of the columns has null values'


y = targets.join(meta[['should_ignore']], how='left')
y = y[~y.should_ignore]
y.sort_index(inplace=True)
y = y.isDropback
y.value_counts(normalize=True)


assert X.shape[0] == y.shape[0]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1337,
                                                    shuffle=True,
                                                    stratify=y)

print(f"{X_train.shape = }")
print(f"{X_test.shape = }")
print(f"{y_train.shape = }")
print(f"{y_test.shape = }")
print(f"\n{y_train.value_counts(normalize=True)}")
print(f"\n{y_test.value_counts(normalize=True)}")


import xgboost as xgb

xgb_model_features = [
    # team categorical? let's at least try it
    # 'possessionTeam',

    # field position
    'down', 'yardsToGo', 'down_x_yardsToGo', 'yards_per_remaining_down',
    'absoluteYardlineNumber', 'ball_hash_cat',

    # game situation
    'quarter', 'possession_team_score_delta',
    'possession_team_score_delta_cat',
    'elapsed_time', 'possession_team_is_home',
    'presnap_possession_team_win_probability',
    'is_under_2_min', 'is_overtime', 'dist_cat',

    # formation information
    'offenseFormation', 'receiverAlignment', 'FB', 'QB', 'pg_cat',
    'had_motion', 'had_motion_d_in_zone',

    # qb injury / replacement
    'starter_is_in', 'prev_wk_starter_is_in',

    # team's tendencies in similar past situations
    'is_dropback_tendency_100', 'score_delta_tendency_100'
]

dm_train = xgb.DMatrix(data=X_train[xgb_model_features],
                       label=y_train,
                       feature_names=xgb_model_features,
                       # feature_types=feature_types,
                       nthread=-1,
                       enable_categorical=True)

dm_test = xgb.DMatrix(data=X_test[xgb_model_features],
                      label=y_test,
                      feature_names=xgb_model_features,
                      # feature_types=feature_types,
                      nthread=-1,
                      enable_categorical=True)


evals_result = {}

b = xgb.train(params={'objective': 'binary:logistic',
                      'eta': 5e-4,
                      'max_depth': 60,
                      'eval_metric': ['error', 'auc', 'logloss']},
              dtrain=dm_train,
              num_boost_round=7_500,
              evals=[(dm_train, 'train'),
                      (dm_test, 'test'),],
              evals_result=evals_result,
              verbose_eval=100,
              early_stopping_rounds=500)


import seaborn as sns

sns.set_theme()


df_eval_test = pd.DataFrame(evals_result['test'])
df_eval_test.loc[:, 'accuracy'] = 1 - df_eval_test.error
df_eval_test.loc[:, 'segment'] = 'test'

df_eval_train = pd.DataFrame(evals_result['train'])
df_eval_train.loc[:, 'accuracy'] = 1 - df_eval_train.error
df_eval_train.loc[:, 'segment'] = 'train'

df_eval = pd.concat([df_eval_test, df_eval_train]).reset_index().rename(columns={'index': 'x'})


ax = df_eval_test.logloss.plot()
ax.set_title('log-loss on test data')


ax = df_eval_test.accuracy.plot()
ax.set_title('accuracy on test data')


max_accuracy = df_eval_test.accuracy.max()
print(f"{max_accuracy = :.2%}")


ax = df_eval_test.auc.plot()
ax.set_title('AUC on test data')


ax = xgb.plot_importance(b)
fig = ax.figure
fig.set_size_inches(10, 10)




