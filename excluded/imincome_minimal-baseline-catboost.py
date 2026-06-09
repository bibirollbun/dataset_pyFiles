import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from scipy.stats import skew, mode


PATH_DATASET = '/kaggle/input/march-machine-learning-mania-2025/'


m_teams_df = pd.read_csv(PATH_DATASET + 'MTeams.csv')
w_teams_df = pd.read_csv(PATH_DATASET + 'WTeams.csv')

m_seasons_df = pd.read_csv(PATH_DATASET + 'MSeasons.csv')
w_seasons_df = pd.read_csv(PATH_DATASET + 'WSeasons.csv')

m_caa_seeds_df = pd.read_csv(PATH_DATASET + 'MNCAATourneySeeds.csv')
w_caa_seeds_df = pd.read_csv(PATH_DATASET + 'WNCAATourneySeeds.csv')

m_seasons_results_df = pd.read_csv(PATH_DATASET + 'MRegularSeasonCompactResults.csv')
w_seasons_results_df = pd.read_csv(PATH_DATASET + 'WRegularSeasonCompactResults.csv')

m_tourney_results_df = pd.read_csv(PATH_DATASET + 'MNCAATourneyCompactResults.csv')
w_tourney_results_df = pd.read_csv(PATH_DATASET + 'WNCAATourneyCompactResults.csv')

sample_submission_stage1_df = pd.read_csv(PATH_DATASET + 'SampleSubmissionStage1.csv')

(m_teams_df.shape, w_teams_df.shape,
 m_seasons_df.shape, w_seasons_df.shape,
 m_caa_seeds_df.shape, w_caa_seeds_df.shape,
 m_seasons_results_df.shape, w_seasons_results_df.shape,
 m_tourney_results_df.shape, w_tourney_results_df.shape,
 sample_submission_stage1_df.shape, 
)


# Create base features
def calc_agg_results(df:pd.DataFrame, type_res:str):
    res_prefix = 'W' if type_res == 'win' else 'L'
    agg_df = df.groupby(by=f'{res_prefix}TeamID').agg(
        min_season=('Season', np.min),
        max_season=('Season', np.max),
        uniq_season =('Season', pd.Series.nunique),
        std_season=('Season', np.std),
        count=('Season', len),
        sum_score=(f'{res_prefix}Score', sum),
        std_score=(f'{res_prefix}Score', np.std),
        min_score=(f'{res_prefix}Score', min),
        max_score=(f'{res_prefix}Score', max),
        freq_loc =(f'{res_prefix}Score', mode),
        skew_loc =(f'{res_prefix}Score', skew),
        sum_num_ot=('NumOT', sum),
        uniq_loc =('WLoc', pd.Series.nunique),
        most_common_loc=('WLoc', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        least_common_loc=('WLoc', lambda x: x.value_counts().idxmin() if not x.empty else None),
    ).reset_index()
    
    agg_df['count_season'] = agg_df['max_season'] - agg_df['min_season'] + 1
    agg_df['norm_count'] = agg_df['count'] / agg_df['count_season']
    agg_df['norm_score'] = agg_df['sum_score'] / agg_df['count_season']
    agg_df['norm_num_ot'] = agg_df['sum_num_ot'] / agg_df['count_season']
    agg_df[['mode_freq_loc', 'count_freq_loc']] = agg_df['freq_loc'].apply(lambda x: pd.Series([x.mode, x.count]))
    agg_df = agg_df.drop(columns=['freq_loc'])
    agg_df = agg_df.set_index(f'{res_prefix}TeamID').add_suffix(f'_{type_res}').reset_index()
    return agg_df

m_win_seasons_results_df = calc_agg_results(df=m_seasons_results_df, type_res='win')
m_lose_seasons_results_df = calc_agg_results(df=m_seasons_results_df, type_res='lose')

m_win_tourney_results_df = calc_agg_results(df=m_tourney_results_df, type_res='win')
m_lose_tourney_results_df = calc_agg_results(df=m_tourney_results_df, type_res='lose')

m_ext_season_df = m_caa_seeds_df.merge(m_seasons_df, on='Season').merge(m_teams_df, on='TeamID')
m_ext_season_df = m_ext_season_df.merge(m_win_seasons_results_df, left_on='TeamID', right_on='WTeamID', suffixes=(None, '_win_seasons')).drop(columns=['WTeamID']).merge(m_lose_seasons_results_df, left_on='TeamID', right_on='LTeamID', suffixes=(None, '_lose_seasons')).drop(columns=['LTeamID'])
m_ext_season_df = m_ext_season_df.merge(m_win_tourney_results_df, left_on='TeamID', right_on='WTeamID', suffixes=(None, '_win_tourney')).drop(columns=['WTeamID']).merge(m_lose_tourney_results_df, left_on='TeamID', right_on='LTeamID', suffixes=(None, '_lose_tourney')).drop(columns=['LTeamID'])


w_win_seasons_results_df = calc_agg_results(df=w_seasons_results_df, type_res='win')
w_lose_seasons_results_df = calc_agg_results(df=w_seasons_results_df, type_res='lose')

w_win_tourney_results_df = calc_agg_results(df=w_tourney_results_df, type_res='win')
w_lose_tourney_results_df = calc_agg_results(df=w_tourney_results_df, type_res='lose')

w_ext_season_df = w_caa_seeds_df.merge(w_seasons_df, on='Season').merge(w_teams_df, on='TeamID')
w_ext_season_df = w_ext_season_df.merge(w_win_seasons_results_df, left_on='TeamID', right_on='WTeamID', suffixes=(None, '_win_seasons')).drop(columns=['WTeamID']).merge(w_lose_seasons_results_df, left_on='TeamID', right_on='LTeamID', suffixes=(None, '_lose_seasons')).drop(columns=['LTeamID'])
w_ext_season_df = w_ext_season_df.merge(w_win_tourney_results_df, left_on='TeamID', right_on='WTeamID', suffixes=(None, '_win_tourney')).drop(columns=['WTeamID']).merge(w_lose_tourney_results_df, left_on='TeamID', right_on='LTeamID', suffixes=(None, '_lose_tourney')).drop(columns=['LTeamID'])

m_ext_season_df.shape, w_ext_season_df.shape


# Create target df
def create_target_df(df):
    win_result = df[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID_1', 'LTeamID': 'TeamID_2'})
    loss_result = df[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID_2', 'LTeamID': 'TeamID_1'})
    win_result['target'] = 1
    loss_result['target'] = 0
    return pd.concat((win_result, loss_result))

m_train_result_df = create_target_df(m_seasons_results_df)
w_train_result_df = create_target_df(w_seasons_results_df)

m_train_result_df.shape, w_train_result_df.shape


# Merge target with base features
m_train_result_df = m_train_result_df.merge(m_ext_season_df.add_suffix('__1'), left_on=['Season', 'TeamID_1'], right_on=['Season__1', 'TeamID__1'], how='left').merge(m_ext_season_df.add_suffix('__2'), left_on=['Season', 'TeamID_2'], right_on=['Season__2', 'TeamID__2'], how='left')
w_train_result_df = w_train_result_df.merge(w_ext_season_df.add_suffix('__1'), left_on=['Season', 'TeamID_1'], right_on=['Season__1', 'TeamID__1'], how='left').merge(w_ext_season_df.add_suffix('__2'), left_on=['Season', 'TeamID_2'], right_on=['Season__2', 'TeamID__2'], how='left')
m_train_result_df.shape, w_train_result_df.shape


def fit_model(train_result_df):
    target_column = 'target'
    numerical_features = list(train_result_df.dtypes[train_result_df.dtypes == np.dtype('int64')].index)
    numerical_features = numerical_features + list(train_result_df.dtypes[train_result_df.dtypes == np.dtype('float64')].index)
    categorical_features = list(train_result_df.dtypes[train_result_df.dtypes == np.dtype('object')].index)
    for rm_col in ['Season__1', 'TeamID__1', 'Season__2','TeamID__2', 'Season', 'TeamID_1', 'TeamID_2', 'target']:
        numerical_features.remove(rm_col)
    
    feature_columns = numerical_features + categorical_features
    train_result_df[numerical_features] = train_result_df[numerical_features].fillna(0)
    train_result_df[categorical_features] = train_result_df[categorical_features].fillna('null')

    # Train Test Split
    X_train, X_val, y_train, y_val = train_test_split(train_result_df[feature_columns], train_result_df[target_column], test_size = 0.05, random_state = 53)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 53)

    model_clf = CatBoostClassifier(
                    eval_metric="AUC", 
                    early_stopping_rounds=200, 
                    iterations=2000,
                    random_state=53, 
                    cat_features=categorical_features, 
                    learning_rate=0.01,
                    task_type='GPU'
    )
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    model_clf.fit(X_train, y_train, eval_set=val_pool, plot=True, verbose=False)
    
    return model_clf, feature_columns, numerical_features, categorical_features
m_model_clf, m_feature_columns, m_numerical_features, m_categorical_features = fit_model(m_train_result_df)
w_model_clf, w_feature_columns, w_numerical_features, w_categorical_features = fit_model(w_train_result_df)


%%time
# Create features submission
sample_submission_stage1_df[['Season', 'TeamID_1', 'TeamID_2']]  = sample_submission_stage1_df['ID'].apply(lambda x: pd.Series(map(int, x.split('_'))))
sample_submission_stage1_df.shape


# Split submission W & M
m_sample_submission_stage1_df = sample_submission_stage1_df[sample_submission_stage1_df['TeamID_1'] < 3000]
w_sample_submission_stage1_df = sample_submission_stage1_df[sample_submission_stage1_df['TeamID_1'] >= 3000]

m_sample_submission_stage1_df.shape, w_sample_submission_stage1_df.shape


# Merge submission with base features
m_submission_result_df = m_sample_submission_stage1_df.merge(m_ext_season_df.add_suffix('__1'), left_on=['Season', 'TeamID_1'], right_on=['Season__1', 'TeamID__1'], how='left').merge(m_ext_season_df.add_suffix('__2'), left_on=['Season', 'TeamID_2'], right_on=['Season__2', 'TeamID__2'], how='left')
w_submission_result_df = w_sample_submission_stage1_df.merge(w_ext_season_df.add_suffix('__1'), left_on=['Season', 'TeamID_1'], right_on=['Season__1', 'TeamID__1'], how='left').merge(w_ext_season_df.add_suffix('__2'), left_on=['Season', 'TeamID_2'], right_on=['Season__2', 'TeamID__2'], how='left')

m_submission_result_df[m_numerical_features] = m_submission_result_df[m_numerical_features].fillna(0)
m_submission_result_df[m_categorical_features] = m_submission_result_df[m_categorical_features].fillna('null')

w_submission_result_df[w_numerical_features] = w_submission_result_df[w_numerical_features].fillna(0)
w_submission_result_df[w_categorical_features] = w_submission_result_df[w_categorical_features].fillna('null')

m_submission_result_df.shape, w_submission_result_df.shape


# Submission predict
m_submission_result_df['Pred'] = m_model_clf.predict_proba(m_submission_result_df[m_feature_columns])[:,1]
w_submission_result_df['Pred'] = w_model_clf.predict_proba(w_submission_result_df[w_feature_columns])[:,1]
all_submission_result_df = pd.concat([m_submission_result_df[['ID', 'Pred']], w_submission_result_df[['ID', 'Pred']]])
all_submission_result_df.shape


all_submission_result_df.to_csv('/kaggle/working/submission.csv', index=False)




