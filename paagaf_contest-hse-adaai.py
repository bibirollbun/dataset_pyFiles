!pip install optuna catboost


import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error


train_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv'
test_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv'
sample_submission_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_submission_path)


train.head()


def augment_inversion(df):
    df_copy = df.copy()
    df_copy['agent1'], df_copy['agent2'] = df['agent2'], df['agent1']
    df_copy['AdvantageP2'] = 1 - df['AdvantageP1']
    return df_copy

def extract_agent_details(df):
    df = df.copy()
    agent1_details = df['agent1'].str.split('-', expand=True, n=4)
    agent1_details.columns = ['agent1_dropcol', 'agent1_selection', 'agent1_expconst', 'agent1_playout', 'agent1_scorebounds']
    df = pd.concat([df, agent1_details], axis=1).drop(columns=['agent1_dropcol'])
    agent2_details = df['agent2'].str.split('-', expand=True, n=4)
    agent2_details.columns = ['agent2_dropcol', 'agent2_selection', 'agent2_expconst', 'agent2_playout', 'agent2_scorebounds']
    df = pd.concat([df, agent2_details], axis=1).drop(columns=['agent2_dropcol'])

    df = df.drop(columns=['agent1', 'agent2'])
    df["adv_diff_adj"] = ((df["AdvantageP1"] - df["AdvantageP2"]) * df["Completion"])
    return df


train= augment_inversion(train)
train.head()


train= extract_agent_details(train)
train.head()


X_train = train.drop(columns=['utility_agent1','num_wins_agent1','num_draws_agent1','num_losses_agent1','Id'])
y_train = train['utility_agent1']

categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Обучаем модель CatBoostRegressor для получения важности признаков
model = CatBoostRegressor(iterations=300, verbose=False, random_seed=42)
model.fit(X_train, y_train, cat_features=categorical_features)

# Получаем важность признаков
importances = model.get_feature_importance()
feature_names = X_train.columns

# Формируем DataFrame с важностями и сортируем по убыванию
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values(by='importance', ascending=False)

top_50_features = fi_df.head(50)['feature'].tolist()


top_50_features


def objective(trial, X, y, used_cols, cat_features):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('depth', 2, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'iterations': trial.suggest_int('iterations', 100, 3000),
        'loss_function': 'RMSE',
        'verbose': 0,
        'random_seed': 42,
    }
    # Разбиение на обучающую и валидационную выборки (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X[used_cols], y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        cat_features=cat_features
    )
    preds = model.predict(X_val)
    score = mean_squared_error(y_val, preds, squared=False)
    return score


#study = optuna.create_study(direction='minimize')
#study.optimize(lambda trial: objective(trial, X, y, used_cols, cat_features), n_trials=n_trials)
#best_params = study.best_trial.params


used_cols = [
   'LudRules',
 'agent2_selection',
 'agent2_expconst',
 'agent1_selection',
 'Drawishness',
 'agent2_playout',
 'agent1_expconst',
 'agent1_playout',
 'EnglishRules',
 'AdvantageP2',
 'OutcomeUniformity',
 'PlayoutsPerSecond',
 'GameTreeComplexity',
 'DecisionFactorMedian',
 'NumVertices',
 'DurationActions',
 'AsymmetricPiecesType',
 'MancalaStyle',
 'Completion',
 'NumComponentsType',
 'DecisionMoves',
 'AsymmetricForces',
 'GameRulesetName',
 'Balance',
 'BranchingFactorMaximum',
 'HopCaptureFrequency',
 'AdvantageP1',
 'NumOuterSites',
 'BoardSitesOccupiedMaxIncrease',
 'DurationMoves',
 'BranchingFactorAverage',
 'NumPlayableSites',
 'BoardSitesOccupiedChangeSign',
 'HopCaptureMoreThanOneFrequency',
 'BoardSitesOccupiedMaxDecrease',
 'StateTreeComplexity',
 'adv_diff_adj',
 'DurationTurnsNotTimeouts',
 'MovesPerSecond',
 'HopDecisionEnemyToEmptyFrequency',
 'BranchingFactorMedian',
 'NumPlayableSitesOnBoard',
 'PieceNumberVariance',
 'BoardSitesOccupiedChangeLineBestFit',
 'PieceNumberAverage',
 'DurationTurns',
 'StarBoard',
 'AddDecisionFrequency',
 'DecisionFactorMaxIncrease',
 'NumStartComponentsHand'
]

# Параметры модели CatBoost (получены optuna)
cb_params = {
    'learning_rate': 0.06855001104481778,
    'depth': 5,
    'l2_leaf_reg': 1.2172993508994574,
    'iterations': 686,
    'verbose': 0
}

cb_model = CatBoostRegressor(**cb_params)



categorical_features = train[used_cols].select_dtypes(include=['object']).columns.tolist()
def train_model(train_df: pl.DataFrame) -> CatBoostRegressor:
    X_train = train_df[used_cols]
    y_train = train_df['utility_agent1']
    cb_model.fit(X_train, y_train,cat_features=categorical_features)
    return cb_model


model = train_model(train)


test.head()


def predict(test, sample_sub):
    if not isinstance(test, pd.DataFrame):
        test = test.to_pandas()
    if not isinstance(sample_sub, pd.DataFrame):
        sample_sub = sample_sub.to_pandas()

    test = augment_inversion(test)
    test = extract_agent_details(test)
    
    pred = model.predict(test[used_cols])
    
    final_prediction = np.clip(pred, -1, 1)
    
    sample_sub['utility_agent1'] = final_prediction
    return sample_sub


import kaggle_evaluation.mcts_inference_server
import os
inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )

