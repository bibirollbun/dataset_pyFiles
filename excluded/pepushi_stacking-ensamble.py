import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression  # for calibration

import lightgbm as lgb
import xgboost as xgb


class EnhancedTournamentPredictor:
    def __init__(self, data_path):
        self.data_path = data_path  # 例: '/kaggle/input/march-machine-learning-mania-2025/**'
        self.data = None
        self.teams = None
        self.seeds = None
        self.games = None
        self.sub = None
        self.gb = None
        self.col = None

        # 前処理オブジェクト
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

        # 各モデルの定義
        self.model_rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=12,
            min_samples_split=3,
            max_features='sqrt'
        )
        self.model_lgb = lgb.LGBMRegressor(
            n_estimators=300,
            random_state=42,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10
        )
        self.model_xgb = xgb.XGBRegressor(
            n_estimators=300,
            random_state=42,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # スタッキングによるアンサンブル（最終メタモデルは RidgeCV）
        estimators = [
            ('rf', self.model_rf),
            ('lgb', self.model_lgb),
            ('xgb', self.model_xgb)
        ]
        self.meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=5,
            n_jobs=-1
        )

    def load_data(self):
        # CSVファイルの読み込み
        files = glob.glob(self.data_path)
        self.data = {
            p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1')
            for p in files
        }

        # チーム情報の処理（男女混合）
        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
        del teams_spelling

        # シーズン・トーナメントの試合結果（コンパクト版・詳細版）
        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])

        # シード、都市、提出ファイルなどの読み込み
        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        gcities = pd.concat([self.data['MGameCities'], self.data['WGameCities']])
        seasons = pd.concat([self.data['MSeasons'], self.data['WSeasons']])

        # seedsを辞書化：キーは "Season_TeamID"
        self.seeds = {
            '_'.join(map(str, [int(row[0]), row[2]])): int(row[1][1:3])
            for row in seeds_df[['Season', 'Seed', 'TeamID']].values
        }

        cities = self.data['Cities']
        self.sub = self.data['SampleSubmissionStage1']
        del seeds_df, cities  # メモリ解放

        # 結果にシーズン (S) またはトーナメント (T) のラベルを付与
        season_cresults['ST'] = 'S'
        season_dresults['ST'] = 'S'
        tourney_cresults['ST'] = 'T'
        tourney_dresults['ST'] = 'T'

        # 詳細結果を統合
        self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        self.games.reset_index(drop=True, inplace=True)
        self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

        # 識別子およびチーム関連の特徴量作成
        self.games['ID'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1
        )
        self.games['IDTeams'] = self.games.apply(
            lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
        )
        self.games['Team1'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1
        )
        self.games['Team2'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1
        )
        self.games['IDTeam1'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
        )
        self.games['IDTeam2'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
        )
        self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
        self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)

        # その他の特徴量
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        self.games['Pred'] = self.games.apply(
            lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1
        )
        self.games['ScoreDiffNorm'] = self.games.apply(
            lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1
        )
        self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        self.games = self.games.fillna(-1)

        # 集約統計量の作成（チームペア毎）
        c_score_col = [
            'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
            'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
        ]
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
        self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]

        # シーズン試合とトーナメント試合を分割（ここではトーナメントのみを学習に使用）
        self.games = self.games[self.games['ST'] == 'T']

        # -------------------------
        # 【追加】シーズン中の各チームの成績算出（チームレベルの特徴量）
        season_games = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        season_games.reset_index(drop=True, inplace=True)
        season_games['win_margin'] = season_games['WScore'] - season_games['LScore']

        # 勝った試合の統計
        w_stats = season_games.groupby(['Season', 'WTeamID']).agg(
            games_played_w=('WTeamID', 'count'),
            win_margin_mean=('win_margin', 'mean'),
            win_margin_std=('win_margin', 'std')
        ).reset_index().rename(columns={'WTeamID': 'TeamID'})

        # 負けた試合の統計
        l_stats = season_games.groupby(['Season', 'LTeamID']).agg(
            games_played_l=('LTeamID', 'count'),
            loss_margin_mean=('win_margin', 'mean'),
            loss_margin_std=('win_margin', 'std')
        ).reset_index().rename(columns={'LTeamID': 'TeamID'})

        team_stats = pd.merge(w_stats, l_stats, how='outer', on=['Season', 'TeamID']).fillna(0)
        team_stats['games_played'] = team_stats['games_played_w'] + team_stats['games_played_l']
        team_stats['wins'] = team_stats['games_played_w']
        team_stats['win_pct'] = team_stats['wins'] / team_stats['games_played']
        team_stats['losses'] = team_stats['games_played_l']
        team_stats['avg_margin'] = (
            team_stats['win_margin_mean'] * team_stats['wins'] + team_stats['loss_margin_mean'] * team_stats['losses']
        ) / team_stats['games_played']

        # -------------------------
        # マージ：学習用のトーナメント試合にシーズン成績を追加（左右のチームで別々に）
        self.games = pd.merge(
            self.games,
            team_stats[['Season', 'TeamID', 'win_pct', 'avg_margin']],
            how='left',
            left_on=['Season', 'Team1'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'win_pct': 'Team1_win_pct', 'avg_margin': 'Team1_avg_margin'}).drop(columns=['TeamID'])
        self.games = pd.merge(
            self.games,
            team_stats[['Season', 'TeamID', 'win_pct', 'avg_margin']],
            how='left',
            left_on=['Season', 'Team2'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'win_pct': 'Team2_win_pct', 'avg_margin': 'Team2_avg_margin'}).drop(columns=['TeamID'])
        self.games['Team1_win_pct'] = self.games['Team1_win_pct'].fillna(0)
        self.games['Team2_win_pct'] = self.games['Team2_win_pct'].fillna(0)
        self.games['Team1_avg_margin'] = self.games['Team1_avg_margin'].fillna(0)
        self.games['Team2_avg_margin'] = self.games['Team2_avg_margin'].fillna(0)
        self.games['win_pct_diff'] = self.games['Team1_win_pct'] - self.games['Team2_win_pct']
        self.games['avg_margin_diff'] = self.games['Team1_avg_margin'] - self.games['Team2_avg_margin']

        # -------------------------
        # 提出用データの処理
        self.sub['WLoc'] = 3
        self.sub['Season'] = self.sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
        self.sub['Team1'] = self.sub['ID'].map(lambda x: x.split('_')[1])
        self.sub['Team2'] = self.sub['ID'].map(lambda x: x.split('_')[2])
        # 型の不一致解消のため、Team1, Team2 を int 型に変換
        self.sub['Team1'] = self.sub['Team1'].astype(int)
        self.sub['Team2'] = self.sub['Team2'].astype(int)
        self.sub['IDTeams'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
        self.sub['IDTeam1'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.sub['IDTeam2'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        self.sub['Team1Seed'] = self.sub['IDTeam1'].map(self.seeds).fillna(0)
        self.sub['Team2Seed'] = self.sub['IDTeam2'].map(self.seeds).fillna(0)
        self.sub['SeedDiff'] = self.sub['Team1Seed'] - self.sub['Team2Seed']
        self.sub = self.sub.fillna(-1)

        # ★ 提出用データにもシーズン成績をマージ ★
        self.sub = pd.merge(
            self.sub,
            team_stats[['Season', 'TeamID', 'win_pct', 'avg_margin']],
            how='left',
            left_on=['Season', 'Team1'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'win_pct': 'Team1_win_pct', 'avg_margin': 'Team1_avg_margin'}).drop(columns=['TeamID'])
        self.sub = pd.merge(
            self.sub,
            team_stats[['Season', 'TeamID', 'win_pct', 'avg_margin']],
            how='left',
            left_on=['Season', 'Team2'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'win_pct': 'Team2_win_pct', 'avg_margin': 'Team2_avg_margin'}).drop(columns=['TeamID'])
        self.sub['Team1_win_pct'] = self.sub['Team1_win_pct'].fillna(0)
        self.sub['Team2_win_pct'] = self.sub['Team2_win_pct'].fillna(0)
        self.sub['Team1_avg_margin'] = self.sub['Team1_avg_margin'].fillna(0)
        self.sub['Team2_avg_margin'] = self.sub['Team2_avg_margin'].fillna(0)
        self.sub['win_pct_diff'] = self.sub['Team1_win_pct'] - self.sub['Team2_win_pct']
        self.sub['avg_margin_diff'] = self.sub['Team1_avg_margin'] - self.sub['Team2_avg_margin']

        # -------------------------
        # 集約統計量（gb）のマージ
        self.games = pd.merge(self.games, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        self.sub = pd.merge(self.sub, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

        # -------------------------
        # 使用する特徴量リストの作成
        # ※ win_pct_diff, avg_margin_diff などは学習に使用するため除外リストには含めない
        exclude_cols = [
            'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
            'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff',
            'ScoreDiffNorm', 'WLoc'
        ] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def train_model(self):
        # 学習用特徴量とターゲットの準備
        X = self.games[self.col].fillna(-1)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y = self.games['Pred']

        # スタッキングモデルの学習
        self.model.fit(X_scaled, y)

        # 学習セット上での予測＆キャリブレーション
        pred = self.model.predict(X_scaled).clip(0.001, 0.999)
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pred, y)
        pred_cal = ir.transform(pred)

        # 評価指標の表示
        print(f'Log Loss: {log_loss(y, pred_cal):.4f}')
        print(f'Mean Absolute Error: {mean_absolute_error(y, pred_cal):.4f}')
        print(f'Brier Score: {brier_score_loss(y, pred_cal):.4f}')
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        print(f'Cross-validated MSE: {-cv_scores.mean():.4f}')

    def predict_submission(self, output_file='submission.csv'):
        # 提出用特徴量の準備と予測
        sub_X = self.sub[self.col].fillna(-1)
        sub_X_imputed = self.imputer.transform(sub_X)
        sub_X_scaled = self.scaler.transform(sub_X_imputed)
        preds = self.model.predict(sub_X_scaled).clip(0.01, 0.99)

        # 学習時と同様のキャリブレーションを再適用
        ir = IsotonicRegression(out_of_bounds='clip')
        X_train = self.imputer.fit_transform(self.games[self.col].fillna(-1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        train_preds = self.model.predict(X_train_scaled).clip(0.001, 0.999)
        ir.fit(train_preds, self.games['Pred'])
        preds_cal = ir.transform(preds)

        self.sub['Pred'] = preds_cal
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        print(f"Submission file saved to {output_file}")

    def run_all(self):
        self.load_data()
        self.train_model()
        self.predict_submission()


if __name__ == "__main__":
    data_path = '/kaggle/input/march-machine-learning-mania-2025/**'
    predictor = EnhancedTournamentPredictor(data_path)
    predictor.run_all()




