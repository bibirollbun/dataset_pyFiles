from IPython.display import display
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import chardet
import os


# Settings
sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")


COMP_DIR = '/kaggle/input/march-machine-learning-mania-2025'


# Tools
def get_csv_filepath(filename: str) -> str:
    return os.path.join(COMP_DIR, filename)

def load_csv_file(csv_file_path: str) -> pd.DataFrame:
    def _detect_file_encoding(file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            raw_data: bytes = file.read()
            encoding_result: Dict[str, Any] = chardet.detect(raw_data)
        return encoding_result

    def _load_csv_to_dataframe(file_path: str, encoding: str) -> pd.DataFrame:
        return pd.read_csv(file_path, encoding=encoding)

    encoding_result: Dict[str, Any] = _detect_file_encoding(csv_file_path)
    dataframe: pd.DataFrame = _load_csv_to_dataframe(csv_file_path, encoding_result["encoding"])
    
    return dataframe

def investigate_teams(genders: List[str], dataframes: List[pd.DataFrame]) -> None:
    def display_team_statistics(gender: str, df: pd.DataFrame) -> None:
        print(f"Investigating {gender} Teams")
        display(df.describe())
        display(df.head())

    for gender, df in zip(genders, dataframes):
        display_team_statistics(gender, df)


class TeamDurationsPlotter:
    def __init__(self, gender: str, df: pd.DataFrame):
        self.__gender = gender
        self.__df = df

    def __has_valid_season_columns(self) -> bool:
        required_columns = ['FirstD1Season', 'LastD1Season']
        return all(col in self.__df.columns for col in required_columns)
    
    def __calculate_participation_duration(self) -> pd.Series:
        return self.__df['LastD1Season'] - self.__df['FirstD1Season']
    
    def __plot_participation_durations(self) -> None:
        self.__prepare_plot()
        self.__create_figure()
        self.__plot_bars()
        self.__adjust_bar_positions()
        self.__set_axes_limits()
        self.__finalize_plot()
        self.__show_plot()        

    def __prepare_plot(self) -> None:
        self.__df['Duration'] = self.__df['LastD1Season'] - self.__df['FirstD1Season']
    
    def __create_figure(self) -> plt.Figure:
        plt.figure(figsize=(15, 60))
    
    def __plot_bars(self) -> None:
        sns.barplot(
            data=self.__df,
            y="TeamName",
            x="Duration",
            orient="h",
            color="blue",
            edgecolor="black"
        )

    def __adjust_bar_positions(self) -> None:
        for i, (start, duration) in enumerate(zip(self.__df["FirstD1Season"], self.__df["Duration"])):
            plt.gca().patches[i].set_x(start)
    
    def __set_axes_limits(self) -> None:
        plt.xlim(self.__df["FirstD1Season"].min(), self.__df["LastD1Season"].max())
    
    def __finalize_plot(self) -> None:
        plt.title(f'Durations of {self.__gender} NCAA Teams')
        plt.tight_layout()
    
    def __show_plot(self) -> None:
        plt.show()

    def __call__(self) -> None:
        if not self.__has_valid_season_columns():
            print(f"Warning: The dataset for {self.__gender} does not have 'FirstD1Season' and 'LastD1Season' columns.")
            return
        self.__df['Duration'] = self.__calculate_participation_duration()
        self.__plot_participation_durations()


class TournamentResultsAnalyzer:
    def __init__(self, mens_df: pd.DataFrame, womens_df: pd.DataFrame):
        self.__mens_df = mens_df
        self.__womens_df = womens_df

    def __call__(self) -> None:
        for gender, df in zip(['Mens', 'Womens'], [self.__mens_df, self.__womens_df]):
            print(f"Investigating {gender} Team")

            df['Score Difference'] = df['WScore'] - df['LScore']

            self.__plot_score_distribution(df, gender, 'WScore', 'Winning Score')
            self.__plot_score_distribution(df, gender, 'LScore', 'Losing Score')
            self.__plot_score_distribution(df, gender, 'Score Difference', 'Score Difference')

    def __plot_score_distribution(self, df: pd.DataFrame, gender: str, score_column: str, score_type: str) -> None:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Season', y=score_column, data=df)
        plt.xticks(rotation=90)
        plt.title(f'{gender} {score_type} Distributions')
        plt.tight_layout()
        plt.show()



class DetailedTeamStatisticsDifferenceDistributionsPlotter:
    def __init__(self, men_detailed_results: pd.DataFrame, womens_detailed_results: pd.DataFrame):
        self.__men_detailed_results = men_detailed_results
        self.__womens_detailed_results = womens_detailed_results
        self.__statistic_suffix_mapping  = {
            'FGM': 'Field goals made', 
            'FGA': 'Field goals attempted', 
            'FGM3': 'Three pointers made', 
            'FGA3': 'Three pointers attempted', 
            'FTM': 'Free throws made', 
            'FTA': 'Free throws attempted', 
            'OR': 'Offensive rebounds', 
            'DR': 'Defensive rebounds', 
            'Ast': 'Assists', 
            'TO': 'Turnovers committed', 
            'Stl': 'Steals', 
            'Blk': 'Blocks', 
            'PF': 'Personal fouls committed',
        }

    def __calculate_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        for key in self.__statistic_suffix_mapping .keys():
            df[f'D{key}'] = df[f'W{key}'] - df[f'L{key}']
        return df

    def __plot_difference_distributions(self, df: pd.DataFrame, gender: str) -> None:
        for key, value in self.__statistic_suffix_mapping .items():
            plt.figure(figsize=(10, 6))
            sns.histplot(df[f'D{key}'], bins=15, kde=True)
            plt.xticks(rotation=90)
            plt.title(f'{gender} {value} Difference')
            plt.tight_layout()
            plt.show()

    def __call__(self):
        for gender, df in zip(['Mens', 'Womens'], [self.__men_detailed_results, self.__womens_detailed_results]):
            print(f"Investigating {gender} Teams")
            df_with_differences = self.__calculate_differences(df)
            self.__plot_difference_distributions(df_with_differences, gender)


men_teams: pd.DataFrame = load_csv_file(get_csv_filepath('MTeams.csv'))
women_teams: pd.DataFrame = load_csv_file(get_csv_filepath('WTeams.csv'))

investigate_teams(['Men', 'Women'], [men_teams, women_teams])


men_team_names: pd.Series = men_teams['TeamName']
women_team_names: pd.Series = women_teams['TeamName']

# Find unique values in each series
unique_in_men: pd.Series = men_team_names[~men_team_names.isin(women_team_names)]
unique_in_women: pd.Series = women_team_names[~women_team_names.isin(men_team_names)]

# Combine results
unique_team_names: pd.Series = pd.concat([unique_in_men, unique_in_women])

print("Unique Team Names:")
print(unique_team_names)


action = TeamDurationsPlotter(gender="Mens", df=men_teams)
action()


men_seasons: pd.DataFrame = load_csv_file(get_csv_filepath('MSeasons.csv'))
women_seasons: pd.DataFrame = load_csv_file(get_csv_filepath('WSeasons.csv'))

investigate_teams(['Men', 'Women'], [men_seasons, women_seasons])


men_seeds: pd.DataFrame = load_csv_file(get_csv_filepath('MNCAATourneySeeds.csv'))
women_seeds: pd.DataFrame = load_csv_file(get_csv_filepath('WNCAATourneySeeds.csv'))

investigate_teams(['Men', 'Women'], [men_seeds, women_seeds])


men_regular_season_compact_results: pd.DataFrame = load_csv_file(get_csv_filepath('MRegularSeasonCompactResults.csv'))
women_regular_season_compact_results: pd.DataFrame = load_csv_file(get_csv_filepath('WRegularSeasonCompactResults.csv'))

investigate_teams(['Men', 'Women'], [men_regular_season_compact_results, women_regular_season_compact_results])


results_analyzer = TournamentResultsAnalyzer(men_regular_season_compact_results, women_regular_season_compact_results)
results_analyzer()


men_ncca_tourney_compact_results: pd.DataFrame = load_csv_file(get_csv_filepath('MNCAATourneyCompactResults.csv'))
women_ncca_tourney_compact_results: pd.DataFrame = load_csv_file(get_csv_filepath('WNCAATourneyCompactResults.csv'))

investigate_teams(['Men', 'Women'], [men_ncca_tourney_compact_results, women_ncca_tourney_compact_results])


results_analyzer = TournamentResultsAnalyzer(men_ncca_tourney_compact_results, women_ncca_tourney_compact_results)
results_analyzer()


stage_one_submission_sample: pd.DataFrame = load_csv_file(get_csv_filepath('SampleSubmissionStage1.csv'))

# Split the 'ID' column into separate columns for Season, Lower TeamID, and Higher TeamID
stage_one_submission_sample[['Season', 'Lower TeamID', 'Higher TeamID']] = stage_one_submission_sample['ID'].str.split('_', expand=True)

# Convert 'Season', 'Lower TeamID', and 'Higher TeamID' to integers for consistency
stage_one_submission_sample['Season'] = stage_one_submission_sample['Season'].astype(int)
stage_one_submission_sample['Lower TeamID'] = stage_one_submission_sample['Lower TeamID'].astype(int)
stage_one_submission_sample['Higher TeamID'] = stage_one_submission_sample['Higher TeamID'].astype(int)

stage_one_submission_sample = stage_one_submission_sample[['Season', 'Lower TeamID', 'Higher TeamID', 'Pred']]
stage_one_submission_sample.head()


men_regular_season_detailed_results: pd.DataFrame = load_csv_file(get_csv_filepath('MRegularSeasonDetailedResults.csv'))
women_regular_season_detailed_results: pd.DataFrame = load_csv_file(get_csv_filepath('WRegularSeasonDetailedResults.csv'))

investigate_teams(['Men', 'Women'], [men_regular_season_detailed_results, women_regular_season_detailed_results])


plotter = DetailedTeamStatisticsDifferenceDistributionsPlotter(men_regular_season_detailed_results, women_regular_season_detailed_results)
plotter()


men_ncca_tourney_detailed_results: pd.DataFrame = load_csv_file(get_csv_filepath('MNCAATourneyDetailedResults.csv'))
women_ncca_tourney_detailed_results: pd.DataFrame = load_csv_file(get_csv_filepath('WNCAATourneyDetailedResults.csv'))

investigate_teams(['Men', 'Women'], [men_ncca_tourney_detailed_results, women_ncca_tourney_detailed_results])

