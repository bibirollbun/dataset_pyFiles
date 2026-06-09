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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data():
    """
    Loads, prepares, and merges data from various CSV files, adding advanced football-specific features
    and including height and weight of the running back as well as positional group averages.

    Returns:
        tuple: X (features), y (target).
    """
    try:
        print("Loading datasets...")

        # Load player-play data
        player_play_data = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/player_play.csv", usecols=[
            "routeRan", "nflId", "playId", "gameId", "inMotionAtBallSnap"
        ]).dropna(subset=["routeRan"])

        # Load play data
        play_data = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/plays.csv", usecols=[
            "quarter", "down", "yardsToGo", "possessionTeam", "defensiveTeam", "yardlineSide", "yardlineNumber",
            "gameClock", "preSnapHomeScore", "preSnapVisitorScore", "absoluteYardlineNumber",
            "preSnapHomeTeamWinProbability", "preSnapVisitorTeamWinProbability", "expectedPoints",
            "offenseFormation", "receiverAlignment", "gameId", "playId"
        ])
        
        # Convert game clock to seconds
        def game_clock_to_seconds(clock):
            try:
                minutes, seconds = map(int, clock.split(":"))
                return minutes * 60 + seconds
            except Exception:
                return np.nan

        # Filter for plays with less than two minutes left in quarter
        play_data["seconds_left_in_quarter"] = play_data["gameClock"].apply(game_clock_to_seconds)
        play_data = play_data[play_data["seconds_left_in_quarter"] <= 120]

        # Load player data
        player_data = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv", usecols=[
            "height", "weight", "collegeName", "nflId", "position"
        ])

        # Load game data
        game_data = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/games.csv", usecols=[
            "gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"
        ])
        # Load and filter tracking data
        tracking_data_combined = pd.concat(
            [
                pd.read_csv(file, usecols=[
                    "gameId", "playId", "nflId", "playDirection", "x", "y", "frameType", "event"
                ]).query("frameType == 'BEFORE_SNAP'")
                for file in [f"/kaggle/input/nfl-big-data-bowl-2025/tracking_week_{i}.csv" for i in range(1, 10)]
            ],
            ignore_index=True
        )

        print("Merging datasets...")

        # Merge datasets
        player_play_merged = pd.merge(player_play_data, player_data, on="nflId")
        play_game_merged = pd.merge(play_data, game_data, on="gameId")
        final_data = pd.merge(player_play_merged, play_game_merged, on=["playId", "gameId"])
        final_data = pd.merge(final_data, tracking_data_combined, on=["gameId", "playId", "nflId"])
        
        # Drop unnecessary columns
        final_data = final_data.drop(columns=["gameId", "frameType", "gameClock"])

        # Create features and target
        x = final_data.drop(columns=["routeRan"])
        y = final_data["routeRan"]

        # Validate shapes
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        if len(x) != len(y):
            raise ValueError("Mismatch in number of rows between features and target.")

        return x, y

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def split_and_save_data(x, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets based on plays (playId) and saves them to CSV files.
    """
    try:
        print("Splitting data into training and test sets based on playId...")

        # Use GroupShuffleSplit to split data by playId
        plays = x['playId']  # Ensure playId is present in x
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y, plays))

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Validate shapes
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
        if len(x_train) != len(y_train) or len(x_test) != len(y_test):
            raise ValueError("Mismatch in lengths after splitting.")

        # Save the split datasets
        logging.info("Saving training and testing datasets...")
        x_train.to_csv(os.path.join("/kaggle/working/x_train.csv"), index=False)
        x_test.to_csv(os.path.join("/kaggle/working/x_test.csv"), index=False)
        y_train.to_csv(os.path.join("/kaggle/working/y_train.csv"), index=False)
        y_test.to_csv(os.path.join("/kaggle/working/y_test.csv"), index=False)

        print("Datasets saved successfully.")
        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"An error occurred during data splitting: {e}")
        raise


def main():
    """Main execution function."""
    try:
        x, y = load_and_prepare_data()
        x_train, x_test, y_train, y_test = split_and_save_data(x, y)
    
        logging.info("Program execution completed successfully.")
    except Exception as e:
        logging.error(f"Program terminated with an exception: {e}")


if __name__ == "__main__":
    main()


import pandas as pd
from xgboost import XGBClassifier  # Import XGBoost Classifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier  # Import XGBClassifier


from sklearn.preprocessing import LabelEncoder

def load_data():
    """Loads and returns the training and testing datasets with encoded labels."""
    try:
        print("Loading datasets...")
        x_train = pd.read_csv("/kaggle/working/x_train.csv")
        x_test = pd.read_csv("/kaggle/working/x_test.csv")
        y_train = pd.read_csv("/kaggle/working/y_train.csv").squeeze()
        y_test = pd.read_csv("/kaggle/working/y_test.csv").squeeze()

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform y_train and y_test to encode labels as numeric
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        logging.info("Data loaded and labels encoded successfully.")
        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_pipeline(x_train):
    """Creates a preprocessing pipeline and returns it."""
    try:
        print("Creating preprocessing pipeline...")

        # Identify numeric and categorical columns
        numeric_columns = x_train.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = x_train.select_dtypes(include=['object', 'bool']).columns

        if numeric_columns.empty and categorical_columns.empty:
            raise ValueError("No numeric or categorical columns found in x_train.")

        # Define preprocessing steps for numeric data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing steps for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ],
            remainder='drop'  # Drop unprocessed columns
        )

        # Create pipeline with XGBoost classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                random_state=42, 
                use_label_encoder=False,  # Avoid warning with label encoding
                eval_metric='logloss',    # For classification tasks
                n_estimators=100,
                max_depth=20,
                scale_pos_weight=1       # Adjust class imbalance if needed
            ))
        ])

        print("Pipeline created successfully.")
        return pipeline

    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import logging
from typing import Tuple

def train_and_evaluate_model(pipeline, x_train, x_test, y_train, y_test) -> Tuple[object, np.ndarray]:
    """
    Trains and evaluates an XGBoost model.
    
    Args:
        pipeline: The preprocessing and training pipeline with XGBoost classifier
        x_train: Training features
        x_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        Tuple containing:
        - Trained pipeline
        - Feature importance scores
        
    Raises:
        Exception: If there's an error during training or evaluation
    """
    try:
        print("Training XGBoost model...")
        pipeline.fit(x_train, y_train)

        # Extract feature importances (using gain importance by default)
        xgb_model = pipeline.named_steps['classifier']
        feature_importances = xgb_model.get_booster().get_score(importance_type='gain')
        
        # If using early stopping, print best iteration
        if hasattr(xgb_model, 'best_iteration_'):
            print(f"Best iteration found: {xgb_model.best_iteration_}")

        print("Evaluating model on the testing set...")
        y_pred = pipeline.predict(x_test)
        
        # Get probability predictions for metrics
        y_prob = pipeline.predict_proba(x_test)

        # Evaluate performance
        test_accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("\nModel Performance Metrics:")
        print(f"Testing Set Accuracy: {test_accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # Print top feature importances
        print("\nTop Feature Importances (Gain):")
        sorted_features = sorted(feature_importances.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        for feature, importance in sorted_features[:10]:  # Show top 10 features
            print(f"{feature}: {importance:.4f}")

        return pipeline, feature_importances

    except Exception as e:
        print(f"Error during model training or evaluation: {str(e)}")
        raise


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from typing import List, Union
import warnings

def visualize_model_results(
    x_train: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    pipeline: object,
    top_n_features: int = 10,
    figsize: tuple = (12, 8)
) -> None:
    """
    Generates visualizations of XGBoost model performance and feature importance.
    
    Args:
        x_train: Training features used in the model
        y_test: True labels for the test set (numpy array)
        y_pred: Predicted labels from the model
        pipeline: Trained XGBoost pipeline
        top_n_features: Number of top features to display in feature importance plots
        figsize: Base figure size for plots
    """
    try:
        print("Generating visualizations...")
        
        # Use a modern style without deprecation warnings
        plt.style.use('seaborn-v0_8')
        
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=unique_classes,
            yticklabels=unique_classes
        )
        plt.title("Confusion Matrix", pad=20)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # 2. Classification Report Heatmap
        plt.subplot(2, 2, 2)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        sns.heatmap(
            report_df.iloc[:-1, :-1], 
            annot=True, 
            cmap="YlGnBu", 
            fmt=".2f"
        )
        plt.title("Classification Report Heatmap", pad=20)

        # 3. XGBoost Feature Importance
        plt.subplot(2, 2, 3)
        xgb_model = pipeline.named_steps['classifier']
        
        # Get feature names
        try:
            feature_names = (
                pipeline.named_steps['preprocessor']
                .transformers_[0][2].tolist() +
                pipeline.named_steps['preprocessor']
                .transformers_[1][1].get_feature_names_out().tolist()
            )
        except (AttributeError, IndexError, KeyError):
            # Fallback to generic feature names
            feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]
            
        # Get feature importance scores
        try:
            importance_gains = xgb_model.get_booster().get_score(importance_type='gain')
            importance_weights = xgb_model.get_booster().get_score(importance_type='weight')
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': list(importance_gains.keys()),
                'gain': list(importance_gains.values()),
                'weight': list(importance_weights.values())
            })
            importance_df = importance_df.sort_values('gain', ascending=False)
            
            # Plot feature importance
            sns.barplot(
                x="gain",
                y="feature",
                data=importance_df.head(top_n_features),
                palette="viridis"
            )
            plt.title("Top Feature Importance (Gain)", pad=20)
            plt.xlabel("Gain")
            plt.ylabel("Feature Names")
        except (AttributeError, KeyError):
            plt.text(0.5, 0.5, "Feature importance not available", 
                    ha='center', va='center')
            plt.title("Feature Importance (Not Available)", pad=20)

        # 4. Distribution of Predicted Routes
        plt.subplot(2, 2, 4)
        
        # Convert predictions to pandas Series for better plotting
        pred_series = pd.Series(y_pred)
        value_counts = pred_series.value_counts()
        
        sns.barplot(x=value_counts.index, y=value_counts.values, palette="muted")
        plt.title("Distribution of Predictions", pad=20)
        plt.xlabel("Predicted Class")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)

        # Adjust layout and display
        plt.tight_layout(pad=3.0)
        plt.show()

        # 5. Learning curves (if available)
        if hasattr(xgb_model, 'evals_result_'):
            plt.figure(figsize=(10, 6))
            results = xgb_model.evals_result_
            
            # Check if we have validation results
            if results and 'validation_0' in results:
                metric_name = list(results['validation_0'].keys())[0]
                epochs = len(results['validation_0'][metric_name])
                x_axis = range(epochs)
                
                plt.plot(x_axis, results['validation_0'][metric_name], 
                        label='Train')
                if 'validation_1' in results:
                    plt.plot(x_axis, results['validation_1'][metric_name], 
                            label='Test')
                
                plt.legend()
                plt.ylabel(f'Loss ({metric_name})')
                plt.xlabel('Number of Boosting Rounds')
                plt.title('XGBoost Learning Curves')
                plt.show()

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise


import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings

def main():
    """
    Main execution function for XGBoost model training and evaluation pipeline.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Load and prepare data
        logger.info("Loading datasets...")
        x_train, x_test, y_train, y_test = load_data()
        
        # Create pipeline with XGBoost
        logger.info("Creating preprocessing pipeline...")
        pipeline = create_pipeline(x_train)
        logger.info("Pipeline created successfully.")

        # Train and evaluate model
        logger.info("Starting model training and evaluation...")
        pipeline, feature_importances = train_and_evaluate_model(
            pipeline=pipeline,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test
        )
        
        # Generate predictions for visualization
        logger.info("Generating predictions for visualization...")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            y_pred = pipeline.predict(x_test)
            
        # Visualize results
        logger.info("Creating visualizations...")
        visualize_model_results(
            x_train=x_train,
            y_test=y_test,
            y_pred=y_pred,
            pipeline=pipeline,
            top_n_features=10
        )

        # Save the model (optional)
        logger.info("Saving model...")
        save_model(pipeline, "xgboost_model.pkl")

        logger.info("Program execution completed successfully.")
        return pipeline, feature_importances

    except Exception as e:
        logger.error(f"Program terminated with an exception: {str(e)}", exc_info=True)
        raise

def save_model(model: Pipeline, filename: str) -> None:
    """
    Saves the trained model to disk.
    
    Args:
        model: Trained pipeline with XGBoost model
        filename: Path where model will be saved
    """
    import joblib
    try:
        joblib.dump(model, filename)
        logging.info(f"Model saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        raise

if __name__ == "__main__":
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(42)
    
    # Suppress XGBoost warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        pipeline, feature_importances = main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
    except Exception as e:
        logging.error("Program failed", exc_info=True)


player_play = pd.read_csv("../input/nfl-big-data-bowl-2025/player_play.csv")
games = pd.read_csv("../input/nfl-big-data-bowl-2025/games.csv")
plays = pd.read_csv("../input/nfl-big-data-bowl-2025/plays.csv")
players = pd.read_csv("../input/nfl-big-data-bowl-2025/players.csv")


players.head()


players.info()


players.isnull().sum()


print("Data type of birthDate column before parsing : ", players["birthDate"].dtypes)
players["birthDate"] = pd.to_datetime(players["birthDate"], format='mixed')
print("Data type of birthDate column after parsing : ", players["birthDate"].dtypes)
print(players["birthDate"].head())



players['birthYear']= pd.DatetimeIndex(players['birthDate']).year
print(players['birthYear'])


print(players['birthYear'].value_counts())


print(2025-max(players['birthYear']))
print(2025-min(players['birthYear']))


hist=players['birthYear'].plot.hist(bins=20,color='orange',edgecolor='black')


college_names=players.pivot_table(index= ['collegeName'], aggfunc='size')
college_names=college_names.reset_index()
college_names.columns=['College Names','Counts']
college_names=college_names.sort_values('Counts',ascending=False)
print(college_names)


top_colleges=college_names[0:10]
print(top_colleges)


fig=plt.figure(figsize=(8,8))
circle=plt.Circle((0,0),0.5,color='white')
plt.pie(top_colleges['Counts'],labels=top_colleges['College Names'])
p=plt.gcf()
p.gca().add_artist(circle)
plt.legend(top_colleges['Counts'])
plt.title("Top 10 Colleges having Highest Number of Players",fontsize=25)
plt.show()


pos_val=players.pivot_table(index=['position'], aggfunc='size')
pos_val = pos_val.reset_index()
pos_val.columns=['Positions','Counts']
pos_val = pos_val.sort_values('Counts',ascending=False)
print(pos_val)


height = players[players['height'] == max(players['height'])]
height


lowheight = players[players['height'] == min(players['height'])]
lowheight


oldest = players[players['birthYear'] == min(players['birthYear'])]
oldest


youngest = players[players['birthYear'] == max(players['birthYear'])]
youngest


mean=np.ceil(players['weight'].mean())


median=np.ceil(players['weight'].median())


plt.figure(figsize=(10, 5))
sns.set_style('white')
hist_plot = sns.histplot(players['weight'], )
hist_plot.axvline(mean, color='r', linestyle='--', linewidth = 4, label = f'mean-{mean}')
hist_plot.axvline(median, color='g', linestyle='-', linewidth = 4, label = f'median-{median}')
plt.suptitle("Players Weight Distribution")
plt.legend();


games.tail()


print('NFL Unique values and Their Counts')
g_season=games.pivot_table(index=['season'], aggfunc='size')
g_season = g_season.reset_index()
g_season.columns = ['Seasons','Counts']
g_season = g_season.sort_values('Counts',ascending=False)
print(g_season)


g_week = games.pivot_table(index = ['week'], aggfunc = 'size') 
g_week = g_week.reset_index()
g_week.columns= ["Weeks", "Counts"]
g_week = g_week.sort_values("Counts", ascending = False)
print(g_week)


bar_plot = g_week.plot.barh()
bar_plot.set_title('Unique NFL Weeks and their Counts')
bar_plot.set_xlabel('Counts')
bar_plot.set_ylabel('Weeks')
bar_plot.invert_yaxis()
plt.show(bar_plot)


print('Unique NFL Dates and Their Counts')
g_date = games.pivot_table(index=['gameDate'],aggfunc = 'size')
g_date = g_date.reset_index()
g_date.columns=['Date','Counts']
g_date=g_date.sort_values('Counts',ascending=False)
print(g_date)


bar_plot1=g_date.plot.barh()
bar_plot1.set_title('NFL Event Dates')
bar_plot1.set_xlabel('COunts')
bar_plot1.set_ylabel('Dates')
bar_plot1.invert_yaxis()
plt.show(bar_plot1)


games['gameDay'] = pd.DatetimeIndex(games['gameDate']).day
print(games['gameDay'])


print("Unique NFL days and their counts :")
g_days = games.pivot_table(index = ['gameDay'], aggfunc = 'size') 
g_days = g_days.reset_index()
g_days.columns= ["Day", "Counts"]
g_days = g_days.sort_values("Counts", ascending = False)
print(g_days)


bar_plot3=g_days.plot.barh()
bar_plot3.set_title('NFL Event Days')
bar_plot3.set_xlabel('Counts')
bar_plot3.set_ylabel('Days')
bar_plot3.invert_yaxis()
plt.show(bar_plot3)


print('Unique NFL Timings and Their Counts')
g_time = games.pivot_table(index= ['gameTimeEastern'],aggfunc='size')
g_time = g_time.reset_index()
g_time.columns = ['Time','Counts']
g_time = g_time.sort_values('Counts',ascending=False)
print(g_time)


games['gameTimeEastern'].value_counts().sort_values().plot.barh(color=['blue','red'], title='NFL Event Time')
plt.xlabel('Counts');


print('Unique NFl home and Their Values')
g_home = games.pivot_table(index = ['homeTeamAbbr'], aggfunc='size')
g_home = g_home.reset_index()
g_home.columns = ['Home Team','Counts']
g_home = g_home.sort_values('Counts', ascending = False)
print(g_home)


g_home['Home Team'].value_counts().head(20).plot.barh(color='blue',title='NFL Home Team')
plt.xlabel('Counts')


print("Unique NFL yards to go and their counts :")
g_yards = plays.pivot_table(index = ['yardsToGo'], aggfunc = 'size') 
g_yards = g_yards.reset_index()
g_yards.columns= ["Yards To Go", "Counts"]
g_yards = g_yards.sort_values("Counts", ascending = False)
print(g_yards)


bar_plot = g_yards.plot.barh()
bar_plot.set_title("NFL, Yards to Go")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Yards to Go ")
bar_plot.invert_yaxis() #order increasing
plt.show(bar_plot)


print("Unique NFL Offense Formation and their counts :")
gp_type = plays.pivot_table(index = ['offenseFormation'], aggfunc = 'size') 
gp_type = gp_type.reset_index()
gp_type.columns= ["Offense Formation", "Counts"]
gp_type = gp_type.sort_values("Counts", ascending = False)
print(gp_type)


plays["offenseFormation"].value_counts().plot.barh(color='orange', title='NFL Offense Formation')
plt.xlabel('Counts');


print("Unique NFL Pre-snap Home Team Win Probability and their counts :")
g_home = plays.pivot_table(index = ['preSnapHomeTeamWinProbability'], aggfunc = 'size') 
g_home = g_home.reset_index()
g_home.columns= ["Pre-Snap HomeTeam Win Probability", "Counts"]
g_home = g_home.sort_values("Counts", ascending = False)
print(g_home)


hist = plays["preSnapHomeTeamWinProbability"].plot.hist(bins=25, color="orange", edgecolor="black")
plt.title('NFL Pre-Snap HomeTeam Win Probability');


print("Unique NFL pass results and their counts :")
g_res = plays.pivot_table(index = ['passResult'], aggfunc = 'size') 
g_res = g_res.reset_index()
g_res.columns= ["Pass Results", "Counts"]
g_res = g_res.sort_values("Counts", ascending = False)
print(g_res)


plays["passResult"].value_counts().sort_values().plot.barh(color='red', title='NFL Pass Results')
plt.xlabel('Counts');


print("Unique NFL absolute yardline numbers and their counts :")
g_abyl = plays.pivot_table(index = ['absoluteYardlineNumber'], aggfunc = 'size') 
g_abyl = g_abyl.reset_index()
g_abyl.columns= ["Absolute YardLine Number", "Counts"]
g_abyl = g_abyl.sort_values("Counts", ascending = False)
print(g_abyl)


plays["absoluteYardlineNumber"].value_counts().head(20).sort_values().plot.barh(color='green', title='NFL Absolute Yard Line Number')
plt.xlabel('Counts');

