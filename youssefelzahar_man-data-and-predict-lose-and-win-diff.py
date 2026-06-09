import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def showing_data(path):
    data=pd.read_csv(path)
    data.head()
    data.info()
    data.describe()
    return data




mteams=showing_data("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
mteams


mseasons=showing_data("/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv")
mseasons





mseeds=showing_data("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
mseeds



mregularseason=showing_data("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
mregularseason


mresults=showing_data("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
mresults



total_rows_mresults=mresults.shape[0]
null_data_mresults = mresults.isna().sum().reset_index().rename(columns = {0: "Nulls_Count", "index": "Column_Name"}).sort_values(by="Nulls_Count", ascending=False)
null_data_mresults['Percentage']=(null_data_mresults['Nulls_Count']/total_rows_mresults)*100
null_data_mresults[null_data_mresults["Nulls_Count"] > 0]



mresults.duplicated().sum()



def check_normality(data):
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Create subplots for each numeric column
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
    
    # If there's only one numeric column, wrap axes in a list to make it 2D
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(numeric_cols):
        # Histogram
        sns.histplot(data=data, x=col, kde=True, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'Distribution of {col}')
        
        # Q-Q plot
        stats.probplot(data[col], dist="norm", plot=axes[idx, 1])
        axes[idx, 1].set_title(f'Q-Q Plot of {col}')
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(data[col])
        print(f"\nShapiro-Wilk test for {col}:")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print("Conclusion:", "Normal distribution" if p_value > 0.05 else "Not normal distribution")
        
        # Skewness and Kurtosis
        skewness = data[col].skew()
        kurtosis = data[col].kurtosis()
        print(f"\nSkewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
    
    plt.tight_layout()
    plt.show()

# First, import required libraries if not already imported
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Check normality for mresults
check_normality(mresults)


mresults_numbers=mresults.select_dtypes(include=["int64"])
mresults_objects=mresults.select_dtypes(include=["object"])



def check_outliers(df):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                print(f"Outliers in {col}:")
                print(outliers)
    return df
outliers=check_outliers(mresults)


def solve_outliers(df,name):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr    
            df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
            print(f"Outliers in {col} solved")
            
    return df
solve_outliers=solve_outliers(mresults, "Tournament Results")



def check_skewness(df,name):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            print(f"Skewness of {col}: {df[col].skew()}")
    return df
skewness=check_skewness(solve_outliers, "Tournament Results")



skewness.head()


wloc=skewness["WLoc"]
wteamid=skewness["WTeamID"]
lteamid=skewness["LTeamID"]
season=skewness["Season"]
mresults=skewness.drop(["Season","WLoc","WTeamID","LTeamID"],axis=1)





mresults


mresults=pd.concat([mresults,wloc,wteamid,lteamid,season],axis=1)
mresults.head()


def process_results(df):
    df["PointDiff"] = df["WScore"] - df["LScore"]
    df["WLoc"] = df["WLoc"].map({"H": 1, "A": -1, "N": 0})  # تشفير الموقع
    return df
mresults=process_results(mresults)


mresults["PointDiff"].unique()


mteams.isnull().sum()


mteams.duplicated().sum()



check_normality(mteams)



check_outliers(mteams[["FirstD1Season","LastD1Season"]])



mresults.head()


mresults=mresults.merge(mteams,left_on="WTeamID",right_on="TeamID",how="left")
mresults.rename(columns={"TeamName": "WTeamName"}, inplace=True)
mresults.drop(columns=["TeamID"], inplace=True)
mresults=mresults.merge(mteams,left_on="LTeamID",right_on="TeamID",how="left")
mresults.rename(columns={"TeamName": "LTeamName"}, inplace=True)
mresults.drop(columns=["TeamID"], inplace=True)
mresults.head()







mseasons.head()


mseasons.isnull().sum()
mseasons.duplicated().sum()
mseasons.info()







# ... existing code ...
# Convert date column to datetime
mseasons["DayZero"] = pd.to_datetime(mseasons['DayZero'], format='%m/%d/%Y')

mseasons.head()



mseasons.info()








mresults=mresults.merge(mseasons,left_on="Season",right_on="Season",how="left")
mresults.head()











mregularseason.head()



mregularseason.info()



mregularseason.isnull().sum()


mregularseason.duplicated().sum()


check_outliers(mregularseason)


def solve_outliers(df):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr    
            df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
            print(f"Outliers in {col} solved")
            
    return df
solve_outliers=solve_outliers(mregularseason)


check_skewness(solve_outliers,"name")


check_normality(solve_outliers[["DayNum",	"WTeamID",	"WScore","LTeamID"]])


def improve_normality(df):
    from scipy import stats
    import numpy as np
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    transformed_df = df.copy()
    
    for col in numeric_columns:
        # Skip if column contains zeros or negative values
        if (df[col] <= 0).any():
            continue
            
        # Try different transformations and keep the one that gives best normality
        original_shapiro = stats.shapiro(df[col])[0]
        
        # Log transformation
        log_transform = np.log1p(df[col])
        log_shapiro = stats.shapiro(log_transform)[0]
        
        # Square root transformation
        sqrt_transform = np.sqrt(df[col])
        sqrt_shapiro = stats.shapiro(sqrt_transform)[0]
        
        # Box-Cox transformation
        try:
            boxcox_transform, _ = stats.boxcox(df[col])
            boxcox_shapiro = stats.shapiro(boxcox_transform)[0]
        except:
            boxcox_shapiro = -1
            
        # Find the best transformation
        transformations = {
            'original': (original_shapiro, df[col]),
            'log': (log_shapiro, log_transform),
            'sqrt': (sqrt_shapiro, sqrt_transform),
            'boxcox': (boxcox_shapiro, boxcox_transform if boxcox_shapiro > 0 else df[col])
        }
        
        best_transform = max(transformations.items(), key=lambda x: x[1][0])
        transformed_df[col] = best_transform[1][1]
        print(f"Column {col}: Best transformation was {best_transform[0]}")
    
    return transformed_df

mregularseason_normalized = improve_normality(solve_outliers[["DayNum",	"WTeamID",	"WScore","LTeamID"]])
# ... existing code ...


mregularseason_normalized


temp_columns = mregularseason_normalized[["DayNum","WTeamID","WScore","LTeamID"]].copy()

mregularseason_rest = mregularseason.drop(["DayNum","WTeamID","WScore","LTeamID"], axis=1)

mregularseason = pd.concat([temp_columns, mregularseason_rest], axis=1)
mregularseason.head()



mregularseason["WLoc"].value_counts()


def process_results(df):
    df["PointDiff"] = df["WScore"] - df["LScore"]
    df["WLoc"] = df["WLoc"].map({"H": 1, "A": -1, "N": 0})  # تشفير الموقع
    return df
regular_season = process_results(mregularseason)



team_stats = mregularseason.groupby("WTeamID").agg(
    avg_points=("WScore", "mean"),
    avg_points_against=("LScore", "mean"),
    games_played=("WTeamID", "count")
).reset_index()
team_stats.rename(columns={"WTeamID": "TeamID"}, inplace=True)



tournament = mresults.merge(team_stats, left_on="WTeamID", right_on="TeamID", how="left")
tournament = mresults.merge(team_stats, left_on="LTeamID", right_on="TeamID", how="left", suffixes=("_W", "_L"))


tournament.head()


tournament.columns


tournamentnumbers=tournament.select_dtypes(["int64","float64"])
features=tournamentnumbers.drop("PointDiff",axis=1)
y = tournamentnumbers["PointDiff"]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train) 




y_pred=model.predict(X_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")



y_pred 

