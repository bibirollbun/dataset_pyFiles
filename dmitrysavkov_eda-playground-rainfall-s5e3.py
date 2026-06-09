import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, mean_squared_error, log_loss
from sklearn.linear_model import LogisticRegression

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


DATA_PATH = "/kaggle/input/playground-series-s5e3/"
RSEED = 42
RESULTS_PATH = "/kaggle/working/"

np.random.seed(RSEED)

# Future results of every model
global_results = pd.DataFrame()


def get_current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

def read_data(fileName, dataPath=DATA_PATH):
    path = dataPath + fileName
    return pd.read_csv(path)


train = read_data("train.csv")
test = read_data("test.csv")
sampleSubmission = read_data("sample_submission.csv")
rainfall = read_data("Rainfall.csv", dataPath="/kaggle/input/rainfall-prediction-using-machine-learning/")


train.shape


train.describe()


test.describe()


train.head()


train.info()


# One sample with negative dewpoint in the train. 
print(train[train["dewpoint"] < 0].shape)
train["dewpoint"] = train["dewpoint"].apply(lambda x: 0 if x < 0 else x)


train["rainfall"].value_counts() # a bit imbalanced to rainfall


sampleSubmission


test.info()


test['winddirection'] = test['winddirection'].fillna(train["winddirection"].median())


for c in train.columns:
    stat, p_value = stats.shapiro(train[c])
    is_normal = p_value > 0.05
    print(f"Column {c} is {'' if is_normal else 'not '}normal. p-value: {p_value:.4f}")


rainfall["rainfall"] = rainfall["rainfall"].apply(lambda x: 1 if x == "yes" else 0)


train = train.drop("id", axis=1)


def compare_distributions(df1, df1_name, df2, df2_name):
    """
    Compare the distributions of common features between two dataframes.
    """

    common_cols = set(df1.columns) & set(df2.columns)
    
    print(f"Comparing distributions between {df1_name} and {df2_name}:")
    print("-" * 80)
    
    for col in common_cols:
        try:
            are_distrs_similar = stats.ttest_ind(df1[col], df2[col]).pvalue > 0.05
            print(f"{col}: mean {df1_name}: {round(df1[col].mean(), 2)}; mean {df2_name}: {round(df2[col].mean(), 2)}; Distributions are {'' if are_distrs_similar else 'not '}similar.")
        except:
            print(f"Couldn't compare {col} - possible mismatch in data types or missing values.")
    
    print("-" * 80)

compare_distributions(train, 'train', rainfall, 'rainfall')
print()
compare_distributions(train, 'train', test, 'test')


train.columns


def plot_pairplot(df, hue_column="rainfall"):
    colors = sns.color_palette("viridis", 2)

    plt.figure(figsize=(12, 10))


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if hue_column:
            pairplot = sns.pairplot(
                df, 
                hue=hue_column,
                palette={0: colors[1], 1: colors[0]}, # Blue is rainfall and green is no rainfall.
                corner=True,  # Shows only the lower triangle to reduce redundancy
                plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
                diag_kws={'alpha': 0.6}
            )
        else:
            pairplot = sns.pairplot(
                df,
                corner=True,  
                plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
                diag_kws={'alpha': 0.6}
            )

    pairplot.figure.suptitle('Pairwise Relationships of Features Colored by Rainfall', y=1.02, fontsize=16)
    plt.tight_layout()

plot_pairplot(train)
plot_pairplot(test.drop("id", axis=1), hue_column="")


def plot_heatmap(corr, title="", annot=True, *args, **kwargs):
    plt.figure(figsize=(12, 10))

    # Create a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, 
                annot=annot, fmt=".2f", cbar_kws={"shrink": .8}, **kwargs)

    plt.title(title, fontsize=16)
    plt.tight_layout()


# Verifying correlations 

kendall_corr = train.corr(method='kendall')
plot_heatmap(kendall_corr, title="Kendall Correlation in train.", xticklabels=train.columns)

corr, pvalue = stats.spearmanr(train)
plot_heatmap(corr, title="Spearman Correlation in train.", xticklabels=train.columns, yticklabels=train.columns)

test_without_id = test.drop("id", axis=1)
corr, pvalue = stats.spearmanr(test_without_id)
plot_heatmap(corr, title="Test set correlation.",  xticklabels=test_without_id.columns, yticklabels=test_without_id.columns)

# Using spearman because of its robustness to non-normal distributions and nonlinear data relationship.
# Besides, the fact that rainfall is a binary variable makes it tolerable for this statistical test. 


def plot_horizontal_bar(x, y, title=""):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=x, y=y)
    plt.title(title)
    plt.xlabel("VIF Score")
    plt.ylabel("Column")
    plt.tight_layout()


def get_vif_multicollinearity(df):
    res = pd.DataFrame({
        "column": df.columns
    })

    res["vif_score"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    res = res.sort_values("vif_score", ascending=False)

    plot_horizontal_bar(res["vif_score"], res["column"], title="Variance Inflation Factor (VIF) by Column")
    
    return res

train_init_multicol = get_vif_multicollinearity(train)
print(train_init_multicol[train_init_multicol["column"].isin(["sunshine", "cloud", "humidity"])])
test_init_multicol = get_vif_multicollinearity(test)
print(test_init_multicol[test_init_multicol["column"].isin(["sunshine", "cloud", "humidity"])])


def check_skeweness(X: pd.DataFrame):
    cols = X.columns
    for c in cols:
        sk = round(stats.skew(X[c]), 3)
        kurt = round(stats.kurtosis(X[c]), 3)
        print(f"Column {c} has skeweness of {sk} and kurtosis of {kurt}")


check_skeweness(train.drop("rainfall", axis=1))


print(train["humidity"].value_counts().iloc[:20])
print()
print(train["cloud"].value_counts().iloc[:20])
print()
print(train["sunshine"].value_counts().iloc[:20])


def add_binary_sunshine(df):
    df["is_sunshine"] = df["sunshine"].apply(lambda x: bool(x))
    return df


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    for c in train.columns.drop(["rainfall"]):
        print(f"Column: {c}")
    
        train_dupl_count = train[c].duplicated().value_counts()
        test_dupl_count = test[c].duplicated().value_counts()
    
        print(f"Train duplicates ratio: {round(train_dupl_count[1] / train_dupl_count[0] * 100, 2)}%")
        print(f"Test duplicates ratio: {round(test_dupl_count[1] / test_dupl_count[0] * 100, 2)}%")
        print()


print(train["day"].value_counts().value_counts())
print(test["day"].value_counts().value_counts())


train["day"].plot()


prev = 365
count = 0
error_idx = []
for idx, row in train.iterrows():
    day = row["day"]

    # Mismatch between current and previous days
    if day != prev + 1 and not (prev == 365 and day == 1):
        count += 1
        x = idx % 365 if idx != 365 else 365
        #print(x, day)
        error_idx.append(idx)

    prev = day

print(f"There are {count} erroneous days. \n")
#print(train.loc[1400:1450, ['day']])
print(train.loc[1020:1070, ['day']])


print(max(train["day"]))


for idx in error_idx:
    prev_idx = [idx - 1, idx - 2, idx - 3, idx -10, idx - 50, idx - 100]
    temps = []

    for prev in prev_idx:
        temps.append(train.iloc[prev]["temparature"])

    temp_value = train.iloc[idx]["temparature"]
    print(f"Current temp: {temp_value}. Prev_temps: {temps}. ")


train.describe()


# temp = train[["temparature", "mintemp", "pressure", "humidity"]].copy() # columns with less varience
temp = train.copy()

temp['is_error'] = temp.index.isin(error_idx)

plot_pairplot(temp, hue_column="is_error");


def fix_error_day(error_idx, df):
    for idx in sorted(error_idx):
        prev = df.loc[idx - 1, "day"]
        day = ((prev + 1) % 365) if prev != 364 else 365
        df.loc[idx, "day"] = day
    return df

test_df = train.copy()
#print(test_df["day"].value_counts().value_counts())
test_df = fix_error_day(error_idx, test_df)
print(test_df["day"].value_counts().value_counts())


temp = fix_error_day(error_idx, temp)
plot_pairplot(temp, hue_column="is_error")


train = fix_error_day(error_idx, train)


print(train.shape)


def add_time_series_day(df):
    """
    Map days of the year (1-365) into months (1-12) and weeks (1-53).
    """
    df = df.copy()
    
    month_ranges = {
        1: (1, 31),       # January
        2: (32, 59),      # February
        3: (60, 90),      # March
        4: (91, 120),     # April
        5: (121, 151),    # May
        6: (152, 181),    # June
        7: (182, 212),    # July
        8: (213, 243),    # August
        9: (244, 273),    # September
        10: (274, 304),   # October
        11: (305, 334),   # November
        12: (335, 365)    # December
    }
    
    def day_to_month(day):
        for month, (start, end) in month_ranges.items():
            if start <= day <= end:
                return month
        return None 
    
    df['month'] = df['day'].apply(day_to_month)
    
    df['week'] = np.ceil(df['day'] / 7).astype(int)
    
    return df


def plot_monthly_averages_across_years(df):
    """
    Creates plots showing monthly averages of weather features across different years
    """

    df = df.copy()
    df = add_time_series_day(df)
    
    if 'year' not in df.columns:
        # Calculate year based on index position (6 years total). 
        # NOTE: index starts from 0. 
        df['year'] = (df.index // 365) + 1
    
    weather_features = [
        'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
        'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'
    ]
    
    monthly_avgs = df.groupby(['year', 'month'])[weather_features].mean().reset_index()
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 25), sharex=True)
    axes = axes.flatten()
    
    cmap = plt.cm.viridis
    years = df['year'].unique()
    colors = cmap(np.linspace(0, 1, len(years)))
    
    for i, feature in enumerate(weather_features):
        ax = axes[i]
        
        # Create pivot table for easier plotting
        pivot_data = monthly_avgs.pivot(index='month', columns='year', values=feature)
        
        for j, year in enumerate(pivot_data.columns):
            ax.plot(pivot_data.index, pivot_data[year], marker='o', 
                    linewidth=2, markersize=8, label=f'Year {year}',
                    color=colors[j])
        
        # Add horizontal line for overall average
        avg_value = df[feature].mean()
        ax.axhline(y=avg_value, color='red', linestyle='--', alpha=0.5,
                   label=f'Overall avg: {avg_value:.2f}')
        
        # Format the plot
        ax.set_title(f'Monthly Average {feature.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:
            ax.legend(loc='best')
    
    plt.tight_layout()
    plt.suptitle('Weather Feature Trends by Month Across Years', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    return fig

fig = plot_monthly_averages_across_years(train)
plt.show()


# Stratifying on rainfall to decrease rainfall imbalance in the data. 
X, Xval, y, yval = train_test_split(train.drop("rainfall", axis=1), train["rainfall"], stratify=train["rainfall"], random_state=RSEED)
X_base = X.copy()
Xval_base = Xval.copy()


def print_metrics(ytrue, yhat, threshold=0.5):
    print("Classification Metrics:")
    
    y_binary = (yhat >= threshold)

    auc = roc_auc_score(ytrue, yhat)
    mse = mean_squared_error(ytrue, yhat)
    log_ls = log_loss(ytrue, yhat)

    accuracy = accuracy_score(ytrue, y_binary)
    cm = confusion_matrix(ytrue, y_binary)

    print(f"Log loss:  {log_ls:.4f}")
    print(f"Roc auc:  {auc:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    '''print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")'''

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    return {"auc": auc, "lg_ls": log_ls, "accuracy": accuracy, "mse": mse}


def handle_prediction(X, y, Xval, yval, model, storing=None, print_yhat=False):
    model.fit(X, y)

    yhat = model.predict_proba(X)[:, 1]
    train_res = print_metrics(y, yhat)

    print()

    yhat = model.predict_proba(Xval)[:, 1]
    val_res = print_metrics(yval, yhat)

    # cross_val
    X = pd.concat([X, Xval])
    y = pd.concat([y, yval])
    cross_res = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring="neg_log_loss")
    print("Cross_val Log Loss results:", cross_res)

    if print_yhat:
        print(yhat)

    if storing:
        cur = []
        for res in [train_res, val_res]:
            for name, metric in res.items():
                cur.append(f"{name}_{metric:.4f}")
        cur.append(f"cross_lg_ls_{abs(cross_res.mean())}")
        global_results[storing] = cur

    return model


model = LogisticRegression(random_state=RSEED, max_iter=100000)

handle_prediction(X_base, y, Xval_base, yval, model, storing="Baseline Logistic")


def save_prediction(id, yhat, filename="", path=RESULTS_PATH):
    if not filename:
        filename = get_current_time() + ".csv"

    res = pd.DataFrame({
        "id": id,
        "rainfall": yhat
    })

    path += filename
    res.to_csv(path, index=False)


id = test["id"]
model.fit(train.drop("rainfall", axis=1), train["rainfall"])
yhat_test = model.predict_proba(test_without_id)[:, 1]
save_prediction(id, yhat_test, filename="baseline_logistic.csv")


model.fit(X, y)

model_res = pd.DataFrame({
    "coef": abs(model.coef_.flatten()),
    "columns": X.columns
})

model_res = model_res.sort_values("coef", ascending=False)

# lazy to change legends
plot_horizontal_bar(model_res["coef"], model_res["columns"], title="Baseline Logistic feature importances")


model_res[model_res["columns"] == "winddirection"]

