import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


from prophet import Prophet
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/mlb-player-digital-engagement-forecasting/train.csv")


df.head()


df["ds"] = pd.to_datetime(df["date"].astype(str), format = "%Y%m%d")


def agg_next_day_targets(cell, how="mean"):
    """
    cell: JSON string for nextDayPlayerEngagement (list of dicts)
    returns: (y1,y2,y3,y4,n_players)
    """
    if pd.isna(cell) or cell == "":
        return pd.Series([np.nan]*5)

    arr = json.loads(cell)  # list of dicts
    if not arr:
        return pd.Series([np.nan]*5)

    t1 = np.array([d.get("target1", np.nan) for d in arr], dtype=float)
    t2 = np.array([d.get("target2", np.nan) for d in arr], dtype=float)
    t3 = np.array([d.get("target3", np.nan) for d in arr], dtype=float)
    t4 = np.array([d.get("target4", np.nan) for d in arr], dtype=float)

    if how == "sum":
        f = np.nansum
    else:
        f = np.nanmean

    return pd.Series([f(t1), f(t2), f(t3), f(t4)])

df[["y1","y2","y3","y4"]] = df["nextDayPlayerEngagement"].apply(agg_next_day_targets)


def count_list_json(cell):
    if pd.isna(cell) or cell == "":
        return 0
    try:
        return len(json.loads(cell))
    except Exception:
        return 0

df["n_games_total"] = df["games"].apply(count_list_json)
df["n_transactions_total"] = df["transactions"].apply(count_list_json)


def standings_summaries(cell):
    # returns avg_win_pct, n_win_streak_teams
    if pd.isna(cell) or cell == "":
        return pd.Series([np.nan, 0])

    try:
        arr = json.loads(cell)
        if not arr:
            return pd.Series([np.nan, 0])

        # pct is often a string like "0.567" or numeric; handle both
        pcts = []
        win_streak = 0
        for d in arr:
            pct = d.get("pct", None)
            if pct is not None and pct != "":
                try:
                    pcts.append(float(pct))
                except:
                    pass

            streak = d.get("streakCode", "")
            if isinstance(streak, str) and streak.startswith("W"):
                win_streak += 1

        avg_win_pct = np.mean(pcts) if len(pcts) else np.nan
        return pd.Series([avg_win_pct, win_streak])

    except Exception:
        return pd.Series([np.nan, 0])

df[["avg_win_pct","n_win_streak_teams"]] = df["standings"].apply(standings_summaries)


# simple calendar regressors 
df["is_weekend"] = (df["ds"].dt.dayofweek >= 5).astype(int)


df["y"] = (df["y1"] + df["y2"] + df["y3"] + df["y4"])/4


df.head()


df_prophet = df[[
    "ds","y",
    "n_games_total","n_transactions_total",
    "avg_win_pct","n_win_streak_teams",
    "is_weekend"
]].copy()

df_prophet = df_prophet.dropna(subset=["y"]).sort_values("ds")
df_prophet["avg_win_pct"] = df_prophet["avg_win_pct"].fillna(df_prophet["avg_win_pct"].median())

for c in ["n_games_total","n_transactions_total","n_win_streak_teams","is_weekend"]:
    df_prophet[c] = df_prophet[c].fillna(0)


color_pal = sns.color_palette()
df_prophet_plot = df[["ds", "y"]]
df_prophet_plot["engagement_day"] = df_prophet_plot["ds"] + pd.Timedelta(days=1)
df_prophet_plot = df_prophet_plot[["engagement_day", "y"]].set_index("engagement_day")


import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

MLB_NAVY = "#041E42"
MLB_RED  = "#BF0D3E"

y_col = "y"
s = df_prophet_plot[y_col].sort_index()
roll7 = s.rolling(7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(13, 5.5), dpi=140)
ax.set_facecolor("#fbfbfd")
fig.patch.set_facecolor("white")

# season shading — but clip to data range later
years = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="YS")
for y in years:
    season_start = pd.Timestamp(year=y.year, month=3, day=1)
    season_end   = pd.Timestamp(year=y.year, month=10, day=31)
    ax.axvspan(season_start, season_end, color=MLB_NAVY, alpha=0.05, lw=0)

# Raw points
ax.scatter(s.index, s.values, s=10, alpha=0.5, color=MLB_NAVY, edgecolors="none", label="Daily")

# Rolling avg line (with white outline)
line, = ax.plot(roll7.index, roll7.values, linewidth=2.8, color=MLB_RED, label="7-day rolling avg")
line.set_path_effects([pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

# Title
ax.set_title("MLB Digital Engagement — Daily Avg (mean of target1–target4)", fontsize=13, pad=10)

ax.set_xlabel("Date")
ax.set_ylabel("Engagement")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# remove empty space: hard set x-limits to actual data
ax.set_xlim(s.index.min(), s.index.max())

# tighter: move MLB text right
ax.text(0.925, 0.03, "MLB", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=16,
        color=MLB_NAVY, alpha=0.85, weight="bold", zorder=10)
# Add logo to the right of the text, small and low
logo_path = "/kaggle/input/mlb-logo/Major_League_Baseball_logo.svg.png"
logo = mpimg.imread(logo_path)

# move logo a bit left so it sits right next to the text
imagebox = OffsetImage(logo, zoom=0.045)
ab = AnnotationBbox(
    imagebox,
    (0.928, 0.055),   # <-- was 0.935; move left toward the text
    xycoords="axes fraction",
    frameon=False,
    box_alignment=(0, 0.5),
    pad=0,
    zorder=10
)
ax.add_artist(ab)

ax.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.show()


from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=['Monday','Tuesday',
                                        'Wednesday',
                                        'Thursday','Friday',
                                        'Saturday','Sunday'],
                            ordered=True)


df_prophet_time_series = df_prophet.copy()
df_prophet_time_series["engagement_day"] = df_prophet_time_series["ds"] + pd.Timedelta(days=1)


def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['dayofweek'] = df['engagement_day'].dt.dayofweek
    df['weekday'] = df['engagement_day'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['engagement_day'].dt.quarter
    df['month'] = df['engagement_day'].dt.month
    df['year'] = df['engagement_day'].dt.year
    df['dayofyear'] = df['engagement_day'].dt.dayofyear
    df['dayofmonth'] = df['engagement_day'].dt.day
    # df['weekofyear'] = df['ds'].dt.weekofyear
    df['date_offset'] = (df['engagement_day'].dt.month*100 + df['engagement_day'].dt.day - 320)%1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                          labels=['Spring', 'Summer', 'Fall', 'Winter']
                   )
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekday',
           'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(df_prophet_time_series, label='y')
features_and_target = pd.concat([X, y], axis=1)


fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=features_and_target.dropna(),
            x='weekday',
            y='y',
            hue='season',
            ax=ax,
            linewidth=1)
ax.set_title("Weekday vs MLB Engagement (by Season)")
ax.set_xlabel("Weekday")
ax.set_ylabel("Next-day Avg Engagement")
ax.legend(bbox_to_anchor=(1, 1))
plt.show()


df_train = df_prophet.iloc[:-200, :]
df_test= df_prophet.iloc[-200:, :]


# Plot to show train vs test engagement
df_train_plot = df_train[["ds", "y"]].copy()
df_test_plot = df_test[["ds", "y"]].copy()
df_train_plot["engagement_day"] = df_train_plot["ds"] + pd.Timedelta(days=1)
df_test_plot["engagement_day"] = df_test_plot["ds"] + pd.Timedelta(days=1)

df_train_plot = df_train_plot[["engagement_day", "y"]].set_index("engagement_day")
df_test_plot = df_test_plot[["engagement_day", "y"]].set_index("engagement_day")


MLB_NAVY = "#041E42"
MLB_RED  = "#BF0D3E"
TEST_GOLD = "#F5B700"

s_train = df_train_plot["y"].sort_index()
s_test  = df_test_plot["y"].sort_index()

r_train = s_train.rolling(7, min_periods=1).mean()
r_test  = s_test.rolling(7, min_periods=1).mean()

split_date = s_test.index.min()

fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
ax.set_facecolor("#fbfbfd")
fig.patch.set_facecolor("white")

# --- background shading for train/test ---
ax.axvspan(s_train.index.min(), split_date, color=MLB_NAVY, alpha=0.04, lw=0)
ax.axvspan(split_date, s_test.index.max(), color=TEST_GOLD, alpha=0.06, lw=0)

# --- scatter (lighter) ---
ax.scatter(s_train.index, s_train.values, s=9,  alpha=0.18, color=MLB_NAVY, edgecolors="none", label="Train (daily)")
ax.scatter(s_test.index,  s_test.values,  s=10, alpha=0.22, color=TEST_GOLD, edgecolors="none", label="Test (daily)")

# --- rolling lines (hero) with white outline ---
l1, = ax.plot(r_train.index, r_train.values, color=MLB_RED, linewidth=2.8, label="Train (7-day avg)")
l1.set_path_effects([pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()])

l2, = ax.plot(r_test.index, r_test.values, color="black", alpha=0.85, linewidth=2.8, label="Test (7-day avg)")
l2.set_path_effects([pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()])

# --- split line + label (cleaner positioning) ---
ax.axvline(split_date, linestyle="--", linewidth=2, alpha=0.55, color=MLB_NAVY)
ax.text(split_date, 0.98, " Test starts", transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=10, color=MLB_NAVY, alpha=0.85)

# --- titles/labels ---
ax.set_title("MLB Next-day Engagement — Train vs Test", fontsize=14, pad=10)
ax.set_xlabel("Engagement day")
ax.set_ylabel("Avg engagement (mean of target1–target4)")

# --- date formatting ---
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# --- grid + spines ---
ax.grid(True, alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- limits so no extra whitespace ---
ax.set_xlim(s_train.index.min(), s_test.index.max())

# --- nicer legend ---
leg = ax.legend(frameon=True, loc="upper left")
leg.get_frame().set_alpha(0.95)
leg.get_frame().set_edgecolor("#dddddd")

plt.tight_layout()
plt.show()


df_train["ds"] = df_train["ds"] + pd.Timedelta(days=1)
df_test["ds"] = df_test["ds"] + pd.Timedelta(days=1)


m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.9
)

# US holidays 
m.add_country_holidays(country_name="US")

# Add regressors
m.add_regressor("n_games_total")
m.add_regressor("n_transactions_total")
m.add_regressor("avg_win_pct")
m.add_regressor("n_win_streak_teams")
m.add_regressor("is_weekend")

m.fit(df_train)


test_fcst = m.predict(df_test)


fig, ax = plt.subplots(figsize=(10, 5))
fig = m.plot(test_fcst, ax=ax)
ax.set_title('Prophet Forecast')
plt.show()


fig = m.plot_components(test_fcst)
plt.show()



MLB_NAVY = "#041E42"
MLB_RED  = "#BF0D3E"
TEST_GOLD = "#F5B700"

# df_test must have ds, y
# test_fcst must have ds, yhat, yhat_lower, yhat_upper
# df_train optional for context (ds,y)

fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
ax.set_facecolor("#fbfbfd")
fig.patch.set_facecolor("white")

# 1) history (context)
if "df_train" in globals():
    ax.scatter(df_train["ds"], df_train["y"],
               s=10, alpha=0.12, color=MLB_NAVY, edgecolors="none",
               label="Train (actual)")

# 2) uncertainty band (subtle)
ax.fill_between(test_fcst["ds"],
                test_fcst["yhat_lower"],
                test_fcst["yhat_upper"],
                alpha=0.18, linewidth=0,
                label="Forecast interval")

# 3) forecast line (hero)
line, = ax.plot(test_fcst["ds"], test_fcst["yhat"],
                color=MLB_RED, linewidth=2.8,
                label="Prophet forecast (yhat)")
line.set_path_effects([pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()])

# 4) actual test points (nice + readable)
ax.scatter(df_test["ds"], df_test["y"],
           s=22, alpha=0.75, color=TEST_GOLD,
           edgecolors="white", linewidth=0.6,
           label="Test (actual)")

# 5) split line
split_date = df_test["ds"].min()
ax.axvline(split_date, linestyle="--", linewidth=2, alpha=0.55, color=MLB_NAVY)
ax.text(split_date, 0.98, " Test starts", transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=10, color=MLB_NAVY, alpha=0.85)

# labels
ax.set_title("Prophet Forecast vs Actuals — MLB Next-day Engagement", fontsize=14, pad=10)
ax.set_xlabel("Date")
ax.set_ylabel("Avg engagement (mean of target1–target4)")

# dates/grid/spines
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(True, alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# x-limits to remove extra whitespace
ax.set_xlim(min(df_train["ds"].min(), df_test["ds"].min()) if "df_train" in globals() else df_test["ds"].min(),
            max(df_test["ds"].max(), test_fcst["ds"].max()))

# legend (clean)
leg = ax.legend(frameon=True, loc="upper left")
leg.get_frame().set_alpha(0.95)
leg.get_frame().set_edgecolor("#dddddd")

plt.tight_layout()
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 1) Prophet eval
merged_prophet = df_test.merge(test_fcst[["ds","yhat"]], on="ds", how="inner").dropna()
print("Prophet MAE:", mean_absolute_error(merged_prophet["y"], merged_prophet["yhat"]))
print("Prophet RMSE:", rmse(merged_prophet["y"], merged_prophet["yhat"]))

# 2) Seasonal naive - same day last year
# Build a lookup table from history (train) keyed by date
history = df_train[["ds","y"]].copy()
history["ds"] = pd.to_datetime(history["ds"])
history = history.sort_values("ds")

test = df_test[["ds","y"]].copy()
test["ds"] = pd.to_datetime(test["ds"])
test = test.sort_values("ds")

# For each test day, look up y from exactly 365 days earlier
test["ds_last_year"] = test["ds"] - pd.Timedelta(days=365)

lookup = history.rename(columns={"ds": "ds_last_year", "y": "y_last_year"})
seasonal = test.merge(lookup, on="ds_last_year", how="left").dropna(subset=["y_last_year"])

print("Seasonal naive (t-365d) MAE:", mean_absolute_error(seasonal["y"], seasonal["y_last_year"]))
print("Seasonal naive (t-365d) RMSE:", rmse(seasonal["y"], seasonal["y_last_year"]))


 import plotly.graph_objects as go

# thresholds from training distribution
p75 = df_train["y"].quantile(0.75)
p90 = df_train["y"].quantile(0.90)

plot = df_test.merge(test_fcst[["ds","yhat","yhat_lower","yhat_upper"]], on="ds", how="inner").dropna()
plot = plot.sort_values("ds")

# action levels
plot["action"] = np.where(plot["yhat"] >= p90, "High",
                  np.where(plot["yhat"] >= p75, "Medium", "Low"))

fig = go.Figure()

# uncertainty band
fig.add_trace(go.Scatter(
    x=pd.concat([plot["ds"], plot["ds"][::-1]]),
    y=pd.concat([plot["yhat_upper"], plot["yhat_lower"][::-1]]),
    fill="toself",
    line=dict(width=0),
    name="Forecast interval",
    hoverinfo="skip",
    opacity=0.2
))

# forecast line
fig.add_trace(go.Scatter(
    x=plot["ds"], y=plot["yhat"],
    mode="lines",
    name="Prophet forecast",
))

# actual points
fig.add_trace(go.Scatter(
    x=plot["ds"], y=plot["y"],
    mode="markers",
    name="Actual",
    marker=dict(size=6),
))

# highlight action-level points (optional markers on top)
for level in ["High","Medium","Low"]:
    sub = plot[plot["action"] == level]
    fig.add_trace(go.Scatter(
        x=sub["ds"], y=sub["yhat"],
        mode="markers",
        name=f"Action: {level}",
        marker=dict(size=7, symbol="circle-open"),
        hovertemplate="Date=%{x}<br>yhat=%{y:.3f}<extra></extra>"
    ))

fig.update_layout(
    title="Decision View: Forecast + Uncertainty + Action Level",
    xaxis_title="Date",
    yaxis_title="Engagement",
    legend_title="",
    template="plotly_white",
    height=520
)

# fig.show()
iplot(fig)


# Merge actual vs predicted
err_df = (
    df_test
    .merge(test_fcst[["ds", "yhat"]], on="ds", how="inner")
    .dropna()
    .sort_values("ds")
)

# Compute errors
err_df["error"] = err_df["y"] - err_df["yhat"]          # positive = underpredicted
err_df["abs_error"] = err_df["error"].abs()

# Top 10 biggest misses
top10 = (
    err_df.sort_values("abs_error", ascending=False)
    .head(10)
    .copy()
)

# Optional: nicer formatting
top10["ds"] = pd.to_datetime(top10["ds"]).dt.date
# top10 = top10[["ds", "y", "yhat", "error", "abs_error"]]

top10

