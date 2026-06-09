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


from IPython.display import Video
Video('/kaggle/input/my-play/my_play.mp4', width=800,height=400,embed=True)


df = pd.read_csv("/kaggle/input/dcp-modeling-data/modeling_data.csv")


import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Features and Target ---
features = [
    'RDD',          # Reaction Decisiveness
    'S_close',      # Closing Speed Component
    'L_score',      # Leverage Quality 
    's_M_status',   # Pre-throw speed (Context)
    'a_M_status',   # Pre-throw acceleration (Context)
]

target = 'Y_frame'

# Drop any tiny remaining NaNs
modeling_df = df.dropna(subset=features + [target])

# --- 2. Split Data by GAME_ID ---
unique_games = modeling_df['game_id'].unique()

# Split 80/20 based on games
np.random.seed(42)
train_games = np.random.choice(unique_games, size=int(len(unique_games) * 0.8), replace=False)

train_df = modeling_df[modeling_df['game_id'].isin(train_games)]
test_df = modeling_df[~modeling_df['game_id'].isin(train_games)]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Training on {len(X_train)} frames ({len(train_games)} games)")
print(f"Testing on {len(X_test)} frames ({len(unique_games) - len(train_games)} games)")

# --- 3. Train XGBoost Model ---
dcp_model = xgb.XGBClassifier(
    n_estimators=200,   # Increased slightly for better convergence
    learning_rate=0.05,
    max_depth=5,       
    subsample=0.8,      # Prevent overfitting
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

dcp_model.fit(X_train, y_train)

# --- 4. Evaluate Performance ---
# Get probabilities
y_pred_prob = dcp_model.predict_proba(X_test)[:, 1]
y_pred = dcp_model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 5. Feature Importance ---
xgb.plot_importance(dcp_model, importance_type='gain', title='Feature Importance (Gain)', show_values=False)
plt.show()


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# ---- Build classification report table ----
report = classification_report(
    y_test,
    y_pred,
    target_names=["Unsuccessful Coverage Position", "Successful Coverage Position"],  # <-- adjust if needed
    output_dict=True
)

report_df = (
    pd.DataFrame(report)
    .T
    .loc[["Unsuccessful Coverage Position", "Successful Coverage Position"], ["precision", "recall", "f1-score"]]
    .round(2)
)

# ---- Confusion matrix ----
cm = confusion_matrix(y_test, y_pred)

# ---- AUC ----
auc = roc_auc_score(y_test, y_pred_prob)

# ---- One combined figure ----
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    height_ratios=[2.2, 2.2, 1.1],  # CM / ROC / Table
    hspace=0.35
)

# --- Confusion Matrix ---
ax_cm = fig.add_subplot(gs[0:2, 0])
ConfusionMatrixDisplay(confusion_matrix=cm).plot(
    ax=ax_cm, values_format="d", colorbar=False
)
ax_cm.set_title("Confusion Matrix", fontsize=12)

# --- ROC Curve ---
ax_roc = fig.add_subplot(gs[0:2, 1])
RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=ax_roc)
ax_roc.set_title(f"ROC Curve (AUC = {auc:.3f})", fontsize=12)

ax_table = fig.add_subplot(gs[2, :])
ax_table.axis("off")

table = ax_table.table(
    cellText=report_df.values,
    rowLabels=report_df.index,
    colLabels=["Precision", "Recall", "F1"],
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.3)

ax_table.set_title("Classification Report", fontsize=12, pad=6)

fig.suptitle("Model Performance Summary", fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 11969 idx of really good play to look at
#choose random play
n = modeling_df.shape[0]
idx = np.random.randint(0,n)
idx = 11969
print(idx)

sample_play = modeling_df.iloc[idx]
GAME_ID = sample_play['game_id']
PLAY_ID = sample_play['play_id']

play_data = modeling_df.loc[(modeling_df['game_id'] == GAME_ID) & 
                        (modeling_df['play_id'] == PLAY_ID)]

play_data = play_data.sort_values(by='frame_id')

#Predict Probabilities
X_play = play_data[features]
play_data['DCP_score'] = dcp_model.predict_proba(X_play)[:, 1]

# 4. Display the Data Table
print(f"Viewing Play: {GAME_ID} - {PLAY_ID}")
play_data[['frame_id', 'DCP_score','Y_frame','L_score', 'RDD', 'S_close','s_M_status','a_M_status',]]


import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- SETUP ---
fig, (ax_field, ax_graph) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(hspace=0.3)

# --- 1. FIELD PLOT SETUP ---
def draw_field(ax):
    # Green Field
    ax.set_facecolor('#79A06D')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    
    # Yard Lines
    for x in range(10, 110, 10):
        ax.axvline(x, color='white', linestyle='-', alpha=0.5)
        # Numbers
        if x < 60: num = x
        else: num = 120 - x
        ax.text(x, 5, str(num), color='white', ha='center', fontsize=10)
        ax.text(x, 53.3-5, str(num), color='white', ha='center', fontsize=10, rotation=180)

draw_field(ax_field)

# Initialize Players
db_dot, = ax_field.plot([], [], 'o', markersize=12, color='blue', label='DB (Assigned)')
wr_dot, = ax_field.plot([], [], 'o', markersize=12, color='red', label='WR (Target)')
ball_dot, = ax_field.plot([], [], 'o', markersize=8, color='brown', label='Ball Land')

# Add Legend
ax_field.legend(loc='upper right')
ax_field.set_title(f"Play: {GAME_ID}-{PLAY_ID} | DB Coverage", fontsize=14, color='black')

# --- 2. PROBABILITY GRAPH SETUP ---
ax_graph.set_xlim(play_data['frame_id'].min(), play_data['frame_id'].max())
ax_graph.set_ylim(0, 1.0)
ax_graph.set_ylabel("Win Probability (DCP)", fontsize=12)
ax_graph.set_xlabel("Frame ID", fontsize=12)
ax_graph.grid(True, linestyle='--', alpha=0.7)

# The Probability Line
prob_line, = ax_graph.plot([], [], color='purple', linewidth=3)
# The "Current Frame" Marker
time_marker = ax_graph.axvline(x=play_data['frame_id'].min(), color='black', linestyle=':')

# Threshold Line (0.5)
ax_graph.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax_graph.text(play_data['frame_id'].min(), 0.52, "Neutral (50%)", color='gray', fontsize=8)

# --- 3. ANIMATION UPDATE FUNCTION ---
def update(frame_idx):
    # Get current frame data
    current_frame = play_data.iloc[frame_idx]
    
    # UPDATE FIELD
    db_dot.set_data([current_frame['x_DB']], [current_frame['y_DB']])
    wr_dot.set_data([current_frame['x_WR']], [current_frame['y_WR']])
    # Show ball land target
    ball_dot.set_data([current_frame['ball_land_x']], [current_frame['ball_land_y']])
    
    # UPDATE GRAPH
    # Get data up to this frame
    history = play_data.iloc[:frame_idx+1]
    prob_line.set_data(history['frame_id'], history['DCP_score'])
    
    # Move the vertical marker
    time_marker.set_xdata([current_frame['frame_id']])
    
    # Update Title with current prob
    prob_val = current_frame['DCP_score']
    ax_graph.set_title(f"Current DCP Score: {prob_val:.1%}", fontsize=14, fontweight='bold', 
                       color='green' if prob_val > 0.5 else 'red')
    
    return db_dot, wr_dot, ball_dot, prob_line, time_marker

# --- 4. GENERATE ANIMATION ---
frames = len(play_data)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Save to file (Change path as needed)
ani.save('dcp_play_animation.mp4', writer='ffmpeg', fps=10)
print("✅ Animation Saved to 'dcp_play_animation.mp4'")

# To display in Jupyter Notebook (optional):
#from IPython.display import HTML
#HTML(ani.to_jshtml())


supp_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')


play_df = modeling_df.copy()
probs = dcp_model.predict_proba(play_df[features])[:, 1]
play_df['DCP_score'] = probs

cols = ["game_id",'play_id',"pass_result"]
pass_results = supp_df[cols]
pass_results['is_incomplete'] = (pass_results['pass_result']=='I').astype(int)

idx_before_arrival = play_df.groupby(['game_id','play_id'])['frame_id'].idxmax()
last_frame_dcp = play_df.loc[idx_before_arrival][['game_id','play_id','DCP_score']]
play_eval_df = (
    pd.merge(last_frame_dcp,pass_results,on=['game_id','play_id'])
)
def calibration_table(df,prob_col='DCP_play',outcome_col='is_incomplete',n_bins=5,custom_bins=None):
    """
    df: play-level DataFrame
    prob_col: column with predicted probability (e.g., DCP_play)
    outcome_col: binary outcome (1 = incompletion, 0 = completion)
    n_bins: number of equal-width bins in [0,1] if custom_bins is None
    custom_bins: explicit list of bin edges, e.g., [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """
    data = df[[prob_col, outcome_col]].dropna().copy()
    
    # Clip probs just in case numerical noise pushed them slightly outside [0,1]
    data[prob_col] = data[prob_col].clip(0, 1)
    
    # Define bins
    if custom_bins is None:
        bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = custom_bins
    
    data['prob_bin'] = pd.cut(
        data[prob_col],
        bins=bins,
        include_lowest=True,
        right=True
    )
    
    # Group by bin and compute stats
    grouped = data.groupby('prob_bin')
    
    calib = grouped.agg(
        n_plays=(outcome_col, 'size'),
        mean_pred_prob=(prob_col, 'mean'),
        incompletion_rate=(outcome_col, 'mean') 
    ).reset_index()
    
    calib['completion_rate'] = 1 - calib['incompletion_rate']
    
    return calib


calib_df = calibration_table(play_eval_df, prob_col='DCP_score', outcome_col='is_incomplete', n_bins=5)
#print(calib_df)


plt.plot(calib_df['mean_pred_prob'], calib_df['incompletion_rate'], marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')  
plt.xlabel('Mean Predicted Coverage Probability')
plt.ylabel('Observed Incompletion Rate')
plt.title('DCP Calibration: Predicted vs Observed Incompletion')
plt.grid(True)
#plt.show()


roc_auc_score(play_eval_df['is_incomplete'], play_eval_df['DCP_score'])


features = ['RDD','S_close', 'L_score', 's_M_status', 'a_M_status']
target = 'Y_frame'
modeling_df = df.dropna(subset=features + [target])
y_pred_prob = dcp_model.predict_proba(modeling_df[features])
modeling_df['DCP_score'] = y_pred_prob[:,1]
modeling_df[['game_id','play_id','frame_id','Y_frame','DCP_score']]


name_map = pd.read_csv('/kaggle/input/name-map/name_map.csv')


scheme = supp_df[['game_id','play_id','team_coverage_type']]
master_df = modeling_df[['game_id','play_id','frame_id','Y_frame','DCP_score','nfl_id_DB']]
master_df = pd.merge(master_df,name_map,how='left',left_on='nfl_id_DB',right_on='nfl_id')
master_df = pd.merge(master_df,scheme,how='left',on=['game_id','play_id'])

arrival_idx = (
    master_df
    .groupby(["game_id", "play_id"])["frame_id"]
    .idxmax()
)

arrival_df = master_df.loc[arrival_idx].reset_index(drop=True)
arrival_df


from sklearn.preprocessing import StandardScaler

cb_summary = (
    arrival_df
    .groupby(["nfl_id_DB", "player_name"])
    .agg(
        mean_dcp=("DCP_score", "mean"),
        win_rate=("Y_frame", "mean"),
        high_dcp_rate=("DCP_score", lambda x: (x >= 0.8).mean()),
        dcp_std=("DCP_score", "std"),
        n_targets=("DCP_score", "count"),
    )
    .reset_index()
)

MIN_TARGETS = 20 
cb_summary = cb_summary.loc[cb_summary['n_targets']>=MIN_TARGETS]
cb_summary.head(10)


metrics = ["mean_dcp", "win_rate", "high_dcp_rate", "dcp_std"]

scaler = StandardScaler()
cb_summary_scaled = cb_summary.copy()
cb_summary_scaled[metrics] = scaler.fit_transform(cb_summary[metrics])

# Lower variance = better → invert
cb_summary_scaled["dcp_std"] *= -1

cb_summary_scaled.head(10)


cb_summary_scaled["CB_score"] = (
    0.3 * cb_summary_scaled["mean_dcp"] +
    0.2 * cb_summary_scaled["win_rate"] +
    0.3 * cb_summary_scaled["high_dcp_rate"] +
    0.2 * cb_summary_scaled["dcp_std"]
)


top_10_cbs = (
    cb_summary_scaled
    .sort_values("CB_score", ascending=False)
    .head(10)
)

top_10_cbs[
    [
        "player_name",
        "CB_score",
        "mean_dcp",
        "win_rate",
        "high_dcp_rate",
        "dcp_std",
        "n_targets",
    ]
]


col_map = {'player_name':"Player Name","CB_score":"CB Score","mean_dcp":"Mean DCP","win_rate":"Win Rate","high_dcp_rate":"High DCP Rate","dcp_std":"DCP Std Dev","n_targets":"Num Targets"}
display_df = cb_summary_scaled[[
    "player_name",
    "CB_score",
    "mean_dcp",
    "win_rate",
    "high_dcp_rate",
    "dcp_std",
    "n_targets"
]].sort_values("CB_score", ascending=False).head(10).rename(col_map,axis=1)

styled_cb_table = (
    display_df
    .style
    .format({
        "CB Score": "{:.2f}",
        "Mean DCP": "{:.2f}",
        "Win Rate": "{:.2f}",
        "High DCP Rate": "{:.2f}",
        "DCP Std Dev": "{:.2f}",
        "Num Targets": "{:d}",
    })
    # Core performance metrics (green = better)
    .background_gradient(
        subset=["CB Score", "Mean DCP", "Win Rate", "High DCP Rate","Num Targets"],
        cmap="YlGn"
    )
    # Variability (lower std = better → invert colormap)
    .background_gradient(
        subset=["DCP Std Dev"],
        cmap="YlOrRd_r"
    )
    .set_properties(
        subset=["Player Name"],
        **{"font-weight": "bold"}
    )
    .set_caption(
        "Top 10 Cornerbacks by Composite DCP Score"
    )
)

styled_cb_table


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.barh(
    top_10_cbs["player_name"][::-1],
    top_10_cbs["CB_score"][::-1]
)
plt.xlabel("CB_score")
plt.title("Top 10 Cornerbacks by Dynamic Coverage Score")
plt.tight_layout()
plt.show()


def radar_plot(cb_df, metrics, title):
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    plt.figure(figsize=(6, 6))

    for _, row in cb_df.iterrows():
        values = row[metrics].values
        values = np.concatenate([values, [values[0]]])
        plt.polar(angles, values, label=row["player_name"])

    plt.xticks(angles[:-1], metrics)
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()

radar_metrics = ["mean_dcp", "high_dcp_rate", "dcp_std"]

radar_plot(
    top_10_cbs.head(3),
    radar_metrics,
    "Technique Profiles of Top CBs"
)


import plotly.graph_objects as go

label_positions = {
    "DaRon Bland": "middle left",
    "Michael Carter II": "middle right",
    "Jonathan Jones": "middle left",
    "L'Jarius Sneed": "top right",
    "Deommodore Lenoir": "top center",
    "Jeff Okudah": "top center",
    "Benjamin St-Juste": "middle left",
    "Emmanuel Forbes": "top left",
    "Pat Surtain II": "bottom center",
    "Charvarius Ward": "middle right",
}
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=cb_summary["mean_dcp"],
        y=cb_summary["dcp_std"],
        mode="markers",
        name="All CBs",
        marker=dict(
            color="rgba(100, 149, 237, 0.4)", 
            size=7
        ),
        hoverinfo="skip" 
    )
)


# Top 10 CBs 
fig.add_trace(
    go.Scatter(
        x=top_10_cbs["mean_dcp"],
        y=top_10_cbs["dcp_std"],
        mode="markers+text",
        name="Top 10 CBs",
        marker=dict(
            color="red",
            size=10,
            line=dict(width=1, color="black")
        ),
        text=top_10_cbs["player_name"],
        textposition=[
            label_positions.get(name, "top center")
            for name in top_10_cbs["player_name"]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Mean DCP: %{x:.3f}<br>"
            "DCP Variability: %{y:.3f}<br>"
            "CB Score: %{customdata[0]:.3f}<br>"
            "Targets: %{customdata[1]}<extra></extra>"
        ),
        customdata=top_10_cbs[["CB_score", "n_targets"]].values
    )
)


# Layout
fig.update_layout(
    title="Coverage Quality vs Consistency (Arrival Frame)",
    xaxis_title="Mean DCP at Arrival",
    yaxis_title="DCP Variability (Std Dev)",
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    template="plotly_white",
    height=550,
    width=750,
    title_x=0.5
)

fig.show()

