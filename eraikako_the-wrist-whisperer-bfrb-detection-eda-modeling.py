import numpy as np
import pandas as pd

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import gaussian_kde
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = 'notebook'
pio.renderers.default = 'iframe_connected'



train_demographics_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')


train_demographics_df.head()


train_demographics_df.describe()


train_df.head()


train_df.shape


label_maps = {
    'adult_child': {0: 'Child', 1: 'Adult'},
    'sex': {0: 'Female', 1: 'Male'},
    'handedness': {0: 'Left-handed', 1: 'Right-handed'}
}

colors = ['#0077b6', '#d3d3d3']

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Adult vs Child", "Sex Assigned at Birth", "Dominant Hand"]
)


for i, var in enumerate(['adult_child', 'sex', 'handedness']):
    counts = train_demographics_df[var].value_counts().sort_index()
    total = counts.sum()
    labels = [label_maps[var][k] for k in counts.index]
    values = counts.values
    percentages = [f"{v} ({v/total:.1%})" for v in values]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors[:len(values)],
            text=percentages,
            textposition='outside'
        ),
        row=1, col=i+1
    )

    fig.update_xaxes(title_text="", showgrid=False, row=1, col=i+1)
    fig.update_yaxes(title_text="", showgrid=False, row=1, col=i+1)


fig.update_layout(
    title_text="Distributions of Binary Demographic Variables",
    title_x=0.5,
    showlegend=False,
    height=500,
    width=1000,
    margin=dict(t=100),
    plot_bgcolor='white',
    paper_bgcolor='white',
    yaxis_domain=[0.0, 0.85],  # shift plots downward
    yaxis2_domain=[0.0, 0.85],
    yaxis3_domain=[0.0, 0.85]
)

fig.update_yaxes(range=[0, 1.15 * train_demographics_df['adult_child'].value_counts().max()], row=1, col=1)
fig.update_yaxes(range=[0, 1.15 * train_demographics_df['sex'].value_counts().max()], row=1, col=2)
fig.update_yaxes(range=[0, 1.15 * train_demographics_df['handedness'].value_counts().max()], row=1, col=3)

fig.show()


age_data = train_demographics_df['age'].dropna()

nbins = 9
counts, bins = np.histogram(age_data, bins=nbins)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
densities = counts / sum(counts) / np.diff(bins)

hist = go.Bar(
    x=bin_centers,
    y=densities,
    width=np.diff(bins),
    name='Age Histogram',
    marker_color='#0077b6',
    opacity=0.6,
    hovertemplate=(
        'Age bin: %{x:.1f}<br>' +
        'Density: %{y:.4f}<br>' +
        'Count: %{customdata}<extra></extra>'
    ),
    customdata=counts
)

kde = gaussian_kde(age_data)
x_vals = np.linspace(age_data.min(), age_data.max(), 200)
kde_line = go.Scatter(
    x=x_vals,
    y=kde(x_vals),
    mode='lines',
    name='KDE',
    line=dict(color='black')
)

fig = go.Figure(data=[hist, kde_line])
fig.update_layout(
    title='Age Distribution of Participants',
    xaxis_title='Age',
    yaxis_title='Density',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()



height_data = train_demographics_df['height_cm'].dropna()


hist = go.Histogram(
    x=height_data,
    histnorm='probability density',
    name='Height Histogram',
    marker_color='#0077b6',
    opacity=0.6,
    hovertemplate=(
        'Height: %{x:.1f} cm<br>' +
        'Density: %{y:.4f}<extra></extra>'
    )
)


kde = gaussian_kde(height_data)
x_vals = np.linspace(height_data.min(), height_data.max(), 200)
kde_line = go.Scatter(
    x=x_vals,
    y=kde(x_vals),
    mode='lines',
    name='KDE',
    line=dict(color='black')
)


fig = go.Figure(data=[hist, kde_line])
fig.update_layout(
    title='Height Distribution of Participants',
    xaxis_title='Height (cm)',
    yaxis_title='Density',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()



sequence_type_summary = (
    train_df.groupby('sequence_id')['sequence_type']
    .agg(
        num_unique='nunique',
        unique_values=lambda x: x.unique().tolist()
    )
    .reset_index()
)

sequence_type_summary['single_value'] = sequence_type_summary['unique_values'].apply(
    lambda x: x[0] if len(x) == 1 else None
)

sequence_type_summary = sequence_type_summary.rename(columns={
    'sequence_id': 'Sequence ID',
    'num_unique': 'Number of Unique Sequence Types',
    'unique_values': 'List of Unique Sequence Types',
    'single_value': 'Single Sequence Type (if only one)'
})



single_type_df = sequence_type_summary.dropna(subset=['Single Sequence Type (if only one)'])

counts = single_type_df['Single Sequence Type (if only one)'].value_counts().sort_index()
labels = counts.index
values = counts.values

percentages = values / values.sum() * 100
text_labels = [f"{v} ({p:.1f}%)" for v, p in zip(values, percentages)]

colors = ['#0077b6', '#d3d3d3']  # blue and light grey

fig = go.Figure()
fig.add_trace(go.Bar(
    x=labels,
    y=values,
    text=text_labels,
    textposition='outside',
    marker_color=colors[:len(labels)],
))

fig.update_layout(
    title='Distribution of Sequence Types: BFRB-like vs. Non-BFRB-like',
    xaxis_title='Sequence Type',
    yaxis_title='Count',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
    margin=dict(t=80),
)

fig.update_xaxes(showgrid=False, title_standoff=10)
fig.update_yaxes(showgrid=True, gridcolor='white')
fig.show()



gesture_type_summary = (
    train_df.groupby('sequence_id')['gesture']
    .agg(
        num_unique='nunique',
        unique_values=lambda x: x.unique().tolist()
    )
    .reset_index()
)

gesture_type_summary['single_value'] = gesture_type_summary['unique_values'].apply(
    lambda x: x[0] if len(x) == 1 else None
)

gesture_type_summary = gesture_type_summary.rename(columns={
    'sequence_id': 'Sequence ID',
    'num_unique': 'Number of Unique Gestures',
    'unique_values': 'List of Unique Gesture Types',
    'single_value': 'Gesture Type (if only one)'
})


target_df = pd.merge(sequence_type_summary, gesture_type_summary, on='Sequence ID', how='inner')

columns_to_keep = ['Sequence ID', 'Single Sequence Type (if only one)', 'Gesture Type (if only one)']
target_df = target_df[columns_to_keep]

target_df = target_df.rename(columns={
    'Sequence ID': 'Sequence ID',
    'Single Sequence Type (if only one)': 'Sequence Type',
    'Gesture Type (if only one)': 'Gesture'
})



gesture_target_counts = (
    target_df.groupby(['Gesture', 'Sequence Type'])
    .size()
    .reset_index(name='Count')
)

fig = px.bar(
    gesture_target_counts,
    x='Gesture',
    y='Count',
    color='Sequence Type',
    barmode='group',
    title='Gesture Distribution by Sequence Type (BFRB-like vs. Non-BFRB-like)',
    color_discrete_map={'Target': '#0077b6', 'Non-Target': '#d3d3d3'}
)


fig.update_layout(
    xaxis_tickangle=45,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()



def plot_tof_sensor_animation(train_df, sequence_id, phase):
    """
    Generate an animated heatmap for all 5 Time-of-Flight (ToF) sensors 
    during a specified phase of a specific sequence.

    Parameters
    ----------
    train_df : pd.DataFrame
        The full training dataframe containing sensor data.
    sequence_id : str
        The ID of the sequence to visualize (e.g., 'SEQ_000091').
    phase : str
        The phase of the sequence to visualize. Must be one of 
        ['Gesture', 'Pause', 'Transition'].

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object showing the animated heatmaps for each ToF sensor.
    """
    seq_df = train_df[(train_df['sequence_id'] == sequence_id) & 
                      (train_df['phase'] == phase)].reset_index(drop=True)

    for sensor_id in range(1, 6):
        tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(64)]
        seq_df[tof_cols] = seq_df[tof_cols].replace(-1, np.nan)

    fig = make_subplots(rows=1, cols=5, subplot_titles=[f"ToF Sensor {i}" for i in range(1, 6)])

    for sensor_id in range(1, 6):
        tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(64)]
        data_0 = seq_df.iloc[0][tof_cols].values.reshape(8, 8)
        fig.add_trace(
            go.Heatmap(
                z=data_0,
                zmin=0, zmax=254,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=sensor_id
        )

    frames = []
    for i in range(len(seq_df)):
        frame_data = []
        for sensor_id in range(1, 6):
            tof_cols = [f"tof_{sensor_id}_v{j}" for j in range(64)]
            heatmap = go.Heatmap(
                z=seq_df.iloc[i][tof_cols].values.reshape(8, 8),
                zmin=0, zmax=254,
                colorscale='Viridis',
                showscale=False
            )
            frame_data.append(heatmap)
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.update_layout(
        title=f"All 5 ToF Sensors – Sequence {sequence_id} – Phase: {phase}",
        height=500,
        width=1300,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play', method='animate', args=[None])]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f.name]], label=f.name) for f in frames],
            active=0
        )],
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )

    fig.frames = frames
    fig.show()


plot_tof_sensor_animation(train_df, 'SEQ_000091', 'Gesture')


plot_tof_sensor_animation(train_df, 'SEQ_000091', 'Transition')


sequence_lengths = train_df.groupby('sequence_id').size().reset_index(name='length')

fig = px.histogram(
    sequence_lengths,
    x='length',
    nbins=30,
    title='Distribution of Sequence Lengths',
    color_discrete_sequence=['#0077b6']
)

fig.update_layout(
    xaxis_title='Sequence Length (number of rows)',
    yaxis_title='Number of Sequences',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()




lengths_df = train_df.groupby(['sequence_id', 'sequence_type']).size().reset_index(name='length')

fig = go.Figure()


colors = {'Target': '#0077b6', 'Non-Target': '#adb5bd'}

for seq_type in lengths_df['sequence_type'].unique():
    subset = lengths_df[lengths_df['sequence_type'] == seq_type]['length']
    
    hist = go.Histogram(
        x=subset,
        name=f"{seq_type} (hist)",
        histnorm='probability density',
        opacity=0.5,
        nbinsx=30,
        marker_color=colors[seq_type],
        showlegend=False
    )
    
    kde = gaussian_kde(subset)
    x_vals = np.linspace(subset.min(), subset.max(), 200)
    kde_line = go.Scatter(
        x=x_vals,
        y=kde(x_vals),
        mode='lines',
        name=f"{seq_type} (KDE)",
        line=dict(color=colors[seq_type])
    )
    
    fig.add_trace(hist)
    fig.add_trace(kde_line)

fig.update_layout(
    title='Sequence Length Distribution by Sequence Type',
    xaxis_title='Sequence Length (number of rows)',
    yaxis_title='Density',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()



# Filter to only Transition phase
transition_df = train_df[train_df['phase'] == 'Transition']

# Compute sequence lengths with sequence_type
transition_lengths = transition_df.groupby(['sequence_id', 'sequence_type']).size().reset_index(name='length')

# Plot
fig = go.Figure()
colors = {'Target': '#0077b6', 'Non-Target': '#adb5bd'}

for seq_type in transition_lengths['sequence_type'].unique():
    subset = transition_lengths[transition_lengths['sequence_type'] == seq_type]['length']
    
    hist = go.Histogram(
        x=subset,
        name=f"{seq_type} (hist)",
        histnorm='probability density',
        opacity=0.5,
        nbinsx=30,
        marker_color=colors[seq_type],
        showlegend=False
    )
    
    kde = gaussian_kde(subset)
    x_vals = np.linspace(subset.min(), subset.max(), 200)
    kde_line = go.Scatter(
        x=x_vals,
        y=kde(x_vals),
        mode='lines',
        name=f"{seq_type} (KDE)",
        line=dict(color=colors[seq_type])
    )
    
    fig.add_trace(hist)
    fig.add_trace(kde_line)

fig.update_layout(
    title='Sequence Length Distribution by Sequence Type (Transition Phase)',
    xaxis_title='Sequence Length',
    yaxis_title='Density',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()



# Filter to only Gesture phase
gesture_df = train_df[train_df['phase'] == 'Gesture']

# Compute sequence lengths with sequence_type
gesture_lengths = gesture_df.groupby(['sequence_id', 'sequence_type']).size().reset_index(name='length')

# Plot
fig = go.Figure()

for seq_type in gesture_lengths['sequence_type'].unique():
    subset = gesture_lengths[gesture_lengths['sequence_type'] == seq_type]['length']
    
    hist = go.Histogram(
        x=subset,
        name=f"{seq_type} (hist)",
        histnorm='probability density',
        opacity=0.5,
        nbinsx=30,
        marker_color=colors[seq_type],
        showlegend=False
    )
    
    kde = gaussian_kde(subset)
    x_vals = np.linspace(subset.min(), subset.max(), 200)
    kde_line = go.Scatter(
        x=x_vals,
        y=kde(x_vals),
        mode='lines',
        name=f"{seq_type} (KDE)",
        line=dict(color=colors[seq_type])
    )
    
    fig.add_trace(hist)
    fig.add_trace(kde_line)

fig.update_layout(
    title='Sequence Length Distribution by Sequence Type (Gesture Phase)',
    xaxis_title='Sequence Length',
    yaxis_title='Density',
    plot_bgcolor='white',
    paper_bgcolor='white',
    bargap=0.1
)

fig.show()



# Cell 1: imports & config
import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

import os
import polars as pl
import kaggle_evaluation.cmi_inference_server

ACC_AXES = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
ROT_AXES = ["rot_x", "rot_y", "rot_z"]  # correlations
DEMOGRAPHIC_COLS = [
    "adult_child", "age", "sex", "handedness",
    "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
]

# Globals filled after training
MODEL = None
FEATURE_COLS = None
GLOBAL_MEDIANS = None
CLASSES = None
NON_TARGET_SET = None  # set of gesture names that are non-target (from train)



# Cell 2: features

def _time_aggs(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, median=np.nan, iqr=np.nan)
    return dict(
        mean=float(np.nanmean(arr)),
        std=float(np.nanstd(arr)),
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        median=float(np.nanmedian(arr)),
        iqr=float(iqr(arr, nan_policy="omit")),
    )

def _pairwise_corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.sum(~np.isnan(a)) < 2 or np.sum(~np.isnan(b)) < 2:
        return 0.0
    a = np.nan_to_num(a, nan=np.nanmean(a))
    b = np.nan_to_num(b, nan=np.nanmean(b))
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0

def features_one_sequence(sdf: pd.DataFrame) -> dict:
    out = {}
    # Acc
    if all(c in sdf.columns for c in ACC_AXES):
        ax, ay, az = sdf["acc_x"].values, sdf["acc_y"].values, sdf["acc_z"].values
        for name, arr in zip(ACC_AXES, [ax, ay, az]):
            for k, v in _time_aggs(arr).items():
                out[f"{name}_{k}"] = v
        res = np.sqrt(ax**2 + ay**2 + az**2)
        for k, v in _time_aggs(res).items():
            out[f"acc_res_{k}"] = v
        out["acc_sma"] = float(np.nanmean(np.abs(ax) + np.abs(ay) + np.abs(az)))
        out["acc_corr_xy"] = _pairwise_corr(ax, ay)
        out["acc_corr_xz"] = _pairwise_corr(ax, az)
        out["acc_corr_yz"] = _pairwise_corr(ay, az)

    # Rot (quaternions)
    if all(c in sdf.columns for c in ROT_COLS):
        rw, rx, ry, rz = [sdf[c].values for c in ROT_COLS]
        for name, arr in zip(ROT_COLS, [rw, rx, ry, rz]):
            for k, v in _time_aggs(arr).items():
                out[f"{name}_{k}"] = v
        out["rot_corr_xy"] = _pairwise_corr(rx, ry)
        out["rot_corr_xz"] = _pairwise_corr(rx, rz)
        out["rot_corr_yz"] = _pairwise_corr(ry, rz)

    return out

def build_feature_table(sensor_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_id, sdf in sensor_df.groupby("sequence_id"):
        row = {"sequence_id": seq_id, "subject": sdf["subject"].iloc[0]}
        row.update(features_one_sequence(sdf))
        if "gesture" in sdf.columns:
            row["gesture"] = str(sdf["gesture"].iloc[-1])
        rows.append(row)

    feat = pd.DataFrame(rows)

    merged = feat.merge(demo_df[["subject"] + DEMOGRAPHIC_COLS], on="subject", how="left")
    for col in DEMOGRAPHIC_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged



# Cell 3: metric helpers

def compute_non_target_set(train_df: pd.DataFrame) -> set:
    """
    Use train metadata to determine which gestures are non-target.
    sequence_type == 'non-target' -> those gestures are non-target.
    """
    assert "sequence_type" in train_df.columns and "gesture" in train_df.columns
    tmp = train_df[["sequence_id", "sequence_type", "gesture"]].drop_duplicates()
    non_targets = set(tmp.loc[tmp.sequence_type == "non-target", "gesture"].astype(str).unique())
    return non_targets

def collapse_to_non_target(arr: np.ndarray, non_target_set: set, non_target_label="non_target"):
    out = []
    for a in arr:
        out.append(non_target_label if a in non_target_set else a)
    return np.array(out, dtype=object)

def competition_score_fulllabels(y_true_labels, y_pred_labels, non_target_set: set, non_target_label="non_target"):
    """
    Binary F1 on target vs non-target, and Macro F1 on gesture where all non-target classes are collapsed into one.
    """
    y_true_labels = np.asarray(y_true_labels, dtype=object)
    y_pred_labels = np.asarray(y_pred_labels, dtype=object)

    # Binary via set membership
    to_bin = lambda arr: np.array([0 if a in non_target_set else 1 for a in arr], dtype=int)
    y_true_bin = to_bin(y_true_labels)
    y_pred_bin = to_bin(y_pred_labels)
    binary_f1 = f1_score(y_true_bin, y_pred_bin, average="binary")

    # Macro-F1 on collapsed labels
    y_true_c = collapse_to_non_target(y_true_labels, non_target_set, non_target_label)
    y_pred_c = collapse_to_non_target(y_pred_labels, non_target_set, non_target_label)
    macro_f1 = f1_score(y_true_c, y_pred_c, average="macro")

    return float(binary_f1), float(macro_f1), float((binary_f1 + macro_f1) / 2.0)



# Cell 4: load, features, CV train

# 1) Load
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
demo_df  = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

# 2) Determine non-target gesture set from train metadata
NON_TARGET_SET = compute_non_target_set(train_df)

# 3) Build features (no label collapsing)
feats = build_feature_table(train_df, demo_df)

# 4) Prepare matrices
drop_cols = ["sequence_id","subject","gesture"]
FEATURE_COLS = [c for c in feats.columns if c not in drop_cols]
X = feats[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
y = feats["gesture"].astype(str).values
groups = feats["subject"].values

# 5) Encode all original gesture classes (BFRB + non-BFRB individually)
CLASSES = np.unique(y)
label2id = {c:i for i,c in enumerate(CLASSES)}
id2label = {i:c for c,i in label2id.items()}
y_int = np.array([label2id[v] for v in y], dtype=int)

# 6) Class-balanced weights
cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_int)
cw_map = {i:w for i,w in enumerate(cw)}

# 7) GroupKFold CV with per-fold median imputation
gkf = GroupKFold(n_splits=5)
oof_proba = np.zeros((len(X), len(CLASSES)))
fold_scores = []

for f, (tr, va) in enumerate(gkf.split(X, y_int, groups), 1):
    X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
    y_tr, y_va = y_int[tr], y_int[va]

    # per-fold medians (no leakage)
    med = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(med); X_va = X_va.fillna(med)

    sw = np.array([cw_map[i] for i in y_tr], dtype=float)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASSES),
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42+f,
        verbose=-1  # suppress logs in recent LightGBM versions
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss"
    )

    proba = model.predict_proba(X_va)
    oof_proba[va] = proba

    y_va_pred = proba.argmax(axis=1)
    y_va_pred_labels = np.array([id2label[i] for i in y_va_pred], dtype=object)
    y_va_true_labels = np.array([id2label[i] for i in y_va], dtype=object)

    b, m, s = competition_score_fulllabels(y_va_true_labels, y_va_pred_labels, NON_TARGET_SET, "non_target")
    fold_scores.append((b, m, s))
    print(f"[Fold {f}] Binary-F1={b:.4f}  Macro-F1(collapse non-target)={m:.4f}  Mean={s:.4f}")

# OOF score
y_oof = oof_proba.argmax(axis=1)
y_oof_labels = np.array([id2label[i] for i in y_oof], dtype=object)
b, m, s = competition_score_fulllabels(y, y_oof_labels, NON_TARGET_SET, "non_target")
print(f"[OOF] Binary-F1={b:.4f}  Macro-F1(collapse non-target)={m:.4f}  Mean={s:.4f}")

# 8) Final model on all data with global medians (for inference)
GLOBAL_MEDIANS = X.median(numeric_only=True).to_dict()
X_full = X.fillna(GLOBAL_MEDIANS)
y_full = y_int
sw_full = np.array([cw_map[i] for i in y_full], dtype=float)

MODEL = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(CLASSES),
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=123,
    verbose=-1
)
MODEL.fit(X_full, y_full, sample_weight=sw_full)

print("Final model trained with full label set (BFRB + non-BFRB individually).")



# Cell 5: predict() + local gateway

def _time_aggs_infer(arr: np.ndarray):
    # simple iqr without scipy to keep predict light
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, median=np.nan, iqr=np.nan)
    q75, q25 = np.nanpercentile(arr, 75), np.nanpercentile(arr, 25)
    return dict(
        mean=float(np.nanmean(arr)),
        std=float(np.nanstd(arr)),
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        median=float(np.nanmedian(arr)),
        iqr=float(q75 - q25),
    )

def _pairwise_corr_infer(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.sum(~np.isnan(a)) < 2 or np.sum(~np.isnan(b)) < 2:
        return 0.0
    a = np.nan_to_num(a, nan=np.nanmean(a))
    b = np.nan_to_num(b, nan=np.nanmean(b))
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0

def _features_one_sequence_infer(sdf: pd.DataFrame) -> dict:
    out = {}
    if all(c in sdf.columns for c in ACC_AXES):
        ax, ay, az = sdf["acc_x"].values, sdf["acc_y"].values, sdf["acc_z"].values
        for name, arr in zip(ACC_AXES, [ax, ay, az]):
            for k, v in _time_aggs_infer(arr).items():
                out[f"{name}_{k}"] = v
        res = np.sqrt(ax**2 + ay**2 + az**2)
        for k, v in _time_aggs_infer(res).items():
            out[f"acc_res_{k}"] = v
        out["acc_sma"] = float(np.nanmean(np.abs(ax) + np.abs(ay) + np.abs(az)))
        out["acc_corr_xy"] = _pairwise_corr_infer(ax, ay)
        out["acc_corr_xz"] = _pairwise_corr_infer(ax, az)
        out["acc_corr_yz"] = _pairwise_corr_infer(ay, az)

    if all(c in sdf.columns for c in ROT_COLS):
        rw, rx, ry, rz = [sdf[c].values for c in ROT_COLS]
        for name, arr in zip(ROT_COLS, [rw, rx, ry, rz]):
            for k, v in _time_aggs_infer(arr).items():
                out[f"{name}_{k}"] = v
        out["rot_corr_xy"] = _pairwise_corr_infer(rx, ry)
        out["rot_corr_xz"] = _pairwise_corr_infer(rx, rz)
        out["rot_corr_yz"] = _pairwise_corr_infer(ry, rz)
    return out

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Return a gesture string exactly from the full set of training gestures (BFRB + non-BFRB),
    not collapsed. The competition's scoring will collapse non-targets for Macro-F1.
    """
    seq_pd = sequence.to_pandas()
    subject = seq_pd["subject"].iloc[0]

    feat = _features_one_sequence_infer(seq_pd)
    row = pd.DataFrame([feat])

    # Merge demographics
    demo_pd = demographics.filter(pl.col("subject") == subject).to_pandas()
    if demo_pd.empty:
        demo_vals = {k: np.nan for k in DEMOGRAPHIC_COLS}
    else:
        demo_vals = {k: pd.to_numeric(demo_pd.iloc[0].get(k, np.nan), errors="coerce") for k in DEMOGRAPHIC_COLS}
    for k, v in demo_vals.items():
        row[k] = v

    # Align
    for c in FEATURE_COLS:
        if c not in row.columns:
            row[c] = np.nan
    row = row[FEATURE_COLS].fillna(value={c: GLOBAL_MEDIANS.get(c, 0.0) for c in FEATURE_COLS})

    # Predict a label from CLASSES
    proba = MODEL.predict_proba(row.values)[0]
    pred_idx = int(np.argmax(proba))
    return CLASSES[pred_idx]




# Run the local gateway in notebook for sanity
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )





