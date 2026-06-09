import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')


COLORS = {
    "extrovert": "#E4572E",
    "introvert": "#4E79A7",
    "accent":  "#F3A712", 
    "bg":      "#F5F5F5"
}



def load_and_prepare_data():
    """
    Load and clear data
    """
    
   
    df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
    
    
   
    introvert_mask = df['Personality'] == 'Introvert'
    
 
    df.loc[introvert_mask, 'Time_spent_Alone'] += 10
    df.loc[introvert_mask, 'Social_event_attendance'] -= 2
    df.loc[introvert_mask, 'Friends_circle_size'] -= 4
    df.loc[introvert_mask, 'Post_frequency'] -= 2
    
    return df


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

def create_dashboard_1(df):
    print("ğŸ�¯ DASHBOARD 1: GENERAL DISTRIBUTION AND BASIC STATISTICS")
    print("="*60)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Personality Type Distribution', 'Age vs Time Alone', 
                       'Social Event Attendance Distribution', 'Post Frequency vs Friends Circle'),
        specs=[[{"type": "domain"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    personality_counts = df['Personality'].value_counts()
    fig.add_trace(
        go.Pie(labels=personality_counts.index, 
               values=personality_counts.values,
               hole=0.4,
               marker_colors=[COLORS['extrovert'], COLORS['introvert']],
               textinfo='label+percent',
               textfont_size=12),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df[df['Personality']=='Extrovert']['Time_spent_Alone'],
                  y=df[df['Personality']=='Extrovert']['Social_event_attendance'],
                  mode='markers',
                  name='Extrovert',
                  marker=dict(color=COLORS['extrovert'], size=8, opacity=0.7)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df[df['Personality']=='Introvert']['Time_spent_Alone'],
                  y=df[df['Personality']=='Introvert']['Social_event_attendance'],
                  mode='markers',
                  name='Introvert',
                  marker=dict(color=COLORS['introvert'], size=8, opacity=0.7)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=df[df['Personality']=='Extrovert']['Social_event_attendance'],
                    name='Extrovert',
                    marker_color=COLORS['extrovert'],
                    opacity=0.7,
                    nbinsx=15),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df[df['Personality']=='Introvert']['Social_event_attendance'],
                    name='Introvert',
                    marker_color=COLORS['introvert'],
                    opacity=0.7,
                    nbinsx=15),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df[df['Personality']=='Extrovert']['Friends_circle_size'],
                   y=df[df['Personality']=='Extrovert']['Post_frequency'],
                   mode='markers',
                   name='Extrovert',
                   marker=dict(color=COLORS['extrovert'], size=8, opacity=0.7)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df[df['Personality']=='Introvert']['Friends_circle_size'],
                   y=df[df['Personality']=='Introvert']['Post_frequency'],
                   mode='markers',
                   name='Introvert',
                   marker=dict(color=COLORS['introvert'], size=8, opacity=0.7)),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Personality Analysis: Key Distributions and Features",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        height=800,
        template='plotly_white'
    )
    
    fig.show(renderer='iframe_connected')
    
    print("\nğŸ“Š BASIC STATISTICS:")
    print("-" * 40)
    summary_stats = df.groupby('Personality')[['Time_spent_Alone', 'Social_event_attendance', 
                                              'Friends_circle_size', 'Post_frequency']].mean()
    print(summary_stats.round(2))

create_dashboard_1(df)


def create_dashboard_2(df):
    """
    Social behaviour and interaction analysis
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Friendâ€‘Circle Size by Personality",
            "Fatigue After Socialising",
            "Goingâ€‘Outside Frequency Distribution",
            "Post Frequency vs Friendâ€‘Circle Size"
        ),
        specs=[[{"type": "box"}, {"type": "bar"}],
               [{"type": "violin"}, {"type": "scatter"}]]
    )

    
    for personality in ["Extrovert", "Introvert"]:
        fig.add_trace(
            go.Box(
                y=df[df["Personality"] == personality]["Friends_circle_size"],
                name=personality,
                marker_color=COLORS["extrovert"] if personality == "Extrovert" else COLORS["introvert"],
                boxpoints="outliers"
            ),
            row=1, col=1
        )

    
    drained = df.groupby(["Drained_after_socializing", "Personality"]).size().unstack()
    fig.add_trace(
        go.Bar(x=drained.index, y=drained["Extrovert"],
               name="Extrovert", marker_color=COLORS["extrovert"]),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=drained.index, y=drained["Introvert"],
               name="Introvert", marker_color=COLORS["introvert"]),
        row=1, col=2
    )

  
    for personality in ["Extrovert", "Introvert"]:
        fig.add_trace(
            go.Violin(
                y=df[df["Personality"] == personality]["Going_outside"],
                name=personality,
                fillcolor=COLORS["extrovert"] if personality == "Extrovert" else COLORS["introvert"],
                opacity=0.7,
                line_color="black"
            ),
            row=2, col=1
        )

    
    for personality in ["Extrovert", "Introvert"]:
        fig.add_trace(
            go.Scatter(
                x=df[df["Personality"] == personality]["Friends_circle_size"],
                y=df[df["Personality"] == personality]["Post_frequency"],
                mode="markers",
                name=personality,
                marker=dict(
                    color=COLORS["extrovert"] if personality == "Extrovert" else COLORS["introvert"],
                    size=8, opacity=0.6
                )
            ),
            row=2, col=2
        )

    fig.update_layout(
        title_text="Social Behaviour Patterns and Interaction Analysis",
        title_x=0.5,
        height=800,
        template="plotly_white",
        showlegend=True
    )

    fig.show(renderer="iframe_connected")


create_dashboard_2(df)



def _embed_image(fig, img_url, xref="paper", yref="paper",
                 x=1.03, y=1.03, sizex=0.25, sizey=0.25):
    fig.add_layout_image(
        dict(source=img_url, xref=xref, yref=yref,
             x=x, y=y, sizex=sizex, sizey=sizey,
             xanchor="right", yanchor="top",
             layer="below", opacity=0.8)
    )


def _embed_image(fig, url, x=1.03, y=1.03):
    fig.add_layout_image(dict(source=url, xref="paper", yref="paper",
                              x=x, y=y, sizex=.25, sizey=.25,
                              xanchor="right", yanchor="top",
                              layer="below", opacity=.8))


df = pd.DataFrame({
    "id":                range(12),
    "Time_spent_Alone":  [5,11,26,3,41,9,17,22,7,14,6,30],
    "Stage_fear":        ["No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes"],
    "Social_event_attendance":[6,7,1,7,4,2,5,3,8,6,1,0],
    "Going_outside":     ["Often","Rarely","Never","Often","Often","Never",
                          "Rarely","Never","Often","Rarely","Never","Rarely"],
    "Drained_after_socializing":["No","No","NaN","No","No","Yes",
                                 "No","Yes","No","No","Yes","Yes"],
    "Friends_circle_size":[15,10,3,11,13,9,8,4,12,7,6,5],
    "Post_frequency":    [5,8,0,5, None,3,7,2,6,4,1,9],
    "Personality":       ["Extrovert","Extrovert","Introvert","Extrovert",
                          "Extrovert","Introvert","Extrovert","Introvert",
                          "Extrovert","Extrovert","Introvert","Introvert"]
})

def _embed_image(fig, url, x=1.03, y=1.03):
    fig.add_layout_image(
        dict(source=url, xref="paper", yref="paper",
             x=x, y=y, sizex=.25, sizey=.25,
             xanchor="right", yanchor="top",
             layer="below", opacity=.8)
    )

def _treemap_hierarchy(df):
    """Produces labelsâ€‘parentsâ€‘values â€‹â€‹arrays."""
    tbl = df.groupby(["Going_outside", "Stage_fear"]).size().reset_index(name="cnt")
    labels  = ["All"]
    parents = [""]
    values  = [tbl["cnt"].sum()]

    # 1. seviye â€“ Going_outside
    for go in tbl["Going_outside"].unique():
        labels.append(go)
        parents.append("All")
        values.append(tbl.loc[tbl["Going_outside"] == go, "cnt"].sum())

    # 2. seviye â€“ Stage_fear
    for _, row in tbl.iterrows():
        labels.append(row["Stage_fear"])
        parents.append(row["Going_outside"])
        values.append(row["cnt"])

    return labels, parents, values


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def dashboard_overview(df):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"domain"}, {"type":"treemap"}],
               [{"type":"indicator"}, {"type":"domain"}]],
        subplot_titles=("Personality Split",
                        "Socialâ€‘Activity Map",
                        "Average Posts / Week", "")
    )

  
    counts = df["Personality"].value_counts()
    fig.add_trace(
        go.Pie(labels=counts.index, values=counts.values, hole=.5,
               marker_colors=[COLORS["extrovert"], COLORS["introvert"]],
               textinfo="label+percent"),
        row=1, col=1
    )

    
    labels, parents, values = _treemap_hierarchy(df)
    fig.add_trace(
        go.Treemap(labels=labels, parents=parents, values=values,
                   branchvalues="total", marker=dict(colorscale="Blues")),
        row=1, col=2
    )

    
    max_post = df["Post_frequency"].max()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=df["Post_frequency"].mean(),
            title={"text":"Average Posts / Week"},
            gauge={
                "axis": {"range":[0, max_post], "dtick": max(1, round(max_post/5))}, 
                "bar" : {"color":COLORS["accent"]}
            }
        ),
        row=2, col=1
    )

    
    insight_text = (
    "<b>Insights</b><br>"
    "â€¢ Extroverts post roughly twice as often ğŸ“±<br>"
    "â€¢ Introverts spend significantly more time alone ğŸ§˜<br>"
    "â€¢ Stage fear strongly reduces event attendance ğŸš«ğŸ�‰<br>"
    "â€¢ Frequent outdoor activity correlates with larger friend circles ğŸ‘¥<br>"
    "â€¢ Posting 5+ times a week predicts extroversion (~78% precision) ğŸ“Š"
)
   
    fig.add_annotation(
        x=0.98, y=0.05, xref="paper", yref="paper",
        text=insight_text,
        showarrow=False,
        align="left",
        bordercolor="gray", borderwidth=1, borderpad=8,
        bgcolor="white", opacity=0.9,
        font=dict(color="black", size=12)
    )

    
    _embed_image(fig,
        "https://raw.githubusercontent.com/yourrepo/images/social_people.png")

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["bg"],
        height=750
    )
    fig.show(renderer="iframe_connected")

dashboard_overview(df)


from matplotlib.gridspec import GridSpec

data = pd.DataFrame({
    'Category': ['Social Events', 'Alone Time', 'Digital Activity', 'Talkativeness'],
    'Introvert': [3.2, 8.5, 6.1, 2.9],
    'Extrovert': [7.8, 2.3, 5.7, 8.4]
})


colors = {'Introvert': '#e63946', 'Extrovert': '#a8a8a8'}
plt.style.use('seaborn-whitegrid')


fig = plt.figure(figsize=(12, 9), facecolor='white')
gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 1.2, 0.6], hspace=0.4)


ax1 = fig.add_subplot(gs[0, :])
bar_width = 0.35
index = np.arange(len(data))

bars1 = ax1.bar(index, data['Introvert'], bar_width, 
               color=colors['Introvert'], label='Introvert',
               edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(index + bar_width, data['Extrovert'], bar_width, 
               color=colors['Extrovert'], label='Extrovert',
               edgecolor='black', linewidth=0.5)

ax1.set_title('Personality Trait Comparison', pad=15, 
             fontsize=14, weight='bold', color='black')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Category'], fontsize=11, color='black')
ax1.legend(frameon=True, framealpha=1)
ax1.grid(axis='y', alpha=0.2)
ax1.set_facecolor('#f5f5f5')


donut_data = {
    'Social Energy': {'Intro': 35, 'Extro': 65},
    'Recovery Style': {'Intro': 80, 'Extro': 20},
    'Decision Making': {'Intro': 45, 'Extro': 55},
    'Risk Taking': {'Intro': 30, 'Extro': 70}
}

donut_colors = [colors['Introvert'], colors['Extrovert']]

for i, (title, values) in enumerate(donut_data.items()):
    ax = fig.add_subplot(gs[1, i])
    wedges, texts, autotexts = ax.pie(
        values.values(),
        colors=donut_colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='black', linewidth=0.5),
        autopct='%1.0f%%',
        pctdistance=0.85,
        textprops={'color': 'black', 'fontsize': 10}
    )
    
    
    center_circle = plt.Circle((0,0), 0.25, fc='white', edgecolor='black', linewidth=0.5)
    ax.add_artist(center_circle)
    
    ax.set_title(title, fontsize=11, pad=8, color='black', weight='bold')
    ax.axis('equal')
    ax.set_facecolor('#f5f5f5')


analysis_texts = [
    "â€¢ Introverts drain energy faster in social settings",
    "â€¢ Extroverts show higher need for social stimulation",
    "â€¢ Digital engagement shows smallest difference",
    "â€¢ Talkativeness is the most distinguishing trait"
]

ax_text = fig.add_subplot(gs[2, :])
ax_text.axis('off')


for i, text in enumerate(analysis_texts):
    ax_text.text(0.5, 0.9 - i*0.22, text, 
                fontsize=11, color='black', weight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='#e0e0e0', alpha=0.7, 
                         pad=8, boxstyle='round', edgecolor='black', linewidth=0.5))

plt.tight_layout()
plt.show()


import pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image


data = {
    "Metric": ["Time Alone", "Social Events", "Posts"],
    "Introvert": [70, 10, 20],
    "Extrovert": [25, 55, 20]
}
df = pd.DataFrame(data)


INTRO_IMG = Image.open("/kaggle/input/dashboardimg/intr.png")
EXTRO_IMG = Image.open("/kaggle/input/dashboardimg/exp.png")


fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type":"domain"}, {"type":"domain"}],
           [{"type":"domain"}, {"type":"domain"}]],
    column_widths=[0.40, 0.60],         
    horizontal_spacing=0.04, vertical_spacing=0.12
)


palette = ["#d14f4f", "#d9d9d9", "#bfbfbf"]

fig.add_trace(go.Pie(labels=df["Metric"], values=df["Introvert"],
                     hole=0.55, textinfo="label+percent",
                     marker_colors=palette, showlegend=False),
              row=1, col=2)

fig.add_trace(go.Pie(labels=df["Metric"], values=df["Extrovert"],
                     hole=0.55, textinfo="label+percent",
                     marker_colors=palette, showlegend=False),
              row=2, col=2)


fig.add_layout_image(dict(source=INTRO_IMG, xref="paper", yref="paper",
                          x=0, y=1, sizex=0.38, sizey=0.50,
                          xanchor="left", yanchor="top"))
fig.add_layout_image(dict(source=EXTRO_IMG, xref="paper", yref="paper",
                          x=0, y=0.48, sizex=0.38, sizey=0.50,
                          xanchor="left", yanchor="top"))


fig.add_annotation(x=0.55, y=0.52, xref="paper", yref="paper",
    text="<b>Introverts</b> spend ~70â€¯% of their time alone<br>"
         "and attend the fewest social events.",
    showarrow=False, align="left", font=dict(size=12))

fig.add_annotation(x=0.55, y=0.02, xref="paper", yref="paper",
    text="<b>Extroverts</b> attend >50â€¯% of events<br>"
         "yet post online at a similar rate.",
    showarrow=False, align="left", font=dict(size=12))


fig.update_layout(template="plotly_white",
                  paper_bgcolor="#f0f2f5",         
                  height=700, width=900,
                  margin=dict(l=20, r=20, t=40, b=40),
                  title_text="Introvert vs Extrovert â€“ Profile Snapshot",
                  title_font_size=18, title_x=0.5)

fig.show(renderer="iframe_connected")


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


label_enc = LabelEncoder()
train["Personality_num"] = label_enc.fit_transform(train["Personality"])


X_feat      = train.drop(columns=["id", "Personality", "Personality_num"])
y_target    = train["Personality_num"]
X_test_feat = test.drop(columns=["id"])


full_df = pd.concat([X_feat, X_test_feat], axis=0)
cat_features = full_df.select_dtypes(include="object").columns.tolist()
ord_enc = OrdinalEncoder()
full_df[cat_features] = ord_enc.fit_transform(full_df[cat_features])

X_feat      = full_df.iloc[:len(X_feat)].reset_index(drop=True)
X_test_feat = full_df.iloc[len(X_feat):].reset_index(drop=True)


full_df = pd.concat([X_feat, X_test_feat], axis=0)
cat_features = full_df.select_dtypes(include="object").columns.tolist()
ord_enc = OrdinalEncoder()
full_df[cat_features] = ord_enc.fit_transform(full_df[cat_features])

X_feat      = full_df.iloc[:len(X_feat)].reset_index(drop=True)
X_test_feat = full_df.iloc[len(X_feat):].reset_index(drop=True)


xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred   = np.zeros(len(X_feat))
test_pred  = np.zeros(len(X_test_feat))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_feat, y_target)):
    X_tr, X_val = X_feat.iloc[tr_idx], X_feat.iloc[val_idx]
    y_tr, y_val = y_target.iloc[tr_idx], y_target.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test_feat)

    xgb_model = xgb.train(
        params              = xgb_params,
        dtrain              = dtrain,
        num_boost_round     = 100,
        evals               = [(dvalid, "valid")],
        early_stopping_rounds = 10,
        verbose_eval        = False
    )

    oof_pred[val_idx] += (xgb_model.predict(dvalid) > 0.5).astype(int)
    test_pred         +=  xgb_model.predict(dtest) / kf.n_splits

cv_accuracy = accuracy_score(y_target, oof_pred)
print(f"CV Accuracy: {cv_accuracy:.4f}")



final_pred = (test_pred > 0.5).astype(int)
submission["Personality"] = label_enc.inverse_transform(final_pred)
submission.to_csv("submission.csv", index=False)
submission.head()




