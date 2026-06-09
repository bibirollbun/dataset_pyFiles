# =============================================================================
# CELL 1: IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from IPython.display import display, Image, Markdown, HTML

# Paths
ANALYTICS_DIR = Path("/kaggle/input/nfl-bdb2026-go-birds-analytics-v1/analytics")
GIFS_DIR = ANALYTICS_DIR / "player_gifs"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1a1a2e',
    'accent': '#e94560',
    'gold': '#f4d03f',
    'green': '#27ae60',
    'blue': '#3498db',
    'gray': '#7f8c8d'
}

import base64

def show_image(filename, width=950):
    filepath = ANALYTICS_DIR / filename
    if filepath.exists():
        display(Image(filename=str(filepath), width=width))

def show_gif(filename, width=750):
    """Display GIF by embedding base64 data (works on Kaggle)."""
    filepath = GIFS_DIR / filename
    if filepath.exists():
        # Read GIF and encode as base64 for embedding
        with open(filepath, 'rb') as f:
            gif_data = base64.b64encode(f.read()).decode('utf-8')
        
        display(HTML(f'''
        <div style="text-align: center; margin: 10px 0;">
            <img src="data:image/gif;base64,{gif_data}" width="{width}" 
                 style="border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        </div>
        '''))
    else:
        print(f"GIF not found: {filename}")

# Load data
wr_te = pd.read_csv(ANALYTICS_DIR / "route_running_scores_wr_te.csv")
rb = pd.read_csv(ANALYTICS_DIR / "route_running_scores_rb.csv")
ballhawk = pd.read_csv(ANALYTICS_DIR / "ballhawk_scores.csv")
matchups = pd.read_csv(ANALYTICS_DIR / "separation_matchups.csv")

wr = wr_te[wr_te['position'] == 'WR']
te = wr_te[wr_te['position'] == 'TE']
cb = ballhawk[ballhawk['position'] == 'CB']
fs = ballhawk[ballhawk['position'] == 'FS']
ss = ballhawk[ballhawk['position'] == 'SS']

print("Setup complete. Storytime.")


# =============================================================================
# THE PROBLEM - BALL NOISE
# =============================================================================

show_image("ball_noise_distribution.png")

# Throw quality breakdown visualization
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')

categories = ['Elite\n(â‰¤2 yds)', 'Good\n(2-5 yds)', 'Off\n(5-10 yds)', 'Bad\n(10-15 yds)', 'Wild\n(15+ yds)']
percentages = [75.4, 21.1, 3.0, 0.4, 0.1]
colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

bars = ax.bar(categories, percentages, color=colors, edgecolor='white', linewidth=2)
ax.set_ylabel('Percentage of Throws', fontsize=12, fontweight='bold')
ax.set_title('QB Throw Accuracy Distribution (n=14,108 plays)', fontsize=16, fontweight='bold', pad=20)

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{pct}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylim(0, 85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/kaggle/working/throw_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

display(HTML("""
<div style="background: #fff3cd; padding: 25px; border-radius: 10px; margin: 30px 0; 
            border-left: 5px solid #f39c12;">
    <h4 style="color: #856404; margin: 0 0 10px 0;">âš ï¸� The Hidden Problem</h4>
    <p style="color: #856404; margin: 0; font-size: 16px;">
        <strong>~560 plays</strong> in our dataset have the ball landing more than 5 yards from the receiver.<br>
        Traditional metrics blame the receiver for these plays. But should they?
    </p>
</div>
"""))


# =============================================================================
# THE KEY INSIGHT
# =============================================================================

show_image("execution_stability.png")

# Execution stability data
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

throw_quality = ['Elite\n(0-2 yds)', 'Good\n(2-5 yds)', 'Off\n(5-10 yds)', 'Bad\n(10-15 yds)', 'Wild\n(15+ yds)']
execution = [0.369, 0.322, 0.270, 0.219, 0.281]
samples = [10644, 2966, 426, 58, 14]
colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

bars = ax.bar(throw_quality, execution, color=colors, edgecolor='white', linewidth=2)

ax.axhline(y=0.369, color='#27ae60', linestyle='--', alpha=0.5, linewidth=2)
ax.text(4.5, 0.375, 'Elite baseline', fontsize=10, color='#27ae60', ha='right')

ax.axhline(y=0.281, color='#8e44ad', linestyle='--', alpha=0.5, linewidth=2)
ax.text(4.5, 0.287, '76% of baseline on WILD throws', fontsize=10, color='#8e44ad', ha='right')

ax.set_ylabel('Endpoint Execution Score', fontsize=12, fontweight='bold')
ax.set_title('Player Execution is STABLE Regardless of Throw Quality', fontsize=16, fontweight='bold', pad=20)

for bar, n in zip(bars, samples):
    ax.text(bar.get_x() + bar.get_width()/2., 0.02,
            f'n={n:,}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

ax.set_ylim(0, 0.45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/kaggle/working/execution_stability_viz.png', dpi=150, bbox_inches='tight')
plt.show()

display(HTML("""
<div style="background: #d4edda; padding: 25px; border-radius: 10px; margin: 30px 0;
            border-left: 5px solid #27ae60;">
    <h4 style="color: #155724; margin: 0 0 15px 0;">âœ“ The Verdict</h4>
    <p style="color: #155724; margin: 0; font-size: 18px; line-height: 1.8;">
        <strong>Players maintain 76% of their execution even on WILD throws.</strong><br><br>
        The receiver who runs a perfect route but has a bad QB looks worse than 
        the receiver who runs a mediocre route with an elite QB.<br><br>
        <em>That's not player evaluation. That's quarterback evaluation wearing a receiver's jersey.</em>
    </p>
</div>
"""))


# =============================================================================
# THE BRANDON MARSHALL PHENOMENON
# =============================================================================

display(HTML("""
<div style="background: #1a1a2e; padding: 30px; border-radius: 10px; margin: 20px 0;">
    <p style="color: #f4d03f; font-size: 20px; text-align: center; margin: 0; font-style: italic;">
        "Amazing Receiver + Bad Quarterback = ?"
    </p>
    <p style="color: #a0a0a0; font-size: 14px; text-align: center; margin-top: 15px;">
        Brandon Marshall was one of the most talented receivers of his era.<br>
        His stats were depressed because of inconsistent QB play.<br>
        <strong style="color: #e94560;">Traditional metrics blamed him for his quarterbacks' inaccuracy.</strong>
    </p>
</div>
"""))

show_image("shatter_plot.png")

# Shatter plot data visualization
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')

noise_levels = [0, 2, 5, 10, 15, 20]
ball_model = [1.59, 2.57, 5.18, 9.75, 14.25, 18.43]
noball_model = [3.36, 3.26, 3.10, 3.21, 3.20, 3.18]

ax.fill_between([10, 20], [0, 0], [20, 20], color='#ffcccc', alpha=0.3, label='Danger Zone')

ax.plot(noise_levels, ball_model, 'o-', color='#e74c3c', linewidth=3, markersize=10,
        label='Ball-Dependent Model')
ax.plot(noise_levels, noball_model, 's-', color='#27ae60', linewidth=3, markersize=10,
        label='No-Ball Model')

ax.annotate('SHATTERS\n(1,057% degradation)', xy=(15, 14.25), xytext=(12, 17),
            fontsize=12, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

ax.annotate('STABLE', xy=(15, 3.20), xytext=(17, 5),
            fontsize=12, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

ax.set_xlabel('QB Throw Error (yards from target)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model Prediction Error', fontsize=12, fontweight='bold')
ax.set_title('The Shatter Plot: Model Robustness vs. QB Accuracy', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(-1, 21)
ax.set_ylim(0, 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/kaggle/working/shatter_plot_viz.png', dpi=150, bbox_inches='tight')
plt.show()

display(HTML("""
<div style="display: flex; gap: 20px; margin: 30px 0;">
    <div style="flex: 1; background: #f8d7da; padding: 25px; border-radius: 10px; text-align: center;">
        <h4 style="color: #721c24; margin: 0 0 10px 0;">Ball-Dependent Model</h4>
        <p style="color: #721c24; margin: 0; font-size: 14px;">
            Error goes from <strong>1.59 â†’ 18.43</strong><br>
            <span style="font-size: 24px; font-weight: bold;">1,057% DEGRADATION</span>
        </p>
    </div>
    <div style="flex: 1; background: #d4edda; padding: 25px; border-radius: 10px; text-align: center;">
        <h4 style="color: #155724; margin: 0 0 10px 0;">No-Ball Model</h4>
        <p style="color: #155724; margin: 0; font-size: 14px;">
            Error stays at <strong>~3.2</strong><br>
            <span style="font-size: 24px; font-weight: bold;">ROCK SOLID</span>
        </p>
    </div>
</div>
"""))


# =============================================================================
# THE SOLUTION
# =============================================================================

display(HTML("""
<h2 style="color: #1a1a2e; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 50px;">
    The Solution
</h2>
<h3 style="color: #7f8c8d; font-weight: 400;">Ball-Free Player Metrics</h3>
"""))

display(HTML("""
<div style="background: #e8f4f8; padding: 30px; border-radius: 10px; margin: 20px 0;">
    <h4 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">
        Our Metrics Use <span style="color: #e94560;">ZERO</span> Ball Information
    </h4>
    
    <div style="display: flex; gap: 30px;">
        <div style="flex: 1;">
            <h5 style="color: #e94560; margin: 0 0 15px 0; border-bottom: 2px solid #e94560; padding-bottom: 5px;">
                Route Running Score (Receivers)
            </h5>
            <table style="width: 100%; font-size: 14px;">
                <tr><td><strong>Separation</strong></td><td style="text-align: right;">40%</td></tr>
                <tr><td><strong>Endpoint Execution</strong></td><td style="text-align: right;">35%</td></tr>
                <tr><td><strong>Route Efficiency</strong></td><td style="text-align: right;">15%</td></tr>
                <tr><td><strong>Consistency</strong></td><td style="text-align: right;">10%</td></tr>
            </table>
        </div>
        <div style="flex: 1;">
            <h5 style="color: #3498db; margin: 0 0 15px 0; border-bottom: 2px solid #3498db; padding-bottom: 5px;">
                Ballhawk Score (Defenders)
            </h5>
            <table style="width: 100%; font-size: 14px;">
                <tr><td><strong>Coverage Tightness</strong></td><td style="text-align: right;">35%</td></tr>
                <tr><td><strong>Pursuit Angle</strong></td><td style="text-align: right;">30%</td></tr>
                <tr><td><strong>Position Discipline</strong></td><td style="text-align: right;">20%</td></tr>
                <tr><td><strong>Consistency</strong></td><td style="text-align: right;">15%</td></tr>
            </table>
        </div>
    </div>
</div>
"""))


# =============================================================================
# WIDE RECEIVER RANKINGS
# =============================================================================

display(HTML("""
<h2 style="color: #e94560; border-bottom: 3px solid #e94560; padding-bottom: 10px; margin-top: 50px;">
    Results: Wide Receivers
</h2>
"""))

show_image("wr_route_running_top30.png")

# Top 10 WR table
display(HTML("<h4 style='color: #1a1a2e; margin-top: 30px;'>Top 10 Wide Receivers</h4>"))

top_wr = wr.head(10)[['player_name', 'ROUTE_RUNNING_SCORE', 'grade', 'avg_separation', 'avg_execution', 'num_plays']]
top_wr.columns = ['Player', 'Score', 'Grade', 'Avg Sep', 'Execution', 'Plays']

# Create styled table
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
table = ax.table(cellText=[[i+1] + list(row) for i, row in enumerate(top_wr.values)],
                 colLabels=['Rank', 'Player', 'Score', 'Grade', 'Avg Sep', 'Execution', 'Plays'],
                 cellLoc='center', loc='center',
                 colColours=['#e94560']*7)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold', color='white')
    cell.set_edgecolor('#ddd')
    
plt.tight_layout()
plt.savefig('/kaggle/working/top_wr_table.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# =============================================================================
# TIGHT END RANKINGS
# =============================================================================

display(HTML("""
<h2 style="color: #f4d03f; border-bottom: 3px solid #f4d03f; padding-bottom: 10px; margin-top: 50px;">
    Results: Tight Ends
</h2>
"""))

show_image("te_route_running_top30.png")

# Top TE insight
display(HTML(f"""
<div style="background: #1a1a2e; padding: 25px; border-radius: 10px; margin: 30px 0;">
    <h4 style="color: #f4d03f; margin: 0 0 15px 0;">ğŸ�† Top Tight End</h4>
    <p style="color: white; font-size: 24px; margin: 0;">
        <strong>{te.iloc[0]['player_name']}</strong> â€” Score: {te.iloc[0]['ROUTE_RUNNING_SCORE']:.1f}
    </p>
    <p style="color: #a0a0a0; font-size: 14px; margin-top: 10px;">
        Avg Separation: {te.iloc[0]['avg_separation']:.1f} yds | 
        Execution: {te.iloc[0]['avg_execution']:.3f} |
        Plays: {int(te.iloc[0]['num_plays'])}
    </p>
</div>
"""))


# =============================================================================
# CORNERBACK RANKINGS
# =============================================================================

display(HTML("""
<h2 style="color: #3498db; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 50px;">
    Results: Cornerbacks
</h2>
"""))

show_image("cb_ballhawk_top30.png")

display(HTML(f"""
<div style="background: #16213e; padding: 25px; border-radius: 10px; margin: 30px 0;">
    <h4 style="color: #3498db; margin: 0 0 15px 0;">ğŸ›¡ï¸� Top Cornerback</h4>
    <p style="color: white; font-size: 24px; margin: 0;">
        <strong>{cb.iloc[0]['player_name']}</strong> â€” Score: {cb.iloc[0]['BALLHAWK_SCORE']:.1f}
    </p>
    <p style="color: #a0a0a0; font-size: 14px; margin-top: 10px;">
        Avg Separation Allowed: {cb.iloc[0]['avg_separation_allowed']:.1f} yds | 
        Closing Speed: {cb.iloc[0]['avg_closing_speed']:.2f} |
        Plays: {int(cb.iloc[0]['num_plays'])}
    </p>
</div>
"""))


# =============================================================================
# SAFETY RANKINGS
# =============================================================================

display(HTML("""
<h2 style="color: #27ae60; border-bottom: 3px solid #27ae60; padding-bottom: 10px; margin-top: 50px;">
    Results: Safeties
</h2>
"""))

show_image("fs_ballhawk_top25.png")
show_image("ss_ballhawk_top25.png")


# =============================================================================
# ELITE PLAYERS IN ACTION
# =============================================================================

display(HTML("""
<h2 style="color: #1a1a2e; border-bottom: 3px solid #f4d03f; padding-bottom: 10px; margin-top: 50px;">
    See It In Action: Elite Route Runners
</h2>
<p style="color: #7f8c8d; font-size: 16px;">
    Watch how our top-rated players execute their routes. These animations show player tracking dataâ€”no ball needed.
</p>
"""))

gifs_to_show = [
    ("Drew_Sample_TE_route.gif", "Drew Sample", "TE", 90.0),
    ("Kadarius_Toney_WR_route.gif", "Kadarius Toney", "WR", 80.8),
    ("Cade_Otton_TE_route.gif", "Cade Otton", "TE", 84.1),
]

for gif_file, name, pos, score in gifs_to_show:
    display(HTML(f"""
    <div style="background: #f8f9fa; padding: 15px 20px; border-radius: 10px; margin: 20px 0 10px 0;
                display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 18px; font-weight: bold; color: #1a1a2e;">{name} ({pos})</span>
        <span style="font-size: 16px; color: #27ae60; font-weight: bold;">RRS: {score}</span>
    </div>
    """))
    show_gif(gif_file)


# =============================================================================
# ELITE DEFENDERS IN ACTION
# =============================================================================

display(HTML("""
<h2 style="color: #1a1a2e; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 50px;">
    See It In Action: Elite Coverage
</h2>
"""))

defender_gifs = [
    ("AJ_Terrell_CB_coverage.gif", "A.J. Terrell", "CB", 83.6),
    ("Sauce_Gardner_CB_coverage.gif", "Sauce Gardner", "CB", 76.8),
    ("Juanyeh_Thomas_FS_coverage.gif", "Juanyeh Thomas", "FS", 86.4),
]

for gif_file, name, pos, score in defender_gifs:
    display(HTML(f"""
    <div style="background: #f8f9fa; padding: 15px 20px; border-radius: 10px; margin: 20px 0 10px 0;
                display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 18px; font-weight: bold; color: #1a1a2e;">{name} ({pos})</span>
        <span style="font-size: 16px; color: #3498db; font-weight: bold;">BHS: {score}</span>
    </div>
    """))
    show_gif(gif_file)


# =============================================================================
# ELITE MATCHUPS
# =============================================================================


display(HTML("""
<h2 style="color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; margin-top: 50px;">
    Elite Matchups: When The Best Face The Best
</h2>
<p style="color: #7f8c8d; font-size: 16px;">
    These plays feature our top-rated receivers against our top-rated defenders.<br>
    <strong>Notice: The ball location is often irrelevant to the quality of the battle.</strong>
</p>
"""))

matchup_gifs = [
    ("MATCHUP_Cade_Otton_vs_AJ_Terrell.gif", "Cade Otton", "A.J. Terrell"),
    ("MATCHUP_Josh_Oliver_vs_AJ_Terrell.gif", "Josh Oliver", "A.J. Terrell"),
    ("MATCHUP_Drew_Sample_vs_Jaylen_Watso.gif", "Drew Sample", "Jaylen Watson"),
]

for gif_file, rec, defender in matchup_gifs:
    display(HTML(f"""
    <div style="background: linear-gradient(90deg, #e94560 0%, #1a1a2e 50%, #3498db 100%); 
                padding: 15px 20px; border-radius: 10px; margin: 20px 0 10px 0;
                display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 18px; font-weight: bold; color: white;">{rec}</span>
        <span style="font-size: 14px; color: #f4d03f; font-weight: bold;">VS</span>
        <span style="font-size: 18px; font-weight: bold; color: white;">{defender}</span>
    </div>
    """))
    show_gif(gif_file)


# =============================================================================
# CONCLUSION
# =============================================================================

display(HTML("""
<h2 style="color: #7f8c8d; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 50px;">
    Technical Appendix
</h2>
"""))

display(HTML("""
<div style="font-family: monospace; background: #f8f9fa; padding: 25px; border-radius: 10px; font-size: 13px;">
<pre style="margin: 0;">
MODEL COMPARISON
================
Ball Model (GNN):     0.577 RMSE  [uses ball_land_x/y]
No-Ball Model:        0.833 RMSE  [player kinematics only]
Gap:                  0.256 RMSE  â†’ This is QB noise

ROUTE RUNNING SCORE (RRS)
=========================
RRS = 0.40 Ã— Separation_pct
    + 0.35 Ã— Execution_pct
    + 0.15 Ã— Efficiency_pct  
    + 0.10 Ã— Consistency_pct

BALLHAWK SCORE (BHS)
====================
BHS = 0.35 Ã— Coverage_pct
    + 0.30 Ã— Pursuit_pct
    + 0.20 Ã— Discipline_pct
    + 0.15 Ã— Consistency_pct

GRADING (Percentile-Based)
==========================
A: Top 15%    |    B: 15-40%
C: 40-65%     |    D: 65-85%
F: Bottom 15%
</pre>
</div>
"""))

display(HTML("""
<div style="text-align: center; margin: 50px 0; padding: 30px; background: #f8f9fa; border-radius: 10px;">
    <p style="color: #7f8c8d; font-size: 14px; margin: 0;">
        NFL Big Data Bowl 2026 - Analytics Track - Ryan Adams UPenn M.S.Ed. Learning Analytics and Artificial Intelligence '27'<br>
        <strong style="color: #1a1a2e;">GO BIRDS</strong>
    </p>
</div>
"""))

