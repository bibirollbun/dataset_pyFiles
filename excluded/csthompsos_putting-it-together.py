import pandas as pd
import numpy as np
import re
import os
from scipy.optimize import minimize
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from scipy.spatial import ConvexHull
import plotly.express as px




folder = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
supp = pd.read_csv(folder + 'supplementary_data.csv', low_memory=False)

# combine the tracking data
input_dfs = []
output_dfs = []
pattern = re.compile(r"(input|output)_(\d{4})_w(\d+)")# Regex pattern: matches input_2023_w17.csv or output_2023_w05.csv
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".csv") and (filename.startswith("input_") or filename.startswith("output_")):
            file_path = os.path.join(dirname, filename)

            # Extract type, year, week
            match = pattern.search(filename)
            if not match:
                continue
            ftype, year, week = match.groups()

            # Read CSV
            df = pd.read_csv(file_path)

            # Add metadata
            df["week"] = int(week)

            # Store in correct list
            if ftype == "input":
                input_dfs.append(df)
            else:
                output_dfs.append(df)

# Combine separately
input_df = pd.concat(input_dfs, ignore_index=True).sort_values(by=["week"]).reset_index(drop=True)
output_df = pd.concat(output_dfs, ignore_index=True).sort_values(by=["week"]).reset_index(drop=True)
# index the tracking data for efficient subsetting
input_df = input_df.set_index(['game_id', 'play_id']).sort_index()
output_df = output_df.set_index(['game_id', 'play_id']).sort_index()


df2 = supp[supp['route_of_targeted_receiver'] == 'OUT']

# Reset indexes so game_id/play_id are columns
out_reset  = output_df.reset_index()
in_reset   = input_df.reset_index()

# Ensure a 'dir' column exists on input side (fallback to play_direction)
if 'dir' not in in_reset.columns and 'play_direction' in in_reset.columns:
    in_reset = in_reset.assign(dir=in_reset['play_direction'])

# --- 1) take last frame per play from output_df ---
last_frame = (
    out_reset.groupby(['game_id','play_id'], as_index=False)['frame_id'].max()
             .rename(columns={'frame_id':'last_post_frame'})
)
last_frame_df = (
    out_reset.merge(last_frame, on=['game_id','play_id'])
             .query('frame_id == last_post_frame')
)

# --- 2) bring in player_role & player_height from input_df (by play + nfl_id) ---
role_height = in_reset[['game_id','play_id','nfl_id','player_role','player_height']].drop_duplicates()
last_frame_df = last_frame_df.merge(
    role_height,
    on=['game_id','play_id','nfl_id'],
    how='left'
)

# --- 3) get dir from the SAME play & frame (from input_df) ---
dir_at_frame = in_reset[['game_id','play_id','frame_id','nfl_id','dir']].drop_duplicates()
last_frame_df = last_frame_df.merge(
    dir_at_frame,
    on=['game_id','play_id','frame_id','nfl_id'],
    how='left',
    suffixes=('', '_from_input')
)

# --- 4) keep only roles we care about ---
roles = ['Targeted Receiver','Defensive Coverage']
filtered = last_frame_df[last_frame_df['player_role'].isin(roles)].copy()

# Index defenders within each play
filtered['role_idx'] = (
    filtered.groupby(['game_id','play_id','player_role']).cumcount() + 1
)

# --- 5) pivot to wide: receiver_* and defender_1_*, defender_2_*, ... ---
wide = (
    filtered
    .pivot_table(
        index=['game_id','play_id'],
        columns=['player_role','role_idx'],
        values=['x','y','dir','player_height'],
        aggfunc='first'
    )
)

# Flatten column names: (x, Targeted Receiver, 1) -> receiver_x
wide.columns = [
    f"{'receiver' if role=='Targeted Receiver' else 'defender'}"
    f"{'' if role=='Targeted Receiver' else f'_{idx}'}_{val}"
    for (val, role, idx) in wide.columns
]
wide = wide.reset_index()

# --- 6) merge with df2, keeping only pass_result ---
df_final = (
    df2[['game_id','play_id','pass_result']]
    .merge(wide, how='left', on=['game_id','play_id'])
)

# 1) Split receiver and defender rows from last_frame_df
rec = (
    last_frame_df[last_frame_df['player_role'] == 'Targeted Receiver']
    [['game_id','play_id','nfl_id','x','y','dir','player_height']]
    .rename(columns={
        'nfl_id':'receiver_id', 'x':'receiver_x', 'y':'receiver_y',
        'dir':'receiver_dir', 'player_height':'receiver_height'
    })
)

defn = (
    last_frame_df[last_frame_df['player_role'] == 'Defensive Coverage']
    [['game_id','play_id','nfl_id','x','y','dir','player_height']]
    .rename(columns={
        'nfl_id':'defender_id', 'x':'defender_x', 'y':'defender_y',
        'dir':'defender_dir', 'player_height':'defender_height'
    })
)

# 2) Pair each receiver with all defenders in the same play
pairs = rec.merge(defn, on=['game_id','play_id'], how='inner')

# 3) Compute distance and keep the nearest defender per play
pairs['defender_distance'] = np.hypot(
    pairs['receiver_x'] - pairs['defender_x'],
    pairs['receiver_y'] - pairs['defender_y']
)

nearest_def = (
    pairs.sort_values(['game_id','play_id','defender_distance'])
         .groupby(['game_id','play_id'], as_index=False)
         .first()
)

# 4) (Optional) Merge with df2 to keep pass_result
closest_join = (
    df2[['game_id','play_id','pass_result']]
    .drop_duplicates()
    .merge(nearest_def, on=['game_id','play_id'], how='left')
)
def height_to_inches(h):
    if pd.isna(h):
        return np.nan
    match = re.match(r'(\d+)[-\' ]?(\d+)?', str(h))
    if not match:
        return np.nan
    feet = int(match.group(1))
    inches = int(match.group(2)) if match.group(2) else 0
    return feet * 12 + inches

# get inches
closest_join['receiver_height'] = closest_join['receiver_height'].apply(height_to_inches)
closest_join['defender_height'] = closest_join['defender_height'].apply(height_to_inches)

# drop nas
closest_join = closest_join.dropna(subset=['receiver_height', 'defender_height'])

# get the play direction
directions = []
for i,row in closest_join.iterrows():
    game_id = row['game_id']
    play_id = row['play_id']
    directions.append(input_df.loc[(game_id,play_id)].iloc[0]['play_direction'])
closest_join['play_direction'] = directions
closest_join.head()

### Standardize x and y
# Mask for plays going left
closest_join_copy = closest_join.copy()
mask_left = closest_join_copy['play_direction'] == 'left'

# Flip coordinates only for left plays
closest_join_copy.loc[mask_left, 'receiver_x'] = 120 - closest_join_copy.loc[mask_left, 'receiver_x']
closest_join_copy.loc[mask_left, 'receiver_y'] = 53.3 - closest_join_copy.loc[mask_left, 'receiver_y']
closest_join_copy.loc[mask_left, 'defender_x'] = 120 - closest_join_copy.loc[mask_left, 'defender_x']
closest_join_copy.loc[mask_left, 'defender_y'] = 53.3 - closest_join_copy.loc[mask_left, 'defender_y']
closest_join = closest_join_copy.copy()

# make all the outs go to the low y side of the field
closest_join_copy = closest_join.copy()
mask = closest_join_copy['receiver_y'] > 25
# Flip coordinates only for left plays
closest_join_copy.loc[mask, 'receiver_y'] = 53.3 - closest_join_copy.loc[mask, 'receiver_y']
closest_join_copy.loc[mask, 'defender_y'] = 53.3 - closest_join_copy.loc[mask, 'defender_y']
closest_join = closest_join_copy.copy()


# Make the final model df
model_df = closest_join.dropna(subset=[
    'receiver_x','receiver_y','defender_x','defender_y',
    'receiver_height','defender_height','pass_result'
]).copy()

#get height and x/y coordinate differences
model_df['x_diff'] = model_df['defender_x'] - model_df['receiver_x']
model_df['y_diff'] = model_df['defender_y'] - model_df['receiver_y']
model_df['height_diff'] = model_df['defender_height'] - model_df['receiver_height']

# create binary outcomes
model_df['pass_result_is_I']  = (model_df['pass_result'] == 'I').astype(int)
model_df['pass_result_is_C']  = (model_df['pass_result'] == 'C').astype(int)
model_df['pass_result_is_IN'] = (model_df['pass_result'] == 'IN').astype(int)


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # ensure correct shape and type
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # gather probabilities of the true classes
        idx = tf.stack([tf.range(tf.shape(y_pred)[0], dtype=tf.int32), y_true], axis=1)
        pt = tf.gather_nd(y_pred, idx)

        # focal loss formula
        loss_val = -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)

    return loss

# --- Define predictors and multiclass target ---
model_df_altered = model_df[~(((model_df['pass_result_is_I']==1) | (model_df['pass_result_is_IN']==1)) & ((np.abs(model_df['x_diff'])>3) | (np.abs(model_df['y_diff'])>3)))]
X = model_df_altered[['x_diff', 'y_diff']]

# Combine your one-hot columns into one label column
# assumes exactly one of these is 1 per row
y = (
    model_df_altered[['pass_result_is_I', 'pass_result_is_C', 'pass_result_is_IN']]
    .idxmax(axis=1)
    .str.replace('pass_result_is_', '')
)

# --- Encode labels as integers ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- Build a simple neural network ---
# nn = models.Sequential([
#     layers.Input(shape=(X.shape[1],)),
#     layers.Dense(128, activation='sigmoid', kernel_initializer='lecun_normal'),
#     layers.Dense(128, activation='sigmoid', kernel_initializer='lecun_normal'),
#     layers.Dense(128, activation='sigmoid', kernel_initializer='lecun_normal'),
#     layers.Dense(len(le.classes_), activation='softmax')
# ])
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='softplus',
                 kernel_initializer='glorot_uniform'),
    layers.Dense(128, activation='softplus',
                 kernel_initializer='glorot_uniform'),
    layers.Dense(128, activation='softplus',
                 kernel_initializer='glorot_uniform'),
    layers.Dense(len(le.classes_), activation='softmax')
])

# compute weights for each class
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)

class_weights = dict(enumerate(class_weights))
# print(class_weights)

# nn.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
model.compile(
    optimizer='adam',
    loss=sparse_categorical_focal_loss(gamma=3.0, alpha=50),
    metrics=['accuracy']
)

# --- Fit on ALL data ---
# nn.fit(X, y_enc, epochs=30, batch_size=32, verbose=0)
model.fit(X, y_enc, epochs=30, batch_size=32, class_weight=class_weights,verbose=0)

# --- Predict on ALL data (or any X you want later) ---
y_pred = model.predict(X).argmax(axis=1)

# --- Optional: classification report ---
# from sklearn.metrics import classification_report
# print(classification_report(y_enc, y_pred, target_names=le.classes_))


from matplotlib.patches import Rectangle

# --- Original observed ranges ---
x0_min, x0_max = model_df["x_diff"].min(), model_df["x_diff"].max()
y0_min, y0_max = model_df["y_diff"].min(), model_df["y_diff"].max()

# --- Extended grid ranges ---
x_pad = 0
y_pad = 0

x_min, x_max = x0_min - x_pad, x0_max + x_pad
y_min, y_max = y0_min - y_pad, y0_max + y_pad

x_range = np.arange(x_min, x_max + 0.1, 0.1)
y_range = np.arange(y_min, y_max + 0.1, 0.1)

X1, X2 = np.meshgrid(x_range, y_range)

# --- Create grid dataframe for predictions ---
grid = pd.DataFrame({
    "x_diff": X1.ravel(),
    "y_diff": X2.ravel()
})

# --- Predict probabilities ---
probs = model.predict(grid)
classes = le.classes_

# Add predictions to grid
for i, cls in enumerate(classes):
    grid[f"pred_{cls}"] = probs[:, i]

# --- Pivot predictions to heatmap matrices ---
Z_dict = {cls: grid.pivot(index="y_diff", columns="x_diff", values=f"pred_{cls}") 
          for cls in classes}

# --- Plot heatmaps ---
fig, axes = plt.subplots(1, len(classes), figsize=(18, 5), sharex=True, sharey=True)
heatmaps = [(cls, Z_dict[cls]) for cls in classes]

for ax, (label, Z) in zip(axes, heatmaps):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],  # extended grid
        aspect="auto",
        cmap="viridis"
    )

    # Mark zero lines
    ax.axhline(0, color="white", linestyle="--", linewidth=1)
    ax.axvline(0, color="white", linestyle="--", linewidth=1)

    # Labels and title
    if label == 'C': label = 'Completion'
    elif label == 'I': label = 'Incompletion'
    elif label == 'IN': label = 'Interception'
    ax.set_title(f"Predicted Probability: {label}")
    ax.set_xlabel("x_diff")
    ax.set_ylabel("y_diff")

    # Independent colorbar
    fig.colorbar(im, ax=ax, label="Predicted Probability")

plt.tight_layout()
plt.show()


def get_model_probs(model,X):
    probs = model.predict(X)
    
    #For multiclass, predict_proba returns a list of arrays (one per class)
    #RandomForestClassifier handles this internally: shape = (n_samples, n_classes)
    probs_df = pd.DataFrame(probs, columns=classes, index=X.index)
    
    return probs_df
get_model_probs(model,X)


def get_data_for_play(game_id,play_id,output_df = output_df,input_df = input_df):
    # get the data    
    play_output_data = output_df.loc[(game_id,play_id)]
    play_input_data = input_df.loc[(game_id,play_id)]
    play_output_data = pd.merge(play_output_data,
             play_input_data[['nfl_id','player_side']].drop_duplicates(),
             how = 'left',
             on = 'nfl_id')

    # process the data
    last_frame = play_output_data['frame_id'].max()
    last_frame_data = play_output_data[play_output_data['frame_id'] == last_frame]
    offense_loc = last_frame_data[last_frame_data['player_side'] == 'Offense'].copy()
    defense_loc = last_frame_data[last_frame_data['player_side'] == 'Defense'].copy()

    # check whether there are even any defenders
    if len(defense_loc) == 0:
        print(f'No defender in {game_id}, {play_id}!')
        return
    
    defense_loc['distance_to_offense'] = defense_loc.apply(
        lambda row: np.sqrt((row['x'] - offense_loc['x'])**2 + (row['y'] - offense_loc['y'])**2),
        axis=1
    )
    
    closest_defender_at_ball_arrive = defense_loc.iloc[defense_loc['distance_to_offense'].argmin()]['nfl_id']

    defender_at_ball_throw = play_input_data[play_input_data['nfl_id'] == closest_defender_at_ball_arrive]
    defender_at_ball_throw = defender_at_ball_throw.iloc[defender_at_ball_throw['frame_id'].argmax()]

    # switch to make it go right
    if defender_at_ball_throw['play_direction'] == 'left':
        defender_at_ball_throw['x'] = 120 - defender_at_ball_throw['x']
        defender_at_ball_throw['y'] = 53.3 - defender_at_ball_throw['y']
        defender_at_ball_throw['dir'] = (defender_at_ball_throw['dir'] + 180) % 360 # rotate the angle
        
        offense_loc['x'] = 120 - offense_loc['x']
        offense_loc['y'] = 53.3 - offense_loc['y']

    # switch to make it go to the bottom of the field
    if defender_at_ball_throw['y'] > 25:
        defender_at_ball_throw['y'] = 53.3 - defender_at_ball_throw['y']
        defender_at_ball_throw['dir'] = (180 - defender_at_ball_throw['dir']) % 360 # reflect the angle
        
        offense_loc['y'] = 53.3 - offense_loc['y']

    return defender_at_ball_throw, offense_loc, last_frame



def optimize_defender_trajectory(defender_at_ball_throw, final_pos, 
                                 k_d=3.84176449, k_a=2.26192453, 
                                 v_max=8.03820468, F_max=1718.26426701):
    """
    Solve for t_1, t_2, t_3, and ω (angular velocity) given defender data and target position.

    Parameters
    ----------
    defender_at_ball_throw : dict or Series
        Must include keys ['player_weight', 's', 'dir', 'x', 'y'].
    final_pos : tuple
        (x_1, y_1) final target coordinates.
    k_d : float
        Deceleration rate (default=2).
    k_a : float
        Acceleration rate (default=2).
    v_max : float
        Maximum speed (m/s, default=9).
    F_max : float
        Maximum centripetal force (default=400 * π).

    Returns
    -------
    dict
        Optimization results including t_1, t_2, t_3, ω and the full scipy result.
    """

    # --- Extract initial values ---
    m = defender_at_ball_throw['player_weight']
    s_0 = defender_at_ball_throw['s']
    theta_0 = np.deg2rad(90 - defender_at_ball_throw['dir'])
    x_0, y_0 = defender_at_ball_throw['x'], defender_at_ball_throw['y']
    x_1, y_1 = final_pos

    # --- Piecewise s(t) ---
    def s_(t, t_1, t_2, t_3):
        s_t1 = s_0 * np.exp(-k_d * t_1)
        if t <= t_1:
            return s_0 * np.exp(-k_d * t)
        elif t <= t_2:
            return s_t1
        elif t <= t_3:
            return s_t1 + (v_max - s_t1) * (1 - np.exp(-k_a * (t - t_2)))
        else:
            return v_max

    # --- Piecewise θ(t) ---
    def theta_(t, t_1, t_2, t_3, omega):
        if t <= t_1:
            return theta_0
        elif t <= t_2:
            return theta_0 + omega * (t - t_1)
        elif t <= t_3:
            return theta_0 + omega * (t_2 - t_1)
        else:
            return theta_0 + omega * (t_2 - t_1)
    
    # --- Equality constraints (Eqs. 3–4) ---
    def constraint_eq(vars):
        t_1, t_2, t_3, omega = vars
        s_t1 = s_(t_1, t_1, t_2, t_3)
        theta_t2 = theta_(t_2,t_1,t_2,t_3,omega)
    
        eps = 1e-8
        if abs(omega) < eps:
            term2_x = s_t1 * (t_2 - t_1) * np.cos(theta_0)
            term2_y = s_t1 * (t_2 - t_1) * np.sin(theta_0)
        else:
            term2_x = (s_t1/omega) * (np.sin(theta_0 + omega*(t_2-t_1)) - np.sin(theta_0))
            term2_y = - (s_t1/omega) * (np.cos(theta_0 + omega*(t_2-t_1)) - np.cos(theta_0))
    
        term1_x = s_0*np.cos(theta_0)/k_d * (1 - np.exp(-k_d * t_1))
        term1_y = s_0*np.sin(theta_0)/k_d * (1 - np.exp(-k_d * t_1))
    
        bracket = ((s_t1 - v_max)/k_a) * (1 - np.exp(-k_a * (t_3 - t_2))) + v_max * (t_3 - t_2)
        term3_x = np.cos(theta_t2) * bracket
        term3_y = np.sin(theta_t2) * bracket
    
        x_final = term1_x + term2_x + term3_x
        y_final = term1_y + term2_y + term3_y
    
        return [
            x_final - (x_1 - x_0),
            y_final - (y_1 - y_0)
        ]


    # --- Inequality constraints (Eqs. 5–7) ---
    def constraint_ineq(vars):
        t_1, t_2, t_3, omega = vars
        s_t1 = s_(t_1, t_1, t_2, t_3)
        return [
            2 * np.pi - abs(omega * (t_2 - t_1)),  # -2π ≤ ω(t₂−t₁) ≤ 2π
            t_1,                                   # t₁ > 0
            t_2 - t_1,                             # t₂ > t₁
            t_3 - t_2,                             # t₃ > t₂
            (F_max / m) - abs(s_t1 * omega)        # −F_max/m ≤ s(t₁)ω ≤ F_max/m
        ]

    # --- Objective function (minimize t₃) ---
    def objective(vars):
        _, _, t_3, _ = vars
        return t_3

    # --- Build constraints for scipy ---
    constraints = [
        {'type': 'eq', 'fun': lambda v: constraint_eq(v)[0]},
        {'type': 'eq', 'fun': lambda v: constraint_eq(v)[1]},
        {'type': 'ineq', 'fun': lambda v: constraint_ineq(v)[0]},
        {'type': 'ineq', 'fun': lambda v: constraint_ineq(v)[1]},
        {'type': 'ineq', 'fun': lambda v: constraint_ineq(v)[2]},
        {'type': 'ineq', 'fun': lambda v: constraint_ineq(v)[3]},
        {'type': 'ineq', 'fun': lambda v: constraint_ineq(v)[4]},
    ]

    # --- Solve optimization ---
    dx = x_1 - x_0
    dy = y_1 - y_0
    dist = np.hypot(dx, dy)
    avg_speed = max(0.1, min(v_max, s_0))
    t3_guess = dist / avg_speed
    t1_guess = 0.1 * t3_guess
    t2_guess = 0.6 * t3_guess
    
    target_angle = np.arctan2(dy, dx)
    desired_rotation = ((target_angle - theta_0 + np.pi) % (2*np.pi)) - np.pi
    omega_guess = desired_rotation / max(1e-3, t2_guess - t1_guess)
    initial_guess = [t1_guess, t2_guess, t3_guess, omega_guess]

    result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

    # --- Package results ---
    return {
        'result': result,
        'start_pos': (x_0, y_0),
        'end_pos': (x_1, y_1),
        'optimized_t1': result.x[0] if result.success else None,
        'optimized_t2': result.x[1] if result.success else None,
        'optimized_t3': result.x[2] if result.success else None,
        'optimized_omega': result.x[3] if result.success else None
    }


def get_grid_of_locations(offense_loc, distance=10, density=1):
    """
    Generate a grid of (x, y) points within a given Manhattan distance
    from an origin point, with adjustable density.

    Parameters:
    - offense_loc: dict or Series with 'x' and 'y' keys
    - distance: maximum Manhattan distance (default=10)
    - density: number of points per unit distance (default=1)
               e.g. density=2 → 0.5 spacing, density=4 → 0.25 spacing

    Returns:
    - DataFrame of (x, y) points within the Manhattan distance
    """
    x0, y0 = offense_loc['x'].values[0], offense_loc['y'].values[0]
    step = 1 / density  # spacing between grid points

    # Generate possible deltas
    deltas = np.arange(-distance, distance + step, step)
    points = [
        (x0 + dx, y0 + dy)
        for dx in deltas
        for dy in deltas
        if abs(dx) + abs(dy) <= distance
    ]

    return pd.DataFrame(points, columns=['x', 'y'])


game_id = 2023102206
play_id = 524
defender_at_ball_throw,offense_loc,frames_in_air = get_data_for_play(game_id,play_id)
# display(defender_at_ball_throw)
# display(offense_loc)
print('Total frames in air:', frames_in_air)

grid = get_grid_of_locations(offense_loc,density=2)
times = []
for i, row in tqdm(grid.iterrows(), total=len(grid), desc="Getting Times"):
    x = row['x']
    y = row['y']
    results = optimize_defender_trajectory(defender_at_ball_throw, (x, y))
    times.append(results['optimized_t3'])
grid['time_to_reach'] = times
grid


game_id = 2023102206
play_id = 524
defender_at_ball_throw,offense_loc,frames_in_air = get_data_for_play(game_id,play_id)
# display(defender_at_ball_throw)
# display(offense_loc)
print('Total frames in air:', frames_in_air)

grid = get_grid_of_locations(offense_loc,density=2)
times = []
for i, row in tqdm(grid.iterrows(), total=len(grid), desc="Getting Times"):
    x = row['x']
    y = row['y']
    results = optimize_defender_trajectory(defender_at_ball_throw, (x, y))
    times.append(results['optimized_t3'])
grid['time_to_reach'] = times
grid = grid[grid['time_to_reach'] <= frames_in_air/10].reset_index(drop=True)
grid


relevant_data = model_df[(model_df['game_id'] == game_id) & (model_df['play_id'] == play_id)]
height_diff = relevant_data['height_diff'].values[0]

offense_x = relevant_data['receiver_x'].values[0]
offense_y = relevant_data['receiver_y'].values[0]

x_diffs = []
y_diffs = []
for i,row in grid.iterrows():
    x_diffs.append(row['x'] - offense_x)
    y_diffs.append(row['y'] - offense_y)

x_diffs.append(relevant_data['x_diff'].values[0])
y_diffs.append(relevant_data['y_diff'].values[0])

grid_data =  pd.DataFrame({'x_diff':x_diffs,
              'y_diff':y_diffs
             })

    
probs = get_model_probs(model,grid_data)

xs = list(grid['x'])
xs.append(relevant_data['defender_x'].values[0])
ys = list(grid['y'])
ys.append(relevant_data['defender_y'].values[0])
grid_data['x'] = xs
grid_data['y'] = ys
grid_data['prob_C'] = probs['C']
grid_data['prob_I'] = probs['I']
grid_data['prob_IN'] = probs['IN']
grid_data



def get_grid_data(game_id,play_id,density = 2,distance = 10):
    defender_at_ball_throw,offense_loc,frames_in_air = get_data_for_play(game_id,play_id)
    # display(defender_at_ball_throw)
    # display(offense_loc)
    print('Total frames in air:', frames_in_air)
    
    grid = get_grid_of_locations(offense_loc,density=density,distance=distance)
    times = []
    for i, row in tqdm(grid.iterrows(), total=len(grid), desc="Getting Times"):
        x = row['x']
        y = row['y']
        results = optimize_defender_trajectory(defender_at_ball_throw, (x, y))
        times.append(results['optimized_t3'])
    grid['time_to_reach'] = times
    grid = grid[grid['time_to_reach'] <= frames_in_air/10].reset_index(drop=True)
    relevant_data = model_df[(model_df['game_id'] == game_id) & (model_df['play_id'] == play_id)]
    height_diff = relevant_data['height_diff'].values[0]
    
    offense_x = relevant_data['receiver_x'].values[0]
    offense_y = relevant_data['receiver_y'].values[0]
    
    x_diffs = []
    y_diffs = []
    for i,row in grid.iterrows():
        x_diffs.append(row['x'] - offense_x)
        y_diffs.append(row['y'] - offense_y)
    
    x_diffs.append(relevant_data['x_diff'].values[0])
    y_diffs.append(relevant_data['y_diff'].values[0])
    
    grid_data =  pd.DataFrame({'x_diff':x_diffs,
                  'y_diff':y_diffs})
    
        
    probs = get_model_probs(model,grid_data)
    
    xs = list(grid['x'])
    xs.append(relevant_data['defender_x'].values[0])
    ys = list(grid['y'])
    ys.append(relevant_data['defender_y'].values[0])
    grid_data['x'] = xs
    grid_data['y'] = ys
    grid_data['prob_C'] = probs['C']
    grid_data['prob_I'] = probs['I']
    grid_data['prob_IN'] = probs['IN']
    return grid_data


role_colors = {
    'Other Route Runner':'green',
    'Passer':'grey',
    'Defensive Coverage':'blue',
    'Targeted Receiver':'lime'
}


def animate_pre_throw(game_id, play_id,grid_data, input_df=input_df,output_df=output_df, supp=supp):
    play_pre = input_df.loc[(game_id, play_id)].copy()
    play_direction = play_pre.iloc[0]['play_direction']
    play_post = output_df.loc[(game_id, play_id)].copy()
    if play_direction == 'left':
        play_pre['x'] = 120 - play_pre['x']
        play_pre['y'] = 53.3 - play_pre['y']
        play_pre['ball_land_x'] = 120 - play_pre['ball_land_x']
        play_pre['ball_land_y'] = 53.3 - play_pre['ball_land_y']
        play_post['x'] = 120 - play_post['x']
        play_post['y'] = 53.3 - play_post['y']

    if play_post['y'].median() > 25:
        play_pre['y'] = 53.3 - play_pre['y']
        play_pre['ball_land_y'] = 53.3 - play_pre['ball_land_y']
        play_post['y'] = 53.3 - play_post['y']
    play_supp = supp[(supp['game_id'] == game_id) & (supp['play_id'] == play_id)]
    description = play_supp['play_description'].values[0] if not play_supp.empty else ""

    pre_frames = sorted(play_pre['frame_id'].unique())
    
    role_dict = play_pre[['nfl_id', 'player_role']].drop_duplicates().set_index('nfl_id')['player_role'].to_dict()
    ball_land_x = play_pre['ball_land_x'].iloc[0]
    ball_land_y = play_pre['ball_land_y'].iloc[0]
    
    frames = []
    sliders_dict = {"active":0, "yanchor":"top", "xanchor":"left", 
                    "currentvalue":{"prefix":"Frame: "}, "pad":{"b":10,"t":50}, "steps":[]}
    
    # Field numbers
    field_x_top = np.arange(20,110,10)
    field_x_bottom = np.arange(20,110,10)
    field_y_top = [5]*len(field_x_top)
    field_y_bottom = [53.5-5]*len(field_x_bottom)
    field_numbers_top = list(map(str, list(np.arange(20, 61, 10)-10) + list(np.arange(40, 9, -10))))
    field_numbers_bottom = field_numbers_top
    
    for frame_id in pre_frames:
        data = []

        # Top numbers
        data.append(go.Scatter(
            x=field_x_top, y=field_y_top, mode='text',
            text=field_numbers_top, textfont_size=30, textfont_family="Courier New, monospace",
            textfont_color="#ffffff", showlegend=False, hoverinfo='none'
        ))
        # Bottom numbers
        data.append(go.Scatter(
            x=field_x_bottom, y=field_y_bottom, mode='text',
            text=field_numbers_bottom, textfont_size=30, textfont_family="Courier New, monospace",
            textfont_color="#ffffff", showlegend=False, hoverinfo='none'
        ))

        # Ball land marker
        data.append(go.Scatter(
            x=[ball_land_x], y=[ball_land_y], mode='markers',
            marker=dict(size=15, color="gold", symbol="diamond", line=dict(width=2, color="black")),
            name="Ball Land",
            hovertext=[f"Ball lands at ({ball_land_x:.1f}, {ball_land_y:.1f})"],
            hoverinfo='text',
            showlegend=True
        ))
        # Plot best location for interception
        temp = grid_data.iloc[grid_data['prob_IN'].argmax()]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="gold",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Optimal location for interception",
            hovertext=[(
                        f'Optimal location for interception<br>'
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        # Plot best location for incompletion
        temp = grid_data.iloc[grid_data['prob_I'].argmax()]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="gold",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Optimal location for incompletion",
            hovertext=[(
                        f'Optimal location for incompletion<br>'
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        # Plot actual final location
        temp = grid_data.iloc[-1]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="blue",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Final location",
            hovertext=[(
                        f"Final location of defender<br>"
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        ## show region defender can reach
        xs = grid_data['x'].values
        ys = grid_data['y'].values
        points = np.column_stack((xs, ys))
        
        # compute convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # close the polygon by repeating the first point
        hull_points = np.vstack([hull_points, hull_points[0]])
        
        data.append(go.Scatter(
            x=hull_points[:,0],
            y=hull_points[:,1],
            fill='toself',           # fill the polygon
            fillcolor='rgba(0,100,255,0.2)',  # semi-transparent blue
            line=dict(color='blue'),
            mode='lines',
            name='Grid coverage region',
            hoverinfo='skip'         # skip hover to keep it clean
        ))


        # Plot players by role and player_to_predict
        for role, color in role_colors.items():
            role_players = [pid for pid in play_pre.loc[play_pre['frame_id']==frame_id, 'nfl_id'] if role_dict.get(pid) == role]
            if role_players:
                frame_data_role = play_pre[(play_pre['frame_id']==frame_id) & (play_pre['nfl_id'].isin(role_players))].sort_values('nfl_id')
                
                # True players (circle)
                true_players = frame_data_role[frame_data_role['player_to_predict'] == True]
                if not true_players.empty:
                    data.append(go.Scatter(
                        x=true_players['x'], y=true_players['y'], mode='markers',
                        marker=dict(size=12, color=color, line=dict(width=1, color="black"), symbol='circle'),
                        name=role,
                        hovertext=[f"ID: {pid}<br>Role: {role}<br>Predict: True" for pid in true_players['nfl_id']],
                        hoverinfo='text',
                        showlegend=True
                    ))

                # False players (square)
                false_players = frame_data_role[frame_data_role['player_to_predict'] == False]
                if not false_players.empty:
                    data.append(go.Scatter(
                        x=false_players['x'], y=false_players['y'], mode='markers',
                        marker=dict(size=12, color=color, line=dict(width=1, color="black"), symbol='square'),
                        name=f"{role} (Not Predicted)",
                        hovertext=[f"ID: {pid}<br>Role: {role}<br>Predict: False" for pid in false_players['nfl_id']],
                        hoverinfo='text',
                        showlegend=True
                    ))

        # Slider step
        slider_step = {"args":[[frame_id], {"frame":{"duration":100,"redraw":False},"mode":"immediate","transition":{"duration":0}}],
                       "label": str(frame_id), "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frame_id)))
    
    scale = 10
    layout = go.Layout(
        autosize=False, width=120*scale, height=60*scale + 60,
        xaxis=dict(range=[0,120], autorange=False, tickmode='array', tickvals=np.arange(10,111,5), showticklabels=False),
        yaxis=dict(range=[0,53.3], autorange=False, showgrid=False, showticklabels=False),
        plot_bgcolor='#00B140',
        title=f"Game {game_id} | Play {play_id} | Direction {play_direction} (Pre-Throw)",
        sliders=[sliders_dict],
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"Play",
                 "method":"animate",
                 "args":[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label":"Pause","method":"animate","args":[[None],{"frame":{"duration":0},"mode":"immediate"}]}
            ]
        }],
        annotations=[dict(
            x=0.5, y=-0.08, xref='paper', yref='paper',
            text=description, showarrow=False, font=dict(size=14),
            xanchor='center', yanchor='top'
        )]
    )

    fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames[1:])
    return fig



def animate_post_throw(game_id, play_id,grid_data, input_df=input_df, output_df=output_df, supp=supp):
    play_pre = input_df.loc[(game_id, play_id)].copy()
    play_direction = play_pre.iloc[0]['play_direction']
    play_post = output_df.loc[(game_id, play_id)].copy()
    if play_direction == 'left':
        play_pre['x'] = 120 - play_pre['x']
        play_pre['y'] = 53.3 - play_pre['y']
        play_pre['ball_land_x'] = 120 - play_pre['ball_land_x']
        play_pre['ball_land_y'] = 53.3 - play_pre['ball_land_y']
        play_post['x'] = 120 - play_post['x']
        play_post['y'] = 53.3 - play_post['y']

    if play_post['y'].median() > 25:
        play_pre['y'] = 53.3 - play_pre['y']
        play_pre['ball_land_y'] = 53.3 - play_pre['ball_land_y']
        play_post['y'] = 53.3 - play_post['y']
    
    play_supp = supp[(supp['game_id'] == game_id) & (supp['play_id'] == play_id)]
    description = play_supp['play_description'].values[0] if not play_supp.empty else ""

    post_frames = sorted(play_post['frame_id'].unique())
    
    role_dict = play_pre[['nfl_id', 'player_role']].drop_duplicates().set_index('nfl_id')['player_role'].to_dict()
    ball_land_x = play_pre['ball_land_x'].iloc[0]
    ball_land_y = play_pre['ball_land_y'].iloc[0]
    
    frames = []
    sliders_dict = {"active":0, "yanchor":"top", "xanchor":"left", "currentvalue":{"prefix":"Frame: "}, "pad":{"b":10,"t":50}, "steps":[]}
    
    # Draw field numbers
    field_x_top = np.arange(20,110,10)
    field_x_bottom = np.arange(20,110,10)
    field_y_top = [5]*len(field_x_top)
    field_y_bottom = [53.5-5]*len(field_x_bottom)
    field_numbers_top = list(map(str, list(np.arange(20, 61, 10)-10) + list(np.arange(40, 9, -10))))
    field_numbers_bottom = field_numbers_top
    
    for frame_id in post_frames:
        data = []

        # Top field numbers
        data.append(go.Scatter(
            x=field_x_top,
            y=field_y_top,
            mode='text',
            text=field_numbers_top,
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo='none'
        ))
        # Bottom field numbers
        data.append(go.Scatter(
            x=field_x_bottom,
            y=field_y_bottom,
            mode='text',
            text=field_numbers_bottom,
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo='none'
        ))

        # Plot Ball Land as a legend item
        data.append(go.Scatter(
            x=[ball_land_x],
            y=[ball_land_y],
            mode='markers',
            marker=dict(
                size=15,
                color="gold",
                symbol="diamond",
                line=dict(width=2, color="black")
            ),
            name="Ball Land",
            hovertext=[f"Ball lands at ({ball_land_x:.1f}, {ball_land_y:.1f})"],
            hoverinfo='text',
            showlegend=True
        ))

        # Plot best location for interception
        temp = grid_data.iloc[grid_data['prob_IN'].argmax()]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="gold",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Optimal location for interception",
            hovertext=[(
                        f'Optimal location for interception<br>'
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        # Plot best location for incompletion
        temp = grid_data.iloc[grid_data['prob_I'].argmax()]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="gold",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Optimal location for incompletion",
            hovertext=[(
                        f'Optimal location for incompletion<br>'
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        # Plot actual final location
        temp = grid_data.iloc[-1]
        data.append(go.Scatter(
            x=[temp['x']],
            y=[temp['y']],
            mode='markers',
            marker=dict(
                size=15,
                color="blue",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Final location",
            hovertext=[(
                        f"Final location of defender<br>"
                        f"Location at ({temp['x']:.2f}, {temp['y']:.2f})<br>"
                        f"Prob Completion: {temp['prob_C']:.2f}<br>"
                        f"Prob Incompletion: {temp['prob_I']:.2f}<br>"
                        f"Prob Interception: {temp['prob_IN']:.2f}"
                        )],
            hoverinfo='text',
            showlegend=True
        ))
        ## show region defender can reach
        xs = grid_data['x'].values
        ys = grid_data['y'].values
        points = np.column_stack((xs, ys))
        
        # compute convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # close the polygon by repeating the first point
        hull_points = np.vstack([hull_points, hull_points[0]])
        
        data.append(go.Scatter(
            x=hull_points[:,0],
            y=hull_points[:,1],
            fill='toself',           # fill the polygon
            fillcolor='rgba(0,100,255,0.2)',  # semi-transparent blue
            line=dict(color='blue'),
            mode='lines',
            name='Grid coverage region',
            hoverinfo='skip'         # skip hover to keep it clean
        ))

        # Plot Players by role so legend works
        for role, color in role_colors.items():
            role_players = [pid for pid in play_post.loc[play_post['frame_id']==frame_id, 'nfl_id'] 
                            if role_dict.get(pid) == role]
            if role_players:
                frame_data_role = play_post[(play_post['frame_id']==frame_id) & 
                                            (play_post['nfl_id'].isin(role_players))].sort_values('nfl_id')
        
                # build hovertext including current location
                hover_texts = [
                    f"ID: {row['nfl_id']}<br>"
                    f"Role: {role}<br>"
                    f"Location: ({row['x']:.2f}, {row['y']:.2f})"
                    for _, row in frame_data_role.iterrows()
                ]
        
                data.append(go.Scatter(
                    x=frame_data_role['x'],
                    y=frame_data_role['y'],
                    mode='markers',
                    marker=dict(size=12, color=color, line=dict(width=1, color="black")),
                    name=role,
                    hovertext=hover_texts,
                    hoverinfo='text',
                ))

        # Add frame to slider
        slider_step = {"args":[[frame_id], {"frame":{"duration":100,"redraw":False},"mode":"immediate","transition":{"duration":0}}],
                       "label": str(frame_id),
                       "method": "animate"}
        sliders_dict["steps"].append(slider_step)

        frames.append(go.Frame(data=data, name=str(frame_id)))
    
    scale = 10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale + 60,  # extra space for description
        xaxis=dict(range=[0,120], autorange=False, tickmode='array', tickvals=np.arange(10,111,5), showticklabels=False),
        yaxis=dict(range=[0,53.3], autorange=False, showgrid=False, showticklabels=False),
        plot_bgcolor='#00B140',
        title=f"Game {game_id} | Play {play_id} | Direction {play_direction}",
        sliders=[sliders_dict],
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"Play",
                 "method":"animate",
                 "args":[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label":"Pause","method":"animate","args":[[None],{"frame":{"duration":0},"mode":"immediate"}]}
            ]
        }],
        annotations=[
            dict(
                x=0.5,
                y=-0.08,  # position below the plot
                xref='paper',
                yref='paper',
                text=description,
                showarrow=False,
                font=dict(size=14),
                xanchor='center',
                yanchor='top'
            )
        ]
    )

    fig = go.Figure(
        data=frames[0]["data"],
        layout=layout,
        frames=frames[1:]
    )

    return fig



game_id = 2023102206
play_id = 524
grid_data = get_grid_data(game_id,play_id)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))


game_id = 2023110600
play_id = 1554
grid_data = get_grid_data(game_id,play_id,distance=15)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))


game_id = 2023101504
play_id = 4602
grid_data = get_grid_data(game_id,play_id)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))


game_id = 2023102909
play_id = 3065
grid_data = get_grid_data(game_id,play_id)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))


game_id = 2023100900
play_id = 3730
grid_data = get_grid_data(game_id,play_id)
prethrow = animate_pre_throw(game_id,play_id,grid_data)
postthrow = animate_post_throw(game_id,play_id,grid_data)
display(prethrow)
display(postthrow)



prethrow.write_html("/kaggle/working/love_to_doubs_prethrow.html")
postthrow.write_html("/kaggle/working/love_to_doubs_postthrow.html")


game_id = 2023121001
play_id = 1138
grid_data = get_grid_data(game_id,play_id)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))


model_df[model_df['pass_result_is_IN']==1]


index = 6
# really good: 6, 25
# eh: 8,10,13,17
play = model_df[model_df['pass_result_is_IN']==1].iloc[index]
game_id = play['game_id']
play_id = play['play_id']
grid_data = get_grid_data(game_id,play_id)
pre = animate_pre_throw(game_id,play_id,grid_data)
post = animate_post_throw(game_id,play_id,grid_data)
display(pre)
display(post)
pre.write_html("/kaggle/working/ridder_interception_prethrow.html")
post.write_html("/kaggle/working/ridder_interception_postthrow.html")


import random
play = model_df.sample(n=1)
game_id = play['game_id'].iloc[0]
play_id = play['play_id'].iloc[0]
grid_data = get_grid_data(game_id,play_id, density=1)
display(animate_pre_throw(game_id,play_id,grid_data))
display(animate_post_throw(game_id,play_id,grid_data))
# Examples of incompletions that could have been interception
# 2023100900 3730 pretty good, could easily have made a play
# 2023111901 3940 pretty good
# 2023100105 1780 pretty good
# 2024010704 2504  ok

# Examples of completions that could have been better
# 2023120301 1790  really good
# 2023110504 2028  ok
# 2023120304 354   technically incomplete, but should have been complete and is a good example
# 2023101507 2930  kind of just a bad play by the defender
# 2023092800 2083  kinda meh, he was following another receiver

# Misc
# 2023092403 2781 was completion, went to incompletion spot, but could have been interception




