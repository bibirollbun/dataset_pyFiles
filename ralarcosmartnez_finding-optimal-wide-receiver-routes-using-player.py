import pandas as pd

path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv' 


df = pd.read_csv(path)
print(df.shape) 
print(df.head())


import pandas as pd
import numpy as np
import glob
import os

path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'

def load_week_data(week_num):
    
    week_str = f"{week_num:02d}" 
    
    # 1. Input (Pre-pass)
    input_path = f"{path}/train/input_2023_w{week_str}.csv"
    if not os.path.exists(input_path):
        print(f"No se encontró: {input_path}")
        return None
        
    print(f"Cargando semana {week_num}...")
    df_input = pd.read_csv(input_path)
    
    df_input['phase'] = 'pre_pass'
    
    # 2. Output (Post-pass)
    output_path = f"{path}/train/output_2023_w{week_str}.csv"
    df_output = pd.read_csv(output_path)
    df_output['phase'] = 'post_pass'
    
    common_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 'phase']
    
    df_full = pd.concat([df_input, df_output], axis=0, ignore_index=True)

    # Define the correct phase order
    df_full['phase'] = pd.Categorical(
        df_full['phase'],
        categories=['pre_pass', 'post_pass'],
        ordered=True
    )
    
    # Order correctly (pre_pass → post_pass)
    df_full.sort_values(
        by=['game_id', 'play_id', 'nfl_id', 'phase', 'frame_id'],
        inplace=True
    )
    
    # Forward fill of static columns
    static_cols = ['player_role', 'play_direction', 'player_side', 'player_position']
    cols_to_fill = [c for c in static_cols if c in df_full.columns]
    
    df_full[cols_to_fill] = (
        df_full
        .groupby(['game_id', 'play_id', 'nfl_id'])[cols_to_fill]
        .ffill()
    )
    
        
    return df_full

df_tracking = load_week_data(1)
print(f"Filas totales semana 1: {len(df_tracking)}")


def standardize_tracking_data(df):
    """
    Normalize the coordinates so that the offense always attacks from left to right.
    X: 0 to 120 (0 is your own end zone, 120 is the opponent's)
    Y: 0 to 53.3
    Direction/Opposite: They rotate
    """
    # Crear copia para no alterar original si no se quiere
    df_norm = df.copy()
    
    # Identificar jugadas hacia la izquierda
    to_left = df_norm['play_direction'] == 'left'
    
    # Invertir X y Y
    df_norm.loc[to_left, 'x'] = 120 - df_norm.loc[to_left, 'x']
    df_norm.loc[to_left, 'y'] = 53.3 - df_norm.loc[to_left, 'y']
    
    # Ajustar Orientación (o) y Dirección (dir)
    # Se suman 180 grados y se usa módulo 360 para mantenerlo en círculo
    df_norm.loc[to_left, 'dir'] = (df_norm.loc[to_left, 'dir'] + 180) % 360
    df_norm.loc[to_left, 'o'] = (df_norm.loc[to_left, 'o'] + 180) % 360
    
    return df_norm

# Aplicar estandarización
df_tracking = standardize_tracking_data(df_tracking)
print("Coordenadas estandarizadas (Ofensiva siempre va -> Derecha)")


# 1. Cargar Supplementary
path_supp = f"{path}/supplementary_data.csv"
df_plays = pd.read_csv(path_supp, low_memory=False)

# 2. Filtrar jugadas nulas por castigo (generalmente ensucian el análisis de rutas óptimas)
df_plays = df_plays[df_plays['play_nullified_by_penalty'] == 'N'].copy()

# 3. Limpieza de Features en Supplementary
# Convertir altura (ft-in) a pulgadas (Esto está en Tracking, pero es bueno tener la función)
def height_to_inches(h):
    if pd.isna(h): return None
    try:
        feet, inches = map(int, h.split('-'))
        return feet * 12 + inches
    except:
        return None

if 'player_height' in df_tracking.columns:
    df_tracking['height_inches'] = df_tracking['player_height'].apply(height_to_inches)

# 4. Merge (Unir Tracking con Contexto de Jugada)
# Usamos inner join para quedarnos solo con jugadas válidas que existen en ambos lados
df_clean = pd.merge(df_tracking, df_plays, on=['game_id', 'play_id'], how='inner')

print(f"Dataset fusionado. Columnas totales: {len(df_clean.columns)}")


# Verificamos los roles disponibles
print(df_clean['player_role'].unique())

# Filtro sugerido: Quedarnos solo con el Receptor Objetivo y quien lo cubre
# Ojo: 'Targeted Receiver' es clave.
target_roles = ['Targeted Receiver', 'Other Route Runner'] 

# Opcional: Filtrar solo el equipo ofensivo si solo te importa la ruta ideal sin interacción defensiva inicial
df_receivers = df_clean[df_clean['player_role'].isin(target_roles)].copy()

print(f"Datos filtrados para receptores: {df_receivers.shape}")


df_clean.columns


df_all = df_clean.copy()

df_rec = df_clean[
    (df_clean['pass_result'] == 'C') &                 # pase completo
    (df_clean['route_of_targeted_receiver'].notna()) & # receptor objetivo
    (df_clean['phase'] == 'post_pass')                   # movimiento real de la ruta
]



import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)



separations = []

for (game_id, play_id, nfl_id), rec in df_rec.groupby(['game_id','play_id','nfl_id']):
    
    rec_frames = rec[['frame_id','x','y']]
    route = rec['route_of_targeted_receiver'].iloc[0]
    
    defenders = df_all[
        (df_all['game_id'] == game_id) &
        (df_all['play_id'] == play_id) &
        (df_all['phase'] == 'post_pass') &
        (df_all['player_side'] == 'Defense')
    ][['frame_id','x','y']]
    
    if defenders.empty:
        continue
    
    for _, r in rec_frames.iterrows():
        same_frame_def = defenders[defenders['frame_id'] == r['frame_id']]
        if same_frame_def.empty:
            continue
            
        dists = np.sqrt(
            (same_frame_def['x'] - r['x'])**2 +
            (same_frame_def['y'] - r['y'])**2
        )
        
        separations.append({
            'game_id': game_id,
            'play_id': play_id,
            'nfl_id': nfl_id,
            'route': route,
            'frame_sep': dists.min()
        })



sep_df = pd.DataFrame(separations)

print("rows:", len(sep_df))
sep_df.head()
#sep_df['route'].value_counts()



sep_df = pd.DataFrame(separations)
sep_df = sep_df[sep_df['frame_sep'] > 0].copy()


play_sep = (
    sep_df
    .groupby(['game_id','play_id','nfl_id','route'])
    .agg(
        avg_sep=('frame_sep','mean'),
        max_sep=('frame_sep','max'),
        std_sep=('frame_sep','std'),
        n_frames=('frame_sep','count')
    )
    .reset_index()
)



play_sep


route_summary = (
    play_sep
    .groupby('route')
    .agg(
        mean_ars=('avg_sep','mean'),
        median_ars=('avg_sep','median'),
        plays=('avg_sep','count')
    )
    .sort_values('mean_ars', ascending=False)
)



route_summary


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# Sort routes by median separation for better readability
order = play_sep.groupby('route')['avg_sep'].median().sort_values(ascending=False).index

sns.boxplot(
    data=play_sep, 
    x='avg_sep', 
    y='route', 
    order=order,
    palette='viridis'
)

plt.title('Distribution of Average Receiver Separation (ARS) by Route Type', fontsize=15)
plt.xlabel('Average Separation (Yards)', fontsize=12)
plt.ylabel('Route Name', fontsize=12)

# Global average line
plt.axvline(play_sep['avg_sep'].mean(), color='red', linestyle='--', label='Global Average')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 7))

scatter = sns.scatterplot(
    data=route_summary, 
    x='plays', 
    y='mean_ars', 
    size='mean_ars', 
    hue='mean_ars',
    palette='magma',
    sizes=(100, 600)
)


for i in range(route_summary.shape[0]):
    plt.text(
        route_summary.plays.iloc[i] + 0.5, 
        route_summary.mean_ars.iloc[i], 
        route_summary.index[i], 
        fontsize=10,
        fontweight='bold'
    )

plt.title('Route Frequency vs. Separation Efficiency', fontsize=15)
plt.xlabel('Number of Plays (Sample Size)', fontsize=12)
plt.ylabel('Mean Receiver Separation (Yards)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def plot_normalized_play(df_tracking):
    
    sample_play = df_tracking[df_tracking['player_role'] == 'Targeted Receiver'].sample(1)
    g_id, p_id = sample_play.iloc[0]['game_id'], sample_play.iloc[0]['play_id']
    
    play_data = df_tracking[(df_tracking['game_id'] == g_id) & (df_tracking['play_id'] == p_id)]
    
    plt.figure(figsize=(12, 5.33)) 
    
    #Defenders (Orange), Offense (Blue)
    for nfl_id, player_df in play_data.groupby('nfl_id'):
        is_offense = player_df['player_side'].iloc[0] == 'Offense'
        is_target = player_df['player_role'].iloc[0] == 'Targeted Receiver'
        
        color = 'royalblue' if is_offense else 'darkorange'
        alpha = 1.0 if is_target else 0.4
        linewidth = 2.5 if is_target else 1.0
        
        plt.plot(player_df['x'], player_df['y'], color=color, alpha=alpha, linewidth=linewidth)
        # Mark current/final position
        plt.scatter(player_df['x'].iloc[-1], player_df['y'].iloc[-1], color=color, alpha=alpha, s=50)

    plt.title(f'Standardized Play Trajectories | Game: {g_id} Play: {p_id}', fontsize=14)
    plt.xlabel('X Coordinate (Yards from Endzone)')
    plt.ylabel('Y Coordinate (Width)')
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    
    #Field Lines
    plt.axvline(10, color='black', alpha=0.2) # Back of Endzone
    plt.axvline(110, color='black', alpha=0.2) # Goal Line
    plt.gca().set_facecolor('#e8f5e9') 
    plt.tight_layout()
    plt.show()


plot_normalized_play(df_clean)


# Updated filter to include both Complete (C) and Incomplete (I) passes
df_res = df_clean[
    (df_clean['pass_result'].isin(['C', 'I'])) & 
    (df_clean['route_of_targeted_receiver'].notna()) & 
    (df_clean['phase'] == 'post_pass')
].copy()

# Recalculate separations including the pass result
res_separations = []

for (game_id, play_id, nfl_id), rec in df_res.groupby(['game_id','play_id','nfl_id']):
    rec_frames = rec[['frame_id','x','y', 'pass_result']]
    route = rec['route_of_targeted_receiver'].iloc[0]
    result = rec['pass_result'].iloc[0]
    
    defenders = df_all[
        (df_all['game_id'] == game_id) &
        (df_all['play_id'] == play_id) &
        (df_all['phase'] == 'post_pass') &
        (df_all['player_side'] == 'Defense')
    ][['frame_id','x','y']]
    
    if defenders.empty: continue
    
    for _, r in rec_frames.iterrows():
        same_frame_def = defenders[defenders['frame_id'] == r['frame_id']]
        if same_frame_def.empty: continue
            
        dists = np.sqrt((same_frame_def['x'] - r['x'])**2 + (same_frame_def['y'] - r['y'])**2)
        
        res_separations.append({
            'route': route,
            'pass_result': result,
            'frame_sep': dists.min()
        })

res_sep_df = pd.DataFrame(res_separations)


plt.figure(figsize=(10, 6))

sns.kdeplot(data=res_sep_df[res_sep_df['pass_result'] == 'C'], x='frame_sep', label='Complete', fill=True, color='green')
sns.kdeplot(data=res_sep_df[res_sep_df['pass_result'] == 'I'], x='frame_sep', label='Incomplete', fill=True, color='red')

plt.title('Probability Density of Separation by Pass Result', fontsize=15)
plt.xlabel('Receiver Separation (Yards)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Create distance bins
res_sep_df['sep_bin'] = pd.cut(res_sep_df['frame_sep'], bins=np.arange(0, 11, 1))

# Calculate completion rate per bin
comp_rate = res_sep_df.groupby('sep_bin')['pass_result'].apply(lambda x: (x == 'C').mean()).reset_index()
comp_rate['sep_center'] = comp_rate['sep_bin'].apply(lambda x: x.mid)

plt.figure(figsize=(10, 6))
sns.lineplot(data=comp_rate, x='sep_center', y='pass_result', marker='o', color='blue', linewidth=2.5)

plt.title('Completion Probability vs. Separation Distance', fontsize=15)
plt.xlabel('Separation Distance (Yards)', fontsize=12)
plt.ylabel('Completion Rate (%)', fontsize=12)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# 1. Calculate Max Acceleration (a) per targeted receiver per play
accel_data = []

# We focus on the targeted receiver during the entire play
df_receivers = df_clean[df_clean['player_role'] == 'Targeted Receiver'].copy()

# Group by play to find the peak acceleration
accel_summary = (
    df_receivers
    .groupby(['game_id', 'play_id', 'nfl_id'])
    .agg(
        max_accel=('a', 'max'),
        avg_speed=('s', 'mean'),
        route=('route_of_targeted_receiver', 'first'),
        pass_result=('pass_result', 'first')
    )
    .reset_index()
)

# 2. Merge with our previous separation data (play_sep)
# This connects 'How much space' with 'How much force'
df_performance = pd.merge(
    play_sep, 
    accel_summary[['game_id', 'play_id', 'nfl_id', 'max_accel']], 
    on=['game_id', 'play_id', 'nfl_id']
)



import seaborn as sns
import matplotlib.pyplot as plt

# Usamos 'reg' para ver la línea de tendencia entre aceleración y separación
g = sns.jointplot(
    data=df_performance,
    x='max_accel', 
    y='avg_sep',
    kind='reg',
    truncate=False,
    color='royalblue',
    scatter_kws={'alpha': 0.4},  # Transparencia para ver densidad de puntos
    line_kws={'color': 'red'}    # Línea de regresión en rojo
)

# Ajuste de títulos y etiquetas en inglés para la competencia
g.fig.suptitle('Impact of Peak Acceleration on Receiver Separation', fontsize=14)
g.ax_joint.set_xlabel('Max Acceleration ($yd/s^2$)', fontsize=12)
g.ax_joint.set_ylabel('Average Separation (Yards)', fontsize=12)

# Ajustar el espacio para que el título no se corte
g.fig.subplots_adjust(top=0.92) 
plt.show()


plt.figure(figsize=(12, 6))

# Boxplot of acceleration by route
sns.boxplot(
    data=df_performance,
    x='max_accel',
    y='route',
    palette='magma',
    order=df_performance.groupby('route')['max_accel'].median().sort_values(ascending=False).index
)

plt.title('Receiver Explosiveness (Max Accel) by Route Type', fontsize=15)
plt.xlabel('Peak Acceleration ($yd/s^2$)', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.show()

