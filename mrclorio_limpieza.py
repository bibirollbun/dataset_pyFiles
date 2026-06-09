import pandas as pd

path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv' 


df = pd.read_csv(path)
print(df.shape) 
print(df.head())


import pandas as pd
import numpy as np
import glob
import os

# Ruta base (Ajusta según tu entorno, ya vimos que termina en la carpeta final)
path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'

def load_week_data(week_num):
    """
    Carga y une los datos de input y output para una semana específica.
    """
    # Formato de archivo esperado: input_2023_w01.csv
    week_str = f"{week_num:02d}" 
    
    # 1. Cargar Input (Pre-pass)
    input_path = f"{path}/train/input_2023_w{week_str}.csv"
    if not os.path.exists(input_path):
        print(f"No se encontró: {input_path}")
        return None
        
    print(f"Cargando semana {week_num}...")
    df_input = pd.read_csv(input_path)
    # Marcamos que estos son frames de entrada
    df_input['phase'] = 'pre_pass'
    
    # 2. Cargar Output (Post-pass)
    output_path = f"{path}/train/output_2023_w{week_str}.csv"
    df_output = pd.read_csv(output_path)
    df_output['phase'] = 'post_pass'
    
    # 3. Preparar Output para la fusión
    # El output tiene menos columnas. Necesitamos alinear las columnas clave.
    # Las columnas estáticas (altura, peso, nombre) están en input.
    # Vamos a pegar 'input' y 'output' verticalmente (concat).
    
    # Identificar columnas comunes para mantener la trayectoria
    common_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 'phase']
    
    # Concatenamos. Pandas rellenará con NaN las columnas que faltan en Output (como player_role)
    # Esto es normal, luego rellenaremos esos valores hacia abajo (ffill)
    df_full = pd.concat([df_input, df_output], axis=0, ignore_index=True)

    # 1️⃣ Definir orden correcto de la fase
    df_full['phase'] = pd.Categorical(
        df_full['phase'],
        categories=['pre_pass', 'post_pass'],
        ordered=True
    )
    
    # 2️⃣ Ordenar correctamente (pre_pass → post_pass)
    df_full.sort_values(
        by=['game_id', 'play_id', 'nfl_id', 'phase', 'frame_id'],
        inplace=True
    )
    
    # 3️⃣ Forward fill de columnas estáticas
    static_cols = ['player_role', 'play_direction', 'player_side', 'player_position']
    cols_to_fill = [c for c in static_cols if c in df_full.columns]
    
    df_full[cols_to_fill] = (
        df_full
        .groupby(['game_id', 'play_id', 'nfl_id'])[cols_to_fill]
        .ffill()
    )
    
        
    return df_full

# --- EJECUCIÓN DE PRUEBA (Solo Semana 1) ---
df_tracking = load_week_data(1)
print(f"Filas totales semana 1: {len(df_tracking)}")


def standardize_tracking_data(df):
    """
    Normaliza las coordenadas para que la ofensiva siempre ataque de Izquierda a Derecha.
    X: 0 a 120 (0 es la endzone propia, 120 la rival)
    Y: 0 a 53.3
    Dir/O: Se rotan
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

