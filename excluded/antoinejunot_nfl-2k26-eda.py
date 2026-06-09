"""
Chargement et exploration des données NFL Big Data Bowl 2026
Analyse des trajectoires des joueurs pendant les passes aériennes
"""

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration des chemins
# =============================================================================

BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
SUPPLEMENT_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'

INPUT_PATTERN = os.path.join(BASE_DIR, 'input_2023_*.csv')
OUTPUT_PATTERN = os.path.join(BASE_DIR, 'output_2023_*.csv')

# =============================================================================
# Récupération des fichiers disponibles
# =============================================================================

input_files = sorted(glob.glob(INPUT_PATTERN))
output_files = sorted(glob.glob(OUTPUT_PATTERN))

print("="*60)
print("EXPLORATION DES FICHIERS")
print("="*60)
print(f"✓ Fichiers input trouvés: {len(input_files)}")
print(f"✓ Fichiers output trouvés: {len(output_files)}")

if input_files:
    print(f"\nExemples de fichiers input:")
    for f in input_files[:2]:
        print(f"  - {os.path.basename(f)}")

if output_files:
    print(f"\nExemples de fichiers output:")
    for f in output_files[:2]:
        print(f"  - {os.path.basename(f)}")

# Vérification de cohérence
if len(input_files) != len(output_files):
    print(f"\n⚠ ATTENTION: Déséquilibre détecté ({len(input_files)} input vs {len(output_files)} output)")


# =============================================================================
# Chargement des données de la semaine 01
# =============================================================================

print(f"\n{'='*60}")
print("CHARGEMENT - SEMAINE 01")
print(f"{'='*60}")

df_input = pd.read_csv(input_files[0])
df_output = pd.read_csv(output_files[0])

# Affichage du résumé des données input
print(f"\n--- Input ---")
print(f"Dimensions: {df_input.shape[0]:,} lignes × {df_input.shape[1]} colonnes")
print(f"\nAperçu des données:\n{df_input.head(3)}")
print(f"\nColonnes: {df_input.columns.tolist()}")

# Affichage du résumé des données output
print(f"\n--- Output ---")
print(f"Dimensions: {df_output.shape[0]:,} lignes × {df_output.shape[1]} colonnes")
print(f"\nAperçu des données:\n{df_output.head(3)}")
print(f"\nColonnes: {df_output.columns.tolist()}")





# =============================================================================
# Analyse de la structure des données
# =============================================================================

print("="*60)
print("ANALYSE DE LA STRUCTURE INPUT")
print("="*60)

# Informations générales
print(f"\nNombre total d'observations: {len(df_input):,}")
print(f"\nTypes de données:\n{df_input.dtypes}")

# Vérification des valeurs manquantes
print(f"\n--- Valeurs manquantes ---")
missing = df_input.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✓ Aucune valeur manquante")

# Colonnes clés pour l'analyse offensive
print(f"\n--- Colonnes disponibles ---")
print(df_input.columns.tolist())




# =============================================================================
# Identification des plays et événements
# =============================================================================

print(f"\n{'='*60}")
print("ANALYSE DES PLAYS")
print(f"{'='*60}")

# Compter les plays uniques avec les bons noms de colonnes
unique_plays = df_input.groupby(['game_id', 'play_id']).size()
print(f"\n✓ Nombre de plays uniques: {len(unique_plays):,}")
print(f"✓ Nombre moyen de frames par play: {unique_plays.mean():.1f}")
print(f"✓ Min/Max frames: {unique_plays.min()} / {unique_plays.max()}")

# Distribution du nombre de frames
print(f"\n--- Distribution du nombre de frames ---")
print(unique_plays.describe())


# =============================================================================
# Identification des joueurs et rôles
# =============================================================================

print(f"\n{'='*60}")
print("ANALYSE DES JOUEURS")
print(f"{'='*60}")

# Positions des joueurs
print(f"\n--- Répartition par position ---")
position_counts = df_input['player_position'].value_counts()
print(position_counts)

# Côté (offense vs defense)
print(f"\n--- Répartition par côté (offense/defense) ---")
side_counts = df_input['player_side'].value_counts()
print(side_counts)

# Rôle des joueurs
print(f"\n--- Répartition par rôle ---")
role_counts = df_input['player_role'].value_counts()
print(role_counts)

# Focus sur les receveurs potentiels (positions offensives)
receiver_positions = ['WR', 'TE', 'RB']
receivers = df_input[df_input['player_position'].isin(receiver_positions)]
print(f"\n✓ Observations de receveurs potentiels (WR/TE/RB): {len(receivers):,}")

# Identifier le joueur ciblé (player_to_predict)
targeted = df_input[df_input['player_to_predict'] == True]
print(f"\n✓ Observations du joueur ciblé (player_to_predict=True): {len(targeted):,}")
print(f"✓ Nombre de joueurs uniques ciblés: {targeted['nfl_id'].nunique():,}")

# Vérifier les positions des joueurs ciblés
if len(targeted) > 0:
    print(f"\n--- Positions des joueurs ciblés ---")
    targeted_positions = targeted['player_position'].value_counts()
    print(targeted_positions)


# =============================================================================
# Analyse du point d'atterrissage du ballon
# =============================================================================

print(f"\n{'='*60}")
print("DONNÉES DU BALLON")
print(f"{'='*60}")

# Statistiques sur ball_land_x et ball_land_y
print(f"\n--- Point d'atterrissage du ballon ---")
print(f"ball_land_x - Min: {df_input['ball_land_x'].min():.1f}, Max: {df_input['ball_land_x'].max():.1f}, Moyenne: {df_input['ball_land_x'].mean():.1f}")
print(f"ball_land_y - Min: {df_input['ball_land_y'].min():.1f}, Max: {df_input['ball_land_y'].max():.1f}, Moyenne: {df_input['ball_land_y'].mean():.1f}")

# Vérifier si le point d'atterrissage est constant par play
sample_play = df_input[(df_input['game_id'] == df_input['game_id'].iloc[0]) & 
                        (df_input['play_id'] == df_input['play_id'].iloc[0])]
unique_landing = sample_play[['ball_land_x', 'ball_land_y']].nunique()
print(f"\n✓ Le point d'atterrissage est {'constant' if unique_landing['ball_land_x'] == 1 else 'variable'} par play")



# =============================================================================
# Analyse de la durée de prédiction
# =============================================================================

print(f"\n{'='*60}")
print("DURÉE DE PRÉDICTION")
print(f"{'='*60}")

print(f"\n--- num_frames_output (nombre de frames à prédire) ---")
print(df_input['num_frames_output'].value_counts().sort_index())
print(f"\nMoyenne: {df_input['num_frames_output'].mean():.1f} frames")
print(f"Min/Max: {df_input['num_frames_output'].min()} / {df_input['num_frames_output'].max()} frames")



# =============================================================================
# Analyse d'un play exemple détaillé
# =============================================================================

print(f"\n{'='*60}")
print("EXEMPLE D'UN PLAY COMPLET")
print(f"{'='*60}")

# Prendre le premier play
first_game = df_input['game_id'].iloc[0]
first_play = df_input['play_id'].iloc[0]

play_data = df_input[(df_input['game_id'] == first_game) & 
                      (df_input['play_id'] == first_play)]

print(f"\nGameId: {first_game}, PlayId: {first_play}")
print(f"Nombre total de frames: {len(play_data)}")
print(f"FrameIds: {play_data['frame_id'].min()} → {play_data['frame_id'].max()}")

n_players = play_data['nfl_id'].nunique()
print(f"Nombre de joueurs trackés: {n_players}")

print(f"\n--- Répartition offense/défense ---")
print(play_data.groupby('player_side')['nfl_id'].nunique())

print(f"\n--- Joueur ciblé ---")
targeted_player = play_data[play_data['player_to_predict'] == True]
if len(targeted_player) > 0:
    print(f"Nom: {targeted_player['player_name'].iloc[0]}")
    print(f"Position: {targeted_player['player_position'].iloc[0]}")
    print(f"Rôle: {targeted_player['player_role'].iloc[0]}")

print(f"\n--- Point d'atterrissage du ballon ---")
print(f"ball_land_x: {play_data['ball_land_x'].iloc[0]:.2f}")
print(f"ball_land_y: {play_data['ball_land_y'].iloc[0]:.2f}")

print(f"\nAperçu des 5 premières frames (joueur ciblé uniquement):")
if len(targeted_player) > 0:
    print(targeted_player[['frame_id', 'player_name', 'x', 'y', 's', 'a', 'dir']].head())


# =============================================================================
# Analyse des données OUTPUT (trajectoires futures)
# =============================================================================

print(f"\n{'='*60}")
print("ANALYSE OUTPUT (trajectoires à prédire)")
print(f"{'='*60}")

print(f"\nDimensions: {df_output.shape[0]:,} observations")
print(f"Colonnes: {df_output.columns.tolist()}")

# Combien de plays dans output ?
output_plays = df_output.groupby(['game_id', 'play_id']).size()
print(f"\n✓ Nombre de plays dans output: {len(output_plays):,}")
print(f"✓ Nombre moyen de frames par play dans output: {output_plays.mean():.1f}")

# Vérifier la correspondance avec input
print(f"\n--- Correspondance INPUT ↔ OUTPUT ---")
print(f"Plays dans INPUT: {len(unique_plays):,}")
print(f"Plays dans OUTPUT: {len(output_plays):,}")
print(f"✓ {'Correspondance parfaite' if len(unique_plays) == len(output_plays) else 'Différence détectée'}")

# Analyser le même play dans output
output_play = df_output[(df_output['game_id'] == first_game) & 
                         (df_output['play_id'] == first_play)]

print(f"\n--- Exemple du même play dans OUTPUT ---")
print(f"Nombre de frames: {len(output_play)}")
if len(output_play) > 0:
    print(f"FrameIds: {output_play['frame_id'].min()} → {output_play['frame_id'].max()}")
    print(f"nfl_id: {output_play['nfl_id'].iloc[0]}")
    print(f"\nAperçu:")
    print(output_play.head())



# =============================================================================
# Résumé de la structure des données
# =============================================================================

print(f"\n{'='*60}")
print("RÉSUMÉ - STRUCTURE DES DONNÉES")
print(f"{'='*60}")

print(f"""
INPUT (observations passées):
  - {len(df_input):,} observations
  - {len(unique_plays):,} plays
  - {unique_plays.mean():.0f} frames en moyenne par play
  - Joueur à prédire identifié par 'player_to_predict'
  - Point d'atterrissage du ballon connu (ball_land_x, ball_land_y)
  - Positions offensives principales: {', '.join(receiver_positions)}

OUTPUT (trajectoires à prédire):
  - {len(df_output):,} observations
  - {len(output_plays):,} plays
  - {output_plays.mean():.0f} frames futures en moyenne par play
  - Contient uniquement les positions (x, y) à prédire

OBJECTIF:
  Prédire la trajectoire du joueur ciblé (player_to_predict=True)
  pendant les {df_input['num_frames_output'].mean():.0f} frames suivantes
  en se basant sur son historique de mouvement et le point d'atterrissage du ballon.
""")


"""
Analyse offensive : Métriques de performance des receveurs ciblés
Focus sur la séparation, l'efficacité de route et la vitesse d'approche
"""

# Configuration de l'affichage
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# Filtrage : Focus sur les vrais receveurs offensifs
# =============================================================================

print("="*60)
print("FILTRAGE DES RECEVEURS OFFENSIFS")
print("="*60)

# Filtrer uniquement les receveurs ciblés (offense)
offensive_receivers = df_input[
    (df_input['player_role'] == 'Targeted Receiver') &
    (df_input['player_position'].isin(['WR', 'TE', 'RB']))
]

print(f"\n✓ Observations filtrées: {len(offensive_receivers):,}")
print(f"✓ Nombre de plays avec receveur offensif: {offensive_receivers.groupby(['game_id', 'play_id']).ngroups:,}")
print(f"✓ Receveurs uniques: {offensive_receivers['nfl_id'].nunique()}")

print(f"\n--- Répartition par position ---")
print(offensive_receivers['player_position'].value_counts())


# =============================================================================
# Fonction : Calculer la distance euclidienne
# =============================================================================

def calculate_distance(x1, y1, x2, y2):
    """Calcule la distance euclidienne entre deux points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# =============================================================================
# Métrique 1 : Distance au point d'atterrissage (par frame)
# =============================================================================

print(f"\n{'='*60}")
print("MÉTRIQUE 1 : DISTANCE AU POINT D'ATTERRISSAGE")
print(f"{'='*60}")

# Calculer la distance pour chaque observation
offensive_receivers = offensive_receivers.copy()
offensive_receivers['distance_to_ball_land'] = calculate_distance(
    offensive_receivers['x'], 
    offensive_receivers['y'],
    offensive_receivers['ball_land_x'],
    offensive_receivers['ball_land_y']
)

print(f"\n--- Statistiques de distance ---")
print(f"Moyenne: {offensive_receivers['distance_to_ball_land'].mean():.2f} yards")
print(f"Médiane: {offensive_receivers['distance_to_ball_land'].median():.2f} yards")
print(f"Min/Max: {offensive_receivers['distance_to_ball_land'].min():.2f} / {offensive_receivers['distance_to_ball_land'].max():.2f} yards")


# =============================================================================
# Métrique 2 : Closing Speed (vitesse de rapprochement)
# =============================================================================

print(f"\n{'='*60}")
print("MÉTRIQUE 2 : CLOSING SPEED")
print(f"{'='*60}")

# Calculer le closing speed par play
# On compare la distance à la première frame vs dernière frame

def calculate_closing_metrics(group):
    """Calcule les métriques de rapprochement pour un play."""
    group = group.sort_values('frame_id')
    
    initial_distance = group['distance_to_ball_land'].iloc[0]
    final_distance = group['distance_to_ball_land'].iloc[-1]
    num_frames = len(group)
    
    # Distance parcourue vers la cible
    distance_closed = initial_distance - final_distance
    
    # Vitesse moyenne de rapprochement (yards/frame)
    closing_speed = distance_closed / num_frames if num_frames > 0 else 0
    
    # Vitesse réelle moyenne du joueur
    avg_speed = group['s'].mean()
    max_speed = group['s'].max()
    
    return pd.Series({
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'distance_closed': distance_closed,
        'closing_speed': closing_speed,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'num_frames': num_frames,
        'player_name': group['player_name'].iloc[0],
        'player_position': group['player_position'].iloc[0]
    })

# Appliquer le calcul par play
play_metrics = offensive_receivers.groupby(['game_id', 'play_id']).apply(calculate_closing_metrics).reset_index()

print(f"\n✓ Métriques calculées pour {len(play_metrics)} plays")
print(f"\n--- Statistiques de closing speed ---")
print(play_metrics[['closing_speed', 'avg_speed', 'max_speed']].describe())


# =============================================================================
# Métrique 3 : Route Efficiency
# =============================================================================

print(f"\n{'='*60}")
print("MÉTRIQUE 3 : ROUTE EFFICIENCY")
print(f"{'='*60}")

def calculate_route_efficiency(group):
    """Calcule l'efficacité de la route."""
    group = group.sort_values('frame_id')
    
    # Distance directe (début → ball_land)
    start_x, start_y = group['x'].iloc[0], group['y'].iloc[0]
    ball_x, ball_y = group['ball_land_x'].iloc[0], group['ball_land_y'].iloc[0]
    direct_distance = calculate_distance(start_x, start_y, ball_x, ball_y)
    
    # Distance réellement parcourue (somme des segments)
    actual_distance = 0
    for i in range(1, len(group)):
        x1, y1 = group['x'].iloc[i-1], group['y'].iloc[i-1]
        x2, y2 = group['x'].iloc[i], group['y'].iloc[i]
        actual_distance += calculate_distance(x1, y1, x2, y2)
    
    # Route efficiency = distance directe / distance parcourue
    # 1.0 = parfaitement droit, < 1.0 = route courbée
    efficiency = direct_distance / actual_distance if actual_distance > 0 else 0
    
    return pd.Series({
        'direct_distance': direct_distance,
        'actual_distance': actual_distance,
        'route_efficiency': efficiency
    })

# Appliquer le calcul
route_metrics = offensive_receivers.groupby(['game_id', 'play_id']).apply(calculate_route_efficiency).reset_index()

# Merger avec play_metrics
play_metrics = play_metrics.merge(route_metrics, on=['game_id', 'play_id'], how='left')

print(f"\n--- Statistiques de route efficiency ---")
print(f"Moyenne: {play_metrics['route_efficiency'].mean():.3f}")
print(f"Médiane: {play_metrics['route_efficiency'].median():.3f}")
print(f"Min/Max: {play_metrics['route_efficiency'].min():.3f} / {play_metrics['route_efficiency'].max():.3f}")

print(f"\nInterprétation:")
print(f"  - 1.0 = Route parfaitement directe")
print(f"  - 0.7-0.9 = Route avec courbes normales")
print(f"  - < 0.7 = Route très courbée (comeback, slant, etc.)")


# =============================================================================
# Visualisation 1 : Distribution des métriques
# =============================================================================

print(f"\n{'='*60}")
print("VISUALISATION DES MÉTRIQUES")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution du closing speed
axes[0, 0].hist(play_metrics['closing_speed'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Closing Speed (yards/frame)')
axes[0, 0].set_ylabel('Nombre de plays')
axes[0, 0].set_title('Distribution du Closing Speed')
axes[0, 0].axvline(play_metrics['closing_speed'].median(), color='red', linestyle='--', label='Médiane')
axes[0, 0].legend()

# 2. Distribution de la route efficiency
axes[0, 1].hist(play_metrics['route_efficiency'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Route Efficiency (ratio)')
axes[0, 1].set_ylabel('Nombre de plays')
axes[0, 1].set_title('Distribution de la Route Efficiency')
axes[0, 1].axvline(play_metrics['route_efficiency'].median(), color='red', linestyle='--', label='Médiane')
axes[0, 1].legend()

# 3. Relation closing speed vs vitesse moyenne
axes[1, 0].scatter(play_metrics['avg_speed'], play_metrics['closing_speed'], alpha=0.5)
axes[1, 0].set_xlabel('Vitesse moyenne (yards/s)')
axes[1, 0].set_ylabel('Closing Speed (yards/frame)')
axes[1, 0].set_title('Closing Speed vs Vitesse Moyenne')

# 4. Route efficiency par position
position_efficiency = play_metrics.groupby('player_position')['route_efficiency'].mean().sort_values()
axes[1, 1].barh(position_efficiency.index, position_efficiency.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 1].set_xlabel('Route Efficiency moyenne')
axes[1, 1].set_title('Efficacité de Route par Position')

plt.tight_layout()
plt.show()

print("\n✓ Graphiques générés")


# =============================================================================
# Top/Bottom performers
# =============================================================================

print(f"\n{'='*60}")
print("TOP/BOTTOM PERFORMERS")
print(f"{'='*60}")

print(f"\n--- Top 10 : Meilleur Closing Speed ---")
top_closing = play_metrics.nlargest(10, 'closing_speed')[
    ['game_id', 'play_id', 'player_name', 'player_position', 'closing_speed', 'avg_speed']
]
print(top_closing.to_string(index=False))

print(f"\n--- Top 10 : Meilleure Route Efficiency ---")
top_efficiency = play_metrics.nlargest(10, 'route_efficiency')[
    ['game_id', 'play_id', 'player_name', 'player_position', 'route_efficiency', 'direct_distance']
]
print(top_efficiency.to_string(index=False))


# =============================================================================
# Sauvegarde des métriques
# =============================================================================

print(f"\n{'='*60}")
print("SAUVEGARDE DES DONNÉES")
print(f"{'='*60}")

# Sauvegarder le DataFrame enrichi
print(f"\n✓ DataFrame 'play_metrics' créé avec {len(play_metrics)} plays")
print(f"✓ Colonnes disponibles: {play_metrics.columns.tolist()}")

