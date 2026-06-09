%%time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
print(seed_df.head())
print(submission_df.head())


submission_df.shape


seed_df.tail()


# Filtrar las filas de la temporada 2024
seed_2024 = seed_df[seed_df['Season'] == 2024]

# Contar las semillas únicas en la temporada 2024
unique_seeds_2024 = len(seed_2024['Seed'].unique())

# Contar los equipos únicos en la temporada 2024
unique_teams_2024 = len(seed_2024['TeamID'].unique())

# Imprimir los resultados
print(f'Número de semillas únicas en 2024: {unique_seeds_2024}')
print(f'Número de equipos únicos en 2024: {unique_teams_2024}')



# Crear un nuevo DataFrame para 2025 replicando las mismas semillas de 2024
seed_df_final = seed_df.copy()

# Aplicar el LabelEncoder si es necesario para convertir las semillas en valores numéricos
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
seed_df_final['SeedEncoded'] = label_encoder.fit_transform(seed_df_final['Seed'])
seed_df_final


seed_2023 = seed_df_final[seed_df_final['Season'] == 2023].reset_index(drop=True)

# Verificar el resultado
print(seed_2023.tail())


# Filtrar solo los datos de la temporada 2024
seed_2024 = seed_df_final[seed_df_final['Season'] == 2024].reset_index(drop=True)

# Verificar el resultado
print(seed_2024.tail())


# Crear un nuevo DataFrame para 2025 replicando las mismas semillas de 2024
seed_2025 = seed_2024.copy()

# Cambiar la columna 'Season' a 2025
seed_2025['Season'] = 2025

# Verificar el resultado
print(seed_2025.head())


# Concatenar seed_df_final con seed_2025 manteniendo seed_df_final
seed_df_combined = pd.concat([seed_df_final, seed_2025], ignore_index=True)

# Verificar el resultado
print(seed_df_combined.tail())



import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error


#w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
#m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
#seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
#submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')


submission_df.head(1)


seed_df.shape, seed_df_combined.shape, submission_df.shape


seed_df.head()


seed_df.tail()


submission_df.info()


submission_df.head()


'''
Esta funciona no devolvía una lista por lo que no se podia ejecutar bien el ,apply
def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')

    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    
    return year, teamID1, teamID2
'''

def extract_game_info(id_str):
    parts = id_str.split('_')
    return [int(parts[0]), int(parts[1]), int(parts[2])]

##No se requiere esta funcion ya qye se obtiene con labal_encoder
'''
def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
       return 16
'''

submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(lambda x: pd.Series(extract_game_info(x)))
submission_df.head(5)



# Aplicar la función y expandir los resultados en 3 columnas
#Season, TeamID1, TeamID2 = submission_df['ID'].apply(lambda x: pd.Series(extract_game_info(x)))




seed_df_combined


submission_df_back = submission_df.copy()


submission_df = submission_df_back.copy()


# Filtrar seed_df_combined para solo incluir los datos de la temporada 2025
seed_df_2025 = seed_df_combined[seed_df_combined['Season'] == 2025]

# Realizar el merge con submission_df
merged_df = pd.merge(submission_df, seed_df_2025, how='left', left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'])
merged_df = pd.merge(merged_df, seed_df_2025, how='left', left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], suffixes=('_Team1', '_Team2'))

# Eliminar las columnas 'TeamID' adicionales que fueron agregadas durante el merge
merged_df.drop(columns=['TeamID_Team1', 'TeamID_Team2'], inplace=True)

# Reemplazar NaN en las columnas Seed_Team1, Seed_Team2 con 0 o el valor que prefieras
merged_df['SeedEncoded_Team1'] = merged_df['SeedEncoded_Team1'].fillna(0.5)
merged_df['SeedEncoded_Team2'] = merged_df['SeedEncoded_Team2'].fillna(0.5)

# Verificar el resultado
print(merged_df.head())





# Mantener solo las columnas relevantes
final_df = merged_df[['ID', 'Pred', 'Season', 'TeamID1', 'TeamID2', 'SeedEncoded_Team1', 'SeedEncoded_Team2']]

# Verificar el resultado
print(final_df.head())



print(final_df.tail())


count_seedencoded_team1 = (final_df['SeedEncoded_Team1'] > 0).sum()
count_seedencoded_team2 = (final_df['SeedEncoded_Team2'] > 0).sum()

print(count_seedencoded_team1, count_seedencoded_team2)



final_df.shape


'''
# Merge para TeamID1 con la columna 'TeamID' en seed_df_combined
submission_df = pd.merge(submission_df, seed_df_combined[['TeamID', 'SeedEncoded']], 
                         left_on='TeamID1', right_on='TeamID', how='left')
submission_df = submission_df.rename(columns={'SeedEncoded': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge para TeamID2 con la columna 'TeamID' en seed_df_combined
submission_df = pd.merge(submission_df, seed_df_combined[['TeamID', 'SeedEncoded']], 
                         left_on='TeamID2', right_on='TeamID', how='left')
submission_df = submission_df.rename(columns={'SeedEncoded': 'SeedValue2'}).drop(columns=['TeamID'])

# Verificar los resultados
print(submission_df.head(5))
'''


submission_df


submission_df.shape, final_df.shape


submission_df = final_df.copy()



submission_df


'''
# Calculate seed difference
#submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']
submission_df['SeedDiff'] = submission_df['SeedEncoded_Team1'] - submission_df['SeedEncoded_Team2']


print(submission_df['SeedDiff'])


# Update 'Pred' column
submission_df['Pred'] = 0.5 + (0.03 * submission_df['SeedDiff'])

# Drop unnecessary columns
submission_df = submission_df[['ID', 'Pred']].fillna(0.5)

# Preview your submission
submission_df.head()
'''


# Asegurarse de que 'SeedDiff' se calcula correctamente
submission_df['SeedDiff'] = final_df['SeedEncoded_Team1'] - final_df['SeedEncoded_Team2']

# Actualizar la columna 'Pred' con base en 'SeedDiff'
submission_df['Pred'] = 0.5 + (0.03 * submission_df['SeedDiff'])

# Recortar los valores de 'Pred' para asegurarse de que estén entre 0 y 1
submission_df['Pred'] = submission_df['Pred'].clip(0, 1)

# Eliminar columnas innecesarias y rellenar NaN con 0.5
submission_df = submission_df[['ID', 'Pred']].fillna(0.5)

# Vista previa de la sumisión
submission_df.head(20)



count_pred_greather_than_0 = (submission_df['Pred'] > 0).sum()
count_pred_greather_than_0


submission_df['Pred'].unique()


stats = submission_df.iloc[:, 1].describe()
print(stats)


# Create a dataframe of ground truth values
solution_df = submission_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")


submission_df.to_csv('/kaggle/working/submission.csv', index=False)


print("End Program...")

