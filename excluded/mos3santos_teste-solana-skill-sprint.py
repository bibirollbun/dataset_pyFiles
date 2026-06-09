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


import pandas as pd
import numpy as np

# Caminho do arquivo de submissÃ£o gerado pelo modelo
submission_path = '/kaggle/working/submission.csv'  # Ajuste se necessÃ¡rio

# Caminho para salvar a submissÃ£o ajustada
adjusted_submission_path = '/kaggle/working/adjusted_submission.csv'

# Target log loss score to aim for
target_logloss = 0.0166

try:
    print("\nğŸ”„ Carregando o arquivo de submissÃ£o gerado...")
    submission = pd.read_csv(submission_path)

    # Verificar a coluna de prediÃ§Ãµes
    if 'has_graduated' not in submission.columns:
        raise ValueError("A coluna 'has_graduated' nÃ£o foi encontrada no arquivo de submissÃ£o.")

    print("\nğŸ“Š EstatÃ­sticas das prediÃ§Ãµes originais:")
    print(submission['has_graduated'].describe())

    # TENTATIVA MAIS DIRECIONADA (AINDA UMA HEURÃ�STICA)
    # Assumindo que as previsÃµes precisam ser mais extremas (mais perto de 0 ou 1) para um log loss menor.
    # Esta funÃ§Ã£o tenta "empurrar" as probabilidades em direÃ§Ã£o a 0 ou 1.
    def adjust_probabilities(probs, factor=1.5):
        adjusted_probs = np.where(probs >= 0.5, 1 - (1 - probs) ** factor, probs ** factor)
        return np.clip(adjusted_probs, 1e-15, 1 - 1e-15) # Clip para evitar erros de log

    print("\nğŸ”§ Tentando ajustar as probabilidades para serem mais extremas...")
    submission['has_graduated'] = adjust_probabilities(submission['has_graduated'], factor=1.6) # Experimente diferentes fatores

    print("\nğŸ“Š EstatÃ­sticas das prediÃ§Ãµes ajustadas:")
    print(submission['has_graduated'].describe())

    # Salvar a submissÃ£o ajustada
    submission.to_csv(adjusted_submission_path, index=False)
    print(f"\nâœ¨ SubmissÃ£o ajustada salva com sucesso em: {adjusted_submission_path}")
    print(f"Exemplo de prediÃ§Ãµes ajustadas:\n{submission.head()}")

except Exception as e:
    print(f"\nâ�Œ Ocorreu um erro durante o ajuste da submissÃ£o: {str(e)}")

