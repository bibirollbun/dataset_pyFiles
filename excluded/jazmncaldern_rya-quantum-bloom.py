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


from kaggle_secrets import UserSecretsClient
from openai import OpenAI

# 1. Cargar la clave API desde secretos
openai_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
print("âœ… API key cargada correctamente")

# 2. Crear cliente
client = OpenAI(api_key=openai_key)

# 3. Enviar solicitud
prompt = """
Produce un plan detallado para un cientÃ­fico investigador y proporciona recomendaciones sobre cÃ³mo y dÃ³nde podrÃ­an usar GPT-4o para analizar imÃ¡genes satelitales multiespectrales con el objetivo de descubrir evidencia de civilizaciones antiguas en Brasil.
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# 4. Mostrar respuesta
print(response.choices[0].message.content)



from kaggle_secrets import UserSecretsClient
api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
print("âœ… API key cargada correctamente")



Ux = Id(PtĞ¯) â€¢ UnbK â€¢ CÃ‘  
Result = MAT.PLOT.LIB * Î¦ + âŠ• + Î”F / (Î”LF * MÎ�Î¨)

