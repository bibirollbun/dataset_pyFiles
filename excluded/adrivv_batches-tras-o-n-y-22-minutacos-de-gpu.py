# ===================================================================
# 2.1 DESCARGA DEL DATASET TINY SHAKESPEARE
# ===================================================================

# Importamos las bibliotecas necesarias
import urllib.request  # Para descargar archivos desde URLs
import os              # Para operaciones del sistema de archivos

# URL del dataset Tiny Shakespeare en el repositorio de Andrej Karpathy
# Este es un archivo de texto plano que contiene todas las obras de Shakespeare
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# Nombre del archivo donde guardaremos el dataset localmente
filename = 'tiny_shakespeare.txt'

# Descargamos el archivo si no existe ya en el sistema
if not os.path.exists(filename):
    print(f'ğŸ“¥ Descargando dataset desde {url}...')
    urllib.request.urlretrieve(url, filename)
    print(f'âœ… Dataset descargado exitosamente como {filename}')
else:
    print(f'â„¹ï¸� El archivo {filename} ya existe, saltando descarga.')

# Leemos el contenido completo del archivo
# 'r' = modo lectura (read)
# 'utf-8' = codificaciÃ³n de caracteres estÃ¡ndar para texto
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

print('\n' + '='*70)
print('INFORMACIÃ“N BÃ�SICA DEL DATASET')
print('='*70)

# Mostramos estadÃ­sticas bÃ¡sicas del dataset
print(f'ğŸ“Š TamaÃ±o total del texto: {len(text):,} caracteres')
print(f'ğŸ“Š TamaÃ±o en bytes: {os.path.getsize(filename):,} bytes')
print(f'ğŸ“Š TamaÃ±o en KB: {os.path.getsize(filename) / 1024:.2f} KB')
print(f'ğŸ“Š TamaÃ±o en MB: {os.path.getsize(filename) / (1024*1024):.2f} MB')

# Contamos el nÃºmero de lÃ­neas
num_lines = text.count('\n')
print(f'ğŸ“‹ NÃºmero de lÃ­neas: {num_lines:,}')

# Mostramos las primeras 500 caracteres del texto para ver su estructura
print('\n' + '='*70)
print('MUESTRA DEL CONTENIDO (primeros 500 caracteres)')
print('='*70)
print(text[:500])
print('\n[...]\n')


# ===================================================================
# 2.2 ANÃ�LISIS DEL VOCABULARIO (CARACTERES ÃšNICOS)
# ===================================================================

# En un modelo de lenguaje a nivel de caracteres, nuestro "vocabulario" es
# el conjunto de todos los caracteres Ãºnicos que aparecen en el texto.
# Cada carÃ¡cter Ãºnico se convertirÃ¡ en un token.

# Obtenemos todos los caracteres Ãºnicos usando set() y los ordenamos
chars = sorted(list(set(text)))
vocab_size = len(chars)

print('='*70)
print('ANÃ�LISIS DEL VOCABULARIO')
print('='*70)
print(f'âœ�ï¸� TamaÃ±o del vocabulario: {vocab_size} caracteres Ãºnicos\n')

# Mostramos todos los caracteres Ãºnicos encontrados
print('ğŸ”¤ Caracteres Ãºnicos encontrados:')
print(''.join(chars))
print('\nğŸ”� Caracteres especiales encontrados:')

# Identificamos caracteres especiales (espacios, saltos de lÃ­nea, etc.)
special_chars = []
for char in chars:
    if char == '\n':
        special_chars.append('\\n (salto de lÃ­nea)')
    elif char == ' ':
        special_chars.append('[espacio]')
    elif char == '\t':
        special_chars.append('\\t (tabulaciÃ³n)')
    elif not char.isprintable():
        special_chars.append(f'\\x{ord(char):02x} (no imprimible)')

if special_chars:
    for sc in special_chars:
        print(f'  - {sc}')
else:
    print('  (ninguno)')

# Mostramos estadÃ­sticas de frecuencia de algunos caracteres
print('\nğŸ“Š Frecuencia de algunos caracteres comunes:')
common_chars = [' ', 'e', 'a', 'o', 'i', 't', 'n', 's', '\n']
for char in common_chars:
    if char in text:
        count = text.count(char)
        percentage = (count / len(text)) * 100
        char_display = '\\n' if char == '\n' else ('[espacio]' if char == ' ' else char)
        print(f'  {char_display}: {count:,} veces ({percentage:.2f}%)')


# ===================================================================
# 2.3 CREACIÃ“N DE MAPEOS (ENCODE/DECODE)
# ===================================================================

# Para entrenar un modelo de lenguaje, necesitamos convertir el texto (cadenas)
# en nÃºmeros (tensores). Esto se hace creando dos diccionarios:
# 1. stoi (string to integer): carÃ¡cter -> nÃºmero
# 2. itos (integer to string): nÃºmero -> carÃ¡cter

print('='*70)
print('CREACIÃ“N DE MAPEOS CARÃ�CTER â†” NÃšMERO')
print('='*70)

# Creamos un diccionario que mapea cada carÃ¡cter a un Ã­ndice Ãºnico
# enumerate() nos da pares (Ã­ndice, carÃ¡cter)
stoi = {ch: i for i, ch in enumerate(chars)}

# Creamos el diccionario inverso: Ã­ndice -> carÃ¡cter
itos = {i: ch for i, ch in enumerate(chars)}

print(f'âœ… Diccionario stoi (string to integer) creado con {len(stoi)} entradas')
print(f'âœ… Diccionario itos (integer to string) creado con {len(itos)} entradas\n')

# Mostramos algunos ejemplos de mapeo
print('ğŸ”� Ejemplos de mapeo carÃ¡cter -> nÃºmero (stoi):')
example_chars = ['a', 'b', 'c', ' ', '\n', 'A', 'B', 'C']
for ch in example_chars:
    if ch in stoi:
        display_char = '\\n' if ch == '\n' else ('[espacio]' if ch == ' ' else ch)
        print(f"  '{display_char}' -> {stoi[ch]}")

print('\nğŸ”� Ejemplos de mapeo nÃºmero -> carÃ¡cter (itos):')
for i in range(min(10, vocab_size)):
    display_char = '\\n' if itos[i] == '\n' else ('[espacio]' if itos[i] == ' ' else itos[i])
    print(f"  {i} -> '{display_char}'")

# Ahora creamos funciones para codificar (texto -> nÃºmeros)
# y decodificar (nÃºmeros -> texto)

def encode(s):
    """
    Codifica una cadena de texto a una lista de enteros.

    ParÃ¡metros:
        s (str): Cadena de texto a codificar

    Retorna:
        list: Lista de enteros donde cada entero representa un carÃ¡cter

    Ejemplo:
        >>> encode("hola")
        [56, 63, 60, 47]
    """
    return [stoi[c] for c in s]

def decode(l):
    """
    Decodifica una lista de enteros a una cadena de texto.

    ParÃ¡metros:
        l (list): Lista de enteros a decodificar

    Retorna:
        str: Cadena de texto reconstruida

    Ejemplo:
        >>> decode([56, 63, 60, 47])
        'hola'
    """
    return ''.join([itos[i] for i in l])

print('\nâœ… Funciones encode() y decode() creadas correctamente\n')

# Probamos las funciones con algunos ejemplos
print('='*70)
print('PRUEBAS DE CODIFICACIÃ“N/DECODIFICACIÃ“N')
print('='*70)

test_strings = [
    "Hello",
    "GPT",
    "To be or not to be",
    # "123" # Removed as '1' is not in the vocabulary
]

for test_str in test_strings:
    encoded = encode(test_str)
    decoded = decode(encoded)
    print(f'\nğŸ“‹ Texto original: "{test_str}"')
    print(f'   Codificado: {encoded}')
    print(f'   Decodificado: "{decoded}"')
    print(f'   âœ… Coincide: {test_str == decoded}')


# ===================================================================
# 2.4 CODIFICACIÃ“N DEL DATASET COMPLETO
# ===================================================================

# Ahora que tenemos nuestras funciones encode/decode, convertimos
# todo el texto en una secuencia de nÃºmeros (tensores)

import torch  # Importamos PyTorch para trabajar con tensores

print('='*70)
print('CODIFICACIÃ“N DEL DATASET')
print('='*70)

# Codificamos todo el texto a una lista de enteros
data = torch.tensor(encode(text), dtype=torch.long)

print(f'âœ… Dataset codificado exitosamente')
print(f'ğŸ“Š Forma del tensor: {data.shape}')
print(f'ğŸ“Š NÃºmero total de tokens: {len(data):,}')
print(f'ğŸ“Š Tipo de dato: {data.dtype}')

# Mostramos los primeros tokens
print(f'\nğŸ”� Primeros 100 tokens codificados:')
print(data[:100])

# Decodificamos de vuelta para verificar
print(f'\nğŸ”� DecodificaciÃ³n de los primeros 100 tokens:')
print(decode(data[:100].tolist()))

print('\nâœ… VerificaciÃ³n exitosa: la codificaciÃ³n y decodificaciÃ³n funcionan correctamente')


# ===================================================================
# 2.5 DIVISIÃ“N TRAIN/VALIDATION
# ===================================================================

# Es fundamental dividir nuestros datos en:
# - Training set (entrenamiento): Para entrenar el modelo
# - Validation set (validaciÃ³n): Para evaluar el modelo durante el entrenamiento
#   y evitar overfitting

# Usaremos una divisiÃ³n 90/10 (90% entrenamiento, 10% validaciÃ³n)
# Este es un ratio estÃ¡ndar para datasets pequeÃ±os

print('='*70)
print('DIVISIÃ“N DEL DATASET EN TRAIN/VALIDATION')
print('='*70)

# Calculamos el Ã­ndice donde hacer el corte (90% del dataset)
n = int(0.9 * len(data))

# Dividimos el tensor en dos partes
train_data = data[:n]  # Primeros 90%
val_data = data[n:]    # Ãšltimos 10%

print(f'âœ… Dataset dividido exitosamente\n')

print(f'ğŸ“‹ Dataset de entrenamiento (train):')
print(f'   TamaÃ±o: {len(train_data):,} tokens')
print(f'   Porcentaje: {(len(train_data) / len(data)) * 100:.2f}%')
print(f'   Forma: {train_data.shape}')

print(f'\nğŸ“‹ Dataset de validaciÃ³n (validation):')
print(f'   TamaÃ±o: {len(val_data):,} tokens')
print(f'   Porcentaje: {(len(val_data) / len(data)) * 100:.2f}%')
print(f'   Forma: {val_data.shape}')

print(f'\nğŸ“Š Total: {len(data):,} tokens')

# Mostramos una muestra de cada conjunto
print('\n' + '='*70)
print('MUESTRAS DE CADA CONJUNTO')
print('='*70)

print('\nğŸ”� Primeros 50 tokens del conjunto de entrenamiento:')
print(decode(train_data[:50].tolist()))

print('\nğŸ”� Primeros 50 tokens del conjunto de validaciÃ³n:')
print(decode(val_data[:50].tolist()))

print('\n' + '='*70)
print('âœ… PREPROCESAMIENTO COMPLETADO EXITOSAMENTE')
print('='*70)
print('\nğŸ�¯ Resumen:')
print(f'   â€¢ Dataset descargado: tiny_shakespeare.txt')
print(f'   â€¢ Caracteres totales: {len(text):,}')
print(f'   â€¢ Vocabulario: {vocab_size} caracteres Ãºnicos')
print(f'   â€¢ Tokens de entrenamiento: {len(train_data):,}')
print(f'   â€¢ Tokens de validaciÃ³n: {len(val_data):,}')
print(f'   â€¢ Funciones: encode() y decode() implementadas')
print('\nğŸš€ El dataset estÃ¡ listo para entrenar el modelo GPT!')


# ==================================================================
# 3.3 IMPLEMENTACIÃ“N: FUNCIÃ“N PARA GENERAR BATCHES
# ==================================================================
# Esta funciÃ³n es el corazÃ³n del entrenamiento. Genera batches aleatorios
# de datos para alimentar al modelo durante el entrenamiento.

import torch

# HiperparÃ¡metros para el entrenamiento
block_size = 8      # Longitud del contexto (nÃºmero de tokens que el modelo ve a la vez)
batch_size = 32     # NÃºmero de secuencias independientes a procesar en paralelo

print('='*70)
print('CONFIGURACIÃ“N DE HIPERPARÃ�METROS')
print('='*70)
print(f'ğŸ�¯ block_size (tamaÃ±o de contexto): {block_size} tokens')
print(f'ğŸ�¯ batch_size (tamaÃ±o de batch): {batch_size} secuencias')
print(f'ğŸ“Š Memoria por batch: {batch_size} x {block_size} = {batch_size * block_size} tokens')

################
# CAMBIO 1: BATCHES TRAS ESPACIOS O NEWLINES PARA MEJOR CONTEXTUALIZACION
def build_valid_starts(data, block_size, chars='\n '):
    """Encuentra todos los Ã­ndices donde empieza un token que es \n o espacio"""
    token_ids = set(encode(c)[0] for c in chars)  # IDs de \n y espacio
    # Encontramos dÃ³nde aparece cualquiera de esos tokens
    positions = torch.where(torch.isin(data[:-1], torch.tensor(list(token_ids))))[0]
    # El inicio vÃ¡lido es el SIGUIENTE token (para que el contexto empiece despuÃ©s del \n/espacio)
    valid_starts = (positions + 1).unique()
    # Filtramos los que dejan espacio para block_size + 1
    valid_starts = valid_starts[valid_starts + block_size < len(data)]
    return valid_starts

valid_train_starts = build_valid_starts(train_data, block_size, chars='\n ')
valid_val_starts   = build_valid_starts(val_data,   block_size, chars='\n ')

print(f"Original train tokens: {len(train_data)}")
print(f"VÃ¡lidos para batching limpio: {len(valid_train_starts)} ({100*len(valid_train_starts)/(len(train_data)-block_size):.1f}%)")
################

def get_batch(split):
    """
    Genera un batch aleatorio de datos para entrenamiento o validaciÃ³n.

    ParÃ¡metros:
        split (str): 'train' o 'val' para seleccionar el conjunto de datos

    Retorna:
        x (torch.Tensor): Tensor de entrada con forma (batch_size, block_size)
        y (torch.Tensor): Tensor de objetivos con forma (batch_size, block_size)

    ExplicaciÃ³n:
        - x contiene las secuencias de entrada (contextos)
        - y contiene los tokens objetivo (lo que el modelo debe predecir)
        - y[i, j] es el token que sigue inmediatamente a x[i, j]

    Ejemplo con batch_size=2, block_size=4:
        Si tenemos la secuencia [1, 2, 3, 4, 5, 6, 7, 8, 9]

        x podrÃ­a ser:  [[1, 2, 3, 4],    <- Secuencia 1
                        [5, 6, 7, 8]]    <- Secuencia 2

        y serÃ­a:      [[2, 3, 4, 5],    <- Objetivos para secuencia 1
                        [6, 7, 8, 9]]    <- Objetivos para secuencia 2

        El modelo aprende a predecir:
        - dado [1, 2, 3, 4], predecir [2, 3, 4, 5]
        - dado [5, 6, 7, 8], predecir [6, 7, 8, 9]
    """
    # # Seleccionamos el dataset correcto segÃºn el parÃ¡metro 'split'
    # data = train_data if split == 'train' else val_data

    # # Generamos batch_size Ã­ndices aleatorios
    # # Restamos block_size para asegurarnos de que hay espacio para los tokens objetivo
    # # torch.randint genera nÃºmeros aleatorios entre 0 y len(data) - block_size
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    
    ################
    # CAMBIO 1: BATCHES TRAS ESPACIOS O NEWLINES PARA MEJOR CONTEXTUALIZACION
    if split == 'train':
        starts = valid_train_starts
    else:
        starts = valid_val_starts
    ################
    
    # Elegimos batch_size Ã­ndices aleatorios entre los vÃ¡lidos
    ix = starts[torch.randint(len(starts), (batch_size,))]
    # # Creamos el tensor de entrada (x)
    # # Para cada Ã­ndice aleatorio, tomamos block_size tokens consecutivos
    # # torch.stack apila todos los tensores en una matriz
    x = torch.stack([data[i:i+block_size] for i in ix])

    # # Creamos el tensor de objetivos (y)
    # # Los objetivos son los mismos tokens, pero desplazados una posiciÃ³n hacia adelante
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])


    return x, y

print('\nâœ… FunciÃ³n get_batch() implementada correctamente')
print('\n' + '='*70)
print('PRUEBA DE LA FUNCIÃ“N get_batch()')
print('='*70)

# Generamos un batch de ejemplo para entrenamiento
xb, yb = get_batch('train')

print(f'\nğŸ“Š Forma del batch de entrada (x): {xb.shape}')
print(f'ğŸ“Š Forma del batch de objetivos (y): {yb.shape}')
print(f'\nğŸ”� InterpretaciÃ³n:')
print(f'   - {xb.shape[0]} secuencias independientes (batch_size)')
print(f'   - {xb.shape[1]} tokens por secuencia (block_size)')
print(f'   - Total: {xb.shape[0] * xb.shape[1]} tokens en este batch')

print('\n' + '='*70)
print('VISUALIZACIÃ“N DETALLADA DE UN BATCH')
print('='*70)
print('\nMostraremos las primeras 3 secuencias del batch para entender la estructura:\n')

for i in range(min(3, batch_size)):
    print(f'ğŸŸ¦ Secuencia {i+1}:')
    print(f'   Entrada (x): {xb[i].tolist()}')
    print(f'   Objetivo (y): {yb[i].tolist()}')
    print(f'   Texto de entrada: "{decode(xb[i].tolist())}"')
    print(f'   Texto objetivo: "{decode(yb[i].tolist())}"')
    print()


# ==================================================================
# 3.4 VISUALIZACIÃ“N: EXTRACCIÃ“N DE EJEMPLOS DE UN BLOQUE
# ==================================================================
# Demostraremos cÃ³mo de un solo bloque de tokens se extraen
# mÃºltiples ejemplos de entrenamiento con contextos de diferente longitud

print('='*70)
print('EXTRACCIÃ“N DE EJEMPLOS DE ENTRENAMIENTO DESDE UN BLOQUE')
print('='*70)

# Tomamos un bloque pequeÃ±o del conjunto de entrenamiento
block_sample = train_data[:block_size+1]

print(f'\nğŸ“� Bloque original ({block_size+1} tokens):')
print(f'   Tokens: {block_sample.tolist()}')
print(f'   Texto: "{decode(block_sample.tolist())}"')

print(f'\nğŸ“¦ De este bloque de {block_size+1} tokens, extraemos {block_size} ejemplos de entrenamiento:\n')

# Iteramos sobre el bloque y mostramos cada ejemplo de entrenamiento
for t in range(block_size):
    # El contexto son los primeros t+1 tokens
    context = block_sample[:t+1]

    # El objetivo es el token que sigue al contexto
    target = block_sample[t+1]

    # Convertimos a texto para visualizaciÃ³n
    context_text = decode(context.tolist())
    target_text = decode([target.item()])

    print(f'ğŸ”¹ Ejemplo {t+1}:')
    print(f'   Contexto (longitud {t+1}): {context.tolist()} = "{context_text}"')
    print(f'   Objetivo: {target.item()} = "{target_text}"')
    print(f'   â�¡ï¸� El modelo aprende: dado "{context_text}" â†’ predecir "{target_text}"')
    print()

print('='*70)
print('CONCLUSIONES IMPORTANTES')
print('='*70)
print(f'''
âœ… De un bloque de {block_size+1} tokens obtuvimos {block_size} ejemplos de entrenamiento

âœ… El modelo aprende a predecir con contextos de longitud variable:
   - Ejemplo 1: contexto de 1 token
   - Ejemplo 2: contexto de 2 tokens
   - ...
   - Ejemplo {block_size}: contexto de {block_size} tokens

âœ… Esto es fundamental porque:
   1. Durante la generaciÃ³n, el modelo empieza con pocos tokens y va creciendo
   2. Aprende patrones tanto locales (tokens cercanos) como distantes
   3. Maximiza el uso de los datos (cada bloque genera mÃºltiples ejemplos)

âœ… En un batch completo de {batch_size} secuencias:
   - Procesamos {batch_size} x {block_size} = {batch_size * block_size} tokens simultÃ¡neamente
   - Cada secuencia genera {block_size} ejemplos de entrenamiento
   - Total: {batch_size * block_size} predicciones por batch
''')

print('='*70)
print('âœ… SECCIÃ“N 3 COMPLETADA EXITOSAMENTE')
print('='*70)
print('\nğŸ�“ Has aprendido:')
print('   â€¢ Por quÃ© dividir datos en train/validation (evitar overfitting)')
print('   â€¢ QuÃ© son los tokens, contextos y batches')
print('   â€¢ CÃ³mo GPT aprende mediante predicciÃ³n del siguiente token')
print('   â€¢ CÃ³mo se extraen mÃºltiples ejemplos de un solo bloque')
print('   â€¢ CÃ³mo implementar la funciÃ³n get_batch() para entrenamiento')
print('\nğŸš€ Â¡Listo para construir la arquitectura del modelo GPT en la siguiente secciÃ³n!')


# ============================================================================
# 4.1 HIPERPARÃ�METROS PARA EL MODELO BIGRAM
# ============================================================================
# Configuraremos los hiperparÃ¡metros necesarios para entrenar el modelo bigram

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- HiperparÃ¡metros del modelo ---
# Ya definidos anteriormente, pero los reiteramos aquÃ­ para claridad
block_size = 8          # Contexto mÃ¡ximo que el modelo puede ver
batch_size = 32         # NÃºmero de secuencias procesadas en paralelo

# --- HiperparÃ¡metros de entrenamiento ---
max_iters = 3000        # NÃºmero total de iteraciones de entrenamiento
eval_interval = 300     # Cada cuÃ¡ntas iteraciones evaluar en validation
learning_rate = 1e-2    # Tasa de aprendizaje (0.01) - relativamente alta para modelo simple
eval_iters = 200        # NÃºmero de batches para promediar la pÃ©rdida de evaluaciÃ³n
# --- ConfiguraciÃ³n de dispositivo (CPU/GPU) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('='*70)
print('CONFIGURACIÃ“N DE HIPERPARÃ�METROS PARA MODELO BIGRAM')
print('='*70)
print('\nğŸ“Š ParÃ¡metros del modelo:')
print(f'   â€¢ Vocabulario (vocab_size): {vocab_size} caracteres')
print(f'   â€¢ TamaÃ±o de contexto (block_size): {block_size} tokens')
print(f'   â€¢ TamaÃ±o de batch (batch_size): {batch_size} secuencias')
print('\nğŸ�¯ ParÃ¡metros de entrenamiento:')
print(f'   â€¢ Iteraciones totales (max_iters): {max_iters:,}')
print(f'   â€¢ Intervalo de evaluaciÃ³n (eval_interval): {eval_interval}')
print(f'   â€¢ Tasa de aprendizaje (learning_rate): {learning_rate}')
print(f'   â€¢ Iteraciones de evaluaciÃ³n (eval_iters): {eval_iters}')
print('\nğŸ’» Dispositivo de cÃ³mputo:')
print(f'   â€¢ Usando: {device.upper()}')
if device == 'cuda':
    print(f'   â€¢ GPU detectada: {torch.cuda.get_device_name(0)}')
    print(f'   â€¢ Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('   â€¢ No se detectÃ³ GPU, usando CPU (el entrenamiento serÃ¡ mÃ¡s lento)')

print('\nâœ… HiperparÃ¡metros configurados correctamente')
print('='*70)


# ============================================================================
# 4.2 IMPLEMENTACIÃ“N DEL MODELO BIGRAM DE LENGUAJE
# ============================================================================

class BigramLanguageModel(nn.Module):
    """Modelo Bigram de Lenguaje - El modelo de lenguaje mÃ¡s simple."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print('=' * 70)
print('MODELO BIGRAM DE LENGUAJE IMPLEMENTADO')
print('=' * 70)
print('\nâœ… Clase BigramLanguageModel creada correctamente')

m = BigramLanguageModel(vocab_size)
m = m.to(device)
print(f'\nğŸš€ Modelo creado y movido a {device.upper()}')
print(f'\nğŸ“Š Resumen del modelo:')
print(f'   â€¢ Vocabulario: {vocab_size} tokens')
print(f'   â€¢ ParÃ¡metros: {sum(p.numel() for p in m.parameters()):,}')
print(f'   â€¢ Dispositivo: {device.upper()}')
print('=' * 70)


# ============================================================================
# 4.3 FUNCIÃ“N PARA ESTIMAR LA PÃ‰RDIDA EN TRAIN Y VALIDATION
# ============================================================================
# Esta funciÃ³n evaluarÃ¡ el rendimiento del modelo en ambos conjuntos de datos
# sin actualizar los parÃ¡metros del modelo

@torch.no_grad()
def estimate_loss():
    """
    Estima la pÃ©rdida promedio del modelo en los conjuntos de entrenamiento y validaciÃ³n.

    Esta funciÃ³n es crucial para:
    1. Monitorear el progreso del entrenamiento
    2. Detectar overfitting (si train loss << val loss)
    3. Decidir cuÃ¡ndo detener el entrenamiento (early stopping)

    Usamos @torch.no_grad() para:
    - Desactivar el cÃ¡lculo de gradientes (no estamos entrenando)
    - Ahorrar memoria y acelerar el cÃ¡lculo
    - Evitar que estas evaluaciones afecten el entrenamiento

    Returns:
        dict: Diccionario con las pÃ©rdidas promedio
              {'train': train_loss, 'val': val_loss}

    Proceso:
    1. Pone el modelo en modo evaluaciÃ³n
    2. Calcula la pÃ©rdida en eval_iters batches de train
    3. Calcula la pÃ©rdida en eval_iters batches de validation
    4. Promedia las pÃ©rdidas
    5. Regresa el modelo a modo entrenamiento
    """
    out = {}

    # Ponemos el modelo en modo evaluaciÃ³n
    # Esto desactiva dropout y otras capas que se comportan diferente en train/eval
    # (aunque nuestro modelo bigram simple no tiene estas capas)
    m.eval()

    # Evaluamos en ambos conjuntos: train y validation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        # Calculamos la pÃ©rdida en eval_iters batches diferentes
        for k in range(eval_iters):
            # Obtenemos un batch aleatorio
            X, Y = get_batch(split)

            # Movemos los datos al dispositivo correcto (GPU o CPU)
            X, Y = X.to(device), Y.to(device)

            # Forward pass: calculamos las predicciones y la pÃ©rdida
            logits, loss = m(X, Y)

            # Guardamos la pÃ©rdida de este batch
            losses[k] = loss.item()

        # Calculamos la pÃ©rdida promedio para este conjunto
        out[split] = losses.mean()

    # Regresamos el modelo a modo entrenamiento
    m.train()

    return out

print('='*70)
print('FUNCIÃ“N DE EVALUACIÃ“N estimate_loss() IMPLEMENTADA')
print('='*70)
print('\nâœ… FunciÃ³n estimate_loss() creada correctamente')
print('\nğŸ”� Â¿QuÃ© hace esta funciÃ³n?')
print(f'   â€¢ EvalÃºa el modelo en {eval_iters} batches de train y validation')
print('   â€¢ Calcula la pÃ©rdida promedio en cada conjunto')
print('   â€¢ No actualiza los parÃ¡metros del modelo (@torch.no_grad())')
print('   â€¢ Nos ayuda a detectar overfitting y monitorear el progreso')
print('\nğŸ“Š Â¿QuÃ© significa la pÃ©rdida?')
print('   â€¢ PÃ©rdida baja (â‰ˆ 0): El modelo predice muy bien')
print('   â€¢ PÃ©rdida alta (â‰ˆ 4.17): El modelo predice al azar (como adivinar)')
print('   â€¢ PÃ©rdida de adivinar = -log(1/vocab_size) = -log(1/65) â‰ˆ 4.17')
print('\nâš ï¸� Signos de overfitting:')
print('   â€¢ train_loss < val_loss (esperado y normal)')
print('   â€¢ train_loss << val_loss (gran diferencia = overfitting severo)')
print('   â€¢ val_loss aumenta mientras train_loss disminuye (overfitting)')
print('\nâœ… Ã‰xito en entrenamiento:')
print('   â€¢ Ambas pÃ©rdidas disminuyen con el tiempo')
print('   â€¢ val_loss se mantiene cercana a train_loss')
print('   â€¢ Ambas pÃ©rdidas convergen a un valor bajo')
print('='*70)



# ============================================================================
# 4.4 BUCLE DE ENTRENAMIENTO DEL MODELO BIGRAM
# ============================================================================
# Implementaremos el bucle de entrenamiento completo que:
# 1. Obtiene batches de datos
# 2. Calcula las predicciones del modelo
# 3. Calcula la pÃ©rdida
# 4. Actualiza los parÃ¡metros mediante backpropagation
# 5. EvalÃºa periÃ³dicamente en el conjunto de validaciÃ³n

import math
import json
from pathlib import Path
from time import perf_counter

model_params = sum(p.numel() for p in m.parameters())
n_embd_cfg = globals().get('n_embd')
n_layer_cfg = globals().get('n_layer')
n_head_cfg = globals().get('n_head')

print('='*70)
print('ENTRENAMIENTO DEL MODELO BIGRAM')
print('='*70)
print('\nğŸ�¯ Iniciando entrenamiento...')
print(f'   â€¢ {max_iters:,} iteraciones totales')
print(f'   â€¢ EvaluaciÃ³n cada {eval_interval} iteraciones')
print(f'   â€¢ Tasa de aprendizaje: {learning_rate}')
print(f'   â€¢ ParÃ¡metros del modelo: {model_params:,}')
if n_embd_cfg is not None:
    print(f'   â€¢ DimensiÃ³n de embeddings (n_embd): {n_embd_cfg}')
if n_layer_cfg is not None:
    print(f'   â€¢ Bloques Transformer (n_layer): {n_layer_cfg}')
if n_head_cfg is not None:
    print(f'   â€¢ NÃºmero de cabezas (n_head): {n_head_cfg}')
print('\nğŸ› ï¸� Preparando optimizador y registro de mÃ©tricas...')

# Creamos el optimizador
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Directorio y estructuras para registrar mÃ©tricas de entrenamiento
artifact_dir = Path('artifacts')
artifact_dir.mkdir(parents=True, exist_ok=True)
history_path = artifact_dir / 'training_history.json'

training_history = []
best_val_loss = float('inf')
t0 = perf_counter()

tokens_per_iter = batch_size * block_size

# Bucle principal de entrenamiento
for iter in range(max_iters):

    # --- EVALUACIÃ“N PERIÃ“DICA ---
    # Cada eval_interval iteraciones, evaluamos el modelo en train y val
    if iter % eval_interval == 0:
        losses = estimate_loss()
        elapsed_s = perf_counter() - t0

        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        lr = optimizer.param_groups[0]['lr']

        tokens_processed = iter * tokens_per_iter
        tokens_per_sec = None
        if elapsed_s > 0 and tokens_processed > 0:
            tokens_per_sec = tokens_processed / elapsed_s
        best_val_loss = min(best_val_loss, val_loss)

        training_history.append({
            'iter': iter,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_minus_train': val_loss - train_loss,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl,
            'learning_rate': lr,
            'elapsed_s': elapsed_s,
            'tokens_processed': tokens_processed,
            'tokens_per_sec': tokens_per_sec,
            'best_val_loss_so_far': best_val_loss,
            'batch_size': batch_size,
            'block_size': block_size,
            'n_embd': n_embd_cfg,
            'n_layer': n_layer_cfg,
            'n_head': n_head_cfg,
            'model_params': model_params,
        })

        print(f'\nğŸ“Š IteraciÃ³n {iter:,}/{max_iters:,}:')
        print(f'   â€¢ PÃ©rdida de entrenamiento: {train_loss:.4f}')
        print(f'   â€¢ PÃ©rdida de validaciÃ³n: {val_loss:.4f}')
        print(f'   â€¢ Perplexity (train): {train_ppl:.2f}')
        print(f'   â€¢ Perplexity (val): {val_ppl:.2f}')
        print(f'   â€¢ Tokens procesados: {tokens_processed:,}')
        if tokens_per_sec is not None:
            print(f'   â€¢ Tokens por segundo: {tokens_per_sec:,.0f}')

        diff = val_loss - train_loss
        if diff > 0.5:
            print(f'   âš ï¸�  Posible overfitting detectado (diferencia: {diff:.4f})')
        elif diff > 0.1:
            print(f'   ğŸŸ¡ Ligero overfitting (diferencia: {diff:.4f})')
        else:
            print('   âœ… Buen equilibrio entre train y val')

    # --- PASO 1: OBTENER UN BATCH DE DATOS ---
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # --- PASO 2: FORWARD PASS ---
    logits, loss = m(xb, yb)

    # --- PASO 3: BACKPROPAGATION ---
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # --- PASO 4: ACTUALIZACIÃ“N DE PARÃ�METROS ---
    optimizer.step()

# EvaluaciÃ³n final
print('\n' + '='*70)
print('ENTRENAMIENTO COMPLETADO')
print('='*70)
final_losses = estimate_loss()
final_elapsed = perf_counter() - t0
final_train_loss = final_losses['train'].item()
final_val_loss = final_losses['val'].item()
final_train_ppl = math.exp(final_train_loss)
final_val_ppl = math.exp(final_val_loss)

print('\nğŸ�¯ Resultados finales:')
print(f'   â€¢ PÃ©rdida de entrenamiento final: {final_train_loss:.4f}')
print(f'   â€¢ PÃ©rdida de validaciÃ³n final: {final_val_loss:.4f}')
print(f'   â€¢ Diferencia (val - train): {final_val_loss - final_train_loss:.4f}')
print(f'   â€¢ Perplexity (train): {final_train_ppl:.2f}')
print(f'   â€¢ Perplexity (val): {final_val_ppl:.2f}')
print(f'   â€¢ Tiempo total: {final_elapsed/60:.2f} min')

# Contexto: Â¿QuÃ© significa la pÃ©rdida?
print('\nğŸ“Š InterpretaciÃ³n de la pÃ©rdida:')
print('   â€¢ PÃ©rdida aleatoria (adivinar): ~4.17 (-log(1/65))')
print(f'   â€¢ Nuestra pÃ©rdida: ~{final_val_loss:.2f}')
improvement = (4.17 - final_val_loss) / 4.17 * 100
print(f'   â€¢ Mejora respecto a adivinar: {improvement:.1f}%')

print('\nâœ… Modelo Bigram entrenado exitosamente')
print('='*70)

# Registramos la evaluaciÃ³n final como Ãºltimo punto de la historia
final_tokens = max_iters * tokens_per_iter
final_tokens_per_sec = None
if final_elapsed > 0 and final_tokens > 0:
    final_tokens_per_sec = final_tokens / final_elapsed

training_history.append({
    'iter': max_iters,
    'train_loss': final_train_loss,
    'val_loss': final_val_loss,
    'val_minus_train': final_val_loss - final_train_loss,
    'train_ppl': final_train_ppl,
    'val_ppl': final_val_ppl,
    'learning_rate': optimizer.param_groups[0]['lr'],
    'elapsed_s': final_elapsed,
    'tokens_processed': final_tokens,
    'tokens_per_sec': final_tokens_per_sec,
    'best_val_loss_so_far': min(best_val_loss, final_val_loss),
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embd': n_embd_cfg,
    'n_layer': n_layer_cfg,
    'n_head': n_head_cfg,
    'model_params': model_params,
})

# Guardamos la historia completa para reutilizarla en visualizaciones
with history_path.open('w', encoding='utf-8') as f:
    json.dump(training_history, f, indent=2)

print(f"\nğŸ’¾ Historial de entrenamiento guardado en: {history_path.resolve()}")




import torch
import torch.nn as nn
from torch.nn import functional as F

# ConfiguraciÃ³n para Self-Attention
n_embd = 64       # DimensiÃ³n de embeddings
head_size = 16    # TamaÃ±o de cada cabeza de atenciÃ³n

class Head(nn.Module):
    """
    Una sola cabeza de self-attention.

    Self-attention permite que cada posiciÃ³n en la secuencia "mire" a todas
    las otras posiciones y determine quÃ© tan relevante es cada una para
    hacer su predicciÃ³n.

    Componentes:
    - Query (Q): "Â¿QuÃ© estoy buscando?"
    - Key (K): "Â¿QuÃ© informaciÃ³n tengo?"
    - Value (V): "Â¿CuÃ¡l es mi contenido real?"

    El mecanismo calcula attention_weights = softmax(Q @ K^T / sqrt(d_k))
    Y luego combina los values: output = attention_weights @ V
    """

    def __init__(self, head_size):
        super().__init__()
        # Matrices lineales que transforman el input en Q, K, V
        # Cada una mapea de n_embd dimensiones a head_size dimensiones
        self.key = nn.Linear(n_embd, head_size, bias=False)    # X -> K
        self.query = nn.Linear(n_embd, head_size, bias=False)  # X -> Q
        self.value = nn.Linear(n_embd, head_size, bias=False)  # X -> V

        # Registro de mÃ¡scara triangular inferior para atenciÃ³n causal
        # Esto asegura que el token en posiciÃ³n i solo puede atender
        # a tokens en posiciones j <= i (no puede "ver el futuro")
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Aplica self-attention a la entrada.

        Args:
            x: Tensor de forma (B, T, C) donde:
               B = batch_size, T = block_size, C = n_embd

        Returns:
            out: Tensor de forma (B, T, head_size) con la informaciÃ³n
                 procesada por attention
        """
        B, T, C = x.shape

        # Paso 1: Generar matrices Q, K, V
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Paso 2: Calcular puntuaciones de atenciÃ³n (compatibilidades)
        # Q @ K^T nos da una matriz donde entry (i,j) indica
        # quÃ© tan compatible es el query en posiciÃ³n i con el key en posiciÃ³n j
        weights = q @ k.transpose(-2, -1) # (B, T, T)

        # Paso 3: Escalar por sqrt(head_size) para estabilidad numÃ©rica
        # Sin esto, las puntuaciones pueden volverse muy grandes y causar
        # gradientes que se desvanecen despuÃ©s del softmax
        weights = weights * (head_size ** -0.5)

        # Paso 4: Aplicar mÃ¡scara causal (triangular inferior)
        # Ponemos -infinito en posiciones donde no deberÃ­a haber atenciÃ³n
        # Esto asegura que despuÃ©s del softmax, esas posiciones tengan probabilidad 0
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Paso 5: Aplicar softmax para obtener probabilidades de atenciÃ³n
        # Cada fila suma 1, representando cÃ³mo se distribuye la atenciÃ³n
        weights = F.softmax(weights, dim=-1) # (B, T, T)

        # Paso 6: Aplicar atenciÃ³n a los values
        # Combinamos los values usando las probabilidades de atenciÃ³n
        out = weights @ v # (B, T, head_size)

        return out

print("âœ… Clase Head (Self-Attention) implementada correctamente")


# Vamos a probar nuestro mecanismo de Self-Attention
# Primero creamos algunos datos de ejemplo

torch.manual_seed(1337)  # Para reproducibilidad

# Creamos un batch pequeÃ±o de ejemplo
B, T, C = 4, 8, 32  # batch_size=4, block_size=8, embedding_dim=32

# Simulamos embeddings de tokens (normalmente vienen de una tabla de embeddings)
x = torch.randn(B, T, C)  # (4, 8, 32) - datos aleatorios que simulan embeddings

# Creamos una cabeza de atenciÃ³n con head_size=16
head_size = 16
n_embd = 32  # Actualizamos para coincidir con nuestro ejemplo
block_size = 8  # Actualizamos para coincidir

# Instanciamos nuestra cabeza de Self-Attention
attention_head = Head(head_size)

print('=' * 70)
print('PRUEBA DE SELF-ATTENTION')
print('=' * 70)

print(f'\nğŸ”� ConfiguraciÃ³n:')
print(f'   â€¢ Batch size: {B}')
print(f'   â€¢ Sequence length: {T}')
print(f'   â€¢ Embedding dimension: {C}')
print(f'   â€¢ Head size: {head_size}')

# Aplicamos Self-Attention
with torch.no_grad():  # No calculamos gradientes para la prueba
    output = attention_head(x)

print(f'\nğŸ“Š Resultados:')
print(f'   â€¢ Input shape: {x.shape}')
print(f'   â€¢ Output shape: {output.shape}')
print(f'   â€¢ ReducciÃ³n dimensional: {C} â†’ {head_size}')

# Veamos cÃ³mo cambian las activaciones
print(f'\nğŸ”� EstadÃ­sticas:')
print(f'   â€¢ Input mean: {x.mean():.4f}, std: {x.std():.4f}')
print(f'   â€¢ Output mean: {output.mean():.4f}, std: {output.std():.4f}')

# Visualicemos los pesos de atenciÃ³n para una secuencia
with torch.no_grad():
    # Recalculamos para obtener los pesos de atenciÃ³n
    k = attention_head.key(x[0:1])  # Solo la primera secuencia
    q = attention_head.query(x[0:1])

    # Calculamos los pesos de atenciÃ³n
    weights = q @ k.transpose(-2, -1) * (head_size ** -0.5)
    weights = weights.masked_fill(attention_head.tril[:T, :T] == 0, float('-inf'))
    attention_weights = F.softmax(weights, dim=-1)

print(f'\nğŸ�­ Matriz de AtenciÃ³n (primera secuencia):')
print('Filas = posiciones que atienden, Columnas = posiciones atendidas')
print('Valores mÃ¡s altos = mayor atenciÃ³n')
print(attention_weights[0].numpy())

print('\nâœ… Self-Attention funciona correctamente!')
print('\nğŸ”� Observaciones importantes:')
print('   â€¢ Cada fila suma 1.0 (softmax)')
print('   â€¢ La matriz es triangular inferior (atenciÃ³n causal)')
print('   â€¢ Posiciones posteriores tienen 0 atenciÃ³n (no pueden ver el futuro)')


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementa mÃºltiples cabezas de atenciÃ³n en paralelo.

    En lugar de tener una sola cabeza de atenciÃ³n que procesa toda la

    informaciÃ³n, dividimos el espacio de embeddings en mÃºltiples "cabezas"
    que pueden especializarse en diferentes tipos de relaciones.

    Arquitectura:
    1. Dividir el input en n_head porciones
    2. Aplicar atenciÃ³n a cada porciÃ³n independientemente
    3. Concatenar todas las salidas
    4. Aplicar una proyecciÃ³n final
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        # Aplicamos cada cabeza de atenciÃ³n y concatenamos los resultados
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, head_size * num_heads)
        out = self.proj(out)  # ProyecciÃ³n final (B, T, n_embd)
        return out


class FeedFoward(nn.Module):
    """
    Feedforward Network (FFN) para el bloque Transformer.

    La FFN es una red completamente conectada que procesa cada token
    independientemente. Consta de dos transformaciones lineales con
    una activaciÃ³n ReLU en el medio.

    Arquitectura:
        Input (n_embd) -> Linear (4*n_embd) -> ReLU -> Linear (n_embd) -> Output

    El factor de expansiÃ³n 4x es estÃ¡ndar en Transformers:
    - Permite representaciones intermedias mÃ¡s ricas
    - El cuello de botella final fuerza compresiÃ³n de informaciÃ³n
    """

    def __init__(self, n_embd):
        super().__init__()
        # Red feedforward de 2 capas con expansiÃ³n 4x
        # Capa 1: ExpansiÃ³n (n_embd -> 4*n_embd)
        # Capa 2: CompresiÃ³n (4*n_embd -> n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expandir a 4x las dimensiones
            nn.ReLU(),                      # ActivaciÃ³n no lineal
            nn.Linear(4 * n_embd, n_embd),  # Comprimir de vuelta a n_embd
        )

    def forward(self, x):
        """
        Aplica la feedforward network a cada token independientemente.

        Args:
            x: Input tensor de forma (B, T, n_embd)

        Returns:
            out: Output tensor de forma (B, T, n_embd)
        """
        return self.net(x)

print("âœ… Clase FeedFoward implementada correctamente")


# Ejemplo de cÃ³mo fluye la informaciÃ³n a travÃ©s de la FFN
print('=' * 60)
print('ğŸ§ª VISUALIZACIÃ“N DEL FLUJO EN FFN')
print('=' * 60)

# ParÃ¡metros de ejemplo
n_embd = 64
B, T = 4, 8  # batch_size, block_size

# Crear algunos datos de prueba
x_test = torch.randn(B, T, n_embd)
ffn = FeedFoward(n_embd)

print(f'ğŸ“Š ConfiguraciÃ³n:')
print(f'   Input shape: {x_test.shape} (batch_size={B}, seq_len={T}, n_embd={n_embd})')
print(f'   FFN expansion factor: 4x')
print(f'   Hidden size: {4 * n_embd} dimensiones')

# Forward pass
with torch.no_grad():
    output = ffn(x_test)

print(f'ğŸ”„ Flujo de datos:')
print(f'   1. Input:         {x_test.shape} -> {n_embd} dims per token')
print(f'   2. Linear expand: {n_embd} -> {4 * n_embd} dims')
print(f'   3. ReLU:          No change in shape, adds non-linearity')
print(f'   4. Linear compress: {4 * n_embd} -> {n_embd} dims')
print(f'   5. Output:        {output.shape} -> Same as input')

print(f'ğŸ“Š EstadÃ­sticas:')
print(f'   Input  - mean: {x_test.mean():.4f}, std: {x_test.std():.4f}')
print(f'   Output - mean: {output.mean():.4f}, std: {output.std():.4f}')

print('âœ… FFN procesa cada posiciÃ³n independientemente')
print('âœ… Mantiene la forma del tensor de entrada')
print('âœ… AÃ±ade capacidad no-lineal al modelo')


class LayerNorm(nn.Module):
    """
    Layer Normalization para estabilizar el entrenamiento.

    A diferencia de Batch Normalization que normaliza a travÃ©s del batch,
    Layer Normalization normaliza a travÃ©s de las features de cada ejemplo.

    Esto es especialmente importante en secuencias donde:
    - Cada token necesita normalizaciÃ³n independiente
    - No queremos dependencia del tamaÃ±o del batch
    - Queremos consistencia entre entrenamiento e inferencia

    FÃ³rmula: LN(x) = Î³ * (x - Î¼) / Ïƒ + Î²
    """

    def __init__(self, ndim, bias=True):
        super().__init__()
        # ParÃ¡metros aprendibles: scale (gamma) y shift (beta)
        self.weight = nn.Parameter(torch.ones(ndim))  # gamma (inicializado en 1)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # beta (inicializado en 0)

    def forward(self, input):
        # Layer norm normaliza en la Ãºltima dimensiÃ³n (features)
        # Para input de forma (B, T, C), normaliza en C
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)

print("âœ… Clase LayerNorm implementada correctamente")


class PositionalEmbedding(nn.Module):
    """
    Embeddings posicionales aprendibles para Transformers.

    Los transformers no tienen nociÃ³n inherente del orden de los tokens.
    Los positional embeddings aÃ±aden informaciÃ³n sobre la posiciÃ³n de cada
    token en la secuencia.

    ImplementaciÃ³n:
    - Tabla de lookup similar a token embeddings
    - Cada posiciÃ³n (0, 1, 2, ..., block_size-1) tiene un vector Ãºnico
    - Se suma al token embedding correspondiente

    Ventajas de embeddings aprendibles vs sinusoidales:
    + Se optimizan durante el entrenamiento para la tarea especÃ­fica
    + Suelen funcionar mejor en prÃ¡ctica para secuencias cortas-medias
    - Limitados a secuencias â‰¤ block_size durante entrenamiento
    """

    def __init__(self, block_size, n_embd):
        super().__init__()
        # Tabla de embeddings posicionales
        # Cada posiciÃ³n (0 a block_size-1) tiene un vector de n_embd dims
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        """
        AÃ±ade embeddings posicionales a los tokens.

        Args:
            idx: Ã�ndices de tokens con forma (B, T)

        Returns:
            pos_emb: Embeddings posicionales con forma (T, n_embd)
                    Se pueden sumar directamente a token embeddings
        """
        B, T = idx.shape

        # Creamos Ã­ndices de posiciÃ³n: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=idx.device)  # (T,)

        # Obtenemos los embeddings posicionales
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)

        return pos_emb

print("âœ… Clase PositionalEmbedding implementada correctamente")


# DemostraciÃ³n de cÃ³mo funcionan los positional embeddings
print('=' * 60)
print('ğŸ§ª VISUALIZACIÃ“N DE POSITIONAL EMBEDDINGS')
print('=' * 60)

# ParÃ¡metros de ejemplo
block_size = 8
n_embd = 64
B = 2  # batch size pequeÃ±o

# Creamos algunos Ã­ndices de tokens de ejemplo
idx_example = torch.tensor([
    [15, 20, 5, 18, 25, 7, 12, 3],      # Secuencia 1
    [8, 30, 14, 22, 9, 11, 6, 28]       # Secuencia 2
])

print(f'ğŸ“Š ConfiguraciÃ³n:')
print(f'   Input tokens shape: {idx_example.shape}')
print(f'   Block size (max positions): {block_size}')
print(f'   Embedding dimension: {n_embd}')

# Instanciamos positional embeddings
pos_emb_layer = PositionalEmbedding(block_size, n_embd)

# Aplicamos positional embeddings
with torch.no_grad():
    pos_embeddings = pos_emb_layer(idx_example)  # (T, n_embd)

print(f'\nğŸ”„ Resultados:')
print(f'   Positional embeddings shape: {pos_embeddings.shape}')
print(f'   PosiciÃ³n 0 (primer token): vector de {pos_embeddings[0].shape} dims')
print(f'   PosiciÃ³n 1 (segundo token): vector de {pos_embeddings[1].shape} dims')
print(f'   ... hasta posiciÃ³n {block_size-1}')

print(f'\nğŸ”� Proceso de combinaciÃ³n:')
print('   1. Token embeddings: cada token â†’ vector semÃ¡ntico')
print('   2. Position embeddings: cada posiciÃ³n â†’ vector posicional')
print('   3. Suma: token_emb + pos_emb â†’ representaciÃ³n completa')

# Ejemplo de cÃ³mo se verÃ­a la suma (simulada)
print(f'\nğŸ¤� Ejemplo de combinaciÃ³n:')
print('   Token "hello" en posiciÃ³n 0:')
print('     â€¢ Token embedding: [0.1, -0.3, 0.8, ...]  (significado)')
print('     â€¢ Position embedding: [0.2, 0.1, -0.1, ...]  (posiciÃ³n 0)')
print('     â€¢ Combinado: [0.3, -0.2, 0.7, ...]  (significado + posiciÃ³n)')

print('\n   Token "hello" en posiciÃ³n 3:')
print('     â€¢ Token embedding: [0.1, -0.3, 0.8, ...]  (mismo significado)')
print('     â€¢ Position embedding: [-0.1, 0.4, 0.2, ...]  (posiciÃ³n 3, diferente)')
print('     â€¢ Combinado: [0.0, 0.1, 1.0, ...]  (significado + posiciÃ³n diferente)')

print('\nâœ… Ahora el modelo puede distinguir tokens idÃ©nticos en posiciones diferentes')
print('âœ… Cada posiciÃ³n tiene una "firma" Ãºnica aprendible')
print('âœ… Se preserva tanto el significado como el orden')


class Block(nn.Module):
    """
    Bloque Transformer completo.

    Un bloque Transformer es la unidad fundamental que combina:
    1. Multi-Head Self-Attention con residual connection y layer norm
    2. Feedforward Network con residual connection y layer norm

    Arquitectura (Pre-Layer Norm):
        x = x + multi_head_attention(layer_norm(x))
        x = x + feedforward(layer_norm(x))

    Esta arquitectura permite:
    - Flujo estable de gradientes
    - Representaciones ricas que combinan attention y processing
    - Escalabilidad a redes muy profundas
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # Calculamos el tamaÃ±o de cada cabeza
        head_size = n_embd // n_head

        # Componentes del bloque
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention
        self.ffwd = FeedFoward(n_embd)                   # Feedforward
        self.ln1 = LayerNorm(n_embd)                     # Layer norm 1
        self.ln2 = LayerNorm(n_embd)                     # Layer norm 2

    def forward(self, x):
        # Aplicamos Pre-Layer Norm con residual connections
        # PatrÃ³n: x = x + transformacion(layer_norm(x))

        # 1. Self-attention block
        x = x + self.sa(self.ln1(x))    # Residual + Multi-Head Attention

        # 2. Feedforward block
        x = x + self.ffwd(self.ln2(x))  # Residual + FFN

        return x

print("âœ… Clase Block (Bloque Transformer) implementada correctamente")


# DemostraciÃ³n del bloque Transformer completo
print('=' * 70)
print('ğŸ§ª VISUALIZACIÃ“N DEL BLOQUE TRANSFORMER')
print('=' * 70)

# ConfiguraciÃ³n
n_embd = 64   # DimensiÃ³n de embeddings
n_head = 4    # NÃºmero de cabezas de atenciÃ³n
B, T = 2, 8   # Batch y secuencia pequeÃ±os

print(f'ğŸ“Š ConfiguraciÃ³n del bloque:')
print(f'   â€¢ Embedding dimension: {n_embd}')
print(f'   â€¢ Number of attention heads: {n_head}')
print(f'   â€¢ Head size: {n_embd // n_head}')
print(f'   â€¢ Input shape: (B={B}, T={T}, C={n_embd})')

# Datos de entrada simulados
torch.manual_seed(42)
x_input = torch.randn(B, T, n_embd)

# Crear bloque Transformer
transformer_block = Block(n_embd, n_head)

print(f'\nğŸ”„ Flujo a travÃ©s del bloque:')
print(f'   1. Input: {x_input.shape}')
print(f'      Mean: {x_input.mean():.4f}, Std: {x_input.std():.4f}')

with torch.no_grad():
    # Paso intermedio 1: Layer norm + Self-attention
    x_ln1 = transformer_block.ln1(x_input)
    x_attn = transformer_block.sa(x_ln1)
    x_after_attn = x_input + x_attn  # Residual connection

    print(f'\n   2. After Self-Attention + Residual: {x_after_attn.shape}')
    print(f'      Mean: {x_after_attn.mean():.4f}, Std: {x_after_attn.std():.4f}')

    # Paso intermedio 2: Layer norm + Feedforward
    x_ln2 = transformer_block.ln2(x_after_attn)
    x_ffwd = transformer_block.ffwd(x_ln2)
    x_final = x_after_attn + x_ffwd  # Residual connection

    print(f'\n   3. After Feedforward + Residual: {x_final.shape}')
    print(f'      Mean: {x_final.mean():.4f}, Std: {x_final.std():.4f}')

    # Comparar con forward completo
    x_complete = transformer_block(x_input)
    print(f'\n   4. Complete forward pass: {x_complete.shape}')
    print(f'      Mean: {x_complete.mean():.4f}, Std: {x_complete.std():.4f}')
    print(f'      Matches step-by-step: {torch.allclose(x_final, x_complete)}')

print('\nğŸ”� Propiedades importantes del bloque:')
print('   â€¢ Entrada y salida tienen la misma forma (residual connections)')
print('   â€¢ Layer normalization estabiliza las activaciones')
print('   â€¢ Self-attention captura dependencias entre tokens')
print('   â€¢ Feedforward aÃ±ade capacidad de procesamiento no-lineal')
print('   â€¢ Residual connections permiten entrenamiento de redes profundas')
print('\nâœ… Bloque Transformer funcionando correctamente!')


# ============================================================================
# ğŸ�¯ BLOQUE DE CONFIGURACIÃ“N PARA EXPERIMENTACIÃ“N
# ============================================================================
#
# âš ï¸�  IMPORTANTE: Este bloque SOBRESCRIBE todas las configuraciones anteriores
#    Modifica estos valores libremente para mejorar tu modelo
#
# ğŸ’¡ ESTRATEGIA: Cambia un parÃ¡metro a la vez y registra los resultados
# ğŸ�† OBJETIVO: Minimizar val_loss para ganar en el Kaggle Challenge
#
# ============================================================================

print("=" * 80)
print("ğŸ”§ CONFIGURANDO PARÃ�METROS PARA EXPERIMENTACIÃ“N")
print("=" * 80)

# ----------------------------------------------------------------------------
# ARQUITECTURA DEL MODELO - Controla la capacidad de aprendizaje
# ----------------------------------------------------------------------------

block_size = 128        # Longitud del contexto (tokens que ve el modelo)
                        # Rango recomendado: 64-512
                        # â†‘ MÃ¡s grande = mÃ¡s contexto, pero mÃ¡s lento y usa mÃ¡s memoria
                        # Ejemplo: 128 es rÃ¡pido, 512 es lento pero mÃ¡s preciso

n_embd = 192           # DimensiÃ³n de los embeddings
                        # Rango recomendado: 192-768
                        # â†‘ MÃ¡s grande = mÃ¡s capacidad de representaciÃ³n
                        # âš ï¸� Debe ser divisible por n_head

n_head = 6              # NÃºmero de cabezas de atenciÃ³n
                        # Rango recomendado: 4-12
                        # Cada cabeza atiende a diferentes aspectos del contexto
                        # âš ï¸� n_embd debe ser divisible por n_head

n_layer = 6             # NÃºmero de capas transformer
                        # Rango recomendado: 4-12
                        # â†‘ MÃ¡s capas = red mÃ¡s profunda, mÃ¡s capacidad de aprendizaje
                        # Pero tambiÃ©n mÃ¡s difÃ­cil de entrenar

dropout = 0.2           # Tasa de dropout para regularizaciÃ³n
                        # Rango recomendado: 0.0-0.3
                        # â†‘ Mayor dropout = menos overfitting pero puede underfittear
                        # 0.2 es un buen punto de partida

# ----------------------------------------------------------------------------
# HIPERPARÃ�METROS DE ENTRENAMIENTO - Controla cÃ³mo aprende el modelo
# ----------------------------------------------------------------------------

batch_size = 64         # NÃºmero de secuencias procesadas simultÃ¡neamente
                        # Rango recomendado: 32-128
                        # â†‘ MÃ¡s grande = entrenamiento mÃ¡s estable pero usa mÃ¡s memoria
                        # Si te quedas sin memoria, reduce este valor

learning_rate = 3e-4   # Tasa de aprendizaje (learning rate)
                        # Rango recomendado: 1e-4 a 1e-3
                        # Este es CRÃ�TICO: muy alto y diverge, muy bajo y no aprende
                        # 3e-4 es el valor clÃ¡sico de Karpathy para GPT pequeÃ±os

max_iters = 10000        # NÃºmero total de iteraciones de entrenamiento
                        # Rango recomendado: 3000-10000
                        # â†‘ MÃ¡s iters = mÃ¡s aprendizaje pero mÃ¡s tiempo
                        # Observa las grÃ¡ficas para saber si necesitas mÃ¡s

# ----------------------------------------------------------------------------
# PARÃ�METROS DE EVALUACIÃ“N Y MONITOREO
# ----------------------------------------------------------------------------

eval_interval = 100     # Cada cuÃ¡ntas iteraciones evaluar en validation set
                        # Valores tÃ­picos: 100-500
                        # MÃ¡s frecuente = mÃ¡s feedback pero entrenamiento mÃ¡s lento

eval_iters = 100        # CuÃ¡ntos batches usar para estimar la pÃ©rdida
                        # Valores tÃ­picos: 100-200
                        # MÃ¡s = estimaciÃ³n mÃ¡s precisa pero mÃ¡s lento

# ============================================================================
# RESUMEN Y VALIDACIÃ“N DE LA CONFIGURACIÃ“N
# ============================================================================

print("\nğŸ“‹ ARQUITECTURA DEL MODELO")
print("-" * 80)
print(f"   TamaÃ±o de contexto (block_size):        {block_size:>6} tokens")
print(f"   DimensiÃ³n de embeddings (n_embd):       {n_embd:>6}")
print(f"   NÃºmero de cabezas de atenciÃ³n (n_head): {n_head:>6}")
print(f"   NÃºmero de capas transformer (n_layer):  {n_layer:>6}")
print(f"   Tasa de dropout:                        {dropout:>6.1%}")

print("\nğŸ�“ HIPERPARÃ�METROS DE ENTRENAMIENTO")
print("-" * 80)
print(f"   TamaÃ±o del batch (batch_size):          {batch_size:>6}")
print(f"   Tasa de aprendizaje (learning_rate):    {learning_rate:>6.0e}")
print(f"   Iteraciones mÃ¡ximas (max_iters):        {max_iters:>6,}")

print("\nğŸ“Š CONFIGURACIÃ“N DE EVALUACIÃ“N")
print("-" * 80)
print(f"   Intervalo de evaluaciÃ³n (eval_interval):{eval_interval:>6}")
print(f"   Iteraciones por evaluaciÃ³n (eval_iters):{eval_iters:>6}")

# Validaciones importantes
print("\nğŸ”� VALIDANDO CONFIGURACIÃ“N...")
print("-" * 80)

validacion_exitosa = True

# ValidaciÃ³n 1: n_embd divisible por n_head
if n_embd % n_head != 0:
    print(f"â�Œ ERROR: n_embd ({n_embd}) debe ser divisible por n_head ({n_head})")
    print(f"   Sugerencia: cambia n_embd a {(n_embd // n_head) * n_head} o n_head a {n_embd // (n_embd // n_head)}")
    validacion_exitosa = False
else:
    print(f"âœ“ n_embd ({n_embd}) es divisible por n_head ({n_head}) âœ“")

# ValidaciÃ³n 2: Advertencias sobre memoria
if batch_size * block_size > 20000:
    print(f"âš ï¸�  ADVERTENCIA: batch_size Ã— block_size = {batch_size * block_size:,}")
    print(f"   Esto puede causar problemas de memoria. Considera reducir batch_size.")
else:
    print(f"âœ“ ConfiguraciÃ³n de memoria razonable (batch_size Ã— block_size = {batch_size * block_size:,}) âœ“")

# ValidaciÃ³n 3: Learning rate razonable
if learning_rate > 1e-3:
    print(f"âš ï¸�  ADVERTENCIA: learning_rate ({learning_rate:.0e}) es muy alto")
    print(f"   Esto puede causar inestabilidad. Rango tÃ­pico: 1e-4 a 1e-3")
elif learning_rate < 1e-5:
    print(f"âš ï¸�  ADVERTENCIA: learning_rate ({learning_rate:.0e}) es muy bajo")
    print(f"   El entrenamiento serÃ¡ muy lento. Rango tÃ­pico: 1e-4 a 1e-3")
else:
    print(f"âœ“ Learning rate ({learning_rate:.0e}) en rango razonable âœ“")

# EstimaciÃ³n del nÃºmero de parÃ¡metros (aproximada)
# FÃ³rmula aproximada: 12 * n_layer * n_embd^2 (para GPT-2 style)
approx_params = 12 * n_layer * n_embd * n_embd + vocab_size * n_embd * 2
approx_params_millions = approx_params / 1_000_000

print(f"\nğŸ“Š ESTADÃ�STICAS ESTIMADAS DEL MODELO")
print("-" * 80)
print(f"   ParÃ¡metros estimados: ~{approx_params_millions:.2f}M parÃ¡metros")
print(f"   TamaÃ±o del vocabulario: {vocab_size} tokens")
print(f"   DimensiÃ³n de cada cabeza: {n_embd // n_head}")

# Tiempo estimado de entrenamiento
tokens_per_iter = batch_size * block_size
total_tokens = tokens_per_iter * max_iters
print(f"\nâ�±ï¸�  ESTIMACIÃ“N DE ENTRENAMIENTO")
print("-" * 80)
print(f"   Tokens por iteraciÃ³n: {tokens_per_iter:,}")
print(f"   Tokens totales a procesar: {total_tokens:,}")
print(f"   Tiempo estimado: depende de tu GPU (ver tokens/sec durante entrenamiento)")

if validacion_exitosa:
    print("\n" + "=" * 80)
    print("âœ… CONFIGURACIÃ“N VALIDADA CORRECTAMENTE - LISTA PARA ENTRENAR")
    print("=" * 80)
    print("\nğŸ’¡ Consejo: Anota estos valores y el val_loss que obtengas para comparar experimentos\n")
else:
    print("\n" + "=" * 80)
    print("â�Œ CONFIGURACIÃ“N CON ERRORES - CORRIGE LOS PROBLEMAS ANTES DE CONTINUAR")
    print("=" * 80)
    raise ValueError("La configuraciÃ³n tiene errores. Por favor corrÃ­gelos antes de continuar.")



class GPTLanguageModel(nn.Module):
    """
    Modelo GPT completo para generaciÃ³n de lenguaje.

    Arquitectura:
    1. Token + Position Embeddings
    2. N bloques Transformer apilados
    3. Layer Normalization final
    4. Capa de salida (language modeling head)

    Componentes:
    - token_embedding_table: Convierte token IDs a vectores
    - position_embedding_table: AÃ±ade informaciÃ³n posicional
    - blocks: Bloques Transformer apilados (self-attention + FFN)
    - ln_f: Layer norm final
    - lm_head: ProyecciÃ³n a vocabulario para predicciÃ³n
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Tablas de embeddings
        # Token embeddings: Cada token tiene un vector de n_embd dimensiones
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embeddings: Cada posiciÃ³n (0 a block_size-1) tiene un vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Bloques Transformer apilados
        # Cada bloque procesa la secuencia completa con attention + FFN
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        # Layer normalization final (despuÃ©s de todos los bloques)
        self.ln_f = LayerNorm(n_embd)

        # Language modeling head: proyecta de n_embd a vocab_size
        # Para cada posiciÃ³n, predice probabilidades sobre todo el vocabulario
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # InicializaciÃ³n de pesos (opcional pero recomendado)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        InicializaciÃ³n de pesos segÃºn el paper 'Attention is All You Need'.
        Usar una buena inicializaciÃ³n ayuda al entrenamiento.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass del modelo GPT.

        Args:
            idx: Tensor de tokens (B, T) donde B=batch_size, T=sequence_length
            targets: (Opcional) Tensor de targets (B, T) para calcular pÃ©rdida

        Returns:
            logits: Predicciones (B, T, vocab_size) - puntuaciones para cada token
            loss: (Si targets != None) PÃ©rdida de entropÃ­a cruzada
        """
        B, T = idx.shape

        # Paso 1: Obtener embeddings de tokens
        # idx: (B, T) -> token_emb: (B, T, C)
        token_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # Paso 2: Obtener embeddings posicionales
        # Creamos un tensor [0, 1, 2, ..., T-1] para las posiciones
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)

        # Paso 3: Combinar token + position embeddings
        # Broadcasting: pos_emb se expande de (T, C) a (B, T, C)
        x = token_emb + pos_emb  # (B, T, n_embd)

        # Paso 4: Pasar por todos los bloques Transformer
        # Cada bloque aplica self-attention + feedforward con residual connections
        x = self.blocks(x)  # (B, T, n_embd)

        # Paso 5: Layer normalization final
        x = self.ln_f(x)  # (B, T, n_embd)

        # Paso 6: Language modeling head
        # Proyectar a espacio de vocabulario para predicciÃ³n
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calcular pÃ©rdida si se proporcionan targets
        if targets is None:
            loss = None
        else:
            # Reshape para calcular cross-entropy
            # PyTorch espera: (N, C) para logits y (N,) para targets
            # Donde N = B*T (todos los tokens), C = vocab_size
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)

            # Calcular cross-entropy loss
            # Mide quÃ© tan bien las predicciones coinciden con los targets
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Genera nuevos tokens autorregressivamente.

        Args:
            idx: Tensor de contexto inicial (B, T)
            max_new_tokens: NÃºmero de tokens nuevos a generar

        Returns:
            idx: Secuencia extendida (B, T + max_new_tokens)

        Proceso:
        1. Tomar los Ãºltimos block_size tokens como contexto
        2. Obtener predicciones del modelo
        3. Muestrear el siguiente token de la distribuciÃ³n de probabilidades
        4. AÃ±adir el token generado al contexto
        5. Repetir
        """
        for _ in range(max_new_tokens):
            # Recortar el contexto a los Ãºltimos block_size tokens
            # El modelo solo puede ver block_size tokens hacia atrÃ¡s
            idx_cond = idx[:, -block_size:]  # (B, T') donde T' <= block_size

            # Obtener predicciones
            logits, loss = self(idx_cond)  # (B, T', vocab_size)

            # Enfocarnos en el Ãºltimo paso de tiempo
            # Solo nos interesa la predicciÃ³n para el siguiente token
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Aplicar softmax para obtener probabilidades
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Muestrear el siguiente token de la distribuciÃ³n de probabilidades
            # Esto aÃ±ade aleatoriedad y diversidad a la generaciÃ³n
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # AÃ±adir el token generado a la secuencia
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

print("âœ… Clase GPTLanguageModel implementada correctamente")
print("\n" + "="*70)
print("ğŸ“Š RESUMEN DE LA ARQUITECTURA GPT")
print("="*70)
print(f"\nğŸ”§ HiperparÃ¡metros configurados:")
print(f"   â€¢ TamaÃ±o del vocabulario: {vocab_size}")
print(f"   â€¢ DimensiÃ³n de embeddings (n_embd): {n_embd}")
print(f"   â€¢ NÃºmero de bloques Transformer (n_layer): {n_layer}")
print(f"   â€¢ NÃºmero de cabezas de atenciÃ³n (n_head): {n_head}")
print(f"   â€¢ TamaÃ±o de cada cabeza: {n_embd // n_head}")
print(f"   â€¢ TamaÃ±o de contexto (block_size): {block_size}")
print(f"\nğŸ�—ï¸� Estructura del modelo:")
print(f"   1. Token Embeddings: {vocab_size} â†’ {n_embd} dims")
print(f"   2. Position Embeddings: {block_size} â†’ {n_embd} dims")
print(f"   3. {n_layer} Bloques Transformer (cada uno con:)")
print(f"      â€¢ Multi-Head Attention: {n_head} cabezas de tamaÃ±o {n_embd // n_head}")
print(f"      â€¢ Feedforward Network: {n_embd} â†’ {4*n_embd} â†’ {n_embd}")
print(f"      â€¢ 2x Layer Normalization + Residual Connections")
print(f"   4. Layer Normalization final")
print(f"   5. Language Modeling Head: {n_embd} â†’ {vocab_size}")


# ============================================================================
# 6.2 ENTRENAMIENTO DEL MODELO NanoGPT
# ============================================================================
import json
import math
from pathlib import Path
from time import perf_counter

print("=" * 80)
print("ğŸš€ INICIANDO ENTRENAMIENTO NanoGPT")
print("=" * 80)

# Asegura que Block use la implementaciÃ³n correcta de la FFN
if "FeedForward" not in globals():
    FeedForward = FeedFoward  # type: ignore[name-defined]

torch.manual_seed(1337)

m = GPTLanguageModel(vocab_size).to(device)
m.train()
model_params = sum(p.numel() for p in m.parameters())

print(f"\nğŸ“¦ ParÃ¡metros entrenables: {model_params:,}")
print(f"ğŸ’» Dispositivo activo: {device.upper()}")

optimizer = torch.optim.AdamW(
    m.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=1e-1,
)

use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
grad_clip = 1.0

artifact_dir = Path("artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)
history_path = artifact_dir / "training_history.json"

training_history = []
best_val_loss = float("inf")
tokens_per_iter = block_size * batch_size
t0 = perf_counter()

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        elapsed_s = perf_counter() - t0
        train_loss = losses["train"].item()
        val_loss = losses["val"].item()
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        tokens_processed = iter * tokens_per_iter
        tokens_per_sec = (
            tokens_processed / elapsed_s
            if elapsed_s > 0 and tokens_processed > 0
            else None
        )
        best_val_loss = min(best_val_loss, val_loss)

        training_history.append(
            {
                "iter": iter,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_minus_train": val_loss - train_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": elapsed_s,
                "tokens_processed": tokens_processed,
                "tokens_per_sec": tokens_per_sec,
                "best_val_loss_so_far": best_val_loss,
                "batch_size": batch_size,
                "block_size": block_size,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": n_head,
                "model_params": model_params,
            }
        )

        print(f"\nğŸ“Š IteraciÃ³n {iter:,}/{max_iters:,}")
        print(f"   â€¢ train_loss: {train_loss:.4f}  |  val_loss: {val_loss:.4f}")
        print(
            f"   â€¢ Perplexity (train/val): {train_ppl:.2f} / {val_ppl:.2f}"
        )
        if tokens_per_sec is not None:
            print(
                f"   â€¢ Tokens procesados: {tokens_processed:,}  "
                f"({int(tokens_per_sec):,} tok/s)"
            )
        diff = val_loss - train_loss
        if diff > 0.5:
            print("   âš ï¸� Gap grande entre train y val (overfitting probable)")
        elif diff > 0.1:
            print("   ğŸŸ¡ Gap moderado entre train y val")
        else:
            print("   âœ… Balance saludable entre train y val")

    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits, loss = m(xb, yb)

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

final_losses = estimate_loss()
final_elapsed = perf_counter() - t0
final_train_loss = final_losses["train"].item()
final_val_loss = final_losses["val"].item()
final_train_ppl = math.exp(final_train_loss)
final_val_ppl = math.exp(final_val_loss)
best_val_loss = min(best_val_loss, final_val_loss)
final_tokens = max_iters * tokens_per_iter
final_tokens_per_sec = (
    final_tokens / final_elapsed if final_elapsed > 0 else None
)

training_history.append(
    {
        "iter": max_iters,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "val_minus_train": final_val_loss - final_train_loss,
        "train_ppl": final_train_ppl,
        "val_ppl": final_val_ppl,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_s": final_elapsed,
        "tokens_processed": final_tokens,
        "tokens_per_sec": final_tokens_per_sec,
        "best_val_loss_so_far": best_val_loss,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "model_params": model_params,
    }
)

with history_path.open("w", encoding="utf-8") as f:
    json.dump(training_history, f, indent=2)

print("\n" + "=" * 80)
print("âœ… ENTRENAMIENTO COMPLETADO")
print("=" * 80)
print(f"   â€¢ train_loss final: {final_train_loss:.4f}")
print(f"   â€¢ val_loss final:   {final_val_loss:.4f}")
print(
    f"   â€¢ Perplexity (train/val): {final_train_ppl:.2f} / {final_val_ppl:.2f}"
)
if final_tokens_per_sec is not None:
    print(f"   â€¢ Rendimiento medio: {int(final_tokens_per_sec):,} tok/s")
print(f"\nğŸ’¾ Historial guardado en: {history_path.resolve()}")




# ============================================================================
# VISUALIZACIÃ“N AUTOMÃ�TICA Y PERSISTENCIA DE MÃ‰TRICAS DE ENTRENAMIENTO
# ============================================================================
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

artifact_dir = Path('artifacts')
history_path = artifact_dir / 'training_history.json'

# Priorizar la historia en memoria si existe en esta sesiÃ³n
if 'training_history' in globals() and training_history:
    history_data = training_history
elif history_path.exists():
    history_data = json.loads(history_path.read_text())
else:
    history_data = []

if not history_data:
    print('âš ï¸� No hay mÃ©tricas de entrenamiento registradas todavÃ­a. Ejecuta el bucle de entrenamiento primero.')
else:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history_data).sort_values('iter').reset_index(drop=True)
    df['tokens_per_sec'] = df['tokens_per_sec'].fillna(0.0)
    df['perplexity_ratio'] = df['val_ppl'] / df['train_ppl']

    display(df.tail(10))

    best_row = df.loc[df['val_loss'].idxmin()]
    last_row = df.iloc[-1]
    print('\nğŸ�� Ãšltima evaluaciÃ³n registrada:')
    print(f"   â€¢ IteraciÃ³n: {int(last_row['iter'])}")
    print(f"   â€¢ train_loss: {last_row['train_loss']:.4f}")
    print(f"   â€¢ val_loss: {last_row['val_loss']:.4f}")
    print(f"   â€¢ Perplexity (train/val): {last_row['train_ppl']:.2f} / {last_row['val_ppl']:.2f}")
    if last_row['tokens_per_sec']:
        print(f"   â€¢ Tokens/s aproximados: {last_row['tokens_per_sec']:,.0f}")
    print(f"   â€¢ Diferencia val-train: {last_row['val_minus_train']:.4f}")

    print('\nğŸ¥‡ Mejor checkpoint por val_loss:')
    print(f"   â€¢ IteraciÃ³n: {int(best_row['iter'])}")
    print(f"   â€¢ val_loss mÃ­nima: {best_row['val_loss']:.4f}")
    print(f"   â€¢ train_loss correspondiente: {best_row['train_loss']:.4f}")
    print(f"   â€¢ Gap val-train: {best_row['val_minus_train']:.4f}")
    print(f"   â€¢ Perplexity (train/val): {best_row['train_ppl']:.2f} / {best_row['val_ppl']:.2f}")

    csv_path = artifact_dir / 'training_history.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nğŸ’¾ Historial tabular guardado en: {csv_path.resolve()}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(df['iter'], df['train_loss'], label='train_loss', marker='o')
    axes[0].plot(df['iter'], df['val_loss'], label='val_loss', marker='o')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('EvoluciÃ³n de train_loss vs val_loss')
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(df['iter'], df['val_minus_train'], label='val - train', color='tab:orange')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Gap')
    axes[1].set_xlabel('IteraciÃ³n')
    axes[1].set_title('Diferencia entre pÃ©rdidas')
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    loss_plot_path = artifact_dir / 'loss_curves.png'
    fig.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"ğŸ“Š GrÃ¡fico de pÃ©rdidas guardado en: {loss_plot_path.resolve()}")

    fig_tokens, ax_tokens = plt.subplots(figsize=(9, 4))
    ax_tokens.plot(df['iter'], df['tokens_per_sec'], color='tab:green')
    ax_tokens.set_title('Tokens procesados por segundo')
    ax_tokens.set_xlabel('IteraciÃ³n')
    ax_tokens.set_ylabel('Tokens/s')
    ax_tokens.grid(alpha=0.2)
    plt.tight_layout()
    tokens_plot_path = artifact_dir / 'throughput.png'
    fig_tokens.savefig(tokens_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig_tokens)
    print(f"âš™ï¸� GrÃ¡fico de throughput guardado en: {tokens_plot_path.resolve()}")




# ============================================================================
# GENERADOR DE CSV PARA EVALUACIÃ“N DEL MODELO
# ============================================================================
import csv
import json
from pathlib import Path
from datetime import datetime

# Directorio de salida
output_dir = Path('artifacts')
output_dir.mkdir(exist_ok=True)

# Nombre del archivo CSV con timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = output_dir / f'model_evaluation_{timestamp}.csv'

# TambiÃ©n mantener un CSV con el Ãºltimo resultado que es lo que se debe entregar
csv_latest = 'submission.csv'

# CSV histÃ³rico que acumula todos los experimentos
csv_history = output_dir / 'model_evaluation_history.csv'

print('ğŸ“Š GENERANDO CSV DE EVALUACIÃ“N DEL MODELO')
print('=' * 80)

# Verificar que existan los datos del entrenamiento
if 'training_history' not in globals() or not training_history:
    print('âš ï¸�  No hay datos de entrenamiento disponibles en esta sesiÃ³n.')
    print('   Ejecuta primero el entrenamiento del modelo (SecciÃ³n 6.3)')
else:
    # Obtener el Ãºltimo registro del entrenamiento
    last_record = training_history[-1]

    # Calcular mÃ©tricas adicionales
    num_parameters = sum(p.numel() for p in m.parameters())

    # Usar perplejidad del registro si existe, sino calcularla
    val_perplexity = last_record.get('val_ppl', math.exp(last_record['val_loss']))
    train_perplexity = last_record.get('train_ppl', math.exp(last_record['train_loss']))
    overfitting_gap = last_record['val_loss'] - last_record['train_loss']

    # Obtener tokens_per_sec de manera segura
    tokens_per_sec = last_record.get('tokens_per_sec', 0.0)
    if tokens_per_sec is None:
        tokens_per_sec = 0.0

    # Obtener elapsed_s y calcular minutos
    elapsed_s = last_record.get('elapsed_s', 0.0)
    if elapsed_s is None:
        elapsed_s = 0.0
    elapsed_min = elapsed_s / 60.0

    # Preparar datos para CSV
    csv_data = {
        'id': 1,
        # IdentificaciÃ³n
        'timestamp': timestamp,
        'experiment_id': f'exp_{timestamp}',

        # MÃ©tricas de pÃ©rdida
        'train_loss': f"{last_record['train_loss']:.6f}",
        'val_loss': f"{last_record['val_loss']:.6f}",
        'val_minus_train': f"{overfitting_gap:.6f}",

        # Perplejidad
        'train_perplexity': f"{train_perplexity:.4f}",
        'val_perplexity': f"{val_perplexity:.4f}",

        # HiperparÃ¡metros del modelo (usar valores del registro si existen)
        'block_size': last_record.get('block_size', block_size),
        'batch_size': last_record.get('batch_size', batch_size),
        'vocab_size': vocab_size,

        # HiperparÃ¡metros de entrenamiento
        'learning_rate': last_record['learning_rate'],
        'max_iters': last_record['iter'],
        'eval_interval': eval_interval if 'eval_interval' in globals() else 'N/A',

        # EstadÃ­sticas del modelo
        'num_parameters': num_parameters,
        'best_val_loss': f"{last_record['best_val_loss_so_far']:.6f}",

        # Rendimiento
        'tokens_per_sec': f"{tokens_per_sec:.2f}",
        'total_tokens_processed': last_record.get('tokens_processed', 0),
        'training_time_minutes': f"{elapsed_min:.2f}",
    }

    # Escribir CSV con timestamp
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        writer.writeheader()
        writer.writerow(csv_data)

    # Escribir CSV latest (sobreescribe el anterior)
    with open(csv_latest, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        writer.writeheader()
        writer.writerow(csv_data)

    # AÃ±adir al CSV histÃ³rico (append)
    file_exists = csv_history.exists()
    with open(csv_history, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_data)

    print('\nâœ… CSV generado exitosamente')
    print('\nğŸ“� Archivos creados:')
    print(f'   â€¢ {csv_filename} (registro con timestamp)')
    print(f'   â€¢ {csv_latest} (Ãºltimo resultado)')
    print(f'   â€¢ {csv_history} (histÃ³rico acumulado)')

    print('\nğŸ“Š Resumen de mÃ©tricas guardadas:')
    print(f'   â€¢ PÃ©rdida de entrenamiento: {csv_data["train_loss"]}')
    print(f'   â€¢ PÃ©rdida de validaciÃ³n: {csv_data["val_loss"]}')
    print(f'   â€¢ Perplejidad de validaciÃ³n: {csv_data["val_perplexity"]}')
    print(f'   â€¢ Gap de overfitting: {csv_data["val_minus_train"]}')
    print(f'   â€¢ ParÃ¡metros del modelo: {csv_data["num_parameters"]:,}')

    print('\nğŸ”§ HiperparÃ¡metros guardados:')
    print(f'   â€¢ block_size: {csv_data["block_size"]}')
    print(f'   â€¢ batch_size: {csv_data["batch_size"]}')
    print(f'   â€¢ learning_rate: {csv_data["learning_rate"]}')

    print('\nğŸ’¡ Uso sugerido:')
    print('   â€¢ Importa el CSV en Excel/Google Sheets para anÃ¡lisis')
    print('   â€¢ Compara diferentes experimentos usando el histÃ³rico')
    print('   â€¢ Usa val_loss para rankear en el Kaggle Challenge')
    print('   â€¢ Analiza la relaciÃ³n entre hiperparÃ¡metros y rendimiento')

    # Mostrar vista previa del CSV
    print('\nğŸ“‹ Vista previa del CSV:')
    print('-' * 80)
    import pandas as pd
    df = pd.read_csv(csv_latest)
    # Transponer para mejor visualizaciÃ³n
    print(df.T.to_string())
    print('-' * 80)


