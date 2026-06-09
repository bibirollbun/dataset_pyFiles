# --- CELDA 1: CONFIGURACIÃ“N EN KAGGLE ---
!pip install -q -U google-generativeai

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient # <--- Esto es lo que cambia respecto a Colab

try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… API Key configurada en Kaggle correctamente.")
except Exception as e:
    print(f"â�Œ Error: {e}")


# --- CELDA 2: HERRAMIENTA DE CÃ�LCULO ROBUSTA ---

# Base de datos ampliada con sinÃ³nimos y correcciÃ³n de errores comunes
FACTORES_EMISION = {
    # ALCANCE 1
    "diesel": {"factor": 2.68, "unidad_ref": "litros", "alcance": 1, "desc": "Combustible para transporte/generadores"},
    "diÃ©sel": {"factor": 2.68, "unidad_ref": "litros", "alcance": 1, "desc": "Combustible para transporte/generadores"}, # <--- Â¡Agregado con tilde!
    "gasoil": {"factor": 2.68, "unidad_ref": "litros", "alcance": 1, "desc": "SinÃ³nimo de diÃ©sel"},
    "gasolina": {"factor": 2.31, "unidad_ref": "litros", "alcance": 1, "desc": "VehÃ­culos ligeros"},
    "gas_natural": {"factor": 2.05, "unidad_ref": "m3", "alcance": 1, "desc": "CalefacciÃ³n"},

    # ALCANCE 2
    "electricidad_mix": {"factor": 0.25, "unidad_ref": "kWh", "alcance": 2, "desc": "Mix elÃ©ctrico red nacional"},
    "electricidad_red": {"factor": 0.25, "unidad_ref": "kWh", "alcance": 2, "desc": "SinÃ³nimo mix"},

    # ALCANCE 3
    "taxi": {"factor": None, "unidad_ref": "km", "alcance": 3, "desc": "Requiere distancia, no importe monetario"},
    "vuelo_corto": {"factor": 0.15, "unidad_ref": "km", "alcance": 3, "desc": "Vuelos < 500km"}
}

def calcular_huella_carbono(actividad_id: str, cantidad: float):
    """
    Calcula emisiones y asigna Alcance ISO 14064.
    """
    # Normalizamos la entrada (quitamos espacios extra y pasamos a minÃºsculas)
    clave = actividad_id.lower().strip()

    # Buscamos en la base de datos
    info = FACTORES_EMISION.get(clave)

    if not info:
        # Fallback inteligente: si no encuentra la clave exacta, busca coincidencias parciales
        for k, v in FACTORES_EMISION.items():
            if k in clave or clave in k:
                info = v
                clave = k # Actualizamos la clave encontrada
                break

    if not info:
        return {"error": f"Actividad '{actividad_id}' no catalogada. Disponibles: {list(FACTORES_EMISION.keys())}"}

    # CÃ¡lculo
    if info['factor'] is None:
        return {
            "aviso": "Faltan datos tÃ©cnicos",
            "alcance_iso": f"Alcance {info['alcance']}",
            "mensaje": f"Para calcular {clave} necesito la unidad '{info['unidad_ref']}', no puedo calcularlo solo con el nombre."
        }

    emision = cantidad * info['factor']

    return {
        "actividad": clave,
        "emision_kgCO2e": round(emision, 2),
        "alcance_iso": f"Alcance {info['alcance']}",
        "factor_utilizado": info['factor'],
        "norma_referencia": "ISO 14064-1 / GHG Protocol"
    }

print("âœ… Herramienta de cÃ¡lculo ACTUALIZADA")


# --- CELDA 3: INSTRUCCIÃ“N DE SISTEMA (PULIDA) ---

system_instruction = """
Eres el 'Eco-Audit Agent', un auditor experto en sostenibilidad (ISO 14064).
Tu objetivo es procesar listas de consumos y generar un informe tÃ©cnico riguroso.

REGLAS DE COMPORTAMIENTO:
1.  **NO ALUCINES:** Usa siempre la herramienta `calcular_huella_carbono`. Si no tienes datos suficientes (ej: euros en vez de km), repÃ³rtalo como "Pendiente de informaciÃ³n".
2.  **VISIÃ“N GLOBAL:** No te centres solo en los errores. Debes reportar CADA UNO de los Ã­tems que te envÃ­e el usuario.
3.  **FORMATO OBLIGATORIO:** Tu respuesta final DEBE incluir siempre una tabla Markdown perfectamente alineada con estas columnas exactas:

| Actividad | Dato Original | Alcance ISO | EmisiÃ³n (kgCO2e) | Estado |
| :--- | :--- | :--- | :--- | :--- |
| DiÃ©sel | 5000 litros | Alcance 1 | 13,400 | âœ… Calculado |
| Taxis | 500 euros | Alcance 3 | - | âš ï¸� Faltan Km |

Al final de la tabla, aÃ±ade un breve comentario tÃ©cnico sobre los hallazgos.
"""

print("âœ… System Prompt DEFINITIVO: Formato de tabla optimizado.")


# --- CELDA 4: DETECTOR AUTOMÃ�TICO DE MODELO (SoluciÃ³n Definitiva) ---
import google.generativeai as genai

print("ğŸ”� Escaneando modelos disponibles en tu cuenta...")

nombre_modelo_final = None

# 1. Buscamos en la lista real de modelos que Google te ofrece
for m in genai.list_models():
    # Filtramos solo los que sirven para generar texto
    if 'generateContent' in m.supported_generation_methods:
        # Prioridad absoluta: Buscar uno que diga "flash" (rÃ¡pido y gratis)
        if 'flash' in m.name:
            nombre_modelo_final = m.name
            break

# 2. Si no encuentra Flash, usamos el clÃ¡sico 'gemini-pro' como respaldo
if not nombre_modelo_final:
    nombre_modelo_final = 'models/gemini-pro'

print(f"âœ… Â¡Conectado! Usaremos el modelo: {nombre_modelo_final}")

try:
    # 3. Inicializamos el modelo con el nombre exacto que encontramos
    model = genai.GenerativeModel(
        model_name=nombre_modelo_final,
        tools=[calcular_huella_carbono],
        system_instruction=system_instruction
    )

    # 4. Iniciamos el chat
    chat = model.start_chat(enable_automatic_function_calling=True)
    print("ğŸ¤– Agente Eco-Audit LISTO y operativo.")

except Exception as e:
    print(f"â�Œ Error fatal: {e}")


# --- CELDA 5: SIMULACIÃ“N CON VISUALIZACIÃ“N PROFESIONAL ---
from IPython.display import display, Markdown

# Tu mensaje de prueba
mensaje_usuario = """
Hola auditor. Necesito calcular estas tres emisiones:
1. Consumo de 5,000 litros de diÃ©sel en maquinaria propia.
2. Consumo de 12,500 kWh de electricidad de red.
3. Gasto de 500 euros en taxis (sin dato de km).
"""

print(f"ğŸ‘¤ CLIENTE:\n{mensaje_usuario}")
print("\nâ�³ El Agente Eco-Audit estÃ¡ analizando la normativa y calculando...\n")
print("-" * 60)

try:
    # 1. Obtenemos la respuesta
    response = chat.send_message(mensaje_usuario)

    # 2. VISUALIZACIÃ“N RICA (AquÃ­ estÃ¡ el truco)
    # En lugar de print(), usamos display(Markdown())
    print("ğŸ¤– AGENTE ECO-AUDIT (Informe Final):")
    display(Markdown(response.text))

except Exception as e:
    print(f"â�Œ Error: {e}")

