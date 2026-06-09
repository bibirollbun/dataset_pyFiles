# ==========================
# LibrerÃ­as estÃ¡ndar de Python
# ==========================
import os                           # Operaciones con el sistema de archivos
import sys                          # Acceso a funciones y variables del intÃ©rprete
import subprocess                   # EjecuciÃ³n de procesos del sistema
import importlib                    # Carga dinÃ¡mica de mÃ³dulos
import logging                      # ConfiguraciÃ³n y manejo de logs



# ConfiguraciÃ³n de Logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    level=getattr(logging, log_level),
    force=True
)
logger = logging.getLogger(__name__)


def install_and_check_dependencies():
    """
    Verifica e instala las dependencias necesarias para el Asistente Ejecutivo.
    Mapea el nombre del paquete en PIP con el nombre del mÃ³dulo en Python.
    """
    print("âš™ï¸� Instalando Tesseract OCR (Sistema Linux)...")
    # Esto es necesario para que funcione pytesseract en Kaggle
    os.system("apt-get update && apt-get install -y tesseract-ocr libtesseract-dev")

    # Diccionario: "nombre-en-pip": "nombre-del-modulo-python"
    dependencies = {
        "feedparser": "feedparser",
        "crewai": "crewai",
        "yfinance": "yfinance",
        "nest_asyncio": "nest_asyncio",
        "google-auth-oauthlib": "google_auth_oauthlib",
        "google-api-python-client": "googleapiclient",
        "twilio": "twilio",
        "pytesseract": "pytesseract",
        "requests": "requests",
        "Pillow": "Pillow",
        "beautifulsoup4": "BeautifulSoup",
    }

    logger.info("ğŸ”� Verificando dependencias del sistema...")

    cambios_realizados = False

    for package, module in dependencies.items():
        try:
            importlib.import_module(module)
            # Si llegamos aquÃ­, el mÃ³dulo existe, no hacemos nada (o logueamos debug)
        except ImportError:
            logger.info(f"â¬‡ï¸�  Instalando {package}...")
            try:
                # Ejecuta pip install en modo silencioso (-q)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
                cambios_realizados = True
                logger.info(f"âœ… {package} instalado correctamente.")
            except subprocess.CalledProcessError:
                logger.error(f"â�Œ Error crÃ­tico al instalar {package}.")

    if cambios_realizados:
        logger.info("ğŸ�‰ Todas las dependencias han sido instaladas/verificadas.")
    else:
        logger.info("âœ… Todo estÃ¡ al dÃ­a. No se requirieron instalaciones.")


install_and_check_dependencies()


# ==========================
# LibrerÃ­as estÃ¡ndar de Python
# ==========================
import json                     # Manejo de datos en formato JSON
import urllib.parse             # Funciones para parsear y manipular URLs
from datetime import datetime, timedelta  # Manejo de fechas y tiempos
from typing import List, Dict, Optional, Any  # Tipado estÃ¡tico para anotaciones

# ==========================
# LibrerÃ­as externas para HTTP y datos
# ==========================
import requests                 # Realizar peticiones HTTP
import feedparser               # Parseo de feeds RSS/Atom
import yfinance as yf           # Acceso a datos financieros (Yahoo Finance)
import pandas as pd             # ManipulaciÃ³n y anÃ¡lisis de datos

# ==========================
# Google APIs
# ==========================
from google.oauth2.credentials import Credentials       # AutenticaciÃ³n OAuth2
from google.auth.transport.requests import Request      # Transporte para OAuth
from google_auth_oauthlib.flow import InstalledAppFlow  # Flujo de autenticaciÃ³n
from googleapiclient.discovery import build             # ConstrucciÃ³n de clientes API
from google.colab import userdata                       # Acceso a datos en Google Colab
from kaggle_secrets import UserSecretsClient
# ==========================
# CrewAI (Agentes y Tareas)
# ==========================
from crewai import Crew, Agent, Task, LLM          # Componentes principales de CrewAI
from crewai.tools import tool                      # Decorador para definir herramientas

# ==========================
# Twilio (MensajerÃ­a)
# ==========================
from twilio.rest import Client                     # Cliente para enviar SMS y llamadas

# ==========================
# VisualizaciÃ³n
# ==========================
from IPython.display import display, Markdown      # Mostrar contenido en notebooks

# ==========================
# Procesamiento de HTML y OCR
# ==========================
from bs4 import BeautifulSoup                      # Parseo de HTML
from PIL import Image                              # ManipulaciÃ³n de imÃ¡genes
import pytesseract                                 # OCR (Reconocimiento Ã³ptico de caracteres)
from io import BytesIO                             # Manejo de datos binarios en memoria

# ==========================
# Utilidades adicionales
# ==========================
from urllib.parse import urlparse                  # Parseo de URLs



# Scopes de Google necesarios
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/tasks.readonly",
    "https://www.googleapis.com/auth/gmail.readonly"
]


# Variables de entorno (Cargadas de manera segura)
# NOTA: En producciÃ³n, usa userdata.get() o variables de sistema. NO hardcodees aquÃ­.
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


# Intentamos cargar credenciales de userdata (Colab) o entorno
try:
    os.environ["PUSHOVER_USER"] = UserSecretsClient().get_secret('PUSHOVER_USER') or os.environ.get('PUSHOVER_USER', '')
    os.environ["PUSHOVER_TOKEN"] = UserSecretsClient().get_secret('PUSHOVER_TOKEN') or os.environ.get('PUSHOVER_TOKEN', '')
    os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret('GOOGLE_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    os.environ["TELEGRAM_TOKEN"] = UserSecretsClient().get_secret('TELEGRAM_TOKEN') or os.environ.get('TELEGRAM_TOKEN', '')
    os.environ["TELEGRAM_CHAT_ID"] = UserSecretsClient().get_secret('TELEGRAM_CHAT_ID') or os.environ.get('TELEGRAM_CHAT_ID', '')
    #os.environ["WHATSAPP_PHONE"] = userdata.get('WHATSAPP_PHONE') or os.environ.get('WHATSAPP_PHONE', '')
    #os.environ["WHATSAPP_API_KEY"] = userdata.get('WHATSAPP_API_KEY') or os.environ.get('WHATSAPP_API_KEY', '')
except Exception as e:
    logger.warning(f"Advertencia cargando secretos: {e}. AsegÃºrate de tener las variables de entorno configuradas.")


# LÃ­mites para evitar errores de API
RPM_LIMIT = 10


# Campos financieros a extraer
CLAVES_PRINCIPALES = [
    "symbol", "shortName", "currency", "exchange",
    "currentPrice", "regularMarketPrice", "previousClose",
    "regularMarketChangePercent", "dayLow", "dayHigh",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "marketCap", "dividendYield"
]


%cp /kaggle/input/token-google/token.json /kaggle/working/token.json


def get_llm():
    """
    Retorna una instancia configurada del LLM (Gemini Flash).

    Returns:
        LLM: Instancia lista para usar en CrewAI.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY no estÃ¡ configurada. Verifica tus secretos.")

    return LLM(
        model="gemini/gemini-flash-latest", # Nombre del modelo actualizado
        verbose=True,
        temperature=0, # Temperatura baja para reducir alucinaciones
        google_api_key=api_key
    )


def authenticate_google_services() -> Optional[Credentials]:
    """
    Maneja el flujo de autenticaciÃ³n OAuth 2.0 para servicios de Google.

    Returns:
        Credentials: Objeto de credenciales vÃ¡lido o None si falla.
    """
    creds = None
    token_file = '/kaggle/working/token.json'

    if os.path.exists(token_file):
         creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("ğŸ”„ Token refrescado automÃ¡ticamente")
            except Exception:
                logger.warning("âš ï¸� Token expirado y no renovable. Re-autenticando...")
                creds = None

        if not creds:
            if not os.path.exists('credentials.json'):
                logger.error("â�Œ Faltan 'credentials.json'. DescÃ¡rgalo de Google Cloud Console.")
                return None

            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            # run_local_server puede fallar en entornos headless (nube), usar con precauciÃ³n
            creds = flow.run_local_server(port=0)

        # Guardar token para la prÃ³xima vez
        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return creds


@tool("Enviar Telegram")
def send_telegram(message: str) -> str:
    """
    EnvÃ­a el reporte a Telegram. Es mÃ¡s rÃ¡pido y fiable que WhatsApp.
    Requiere: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID.
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return "Error: Faltan credenciales de Telegram."

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Telegram soporta Markdown si quieres negritas
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return "Telegram enviado."
        return f"Error Telegram: {response.text}"
    except Exception as e:
        return f"ExcepciÃ³n Telegram: {e}"


@tool("Noticias RSS")
def get_financial_news(busqueda: str) -> str:
    """
    Busca noticias recientes en Google News (RSS).
    Args:
        busqueda (str): TÃ©rmino a buscar (ej: "El Corte InglÃ©s").
    Returns:
        str: JSON con lista de noticias (tÃ­tulo, link, fecha, fuente).
    """
    base_url = "https://news.google.com/rss/search"
    params = {"q": busqueda, "hl": "es-ES", "gl": "ES", "ceid": "ES:es"}

    # Headers vitales para evitar bloqueo (403/429)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/115.0.0.0 Safari/537.36"}

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        f = feedparser.parse(response.content)
        noticias_procesadas = []

        if not f.entries:
            return json.dumps({"news": [], "mensaje": f"Sin noticias para '{busqueda}'."}, ensure_ascii=False)

        for entry in f.entries[:5]: # Top 5 noticias
            noticias_procesadas.append({
                "titulo": entry.title,
                "link": entry.link,
                "fecha": entry.get("published", "Fecha desconocida"),
                "fuente": entry.get("source", {}).get("title", "Google News")
            })

        return json.dumps({"news": noticias_procesadas}, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error noticias: {e}")
        return json.dumps({"error": str(e), "news": []}, ensure_ascii=False)


@tool("Bolsa")
def get_stock_price(symbol: str) -> str:
    """
    Obtiene datos financieros filtrados de Yahoo Finance.
    Args:
        symbol (str): Ticker (ej: "REP.MC").
    Returns:
        str: JSON con datos clave de mercado.
    """
    try:
        symbol = symbol.strip().upper()
        t = yf.Ticker(symbol)
        info = t.info

        if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
             return json.dumps({"error": "Ticker invÃ¡lido", "stock": {}}, ensure_ascii=False)

        datos_filtrados = {k: info.get(k) for k in CLAVES_PRINCIPALES if info.get(k) is not None}
        return json.dumps({"stock": datos_filtrados}, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error bolsa: {e}")
        return json.dumps({"error": str(e), "stock": {}}, ensure_ascii=False)


@tool("Enviar Pushover")
def send_pushover(msg: str) -> str:
    """EnvÃ­a notificaciones vÃ­a Pushover."""
    if DRY_RUN: return "SimulaciÃ³n: Enviado."

    user = os.environ.get("PUSHOVER_USER")
    token = os.environ.get("PUSHOVER_TOKEN")

    if not user or not token: return "Error: Faltan credenciales Pushover."

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": msg, "title": "Briefing IA"},
            timeout=10
        )
        return "Enviado." if resp.status_code == 200 else f"Error API: {resp.text}"
    except Exception as e:
        return f"ExcepciÃ³n Pushover: {e}"


@tool("Enviar WhatsApp")
def send_whatsapp(message: str) -> str:
    """
    EnvÃ­a un mensaje de WhatsApp utilizando la API de Twilio.
    Requiere las variables de entorno: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER y WHATSAPP_PHONE.
    """
    # 1. Verificar si es una simulaciÃ³n (DRY_RUN)
    if os.environ.get("DRY_RUN", "0") == "1":
        logger.info(f"SimulaciÃ³n Twilio: {message[:50]}...")
        return json.dumps({"resultado": "SimulaciÃ³n: Enviado (DRY_RUN)"}, ensure_ascii=False)

    # 2. Obtener credenciales
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER") # Ej: +14155238886
    to_number = os.environ.get("WHATSAPP_PHONE")       # Tu nÃºmero personal

    # ValidaciÃ³n de seguridad
    if not all([account_sid, auth_token, from_number, to_number]):
        logger.error("Faltan credenciales de Twilio.")
        return json.dumps({"error": "ConfiguraciÃ³n incompleta de Twilio"}, ensure_ascii=False)

    try:
        # 3. Inicializar Cliente Twilio
        client = Client(account_sid, auth_token)

        # Twilio exige el prefijo 'whatsapp:' antes del nÃºmero E.164
        # Nos aseguramos de que el prefijo estÃ© presente
        if not from_number.startswith("whatsapp:"):
            from_number = f"whatsapp:{from_number}"

        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"

        # 4. Enviar Mensaje
        # Nota: Twilio trunca mensajes muy largos (>1600 caracteres),
        # pero es seguro cortar a 1000 para asegurar entrega rÃ¡pida.
        msg = client.messages.create(
            body=message[:1500],
            from_=from_number,
            to=to_number
        )

        logger.info(f"WhatsApp enviado. SID: {msg.sid}")
        return json.dumps({
            "resultado": "Enviado correctamente",
            "id_mensaje": msg.sid
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error enviando WhatsApp con Twilio: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool("Enviar WhatsApp")
def send_whatsapp_old(message: str) -> str:
    """EnvÃ­a WhatsApp vÃ­a CallMeBot."""
    if DRY_RUN: return "SimulaciÃ³n: Enviado."

    phone = os.environ.get("WHATSAPP_PHONE")
    apikey = os.environ.get("WHATSAPP_API_KEY")

    if not phone or not apikey: return json.dumps({"error": "Faltan credenciales WhatsApp"})

    try:
        # Usamos params para codificaciÃ³n automÃ¡tica y segura
        url = "https://api.callmebot.com/whatsapp.php"
        params = {"phone": phone, "text": message[:1000], "apikey": apikey} # Truncar a 1000 chars

        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            return json.dumps({"resultado": "Enviado correctamente"}, ensure_ascii=False)
        return json.dumps({"error": f"API Error: {response.text}"}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool("Enviar WhatsApp Meta")
def send_whatsapp_meta(message: str) -> str:
    """
    EnvÃ­a mensaje usando la API oficial de WhatsApp Cloud (Meta).
    Requiere: META_TOKEN, META_PHONE_ID, WHATSAPP_PHONE.
    """
    token = os.environ.get("META_TOKEN")
    phone_id = os.environ.get("META_PHONE_ID")
    to_number = os.environ.get("WHATSAPP_PHONE") # Debe incluir cÃ³digo paÃ­s sin '+' (ej: 34666...)

    if not all([token, phone_id, to_number]):
        return "Error: Faltan credenciales de Meta."

    url = f"https://graph.facebook.com/v22.0/{phone_id}/messages"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Payload para mensaje de texto simple
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message[:1000]} # LÃ­mite de caracteres por seguridad
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            return "WhatsApp enviado (Meta)."
        return f"Error Meta: {response.text}"
    except Exception as e:
        return f"ExcepciÃ³n: {e}"


@tool("Leer correo")
def read_emails() -> str:
    """
    Lee correos de hoy filtrando 'category:primary' para evitar spam/promociones.
    Returns: JSON con lista de correos.
    """
    try:
        creds = authenticate_google_services()
        service = build("gmail", "v1", credentials=creds)

        hoy = datetime.now().strftime("%Y/%m/%d")
        maÃ±ana = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d")

        # MEJORA: Filtramos por category:primary para reducir ruido
        query = f"after:{hoy} before:{maÃ±ana}"

        results = service.users().messages().list(userId='me', q=query, maxResults=50).execute()
        messages = results.get('messages', [])

        correos = []
        if not messages:
            return json.dumps({"correos": [], "mensaje": "Bandeja de entrada absolutamente vacÃ­a hoy."}, ensure_ascii=False)
        for msg in messages:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = txt.get('payload', {}).get('headers', [])
            snippet = txt.get('snippet', '')

            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(Sin Asunto)')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '(Desconocido)')

            # Limpiamos el remitente para que sea mÃ¡s legible (ej: "Google <no-reply...>" -> "Google")
            sender_name = sender.split('<')[0].strip().replace('"', '')

            correos.append({"remitente": sender, "asunto": subject, "resumen": txt.get('snippet', '')[:200]})

        if not correos: return json.dumps({"correos": [], "mensaje": "Sin correos importantes hoy."}, ensure_ascii=False)
        return json.dumps({"correos": correos}, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Calendario")
def get_todays_agenda() -> str:
    """Obtiene eventos de hoy del calendario principal."""
    try:
        creds = authenticate_google_services()
        service = build("calendar", "v3", credentials=creds)

        now = datetime.now()
        time_min = now.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
        time_max = now.replace(hour=23, minute=59, second=59).isoformat() + 'Z'

        events_result = service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime'
        ).execute()

        agenda = []
        for event in events_result.get('items', []):
            start = event['start'].get('dateTime', event['start'].get('date'))
            agenda.append({
                "titulo": event.get('summary', 'Sin tÃ­tulo'),
                "inicio": start,
                "ubicacion": event.get('location', 'N/A')
            })

        if not agenda: return json.dumps({"agenda": [], "mensaje": "Agenda libre."}, ensure_ascii=False)
        return json.dumps({"agenda": agenda}, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})



@tool("Tareas")
def get_todays_tasks() -> str:
    """Obtiene tareas que vencen hoy."""
    try:
        creds = authenticate_google_services()
        service = build("tasks", "v1", credentials=creds)

        hoy_str = datetime.now().strftime("%Y-%m-%d") # ComparaciÃ³n por string es mÃ¡s segura en ISO
        tareas = []

        # Listamos todas las listas de tareas
        tasklists = service.tasklists().list(maxResults=5).execute()

        for lista in tasklists.get('items', []):
            tasks = service.tasks().list(tasklist=lista['id'], showCompleted=False).execute()
            for t in tasks.get('items', []):
                due = t.get('due')
                if due and due.startswith(hoy_str): # Chequeo simple de fecha
                    tareas.append({
                        "lista": lista['title'],
                        "titulo": t['title'],
                        "notas": t.get('notes', '')
                    })

        if not tareas: return json.dumps({"tareas": [], "mensaje": "Sin tareas para hoy."}, ensure_ascii=False)
        return json.dumps({"tareas": tareas}, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Transporte")
def inc_transport():
  """Obtiene problemas en el transporte."""
  URL = 'https://tmpmurcia.es/ultima.asp'

  try:
    parsed = urlparse(URL)
    dominio = f"{parsed.scheme}://{parsed.netloc}/"  # esquema + dominio

    resp = requests.get(URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    for a in soup.find_all('a', href=True): # Iterate through all <a> tags
      if "Cuerpo.asp?codigo=" in a['href']:
        enlace = dominio + a['href']
        break

    r = requests.get(enlace)
    r.raise_for_status()
    sub_soup = BeautifulSoup(r.text, 'html.parser')

    # Buscar imagen en /fotos/noticias/
    img_tag = sub_soup.find('img', src=lambda x: x and '/fotos/noticias/' in x)
    if not img_tag:
        return {
            'pagina': enlace,
            'error': 'No se encontrÃ³ imagen en /fotos/noticias/'
            }

    img_url = img_tag['src']
    if not img_url.startswith('http'):
        img_url = f"https://tmpmurcia.es/{img_url.lstrip('/')}"

    # Descargar imagen
    img_resp = requests.get(img_url)
    img_resp.raise_for_status()
    imagen = Image.open(BytesIO(img_resp.content))

    # OCR
    texto = pytesseract.image_to_string(imagen, lang='spa')
    lineas = texto.splitlines()

    # Buscar lÃ­nea 44
    ocurrencias_44 = [line for line in lineas if '44' in line]

    return {
        'pagina': enlace,
        'imagen': img_url,
        'ocurrencias_44': ocurrencias_44,
        'texto': texto,
    }

  except Exception as e:
    return {
        'pagina': enlace,
        'error': str(e)
        }


# Agentes con roles mÃ¡s definidos y anti-alucinaciÃ³n
mail_agent = Agent(
    role="Analista de Comunicaciones General",
    goal="Revisar TODO el correo entrante y reportar ofertas interesantes, alertas de seguridad y notificaciones.",
    backstory="Eres un asistente EA de alto nivel. Tu trabajo es que el usuario no se pierda ninguna oportunidad, ni alerta importante.",
    llm=get_llm(),
    tools=[read_emails],
    verbose=False
)


calendar_agent = Agent(
    role="Lector de Datos de Calendario",
    goal="Extraer y listar LITERALMENTE los eventos que devuelve la herramienta. NO modificar ni inventar nada.",
    backstory="""Eres un programa automatizado, no un humano.
    Tu Ãºnica funciÃ³n es leer la salida JSON de la herramienta de calendario y listarla tal cual.
    Tienes PROHIBIDO inventar reuniones como 'PlanificaciÃ³n', 'Almuerzos' o 'Entrevistas'.
    Si la herramienta dice 'Turno Tardes', tÃº escribes 'Turno Tardes'.
    Si la herramienta no devuelve nada, tÃº dices 'No hay eventos'.""",
    llm=get_llm(),
    tools=[get_todays_agenda],
    verbose=False
)


task_agent = Agent(
    role="Lector de tareas",
    goal="Listar tareas vencidas, para hoy o los prÃ³ximos dos dÃ­as. No sugerir tareas genÃ©ricas.",
    backstory="Solo te importan las tareas registradas en el sistema (Google Tasks).",
    llm=get_llm(),
    tools=[get_todays_tasks],
    verbose=False
)


analyst_agent = Agent(
    role="Analista de Inteligencia de Mercado",
    goal="Proveer datos duros sobre 'El Corte InglÃ©s' y 'Repsol'.",
    backstory="Analista bursÃ¡til. Te basas en datos, no en especulaciones. Usas las herramientas para buscar precios y noticias frescas.",
    llm=get_llm(),
    tools=[get_stock_price, get_financial_news],
    verbose=False
)



transport_agent = Agent(
    role="Analista de incidencias de transporte",
    goal="Identificar y listar los problemas de transporte reportados para la fecha actual.",
    backstory="Este agente se centra exclusivamente en los problemas que el sistema indica, y Ãºnicamente si corresponden al dÃ­a en curso.",
    llm=get_llm(),
    tools=[inc_transport],
    verbose=False
)


briefing_agent = Agent(
    role="Redactor de Reporte Factual",
    goal="Consolidar toda la informaciÃ³n veraz en un reporte ejecutivo.",
    backstory="""
    Eres el Jefe de Gabinete. Tu responsabilidad es generar el 'Briefing' diario.
    REGLA DE ORO: NO INVENTES INFORMACIÃ“N.
    Si no hay correos, dÃ­ 'Sin correos.'.
    Si no hay tareas, NO inventes 'Aprobar pago'.
    Usa estrictamente los datos que te pasan los otros agentes.
    JamÃ¡s agregues eventos que no te hayan pasado explÃ­citamente los otros agentes.
    """,
    llm=get_llm(),
    tools=[send_pushover, send_telegram],
    verbose=False
)


fecha_hoy = datetime.now().strftime('%d/%m/%Y')


mail_scan = Task(
    description=f"Lee TODOS los correos de hoy ({fecha_hoy}). Identifica cualquier notificaciÃ³n relevante y categorizala.",
    expected_output="Lista JSON o bullet points de correos categorizados.",
    agent=mail_agent
)


calendar_scan = Task(
    description=f"""ObtÃ©n la agenda de hoy ({fecha_hoy}). MIRA los datos reales que devuelve. GENERA una lista usando SOLO esos datos.
    âš ï¸� ALERTA DE SEGURIDAD:
    - NO INVENTES "Reuniones de Estrategia".
    - NO INVENTES "Almuerzos de equipo".
    - Si el JSON recuperado contiene eventos reales, ÃšSALOS.
    - Si ignoras los datos reales ("Turno Tardes") para inventar datos falsos, fallarÃ¡s tu misiÃ³n.""",
    expected_output="Resumen cronolÃ³gico de eventos reales del calendario.",
    agent=calendar_agent
)


task_scan = Task(
    description=f"Lista las tareas con vencimiento hoy ({fecha_hoy}).",
    expected_output="Lista de tareas pendientes reales extraÃ­das de Google Tasks.",
    agent=task_agent
)


analyst_scan = Task(
    description="1. ObtÃ©n precio de REP.MC. 2. Busca noticias de 'El Corte InglÃ©s'.",
    expected_output="Datos financieros y resumen de 3 noticias principales.",
    agent=analyst_agent
)


transport_scan = Task(
    description="Verificar si existen incidencias en el transporte para la fecha actual.",
    expected_output="Generar una alerta crÃ­tica Ãºnicamente si se detectan incidencias correspondientes al dÃ­a en curso.",
    agent=transport_agent
)


task_notify = Task(
    description=f"""
    Genera el BRIEFING -{fecha_hoy}.

    ESTRUCTURA OBLIGATORIA:
    1. ğŸ“… Agenda: Resumen de eventos (Solo si existen).
    2. âœ… Tareas: Lista de tareas (Solo si existen).
    3. ğŸ“§ Correos Clave: (Solo si hay importantes).
    4. ğŸ“ˆ Mercado: Precio REP.MC y 1 titular clave de El Corte InglÃ©s.
    5. ğŸšš Transporte: Incidencias de transporte.

    NOTA: NO inventes datos. NO inventes tareas. NO inventes eventos. NO inventes correos. NO inventes noticias. Las incidencias de trasporte son CRITICAS.

    ADVERTENCIA: NO agregues secciones de 'Prioridades CrÃ­ticas' si las tareas estÃ¡n vacÃ­as. SÃ© fiel a los datos.
    Al final, envÃ­a el resumen por Pushover y WhatsApp.
    """,
    expected_output="Reporte enviado y texto final de confirmaciÃ³n.",
    agent=briefing_agent,
    context=[mail_scan, calendar_scan, task_scan, analyst_scan, transport_scan]
)


morning_crew = Crew(
    agents=[mail_agent, calendar_agent, task_agent, analyst_agent, transport_agent, briefing_agent],
    tasks=[mail_scan, calendar_scan, task_scan, analyst_scan, transport_scan, task_notify],
    verbose=False
)


def run_shopping_assistant():
    logger.info("ğŸš€ Iniciando Asistente Ejecutivo IA...")
    try:
        result = morning_crew.kickoff()
        logger.info("âœ… EjecuciÃ³n finalizada con Ã©xito.")
        return result
    except Exception as e:
        logger.critical(f"ğŸ”¥ Error crÃ­tico en la ejecuciÃ³n del Crew: {e}")
        return str(e)


if __name__ == "__main__":
    resultado_crew = run_shopping_assistant()


# COmprobaciÃ³n
print("\n" + "="*40)
print("ğŸ“± VISTA PREVIA DEL MENSAJE ENVIADO")
print("="*40 + "\n")

# Verificamos si la ejecuciÃ³n anterior guardÃ³ el resultado
if 'resultado_crew' in locals() and resultado_crew:
    # Convertimos el resultado a string (CrewAI devuelve un objeto CrewOutput)
    reporte_texto = str(resultado_crew)

    # 1. Mostrar renderizado (Bonito)
    display(Markdown(reporte_texto))

    # 2. (Opcional) Mostrar caracteres raw por si hay errores de formato
    # print("\n--- Texto Plano ---\n", reporte_texto)
else:
    print("âš ï¸� No se encontrÃ³ el reporte ('resultado_crew').")
    print("AsegÃºrate de haber ejecutado la celda principal: resultado_crew = run_shopping_assistant()")


    # Esto realiza llamadas extra a las APIs para generar las tablas visuales
    # Ãštil para depuraciÃ³n o reportes visuales en Notebooks.

    print("\n" + "="*50 + "\n GENERANDO REPORTE VISUAL (TABLAS) \n" + "="*50)

    try:
        # Funciones helper para extracciÃ³n segura
        def safe_json_load(tool_func, **kwargs):
            try:
                res = tool_func._run(**kwargs)
                return json.loads(res)
            except:
                return {}

        raw_emails = safe_json_load(read_emails).get("correos", [])
        raw_agenda = safe_json_load(get_todays_agenda).get("agenda", [])
        raw_tasks  = safe_json_load(get_todays_tasks).get("tareas", [])
        raw_news   = safe_json_load(get_financial_news, busqueda="El Corte InglÃ©s").get("news", [])
        raw_stock  = safe_json_load(get_stock_price, symbol="REP.MC").get("stock", {})

        # Mostrar Markdown
        reporte_md = f"""
        # ğŸ“Š Tablero de Control - {datetime.now().strftime('%Y-%m-%d')}
        """
        display(Markdown(reporte_md))

        if raw_agenda:
            display(Markdown("### ğŸ“… Agenda"))
            display(pd.DataFrame(raw_agenda))
        else:
            display(Markdown("ğŸ“… **Agenda:** Sin eventos."))

        if raw_tasks:
            display(Markdown("### âœ… Tareas"))
            display(pd.DataFrame(raw_tasks))

        if raw_emails and "mensaje" not in raw_emails[0]:
            display(Markdown("### ğŸ“§ Correos Recientes"))
            display(pd.DataFrame(raw_emails))

        if raw_stock:
            display(Markdown(f"### ğŸ“ˆ Mercado ({raw_stock.get('symbol', 'N/A')})"))
            df_stock = pd.DataFrame([raw_stock])
            display(df_stock[['currentPrice', 'dayHigh', 'dayLow', 'regularMarketChangePercent']])

        if raw_news:
            display(Markdown("### ğŸ“° Noticias ECI"))
            display(pd.DataFrame(raw_news)[['titulo', 'fuente', 'fecha']])

    except Exception as e:
        logger.error(f"Error generando visualizaciÃ³n final: {e}")

