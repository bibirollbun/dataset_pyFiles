%%time

#BASIC INFO ABOUT VM

!echo -e "*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "                                 GENERAL SYSTEM INFO:"
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "OS DETAILS:"
!uname -a
!lsb_release -a
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "RAM DETAILS:"
!free -h
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "CPU DETAILS:"
!lscpu
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "GPU DETAILS:"
!nvidia-smi #GPU details
!nvcc --version #GPU visor
!lspci -v #NO gpu visor
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "CONDA DETAILS:"
!conda --version && echo
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "MY CONDA DETAILS:"
!conda info --envs && echo
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"
!echo "CONDA CHANNELS:"
!conda config --show channels && echo
!echo -e "\n<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>\n"


#Install addtional tools to reproduce this workflow if you reproduce from other kaggle VM
%pip install rioxarray rasterio keybert earthengine-api google-auth-oauthlib pystac_client pystac laspy open3d planetary-computer -q


#Cpu version
#%pip freeze > requirements_cpu.txt


#Gpu version
#%pip freeze > requirements_gpu.txt



#Cpu version
#%cat requirements_cpu.txt

#Gpu version
#%cat requirements_gpu.txt


from IPython.display import Image, display

# Mostrar imagen desde ruta local
display(Image(filename='/kaggle/input/amaztest/my_imgs/ETLsd.png'))



%%time
import os
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm # tqdm se mantiene para la barra de progreso de descargas
from typing import List, Optional, Set
from pathlib import Path

class KMLDownloader:
    """
    Descarga archivos KML y KMZ desde una lista de URLs base.
    """
    def __init__(self, base_urls: List[str], max_retries: int = 3, download_dir: str = "kml_downloads"):
        self.base_urls = self._validate_urls(base_urls)
        self.max_retries = max_retries
        self.session = self._create_session()
        
        self.downloads_dir = Path(download_dir)
        self._setup_directories()
        
        self.downloaded_kml_urls = set() # URLs de KML/KMZ que ya se intentaron descargar
        self.failed_page_fetches = set() # URLs de pÃ¡ginas HTML que no se pudieron obtener
        self.failed_kml_downloads = set() # URLs de KML/KMZ que fallaron al descargar
        self.successful_downloads_count = 0
        self.already_exists_count = 0

    def _validate_urls(self, urls: List[str]) -> List[str]:
        validated = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                validated.append(url)
            else:
                logging.warning(f"URL base invÃ¡lida descartada: {url}")
        return validated

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        })
        return session

    def _setup_directories(self) -> None:
        self.downloads_dir.mkdir(exist_ok=True)
        logging.info(f"Los archivos se guardarÃ¡n en: {self.downloads_dir.resolve()}")

    def _get_clean_filename(self, url: str) -> Path:
        """Genera un nombre de archivo seguro desde una URL para KML/KMZ."""
        parsed_url = urlparse(url)
        basename = os.path.basename(parsed_url.path)
        
        # Si basename estÃ¡ vacÃ­o o es solo '/', intenta construir uno mejor
        if not basename or basename == '/':
            path_parts = [part for part in parsed_url.path.split('/') if part]
            if path_parts:
                basename = path_parts[-1]
            else: # Si no hay path, usa parte del netloc
                basename = parsed_url.netloc.replace('.', '_')
        
        # Limpiar el nombre base
        clean_name_base = ''.join(c for c in basename if c.isalnum() or c in ('-', '_'))
        
        # Determinar extensiÃ³n
        if url.lower().endswith('.kmz'):
            extension = '.kmz'
        elif url.lower().endswith('.kml'):
            extension = '.kml'
        else: # Si no tiene extensiÃ³n clara, adivinar o poner .kml por defecto
            extension = '.kml' 
            logging.debug(f"URL {url} sin extensiÃ³n clara, asumiendo {extension}")

        # Si el nombre base ya tiene una extensiÃ³n correcta, no duplicarla
        if clean_name_base.lower().endswith(extension):
            clean_name = clean_name_base
        else:
            clean_name = clean_name_base + extension
            
        # Si el nombre resultante es solo la extensiÃ³n (ej: ".kml"), prefijar con "file"
        if clean_name == extension:
            clean_name = "file" + extension
            
        return self.downloads_dir / clean_name

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Descarga el contenido HTML de una pÃ¡gina."""
        logging.info(f"Accediendo a la pÃ¡gina: {url}")
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status() # Lanza HTTPError para respuestas 4xx/5xx
                return response.text
            except requests.RequestException as e:
                logging.warning(f"Intento {attempt+1}/{self.max_retries} fallido para obtener {url}: {e}")
        logging.error(f"No se pudo acceder a la pÃ¡gina {url} despuÃ©s de {self.max_retries} intentos.")
        self.failed_page_fetches.add(url)
        return None

    def extract_kml_links_from_html(self, html_content: str, base_page_url: str) -> Set[str]:
        """Extrae todos los enlaces a archivos .kml o .kmz del HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        kml_links = set()
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href'].strip()
            if href.lower().endswith(('.kml', '.kmz')):
                # Construir URL absoluta si es relativa
                full_url = urljoin(base_page_url, href)
                kml_links.add(full_url)
        if kml_links:
            logging.info(f"Encontrados {len(kml_links)} enlaces KML/KMZ en {base_page_url}")
        return kml_links

    def download_single_kml_file(self, kml_url: str) -> bool:
        """Descarga un Ãºnico archivo KML/KMZ."""
        if kml_url in self.downloaded_kml_urls:
            # Ya se intentÃ³, no reintentar a menos que se implemente una lÃ³gica mÃ¡s compleja
            return False 

        self.downloaded_kml_urls.add(kml_url)
        target_filepath = self._get_clean_filename(kml_url)

        if target_filepath.exists() and target_filepath.stat().st_size > 0:
            logging.info(f"Archivo ya existe y no estÃ¡ vacÃ­o: {target_filepath.name}")
            self.already_exists_count += 1
            return True # Considerar como Ã©xito si ya existe

        temp_filepath = target_filepath.with_suffix(target_filepath.suffix + '.tmpdownload')
        
        try:
            with self.session.get(kml_url, stream=True, timeout=60) as response: # Timeout mÃ¡s largo para descargas
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(temp_filepath, 'wb') as f, tqdm(
                    total=total_size, 
                    unit='iB', 
                    unit_scale=True,
                    desc=f"Descargando {target_filepath.name}",
                    leave=False, # La barra desaparece despuÃ©s de completarse
                    mininterval=0.5 # Actualizar no mÃ¡s de 2 veces por segundo
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: # Filtrar chunks de keep-alive
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verificar si el archivo temporal se escribiÃ³ y no estÃ¡ vacÃ­o
            if temp_filepath.exists() and temp_filepath.stat().st_size > 0:
                temp_filepath.rename(target_filepath)
                logging.info(f"Descargado: {target_filepath.name}")
                self.successful_downloads_count += 1
                return True
            elif temp_filepath.exists(): # Existe pero estÃ¡ vacÃ­o
                logging.warning(f"Descarga de {kml_url} resultÃ³ en un archivo vacÃ­o ({target_filepath.name}). Eliminando.")
                temp_filepath.unlink()
                self.failed_kml_downloads.add(kml_url)
                return False
            else: # No se creÃ³ el archivo temporal
                logging.error(f"Archivo temporal para {kml_url} no se creÃ³.")
                self.failed_kml_downloads.add(kml_url)
                return False

        except requests.RequestException as e:
            logging.error(f"Error de red al descargar {kml_url}: {e}")
        except IOError as e:
            logging.error(f"Error de E/S al guardar {kml_url} como {target_filepath.name}: {e}")
        except Exception as e:
            logging.error(f"Error inesperado al descargar {kml_url}: {e}", exc_info=False) # exc_info=False para ser menos verboso
        
        # Si hubo una excepciÃ³n, limpiar el archivo temporal si existe
        if temp_filepath.exists():
            temp_filepath.unlink(missing_ok=True)
        self.failed_kml_downloads.add(kml_url)
        return False

    def process_single_page_url(self, page_url: str):
        """Obtiene una pÃ¡gina, extrae enlaces KML/KMZ y los descarga."""
        html_content = self.fetch_page_content(page_url)
        if not html_content:
            return # fetch_page_content ya registrÃ³ el fallo

        kml_file_urls = self.extract_kml_links_from_html(html_content, page_url)
        if not kml_file_urls:
            logging.info(f"No se encontraron archivos KML/KMZ en la pÃ¡gina: {page_url}")
            return

        logging.info(f"Procesando {len(kml_file_urls)} KML/KMZ links de {page_url}")
        for kml_url in kml_file_urls: # No usar tqdm aquÃ­, download_single_kml_file tiene su propia barra
            self.download_single_kml_file(kml_url)

    def run_downloader(self):
        """Ejecuta el proceso completo de descarga para todas las URLs base."""
        logging.info(f"ğŸš€ Iniciando descarga de archivos KML/KMZ...")
        if not self.base_urls:
            logging.warning("No hay URLs base vÃ¡lidas para procesar.")
            return

        for page_url in self.base_urls: # No usar tqdm para las pÃ¡ginas base
            self.process_single_page_url(page_url)
        
        logging.info("âœ… Proceso de descarga completado.")
        logging.info(f"Resumen:")
        logging.info(f"  PÃ¡ginas HTML procesadas: {len(self.base_urls) - len(self.failed_page_fetches)} de {len(self.base_urls)}")
        logging.info(f"  Descargas KML/KMZ exitosas: {self.successful_downloads_count}")
        logging.info(f"  Archivos KML/KMZ que ya existÃ­an: {self.already_exists_count}")
        logging.info(f"  Fallos al obtener pÃ¡ginas HTML: {len(self.failed_page_fetches)}")
        logging.info(f"  Fallos al descargar archivos KML/KMZ: {len(self.failed_kml_downloads)}")

        if self.failed_page_fetches:
            logging.warning("PÃ¡ginas HTML que fallaron:")
            for url in sorted(list(self.failed_page_fetches)):
                logging.warning(f"  - {url}")
        if self.failed_kml_downloads:
            logging.warning("Archivos KML/KMZ que fallaron al descargar:")
            for url in sorted(list(self.failed_kml_downloads)):
                logging.warning(f"  - {url}")

if __name__ == "__main__":
    # ConfiguraciÃ³n del logging
    logging.basicConfig(
        level=logging.INFO, # Cambiar a logging.DEBUG para mÃ¡s detalles
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # URLs base de donde extraer los enlaces a KML/KMZ
    target_urls = [
        "https://www.jqjacobs.net/blog/",
        *[f"https://www.jqjacobs.net/blog/{year}_posts.html" for year in range(2007, 2024)]
    ]

    # Crear una instancia del descargador y ejecutarlo
    # Los archivos se guardarÃ¡n en "kml_downloads" en el directorio actual por defecto
    downloader = KMLDownloader(base_urls=target_urls, download_dir="downloads_jqjacobs")
    downloader.run_downloader()


%%time
import os
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
import zipfile
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from io import BytesIO

class KMLKMZToParquetConverter:
    def __init__(self, input_dir, output_dir, max_workers=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        self.max_workers = max_workers or min(8, cpu_count() * 2)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        files = list(self.input_dir.glob("*.kml")) + list(self.input_dir.glob("*.kmz"))

        with tqdm(total=len(files), desc="ğŸ“¦ Procesando archivos", unit="file") as pbar:
            for file_path in files:
                try:
                    self.process_file(file_path)
                except Exception as e:
                    tqdm.write(f"â�Œ Error procesando {file_path.name}: {str(e)}")
                pbar.update(1)

    def process_file(self, file_path):
        layers = self._extract_layers(file_path)

        for layer_name, features in layers.items():
            if not features:
                continue

            output_name = re.sub(r'[\\/*?:"<>|]', "_", layer_name)
            output_path = self.output_dir / f"{file_path.stem}_{output_name}.parquet"

            df = pd.DataFrame(features)
            df = self._flatten_coordinates(df)
            df.to_parquet(output_path, engine='pyarrow', index=False)

    def _extract_layers(self, file_path):
        if file_path.suffix.lower() == ".kmz":
            with zipfile.ZipFile(file_path, 'r') as z:
                kml_file = next((f for f in z.namelist() if f.endswith(".kml")), None)
                if not kml_file:
                    return {}
                with z.open(kml_file) as kml:
                    tree = ET.parse(kml)
        else:
            tree = ET.parse(file_path)

        root = tree.getroot()
        folders = root.findall('.//kml:Folder', self.ns)

        if not folders:
            features = self._parse_features(root)
            return {"main_layer": features} if features else {}

        layers = {}
        for i, folder in enumerate(folders):
            name_elem = folder.find('kml:name', self.ns)
            name = name_elem.text.strip() if name_elem is not None else f"layer_{i}"
            features = self._parse_features(folder)
            if features:
                layers[name] = features

        return layers

    def _parse_features(self, element):
        placemarks = element.findall('.//kml:Placemark', self.ns)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(filter(None, executor.map(self._parse_feature, placemarks)))

    def _parse_feature(self, pm):
        def coords_from_text(text):
            parts = [c.strip() for c in text.strip().split()]
            coords = []
            for c in parts:
                try:
                    lon, lat = map(float, c.split(',')[:2])
                    coords.append((lon, lat))
                except ValueError:
                    continue
            return coords

        try:
            name = self._get_text(pm, 'kml:name')
            desc = self._get_text(pm, 'kml:description')
            style = self._get_text(pm, 'kml:styleUrl')

            for geom_type, xpath in [('Point', './/kml:Point/kml:coordinates'),
                                     ('LineString', './/kml:LineString/kml:coordinates'),
                                     ('Polygon', './/kml:Polygon//kml:coordinates')]:
                elem = pm.find(xpath, self.ns)
                if elem is not None and elem.text:
                    coords = coords_from_text(elem.text)
                    if coords:
                        if geom_type == 'Polygon':
                            coords = [coords]  # wrap polygon
                        return {
                            'name': name,
                            'description': desc,
                            'styleUrl': style,
                            'geometry_type': geom_type,
                            'coordinates': coords
                        }
        except Exception:
            pass
        return None

    def _get_text(self, elem, path):
        node = elem.find(path, self.ns)
        return node.text.strip() if node is not None and node.text else ""

    def _flatten_coordinates(self, df):
        result = df.copy()

        for geom in ['Point', 'LineString', 'Polygon']:
            mask = result['geometry_type'] == geom
            if mask.any():
                if geom == 'Polygon':
                    coords = result.loc[mask, 'coordinates'].apply(
                        lambda x: {'longitud': x[0][0], 'latitud': x[0][1]} if x else {}
                    )
                else:
                    coords = result.loc[mask, 'coordinates'].apply(
                        lambda x: {'longitud': x[0][0], 'latitud': x[0][1]} if x else {}
                    )
                coords_df = pd.json_normalize(coords)
                result.loc[mask, ['longitud', 'latitud']] = coords_df[['longitud', 'latitud']].values

        result.drop(columns=['coordinates'], inplace=True, errors='ignore')
        for col in ['longitud', 'latitud']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        for col in ['name', 'description', 'styleUrl', 'geometry_type']:
            if col in result.columns:
                result[col] = result[col].astype('string')

        return result

if __name__ == "__main__":
    INPUT_DIR = "/kaggle/working/downloads_jqjacobs/"
    OUTPUT_DIR = "/kaggle/working/parquet_jqjacobs/"

    converter = KMLKMZToParquetConverter(INPUT_DIR, OUTPUT_DIR)
    converter.run()


%%time
import pandas as pd
dt1 = pd.read_csv("/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv")
dt2 = pd.read_csv("/kaggle/input/archaeological-survey-data/submit.csv")
dt3 = pd.read_csv("/kaggle/input/casarabe-sites-utm/casarabe_sites_utm.csv")
dt4 = pd.read_csv("/kaggle/input/mound-villages-acre/mound_villages_acre.csv")
dt5 = pd.read_csv("/kaggle/input/science-data/science.ade2541_data_s2.csv")

print(f" amazon_geoglyphs_sites >>  {dt1.head(2)} \n", f" submit >>  {dt2.head(2)} \n",f" casarabe_sites_utm >> {dt3.head(2)} \n", f" mound_villages_acre >> {dt4.head(2)} \n",f" ade2541_data_s2 >> {dt5.head(2)}")
print(f" amazon_geoglyphs_sites >>  {dt1.dtypes} \n", f" submit >>  {dt2.dtypes} \n",f" casarabe_sites_utm >> {dt3.dtypes} \n", f" mound_villages_acre >> {dt4.dtypes} \n",f" ade2541_data_s2 >> {dt5.dtypes}")


%%time
"""Ver datos nuevos de las colecciones dt_n"""
from pyproj import Transformer
import pandas as pd
from pathlib import Path
import dask.dataframe as dd

class FilterTags:
    def __init__(self, parquet_folder):
        self.parquet_folder = Path(parquet_folder)
        self.datasets = {}
        
    def load_datasets(self):
        """Carga y procesa cada dataset extrayendo sus coordenadas como (lon, lat)"""
        
        dt1 = pd.read_csv("/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv")
        self.datasets["amazon_geoglyphs_sites"] = self._extract_points(
            dt1, lat_col="latitude", lon_col="longitude"
        )
        
        dt2 = pd.read_csv("/kaggle/input/archaeological-survey-data/submit.csv")
        self.datasets["submit"] = self._extract_points(
            dt2, lat_col="y", lon_col="x"
        )
        
        dt3 = pd.read_csv("/kaggle/input/casarabe-sites-utm/casarabe_sites_utm.csv")
        self.datasets["casarabe_sites_utm"] = self._convert_utm_to_latlon(
            dt3, zone_number=20, zone_letter='N', name="casarabe_sites_utm"
        )
        
        dt4 = pd.read_csv("/kaggle/input/mound-villages-acre/mound_villages_acre.csv")
        self.datasets["mound_villages_acre"] = self._convert_utm_to_latlon(
            dt4, easting_col="UTM X (Easting)", northing_col="UTM Y (Northing)", 
            zone_number=19, zone_letter='M', name="mound_villages_acre"
        )
        
        dt5 = pd.read_csv("/kaggle/input/science-data/science.ade2541_data_s2.csv")
        self.datasets["science_data"] = self._extract_points(
            dt5, lat_col="Latitude", lon_col="Longitude"
        )
    
    def _extract_points(self, df, lat_col, lon_col):
        """Extrae tuplas (lon, lat) desde columnas de lat/lon"""
        points = set()
        for _, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                points.add((lon, lat))
            except (ValueError, KeyError, TypeError):
                continue
        return list(points)
    
    def _convert_utm_to_latlon(self, df, easting_col="UTM X (Easting)", northing_col="UTM Y (Northing)", 
                             zone_number=None, zone_letter=None, name="Unknown"):
        """Convierte coordenadas UTM a Latitud/Longitud usando pyproj"""
        if zone_number is None or zone_letter is None:
            print(f"âš ï¸� Zona UTM no especificada para: {name}")
            return []
            
        points = set()
        transformer = Transformer.from_crs(
            f"EPSG:326{zone_number}",  # CÃ³digo EPSG para UTM zona norte
            "EPSG:4326",               # CÃ³digo EPSG para WGS84 (lat/lon)
            always_xy=True             # Para obtener (lon, lat) en lugar de (lat, lon)
        )
        
        for _, row in df.iterrows():
            try:
                easting = float(row[easting_col])
                northing = float(row[northing_col])
                lon, lat = transformer.transform(easting, northing)
                points.add((lon, lat))
            except (ValueError, KeyError, TypeError) as e:
                print(f"âš ï¸� Error procesando fila en {name}: {e}")
                continue
                
        print(f"âœ… Converted {len(points)} UTM points to Lat/Lon for: {name}")
        return list(points)
    
    def collect_parquet_points(self):
        """Recoge todos los puntos Ãºnicos desde los archivos Parquet"""
        all_parquet_points = set()
        parquet_files = [f for f in self.parquet_folder.glob("*.parquet")]
        if not parquet_files:
            print("âš ï¸� No files found Parquet")
            return set()
        for file in parquet_files:
            try:
                df_pq = dd.read_parquet(file).compute()
                if "latitud" in df_pq.columns and "longitud" in df_pq.columns:
                    for _, row in df_pq.iterrows():
                        try:
                            lat = float(row["latitud"])
                            lon = float(row["longitud"])
                            all_parquet_points.add((lon, lat))
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                print(f"â�Œ Error leyendo {file.name}: {e}")
        return all_parquet_points
    
    def report_new_unique_points(self, parquet_points):
        """Genera informe de puntos NUEVOS (no presentes en Parquet)"""
        print("\nğŸ†• NEW Sites Report:")
        total_new = 0
        for name, points in self.datasets.items():
            new_points = [pt for pt in points if pt not in parquet_points]
            count = len(new_points)
            if count > 0:
                print(f"ğŸ“� {name}: {count} new sites")
                total_new += count
        print(f"âœ… TTotal number of NEW sites: {total_new}")

# --- EjecuciÃ³n principal ---
if __name__ == "__main__":
    FOLDER_PATH = "/kaggle/working/parquet_jqjacobs"

    print("ğŸ“¥ Loading datasets and extracting coordinates...")
    tag_filter = FilterTags(FOLDER_PATH)
    tag_filter.load_datasets()

    print("ğŸ”� Collecting existing points in Parquet...")
    parquet_points = tag_filter.collect_parquet_points()

    print("ğŸ§¾ Generating new site report...")
    tag_filter.report_new_unique_points(parquet_points)

    print("\nâœ… Process completed.")


%%time
import folium
import requests
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from io import StringIO
import os

class AmazonMapGenerator:
    """Genera un mapa interactivo del Amazonas con capacidad de guardar datos intermedios como Parquet."""
    
    def __init__(self, location=(-3.4653, -62.2159), zoom_start=4):
        # Crear directorios independientes (no anidados)
        os.makedirs("amazon_maps", exist_ok=True)
        os.makedirs("amazon_data", exist_ok=True)
        
        self.map = folium.Map(
            location=location,
            zoom_start=zoom_start,
            control_scale=True,
            tiles='CartoDB Positron'
        )
        self.raisg_rest_api_base_url = "https://geo2.socioambiental.org/raisg/rest/services/raisg/raisg_base_N/MapServer"

    def _fetch_geojson(self, layer_id: int, layer_name: str):
        """Descarga GeoJSON desde la API REST de RAISG."""
        url = f"{self.raisg_rest_api_base_url}/{layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson"
        }
        print(f"â�³ Descargando '{layer_name}' (ID {layer_id})...")
        try:
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            data = response.json()
            print(f"âœ… GeoJSON recibido para '{layer_name}' ({len(data['features'])} features).")
            return data
        except Exception as e:
            print(f"â�Œ Error al obtener '{layer_name}': {e}")
            return None

    def _flatten_to_parquet(self, geojson_data: dict, layer_name: str):
        """Convierte GeoJSON a GeoDataFrame y guarda como Parquet."""
        try:
            # Convertir a GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            
            # Aplanar geometrÃ­as complejas
            gdf = gdf.explode(index_parts=True).reset_index(drop=True)
            
            # Nombre del archivo seguro
            safe_name = layer_name.replace(" ", "_").lower()
            parquet_path = os.path.join("amazon_data", f"{safe_name}.parquet")
            
            # Guardar como Parquet
            gdf.to_parquet(parquet_path)
            print(f"ğŸ’¾ Datos de '{layer_name}' guardados como Parquet en: {parquet_path}")
            
            # EstadÃ­sticas
            print(f"ğŸ“Š Resumen de datos ({layer_name}):")
            print(f"- Columnas: {list(gdf.columns)}")
            print(f"- Registros: {len(gdf)}")
            print(f"- CRS: {gdf.crs}")
            
            return gdf
        except Exception as e:
            print(f"â�Œ Error al procesar '{layer_name}': {str(e)}")
            return None

    def add_layers(self, save_parquet=True):
        """Agrega capas al mapa con opciÃ³n de guardar datos intermedios."""
        layers = {
            7: {
                "name": "Contorno del Amazonas (LÃ­mite RAISG)",
                "style": lambda f: {'color': '#E300FF', 'weight': 3, 'fillOpacity': 0.0}
            },
            9: {
                "name": "Ã�rea BiogeogrÃ¡fica (Relleno)",
                "style": lambda f: {'fillColor': '#008000', 'fillOpacity': 0.5, 'weight': 0},
                "show": False
            }
        }
        for layer_id, cfg in layers.items():
            data = self._fetch_geojson(layer_id, cfg["name"])
            if data:
                if save_parquet:
                    self._flatten_to_parquet(data, cfg["name"])
                
                folium.GeoJson(
                    data,
                    name=cfg["name"],
                    style_function=cfg["style"],
                    show=cfg.get("show", True)
                ).add_to(self.map)
                print(f"ğŸŸ¢ Capa '{cfg['name']}' aÃ±adida al mapa.")
        folium.LayerControl(collapsed=False).add_to(self.map)

    def save_map(self, filename="amazon_map_definitivo_geojson.html"):
        """Guarda el mapa en un archivo HTML."""
        output_path = os.path.join("amazon_maps", filename)
        self.map.save(output_path)
        print(f"ğŸ’¾ Mapa guardado como '{output_path}'")

if __name__ == "__main__":
    print("ğŸš€ Iniciando generaciÃ³n del mapa...")
    mapa = AmazonMapGenerator()
    
    mapa.add_layers(save_parquet=True)
    mapa.save_map()
    
    print("ğŸ�‰ Â¡Proceso completo!")    


%%time
import os
import geopandas as gpd
import folium

class AmazonTerritoriosIndigenas:
    def __init__(self, base_path):
        """Inicializa el procesador con rutas organizadas"""
        self.base_path = base_path
        self.territories_gdf = None
        
        # Crear estructura de directorios
        self.data_dir = "amazon_data"
        self.maps_dir = "amazon_maps"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)

    def load_and_flatten(self):
        """Carga y aplana el shapefile de territorios indÃ­genas"""
        shp_path = os.path.join(self.base_path, "Tis_TerritoriosIndigenas.shp")
        
        if not os.path.exists(shp_path):
            raise FileNotFoundError("â�Œ Archivo Tis_TerritoriosIndigenas.shp no encontrado")
        
        # Cargar y procesar datos
        self.territories_gdf = gpd.read_file(shp_path)
        print(f"ğŸ“Œ Cargados {len(self.territories_gdf)} territorios indÃ­genas")
        
        # Aplanar geometrÃ­as complejas
        flattened = self.territories_gdf.explode(index_parts=True).reset_index(drop=True)
        
        # Guardar datos procesados
        parquet_path = os.path.join(self.data_dir, "territorios_indigenas.parquet")
        flattened.to_parquet(parquet_path)
        print(f"ğŸ’¾ Datos guardados en: {parquet_path}")
        
        return flattened

    def generate_map(self, map_name="mapa_territorios.html"):
        """Genera mapa interactivo con los territorios indÃ­genas"""
        if self.territories_gdf is None:
            self.load_and_flatten()
        
        # ConfiguraciÃ³n del mapa
        m = folium.Map(
            location=[-3.4653, -62.2159],
            zoom_start=5,
            tiles="CartoDB positron",
            control_scale=True
        )
        
        # Capa de territorios indÃ­genas
        folium.GeoJson(
            self.territories_gdf,
            name="Territorios IndÃ­genas",
            style_function=lambda x: {
                'fillColor': '#FFD37F',
                'color': '#800000',
                'weight': 1,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nombre", "pais"],
                aliases=["Territorio: ", "PaÃ­s: "],
                localize=True
            )
        ).add_to(m)
        
        # Control de capas y guardado
        folium.LayerControl().add_to(m)
        output_path = os.path.join(self.maps_dir, map_name)
        m.save(output_path)
        print(f"ğŸŒ� Mapa guardado en: {output_path}")
        return output_path

    def process_all(self):
        """Ejecuta todo el pipeline de procesamiento"""
        print("\n=== PROCESAMIENTO DE TERRITORIOS INDÃ�GENAS ===")
        self.load_and_flatten()
        map_path = self.generate_map()
        print("\nâœ… Proceso completado")
        print(f"ğŸ“‚ Datos: {self.data_dir}/")
        print(f"ğŸ—ºï¸�  Mapa: {map_path}")
        return map_path


if __name__ == "__main__":    
    BASE_PATH = "/kaggle/input/amaztest/Raisg24Tis"  # Ruta a los datos fuente
    
    # EjecuciÃ³n
    processor = AmazonTerritoriosIndigenas(BASE_PATH)
    processor.process_all()


display(Image(filename='/kaggle/input/amaztest/my_imgs/Tagged.png'))


%%time
"""IdentificaciÃ³n de comunidades desconocidas en el Amazonas"""
import geopandas as gpd
import folium
from shapely.ops import unary_union
from pathlib import Path


class UnlabeledTerritoryProcessor:
    def __init__(self, amazon_path, indigenous_path):
        """
        Inicializa el procesador usando rutas fijas de salida en carpetas existentes.
        
        Args:
            amazon_path (str): Ruta al archivo Parquet del contorno amazÃ³nico.
            indigenous_path (str): Ruta al archivo Parquet de territorios indÃ­genas.
        """
        self.amazon_path = amazon_path
        self.indigenous_path = indigenous_path
        
        # Usamos las carpetas ya existentes
        self.data_dir = Path("amazon_data")
        self.maps_dir = Path("amazon_maps")

        # Validar que existan
        if not self.data_dir.exists():
            raise FileNotFoundError(f"No se encontrÃ³ la carpeta: {self.data_dir}")
        if not self.maps_dir.exists():
            raise FileNotFoundError(f"No se encontrÃ³ la carpeta: {self.maps_dir}")

        # Atributos para datos
        self.amazon = None
        self.indigenous = None
        self.unlabeled = None

    def _load_data(self):
        """Carga y prepara los datasets geoespaciales con validaciÃ³n de CRS"""
        print("ğŸ“¥ Cargando datos geoespaciales...")
        self.amazon = gpd.read_parquet(self.amazon_path)
        self.indigenous = gpd.read_parquet(self.indigenous_path)

        for gdf in [self.amazon, self.indigenous]:
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf.to_crs("EPSG:4326", inplace=True)

    def compute_difference(self):
        """Calcula Ã¡reas no etiquetadas como diferencia espacial"""
        self._load_data()
        print("ğŸ§® Calculando Ã¡reas no etiquetadas...")

        amazon_union = unary_union(self.amazon.geometry)
        indigenous_union = unary_union(self.indigenous.geometry)
        difference_geom = amazon_union.difference(indigenous_union)

        if difference_geom.is_empty:
            print("âš ï¸� No se encontraron territorios no etiquetados.")
            self.unlabeled = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            self.unlabeled = gpd.GeoDataFrame(geometry=[difference_geom], crs="EPSG:4326")

    def save_results(self):
        """Guarda los resultados en formato Parquet"""
        if self.unlabeled is None:
            raise ValueError("Primero debe calcular las diferencias")

        output_path = self.data_dir / "unlabeled_territories.parquet"
        self.unlabeled.to_parquet(output_path, index=False)
        print(f"ğŸ’¾ Datos guardados en: {output_path}")
        return output_path

    def generate_map(self):
        """Genera mapa interactivo con estilos profesionales"""
        if self.unlabeled is None or self.unlabeled.empty:
            print("âš ï¸� No se generÃ³ mapa: geometrÃ­as vacÃ­as.")
            return None

        centroid = self.unlabeled.unary_union.centroid
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True
        )

        folium.GeoJson(
            self.unlabeled,
            name="Territorios no etiquetados",
            style_function=lambda x: {
                "fillColor": "#FF9900",
                "color": "#CC6600",
                "weight": 2,
                "fillOpacity": 0.5,
                "dashArray": "5, 5"
            },
            tooltip="Territorio sin clasificaciÃ³n"
        ).add_to(m)

        folium.LayerControl(position="topright").add_to(m)

        map_path = self.maps_dir / "unlabeled_territories_map.html"
        m.save(map_path)
        print(f"ğŸ—ºï¸� Mapa interactivo guardado en: {map_path}")
        return map_path

    def run_pipeline(self):
        """Ejecuta el flujo completo de procesamiento"""
        print("\n=== PROCESAMIENTO DE TERRITORIOS NO ETIQUETADOS ===")
        try:
            self.compute_difference()
            data_path = self.save_results()
            map_path = self.generate_map()

            print("\nâœ… Proceso completado con Ã©xito")
            print(f"ğŸ“Š Datos: {data_path}")
            print(f"ğŸŒ� Mapa: {map_path}")

            return {
                "data_path": str(data_path),
                "map_path": str(map_path),
                "success": True
            }

        except Exception as e:
            print(f"\nâ�Œ Error en el procesamiento: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Ejemplo de uso
if __name__ == "__main__":
    processor = UnlabeledTerritoryProcessor(
        amazon_path="amazon_data/contorno_del_amazonas_(lÃ­mite_raisg).parquet",
        indigenous_path="amazon_data/territorios_indigenas.parquet"
    )
    results = processor.run_pipeline()


display(Image(filename='/kaggle/input/amaztest/my_imgs/noTagged.png'))


%%time
"""GeneraciÃ³n de Bounding Box para el Ã¡rea de estudio del Amazonas"""

import geopandas as gpd
import folium
from shapely.geometry import Polygon
from pathlib import Path

class AmazonStudyArea:
    def __init__(self, parquet_path):
        """Inicializa el procesador con estructura de directorios organizada"""
        self.parquet_path = parquet_path
        self.gdf = None
        self.bbox = None
        
        # Configurar estructura de directorios
        self.maps_dir = Path("amazon_maps")
        self.maps_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Carga y valida los datos geoespaciales"""
        self.gdf = gpd.read_parquet(self.parquet_path)
        
        # Validar y asegurar el sistema de referencia
        if self.gdf.crs is None:
            self.gdf = self.gdf.set_crs("EPSG:4326")
        else:
            self.gdf = self.gdf.to_crs("EPSG:4326")
            
        print(f"ğŸ“Œ Datos cargados ({len(self.gdf)} features, CRS: {self.gdf.crs})")
        return self.gdf

    def calculate_bounding_box(self):
        """Calcula el bounding box del Ã¡rea de estudio"""
        if self.gdf is None:
            self.load_data()
            
        total_bounds = self.gdf.total_bounds
        self.bbox = {
            'minx': total_bounds[0],
            'miny': total_bounds[1],
            'maxx': total_bounds[2],
            'maxy': total_bounds[3]
        }
        
        print("\nğŸ“� Bounding Box (WGS84):")
        print(f"SO: ({self.bbox['minx']:.6f}, {self.bbox['miny']:.6f})")
        print(f"NE: ({self.bbox['maxx']:.6f}, {self.bbox['maxy']:.6f})")
        return self.bbox

    def _create_bbox_polygon(self):
        """Crea un polÃ­gono del bounding box"""
        if self.bbox is None:
            self.calculate_bounding_box()
            
        return Polygon([
            [self.bbox['minx'], self.bbox['miny']],
            [self.bbox['maxx'], self.bbox['miny']],
            [self.bbox['maxx'], self.bbox['maxy']],
            [self.bbox['minx'], self.bbox['maxy']]
        ])

    def generate_study_area_map(self, filename="amazon_study_area.html"):
        """Genera mapa interactivo con el Ã¡rea de estudio delimitada"""
        if self.gdf is None:
            self.load_data()
        if self.bbox is None:
            self.calculate_bounding_box()
            
        # ConfiguraciÃ³n del mapa
        center_y = (self.bbox['miny'] + self.bbox['maxy']) / 2
        center_x = (self.bbox['minx'] + self.bbox['maxx']) / 2
        
        m = folium.Map(
            location=[center_y, center_x],
            zoom_start=5,
            tiles="CartoDB positron",
            control_scale=True
        )
        
        # Capa del Amazonas original
        folium.GeoJson(
            self.gdf,
            name="Amazonas Original",
            style_function=lambda x: {
                'color': '#008000',
                'weight': 2,
                'fillOpacity': 0.1
            },
            tooltip="Ã�rea del Amazonas"
        ).add_to(m)
        
        # Bounding Box
        bbox_polygon = gpd.GeoDataFrame(
            geometry=[self._create_bbox_polygon()], 
            crs="EPSG:4326"
        )
        
        folium.GeoJson(
            bbox_polygon,
            name="Ã�rea de Estudio",
            style_function=lambda x: {
                'color': '#FF0000',
                'weight': 3,
                'fillOpacity': 0,
                'dashArray': '5, 5'
            },
            tooltip="LÃ­mite rectangular de estudio"
        ).add_to(m)
        
        # Marcadores de esquinas con informaciÃ³n detallada
        corners = [
            ("SO", self.bbox['minx'], self.bbox['miny']),
            ("SE", self.bbox['maxx'], self.bbox['miny']),
            ("NE", self.bbox['maxx'], self.bbox['maxy']),
            ("NO", self.bbox['minx'], self.bbox['maxy'])
        ]
        
        for label, lon, lat in corners:
            folium.Marker(
                location=[lat, lon],
                popup=f"{label}: {abs(lat):.4f}Â°{'N' if lat >=0 else 'S'}, {abs(lon):.4f}Â°{'E' if lon >=0 else 'W'}",
                icon=folium.Icon(color='red', icon='map-pin')
            ).add_to(m)
        
        # Control de capas y guardado
        folium.LayerControl(position="topright").add_to(m)
        output_path = self.maps_dir / filename
        m.save(output_path)
        
        print(f"\nğŸŒ� Mapa guardado en: {output_path}")
        return output_path

    def process_all(self):
        """Ejecuta todo el pipeline de procesamiento"""
        print("\n=== DELIMITACIÃ“N DEL Ã�REA DE ESTUDIO ===")
        self.load_data()
        self.calculate_bounding_box()
        map_path = self.generate_study_area_map()
        
        print("\nâœ… Proceso completado")
        print(f"ğŸ—ºï¸� Mapa generado: {map_path}")
        return map_path


# Ejemplo de uso
if __name__ == "__main__":
    try:
        processor = AmazonStudyArea(
            "amazon_data/contorno_del_amazonas_(lÃ­mite_raisg).parquet"
        )
        processor.process_all()
    except Exception as e:
        print(f"â�Œ Error en el procesamiento: {str(e)}")
        raise


display(Image(filename='/kaggle/input/amaztest/my_imgs/amz_region.png'))


%%time
"""DivisiÃ³n del Ã¡rea del Amazonas en chunks para anÃ¡lisis distribuido"""

import folium
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from pathlib import Path

class AmazonGridDivider:
    def __init__(self, contour_path):
        """
        Inicializa el divisor con estructura de directorios organizada
        
        Args:
            contour_path: Ruta al archivo Parquet del contorno RAISG
        """
        self.contour_path = contour_path
        self.contour = None
        self.bbox = None
        
        # Configurar estructura de directorios
        self.data_dir = Path("amazon_data")
        self.maps_dir = Path("amazon_maps")
        self.data_dir.mkdir(exist_ok=True)
        self.maps_dir.mkdir(exist_ok=True)

    def _load_contour(self):
        """Carga y valida el contorno del Amazonas"""
        gdf = gpd.read_parquet(self.contour_path)
        
        # ValidaciÃ³n y limpieza de datos
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
            
        gdf = (
            gdf.drop_duplicates(subset=['geometry'])
              .loc[gdf.geometry.is_valid]
              .to_crs("EPSG:4326")
        )
        
        self.contour = gdf
        self.bbox = gdf.total_bounds
        return gdf

    def _generate_grid_chunks(self, n_splits=5):
        """
        Genera una cuadrÃ­cula de chunks que intersectan significativamente con el contorno
        
        Args:
            n_splits: NÃºmero de divisiones en cada eje (cuadrÃ­cula n_splits x n_splits)
            
        Returns:
            Tuple: (lista de chunks, rangos en X, rangos en Y)
        """
        if self.contour is None:
            self._load_contour()
            
        amazon_union = unary_union(self.contour.geometry)
        chunks = []
        x_range = np.linspace(self.bbox[0], self.bbox[2], n_splits + 1)
        y_range = np.linspace(self.bbox[1], self.bbox[3], n_splits + 1)

        for i in range(n_splits):
            for j in range(n_splits):
                chunk_box = box(x_range[i], y_range[j], x_range[i+1], y_range[j+1])
                intersection = chunk_box.intersection(amazon_union)
                
                # Filtrar chunks con intersecciÃ³n significativa (>1% del Ã¡rea)
                if not intersection.is_empty and (intersection.area / chunk_box.area) > 0.01:
                    chunks.append({
                        'id': f"{i}_{j}",
                        'x_index': i,
                        'y_index': j,
                        'bounds': {
                            'minx': x_range[i], 'miny': y_range[j],
                            'maxx': x_range[i+1], 'maxy': y_range[j+1]
                        },
                        'geometry': mapping(chunk_box),
                        'intersection_area': intersection.area,
                        'intersection_ratio': intersection.area / chunk_box.area
                    })
        
        print(f"ğŸ”² Generada cuadrÃ­cula de {n_splits}x{n_splits} con {len(chunks)} chunks vÃ¡lidos")
        return chunks, x_range, y_range

    def export_grid_to_geojson(self, n_splits=5, filename="amazon_grid.geojson"):
        """
        Exporta la cuadrÃ­cula a formato GeoJSON
        
        Args:
            n_splits: NÃºmero de divisiones en cada eje
            filename: Nombre del archivo de salida
            
        Returns:
            Path: Ruta al archivo generado
        """
        chunks, _, _ = self._generate_grid_chunks(n_splits)
        
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "id": chunk['id'],
                    "x_index": chunk['x_index'],
                    "y_index": chunk['y_index'],
                    "intersection_ratio": round(chunk['intersection_ratio'], 4)
                },
                "geometry": chunk['geometry']
            } for chunk in chunks]
        }
        
        output_path = self.data_dir / filename
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
            
        print(f"ğŸ’¾ CuadrÃ­cula guardada en: {output_path}")
        return output_path

    def _add_grid_to_map(self, map_obj, x_range, y_range):
        """AÃ±ade lÃ­neas de cuadrÃ­cula al mapa"""
        for x in x_range:
            folium.PolyLine(
                locations=[[self.bbox[1], x], [self.bbox[3], x]],
                color='#555555',
                weight=1,
                dash_array='5,2',
                opacity=0.7
            ).add_to(map_obj)
            
        for y in y_range:
            folium.PolyLine(
                locations=[[y, self.bbox[0]], [y, self.bbox[2]]],
                color='#555555',
                weight=1,
                dash_array='5,2',
                opacity=0.7
            ).add_to(map_obj)

    def generate_grid_map(self, n_splits=5, filename="amazon_grid.html"):
        """
        Genera mapa interactivo con la cuadrÃ­cula
        
        Args:
            n_splits: NÃºmero de divisiones en cada eje
            filename: Nombre del archivo de salida
            
        Returns:
            Path: Ruta al mapa generado
        """
        chunks, x_range, y_range = self._generate_grid_chunks(n_splits)
        center = [(self.bbox[1]+self.bbox[3])/2, (self.bbox[0]+self.bbox[2])/2]
        
        m = folium.Map(
            location=center,
            zoom_start=5,
            tiles="CartoDB positron",
            control_scale=True
        )

        # Capa del contorno original
        folium.GeoJson(
            self.contour,
            name='LÃ­mite del Amazonas',
            style_function=lambda x: {
                'color': '#0066FF',
                'weight': 3,
                'fillOpacity': 0.1
            },
            tooltip="Contorno RAISG"
        ).add_to(m)

        # Capa de chunks
        for chunk in chunks:
            folium.GeoJson(
                chunk['geometry'],
                style_function=lambda x: {
                    'fillColor': '#00AA00',
                    'color': '#005500',
                    'weight': 1.5,
                    'fillOpacity': 0.15
                },
                tooltip=f"Chunk {chunk['id']}<br>IntersecciÃ³n: {chunk['intersection_ratio']:.1%}"
            ).add_to(m)

        # AÃ±adir cuadrÃ­cula
        self._add_grid_to_map(m, x_range, y_range)
        folium.LayerControl(position="topright").add_to(m)
        
        # Guardar mapa
        output_path = self.maps_dir / filename
        m.save(output_path)
        
        print(f"ğŸ—ºï¸� Mapa de cuadrÃ­cula guardado en: {output_path}")
        return output_path

    def process_all(self, n_splits=5):
        """
        Ejecuta todo el pipeline de procesamiento
        
        Args:
            n_splits: NÃºmero de divisiones en cada eje
            
        Returns:
            dict: Rutas de los archivos generados
        """
        print("\n=== DIVISIÃ“N DEL AMAZONAS EN CHUNKS ===")
        self._load_contour()
        
        grid_path = self.export_grid_to_geojson(n_splits)
        map_path = self.generate_grid_map(n_splits)
        
        print("\nâœ… Proceso completado:")
        print(f"ğŸ“Š Datos: {grid_path}")
        print(f"ğŸ—ºï¸� Mapa: {map_path}")
        
        return {
            'grid_data': grid_path,
            'grid_map': map_path
        }


if __name__ == "__main__":
    try:
        processor = AmazonGridDivider(
            "amazon_data/contorno_del_amazonas_(lÃ­mite_raisg).parquet"
        )
        results = processor.process_all(n_splits=8)
    except Exception as e:
        print(f"â�Œ Error en el procesamiento: {str(e)}")
        raise


display(Image(filename='/kaggle/input/amaztest/my_imgs/amz_grid.png'))


%%time
"""IdentificaciÃ³n de Ã¡reas crÃ­ticas con alto porcentaje de territorio desconocido"""

import folium
import geopandas as gpd
import json
from shapely.geometry import shape
from shapely.ops import unary_union
from pathlib import Path

class CriticalAreaAnalyzer:
    def __init__(self, contour_path, untagged_path):
        """
        Inicializa el analizador con estructura de directorios organizada
        
        Args:
            contour_path: Ruta al archivo Parquet del contorno RAISG
            untagged_path: Ruta al archivo Parquet de territorios no etiquetados
        """
        self.contour_path = contour_path
        self.untagged_path = untagged_path
        
        # Configurar estructura de directorios
        self.data_dir = Path("amazon_data")
        self.maps_dir = Path("amazon_maps")
        self.data_dir.mkdir(exist_ok=True)
        self.maps_dir.mkdir(exist_ok=True)
        
        # Cargar datos
        self.contour = self._load_geodata(contour_path)
        self.untagged = self._load_geodata(untagged_path)
        self.bbox = self.contour.total_bounds
        
        print("âœ… Datos cargados exitosamente")
        print(f"Contorno CRS: {self.contour.crs}")
        print(f"Ã�rea no etiquetada CRS: {self.untagged.crs}")

    def _load_geodata(self, path):
        """Carga y valida datos geoespaciales"""
        gdf = gpd.read_parquet(path)
        
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
            
        return gdf.to_crs("EPSG:4326")

    def _identify_critical_areas(self, chunks_json_path, min_percentage=0.8):
        """
        Identifica Ã¡reas crÃ­ticas con alto porcentaje de territorio desconocido
        
        Args:
            chunks_json_path: Ruta al archivo JSON con la cuadrÃ­cula
            min_percentage: Porcentaje mÃ­nimo para considerar Ã¡rea crÃ­tica (0-1)
            
        Returns:
            Tuple: (GeoJSON para exportar, Datos para visualizaciÃ³n)
        """
        with open(chunks_json_path) as f:
            chunks_data = json.load(f)
            
        untagged_union = unary_union(self.untagged.geometry)
        critical_features = []
        visualization_data = []
        
        for feature in chunks_data['features']:
            chunk_geom = shape(feature['geometry'])
            intersection = chunk_geom.intersection(untagged_union)
            
            if not intersection.is_empty:
                percentage = intersection.area / chunk_geom.area
                
                if percentage >= min_percentage:
                    # Datos para GeoJSON
                    critical_features.append({
                        "type": "Feature",
                        "properties": {
                            "id": feature['properties']['id'],
                            "x_index": feature['properties']['x_index'],
                            "y_index": feature['properties']['y_index'],
                            "percentage_unknown": round(percentage * 100, 2),
                            "area_km2": round(chunk_geom.area * 111.32**2, 2)  # ConversiÃ³n aproximada a kmÂ²
                        },
                        "geometry": feature['geometry']
                    })
                    
                    # Datos para visualizaciÃ³n
                    visualization_data.append({
                        'id': feature['properties']['id'],
                        'geometry': feature['geometry'],
                        'percentage': round(percentage * 100, 2),
                        'area_km2': round(chunk_geom.area * 111.32**2, 2)
                    })
        
        return {
            "type": "FeatureCollection",
            "name": "critical_areas",
            "features": critical_features
        }, visualization_data

    def export_critical_areas(self, chunks_json_path, filename="critical_areas.geojson"):
        """
        Exporta Ã¡reas crÃ­ticas a GeoJSON
        
        Args:
            chunks_json_path: Ruta al archivo JSON con la cuadrÃ­cula
            filename: Nombre del archivo de salida
            
        Returns:
            Path: Ruta al archivo generado
        """
        geojson, _ = self._identify_critical_areas(chunks_json_path)
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
            
        print(f"ğŸ’¾ Ã�reas crÃ­ticas guardadas en: {output_path}")
        print(f"Total de Ã¡reas identificadas: {len(geojson['features'])}")
        return output_path

    def _create_legend(self):
        """Crea leyenda profesional para el mapa"""
        return '''
        <div style="
            position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 250px;
            height: 110px;
            border: 2px solid #1f4e79;
            border-radius: 5px;
            z-index: 9999;
            font-size: 14px;
            font-family: Arial;
            background: white;
            opacity: 0.85;
            padding: 10px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
        ">
            <div style="font-weight: bold; color: #1f4e79; margin-bottom: 8px; font-size: 15px;">
                Ã�reas CrÃ­ticas (>80% desconocido)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #006d32; height: 12px; width: 12px; margin-right: 8px;"></div>
                <span>LÃ­mite del Amazonas</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #3a7ca5; height: 12px; width: 12px; margin-right: 8px;"></div>
                <span>Ã�rea crÃ­tica</span>
            </div>
            <div style="color: #555; font-size: 12px; margin-top: 5px;">
                Click para mÃ¡s detalles
            </div>
        </div>
        '''

    def generate_critical_map(self, chunks_json_path, filename="critical_areas_map.html"):
        """
        Genera mapa interactivo de Ã¡reas crÃ­ticas
        
        Args:
            chunks_json_path: Ruta al archivo JSON con la cuadrÃ­cula
            filename: Nombre del archivo de salida
            
        Returns:
            Path: Ruta al mapa generado
        """
        _, critical_areas = self._identify_critical_areas(chunks_json_path)
        
        if not critical_areas:
            print("âš ï¸� No se encontraron Ã¡reas con >80% de territorio desconocido")
            return None
            
        # Centrar mapa en la primera Ã¡rea crÃ­tica
        first_area = critical_areas[0]
        centroid = shape(first_area['geometry']).centroid
        center = [centroid.y, centroid.x]
        
        m = folium.Map(
            location=center,
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True
        )

        # Capa del contorno del Amazonas
        folium.GeoJson(
            self.contour,
            name='LÃ­mite del Amazonas',
            style_function=lambda x: {
                'color': '#006d32',
                'weight': 2.5,
                'fillOpacity': 0
            },
            tooltip="Territorio AmazÃ³nico"
        ).add_to(m)

        # Capa de Ã¡reas crÃ­ticas
        for area in critical_areas:
            folium.GeoJson(
                area['geometry'],
                style_function=lambda x: {
                    'fillColor': '#3a7ca5',
                    'color': '#1f4e79',
                    'weight': 1.5,
                    'fillOpacity': 0.35
                },
                tooltip=f"Ã�rea {area['id']}<br>"
                       f"Desconocido: {area['percentage']}%<br>"
                       f"Ã�rea: {area['area_km2']} kmÂ²"
            ).add_to(m)

        # AÃ±adir leyenda
        m.get_root().html.add_child(folium.Element(self._create_legend()))
        folium.LayerControl(position="topright").add_to(m)
        
        # Guardar mapa
        output_path = self.maps_dir / filename
        m.save(output_path)
        
        print(f"ğŸ—ºï¸� Mapa de Ã¡reas crÃ­ticas guardado en: {output_path}")
        return output_path

    def process_all(self, chunks_json_path):
        """
        Ejecuta todo el pipeline de anÃ¡lisis
        
        Args:
            chunks_json_path: Ruta al archivo JSON con la cuadrÃ­cula
            
        Returns:
            dict: Rutas de los archivos generados
        """
        print("\n=== ANÃ�LISIS DE Ã�REAS CRÃ�TICAS ===")
        
        data_path = self.export_critical_areas(chunks_json_path)
        map_path = self.generate_critical_map(chunks_json_path)
        
        print("\nâœ… Proceso completado:")
        print(f"ğŸ“Š Datos: {data_path}")
        print(f"ğŸ—ºï¸� Mapa: {map_path}")
        
        return {
            'critical_areas': data_path,
            'critical_map': map_path
        }


if __name__ == "__main__":
    try:
        analyzer = CriticalAreaAnalyzer(
            contour_path="amazon_data/contorno_del_amazonas_(lÃ­mite_raisg).parquet",
            untagged_path="/kaggle/working/amazon_data/unlabeled_territories.parquet"
        )
        
        results = analyzer.process_all(
            chunks_json_path="amazon_data/amazon_grid.geojson"
        )
    except Exception as e:
        print(f"â�Œ Error en el procesamiento: {str(e)}")
        raise


from IPython.display import display
from ipywidgets import HBox, Image as widgets_Image  # Renombramos para evitar conflicto

# Rutas de las imÃ¡genes
img1 = open('/kaggle/input/amaztest/my_imgs/65p.png', 'rb').read()  # Leer como bytes
img2 = open('/kaggle/input/amaztest/my_imgs/75p.png', 'rb').read()
img3 = open('/kaggle/input/amaztest/my_imgs/80p.png', 'rb').read()

# Crear widgets de imÃ¡genes
widget_img1 = widgets_Image(value=img1, format='png')
widget_img2 = widgets_Image(value=img2, format='png')
widget_img3 = widgets_Image(value=img3, format='png')

# Mostrar en HBox
display(HBox([widget_img1, widget_img2, widget_img3]))



display(Image(filename='/kaggle/input/amaztest/my_imgs/65p.png'))
display(Image(filename='/kaggle/input/amaztest/my_imgs/75p.png'))
display(Image(filename='/kaggle/input/amaztest/my_imgs/80p.png'))


%%time
"""BÃºsqueda y visualizaciÃ³n de datos STAC para Ã¡reas de interÃ©s en el Amazonas"""

import json
import geopandas as gpd
from pystac_client import Client
from tqdm import tqdm
import itertools
import folium
from pathlib import Path

class STACDataExplorer:
    def __init__(self, chunks_json_path):
        """
        Inicializa el explorador con estructura de directorios organizada
        
        Args:
            chunks_json_path: Ruta al archivo JSON con los chunks pre-filtrados
        """
        # Configurar estructura de directorios
        self.data_dir = Path("amazon_data")
        self.maps_dir = Path("amazon_maps")
        self.data_dir.mkdir(exist_ok=True)
        self.maps_dir.mkdir(exist_ok=True)
        
        # Cargar chunks y conectar a catÃ¡logo STAC
        self.chunks = self._load_and_prepare_chunks(chunks_json_path)
        self.catalog = Client.open("https://landsatlook.usgs.gov/stac-server/")
        
        print(f"âœ… {len(self.chunks)} Ã¡reas de interÃ©s listas para bÃºsqueda STAC")

    def _load_and_prepare_chunks(self, path):
        """
        Carga y prepara los chunks desde el JSON con validaciÃ³n
        
        Args:
            path: Ruta al archivo JSON con los chunks
            
        Returns:
            GeoDataFrame con los chunks preparados
        """
        with open(path) as f:
            data = json.load(f)
        
        # Generar IDs consistentes si no existen
        for idx, feat in enumerate(data['features']):
            if 'id' not in feat['properties']:
                x_idx = feat['properties'].get('x_index', idx)
                y_idx = feat['properties'].get('y_index', idx)
                feat['properties']['id'] = f"area_{x_idx}_{y_idx}"
        
        gdf = gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")
        
        # Validar columnas requeridas
        required_columns = ['id', 'x_index', 'y_index']
        for col in required_columns:
            if col not in gdf.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
        
        return gdf

    def fetch_stac_items(self, collection="landsat-c2l2-sr", max_items=2, 
                        datetime_range="2022-01-01/2022-12-31", cloud_cover=20):
        """
        Busca Ã­tems STAC para cada Ã¡rea de interÃ©s
        
        Args:
            collection: ColecciÃ³n STAC a buscar
            max_items: MÃ¡ximo de Ã­tems por Ã¡rea
            datetime_range: Rango de fechas (formato ISO)
            cloud_cover: Porcentaje mÃ¡ximo de nubes permitido
            
        Returns:
            Lista de Ã­tems STAC encontrados
        """
        all_items = []
        pbar = tqdm(self.chunks.iterrows(), total=len(self.chunks), 
                   desc="ğŸ”� Buscando datos STAC")
        
        for idx, chunk in pbar:
            chunk_id = chunk['id']
            bbox = chunk.geometry.bounds
            
            try:
                search = self.catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=datetime_range,
                    query={"eo:cloud_cover": {"lt": cloud_cover}},
                    limit=max_items
                )
                items = list(itertools.islice(search.items(), max_items))
                all_items.extend(items)
                pbar.set_postfix({"Ã�rea": chunk_id, "Ã�tems": len(items)})
            except Exception as e:
                pbar.set_postfix({"Ã�rea": chunk_id, "Error": str(e)[:30] + "..."})
                continue
        
        print(f"\nğŸ“Š Total de Ã­tems encontrados: {len(all_items)}")
        return all_items

    def _create_stac_map(self, items):
        """
        Crea mapa interactivo con Ã¡reas de interÃ©s e Ã­tems STAC
        
        Args:
            items: Lista de Ã­tems STAC a visualizar
            
        Returns:
            Mapa Folium configurado
        """
        center = self.chunks.unary_union.centroid
        m = folium.Map(
            location=[center.y, center.x],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True
        )
        
        # Capa de Ã¡reas de interÃ©s
        folium.GeoJson(
            self.chunks,
            name="Ã�reas de interÃ©s",
            style_function=lambda x: {
                'fillColor': '#2ECC71',
                'color': '#27AE60',
                'weight': 1.5,
                'fillOpacity': 0.2
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['id', 'x_index', 'y_index'],
                aliases=['ID:', 'Fila:', 'Columna:'],
                localize=True
            )
        ).add_to(m)
        
        # Capa de Ã­tems STAC
        for item in items:
            try:
                props = item.properties
                tooltip = f"""
                <div style="width: 250px">
                    <b>ID:</b> {item.id}<br>
                    <b>Fecha:</b> {item.datetime.date() if item.datetime else 'N/A'}<br>
                    <b>Nubes:</b> {props.get('eo:cloud_cover', 'N/A')}%<br>
                    <b>SatÃ©lite:</b> {props.get('platform', 'N/A')}
                </div>
                """
                
                folium.GeoJson(
                    item.geometry,
                    style_function=lambda x: {
                        'color': '#E74C3C',
                        'weight': 2.5,
                        'fillOpacity': 0.1
                    },
                    tooltip=tooltip  # Eliminamos el popup para evitar el error
                ).add_to(m)
            except Exception as e:
                print(f"âš ï¸� Error visualizando Ã­tem {getattr(item, 'id', 'unknown')}: {str(e)}")
                continue
        
        # Control de capas y leyenda
        folium.LayerControl(position="topright").add_to(m)
        return m

    def visualize_stac_results(self, items, filename="stac_results.html"):
        """
        Visualiza y guarda los resultados STAC en un mapa interactivo
        
        Args:
            items: Lista de Ã­tems STAC a visualizar
            filename: Nombre del archivo de salida
            
        Returns:
            Path: Ruta al mapa generado
        """
        if not items:
            raise ValueError("No hay Ã­tems STAC para visualizar")
        
        m = self._create_stac_map(items)
        output_path = self.maps_dir / filename
        m.save(output_path)
        
        print(f"ğŸ—ºï¸� Mapa de resultados STAC guardado en: {output_path}")
        return output_path

    def process_all(self, collection="landsat-c2l2-sr", max_items=2):
        """
        Ejecuta todo el pipeline de bÃºsqueda y visualizaciÃ³n STAC
        
        Args:
            collection: ColecciÃ³n STAC a buscar
            max_items: MÃ¡ximo de Ã­tems por Ã¡rea
            
        Returns:
            dict: Resultados del procesamiento
        """
        print("\n=== BÃšSQUEDA DE DATOS STAC ===")
        
        # Buscar datos STAC
        items = self.fetch_stac_items(collection=collection, max_items=max_items)
        
        # Visualizar resultados
        map_path = self.visualize_stac_results(items)
        
        print("\nâœ… Proceso completado")
        print(f"ğŸ“Š Ã�tems encontrados: {len(items)}")
        print(f"ğŸ—ºï¸� Mapa generado: {map_path}")
        
        return {
            'stac_items': items,
            'stac_map': map_path
        }


if __name__ == "__main__":
    try:
        explorer = STACDataExplorer(
            chunks_json_path="amazon_data/amazon_grid.geojson"
        )
        
        results = explorer.process_all(
            collection="landsat-c2l2-sr",
            max_items=2
        )
    except Exception as e:
        print(f"â�Œ Error en el procesamiento: {str(e)}")
        raise


%%time
"""Mapeo de Ã¡reas crÃ­ticas xtrasmall con datos STAC"""


import json
import geopandas as gpd
from pystac_client import Client
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import folium
from shapely.geometry import shape
from pathlib import Path


class STACDataFetcher:
    def __init__(self, chunks_json_path):
        """Inicializa el fetcher con los chunks pre-filtrados del JSON"""
        self.chunks = self._load_chunks(chunks_json_path)
        print(f"âœ… {len(self.chunks)} chunks cargados listos para bÃºsqueda")

    def _load_chunks(self, path):
        """Carga los chunks desde el JSON (ya pre-filtrados)"""
        with open(path) as f:
            data = json.load(f)

        # Generar IDs consistentes basados en x_index y y_index si no existen
        for feat in data['features']:
            if 'id' not in feat:
                x_idx = feat['properties']['x_index']
                y_idx = feat['properties']['y_index']
                feat['id'] = f"chunk_{x_idx}_{y_idx}"

        gdf = gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")

        return gdf

    def fetch_all_stac_items(self, stac_endpoint, collection, datetime_range, cloud_limit=20):
        """
        Busca TODOS los Ã­tems STAC por chunk en paralelo
        
        Args:
            stac_endpoint (str): URL del servidor STAC
            collection (str): ColecciÃ³n STAC a consultar
            datetime_range (str): Rango de fechas (ej: "2022-01-01/2022-12-31")
            cloud_limit (int): Porcentaje mÃ¡ximo de nubes permitido
        """
        self.catalog = Client.open(stac_endpoint)

        def process_chunk(chunk):
            try:
                bbox = chunk.geometry.bounds
                search = self.catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=datetime_range,
                    query={"eo:cloud_cover": {"lt": cloud_limit}}
                )
                items = list(search.items())
                return [(chunk.id, item) for item in items]
            except Exception as e:
                return []

        all_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_chunk, chunk) for _, chunk in self.chunks.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="ğŸ”� Buscando todos los Ã­tems STAC"):
                all_results.extend(future.result())

        stac_items = [item for (_, item) in all_results]
        print(f"\nğŸ“Š Total de Ã­tems encontrados: {len(stac_items)}")
        return stac_items

    def visualize_stac_data(self, items, output_html="stac_resultados.html"):
        """Visualiza los Ã­tems STAC en un mapa interactivo"""

        if not items:
            raise ValueError("No hay Ã­tems para visualizar")

        maps_dir = Path("amazon_maps")
        maps_dir.mkdir(exist_ok=True)

        center = self.chunks.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=6, tiles="CartoDB positron")

        # Capa de chunks
        folium.GeoJson(
            self.chunks,
            name="Chunks",
            style_function=lambda x: {
                'fillColor': '#00FF00',
                'color': '#005500',
                'weight': 0.5,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(fields=['id'])
        ).add_to(m)

        # Capa de Ã­tems STAC
        for item in items:
            try:
                props = item.properties
                tooltip = f"<b>ID:</b> {item.id}<br><b>Nubes:</b> {props.get('eo:cloud_cover', 'N/A')}%"
                simplified_geom = shape(item.geometry).simplify(0.01)

                folium.GeoJson(
                    simplified_geom,
                    style_function=lambda x: {'color': '#E74C3C', 'weight': 1.5},
                    tooltip=tooltip
                ).add_to(m)
            except Exception as e:
                continue

        folium.LayerControl().add_to(m)

        output_path = maps_dir / output_html
        m.save(output_path)
        print(f"ğŸŒ� Mapa guardado en: {output_path}")
        return m


# === EjecuciÃ³n parametrizada ===
if __name__ == "__main__":
    # ConfiguraciÃ³n flexible desde aquÃ­
    config = {
        "chunks_json_path": "/kaggle/working/amazon_data/critical_areas.geojson",
        "stac_endpoint": "https://landsatlook.usgs.gov/stac-server/", 
        "collection": "landsat-c2l2-sr",
        "datetime_range": "2022-01-01/2022-12-31",
        "cloud_limit": 20,
        "map_filename": "xs_stac_resultados.html"
    }

    try:
        # Inicializar
        fetcher = STACDataFetcher(config["chunks_json_path"])

        # Descargar datos STAC usando los parÃ¡metros externos
        items = fetcher.fetch_all_stac_items(
            stac_endpoint=config["stac_endpoint"],
            collection=config["collection"],
            datetime_range=config["datetime_range"],
            cloud_limit=config["cloud_limit"]
        )

        # Visualizar resultados
        fetcher.visualize_stac_data(items, output_html=config["map_filename"])

    except Exception as e:
        print(f"â�Œ Error: {e}")


!pip install pystac_client pystac laspy open3d planetary-computer laszip laspy[lazrs] lazr copclib -q


%%time
from pystac_client import Client
from pathlib import Path
import requests
import planetary_computer
import time

class LidarDownloader:
    def __init__(self, output_dir="lidar_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_laz_file(self):
        print("ğŸ”� Buscando datos LiDAR...")
        try:
            client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = client.search(
                collections=["3dep-lidar-copc"],
                bbox=[-74.05, 40.68, -73.90, 40.79],
                query={"pc:type": {"eq": "lidar"}},
                limit=1
            )
            item = next(search.get_items(), None)
            if not item:
                print("âš ï¸� No se encontraron items con datos LiDAR")
                return None
        except Exception as e:
            print(f"â�Œ Error en bÃºsqueda STAC: {e}")
            return None

        laz_file = self.output_dir / f"{item.id}.laz"
        try:
            asset = item.assets["data"]
            signed_url = planetary_computer.sign(asset.href)
            print(f"â¬‡ï¸� Descargando {Path(asset.href).name}...")

            start_time = time.time()
            with requests.get(signed_url, stream=True) as r:
                r.raise_for_status()
                with open(laz_file, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

            print(f"âœ… Descarga completada en {time.time()-start_time:.1f}s - {laz_file.stat().st_size/1e6:.1f}MB")
            return laz_file
        except Exception as e:
            print(f"â�Œ Error en descarga: {e}")
            return None

if __name__ == "__main__":
    downloader = LidarDownloader()
    downloader.download_laz_file()



%%time
from pathlib import Path
import copclib as copc
import pandas as pd
import numpy as np
import open3d as o3d

class LidarConverter:
    def __init__(self, copc_file: Path, output_dir="lidar_data"):
        self.copc_file = Path(copc_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.item_id = self.copc_file.stem

    def convert(self):
        print("\nğŸ”„ Procesando archivo COPC...")
        try:
            reader = copc.FileReader(str(self.copc_file))
            nodes = reader.GetAllNodes()
            if not nodes:
                print("âš ï¸� No se detectaron nodos")
                return

            sample = reader.GetPoints(nodes[0])
            extras = [attr for attr in dir(sample)
                      if not attr.startswith('_') and attr.islower() and attr not in ('x','y','z')]
            print("Dimensiones adicionales detectadas:", extras)

            data = {'x': [], 'y': [], 'z': []}
            for a in extras:
                data[a] = []

            has_rgb = False
            total = 0
            for node in nodes:
                pts = reader.GetPoints(node)
                count = len(pts.x)
                if count == 0:
                    continue # Omitir nodos vacÃ­os
                total += count

                data['x'].append(pts.x)
                data['y'].append(pts.y)
                data['z'].append(pts.z)

                for a in extras:
                    try:
                        arr = getattr(pts, a)
                        # --- INICIO DE LA MODIFICACIÃ“N ---
                        # Si 'arr' es un solo valor (escalar), conviÃ©rtelo en un array
                        # con ese valor repetido para cada punto del nodo.
                        if np.isscalar(arr):
                            arr = np.full(count, arr)
                        # --- FIN DE LA MODIFICACIÃ“N ---
                        data[a].append(arr)
                        if a in ('red','green','blue'):
                            has_rgb = True
                    except Exception:
                        data[a].append(np.full(count, np.nan))

            print(f"  - Puntos leÃ­dos: {total:,}")

            for k in data:
                data[k] = np.concatenate(data[k])

            df = pd.DataFrame({k.upper(): v for k, v in data.items()})

            parquet_file = self.output_dir / f"{self.item_id}.parquet"
            df.to_parquet(parquet_file, compression='zstd')
            print(f"âœ… Parquet generado: {parquet_file.stat().st_size/1e6:.1f}â€¯MB")

            ply_file = self.output_dir / f"{self.item_id}.ply"
            pcd = o3d.geometry.PointCloud()
            xyz = np.vstack((data['x'], data['y'], data['z'])).T
            pcd.points = o3d.utility.Vector3dVector(xyz)

            if has_rgb and 'red' in data and 'green' in data and 'blue' in data:
                # Usar nan_to_num para evitar errores con valores nulos
                r = np.nan_to_num(data['red']) / 65535.0
                g = np.nan_to_num(data['green']) / 65535.0
                b = np.nan_to_num(data['blue']) / 65535.0
                colors = np.vstack((r, g, b)).T
                pcd.colors = o3d.utility.Vector3dVector(colors)
                print("âœ… Colores incluidos en el PLY (con algunos valores por defecto)")
            else:
                print("âš ï¸� No se encontraron colores vÃ¡lidos â€” PLY guardado sin color")

            o3d.io.write_point_cloud(str(ply_file), pcd)
            print(f"âœ… PLY generado: {ply_file.stat().st_size/1e6:.1f}â€¯MB")

        except Exception as e:
            print(f"â�Œ Error en procesamiento: {e}")

if __name__ == "__main__":
    file = Path("/kaggle/working/lidar_data/18TWL850150.laz")
    converter = LidarConverter(file)
    converter.convert()



%%time
"""Mapping data per region in microsoft"""
import pystac_client
import pystac


class LidarExplorer:
    """
    Una clase para explorar el catÃ¡logo STAC de Microsoft Planetary Computer,
    buscar colecciones LiDAR y verificar si contienen datos dentro de una regiÃ³n especÃ­fica.
    """

    def __init__(self, catalog_url: str):
        self.catalog_url = catalog_url
        self.client = None
        print(f"Inicializando explorador para el catÃ¡logo: {self.catalog_url}")
        try:
            self.client = pystac_client.Client.open(self.catalog_url)
            print("âœ… ConexiÃ³n al catÃ¡logo establecida con Ã©xito.")
        except Exception as e:
            print(f"â�Œ No se pudo conectar al catÃ¡logo: {e}")

    def find_lidar_collections(self, keywords=None):
        """Busca colecciones que coincidan con palabras clave relacionadas con LiDAR."""
        if not self.client:
            return []

        if keywords is None:
            keywords = ['lidar', 'copc', 'pointcloud', '3dep']

        print(f"\nğŸ”� Buscando colecciones con las palabras clave: {keywords}...")
        lidar_collections = []
        for collection in self.client.get_collections():
            text_to_match = f"{collection.id} {collection.title}".lower()
            if any(kw in text_to_match for kw in keywords):
                lidar_collections.append(collection)
                print(f"   - ColecciÃ³n encontrada: '{collection.id}' | TÃ­tulo: {collection.title}")
        return lidar_collections

    def has_data_in_bbox(self, collection_id: str, bbox: list) -> bool:
        """
        Verifica si una colecciÃ³n tiene al menos un item dentro del bounding box dado.
        """
        print(f"\nğŸ“¡ Verificando datos en la colecciÃ³n '{collection_id}' para el Ã¡rea definida...")
        try:
            search = self.client.search(
                collections=[collection_id],
                bbox=bbox,
                max_items=1
            )
            item = next(search.items(), None)
            if item:
                print(f"   âœ… Se encontrÃ³ al menos un item. ID: {item.id}")
                print(f"      BBOX del item: {item.bbox}")
                return True
            else:
                print("   â�Œ No se encontraron items en esta colecciÃ³n para el Ã¡rea especificada.")
                return False
        except Exception as e:
            print(f"âš ï¸� Error al buscar datos en '{collection_id}': {e}")
            return False

    def generate_report(self, results: dict):
        """Genera un reporte tabular con los resultados."""
        print("\nğŸ“Š" + "-" * 45)
        print("     Reporte Final de Disponibilidad de Datos")
        print("-" * 45)
        for collection, status in results.items():
            print(f"- {collection.ljust(20)} | Â¿Datos disponibles?: {status}")
        print("-" * 45)


if __name__ == "__main__":
    # ConfiguraciÃ³n inicial
    CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1" 
    BBOX = [-79.617211, -20.535150, -43.399318, 10.059151]  # [minx, miny, maxx, maxy]

    # Iniciar explorador
    explorer = LidarExplorer(catalog_url=CATALOG_URL)

    # Resultados finales
    availability_results = {}

    if explorer.client:
        lidar_collections = explorer.find_lidar_collections()

        if lidar_collections:
            print("\nğŸ”� Revisando cada colecciÃ³n para ver si contiene datos en el Ã¡rea definida...")

            for collection in lidar_collections:
                has_data = explorer.has_data_in_bbox(collection.id, BBOX)
                availability_results[collection.id] = "âœ… SÃ­" if has_data else "â�Œ No"

            # Mostrar reporte final
            explorer.generate_report(availability_results)
        else:
            print("â�Œ No se encontraron colecciones LiDAR en el catÃ¡logo.")


%%time
import requests
import json
import re

class LidarSoilDataFinder:
    def __init__(self):
        # URLs y parÃ¡metros de bÃºsqueda
        self.url = "https://cmr.earthdata.nasa.gov:443/search/collections.json" 
        self.params = {
            "keyword": "lidar",
            "bounding_box": "-79.617211,-20.535150,-43.399318,10.059151"
        }

        # Palabras clave mejoradas
        self.LIDAR_STRONG = {'gedi', 'icesat', 'pointcloud', 'als', 'dtm', 'dem'}
        self.LIDAR_MEDIUM = {'lidar', 'canopy', 'height', 'pulse', 'waveform', 'atl'}

        self.SOIL_STRONG = {'bare earth', 'terrain', 'topography', 'dem', 'dtm', 'ground surface'}
        self.SOIL_MEDIUM = {'elevation', 'surface', 'ground', 'geoid', 'bathymetry'}

        self.NEGATIVE_KEYWORDS = {'aerosol', 'atmosphere', 'cloud', 'plankton', 'ocean color', 'ionosphere'}

    def fetch_collections(self):
        """Realiza la solicitud HTTP y guarda la respuesta como JSON."""
        print("ğŸ”� Buscando colecciones LIDAR...")
        response = requests.get(self.url, params=self.params)

        if response.status_code == 200:
            try:
                data = response.json()
                with open("respuesta_cmr.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
                print("âœ… Datos guardados en 'respuesta_cmr.json'")
            except json.JSONDecodeError as e:
                print("â�Œ Error al parsear JSON:", e)
                print("ğŸ“„ Respuesta cruda:")
                print(response.text)
        else:
            print(f"â�Œ Error en la solicitud. CÃ³digo HTTP: {response.status_code}")
            print("ğŸ“„ Respuesta cruda:")
            print(response.text)

    def evaluate_entry(self, entry: dict) -> dict | None:
        """EvalÃºa si una entrada es relevante para LIDAR y estudio del suelo."""
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        full_text = f"{title} {summary}".lower()

        # Filtrado negativo
        for keyword in self.NEGATIVE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', full_text):
                return None

        # Puntaje LIDAR
        lidar_score = 0
        if any(re.search(r'\b' + re.escape(kw) + r'\b', full_text) for kw in self.LIDAR_STRONG):
            lidar_score += 2
        if any(re.search(r'\b' + re.escape(kw) + r'\b', full_text) for kw in self.LIDAR_MEDIUM):
            lidar_score += 1

        # Puntaje Suelo
        soil_score = 0
        if any(re.search(r'\b' + re.escape(kw) + r'\b', full_text) for kw in self.SOIL_STRONG):
            soil_score += 2
        if any(re.search(r'\b' + re.escape(kw) + r'\b', full_text) for kw in self.SOIL_MEDIUM):
            soil_score += 1

        # CondiciÃ³n final
        if lidar_score > 0 and soil_score > 0:
            return {
                "title": title,
                "summary": summary,
                "id": entry.get("id"),
                "lidar_score": lidar_score,
                "soil_score": soil_score
            }
        return None

    def filter_collections(self):
        """Carga el JSON y filtra las colecciones relevantes."""
        try:
            with open("respuesta_cmr.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("â�Œ El archivo 'respuesta_cmr.json' no se encontrÃ³.")
            return []

        entries = data.get("feed", {}).get("entry", [])
        filtered = []
        for entry in entries:
            result = self.evaluate_entry(entry)
            if result:
                filtered.append(result)

        # Ordenar por puntaje total descendente
        filtered.sort(key=lambda x: (x['lidar_score'] + x['soil_score']), reverse=True)
        return filtered

    def save_results(self, results):
        """Guarda los resultados filtrados en un archivo JSON."""
        output_file = "colecciones_lidar_suelo_filtrado_inteligente.json"
        clean_results = [{k: v for k, v in item.items() if k not in ['lidar_score', 'soil_score']} for item in results]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"\nğŸ“„ Resultados guardados en: {output_file}")

    def display_results(self, results):
        """Muestra los resultados por consola."""
        if results:
            print(f"âœ… Se encontraron {len(results)} colecciones LIDAR relacionadas con el suelo:")
            for item in results:
                print(f"- [LIDAR: {item['lidar_score']}, SUELO: {item['soil_score']}] {item['title']}")
        else:
            print("â�Œ No se encontraron colecciones relevantes despuÃ©s del filtrado.")



if __name__ == "__main__":
    finder = LidarSoilDataFinder()

    # Paso 1: Obtener datos de la API
    finder.fetch_collections()

    # Paso 2: Filtrar colecciones relevantes
    results = finder.filter_collections()

    # Paso 3: Mostrar y guardar resultados
    finder.display_results(results)
    finder.save_results(results)


%%time
"""QUERY 1: Se consulta mediante la region definida y analisis en bruto"""
import os
import re
import json
import requests
from tqdm import tqdm
from urllib.parse import urlparse

# === ConfiguraciÃ³n de tÃ©rminos ===
OUTPUT_FOLDER = "/kaggle/working/amazon_data"
DOWNLOAD_FOLDER = "/kaggle/working/usgob"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

MAP_FORMATS = {'tif', 'shp', 'las', 'laz', 'gpkg', 'geopackage', 'geojson'}
ZIP_FORMATS = {'zip', 'tar.gz'}
CSV_FORMATS = {'csv'}
LOW_PRIORITY_FORMATS = {'xml', 'html'}


class CKANLidarHarvester:
    API_URL = "https://catalog.data.gov/api/3/action"

    def __init__(self, bbox: str, max_results=7000, truncate=False):
        self.bbox = bbox
        self.max_results = max_results
        self.truncate = truncate
        self.brute_file = os.path.join(OUTPUT_FOLDER, "usgob_selected_brute.json")
        self.blessed_file = os.path.join(OUTPUT_FOLDER, "usgob_selected_blessed.json")
        self.final_file = os.path.join(OUTPUT_FOLDER, "usgob_selected_batch.json")

    def stage_1_scrape_all(self):
        collected = []
        rows = 100
        for start in tqdm(range(0, self.max_results, rows), desc="ğŸ“¥ Etapa 1: Recolectando datasets"):
            payload = {
                "q": "*:*",
                "rows": rows,
                "start": start,
                "extras": {"ext_bbox": self.bbox}
            }
            try:
                resp = requests.post(f"{self.API_URL}/package_search", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"â�Œ Error en request: {e}")
                break

            if not data.get("success"):
                print("â�Œ Fallo en respuesta CKAN.")
                break

            for item in data["result"]["results"]:
                resources = item.get("resources", [])
                formats = list({r.get("format", "").lower() for r in resources if r.get("format")})
                collected.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "title": item.get("title"),
                    "notes": item.get("notes", ""),
                    "num_resources": len(resources),
                    "formats": formats
                })

            if self.truncate and len(collected) >= 100:
                break

        with open(self.brute_file, "w") as f:
            json.dump(collected, f, indent=2)
        print(f"âœ… Guardado {len(collected)} datasets en {self.brute_file}")

    def _analyze_lidar(self, notes: str) -> dict:
        text = notes.lower()
        has_lidar = "lidar" in text
        has_dem = "dem" in text
        return {
            "has_lidar": has_lidar,
            "has_dem": has_dem,
            "is_lidar": has_lidar and has_dem
        }

    def stage_2_filter_lidar(self):
        with open(self.brute_file, "r") as f:
            raw = json.load(f)

        blessed = []
        for entry in tqdm(raw, desc="ğŸ”� Etapa 2: Filtrando LIDAR"):
            analysis = self._analyze_lidar(entry["notes"])
            if analysis["is_lidar"]:
                blessed.append({
                    "id": entry["id"],
                    "title": entry["title"],
                    "url": f"https://catalog.data.gov/dataset/{entry['name']}",
                    "notes": entry["notes"],
                    "diagnostic": analysis
                })

        with open(self.blessed_file, "w") as f:
            json.dump(blessed, f, indent=2)
        print(f"âœ… Guardado {len(blessed)} datasets LIDAR en {self.blessed_file}")

    def _prioritize_format(self, fmt: str) -> int:
        fmt = fmt.lower()
        if fmt in MAP_FORMATS:
            return 0
        if fmt in ZIP_FORMATS:
            return 1
        if fmt in CSV_FORMATS:
            return 2
        if fmt in LOW_PRIORITY_FORMATS:
            return 3
        return 9

    def _download_to_folder(self, url: str, folder: str, name: str) -> str:
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                os.makedirs(folder, exist_ok=True)
                filename = re.sub(r'[^\w\-_.]', '_', name.lower())[:80]
                ext = os.path.splitext(urlparse(url).path)[1]
                if not ext:
                    ext = '.bin'
                path = os.path.join(folder, f"{filename}{ext}")
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return path
        except Exception as e:
            print(f"â�Œ Error al descargar {url}: {e}")
        return None

    def stage_3_fetch_resources(self):
        with open(self.blessed_file, "r") as f:
            blessed = json.load(f)
    
        output = []
        for ds in tqdm(blessed, desc="ğŸ“¦ Etapa 3: Descargando TODOS los recursos"):
            try:
                resp = requests.get(f"{self.API_URL}/package_show?id={ds['id']}", timeout=20)
                data = resp.json()
                if not data.get("success"):
                    continue
                resources = data["result"].get("resources", [])
                if not resources:
                    continue
    
                dataset_slug = re.sub(r'\W+', '_', ds["title"].lower())[:60]
                folder = os.path.join(DOWNLOAD_FOLDER, dataset_slug)
    
                downloaded = []
                for res in resources:
                    fmt = res.get("format", "")
                    url = res.get("url", "")
                    name = res.get("name", "resource")
                    if not url or not url.startswith("http"):
                        continue
                    try:
                        head = requests.head(url, timeout=10, allow_redirects=True)
                        if head.status_code == 200:
                            local_path = self._download_to_folder(url, folder, name)
                            if local_path:
                                downloaded.append({
                                    "name": name,
                                    "format": fmt,
                                    "url": url,
                                    "size": res.get("size", None),
                                    "local_path": local_path
                                })
                    except Exception as e:
                        print(f"âš ï¸� HEAD fallido: {e}")
                        continue
    
                if downloaded:
                    output.append({
                        "dataset_title": ds["title"],
                        "dataset_url": ds["url"],
                        "resources": downloaded
                    })
    
                if self.truncate and len(output) >= 10:
                    break
    
            except Exception as e:
                print(f"âš ï¸� Fallo en dataset {ds['id']}: {e}")
    
        with open(self.final_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"âœ… Guardado {len(output)} datasets con recursos en {self.final_file}")
        



# === Punto de entrada ===
if __name__ == "__main__":
    bbox = "-79.617211,-20.535150,-43.399318,10.059151"  # RegiÃ³n geogrÃ¡fica
    truncate = False  # Cambiar a False para full run

    pipeline = CKANLidarHarvester(bbox=bbox, truncate=truncate)
    pipeline.stage_1_scrape_all()
    pipeline.stage_2_filter_lidar()
    pipeline.stage_3_fetch_resources()

    print("\nğŸ�� Flujo completo finalizado.")



%%time
"""QUERY 2: Filtra por la palabra amazon """
import os
import re
import json
import requests
from tqdm import tqdm
from urllib.parse import urlparse, unquote

# ========== CONFIGURACIÃ“N ==========
class Config:
    OUTPUT_FOLDER = "/kaggle/working/amazon_data"
    DOWNLOAD_FOLDER = "/kaggle/working/amzusgob"
    API_URL = "https://catalog.data.gov/api/3/action"

    MAP_FORMATS = {'tif', 'shp', 'las', 'laz', 'gpkg', 'geojson'}
    ZIP_FORMATS = {'zip', 'tar.gz'}
    CSV_FORMATS = {'csv'}
    LOW_PRIORITY_FORMATS = {'xml', 'html'}

    @staticmethod
    def setup():
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.DOWNLOAD_FOLDER, exist_ok=True)

    @staticmethod
    def prioritize_format(fmt: str) -> int:
        fmt = fmt.lower()
        if fmt in Config.MAP_FORMATS:
            return 0
        if fmt in Config.ZIP_FORMATS:
            return 1
        if fmt in Config.CSV_FORMATS:
            return 2
        if fmt in Config.LOW_PRIORITY_FORMATS:
            return 3
        return 9

    @staticmethod
    def download_to_folder(url: str, folder: str, name: str) -> str:
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                os.makedirs(folder, exist_ok=True)
                filename = re.sub(r'[^\w\-_.]', '_', name.lower())[:80]
                ext = os.path.splitext(urlparse(url).path)[1]
                if not ext:
                    ct = response.headers.get("Content-Type", "")
                    if "las" in ct: ext = ".las"
                    elif "laz" in ct: ext = ".laz"
                    elif "zip" in ct: ext = ".zip"
                    else: ext = ".bin"
                path = os.path.join(folder, f"{filename}{ext}")
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return path
        except Exception as e:
            print(f"â�Œ Error al descargar {url}: {e}")
        return None

# ========== HARVESTER AMAZON ==========
class CKANAmazonHarvester:
    def __init__(self, max_results=3000, truncate=False):
        self.max_results = max_results
        self.truncate = truncate
        self.brute_file = os.path.join(Config.OUTPUT_FOLDER, "amazon_brute.json")
        self.blessed_file = os.path.join(Config.OUTPUT_FOLDER, "amazon_blessed.json")
        self.final_file = os.path.join(Config.OUTPUT_FOLDER, "amazon_batch.json")

    def stage_1_scrape_all(self):
        collected = []
        rows = 100
        for start in tqdm(range(0, self.max_results, rows), desc="ğŸ“¥ Etapa 1: Buscando 'amazon'"):
            payload = {"q": "amazon", "rows": rows, "start": start}
            try:
                resp = requests.post(f"{Config.API_URL}/package_search", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"â�Œ Error en request: {e}")
                break
            if not data.get("success"):
                break
            for item in data["result"]["results"]:
                resources = item.get("resources", [])
                formats = list({r.get("format", "").lower() for r in resources if r.get("format")})
                collected.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "title": item.get("title"),
                    "notes": item.get("notes", ""),
                    "num_resources": len(resources),
                    "formats": formats
                })
            if self.truncate and len(collected) >= 100:
                break
        with open(self.brute_file, "w") as f:
            json.dump(collected, f, indent=2)
        print(f"âœ… Guardado {len(collected)} datasets en {self.brute_file}")

    def stage_2_filter_amazon(self):
        with open(self.brute_file, "r") as f:
            raw = json.load(f)
        blessed = []
        for entry in tqdm(raw, desc="ğŸ”� Etapa 2: Filtrando 'amazon' en descripciÃ³n"):
            if "amazon" in entry["notes"].lower():
                blessed.append({
                    "id": entry["id"],
                    "title": entry["title"],
                    "url": f"https://catalog.data.gov/dataset/{entry['name']}",
                    "notes": entry["notes"]
                })
        with open(self.blessed_file, "w") as f:
            json.dump(blessed, f, indent=2)
        print(f"âœ… Guardado {len(blessed)} datasets Amazon en {self.blessed_file}")

    def stage_3_fetch_resources(self):
        with open(self.blessed_file, "r") as f:
            blessed = json.load(f)
        output = []
        for ds in tqdm(blessed, desc="ğŸ“¦ Etapa 3: Descargando recursos"):
            try:
                resp = requests.get(f"{Config.API_URL}/package_show?id={ds['id']}", timeout=20)
                data = resp.json()
                if not data.get("success"):
                    continue
                resources = data["result"].get("resources", [])
                resources = sorted(
                    [r for r in resources if r.get("url")],
                    key=lambda r: Config.prioritize_format(r.get("format", ""))
                )
                folder = os.path.join(Config.DOWNLOAD_FOLDER, re.sub(r'\W+', '_', ds["title"].lower())[:60])
                downloaded = []
                for res in resources:
                    url = unquote(res.get("url", ""))
                    if not url.startswith("http"):
                        continue
                    try:
                        head = requests.head(url, timeout=10, allow_redirects=True)
                        if head.status_code == 200:
                            local_path = Config.download_to_folder(url, folder, res.get("name", "resource"))
                            if local_path:
                                downloaded.append({
                                    "name": res.get("name", ""),
                                    "format": res.get("format", ""),
                                    "url": url,
                                    "size": res.get("size", None),
                                    "local_path": local_path
                                })
                    except Exception as e:
                        print(f"âš ï¸� HEAD fallido: {e}")
                if downloaded:
                    output.append({
                        "dataset_title": ds["title"],
                        "dataset_url": ds["url"],
                        "resources": downloaded
                    })
                if self.truncate and len(output) >= 10:
                    break
            except Exception as e:
                print(f"âš ï¸� Fallo en dataset {ds['id']}: {e}")
        with open(self.final_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"âœ… Guardado {len(output)} datasets con recursos en {self.final_file}")

# ========== EJECUCIÃ“N ==========
if __name__ == "__main__":
    Config.setup()
    print("ğŸš€ Iniciando pipeline AMAZON...\n")

    amazon = CKANAmazonHarvester(truncate=True) #Poner false para traer toda la cola
    amazon.stage_1_scrape_all()
    amazon.stage_2_filter_amazon()
    amazon.stage_3_fetch_resources()

    print("\nâœ… Pipeline AMAZON finalizado.")



%%time
"""Resultados de la query 2: ver descripciones sospechosas"""
import requests
import json
from tqdm import tqdm

class AmazonLidarInspector:
    API_URL = "https://catalog.data.gov/api/3/action"

    # Diccionario de datasets LIDAR/DEM
    LIDAR_DATASETS = {
        "lidar_surveys_brazil": "lidar-surveys-over-selected-forest-research-sites-brazilian-amazon-2008-2018-38601",
        "amazon_forest_structure_lidar": "amazon-forest-structure-from-airborne-lidar-ed2-initial-condition-files-2016",
        "lidar_grid_tiles": "lidar-grid-tiles",
        "srtm_90m_northern_ecuador": "lba-eco-lc-01-srtm-90-meter-digital-elevation-model-northern-ecuadorian-amazon-49ccb",
        "selective_logging_brazil": "lba-eco-lc-21-selective-logging-activity-in-the-brazilian-amazon-1999-2002",
        "gedi_tandemx_biomass": "pantropical-forest-height-and-biomass-from-gedi-and-tandem-x-data-fusion-30c83",
        "srtm30_amazon_basin": "lba-eco-lc-15-srtm30-digital-elevation-model-data-amazon-basin-2000-13076",
        "sar_landcover_biomass": "lba-eco-lc-03-sar-images-land-cover-and-biomass-four-areas-across-brazilian-amazon-72c32"
    }

    def __init__(self):
        self.output = []

    def fetch_all_batches(self):
        for key, dataset_name in tqdm(self.LIDAR_DATASETS.items(), desc="ğŸ”� Consultando datasets LIDAR"):
            try:
                resp = requests.get(f"{self.API_URL}/package_show?id={dataset_name}", timeout=20)
                data = resp.json()
                if not data.get("success"):
                    print(f"â�Œ Fallo API para {dataset_name}")
                    continue

                result = data["result"]
                resources = result.get("resources", [])
                self.output.append({
                    "dataset_id": dataset_name,
                    "dataset_title": result.get("title"),
                    "dataset_url": f"https://catalog.data.gov/dataset/{dataset_name}",
                    "resources": [
                        {
                            "name": r.get("name"),
                            "format": r.get("format"),
                            "url": r.get("url"),
                            "size": r.get("size"),
                            "created": r.get("created"),
                            "mimetype": r.get("mimetype")
                        }
                        for r in resources if r.get("url")
                    ]
                })

            except Exception as e:
                print(f"âš ï¸� Error al consultar {dataset_name}: {e}")

    def save_to_file(self, path="/kaggle/working/amazon_data/amazon_lidar_batches_observer.json"):
        with open(path, "w") as f:
            json.dump(self.output, f, indent=2)
        print(f"âœ… Datos guardados en {path}")

if __name__ == "__main__":
    print("ğŸ”� Inspeccionando datasets LIDAR del Amazonas...")
    inspector = AmazonLidarInspector()
    inspector.fetch_all_batches()
    inspector.save_to_file()



%%time
"""De la query 2: Aqui se bajan los metadatos de los datasets sospechosos """
import os
import re
import json
import requests
from urllib.parse import unquote, urlparse
from tqdm import tqdm

class AmazonLidarDownloader:
    def __init__(self, json_path="/kaggle/working/amazon_data/amazon_lidar_batches_observer.json"):
        self.json_path = json_path
        self.download_dir = "/kaggle/working/amzgob_data"
        os.makedirs(self.download_dir, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[^\w\-_.]', '_', name.lower())[:80]

    def _infer_extension(self, url: str, content_type: str) -> str:
        ext = os.path.splitext(urlparse(url).path)[1]
        if ext:
            return ext
        if "las" in content_type: return ".las"
        if "laz" in content_type: return ".laz"
        if "zip" in content_type: return ".zip"
        if "pdf" in content_type: return ".pdf"
        return ".bin"

    def _download(self, url: str, out_folder: str, name_hint: str) -> str:
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                filename = self._sanitize_filename(name_hint)
                ext = self._infer_extension(url, response.headers.get("Content-Type", ""))
                full_path = os.path.join(out_folder, f"{filename}{ext}")
                with open(full_path, "wb") as f:
                    for chunk in response.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                return full_path
        except Exception as e:
            print(f"â�Œ Error al descargar: {url} â†’ {e}")
        return None

    def download_all(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        for dataset in tqdm(data, desc="ğŸ“¦ Descargando recursos por dataset"):
            folder_name = self._sanitize_filename(dataset["dataset_title"])
            folder_path = os.path.join(self.download_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            for res in dataset.get("resources", []):
                raw_url = res.get("url", "")
                name = res.get("name", "resource")
                decoded_url = unquote(raw_url)
                if not decoded_url.startswith("http"):
                    continue

                try:
                    head = requests.head(decoded_url, timeout=10, allow_redirects=True)
                    if head.status_code != 200:
                        continue
                except:
                    continue

                self._download(decoded_url, folder_path, name)

        print(f"\nâœ… Todos los datasets se han descargado en: {self.download_dir}")

if __name__ == "__main__":
    print("ğŸšš Descargando recursos LIDAR del Amazonas...")
    downloader = AmazonLidarDownloader()
    downloader.download_all()



!pip install earthengine-api google-auth-oauthlib -q


%%time
"""Este script mapea el area ortogonal en bruto"""
import os
import ee
import time
import requests
from itertools import product
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient
from google.oauth2 import service_account

class GeeDownloader:
    """
    A definitive, fully optimized class to download GEE data. It uses pre-determined 
    minimum scales for each dataset to ensure a clean, single-attempt download.
    """
    CATALOG = [
        {'name': 'Copernicus_DEM_GLO30', 'id': 'COPERNICUS/DEM/GLO30', 'type': 'IMAGE_COLLECTION', 'min_scale': 240},
        {'name': 'ALOS_DEM_AW3D30', 'id': 'JAXA/ALOS/AW3D30/V4_1', 'type': 'IMAGE_COLLECTION', 'min_scale': 120},
        {'name': 'GEDI_Monthly', 'id': 'LARSE/GEDI/GEDI02_A_002_MONTHLY', 'type': 'IMAGE_COLLECTION', 'min_scale': 1920},
        {'name': 'Global_SRTM_Topographic_Diversity', 'id': 'CSP/ERGo/1_0/Global/SRTM_topoDiversity', 'type': 'IMAGE', 'min_scale': 120},
    ]

    def __init__(self, download_dir="/kaggle/working/gee_downloads"):
        print("--- STEP 1: Authenticating with GEE ---")
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        try:
            # (Authentication code remains the same)
            user_secrets = UserSecretsClient()
            gee_mail = user_secrets.get_secret("gee_mail")
            gee_pkey = user_secrets.get_secret("gee_pkey").replace('\\n', '\n')
            project_id = user_secrets.get_secret("gee_project_id")
            scopes = ['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
            info = {
                "type": "service_account", "project_id": project_id,
                "private_key": gee_pkey, "client_email": gee_mail,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
            ee.Initialize(credentials=creds, project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
            print("âœ… Authentication successful.")
        except Exception as e:
            raise RuntimeError(f"Error initializing GEE: {e}")

    def check_collections_for_data(self, region_of_interest: dict) -> dict:
        geometry = ee.Geometry.Rectangle(region_of_interest['bbox'], proj='EPSG:4326', geodesic=False)
        print("\n--- STEP 2: Checking data availability ---")
        report = {}
        for item in self.CATALOG:
            name, asset_id, typ = item['name'], item['id'], item['type']
            count = 0
            is_available = False
            try:
                if typ == 'IMAGE_COLLECTION':
                    coll = ee.ImageCollection(asset_id).filterBounds(geometry)
                    count = coll.size().getInfo()
                    if count > 0: is_available = True
                elif typ == 'IMAGE':
                    img = ee.Image(asset_id)
                    if img.geometry(1).intersects(geometry, 1).getInfo():
                        count = 1
                        is_available = True
            except Exception as e:
                print(f"   - Could not check {name}. Reason: {e}")
            report[name] = {'available': is_available, 'count': count}
        return report
    
    def download_available_data(self, bbox: dict, availability_report: dict, tile_size_deg: float = 1.0, test_mode: bool = False):
        print("\n--- STEP 3: Starting download process ---")
        if test_mode:
            print("âš ï¸�  SMART TEST MODE ENABLED: Will find and download only one data-containing tile per dataset.")

        lon_min, lat_min, lon_max, lat_max = bbox['bbox']
        all_tiles = list(product(self._frange(lon_min, lon_max, tile_size_deg), self._frange(lat_min, lat_max, tile_size_deg)))

        for entry in self.CATALOG:
            name, asset_id, typ = entry["name"], entry["id"], entry["type"]
            min_scale = entry["min_scale"]

            if not availability_report.get(name, {}).get('available'):
                continue

            print(f"\nğŸ“¦ Processing dataset: {name} (using optimal scale: {min_scale}m)")
            out_dir = os.path.join(self.download_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            
            if test_mode:
                self._run_test_mode_for_dataset(all_tiles, name, asset_id, typ, out_dir, tile_size_deg, min_scale)
            else:
                self._run_full_download_for_dataset(all_tiles, name, asset_id, typ, out_dir, lon_max, lat_max, tile_size_deg, min_scale)

    def _run_test_mode_for_dataset(self, all_tiles, name, asset_id, typ, out_dir, tile_size_deg, scale):
        """Finds and downloads a single test tile for a dataset using its optimal scale."""
        found_tile_info = None
        for x0, y0 in tqdm(all_tiles, desc=f"Searching for data in {name}"):
            geom = ee.Geometry.Rectangle([x0, y0, x0 + tile_size_deg, y0 + tile_size_deg], proj="EPSG:4326", geodesic=False)
            image = self._get_image_for_region(asset_id, typ, geom)
            if image:
                found_tile_info = {'image': image, 'geom': geom, 'x0': x0, 'y0': y0}
                break
        
        if found_tile_info:
            x0, y0 = found_tile_info['x0'], found_tile_info['y0']
            print(f"-> Data tile found near (lon: {x0:.2f}, lat: {y0:.2f}). Proceeding to download.")
            fname = f"{name}_test_tile"
            self._download_tile(found_tile_info['image'], found_tile_info['geom'], out_dir, fname, scale)
        else:
            print(f"-> No data-containing tiles found for {name} in the specified region.")

    def _run_full_download_for_dataset(self, all_tiles, name, asset_id, typ, out_dir, lon_max, lat_max, tile_size_deg, scale):
        """Downloads all tiles for a dataset using its optimal scale."""
        with tqdm(all_tiles, desc=f"Downloading {name}") as pbar:
            for x0, y0 in pbar:
                pbar.set_postfix_str("Preparing tile...")
                x1, y1 = min(x0 + tile_size_deg, lon_max), min(y0 + tile_size_deg, lat_max)
                geom = ee.Geometry.Rectangle([x0, y0, x1, y1], proj="EPSG:4326", geodesic=False)
                
                image = self._get_image_for_region(asset_id, typ, geom)
                
                if image:
                    fname = f"{name}_lon{x0:.4f}_lat{y0:.4f}".replace(".", "_").replace("-", "m")
                    success = self._download_tile(image, geom, out_dir, fname, scale)
                    status = "âœ… Success" if success else "â�Œ Failed"
                    pbar.set_postfix_str(f"{status} (scale: {scale}m)")
                    time.sleep(0.5)
                else:
                    pbar.set_postfix_str("No data in tile, skipping.")

    def _get_image_for_region(self, asset_id: str, typ: str, geom: ee.Geometry) -> ee.Image | None:
        try:
            if typ == "IMAGE_COLLECTION":
                coll = ee.ImageCollection(asset_id).filterBounds(geom)
                if coll.size().getInfo() > 0: return coll.first().clip(geom)
            elif typ == "IMAGE":
                img = ee.Image(asset_id)
                if img.geometry(1).intersects(geom, 1).getInfo(): return img.clip(geom)
        except ee.EEException:
            return None
        return None

    def _download_tile(self, image: ee.Image, region: ee.Geometry, folder: str, name: str, scale: int) -> bool:
        """
        Attempts to download a tile exactly once using the provided optimal scale.
        """
        try:
            url = image.getDownloadURL({
                "name": name,
                "scale": scale,
                "region": region.getInfo(), # Must be client-side dictionary
                "fileFormat": "GeoTIFF"
            })
            response = requests.get(url, stream=True, timeout=180)
            
            if response.status_code == 200:
                out_path = os.path.join(folder, f"{name}.zip")
                with open(out_path, "wb") as f:
                    f.write(response.content)
                print(f"   âœ… Success: Saved {name}.zip (scale: {scale}m)")
                return True
            else:
                # If the server gives an error even with the optimal scale
                print(f"   â�Œ Download Failed (HTTP {response.status_code}): {response.text[:120]}")
                return False
        except Exception as e:
            # For timeouts or other client-side issues
            print(f"   â�Œ Download Failed (Exception): {str(e)[:120]}")
            return False

    def _frange(self, start, stop, step):
        vals = []
        current = start
        while current < stop:
            vals.append(round(current, 6))
            current += step
        return vals

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Set to False to download the entire region using the optimized scales.
        run_in_test_mode = True 
        
        amazon_bbox = {"bbox": [-79.617211, -20.535150, -43.399318, 10.059151]}
        
        downloader = GeeDownloader(download_dir="/kaggle/working/gee_amz_data")
        availability = downloader.check_collections_for_data(amazon_bbox)
        
        print("\n--- AVAILABILITY REPORT ---")
        print(f"{'Dataset':<35} | {'Status':<12} | {'Items Found'}")
        print("-" * 65)
        for name, report in availability.items():
            status = "âœ… Available" if report['available'] else "â�Œ Unavailable"
            print(f"- {name:<35} | {status:<12} | {report['count']}")
        
        downloader.download_available_data(
            bbox=amazon_bbox,
            availability_report=availability,
            tile_size_deg=2.0,
            test_mode=run_in_test_mode 
        )
        print("\nğŸ�‰ Process completed.")

    except Exception as e:
        print(f"\nâ�Œ FATAL ERROR in main process: {e}")


%%time
"""Etl en bruto xs"""
import os
import ee
import time
import json
import requests
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor, as_completed

class GeeDownloaderXS:
    """
    A definitive class to download ALL intersecting data for each polygon in a GeoJSON file.
    It accurately pre-calculates the total number of images and downloads them in parallel
    to provide a fast and clear user experience.
    """
    CATALOG = [
        {'name': 'Copernicus_DEM_GLO30', 'id': 'COPERNICUS/DEM/GLO30', 'type': 'IMAGE_COLLECTION', 'min_scale': 480},
        {'name': 'ALOS_DEM_AW3D30', 'id': 'JAXA/ALOS/AW3D30/V4_1', 'type': 'IMAGE_COLLECTION', 'min_scale': 240},
        {'name': 'GEDI_Monthly', 'id': 'LARSE/GEDI/GEDI02_A_002_MONTHLY', 'type': 'IMAGE_COLLECTION', 'min_scale': 3840},
        {'name': 'Global_SRTM_Topographic_Diversity', 'id': 'CSP/ERGo/1_0/Global/SRTM_topoDiversity', 'type': 'IMAGE', 'min_scale': 240},
    ]

    def __init__(self, download_dir="/kaggle/working/gee_downloads_geojson", workers: int = 2):
        print("--- STEP 1: Authenticating with GEE ---")
        self.download_dir = download_dir
        self.workers = workers
        os.makedirs(self.download_dir, exist_ok=True)
        try:
            user_secrets = UserSecretsClient()
            gee_mail = user_secrets.get_secret("gee_mail")
            gee_pkey = user_secrets.get_secret("gee_pkey").replace('\\n', '\n')
            project_id = user_secrets.get_secret("gee_project_id")
            scopes = ['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
            info = { "type": "service_account", "project_id": project_id, "private_key": gee_pkey, "client_email": gee_mail, "token_uri": "https://oauth2.googleapis.com/token" }
            creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
            ee.Initialize(credentials=creds, project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
            print("âœ… Authentication successful.")
        except Exception as e:
            raise RuntimeError(f"Error initializing GEE: {e}")

    def download_from_geojson(self, geojson_path: str):
        """Processes a GeoJSON file by planning all jobs first, then executing them in parallel."""
        print(f"\n--- STEP 2: Loading features from {geojson_path} ---")
        try:
            with open(geojson_path, 'r') as f:
                all_features = json.load(f)['features']
            print(f"âœ… Found {len(all_features)} areas to process.")
        except Exception as e:
            raise IOError(f"Could not read or parse GeoJSON file: {e}")

        # Phase 1: Planning - Create a definitive list of all download jobs.
        print("\n--- STEP 3: Planning all download jobs (this may take a moment)... ---")
        all_jobs_by_dataset = self._plan_download_jobs(all_features)
        
        # Phase 2: Reporting - Display the accurate, planned job counts.
        print("\n--- ACCURATE AVAILABILITY REPORT (FROM PLANNED JOBS) ---")
        print(f"{'Dataset':<35} | {'Status':<12} | {'Items to Download'}")
        print("-" * 75)
        for name, jobs in all_jobs_by_dataset.items():
            status = "âœ… Available" if jobs else "â�Œ Not found"
            print(f"- {name:<35} | {status:<12} | {len(jobs)}")

        # Phase 3: Execution - Process the planned jobs with parallel workers.
        print("\n--- STEP 4: Starting parallel download process ---")
        self._execute_all_jobs(all_jobs_by_dataset)

    def _plan_download_jobs(self, features: list) -> dict:
        """Iterates through all features and datasets to create a definitive list of download tasks."""
        jobs_by_dataset = {entry['name']: [] for entry in self.CATALOG}
        for feature in tqdm(features, desc="Scanning all areas and datasets"):
            geometry = ee.Geometry.Polygon(feature['geometry']['coordinates'])
            feature_id = feature['properties'].get('id', 'unknown_id')
            for entry in self.CATALOG:
                d_name, d_id, d_typ, d_scale = entry['name'], entry['id'], entry['type'], entry['min_scale']
                try:
                    if d_typ == 'IMAGE_COLLECTION':
                        coll = ee.ImageCollection(d_id).filterBounds(geometry)
                        count = coll.size().getInfo()
                        if count > 0:
                            image_list = coll.toList(count)
                            for i in range(count):
                                image = ee.Image(image_list.get(i))
                                jobs_by_dataset[d_name].append({
                                    'image': image.clip(geometry), 'geometry': geometry,
                                    'base_name': f"{d_name}_{feature_id}_{i}", 'scale': d_scale })
                    elif d_typ == 'IMAGE':
                        image = ee.Image(d_id)
                        if image.geometry(1).intersects(geometry, 1).getInfo():
                            jobs_by_dataset[d_name].append({
                                'image': image.clip(geometry), 'geometry': geometry,
                                'base_name': f"{d_name}_{feature_id}", 'scale': d_scale })
                except Exception as e:
                    tqdm.write(f"Warning: Could not plan job for area {feature_id} in {d_name}. Reason: {e}")
        return jobs_by_dataset

    def _execute_all_jobs(self, all_jobs_by_dataset: dict):
        """Executes all planned jobs using a thread pool."""
        for dataset_name, jobs in all_jobs_by_dataset.items():
            if not jobs:
                print(f"\nğŸ“¦ No items to download for dataset: {dataset_name}. Skipping.")
                continue
            
            print(f"\nğŸ“¦ Processing dataset: {dataset_name}")
            out_dir = os.path.join(self.download_dir, dataset_name)
            os.makedirs(out_dir, exist_ok=True)
            
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                with tqdm(total=len(jobs), desc=f"Downloading {dataset_name}") as pbar:
                    futures = [executor.submit(self._execute_single_download, job, out_dir) for job in jobs]
                    for future in as_completed(futures):
                        pbar.update(1)

    def _execute_single_download(self, job: dict, folder: str, max_retries: int = 4):
        """Downloads a single item, with retries. This function is run by each worker."""
        image, region, name, scale = job['image'], job['geometry'], job['base_name'], job['scale']
        current_scale = scale
        for attempt in range(max_retries):
            try:
                url = image.getDownloadURL({"name": name, "scale": current_scale, "region": region.getInfo(), "fileFormat": "GeoTIFF"})
                response = requests.get(url, stream=True, timeout=300)
                if response.status_code == 200:
                    with open(os.path.join(folder, f"{name}.zip"), "wb") as f: f.write(response.content)
                    return
                if "total request size" not in response.text.lower():
                    tqdm.write(f"   â�Œ Download Failed for {name} (HTTP {response.status_code}): {response.text[:120]}")
                    return
            except Exception as e:
                if "request size" not in str(e).lower() and "computation" not in str(e).lower():
                    tqdm.write(f"   â�Œ Download Failed for {name} (Unrecoverable Exception): {str(e)[:120]}")
                    return
            time.sleep(2); current_scale *= 2
        tqdm.write(f"   â�Œ Final failure for {name} after all attempts.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        geojson_file_path = "/kaggle/working/amazon_data/critical_areas.geojson"
        
        downloader = GeeDownloaderXS(download_dir="/kaggle/working/gee_amz_xs", workers=2)
        
        # This single call will now perform the planning, reporting, and full parallel download.
        downloader.download_from_geojson(geojson_path=geojson_file_path)
        
        print("\nğŸ�‰ Process completed.")
    except Exception as e:
        print(f"\nâ�Œ FATAL ERROR in main process: {e}")


%%time
import os
import zipfile

def unzip_and_remove_zip_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.zip'):
                file_path = os.path.join(foldername, filename)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(foldername)
                print(f"Descomprimido: {file_path}")
                os.remove(file_path)
                print(f"Eliminado: {file_path}")

if __name__ == "__main__":
    root_directory = '/kaggle/working/gee_amz_xs'
    unzip_and_remove_zip_files(root_directory)



%%time
"""Archeo blog map"""
import os
import folium
import dask.dataframe as dd
from pathlib import Path

class GeoLayerVisualizer:
    def __init__(self, folder_path, output_map="locations_per_parquet.html"):
        self.folder_path = folder_path
        self.output_map = output_map
        self.map_dir = Path("amazon_maps")
        self.map_dir.mkdir(exist_ok=True)

    def _get_parquet_files(self):
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".parquet")
        ]

    def _load_unique_points(self, file_path):
        df = dd.read_parquet(file_path)
        required_cols = {"latitud", "longitud"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Faltan columnas en {file_path}")
        df["latitud"] = dd.to_numeric(df["latitud"], errors="coerce")
        df["longitud"] = dd.to_numeric(df["longitud"], errors="coerce")
        df = df.dropna(subset=["latitud", "longitud"])
        return df[["latitud", "longitud"]].drop_duplicates().compute()

    def create_interactive_layered_map(self):
        parquet_files = self._get_parquet_files()
        if not parquet_files:
            print("âš ï¸� No se encontraron archivos Parquet")
            return

        all_points = []
        layers = []

        for file in parquet_files:
            filename = os.path.basename(file)
            #print(f"ğŸ“Š Procesando archivo: {filename}")
            try:
                points_df = self._load_unique_points(file)
                if len(points_df) == 0:
                    print(f"âš ï¸� Sin datos en {filename}")
                    continue
                all_points.extend(points_df.to_dict("records"))

                layer = folium.FeatureGroup(name=f"{filename} ({len(points_df)} sites)", show=True)

                for _, row in points_df.iterrows():
                    folium.Rectangle(
                        bounds=[
                            [row["latitud"] - 0.005, row["longitud"] - 0.005],
                            [row["latitud"] + 0.005, row["longitud"] + 0.005],
                        ],
                        color="blue",
                        fill=True,
                        fill_opacity=0.6,
                        tooltip=f"{filename}<br>{row['latitud']:.5f}, {row['longitud']:.5f}",
                    ).add_to(layer)

                layer.add_to(folium.Map())  # dummy just to register
                layers.append(layer)

            except Exception as e:
                print(f"â�Œ Error procesando {filename}: {e}")

        if not all_points:
            print("âš ï¸� No se pudieron cargar puntos vÃ¡lidos.")
            return

        # Obtener el bounding box general
        lats = [p["latitud"] for p in all_points]
        lons = [p["longitud"] for p in all_points]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

        # Crear mapa centrado automÃ¡ticamente
        m = folium.Map(tiles="CartoDB positron")
        m.fit_bounds(bounds)

        title_html = """<h3 align="center" style="font-family:Arial; font-size:20px">
        <b>Archeoblog registered sites</b></h3>"""
        m.get_root().html.add_child(folium.Element(title_html))

        # Agregar capas
        for layer in layers:
            layer.add_to(m)

        # Agregar control de capas con scroll si hay muchas capas
        folium.LayerControl(collapsed=False).add_to(m)

        # Guardar mapa
        map_path = self.map_dir / self.output_map
        m.save(map_path)
        print(f"\nâœ… Mapa guardado en: {map_path}")
        return map_path

# --- Bloque principal ---
if __name__ == "__main__":
    FOLDER_PATH = "/kaggle/working/parquet_jqjacobs"
    visualizer = GeoLayerVisualizer(FOLDER_PATH)
    visualizer.create_interactive_layered_map()



display(Image(filename='/kaggle/input/amaztest/my_imgs/amz_all_arch.png'))


%%time
"""Se guardan solo los ejes x,y para el etiquetado"""
import os
import json
import pandas as pd
import folium
from pathlib import Path
from shapely.geometry import Point, box
from folium.plugins import GroupedLayerControl
from pyproj import Transformer
import dask.dataframe as dd

# Paleta de colores para capas
COLORS = {
    "amazon_geoglyphs_sites": "#FF6B6B",
    "submit": "#4ECDC4",
    "casarabe_sites_utm": "#45B7D1",
    "mound_villages_acre": "#9C27B0",
    "science_data": "#FFA726",
    "parquet_data": "#666666"
}

# Mapeo de nombres de dataset a sus paths de origen
DATASET_PATHS = {
    "amazon_geoglyphs_sites": "/kaggle/input/amazon-geoglyphs-sites",
    "submit": "/kaggle/input/archaeological-survey-data",
    "casarabe_sites_utm": "/kaggle/input/casarabe-sites-utm",
    "mound_villages_acre": "/kaggle/input/mound-villages-acre",
    "science_data": "/kaggle/input/science-data",
    "parquet_data": "/kaggle/working/parquet_jqjacobs"
}

class FilterTags:
    def __init__(self, parquet_folder):
        self.parquet_folder = Path(parquet_folder)
        self.datasets = {}
        self.original_dfs = {}
        self.point_references = {}  # Nuevo: Almacena referencias a los registros originales

    def load_datasets(self):
        # Cargar amazon_geoglyphs_sites
        dt1 = pd.read_csv("/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv")
        self.original_dfs["amazon_geoglyphs_sites"] = dt1
        self.datasets["amazon_geoglyphs_sites"] = self._extract_points(dt1, "latitude", "longitude", "amazon_geoglyphs_sites")

        # Cargar submit
        dt2 = pd.read_csv("/kaggle/input/archaeological-survey-data/submit.csv")
        self.original_dfs["submit"] = dt2
        self.datasets["submit"] = self._extract_points(dt2, "y", "x", "submit")

        # Cargar casarabe_sites_utm
        dt3 = pd.read_csv("/kaggle/input/casarabe-sites-utm/casarabe_sites_utm.csv")
        self.original_dfs["casarabe_sites_utm"] = dt3
        self.datasets["casarabe_sites_utm"] = self._convert_utm_to_latlon(dt3, 20, 'S', "casarabe_sites_utm")

        # Cargar mound_villages_acre
        dt4 = pd.read_csv("/kaggle/input/mound-villages-acre/mound_villages_acre.csv")
        self.original_dfs["mound_villages_acre"] = dt4
        self.datasets["mound_villages_acre"] = self._convert_utm_to_latlon(
            dt4, 19, 'S', "mound_villages_acre", "UTM X (Easting)", "UTM Y (Northing)"
        )

        # Cargar science_data
        dt5 = pd.read_csv("/kaggle/input/science-data/science.ade2541_data_s2.csv")
        self.original_dfs["science_data"] = dt5
        self.datasets["science_data"] = self._extract_points(dt5, "Latitude", "Longitude", "science_data")

    def _extract_points(self, df, lat_col, lon_col, dataset_name=None):
        points = set()
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                point = (round(lon, 5), round(lat, 5))
                points.add(point)
                
                # Guardar referencia mÃ­nima al registro original
                if dataset_name:
                    self.point_references[point] = {
                        'dataset': dataset_name,
                        'index': idx,
                        'source_path': DATASET_PATHS.get(dataset_name, "unknown")
                    }
            except:
                continue
        return list(points)

    def _convert_utm_to_latlon(self, df, zone_number, zone_letter, name,
                             easting_col="UTM X (Easting)", northing_col="UTM Y (Northing)"):
        points = set()
        hemisphere = 'north' if zone_letter.upper() in ['N', 'P'] else 'south'
        crs_utm = f"+proj=utm +zone={zone_number} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

        for idx, row in df.iterrows():
            try:
                easting = float(row[easting_col])
                northing = float(row[northing_col])
                lon, lat = transformer.transform(easting, northing)
                point = (round(lon, 5), round(lat, 5))
                points.add(point)
                
                # Guardar referencia mÃ­nima al registro original
                self.point_references[point] = {
                    'dataset': name,
                    'index': idx,
                    'source_path': DATASET_PATHS.get(name, "unknown")
                }
            except:
                continue
        return list(points)

    def collect_parquet_points(self):
        parquet_files = list(self.parquet_folder.glob("*.parquet"))
        points = set()
        for file in parquet_files:
            try:
                df = dd.read_parquet(file).compute()
                if {"latitud", "longitud"}.issubset(df.columns):
                    df = df.dropna(subset=["latitud", "longitud"])
                    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
                    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
                    df = df.dropna(subset=["latitud", "longitud"])
                    for idx, row in df.iterrows():
                        point = (round(row["longitud"], 5), round(row["latitud"], 5))
                        points.add(point)
                        
                        # Guardar referencia mÃ­nima al registro parquet
                        self.point_references[point] = {
                            'dataset': "parquet_data",
                            'index': idx,
                            'source_path': str(file),
                            'file_stem': file.stem
                        }
            except Exception as e:
                print(f"â�Œ Error leyendo {file.name}: {e}")
        return points


class GeoJSONExporter:
    def __init__(self, output_path, grid_geojson_path):
        self.output_path = Path(output_path)
        self.grid_geojson_path = Path(grid_geojson_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.features = []
        self.region_bounds = self._load_grid_bounds()
        self.layer_counts = {}

    def _load_grid_bounds(self):
        """Carga el bounding box del grid de Amazon"""
        with open(self.grid_geojson_path) as f:
            grid_data = json.load(f)
        all_coords = []
        for feature in grid_data['features']:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                all_coords.extend(geom['coordinates'][0])
            elif geom['type'] == 'MultiPolygon':
                for polygon in geom['coordinates']:
                    all_coords.extend(polygon[0])
        lons, lats = zip(*all_coords)
        return box(min(lons), min(lats), max(lons), max(lats))

    def _point_in_region(self, point):
        """Verifica si un punto estÃ¡ dentro de la regiÃ³n del grid"""
        lon, lat = point
        return self.region_bounds.contains(Point(lon, lat))

    def add_points(self, dataset_name, points, filter_tags=None):
        """AÃ±ade puntos dentro del grid con referencias mÃ­nimas"""
        points_in_region = [pt for pt in points if self._point_in_region(pt)]
        self.layer_counts[dataset_name] = len(points_in_region)
        
        for idx, (lon, lat) in enumerate(points_in_region):
            # Obtener referencia al registro original
            reference = filter_tags.point_references.get((lon, lat), {}) if filter_tags else {}
            
            # Generar bbox alrededor del punto
            bbox = {
                "lon_min": round(lon - 0.009, 5),
                "lat_min": round(lat - 0.009, 5),
                "lon_max": round(lon + 0.009, 5),
                "lat_max": round(lat + 0.009, 5)
            }
            
            feature_properties = {
                "ffather": reference.get('source_path', DATASET_PATHS.get(dataset_name, "unknown")),
                "dataset": dataset_name,
                "id": f"{dataset_name}_{reference.get('index', idx)}",
                "bbox": bbox
            }
            
            # Para datos parquet, aÃ±adir referencia al archivo
            if dataset_name == "parquet_data":
                feature_properties["source_file"] = reference.get('file_stem', "")
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": feature_properties
            }
            self.features.append(feature)
        
        return len(points_in_region)

    def save(self):
        geojson = {"type": "FeatureCollection", "features": self.features}
        with open(self.output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"âœ… GeoJSON guardado con {len(self.features)} features en: {self.output_path}")
        return self.output_path


class TaggedSitesVisualizer:
    def __init__(self, geojson_path, grid_geojson_path, output_html):
        self.geojson_path = geojson_path
        self.grid_geojson_path = grid_geojson_path
        self.output_html = output_html
        self.map = None
        self.layer_groups = {}
    
    def create_map(self):
        # Cargar grid para centrar mapa
        with open(self.grid_geojson_path) as f:
            grid_data = json.load(f)
        coords = grid_data['features'][0]['geometry']['coordinates'][0]
        center_lat = sum(c[1] for c in coords)/len(coords)
        center_lon = sum(c[0] for c in coords)/len(coords)
        
        # Crear mapa centrado
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles="CartoDB positron"
        )
        
        # AÃ±adir grid como capa base semitransparente
        folium.GeoJson(
            self.grid_geojson_path,
            name="ğŸ“Œ Amazon Grid",
            style_function=lambda x: {
                'fillColor': '#a8dadc',
                'color': '#336633',
                'weight': 1,
                'fillOpacity': 0.2
            }
        ).add_to(self.map)
        
        # Procesar GeoJSON y crear capas
        with open(self.geojson_path) as f:
            sites_data = json.load(f)
        
        # Agrupar features por dataset
        features_by_dataset = {}
        for feature in sites_data['features']:
            dataset = feature['properties']['dataset']
            if dataset not in features_by_dataset:
                features_by_dataset[dataset] = []
            features_by_dataset[dataset].append(feature)
                
        
        # Diccionario para mantener las capas individuales
        self.layer_groups = {}
        
        for dataset, features in features_by_dataset.items():
            color = COLORS.get(dataset, "#666666")
            
            # Crear capa individual para cada dataset
            individual_layer = folium.FeatureGroup(name=f"ğŸ“‚ {dataset} ({len(features)})", show=True)
            
            for feature in features:
                props = feature['properties']
                point = feature['geometry']['coordinates']
                lat, lon = point[1], point[0]
                
                # Crear cuadrado para el punto
                offset = 0.005  # TamaÃ±o del cuadrado
                rectangle = folium.Rectangle(
                    bounds=[
                        [lat - offset, lon - offset],
                        [lat + offset, lon + offset]
                    ],
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{dataset}: {props.get('name', 'No name')}"
                )                                        
                rectangle.add_to(individual_layer)
            
            # Guardar capa individual
            self.layer_groups[dataset] = individual_layer
            individual_layer.add_to(self.map)
        
        
        
        
        # Control de capas estÃ¡ndar (no agrupado)
        folium.LayerControl(
            position='topright',
            collapsed=False
        ).add_to(self.map)
        
        self.map.save(self.output_html)
        print(f"âœ… Mapa guardado en: {self.output_html}")

if __name__ == "__main__":
    # ConfiguraciÃ³n de paths
    FOLDER_PATH = "/kaggle/working/parquet_jqjacobs"
    GRID_GEOJSON = "/kaggle/working/amazon_data/amazon_grid.geojson"
    OUTPUT_GEOJSON = "/kaggle/working/amazon_data/tagged_sites_filtered.geojson"
    OUTPUT_HTML = "/kaggle/working/amazon_maps/tagged_sites_map.html"

    # 1. Cargar y filtrar datos
    filter_tags = FilterTags(FOLDER_PATH)
    filter_tags.load_datasets()
    parquet_coords = filter_tags.collect_parquet_points()

    # 2. Exportar a GeoJSON (ya filtrado por grid)
    exporter = GeoJSONExporter(OUTPUT_GEOJSON, GRID_GEOJSON)
    
    # Procesar datasets externos
    for name, points in filter_tags.datasets.items():
        count = exporter.add_points(name, points, filter_tags)
        print(f"ğŸ“Š {name}: {count} sitios dentro del grid")
    
    # Procesar datos parquet
    parquet_points = [(round(x,5), round(y,5)) for x,y in parquet_coords]
    count = exporter.add_points("parquet_data", parquet_points, filter_tags)
    print(f"ğŸ“Š parquet_data: {count} sitios dentro del grid")
    
    # Guardar GeoJSON
    geojson_path = exporter.save()

    # 3. VisualizaciÃ³n
    print("\nğŸ—ºï¸� Generando mapa interactivo...")
    visualizer = TaggedSitesVisualizer(geojson_path, GRID_GEOJSON, OUTPUT_HTML)
    visualizer.create_map()

    print("\nâœ… Proceso completado:")
    print(f"- GeoJSON filtrado: {geojson_path}")
    print(f"- Mapa interactivo: {OUTPUT_HTML}")


display(Image(filename='/kaggle/input/amaztest/my_imgs/amz_arch_tagged.png'))


%%time
import folium
import geopandas as gpd
import json
from shapely.geometry import shape
from shapely.ops import unary_union
import branca.colormap as cm
import os  

class HeatmapVisualizer:
    def __init__(self, contour_path, untagged_path):
        """
        contour_path: Path to RAISG contour Parquet file
        untagged_path: Path to untagged territories Parquet file
        """
        self.contour = self._load_data(contour_path)
        self.untagged = self._load_data(untagged_path)
        self.bbox = self.contour.total_bounds
        self.colormap = self._create_colormap()
        print("âœ… Data loaded successfully with defined CRS")

    def _load_data(self, path):
        """Load geospatial data ensuring it has CRS"""
        gdf = gpd.read_parquet(path)
        
        # Ensure it has CRS defined (EPSG:4326 for lat/lon)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        
        # Convert to WGS84 if not in that CRS
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Validate geometries
        gdf = gdf[gdf.geometry.is_valid]
        return gdf

    def _create_colormap(self):
        """Create a custom color scale"""
        return cm.LinearColormap(
            colors=['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
            index=[0, 25, 50, 75, 100],
            vmin=0,
            vmax=100,
            caption='Percentage of unknown area'
        )

    def create_heatmap(self, chunks_json_path, output_dir="/kaggle/working/amazon_maps", output_html="heatmap_batches.html"):
        """Generate an interactive heatmap"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_html)
        
        # Load and validate chunks
        with open(chunks_json_path) as f:
            chunks_data = json.load(f)
        
        # Map center
        center = [
            (self.bbox[1] + self.bbox[3]) / 2,
            (self.bbox[0] + self.bbox[2]) / 2
        ]
        
        # Create base map with better parameters
        m = folium.Map(
            location=center,
            zoom_start=5,
            tiles='CartoDB positron',
            control_scale=True,
            prefer_canvas=True  # Better performance for many geometries
        )
        
        # Pre-process: union of untagged areas
        untagged_union = unary_union(self.untagged.geometry)
        
        # Add Amazon contour with improved style
        folium.GeoJson(
            self.contour,
            name='Amazon Boundary',
            style_function=lambda x: {
                'color': '#006d32',
                'weight': 2.5,
                'fillOpacity': 0,
                'dashArray': '5, 3'
            },
            zoom_on_click=False
        ).add_to(m)
        
        # Process each chunk for the heatmap
        for feature in chunks_data['features']:
            try:
                chunk_geom = shape(feature['geometry'])
                if not chunk_geom.is_valid:
                    continue
                
                # Calculate percentage of unknown area
                intersection = chunk_geom.intersection(untagged_union)
                if intersection.is_empty:
                    percentage = 0
                else:
                    percentage = (intersection.area / chunk_geom.area) * 100
                
                # Add to map with conditional style
                folium.GeoJson(
                    chunk_geom,
                    style_function=lambda x, p=percentage: {
                        'fillColor': self.colormap(p),
                        'color': '#333333',
                        'weight': 0.8,
                        'fillOpacity': 0.7,
                        'opacity': 0.9
                    },
                    tooltip=self._create_tooltip(feature, percentage)
                ).add_to(m)
                        
            except Exception as e:
                print(f"âš ï¸� Error processing chunk {feature.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Add control elements
        self._add_map_controls(m)
        m.save(output_path)
        print(f"ğŸ”¥ Heatmap saved to: {output_path}")
        return m

    def _create_tooltip(self, feature, percentage):
        """Create informative tooltip with improved HTML"""
        props = feature.get('properties', {})
        return f"""
        <div style="
            font-family: Arial; 
            width: 220px;
            padding: 5px;
        ">
            <div style="
                background: #1f4e79;
                color: white;
                padding: 5px;
                margin: -5px -5px 5px -5px;
                font-weight: bold;
            ">
                Batch {props.get('id', 'N/A')}
            </div>
            <div style="margin-bottom: 3px;">
                <span style="font-weight: bold;">Unknown:</span> 
                <span style="color: {self._get_percentage_color(percentage)};">
                    {percentage:.1f}%
                </span>
            </div>
            <div style="
                background: #f5f5f5;
                padding: 3px;
                font-size: 0.8em;
                margin-top: 5px;
            ">
                X: {props.get('x_index', '')}, Y: {props.get('y_index', '')}
            </div>
        </div>
        """

    def _get_percentage_color(self, percentage):
        """Returns color based on percentage"""
        if percentage < 25:
            return '#2c7bb6'
        elif percentage < 50:
            return '#67a9cf'
        elif percentage < 75:
            return '#fdae61'
        else:
            return '#d7191c'

    def _add_map_controls(self, map_obj):
        """Add controls and legend to the map"""
        # Add colormap to the map
        self.colormap.add_to(map_obj)
        
        # Improved professional legend
        legend_html = '''
        <div style="
            position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 300px;
            border: 2px solid #1f4e79;
            border-radius: 8px;
            z-index: 9999;
            font-family: Arial;
            background: rgba(255,255,255,0.93);
            padding: 10px 15px;
            box-shadow: 0 3px 14px rgba(0,0,0,0.2);
        ">
            <h4 style="
                margin: 0 0 10px 0;
                color: #1f4e79;
                font-size: 16px;
                text-align: center;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            ">
                Percentage of Unknown Area
            </h4>
            <div style="display: flex; margin-bottom: 5px;">
                <div style="flex-grow: 1; text-align: center; color: #2c7bb6;">
                    0%
                </div>
                <div style="flex-grow: 1; text-align: center; color: #d7191c;">
                    100%
                </div>
            </div>
            <div style="
                height: 20px;
                width: 100%;
                background: linear-gradient(to right, #2c7bb6, #abd9e9, #ffffbf, #fdae61, #d7191c);
                margin-bottom: 15px;
                border-radius: 4px;
                border: 1px solid #ddd;
            "></div>
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 0.85em;
                color: #555;
            ">
                <span>Well documented</span>
                <span>Poorly documented</span>
            </div>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

# Example usage
if __name__ == "__main__":
    try:
        # Path configuration
        contour_path = "/kaggle/working/amazon_data/contorno_del_amazonas_(lÃ­mite_raisg).parquet"
        untagged_path = "/kaggle/working/amazon_data/unlabeled_territories.parquet"
        chunks_json = "/kaggle/working/amazon_data/amazon_grid.geojson"
        
        print("ğŸ”¥ Starting heatmap generation...")
        
        # Create visualizer
        visualizer = HeatmapVisualizer(contour_path, untagged_path)
        
        # Generate map
        visualizer.create_heatmap(chunks_json)
        
        print("âœ… Process completed successfully")
        print(f"- Map generated: /kaggle/working/amazon_maps/heatmap_batches.html")
        
    except Exception as e:
        print(f"â�Œ Critical error: {str(e)}")


display(Image(filename='/kaggle/input/amaztest/my_imgs/amz_heatmap.png'))


%%time
"""Ver como viene los datos lidar_2412"""
import pandas as pd

# Leer un archivo .sss (sitio)
df_sss = pd.read_csv("/kaggle/input/amaztest/Amazon_ForestStructure_LIDAR_2412/Amazon_ForestStructure_LIDAR_2412/data/amzbr_0004.lat-12.5lon-62.5.sss", delim_whitespace=True)

# Leer un archivo .pss (parche)
df_pss = pd.read_csv("/kaggle/input/amaztest/Amazon_ForestStructure_LIDAR_2412/Amazon_ForestStructure_LIDAR_2412/data/amzbr_0004.lat-12.5lon-62.5.pss", delim_whitespace=True)

# Leer un archivo .css (cohort)
df_css = pd.read_csv("/kaggle/input/amaztest/Amazon_ForestStructure_LIDAR_2412/Amazon_ForestStructure_LIDAR_2412/data/amzbr_0004.lat-12.5lon-62.5.css", delim_whitespace=True)

print(f" sss file >>> {df_sss.head(3)}",f" pss file >>> {df_pss.head(3)}", f" css file >>> {df_css.head(3)}")

print(f" sss data types >>> {df_sss.dtypes}",f" pss data types >>> {df_pss.dtypes}", f" css data types >>> {df_css.dtypes}")

print(f" sss shape >>> {df_sss.shape}",f" pss shape >>> {df_pss.shape}", f" css shape >>> {df_css.shape}")


%%time
"""Mapear coordenadas para dataset lidar_2412"""
import os
import re
import pandas as pd

class EDForestStructureParser:
    def __init__(self, data_folder, output_csv):
        self.data_folder = data_folder
        self.output_csv = output_csv
        self.pattern = re.compile(
            r"amzbr_(\d{4})\.lat(-?\d+(?:\.\d+)?)lon(-?\d+(?:\.\d+)?)\.(css|pss|sss)$"
        )

    def run(self):
        print(f"ğŸ”� Scanning directory: {self.data_folder}")
        records = []

        for fname in os.listdir(self.data_folder):
            match = self.pattern.match(fname)
            if match:
                grid_id, lat, lon, ext = match.groups()
                records.append({
                    "grid_id": grid_id,
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": ext,
                    "filename": fname
                })

        if not records:
            print("âš ï¸� No matching files found.")
            return

        df = pd.DataFrame(records)
        grouped = df.groupby(["lat", "lon"]).agg({
            "grid_id": "first",
            "filename": list,
            "type": list
        }).reset_index()

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        grouped.to_csv(self.output_csv, index=False)

        print(f"âœ… Parsed {len(df)} files into {len(grouped)} unique location points.")
        print(f"ğŸ“„ CSV saved to: {self.output_csv}")


if __name__ == "__main__":
    parser = EDForestStructureParser(
        data_folder="/kaggle/input/amaztest/Amazon_ForestStructure_LIDAR_2412/Amazon_ForestStructure_LIDAR_2412/data",
        output_csv="/kaggle/working/amazon_data/amz_lidar_points2412.csv"
    )
    parser.run()



%%time
"""Graficar puntos lidar_2412"""
import folium
import pandas as pd
from folium.plugins import Fullscreen

class LidarPointTileMap:
    def __init__(self, csv_path, output_html="/kaggle/working/amazon_data/lidar_point_map.html", tile_size_deg=0.01):
        """
        tile_size_deg: tamaÃ±o del bbox en grados (aprox. 0.01 ~ 1km a nivel del ecuador)
        """
        self.csv_path = csv_path
        self.output_html = output_html
        self.tile_size = tile_size_deg
        self.center = [-5, -60]  # Centro aproximado del Amazonas

    def draw_map(self):
        print("ğŸ—ºï¸� Creando mapa desde CSV...")
        df = pd.read_csv(self.csv_path)
        fmap = folium.Map(location=self.center, zoom_start=5, tiles="CartoDB positron")
        Fullscreen().add_to(fmap)
        layer = folium.FeatureGroup(name="Amazon ED Points", show=True)

        for _, row in df.iterrows():
            lat = row["lat"]
            lon = row["lon"]
            filenames = eval(row["filename"]) if isinstance(row["filename"], str) else row["filename"]
            popup_text = "<br>".join(filenames)

            bounds = [
                [lat - self.tile_size / 2, lon - self.tile_size / 2],
                [lat + self.tile_size / 2, lon + self.tile_size / 2]
            ]
            folium.Rectangle(
                bounds=bounds,
                color="green",
                fill=True,
                fill_opacity=0.4,
                weight=1,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(layer)

        layer.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.save(self.output_html)
        print(f"âœ… Mapa guardado en: {self.output_html}")

if __name__ == "__main__":
    viewer = LidarPointTileMap(
        csv_path="/kaggle/working/amazon_data/amz_lidar_points2412.csv",
        output_html="/kaggle/working/amazon_maps/lidar_point_map2412.html"
    )
    viewer.draw_map()




display(Image(filename='/kaggle/input/amaztest/my_imgs/2412m.png'))


%%time
import os
import rasterio
from rasterio import warp
import folium

class GeoTIFFMapVisualizer:
    def __init__(self, input_folder, output_map_path):
        self.input_folder = input_folder
        self.output_map_path = output_map_path
        self.unique_bboxes = set()  # bbox Ãºnicos en formato (minlon,minlat, maxlon,maxlat)

    def extract_bbox(self, file_path):
        """Extrae el bounding box geogrÃ¡fico en lat/lon EPSG:4326"""
        try:
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                src_crs = src.crs

                # Esquinas del bounding box en coordenadas del archivo
                ul = (bounds.left, bounds.top)
                lr = (bounds.right, bounds.bottom)

                # Transformar coordenadas a EPSG:4326 lat,lon
                xs = [bounds.left, bounds.right]
                ys = [bounds.top, bounds.bottom]
                lons, lats = warp.transform(src_crs, "EPSG:4326", xs, ys)

                min_lon, max_lon = sorted(lons)
                min_lat, max_lat = sorted(lats)

                return ((min_lon, max_lat), (max_lon, min_lat))  # UL, LR en lat/lon

        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            return None

    def collect_unique_bboxes(self):
        """Recorre todos los .tif y recolecta bbox Ãºnicos redondeados para evitar duplicados espurios"""
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".tif"):
                file_path = os.path.join(self.input_folder, filename)
                bbox = self.extract_bbox(file_path)
                if bbox:
                    rounded_bbox = (
                        (round(bbox[0][0], 6), round(bbox[0][1], 6)),  # UL
                        (round(bbox[1][0], 6), round(bbox[1][1], 6))   # LR
                    )
                    self.unique_bboxes.add((rounded_bbox[0], rounded_bbox[1]))

    def create_map(self):
        """Crea el mapa Folium y dibuja cada bbox Ãºnico"""
        if not self.unique_bboxes:
            raise ValueError("No se encontraron bbox vÃ¡lidos")

        # Centramos el mapa aproximadamente
        example_bbox = next(iter(self.unique_bboxes))
        center_lat = (example_bbox[0][1] + example_bbox[1][1]) / 2
        center_lon = (example_bbox[0][0] + example_bbox[1][0]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

        for bbox in self.unique_bboxes:
            ul_lon, ul_lat = bbox[0]
            lr_lon, lr_lat = bbox[1]

            # Formato para folium.Rectangle: [[lat1, lon1], [lat2, lon2]]
            bounds = [[ul_lat, ul_lon], [lr_lat, lr_lon]]

            folium.Rectangle(
                bounds=bounds,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.4,
                popup="Ã�rea de imagen"
            ).add_to(m)

        # Guardar el mapa
        m.save(self.output_map_path)
        print(f"âœ… Mapa guardado en: {self.output_map_path}")

    def run(self):
        """Ejecuta todo el flujo"""
        self.collect_unique_bboxes()
        self.create_map()


if __name__ == "__main__":
    input_folder = "/kaggle/input/amaztest/LC03_SAR_LC_Biomass_1093/LC03_SAR_LC_Biomass_1093/data"
    output_map_path = "/kaggle/working/amazon_maps/LC03_SAR_LC_Biomass_1093_map.html"

    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)

    visualizer = GeoTIFFMapVisualizer(input_folder, output_map_path)
    visualizer.run()


display(Image(filename='/kaggle/input/amaztest/my_imgs/1093m.png'))


%%time
"""Mapeando s6 para ver que traer"""
import os
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import box

class BrazilLidarTileFilter:
    def __init__(self,
                 csv_path,
                 critical_geojson_path,
                 output_bbox_json,
                 output_critical_json,
                 bbox=(-79.617211, -20.535150, -43.399318, 10.059151)):
        
        self.csv_path = csv_path
        self.critical_geojson_path = critical_geojson_path
        self.output_bbox_json = output_bbox_json
        self.output_critical_json = output_critical_json
        self.bbox = bbox

    def run(self):
        print("ğŸ“¥ Leyendo CSV del inventario...")
        df = pd.read_csv(self.csv_path)

        required_cols = {"filename", "max_lat", "min_lat", "max_lon", "min_lon"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError("â�Œ El CSV no tiene las columnas requeridas.")

        print("ğŸŒ� Generando geometrÃ­as...")
        df["geometry"] = df.apply(
            lambda row: box(row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"]), axis=1)
        gdf_tiles = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        print("ğŸ§­ Filtrando por BBOX...")
        bbox_geom = box(*self.bbox)
        gdf_bbox = gdf_tiles[gdf_tiles.geometry.intersects(bbox_geom)]

        print("ğŸš¨ Filtrando por Ã¡reas crÃ­ticas...")
        gdf_crit = gpd.read_file(self.critical_geojson_path)
        gdf_critical = gdf_tiles[gdf_tiles.geometry.intersects(gdf_crit.unary_union)]

        print("ğŸ’¾ Guardando resultados...")
        gdf_bbox.drop(columns="geometry").to_json(self.output_bbox_json, orient="records", indent=2)
        gdf_critical.drop(columns="geometry").to_json(self.output_critical_json, orient="records", indent=2)

        print(f"âœ… Guardado: {self.output_bbox_json} ({len(gdf_bbox)} tiles)")
        print(f"âœ… Guardado: {self.output_critical_json} ({len(gdf_critical)} tiles)")

if __name__ == "__main__":
    filtro = BrazilLidarTileFilter(
        csv_path="/kaggle/input/amaztest/LidarIndex/cms_brazil_lidar_tile_inventory.csv",
        critical_geojson_path="/kaggle/working/amazon_data/critical_areas.geojson",
        output_bbox_json="/kaggle/working/amazon_data/lidar_amz_18_bbox.json",
        output_critical_json="/kaggle/working/amazon_data/lidar_amz_18_critical.json"
    )
    filtro.run()



%%time
"""Estimar peso en GB para los conjuntos"""
import json

# Cargar los dos archivos de salida generados
bbox_json_path = "/kaggle/working/amazon_data/lidar_amz_18_bbox.json"
critical_json_path = "/kaggle/working/amazon_data/lidar_amz_18_critical.json"

with open(bbox_json_path, "r") as f:
    bbox_data = json.load(f)

with open(critical_json_path, "r") as f:
    critical_data = json.load(f)

# Sumar los tamaÃ±os en MB y convertir a GB
bbox_gb = sum(float(tile["file_size_mb"]) for tile in bbox_data if tile["file_size_mb"]) / 1024
critical_gb = sum(float(tile["file_size_mb"]) for tile in critical_data if tile["file_size_mb"]) / 1024

round(bbox_gb, 2), round(critical_gb, 2)



%%time
"""Graficar puntos por bbox"""
import os
import json
import folium
from folium.plugins import Fullscreen
from shapely.geometry import box

class LidarTileMapViewer:
    def __init__(self,
                 bbox_json_path,
                 critical_json_path,
                 output_folder="/kaggle/working/amazon_maps",
                 output_file="lidar_tile_map.html"):
        self.bbox_json_path = bbox_json_path
        self.critical_json_path = critical_json_path
        self.output_folder = output_folder
        self.output_html = os.path.join(output_folder, output_file)
        self.map_center = [-5.0, -60.0]  # Approximate Amazon center

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _add_tile_layer(self, fmap, data, name, color):
        layer = folium.FeatureGroup(name=name, show=True)
        for tile in data:
            try:
                bounds = [
                    [tile["min_lat"], tile["min_lon"]],
                    [tile["max_lat"], tile["max_lon"]]
                ]
                folium.Rectangle(
                    bounds=bounds,
                    color=color,
                    fill=True,
                    weight=1,
                    fill_opacity=0.3,
                    popup=folium.Popup(tile["filename"], parse_html=False)
                ).add_to(layer)
            except KeyError:
                continue
        layer.add_to(fmap)

    def render(self):
        print("ğŸ—ºï¸� Creating interactive map...")
        os.makedirs(self.output_folder, exist_ok=True)
        fmap = folium.Map(location=self.map_center, zoom_start=5, tiles="CartoDB positron")
        Fullscreen().add_to(fmap)

        # Load tile data
        bbox_data = self._load_json(self.bbox_json_path)
        critical_data = self._load_json(self.critical_json_path)

        # Add each tile set as its own layer
        self._add_tile_layer(
            fmap,
            bbox_data,
            name="ğŸ“¦ All Amazon LIDAR Tiles (BBOX ~101.91 GB)",
            color="blue"
        )
        self._add_tile_layer(
            fmap,
            critical_data,
            name="ğŸš¨ Critical Tiles (Priority ~64.26 GB)",
            color="red"
        )

        # Layer toggle control
        folium.LayerControl(collapsed=False).add_to(fmap)

        # Save map
        fmap.save(self.output_html)
        print(f"âœ… Map saved to: {self.output_html}")

if __name__ == "__main__":
    viewer = LidarTileMapViewer(
        bbox_json_path="/kaggle/working/amazon_data/lidar_amz_18_bbox.json",
        critical_json_path="/kaggle/working/amazon_data/lidar_amz_18_critical.json"
    )
    viewer.render()



display(Image(filename='/kaggle/input/amaztest/my_imgs/tilesamz.png'))


%%time
import pandas as pd

# Leer el archivo CSV corregido
df = pd.read_csv("/kaggle/input/amaztest/LidarIndex/Pantropikal_map.csv")

# Mostrar las primeras filas para ver cÃ³mo se ve
print("Primeras filas del archivo:")
print(df.head(12))
print("\n")

# Filtrar archivos relacionados con el Amazonas
amazon_files = df[df['Data File'].str.contains(r'_.+_amazon_.+\.tif', regex=True)]

# Mostrar resultados filtrados
print("Archivos relacionados con el Amazonas:")
print(amazon_files[['Data File', 'Size']])
print("\n")

# FunciÃ³n para convertir tamaÃ±o a GB
def size_to_gb(size_str):
    if 'KB' in size_str:
        return float(size_str.replace('KB', '').strip()) / (1024 ** 2)
    elif 'MB' in size_str:
        return float(size_str.replace('MB', '').strip()) / 1024
    elif 'GB' in size_str:
        return float(size_str.replace('GB', '').strip())
    else:
        return 0  # Si no tiene unidad reconocida

# Aplicar conversiÃ³n y sumar
amazon_files['Size_GB'] = amazon_files['Size'].apply(size_to_gb)
total_size_gb = amazon_files['Size_GB'].sum()

# Imprimir tamaÃ±o total estimado
print(f"TamaÃ±o total estimado del dataset para el Amazonas: {total_size_gb:.2f} GB")


%%time

import os
import rasterio
from rasterio import warp
import folium

class GeoTIFFMapVisualizer:
    def __init__(self, input_folder, output_map_path):
        self.input_folder = input_folder
        self.output_map_path = output_map_path
        self.unique_bboxes = set()  # bbox Ãºnicos en formato ((min_lon, min_lat), (max_lon, max_lat))

    def extract_bbox(self, file_path):
        """Extrae el bounding box geogrÃ¡fico en lat/lon (EPSG:4326)"""
        try:
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                src_crs = src.crs

                # Esquinas del bounding box en coordenadas del archivo
                ul = (bounds.left, bounds.top)
                lr = (bounds.right, bounds.bottom)

                # Transformar coordenadas a EPSG:4326 (lat, lon)
                xs = [bounds.left, bounds.right]
                ys = [bounds.top, bounds.bottom]
                lons, lats = warp.transform(src_crs, "EPSG:4326", xs, ys)

                min_lon, max_lon = sorted(lons)
                min_lat, max_lat = sorted(lats)

                return ((min_lon, max_lat), (max_lon, min_lat))  # UL, LR en lat/lon

        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            return None

    def collect_unique_bboxes(self):
        """Recorre todos los .tif y recolecta bbox Ãºnicos redondeados para evitar duplicados espurios"""
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".tif"):
                file_path = os.path.join(self.input_folder, filename)
                bbox = self.extract_bbox(file_path)
                if bbox:
                    rounded_bbox = (
                        (round(bbox[0][0], 6), round(bbox[0][1], 6)),  # UL
                        (round(bbox[1][0], 6), round(bbox[1][1], 6))   # LR
                    )
                    self.unique_bboxes.add((rounded_bbox[0], rounded_bbox[1]))

    def create_map(self):
        """Crea el mapa Folium y dibuja cada bbox Ãºnico"""
        if not self.unique_bboxes:
            raise ValueError("No se encontraron bbox vÃ¡lidos")

        # Centramos el mapa aproximadamente
        example_bbox = next(iter(self.unique_bboxes))
        center_lat = (example_bbox[0][1] + example_bbox[1][1]) / 2
        center_lon = (example_bbox[0][0] + example_bbox[1][0]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

        for bbox in self.unique_bboxes:
            ul_lon, ul_lat = bbox[0]
            lr_lon, lr_lat = bbox[1]

            # Formato para folium.Rectangle: [[lat1, lon1], [lat2, lon2]]
            bounds = [[ul_lat, ul_lon], [lr_lat, lr_lon]]

            folium.Rectangle(
                bounds=bounds,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.4,
                popup="Ã�rea de imagen"
            ).add_to(m)

        # Guardar el mapa
        m.save(self.output_map_path)
        print(f"âœ… Mapa guardado en: {self.output_map_path}")

    def run(self):
        """Ejecuta todo el flujo"""
        self.collect_unique_bboxes()
        self.create_map()


if __name__ == "__main__":
    input_folder = "/kaggle/input/amaztest/Estimated_Biomass_Stock_Amazon_1648/Estimated_Biomass_Stock_Amazon_1648/data"
    output_map_path = "/kaggle/working/amazon_maps/Estimated_Biomass_Stock_Amazon_1648_map.html"

    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)

    visualizer = GeoTIFFMapVisualizer(input_folder, output_map_path)
    visualizer.run()


display(Image(filename='/kaggle/input/amaztest/my_imgs/1648m.png'))


%%time
import folium
import rasterio
import numpy as np
import os
import glob
import matplotlib
import matplotlib.colors as colors
from folium.raster_layers import ImageOverlay
from folium.plugins import MiniMap
import warnings

class RasterMapVisualizer:
    def __init__(self, input_dir, output_html_path):
        """
        Inicializa el visualizador de mapas raster.
        
        Args:
            input_dir (str): Directorio con los archivos raster .tif
            output_html_path (str): Path donde guardar el mapa HTML
        """
        self.input_dir = input_dir
        self.output_html_path = output_html_path
        self.map = None
        self.raster_paths = self._find_raster_files()
        
    def _find_raster_files(self):
        """Encuentra y ordena todos los archivos .tif en el directorio"""
        return sorted(glob.glob(os.path.join(self.input_dir, '*.tif')))
    
    def _get_raster_bounds(self, raster_path):
        """Obtiene los lÃ­mites geogrÃ¡ficos del raster"""
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    
    def _process_raster_data(self, array, cmap_name='viridis'):
        """Procesa los datos raster para visualizaciÃ³n"""
        # Convertir a array numpy estÃ¡ndar y manejar valores especiales
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.ma.filled(array, fill_value=np.nan)
            data = np.where(data == -9999, np.nan, data)  # Manejar valores nodata comunes
            
            # Calcular percentiles de forma robusta
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return None
                
            p2, p98 = np.percentile(valid_data, [2, 98])
            
            # NormalizaciÃ³n
            normalized = np.clip((data - p2) / (p98 - p2 + 1e-10), 0, 1)
            
            # Aplicar colormap
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
            colored = cmap(normalized)
            
            return (colored[..., :3] * 255).astype(np.uint8)
    
    def _read_raster(self, raster_path):
        """Lee un archivo raster con manejo robusto de datos"""
        with rasterio.open(raster_path) as src:
            # Leer datos y aplicar escala si existe
            array = src.read(1)
            if src.scales[0] != 1:
                array = array * src.scales[0]
                
            # Manejar valores nodata
            if src.nodata is not None:
                array[array == src.nodata] = np.nan
                
            bounds = self._get_raster_bounds(raster_path)
            return array, bounds
    
    def create_map(self):
        """Crea el mapa Folium con todas las capas raster"""
        if not self.raster_paths:
            raise ValueError("No se encontraron archivos .tif en el directorio")
        
        # ConfiguraciÃ³n inicial del mapa
        first_bounds = self._get_raster_bounds(self.raster_paths[0])
        center = [(first_bounds[0][0] + first_bounds[1][0]) / 2,
                 (first_bounds[0][1] + first_bounds[1][1]) / 2]
        
        # Mapa base con mÃºltiples opciones
        self.map = folium.Map(
            location=center,
            zoom_start=8,
            control_scale=True,
            tiles='OpenStreetMap',
            attr='Â© OpenStreetMap contributors'
        )
        
        # AÃ±adir capa satelital
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri, Maxar, Earthstar Geographics',
            name='Imagen Satelital',
            overlay=False
        ).add_to(self.map)
        
        # Capas de visualizaciÃ³n
        colormaps = ['viridis', 'plasma', 'coolwarm', 'magma', 'YlOrRd']
        
        for idx, raster_path in enumerate(self.raster_paths):
            try:
                array, bounds = self._read_raster(raster_path)
                
                # Verificar datos vÃ¡lidos
                if np.all(np.isnan(array)):
                    print(f"Advertencia: {os.path.basename(raster_path)} contiene solo valores NaN")
                    continue
                    
                layer_name = os.path.splitext(os.path.basename(raster_path))[0]
                colored_array = self._process_raster_data(array, colormaps[idx % len(colormaps)])
                
                if colored_array is None:
                    print(f"Advertencia: No se pudo procesar {layer_name} - datos invÃ¡lidos")
                    continue
                
                # AÃ±adir capa al mapa
                img_overlay = ImageOverlay(
                    image=colored_array,
                    bounds=bounds,
                    name=layer_name,
                    opacity=0.85,
                    interactive=True,
                    mercator_project=True
                )
                img_overlay.add_to(self.map)
                
                # AÃ±adir barra de colores
                self._add_colorbar(layer_name, array, colormaps[idx % len(colormaps)])
                
            except Exception as e:
                print(f"Error procesando {raster_path}: {str(e)}")
                continue
        
        # Controles del mapa
        folium.LayerControl(position='topright', collapsed=False).add_to(self.map)
        MiniMap(position='bottomleft').add_to(self.map)
        
        # AÃ±adir tÃ­tulo
        title_html = '''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                   z-index: 1000; background: rgba(255,255,255,0.9); padding: 5px 15px;
                   border: 2px solid grey; border-radius: 5px;
                   font-family: Arial; font-size: 14px; font-weight: bold;">
            Visualization of GEDI L3 rasters
        </div>
        '''
        self.map.get_root().html.add_child(folium.Element(title_html))
    
    def _add_colorbar(self, layer_name, data, cmap_name):
        """AÃ±ade una barra de colores interactiva"""
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return
            
        stats = {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'median': np.median(valid_data)
        }
        
        html = f'''
        <div style="position: fixed; bottom: 50px; right: 50px; 
                    width: 120px; height: 300px; padding: 5px;
                    border: 2px solid grey; z-index: 9999; 
                    background: rgba(255,255,255,0.9);
                    font-family: Arial; font-size: 11px;">
            <div style="text-align: center; font-weight: bold; border-bottom: 1px solid #ddd;">
                {layer_name}
            </div>
            <div style="height: 150px; margin: 5px 0; 
                        background: linear-gradient(to top, 
                            {matplotlib.colors.to_hex(cmap(0.0))}, 
                            {matplotlib.colors.to_hex(cmap(1.0))});">
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>{stats['min']:.2f}</span>
                <span>{stats['max']:.2f}</span>
            </div>
            <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 5px;">
                <div>Mean: {stats['mean']:.2f}</div>
                <div>Median: {stats['median']:.2f}</div>
            </div>
        </div>
        '''
        self.map.get_root().html.add_child(folium.Element(html))
    
    def save_map(self):
        """Guarda el mapa con manejo de errores"""
        try:
            os.makedirs(os.path.dirname(self.output_html_path), exist_ok=True)
            self.map.save(self.output_html_path)
            print(f"Mapa guardado exitosamente en: {self.output_html_path}")
            return True
        except Exception as e:
            print(f"Error al guardar el mapa: {str(e)}")
            return False

if __name__ == '__main__':
    # ConfiguraciÃ³n de paths
    input_dir = '/kaggle/input/amaztest/rasters_GEDIL3'
    output_html = '/kaggle/working/amazon_maps/rasters_GEDIL3_map.html'
    
    # EjecuciÃ³n principal
    print("Iniciando visualizaciÃ³n de rasters GEDI L3...")
    visualizer = RasterMapVisualizer(input_dir, output_html)
    
    print(f"Procesando {len(visualizer.raster_paths)} archivos raster...")
    try:
        visualizer.create_map()
        if visualizer.save_map():
            print("Proceso completado exitosamente!")
        else:
            print("Hubo problemas al guardar el mapa")
    except Exception as e:
        print(f"Error crÃ­tico: {str(e)}")


display(Image(filename='/kaggle/input/amaztest/my_imgs/GediL3.png'))


%%time
"""Este script brinda un reporte global de los datos"""
import os
import numpy as np
import rasterio
from rasterio.merge import merge

class DEMReporter:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.results = {}
    
    def load_layer(self, layer_type):
        """Load and merge layer files with error handling"""
        try:
            files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                    if f.endswith(f'{layer_type}.tif')]
            
            if not files:
                print(f"Note: No {layer_type} files found in directory")
                return None
                
            src_files = [rasterio.open(f) for f in files]
            mosaic, _ = merge(src_files)
            for src in src_files:
                src.close()
            return mosaic[0] if mosaic.ndim == 3 else mosaic
        except Exception as e:
            print(f"Could not process {layer_type} data: {str(e)}")
            return None
    
    def safe_stat(self, func, data, default=np.nan):
        """Safe calculation of statistics"""
        try:
            return func(data)
        except:
            return default
    
    def calculate_stats(self, data, name):
        """Calculate statistics with error protection"""
        if data is None:
            return None
            
        stats = {
            'min': self.safe_stat(np.nanmin, data),
            'max': self.safe_stat(np.nanmax, data),
            'mean': self.safe_stat(np.nanmean, data),
            'median': self.safe_stat(np.nanmedian, data),
            'std': self.safe_stat(np.nanstd, data),
            'q1': self.safe_stat(lambda x: np.nanpercentile(x, 25), data),
            'q3': self.safe_stat(lambda x: np.nanpercentile(x, 75), data),
            'valid': self.safe_stat(lambda x: np.count_nonzero(~np.isnan(x)), data, 0)
        }
        
        # Elevation-specific calculations
        if name == 'Elevation':
            try:
                slope = np.degrees(np.arctan(np.sqrt(np.gradient(data)[0]**2 + np.gradient(data)[1]**2)))
                stats.update({
                    'mean_slope': self.safe_stat(np.nanmean, slope),
                    'max_slope': self.safe_stat(np.nanmax, slope)
                })
            except:
                stats.update({
                    'mean_slope': np.nan,
                    'max_slope': np.nan
                })
        
        # Water-specific calculations
        if name == 'Water':
            try:
                stats['water_pct'] = np.count_nonzero(data > 0) / stats['valid'] * 100 if stats['valid'] > 0 else 0
            except:
                stats['water_pct'] = np.nan
        
        self.results[name] = stats
        return stats
    
    def print_section(self, title, stats):
        """Print a formatted section of the report"""
        if stats is None:
            print(f"\n{title}\nNo data available\n")
            return
            
        print(f"\n{title}")
        print("-" * 40)
        print(f"Minimum elevation: {stats['min']:.2f}")
        print(f"Maximum elevation: {stats['max']:.2f}")
        print(f"Average elevation: {stats['mean']:.2f}")
        print(f"Median elevation: {stats['median']:.2f}")
        print(f"Variation (std dev): {stats['std']:.2f}")
        print(f"25th percentile: {stats['q1']:.2f}")
        print(f"75th percentile: {stats['q3']:.2f}")
        print(f"Valid data points: {stats['valid']:,}")
        
        if 'mean_slope' in stats:
            print(f"\nTerrain slope:")
            print(f"Average slope: {stats['mean_slope']:.2f}Â°")
            print(f"Maximum slope: {stats['max_slope']:.2f}Â°")
        
        if 'water_pct' in stats:
            print(f"\nWater coverage: {stats['water_pct']:.2f}% of area")
    
    def generate_report(self):
        """Generate the complete analysis report"""
        print("\nCOPERNICUS DEM TERRAIN ANALYSIS")
        print("===============================")
        
        # Process elevation data
        elev_data = self.load_layer('DEM')
        elev_stats = self.calculate_stats(elev_data, 'Elevation')
        self.print_section("ELEVATION PROFILE", elev_stats)
        
        # Process water data
        water_data = self.load_layer('WBM')
        water_stats = self.calculate_stats(water_data, 'Water')
        self.print_section("WATER BODIES ANALYSIS", water_stats)
        
        # Process editing data
        edit_data = self.load_layer('EDM')
        edit_stats = self.calculate_stats(edit_data, 'Editing')
        self.print_section("DATA EDITING ANALYSIS", edit_stats)
        
        # Process error data
        error_data = self.load_layer('HEM')
        error_stats = self.calculate_stats(error_data, 'Height Error')
        self.print_section("HEIGHT ACCURACY ANALYSIS", error_stats)
        
        print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    analyzer = DEMReporter("/kaggle/working/gee_amz_xs/Copernicus_DEM_GLO30")
    analyzer.generate_report()


%%time
"""Graficar tiles en el mapa"""
import os
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from folium.raster_layers import ImageOverlay

class CopernicusDEMVisualizer:
    def __init__(self, tif_directory, output_dir):
        """
        Inicializa el visualizador con el directorio de archivos TIF y el directorio de salida
        
        Args:
            tif_directory (str): Ruta al directorio con archivos TIF de Copernicus
            output_dir (str): Ruta para guardar los mapas generados
        """
        self.tif_directory = tif_directory
        self.output_dir = output_dir
        self.dem_files = []
        self.mask_files = []
        
        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clasificar los archivos por tipo
        for file in os.listdir(self.tif_directory):
            file_path = os.path.join(self.tif_directory, file)
            if file.endswith('.DEM.tif'):
                self.dem_files.append(file_path)
            elif file.endswith('.EDM.tif'):
                self.mask_files.append(file_path)
    
    def get_bounds(self, tif_path):
        """Obtiene los lÃ­mites geogrÃ¡ficos de un archivo TIF"""
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    
    def normalize_array(self, array):
        """Normaliza un array numpy para visualizaciÃ³n"""
        array = np.nan_to_num(array)
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else array
    
    def apply_mask(self, dem_data, mask_file):
        """Aplica mÃ¡scara de calidad a los datos DEM si estÃ¡ disponible"""
        if os.path.exists(mask_file):
            with rasterio.open(mask_file) as mask_src:
                mask_data = mask_src.read(1)
                # Asumimos que valores altos en la mÃ¡scara indican mejor calidad
                dem_data[mask_data < 1] = np.nan  # Ajustar segÃºn criterios de calidad
        return dem_data
    
    def create_folium_map(self, output_filename='CopernicusMap.html', show_masks=False):
        """
        Crea un mapa Folium con los archivos DEM superpuestos
        
        Args:
            output_filename (str): Nombre del archivo HTML de salida
            show_masks (bool): Si True, muestra las mÃ¡scaras de calidad
        """
        if not self.dem_files:
            print("No se encontraron archivos DEM.tif en el directorio.")
            return
        
        # Usar el primer archivo para centrar el mapa
        first_bounds = self.get_bounds(self.dem_files[0])
        center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
        center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2
        
        # Crear mapa Folium
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery'
        )
        
        # AÃ±adir capa de elevaciÃ³n (DEM)
        for dem_file in self.dem_files:
            base_name = os.path.basename(dem_file).replace('.DEM.tif', '')
            mask_file = dem_file.replace('.DEM.tif', '.EDM.tif')
            
            with rasterio.open(dem_file) as src:
                data = src.read(1)
                
                # Aplicar mÃ¡scara de calidad si existe
                if os.path.exists(mask_file):
                    data = self.apply_mask(data, mask_file)
                
                bounds = self.get_bounds(dem_file)
                
                # Normalizar datos para visualizaciÃ³n
                norm_data = self.normalize_array(data)
                
                # Crear figura matplotlib
                fig, ax = plt.subplots(figsize=(10, 10))
                img = ax.imshow(norm_data, cmap='terrain', vmin=0, vmax=1)
                plt.axis('off')
                plt.tight_layout()
                
                # Guardar temporalmente la imagen
                temp_img_path = os.path.join(self.output_dir, f'temp_dem_{base_name}.png')
                plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                
                # AÃ±adir al mapa Folium
                img_overlay = ImageOverlay(
                    image=temp_img_path,
                    bounds=bounds,
                    opacity=0.7,
                    interactive=True,
                    cross_origin=False,
                    zindex=1,
                    name=f'DEM_{base_name}'
                )
                img_overlay.add_to(m)
                
                # Eliminar archivo temporal
                os.remove(temp_img_path)
        
        # AÃ±adir mÃ¡scaras de calidad si se solicita
        if show_masks and self.mask_files:
            for mask_file in self.mask_files:
                base_name = os.path.basename(mask_file).replace('.EDM.tif', '')
                
                with rasterio.open(mask_file) as src:
                    mask_data = src.read(1)
                    bounds = self.get_bounds(mask_file)
                    
                    # Normalizar datos para visualizaciÃ³n
                    norm_mask = self.normalize_array(mask_data)
                    
                    # Crear figura matplotlib
                    fig, ax = plt.subplots(figsize=(10, 10))
                    img = ax.imshow(norm_mask, cmap='viridis', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Guardar temporalmente la imagen
                    temp_img_path = os.path.join(self.output_dir, f'temp_mask_{base_name}.png')
                    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0, transparent=True)
                    plt.close()
                    
                    # AÃ±adir al mapa Folium
                    mask_overlay = ImageOverlay(
                        image=temp_img_path,
                        bounds=bounds,
                        opacity=0.5,
                        interactive=True,
                        cross_origin=False,
                        zindex=2,
                        name=f'MÃ¡scara_{base_name}'
                    )
                    mask_overlay.add_to(m)
                    
                    # Eliminar archivo temporal
                    os.remove(temp_img_path)
        
        # AÃ±adir control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa
        output_path = os.path.join(self.output_dir, output_filename)
        m.save(output_path)
        print(f"Mapa de Copernicus DEM guardado en: {output_path}")
        
        return m

if __name__ == '__main__':
    # ConfiguraciÃ³n de rutas
    tif_directory = '/kaggle/working/gee_amz_xs/Copernicus_DEM_GLO30'
    output_dir = '/kaggle/working/amazon_maps'
    
    # Crear visualizador y generar mapa
    visualizer = CopernicusDEMVisualizer(tif_directory, output_dir)
    folium_map = visualizer.create_folium_map('CopernicusMap.html', show_masks=True)


display(Image(filename='/kaggle/input/amaztest/my_imgs/CopM.png'))


%%time
"""Se procede a crear un reporte por capas"""
import os
import numpy as np
import rasterio
from rasterio.merge import merge

class ALOSDEMAnalyzer:
    def __init__(self, data_dir):
        """Initialize with ALOS DEM data directory"""
        self.data_dir = data_dir
        self.results = {}
        
    def safe_calculation(self, func, data, default=None):
        """Safe wrapper for numpy calculations"""
        try:
            with np.errstate(all='ignore'):
                result = func(data)
                if np.isnan(result) or np.isinf(result):
                    return default
                return result
        except:
            return default
    
    def load_and_merge_layer(self, layer_type):
        """Load and merge tiles with memory optimization"""
        print(f"Processing {layer_type} layer...")
        try:
            files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                    if f.endswith(f'{layer_type}.tif')]
            
            if not files:
                print(f"Note: No {layer_type} files found in directory")
                return None
                
            # Process in chunks if memory is an issue
            src_files = [rasterio.open(f) for f in files]
            mosaic, _ = merge(src_files)
            
            for src in src_files:
                src.close()
                
            data = mosaic[0] if mosaic.ndim == 3 else mosaic
            return data.astype(np.float32)  # Reduce memory usage
            
        except Exception as e:
            print(f"Error processing {layer_type}: {str(e)}")
            return None
    
    def calculate_layer_stats(self, data, layer_name):
        """Calculate statistics with numerical stability"""
        if data is None:
            return None
            
        stats = {
            'layer': layer_name,
            'min': self.safe_calculation(np.nanmin, data),
            'max': self.safe_calculation(np.nanmax, data),
            'mean': self.safe_calculation(np.nanmean, data),
            'median': self.safe_calculation(np.nanmedian, data),
            'std_dev': self.safe_calculation(np.nanstd, data),
            'q1': self.safe_calculation(lambda x: np.nanpercentile(x, 25), data),
            'q3': self.safe_calculation(lambda x: np.nanpercentile(x, 75), data),
            'valid_pixels': self.safe_calculation(lambda x: np.count_nonzero(~np.isnan(x)), data, 0),
            'null_pixels': self.safe_calculation(lambda x: np.count_nonzero(np.isnan(x)), data, 0)
        }
        
        # DSM-specific calculations with numerical safety
        if layer_name == 'DSM':
            stats['elevation_range'] = self.safe_calculation(
                lambda: stats['max'] - stats['min'], None) if None not in [stats['max'], stats['min']] else None
            
            # Slope calculations with chunking for large arrays
            try:
                with np.errstate(all='ignore'):
                    x, y = np.gradient(data.astype(np.float64))  # Higher precision for derivatives
                    slope = np.degrees(np.arctan(np.sqrt(x**2 + y**2)))
                    stats.update({
                        'mean_slope': self.safe_calculation(np.nanmean, slope),
                        'max_slope': self.safe_calculation(np.nanmax, slope),
                        'slope_std': self.safe_calculation(np.nanstd, slope)
                    })
            except Exception as e:
                print(f"Slope calculation warning: {str(e)}")
                stats.update({
                    'mean_slope': None,
                    'max_slope': None,
                    'slope_std': None
                })
        
        # MSK-specific calculations
        elif layer_name == 'MSK':
            try:
                unique, counts = np.unique(data, return_counts=True)
                stats['mask_classes'] = dict(zip(unique.astype(int), counts))
            except:
                stats['mask_classes'] = {}
        
        # STK-specific calculations
        elif layer_name == 'STK':
            stats['stack_range'] = self.safe_calculation(
                lambda: stats['max'] - stats['min'], None) if None not in [stats['max'], stats['min']] else None
            stats['unique_stacks'] = self.safe_calculation(lambda x: len(np.unique(x)), data)
        
        self.results[layer_name] = stats
        return stats
    
    def format_number(self, value):
        """Format numbers consistently"""
        if value is None:
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return f"{float(value):.2f}"
    
    def print_layer_report(self, stats):
        """Print formatted statistics with numerical safety"""
        if stats is None:
            print("No data available for this layer")
            return
            
        print(f"\n{stats['layer']} ANALYSIS")
        print("-" * 40)
        print(f"Minimum value: {self.format_number(stats['min'])}")
        print(f"Maximum value: {self.format_number(stats['max'])}")
        print(f"Mean value: {self.format_number(stats['mean'])}")
        print(f"Median value: {self.format_number(stats['median'])}")
        print(f"Standard deviation: {self.format_number(stats['std_dev'])}")
        print(f"First quartile (Q1): {self.format_number(stats['q1'])}")
        print(f"Third quartile (Q3): {self.format_number(stats['q3'])}")
        print(f"Valid pixels: {self.format_number(stats['valid_pixels'])}")
        print(f"Null pixels: {self.format_number(stats['null_pixels'])}")
        
        if stats['layer'] == 'DSM' and stats.get('elevation_range') is not None:
            print(f"\nElevation range: {self.format_number(stats['elevation_range'])} meters")
            if stats.get('mean_slope') is not None:
                print(f"Mean slope: {self.format_number(stats['mean_slope'])} degrees")
                print(f"Maximum slope: {self.format_number(stats['max_slope'])} degrees")
                print(f"Slope variability: {self.format_number(stats['slope_std'])}")
        
        elif stats['layer'] == 'MSK' and stats.get('mask_classes'):
            print("\nMask class distribution:")
            for val, count in sorted(stats['mask_classes'].items()):
                print(f"Class {val}: {self.format_number(count)} pixels")
        
        elif stats['layer'] == 'STK':
            if stats.get('stack_range') is not None:
                print(f"\nStack value range: {self.format_number(stats['stack_range'])}")
            if stats.get('unique_stacks') is not None:
                print(f"Unique stack values: {self.format_number(stats['unique_stacks'])}")
    
    def generate_summary_report(self):
        """Generate comparative summary with safe formatting"""
        if not self.results:
            print("No data available for summary report")
            return
            
        print("\nSUMMARY COMPARISON")
        print("-" * 40)
        print("{:<15} {:<12} {:<12} {:<12} {:<12}".format(
            'Layer', 'Min', 'Max', 'Mean', 'Std Dev'))
        
        for layer, stats in self.results.items():
            print("{:<15} {:<12} {:<12} {:<12} {:<12}".format(
                layer,
                self.format_number(stats['min']),
                self.format_number(stats['max']),
                self.format_number(stats['mean']),
                self.format_number(stats['std_dev'])))
    
    def analyze_all_layers(self):
        """Run complete analysis with memory management"""
        print("\nALOS DEM AW3D30 ANALYSIS REPORT")
        print("=" * 40)
        
        # Process each layer with cleanup
        layers = ['DSM', 'MSK', 'STK']
        for layer in layers:
            data = self.load_and_merge_layer(layer)
            stats = self.calculate_layer_stats(data, layer)
            self.print_layer_report(stats)
            del data  # Free memory
            
        self.generate_summary_report()
        print("\nAnalysis completed successfully")


if __name__ == "__main__":
    analyzer = ALOSDEMAnalyzer("/kaggle/working/gee_amz_xs/ALOS_DEM_AW3D30")
    analyzer.analyze_all_layers()


%%time
"""Graficar tiles en el mapa"""
import os
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from folium.raster_layers import ImageOverlay

class ALOSDEMVisualizer:
    def __init__(self, tif_directory, output_dir):
        """
        Inicializa el visualizador con el directorio de archivos TIF y el directorio de salida
        
        Args:
            tif_directory (str): Ruta al directorio con archivos TIF
            output_dir (str): Ruta para guardar los mapas generados
        """
        self.tif_directory = tif_directory
        self.output_dir = output_dir
        self.dsm_files = []
        
        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Encontrar todos los archivos DSM
        for file in os.listdir(self.tif_directory):
            if file.endswith('.DSM.tif'):
                self.dsm_files.append(os.path.join(self.tif_directory, file))
    
    def get_bounds(self, tif_path):
        """Obtiene los lÃ­mites geogrÃ¡ficos de un archivo TIF"""
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    
    def normalize_array(self, array):
        """Normaliza un array numpy para visualizaciÃ³n"""
        array = np.nan_to_num(array)
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)
    
    def create_folium_map(self, output_filename='CopernicusMap.html'):
        """
        Crea un mapa Folium con todos los archivos DSM superpuestos
        
        Args:
            output_filename (str): Nombre del archivo HTML de salida
        """
        if not self.dsm_files:
            print("No se encontraron archivos DSM.tif en el directorio.")
            return
        
        # Usar el primer archivo para centrar el mapa
        first_bounds = self.get_bounds(self.dsm_files[0])
        center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
        center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2
        
        # Crear mapa Folium
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery'
        )
        
        # AÃ±adir cada archivo DSM como capa
        for dsm_file in self.dsm_files:
            with rasterio.open(dsm_file) as src:
                data = src.read(1)
                bounds = self.get_bounds(dsm_file)
                
                # Normalizar datos para visualizaciÃ³n
                norm_data = self.normalize_array(data)
                
                # Crear figura matplotlib
                fig, ax = plt.subplots(figsize=(10, 10))
                img = ax.imshow(norm_data, cmap='terrain')
                plt.axis('off')
                plt.tight_layout()
                
                # Guardar temporalmente la imagen
                temp_img_path = os.path.join(self.output_dir, 'temp_img.png')
                plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                
                # AÃ±adir al mapa Folium
                img_overlay = ImageOverlay(
                    image=temp_img_path,
                    bounds=bounds,
                    opacity=0.7,
                    interactive=True,
                    cross_origin=False,
                    zindex=1,
                    name=os.path.basename(dsm_file)
                )
                img_overlay.add_to(m)
                
                # Eliminar archivo temporal
                os.remove(temp_img_path)
        
        # AÃ±adir control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa
        output_path = os.path.join(self.output_dir, output_filename)
        m.save(output_path)
        print(f"Mapa guardado en: {output_path}")
        
        return m

if __name__ == '__main__':
    # ConfiguraciÃ³n de rutas
    tif_directory = '/kaggle/working/gee_amz_xs/ALOS_DEM_AW3D30'
    output_dir = '/kaggle/working/amazon_maps'
    
    # Crear visualizador y generar mapa
    visualizer = ALOSDEMVisualizer(tif_directory, output_dir)
    folium_map = visualizer.create_folium_map('AlosMap.html')



display(Image(filename='/kaggle/input/amaztest/my_imgs/AlosM.png'))


%%time
"""Se crea el reporte por capas"""
import os
import glob
import numpy as np
import rasterio
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Worker Function ---
def analyze_area_task(area_data):
    area_id, file_dict = area_data
    area_results = []
    for band_name, file_path in file_dict.items():
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1, masked=True)
                valid_data = data.compressed()
                if valid_data.size > 0:
                    area_results.append({'band': band_name, 'data': valid_data})
        except Exception:
            continue
    return area_results

class GEDIParallelAnalyzer:
    """
    A high-performance, parallel class to analyze thousands of individual GEDI band files
    by distributing work by area, avoiding I/O contention, and producing professional reports.
    """
    def __init__(self, base_path: str, num_workers: int = 2):
        print(f"--- Initializing GEDI Parallel Analysis for: {base_path} ---")
        self.num_workers = min(num_workers, cpu_count())
        self.area_files = self._group_files_by_area(base_path)
        print(f"âœ… Found {len(self.area_files)} areas. Will use {self.num_workers} worker processes.")
        if self.area_files:
            first_area_key = next(iter(self.area_files))
            first_band_file = next(iter(self.area_files[first_area_key].values()))
            with rasterio.open(first_band_file) as src:
                self.pixels_per_raster = src.width * src.height
        else:
            self.pixels_per_raster = 0

    def _group_files_by_area(self, base_path: str) -> dict:
        all_tiffs = glob.glob(os.path.join(base_path, "*.tif"))
        grouped = {}
        for f_path in all_tiffs:
            try:
                basename = os.path.basename(f_path).replace('.tif', '')
                parts = basename.split('.')
                band_name, area_id = parts[-1], parts[0].split('_')[-1]
                if area_id not in grouped: grouped[area_id] = {}
                grouped[area_id][band_name] = f_path
            except IndexError: continue
        return grouped

    def analyze_in_parallel(self) -> pd.DataFrame | None:
        if not self.area_files: return None
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_area_task, self.area_files.items()), total=len(self.area_files), desc="Analyzing Areas in Parallel"))
        if not results: return None
        print("\n[INFO] Aggregating results from all files...")
        all_band_data_list = [item for sublist in results for item in sublist]
        if not all_band_data_list: return None
        band_data_agg = {}
        for item in all_band_data_list:
            band = item['band']
            if band not in band_data_agg: band_data_agg[band] = []
            band_data_agg[band].append(item['data'])
        final_stats = []
        for band, data_arrays in tqdm(band_data_agg.items(), desc="Calculating Final Stats"):
            full_array = np.concatenate(data_arrays)
            clean_array = full_array[np.isfinite(full_array)]
            if clean_array.size == 0: continue
            stats = {
                'band': band, 'min': np.min(clean_array), 'max': np.max(clean_array),
                'mean': np.mean(clean_array), 'median': np.median(clean_array),
                'std': np.std(clean_array), 'q1': np.percentile(clean_array, 25),
                'q3': np.percentile(clean_array, 75), 'valid_pixels': clean_array.size,
            }
            if np.issubdtype(clean_array.dtype, np.integer):
                unique_vals, counts = np.unique(clean_array, return_counts=True)
                if len(unique_vals) <= 50:
                    stats['distribution'] = dict(zip(unique_vals.astype(int), counts.astype(int)))
            final_stats.append(stats)
        if not final_stats: return None
        stats_df = pd.DataFrame(final_stats).set_index('band')
        total_possible_pixels = self.pixels_per_raster * len(self.area_files)
        stats_df['null_pixels'] = total_possible_pixels - stats_df['valid_pixels']
        return stats_df

    def generate_formatted_report(self, stats_df: pd.DataFrame, relevant_bands: list):
        """Generates a detailed text report focusing on relevant bands."""
        if stats_df.empty:
            print("[WARNING] Cannot generate report from empty stats DataFrame.")
            return
        print("\n" + "="*80)
        print("GEDI SOIL-RELEVANT ANALYSIS REPORT".center(80))
        report_df = stats_df.reindex(relevant_bands).dropna(subset=['mean'])
        for band_name, row in report_df.iterrows():
            print("\n" + "="*80)
            print(f"Processing {band_name.upper()} layer...")
            print(f"{band_name.upper()} ANALYSIS".center(80))
            print("----------------------------------------".center(80))
            print(f"{'Minimum value:':<25} {row['min']:.2f}")
            print(f"{'Maximum value:':<25} {row['max']:.2f}")
            print(f"{'Mean value:':<25} {row['mean']:.2f}")
            print(f"{'Median value:':<25} {row['median']:.2f}")
            print(f"{'Standard deviation:':<25} {row['std']:.2f}")
            print(f"{'First quartile (Q1):':<25} {row['q1']:.2f}")
            print(f"{'Third quartile (Q3):':<25} {row['q3']:.2f}")
            print(f"{'Valid pixels:':<25} {int(row['valid_pixels']):,}")
            print(f"{'Null pixels:':<25} {int(row['null_pixels']):,}")
            if isinstance(row.get('distribution'), dict):
                print("\nMask class distribution:")
                for class_val, count in sorted(row['distribution'].items()):
                    print(f"  Class {int(class_val)}: {int(count):,} pixels")
        print("\n" + "="*80)
        print("SUMMARY COMPARISON".center(80))
        print("----------------------------------------".center(80))
        summary_table = report_df[['min', 'max', 'mean', 'std']].copy()
        summary_table.index.name = "Layer"
        print(summary_table.round(2).to_string(justify='right', header=True))
        print("="*80)

# --- Main Execution Block ---
if __name__ == "__main__":
    base_data_path = "/kaggle/working/gee_amz_xs/GEDI_Monthly"
    
    # Bandas mÃ¡s importantes para el estudio del suelo y el ecosistema
    RELEVANT_BANDS_FOR_SOIL_STUDY = [
        'digital_elevation_model',
        'landsat_treecover',
        'landsat_water_persistence',
        'rh25', 'rh50', 'rh75', 'rh95', 'rh100',
        'quality_flag'
    ]
    
    try:
        analyzer = GEDIParallelAnalyzer(base_path=base_data_path, num_workers=2)
        if analyzer.area_files:
            stats_df = analyzer.analyze_in_parallel()
            if stats_df is not None and not stats_df.empty:
                analyzer.generate_formatted_report(stats_df, RELEVANT_BANDS_FOR_SOIL_STUDY)
            else:
                print("[ERROR] Analysis completed, but no statistics were generated.")
        
        print("\nğŸ�‰ GEDI parallel analysis completed successfully.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] The analysis failed: {e}")


%%time
"""Super grafico"""
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Bandas relevantes para anÃ¡lisis
RELEVANT_BANDS = [
    'digital_elevation_model',
    'landsat_treecover',
    'landsat_water_persistence',
    'rh25', 'rh50', 'rh75', 'rh95', 'rh100',
    'quality_flag'
]

def plot_raster_and_violin(tif_path, band_name):
    """Muestra lado a lado la imagen raster y el grÃ¡fico de violÃ­n."""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            print(f"[DEBUG] {band_name} - Datos cargados. Shape: {data.shape}, dtype: {data.dtype}")

            # MÃ¡scara de valores invÃ¡lidos
            data = np.ma.masked_invalid(data)
            valid_data = data.compressed()

            if valid_data.size == 0:
                print(f"[DEBUG] {band_name} - No hay datos vÃ¡lidos (todos NaN o vacÃ­o).")
                return

            # Mostrar estadÃ­sticas bÃ¡sicas
            print(f"[DEBUG] {band_name} - Min: {np.min(valid_data):.2f}, Max: {np.max(valid_data):.2f}, Media: {np.mean(valid_data):.2f}")

            # Mejorar contraste con percentiles
            p_low, p_high = np.nanpercentile(valid_data, [2, 98])
            clipped_data = np.clip(data, p_low, p_high)

            # Plotear
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Vista raster
            img = axs[0].imshow(clipped_data, cmap='viridis')
            axs[0].set_title(f"{band_name} - Raster view")
            axs[0].axis('off')
            fig.colorbar(img, ax=axs[0], label="Value")

            # GrÃ¡fico de violÃ­n
            axs[1].violinplot(valid_data, showmeans=False, showmedians=True)
            axs[1].set_title(f"Value Distribution ({band_name})")
            axs[1].set_xticks([])
            axs[1].grid(True, axis='y')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"[ERROR] No se pudo graficar '{band_name}': {e}")



def extract_band_statistics(tif_path, band_name):
    """Extrae estadÃ­sticas relevantes de una capa raster."""
    stats = {
        'band': band_name,
        'file': os.path.basename(tif_path),
        'mean': np.nan,
        'median': np.nan,
        'std': np.nan,
        'min': np.nan,
        'max': np.nan,
        'p05': np.nan,
        'p25': np.nan,
        'p75': np.nan,
        'p95': np.nan,
        'valid_pixels': 0
    }
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            data = np.ma.masked_invalid(data)
            valid_data = data.compressed()
            if valid_data.size == 0:
                print(f"[WARN] {band_name} - Sin datos vÃ¡lidos.")
                return stats
            stats.update({
                'mean': np.mean(valid_data),
                'median': np.median(valid_data),
                'std': np.std(valid_data),
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'p05': np.percentile(valid_data, 5),
                'p25': np.percentile(valid_data, 25),
                'p75': np.percentile(valid_data, 75),
                'p95': np.percentile(valid_data, 95),
                'valid_pixels': valid_data.size
            })
    except Exception as e:
        print(f"[ERROR] No se pudo extraer estadÃ­sticas de '{band_name}': {e}")
    return stats



def find_and_plot_all_bands(base_folder):
    """Busca, grafica y resume estadÃ­sticas de cada banda relevante en la carpeta."""
    if not os.path.exists(base_folder):
        print(f"[ERROR] Carpeta no encontrada: {base_folder}")
        return

    all_files = [f for f in os.listdir(base_folder) if f.lower().endswith(".tif")]
    found_bands = set()
    stats_list = []

    print(f"[INFO] Seeking {len(RELEVANT_BANDS)} bands in {len(all_files)} TIFF files...")

    for band in RELEVANT_BANDS:
        found = False
        for file in all_files:
            if file.endswith(f".{band}") or f".{band}." in file:
                full_path = os.path.join(base_folder, file)
                print(f"[INFO] Procesando banda: {band}")
                plot_raster_and_violin(full_path, band)
                stats = extract_band_statistics(full_path, band)
                stats_list.append(stats)
                found_bands.add(band)
                found = True
                break
        if not found:
            print(f"[ERROR] No se encontrÃ³ archivo para la banda '{band}'.")

    # Guardar estadÃ­sticas en CSV
    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        output_csv = os.path.join(base_folder, "/kaggle/working/amazon_data/GEDI_band_statistics_summary.csv")
        df_stats.to_csv(output_csv, index=False)
        print(f"\n[âœ“] EstadÃ­sticas guardadas en: {output_csv}")

    # Resumen final
    print("\n[SUMMARY]")
    print(f"Found and plotted bands: {', '.join(found_bands)}")
    missing = set(RELEVANT_BANDS) - found_bands
    if missing:
        print(f"Missing bands: {', '.join(missing)}")
    else:
        print("âœ… Â¡Todas las bandas relevantes fueron graficadas!")



# --- Ejecutar ---
if __name__ == "__main__":
    base_data_path = "/kaggle/working/gee_amz_xs/GEDI_Monthly"
    find_and_plot_all_bands(base_data_path)


dt = pd.read_csv("/kaggle/working/amazon_data/GEDI_band_statistics_summary.csv")
print(dt.head(9) )
print(f"The shape is: {dt.shape}")


%%time
"""Analizar datos en bruto por conjunto"""
import os
import glob
import rasterio
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def analyze_band_stats(file_path):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            dtype = str(data.dtype)
            nodata = src.nodata

            # MÃ¡scara de valores finitos
            finite_mask = np.isfinite(data)
            valid_data = data[finite_mask]

            stats = {
                'total_pixels': data.size,
                'valid_count': valid_data.size,
                'nan_count': np.isnan(data).sum(),
                'inf_count': np.isinf(data).sum(),
                'min_val': float(valid_data.min()) if valid_data.size > 0 else None,
                'max_val': float(valid_data.max()) if valid_data.size > 0 else None,
                'mean_val': float(valid_data.mean()) if valid_data.size > 0 else None,
                'std_val': float(valid_data.std()) if valid_data.size > 0 else None,
                'q1_val': float(np.percentile(valid_data, 25)) if valid_data.size > 0 else None,
                'q2_val': float(np.percentile(valid_data, 50)) if valid_data.size > 0 else None,
                'q3_val': float(np.percentile(valid_data, 75)) if valid_data.size > 0 else None,
                'dtype': dtype,
                'nodata': str(nodata),
            }
            return stats
    except Exception as e:
        return {'error': str(e)}

def analyze_all_bands(folder_path):
    print("ğŸ”� Starting exploratory analysis of all GEDI bands...")
    
    # Buscar todos los archivos TIFF recursivamente
    tif_files = glob.glob(os.path.join(folder_path, "**", "*.tif"), recursive=True)
    if not tif_files:
        print("[ERROR] No .tif files found")
        return

    print(f"[INFO] Total TIFF files found: {len(tif_files)}")

    # Diccionarios para agrupar por banda
    files_by_band = defaultdict(list)
    
    # Agrupar archivos por banda
    for file in tif_files:
        filename = os.path.basename(file).replace('.tif', '')
        parts = filename.split('.')
        if len(parts) < 2:
            continue
        band_name = parts[-1]
        files_by_band[band_name].append(file)

    print(f"[INFO] Unique bands detected: {len(files_by_band)}\n")

    # Resultado final
    band_summary = {}

    for band, files in tqdm(files_by_band.items(), desc="Analyzing bands"):
        stats_list = []
        example_files = files[:3]  # Solo tomamos ejemplos

        for file in example_files:
            stats = analyze_band_stats(file)
            if 'error' not in stats:
                stats_list.append(stats)

        # Tomar el primer archivo para tipo de dato y nodata
        first_stat = stats_list[0] if stats_list else {}
        
        # Valores tÃ­picos
        min_vals = [s['min_val'] for s in stats_list if s['min_val'] is not None]
        max_vals = [s['max_val'] for s in stats_list if s['max_val'] is not None]
        mean_vals = [s['mean_val'] for s in stats_list if s['mean_val'] is not None]
        std_vals = [s['std_val'] for s in stats_list if s['std_val'] is not None]
        q1_vals = [s['q1_val'] for s in stats_list if s['q1_val'] is not None]
        q2_vals = [s['q2_val'] for s in stats_list if s['q2_val'] is not None]
        q3_vals = [s['q3_val'] for s in stats_list if s['q3_val'] is not None]

        band_summary[band] = {
            'file_count': len(files),
            'dtype': first_stat.get('dtype', 'unknown'),
            'nodata': first_stat.get('nodata', 'unknown'),
            'examples': [os.path.basename(f) for f in example_files],
            'min_global': min(min_vals) if min_vals else None,
            'max_global': max(max_vals) if max_vals else None,
            'mean_avg': np.mean(mean_vals).round(2) if mean_vals else None,
            'std_avg': np.mean(std_vals).round(2) if std_vals else None,
            'q1_avg': np.mean(q1_vals).round(2) if q1_vals else None,
            'q2_avg': np.mean(q2_vals).round(2) if q2_vals else None,
            'q3_avg': np.mean(q3_vals).round(2) if q3_vals else None,
            'valid_pct_avg': np.mean([100 * s['valid_count']/s['total_pixels'] for s in stats_list]).round(2) if stats_list else None,
            'nan_pct_avg': np.mean([100 * s['nan_count']/s['total_pixels'] for s in stats_list]).round(2) if stats_list else None,
            'inf_pct_avg': np.mean([100 * s['inf_count']/s['total_pixels'] for s in stats_list]).round(2) if stats_list else None,
        }

    # Imprimir resultados
    print("\nğŸ“Š BAND SUMMARY\n")
    print("-" * 120)
    print(f"{'Band':<20} | Files | Dtype     | NoData | %Valid | %NaN   | %Inf   | Min       | Max       | Mean      | Std       | Q1        | Q2(Med)   | Q3")
    print("-" * 120)

    for band, summary in sorted(band_summary.items(), key=lambda x: -x[1]['file_count']):
        print(
            f"{band:<20} | {summary['file_count']:<5} | {summary['dtype']:<9} | {summary['nodata']:<6} | "
            f"{summary['valid_pct_avg'] or 0:<6.2f} | {summary['nan_pct_avg'] or 0:<6.2f} | "
            f"{summary['inf_pct_avg'] or 0:<6.2f} | {summary['min_global'] or 0:<9.2f} | {summary['max_global'] or 0:<9.2f} | "
            f"{summary['mean_avg'] or 0:<9.2f} | {summary['std_avg'] or 0:<9.2f} | {summary['q1_avg'] or 0:<9.2f} | "
            f"{summary['q2_avg'] or 0:<9.2f} | {summary['q3_avg'] or 0:<9.2f}"
        )

    print("-" * 120 + "\n")

    print("ğŸ“Œ Examples of filenames by band:")
    for band, summary in list(band_summary.items())[:5]:
        print(f" â†’ Band: {band}")
        for ex in summary['examples']:
            print(f"    - {ex}")

# --- RUN --- 
if __name__ == "__main__":
    BASE_FOLDER = "/kaggle/working/gee_amz_xs/GEDI_Monthly"
    analyze_all_bands(BASE_FOLDER)


%%time
"""Se procede a analizar la diversidad por locacion"""
import os
import glob
import numpy as np
import rasterio
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Worker Function ---
# This function must be at the top level. It processes one single .tif file.
def analyze_diversity_file_task(file_path: str) -> np.ndarray | None:
    """
    Worker task: opens a single Topographic Diversity GeoTIFF, reads its valid data,
    and returns it as a NumPy array.
    """
    try:
        with rasterio.open(file_path) as src:
            # Read the data, automatically applying the NoData mask
            data = src.read(1, masked=True)
            # Filter for finite values to remove potential -inf, inf, NaN
            valid_data = data.compressed()
            clean_data = valid_data[np.isfinite(valid_data)]
            if clean_data.size > 0:
                return clean_data
    except Exception:
        # Return None if the file is corrupt or cannot be processed
        return None
    return None

class SRTMDiversityAnalyzer:
    """
    A high-performance, parallel class to analyze Topographic Diversity data,
    providing a detailed, professionally formatted statistical report.
    """
    def __init__(self, base_path: str, num_workers: int = 2):
        print(f"--- Initializing Topographic Diversity Analysis for: {base_path} ---")
        self.num_workers = min(num_workers, cpu_count())
        self.files = glob.glob(os.path.join(base_path, "*.tif"))
        print(f"âœ… Found {len(self.files)} files. Will use {self.num_workers} worker processes.")

        if self.files:
            with rasterio.open(self.files[0]) as src:
                self.pixels_per_raster = src.width * src.height
        else:
            self.pixels_per_raster = 0

    def analyze_in_parallel(self) -> np.ndarray | None:
        """Orchestrates the parallel analysis by assigning one file per task."""
        if not self.files: return None
        
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_diversity_file_task, self.files), total=len(self.files), desc="Analyzing Diversity Files"))

        # Filter out any None results and concatenate all data into one giant array
        valid_results = [res for res in results if res is not None and res.size > 0]
        if not valid_results: return None
        
        print("\n[INFO] Aggregating results from all files...")
        overall_data = np.concatenate(valid_results)
        return overall_data

    def generate_formatted_report(self, data: np.ndarray):
        """Generates a detailed, professionally formatted text report."""
        if data.size == 0:
            print("[WARNING] Cannot generate report from empty data array.")
            return

        print("\n" + "="*80)
        print("SRTM TOPOGRAPHIC DIVERSITY ANALYSIS REPORT".center(80))
        
        # --- Detailed Analysis Section ---
        print("\n" + "="*80)
        print("Processing DIVERSITY layer...")
        print("DIVERSITY ANALYSIS".center(80))
        print("----------------------------------------".center(80))
        
        min_val, max_val = np.min(data), np.max(data)
        mean_val, median_val = np.mean(data), np.median(data)
        std_val = np.std(data)
        q1_val, q3_val = np.percentile(data, [25, 75])
        valid_pixels = data.size
        null_pixels = (self.pixels_per_raster * len(self.files)) - valid_pixels

        print(f"{'Minimum value:':<25} {min_val:.2f}")
        print(f"{'Maximum value:':<25} {max_val:.2f}")
        print(f"{'Mean value:':<25} {mean_val:.2f}")
        print(f"{'Median value:':<25} {median_val:.2f}")
        print(f"{'Standard deviation:':<25} {std_val:.2f}")
        print(f"{'First quartile (Q1):':<25} {q1_val:.2f}")
        print(f"{'Third quartile (Q3):':<25} {q3_val:.2f}")
        print(f"{'Valid pixels:':<25} {valid_pixels:,}")
        print(f"{'Null pixels:':<25} {null_pixels:,}")

        # --- Final Summary Table ---
        print("\n" + "="*80)
        print("SUMMARY COMPARISON".center(80))
        print("----------------------------------------".center(80))
        
        summary_data = {
            'Layer': ['DIVERSITY'],
            'Min': [min_val],
            'Max': [max_val],
            'Mean': [mean_val],
            'Std Dev': [std_val]
        }
        summary_df = pd.DataFrame(summary_data).set_index('Layer')
        print(summary_df.round(2).to_string(justify='right', header=True))
        print("="*80)

# --- Main Execution Block ---
if __name__ == "__main__":
    # The script is now hardcoded to process ONLY this specific directory
    base_data_path = "/kaggle/working/gee_amz_xs/Global_SRTM_Topographic_Diversity"

    if not os.path.exists(base_data_path) or not os.listdir(base_data_path):
        print(f"[ERROR] Directory not found or is empty: {base_data_path}")
        print("Please ensure the data has been downloaded correctly.")
    else:
        try:
            analyzer = SRTMDiversityAnalyzer(base_path=base_data_path, num_workers=2)
            if analyzer.files:
                # The analysis returns a single, massive NumPy array of all valid data
                overall_data = analyzer.analyze_in_parallel()
                
                if overall_data is not None and overall_data.size > 0:
                    # Generate the report from the aggregated data
                    analyzer.generate_formatted_report(overall_data)
                else:
                    print("[ERROR] Analysis completed, but no valid statistics were generated.")
            
            print("\nğŸ�‰ SRTM Topographic Diversity analysis completed successfully.")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] The analysis failed: {e}")


%%time
"""Graficar tiles en el mapa"""
import os
import folium
import rasterio
import numpy as np
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap

class SRTMTopographicDiversityVisualizer:
    def __init__(self, tif_directory, output_dir):
        """
        Visualizador mejorado de diversidad topogrÃ¡fica SRTM
        
        Args:
            tif_directory (str): Directorio con archivos .constant.tif
            output_dir (str): Directorio para guardar los mapas
        """
        self.tif_directory = tif_directory
        self.output_dir = output_dir
        self.tif_files = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._find_tif_files()
    
    def _find_tif_files(self):
        """Encuentra todos los archivos .constant.tif en el directorio"""
        for file in os.listdir(self.tif_directory):
            if file.endswith('.constant.tif'):
                self.tif_files.append(os.path.join(self.tif_directory, file))
    
    def get_bounds(self, tif_path):
        """Obtiene los lÃ­mites geogrÃ¡ficos del archivo"""
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        except Exception as e:
            print(f"Error leyendo bounds de {tif_path}: {str(e)}")
            return None
    
    def normalize_data(self, data):
        """Normaliza los datos usando percentiles para mejor visualizaciÃ³n"""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return np.zeros_like(data, dtype=np.float32)
        
        vmin, vmax = np.percentile(valid_data, [2, 98])  # Usar percentiles mÃ¡s conservadores
        if vmax - vmin <= 0:
            return np.zeros_like(data, dtype=np.float32)
        
        norm_data = (data - vmin) / (vmax - vmin)
        return np.clip(norm_data, 0, 1).astype(np.float32)
    
    def array_to_png(self, data):
        """Convierte array numpy a imagen PNG en formato RGBA"""
        # Aplicar colormap directamente
        colors = [
            [43, 131, 186, 255],  # Azul
            [171, 221, 164, 255],  # Verde claro
            [255, 255, 191, 255],  # Amarillo
            [253, 174, 97, 255],   # Naranja
            [215, 25, 28, 255]     # Rojo
        ]
        
        # Discretizar los valores normalizados (0-1) en 5 categorÃ­as
        classified = np.digitize(data, bins=[0.2, 0.4, 0.6, 0.8], right=False)
        
        # Crear imagen RGBA
        height, width = data.shape
        image = np.zeros((height, width, 4), dtype=np.uint8)
        
        for i in range(5):
            mask = (classified == i)
            image[mask] = colors[i]
        
        return image
    
    def create_folium_map(self, output_filename='SRTM_Map.html'):
        """
        Crea un mapa Folium con la diversidad topogrÃ¡fica
        
        Args:
            output_filename (str): Nombre del archivo de salida
        """
        if not self.tif_files:
            print("No se encontraron archivos .constant.tif")
            return None
        
        # Obtener centro del primer archivo
        first_bounds = self.get_bounds(self.tif_files[0])
        if first_bounds is None:
            print("No se pudieron obtener los lÃ­mites geogrÃ¡ficos")
            return None
            
        center_lat = (first_bounds[0][0] + first_bounds[1][0]) / 2
        center_lon = (first_bounds[0][1] + first_bounds[1][1]) / 2
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery',
            control_scale=True
        )
        
        # Crear y aÃ±adir colormap
        colormap = LinearColormap(
            colors=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'],
            index=[0, 0.25, 0.5, 0.75, 1],
            vmin=0,
            vmax=1,
            caption='Diversidad TopogrÃ¡fica (normalizada)'
        )
        colormap.add_to(m)
        
        # Procesar cada archivo
        for tif_file in self.tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    data = src.read(1)
                    bounds = self.get_bounds(tif_file)
                    if bounds is None:
                        continue
                    
                    # Normalizar datos
                    norm_data = self.normalize_data(data)
                    
                    # Convertir a imagen RGBA
                    image_data = self.array_to_png(norm_data)
                    
                    # Crear overlay directamente
                    overlay = ImageOverlay(
                        image=image_data,
                        bounds=bounds,
                        opacity=0.8,
                        interactive=True,
                        name=os.path.basename(tif_file),
                        origin='lower'
                    )
                    overlay.add_to(m)
                    
            except Exception as e:
                print(f"Error procesando {tif_file}: {str(e)}")
                continue
        
        # AÃ±adir control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa
        output_path = os.path.join(self.output_dir, output_filename)
        m.save(output_path)
        print(f"Mapa guardado en: {output_path}")
        return m

if __name__ == '__main__':
    # ConfiguraciÃ³n de rutas
    tif_directory = '/kaggle/working/gee_amz_xs/Global_SRTM_Topographic_Diversity'
    output_dir = '/kaggle/working/amazon_maps'
    
    # Crear y ejecutar visualizador
    visualizer = SRTMTopographicDiversityVisualizer(tif_directory, output_dir)
    srtm_map = visualizer.create_folium_map()


display(Image(filename='/kaggle/input/amaztest/my_imgs/SrtmMap.png'))


from IPython.display import Image, display

# Mostrar imagen desde ruta local
display(Image(filename='/kaggle/input/amaztest/LitRev/s1.png'))


%%time

from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
from collections import Counter


class SankeyVisualizer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.df = None
        self.all_nodes = set()
        self.links = []
        self.sources = []
        self.targets = []
        self.values = []
        self.node_colors = []
        self.link_colors = []
        self.all_nodes_list = []

    def parse_html_table(self):
        """Parsea la tabla HTML y crea un DataFrame"""
        soup = BeautifulSoup(self.html_content, 'html.parser')
        rows = soup.find_all('tr')[1:]  # Omitimos encabezado

        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 4:
                paper = cols[0].get_text(strip=True).replace('\xa0', '')
                methodology = cols[1].get_text(strip=True)
                techniques = cols[2].get_text(strip=True)
                approach = cols[3].get_text(strip=True)

                data.append({
                    "Paper": paper,
                    "Methodology": methodology,
                    "Techniques": techniques,
                    "Approach": approach
                })

        self.df = pd.DataFrame(data)

    def build_graph_structure(self):
        """Construye las tripletas source-target-value para Sankey"""
        for _, row in self.df.iterrows():
            paper = f"<b>{row['Paper']}</b>"
            method = f"<b>{row['Methodology']}</b>"

            self.links.append((paper, method, 1))
            self.all_nodes.add(paper)
            self.all_nodes.add(method)

            techniques = [f"<b>{t.strip()}</b>" for t in row['Techniques'].split(',')]

            for tech in techniques:
                self.links.append((method, tech, 1))
                self.all_nodes.add(tech)

        self.all_nodes_list = list(self.all_nodes)
        self.node_id = {node: i for i, node in enumerate(self.all_nodes_list)}

        self.sources = [self.node_id[source] for source, target, value in self.links]
        self.targets = [self.node_id[target] for source, target, value in self.links]
        self.values = [value for source, target, value in self.links]

    def assign_colors(self):
        """Asigna colores por tipo de nodo y transparencia a los enlaces"""

        def get_node_type(node):
            node_clean = node.replace("<b>", "").replace("</b>", "")
            if "[" in node_clean or "Pre-" in node_clean or "Urban" in node_clean:
                return "paper"
            elif "Method" in node_clean or "Modeling" in node_clean or "Learning" in node_clean \
                    or "Analysis" in node_clean or "Technology" in node_clean:
                return "methodology"
            else:
                return "technique"

        node_types = [get_node_type(node) for node in self.all_nodes_list]

        color_map = {
            "paper": "#FFA500",         # Naranja
            "methodology": "#4682B4",   # Azul acero
            "technique": "#32CD32"      # Verde lima
        }

        self.node_colors = [color_map[nt] for nt in node_types]

        # Colores con alpha para enlaces
        self.link_colors = []
        for source_idx, target_idx in zip(self.sources, self.targets):
            source_type = node_types[source_idx]
            base_color = color_map.get(source_type, "#888")
            rgba_color = base_color.replace(")", ", 0.7)").replace("rgb", "rgba")
            self.link_colors.append(rgba_color)

    def generate_sankey(self):
        """Genera y muestra el diagrama de Sankey"""
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=self.all_nodes_list,
                color=self.node_colors
            ),
            link=dict(
                source=self.sources,
                target=self.targets,
                value=self.values,
                color=self.link_colors
            )
        )])

        fig.update_layout(
            title_text="<b>Metodological Flow: Papers â†’ Methodologies â†’ Techniques</b>",
            font_size=12,
            height=800,
            width=1600,  # Ancho aumentado para mejor visibilidad
            hovermode='x'
        )

        fig.show()


if __name__ == "__main__":
    html = '''[<table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;">    <thead>        <tr style="background-color:#f2f2f2;">            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Main Model/Methodology</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Specific Algorithms/Techniques</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Key Methodological Approach</th>        </tr>    </thead>    <tbody>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian earth-builders settled... <a href="#cita4" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[4]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Statistical Modeling (MaxEnt)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">MaxEnt (Maximum Entropy Modeling).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting probable location of ADEs using environmental variables (soil, elevation, dist. to water, vegetation, climate). Spatial cross-validation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Urban Archaeology in the Lower Amazon... <a href="#cita5" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[5]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Traditional Archaeology & Geophysics (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Systematic/non-systematic excavations, GPR (Ground Penetrating Radar).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Exploring pre-colonial structures under urban areas. Artifact analysis, cultural/historical perspective, regional contextualization.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Geometry by Design: Contribution of Lidar... <a href="#cita6" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[6]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Visual & Geometric Analysis of LiDAR Data (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR acquisition (ALS for DTM), analysis of form, orientation, and distribution.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mapping and study of settlement patterns of mound villages. Comparison with previous records, validation with satellite imagery and fieldwork.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Using UAV-Based Lidar for Archaeological Prospection... <a href="#cita7" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[7]*</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">UAV-Based LiDAR Technology (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR on UAVs for high-res DTMs. Hillshade, Slope, Sky-View Factor.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mapping archaeological structures in SantarÃ©m region. Visual expert analysis, field validation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Lidar reveals pre-Hispanic low-density urbanism... <a href="#cita8" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[8]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Technology (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">ALS for DTM, Hillshade, Slope, Sky-View Factor.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mapping hidden structures in Llanos de Moxos. Expert analysis, validation with excavations and radiocarbon dating.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Fast computation of DTM anomalies... <a href="#cita10" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[10]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Image Processing & Topographic Analysis (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Multi-resolution DTM generation (.las/.laz), elevation normalization, DTM subtraction (similar to SLRM by Hesse 2010).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Detection of DTM anomalies to identify geoglyphs. Computationally optimized, avoiding convolutional filters.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting geographic distribution... <a href="#cita11" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[11]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Multi-class Machine Learning</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Neural Networks, XGBoost, SVM, Logistic Regression with Lasso.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting location of 3 site types (Earthworks, ADEs, Other) using geospatial data (climate, soil, dist. to rivers). Spatial cross-validation (blockCV in R).</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Contours of the Past: LiDAR Data... <a href="#cita12" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[12]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Data Analysis (No ML/DL)</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">ALS (Airborne Laser Scanning) for DTM, visualizations (Hillshade, Slope, Sky-View Factor).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mapping and analysis of pre-Columbian settlement remains. Visual interpretation by specialists, validation with excavations and artifact analysis.</p></td>        </tr>    </tbody></table>]'''  # Reemplaza esto por tu contenido HTML real

    visualizer = SankeyVisualizer(html)
    visualizer.parse_html_table()
    visualizer.build_graph_structure()
    visualizer.assign_colors()
    visualizer.generate_sankey()


display(Image(filename='/kaggle/input/amaztest/LitRev/f1.png'))


%%time

import re
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import plotly.express as px


class FactorAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_factors = []
        self.standardized_factors = []
        self.factor_counts = None
        self.df_factors = None

    def parse_html_table(self):
        """Parsea el HTML y extrae los 'Primary Factors Aiding Discovery'"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        rows = soup.find_all("tr")[1:]  # Omitimos encabezado

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                factors_text = cols[2].get_text(strip=True).lower()
                self.raw_factors.append(factors_text)

    def extract_discovery_factors(self, text):
        """Busca patrones clave en el texto de descubrimiento"""
        factors = []

        if re.search(r'\bwater\b|\briver\b|\bdistance to water\b', text):
            factors.append("Proximity to Water")
        if re.search(r'\bade\b|\bdark earth|fertile soil', text):
            factors.append("Amazonian Dark Earths (ADEs)")
        if re.search(r'lidar|laser scanning|airborne', text):
            factors.append("LiDAR Data Availability/Use")
        if re.search(r'cultural landscape|agriculture|cultivation', text):
            factors.append("Cultural Landscape/Agriculture")
        if re.search(r'topography|elevation|slope|relief|geo[-\s]context', text):
            factors.append("Favorable Topography/Geo-Context")
        if re.search(r'human-modified|earthworks|mounds|canals|design', text):
            factors.append("Human-Modified Topography/Geometry")
        if re.search(r'occupation|long-term use|settlement pattern', text):
            factors.append("Prolonged Occupation Indicators")
        if re.search(r'vegetation|forest|bamboo', text):
            factors.append("Vegetation Cover / Biomass")
        if re.search(r'satellite imagery|remote sensing', text):
            factors.append("Satellite Imagery Use")

        return factors

    def standardize_factors(self):
        """Extrae y normaliza los factores desde el texto de cada fila"""
        all_factors = []
        for text in self.raw_factors:
            factors = self.extract_discovery_factors(text)
            all_factors.extend(factors)
        self.standardized_factors = all_factors

    def build_dataframe(self):
        """Construye el DataFrame de frecuencias"""
        self.factor_counts = Counter(self.standardized_factors)
        self.df_factors = pd.DataFrame(self.factor_counts.items(),
                                       columns=["Factor", "Frequency"]) \
                             .sort_values("Frequency", ascending=False)

    def generate_bar_chart(self):
        """Genera y muestra el grÃ¡fico de barras"""
        fig = px.bar(
            self.df_factors,
            x="Frequency",
            y="Factor",
            orientation="h",
            title="<b>Frequency of Key Discovery Factors</b>",
            labels={
                "Frequency": "Number of Papers Mentioning Factor",
                "Factor": "Discovery Factor"
            },
            height=600,
            width=900
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            font=dict(size=12),
            title_x=0.5,
            margin=dict(l=200)
        )

        fig.show()

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        self.standardize_factors()
        self.build_dataframe()
        self.generate_bar_chart()


if __name__ == "__main__":
    # Reemplaza esto por tu contenido HTML real
    html = '''[<table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;">    <thead>        <tr style="background-color:#f2f2f2;">            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Archaeological Objects Found/Studied</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Primary Factors Aiding Discovery</th>        </tr>    </thead>    <tbody>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian earth-builders settled... <a href="#cita4" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[4]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Amazonian Dark Earths (ADEs).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Proximity to water, favorable topography (slightly elevated, non-flood prone), clayey and well-drained soils, historical agricultural activity, connection to cultural landscapes (geoglyphs, mounds).</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Urban Archaeology in the Lower Amazon... <a href="#cita5" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[5]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Large pre-colonial villages (artificial mounds, ceremonial plazas, earth systems).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography, modern urban context (sites under current constructions), relation to cultural landscapes (ritual practices), presence of ADEs, history of prolonged occupation (stratigraphic layers).</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Geometry by Design: Contribution of Lidar... <a href="#cita6" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[6]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Villages with circular/rectangular mounds, earth architecture.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography (deliberately built mounds), geographic context (near rivers), geometric design patterns (symmetric, repetitive), relation to cultural landscapes (raised fields, canals), history of prolonged occupation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Using UAV-Based Lidar for Archaeological Prospection... <a href="#cita7" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[7]*</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian archaeological structures (mounds, canals, terraces).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography, geographic context (near rivers), relation to cultural landscapes (ancient agricultural systems), history of prolonged occupation, presence of ADEs.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Lidar reveals pre-Hispanic low-density urbanism... <a href="#cita8" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[8]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Complex low-density urban structures (mounds, canals, raised causeways, ceremonial plazas).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography, geographic context (Llanos de Moxos, proximity to water, slightly elevated areas), spatial settlement patterns (connected sites), relation to cultural landscapes (ceremonial/symbolic functions), history of prolonged occupation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Fast computation of DTM anomalies... <a href="#cita10" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[10]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian geoglyphs.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography (geometric soil markings), specific geographic context (SW Brazilian Amazon, dry forest, uplands), high-resolution LiDAR data, local relief anomalies, prior archaeological validation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting geographic distribution... <a href="#cita11" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[11]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Earthworks, ADEs, Other sites.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Vegetation (semelparous bamboo for geoglyphs), proximity to water, soil characteristics (ADEs), topography (plains/moderate relief for earthworks), historical distribution patterns, cultivation/domestication areas.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Contours of the Past: LiDAR Data... <a href="#cita12" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[12]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian settlements (mounds, terraces, canals).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography (circular/elongated mounds, canals/trenches), relation to fertile soils (ADEs), organized spatial settlement patterns, elevated geographical context near major rivers, presence of ceramics.</p></td>        </tr>    </tbody></table>]'''  # Reemplaza esto por tu fragmento HTML real

    analyzer = FactorAnalyzer(html)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/LitRev/h1.png'))


%%time
import re
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px


class ConditionAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_data = []
        self.df_conditions = None
        self.treemap_data = None

    def parse_html_table(self):
        """Parsea el HTML y extrae Paper, Limiting y Favoring Conditions"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        table = soup.find("table")

        if not table:
            print("Error: No se encontrÃ³ ninguna tabla en el contenido HTML.")
            return

        rows = table.find_all("tr")[1:]  # Omitir fila de encabezados

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                paper_ref_tag = cols[0].find("a")
                paper_ref = paper_ref_tag.text.strip() if paper_ref_tag else "Unknown"

                limiting_text = cols[1].get_text(strip=True).lower()
                favoring_text = cols[2].get_text(strip=True).lower()

                self.raw_data.append({
                    "Paper": paper_ref,
                    "Limiting_Text": limiting_text,
                    "Favoring_Text": favoring_text
                })

    def extract_conditions(self, text, category):
        """
        Busca patrones clave en el texto de condiciones.
        Devuelve lista de tuplas (categorÃ­a, condiciÃ³n)
        """
        conditions = []

        if category == "Limiting":
            if re.search(r'vegetation|dense|forest', text):
                conditions.append("Dense Vegetation")
            if re.search(r'urban expansion|destruction|alteration', text):
                conditions.append("Urban Expansion")
            if re.search(r'natural erosion', text):
                conditions.append("Natural Erosion")
            if re.search(r'seasonal flooding', text):
                conditions.append("Seasonal Flooding")
            if re.search(r'restricted access|remote|dry season', text):
                conditions.append("Restricted Access")
            if re.search(r'incomplete data', text):
                conditions.append("Incomplete Data")
            if re.search(r'modern deforestation', text):
                conditions.append("Modern Deforestation")
            if re.search(r'complex relief|false positives', text):
                conditions.append("Complex Natural Relief")

        elif category == "Favoring":
            if re.search(r'lidar|airborne scanning', text):
                conditions.append("LiDAR Use")
            if re.search(r'gpr|ground penetrating radar', text):
                conditions.append("GPR Use")
            if re.search(r'ade|dark earths|fertile soil', text):
                conditions.append("Presence of ADEs")
            if re.search(r'geospatial data|climate|soil|elevation', text):
                conditions.append("Geospatial Data Available")
            if re.search(r'visible earthworks|clear signals', text):
                conditions.append("Visible Earthworks/ADEs")
            if re.search(r'prior historical data|archaeological records', text):
                conditions.append("Prior Archaeological Records")
            if re.search(r'favorable landscape|moderate relief', text):
                conditions.append("Favorable Landscape Features")
            if re.search(r'spatial cross-validation|blockcv', text):
                conditions.append("Spatial Cross-Validation")

        # Si no hay coincidencias pero hay texto, aÃ±adir 'Other'
        if not conditions and text.strip():
            conditions.append(f"Other {category} (see paper)")

        return [(category, cond) for cond in conditions]

    def build_condition_dataframe(self):
        """Construye el DataFrame con Paper, Category, Condition"""
        all_entries = []

        for item in self.raw_data:
            paper = item["Paper"]

            # Procesar Limiting Conditions
            limiting_conds = self.extract_conditions(item["Limiting_Text"], "Limiting")
            for cat, cond in limiting_conds:
                all_entries.append({"Paper": paper, "Category": cat, "Condition": cond})

            # Procesar Favoring Conditions
            favoring_conds = self.extract_conditions(item["Favoring_Text"], "Favoring")
            for cat, cond in favoring_conds:
                all_entries.append({"Paper": paper, "Category": cat, "Condition": cond})

        self.df_conditions = pd.DataFrame(all_entries)

        if self.df_conditions.empty:
            print("Advertencia: El DataFrame de condiciones estÃ¡ vacÃ­o.")

    def build_treemap_data(self):
        """Agrupa por CategorÃ­a y CondiciÃ³n para generar datos del treemap"""
        if self.df_conditions is None or self.df_conditions.empty:
            print("Error: DataFrame de condiciones vacÃ­o. No se puede generar datos para el treemap.")
            self.treemap_data = pd.DataFrame(columns=["Category", "Condition", "Count"])
            return

        # Contamos cuÃ¡ntas veces aparece cada condiciÃ³n
        self.treemap_data = (
            self.df_conditions.groupby(["Category", "Condition"])
            .size()
            .reset_index(name="Count")
        )

    def generate_treemap(self):
        """Genera y muestra el treemap interactivo usando Plotly"""
        if self.treemap_data is None or self.treemap_data.empty:
            print("Error: Datos para treemap vacÃ­os. No se puede generar el treemap.")
            return

        fig = px.treemap(
            self.treemap_data,
            path=[px.Constant("All Conditions"), "Category", "Condition"],
            values="Count",
            color="Category",
            color_discrete_map={"(?)": "lightgrey", "Limiting": "#EF553B", "Favoring": "#00CC96"},
            title="<b>Hierarchical View of Environmental & Technical Conditions</b> (by mention count across papers)",
            hover_name="Condition"
        )

        # Mostrar etiquetas: nombre + valor + porcentaje respecto al total
        fig.update_traces(textinfo="label+value+percent parent")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=700)

        fig.show()

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        self.build_condition_dataframe()
        self.build_treemap_data()
        self.generate_treemap()


if __name__ == "__main__":
    # Reemplaza esto con tu contenido HTML real o carga desde un archivo
    html_input = '''[<table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;">    <thead>        <tr style="background-color:#f2f2f2;">            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Limiting Conditions</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Favoring Conditions / Techniques</th>        </tr>    </thead>    <tbody>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian earth-builders settled... <a href="#cita4" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[4]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, natural erosion, incomplete data (fraction of sites known), regional variability.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Availability of environmental data, clear anthropogenic signals (ADEs), recent LiDAR use complementing MaxEnt, validity of MaxEnt model in similar contexts.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Urban Archaeology in the Lower Amazon... <a href="#cita5" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[5]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Modern urban expansion (destruction/alteration of sites), natural erosion (near rivers), restricted access (urban infrastructure).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Presence of ADEs, use of GPR (subsurface detection), prior historical/archaeological context, favorable climate/vegetation in some less dense/cleared areas.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Geometry by Design: Contribution of Lidar... <a href="#cita6" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[6]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation (precision loss), natural erosion/climatic changes, restricted access.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of LiDAR technology (reveals subtle details), favorable landscape features (slightly elevated, moderate relief), presence of clear cultural patterns, prior historical/archaeological data.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Using UAV-Based Lidar for Archaeological Prospection... <a href="#cita7" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[7]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, seasonal flooding, restricted access.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of UAV-LiDAR (flexibility, precision in small/difficult areas), favorable landscape (elevated, moderate relief), clear cultural patterns, prior data.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Lidar reveals pre-Hispanic low-density urbanism... <a href="#cita8" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[8]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation (reduced laser penetration), seasonal flooding, restricted access (remote areas, dry season only).</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of LiDAR technology, favorable landscape features (slightly elevated, moderate relief), clear cultural patterns (geometric design), prior historical/archaeological data.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Fast computation of DTM anomalies... <a href="#cita10" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[10]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation (can reduce LiDAR precision), complex natural relief (can cause false positives), restricted access.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Availability of high-resolution LiDAR, landscape characteristics (uplands, dry forests better preserve geoglyphs), high processing capabilities, prior archaeological experience/records.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting geographic distribution... <a href="#cita11" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[11]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense forests, difficult access, modern deforestation, landscape complexity, incomplete data.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Availability of geospatial data (climate, soil, elevation), existence of visible ADEs/earthworks, recent LiDAR use, persistent cultural patterns, spatial cross-validation (blockCV).</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Contours of the Past: LiDAR Data... <a href="#cita12" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[12]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, natural erosion, modern human activity altering landscape.</p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of LiDAR technology, existence of ADEs, favorable topographic features, documented cultural history, prior archaeological data.</p></td>        </tr>    </tbody></table>]'''  # AquÃ­ deberÃ­as insertar el HTML con la tabla Table 3

    analyzer = ConditionAnalyzer(html_input)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/LitRev/cloud1.png'))


!pip install keybert


%%time
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
import re


class ConclusionAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_conclusions = []
        self.full_text = ""
        self.keyphrases = []

    def parse_html_table(self):
        """Parsea la tabla HTML y extrae las conclusiones"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        rows = soup.find_all("tr")[1:]  # Omitimos encabezado

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                conclusion_text = cols[1].get_text(strip=True)
                self.raw_conclusions.append(conclusion_text)

        self.full_text = " ".join(self.raw_conclusions)

    def clean_text(self):
        """Limpia el texto para anÃ¡lisis"""
        text = self.full_text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Elimina puntuaciÃ³n
        return text

    def extract_keyphrases_with_bert(self, top_n=30, diversity=0.7):
        """Usa KeyBERT para extraer frases clave con mecanismo de atenciÃ³n"""
        kw_model = KeyBERT(model="distilbert-base-nli-mean-tokens")  # Modelo ligero pero efectivo
        keywords = kw_model.extract_keywords(
            self.full_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,  # Maximal Marginal Relevance (mejora diversidad)
            diversity=diversity
        )
        self.keyphrases = [kw[0] for kw in keywords]

    def generate_wordcloud_from_keyphrases(self):
        """Genera una nube de palabras con las frases clave extraÃ­das por BERT"""
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english").fit(self.keyphrases)
        bag_of_words = vectorizer.transform(self.keyphrases)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=set(STOPWORDS),
                              min_font_size=10).generate_from_frequencies(dict(words_freq))

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("BERT-Based Word Cloud of Key Contributions", fontsize=16)
        plt.tight_layout(pad=0)
        plt.savefig("cloud1.png")
        plt.show()

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        self.clean_text()
        self.extract_keyphrases_with_bert(top_n=30)
        self.generate_wordcloud_from_keyphrases()


if __name__ == "__main__":
    html = '''[<table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;">    <thead>        <tr style="background-color:#f2f2f2;">            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th>            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Brief Conclusion / Key Contribution</th>        </tr>    </thead>    <tbody>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Pre-Columbian earth-builders settled... <a href="#cita4" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[4]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">MaxEnt modeling can effectively predict ADE locations, indirect indicators of pre-Columbian occupation, offering a tool to guide research despite landscape and data challenges.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Urban Archaeology in the Lower Amazon... <a href="#cita5" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[5]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Despite urban expansion, a combination of archaeological excavation, geophysics, and contextual analysis can recover information on large pre-colonial villages, offering insights into past social complexity.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Geometry by Design: Contribution of Lidar... <a href="#cita6" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[6]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR is a powerful tool to reveal/analyze complex archaeological structures (mound villages) in Amazonia, highlighting sophisticated pre-Columbian settlement patterns.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Using UAV-Based Lidar for Archaeological Prospection... <a href="#cita7" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[7]*</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">UAV-mounted LiDAR combined with detailed archaeological analysis can reveal pre-Columbian structures in complex Amazonian environments, highlighting sophisticated societies.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Lidar reveals pre-Hispanic low-density urbanism... <a href="#cita8" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[8]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR reveals sophisticated, low-density urbanism in Llanos de Moxos, challenging previous notions of pre-Columbian societal complexity and adaptation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Fast computation of DTM anomalies... <a href="#cita10" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[10]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><P style="font-size: 1.2rem; line-height: 1.6;">Presents an innovative, computationally efficient method for geoglyph detection in Amazonia using only LiDAR and image processing, useful for subtle features in dense vegetation.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Predicting geographic distribution... <a href="#cita11" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[11]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Multi-class ML can predict archaeological site locations in Amazonia using environmental/geographic variables, guiding future research despite landscape complexity and data gaps.</p></td>        </tr>        <tr>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Contours of the Past: LiDAR Data... <a href="#cita12" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[12]</a></p></td>            <td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR significantly expands understanding of extent/complexity of pre-Columbian settlements in SantarÃ©m region, revealing features previously obscured by vegetation.</p></td>        </tr>    </tbody></table>]'''  # Reemplaza esto por tu contenido HTML real

    analyzer = ConclusionAnalyzer(html)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/LitRev/s2.png'))


%%time
import re
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

class SankeyVisualizer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.df = None
        self.all_nodes = set()
        self.links = []
        self.sources = []
        self.targets = []
        self.values = []
        self.node_colors = []
        self.link_colors = []
        self.all_nodes_list = []

    def parse_html_table(self):
        """Parsea la tabla HTML y crea un DataFrame"""
        soup = BeautifulSoup(self.html_content, 'html.parser')
        # Encuentra la tabla especÃ­fica (asumiendo que es la primera con estas caracterÃ­sticas)
        # O podrÃ­as buscarla por un h3 precedente si hay mÃºltiples tablas
        table_title = soup.find('h3', string="Table 1: Machine Learning / Deep Learning and Other Techniques Employed")
        if not table_title:
            print("Error: No se encontrÃ³ el tÃ­tulo 'Table 1...'")
            self.df = pd.DataFrame() # DataFrame vacÃ­o para evitar errores posteriores
            return
        
        table = table_title.find_next_sibling('table')
        if not table:
            print("Error: No se encontrÃ³ la tabla despuÃ©s del tÃ­tulo 'Table 1...'")
            self.df = pd.DataFrame()
            return

        rows = table.find('tbody').find_all('tr') # Buscamos en tbody, omitimos thead
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 4:
                # El texto estÃ¡ dentro de una etiqueta <p> en cada <td>
                paper_p = cols[0].find('p')
                paper = paper_p.get_text(strip=True).replace('\xa0', '') if paper_p else "N/A"
                
                methodology_p = cols[1].find('p')
                methodology = methodology_p.get_text(strip=True) if methodology_p else "N/A"
                
                techniques_p = cols[2].find('p')
                techniques = techniques_p.get_text(strip=True) if techniques_p else "N/A"
                
                approach_p = cols[3].find('p')
                approach = approach_p.get_text(strip=True) if approach_p else "N/A"
                
                data.append({
                    "Paper": paper,
                    "Methodology": methodology,
                    "Techniques": techniques,
                    "Approach": approach # Aunque no se usa en el Sankey actual, lo parseamos
                })
        self.df = pd.DataFrame(data)
        if self.df.empty:
            print("Advertencia: El DataFrame creado para SankeyVisualizer estÃ¡ vacÃ­o.")

    def build_graph_structure(self):
        """Construye las tripletas source-target-value para Sankey"""
        if self.df is None or self.df.empty:
            print("Error: DataFrame no inicializado o vacÃ­o. No se puede construir la estructura del grafo.")
            return

        for _, row in self.df.iterrows():
            # Usamos el texto completo del paper como nodo, incluyendo la referencia [xx]
            paper_node = f"<b>{row['Paper']}</b>" 
            method_node = f"<b>{row['Methodology']}</b>"
            
            self.links.append((paper_node, method_node, 1))
            self.all_nodes.add(paper_node)
            self.all_nodes.add(method_node)
            
            techniques_list = [f"<b>{t.strip()}</b>" for t in row['Techniques'].split(',') if t.strip()]
            for tech_node in techniques_list:
                self.links.append((method_node, tech_node, 1))
                self.all_nodes.add(tech_node)
                
        self.all_nodes_list = list(self.all_nodes)
        if not self.all_nodes_list:
            print("Advertencia: No se generaron nodos para el Sankey.")
            return

        self.node_id = {node: i for i, node in enumerate(self.all_nodes_list)}
        self.sources = [self.node_id[source] for source, target, value in self.links]
        self.targets = [self.node_id[target] for source, target, value in self.links]
        self.values = [value for source, target, value in self.links]

    def assign_colors(self):
        """Asigna colores por tipo de nodo y transparencia a los enlaces"""
        if not self.all_nodes_list:
            print("Advertencia: No hay nodos para asignar colores.")
            self.node_colors = []
            self.link_colors = []
            return

        def get_node_type(node):
            node_clean = node.replace("<b>", "").replace("</b>", "")
            # Si el nodo contiene "[...]" al final, es probable que sea un paper
            if re.search(r'\[\d+\]$', node_clean): 
                return "paper"
            # Palabras clave para metodologÃ­a
            elif any(keyword in node_clean for keyword in ["Learning", "Analysis", "Modeling", "Approach", "Sensing", "Decomposition", "Methodology"]):
                return "methodology"
            # El resto se considera tÃ©cnica
            else:
                return "technique"

        node_types = [get_node_type(node) for node in self.all_nodes_list]
        
        color_map = {
            "paper": "rgb(255, 165, 0)",    # Naranja
            "methodology": "rgb(70, 130, 180)", # Azul acero
            "technique": "rgb(50, 205, 50)"     # Verde lima
        }
        self.node_colors = [color_map.get(nt, "rgb(136, 136, 136)") for nt in node_types] # Gris por defecto

        # Colores con alpha para enlaces
        self.link_colors = []
        for source_idx, target_idx in zip(self.sources, self.targets):
            source_node_type = node_types[source_idx] # Usar tipo del nodo origen para el color del enlace
            base_color = color_map.get(source_node_type, "rgb(136,136,136)") # Gris por defecto
            # Convertir rgb(r,g,b) a rgba(r,g,b,a)
            rgba_color = base_color.replace("rgb", "rgba").replace(")", ", 0.6)") 
            self.link_colors.append(rgba_color)

    def generate_sankey(self):
        """Genera y muestra el diagrama de Sankey"""
        if not self.sources or not self.targets or not self.values:
            print("Error: Faltan datos (sources, targets o values) para generar el Sankey.")
            return

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=self.all_nodes_list,
                color=self.node_colors
            ),
            link=dict(
                source=self.sources,
                target=self.targets,
                value=self.values,
                color=self.link_colors
            )
        )])
        fig.update_layout(
            title_text="<b>Methodological Flow: Papers â†’ Methodologies â†’ Techniques</b>",
            font_size=12,
            height=1000, # Aumentado para mÃ¡s espacio vertical si hay muchos nodos
            width=1600,  # Ancho aumentado para mejor visibilidad
            hovermode='x'
        )
        fig.show()

    def run_visualization(self):
        self.parse_html_table()
        if self.df is not None and not self.df.empty:
            self.build_graph_structure()
            self.assign_colors()
            self.generate_sankey()
        else:
            print("Proceso de visualizaciÃ³n Sankey detenido debido a un DataFrame vacÃ­o o no inicializado.")

if __name__ == "__main__":
    # Reemplaza esto por tu contenido HTML real de la Table 1
    html_table1 = ''' 
    [<h3 style="color:#71b12c;font-size: 2.2rem;">Table 1: Machine Learning / Deep Learning and Other Techniques Employed</h3><table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;"><thead><tr style="background-color:#f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Main Model/Methodology</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Specific Algorithms/Techniques</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Key Methodological Approach</th></tr></thead><tbody><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The lowland Maya settlement landscape: Environmental LiDAR and ecology <a href="#cita13" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[13]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Data Analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Digital Terrain Models (DTM), Hillshade, Slope</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Visual and digital image processing analysis to identify archaeological structures in lowland Maya landscapes.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Combined Detection and Segmentation of Archeological Structures from LiDAR Data Using a Deep Learning Approach <a href="#cita14" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[14]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mask R-CNN</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Detection and segmentation of archaeological structures from LiDAR data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era <a href="#cita15" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[15]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">U-Net, DeepLabv3, Vision Transformers (ViTs), HybViT</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Semantic segmentation of LiDAR data to detect archaeological structures in complex landscapes.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Ancient lowland Maya complexity as revealed by airborne laser scanning of northern Guatemala <a href="#cita16" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[16]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Data Analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Red Relief Image Map (RRIM), Sky-View Factor (SVF), Simple Local Relief Model (SLRM), Prismatic Openness</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Advanced visualization of LiDAR data and manual archaeological interpretation.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Beyond Visualization: Remote Sensing Applications in Prehispanic Settlements to Understand Ancient Anthropogenic Land Use and Occupation in the Sierra Nevada de Santa Marta, Colombia <a href="#cita17" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[17]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Remote Sensing</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Photogrammetry, Terrestrial Laser Scanning (TLS)</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Mapping and analysis of anthropogenic landscapes at a local scale.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The domestication of Amazonia before European conquest <a href="#cita18" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[18]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Interdisciplinary Approach</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeological mapping, analysis of anthropogenic soils, current vegetation data, Geographic Information Systems (GIS)</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Integration of archaeological, ecological, geomorphological, and remote sensing data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning for Archaeological Object Detection on LiDAR: New Evaluation Measures and Insights <a href="#cita19" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[19]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Faster R-CNN</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Automated detection of archaeological objects in LiDAR data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Caracol, Belize, and Changing Perceptions of Ancient Maya Society <a href="#cita20" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[20]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Data Analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR, Geographic Information Systems (GIS), geochemical analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Automated processing of spatial data to identify archaeological patterns or features.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Unraveling pre-Columbian occupation patterns in the tropical forests of French Guiana using an anthracological approach <a href="#cita21" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[21]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Anthracological Approach</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Collection of charcoal samples, radiocarbon dating, taxonomic identification of charcoal, analysis of anthropogenic soils</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Analysis of charcoal to investigate human presence and impact in tropical forests.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Wetland Farming and the Early Anthropocene: Globally Upscaling from the Maya Lowlands with LiDAR and Multiproxy Verification <a href="#cita22" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[22]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">LiDAR Data Analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Aerial LiDAR, multiproxy analysis, Geographic Information Systems (GIS), radiocarbon dating</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Integration of paleoenvironmental, geochemical, archaeological, and sedimentological data to validate the age and function of identified agricultural systems.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">TensorMap: LIDAR-Based Topological Mapping and Localization via Tensor Decompositions <a href="#cita35" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[35]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Tensor Decomposition</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Tucker Decomposition</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topological mapping and localization based on pattern similarity.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Exploring Topological Information Beyond Persistent Homology to Detect Geospatial Objects <a href="#cita36" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[36]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topological Analysis</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Persistent Homology (PH), geometric and contextual information</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Integration of topological, geometric, and contextual information to detect geospatial objects.</p></td></tr></tbody></table>]
    ''' 
    visualizer = SankeyVisualizer(html_table1)
    visualizer.run_visualization()


display(Image(filename='/kaggle/input/amaztest/LitRev/f2.png'))


%%time
import re
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import plotly.express as px

class FactorAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_factors_texts = [] # Cambiado para reflejar que es el texto completo
        self.standardized_factors = []
        self.factor_counts = None
        self.df_factors = None

    def parse_html_table(self):
        """Parsea el HTML y extrae los 'Primary Factors Aiding Discovery' de la Table 2"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        table_title = soup.find('h3', string="Table 2: Key Factors for the Discovery of Archaeological Objects")
        if not table_title:
            print("Error: No se encontrÃ³ el tÃ­tulo 'Table 2...'")
            return
        
        table = table_title.find_next_sibling('table')
        if not table:
            print("Error: No se encontrÃ³ la tabla despuÃ©s del tÃ­tulo 'Table 2...'")
            return

        rows = table.find('tbody').find_all('tr')
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3: # La tercera columna (Ã­ndice 2) contiene los factores
                factors_p = cols[2].find('p')
                if factors_p:
                    factors_text = factors_p.get_text(strip=True).lower()
                    self.raw_factors_texts.append(factors_text)
                else:
                    self.raw_factors_texts.append("") # AÃ±adir string vacÃ­o si no hay <p>

    def extract_discovery_factors(self, text):
        """Busca patrones clave en el texto de descubrimiento"""
        factors = []
        # Se mantienen tus patrones regex, podrÃ­an necesitar ajustes si el texto fuente cambia mucho
        if re.search(r'\bwater\b|\briver\b|\bdistance to water\b|hydrological', text):
            factors.append("Proximity to Water / Hydrology")
        if re.search(r'\bade\b|\bdark earth|fertile soil|anthropogenic soils', text): # AÃ±adido 'anthropogenic soils'
            factors.append("Amazonian Dark Earths (ADEs) / Anthrosols")
        if re.search(r'lidar|laser scanning|airborne|dtm', text): # AÃ±adido dtm
            factors.append("LiDAR Data Availability/Use (DTM)")
        if re.search(r'cultural landscape|agriculture|cultivation|land use', text): # AÃ±adido 'land use'
            factors.append("Cultural Landscape/Agriculture/Land Use")
        if re.search(r'topography|elevation|slope|relief|geomorphological|geo[-\s]context', text): # AÃ±adido geomorphological
            factors.append("Favorable Topography/Geo-Context")
        if re.search(r'human-modified|earthworks|mounds|canals|design|settlement patterns|spatial patterns', text): # AÃ±adido settlement/spatial patterns
            factors.append("Human-Modified Topography/Geometry/Patterns")
        if re.search(r'occupation|long-term use|settlement pattern|historical context', text): # AÃ±adido historical context
            factors.append("Prolonged Occupation/Historical Indicators")
        if re.search(r'vegetation|forest|bamboo|ecological context', text): # AÃ±adido ecological context
            factors.append("Vegetation Cover / Biomass / Ecology")
        if re.search(r'satellite imagery|remote sensing', text):
            factors.append("Satellite Imagery / Remote Sensing Use")
        if re.search(r'gis|geographic information system', text):
            factors.append("GIS Use / Spatial Analysis")
        if re.search(r'multidisciplinary|integration of multiple lines|multiple lines of evidence', text):
            factors.append("Multidisciplinary Approach / Data Integration")
        if not factors and text.strip(): # Si no se encontrÃ³ nada pero hay texto, aÃ±adir "Other specific factors"
            factors.append("Other Specific Factors Detailed in Paper")
        
        return factors

    def standardize_factors(self):
        """Extrae y normaliza los factores desde el texto de cada fila"""
        all_factors = []
        for text in self.raw_factors_texts:
            extracted = self.extract_discovery_factors(text)
            all_factors.extend(extracted)
        self.standardized_factors = all_factors

    def build_dataframe(self):
        """Construye el DataFrame de frecuencias"""
        if not self.standardized_factors:
            print("Advertencia: No se estandarizaron factores. El DataFrame estarÃ¡ vacÃ­o.")
            self.df_factors = pd.DataFrame(columns=["Factor", "Frequency"])
            return
            
        self.factor_counts = Counter(self.standardized_factors)
        self.df_factors = pd.DataFrame(self.factor_counts.items(),
                                       columns=["Factor", "Frequency"]) \
                             .sort_values("Frequency", ascending=False)

    def generate_bar_chart(self):
        """Genera y muestra el grÃ¡fico de barras"""
        if self.df_factors is None or self.df_factors.empty:
            print("Error: DataFrame de factores vacÃ­o o no generado. No se puede crear el grÃ¡fico de barras.")
            return

        fig = px.bar(
            self.df_factors,
            x="Frequency",
            y="Factor",
            orientation="h",
            title="<b>Frequency of Key Discovery Factors</b>",
            labels={
                "Frequency": "Number of Papers Mentioning Factor",
                "Factor": "Discovery Factor"
            },
            height=max(600, len(self.df_factors["Factor"]) * 30), # Altura dinÃ¡mica
            width=1000 # Aumentado para mejor legibilidad de etiquetas largas
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            font=dict(size=12),
            title_x=0.5,
            margin=dict(l=350, r=50, t=50, b=50) # Aumentado margen izquierdo
        )
        fig.show()

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        self.standardize_factors()
        self.build_dataframe()
        self.generate_bar_chart()

if __name__ == "__main__":
    # Reemplaza esto por tu contenido HTML real de la Table 2
    html_table2 = '''
    [<h3 style="color:#71b12c;font-size: 2.2rem;">Table 2: Key Factors for the Discovery of Archaeological Objects</h3><table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;"><thead><tr style="background-color:#f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Archaeological Objects Found/Studied</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Primary Factors Aiding Discovery</th></tr></thead><tbody><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The lowland Maya settlement landscape: Environmental LiDAR and ecology <a href="#cita13" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[13]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Maya archaeological structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Human-modified topography, relationship with natural resources, spatial settlement patterns, prolonged occupation history, ecological context.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Combined Detection and Segmentation of Archeological Structures from LiDAR Data Using a Deep Learning Approach <a href="#cita14" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[14]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeological structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Quality of the Digital Terrain Model (DTM) derived from LiDAR, high spatial resolution, use of multiple relief representations, supervised training, GIS-based semi-automatic approach.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era <a href="#cita15" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[15]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeological structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Subtle topographic patterns, broad spatial context, classes manually defined by experts, combination of RGB + nDTM data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Ancient lowland Maya complexity as revealed by airborne laser scanning of northern Guatemala <a href="#cita16" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[16]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Maya archaeological structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Quality of the Digital Terrain Model (DTM) derived from LiDAR, hydrological and geomorphological context, integration of multiple lines of evidence, scalability of results.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Beyond Visualization: Remote Sensing Applications in Prehispanic Settlements to Understand Ancient Anthropogenic Land Use and Occupation in the Sierra Nevada de Santa Marta, Colombia <a href="#cita17" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[17]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeological structures and landscape modifications</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Integration of multiple spatial variables, analysis scales, multidisciplinary approach, use of high-precision technologies.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The domestication of Amazonia before European conquest <a href="#cita18" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[18]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Anthropogenic soils, domesticated plant species, artificial structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Concentrations of anthropogenic soils, distribution patterns of domesticated plant species, artificial landscape structures, relationship between hydrology and human occupation, archaeological research history.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning for Archaeological Object Detection on LiDAR: New Evaluation Measures and Insights <a href="#cita19" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[19]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Funeral mounds, Celtic fields</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topographic and morphological features visible in LiDAR-derived images, visual patterns consistent with known archaeological structures, precise segmentation of objects.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Caracol, Belize, and Changing Perceptions of Ancient Maya Society <a href="#cita20" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[20]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Maya archaeological structures</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Spatial and cosmological context, functional and domestic context, environmental and landscape context, historical and political context.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Unraveling pre-Columbian occupation patterns in the tropical forests of French Guiana using an anthracological approach <a href="#cita21" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[21]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Occupation patterns and use of fire</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Spatial distribution of archaeological sites, composition of charcoal, presence of anthropogenic soils and artifacts, ecological and vegetation context, radiometric dating.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Wetland Farming and the Early Anthropocene: Globally Upscaling from the Maya Lowlands with LiDAR and Multiproxy Verification <a href="#cita22" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[22]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Wetland farming systems</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Quality of the Digital Terrain Model (DTM) derived from LiDAR, hydrological and geomorphological context, integration of multiple lines of evidence, scalability of results.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">TensorMap: LIDAR-Based Topological Mapping and Localization via Tensor Decompositions <a href="#cita35" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[35]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topological mapping of the environment</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topological representation using tensors, Tucker decomposition, localization based on pattern similarity.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Exploring Topological Information Beyond Persistent Homology to Detect Geospatial Objects <a href="#cita36" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[36]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Landslides</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Topological factors, geometric factors, contextual factors, processing conditions.</p></td></tr></tbody></table>]
    '''
    analyzer = FactorAnalyzer(html_table2)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/LitRev/h2.png'))


%%time
import re
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import plotly.express as px

class ConditionAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_data = []
        self.df_conditions = None
        self.treemap_data = None

    def parse_html_table(self):
        """Parsea el HTML y extrae Paper, Limiting y Favoring Conditions de la Table 3"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        table_title = soup.find('h3', string="Table 3: Environmental and Technical Conditions Limiting or Favoring Studies")
        if not table_title:
            print("Error: No se encontrÃ³ el tÃ­tulo 'Table 3...'")
            return
        
        table = table_title.find_next_sibling('table')
        if not table:
            print("Error: No se encontrÃ³ la tabla despuÃ©s del tÃ­tulo 'Table 3...'")
            return
            
        rows = table.find('tbody').find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                paper_p = cols[0].find("p")
                # Extraer solo el texto de la referencia [xx] del tag <a> dentro de <p>
                paper_ref_tag = paper_p.find("a") if paper_p else None
                paper_ref = paper_ref_tag.text.strip() if paper_ref_tag else "Unknown Ref"
                
                limiting_p = cols[1].find('p')
                limiting_text = limiting_p.get_text(strip=True).lower() if limiting_p else ""
                
                favoring_p = cols[2].find('p')
                favoring_text = favoring_p.get_text(strip=True).lower() if favoring_p else ""
                
                self.raw_data.append({
                    "Paper_Ref": paper_ref, # Cambiado a Paper_Ref para claridad
                    "Limiting_Text": limiting_text,
                    "Favoring_Text": favoring_text
                })

    def extract_conditions(self, text, category):
        """Busca patrones clave en el texto de condiciones"""
        conditions = []
        # Tus patrones regex se mantienen, pueden necesitar ajustes
        if category == "Limiting":
            if re.search(r'vegetation|dense|forest|canopy', text):
                conditions.append("Dense Vegetation/Canopy")
            if re.search(r'urban expansion|destruction|alteration|modern deforestation', text): # Agrupado
                conditions.append("Site Alteration (Urban/Deforestation)")
            if re.search(r'natural erosion|degradation', text): # AÃ±adido degradation
                conditions.append("Natural Erosion/Degradation")
            if re.search(r'seasonal flooding|humid environments', text): # AÃ±adido humid
                conditions.append("Seasonal Flooding/Humidity")
            if re.search(r'restricted access|remote|dry season|difficult access|weather conditions', text): # Agrupado
                conditions.append("Restricted/Difficult Access")
            if re.search(r'incomplete data|scarcity of training data|secondary data|lack of representative data|limitations in coverage', text): # Agrupado
                conditions.append("Data Limitations (Incomplete/Scarce)")
            if re.search(r'complex relief|false positives|complex landscapes|ambiguity', text): # Agrupado
                conditions.append("Complex Natural Relief/Ambiguity")
            if re.search(r'computational complexity|intensive processing', text):
                conditions.append("Computational/Processing Demands")
            if re.search(r'subjective interpretation|dependence on manual|model scalability', text):
                conditions.append("Methodological Limitations (Subjectivity/Scalability)")


        elif category == "Favoring":
            if re.search(r'lidar|airborne scanning|dtm', text): # AÃ±adido dtm
                conditions.append("LiDAR/DTM Use")
            if re.search(r'gpr|ground penetrating radar', text):
                conditions.append("GPR Use")
            if re.search(r'ade|dark earths|fertile soil|anthropogenic soils', text): # AÃ±adido anthropogenic soils
                conditions.append("Presence of ADEs/Anthrosols")
            if re.search(r'geospatial data|gis|remote sensing|public data', text): # Agrupado
                conditions.append("Geospatial Data/GIS/RS Availability")
            if re.search(r'visible earthworks|clear signals|well-defined characteristics', text): # Agrupado
                conditions.append("Clear Archaeological Signatures")
            if re.search(r'prior historical data|archaeological records|previous studies', text): # Agrupado
                conditions.append("Prior Archaeological/Historical Data")
            if re.search(r'favorable landscape|moderate relief|homogeneous landscape', text): # Agrupado
                conditions.append("Favorable Landscape Features")
            if re.search(r'spatial cross-validation|blockcv|transfer learning|hybrid models|deep learning', text): # Agrupado
                conditions.append("Advanced ML/DL Techniques")
            if re.search(r'interdisciplinary|multidisciplinary', text):
                conditions.append("Interdisciplinary Collaboration")
            if re.search(r'high-resolution data|wide coverage', text):
                conditions.append("High-Quality/Extensive Data")

        # Si no se encontrÃ³ ninguna condiciÃ³n especÃ­fica pero hay texto, aÃ±adirlo como "Other"
        if not conditions and text.strip():
            conditions.append(f"Other {category} (see paper)")
            
        return [(category, cond) for cond in conditions]

    def build_condition_dataframe(self):
        """Construye el DataFrame con Paper_Ref, Category, Condition"""
        all_entries = []
        for item in self.raw_data:
            paper_ref = item["Paper_Ref"]
            
            limiting_conds = self.extract_conditions(item["Limiting_Text"], "Limiting")
            for cat, cond in limiting_conds:
                all_entries.append({"Paper_Ref": paper_ref, "Category": cat, "Condition": cond})
            
            favoring_conds = self.extract_conditions(item["Favoring_Text"], "Favoring")
            for cat, cond in favoring_conds:
                all_entries.append({"Paper_Ref": paper_ref, "Category": cat, "Condition": cond})
        
        self.df_conditions = pd.DataFrame(all_entries)
        if self.df_conditions.empty:
            print("Advertencia: DataFrame de condiciones estÃ¡ vacÃ­o.")


    def build_treemap_data(self):
        """Agrupa por CategorÃ­a y CondiciÃ³n para treemap"""
        if self.df_conditions is None or self.df_conditions.empty:
            print("Error: DataFrame de condiciones vacÃ­o. No se puede generar data para treemap.")
            self.treemap_data = pd.DataFrame(columns=["Category", "Condition", "Count"])
            return
        self.treemap_data = self.df_conditions.groupby(["Category", "Condition"]) \
                                              .size().reset_index(name="Count")

    def generate_treemap(self):
        """Genera y muestra el treemap interactivo"""
        if self.treemap_data is None or self.treemap_data.empty:
            print("Error: Datos para treemap vacÃ­os. No se puede generar el treemap.")
            return

        fig = px.treemap(
            self.treemap_data,
            path=[px.Constant("All Conditions"), "Category", "Condition"],
            values="Count",
            color="Category",
            color_discrete_map={"(?)": "lightgrey", "Limiting": "#EF553B", "Favoring": "#00CC96"}, # Rojo para Limiting, Verde para Favoring
            title="<b>Hierarchical View of Environmental & Technical Conditions</b> (by mention count)",
            hover_name="Condition"
        )
        fig.update_traces(textinfo="label+value+percent parent") # AÃ±adido % del padre
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=700)
        fig.show()

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        self.build_condition_dataframe()
        self.build_treemap_data()
        self.generate_treemap()

if __name__ == "__main__":
    # Reemplaza esto por tu contenido HTML real de la Table 3
    html_table3 = '''
    [<h3 style="color:#71b12c;font-size: 2.2rem;">Table 3: Environmental and Technical Conditions Limiting or Favoring Studies</h3><table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;"><thead><tr style="background-color:#f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Limiting Conditions</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Favoring Conditions / Techniques</th></tr></thead><tbody><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The lowland Maya settlement landscape: Environmental LiDAR and ecology <a href="#cita13" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[13]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, steep slopes, restricted access, secondary data.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of LiDAR technology, characteristics of the Maya cultural landscape, availability of public data, previous historical and archaeological context.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Combined Detection and Segmentation of Archeological Structures from LiDAR Data Using a Deep Learning Approach <a href="#cita14" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[14]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Scarcity of training data, complexity of natural and anthropic landscapes, dependence on the type of structure.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Use of airborne LiDAR, study region with rich megalithic heritage, eMSTP visualization.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era <a href="#cita15" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[15]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, variability of natural relief, model scalability, ambiguity in annotations.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">High-resolution LiDAR data, wide geographic coverage, prior archaeological experience, use of hybrid models (CNN + Transformer).</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Ancient lowland Maya complexity as revealed by airborne laser scanning of northern Guatemala <a href="#cita16" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[16]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Subjective interpretation of data, limitations in total area coverage, no predictive modeling or machine learning techniques applied, variable preservation of archaeological remains.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Relatively homogeneous vegetation cover, preservation of anthropic landscapes, relative accessibility of certain study areas, availability of historical data and previous studies.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Beyond Visualization: Remote Sensing Applications in Prehispanic Settlements to Understand Ancient Anthropogenic Land Use and Occupation in the Sierra Nevada de Santa Marta, Colombia <a href="#cita17" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[17]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Incomplete coverage of the archaeological site, lack of representative vegetation and soil data, difficulty of access and weather conditions, current methodological limitations.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Accessible complex topography, relative preservation of the cultural landscape, availability of modern technologies, interdisciplinary collaboration.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The domestication of Amazonia before European conquest <a href="#cita18" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[18]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Low coverage of archaeological data in certain regions, impact of modern deforestation, complexity of the Amazonian landscape, lack of precise chronologies.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Preservation of anthropogenic soils, ecological diversity and presence of long-lived trees, access to previous archaeological records, use of GIS and remote sensing.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning for Archaeological Object Detection on LiDAR: New Evaluation Measures and Insights <a href="#cita19" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[19]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Complexity of some archaeological landscapes, dependence on manually labeled data, variability in the morphology of archaeological objects, limitations in spatial generalization.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Terrains with well-defined characteristics, availability of high-resolution LiDAR and DTM data, landscape homogeneity, combined use of diverse information.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Caracol, Belize, and Changing Perceptions of Ancient Maya Society <a href="#cita20" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[20]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation and tropical relief, degradation of organic materials, past climate changes.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Preservation of stone structures, use of LiDAR, preservation of chemical and ecological records.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Unraveling pre-Columbian occupation patterns in the tropical forests of French Guiana using an anthracological approach <a href="#cita21" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[21]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Natural degradation of charcoal in humid environments, lack of continuous chronologies, limitations in sample representativeness, indirect interpretation of human occupation.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Preservation of charcoal in acidic soils, ecological variability of the region, presence of modified soils and artifacts, access to specialized laboratories.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Wetland Farming and the Early Anthropocene: Globally Upscaling from the Maya Lowlands with LiDAR and Multiproxy Verification <a href="#cita22" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[22]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation and microtopographic changes, intensive LiDAR data processing, complexity of multiproxy interpretation, lack of complete coverage in certain areas.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Preservation of anthropic landscapes in wetland areas, relative accessibility of certain study areas, low modern intervention in some regions, availability of historical data and previous studies.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">TensorMap: LIDAR-Based Topological Mapping and Localization via Tensor Decompositions <a href="#cita35" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[35]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Noise in LiDAR data, dynamic scenarios, very similar or repetitive spaces.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Structured environments or with defined geometry, stability of the scenario, high-resolution data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Exploring Topological Information Beyond Persistent Homology to Detect Geospatial Objects <a href="#cita36" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[36]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Dense vegetation, quality and resolution of the DTM, dependence on manually defined thresholds, computational complexity.</p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Terrains with well-defined characteristics, availability of high-resolution LiDAR and DTM data, landscape homogeneity, combined use of diverse information.</p></td></tr></tbody></table>]
    '''
    analyzer = ConditionAnalyzer(html_table3)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/LitRev/cloud2.png'))


%%time
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# from sklearn.feature_extraction.text import CountVectorizer # No se usa si generamos desde kw[0] y kw[1]
from keybert import KeyBERT
import re

class ConclusionAnalyzer:
    def __init__(self, html_content):
        self.html_content = html_content
        self.raw_conclusions = []
        self.full_text = ""
        self.keyphrases_with_scores = [] # AlmacenarÃ¡ tuplas (frase, score)

    def parse_html_table(self):
        """Parsea la tabla HTML y extrae las conclusiones de la Table 4"""
        soup = BeautifulSoup(self.html_content, "html.parser")
        table_title = soup.find('h3', string="Table 4: Brief Conclusions or Key Contributions")
        if not table_title:
            print("Error: No se encontrÃ³ el tÃ­tulo 'Table 4...'")
            return
        
        table = table_title.find_next_sibling('table')
        if not table:
            print("Error: No se encontrÃ³ la tabla despuÃ©s del tÃ­tulo 'Table 4...'")
            return

        rows = table.find('tbody').find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2: # La segunda columna (Ã­ndice 1) tiene la conclusiÃ³n
                conclusion_p = cols[1].find('p')
                if conclusion_p:
                    conclusion_text = conclusion_p.get_text(strip=True)
                    self.raw_conclusions.append(conclusion_text)
        self.full_text = " ".join(self.raw_conclusions)
        if not self.full_text.strip():
            print("Advertencia: No se extrajeron conclusiones o el texto estÃ¡ vacÃ­o.")


    def clean_text_for_bert(self):
        """Limpia el texto para anÃ¡lisis BERT (mÃ­nima limpieza)"""
        # KeyBERT maneja bien la puntuaciÃ³n y mayÃºsculas/minÃºsculas,
        # por lo que una limpieza agresiva no siempre es necesaria.
        # PodrÃ­amos quitar URLs o caracteres muy especÃ­ficos si fuera un problema.
        text = self.full_text
        text = re.sub(r'\s+', ' ', text).strip() # Normalizar espacios mÃºltiples
        return text

    def extract_keyphrases_with_bert(self, top_n=30, diversity=0.7):
        """Usa KeyBERT para extraer frases clave con mecanismo de atenciÃ³n"""
        if not self.full_text.strip():
            print("Error: El texto para KeyBERT estÃ¡ vacÃ­o.")
            self.keyphrases_with_scores = []
            return

        cleaned_text = self.clean_text_for_bert()
        if not cleaned_text:
            print("Error: El texto limpiado para KeyBERT estÃ¡ vacÃ­o.")
            self.keyphrases_with_scores = []
            return
            
        # Puedes probar otros modelos de sentence-transformers si lo deseas
        # ej. 'all-MiniLM-L6-v2' es popular y eficiente
        kw_model = KeyBERT(model='all-MiniLM-L6-v2') 
        
        # Extraer palabras clave con sus scores
        keywords = kw_model.extract_keywords(
            cleaned_text,
            keyphrase_ngram_range=(1, 3), # Considera frases de 1 a 3 palabras
            stop_words='english', # Puedes pasar una lista custom si es necesario
            top_n=top_n,
            use_mmr=True,  # Maximal Marginal Relevance (mejora diversidad)
            diversity=diversity # 0.7 es un buen punto de partida para diversidad
        )
        # keywords es una lista de tuplas: (frase, score)
        self.keyphrases_with_scores = keywords if keywords else []
        if not self.keyphrases_with_scores:
            print("Advertencia: KeyBERT no extrajo frases clave.")


    def generate_wordcloud_from_keyphrases(self):
        """Genera una nube de palabras con las frases clave extraÃ­das por BERT y sus scores"""
        if not self.keyphrases_with_scores:
            print("Error: No hay frases clave para generar la nube de palabras.")
            return

        # Crear un diccionario de frecuencias donde la 'frecuencia' es el score de KeyBERT
        # Multiplicamos por un factor para que los scores (0-1) den tamaÃ±os de fuente visibles
        # Y aseguramos que sean enteros para WordCloud
        freq_dict = {phrase: int(score * 100) + 1 for phrase, score in self.keyphrases_with_scores}
        
        if not freq_dict:
            print("Error: Diccionario de frecuencias para WordCloud estÃ¡ vacÃ­o.")
            return

        # Lista de stopwords comunes y especÃ­ficas del dominio que podrÃ­as querer aÃ±adir
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["study", "article", "paper", "research", "shows", "demonstrates", 
                                 "highlighted", "presents", "contribution", "approach", "based",
                                 "results", "analysis", "data", "archaeological", "lidar", "use"])


        wordcloud = WordCloud(width=1200, height=600, # Aumentado tamaÃ±o
                              background_color='white',
                              stopwords=custom_stopwords,
                              min_font_size=10,
                              colormap='viridis', # Probar diferentes colormaps
                              prefer_horizontal=0.9 # MayorÃ­a de palabras horizontales
                              ).generate_from_frequencies(freq_dict)
        
        plt.figure(figsize=(15, 7.5)) # Aumentado tamaÃ±o de figura
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("KeyBERT-Extracted Key Contributions & Concepts", fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        plt.savefig("cloud2.png")
        plt.show()
        

    def run_analysis(self):
        """Ejecuta todo el flujo de anÃ¡lisis"""
        self.parse_html_table()
        # No se necesita clean_text() separado si clean_text_for_bert() hace lo necesario
        self.extract_keyphrases_with_bert(top_n=50, diversity=0.6) # Aumentar top_n y probar diversidad
        self.generate_wordcloud_from_keyphrases()

if __name__ == "__main__":
    # Reemplaza esto por tu contenido HTML real de la Table 4
    html_table4 = '''
    [<h3 style="color:#71b12c;font-size: 2.2rem;">Table 4: Brief Conclusions or Key Contributions</h3><table style="width:100%; border-collapse: collapse; font-size: 1.2rem; line-height: 1.5;"><thead><tr style="background-color:#f2f2f2;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Paper (Reference)</th><th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 1.4rem; color:#8A26BD;">Brief Conclusion / Key Contribution</th></tr></thead><tbody><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The lowland Maya settlement landscape: Environmental LiDAR and ecology <a href="#cita13" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[13]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">This article shows how LiDAR data, although originally designed for environmental studies, can be successfully reused to explore and map complex archaeological landscapes, especially in forested regions like the Maya basin.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Combined Detection and Segmentation of Archeological Structures from LiDAR Data Using a Deep Learning Approach <a href="#cita14" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[14]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study demonstrates that Mask R-CNN, combined with transfer learning and data augmentation techniques, can be effectively used to detect and segment archaeological structures automatically from LiDAR data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era <a href="#cita15" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[15]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study highlighted that, although current artificial vision models offer promising tools for the automatic detection of archaeological structures, they still face unique challenges in environments with dense vegetation and subtle patterns.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Ancient lowland Maya complexity as revealed by airborne laser scanning of northern Guatemala <a href="#cita16" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[16]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">This article shows how LiDAR data, although originally designed for environmental studies, can be successfully reused to explore and map complex archaeological landscapes, especially in forested regions like the Maya basin.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Beyond Visualization: Remote Sensing Applications in Prehispanic Settlements to Understand Ancient Anthropogenic Land Use and Occupation in the Sierra Nevada de Santa Marta, Colombia <a href="#cita17" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[17]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study demonstrates that hybrid models and hierarchical architectures, along with the appropriate use of elevation channels, are crucial for advancing in this field.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The domestication of Amazonia before European conquest <a href="#cita18" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[18]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">This study represents an important contribution to the field of Amazonian archaeology, proposing an alternative approach that combines tensor algebra and spatial perception.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Deep Learning for Archaeological Object Detection on LiDAR: New Evaluation Measures and Insights <a href="#cita19" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[19]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study presents an innovative method for the detection of geospatial objects based on the combined use of topological, geometric, and contextual information.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Caracol, Belize, and Changing Perceptions of Ancient Maya Society <a href="#cita20" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[20]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study was based on the integration of archaeological, GIS, and ecological analysis data.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Unraveling pre-Columbian occupation patterns in the tropical forests of French Guiana using an anthracological approach <a href="#cita21" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[21]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The study was based on anthracological analysis, radiocarbon dating, and soil characterization.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Wetland Farming and the Early Anthropocene: Globally Upscaling from the Maya Lowlands with LiDAR and Multiproxy Verification <a href="#cita22" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[22]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">The analysis was based on LiDAR, GIS, multiproxy analysis, and radiocarbon dating.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">TensorMap: LIDAR-Based Topological Mapping and Localization via Tensor Decompositions <a href="#cita35" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[35]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">This work represents an important contribution to the field of mobile robotics and LiDAR data processing, proposing an innovative approach that combines tensor algebra and spatial perception.</p></td></tr><tr><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">Exploring Topological Information Beyond Persistent Homology to Detect Geospatial Objects <a href="#cita36" target="_blank" style="color:#b5ff46; font-size:1.2rem;">[36]</a></p></td><td style="border: 1px solid #ddd; padding: 8px;"><p style="font-size: 1.2rem; line-height: 1.6;">This work establishes a solid foundation for future research aimed at optimizing and expanding the use of knowledge-based methods in the field of geospatial analysis.</p></td></tr></tbody></table>]
    '''
    analyzer = ConclusionAnalyzer(html_table4)
    analyzer.run_analysis()


display(Image(filename='/kaggle/input/amaztest/my_imgs/feat4.png'))


%%time
"""Vistazo de la info"""
import os
import rasterio
import numpy as np

class GeoTiffAnalyzer:
    def __init__(self, paths):
        self.paths = paths  # Diccionario con las rutas
    
    def get_tif_files(self, directory):
        """Obtiene todos los archivos .tif en un directorio"""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")]
    
    def analyze_geotiff(self, filepath):
        """Analiza profundamente un archivo GeoTIFF y muestra info simple"""
        try:
            with rasterio.open(filepath) as src:
                print(f"\nArchivo: {os.path.basename(filepath)}")

                # InformaciÃ³n geoespacial bÃ¡sica
                print(f" - CRS: {src.crs}")
                print(f" - ResoluciÃ³n: {src.res[0]:.6f} x {src.res[1]:.6f}")
                
                # Banda(s)
                print(f" - Bandas: {src.count}")
                for i in range(1, src.count + 1):
                    dtype = src.dtypes[i - 1]
                    description = src.descriptions[i - 1] or "Sin descripciÃ³n"
                    
                    data = src.read(i).astype(dtype)
                    valid_data = data[(data != src.nodata) & (~np.isnan(data))] if src.nodata is not None else data[~np.isnan(data)]

                    print(f"   Canal {i}:")
                    print(f"     DescripciÃ³n: {description}")
                    print(f"     Tipo de dato: {dtype}")
                    print(f"     Valores vÃ¡lidos: {len(valid_data)}")
                    if len(valid_data) > 0:
                        print(f"     Rango: {np.min(valid_data):.2f} - {np.max(valid_data):.2f}")
                        print(f"     Media Â± Std: {np.mean(valid_data):.2f} Â± {np.std(valid_data):.2f}")
                    else:
                        print("     No hay datos vÃ¡lidos.")

                print("-" * 50)

        except Exception as e:
            print(f"Error al procesar {filepath}: {e}")

    def process_directories(self):
        """Procesa todos los archivos .tif en cada directorio"""
        for key, path in self.paths.items():
            print(f"\n{'='*60}\nDirectorio [{key.upper()}]: {path}\n{'='*60}")
            tif_files = self.get_tif_files(path)
            if not tif_files:
                print("No se encontraron archivos .tif.")
                continue
            for tif_file in tif_files:
                self.analyze_geotiff(tif_file)

# Rutas proporcionadas
paths = {
    "s3": "/kaggle/input/amaztest/LC03_SAR_LC_Biomass_1093/LC03_SAR_LC_Biomass_1093/data",
    "sm": "/kaggle/input/amaztest/Estimated_Biomass_Stock_Amazon_1648/Estimated_Biomass_Stock_Amazon_1648/data",
    "os": "/kaggle/input/amaztest/rasters_GEDIL3"
}

if __name__ == "__main__":
    analyzer = GeoTiffAnalyzer(paths)
    analyzer.process_directories()


%%time
"""Se cuenta en bruto los puntos"""
import os
import rasterio
import numpy as np

class GeoTiffValueCounter:
    def __init__(self, paths):
        self.paths = paths  # Diccionario con las rutas
        self.total_counts = {key: 0 for key in paths}  # Inicializa contadores
    
    def get_tif_files(self, directory):
        """Obtiene todos los archivos .tif en un directorio"""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")]
    
    def count_valid_values_in_file(self, filepath):
        """Cuenta los valores vÃ¡lidos (no NaN, no nodata) en un archivo GeoTIFF"""
        total = 0
        try:
            with rasterio.open(filepath) as src:
                for i in range(1, src.count + 1):
                    data = src.read(i)
                    nodata = src.nodata
                    # Filtra valores vÃ¡lidos
                    if nodata is not None:
                        valid = (data != nodata) & (~np.isnan(data))
                    else:
                        valid = ~np.isnan(data)
                    total += np.sum(valid)
        except Exception as e:
            print(f"Error procesando {filepath}: {e}")
        return total

    def process_directories(self):
        """Procesa todos los archivos .tif en cada directorio y cuenta valores"""
        for key, path in self.paths.items():
            print(f"[{key.upper()}] Procesando directorio: {path}")
            tif_files = self.get_tif_files(path)
            if not tif_files:
                print(f"  â�Œ No se encontraron archivos .tif.")
                continue
            
            dir_total = 0
            for tif_file in tif_files:
                filename = os.path.basename(tif_file)
                count = self.count_valid_values_in_file(tif_file)
                dir_total += count
                print(f"  âœ”ï¸� {filename}: {count:,} valores vÃ¡lidos")
            
            self.total_counts[key] = dir_total
            print(f"  ğŸ“¦ Total en [{key.upper()}]: {dir_total:,} valores vÃ¡lidos\n")

    def show_summary(self):
        """Muestra un resumen final con el total de valores por conjunto"""
        print("\nğŸ“Š Resumen total de valores vÃ¡lidos:")
        for key, total in self.total_counts.items():
            print(f" - [{key.upper()}]: {total:,} puntos")
        print("\nâœ… Â¡AnÃ¡lisis completado!")

# Rutas proporcionadas
paths = {
    "s3": "/kaggle/working/min_dots/s3",
    "sm": "/kaggle/working/min_dots/sm",
    "os": "/kaggle/working/min_dots/os"
}

if __name__ == "__main__":
    counter = GeoTiffValueCounter(paths)
    counter.process_directories()
    counter.show_summary()

# """[S3] Procesando directorio: /kaggle/input/amaztest/LC03_SAR_LC_Biomass_1093/LC03_SAR_LC_Biomass_1093/data  âœ”ï¸� manau_classification.tif: 29,614,377 valores vÃ¡lidos  âœ”ï¸� ron2_radar.tif: 9,818,820 valores vÃ¡lidos  âœ”ï¸� ron2_biomass.tif: 3,272,940 valores vÃ¡lidos  âœ”ï¸� tap_classification.tif: 19,784,704 valores vÃ¡lidos  âœ”ï¸� tap_radar.tif: 59,354,112 valores vÃ¡lidos  âœ”ï¸� manaus_radar.tif: 89,830,566 valores vÃ¡lidos  âœ”ï¸� ron2_classification.tif: 3,272,940 valores vÃ¡lidos  âœ”ï¸� manaus_biomass.tif: 29,943,522 valores vÃ¡lidos  âœ”ï¸� rio2_radar.tif: 12,416,652 valores vÃ¡lidos  âœ”ï¸� rio2_classification.tif: 4,138,884 valores vÃ¡lidos  âœ”ï¸� rio2_biomass.tif: 4,138,884 valores vÃ¡lidos  âœ”ï¸� tap_biomass.tif: 19,784,704 valores vÃ¡lidos  ğŸ“¦ Total en [S3]: 285,371,105 valores vÃ¡lidos[SM] Procesando directorio: /kaggle/input/amaztest/Estimated_Biomass_Stock_Amazon_1648/Estimated_Biomass_Stock_Amazon_1648/data  âœ”ï¸� paragominas_predicted_agb.tif: 1,931,909 valores vÃ¡lidos  âœ”ï¸� and_predicted_agb_mean.tif: 1,000 valores vÃ¡lidos  âœ”ï¸� par_predicted_agb_mean.tif: 1,074 valores vÃ¡lidos  âœ”ï¸� par_predicted_agb_sd.tif: 1,074 valores vÃ¡lidos  âœ”ï¸� and_predicted_agb_sd.tif: 1,000 valores vÃ¡lidos  âœ”ï¸� cau_predicted_agb_mean.tif: 1,271 valores vÃ¡lidos  âœ”ï¸� cau_predicted_agb_sd.tif: 1,271 valores vÃ¡lidos  ğŸ“¦ Total en [SM]: 1,938,599 valores vÃ¡lidos[OS] Procesando directorio: /kaggle/input/amaztest/rasters_GEDIL3  âœ”ï¸� output_vh.tif: 6,566,319 valores vÃ¡lidos  âœ”ï¸� output_be.tif: 6,566,319 valores vÃ¡lidos  ğŸ“¦ Total en [OS]: 13,132,638 valores vÃ¡lidosğŸ“Š Resumen total de valores vÃ¡lidos: - [S3]: 285,371,105 puntos - [SM]: 1,938,599 puntos - [OS]: 13,132,638 puntosâœ… Â¡AnÃ¡lisis completado!CPU times: user 2 s, sys: 843 ms, total: 2.84 sWall time: 7.2 s""""""[S3] Procesando directorio: /kaggle/working/min_dots/s3  âœ”ï¸� 5_4_tap_biomass.tif: 19,784,704 valores vÃ¡lidos  âœ”ï¸� 2_2_rio2_radar.tif: 12,416,652 valores vÃ¡lidos  âœ”ï¸� 5_4_tap_radar.tif: 59,354,112 valores vÃ¡lidos  âœ”ï¸� 2_2_rio2_classification.tif: 4,138,884 valores vÃ¡lidos  âœ”ï¸� 4_4_manaus_biomass.tif: 29,943,522 valores vÃ¡lidos  âœ”ï¸� 5_4_tap_classification.tif: 19,784,704 valores vÃ¡lidos  âœ”ï¸� 2_2_rio2_biomass.tif: 4,138,884 valores vÃ¡lidos  âœ”ï¸� 4_4_manaus_radar.tif: 89,830,566 valores vÃ¡lidos  âœ”ï¸� 4_4_manau_classification.tif: 29,614,377 valores vÃ¡lidos  ğŸ“¦ Total en [S3]: 269,006,405 valores vÃ¡lidos[SM] Procesando directorio: /kaggle/working/min_dots/sm  âœ”ï¸� 6_4_cau_predicted_agb_mean.tif: 1,271 valores vÃ¡lidos  âœ”ï¸� 6_4_cau_predicted_agb_sd.tif: 1,271 valores vÃ¡lidos  âœ”ï¸� 6_4_paragominas_predicted_agb.tif: 600,573 valores vÃ¡lidos  ğŸ“¦ Total en [SM]: 603,115 valores vÃ¡lidos[OS] Procesando directorio: /kaggle/working/min_dots/os  âœ”ï¸� 3_4_output_be.tif: 120,362 valores vÃ¡lidos  âœ”ï¸� 4_4_output_be.tif: 91,018 valores vÃ¡lidos  âœ”ï¸� 2_2_output_vh.tif: 130,470 valores vÃ¡lidos  âœ”ï¸� 4_1_output_vh.tif: 138,850 valores vÃ¡lidos  âœ”ï¸� 6_2_output_vh.tif: 144,499 valores vÃ¡lidos  âœ”ï¸� 2_2_output_be.tif: 130,470 valores vÃ¡lidos  âœ”ï¸� 6_4_output_vh.tif: 132,617 valores vÃ¡lidos  âœ”ï¸� 5_4_output_be.tif: 122,975 valores vÃ¡lidos  âœ”ï¸� 3_4_output_vh.tif: 120,362 valores vÃ¡lidos  âœ”ï¸� 4_4_output_vh.tif: 91,018 valores vÃ¡lidos  âœ”ï¸� 4_1_output_be.tif: 138,850 valores vÃ¡lidos  âœ”ï¸� 6_2_output_be.tif: 144,499 valores vÃ¡lidos  âœ”ï¸� 6_1_output_be.tif: 144,234 valores vÃ¡lidos  âœ”ï¸� 5_6_output_vh.tif: 109,632 valores vÃ¡lidos  âœ”ï¸� 6_1_output_vh.tif: 144,234 valores vÃ¡lidos  âœ”ï¸� 5_1_output_be.tif: 145,629 valores vÃ¡lidos  âœ”ï¸� 5_6_output_be.tif: 109,632 valores vÃ¡lidos  âœ”ï¸� 6_4_output_be.tif: 132,617 valores vÃ¡lidos  âœ”ï¸� 5_1_output_vh.tif: 145,629 valores vÃ¡lidos  âœ”ï¸� 5_4_output_vh.tif: 122,975 valores vÃ¡lidos  ğŸ“¦ Total en [OS]: 2,560,572 valores vÃ¡lidosğŸ“Š Resumen total de valores vÃ¡lidos: - [S3]: 269,006,405 puntos - [SM]: 603,115 puntos - [OS]: 2,560,572 puntosâœ… Â¡AnÃ¡lisis completado!CPU times: user 767 ms, sys: 427 ms, total: 1.19 sWall time: 1.19 s"""


%%time
"""tiff acotados y persistidos """
import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
import geopandas as gpd
from shapely.ops import transform
import pyproj

class RasterAreaCropper:
    def __init__(self, geojson_path, input_paths, output_base_dir):
        self.geojson_path = geojson_path
        self.input_paths = input_paths
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

        print("ğŸ“Œ Cargando archivo GeoJSON...")
        self.gdf = gpd.read_file(geojson_path)
        if self.gdf.crs != "EPSG:4326":
            self.gdf = self.gdf.to_crs("EPSG:4326")
        print(f"âœ… GeoJSON cargado con {len(self.gdf)} Ã¡reas")

    def reproject_shape_to_raster_crs(self, geom, raster_crs):
        """Reproyecta una geometrÃ­a al CRS del raster"""
        project = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS(raster_crs), always_xy=True
        ).transform
        return transform(project, geom)

    def process_all_areas(self):
        for idx, row in self.gdf.iterrows():
            area_id = row["id"]
            print(f"\nğŸ”§ Procesando Ã¡rea: {area_id}")
            geom = shape(row['geometry'])

            for key, path in self.input_paths.items():
                input_files = [f for f in os.listdir(path) if f.endswith(".tif")]
                output_dir = os.path.join(self.output_base_dir, key)
                os.makedirs(output_dir, exist_ok=True)

                for file in input_files:
                    input_path = os.path.join(path, file)
                    output_path = os.path.join(output_dir, f"{area_id}_{file}")

                    if os.path.exists(output_path):
                        print(f" âš ï¸� Ya existe: {output_path}, saltando...")
                        continue

                    try:
                        with rasterio.open(input_path) as src:
                            # Reproyectar geometrÃ­a al sistema del raster
                            geom_reprojected = self.reproject_shape_to_raster_crs(geom, src.crs)

                            # Verificar si hay intersecciÃ³n REAL
                            raster_bounds = box(*src.bounds)
                            if not raster_bounds.intersects(geom_reprojected):
                                print(f"ğŸš« Bounding box no se solapa con {file}. Saltando.")
                                continue

                            try:
                                out_image, out_transform = mask(
                                    src,
                                    [geom_reprojected.__geo_interface__],
                                    crop=True,
                                    all_touched=True,
                                    nodata=src.nodata if src.nodata is not None else -9999
                                )

                                # Contar pÃ­xeles vÃ¡lidos en la mÃ¡scara
                                valid_pixels = np.count_nonzero(~np.isnan(out_image))
                                if valid_pixels == 0:
                                    print(f"ğŸŸ¨ {file}: Recorte hecho, pero sin pÃ­xeles vÃ¡lidos. Saltando guardado.")
                                    continue

                                print(f"ğŸŸ¢ {file}: Tiene {valid_pixels} pÃ­xeles dentro del Ã¡rea. Guardando...")

                                out_meta = src.meta.copy()
                                out_meta.update({
                                    "driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform,
                                    "crs": src.crs
                                })

                                with rasterio.open(output_path, "w", **out_meta) as dest:
                                    dest.write(out_image)

                                print(f" âœ”ï¸� Guardado: {output_path}")

                            except ValueError as ve:
                                if "Input shapes do not overlap raster" in str(ve):
                                    print(f"â�Œ {file}: Sin solapamiento real. Saltando.")
                                else:
                                    print(f"âš ï¸� Error desconocido en {file}: {ve}")

                    except Exception as e:
                        print(f"ğŸš¨ Error procesando {input_path}: {e}")

# ConfiguraciÃ³n
geojson_path = "/kaggle/working/amazon_data/critical_areas.geojson"
input_paths = {
    "s3": "/kaggle/input/amaztest/LC03_SAR_LC_Biomass_1093/LC03_SAR_LC_Biomass_1093/data",
    "sm": "/kaggle/input/amaztest/Estimated_Biomass_Stock_Amazon_1648/Estimated_Biomass_Stock_Amazon_1648/data",
    "os": "/kaggle/input/amaztest/rasters_GEDIL3"
}
output_base_dir = "/kaggle/working/min_dots"

if __name__ == "__main__":
    cropper = RasterAreaCropper(geojson_path, input_paths, output_base_dir)
    cropper.process_all_areas()
    print("\nâœ… Â¡Proceso finalizado!")


%%time
import os

class SizeComparator:
    def __init__(self, original_paths, cropped_dir):
        self.original_paths = original_paths
        self.cropped_dir = cropped_dir

    def get_folder_size_gb(self, folder):
        """Calcula el tamaÃ±o total de archivos en GB"""
        total_size = 0
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    continue
        return total_size / (1024 ** 3)  # Convertir a GB

    def compare_sizes(self):
        """Muestra comparaciÃ³n de tamaÃ±os"""
        print("ğŸ“Š Comparando tamaÃ±os...\n")

        # TamaÃ±o de datos originales
        original_total = 0
        for key, path in self.original_paths.items():
            size = self.get_folder_size_gb(path)
            print(f"ğŸ“� [{key.upper()}] Datos originales: {size:.3f} GB")
            original_total += size

        # TamaÃ±o de datos recortados
        cropped_total = self.get_folder_size_gb(self.cropped_dir)
        print(f"\nğŸ“¦ [min_dots] Datos recortados: {cropped_total:.3f} GB")

        # Diferencia
        reduction = original_total - cropped_total
        percent_reduction = (reduction / original_total * 100) if original_total > 0 else 0
        print("\nğŸ“‰ Resumen:")
        print(f"  Original: {original_total:.3f} GB")
        print(f"  Recortado: {cropped_total:.3f} GB")
        print(f"  ReducciÃ³n: {reduction:.3f} GB (-{percent_reduction:.2f}%)\n")

# ConfiguraciÃ³n
original_paths = {
    "s3": "/kaggle/input/amaztest/LC03_SAR_LC_Biomass_1093/LC03_SAR_LC_Biomass_1093/data",
    "sm": "/kaggle/input/amaztest/Estimated_Biomass_Stock_Amazon_1648/Estimated_Biomass_Stock_Amazon_1648/data",
    "os": "/kaggle/input/amaztest/rasters_GEDIL3"
}
cropped_dir = "/kaggle/working/min_dots"

if __name__ == "__main__":
    comparator = SizeComparator(original_paths, cropped_dir)
    comparator.compare_sizes()


%%time
"""Se procesa rapido o prueba rapido"""
import os
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ConfiguraciÃ³n principal
INPUT_DIR = "/kaggle/input/amaztest/Amazon_ForestStructure_LIDAR_2412/Amazon_ForestStructure_LIDAR_2412/data"
GEOJSON_PATH = "/kaggle/working/amazon_data/critical_areas.geojson"
OUTPUT_DIR = "/kaggle/working/min_dots/s1"

# Modo truncado para diagnÃ³stico
TRUNCATE_MODE = False      # True = modo de prueba con N archivos, False = proceso completo
TRUNCATE_LIMIT = 2        # Archivos a procesar en modo truncado

# Cargar Ã¡reas crÃ­ticas
areas_gdf = gpd.read_file(GEOJSON_PATH)
if areas_gdf.crs != "EPSG:4326":
    areas_gdf = areas_gdf.to_crs("EPSG:4326")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def extract_lat_lon_from_filename(filename):
    """Extrae lat y lon desde el nombre del archivo"""
    match = re.search(r'lat(-?\d+\.?\d*)lon(-?\d+\.?\d*)', filename)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    return None, None

def process_file_debug(file):
    try:
        file_path = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        lat, lon = extract_lat_lon_from_filename(file)

        if lat is None or lon is None:
            return f"{file} â†’ â�Œ sin coordenadas en nombre"

        point = Point(lon, lat)
        in_area = any(areas_gdf.contains(point))

        return f"{file} â†’ {'âœ… dentro de cuadrante' if in_area else 'â›” fuera de cuadrantes'}"

    except Exception as e:
        return f"â�Œ Error en {file}: {e}"

def process_file_prod(file):
    try:
        file_path = os.path.join(INPUT_DIR, file)
        base_name, ext = os.path.splitext(file)
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        lat, lon = extract_lat_lon_from_filename(file)

        if lat is None or lon is None:
            return  # No se puede ubicar espacialmente

        point = Point(lon, lat)
        matched = areas_gdf[areas_gdf.contains(point)]

        if matched.empty:
            return  # El archivo completo estÃ¡ fuera de Ã¡reas crÃ­ticas

        for _, row in matched.iterrows():
            quadrant_id = row["id"]
            output_name = f"{base_name}_q{quadrant_id}.parquet"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            df.to_parquet(output_path, engine="pyarrow")

    except Exception as e:
        return f"â�Œ Error en {file}: {e}"

def main():
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".sss", ".pss", ".css"))]
    files = all_files[:TRUNCATE_LIMIT] if TRUNCATE_MODE else all_files

    print(f"ğŸ“¦ Modo truncado: {'ON' if TRUNCATE_MODE else 'OFF'} â€” Archivos a procesar: {len(files)}")

    if TRUNCATE_MODE:
        results = [process_file_debug(file) for file in tqdm(files, desc="ğŸ”� DiagnÃ³stico")]
        print("\nğŸ“Š Resultados del diagnÃ³stico:")
        for line in results:
            print(line)
    else:
        with ProcessPoolExecutor(max_workers=3) as executor:
            list(tqdm(executor.map(process_file_prod, files), total=len(files), desc="ğŸš€ Procesando"))

        print("\nâœ… Â¡Procesamiento completo y optimizado!")

if __name__ == "__main__":
    main()



%%time
"""Se acotan las etiquetas"""
import geopandas as gpd
from pathlib import Path

class GeoTagFilter:
    def __init__(self, points_path, polygons_path, output_path):
        self.points_path = points_path
        self.polygons_path = polygons_path
        self.output_path = output_path
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        print("ğŸ“Œ Cargando puntos...")
        self.points_gdf = gpd.read_file(self.points_path)
        print(f"âœ… Puntos cargados: {len(self.points_gdf)}")

        print("ğŸ“Œ Cargando Ã¡reas crÃ­ticas...")
        self.polygons_gdf = gpd.read_file(self.polygons_path)
        print(f"âœ… Ã�reas cargadas: {len(self.polygons_gdf)}")

        # Asegurar CRS uniforme
        if self.points_gdf.crs != "EPSG:4326":
            self.points_gdf = self.points_gdf.to_crs("EPSG:4326")
        if self.polygons_gdf.crs != "EPSG:4326":
            self.polygons_gdf = self.polygons_gdf.to_crs("EPSG:4326")

    def filter_points(self):
        print("ğŸ”� Filtrando puntos dentro de Ã¡reas...")
        self.filtered_gdf = gpd.sjoin(self.points_gdf, self.polygons_gdf, how="inner", predicate="within")
        print(f"ğŸŸ¢ Puntos filtrados: {len(self.filtered_gdf)}")

    def save_filtered(self):
        print(f"ğŸ’¾ Guardando resultados en: {self.output_path}")
        self.filtered_gdf.drop(columns="index_right").to_file(self.output_path, driver="GeoJSON")
        print("âœ… Archivo guardado correctamente.")

    def run(self):
        self.load_data()
        self.filter_points()
        self.save_filtered()

# EjecuciÃ³n
if __name__ == "__main__":
    points_path = "/kaggle/working/amazon_data/tagged_sites_filtered.geojson"
    polygons_path = "/kaggle/working/amazon_data/critical_areas.geojson"
    output_path = "/kaggle/working/min_dots/mini_tags.geojson"

    processor = GeoTagFilter(points_path, polygons_path, output_path)
    processor.run()



%%time
"""ver las etiquetas en el mapa"""
import folium
import geopandas as gpd
import json
import matplotlib.colors as mcolors
from pathlib import Path


class MapGenerator:
    def __init__(self, geojson_path, output_path, overlay_path=None):
        self.geojson_path = geojson_path
        self.output_path = output_path
        self.overlay_path = overlay_path
        self.gdf = gpd.read_file(geojson_path)
        self.overlay_gdf = gpd.read_file(overlay_path) if overlay_path else None
        self.group_counts = self.gdf['ffather'].value_counts()
        self.top_groups = self.group_counts.index.tolist()
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.group_colors = {group: self.colors[i % len(self.colors)] for i, group in enumerate(self.top_groups)}
        self.m = None

    def bbox_to_polygon(self, bbox_str):
        try:
            if isinstance(bbox_str, str):
                bbox = json.loads(bbox_str.replace("'", "\""))
            else:
                bbox = bbox_str
            return [[
                (bbox['lat_min'], bbox['lon_min']),
                (bbox['lat_min'], bbox['lon_max']),
                (bbox['lat_max'], bbox['lon_max']),
                (bbox['lat_max'], bbox['lon_min']),
                (bbox['lat_min'], bbox['lon_min'])
            ]]
        except Exception as e:
            print(f"Error procesando bbox: {e}")
            return None

    def get_center(self):
        all_bboxes = []
        for bbox in self.gdf['bbox']:
            try:
                bbox_data = json.loads(bbox.replace("'", "\"")) if isinstance(bbox, str) else bbox
                all_bboxes.append([
                    (bbox_data['lat_min'] + bbox_data['lat_max']) / 2,
                    (bbox_data['lon_min'] + bbox_data['lon_max']) / 2
                ])
            except:
                continue
        if all_bboxes:
            avg_lat = sum(x[0] for x in all_bboxes) / len(all_bboxes)
            avg_lon = sum(x[1] for x in all_bboxes) / len(all_bboxes)
        else:
            avg_lat = self.gdf.geometry.y.mean()
            avg_lon = self.gdf.geometry.x.mean()
        return avg_lat, avg_lon

    def create_map(self):
        center_lat, center_lon = self.get_center()
        self.m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap',
            control_scale=True
        )
        title_html = '''
            <div style="position: fixed; width: 100%; top: 10px; z-index: 1000; text-align: center;">
                <h2 style="background-color: white; display: inline-block; padding: 10px 20px; 
                           border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.3);">
                    Study Area Tags by <em>Ffather</em>
                </h2>
            </div>
        '''
        self.m.get_root().html.add_child(folium.Element(title_html))

    def add_main_polygons(self):
        feature_groups = {
            group: folium.FeatureGroup(name=f"{group[:50]} ({self.group_counts[group]})", show=True)
            for group in self.top_groups
        }

        for idx, row in self.gdf.iterrows():
            group = row['ffather']
            color = self.group_colors.get(group, 'gray')
            feature_group = feature_groups[group]

            polygon_coords = self.bbox_to_polygon(row['bbox']) or [[
                (y, x) for x, y in (
                    row.geometry.buffer(0.01).exterior.coords
                    if row.geometry.geom_type == 'Point'
                    else row.geometry.exterior.coords
                )
            ]]

            dataset_name = Path(str(row['dataset'])).stem  
            popup_text = f"""
                <b>Ffather:</b> {group}<br>
                <b>Dataset:</b> {dataset_name}<br>
                <b>ID:</b> {row['id_left']}<br>
                <b>Area:</b> {row['area_km2']:.2f} kmÂ²<br>
                <b>Unknown %:</b> {row['percentage_unknown']:.2f}%
            """

            folium.Polygon(
                locations=polygon_coords[0],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                weight=1,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(feature_group)

        for fg in feature_groups.values():
            self.m.add_child(fg)

    def add_overlay_layer(self):
        if self.overlay_gdf is None:
            return

        overlay_group = folium.FeatureGroup(name="Critical Quadrants", show=True)

        for _, row in self.overlay_gdf.iterrows():
            popup_text = f"""
                <b>ID:</b> {row['id']}<br>
                <b>X, Y Index:</b> ({row['x_index']}, {row['y_index']})<br>
                <b>Area:</b> {row['area_km2']:.2f} kmÂ²<br>
                <b>Unknown %:</b> {row['percentage_unknown']:.2f}%
            """

            folium.GeoJson(
                row['geometry'],
                name="CriticalArea",
                style_function=lambda x: {
                    "color": "#8888ff",
                    "weight": 1,
                    "fillColor": "#8888ff",
                    "fillOpacity": 0.2,
                    "dashArray": "5, 5"
                },
                tooltip=folium.Tooltip(popup_text)
            ).add_to(overlay_group)

        self.m.add_child(overlay_group)

    def save(self):
        folium.LayerControl(collapsed=False).add_to(self.m)
        self.m.save(self.output_path)
        print(f"Mapa guardado en: {self.output_path}")
        print("\nSummary by ffather:")
        print(self.group_counts)

    def run(self):
        self.create_map()
        self.add_main_polygons()
        self.add_overlay_layer()
        self.save()


# ====== Ejecutar ======
if __name__ == "__main__":
    geojson_path = '/kaggle/working/min_dots/mini_tags.geojson'
    output_path = '/kaggle/working/min_dots/tags_map.html'
    overlay_path = '/kaggle/working/amazon_data/critical_areas.geojson'

    generator = MapGenerator(geojson_path, output_path, overlay_path)
    generator.run()



display(Image(filename='/kaggle/input/amaztest/my_imgs/train_tags.png'))


%%time
#
# --- Advanced Archaeological Concept Analysis ---
#
# This script builds upon the previous framework to quantitatively measure the
# relationship between different research concepts found in the literature.
#
# It introduces a ConceptCorrelator class that:
# 1. Maps high-level concepts to the specific raw data features needed to implement them.
# 2. Calculates a correlation/similarity matrix between all concepts using the
#    Jaccard Index, a statistical metric for set similarity.
# 3. This reveals which research paths are closely related (share data) and which are
#    distinct, providing a structured map of the research landscape.
#

import itertools
import pandas as pd
from typing import List, Dict, Any, Set

# --- DICTIONARY 1: FINDINGS FROM THE LITERATURE REVIEW ---
# (Same as before, weights are used for context)
LITERATURE_CONCEPTS: Dict[str, Dict[str, Any]] = {
    'LiDAR_DTM_Analysis': {
        'weight': 5,
        'description': 'Models prioritizing the direct analysis of high-resolution LiDAR data to find terrain anomalies.'
    },
    'Favorable_Geo_Context': {
        'weight': 4,
        'description': 'Models based on the hypothesis that settlements are in strategic geographic locations (e.g., near water, on terraces).'
    },
    'Multidisciplinary_Integration': {
        'weight': 3,
        'description': 'Holistic models that assume combining multiple, diverse data types is most effective.'
    },
    'Cultural_Landscape_Signs': {
        'weight': 2,
        'description': 'Models looking for indirect evidence of human activity, like soil changes (ADEs) or vegetation anomalies.'
    },
    'Human_Modified_Geometry': {
        'weight': 1,
        'description': 'Models focused purely on the shape and patterns of the terrain, often using advanced topographic derivatives.'
    },
    'GIS_Spatial_Analysis': {
        'weight': 1,
        'description': 'Models that rely on classic GIS operations like buffering and weighted overlays.'
    }
}

# --- DICTIONARY 2: MAPPING RAW DATA TO LITERARY CONCEPTS ---
# (Same as before, this is the crucial link)
DATA_MAPPING: Dict[str, List[str]] = {
    'gedi_dtm': ['LiDAR_DTM_Analysis', 'Human_Modified_Geometry'],
    'slope': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'aspect': ['Human_Modified_Geometry'],
    'tpi': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'hillshade': ['Human_Modified_Geometry', 'LiDAR_DTM_Analysis'],
    'topographic_diversity': ['Favorable_Geo_Context'],
    'gedi_canopy_height': ['LiDAR_DTM_Analysis', 'Cultural_Landscape_Signs'],
    'agb_anomaly': ['Cultural_Landscape_Signs'],
    'slsoc_ED2': ['Cultural_Landscape_Signs'],
    'distance_to_water': ['Favorable_Geo_Context', 'GIS_Spatial_Analysis'],
    'is_fluvial_terrace': ['Favorable_Geo_Context'],
    'all_features_stacked': ['Multidisciplinary_Integration']
}

class ConceptCorrelator:
    """
    Analyzes the relationships between research concepts based on shared data requirements.
    """
    def __init__(self, concepts_db: dict, data_map: dict):
        self.concepts_db = concepts_db
        self.data_map = data_map
        self.concept_names = list(self.concepts_db.keys())
        self.concept_feature_sets = self._map_concepts_to_features()
        self.correlation_matrix = self._calculate_correlation_matrix()

    def _map_concepts_to_features(self) -> Dict[str, Set[str]]:
        """
        Inverts the DATA_MAPPING to create a dictionary where each concept
        maps to a set of required raw data features.
        """
        concept_features = {name: set() for name in self.concept_names}
        for feature, associated_concepts in self.data_map.items():
            for concept in associated_concepts:
                if concept in concept_features:
                    concept_features[concept].add(feature)
        return concept_features

    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """
        Calculates the Jaccard Index between two sets.
        Jaccard = |Intersection| / |Union|
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0.0

    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculates the Jaccard similarity between all pairs of concepts
        and returns the result as a pandas DataFrame.
        """
        matrix = pd.DataFrame(index=self.concept_names, columns=self.concept_names, dtype=float)
        
        for concept1 in self.concept_names:
            for concept2 in self.concept_names:
                set1 = self.concept_feature_sets[concept1]
                set2 = self.concept_feature_sets[concept2]
                similarity = self._jaccard_similarity(set1, set2)
                matrix.loc[concept1, concept2] = similarity
        
        return matrix

    def display_correlation_matrix(self):
        """Prints the calculated correlation matrix to the console."""
        print("\n--- Concept Correlation Matrix (Jaccard Similarity) ---")
        print("This matrix shows how related concepts are based on their shared data requirements.")
        print("Score 1.0 = Identical data needs. Score 0.0 = Completely distinct data needs.\n")
        
        # Use pandas' built-in string formatting for neat alignment
        print(self.correlation_matrix.to_string(float_format="%.3f"))

    def get_most_related_pairs(self, top_n=3):
        """Identifies and prints the most closely related concept pairs."""
        # Unstack the matrix to get a series of pairs, then sort
        matrix_unstacked = self.correlation_matrix.unstack()
        # Remove self-correlations (where index == column)
        matrix_unstacked = matrix_unstacked[matrix_unstacked.index.get_level_values(0) != matrix_unstacked.index.get_level_values(1)]
        
        # Sort and get the top pairs (each pair appears twice, so we divide by 2)
        top_pairs = matrix_unstacked.sort_values(ascending=False).iloc[:top_n*2:2]

        print("\n--- Most Related Concept Pairs ---")
        print("These pairs of concepts are highly interdependent as they rely on similar data features.\n")
        for (concept1, concept2), score in top_pairs.items():
            print(f"  - Pair: [{concept1}] <--> [{concept2}]")
            print(f"    Similarity Score: {score:.3f}")

    def get_most_distinct_pairs(self, top_n=3):
        """Identifies and prints the most distinct (uncorrelated) concept pairs."""
        matrix_unstacked = self.correlation_matrix.unstack()
        # Find pairs with the lowest correlation score
        distinct_pairs = matrix_unstacked.sort_values(ascending=True).iloc[:top_n*2:2]

        print("\n--- Most Distinct Concept Pairs ---")
        print("These pairs represent orthogonal research paths, relying on different data.\n")
        for (concept1, concept2), score in distinct_pairs.items():
            print(f"  - Pair: [{concept1}] <--> [{concept2}]")
            print(f"    Similarity Score: {score:.3f}")


if __name__ == "__main__":
    print("\n" + "*"*20 + " Measuring the Space Between Research Concepts " + "*"*20)

    # Initialize the correlator with our knowledge bases
    correlator = ConceptCorrelator(LITERATURE_CONCEPTS, DATA_MAPPING)

    # Display the primary output: the correlation matrix
    correlator.display_correlation_matrix()

    # Provide summary insights
    correlator.get_most_related_pairs()
    correlator.get_most_distinct_pairs()

    print("\n\nAnalysis complete. This matrix can now guide the strategic combination of concepts for modeling.")
    print("For example, combining two highly distinct concepts could yield a truly novel, multidisciplinary model.")


%%time
#
# --- Archaeological Model Success Tree Generator ---
#
# This script visualizes modeling strategies as a hierarchical "Success Tree".
# The tree is built by starting with the most frequently cited concept in the
# literature and progressively branching out by adding other concepts in order of their importance.
# Each path from the root to a node represents a viable, increasingly
# complex modeling strategy, with a cumulative "success score".
#

import pandas as pd
from typing import List, Dict, Any, Set, Optional

# --- DICTIONARY 1: FINDINGS FROM THE LITERATURE REVIEW ---
# (Weights derived from literature frequency)
LITERATURE_CONCEPTS: Dict[str, Dict[str, Any]] = {
    'LiDAR_DTM_Analysis': {
        'weight': 5,
        'description': 'Models prioritizing the direct analysis of high-resolution LiDAR data to find terrain anomalies.'
    },
    'Favorable_Geo_Context': {
        'weight': 4,
        'description': 'Models based on the hypothesis that settlements are in strategic geographic locations (e.g., near water, on terraces).'
    },
    'Multidisciplinary_Integration': {
        'weight': 3,
        'description': 'Holistic models that assume combining multiple, diverse data types is most effective.'
    },
    'Cultural_Landscape_Signs': {
        'weight': 2,
        'description': 'Models looking for indirect evidence of human activity, like soil changes (ADEs) or vegetation anomalies.'
    },
    'Human_Modified_Geometry': {
        'weight': 1,
        'description': 'Models focused purely on the shape and patterns of the terrain, often using advanced topographic derivatives.'
    },
    'GIS_Spatial_Analysis': {
        'weight': 1,
        'description': 'Models that rely on classic GIS operations like buffering and weighted overlays.'
    }
}

# --- DICTIONARY 2: MAPPING RAW DATA TO LITERARY CONCEPTS ---
DATA_MAPPING: Dict[str, List[str]] = {
    'gedi_dtm': ['LiDAR_DTM_Analysis', 'Human_Modified_Geometry'],
    'slope': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'aspect': ['Human_Modified_Geometry'],
    'tpi': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'hillshade': ['Human_Modified_Geometry', 'LiDAR_DTM_Analysis'],
    'topographic_diversity': ['Favorable_Geo_Context'],
    'gedi_canopy_height': ['LiDAR_DTM_Analysis', 'Cultural_Landscape_Signs'],
    'agb_anomaly': ['Cultural_Landscape_Signs'],
    'slsoc_ED2': ['Cultural_Landscape_Signs'],
    'distance_to_water': ['Favorable_Geo_Context', 'GIS_Spatial_Analysis'],
    'is_fluvial_terrace': ['Favorable_Geo_Context'],
    'all_features_stacked': ['Multidisciplinary_Integration']
}

class ModelNode:
    """Represents a single node in the Success Tree."""
    def __init__(self, concept_name: str, path: List[str], score: int, features: Set[str]):
        self.concept_name = concept_name
        self.path = path
        self.cumulative_score = score
        self.required_features = features
        self.children: List['ModelNode'] = []

    def __repr__(self):
        return f"Node({self.concept_name}, Score={self.cumulative_score})"

class SuccessTreeGenerator:
    """Builds and displays a hierarchical tree of modeling strategies."""
    def __init__(self, concepts_db: dict, data_map: dict):
        self.concepts_db = concepts_db
        self.data_map = data_map
        # Sort concepts by weight to build the tree from the most successful downwards
        self.sorted_concepts = sorted(
            concepts_db.items(),
            key=lambda item: item[1]['weight'],
            reverse=True
        )
        self.root = None

    def _get_features_for_path(self, path: List[str]) -> Set[str]:
        """Gets the set of unique raw data features for a list of concepts."""
        features = set()
        for feature, associated_concepts in self.data_map.items():
            if any(concept in path for concept in associated_concepts):
                features.add(feature)
        return features

    def build_tree(self):
        """Builds the success tree recursively."""
        # Create a virtual root to hold the first level of concepts
        self.root = ModelNode("START", [], 0, set())
        
        # Start the recursive build process
        self._build_recursive(self.root, self.sorted_concepts)

    def _build_recursive(self, parent_node: ModelNode, concepts_to_add: list):
        """The recursive function to add children to a node."""
        for i, (concept_name, concept_data) in enumerate(concepts_to_add):
            
            # Create the new path and calculate its properties
            new_path = parent_node.path + [concept_name]
            new_score = parent_node.cumulative_score + concept_data['weight']
            new_features = self._get_features_for_path(new_path)

            # Create the new node
            child_node = ModelNode(concept_name, new_path, new_score, new_features)
            parent_node.children.append(child_node)

            # The concepts remaining for the next level of recursion are those
            # that come *after* the current one in the sorted list.
            # This ensures we don't have duplicate paths (e.g., A+B and B+A).
            remaining_concepts_for_child = concepts_to_add[i+1:]
            if remaining_concepts_for_child:
                self._build_recursive(child_node, remaining_concepts_for_child)

    def display_tree(self):
        """Displays the entire tree in a human-readable format."""
        if not self.root:
            print("Tree has not been built yet. Call build_tree() first.")
            return
            
        print("\n" + "="*80)
        print("--- Archaeological Model Success Tree ---")
        print("Each path represents a modeling strategy, ordered by literature-based importance.")
        print("Score = Cumulative importance. Features = Raw data needed for this model.")
        print("="*80 + "\n")

        # Start the recursive display from the children of the virtual root
        for child in self.root.children:
            self._display_node_recursive(child, "", is_last=child == self.root.children[-1])

    def _display_node_recursive(self, node: ModelNode, prefix: str, is_last: bool):
        """The recursive function to print a node and its children."""
        connector = "â””â”€â”€ " if is_last else "â”œâ”€â”€ "
        summary = f"(Score: {node.cumulative_score}, Features: {len(node.required_features)})"
        print(f"{prefix}{connector}{node.concept_name} {summary}")
        
        new_prefix = prefix + ("    " if is_last else "â”‚   ")
        for i, child in enumerate(node.children):
            self._display_node_recursive(child, new_prefix, is_last=i == len(node.children) - 1)


if __name__ == "__main__":
    # Initialize the generator
    tree_generator = SuccessTreeGenerator(LITERATURE_CONCEPTS, DATA_MAPPING)

    # Build the tree structure in memory
    tree_generator.build_tree()

    # Display the final, formatted tree
    tree_generator.display_tree()

    print("\n\n--- How to Interpret the Tree ---")
    print("1. Top-Level Nodes: These are the most critical, standalone strategies (e.g., 'LiDAR_DTM_Analysis').")
    print("2. Deeper Paths: A path like 'LiDAR_DTM_Analysis -> Favorable_Geo_Context' represents a more complex, integrated model.")
    print("3. Score: Use the score to rank the overall 'success potential' of any given path (model).")
    print("4. Features: The number of features tells you the data engineering cost for that model.")


%%time
#
# --- Topological Analysis of an Archaeological Research Space ---
#
# This script applies concepts from finite topology to model the landscape of
# archaeological research. It moves beyond simple correlation to describe the
# inherent structure and relationships between research concepts.
#
# Core Idea:
# - The "Space" (X) is the set of all available raw data features.
# - The "Open Sets" are the fundamental research themes, defined by the
#   features required by each concept in the literature. These form a 'basis'
#   for a topology on X.
# - Topological operations (Closure, Interior, Boundary) reveal the dependencies,
#   completeness, and "missing links" between different research strategies.
#

from typing import List, Dict, Any, Set
from itertools import chain, combinations

# --- DICTIONARY 1: FINDINGS FROM THE LITERATURE REVIEW (The Basis for our Topology) ---
LITERATURE_CONCEPTS: Dict[str, Dict[str, Any]] = {
    'LiDAR_DTM_Analysis': {'weight': 5},
    'Favorable_Geo_Context': {'weight': 4},
    'Multidisciplinary_Integration': {'weight': 3},
    'Cultural_Landscape_Signs': {'weight': 2},
    'Human_Modified_Geometry': {'weight': 1},
    'GIS_Spatial_Analysis': {'weight': 1}
}

# --- DICTIONARY 2: MAPPING RAW DATA TO CONCEPTS (Defines the Space) ---
DATA_MAPPING: Dict[str, List[str]] = {
    'gedi_dtm': ['LiDAR_DTM_Analysis', 'Human_Modified_Geometry'],
    'slope': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'aspect': ['Human_Modified_Geometry'],
    'tpi': ['Human_Modified_Geometry', 'Favorable_Geo_Context'],
    'hillshade': ['Human_Modified_Geometry', 'LiDAR_DTM_Analysis'],
    'topographic_diversity': ['Favorable_Geo_Context'],
    'gedi_canopy_height': ['LiDAR_DTM_Analysis', 'Cultural_Landscape_Signs'],
    'agb_anomaly': ['Cultural_Landscape_Signs'],
    'slsoc_ED2': ['Cultural_Landscape_Signs'],
    'distance_to_water': ['Favorable_Geo_Context', 'GIS_Spatial_Analysis'],
    'is_fluvial_terrace': ['Favorable_Geo_Context'],
}

class TopologicalResearchSpace:
    """
    Models the relationships between research concepts as a finite topological space.
    The space itself is the set of all raw data features. The open sets are the
    collections of features required to address a specific research theme.
    """

    def __init__(self, concepts_db: dict, data_map: dict):
        # 1. Define the Universal Set (The Space X)
        self.X: Set[str] = set(data_map.keys())

        # 2. Define the Basis for the Topology
        # Each "basis set" is the group of features for one literary concept.
        self.basis: Dict[str, Set[str]] = self._generate_basis(concepts_db, data_map)

        # 3. Generate the full Topology
        # The open sets are all possible unions of the basis sets.
        self.topology: List[Set[str]] = self._generate_topology_from_basis()

        print("--- Topological Research Space Initialized ---")
        print(f"Space Cardinality |X|: {len(self.X)} features")
        print(f"Basis Cardinality: {len(self.basis)} fundamental research themes")
        print(f"Topology Cardinality (Total Open Sets): {len(self.topology)}")
        print("-" * 50 + "\n")

    def _generate_basis(self, concepts_db: dict, data_map: dict) -> Dict[str, Set[str]]:
        """Creates the basis sets from the literature concepts."""
        basis_sets = {name: set() for name in concepts_db.keys()}
        for feature, associated_concepts in data_map.items():
            for concept in associated_concepts:
                if concept in basis_sets:
                    basis_sets[concept].add(feature)
        
        # Add a special concept for the full space, required by topology axioms
        basis_sets['Multidisciplinary_Integration'] = self.X
        return basis_sets

    def _generate_topology_from_basis(self) -> List[Set[str]]:
        """Generates all open sets by taking all possible unions of basis sets."""
        all_unions = {frozenset()} # Start with the empty set
        basis_values = list(self.basis.values())
        for i in range(1, len(basis_values) + 1):
            for combo in combinations(basis_values, i):
                union_set = set.union(*combo)
                all_unions.add(frozenset(union_set))
        return [set(s) for s in all_unions]

    def find_closure(self, A: Set[str]) -> Set[str]:
        """
        Calculates the closure of a set A.
        The closure is the smallest "complete research topic" (closed set) containing A.
        It's the intersection of all closed sets containing A.
        """
        closed_sets = [self.X - open_set for open_set in self.topology]
        closure = self.X.copy()
        for closed_set in closed_sets:
            if A.issubset(closed_set):
                closure.intersection_update(closed_set)
        return closure

    def find_interior(self, A: Set[str]) -> Set[str]:
        """
        Calculates the interior of a set A.
        The interior is the largest "complete research topic" (open set) contained within A.
        It's the union of all open sets contained in A.
        """
        interior = set()
        for open_set in self.topology:
            if open_set.issubset(A):
                interior.update(open_set)
        return interior

    def analyze_feature_set(self, name: str, feature_set: Set[str]):
        """
        Performs a full topological analysis on a given set of features and
        provides a human-readable interpretation.
        """
        print(f"--- Topological Analysis of Scenario: '{name}' ---")
        print(f"Input Features (Set A): {sorted(list(feature_set))}\n")

        interior = self.find_interior(feature_set)
        closure = self.find_closure(feature_set)
        boundary = closure - interior

        print(">> INTERIOR(A): The largest complete research topic you can MASTER with your data.")
        print(f"   - Features: {sorted(list(interior)) if interior else 'None'}")
        print("   - Interpretation: With the data you have, you can fully and rigorously explore the themes defined by these features.\n")

        print(">> CLOSURE(A): The smallest complete research topic your data TOUCHES UPON.")
        print(f"   - Features: {sorted(list(closure))}")
        print("   - Interpretation: Your data is part of this larger, complete topic. To fully understand it, you must consider all features in the closure.\n")

        print(">> BOUNDARY(A): The 'missing link' features.")
        print(f"   - Features: {sorted(list(boundary)) if boundary else 'None'}")
        print("   - Interpretation: These are the features you must ADD to your set to complete the minimal research topic (the Closure) your data belongs to.\n")
        print("-" * (len(name) + 40) + "\n")


if __name__ == "__main__":
    # --- Setup ---
    space = TopologicalResearchSpace(LITERATURE_CONCEPTS, DATA_MAPPING)

    # --- Analysis Scenarios ---
    
    # Scenario 1: A researcher focuses *only* on classic topographic geometry.
    # What is their position in the wider research space?
    scenario_1_features = {'slope', 'aspect', 'tpi', 'hillshade'}
    space.analyze_feature_set("Classic Topography Focus", scenario_1_features)

    # Scenario 2: A researcher has access *only* to GEDI LiDAR data.
    # What are the implied research topics and what are they missing?
    scenario_2_features = {'gedi_dtm', 'gedi_canopy_height'}
    space.analyze_feature_set("LiDAR-Only Data", scenario_2_features)

    # Scenario 3: A researcher is interested in the link between soil and water.
    # They have soil data and hydrological data.
    scenario_3_features = {'slsoc_ED2', 'distance_to_water'}
    space.analyze_feature_set("Soil & Water Connection", scenario_3_features)


%%time
"""Testing purposes, Full reporte"""
import os
import pandas as pd
from collections import defaultdict

class ParquetValidator:
    def __init__(self, directory):
        self.directory = directory
        self.file_info = {}  # InformaciÃ³n de cada archivo
        self.column_patterns = defaultdict(list)  # Archivos por patrÃ³n de columnas
        self.dtype_patterns = defaultdict(list)   # Archivos por patrÃ³n de dtypes
        self.data_stats = {}  # EstadÃ­sticas por patrÃ³n

    def load_files(self):
        """Carga todos los archivos .parquet y extrae informaciÃ³n relevante."""
        files = [f for f in os.listdir(self.directory) if f.endswith('.parquet')]

        for file in files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_parquet(file_path)

            base_name = os.path.splitext(file)[0]  # Nombre sin extensiÃ³n
            columns = tuple(df.columns)
            dtypes = tuple(df.dtypes.items())
            shape = df.shape

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std']).to_dict() if numeric_cols else {}

            self.file_info[base_name] = {
                'columns': columns,
                'dtypes': dtypes,
                'shape': shape,
                'numeric_stats': stats,
                'numeric_columns': numeric_cols
            }

            self.column_patterns[columns].append(base_name)
            self.dtype_patterns[dtypes].append(base_name)

    def analyze_column_patterns(self):
        """Genera estadÃ­sticas por patrÃ³n de columnas."""
        print("ğŸ§© PATRONES DETECTADOS EN COLUMNAS:")
        for idx, (pattern, files) in enumerate(self.column_patterns.items(), start=1):
            print(f"\nğŸ”¹ PatrÃ³n {idx} ({len(files)} archivos):")
            print(f"   Columnas: {list(pattern)}")
            print("   Archivos (sin extensiÃ³n):")
            print('\n'.join(f"   - {f}" for f in files))

            # Agregar estadÃ­sticas combinadas
            combined_stats = defaultdict(list)
            for file in files:
                info = self.file_info[file]
                for col, stat_dict in info['numeric_stats'].items():
                    for stat, value in stat_dict.items():
                        combined_stats[(col, stat)].append(value)

            # Promediar estadÃ­sticas por columna
            averaged_stats = {}
            for (col, stat), values in combined_stats.items():
                try:
                    avg = sum(v for v in values if pd.notna(v)) / len(values)
                    averaged_stats.setdefault(col, {})[stat] = round(avg, 4)
                except ZeroDivisionError:
                    averaged_stats.setdefault(col, {})[stat] = None

            print("\n   ğŸ”� EstadÃ­sticas promedio por columna (numÃ©ricas):")
            for col, stats in sorted(averaged_stats.items()):
                print(f"      [{col}] -> {stats}")

    def analyze_dtype_patterns(self):
        """Imprime resumen de patrones en dtypes."""
        print("\nğŸ§¬ PATRONES DETECTADOS EN DTYPES:")
        for idx, (pattern, files) in enumerate(self.dtype_patterns.items(), start=1):
            print(f"\nğŸ”¹ PatrÃ³n {idx} ({len(files)} archivos):")
            print(f"   Tipos: {dict(pattern)}")
            print("   Archivos (sin extensiÃ³n):")
            print('\n'.join(f"   - {f}" for f in files))

    def full_report(self):
        """Reporte completo de validaciÃ³n con estadÃ­sticas."""
        print("ğŸ“Š VALIDACIÃ“N COMPLETA DE ARCHIVOS PARQUET")
        print(f"Cantidad total de archivos analizados: {len(self.file_info)}\n")
        self.analyze_column_patterns()
        self.analyze_dtype_patterns()


if __name__ == "__main__":
    dir_path = '/kaggle/working/min_dots/s1'
    validator = ParquetValidator(dir_path)
    validator.load_files()
    validator.full_report()


%%time
"""Tranformar y normalizar"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
import warnings
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)

# DefiniciÃ³n de esquemas esperados
SCHEMAS = {
    'sss': {
        'columns': ['site', 'area', 'depth', 'nscol', 'ntext', 'sand', 'clay', 'slsoc', 'slph', 'slcec', 'sldbd',
                   'elevation', 'slope', 'aspect', 'tci', 'moist_f', 'moist_w'],
        'dtypes': {
            'site': object, 'area': float, 'depth': float, 'nscol': int, 'ntext': int,
            'sand': float, 'clay': float, 'slsoc': float, 'slph': float, 'slcec': float,
            'sldbd': float, 'elevation': int, 'slope': int, 'aspect': int, 'tci': int,
            'moist_f': int, 'moist_w': float
        }
    },
    'pss': {
        'columns': ['time', 'site', 'patch', 'dtype', 'age', 'area', 'fgc', 'fsc', 'stgc', 'stgl', 'stsc',
                   'stsl', 'msc', 'ssc', 'psc', 'fsn', 'msn', 'npl', 'agb', 'bsa', 'lai'],
        'dtypes': {
            'time': int, 'site': object, 'patch': object, 'dtype': int,
            'age': float, 'area': float, 'fgc': float, 'fsc': float,
            'stgc': float, 'stgl': float, 'stsc': float, 'stsl': float,
            'msc': float, 'ssc': float, 'psc': float, 'fsn': float,
            'msn': float, 'npl': float, 'agb': float, 'bsa': float, 'lai': float
        }
    },
    'css': {
        'columns': ['time', 'site', 'patch', 'cohort', 'dbh', 'height', 'pft', 'nplant', 'bdead', 'balive', 'agb', 'lai'],
        'dtypes': {
            'time': int, 'site': object, 'patch': object, 'cohort': int,
            'dbh': float, 'height': float, 'pft': int, 'nplant': float,
            'bdead': float, 'balive': float, 'agb': float, 'lai': float
        }
    }
}

class DataCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.sss_data = []
        self.pss_data = []
        self.css_data = []
        self.scaler = MinMaxScaler()
        self.temp_files = []  # Para rastrear archivos temporales
        
    def ensure_output_dir(self):
        """Asegura que el directorio de salida exista y estÃ© vacÃ­o"""
        try:
            # Limpiar el directorio de salida si existe
            if os.path.exists(self.output_dir):
                for f in os.listdir(self.output_dir):
                    os.remove(os.path.join(self.output_dir, f))
            else:
                os.makedirs(self.output_dir)
            print(f"ğŸ“� Directorio de salida preparado: {self.output_dir}")
        except Exception as e:
            print(f"â�Œ Error preparando directorio de salida: {str(e)}")
            raise
    
    def detect_file_type(self, df):
        """Detecta el tipo de archivo basado en sus columnas"""
        for ftype, schema in SCHEMAS.items():
            if set(df.columns) == set(schema['columns']):
                return ftype
        return None
    
    def enforce_schema(self, df, ftype):
        """Aplica el esquema esperado al DataFrame"""
        schema = SCHEMAS[ftype]
        
        # Reordenar columnas
        df = df.reindex(columns=schema['columns'])
        
        # Convertir tipos de datos
        for col, dtype in schema['dtypes'].items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"âš ï¸� Error convirtiendo '{col}' a {dtype}: {str(e)}")
                    if dtype in [int, float]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def clean_data(self, df, ftype):
        """Aplica limpieza y transformaciones al DataFrame"""
        # ImputaciÃ³n de valores faltantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Eliminar filas con NaN en campos crÃ­ticos
        critical_cols = [c for c in df.columns if c not in ['slope', 'aspect', 'tci']]
        df.dropna(subset=critical_cols, inplace=True)
        
        # Filtrar outliers usando Z-score
        for col in numeric_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    if df[col].nunique() > 5 and not np.isnan(df[col]).all():
                        z_vals = np.abs(zscore(df[col], nan_policy='omit'))
                        df = df[z_vals < 3]
                except Exception as e:
                    print(f"âš ï¸� Error aplicando filtro Z-score a '{col}': {e}")
        
        # Escalado Min-Max
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def process_file(self, filename):
        """Procesa un archivo individual"""
        file_path = os.path.join(self.input_dir, filename)
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"â�Œ Error leyendo {filename}: {str(e)}")
            return None
        
        ftype = self.detect_file_type(df)
        if not ftype:
            print(f"âš ï¸� No se pudo identificar el tipo de archivo: {filename}")
            return None
        
        try:
            df = self.enforce_schema(df, ftype)
            df = self.clean_data(df, ftype)
            
            # Guardar temporalmente (luego serÃ¡ eliminado)
            temp_path = os.path.join(self.output_dir, f"temp_{ftype}_{filename}")
            df.to_parquet(temp_path)
            self.temp_files.append(temp_path)
            
            # Almacenar para combinaciÃ³n posterior
            if ftype == 'sss':
                self.sss_data.append(df)
            elif ftype == 'pss':
                self.pss_data.append(df)
            elif ftype == 'css':
                self.css_data.append(df)
            
            return df
        except Exception as e:
            print(f"â�Œ Error procesando {filename}: {str(e)}")
            return None
    
    def save_combined_files(self):
        """Guarda los archivos combinados y limpia los temporales"""
        saved_files = []
        
        if self.sss_data:
            combined_sss = pd.concat(self.sss_data, ignore_index=True)
            output_path = os.path.join(self.output_dir, "all_sss.parquet")
            combined_sss.to_parquet(output_path, index=False)
            saved_files.append(output_path)
            print(f"ğŸ’¾ Guardados {len(self.sss_data)} archivos SSS en all_sss.parquet")
        
        if self.pss_data:
            combined_pss = pd.concat(self.pss_data, ignore_index=True)
            output_path = os.path.join(self.output_dir, "all_pss.parquet")
            combined_pss.to_parquet(output_path, index=False)
            saved_files.append(output_path)
            print(f"ğŸ’¾ Guardados {len(self.pss_data)} archivos PSS en all_pss.parquet")
        
        if self.css_data:
            combined_css = pd.concat(self.css_data, ignore_index=True)
            output_path = os.path.join(self.output_dir, "all_css.parquet")
            combined_css.to_parquet(output_path, index=False)
            saved_files.append(output_path)
            print(f"ğŸ’¾ Guardados {len(self.css_data)} archivos CSS en all_css.parquet")
        
        # Eliminar archivos temporales
        self.cleanup_temp_files()
        
        return saved_files
    
    def cleanup_temp_files(self):
        """Elimina todos los archivos temporales"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"âš ï¸� Error eliminando archivo temporal {temp_file}: {str(e)}")
        
        # TambiÃ©n eliminamos cualquier otro archivo que no sea los 3 combinados
        for f in os.listdir(self.output_dir):
            if f not in ["all_sss.parquet", "all_pss.parquet", "all_css.parquet"]:
                try:
                    os.remove(os.path.join(self.output_dir, f))
                except Exception as e:
                    print(f"âš ï¸� Error eliminando archivo no deseado {f}: {str(e)}")
    
    def verify_output(self):
        """Verifica que solo existan los 3 archivos combinados"""
        files = os.listdir(self.output_dir)
        expected_files = {"all_sss.parquet", "all_pss.parquet", "all_css.parquet"}
        
        # Verificar archivos esperados
        for fname in expected_files:
            fpath = os.path.join(self.output_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                if size > 0:
                    print(f"âœ… {fname} creado correctamente ({size/1024:.2f} KB)")
                else:
                    print(f"âš ï¸� {fname} creado pero vacÃ­o")
            else:
                print(f"â�Œ {fname} no fue creado")
        
        # Verificar archivos no deseados
        unexpected_files = set(files) - expected_files
        if unexpected_files:
            print(f"âš ï¸� Archivos no deseados encontrados: {unexpected_files}")
        else:
            print("âœ… Solo los archivos combinados permanecen en el directorio")
    
    def run_processing(self):
        """Ejecuta el procesamiento completo de los archivos"""
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.parquet')]
        print(f"ğŸ”� Encontrados {len(files)} archivos para procesar")
        
        # Procesar archivos con barra de progreso
        for file in tqdm(files, desc="Procesando archivos", unit="file"):
            self.process_file(file)
        
        # Guardar archivos combinados (esto eliminarÃ¡ los temporales)
        saved_files = self.save_combined_files()
        
        # VerificaciÃ³n final
        self.verify_output()
        
        return saved_files


if __name__ == "__main__":
    input_dir = '/kaggle/working/min_dots/s1'
    output_dir = '/kaggle/working/ag_data/s1'
    
    cleaner = DataCleaner(input_dir, output_dir)
    cleaner.run_processing()


%%time
"""ver datos a profundidad"""

import os
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict

class GeoTiffStatsAnalyzer:
    def __init__(self, dirs):
        self.dirs = dirs
        self.stats = defaultdict(list)
        self.band_info = defaultdict(dict)  # Guarda info de bandas por archivo

    def get_band_info(self, filepath):
        """Detecta nÃºmero de bandas, tipos de datos y metadatos por banda"""
        try:
            with rasterio.open(filepath) as src:
                file_name = Path(filepath).name
                print(f"\nğŸ”� Analizando bandas de: {file_name}")

                for band_idx in range(1, src.count + 1):
                    band = src.read(band_idx)
                    dtype = band.dtype.name

                    # Leer metadatos de banda si existen
                    description = src.descriptions[band_idx - 1] if src.descriptions else f"Banda {band_idx}"
                    nodata = src.nodatavals[band_idx - 1] if src.nodatavals else None

                    band_metadata = {
                        'band': band_idx,
                        'dtype': dtype,
                        'description': description,
                        'nodata': nodata,
                        'shape': band.shape,
                        'min': float(np.nanmin(band)) if band.size > 0 else None,
                        'max': float(np.nanmax(band)) if band.size > 0 else None,
                        'unique_count': len(np.unique(band))
                    }

                    self.band_info[file_name][f"band_{band_idx}"] = band_metadata
                    print(f"   â�¤ Banda {band_idx}: {description} | Tipo: {dtype} | NaN: {nodata}")
        except Exception as e:
            print(f"â�Œ Error leyendo bandas de {filepath}: {str(e)}")

    def analyze_file(self, filepath, folder_name):
        """Analiza estadÃ­sticas de un archivo tiff"""
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                data = data[~np.isnan(data)]  # Ignorar NaN

                if data.size == 0:
                    print(f"âš ï¸� Archivo vacÃ­o o solo NaN: {filepath}")
                    return

                stats = {
                    'filename': Path(filepath).name,
                    'folder': folder_name,
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'median': float(np.median(data)),
                    'p5': float(np.percentile(data, 5)),
                    'p95': float(np.percentile(data, 95)),
                    'unique_count': len(np.unique(data)),
                    'total_pixels': int(data.size),
                    'valid_pixels': int(data.size - np.isnan(data).sum())
                }

                self.stats[folder_name].append(stats)
                self.get_band_info(filepath)  # Nuevo mÃ©todo para leer info de bandas
                return stats

        except Exception as e:
            print(f"â�Œ Error procesando {filepath}: {str(e)}")
            return None

    def analyze_all(self):
        """Recorre todos los archivos tiff en cada carpeta y analiza"""
        for folder_name, folder_path in self.dirs.items():
            if not os.path.exists(folder_path):
                print(f"âš ï¸� Carpeta no encontrada: {folder_path}")
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
            print(f"ğŸ”� Analizando {len(files)} archivos en '{folder_name}'")

            for file in files:
                file_path = os.path.join(folder_path, file)
                self.analyze_file(file_path, folder_name)

    def summarize_stats(self):
        """Genera un resumen estadÃ­stico por carpeta"""
        summary = {}

        for folder, records in self.stats.items():
            all_data = [r['mean'] for r in records]
            summary[folder] = {
                'total_archivos': len(records),
                'mean_global': round(float(np.mean(all_data)), 4),
                'std_global': round(float(np.std(all_data)), 4),
                'min_global': float(np.min([r['min'] for r in records])),
                'max_global': float(np.max([r['max'] for r in records])),
                'valores_unicos_promedio': round(float(np.mean([r['unique_count'] for r in records])), 2)
            }

        return summary

    def report_band_info(self):
        """Imprime informaciÃ³n de bandas por archivo"""
        print("\nğŸ“¡ INFORMACIÃ“N DE BANDAS:")
        for filename, bands in self.band_info.items():
            print(f"\nğŸ“� Archivo: {filename}")
            for bname, bdata in bands.items():
                print(f"   ğŸ”¹{bname}")
                print(f"      DescripciÃ³n: {bdata['description']}")
                print(f"      Tipo de dato: {bdata['dtype']}")
                print(f"      Valores Ãºnicos: {bdata['unique_count']}")
                print(f"      Min/Max: {bdata['min']} / {bdata['max']}")

    def report(self):
        """Imprime un informe detallado de estadÃ­sticas"""
        summary = self.summarize_stats()

        print("\nğŸ“Š RESUMEN GLOBAL POR CARPETA:")
        for folder, data in summary.items():
            print(f"\nğŸ“� [{folder}]")
            print(f"  Total de archivos: {data['total_archivos']}")
            print(f"  Media global: {data['mean_global']}")
            print(f"  Desv. Est. global: {data['std_global']}")
            print(f"  MÃ­nimo global: {data['min_global']}")
            print(f"  MÃ¡ximo global: {data['max_global']}")
            print(f"  Promedio de valores Ãºnicos: {data['valores_unicos_promedio']}")

        print("\nğŸ“ˆ ESTADÃ�STICAS DETALLADAS POR ARCHIVO:")
        for folder, records in self.stats.items():
            print(f"\nğŸ“‚ Carpeta: {folder}")
            for rec in records:
                print(f" - {rec['filename']}")
                print(f"     min: {rec['min']:.4f}, max: {rec['max']:.4f}")
                print(f"     mean: {rec['mean']:.4f}, std: {rec['std']:.4f}")
                print(f"     Pixeles vÃ¡lidos: {rec['valid_pixels']} / {rec['total_pixels']}")

    def save_band_info_to_csv(self, output_path='/kaggle/working/ag_data/tiff_band_info.csv'):
        """Guarda la informaciÃ³n de las bandas en un archivo CSV"""
        import pandas as pd

        rows = []
        for filename, bands in self.band_info.items():
            for bname, bdata in bands.items():
                row = {
                    'filename': filename,
                    'band': bdata['band'],
                    'description': bdata['description'],
                    'dtype': bdata['dtype'],
                    'min': bdata['min'],
                    'max': bdata['max'],
                    'unique_count': bdata['unique_count'],
                    'nodata': bdata['nodata']
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nğŸ’¾ InformaciÃ³n de bandas guardada en: {output_path}")

    def get_band_df(self):
        """Devuelve la informaciÃ³n de bandas como DataFrame"""
        import pandas as pd
        rows = []
        for filename, bands in self.band_info.items():
            for bname, bdata in bands.items():
                row = {
                    'filename': filename,
                    'band': bdata['band'],
                    'description': bdata['description'],
                    'dtype': bdata['dtype'],
                    'min': bdata['min'],
                    'max': bdata['max'],
                    'unique_count': bdata['unique_count'],
                    'nodata': bdata['nodata']
                }
                rows.append(row)
        return pd.DataFrame(rows)

if __name__ == "__main__":
    dirs = {
        's3': '/kaggle/working/min_dots/s3',
        'os': '/kaggle/working/min_dots/os',
        'sm': '/kaggle/working/min_dots/sm'
    }

    analyzer = GeoTiffStatsAnalyzer(dirs)
    analyzer.analyze_all()
    analyzer.report()
    analyzer.report_band_info()
    analyzer.save_band_info_to_csv()


%%time
"""Transformar y masajear"""
import os
import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
from collections import defaultdict

class GeoTiffNormalizer:
    def __init__(self, dirs, output_dirs):
        """
        Inicializa el normalizador con mÃºltiples directorios de archivos .tiff
        
        dirs: dict {nombre_carpeta: ruta_de_entrada}
        output_dirs: dict {nombre_carpeta: ruta_de_salida}
        """
        self.dirs = dirs
        self.output_dirs = output_dirs
        self.file_stats = []

        # Crear directorios de salida si no existen
        for folder in output_dirs.values():
            os.makedirs(folder, exist_ok=True)

    def _get_dataset_rules(self, folder):
        """Devuelve reglas de transformaciÃ³n basadas en documentaciÃ³n tÃ©cnica"""
        if folder == 's3':  # LBA-ECO LC-03 Biomass & Classification (45docs.pdf)
            return {
                'nodata_values': [255.0],
                'dtype': np.uint8,
                'valid_range': (0, 255),
                'normalize': False,
                'convert_to_float': False,
                'description': "Land cover and biomass estimates"
            }
        elif folder == 'os':  # GEDI L3 Gridded Land Surface Metrics (50docs.pdf)
            return {
                'nodata_values': [-9999.0],
                'dtype': np.float32,
                'valid_range': (-10000, 10000),
                'normalize': True,
                'convert_to_float': True,
                'description': "Canopy height, ground elevation, footprint counts"
            }
        elif folder == 'sm':  # LiDAR and PALSAR-Derived Forest AGB (48docs.pdf)
            return {
                'nodata_values': [-99999.0],
                'dtype': np.float64,
                'valid_range': (0, 1000),
                'normalize': False,
                'convert_to_float': True,
                'description': "Aboveground biomass estimates (Mg/ha)"
            }
        else:
            return {}

    def _apply_transformations(self, data, folder):
        """Aplica transformaciones segÃºn documento tÃ©cnico del conjunto"""
        rules = self._get_dataset_rules(folder)

        # Convertir a float para trabajar sin errores
        data = data.astype(np.float32)

        # Reemplazar nodata
        for nd in rules.get('nodata_values', []):
            data[data == nd] = np.nan

        # Limitar al rango vÃ¡lido
        min_val, max_val = rules['valid_range']
        data[(data < min_val) | (data > max_val)] = np.nan

        # Imputar NaN solo si hay datos vÃ¡lidos
        if np.isnan(data).any():
            if rules['dtype'] in [np.uint8, np.int64]:
                median = int(np.floor(np.nanmedian(data)))
            else:
                median = np.nanmedian(data)
            data = np.nan_to_num(data, nan=median)

        # NormalizaciÃ³n opcional
        if rules.get('normalize'):
            valid_mask = data != 0
            if valid_mask.any():
                data_min = np.min(data[valid_mask])
                data_max = np.max(data[valid_mask])
                data[valid_mask] = (data[valid_mask] - data_min) / (data_max - data_min)

        # Forzar tipo final
        return data.astype(rules['dtype'])

    def normalize_file(self, filepath, folder_name):
        """Procesa un archivo tiff aplicando buenas prÃ¡cticas de transformaciÃ³n"""
        try:
            with rasterio.open(filepath) as src:
                meta = src.meta.copy()
                data_all_bands = []

                for band_idx in range(1, src.count + 1):
                    data = src.read(band_idx).astype(meta['dtype'])
                    data_clean = self._apply_transformations(data, folder_name)
                    data_all_bands.append(data_clean)

                # Actualizar metadatos
                meta.update(dtype=rasterio.dtypes.get_minimum_dtype(data_all_bands[0]))

                # Ruta de salida
                output_path = os.path.join(self.output_dirs[folder_name], Path(filepath).name)

                # Escribir archivo procesado
                with rasterio.open(output_path, 'w', **meta) as dst:
                    for i, band_data in enumerate(data_all_bands, start=1):
                        dst.write(band_data, i)

                print(f"âœ… Archivo guardado: {output_path}")

                # Registrar estadÃ­sticas por banda
                for idx, band_data in enumerate(data_all_bands, start=1):
                    stats = {
                        'filename': Path(filepath).name,
                        'folder': folder_name,
                        'band': idx,
                        'min': float(np.min(band_data)),
                        'max': float(np.max(band_data)),
                        'mean': float(np.mean(band_data)),
                        'std': float(np.std(band_data)),
                        'unique_count': len(np.unique(band_data)),
                        'dtype': str(band_data.dtype)
                    }
                    self.file_stats.append(stats)

        except Exception as e:
            print(f"â�Œ Error procesando {filepath}: {str(e)}")

    def run_normalization(self):
        """Recorre todos los archivos tiff en cada carpeta y analiza"""
        for folder_name, folder_path in self.dirs.items():
            if not os.path.exists(folder_path):
                print(f"âš ï¸� Carpeta no encontrada: {folder_path}")
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
            print(f"\nğŸ”� Procesando {len(files)} archivos en '{folder_name}'")

            for file in files:
                file_path = os.path.join(folder_path, file)
                self.normalize_file(file_path, folder_name)

        # Guardar reporte final
        df_stats = pd.DataFrame(self.file_stats)
        stats_path = '/kaggle/working/ag_data/tiff_normalization_report.csv'
        df_stats.to_csv(stats_path, index=False)
        print(f"\nğŸ“Š EstadÃ­sticas guardadas en: {stats_path}")

        return df_stats

if __name__ == "__main__":
    input_dirs = {
        's3': '/kaggle/working/min_dots/s3',
        'os': '/kaggle/working/min_dots/os',
        'sm': '/kaggle/working/min_dots/sm'
    }

    output_dirs = {
        's3': '/kaggle/working/ag_data/s3',
        'os': '/kaggle/working/ag_data/os',
        'sm': '/kaggle/working/ag_data/sm'
    }

    normalizer = GeoTiffNormalizer(input_dirs, output_dirs)
    df_stats = normalizer.run_normalization()

    print("\nğŸ“ˆ Resumen estadÃ­stico:")
    print(df_stats.groupby('folder')[['min', 'max', 'mean']].mean())


%%time
"""Evidenciar cambios de normalizacion"""
import pandas as pd
import plotly.express as px


class DataNormalizerVisualizer:
    def __init__(self, original_csv_path, normalized_csv_path):
        """
        Inicializa el visualizador con dos archivos CSV:
        - Datos originales
        - Datos normalizados
        """
        self.original_df = pd.read_csv(original_csv_path)
        self.normalized_df = pd.read_csv(normalized_csv_path)

        # Agregar estado para diferenciarlos
        self.original_df['status'] = 'Original'
        self.normalized_df['status'] = 'Normalized'

    def _get_numeric_columns(self):
        """Devuelve las columnas numÃ©ricas comunes entre ambos datasets"""
        numeric_original = self.original_df.select_dtypes(include='number').columns.tolist()
        numeric_normalized = self.normalized_df.select_dtypes(include='number').columns.tolist()
        return list(set(numeric_original) & set(numeric_normalized))

    def plot_all_violins(self):
        """
        Genera grÃ¡ficos de violÃ­n para comparar todas las columnas numÃ©ricas
        antes y despuÃ©s de la normalizaciÃ³n
        """
        numeric_cols = self._get_numeric_columns()

        if not numeric_cols:
            print("â�Œ No hay columnas numÃ©ricas comunes entre los datasets")
            return

        print(f"ğŸ”¹ Showing {len(numeric_cols)} violin graphs...")

        for col in numeric_cols:
            combined = pd.concat([
                self.original_df[[col, 'status']],
                self.normalized_df[[col, 'status']]
            ], axis=0)

            fig = px.violin(
                combined,
                y=col,
                color='status',
                box=True,
                points="outliers",
                title=f"ğŸ“Š Distribution of '{col}' - Before vs After Normalization",
                violinmode='group'
            )

            fig.update_layout(
                yaxis_title="Value",
                xaxis_title="Distribution",
                legend_title="Status"
            )

            fig.show()

if __name__ == "__main__":
    visualizer = DataNormalizerVisualizer(
        original_csv_path="/kaggle/working/ag_data/tiff_band_info.csv",
        normalized_csv_path="/kaggle/working/ag_data/tiff_normalization_report.csv"
    )
    
    visualizer.plot_all_violins()


%%time
"""Procesar puntos validos de ruido"""
import os
import glob
import numpy as np
import rasterio
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

RELEVANT_BANDS = [
    'rh25', 'rh50', 'rh75', 'rh95', 'rh100',
    'digital_elevation_model', 'digital_elevation_model_srtm',
    'landsat_treecover', 'landsat_water_persistence'
]

MASK_BANDS = ['quality_flag', 'degrade_flag', 'sensitivity']

class GEDIDataLiteProcessor:
    def __init__(self, input_dir, output_dir, report_path, num_workers=4, batch_size=50):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.report_path = report_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.area_files = self._group_files()

    def _group_files(self):
        tifs = glob.glob(os.path.join(self.input_dir, "*.tif"))
        grouped = {}
        for path in tifs:
            name = os.path.basename(path).replace(".tif", "")
            area_id = name.split('_')[-1]
            band = name.split('.')[-1]
            grouped.setdefault(area_id, {})[band] = path
        return grouped

    def _read_band(self, path):
        with rasterio.open(path) as src:
            return src.read(1), src.profile

    def _create_batches(self):
        items = list(self.area_files.items())
        return [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]

    def _process_batch(self, batch):
        results = []

        for area_id, band_dict in batch:
            # Cargar mÃ¡scaras si existen
            masks = {}
            for mask_band in MASK_BANDS:
                if mask_band in band_dict:
                    mask, _ = self._read_band(band_dict[mask_band])
                    masks[mask_band] = mask

            for band in RELEVANT_BANDS:
                if band not in band_dict:
                    continue

                try:
                    data, profile = self._read_band(band_dict[band])
                    data = data.astype(np.float32)

                    # Curado nodata por banda
                    if band.startswith("rh"):
                        data = np.where((data < -213) | (data > 213), np.nan, data)
                    elif "elevation" in band:
                        data = np.where((data < -1000) | (data > 25000), np.nan, data)
                    elif "landsat" in band:
                        data = np.where((data < 0) | (data > 100), np.nan, data)
                    else:
                        data = np.where(data < -1e6, np.nan, data)

                    # Aplicar mÃ¡scaras si estÃ¡n disponibles
                    valid_mask = np.ones_like(data, dtype=bool)
                    if "quality_flag" in masks:
                        valid_mask &= (masks["quality_flag"] == 1)
                    if "degrade_flag" in masks:
                        valid_mask &= (masks["degrade_flag"] == 0)
                    if "sensitivity" in masks:
                        valid_mask &= (masks["sensitivity"] >= 0) & (masks["sensitivity"] <= 1)

                    masked = np.where(valid_mask, data, np.nan)
                    valid_vals = masked[np.isfinite(masked)]

                    if valid_vals.size < 10:
                        continue

                    p5, p95 = np.percentile(valid_vals, [5, 95])
                    norm = np.clip(masked, p5, p95)

                    out_name = f"{band}_{area_id}.tif"
                    out_path = os.path.join(self.output_dir, out_name)
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(np.nan_to_num(norm).astype(profile['dtype']), 1)

                    results.append({
                        'area': area_id, 'band': band,
                        'min': np.nanmin(norm), 'max': np.nanmax(norm),
                        'mean': np.nanmean(norm), 'median': np.nanmedian(norm),
                        'std': np.nanstd(norm), 'p05': p5, 'p95': p95,
                        'valid_pixels': int(np.sum(np.isfinite(norm)))
                    })

                except Exception as e:
                    #print(f"[ERROR] Banda {band} ({area_id}): {e}")
                    continue

        return results

    def run(self):
        print(f"ğŸš€ Procesando con {self.num_workers} procesos...")
        batches = self._create_batches()

        with Pool(self.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(self._process_batch, batches), total=len(batches)))

        flat_results = [r for sub in results if sub for r in sub]
        if not flat_results:
            print("âš ï¸� No se generaron resultados vÃ¡lidos.")
            return

        df = pd.DataFrame(flat_results)
        df.to_csv(self.report_path, index=False)
        print(f"âœ… Reporte guardado en {self.report_path}")


if __name__ == "__main__":
    input_dir = "/kaggle/working/gee_amz_xs/GEDI_Monthly"
    output_dir = "/kaggle/working/ag_data/GEDI_Monthly"
    report_path = "/kaggle/working/ag_data/repor_GEDI_Monthly.csv"

    processor = GEDIDataLiteProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        num_workers=4,
        batch_size=80
    )
    processor.run()



%%time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

class GEDIPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.normalized_df = None
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        """Carga los datos desde el archivo CSV."""
        self.data = pd.read_csv(self.file_path)
        print("Datos cargados correctamente.")
    
    def normalize_data(self):
        """Normaliza las columnas numÃ©ricas entre 0 y 1."""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        normalized_values = self.scaler.fit_transform(self.data[numeric_cols])
        self.normalized_df = pd.DataFrame(normalized_values, columns=numeric_cols)
        print("Datos normalizados correctamente.")
    
    def plot_violin(self):
        """Genera un violin plot interactivo con Plotly."""
        if self.normalized_df is None:
            raise ValueError("Primero debes normalizar los datos.")

        # Convertir a formato largo para Plotly
        df_long = self.normalized_df.melt(var_name='Variable', value_name='Valor')

        # Crear el violin plot
        fig = px.violin(df_long, 
                        x='Variable', 
                        y='Valor', 
                        box=True, 
                        points='all',
                        title='Violin Plot of Normalized Variables (0-1)',
                        )

        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Normalized Value",
            violinmode='group',  # Valores vÃ¡lidos: 'group' o 'overlay'
            height=600,
            width=1000
        )

        # Personalizar cada violÃ­n
        fig.update_traces(
            meanline_visible=True,
            box_visible=True,   # Mostrar caja dentro del violÃ­n
            points='outliers'   # Mostrar puntos extremos
        )
        
        fig.show()

if __name__ == "__main__":
    # Ruta del archivo
    file_path = "/kaggle/working/ag_data/repor_GEDI_Monthly.csv"
    
    # Instanciar y ejecutar
    plotter = GEDIPlotter(file_path)
    plotter.load_data()
    plotter.normalize_data()
    plotter.plot_violin()


%%time
"""Masajear conjuntos, atacando su variabilidad segun los docs """
import os
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import threading
import warnings

# Ignorar advertencias de divisiÃ³n no vÃ¡lida
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


class DEMProcessor:
    def __init__(self, input_paths, output_paths, normalization_method='robust', max_workers=4):
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.normalization_method = normalization_method
        self.max_workers = max_workers
        self.report = []
        self.report_lock = threading.Lock()

        # ConfiguraciÃ³n especÃ­fica por dataset
        self.dataset_info = {
            "ALOS": {
                "bands": ["DSM", "MSK", "STK"],
                "nodata_values": {"DSM": -32768, "MSK": 255, "STK": 0},
                "valid_ranges": {"DSM": (-433, 8768), "STK": (1, 54)},
                "normalization_strategy": "value_range"
            },
            "Copernicus": {
                "bands": ["DEM", "EDM", "FLM", "WBM"],
                "nodata_values": {"DEM": -32767, "EDM": 255, "FLM": 255, "WBM": 255},
                "valid_ranges": {"DEM": None, "EDM": (0, 13), "WBM": (0, 3)},
                "normalization_strategy": "band_specific"
            },
            "SRTM": {
                "bands": ["constant"],
                "nodata_values": {"constant": -32768},
                "valid_ranges": {"constant": (0, 1)},
                "normalization_strategy": "global"
            }
        }

    def _ensure_output_dirs(self):
        """Crea carpetas de salida si no existen"""
        for path in self.output_paths.values():
            os.makedirs(path, exist_ok=True)

    def _load_tif(self, file_path):
        with rasterio.open(file_path) as src:
            data = src.read(1)
            profile = src.profile
        return data.astype(np.float32), profile

    def _save_tif(self, data, profile, output_file):
        profile.update(dtype=rasterio.float32, nodata=None)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)

    def _extract_band_name(self, filename, dataset_name):
        if dataset_name == "ALOS":
            if ".DSM.tif" in filename:
                return "DSM"
            elif ".MSK.tif" in filename:
                return "MSK"
            elif ".STK.tif" in filename:
                return "STK"
        elif dataset_name == "Copernicus":
            if "Copernicus_DEM" in filename:
                return "DEM"
            elif "Copernicus_EDM" in filename:
                return "EDM"
            elif "Copernicus_FLM" in filename:
                return "FLM"
            elif "Copernicus_WBM" in filename:
                return "WBM"
        elif dataset_name == "SRTM":
            if ".constant.tif" in filename:
                return "constant"
        return None


    def _get_nodata_value(self, dataset_name, band_name):
        return self.dataset_info[dataset_name]["nodata_values"].get(band_name, -32768)

    def _clean_data(self, data, dataset_name, band_name):
        """Limpia datos reemplazando no-data por mediana"""
        no_data_value = self._get_nodata_value(dataset_name, band_name)
        data[data == no_data_value] = np.nan
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return np.zeros_like(data)
        fill_value = np.nanmedian(valid_data)
        cleaned = np.nan_to_num(data, nan=fill_value)
        if dataset_name == "Copernicus" and band_name == "EDM":
            cleaned = np.clip(cleaned, 0, 13)
        return cleaned

    def _normalize_data(self, data, dataset_name, band_name):
        strategy = self.dataset_info[dataset_name]["normalization_strategy"]
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return np.zeros_like(data)
        if strategy == "value_range" or strategy == "band_specific":
            ranges = self.dataset_info[dataset_name]["valid_ranges"].get(band_name)
            if ranges and ranges[0] is not None and ranges[1] is not None:
                min_val, max_val = ranges
                if np.isclose(max_val, min_val):
                    return np.zeros_like(data)
                normalized = (data - min_val) / (max_val - min_val)
                return np.clip(normalized, 0, 1).astype(np.float32)
        q1, q99 = np.percentile(valid_data, [1, 99])
        normalized = np.divide(data - q1, q99 - q1, out=np.zeros_like(data), where=(q99 - q1) != 0)
        return np.clip(normalized, 0, 1).astype(np.float32)


    def _augment_data(self, data, dataset_name, band_name):
        if len(data.shape) != 2:
            return data
        if dataset_name == "ALOS" and band_name == "DSM":
            x, y = np.gradient(data)
            slope = np.sqrt(x**2 + y**2)
            return np.dstack([data, slope])
        elif dataset_name == "Copernicus" and band_name == "DEM":
            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
            roughness = np.abs(convolve2d(data, kernel, mode='same', boundary='symm'))
            return np.dstack([data, roughness])
        elif dataset_name == "SRTM" and band_name == "constant":
            mean = uniform_filter(data, size=3)
            tpi = data - mean
            return np.dstack([data, tpi])
        return data

    def _get_stats(self, data):
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return {
                "min": 0, "max": 0, "mean": 0, "std": 0, "median": 0
            }
        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "median": float(np.median(valid_data))
        }

    def _process_file(self, file_info):
        file_path, dataset_name, output_dir = file_info
        filename = os.path.basename(file_path)
        try:
            band_name = self._extract_band_name(filename, dataset_name)
            if not band_name:
                raise ValueError(f"No se pudo identificar la banda en {filename}")
            raw_data, profile = self._load_tif(file_path)
            cleaned_data = self._clean_data(raw_data, dataset_name, band_name)
            stats_before = self._get_stats(cleaned_data)
            normalized_data = self._normalize_data(cleaned_data, dataset_name, band_name)
            augmented_data = self._augment_data(normalized_data, dataset_name, band_name)
            stats_after = self._get_stats(normalized_data)
            output_file = os.path.join(output_dir, filename)
            if augmented_data.ndim == 2:
                self._save_tif(augmented_data, profile, output_file)
            else:
                base_name, ext = os.path.splitext(output_file)
                for i in range(augmented_data.shape[2]):
                    band_out = f"{base_name}_band{i}{ext}"
                    self._save_tif(augmented_data[:, :, i], profile, band_out)
            report_entry = {
                "dataset": dataset_name,
                "band": band_name,
                "filename": filename,
                "before_min": stats_before["min"],
                "before_max": stats_before["max"],
                "before_mean": stats_before["mean"],
                "before_std": stats_before["std"],
                "after_min": stats_after["min"],
                "after_max": stats_after["max"],
                "after_mean": stats_after["mean"],
                "after_std": stats_after["std"],
            }
            with self.report_lock:
                self.report.append(report_entry)
            return report_entry
        except Exception as e:
            warnings.warn(f"Error procesando {filename}: {str(e)}")
            return None

    def _process_dataset(self, root, output_dir, dataset_name, file_filter):
        file_list = []
        for dirpath, _, files in os.walk(root):
            for file in files:
                if file_filter(file):
                    file_list.append((os.path.join(dirpath, file), dataset_name, output_dir))
        if not file_list:
            print(f"\nâš ï¸� No se encontraron archivos para {dataset_name}")
            return
        print(f"\nğŸ”� Encontrados {len(file_list)} archivos en {dataset_name}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_file, info) for info in file_list]
            successful = 0
            with tqdm(as_completed(futures), total=len(futures), desc=f"Procesando {dataset_name}", unit="archivo") as pbar:
                for future in pbar:
                    result = future.result()
                    if result:
                        successful += 1
                    pbar.set_postfix({"Ã©xitos": successful})
            print(f"âœ… {dataset_name} terminado: {successful}/{len(file_list)}")

    def process_alos(self):
        root = self.input_paths['a']
        output_dir = self.output_paths["/kaggle/working/ag_data/ALOS"]
        self._process_dataset(
            root, output_dir, "ALOS",
            lambda f: any(f.endswith(ext) for ext in [".DSM.tif", ".MSK.tif", ".STK.tif"])
        )

    def process_copernicus(self):
        root = self.input_paths['c']
        output_dir = self.output_paths["/kaggle/working/ag_data/Copernicus"]
        self._process_dataset(
            root, output_dir, "Copernicus",
            lambda f: any(band in f for band in ["DEM", "EDM", "FLM", "WBM"] if ".tif" in f)
        )

    def process_srtm(self):
        root = self.input_paths['s']
        output_dir = self.output_paths["/kaggle/working/ag_data/SRTM"]
        self._process_dataset(
            root, output_dir, "SRTM",
            lambda f: ".constant.tif" in f
        )

    def save_report(self):
        df = pd.DataFrame(self.report)
        report_path = "/kaggle/working/ag_data/top_normalization_report.csv"
        df.to_csv(report_path, index=False)
        print(f"\nğŸ“Š Informe guardado en: {report_path}")

    def run(self):
        self._ensure_output_dirs()
        print("\nğŸš€ Iniciando pipeline de procesamiento de DEM...\n")
        self.process_alos()
        self.process_copernicus()
        self.process_srtm()
        self.save_report()
        print("\nğŸ�‰ Pipeline completado exitosamente.")

if __name__ == "__main__":
    input_paths = {
        'a': "/kaggle/working/gee_amz_xs/ALOS_DEM_AW3D30",
        'c': "/kaggle/working/gee_amz_xs/Copernicus_DEM_GLO30",
        's': "/kaggle/working/gee_amz_xs/Global_SRTM_Topographic_Diversity"
    }
    output_paths = {
        "/kaggle/working/ag_data/ALOS": "/kaggle/working/ag_data/ALOS",
        "/kaggle/working/ag_data/Copernicus": "/kaggle/working/ag_data/Copernicus",
        "/kaggle/working/ag_data/SRTM": "/kaggle/working/ag_data/SRTM"
    }
    processor = DEMProcessor(input_paths, output_paths, normalization_method='robust')
    processor.run()



%%time
"""Graficar normalizacion"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class NormalizationVisualizer:
    def __init__(self, report_path):
        self.report_df = pd.read_csv(report_path)
        self._validate_data()
        self.datasets = self.report_df['dataset'].unique()
        self.bands = self._get_valid_bands()

    def _validate_data(self):
        required_columns = {
            'dataset', 'band', 'filename', 'before_min', 'before_max', 'before_mean', 'before_std',
            'after_min', 'after_max', 'after_mean', 'after_std'
        }
        missing = required_columns - set(self.report_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in report: {missing}")

    def _get_valid_bands(self):
        valid_bands = []
        for band in self.report_df['band'].unique():
            band_df = self.report_df[self.report_df['band'] == band]
            if not band_df.empty and all(col in band_df.columns for col in ['before_mean', 'after_mean']):
                valid_bands.append(band)
        return valid_bands

    def _prepare_plot_data(self):
        metrics = ['min', 'max', 'mean', 'std']
        plot_data = []

        for dataset in self.datasets:
            dataset_df = self.report_df[self.report_df['dataset'] == dataset]

            for band in self.bands:
                band_df = dataset_df[dataset_df['band'] == band]
                if band_df.empty:
                    continue

                for metric in metrics:
                    before_col = f'before_{metric}'
                    after_col = f'after_{metric}'

                    if before_col in band_df.columns and after_col in band_df.columns:
                        plot_data.append({
                            'dataset': dataset,
                            'band': band,
                            'metric': metric,
                            'before': band_df[before_col].mean(),
                            'after': band_df[after_col].mean(),
                            'count': len(band_df)
                        })
        return pd.DataFrame(plot_data)

    def plot_normalization_comparison(self):
        plot_data = self._prepare_plot_data()
        if plot_data.empty:
            raise ValueError("No valid data to visualize")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Minimum Values Comparison',
                'Maximum Values Comparison',
                'Mean Values Comparison',
                'Standard Deviation Comparison'
            ),
            horizontal_spacing=0.15,
            vertical_spacing=0.2
        )

        colors = {
            'before': '#1f77b4',
            'after': '#ff7f0e'
        }

        metrics = ['min', 'max', 'mean', 'std']
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1

            metric_data = plot_data[plot_data['metric'] == metric]
            if metric_data.empty:
                continue

            x_labels = []
            x_positions = []
            pos_counter = 1

            for dataset in self.datasets:
                dataset_bands = metric_data[metric_data['dataset'] == dataset]['band'].unique()
                for band in dataset_bands:
                    x_labels.append(f"{dataset}<br>{band}")
                    x_positions.append(pos_counter)
                    pos_counter += 1
                pos_counter += 1

            for idx, (dataset, band) in enumerate(zip(metric_data['dataset'], metric_data['band'])):
                subset = metric_data[(metric_data['dataset'] == dataset) & (metric_data['band'] == band)]
                if subset.empty:
                    continue

                fig.add_trace(
                    go.Violin(
                        x=[x_positions[idx]] * len(subset),
                        y=subset['before'],
                        name=f'{dataset} {band} before',
                        legendgroup=f'{dataset}_{band}',
                        scalegroup=f'{dataset}_{band}_before',
                        side='negative',
                        line_color=colors['before'],
                        hoverinfo='y',
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        width=0.8
                    ),
                    row=row,
                    col=col
                )

                fig.add_trace(
                    go.Violin(
                        x=[x_positions[idx]] * len(subset),
                        y=subset['after'],
                        name=f'{dataset} {band} after',
                        legendgroup=f'{dataset}_{band}',
                        scalegroup=f'{dataset}_{band}_after',
                        side='positive',
                        line_color=colors['after'],
                        hoverinfo='y',
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        width=0.8
                    ),
                    row=row,
                    col=col
                )

            fig.update_xaxes(
                tickvals=x_positions,
                ticktext=x_labels,
                title='Dataset & Band',
                row=row,
                col=col
            )
            fig.update_yaxes(title='Value', row=row, col=col)

        fig.update_layout(
            title='<b>DEM Data Normalization Results Comparison</b>',
            title_x=0.5,
            violinmode='overlay',
            height=1200,
            width=1400,
            showlegend=True
        )
        return fig

    def plot_metric_distribution(self, metric='mean'):
        valid_metrics = ['min', 'max', 'mean', 'std']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")

        fig = go.Figure()
        colors = {
            'before': '#1f77b4',
            'after': '#ff7f0e'
        }

        for dataset in self.datasets:
            dataset_df = self.report_df[self.report_df['dataset'] == dataset]

            for band in self.bands:
                band_df = dataset_df[dataset_df['band'] == band]
                if band_df.empty:
                    continue

                fig.add_trace(
                    go.Violin(
                        x=[f"{dataset}<br>{band}"] * len(band_df),
                        y=band_df[f'before_{metric}'],
                        name=f'{dataset} {band} before',
                        legendgroup=f'{dataset}_{band}',
                        scalegroup=f'{dataset}_{band}',
                        side='negative',
                        line_color=colors['before'],
                        hoverinfo='y',
                        box_visible=True,
                        meanline_visible=True
                    )
                )

                fig.add_trace(
                    go.Violin(
                        x=[f"{dataset}<br>{band}"] * len(band_df),
                        y=band_df[f'after_{metric}'],
                        name=f'{dataset} {band} after',
                        legendgroup=f'{dataset}_{band}',
                        scalegroup=f'{dataset}_{band}',
                        side='positive',
                        line_color=colors['after'],
                        hoverinfo='y',
                        box_visible=True,
                        meanline_visible=True
                    )
                )

        fig.update_layout(
            title=f'<b>Distribution of {metric.capitalize()} Values</b>',
            title_x=0.5,
            violinmode='overlay',
            height=600,
            width=1200,
            yaxis_title=f'{metric.capitalize()} Value',
            xaxis_title='Dataset & Band',
            showlegend=True
        )
        return fig

    def show_summary_stats(self):
        if self.report_df.empty:
            return pd.DataFrame()

        stats = self.report_df.groupby(['dataset', 'band']).agg({
            'before_min': ['count', 'min', 'max', 'mean', 'std'],
            'after_min': ['min', 'max', 'mean', 'std'],
            'before_max': ['min', 'max', 'mean', 'std'],
            'after_max': ['min', 'max', 'mean', 'std'],
            'before_mean': ['mean', 'std'],
            'after_mean': ['mean', 'std'],
            'before_std': ['mean', 'std'],
            'after_std': ['mean', 'std']
        })

        print("ğŸ“Š Detailed Statistics Before/After Normalization:")
        return stats.round(4)

if __name__ == "__main__":
    try:
        visualizer = NormalizationVisualizer("/kaggle/working/ag_data/top_normalization_report.csv")
        display(visualizer.show_summary_stats())
        fig_comparison = visualizer.plot_normalization_comparison()
        fig_comparison.show()
        fig_distribution = visualizer.plot_metric_distribution('mean')
        fig_distribution.show()
    except Exception as e:
        print(f"Error: {str(e)}")



!pip install rioxarray rasterio -q


%%time
"""Exportar datos aumentados para sacar la bigpicture en otra vm"""
import zipfile
import os

# Ruta de la carpeta a comprimir
folder_path = '/kaggle/working/ag_data'
# Nombre del archivo ZIP de salida
zip_path = '/kaggle/working/ag_data.zip'

# Crear un archivo ZIP
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, folder_path)
            zipf.write(file_path, arcname)

print(f"Â¡Carpeta comprimida en {zip_path}!")


%%time
"""NO CORRER EN LA VM PRINCIPAL DEL FLUJO: Esta clase intenta conciliar las diferentes caracteristicas a 
traves de los origenes, no es muy limpio ni modular, cada conjunto es un mundo y eso agrega complejidad
para una conciliacion perfecta"""
import os
import gc
import tempfile
import warnings
import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.array as da
import rasterio
from rasterio.merge import merge
from rasterio.errors import RasterioIOError
from dask.distributed import Client, LocalCluster
from pathlib import Path
from glob import glob
from tqdm import tqdm
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

# === NormalizaciÃ³n ===
def normalize_block_cpu(block):
    if isinstance(block, np.ndarray):
        min_val = np.nanmin(block)
        max_val = np.nanmax(block)
        norm = (block - min_val) / (max_val - min_val + 1e-6)
        return norm.astype("float32")
    return block

def to_dask_array_normalized(xr_data, chunk_size=(1024, 1024)):
    darr = xr_data.data
    if isinstance(darr, np.ndarray):
        darr = da.from_array(darr, chunks=chunk_size)
    darr = darr.map_blocks(normalize_block_cpu, dtype="float32")
    return xr.DataArray(darr, dims=xr_data.dims, coords=xr_data.coords, attrs=xr_data.attrs)

# === Batching ===
def process_layer_in_batches(layer_name, tiffs, batch_size=10, target_crs="EPSG:4326", error_log=None, max_files=None):
    if max_files:
        tiffs = tiffs[:max_files]

    print(f"\nğŸ”„ Procesando capa: {layer_name} ({len(tiffs)} archivos)")
    temp_paths = []

    for i in range(0, len(tiffs), batch_size):
        batch = tiffs[i:i + batch_size]
        batch_arrays = []

        for tiff_path in tqdm(batch, desc=f"{layer_name} [{i}-{i + len(batch)}]"):
            try:
                da = rxr.open_rasterio(str(tiff_path), chunks={"x": 512, "y": 512}, masked=True)
                da = da.squeeze("band", drop=True)
                da = da.rio.reproject(target_crs)
                da = to_dask_array_normalized(da)
                da.name = layer_name
                batch_arrays.append(da)
            except Exception as e:
                msg = f"â�Œ {os.path.basename(tiff_path)}: {e}"
                tqdm.write(msg)
                if error_log:
                    with open(error_log, "a") as f:
                        f.write(msg + "\n")

        if not batch_arrays:
            continue

        try:
            batch_mosaic = xr.concat(batch_arrays, dim="band").mean("band", keep_attrs=True)
            if batch_mosaic.rio.transform()[1] < 0:
                print(f"ğŸ”� Corrigiendo orientaciÃ³n vertical del batch {i//batch_size}")
                batch_mosaic = batch_mosaic[::-1]

            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp").name
            batch_mosaic.rio.to_raster(temp_path)
            temp_paths.append(temp_path)
            del batch_arrays, batch_mosaic
            gc.collect()
        except Exception as e:
            msg = f"â�Œ Error procesando batch {i//batch_size}: {e}"
            print(msg)
            if error_log:
                with open(error_log, "a") as f:
                    f.write(msg + "\n")

    return temp_paths

# === FusiÃ³n ===
def merge_tiffs_to_one(tiff_paths, output_path):
    fixed_paths = []

    for p in tiff_paths:
        with rasterio.open(p) as src:
            if src.transform.e > 0:
                print(f"ğŸ”� Corrigiendo orientaciÃ³n vertical: {p}")
                flipped = src.read(1)[::-1, :]
                transform = rasterio.Affine(
                    src.transform.a,
                    src.transform.b,
                    src.transform.c,
                    src.transform.d,
                    -abs(src.transform.e),
                    src.transform.f + (src.height * abs(src.transform.e))
                )
                meta = src.meta.copy()
                meta.update({
                    "transform": transform,
                    "height": flipped.shape[0],
                    "width": flipped.shape[1]
                })
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp").name
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    dst.write(flipped, 1)
                fixed_paths.append(tmp_path)
            else:
                fixed_paths.append(p)

    datasets = [rasterio.open(p) for p in fixed_paths]
    mosaic, transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "count": 1
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)

    for ds in datasets:
        ds.close()
    for fp in fixed_paths:
        try:
            os.remove(fp)
        except:
            pass


# === AlineaciÃ³n ===
def align_rasters_to_reference(paths, ref_path):
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    aligned_arrays = []
    for path in paths:
        with rasterio.open(path) as src:
            data = np.empty(ref_shape, dtype="float32")
            reproject(
                source=src.read(1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
            aligned_arrays.append(data)
    return aligned_arrays, ref_crs, ref_transform

# === Escritura ===
def write_multiband_stack(arrays, ref_crs, ref_transform, out_path):
    if not arrays:
        print("â�Œ No hay datos para guardar.")
        return

    height, width = arrays[0].shape
    meta = {
        "driver": "GTiff",
        "count": len(arrays),
        "dtype": "float32",
        "height": height,
        "width": width,
        "crs": ref_crs,
        "transform": ref_transform
    }

    with rasterio.open(out_path, "w", **meta) as dst:
        for i, arr in enumerate(arrays):
            dst.write(arr, i + 1)
    print(f"\nâœ… Stack multibanda final guardado en: {out_path}")

# === EjecuciÃ³n principal ===
def main(batch_size=30, max_files=None, selected_layers=None):
    cluster = LocalCluster()
    client = Client(cluster)
    print("âœ… Dask en CPU activo:", client.dashboard_link)

    output_dir = Path("/kaggle/working/ag_data/big_picture")
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "error_log.txt"
    if error_log.exists(): error_log.unlink()

    layer_config = {
        "dsm": "/kaggle/input/tvring/ag_data/ALOS",
        "dem": "/kaggle/input/tvring/ag_data/Copernicus",
        "rh95": "/kaggle/input/tvring/ag_data/GEDI_Monthly",        
        "vh": "/kaggle/input/tvring/ag_data/os",
        "agb_mean": "/kaggle/input/tvring/ag_data/sm"
    }

    stack_paths = []
    for name, path in layer_config.items():
        if selected_layers and name not in selected_layers:
            continue
        tiffs = sorted(Path(path).rglob("*.tif"))
        print(f"\nğŸ”„ Procesando capa: {name}\nğŸ§¾ Archivos encontrados: {len(tiffs)}")
        if not tiffs: continue
        temp_paths = process_layer_in_batches(name, tiffs, batch_size=batch_size, error_log=str(error_log), max_files=max_files)
        if not temp_paths: continue
        merged = output_dir / f"{name}.tif"
        merge_tiffs_to_one(temp_paths, str(merged))
        stack_paths.append(merged)

    if stack_paths:
        ref_path = stack_paths[0]
        arrays, crs, transform = align_rasters_to_reference(stack_paths, ref_path)
        out_path = output_dir / "big_picture.tif"
        write_multiband_stack(arrays, crs, transform, str(out_path))
    else:
        print("â�Œ No se generÃ³ ningÃºn stack multibanda.")

if __name__ == "__main__":
    main(batch_size=30, max_files=None, selected_layers=None)

#["dsm", "dem", "rh95", "vh", "agb_mean"]


%%time
"""NO CORRER EN VM PRINCIPAL DEL FLUJO: Esta clase intenta conciliar la capa de biomasa"""

import os
import gc
import tempfile
import warnings
import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.array as da
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from dask.distributed import Client, LocalCluster
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === NormalizaciÃ³n ===
def normalize_block_cpu(block):
    if isinstance(block, np.ndarray):
        min_val = np.nanmin(block)
        max_val = np.nanmax(block)
        norm = (block - min_val) / (max_val - min_val + 1e-6)
        return norm.astype("float32")
    return block

def to_dask_array_normalized(xr_data, chunk_size=(1024, 1024)):
    darr = xr_data.data
    if isinstance(darr, np.ndarray):
        darr = da.from_array(darr, chunks=chunk_size)
    darr = darr.map_blocks(normalize_block_cpu, dtype="float32")
    return xr.DataArray(darr, dims=xr_data.dims, coords=xr_data.coords, attrs=xr_data.attrs)

# === Batching especÃ­fico para biomass ===
def process_biomass_batches(tiffs, batch_size=10, target_crs="EPSG:4326", error_log=None, max_files=None):
    if max_files:
        tiffs = tiffs[:max_files]
    print(f"\nğŸ”„ Procesando biomass ({len(tiffs)} archivos)")
    temp_paths = []

    for i in range(0, len(tiffs), batch_size):
        batch = tiffs[i:i + batch_size]
        batch_arrays = []

        for tiff_path in tqdm(batch, desc=f"biomass [{i}-{i + len(batch)}]"):
            try:
                da = rxr.open_rasterio(
                    str(tiff_path), chunks=(1, 1024, 1024), masked=True
                )

                if da.sizes.get("band", 1) > 1:
                    print(f"âš ï¸� {os.path.basename(tiff_path)} tiene mÃ¡s de una banda, se usarÃ¡ la primera")
                    da = da.isel(band=0, drop=True)
                else:
                    da = da.squeeze("band", drop=True)

                da = da.rio.reproject(target_crs, resolution=0.0005, resampling=Resampling.nearest)
                da = to_dask_array_normalized(da)
                da.name = "biomass"
                batch_arrays.append(da)

            except Exception as e:
                msg = f"â�Œ {os.path.basename(tiff_path)}: {e}"
                tqdm.write(msg)
                if error_log:
                    with open(error_log, "a") as f:
                        f.write(msg + "\n")

        if not batch_arrays:
            continue

        try:
            batch_mosaic = xr.concat(batch_arrays, dim="band").mean("band", keep_attrs=True)
            if batch_mosaic.rio.transform()[1] < 0:
                print(f"ğŸ”� Corrigiendo orientaciÃ³n vertical del batch {i//batch_size}")
                batch_mosaic = batch_mosaic[::-1]

            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp").name
            batch_mosaic.rio.to_raster(temp_path)
            temp_paths.append(temp_path)
            del batch_arrays, batch_mosaic
            gc.collect()
        except Exception as e:
            msg = f"â�Œ Error procesando batch {i//batch_size}: {e}"
            print(msg)
            if error_log:
                with open(error_log, "a") as f:
                    f.write(msg + "\n")
    return temp_paths

# === FusiÃ³n ===
def merge_tiffs_to_one(tiff_paths, output_path):
    fixed_paths = []
    for p in tiff_paths:
        with rasterio.open(p) as src:
            if src.transform.e > 0:
                print(f"ğŸ”� Corrigiendo orientaciÃ³n vertical: {p}")
                flipped = src.read(1)[::-1, :]
                transform = rasterio.Affine(
                    src.transform.a,
                    src.transform.b,
                    src.transform.c,
                    src.transform.d,
                    -abs(src.transform.e),
                    src.transform.f + (src.height * abs(src.transform.e))
                )
                meta = src.meta.copy()
                meta.update({
                    "transform": transform,
                    "height": flipped.shape[0],
                    "width": flipped.shape[1]
                })
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp").name
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    dst.write(flipped, 1)
                fixed_paths.append(tmp_path)
            else:
                fixed_paths.append(p)

    datasets = [rasterio.open(p) for p in fixed_paths]
    mosaic, transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "count": 1
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)

    for ds in datasets: ds.close()
    for fp in fixed_paths:
        try: os.remove(fp)
        except: pass

# === Ejecutar solo biomass ===
def main(batch_size=30, max_files=None):
    cluster = LocalCluster()
    client = Client(cluster)
    print("âœ… Dask en CPU activo:", client.dashboard_link)

    input_path = "/kaggle/input/tvring/ag_data/s3"
    output_dir = Path("/kaggle/working/ag_data/big_picture")
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log_biomass.txt"
    if error_log.exists(): error_log.unlink()

    tiffs = sorted(Path(input_path).rglob("*.tif"))
    print(f"\nğŸ”„ Procesando capa: biomass\nğŸ§¾ Archivos encontrados: {len(tiffs)}")

    if not tiffs:
        print("â�Œ No se encontraron archivos.")
        return

    temp_paths = process_biomass_batches(tiffs, batch_size=batch_size, max_files=max_files, error_log=str(error_log))
    if not temp_paths:
        print("â�Œ No se generaron mosaicos temporales.")
        return

    output_file = output_dir / "biomass.tif"
    merge_tiffs_to_one(temp_paths, str(output_file))
    print(f"\nâœ… Archivo final guardado: {output_file}")

if __name__ == "__main__":
    main(batch_size=30, max_files=None)



%%time
"""Graficar capas tiff por area de estudio"""
import os
import rasterio
import numpy as np
import folium
import geopandas as gpd
import matplotlib
from folium.raster_layers import ImageOverlay
from folium.plugins import MiniMap
from pathlib import Path
import concurrent.futures
import warnings


class GeoTIFFMapExplorerBatch:
    def __init__(self, tif_paths, geojson_overlay, output_path, batch_size, max_workers):
        self.tif_paths = [Path(p) for p in tif_paths]
        self.geojson_overlay = geojson_overlay
        self.output_path = output_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.layers = []
        self.map = None

    def _get_bounds(self, tif_path):
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    def _choose_colormap(self, name):
        name = name.lower()
        if "agb" in name or "biomass" in name:
            return "YlGn"
        elif "dem" in name or "dsm" in name:
            return "terrain"
        elif "vh" in name:
            return "inferno"
        elif "rh95" in name:
            return "plasma"
        else:
            return "viridis"

    def _read_and_process_large_raster(self, tif_path, cmap_name='viridis'):
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                blocks = list(src.block_windows(1))  # ventana por bloque
                height, width = src.height, src.width

                full_image = np.zeros((height, width, 3), dtype=np.uint8)

                for idx, (ji, window) in enumerate(blocks):
                    block = src.read(1, window=window)
                    if nodata is not None:
                        block = np.where(block == nodata, np.nan, block)

                    valid = block[~np.isnan(block)]
                    if len(valid) == 0:
                        continue

                    p2, p98 = np.percentile(valid, [2, 98])
                    norm = np.clip((block - p2) / (p98 - p2 + 1e-10), 0, 1)
                    cmap = matplotlib.colormaps.get_cmap(cmap_name)
                    colored = cmap(norm)[..., :3] * 255

                    row_off, col_off = window.row_off, window.col_off
                    rows, cols = window.height, window.width
                    full_image[row_off:row_off+rows, col_off:col_off+cols] = colored.astype(np.uint8)

                bounds = self._get_bounds(tif_path)

                layer = ImageOverlay(
                    image=full_image,
                    bounds=bounds,
                    name=tif_path.stem,
                    opacity=0.85,
                    interactive=True,
                    mercator_project=True,
                    show=tif_path.stem == "agb_mean"
                )
                print(f"âœ… Procesado (por bloques): {tif_path.name}")
                return layer

        except Exception as e:
            print(f"â�Œ Error en bloques para {tif_path.name}: {e}")
            return None

    def _add_tiff_layers(self):
        batches = [self.tif_paths[i:i + self.batch_size] for i in range(0, len(self.tif_paths), self.batch_size)]

        for batch in batches:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for tif_path in batch:
                    cmap = self._choose_colormap(tif_path.stem)
                    futures.append(executor.submit(self._read_and_process_large_raster, tif_path, cmap))
                for f in concurrent.futures.as_completed(futures):
                    layer = f.result()
                    if layer:
                        self.layers.append(layer)

    def _add_geojson_overlay(self):
        if not self.geojson_overlay:
            return
        gdf = gpd.read_file(self.geojson_overlay)
        overlay = folium.FeatureGroup(name="Ã�reas CrÃ­ticas", show=True)

        for _, row in gdf.iterrows():
            folium.GeoJson(
                data=row['geometry'],
                style_function=lambda x: {
                    'color': 'red',
                    'weight': 2,
                    'fillOpacity': 0.2
                },
                tooltip=folium.Tooltip(f"ID: {row.get('id', 'N/A')}<br>Ã�rea: {row.get('area_km2', 0):.2f} kmÂ²")
            ).add_to(overlay)

        overlay.add_to(self.map)

    def create_map(self):
        bounds = self._get_bounds(self.tif_paths[0])
        center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        self.map = folium.Map(location=center, zoom_start=7, control_scale=True)

        self._add_tiff_layers()

        for layer in self.layers:
            layer.add_to(self.map)

        self._add_geojson_overlay()

        folium.LayerControl(collapsed=False).add_to(self.map)
        MiniMap().add_to(self.map)

        title = '''<div style="position: fixed; top: 10px; left: 50%;
            transform: translateX(-50%); z-index:9999;
            background: white; padding: 8px; border: 1px solid gray;
            border-radius: 5px; font-weight: bold;">
            Map: GeoTIFF Layers and Critical Areas</div>'''
        self.map.get_root().html.add_child(folium.Element(title))

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.map.save(self.output_path)
        print(f"âœ… Mapa guardado en: {self.output_path}")

    def run(self):
        self.create_map()
        self.save()

if __name__ == "__main__":
    tif_files = [
        "/kaggle/input/amaztest/big_picture/agb_mean.tif",
        "/kaggle/input/amaztest/big_picture/biomass.tif",
        "/kaggle/input/amaztest/big_picture/dem.tif",
        #"/kaggle/input/amaztest/big_picture/dsm.tif", TIFF GIGANTE
        "/kaggle/input/amaztest/big_picture/rh95.tif",
        "/kaggle/input/amaztest/big_picture/vh.tif"
    ]

    geojson_overlay = "/kaggle/working/amazon_data/critical_areas.geojson"
    output_path = "/kaggle/working/amazon_maps/big_picture_map.html"

    explorer = GeoTIFFMapExplorerBatch(
        tif_paths=tif_files,
        geojson_overlay=geojson_overlay,
        output_path=output_path,
        batch_size=1,         # 1 archivo a la vez
        max_workers=3         # 3 hilos paralelos
    )
    explorer.run()





from IPython.display import Image, display

display(Image(filename='/kaggle/input/amaztest/my_imgs/bigpicture.png'))


%%time
"""Una red trivial"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 1. Generar datos sintÃ©ticos ---------------------------------

class SyntheticTiffDataset(Dataset):
    def __init__(self, num_samples=10, image_size=(3, 64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.tiff = torch.randn(image_size)  # [3, 64, 64]
        self.points = torch.randint(0, 64, (num_samples, 2))  # [10, 2]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x, y = self.points[idx]
        window_size = 16
        half = window_size // 2
        
        # Calcular bordes con clamp para evitar Ã­ndices negativos
        x_min = max(0, x - half)
        x_max = min(self.image_size[1], x + half)
        y_min = max(0, y - half)
        y_max = min(self.image_size[2], y + half)
        
        # Extraer parche
        patch = self.tiff[:, x_min:x_max, y_min:y_max]
        
        # Padding si el parche es mÃ¡s pequeÃ±o que 16x16
        pad_x = max(0, window_size - (x_max - x_min))
        pad_y = max(0, window_size - (y_max - y_min))
        
        # Aplicar padding simÃ©trico
        patch = F.pad(patch, (pad_y // 2, pad_y - pad_y // 2, 
                               pad_x // 2, pad_x - pad_x // 2), 
                      mode='constant', value=0)
        
        return patch, 1  # Label 1 = punto positivo

# 2. Arquitectura Mini-CNN ---------------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)  # 3 bandas de entrada
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 7 * 7, 1)  # 7x7 por reducciÃ³n de tamaÃ±o
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [B, 8, 14, 14]
        x = self.pool(x)               # [B, 8, 7, 7]
        x = x.view(x.size(0), -1)      # [B, 392]
        return torch.sigmoid(self.fc(x))

# 3. Entrenamiento -----------------------------------------
dataset = SyntheticTiffDataset()
dataloader = DataLoader(dataset, batch_size=2)

model = TinyModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2):
    for patches, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. Prueba predictiva ------------------------------------
test_patch, _ = dataset[0]
with torch.no_grad():
    pred = model(test_patch.unsqueeze(0))
print(f"\nPredicciÃ³n para un punto: {pred.item():.2f}")  # Debe ser ~1 (positivo)


%%time
"""Mapear puntos unicos como bbox para usarlos como etiqueta (tags set)"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import geopandas as gpd

class ParquetAnalyzer:
    def __init__(self, geojson_path):
        self.geojson_path = geojson_path
        self.gdf = gpd.read_file(geojson_path)
        self.parquet_info = {}
        self.unique_locations = None
    
    def analyze_parquet_files(self):
        """Identifica y cuenta archivos Parquet por grupo 'ffather'"""
        print("\nAnalyzing files Parquet...")
        self.parquet_info = {}
        
        # Agrupar por 'ffather' y contar archivos Parquet Ãºnicos
        for ffather, group in tqdm(self.gdf.groupby('ffather'), desc="Reading available files"):
            unique_parquets = group['dataset'].apply(lambda x: Path(x).stem).unique()
            self.parquet_info[ffather] = {
                'count': len(unique_parquets),
                'files': sorted(unique_parquets)
            }
    
    def count_unique_locations(self):
        """Calcula ubicaciones Ãºnicas basadas en geometrÃ­a"""
        print("\nCalculating unique locations...")
        locations = []
        
        for idx, row in tqdm(self.gdf.iterrows(), total=len(self.gdf), desc="Processing geometries"):
            if row.geometry.geom_type == 'Point':
                locations.append((row.geometry.x, row.geometry.y))
            else:
                # Para polÃ­gonos, usamos el centroide
                centroid = row.geometry.centroid
                locations.append((centroid.x, centroid.y))
        
        self.unique_locations = pd.DataFrame(locations, columns=['lon', 'lat']).drop_duplicates()
    
    def print_summary(self):
        """Muestra resumen de los anÃ¡lisis"""
        print("\n=== Analysis summary ===")
        print(f"\nTotal features in GeoJSON: {len(self.gdf)}")
        
        # Resumen de archivos Parquet
        print("\nParquet files by ffather:")
        for ffather, info in self.parquet_info.items():
            print(f" - {ffather[:50]}...: {info['count']} files (ej: {info['files'][0]}...)")
        
        # Resumen de ubicaciones Ãºnicas
        if self.unique_locations is not None:
            print(f"\nUnique locations found: {len(self.unique_locations)}")
            print("Example locations:")
            print(self.unique_locations.head())
    
    def run_analysis(self):
        """Ejecuta todos los anÃ¡lisis"""
        self.analyze_parquet_files()
        self.count_unique_locations()
        self.print_summary()
        return self

# ===== Uso =====
if __name__ == "__main__":
    geojson_path = '/kaggle/working/min_dots/mini_tags.geojson'
    analyzer = ParquetAnalyzer(geojson_path)
    analyzer.run_analysis()


%%time
"""Se divide las etiquetas para alimentar el modelo"""
import geopandas as gpd
import random
from typing import List, Tuple

class QuadrantSplitter:
    def __init__(self, geojson_path: str, test_ratio: float = 0.2, seed: int = 42):
        self.geojson_path = geojson_path
        self.test_ratio = test_ratio
        self.seed = seed
        self.gdf = gpd.read_file(geojson_path)
        self.train_quadrants = None
        self.test_quadrants = None

    def split(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        random.seed(self.seed)

        if 'id_left' not in self.gdf.columns or 'ffather' not in self.gdf.columns:
            raise KeyError("El GeoJSON debe contener las columnas 'id_left' y 'ffather'.")

        unique_ids = self.gdf['id_left'].unique().tolist()
        random.shuffle(unique_ids)

        split_idx = int(len(unique_ids) * (1 - self.test_ratio))
        train_ids = unique_ids[:split_idx]
        test_ids = unique_ids[split_idx:]

        self.train_quadrants = self.gdf[self.gdf['id_left'].isin(train_ids)]
        self.test_quadrants = self.gdf[self.gdf['id_left'].isin(test_ids)]

        return self.train_quadrants, self.test_quadrants

    def print_summary(self):
        total = len(self.gdf)
        train_count = len(self.train_quadrants)
        test_count = len(self.test_quadrants)

        total_tags = self.gdf['ffather'].nunique()
        train_tags = self.train_quadrants['ffather'].nunique()
        test_tags = self.test_quadrants['ffather'].nunique()
        
        print("\n=== Tags division ===")
        print(f"Total locations: {total}")
        print(f"Number in training: {train_count} locations")
        print(f"Number in test: {test_count} locations")
 

    def get_test_ids(self) -> List:
        if self.test_quadrants is not None:
            return self.test_quadrants['id_left'].tolist()
        else:
            raise ValueError("Primero debes ejecutar el mÃ©todo split().")

    def export_test_ids(self, output_path: str):
        if self.test_quadrants is not None:
            self.test_quadrants[['id_left']].to_csv(output_path, index=False)
            print(f"\nIDs del set de prueba exportados a: {output_path}")
        else:
            raise ValueError("Test set no generado. Ejecuta primero split().")

# ===== Uso desde script principal =====
if __name__ == "__main__":
    geojson_path = "/kaggle/working/min_dots/mini_tags.geojson"
    output_test_ids = "/kaggle/working/ag_data/test_quadrants.csv"

    splitter = QuadrantSplitter(geojson_path, test_ratio=0.3)
    splitter.split()
    splitter.print_summary()
    splitter.export_test_ids(output_test_ids)



!pip install gudhi -q



%%time
"""Cubical complex, persistence"""
from gudhi import CubicalComplex
import numpy as np

class CubicalPersistenceExample:
    def __init__(self, matrix):
        self.matrix = matrix
        self.cubical_complex = None
        self.persistence = []

    def build_complex(self):
        """Construye el CubicalComplex a partir de la matriz de entrada."""
        # Se pasa como top-dimensional cells
        self.cubical_complex = CubicalComplex(top_dimensional_cells=self.matrix)

    def compute_persistence(self, homology_coeff_field=11, min_persistence=0.0):
        """Calcula la persistencia del complejo."""
        if self.cubical_complex is None:
            raise ValueError("Primero construye el complejo con build_complex().")

        self.persistence = self.cubical_complex.persistence(
            homology_coeff_field=homology_coeff_field,
            min_persistence=min_persistence
        )

    def display_persistence(self):
        """Muestra los intervalos de persistencia agrupados por dimensiÃ³n."""
        print("Intervalos de persistencia:")
        for dim, birth, death in [(p[0], p[1][0], p[1][1]) for p in self.persistence]:
            print(f"Dim {dim}: [{birth}, {death})")

if __name__ == "__main__":
    # Una matriz 3x3 simple que simula una imagen binaria (escalada)
    matrix = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0]
    ])

    # Instanciar el ejemplo
    example = CubicalPersistenceExample(matrix)

    # Construir el complejo cubical
    example.build_complex()

    # Calcular la persistencia
    example.compute_persistence()

    # Mostrar resultados
    example.display_persistence()




%%time
"""Simplicial Complex, persistence"""
import numpy as np
import matplotlib.pyplot as plt
from gudhi import RipsComplex, SimplexTree, plot_persistence_barcode, plot_persistence_diagram, plot_persistence_density


def generate_chaotic_data(n_points=100, noise_level=0.3, seed=None):
    """Genera datos caÃ³ticos: puntos distribuidos al azar con ruido."""
    if seed is not None:
        np.random.seed(seed)

    # Puntos base formando un cÃ­rculo imperfecto (estructura difÃ­cil de ver por el ojo humano)
    theta = np.linspace(0, 2 * np.pi, n_points // 2, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, noise_level, size=(n_points // 2, 2))

    # Puntos aleatorios adicionales (caos)
    random_points = np.random.uniform(-2, 2, size=(n_points // 2, 2))

    return np.vstack((circle, random_points))


class PersistenceVisualizer:
    def __init__(self, points):
        self.points = points
        self.simplex_tree = None
        self.persistence = []

    def build_rips_complex(self, max_edge_length=2.0, max_dimension=2):
        """Construye el complejo de Rips."""
        rips_complex = RipsComplex(points=self.points, max_edge_length=max_edge_length)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    def compute_persistence(self, homology_coeff_field=11, min_persistence=0.0):
        """Calcula la persistencia del complejo."""
        if self.simplex_tree is None:
            raise ValueError("Primero construye el complejo con build_rips_complex().")
        self.persistence = self.simplex_tree.persistence(homology_coeff_field=homology_coeff_field,
                                                         min_persistence=min_persistence)

    def plot_barcodes(self):
        """Dibuja los barcodes."""
        print("Mostrando Barcode...")
        plot_persistence_barcode(persistence=self.persistence)
        plt.title("Barcode de Persistencia")
        plt.show()

    def plot_diagram(self):
        """Dibuja el diagrama de persistencia."""
        print("Mostrando Diagrama de Persistencia...")
        plot_persistence_diagram(persistence=self.persistence)
        plt.title("Diagrama de Persistencia")
        plt.show()

    def plot_density(self):
        """Dibuja la densidad de persistencia en dimensiÃ³n 1."""
        print("Mostrando Densidad de Persistencia...")
        persistence_dim1 = [p[1] for p in self.persistence if p[0] == 1]
        if len(persistence_dim1) == 0:
            print("No hay puntos en dimensiÃ³n 1 para graficar densidad.")
            return
        plot_persistence_density(persistence=persistence_dim1)
        plt.title("Densidad de Persistencia (DimensiÃ³n 1)")
        plt.show()


if __name__ == "__main__":
    # Generar datos caÃ³ticos
    chaotic_points = generate_chaotic_data(n_points=150, noise_level=0.4, seed=42)

    # Mostrar los puntos generados
    plt.scatter(chaotic_points[:, 0], chaotic_points[:, 1], s=10, color='black')
    plt.title("Datos CaÃ³ticos Generados")
    plt.axis('equal')
    plt.show()

    # Instanciar visualizador
    pv = PersistenceVisualizer(chaotic_points)

    # Construir complejo
    pv.build_rips_complex(max_edge_length=1.8, max_dimension=2)

    # Calcular persistencia
    pv.compute_persistence(min_persistence=0.2)

    # Graficar resultados
    pv.plot_barcodes()
    pv.plot_diagram()
    pv.plot_density()


%%time 
"""Bottleneck distance"""

import gudhi

# Dos diagramas de persistencia (dimensiÃ³n 1)
diag1 = [[2.7, 3.7], [9.6, 14.0], [34.2, 34.974], [3.0, float('inf')]]
diag2 = [[2.8, 4.45], [9.5, 14.1], [3.2, float('inf')]]

# Distancia de Bottleneck aproximada (con e = 0.1)
approx_dist = gudhi.bottleneck_distance(diag1, diag2, e=0.1)
print(f"Bottleneck distance (aproximado): {approx_dist:.2f}")

# Distancia de Bottleneck exacta (e = None -> exacto)
exact_dist = gudhi.bottleneck_distance(diag1, diag2)
print(f"Bottleneck distance (exacto): {exact_dist:.2f}")


%%time 
"""Cohomology"""

import numpy as np
import matplotlib.pyplot as plt
from gudhi import RipsComplex, SimplexTree


def generate_circle_with_noise(n_points=100, noise_level=0.1, seed=42):
    """Genera puntos en forma de cÃ­rculo imperfecto."""
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, noise_level, size=(n_points, 2))
    return circle


def plot_persistence_diagram(persistence, title="Diagrama de Persistencia"):
    """Dibuja un diagrama de persistencia bÃ¡sico."""
    birth_death = np.array([p[1] for p in persistence if p[0] == 1])

    if len(birth_death) > 0:
        plt.scatter(birth_death[:, 0], birth_death[:, 1], label="Ciclo", alpha=0.7)
        plt.plot([0, 10], [0, 10], "k--", lw=1)  # Diagonal
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title(title)
        plt.legend()
        plt.axis("equal")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    else:
        print("No se encontraron ciclos persistentes.")


if __name__ == "__main__":
    # Generar datos: cÃ­rculo imperfecto
    points = generate_circle_with_noise(n_points=80, noise_level=0.1)

    # Construir complejo de Rips
    rips_complex = RipsComplex(points=points, max_edge_length=1.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Calcular persistencia usando cohomologÃ­a
    persistence = simplex_tree.persistence(homology_coeff_field=11, min_persistence=0.1)

    # Mostrar resultado
    print("Persistencia calculada:")
    for dim, (birth, death) in persistence:
        print(f"Dim {dim}: [{birth:.3f}, {death:.3f})")

    # Graficar diagrama de persistencia
    plot_persistence_diagram(persistence, title="Persistencia por CohomologÃ­a")




%%time
"""Witness complex, filter recontructor"""


import numpy as np
import matplotlib.pyplot as plt
from gudhi import EuclideanWitnessComplex, plot_persistence_barcode, plot_persistence_diagram
from gudhi.simplex_tree import SimplexTree


def generate_chaotic_data(n_points=200, noise_level=0.3, seed=None):
    """Genera un cÃ­rculo imperfecto con ruido."""
    if seed is not None:
        np.random.seed(seed)
    # CÃ­rculo imperfecto
    theta = np.linspace(0, 2 * np.pi, n_points // 2)
    circle = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, noise_level, size=(n_points // 2, 2))
    # Puntos aleatorios
    random_points = np.random.uniform(-2, 2, size=(n_points // 2, 2))
    return np.vstack((circle, random_points))


class WitnessPersistenceVisualizer:
    def __init__(self, witnesses, num_landmarks=20):
        self.witnesses = witnesses
        self.num_landmarks = num_landmarks
        self.landmarks = self._pick_landmarks()
        self.simplex_tree = None
        self.persistence = []

    def _pick_landmarks(self):
        indices = np.random.choice(len(self.witnesses), self.num_landmarks, replace=False)
        return [self.witnesses[i].tolist() for i in indices]

    def build_witness_complex(self, max_alpha_square=2.0):
        ewc = EuclideanWitnessComplex(witnesses=self.witnesses.tolist(), landmarks=self.landmarks)
        self.simplex_tree = ewc.create_simplex_tree(max_alpha_square=max_alpha_square)

    def compute_persistence(self, homology_coeff_field=11, min_persistence=0.1):
        if self.simplex_tree is None:
            raise ValueError("Primero construye el complejo con build_witness_complex().")
        self.persistence = self.simplex_tree.persistence(homology_coeff_field=homology_coeff_field,
                                                         min_persistence=min_persistence)

    def plot_barcodes(self):
        print("Mostrando Barcode...")
        plot_persistence_barcode(persistence=self.persistence)
        plt.title("Barcode de Persistencia (Witness Complex)")
        plt.show()

    def plot_diagram(self):
        print("Mostrando Diagrama de Persistencia...")
        plot_persistence_diagram(persistence=self.persistence)
        plt.title("Diagrama de Persistencia (Witness Complex)")
        plt.show()


if __name__ == "__main__":
    # Generar datos caÃ³ticos
    chaotic_points = generate_chaotic_data(n_points=200, noise_level=0.4, seed=42)

    # Mostrar puntos
    plt.scatter(chaotic_points[:, 0], chaotic_points[:, 1], s=10, color='black', alpha=0.7)
    plt.title("Datos CaÃ³ticos")
    plt.axis('equal')
    plt.show()

    # Instanciar visualizador
    pv = WitnessPersistenceVisualizer(chaotic_points, num_landmarks=25)

    # Construir complejo
    pv.build_witness_complex(max_alpha_square=2.0)

    # Calcular persistencia
    pv.compute_persistence(min_persistence=0.2)

    # Graficar resultados
    pv.plot_barcodes()
    pv.plot_diagram()


%%time
"""Mapper complex, filter reconstruction"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gudhi.cover_complex import MapperComplex
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

def generate_chaotic_data(n_points=300, noise_level=0.3, seed=42):
	"""Genera un cÃ­rculo imperfecto + ruido aleatorio."""
	np.random.seed(seed)
	theta = np.linspace(0, 2 * np.pi, n_points // 2)
	circle = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, noise_level, size=(n_points // 2, 2))
	random = np.random.uniform(-3, 3, size=(n_points // 2, 2))
	return np.vstack((circle, random))

class MapperVisualizer:
	def __init__(self, resolution=10, gain=0.5):
		self.resolution = resolution
		self.gain = gain
		self.mapper_complex = None
		self.data = None
		self.projected = None

	def build_mapper(self, data, filter_function="pca"):
		"""Construye el complejo Mapper usando una funciÃ³n filtro."""
		if filter_function == "pca":
			self.projected = PCA(n_components=1).fit_transform(data)
		elif filter_function == "x":
			self.projected = data[:, 0].reshape(-1, 1)
		elif filter_function == "y":
			self.projected = data[:, 1].reshape(-1, 1)
		else:
			raise ValueError("Filtro desconocido. Opciones vÃ¡lidas: 'pca', 'x', 'y'")

		clusterer = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=0.5,
			linkage='single'
		)

		self.mapper_complex = MapperComplex(
			input_type='point cloud',
			min_points_per_node=3,
			resolutions=np.array([self.resolution]),
			gains=np.array([self.gain]),
			clustering=clusterer,
			verbose=False
		)

		self.mapper_complex.fit(X=data, filters=self.projected)
		self.data = data

	def plot_mapper_graph(self):
		"""Dibuja el grafo Mapper usando networkx."""
		if self.mapper_complex is None:
			raise RuntimeError("Primero construye el complejo con build_mapper().")

		graph = self.mapper_complex.get_networkx()
		print(f"NÃºmero de nodos: {graph.number_of_nodes()}")
		print(f"NÃºmero de aristas: {graph.number_of_edges()}")

		plt.figure(figsize=(10, 6))
		pos = nx.spring_layout(graph)
		nx.draw(
			graph,
			pos,
			with_labels=True,
			node_size=200,
			font_size=8,
			alpha=0.8
		)
		plt.title("Mapper Grafo (Datos SintÃ©ticos CaÃ³ticos)")
		plt.show()

	def plot_original_data(self):
		"""Muestra los datos originales para comparar."""
		plt.figure(figsize=(8, 6))
		plt.scatter(
			self.data[:, 0],
			self.data[:, 1],
			s=10,
			color='black',
			alpha=0.7
		)
		plt.title("Datos CaÃ³ticos Originales")
		plt.axis('equal')
		plt.show()

if __name__ == "__main__":
	chaotic_data = generate_chaotic_data(n_points=300, noise_level=0.3, seed=42)
	mapper_vis = MapperVisualizer(resolution=10, gain=0.5)
	mapper_vis.build_mapper(chaotic_data, filter_function="pca")
	mapper_vis.plot_original_data()
	mapper_vis.plot_mapper_graph()


%%time
"""Rips complex, filter recontruction"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gudhi import RipsComplex, SimplexTree, plot_persistence_diagram, plot_persistence_barcode


def generate_chaotic_data(n_points=100, noise_level=0.3, seed=42):
    """Genera puntos caÃ³ticos con un cÃ­rculo imperfecto."""
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points // 2)
    circle = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.normal(0, noise_level, size=(n_points // 2, 2))
    random = np.random.uniform(-2, 2, size=(n_points // 2, 2))
    return np.vstack((circle, random))


class RipsPersistenceVisualizer:
    def __init__(self, data, max_edge_length=2.0, max_dimension=2):
        self.data = data
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.simplex_tree = None
        self.persistence = []

    def build_rips_complex(self):
        """Construye el complejo de Rips."""
        rips_complex = RipsComplex(points=self.data, max_edge_length=self.max_edge_length)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)

    def compute_persistence(self, homology_coeff_field=11, min_persistence=0.1):
        """Calcula la persistencia del complejo."""
        if self.simplex_tree is None:
            raise ValueError("Primero construye el complejo con build_rips_complex().")
        self.persistence = self.simplex_tree.persistence(homology_coeff_field=homology_coeff_field,
                                                         min_persistence=min_persistence)

    def plot_barcodes(self):
        """Dibuja los barcodes."""
        print("Mostrando Barcode...")
        plot_persistence_barcode(self.persistence)
        plt.title("Barcode de Persistencia (Rips)")
        plt.show()

    def plot_diagram(self):
        """Dibuja el diagrama de persistencia."""
        print("Mostrando Diagrama de Persistencia...")
        plot_persistence_diagram(self.persistence)
        plt.title("Diagrama de Persistencia (Rips)")
        plt.show()


if __name__ == "__main__":
    chaotic_data = generate_chaotic_data(n_points=150, noise_level=0.4, seed=42)

    plt.scatter(chaotic_data[:, 0], chaotic_data[:, 1], s=10, color='black', alpha=0.7)
    plt.title("Datos CaÃ³ticos Generados")
    plt.axis('equal')
    plt.show()

    rpv = RipsPersistenceVisualizer(chaotic_data, max_edge_length=1.8, max_dimension=2)
    rpv.build_rips_complex()
    rpv.compute_persistence(min_persistence=0.2)

    rpv.plot_barcodes()
    rpv.plot_diagram()


%%time
"""Delanuay complex, filter reconstruction"""

import numpy as np
import matplotlib.pyplot as plt
from gudhi import DelaunayComplex


class DelaunayVisualizer:
    def __init__(self, points):
        self.points = np.array(points)
        self.delaunay_complex = DelaunayComplex(points=self.points)
        self.simplex_tree = None

    def build_complex(self):
        """Construye el simplex tree del complejo de Delaunay."""
        self.simplex_tree = self.delaunay_complex.create_simplex_tree()
        print(f"DimensiÃ³n del complejo: {self.simplex_tree.dimension()}")
        print(f"NÃºmero de sÃ­mplices: {self.simplex_tree.num_simplices()}")

    def plot_delaunay(self):
        """Dibuja los triÃ¡ngulos de Delaunay."""
        if self.simplex_tree is None:
            raise RuntimeError("Primero construye el complejo con build_complex().")

        # Extraer los triÃ¡ngulos (sÃ­mplices de dimensiÃ³n 2)
        triangles = []
        for simplex, _ in self.simplex_tree.get_skeleton(2):
            if len(simplex) == 3:
                triangles.append([self.points[i] for i in simplex])

        # Dibujar puntos y triÃ¡ngulos
        plt.figure(figsize=(8, 8))
        plt.triplot(self.points[:, 0], self.points[:, 1], np.array([[i for i in s] for s in self.simplex_tree.get_skeleton(2) if len(s) == 3]), c='blue', linewidth=0.7)
        plt.scatter(self.points[:, 0], self.points[:, 1], c='red', s=20, zorder=5)
        plt.title("Complejo de Delaunay")
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


if __name__ == "__main__":
    # Generar puntos aleatorios
    np.random.seed(42)
    num_points = 20
    points = np.random.rand(num_points, 2) * 10  # Puntos entre [0,10] x [0,10]

    # Instanciar visualizador
    delaunay_vis = DelaunayVisualizer(points)

    # Construir complejo
    delaunay_vis.build_complex()

    # Graficar
    delaunay_vis.plot_delaunay()



%%time
import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# FunciÃ³n para cargar el secreto desde Kaggle
def load_secret(secret_label):
    user_secrets = UserSecretsClient()
    return user_secrets.get_secret(secret_label)

# Cargar la clave API
openai_key = load_secret("openk3y")  

# Crear cliente OpenAI
client = OpenAI(api_key=openai_key)

# Verificar conexiÃ³n listando modelos
try:
    models = client.models.list()
    print("âœ… ConexiÃ³n exitosa. Modelos disponibles:")
    for model in models.data:
        print("-", model.id)
except Exception as e:
    print("â�Œ Error al conectar con OpenAI:", str(e))


from IPython.display import display, Math

# FÃ³rmula trivial: cuadrado de una suma
display(Math(r'a+b^2 = a^2 + 2ab + b^2'))

# Otra prueba rÃ¡pida
display(Math(r'\text{Hola, } \LaTeX'))


%%time
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown


class GPTTextQuery:
    def __init__(self, model, max_tokens, temperature, secret_label):
        """
        Inicializa el cliente OpenAI con parÃ¡metros ajustables
        """
        # Cargar clave API desde Kaggle Secrets
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret(secret_label)
        self.client = OpenAI(api_key=api_key)

        # ParÃ¡metros del modelo
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def query(self, prompt):
        """
        EnvÃ­a un prompt al modelo y devuelve la respuesta
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error al comunicarse con OpenAI: {e}"

    def display_query(self, prompt):
        """
        EnvÃ­a el prompt y muestra la respuesta interpretando Markdown + LaTeX
        """
        response = self.query(prompt)
        display(Markdown(response))


if __name__ == "__main__":
    
    # ConfiguraciÃ³n personalizable
    MODEL_NAME = "gpt-4.1-mini-2025-04-14"
    MAX_TOKENS = 10000
    TEMPERATURE = 0.7
    SECRET_LABEL = "openk3y"  # nombre del env

    # Crear instancia del consultor de GPT
    gpt = GPTTextQuery(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        secret_label=SECRET_LABEL
    )

    # Prompt bÃ¡sico para prueba
    prompt = """You are an expert in mathematical modeling and topology with a deep understanding of complex data structures and advanced modeling techniques. I need your critical and impartial assessment of the following scenarios and methodologies related to the modeling and analysis of 3D archaeological data, particularly focusing on the Amazon region.### Context and Scenarios#### 1. Data Structure and Topological ChallengesWe are dealing with raster layers \( \mathcal{L} = \{L_i\}_{i=1}^6 \), where each layer defines a topological space \( (X_i, \tau_i) \). Here, \( X_i \subseteq \mathbb{R}^2 \) represents the spatial domain, and \( \tau_i \) is the topology induced by the resolution and rotation of \( L_i \). The central problem is the lack of a homeomorphism \( h: X_i \to X_j \) that preserves the spatial structure between layers, implying that \( \mathcal{L} \) is not a vector bundle (no common base space).#### 2. Methodologies for Robustness and ValidationWe have implemented several strategies to ensure the robustness and validity of our models:- **Cross-Validation by Quadrants**: Modified k-fold cross-validation where "folds" are groups of geographic quadrants to prevent the model from memorizing local patterns.- **Geographic Augmentation**: Techniques such as rotations, reflections, and controlled translations to simulate different terrain orientations and enhance model generalization.- **Synthetic Over-Sampling**: Generating synthetic quadrants using Gaussian Processes to interpolate features from neighboring quadrants.- **Stress Testing**: Introducing adversarial noise to test layers and measuring performance degradation to understand model robustness.- **Spatial Error Analysis**: Using metrics like Shannon Entropy and Moran's I to detect spatial dependencies and localized overfitting.#### 3. Advanced Techniques and TechnologiesWe are leveraging advanced technologies to handle complex and non-connected data structures:- **Graph Neural Networks (GNNs)**: For modeling relationships between incongruent components and propagating information between layers.- **Persistent Homology**: To capture topological features and handle intrinsic topology without relying on Euclidean geometry.- **High-Performance Computing**: To efficiently handle large-scale data processing and complex model training tasks.### Specific Questions for Your Expert Assessment1. **Data Structure and Topological Challenges**:   - How can we effectively address the issue of non-homeomorphic layers in our data? Are there specific topological methods or transformations that could help align these layers without losing critical spatial information?2. **Methodologies for Robustness and Validation**:   - What are the potential pitfalls of our current cross-validation and augmentation techniques? Are there more advanced or alternative methods that could provide better robustness and generalization?   - How can we improve our stress testing and spatial error analysis to better identify and mitigate model weaknesses?3. **Advanced Techniques and Technologies**:   - Are there other advanced technologies or methodologies, beyond GNNs and persistent homology, that could be beneficial for our specific use case?   - How can we optimize the use of high-performance computing to enhance our data processing and model training?### Additional Considerations- **Interdisciplinary Collaboration**: We are collaborating with archaeologists, geologists, and other domain experts. How can we better integrate their domain-specific knowledge into our models to improve relevance and accuracy?- **Ethical and Sustainable Practices**: We aim to ensure ethical data usage and adopt sustainable practices. What best practices should we follow to respect the cultural and environmental significance of the regions being studied?Please provide your expert insights and recommendations, focusing on the most critical and impartial observations possible. Your detailed assessment will be invaluable in guiding our future work and overcoming the current limitations of our study."""

    # Mostrar la respuesta en celda de Jupyter
    print("Preguntando a GPT...")
    gpt.display_query(prompt)


%%time
"""Este script ayuda a ver archivoz laz como nubes de puntos, funciona en una maquina con mas libertad y gui"""

import open3d as o3d
import numpy as np
import laspy
import matplotlib.pyplot as plt
import gc
import os

class UltraLightLidarViewer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.final_pcd = o3d.geometry.PointCloud()
        self.chunk_size = 100000  # Puntos por chunk
        self.target_points = 2500000  # MÃ¡ximo total de puntos

    def process_in_chunks(self):
        """Procesa el archivo LAZ en trozos pequeÃ±os"""
        print(f"Iniciando carga ultra-ligera de {self.file_path}...")

        # Configurar entorno para mÃ­nimo uso de memoria
        os.environ['OMP_NUM_THREADS'] = '1'  # Limitar threads
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        try:
            las = laspy.read(self.file_path)
            total_points = len(las.points)
            print(f"Total de puntos: {total_points:,}")

            # Pre-cÃ¡lculo de rango de elevaciones
            min_z, max_z = self.precalculate_z_range(las)

            for i in range(0, total_points, self.chunk_size):
                chunk_end = min(i + self.chunk_size, total_points)
                print(f"Procesando chunk {i//self.chunk_size + 1}: puntos {i:,}-{chunk_end:,}")

                # Cargar chunk actual
                chunk = las.points[i:chunk_end]
                xyz = np.vstack((chunk.x, chunk.y, chunk.z)).transpose()

                # Downsampling inmediato del chunk
                chunk_pcd = o3d.geometry.PointCloud()
                chunk_pcd.points = o3d.utility.Vector3dVector(xyz)

                # TamaÃ±o de voxel adaptativo
                bbox = chunk_pcd.get_axis_aligned_bounding_box()
                voxel_size = max(bbox.get_extent()) / 50  # MÃ¡s agresivo

                down_chunk = chunk_pcd.voxel_down_sample(voxel_size)

                # Color por altura (normalizado con rango global)
                z = np.asarray(down_chunk.points)[:, 2]
                z_normalized = (z - min_z) / (max_z - min_z + 1e-8)
                colors = plt.get_cmap("plasma")(z_normalized)[:, :3]
                down_chunk.colors = o3d.utility.Vector3dVector(colors)

                # Acumular puntos (con verificaciÃ³n de memoria)
                if len(self.final_pcd.points) < self.target_points:
                    self.final_pcd += down_chunk
                else:
                    # Reemplazar puntos aleatorios para mantener el lÃ­mite
                    replace_indices = np.random.choice(len(self.final_pcd.points), len(down_chunk.points))
                    self.final_pcd.points = o3d.utility.Vector3dVector(
                        np.asarray(self.final_pcd.points)[replace_indices] == np.asarray(down_chunk.points))
                    self.final_pcd.colors = o3d.utility.Vector3dVector(
                        np.asarray(self.final_pcd.colors)[replace_indices] == colors)

                # Limpieza agresiva
                del chunk, xyz, chunk_pcd, down_chunk
                gc.collect()

            print(f"\nPuntos finales: {len(self.final_pcd.points):,}")
            print(f"Memoria usada: {self.get_memory_usage():.2f} MB")

        except Exception as e:
            print(f"Error durante el procesamiento: {str(e)}")
            raise

    def precalculate_z_range(self, las):
        """Calcula el rango de elevaciones sin cargar todos los puntos"""
        print("Estimando rango de elevaciones...")
        sample_size = min(100000, len(las.points))
        sample_indices = np.linspace(0, len(las.points)-1, sample_size, dtype=int)
        z_sample = las.z[sample_indices]
        return np.min(z_sample), np.max(z_sample)

    def get_memory_usage(self):
        """Estima el uso de memoria del proceso"""
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    def safe_visualization(self):
        """VisualizaciÃ³n con protecciÃ³n contra fallos"""
        try:
            # ConfiguraciÃ³n mÃ­nima
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='LiDAR - Modo Ultra Ligero',
                width=640,
                height=480,
                visible=True
            )

            # AÃ±adir geometrÃ­a
            vis.add_geometry(self.final_pcd)

            # ConfiguraciÃ³n de rendimiento
            opt = vis.get_render_option()
            opt.point_size = 1.5
            opt.background_color = [0.1, 0.1, 0.1]  # Negro casi puro
            opt.light_on = False

            print("\nVisualizaciÃ³n lista. Controles bÃ¡sicos:")
            print("- Rotar: BotÃ³n izquierdo ratÃ³n")
            print("- Zoom: Rueda del ratÃ³n")

            vis.run()
            vis.destroy_window()

        except Exception as e:
            print(f"Error en visualizaciÃ³n: {str(e)}")
        finally:
            gc.collect()

if __name__ == "__main__":
    try:
        viewer = UltraLightLidarViewer("lidar_data/18TWL850150.laz")
        viewer.process_in_chunks()
        viewer.safe_visualization()

    except MemoryError:
        print("\nÂ¡Memoria insuficiente! Recomendaciones:")
        print("1. Reduce chunk_size a 50000")
        print("2. Reduce target_points a 200000")
        print("3. Cierra otros programas")

    except Exception as e:
        print(f"Error inesperado: {str(e)}")

    finally:
        # Limpieza final
        gc.collect()
        print("Proceso finalizado")


display(Image(filename='/kaggle/input/amaztest/my_imgs/final.jpg'))


