%%time
import matplotlib.pyplot as plt # Inne method plt
import seaborn as sns 
from matplotlib.colors import ListedColormap


# Custom colors
my_coolors = ["#407e31", "#71b12c", "#8A26BD", "#C7283A","#b5ff46", "#858585","#8db084", "#0c0a0b", "#edf2f8"]
class mainK:
    c = '\033[1m' + '\033[1;38;2;138;38;189m'
    f = '\033[0m'
class secK:
    c = '\033[1m' + '\033[1;38;2;181;255;70m'
    f = '\033[0m'  
class terK:
    c = '\033[1m' + '\033[1;38;2;199;40;58m'
    f = '\033[0m'      

pltKolors = ListedColormap(my_coolors)
sns.palplot(sns.color_palette(my_coolors))
print(mainK.c+"Main color Palette:"+mainK.f,"\n")
print(secK.c+"Second color Palette:"+secK.f,"\n")
print(terK.c+"Third color Palette:"+terK.f,"\n")
plt.show()


%%time
import os
import dask
from dask.distributed import LocalCluster, Client, SSHCluster
import concurrent.futures
import dask.dataframe as dd
import dask.array as da
import traceback
import cProfile

dask_client = None  # Global reference to Dask client

class DaskClusterManager:
    """Manages the Dask cluster and client."""
    def __init__(self, remote_addresses=None):
        global dask_client
        self.remote_addresses = remote_addresses if remote_addresses else []
        self.client = dask_client  # Use global reference

    def create_cluster(self):
        """Creates and returns a Dask client."""
        global dask_client
        try:
            num_local_workers = os.cpu_count()
            local_cluster = LocalCluster(n_workers=num_local_workers)
            add_remote_machines = len(self.remote_addresses) > 0

            if add_remote_machines:
                remote_cluster = SSHCluster(self.remote_addresses)
                combined_cluster = local_cluster + remote_cluster
            else:
                combined_cluster = local_cluster

            self.client = Client(combined_cluster)
            dask_client = self.client  # Assign to global variable

                        
            print(mainK.c + f'The link to the dashboard is  >>>>>>>>>>' + mainK.f,
                  secK.c + f'{self.client.dashboard_link}' + secK.f,
                  terK.c+f"\nğŸ§‘â€�ğŸ’» Using client as ğŸ§‘â€�ğŸ’» >>>>>>>>>> {self.client}\n"+terK.f)            
            return self.client
        except Exception as e:
            print(secK.c + f" \u2620\ufe0f An error occurred while creating the Dask cluster \u2620\ufe0f: {e}" + secK.f)
            traceback.print_exc()
            return None

    def get_client(self):
        """Returns the existing client or creates a new one if it doesn't exist."""
        global dask_client
        if self.client is None:
            self.client = self.create_cluster()
            dask_client = self.client
        return self.client

    def close_client(self):
        """Closes the dask client"""
        global dask_client
        if self.client:
            self.client.close()
            dask_client = None

    def get_workers_id(self):
        """Returns a dictionary with worker IDs and their addresses."""
        if self.client is None:
            print("No client available. Please create a cluster first.")
            return {}
        scheduler_info = self.client.scheduler_info()
        workers = scheduler_info.get('workers', {})
        worker_info = {}
        for worker_id, worker_details in workers.items():
            worker_info[worker_id] = worker_details['host']
        return worker_info


if __name__ == "__main__":
    cluster_manager = DaskClusterManager()  # Create the Dask manager
    dask_client = cluster_manager.get_client()  # Get the dask client

    if dask_client:
        # Get worker IDs and addresses
        workers = cluster_manager.get_workers_id()
        for worker_id, address in workers.items():
            print(secK.c+f"âš™ï¸� Worker ID âš™ï¸�: {worker_id}, Address: {address}"+secK.f)

        # Optional profiling or cleanup
        # cProfile.run('dask_client')
        # cluster_manager.close_client()


%%time
"""How the metadata comes"""

import polars as pl

# Define the dataset
MetadataSet = [
    "/kaggle/input/ariel-data-challenge-2025/adc_info.csv",
    "/kaggle/input/ariel-data-challenge-2025/axis_info.parquet",
    "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv",
    "/kaggle/input/ariel-data-challenge-2025/test_star_info.csv",
    "/kaggle/input/ariel-data-challenge-2025/train.csv",
    "/kaggle/input/ariel-data-challenge-2025/train_star_info.csv",
    "/kaggle/input/ariel-data-challenge-2025/wavelengths.csv"
]

def analyze_file(file_path):
    """Analyze a single file and print its characteristics"""
    print(f"\n=== Analyzing: {file_path.split('/')[-1]} ===")
    
    try:
        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pl.read_parquet(file_path)
        else:
            print("Unsupported file format")
            return
        
        # Display basic information
        print("\n\033[1mFirst 2 rows:\033[0m")
        print(df.head(2))
        
        print("\n\033[1mData types:\033[0m")
        for name, dtype in zip(df.columns, df.dtypes):
            print(f"{name}: {dtype}")
        
        print("\n\033[1mDataframe Shape:\033[0m")
        print(f"Rows: {df.height:,}, Columns: {df.width}")
        
        # Display statistics for numeric columns
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Int64, pl.Float64)]
        if numeric_cols:
            print("\n\033[1mBasic statistics:\033[0m")
            print(df.select(numeric_cols).describe())
        else:
            print("\nNo numeric columns for statistics")
            
    except Exception as e:
        print(f"\nError processing file: {str(e)}")


if __name__ == "__main__":
    # Analyze all files
    for file_path in MetadataSet:
        analyze_file(file_path)
        print("\n" + "="*60)


%%time
"""Folder explorer + plot"""

import os
import re
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataExplorer:
    def __init__(self, root_path):
        self.root_path = root_path
        self.structure = {}
        self.patterns = {
            'folder_patterns': defaultdict(int),
            'file_patterns': defaultdict(int),
            'extensions': defaultdict(int)
        }
    
    def explore(self):
        """Explora recursivamente la estructura del directorio."""
        self._explore_recursive(self.root_path, level=0)
    
    def _explore_recursive(self, path, level):
        """FunciÃ³n recursiva para mapear carpetas y archivos."""
        if level not in self.structure:
            self.structure[level] = {'folders': [], 'files': []}
        
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    self.structure[level]['folders'].append(item)
                    pattern = self._generalize_name(item)
                    self.patterns['folder_patterns'][pattern] += 1
                    self._explore_recursive(full_path, level + 1)
                else:
                    self.structure[level]['files'].append(item)
                    ext = os.path.splitext(item)[1]
                    pattern = self._generalize_name(item)
                    self.patterns['file_patterns'][pattern] += 1
                    self.patterns['extensions'][ext] += 1
        except PermissionError:
            print(f"[!] No se puede acceder a {path}: permiso denegado.")
    
    def _generalize_name(self, name):
        """
        Convierte nombres especÃ­ficos en patrones generalizados.
        Ejemplo: P001 -> ID_*, batch_001 -> batch_*
        """
        # Reemplaza nÃºmeros con *
        generalized = re.sub(r'\d+', '*', name)
        # Reemplaza IDs tipo P001 -> ID_*
        generalized = re.sub(r'[pP]\d+', 'ID_*', generalized)
        # Reemplaza fechas tipo 20250401 -> DATE_*
        generalized = re.sub(r'\b\d{8}\b', 'DATE_*', generalized)
        # Reemplaza UUIDs u otros patrones alfanumÃ©ricos largos
        generalized = re.sub(r'^[a-zA-Z0-9]{6,}$', 'CODE_*', generalized)
        return generalized
    
    def summary(self):
        """Imprime un resumen de la estructura y patrones encontrados."""
        print("\n=== Directory structure ===")
        for level, content in self.structure.items():
            print(f"\nLevel {level}:")
            if content['folders']:
                print("  Folders:", ', '.join(content['folders'][:5]) + ('...' if len(content['folders']) > 5 else ''))
            if content['files']:
                print("  Files:", ', '.join(content['files'][:5]) + ('...' if len(content['files']) > 5 else ''))
        
        print("\n=== Generalized patterns ===")
        print("\nFolder patterns:")
        for pattern, count in sorted(self.patterns['folder_patterns'].items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")
        
        print("\nFile patterns:")
        for pattern, count in sorted(self.patterns['file_patterns'].items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")
        
        print("\nFrequency of extensions:")
        for ext, count in sorted(self.patterns['extensions'].items(), key=lambda x: -x[1]):
            print(f"  {ext}: {count}")
    
    def plot_results(self, top_n=10, output_dir="plots"):
        """
        Genera grÃ¡ficos interactivos con Plotly y los guarda como HTML.
        
        Args:
            top_n (int): NÃºmero mÃ¡ximo de elementos a mostrar en cada grÃ¡fico
            output_dir (str): Directorio donde guardar los grÃ¡ficos HTML
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # GrÃ¡fico para patrones de carpetas
        folder_patterns = sorted(self.patterns['folder_patterns'].items(), key=lambda x: -x[1])[:top_n]
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=[p[0] for p in folder_patterns],
            y=[p[1] for p in folder_patterns],
            name="Folder patterns"
        ))
        fig1.update_layout(
            title="Folder naming patterns (Top {})".format(top_n),
            xaxis_title="Pattern",
            yaxis_title="Frecuency"
        )
        fig1.write_html(os.path.join(output_dir, "folder_patternsTest.html"))
        
        # GrÃ¡fico para patrones de archivos
        file_patterns = sorted(self.patterns['file_patterns'].items(), key=lambda x: -x[1])[:top_n]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[p[0] for p in file_patterns],
            y=[p[1] for p in file_patterns],
            name="File patterns"
        ))
        fig2.update_layout(
            title="Folder naming patterns (Top {})".format(top_n),
            xaxis_title="Pattern",
            yaxis_title="Frecuency"
        )
        fig2.write_html(os.path.join(output_dir, "file_patternsTrain.html"))
        
        # GrÃ¡fico para extensiones de archivos
        extensions = sorted(self.patterns['extensions'].items(), key=lambda x: -x[1])[:top_n]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[p[0] for p in extensions],
            y=[p[1] for p in extensions],
            name="File extensions"
        ))
        fig3.update_layout(
            title="Most common file extensions (Top {})".format(top_n),
            xaxis_title="Extension",
            yaxis_title="Frecuency"
        )
        fig3.write_html(os.path.join(output_dir, "file_extensionsTest.html"))
        
        # GrÃ¡fico combinado para resumen
        fig_combined = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Folder patterns", "Files patterns", "File extensions")
        )
        
        fig_combined.add_trace(
            go.Bar(
                x=[p[0] for p in folder_patterns],
                y=[p[1] for p in folder_patterns],
                name="Folders"
            ),
            row=1, col=1
        )
        
        fig_combined.add_trace(
            go.Bar(
                x=[p[0] for p in file_patterns],
                y=[p[1] for p in file_patterns],
                name="File"
            ),
            row=2, col=1
        )
        
        fig_combined.add_trace(
            go.Bar(
                x=[p[0] for p in extensions],
                y=[p[1] for p in extensions],
                name="Extensions"
            ),
            row=3, col=1
        )
        
        fig_combined.update_layout(
            height=1200,
            title_text="Summary of patterns in the directory structure for Test",
            showlegend=False
        )
        
        fig_combined.write_html(os.path.join(output_dir, "combined_summaryTest.html"))
        
        print(f"\nGrÃ¡ficos guardados en el directorio '{output_dir}' como archivos HTML.")

if __name__ == "__main__":
    # Ruta a tu carpeta principal
    ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025/test"
    
    # Crear instancia del explorador
    explorer = DataExplorer(ROOT_PATH)
    
    # Explorar la estructura
    explorer.explore()
    
    # Mostrar resumen
    explorer.summary()
    
    # Generar grÃ¡ficos
    explorer.plot_results(top_n=15)


%%time

"""ULTRA FILTRO + plot"""

import json
from pathlib import Path
from collections import defaultdict
import os
import plotly.express as px
import pandas as pd


class ArielUltraLightMapper:
    def __init__(self, root_path):
        self.root_path = Path(root_path).absolute()
        self.data = {
            '_meta': {
                'root_path': str(self.root_path),
                'file_patterns': {
                    'signal': '{instrument}_signal_{obs_type}.parquet',
                    'calibration': '{instrument}_calibration_{set_num}/{calib_type}.parquet'
                },
                'stats': defaultdict(int)
            },
            'index': {
                'planets': set(),  # Usamos set para planetas Ãºnicos
                'instruments': defaultdict(list),  # instrument: [planet_ids]
                'signals': defaultdict(lambda: defaultdict(list)),  # instrument: {type: [keys]}
                'calibrations': defaultdict(lambda: defaultdict(dict))  # instrument: {type: {set_num: key}}
            },
            'paths': {}  # key: relative_path
        }

    def build_mapping(self):
        """Construye un mapeo minimalista con Ã­ndices cruzados"""
        for planet_dir in self.root_path.iterdir():
            if not planet_dir.is_dir():
                continue
            planet_id = planet_dir.name
            self.data['index']['planets'].add(planet_id)
            self.data['_meta']['stats']['planets'] += 1

            # Procesar seÃ±ales
            for signal_file in planet_dir.glob('*_signal_*.parquet'):
                instrument, obs_type = self._parse_signal(signal_file.name)
                if not instrument:
                    continue
                rel_path = str(signal_file.relative_to(self.root_path))
                key = f"signal_{instrument}_{obs_type}"
                self.data['paths'][key] = rel_path

                self.data['index']['signals'][instrument][obs_type].append(key)
                if planet_id not in self.data['index']['instruments'][instrument]:
                    self.data['index']['instruments'][instrument].append(planet_id)
                self.data['_meta']['stats']['signals'] += 1

            # Procesar calibraciones
            for calib_dir in planet_dir.glob('*_calibration*'):
                instrument, calib_set = self._parse_calib(calib_dir.name)
                if not instrument:
                    continue
                for calib_file in calib_dir.glob('*.parquet'):
                    calib_type = calib_file.stem
                    rel_path = str(calib_file.relative_to(self.root_path))
                    key = f"calib_{instrument}_{calib_type}_{calib_set}"
                    self.data['paths'][key] = rel_path

                    self.data['index']['calibrations'][instrument][calib_type][calib_set] = key
                    if planet_id not in self.data['index']['instruments'][instrument]:
                        self.data['index']['instruments'][instrument].append(planet_id)
                    self.data['_meta']['stats']['calibrations'] += 1

        self._optimize_structure()
        return self.data

    def _parse_signal(self, filename):
        parts = filename.rsplit('_', 2)
        if len(parts) == 3 and parts[-1].split('.')[0] in ('0', '1'):
            return parts[0], parts[-1].split('.')[0]
        return None, None

    def _parse_calib(self, dirname):
        parts = dirname.split('_')
        if len(parts) >= 3:
            return parts[0], parts[-1]
        return None, None

    def _optimize_structure(self):
        self.data['index']['planets'] = sorted(self.data['index']['planets'])
        self.data['index']['instruments'] = dict(self.data['index']['instruments'])
        self.data['index']['signals'] = {
            instr: dict(types)
            for instr, types in self.data['index']['signals'].items()
        }
        self.data['index']['calibrations'] = {
            instr: dict(types)
            for instr, types in self.data['index']['calibrations'].items()
        }

    def save_json(self, output_file="ariel_ultralight.json"):
        with open(output_file, 'w') as f:
            json.dump(self.data, f, separators=(',', ':'), indent=2 if __debug__ else None)

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"âœ… JSON ultra-ligero guardado ({size_mb:.2f} MB)")
        print("ğŸ“Š EstadÃ­sticas:")
        for k, v in self.data['_meta']['stats'].items():
            print(f"- {k}: {v}")

    def generate_plotly_html(self, html_output="ariel_data_visualization.html"):
        """Genera un grÃ¡fico interactivo con Plotly y lo guarda como HTML"""

        # Extraer informaciÃ³n del JSON
        instruments = list(self.data["index"]["instruments"].keys())
        signals_count = []
        calibrations_count = []
        planets_per_instrument = []

        for inst in instruments:
            signal_types = self.data["index"]["signals"].get(inst, {})
            total_signals = sum(len(keys) for keys in signal_types.values())
            signals_count.append(total_signals)

            calib_types = self.data["index"]["calibrations"].get(inst, {})
            total_calibs = sum(len(set_nums) for set_nums in calib_types.values())
            calibrations_count.append(total_calibs)

            # Planetas asociados al instrumento
            planets_per_instrument.append(len(self.data["index"]["instruments"][inst]))

        # Crear DataFrame para Plotly
        df = pd.DataFrame({
            "Instrument": instruments,
            "Total Signals": signals_count,
            "Total Calibrations": calibrations_count,
            "Associated Planets": planets_per_instrument
        })

        # GrÃ¡fico 1: DistribuciÃ³n de SeÃ±ales y Calibraciones por Instrumento
        fig1 = px.bar(df, x="Instrument", y=["Total Signals", "Total Calibrations"],
                      title="Signal Distribution and Calibrations per Instrument",
                      labels={"value": "Quantity", "variable": "Data type"},
                      barmode='group')

        # GrÃ¡fico 2: Planetas Asociados por Instrumento
        fig2 = px.bar(df, x="Instrument", y="Associated Planets",
                      title="Number of Associated Planets per Instrument",
                      labels={"value": "Planet quantities", "x": "Instrument"})

        # Guardar ambos grÃ¡ficos en un archivo HTML
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Signal Distribution and Calibrations", "Associated Planets"))

        # Agregar grÃ¡ficos
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig1.data[1], row=1, col=1)
        fig.add_trace(fig2.data[0], row=2, col=1)

        # Actualizar diseÃ±o
        fig.update_layout(height=800, showlegend=True)

        # Guardar el HTML
        fig.write_html(html_output)
        print(f"ğŸ“Š GrÃ¡fico interactivo guardado como {html_output}")


# Ejemplo de uso
if __name__ == "__main__":
    mapper = ArielUltraLightMapper("/kaggle/input/ariel-data-challenge-2025/train")
    mapper.build_mapping()
    mapper.save_json()
    mapper.generate_plotly_html()


%%time
"""Truncar lectura, query disk"""

import json
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm
import os
import glob


def truncate_planets(planets: List[str], truncate: Union[float, int, None]) -> List[str]:
    if truncate is None:
        return planets
    if isinstance(truncate, float) and 0 < truncate <= 1:
        return planets[:int(len(planets) * truncate)]
    if isinstance(truncate, int) and truncate > 0:
        return planets[:truncate]
    raise ValueError("truncate must be None, float (0<x<=1) or positive int")


class DiskParquetCounter:
    @staticmethod
    def count_parquet_files(path: Path) -> int:
        if path.is_file() and path.suffix == '.parquet':
            return 1
        elif path.is_dir():
            return len(list(path.rglob("*.parquet")))
        return 0


class ArielContext:
    def __init__(self, json_path: str, planet_id: Optional[str] = None, truncate_ids: Union[int, float, None] = None):
        with open(json_path) as f:
            self.data = json.load(f)

        self.base_path = Path(self.data["_meta"]["root_path"])
        self.signal_pattern = self.data["_meta"]["file_patterns"]["signal"]
        self.calib_pattern = self.data["_meta"]["file_patterns"]["calibration"]

        self.planets = self.data["index"]["planets"]
        self.instruments = self.data["index"]["instruments"]
        self.calibrations = self.data["index"]["calibrations"]
        self.paths_index = self.data.get("paths", {})

        self.planet_id = planet_id
        self.truncate_ids = truncate_ids
        self.active_planets = [planet_id] if planet_id else truncate_planets(self.planets, truncate_ids)

    def planet_has_instrument(self, planet: str, instr: str) -> bool:
        return planet in self.instruments.get(instr, [])


class SignalFinder:
    def __init__(self, context: ArielContext):
        self.ctx = context

    def find(self, instrument: Optional[str] = None, signal_type: Optional[Union[str, int]] = None, count_files: bool = False) -> List[str]:
        results = []
        file_count = 0
        instruments = [instrument] if instrument else list(self.ctx.instruments.keys())

        if signal_type == "2" or signal_type == 2:
            types = ["0", "1"]
        elif signal_type in ["0", "1"]:
            types = [signal_type]
        else:
            types = ["0", "1"]  # default fallback

        for planet in tqdm(self.ctx.active_planets, desc="Planets(signal)"):
            for instr in instruments:
                if not self.ctx.planet_has_instrument(planet, instr):
                    continue
                for stype in types:
                    rel = f"{planet}/{self.ctx.signal_pattern.format(instrument=instr, obs_type=stype)}"
                    full = self.ctx.base_path / rel
                    if full.exists():
                        results.append(str(full))
                        if count_files:
                            file_count += DiskParquetCounter.count_parquet_files(full)
        print(f"ğŸ“� Total signal paths: {len(results)}")
        if count_files:
            print(f"ğŸ“Š Total actual signal .parquet files on disk: {file_count}")
        return results


class CalibrationFinder:
    def __init__(self, context: ArielContext):
        self.ctx = context

    def find(self, instrument: Optional[str] = None, calib_types: Optional[Union[List[str], int, str]] = None, calib_set: Optional[Union[str, int]] = None, count_files: bool = False) -> List[str]:
        results = []
        file_count = 0
        instruments = [instrument] if instrument else list(self.ctx.calibrations.keys())

        if calib_types == "2" or calib_types == 2:
            if instrument:
                types = list(self.ctx.calibrations.get(instrument, {}).keys())
            else:
                types = list(set().union(*[self.ctx.calibrations[i].keys() for i in self.ctx.calibrations]))
        elif calib_types:
            types = calib_types
        else:
            if instrument:
                types = list(self.ctx.calibrations.get(instrument, {}).keys())
            else:
                types = list(set().union(*[self.ctx.calibrations[i].keys() for i in self.ctx.calibrations]))

        sets_map = self.ctx.calibrations
        known_paths = set(self.ctx.paths_index.values())

        for planet in tqdm(self.ctx.active_planets, desc="Planets(calibration)"):
            for instr in instruments:
                if not self.ctx.planet_has_instrument(planet, instr):
                    continue
                for ctype in types:
                    if calib_set == "2" or calib_set == 2:
                        available_sets = list(sets_map.get(instr, {}).get(ctype, {}).keys())
                    elif calib_set:
                        available_sets = [calib_set]
                    else:
                        available_sets = list(sets_map.get(instr, {}).get(ctype, {}).keys())

                    for s in available_sets:
                        rel_path = f"{planet}/{self.ctx.calib_pattern.format(instrument=instr, set_num=s, calib_type=ctype)}"
                        full = self.ctx.base_path / rel_path
                        if full.exists() or rel_path in known_paths:
                            results.append(str(full))
                            if count_files:
                                file_count += DiskParquetCounter.count_parquet_files(full)
        print(f"ğŸ“� Total calibration paths: {len(results)}")
        if count_files:
            print(f"ğŸ“Š Total actual calibration .parquet files on disk: {file_count}")
        return results



# ==================== EJEMPLOS ====================
if __name__ == "__main__":
    # Crear contexto una vez y compartirlo
    ctx1 = ArielContext("ariel_ultralight.json", truncate_ids=None)
    ctx2 = ArielContext("ariel_ultralight.json", planet_id="1810380816")
    ctx3 = ArielContext("ariel_ultralight.json", truncate_ids=0.5)
    ctx4 = ArielContext("ariel_ultralight.json", planet_id="989956432")

    # 1) SeÃ±ales AIRS-CH0, tipo 0, primeros 10 planetas
    sigs = SignalFinder(ctx1).find(instrument="AIRS-CH0", signal_type="2", )
    print(f"Found {len(sigs)} signal paths.")

    # 2) Todas las seÃ±ales de un planeta especÃ­fico
    sigs2 = SignalFinder(ctx2).find(signal_type="1",)
    print(f"Planet 1810380816 signals: {len(sigs2)} files.")

    # 3) Calibraciones dark y flat de FGS1 en 50% de planetas
    cals = CalibrationFinder(ctx3).find(instrument="FGS1", calib_types=["dark", "flat"])
    print(f"Found {len(cals)} calibration files.")

    # 4) Calibraciones dead y dark de FGS1 para un planeta
    cals2 = CalibrationFinder(ctx4).find(instrument="FGS1", calib_types=["dead", "dark"], calib_set="2")
    print(f"Planet 989956432 FGS1 calibrations: {len(cals2)} files.")


%%time
"""Dask sin computar """
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dask.distributed import Queue
import dask.dataframe as dd


class ChunkedSignalReaderWithQueue:
    def __init__(self, context, dask_client, queue_name="signal-queue"):
        self.ctx = context
        self.client = dask_client
        self.queue = Queue(name=queue_name, client=self.client)

    def feed_queue(self, instrument, signal_type, chunk_size):
        paths = SignalFinder(self.ctx).find(instrument=instrument, signal_type=signal_type)
        total_chunks = int(np.ceil(len(paths) / chunk_size))
        self.total_chunks = total_chunks  # Store for tqdm in processing
        with tqdm(total=total_chunks, desc="ğŸ“¤ Enqueuing chunks", unit="chunk") as pbar:
            for i in range(0, len(paths), chunk_size):
                batch_paths = paths[i:i + chunk_size]
                try:
                    self.queue.put(batch_paths)
                except Exception as e:
                    print(f"â�Œ Error queueing batch: {e}")
                pbar.update(1)

    def process_queue(self):
        processed = 0
        with tqdm(total=getattr(self, 'total_chunks', None), desc="ğŸ“¥ Processing chunks", unit="chunk") as pbar:
            while True:
                try:
                    batch_paths = self.queue.get(timeout=5)
                    ddf = dd.read_parquet(batch_paths, engine="pyarrow")
                    result = ddf.columns[:3]  # Light dummy access
                    _ = result  # avoid print
                    processed += 1
                    pbar.update(1)
                except Exception:
                    print("âœ… Finished all queued chunks.")
                    break


if __name__ == "__main__":
    #from DaskClusterManager import DaskClusterManager

    # âœ… ParÃ¡metros
    JSON_PATH = "ariel_ultralight.json"
    TRUNCATE_PLANETS = None
    INSTRUMENT = "AIRS-CH0"
    SIGNAL_TYPE = "0"
    CHUNK_SIZE = 64 # More size less time
    QUEUE_NAME = "signal-queue"

    ctx = ArielContext(JSON_PATH, truncate_ids=TRUNCATE_PLANETS)
    client = DaskClusterManager().get_client()

    reader = ChunkedSignalReaderWithQueue(ctx, client, queue_name=QUEUE_NAME)
    reader.feed_queue(INSTRUMENT, SIGNAL_TYPE, CHUNK_SIZE)
    reader.process_queue()

#16 ch = 8:53min
#32 ch = 4:10min
#64 ch = 2:11min


%%time
"""Sin dask"""

import numpy as np
import pandas as pd
import queue
import threading
from pathlib import Path
from tqdm import tqdm
import concurrent.futures


class ChunkedSignalReaderMultiprocess:
    def __init__(self, context, queue_size=50):
        self.ctx = context
        self.paths = []
        self.queue = queue.Queue(maxsize=queue_size)

    def feed_queue(self, instrument, signal_type, chunk_size):
        from time import sleep
        self.paths = SignalFinder(self.ctx).find(instrument=instrument, signal_type=signal_type)
        total_chunks = int(np.ceil(len(self.paths) / chunk_size))

        def producer():
            for i in tqdm(range(0, len(self.paths), chunk_size), desc="ğŸ“¤ Enqueuing chunks", unit="chunk"):
                batch_paths = self.paths[i:i + chunk_size]
                try:
                    dfs = [pd.read_parquet(p) for p in batch_paths]
                    combined = pd.concat(dfs, ignore_index=True)
                    self.queue.put(combined)  # bloquea si estÃ¡ llena
                except Exception as e:
                    print(f"â�Œ Error loading batch: {e}")

            # seÃ±al de cierre
            for _ in range(3):
                self.queue.put(None)

        threading.Thread(target=producer, daemon=True).start()

    def process_queue(self, num_workers=3):
        def worker(worker_id):
            while True:
                df = self.queue.get()
                if df is None:
                    break
                print(f"ğŸ§ª [Worker {worker_id}] Head:\n", df.head(2))
                self.queue.task_done()

        with tqdm(total=self.queue.qsize(), desc="ğŸ“¥ Processing chunks", unit="chunk") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker, wid) for wid in range(num_workers)]
                self.queue.join()


if __name__ == "__main__":
    # âœ… ParÃ¡metros
    JSON_PATH = "ariel_ultralight.json"
    TRUNCATE_PLANETS = None
    INSTRUMENT = "AIRS-CH0"
    SIGNAL_TYPE = "0"
    CHUNK_SIZE = 2

    ctx = ArielContext(JSON_PATH, truncate_ids=TRUNCATE_PLANETS)

    reader = ChunkedSignalReaderMultiprocess(ctx)
    reader.feed_queue(INSTRUMENT, SIGNAL_TYPE, CHUNK_SIZE)
    reader.process_queue(num_workers=3)



%%time
"""TESTING PURPOSES: This class needs to be optimized in the workers memory to process all the queue"""
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

class BoundedSignalVisualizer:
    def __init__(self, context, output_dir, chunk_size):
        self.ctx = context
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _safe_sum(self, signal):
        """Safe summation handling different array dimensions"""
        if signal.ndim == 1:
            return signal
        return signal.sum(axis=tuple(range(1, signal.ndim)))
    
    def process_signals(self, instrument, signal_type):
        """Process signals in chunks and calculate min/max bounds"""
        paths = SignalFinder(self.ctx).find(instrument=instrument, signal_type=signal_type)
        total_chunks = int(np.ceil(len(paths) / self.chunk_size))
        
        all_min, all_max, all_time = [], [], []
        
        with tqdm(total=total_chunks, desc=f"Processing {instrument}", leave=False) as pbar:
            for i in range(0, len(paths), self.chunk_size):
                batch = paths[i:i + self.chunk_size]
                chunk_data = []
                
                for path in batch:
                    try:
                        signal = pd.read_parquet(path).values.astype(np.float64)
                        chunk_data.append(self._safe_sum(signal))
                    except Exception as e:
                        continue
                
                if chunk_data:
                    chunk_stack = np.array(chunk_data)
                    time_points = np.arange(chunk_stack.shape[1])
                    all_min.append(np.min(chunk_stack, axis=0))
                    all_max.append(np.max(chunk_stack, axis=0))
                    all_time.append(time_points)
                pbar.update(1)
        
        return all_time, all_min, all_max
    
    def save_bounded_plot(self, instrument, signal_type):
        """Generate and save interactive plot as HTML"""
        try:
            times, mins, maxs = self.process_signals(instrument, signal_type)
            if not times:
                return
            
            fig = go.Figure()
            
            for i, (t, min_vals, max_vals) in enumerate(zip(times, mins, maxs)):
                if len(t) == 0:
                    continue
                
                norm_factor = np.mean(max_vals) or 1
                min_norm = min_vals / norm_factor
                max_norm = max_vals / norm_factor
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t, t[::-1]]),
                    y=np.concatenate([max_norm, min_norm[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba(100, 150, 200, {0.3 + 0.7*(i/len(times))})',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Chunk {i+1}'
                ))
            
            fig.update_layout(
                title=f"{instrument} Signal Bounds (Normalized)",
                xaxis_title="Time (frame index)",
                yaxis_title="Normalized Flux",
                height=700,
                template="plotly_dark"
            )
            
            output_file = self.output_dir / f"{instrument}_{signal_type}_bounds.html"
            fig.write_html(output_file)
            
        except Exception as e:
            print(f"Error processing {instrument}: {str(e)}")

if __name__ == "__main__":    
    CONFIG = {
        "json_path": "ariel_ultralight.json",
        "output_dir": "signal_plots",
        "chunk_size": 64,
        "truncate_ids": 0.1,  # Truncate percentage, None for full dataset
        "instruments": ["AIRS-CH0", "FGS1"],
        "signal_types": ["2"]
    }
    
    # ===== EXECUTION =====
    ctx = ArielContext(CONFIG["json_path"], truncate_ids=CONFIG["truncate_ids"])
    visualizer = BoundedSignalVisualizer(
        ctx,
        output_dir=CONFIG["output_dir"],
        chunk_size=CONFIG["chunk_size"]
    )
    
    for instrument in CONFIG["instruments"]:
        for signal_type in CONFIG["signal_types"]:
            print(f"âš™ï¸� Processing {instrument} signal {signal_type}...")
            visualizer.save_bounded_plot(instrument, signal_type)
    
    print(f"âœ… All plots saved to {CONFIG['output_dir']} directory")


%%time
"""Testing Purposes"""
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

class NonLinearityVisualizer:
    def __init__(self, context, output_dir, sample_pixels):
        self.ctx = context
        self.output_dir = Path(output_dir)
        self.sample_pixels = sample_pixels  # NÃºmero de pÃ­xeles a muestrear
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_calibration_data(self, planet_id, instrument):
        """Carga los coeficientes de no linealidad para un planeta e instrumento"""
        calib_files = CalibrationFinder(self.ctx).find(
            instrument=instrument,
            calib_types=["linear_corr"],
            calib_set="0"
        )
        
        # Filtrar por planeta especÃ­fico
        planet_files = [f for f in calib_files if f"/{planet_id}/" in f]
        
        if not planet_files:
            return None
            
        try:
            df = pd.read_parquet(planet_files[0])
            return df.values.astype(np.float64)
        except Exception as e:
            print(f"Error loading {planet_files[0]}: {str(e)}")
            return None
    
    def generate_response_curves(self, coefficients, pixel_indices=None):
        """Genera curvas de respuesta usando los coeficientes polinomiales"""
        if pixel_indices is None:
            # Selecciona pÃ­xeles aleatorios si no se especifican
            rng = np.random.default_rng()
            pixel_indices = rng.choice(
                coefficients.shape[1], 
                size=min(self.sample_pixels, coefficients.shape[1]), 
                replace=False
            )
        
        curves = {}
        x = np.linspace(0, 1, 100)  # Rango normalizado de electrones
        
        for idx in pixel_indices:
            # Los coeficientes estÃ¡n en orden de mayor a menor grado
            poly = np.poly1d(coefficients[:, idx])
            y = poly(x)
            curves[f"Pixel_{idx}"] = (x, y)
            
        return curves
    
    def save_nonlinearity_plot(self, planet_id, instrument):
        """Genera y guarda el grÃ¡fico de no linealidad"""
        coefficients = self.load_calibration_data(planet_id, instrument)
        if coefficients is None:
            return
            
        # Tomamos los primeros 5 pÃ­xeles para el ejemplo
        pixel_indices = range(min(5, coefficients.shape[1]))
        curves = self.generate_response_curves(coefficients, pixel_indices)
        
        fig = go.Figure()
        
        for name, (x, y) in curves.items():
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                name=name,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Curva de Respuesta No Lineal<br>{instrument} - Planeta {planet_id}",
            xaxis_title="Electrones (normalizado)",
            yaxis_title="SeÃ±al del pixel",
            legend_title="PÃ­xeles",
            template="plotly_white",
            height=600
        )
        
        output_file = self.output_dir / f"nonlinearity_{instrument}_{planet_id}.html"
        fig.write_html(output_file)

if __name__ == "__main__":
    # ===== CONFIGURACIÃ“N =====
    CONFIG = {
        "json_path": "ariel_ultralight.json",
        "output_dir": "nonlinearity_plots",
        "sample_planets": 3,  # NÃºmero de planetas a muestrear
        "sample_pixels": 5,   # PÃ­xeles por grÃ¡fico
        "instruments": ["AIRS-CH0", "FGS1"]
    }
    
    # ===== EJECUCIÃ“N =====
    ctx = ArielContext(CONFIG["json_path"])
    visualizer = NonLinearityVisualizer(
        ctx,
        output_dir=CONFIG["output_dir"],
        sample_pixels=CONFIG["sample_pixels"]
    )
    
    # Seleccionar subconjunto de planetas
    planet_ids = ctx.planets[:CONFIG["sample_planets"]]
    
    for planet_id in planet_ids:
        for instrument in CONFIG["instruments"]:
            print(f"âš™ï¸� Procesando {instrument} para planeta {planet_id}...")
            visualizer.save_nonlinearity_plot(planet_id, instrument)
    
    print(f"âœ… Todos los grÃ¡ficos guardados en {CONFIG['output_dir']}")

