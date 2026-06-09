#@title Authenticate (Use your own GEE credentials and project name)

import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime # For potential client-side date parsing if needed

# Initialize Earth Engine if not already done (geemap.ee_initialize() handles this)
# Initialize the library.
# Trigger the authentication flow.
ee.Authenticate()



project_name = input("Enter the name of the gee project")  # Change it your GEE project name 


ee.Initialize(project=project_name)


#@title Helpers and ingestion of exhaustive database of known sites in Acre

import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
import re
import csv

# =========================================================================================
# 0. Helper Functions & Site Data Loading
# =========================================================================================
def parse_source_date_py(source_string, site_code_for_debug="N/A"):
    if not source_string or not isinstance(source_string, str): return None
    date_part = source_string.replace('GE ', '').strip().replace('.', '-').replace('/', '-')
    try:
        parts = date_part.split('-')
        if len(parts) == 3:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if not (1000 <= year <= 3000 and 1 <= month <= 12 and 1 <= day <= 31): return None
            date_part = f"{year:04d}-{month:02d}-{day:02d}"
        else: return None
    except ValueError: return None
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_part): return None
    try:
        if not ee.data._initialized:
            print("Warning (parse_source_date_py): GEE not initialized. Cannot parse date string for GEE.")
            return None
        return ee.Date.parse('yyyy-MM-dd', date_part)
    except Exception: return None

def load_sites_from_csv(csv_filepath):
    sites_list = []
    try:
        with open(csv_filepath, mode='r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            header_row_index, raw_headers_from_file = -1, []
            for i, row in enumerate(reader):
                if not row: continue
                temp_headers = [h.strip().lower() for h in row]
                # MODIFICATION START: Only require 'lat' and 'lon' in headers
                if 'lat' in temp_headers and 'lon' in temp_headers:
                    raw_headers_from_file, header_row_index = [h.strip() for h in row], i
                    break
            if header_row_index == -1:
                print(f"ERROR (load_sites_from_csv): No valid header (lat, lon) in {csv_filepath}.") # Updated error message
                return []
            # MODIFICATION END

            actual_headers_for_dict_keys = [h.lower() for h in raw_headers_from_file if h.strip()]
            data_offset = 1 if raw_headers_from_file and (raw_headers_from_file[0] == '' or raw_headers_from_file[0].isdigit()) else 0
            if data_offset: actual_headers_for_dict_keys = [h.lower() for h in raw_headers_from_file[1:] if h.strip()]
            if not actual_headers_for_dict_keys:
                 print(f"ERROR (load_sites_from_csv): No valid header columns after processing {csv_filepath}.")
                 return []
            infile.seek(0)
            for _ in range(header_row_index + 1): next(reader)
            site_counter = 0 # To generate codes if missing
            for i, row in enumerate(reader):
                if not any(field.strip() for field in row): continue
                data_values = row[data_offset:]
                if len(data_values) < len(actual_headers_for_dict_keys): data_values.extend([None] * (len(actual_headers_for_dict_keys) - len(data_values)))
                elif len(data_values) > len(actual_headers_for_dict_keys): data_values = data_values[:len(actual_headers_for_dict_keys)]
                site_dict = dict(zip(actual_headers_for_dict_keys, data_values))

                # MODIFICATION START: Ensure 'code' is always present, even if generated
                site_code_val = site_dict.get('code')
                if site_code_val is None or str(site_code_val).strip() == '':
                    site_code_val = f"SITE_{site_counter:05d}"
                    site_counter += 1
                # MODIFICATION END

                final_site = {
                    'code': site_code_val, # Use the extracted or generated code
                    'place': site_dict.get('place'),
                    'lat': None, 'lon': None, 'elev': None,
                    'source': site_dict.get('source'),
                    'a_width': site_dict.get('a width'), 'b_width': site_dict.get('b width'), 'form': site_dict.get('form')
                }
                try:
                    if site_dict.get('lat') and str(site_dict.get('lat')).strip(): final_site['lat'] = float(site_dict.get('lat'))
                    if site_dict.get('lon') and str(site_dict.get('lon')).strip(): final_site['lon'] = float(site_dict.get('lon'))
                    # These can be uncommented if these columns are added to your CSV and needed later
                    if site_dict.get('elev') and str(site_dict.get('elev')).strip(): final_site['elev'] = int(float(site_dict.get('elev')))
                    if site_dict.get('a width') and str(site_dict.get('a width')).strip(): final_site['a_width'] = int(float(site_dict.get('a width')))
                    if site_dict.get('b width') and str(site_dict.get('b width')).strip(): final_site['b_width'] = int(float(site_dict.get('b width')))

                except (ValueError, TypeError): print(f"Warning (load_sites_from_csv): Value parsing error for site {site_code_val}.") # Use generated code for warning
                if final_site['lat'] is not None and final_site['lon'] is not None: sites_list.append(final_site)
                else: print(f"Info (load_sites_from_csv): Site {final_site.get('code', 'N/A')} skipped (missing lat/lon).")
    except FileNotFoundError: print(f"ERROR (load_sites_from_csv): File not found: {csv_filepath}"); return []
    except Exception as e: print(f"ERROR (load_sites_from_csv): Reading {csv_filepath}: {e}"); return []
    return sites_list

# =========================================================================================
# 1. Known Archaeological Sites Data - LOADED FROM CSV (Keep as is, but add check for empty FC later)
# =========================================================================================
CSV_FILENAME = '/kaggle/input/amazon-geoglyphs/amazon_geoglyphs.csv' # ADJUSTED TO ALL_GEOGLYPHS.CSV
sites_data_array_py = load_sites_from_csv(CSV_FILENAME)
if not sites_data_array_py: print(f"CRITICAL ERROR: No sites loaded from {CSV_FILENAME}.")
else: print(f"Successfully loaded {len(sites_data_array_py)} sites from {CSV_FILENAME}")




import ee
import geemap
import geemap.foliumap as geemap_folium
from IPython.display import display, Image as IPyImage
import pandas as pd
import os
import folium

# Import the Python Imaging Library (Pillow)
from PIL import Image, ImageDraw, ImageFont



# --- Helper Function to Generate Custom Marker Images ---
def create_shape_icon(shape, color, filepath, size=40):
    """
    Generates a PNG image of a shape and saves it.
    This version correctly handles all specified forms.
    """
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    padding = 4
    bbox = [(padding, padding), (size - padding, size - padding)]

    if shape in ['circle', 'oval', 'ellipse', 'enclosure', 'default']:
        draw.ellipse(bbox, fill=color, outline='#000000', width=1)
    elif shape in ['square', 'rectangle', 'quadrangle', 'parallelogram']:
        draw.rectangle(bbox, fill=color, outline='#000000', width=1)
    elif shape == 'pentagon':
        draw.regular_polygon(
            bounding_circle=(size / 2, size / 2, size / 2 - padding),
            n_sides=5, rotation=18, fill=color, outline='#000000'
        )
    img.save(filepath)



# 1. Convert to DataFrame and filter
df = pd.DataFrame(sites_data_array_py)
df.dropna(subset=['lat', 'lon'], inplace=True)
df['form'] = df['form'].str.lower()

if df.empty:
    print("No sites with valid coordinates to add to the map.")
else:
    # 2. Define Comprehensive Style Guide and Generate Icons
    icon_dir = 'custom_icons'
    os.makedirs(icon_dir, exist_ok=True)

    style_guide = {
        'circle': {'color': '#008000', 'shape': 'circle'},
        'oval': {'color': '#FFA500', 'shape': 'oval'},
        'ellipse': {'color': '#FF8C00', 'shape': 'ellipse'},
        'square': {'color': '#FF0000', 'shape': 'square'},
        'rectangle': {'color': '#8B0000', 'shape': 'rectangle'},
        'quadrangle': {'color': '#F08080', 'shape': 'quadrangle'},
        'parallelogram': {'color': '#FF69B4', 'shape': 'parallelogram'},
        'pentagon': {'color': '#800080', 'shape': 'pentagon'},
        'enclosure': {'color': '#A52A2A', 'shape': 'enclosure'},
        'geoglyph (undefined shape)': {'color': '#808080', 'shape': 'oval'},
        'default': {'color': '#000000', 'shape': 'default'}
    }

    print("Generating custom marker icons...")
    for form, style in style_guide.items():
        safe_filename = "".join(c for c in form if c.isalnum()).rstrip()
        icon_path = os.path.join(icon_dir, f'icon_{safe_filename}.png')
        create_shape_icon(style['shape'], style['color'], icon_path)
        style['icon_path'] = icon_path

    # 3. Create the base map
    Map = geemap_folium.Map(center=[-9.97, -67.81], zoom=12, tiles='Esri.WorldImagery')

    # 4. Manually create and add individually-sized markers (Static Version)
    MIN_ICON_SIZE_PX, MAX_ICON_SIZE_PX, SCALING_FACTOR, DEFAULT_SIZE_PX = 15, 150, 0.4, 30

    for form_name, form_df in df.groupby('form'):
        style = style_guide.get(form_name, style_guide['default'])
        icon_path = style['icon_path']
        feature_group = folium.FeatureGroup(name=form_name.title())

        for _, row in form_df.iterrows():
            a_width = row.get('a_width')
            icon_size = DEFAULT_SIZE_PX
            if pd.notna(a_width):
                try:
                    calculated_size = int(float(a_width) * SCALING_FACTOR)
                    icon_size = max(MIN_ICON_SIZE_PX, min(calculated_size, MAX_ICON_SIZE_PX))
                except (ValueError, TypeError):
                    pass # Keep the default size

            popup_html = (f"<b>Code:</b> {row['code']}<br><b>Form:</b> {row['form']}<br>"
                          f"<b>Width:</b> {row.get('a_width', 'N/A')} m<br><b>Icon Size:</b> {icon_size} px")

            # Use the simple and reliable CustomIcon
            custom_icon = folium.CustomIcon(
                icon_image=icon_path,
                icon_size=(icon_size, icon_size)
            )

            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=popup_html,
                icon=custom_icon
            ).add_to(feature_group)

        feature_group.add_to(Map)

    # 5. Add legend and layer control
    Map.add_layer_control()


    # 6. SAVE TO HTML FOR FULL-SCREEN VIEW
    output_filename = 'fullscreen_static_map.html'
    Map.to_html(output_filename)

    print("-" * 50)
    print(f"\n✅ Map saved to '{output_filename}'.")
    print("This version has STATIC marker sizes (they do not change with zoom).")
    print("Download and open this file /kaggle/working/fullscreen_static_map.html in your browser for the full effect.")
    print("-" * 50)
    display(Map)




