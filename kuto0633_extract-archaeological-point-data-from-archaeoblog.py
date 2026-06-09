from pathlib import Path
import pandas as pd
import json
import xml.etree.ElementTree as ET


OUTPUT_DIR = "."


# !mkdir {OUTPUT_DIR}


!curl -o {OUTPUT_DIR}/archaeogeodesy.kml https://jqjacobs.net/kml/archaeogeodesy.kml
!curl -o {OUTPUT_DIR}/amazon_geoglyphs.kml https://www.jqjacobs.net/amazon/amazon_geoglyphs.kml
!curl -o {OUTPUT_DIR}/amazon_results.kml https://www.jqjacobs.net/amazon/amazon_results.kml


path1 = f"{OUTPUT_DIR}/archaeogeodesy.kml"
path2 = f"{OUTPUT_DIR}/amazon_geoglyphs.kml"
path3 = f"{OUTPUT_DIR}/amazon_results.kml"


def parse_kml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    placemarks = root.findall('.//kml:Placemark', ns)
    data = []
    for pm in placemarks:
        # pointデータのみ抽出（Pointジオメトリを持つもの）
        point_elem = pm.find('.//kml:Point', ns)
        if point_elem is None:
            continue  # LineStringやその他のジオメトリはスキップ
            
        name_elem = pm.find('kml:name', ns)
        name = name_elem.text if name_elem is not None else ''
        coord_elem = point_elem.find('kml:coordinates', ns)
        if coord_elem is not None and coord_elem.text:
            raw = coord_elem.text.replace('>', '').strip()
            coords = raw.split()
            lon, lat, *_ = coords[0].split(',')
            try:
                data.append({'name': name, 'latitude': float(lat), 'longitude': float(lon)})
            except ValueError:
                print(f"Skipping invalid coordinates for {name}: {raw}")
    return pd.DataFrame(data)


df1 = parse_kml(path1)
df2 = parse_kml(path2)
df3 = parse_kml(path3)


df1


df2


df3


df1['source'] = "archaeogeodesy"
df2['source'] = "amazon_geoglyphs"
df3['source'] = "amazon_results"
all_df = pd.concat([df1, df2, df3], ignore_index=True)
all_df = all_df.reset_index().rename(columns={"index": "id"})
all_df.to_csv(f"{OUTPUT_DIR}/all_points.csv", index=False)


!pip install pydeck


import pydeck as pdk

def plot_points(df: pd.DataFrame) -> pdk.Deck:
    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=1000,  # Radius is given in meters
    get_fill_color=[240, 0, 200, 140],
    pickable=True
    )

    view_state = pdk.ViewState(
        longitude=-70, 
        latitude=-10, 
        zoom=2,
    )

    # Render
    r = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_ROAD,
    )
    return r


plot_points(all_df)

