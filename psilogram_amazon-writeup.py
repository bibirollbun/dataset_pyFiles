!pip install -q markdown2


from pathlib import Path
import shutil
import pandas as pd
Path("outputs").mkdir(exist_ok=True)

src = "/kaggle/input/model-description-prompt/Feature_importance_trends.png"
dst = "outputs/Feature_importance_trends.png"
shutil.copy(src, dst)

src = "/kaggle/input/create-amazon-cell-map/final_prediction_map_1000.0.png"
dst = "outputs/final_prediction_map.png"
shutil.copy(src, dst)

src = "/kaggle/input/create-amazon-cell-map/prospecting_score_map_1000.0.html"
dst = "outputs/prospecting_score_map.html"
shutil.copy(src, dst)


import pandas as pd

import pandas as pd

def generate_cluster_markdown(df, n=None):
    """
    Generates markdown text and a DataFrame for the top-scoring cell in each cluster (1 to n).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'cluster_rank', 'prospect_score', 
                           'lat_center', and 'lon_center'.
        n (int, optional): Number of clusters to include. If None, includes all unique cluster_ranks.

    Returns:
        tuple: (markdown_str, coordinates_df)
            - markdown_str (str): Markdown-formatted string.
            - coordinates_df (pd.DataFrame): DataFrame with columns 'Name', 'Latitude', 'Longitude'.
    """
    # Ensure proper sorting and grouping
    df_sorted = df.sort_values('prospect_score', ascending=False)
    top_cells = df_sorted.groupby('cluster_rank').first().reset_index()

    if n is not None:
        top_cells = top_cells[top_cells['cluster_rank'] <= n]

    # Generate markdown and coordinates DataFrame
    markdown_lines = []
    names = []
    latitudes = []
    longitudes = []

    for _, row in top_cells.sort_values('cluster_rank').iterrows():
        cluster_id = int(row['cluster_rank'])
        name = f"Cluster {cluster_id}"
        lat = row['lat_center']
        lon = row['lon_center']

        markdown_lines.append(
            f"### {name}\n"
            f"- **Latitude:** {lat:.6f}\n"
            f"- **Longitude:** {lon:.6f}\n"
        )

        names.append(name)
        latitudes.append(lat)
        longitudes.append(lon)

    coordinates_df = pd.DataFrame({
        'Name': names,
        'Latitude': latitudes,
        'Longitude': longitudes
    })

    return "\n".join(markdown_lines), coordinates_df



df = pd.read_csv("/kaggle/input/create-amazon-cell-map/clustered_cells.csv")
clusters, clusters_df = generate_cluster_markdown(df, n=10)
clusters_df.to_csv("clusters_df.csv", index=False)
clusters


from pathlib import Path
import json
from IPython.display import Markdown, display  # Imported for potential notebook rendering


def read_text(path: str) -> str:
    """Read a UTFâ€‘8 text file and return its stripped contents."""
    return Path(path).read_text(encoding="utf-8").strip()


# Dynamic content
terrain_analysis = read_text(
    "/kaggle/input/model-description-prompt/OpenAI_model_description.txt"
)

# Enhanced Markdown report without embedded images
markdown_report = f"""
# ğŸŒ� **Discovering Amazonian Archaeological Sites with AI**

## ğŸ—ºï¸� **Interactive Map & Archaeological Potential**

Explore the interactive HTML map (prospecting score map.html) to visualise and analyse promising areas for undiscovered 
archaeological sites:

- **Map Legend & Instructions:**
  - **Red Cells:** Highest likelihood of undiscovered sites
  - **Blue/White Cells:** Known archaeological locations; deprioritised for exploration
  - **Purple Clusters:** Groups of highâ€‘scoring cells
  - **Green Markers:** Known archaeological sites (marker size indicates significance)

**Interactions**

- **Hover** over cells/clusters to view detailed scores  
- **Click** cells for a Google Earth zoom  
- **Click** markers for detailed site information  
- **Toggle layers** for customised views; for example view results at different resolutions 

---

## ğŸŒ³ **Terrain Analysis & Archaeological Insights (GPTâ€‘4.1)**

{terrain_analysis}

---

## ğŸ“Š **Feature Importance Across Spatial Scales**

Refer to the separately uploaded figure for a visualisation of how feature importance shifts as spatial resolution 
increases (importance scores normalised per scale, 0â€“100).

---

## **Top Cluster Locations**
{clusters}

## ğŸ�¯ **Strategic Overview**

- Identify archaeological potential via distinct environmental signatures  
- Target regions minimally disturbed by modern activities  
- Use unsupervised learning (oneâ€‘class SVM) because negative cases are uncertain  
- Leverage GPTâ€‘4.1 for nuanced feature interpretation and archaeological relevance  
- **Note:** LIDAR images, while extensively processed during the exploration stages, are not used in the final model. 
LIDAR coverage is sparse and existing datasets have probably already been thoroughly analysed. Promising areas identified
here should, however, be surveyed with LIDAR in future work.

---

## ğŸ› ï¸� **Detailed Methodology**

1. **Data Acquisition & Preparation**  
   Compile and weight archaeological site locations  
2. **Hierarchical Spatial Analysis**  
   Partition the Amazon Basin into grids, refining resolution progressively, and filter by forest cover to focus exploration  
3. **Environmental Feature Engineering**  
   Generate key variablesâ€”elevation, slope, flood risk, proximity to rivers. A key part of the data preparation is
   converting DEM values to relative elevation above the closest river  
4. **Predictive Modelling**  
   Train a oneâ€‘class SVM on known site data only, then rank grid cells by archaeological potential  
5. **Iterative Resolution Enhancement**  
   Refine predictions iteratively at higher spatial detail  
6. **AIâ€‘Driven Interpretation**  
   Apply GPTâ€‘4.1 for deeper insight into landscape suitability  

---

## âš ï¸� **Recommendations & Next Steps**

This analysis provides a framework for archaeological exploration, but improvements in data quality, feature refinement
and validation are necessary. Future iterations should include automated satellite analysis for geometric feature detection
and additional processing power to enable finer spatial resolutions. The current version downsamples DEM values to 270â€¯m,
adequate for large landforms such as plateaus, but potentially too coarse for subtle terrain modifications created by ancient civilisations.

---

## ğŸ—ƒï¸� **Data Sources & References**

- **Topographic Data:** [MERIT DEM](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/) (Yamazaki etâ€¯al.)  
- **Biomass:** [NASA Biomass Map](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1847)  
- **Flood Zones:** [NASA Amazon Flood Map](https://earthdata.nasa.gov/)  
- **River Network:** [ORNL DAAC](https://daac.ornl.gov/LBA/guides/CD06_Amazon_River_Network.html)  
  - [Amazon Geoglyphs](https://www.jqjacobs.net/blog/)  
  - [Science Data Repository](https://www.science.org/doi/10.1126/science.ade2541)  
  - [Preâ€‘Columbian Earthbuilders Study](https://www.nature.com/articles/s41467-018-03510-7)  
  - [Llanos de Mojos site from LIDAR](https://www.nature.com/articles/s41586-022-04780-4)  
  - ChatGPTâ€‘Scholar insights  

---

## ğŸ“’ **Project Notebooks**

- `collect-amazon-sites.ipynb` â€” Site data preparation  
- `make-amazon-polygon.ipynb` â€” Define Amazon region  
- `download_MERIT_DEM_Amazon.ipynb` â€” Elevation data retrieval  
- `download_SWOT_Amazon.ipynb` â€” Floodâ€‘zone data retrieval  
- `elevation-above-river-amazon.ipynb` â€” Elevation relative to rivers  
- `merit-dem-slope-curv-prof-tpi-tri.ipynb` â€” Terrain feature derivation  
- `amazon-grid-analysis.ipynb` â€” Predictive modelling  
- `create-amazon-cell-map.ipynb` â€” Visualisation generation  
- `model_description_prompt.ipynb` â€” AI terrain interpretation  
- `amazon-writeup.ipynb` â€” Final documentation  
"""

# Write Markdown to file
Path("submission.md").write_text(markdown_report, encoding="utf-8")





import markdown2
from IPython.display import HTML

md = Path("submission.md").read_text()
HTML(markdown2.markdown(md))


print(md)




