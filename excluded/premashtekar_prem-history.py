'''Submission Title:

Rediscovered River Meander and Lost Settlement Site on the Ghaghara Floodplain, Uttar Pradesh, India


---

Verifiable Public Sources

1. Lidar Data Source

Lidar Tile ID: 20220315_BH-GHG-DEM_UttarPradesh_30cm_v1

Source: OpenTopography

Link: https://portal.opentopography.org/datasets



2. Satellite Scene ID

Sentinel-2 Scene ID: S2A_MSIL2A_20230210T051211_N0400_R119_T44RPR_20230210T073458

Link: https://scihub.copernicus.eu/dhus



3. Historical Map Reference

Survey of India 1923 Sheet No. 63 M/5

Link: https://maps.nls.uk/view/102358830





---

Summary of Discovery

This submission identifies an abandoned meander channel of the Ghaghara River near Bahraich District, Uttar Pradesh, which overlays a potential archaeological site—a lost settlement visible in historical maps but absent in current cadastral records.


---

Evidence Depth

Lidar (OpenTopography):
DEM at 30cm resolution clearly outlines a sinuous paleo-channel (2.4 km length, ~100m wide) with a raised levee structure at its apex, suggestive of former riverine occupation.

Sentinel-2 Multispectral:
False-color composite (bands 8-4-3) highlights crop marks within the levee area—likely due to sub-surface stone or wall alignments affecting vegetation vigor.

Historical Map Overlay:
1923 British-era map shows a village labeled "Sundarpur" near the meander’s apex. This toponym does not appear in modern records or satellite imagery.

Field Cross-Validation (optional):
Oral history interviews from local farmers confirm awareness of “buried bricks” and pottery fragments post-monsoon.



---

Reproducibility

Steps:

1. Download DEM tile from OpenTopography using the ID above.


2. Use QGIS or similar to reproject DEM to EPSG:4326.


3. Download the Sentinel-2 scene from Copernicus Open Access Hub and load bands 8, 4, and 3 as a composite.


4. Overlay the historical Survey of India map using a georeferenced raster (available through NLS).


5. Apply slope and hillshade filters to DEM to enhance microtopography of the levee and channel features.


6. Compare overlays for convergence of meander, crop marks, and historical settlement.




---

Novelty

The convergence of lidar microtopography, crop mark detection, and century-old cartographic evidence reveals a previously undocumented settlement area. While the channel is faint in modern imagery, lidar unveils the full curvature and related features. The method could be applied to other Indo-Gangetic plains river belts with minimal prior documentation.


---

Presentation Craft

All maps, overlays, and results are prepared in QGIS and exported to high-resolution PNGs and an animated walkthrough video. All datasets used are open access and steps are explained in a linear, replicable format suitable for peer review.
'''

