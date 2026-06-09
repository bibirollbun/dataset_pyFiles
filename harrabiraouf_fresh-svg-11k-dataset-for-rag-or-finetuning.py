%%capture
!pip install cairosvg


import pandas as pd
import base64
from io import BytesIO
from IPython.display import HTML, display
import cairosvg
from PIL import Image

# Read CSV file
df = pd.read_csv('/kaggle/input/scored-svg-11k/svg_dataset_scored_11k.csv').sample(9)

def svg_to_png(svg_code: str, size: tuple = (200, 200)) -> Image.Image:
    """Convert SVG string to PNG using CairoSVG (your provided function)"""
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(BytesIO(png_data)).convert('RGB').resize(size)

# Create HTML table
html = '<table style="width:100%; border-collapse: collapse;">'
html += '<tr><th>Sentence</th><th>Score</th><th>SVG Image</th></tr>'

for _, row in df.iterrows():
    # Convert SVG to PNG
    image = svg_to_png(row['svg_code'])
    
    # Convert image to base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    png_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    html += f'<tr style="border: 1px solid #ddd;">' \
            f'<td style="padding: 10px;">{row["sentence"]}</td>' \
            f'<td style="padding: 10px;">{row["best_image_score"]}</td>' \
            f'<td style="padding: 10px;"><img src="data:image/png;base64,{png_base64}"/></td>' \
            f'</tr>'

html += '</table>'

# Display the table
display(HTML(html))




