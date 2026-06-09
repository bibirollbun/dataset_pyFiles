import base64
from IPython.display import HTML, display, Image
import sys 
sys.path.insert(1, '/kaggle/input/bdb2025')  # Full path
from nfl_animator import Play
import plotly.offline as pyo


def gif_to_html(gif_path):
    with open(gif_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{encoded}" width="1200" height="700">'

# Path to the GIF
gif_path = '/kaggle/input/gif-vfinal/output.gif'

# Display the GIF
HTML(gif_to_html(gif_path))


def gif_to_html(gif_path):
    with open(gif_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{encoded}" width="1200" height="700">'

# Path to the GIF
gif_path = '/kaggle/input/updated-assets/coverage_metrics_with_roc (3).gif'

# Display the GIF
HTML(gif_to_html(gif_path))


# Load and display the image
display(Image(filename='/kaggle/input/attention/panel_attention.png'))



pyo.init_notebook_mode(connected=True) 
pyo.iplot(Play(game_id=2022101610, play_id=807)._figure)


sys.path.insert(1, '/kaggle/input/bdb2025')  # Full path
from nfl_animator import Play
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True) 
pyo.iplot(Play(game_id=2022090800, play_id=212)._figure)


def gif_to_html(gif_path):
    with open(gif_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{encoded}" width="1200" height="700">'

# Path to the GIF
gif_path = '/kaggle/input/212play/coverage_classification_212.gif'

# Display the GIF
HTML(gif_to_html(gif_path))



# Display the visualization for Misclassification Rate
from IPython.display import display, Image

# Path to the updated image in the Kaggle environment
image_path = '/kaggle/input/leaderboardsv10/leaderboardsv10.png'
display(Image(filename=image_path, width=1200, height=700))


# Path to the updated image in the Kaggle environment
image_path = '/kaggle/input/submission-assets-v2/submission_assets/kde_bdb2025.png'
display(Image(filename=image_path, width=1200, height=700))


# Path to the updated image in the Kaggle environment
image_path = '/kaggle/input/corplotv5/corrv5.png'
display(Image(filename=image_path, width=1200, height=700))


def gif_to_html(gif_path):
    with open(gif_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{encoded}" width="1200" height="700">'

# Path to the GIF
gif_path = '/kaggle/input/coverage-classification/coverage_classification (3).gif'

# Display the GIF
HTML(gif_to_html(gif_path))


from IPython.display import Image, display

# Load and display the image
display(Image(filename='/kaggle/input/arch-diagram-v2/model_arch.png',  width=1200, height=700))


