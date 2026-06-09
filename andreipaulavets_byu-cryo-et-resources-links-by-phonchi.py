import requests
from IPython.display import Markdown, display
import base64

github_api_url = "https://api.github.com/repos/phonchi/Computational-CryoET/contents/README.md"

response = requests.get(github_api_url)
data = response.json()
content = base64.b64decode(data['content']).decode('utf-8')

# intro
intro_text = """# Intro
Thanks to [phonchi](https://github.com/phonchi) we have this amazing [list](https://github.com/phonchi/Computational-CryoET) of resources to check before diving into the competition.
Fetched directly from GitHub for transparency and to ensure up-to-date information. Enjoy!
"""

display(Markdown(intro_text))

display(Markdown(content))

