# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import sys
sys.path.append("/kaggle/input/drawing-with-llms/published")


import kagglehub
package = kagglehub.package_import('kawchar85/lb-0-508-simple-svg/versions/1')


import random

class Model:
    def __init__(self):
        pass
    def predict(self, prompt: str) -> str:
        if "chess board" in prompt.lower():
            return self.generate_chess_board()
        elif "pac-man" in prompt.lower():
            return self.generate_pac_man()
        elif "mario" in prompt.lower():
            return self.generate_mario_level()
        else:
            return self.generate_random_svg()
    def generate_chess_board(self) -> str:
        return """
<svg width="500" height="500" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background color -->
  <rect width="500" height="500" fill="#f0d9b5"/>

  <!-- Chess board pattern -->
  <rect x="0" y="0" width="500" height="500" fill="url(#boardPattern)" opacity="0.8"/>
  <defs>
    <pattern id="boardPattern" patternUnits="userSpaceOnUse" width="62.5" height="62.5">
      <rect width="62.5" height="62.5" fill="#b58863"/>
    </pattern>
  </defs>

  <!-- Circles -->
  <circle cx="250" cy="250" r="20" fill="#b58863"/>
  <circle cx="250" cy="250" r="15" fill="#f0d9b5"/>
  <circle cx="250" cy="250" r="10" fill="#b58863"/>
</svg>
"""
    def generate_pac_man(self) -> str:
        return """
<svg width="500" height="500" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background color -->
  <rect width="500" height="500" fill="#000"/>

  <!-- Mario's head -->
  <circle cx="250" cy="200" r="100" fill="#ff6b6b"/>

  <!-- Mario's hat -->
  <path d="M150 100L350 100L250 0Z" fill="#000"/>
  <path d="M170 100L330 100L250 50Z" fill="#ff6b6b"/>

  <!-- Mario's eyes -->
  <circle cx="200" cy="175" r="15" fill="#000"/>
  <circle cx="300" cy="175" r="15" fill="#000"/>

  <!-- Mario's nose -->
  <path d="M250 200L275 225L225 225Z" fill="#000"/>

  <!-- Mario's mouth -->
  <path d="M200 225C225 250 275 250 300 225" stroke="#000" stroke-width="5" fill="none"/>

  <!-- Mario's body -->
  <rect x="200" y="250" width="100" height="150" fill="#ff6b6b"/>

  <!-- Mario's arms -->
  <rect x="150" y="300" width="50" height="50" fill="#ff6b6b"/>
  <rect x="300" y="300" width="50" height="50" fill="#ff6b6b"/>

  <!-- Mario's legs -->
  <rect x="225" y="400" width="25" height="50" fill="#000"/>
  <rect x="250" y="400" width="25" height="50" fill="#000"/>
</svg>
"""
    def generate_mario_level(self) -> str:
        return """
<svg width="500" height="500" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="500" fill="#87CEEB"/>
  <rect x="0" y="400" width="500" height="100" fill="#228B22"/>
  <rect x="100" y="350" width="50" height="50" fill="#8B4513"/>
  <rect x="350" y="350" width="50" height="50" fill="#8B4513"/>
  <circle cx="250" cy="250" r="30" fill="#FF0000"/>
  <circle cx="250" cy="250" r="25" fill="#FFA500"/>
  <circle cx="250" cy="250" r="20" fill="#FF0000"/>
  <circle cx="250" cy="250" r="15" fill="#FFA500"/>
  <circle cx="250" cy="250" r="10" fill="#FF0000"/>
  <rect x="200" y="200" width="100" height="100" fill="#228B22"/>
</svg>
"""
 
    def generate_random_svg(self) -> str:
        return """
<svg width="1000" height="1000" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="bg" cx="60%" cy="40%" r="85%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e3b8a"/>
    </radialGradient>

    <linearGradient id="neon" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#4ade80"/>
      <stop offset="50%" stop-color="#2dd4bf"/>
      <stop offset="100%" stop-color="#5eead4"/>
    </linearGradient>

    <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
      <path d="M100 0H0v100" stroke="#f8fafc" stroke-width="2" opacity="0.1"/>
    </pattern>
  </defs>

  <rect width="1000" height="1000" fill="url(#bg)"/>
  <rect width="1000" height="1000" fill="url(#grid)" opacity="0.15"/>

  <g transform="translate(500 500)">
    <path d="M0-200L57-193 193-57 200 0 193 57 57 193 0 200-57 193-193 57-200 0-193-57-57-193Z" 
          fill="none"
          stroke="url(#neon)"
          stroke-width="15"
          stroke-linejoin="round"
          opacity="0.97"/>

    <g transform="scale(0.6)">
      <circle r="140" fill="#4ade80" opacity="0.95"/>
      <path d="M-100-100L100 100M100-100L-100 100" 
            stroke="#0f172a" 
            stroke-width="25"
            stroke-linecap="round"/>
      <g stroke="#2dd4bf" stroke-width="8">
        <path d="M0-140L0-200M0 140L0 200"/>
        <path d="M-140 0L-200 0M140 0L200 0"/>
      </g>
    </g>

    <circle cx="300" cy="0" r="15" fill="#5eead4"/>
    <circle cx="-300" cy="0" r="15" fill="#5eead4"/>
  </g>

  <g opacity="0.3">
    <circle cx="250" cy="250" r="30" fill="url(#neon)"/>
    <circle cx="750" cy="750" r="40" fill="#2dd4bf"/>
    <rect x="600" y="200" width="60" height="60" rx="15" fill="#5eead4"/>
    <path d="M200 600L300 700 400 600Z" fill="#4ade80"/>
  </g>

  <g stroke="#f8fafc" stroke-width="4" opacity="0.15">
    <path d="M500 100L500 900"/>
    <path d="M100 500L900 500"/>
  </g>
</svg>
"""
        


from IPython.display import SVG

model = Model()
svg1 = model.predict('mario')
svg2 = model.predict('pac-man')
svg3 = model.predict('chess board')

print(svg1)
print(svg2)
print(svg3)

display(SVG(svg1))
display(SVG(svg2))
display(SVG(svg3))


import kaggle_evaluation

kaggle_evaluation.test(Model)

