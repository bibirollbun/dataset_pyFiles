# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/travel-dataset-guide-to-indias-must-see-places'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


Claude Code sessions:
Sessions
Active
Add divider and scrollable area for destinations

yifon8/Yatra

Add beach destination types to mapping

yifon8/Yatra

+1
-1
Merged
Fix destination type enum selections 2-4

yifon8/Yatra

+34
-14
Merged
Fix visit duration assigned to travel time variable

yifon8/Yatra

+10
-13
Merged
Check Yatra codebase for MCP server usage

yifon8/Yatra

Remove family travel UI elements and constraint

yifon8/Yatra

+20
-116
Merged
Fix unreliable search results for user inputs

yifon8/Yatra

+130
-20
Merged
Include destinations with unknown entry fees

yifon8/Yatra

+29
-3
Merged
Fix budget constraint validation for destinations

yifon8/Yatra

+2
-1
Merged
Fix Yatra agent mountain destination search

yifon8/Yatra

+22
-3
Merged
Fix server 400 error in recommendations

yifon8/Yatra

+18
-7
Merged
Fix null list error in web form input

yifon8/Yatra

+25
-0
Merged
Update all models to Gemini 2.5 Flash Lite

yifon8/Yatra

+3
-4
Merged
Add new command console feature

yifon8/Yatra

+1
-1
Closed
Debug pandas filter search in CLI console

yifon8/Yatra

+1
-1
Merged
Fix invalid Google API key error

yifon8/Yatra

+3
-3
Merged
Fix filter submission results display

yifon8/Yatra

+128
-17
Merged
Add pandas filtering to yatra_agent destinations

yifon8/Yatra

+239
-14
Merged
Make budget field optional with zero validation

yifon8/Yatra

+17
-12
Merged
Update visit duration to decimal hours format

yifon8/Yatra

+33
-28
Merged
Fix agent initialization error on Windows

yifon8/Yatra

+45
-1
Merged
Fix import error in web server startup

yifon8/Yatra

+1
-1
Merged
Add user input handling from web form

yifon8/Yatra

+483
-0
Merged
Rename destination_suggester to yatra_agent

yifon8/Yatra

+0
-0
Merged
Build travel destination suggestion agent

yifon8/Yatra

+1969
-0
Merged
Add retry button to results container

yifon8/Yatra

+31
-0
Merged
Rename Travel Duration to Visit Duration

yifon8/Yatra

+1
-1
Merged
Add results container and fix styling layout

yifon8/Yatra

+96
-73
Merged
Update color scheme and add destination suggestions

yifon8/Yatra

+87
-5
Merged
Create travel planning form with destination and preferences

yifon8/Yatra

