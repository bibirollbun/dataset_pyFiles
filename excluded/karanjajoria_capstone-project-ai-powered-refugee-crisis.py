!git clone https://github.com/karanjajoria/RUBIX.git /kaggle/working/project
%cd /kaggle/working/project


!pip install -r requirements.txt


import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyDQ4Xl7BXVEwt6ive_uJymsUl7-WD6gejA'


!python main.py --mode demo

