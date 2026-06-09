# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sub = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/sample_submission.csv')
sub.tail()


train = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
train.tail()


labels = 'Secondary', 'Missing','Primary'
sizes = [449, 309, 270]  #must have same number labels, sizes and explode
explode = (0, 0.2, 0)  # only "explode" the 2nd slice 

fig1, ax1 = plt.subplots(figsize=(4,4))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Citation types')
plt.show()


train['dataset_id'].value_counts()


#By Birolkuymcu https://www.kaggle.com/code/birolkuyumcu/library-of-inventions-and-innovations

import xml.etree.ElementTree as x
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud,STOPWORDS

d = []
s = ""
f = open("../input/make-data-count-finding-data-references/train/XML/10.1002_anie.201916483.xml")
i = 0
for l in f:
    s += l
    print(i,l)
    i += 1
    if i < 160:
        continue
    if i == 200 :
        break
d.append(s)


#By Matt Fortier https://www.kaggle.com/code/fortiema/library-of-inventions-and-innovations

d = []
s = ""
f = open("../input/make-data-count-finding-data-references/train/XML/10.1002_anie.201916483.xml")
for l in f:
    if l == "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n":
        if len(s)>0:
            d.append(s)
        s = ""
    s += l
d.append(s)

s = []

for xm in d:
    root = x.fromstring(xm)
    for e in root.iter(tag="institution"):
        s.append(str(e.text))

print(s[:3])


!pip install PyPDF2


# note the capitalization
import PyPDF2


# Notice we read it as a binary with 'rb'
f = open('../input/make-data-count-finding-data-references/train/PDF/10.1021_acs.jcim.9b01185.pdf','rb')


pdf_reader = PyPDF2.PdfReader(f)


len(pdf_reader.pages)


page_36 = pdf_reader.pages[36]


from PyPDF2 import PdfReader

reader = PdfReader("../input/make-data-count-finding-data-references/train/PDF/10.1021_acs.jcim.9b01185.pdf")
page = reader.pages[35]#Below it's page 36
print(page.extract_text())


!pip install gTTS


#By Gaurav Dutta https://www.kaggle.com/gauravduttakiit/working-with-pdf-files

f = open('../input/make-data-count-finding-data-references/train/PDF/10.1021_acs.jcim.9b01185.pdf','rb')

# List of every page's text.
# The index will correspond to the page number.
pdf_text = [35]  # Original was 0. zero is a placehoder to make page 1 = index 1

pdf_reader = PyPDF2.PdfReader(f)#PdfFileReader is deprecated

for p in range(len(pdf_reader.pages)):
    
    page = pdf_reader.pages[35]#getPage(p) is deprecated
    
    pdf_text.append(page.extract_text())#extractText is deprecated

f.close()


pdf_text[35]


#By Gaurav Dutta https://www.kaggle.com/gauravduttakiit/working-with-pdf-files

print(pdf_text[35])


#By Gaurav Dutta https://www.kaggle.com/gauravduttakiit/working-with-pdf-files

from gtts import gTTS
tts = gTTS(pdf_text[35])
tts.save('output.mp3')


#By Gaurav Dutta https://www.kaggle.com/gauravduttakiit/working-with-pdf-files

import IPython
file = "./output.mp3"# change the folder and file accordingly 
IPython.display.display(IPython.display.Audio(file))

