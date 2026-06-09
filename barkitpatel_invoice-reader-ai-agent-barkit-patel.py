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


!pip install pytesseract pillow pandas python-docx
!apt-get update
!apt-get install -y tesseract-ocr


import pytesseract
from PIL import Image
import pandas as pd
import json
import re
from google.colab import files

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


uploaded = files.upload()

for fn in uploaded.keys():
    print("Uploaded File:", fn)
    invoice_path = fn


image = Image.open(invoice_path)
extracted_text = pytesseract.image_to_string(image)

print("ðŸ“Œ Extracted Text:")
print(extracted_text)


def find_field(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not Found"

data = {
    "Vendor": find_field(r"Vendor[:\s]*(.*)", extracted_text),
    "Invoice Number": find_field(r"Invoice\s*No[:\s]*(.*)", extracted_text),
    "Date": find_field(r"Date[:\s]*(.*)", extracted_text),
    "Total Amount": find_field(r"Total[:\s$]*(.*)", extracted_text)
}

df = pd.DataFrame([data])
df


json_output = "invoice_extracted.json"
with open(json_output, "w") as f:
    json.dump(data, f, indent=4)

print("JSON saved:", json_output)
files.download(json_output)


csv_output = "invoice_extracted.csv"
df.to_csv(csv_output, index=False)

print("CSV saved:", csv_output)
files.download(csv_output)

