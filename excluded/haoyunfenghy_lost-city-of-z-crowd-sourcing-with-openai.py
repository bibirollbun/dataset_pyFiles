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
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Baseline - use openAI model directly 
from kaggle_secrets import UserSecretsClient
import os
from openai import OpenAI

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = secret_value_0

client = OpenAI()


# I have encountered an issue using chatGPT API to handle images. 
# Although correct image format was given, the API reported error of incorrect image type.
# However chatGPT frontend works fine. I'll directly use the frontend for the baseline

"""
prompt = "You are an expert for discovering archeology sites. Can you tell me if there is a site in this image. Tell me how did you reach the conclusion."

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": "https://storage.cloud.google.com/arcknow-test-public/new_site_gee_landsat8.png",
            },
        ],
    }],
)

print(response.output_text)
"""


from kaggle_secrets import UserSecretsClient
import os
user_secrets = UserSecretsClient()
secret_value_1 = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = secret_value_1


!pip install google-genai


from google import genai

client = genai.Client()


from google.genai import types

#myfile = client.files.upload(file="/kaggle/input/googleearth-download/google_earth_download/known_sites/abuna_oval.jpg")
fp="/kaggle/input/googleearth-download/google_earth_download/known_sites/abuna_oval.jpg"
with open(fp, 'rb') as f:
    img_bytes = f.read()
prompt = """
You are an expert for discovering archeology sites. 
Tell me if there is a archeological site in the given satellite image. 
Tell me how did you reach the conclusion.
"""

from pydantic import BaseModel

class ExamineResult(BaseModel):
    contain_site: bool
    explanation: str

result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ExamineResult,
        },
    )

result.text


from enum import Enum
import json
from tqdm import tqdm

class Task(Enum):
    teacher = 0,
    student = 1,
    end = 2

class ExamineResult(BaseModel):
    contain_site: bool
    explanation: str

class PromptUpdate(BaseModel):
    new_prompt: str
    explanation: str

class SiteExaminer:
    def __init__(self, prompt, train_data):
        self.prompt = [prompt]
        self.todo = Task.student
        self.img_id = 0
        self.train_data = train_data #[{"image":<file path>, "gt":<ground truth in bool>}]
        self.explain = None
    
    def studentRound(self):
        if self.img_id >= len(self.train_data):
            self.todo = Task.end
            return 0
        prompt = self.prompt[-1]
        imagef = self.train_data[self.img_id]["image"]
        gt = self.train_data[self.img_id]["gt"]
        print(imagef)
        #myfile = client.files.upload(file=imagef)
        with open(imagef, 'rb') as f:
            img_bytes = f.read()
        result = client.models.generate_content(
            model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExamineResult,
                },
        )
        r = result.parsed
        if r.contain_site != gt:
            self.todo = Task.teacher
            self.explain = r.explanation
        else:
            self.todo = Task.student
            self.img_id += 1
        return 0
        
    def teacherRound(self):
        prompt = f"""
        You are an expert for discovering archeological sites. 
        You will give prompt instructions to a student 
        for discovering archelogical sites from satellite images correctly.
        In the previous examnination, the student made a mistake on the given image. 
        You will adjust previous prompt instruction, 
        so the student will make correct prediction for images with similar pattern.
        Output the exact prompt in new_prompt field and leave other context in explanation field.

        Previous examnination result: 
        The expected results is {self.train_data[self.img_id]["gt"]},
        while the student precited {not self.train_data[self.img_id]["gt"]}
        If the result from examination is True, 
        it means there IS evidence to archeological site in the satellite image. 
        Most images with archelogical sites show unusal square shaped dent on earth. 
        Make sure student pays attention to it. 
        If the result from examination is False, 
        it means there is NO evidence to archeological site in the satellite image. 
        
        Previous prompt:
        {self.prompt[-1]}

        Previous explanation from student on why was the incorrect prediction made:
        {self.explain}

        Image: 
        
        """
        #myfile = client.files.upload(file=self.train_data[self.img_id]["image"])
        imagef = self.train_data[self.img_id]["image"]
        with open(imagef, 'rb') as f:
            img_bytes = f.read()
        result = client.models.generate_content(
            model="gemini-2.0-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    )
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PromptUpdate,
                },
        )
        self.prompt.append(result.parsed.new_prompt)
        self.img_id += 1
        self.todo = Task.student
        return 0

    def train(self):
        while self.todo != Task.end:
            if self.todo == Task.student:
                print(f"Student round on image_id: {self.img_id}, out of {len(self.train_data)}")
                self.studentRound()
            elif self.todo == Task.teacher:
                print(f"Teacher round on image_id: {self.img_id}, out of {len(self.train_data)}")
                self.teacherRound()
            else:
                print(f"Invalid todo task: {self.todo}")
                return 0
        return 0

    def test(self, test_data):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pred = []
        for data in tqdm(test_data):
            imagef = data["image"]
            gt = data["gt"]
            #myfile = client.files.upload(file=imagef)
            with open(imagef, 'rb') as f:
                img_bytes = f.read()
            prompt = self.prompt[-1]
            result = client.models.generate_content(
                model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type='image/jpeg',
                        ),
                        prompt
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ExamineResult,
                    },
            )
            pred.append(result.parsed)
            if gt and result.parsed.contain_site:
                TP += 1
            elif gt and not result.parsed.contain_site:
                FN += 1
            elif not gt and not result.parsed.contain_site:
                TN += 1
            else:
                FP += 1
        print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
        return pred

    def infer(self, imagef):
        #myfile = client.files.upload(file=imagef)
        with open(imagef, 'rb') as f:
            img_bytes = f.read()
        prompt = self.prompt[-1]
        result = client.models.generate_content(
            model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExamineResult,
                },
        )
        return result.parsed
            
        


import random

train_data = []
train_data_o = []
test_data = []

for dirname, _, filenames in os.walk('/kaggle/input/googleearth-download/google_earth_download/known_sites'):
    for filename in filenames:
        if filename != "abuna_oval.png":
            train_data.append({
                "image":os.path.join(dirname, filename),
                "gt": True
            })
            train_data_o.append({
                "image":os.path.join(dirname, filename),
                "gt": True
            })

for dirname, _, filenames in os.walk('/kaggle/input/googleearth-download/google_earth_download/random_pics'):
    for filename in filenames[:10]:
        if filename != "arcs7.jpg":
            train_data.append({
                "image":os.path.join(dirname, filename),
                "gt": False
            })
            train_data_o.append({
                "image":os.path.join(dirname, filename),
                "gt": False
            })
for dirname, _, filenames in os.walk('/kaggle/input/googleearth-download/google_earth_download/random_pics'):
    for filename in filenames[:10]:
        if filename != "arcs7.jpg":
            train_data.append({
                "image":os.path.join(dirname, filename),
                "gt": False
            })
random.shuffle(train_data)

for dirname, _, filenames in os.walk('/kaggle/input/googleearth-download/google_earth_download/known_sites_2'):
    for filename in filenames:
        test_data.append({
            "image":os.path.join(dirname, filename),
            "gt": True
        })

for dirname, _, filenames in os.walk('/kaggle/input/googleearth-download/google_earth_download/random_pics'):
    for filename in filenames[10:]:
        test_data.append({
            "image":os.path.join(dirname, filename),
            "gt": False
        })
        

prompt = """
You are an expert for discovering archeology sites. 
Can you tell me if there is a site in this image. 
Tell me how did you reach the conclusion.
"""

#siteE = SiteExaminer(prompt, train_data)


siteE = SiteExaminer(prompt, train_data_o)


r1 = siteE.test(train_data_o)


r2 = siteE.test(test_data)


siteE.train()


siteE.img_id = 0
siteE.todo = Task.student
siteE.train()


r3 = siteE.test(train_data_o)


r4 = siteE.test(test_data)


def infer(imagef):
    with open(imagef, 'rb') as f:
        img_bytes = f.read()
    with open("/kaggle/input/googleearth-download/google_earth_download/known_sites/abuna_oval.jpg", 'rb') as f:
        img_bytes_0 = f.read()

    with open("/kaggle/input/googleearth-download/google_earth_download/random_pics/arcs5.jpg", 'rb') as f:
        img_bytes_2 = f.read()

    with open("/kaggle/input/googleearth-download/google_earth_download/known_sites/acrs_22.jpg", 'rb') as f:
        img_bytes_1 = f.read()

    prompt0 = """
    The above satellite images contain archeological site evidence. 
    Close to the center of the images, there are unusal square shapes. 
    The land is depressed on the edge of the square outline. 
    The following image does not contain archelogical site evidence.
    Refer to this image to minimize false positive. 
    """

    prompt1 = """
    Examine the following picture closely and tell me 
    if you observed similar archeological evidence as the above given images. 
    In this prediction, you want to minimize false positive.
    """
    result = client.models.generate_content(
        model="gemini-1.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=img_bytes_0,
                    mime_type='image/jpeg',
                ),
                types.Part.from_bytes(
                    data=img_bytes_1,
                    mime_type='image/jpeg',
                ),
                prompt0,
                types.Part.from_bytes(
                    data=img_bytes_2,
                    mime_type='image/jpeg',
                ),
                prompt1,
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/jpeg',
                )
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ExamineResult,
            },
    )
    return result.parsed
    

def test(test_data):
    pred = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pred = []
    for data in tqdm(test_data):
        imagef = data["image"]
        gt = data["gt"]
        result = infer(imagef)
        pred.append((result, data))
        if gt and result.contain_site:
            TP += 1
        elif gt and not result.contain_site:
            FN += 1
        elif not gt and not result.contain_site:
            TN += 1
        else:
            FP += 1
    print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
    return pred


pred = test(train_data_o)


pred = test(test_data)


pred


r = infer("/kaggle/input/test-png/test.png")
r


#!pip install split-image


#from split_image import split_image

#split_image("/kaggle/input/lostz-inference/around_acdoc_unmarked.jpg", 15, 15, False, False)


def discover(filename, start, end):
    predictions=[]
    failed = []
    for i in tqdm(range(start, end)):
        try:
            imagef = f"/kaggle/input/lostz-inference-1/inference/{filename}_{i}.jpg"
            pred = infer(imagef)
            predictions.append(
                {
                    "filename": imagef,
                    "pred": pred
                }
            )
        except:
            print(f"Failed on {imagef}")
            failed.append(imagef)
            pass
    return predictions, failed


r = discover("around_acdoc_unmarked", 0, 225)


[i for i,p in enumerate(r[0]) if p["pred"].contain_site]


r1 = discover("around_site_unmarked", 0, 225)


len(r1[0])


r1[0][0]


def dms_to_dd(degrees, minutes, seconds):
    """Converts degrees, minutes, and seconds to decimal degrees.

    Args:
        degrees: The degrees value.
        minutes: The minutes value.
        seconds: The seconds value.

    Returns:
        The decimal degree value.
    """
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    return dd


def decimal_to_dms(decimal_degrees):
    """
    Converts a decimal degree value to degrees, minutes, and seconds (DMS).

    Args:
        decimal_degrees (float): The decimal degree value.

    Returns:
        tuple: A tuple containing the degrees, minutes, and seconds.
    """
    # Determine the sign for direction (N/S or E/W)
    sign = -1 if decimal_degrees < 0 else 1
    abs_decimal_degrees = abs(decimal_degrees)

    # Extract degrees
    degrees = int(abs_decimal_degrees)

    # Calculate minutes
    remaining_decimal = abs_decimal_degrees - degrees
    minutes_float = remaining_decimal * 60
    minutes = int(minutes_float)

    # Calculate seconds
    seconds = (minutes_float - minutes) * 60
    seconds = round(seconds, 2) # Round to two decimal places for seconds

    return degrees * sign, minutes, seconds


dms_to_dd(61, 17, 24)


#acdoc
-9.81, -67.26
-9.94, -67.06

#site
-10.45, -61.47
-10.59, -61.25


def coord(n, scope):
    coord = [None, None]
    row = int(n/15)
    col = n - row * 15
    coord[0] = scope[0][0] + (scope[1][0] - scope[0][0])/15 * row
    coord[1] = scope[0][1] + (scope[1][1] - scope[0][1])/15 * col
    return coord


scope = ((-9.81, -67.26), (-9.94, -67.06))
for n in [i for i,p in enumerate(r[0][:225]) if p["pred"].contain_site]:
    c = coord(n, scope)
    print(r[0][n]["filename"])
    print(c)
    print(r[0][n]["pred"])


scope = ((-10.45, -61.47), (-10.59, -61.25))
for n in [i for i,p in enumerate(r[0][225:]) if p["pred"].contain_site]:
    c = coord(n, scope)
    print(r[0][n+225]["filename"])
    print(c)
    print(r[0][n+225]["pred"])

