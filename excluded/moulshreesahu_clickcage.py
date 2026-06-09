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


import re, os, time, json, hashlib, urllib.parse, random
from typing import List
print('Environment ready.')



URL_RE = re.compile(r'(https?://[^\s]+)')

def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text)

def normalize_url(url: str) -> str:
    p = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qsl(p.query)
    q = [(k,v) for k,v in q if not k.lower().startswith('utm_')]
    return urllib.parse.urlunparse((p.scheme, p.netloc, p.path, p.params, urllib.parse.urlencode(q), p.fragment))



import hashlib
class MockVirusTotal:
    def __init__(self):
        pass
    def url_report(self, url):
        h = int(hashlib.sha256(url.encode()).hexdigest(), 16)
        score = (h % 100) / 100.0
        if score < 0.05:
            return {'verdict':'malicious','positives':20,'total':70,'scanner_score':score}
        elif score < 0.25:
            return {'verdict':'suspicious','positives':3,'total':70,'scanner_score':score}
        else:
            return {'verdict':'clean','positives':0,'total':70,'scanner_score':score}
VT = MockVirusTotal()



def mock_gemini_classify(message_text, url=None):
    score = 0.0
    t = message_text.lower()
    if 'crack' in t or 'free' in t or 'download' in t:
        score += 0.5
    if url and any(ext in url for ext in ['.exe','.zip','.scr','.bat']):
        score += 0.4
    if 'steam' in (url or '').lower() or 'epicgames' in (url or '').lower():
        score -= 0.2
    confidence = max(0.2, min(0.98, score + 0.1))
    if score > 0.6:
        level = 'high'
        action = 'block'
        reason = 'Strong indicators of binary downloads + suspicious language.'
    elif score > 0.25:
        level = 'medium'
        action = 'warn'
        reason = 'Some suspicious indicators (short link or download keywords).'
    else:
        level = 'low'
        action = 'allow'
        reason = 'No obvious indicators.'
    return {'risk_level': level, 'reason': reason, 'confidence': round(confidence,2), 'recommended_action': action, 'iocs': []}



def decision_engine(url, message_text):
    vt = VT.url_report(url)
    gem = mock_gemini_classify(message_text, url=url)
    heur_score = vt['positives'] / (vt['total'] + 1)
    heur_score += 0.5 * (0 if gem['risk_level']=='low' else (1 if gem['risk_level']=='high' else 0.5))
    if heur_score > 0.6:
        action = 'BLOCK'
    elif heur_score > 0.25:
        action = 'WARN'
    else:
        action = 'ALLOW'
    return {'action': action, 'vt': vt, 'gemini': gem, 'score': round(heur_score,2)}



sample_messages = [
    {"author":"alex","content":"Free GTA papega injector download: http://mal.example.com/gta_installer.exe"},
    {"author":"sam","content":"Check this mod: https://bit.ly/123abc"},
    {"author":"rita","content":"GTA on Steam: https://store.steampowered.com/app/271590/Grand_Theft_Auto_V/"},
    {"author":"vishal","content":"Roblox cracked build: http://dodgy.example/download.zip"}
]

for m in sample_messages:
    urls = extract_urls(m['content'])
    for u in urls:
        res = decision_engine(u, m['content'])
        print(f"{m['author']:6} | {u:50} | ACTION={res['action']:5} | VT={res['vt']['verdict']:9} | GEM={res['gemini']['risk_level']:6} | SCORE={res['score']}")



def run_basic_tests():
    assert extract_urls("no url here") == []
    u = "http://mal.example.com/evil.exe"
    r = decision_engine(u, "download this free")
    assert r['action'] in ('BLOCK','WARN')
    print("Basic tests passed.")
run_basic_tests()


