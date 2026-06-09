import pandas as pd
import numpy as np
import requests
import re
import asyncio
import aiohttp

from tqdm import tqdm

tqdm.pandas()


def get_type_by_request(prefix):
    url = f"https://hdl.handle.net/api/handles/{prefix}"
    response = requests.get(url)

    try:
        data = response.json()

        for val in data['values']:
            if val['type'] == 'HS_SERV':
                return val['data']['value']
        return np.nan

    except:
        return np.nan


async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.json()


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results


prefixes = [f'10.{i}' for i in range(0, 100000)]
all_prefixes = pd.DataFrame({'prefix': prefixes})


urls = [f"https://hdl.handle.net/api/handles/10.{i}" for i in range(0, 100000)]
data = await main()


result = []
for json in tqdm(data):
    try:
        type_cit = ''
        for val in json['values']:
            if val['type'] == 'HS_SERV':
                type_cit += val['data']['value']
                
        if len(type_cit) == 0:
            type_cit = np.nan
        result.append(type_cit)
    except:
        result.append(np.nan)


descr = []
for json in tqdm(data):
    try:
        desc = ''
        for val in json['values']:
            if val['type'] == 'DESC':
                desc += val['data']['value']

        if len(desc) == 0:
            desc = np.nan
        descr.append(desc)
    except:
        descr.append(np.nan)


all_prefixes['type'] = result
all_prefixes['description'] = descr
all_prefixes.dropna(subset = ['type'], inplace = True)


unknown = all_prefixes[all_prefixes['description'].progress_apply(lambda info: re.search(re.compile(r'data\s*cite', re.IGNORECASE), info) is not None if type(info) is str else False)]


unknown = unknown[unknown['type'] != '10.SERV/DATACITE']
all_prefixes.loc[unknown.index, ['type']] = '10.SERV/MIXED'
all_prefixes.drop(columns = ['description']).to_csv('prefixes.csv', index = False)

