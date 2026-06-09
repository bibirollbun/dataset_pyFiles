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


## Get Tour Info from GetYourGuide

import requests

# Endpoint
url = "https://api.getyourguide.com/1/tours"

# Query parameters
params = {
    "cnt_language": "en",
    "currency": "GBP",
    "preformatted": "full",
    "q": "Paris",
    "coordinates[]": ["48.85693", "2.3412", "10"],
    "date[]": ["2025-12-01T00:00:00", "2025-12-02T23:59:59"],
    "cond_language[]": ["fr", "en"],
    "price[]": ["500"],
    "categories[]": ["55", "56"],
    "rating[]": ["2", "5"],
    "duration[]": ["1", "3"],
    "flags[]": ["private"],
    "sortfield": "popularity",
    "sortdirection": "DESC",
    "limit": "10",
    "offset": "0"
}

# Headers (replace API key)
headers = {
    "Accept": "application/json",
    "User-Agent": "PythonRequests/1.0",
    "Content-Type": "application/json",
    "X-Api-Key": ""
}

# Make request
#response = requests.get(url, headers=headers, params=params)

# Raise an error if the request failed
response.raise_for_status()

# Parse JSON
data = response.json()

# Print response JSON
print("Status Code:", response.status_code)
print("Response JSON:")
print(data)


## Another exapmle code for GetYourGuide with error logging

import aiohttp
import asyncio
import logging
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("getyourguide")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)


# ---------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------
class GetYourGuideAPIError(Exception):
    """General API call failure."""


class GetYourGuideAuthError(GetYourGuideAPIError):
    """Authentication or API key failure."""


class GetYourGuideNotFound(GetYourGuideAPIError):
    """404 results not found."""


# ---------------------------------------------------------------------
# Async API Client
# ---------------------------------------------------------------------
class GetYourGuideClient:
    BASE_URL = "https://api.getyourguide.com/1"

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "X-Api-Key": self.api_key,
                "Accept": "application/json",
                "User-Agent": "GYG-AsyncClient/1.0"
            }
        )
        return self   # FIXED

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    # Core request handler
    async def _request(self, method: str, path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"

        logger.debug(f"Request: {method} {url}")
        logger.debug(f"Params: {params}")

        try:
            async with self.session.request(method, url, params=params) as resp:
                status = resp.status

                logger.debug(f"Response status: {status}")

                if status == 401:
                    raise GetYourGuideAuthError("Invalid or missing API key.")
                elif status == 404:
                    raise GetYourGuideNotFound("Requested resource not found.")
                elif status >= 400:
                    text = await resp.text()
                    raise GetYourGuideAPIError(f"API Error {status}: {text}")

                return await resp.json()

        except asyncio.TimeoutError:
            logger.error("Request timed out")
            raise GetYourGuideAPIError("The request timed out.")
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise GetYourGuideAPIError(f"Network failure: {e}")

    # Public API: Search for tours
    async def search_tours(
        self,
        q: str,
        cnt_language: str = "en",
        currency: str = "GBP",
        coordinates: List[str] = None,
        date_range: List[str] = None,
        cond_language: List[str] = None,
        price: List[str] = None,
        categories: List[str] = None,
        rating: List[str] = None,
        duration: List[str] = None,
        flags: List[str] = None,
        sortfield: str = "popularity",
        sortdirection: str = "DESC",
        limit: int = 10,
        offset: int = 0,
    ):
        params = {
            "q": q,
            "cnt_language": cnt_language,
            "currency": currency,
            "sortfield": sortfield,
            "sortdirection": sortdirection,
            "limit": limit,
            "offset": offset,
            "preformatted": "full",
        }

        def add_list(key: str, values: Optional[List[str]]):
            if values:
                params[f"{key}[]"] = values

        add_list("coordinates", coordinates)
        add_list("date", date_range)
        add_list("cond_language", cond_language)
        add_list("price", price)
        add_list("categories", categories)
        add_list("rating", rating)
        add_list("duration", duration)
        add_list("flags", flags)

        return await self._request("GET", "/tours", params=params)


# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------
async def main():
    API_KEY = "API_KEY"

    async with GetYourGuideClient(API_KEY) as client:
        try:
            data = await client.search_tours(
                q="Paris",
                coordinates=["48.85693", "2.3412", "10"],
                date_range=["2025-12-01T00:00:00", "2025-12-02T23:59:59"],
                cond_language=["fr", "en"],
                price=["500"],
                categories=["55", "56"],
                rating=["2", "5"],
                duration=["1", "3"],
                flags=["private"]
            )

            print(data)

        except GetYourGuideAPIError as e:
            logger.error(f"API error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())


