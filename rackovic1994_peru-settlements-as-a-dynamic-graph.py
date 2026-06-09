import requests
from bs4 import BeautifulSoup
import time
import csv
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import great_circle
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image

# Base URL for Wikipedia
BASE_URL = "https://en.wikipedia.org"


def get_site_links(category_url:str) -> list:
    """
    Extracts links from the provided wikipedia category.

    -------
    :param category_url: url pointing to the wikipedia list of relevant sites

    -------
    :returns: List of urls corresponding to the liked wikipedia pages within the category
    """
    response = requests.get(category_url)
    soup = BeautifulSoup(response.text, "html.parser")
    site_links = []

    content_div = soup.find("div", {"class": "mw-parser-output"})
    if content_div:
        for ul in content_div.find_all("ul"):
            for li in ul.find_all("li"):
                a_tag = li.find("a")
                if a_tag and a_tag.has_attr("href") and not a_tag['href'].startswith("#"):
                    href = a_tag['href']
                    if href.startswith('http'):
                        full_url = href
                    else:
                        full_url = BASE_URL + a_tag['href']
                    if 'https://en.wikipedia.org/wiki/' in full_url:
                        site_links.append(full_url)
    return site_links


# Category page for Archaeological sites in Peru
CATEGORY_URL = f"{BASE_URL}/wiki/List_of_archaeological_sites_in_Peru"
site_links = get_site_links(CATEGORY_URL)
print(f'Found {len(site_links)} potential links.')
for i in range(10):
    print(site_links[np.random.randint(len(site_links))])


def extract_site_data(url:str) -> dict:
    '''
    Extracts the data about the archeological site. 
    If it can be located precieslly, with geo coordinates.
    Extracts location, and foundataion and abandonment dates. If on of hte later is not available, also extracts intro text and culture info, that can be used to infer the dates. 
        
    -------
    :param url: url pointing to the wikipedia page

    -------
    :returns: Dictionary containing the data about the settlement, extracted from the given url. 
    '''

    PERIOD_INDICATORS = ['years ago', 'years old']
    period_indicators_0 = ['B.P.','A.D.','C.E.','B.C.', 'B.C.E.']
    for element in period_indicators_0:
        PERIOD_INDICATORS.append(element)
        PERIOD_INDICATORS.append(element.replace(".",". ").strip())
        PERIOD_INDICATORS.append(''.join(element.split('.')))

    # Synonyms for 'founded' and 'abandoned'
    FOUNDED_SYNONYMS = ['founded', 'built', 'establish', 'set up', 'create', 'construct']
    ABANDON_SYNONYMS = ['abandon', 'destroy', 'deserted']

    def extract_lat_lon(soup:object) -> tuple[float,float]:
        """
        Extract geo coordinates of the scraped wiki link, if they exist

        -------
        :param soup: BeautifulSoup object, html representation of the wikipedia page

        -------
        :returns: latitude and longitude in decimal format, if available in the wiki page header.
        """

        def convert_coordinate(coordinate:str)->str:
            """
            Takes in a coordiante, such as a latitude or longitude, assuming it has degrees and optionally minutes, seconds '15°49′S', and makes sure it is writtten properly
            
            -------
            :param coordinate: Geo-coordinate, latitude or longitude, in a form of degrees/minutes/seconds, like '13°37′05″S', with possibly missing one of the three

            -------
            :returns: corrected geo-coordinate, i.e., the same value but made sure to obey the strict rule of two digits in each of teh three units
            """
            latitude = ''
            if '°' in coordinate:
                degrees = coordinate.split('°')
                deg = re.sub('[^0-9]',' ', degrees[0]).strip()[:2]
                latitude += deg + '°'
                rest = degrees[1]
                if '′' in rest:
                    minutes = rest.split('′')
                    mins = minutes[0].split('.')[0].strip()
                    if len(mins) == 1:
                        mins = '0'+mins
                    latitude += mins + '′'
                    rest = minutes[1]
                    if '″' in rest:
                        seconds = rest.split('″')
                        sec = seconds[0].split('.')[0].strip()
                        if len(sec) == 1:
                            sec = '0'+sec
                        latitude += sec + '″'
                    else:
                        latitude += '00″'
                else:
                    latitude += '00′00″'
            latitude += coordinate[-1]
            return latitude

        def dms_to_decimal(coordinate:str) -> float:
            """
            Convert a DMS (degrees, minutes, seconds) coordinate string to decimal degrees.
            Example input: '13°37′05″S'

            -------
            :param coordinate: Geo-coordinate, latitude or longitude, in a form of degrees/minutes/seconds

            -------
            :returns: decimal form of the inserted coordinate
            """
            # Regex to extract degrees, minutes, seconds, and direction
            pattern = r"(\d+)°(\d+)′(\d+)″([NSEW])"
            match = re.match(pattern, coordinate.strip())
            if not match:
                raise ValueError(f"Invalid coordinate format: {coordinate}")
            degrees, minutes, seconds, direction = match.groups()
            decimal = int(degrees) + int(minutes) / 60 + int(seconds) / 3600
            if direction in ['S', 'W']:
                decimal *= -1
            return decimal

        # Might be this format, with separate classes for lat and lon
        lat = soup.find('span', {'class':'latitude'})
        lon = soup.find('span', {'class':'longitude'})
        if lat and lon:
            lat = lat.text.strip()
            lat = convert_coordinate(lat)
            lon = lon.text.strip()
            lon = convert_coordinate(lon)
            return dms_to_decimal(lat), dms_to_decimal(lon)

        # Or this format, for geo decimal coordinates
        geo_dec = soup.find('span', class_='geo-dec')
        if geo_dec:
            coords = geo_dec.get_text(strip=True).split(',')
            if len(coords) == 2:
                lat, lon = coords[0].strip(), coords[1].strip()
                return lat, lon
        # if both fail, return None
        return None

    def extract_age(soup:object)->tuple[bool,bool,bool,int,int,str]:
        """
        Extract the age / fundation period of the scraped wiki link, if available

        -------
        :param soup: BeautifulSoup object, html representation of the wikipedia page

        -------
        :returns: age_founded_flag, age_abandoned_flag, culture_flag - booleans indicating if the info about foundation date, abandonment date, or culture, repsectivelly, is extracted. age_founded - settlement foundation date, age_abandoned - settlement abandonmebt date, culture - culture that inhabited the settlement
        """

        age_founded = None
        age_founded_flag = False
        age_abandoned = None
        age_abandoned_flag = False
        culture = None
        culture_flag = False

        # First case, if it is in the info box
        infobox = soup.find('table', class_='infobox vcard')
        if infobox:
            rows = infobox.find_all('tr')
            for row in rows:
                header = row.find('th', class_="infobox-label")
                # If there is a header sayin 'Founded', extract the date from there
                if header and 'Founded' in header.get_text(strip=True):
                    boxdata = row.find('td', class_='infobox-data')
                    if boxdata:
                        age_founded = boxdata.get_text(strip=True)
                        age_founded = re.sub("[\[].*?[\]]", "", age_founded)
                        if any(char.isdigit() for char in age_founded):
                            age_founded_flag = True

                if header and 'Abandoned' in header.get_text(strip=True):
                    boxdata = row.find('td', class_='infobox-data')
                    if boxdata:
                        age_abandoned = boxdata.get_text(strip=True)
                        age_abandoned = re.sub("[\[].*?[\]]", "", age_abandoned)
                        if any(char.isdigit() for char in age_abandoned):
                            age_abandoned_flag = True

                # Check if there is a cluture it belongs to
                if header and 'Culture' in header.get_text(strip=True):
                    boxdata = row.find('td', class_='infobox-data')
                    if boxdata:
                        culture = boxdata.get_text(strip=True)
                        culture = re.sub("[\[].*?[\]]", "", culture).lower()
                        culture_flag = True

        return age_founded_flag, age_abandoned_flag, culture_flag, age_founded, age_abandoned, culture


    def transforimg_textual_years(textual_years:list)->list:
        """
        Takes a list of strings containing period indicators (years), and transforms them into numerical values.
        Transformed years are integers, where years BC are negative, and AD are positive intigers.

        -------
        :param textual_years: List of strings of the form "1200 BC" or similar

        -------
        :returns: list of integers corresponding to the string values from 'textual_years'. E.g.,: transforimg_textual_years(['1200 BC']) -> [-1200] 
        """
        remove_char = ['.',',']
        transformed_ages = []

        for ty in textual_years:
            indicators_flag = False
            for i, pe in enumerate(PERIOD_INDICATORS):
                if pe in ty:
                    indicators_flag = True
                    for rc in remove_char:
                        ty = ty.replace(rc, "")
                    ty = re.sub('[^0-9]',' ', ty).strip().split(' ')
                    ty = [y for y in ty if len(y)>0]
                    if i < 5:
                        transformed_ages += [2000-int(y) for y in ty]
                    elif i < 11:
                        transformed_ages += [-int(y) for y in ty]
                    else:
                        transformed_ages += [int(y) for y in ty]
                    break
            if not indicators_flag:
                for rc in remove_char:
                    ty = ty.replace(rc, "")
                ty = re.sub('[^0-9]',' ', ty).strip().split(' ')
                ty = [y for y in ty if len(y)>0]
                transformed_ages += [int(y) for y in ty]

        return transformed_ages

    def extract_intro(soup:object) -> tuple[str, list, bool, bool, str, str]:
        """
        Extract the inttro section from the wikipedia page.
        Removes the citations, and isolates parts of text with dates, that might indicate the age of founding or deserting the settlement.

        -------
        :param soup: BeautifulSoup object, html representation of the wikipedia page

        -------
        :returns: lead_text - intro text of the wikipedia page. encountered_ages - list of ages with their era indicators, from the intro text. age_founded_flag/age_abandoned_flag - flag indicating if the foundation/abandonment date is found in the text. age_founded/age_abandoned - textual year of the foundation/abandonment, if found.
    
        """
        # Text content of the leading paragraphs (intro of the wiki page)
        content_body = soup.find('div', {'id': 'mw-content-text'})
        content_div = content_body.find('div', {'class': 'mw-parser-output'})

        lead_text = []
        for p in content_div.find_all('p', recursive=True):
            text = p.get_text(strip=False)
            if text:
                lead_text.append(text)
            # Stop collecting once we reach a <p> after which comes a heading or non-intro content
            next_sibling = p.find_next_sibling()
            if next_sibling and next_sibling.name == 'div':
                break
        lead_text = ''.join(lead_text)
        lead_text = re.sub("[\[].*?[\]]", "", lead_text)

        age_founded = None
        age_founded_flag = False
        age_abandoned = None
        age_abandoned_flag = False

        # Observe parts of text that contain age info
        encountered_ages = []
        for i, pe in enumerate(PERIOD_INDICATORS):
            idx = lead_text.find(pe)
            if idx > 0:
                idx_25 = max(0,idx-25)
                text_part = lead_text[idx_25:idx].lower() # check a short text part prior to indicator
                text_year = text_part.split()[-1] # take only the very last word
                if any(char.isdigit() for char in text_year):
                    encountered_ages.append(text_year+' '+pe)
                    for fs in FOUNDED_SYNONYMS:
                        if fs in text_part:
                            age_founded = text_year+' '+pe
                            age_founded_flag = True
                    for abs in ABANDON_SYNONYMS:
                        if abs in text_part:
                            age_abandoned = text_year+' '+pe
                            age_abandoned_flag = True
                else: # If no digits prior to age indicators, check the word after
                    idx_25 = min(len(lead_text),idx+len(pe)+25)
                    text_part_posterior = lead_text[idx+len(pe):idx_25].lower()
                    text_year = text_part_posterior.split()[0]
                    if any(char.isdigit() for char in text_year):
                        encountered_ages.append(text_year+' '+pe)
                        for fs in FOUNDED_SYNONYMS:
                            if fs in text_part:
                                age_founded = text_year+' '+pe
                                age_founded_flag = True
                        for abs in ABANDON_SYNONYMS:
                            if abs in text_part:
                                age_abandoned = text_year+' '+pe
                                age_abandoned_flag = True


        return lead_text, encountered_ages, age_founded_flag, age_abandoned_flag, age_founded, age_abandoned



    data = {}
    time.sleep(1)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        title = soup.find('span', {'class':'mw-page-title-main'}).text
    except:
        return None
    # Name of the site as a wiki page title - it hte page doesn't ahve a title, we don't want it
    data['NAME'] = title
    lat_lon = extract_lat_lon(soup)
    if lat_lon:
        data['LATITUDE'] = lat_lon[0]
        data['LONGITUDE'] = lat_lon[1]
    else: # If we don't have the loaction, we don't want this data at all
        return None

    print("Scraping: ", url)

    age_founded_flag, age_abandoned_flag, culture_flag, age_founded, age_abandoned, culture = extract_age(soup)
    if age_founded_flag:
        data['FOUNDED'] = age_founded
    if age_abandoned_flag:
        data['ABANDONED'] = age_abandoned
    if culture_flag:
        data['CULTURE'] = culture

    # If we have both the construction and abandonment dates, that's it.
    # Otherwise, we want to extract the intro text and try to fidn the age indicators there.
    if not (age_founded_flag and age_abandoned_flag):
        lead_text, encountered_ages, age_founded_flag, age_abandoned_flag, age_founded, age_abandoned = extract_intro(soup)
        if age_founded_flag and 'FOUNDED' not in data.keys():
            data['FOUNDED'] = age_founded
        if age_abandoned_flag and 'ABANDONED' not in data.keys():
            data['ABANDONED'] = age_abandoned

    # If any of the extracted ages from the intro file was next to the words like 'Foudned' or 'abandoned', we take that as a definite data.
    # Otherwise, we take a list of all extracted years, and the earliest one is assumed to be the foundation date, while the most recent one (if different) is the abandonmenet year.
    if 'FOUNDED' in data.keys():
        numeric_years = transforimg_textual_years([data['FOUNDED']])
        data['FOUNDED NUMERIC'] = min(numeric_years)
    elif len(encountered_ages):
        numeric_years = transforimg_textual_years(encountered_ages)
        data['FOUNDED NUMERIC'] = min(numeric_years)
    if 'ABANDONED' in data.keys():
        numeric_years = transforimg_textual_years([data['ABANDONED']])
        data['ABANDONED NUMERIC'] = max(numeric_years)
    elif len(encountered_ages):
        numeric_years = transforimg_textual_years(encountered_ages)
        if len(numeric_years)>1:
            data['ABANDONED NUMERIC'] = max(numeric_years)

    # It's (almost) safe to assum that if the given year is greater than 2000, it was supposed to be 2000BC, not AD
    if 'FOUNDED NUMERIC' in data.keys(): 
        if data['FOUNDED NUMERIC'] >= 2000:
            data['FOUNDED NUMERIC'] *= -1
    if 'ABANDONED NUMERIC' in data.keys():
        if data['ABANDONED NUMERIC'] >= 2000:
            data['ABANDONED NUMERIC'] *= -1
            
    # If we still don't have these, we keep the text
    if 'FOUNDED NUMERIC' not in data.keys() or 'ABANDONED NUMERIC' not in data.keys():
        data['TEXT'] = lead_text

    return data




all_data = []
cultures = {}
for url in site_links:
    data = extract_site_data(url)
    if data:
        all_data.append(data)
        if 'CULTURE' in data.keys():
            culture = data['CULTURE']
            if culture not in cultures.keys():
                cultures[culture] = 1
            else:
                cultures[culture] += 1


print(f"We extracted the total of {len(all_data)} sites.")
print("The cultures encountered within info-boxes: ")
print(cultures)


# https://en.wikipedia.org/wiki/Pre-Columbian_Peru
CulturesPeru = {
    "paiján" :      [-11000, -8000],
    "lauricocha" :  [-10000, -2500],
    "casma–sechin": [-3500, 200],
    "norte chico" : [-3500, -1800],
    "caral-supe" :  [-3500, -1800],
    "cupisnique" :  [-1500, -500],
    "chavín" :      [-900, -250],
    "paracas" :     [-800, -100],
    "nazca" :       [-100, 800],
    "moche" :       [100, 800],
    "wari" :        [500, 1000],
    "tiwanaku" :    [600, 1000],
    "chachapoya" :  [800, 1470],
    "chimú" :       [900, 1470],
    "chimor" :      [900, 1470],
    "ichma" :       [1100, 1469],
    "chanka" :      [1200, 1438],
    "inca" :        [1200, 1572]
}


def plot_cultures_timelines(CulturesDict:dict, title:str=None):
    """
    Plots temporal spread of each per-Colombian culture.
    
    -------
    :param CulturesDict: Dictionary with culture names as keys and touple of their starting and ending year, as a value.
    :param title: Figure title
    """
    left_limit, right_limit = 0, 0
    fig, ax = plt.subplots(figsize=(15,4))
    for i, key in enumerate(CulturesDict.keys()):
        begin, end = CulturesDict[key][0], CulturesDict[key][1]
        if begin < left_limit:
            left_limit = 0.+begin
        if end > right_limit:
            right_limit = 0.+end
        duration = end-begin
        plt.barh(i, width = duration, left = begin, height = 0.9)
        plt.text(begin + duration + 50, i, key.capitalize(), va='center')  # adjust `+10` as needed for spacing
    ax.spines[['right', 'top', 'left']].set_visible(False)
    plt.yticks([])
    plt.xlim(left_limit-100, right_limit+300)
    plt.xlabel('Years')
    if title:
        plt.title(title)
    plt.show()


plot_cultures_timelines(CulturesPeru, title='Pre-Columbian Cultures in Peru')


encountered_cultures = {}
for key in CulturesPeru.keys():
    encountered_cultures[key] = 0
for key in cultures.keys():
    for key1 in encountered_cultures.keys():
        if key1 in key:
            encountered_cultures[key1] += cultures[key]
            break
            
fig, ax = plt.subplots(figsize=(10,3))
i = 0
for key in encountered_cultures:
    if encountered_cultures[key] > 0:
        i += 1
        plt.barh(i, width = encountered_cultures[key])
        plt.text(encountered_cultures[key]+0.2, i, key.capitalize(), va='center')  # adjust `+10` as needed for spacing
ax.spines[['right', 'top', 'left']].set_visible(False)
plt.yticks([])
plt.xlabel('Count')
plt.title("Occurances per culture")
plt.show()


def culture_based_foundation(data:dict, CulturesDict:dict) ->tuple[dict, bool]:
    """
    Based on the culture the settlement beloged to, assigns correpsonding foundation and abandonment dates.
    
    -------
    :param data: Dictionary containing main data about the settlment
    :param CulturesDict: Dictionary with culture names as keys and touple of their starting and ending year, as a value.

    -------
    :returns: updated data dictionary, and a flag indicating if any edits were made
    """
    if 'CULTURE' in data.keys():
        for culture in CulturesDict.keys():
            if culture in data['CULTURE']:
                if 'FOUNDED NUMERIC' not in data.keys():
                    data['FOUNDED NUMERIC'] = CulturesDict[culture][0]
                if 'ABANDONED NUMERIC' not in data.keys():
                    data['ABANDONED NUMERIC'] = CulturesDict[culture][1]
                if 'TEXT' in data.keys():
                    data.pop('TEXT')
            return data, True
    return data, False


complete_years, edited_years, missing_years = 0, 0, 0
for i, data in enumerate(all_data):
    if 'FOUNDED NUMERIC' in data and 'ABANDONED NUMERIC' in data:
        complete_years += 1
    else:
        data, flag = culture_based_foundation(data, CulturesPeru)
        if flag:
            complete_years += 1
            edited_years += 1
        else:
            missing_years += 1
    all_data[i] = data
print(f"Number of sites with complete start and end dates: {complete_years}")
print(f"Number of sites with start and end dates edited based on their culture: {edited_years}")
print(f"Number of sites with missing start or end dates: {missing_years}")


# Set up a basic map
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)

# Plot points
all_foundation_dates = []
for data in all_data:
    lat, lon = data['LATITUDE'], data['LONGITUDE']
    ax.plot(lon, lat, marker='x', color='brown', transform=ccrs.Geodetic())
    if 'FOUNDED NUMERIC' in data.keys():
        all_foundation_dates.append(data['FOUNDED NUMERIC'])

plt.title('Archeological Sites in Peru')
plt.show()


fig, ax = plt.subplots(figsize=(8,4))
ax.spines[['right', 'top']].set_visible(False)
plt.hist(all_foundation_dates,bins=50)
plt.title("Distribution of Foundation Dates")
plt.ylabel('Count')
plt.xlabel('Years')
plt.show()


def build_adjecency_matrix(all_data:list, all_foundation_dates:list, timestamps:list, distance_threshold:int, plot_flag:bool=True):
    """
    Computes adjecency matrix based on world-distances between sites. 
    Further, creates a matrix of age based occurences of sites, so taht at any given timestamp we can which were the cities that were active.

    
    -------
    :param all_data: List of dictionaries containing main data about the settlment
    :param all_foundation_dates: List of foundation years (numeric) for the correpsonding settlemtns
    :param timestamps: List of timestamps/years for which we check eistance for the correpsonding settlemtns
    :param distance_threshold: Threshold value for the straight-line distance between two setlmenets (in kilometers) to consider them connected 
    :param plot_flag: If set to True, plots the extracted things

    -------
    :returns: ajdacency matrix for hte settlements graph, matrix of temporal occurances for each settlemnt and their geo locations (latitude and longitude)    
    """
    n= len(all_data)
    adjecency_matrix = np.zeros((n,n))
    locations_matrix = np.zeros((n,2))
    times_matrix = np.zeros((n,2)).astype(int)
    temporal_occurance_matrix = np.zeros((n, len(timestamps)))
    for i, data in enumerate(all_data):
        locations_matrix[i] += (data['LATITUDE'], data['LONGITUDE'])
        if 'FOUNDED NUMERIC' in data.keys():
            times_matrix[i,0] = data['FOUNDED NUMERIC']
        else:
            times_matrix[i,0] = min(all_foundation_dates)
        if 'ABANDONED NUMERIC' in data.keys():
            times_matrix[i,1] = data['ABANDONED NUMERIC']
        else:
            times_matrix[i,1] = 1500
        if times_matrix[i,1] < times_matrix[i,0]:
            times_matrix[i,0] = times_matrix[i,1]
            times_matrix[i,1] = 1500
    for i in range(n):
        for j in range(i,n):
            adjecency_matrix[i, j] = great_circle(locations_matrix[i], locations_matrix[j]).km
            adjecency_matrix[j, i] += adjecency_matrix[i,j]
        temporal_occurance = (times_matrix[i,0] <= np.array(timestamps))*(times_matrix[i,1] >= np.array(timestamps))*1
        temporal_occurance_matrix[i] += temporal_occurance

    if plot_flag:
        distance_matrix = adjecency_matrix.copy()
        distance_matrix[distance_matrix==0] = adjecency_matrix.max()
        distance_vector = distance_matrix.min(1)
        fig, ax = plt.subplots(figsize=(12,4))
        ax.spines[['right', 'top']].set_visible(False)
        plt.hist(distance_vector,bins=230,color='g')
        plt.title("Histogram of pair-wise site distances")
        plt.ylabel("count")
        plt.xlabel("Distances in km")
        plt.scatter(distance_threshold,0,color='yellow',marker='d',s=100)
        plt.show()
    
    adjecency_matrix[adjecency_matrix <= distance_threshold]=1
    adjecency_matrix[adjecency_matrix > distance_threshold]=0
    adjecency_matrix -= np.eye(n)
    if plot_flag:
        fig, ax = plt.subplots(figsize=(7,7))
        order = np.argsort(np.mean(adjecency_matrix,1))
        adjecency_matrix_sorted = adjecency_matrix[order]
        adjecency_matrix_sorted = adjecency_matrix_sorted[:,order]
        plt.imshow(adjecency_matrix_sorted, cmap='binary')
        ax.spines[['right', 'top','left','bottom']].set_visible(False)
        plt.title("Adjecency matrix")
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
        fig, ax = plt.subplots(figsize=(7,7))
        plt.imshow(temporal_occurance_matrix[order], cmap='binary')
        plt.title("Temporal occurances matrix")
        ax.spines[['right', 'top','left','bottom']].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Time')
        plt.ylabel('Sites')
        plt.show()

    return adjecency_matrix, temporal_occurance_matrix.astype(int), locations_matrix



distance_threshold = 100
timestamps = np.linspace(min(all_foundation_dates), -3500, 10).tolist() + np.linspace(-3250, -1500, 10).tolist()+ np.linspace(-1400, 1500, 30).tolist()
timestamps = [int(t) for t in timestamps]
print(f"Time stamps: {timestamps}\n")
adjecency_matrix, temporal_occurance_matrix, locations_matrix = build_adjecency_matrix(all_data, all_foundation_dates, timestamps, distance_threshold)
np.save('Peru_adjecency_matrix.npy',adjecency_matrix)
np.save('Peru_temporal_occurance_matrix.npy',temporal_occurance_matrix)
np.save('Peru_locations_matrix.npy',locations_matrix)



graph =nx.from_numpy_array(adjecency_matrix)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.OCEAN)
pos = {i: (locations_matrix[i][1], locations_matrix[i][0]) for i in range(len(locations_matrix))}  # (lon, lat)

def update(t):
    ax.clear()
    # Redraw static map features each frame
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    
    t_indices = np.where(temporal_occurance_matrix[:, t])[0]
    t_graph = graph.subgraph(t_indices).copy()

    # Plot edges manually
    for u, v in t_graph.edges():
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        ax.plot(x_coords, y_coords, transform=ccrs.Geodetic(), color='k', linewidth=0.1)
    
    # Plot nodes
    for node in t_graph.nodes():
        lon, lat = pos[node]
        ax.plot(lon, lat, marker='o', color='brown', transform=ccrs.Geodetic(), markersize=4)

    ax.set_title(f"Settlements Progression in Peru\nYear: {int(timestamps[t])}", fontsize=12)

# Animate and save
anim = FuncAnimation(fig, update, frames=10, interval=600)
anim.save("temporal_graph_map.gif", writer=PillowWriter(fps=2))

# Display in notebook
Image(filename="temporal_graph_map.gif")


