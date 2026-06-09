import matplotlib.pyplot as plt
import pandas as pd
import requests


worldbank_api_url = 'https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
alpha3 = {'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT', 'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'}
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

def get_gdp_per_capita(country, year):
    url = worldbank_api_url.format(alpha3[country], year)
    response = requests.get(url).json()
    return response[1][0]['value']

gdp_list = [[get_gdp_per_capita(country, year) for year in years] for country in countries]
gdp_worldbank = pd.DataFrame(gdp_list, index=countries, columns=years)
gdp_worldbank.reset_index(names='Country Name').to_csv('gdp_worldbank.csv', index=False)


gdp_dataset = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')
gdp_dataset = gdp_dataset[gdp_dataset['Country Name'].isin(countries)].set_index('Country Name')[[str(y) for y in years]]


gdp_worldbank


gdp_dataset


plt.figure(figsize=(24, 6))
for c in countries:
    x = gdp_worldbank.loc[c, :] / gdp_dataset.loc[c, :]
    plt.plot(gdp_worldbank.loc[c] / gdp_dataset.loc[c].to_numpy(), label=c)

plt.legend()
plt.show()

