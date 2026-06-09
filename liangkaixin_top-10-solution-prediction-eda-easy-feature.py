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


from copy import deepcopy
# from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



train_csv = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test_csv = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
train_csv.info()


# maybe useful
REGION_CODES = {
    "Republic of Adygea": ["01"],
    "Altai Republic": ["04"],
    "Republic of Bashkortostan": ["02", "102", "702"],
    "Republic of Buryatia": ["03"],
    "Republic of Dagestan": ["05"],
    "Donetsk People's Republic": ["80", "180"],
    "Republic of Ingushetia": ["06"],
    "Kabardino-Balkarian Republic": ["07"],
    "Republic of Kalmykia": ["08"],
    "Karachay-Cherkess Republic": ["09"],
    "Republic of Karelia": ["10"],
    "Komi Republic": ["11"],
    "Republic of Crimea": ["82"],
    "Luhansk People's Republic": ["81", "181"],
    "Republic of Mari El": ["12"],
    "Republic of Mordovia": ["13", "113"],
    "Sakha Republic": ["14"],
    "Republic of North Ossetia": ["15"],
    "Republic of Tatarstan": ["16", "116", "716"],
    "Republic of Tyva (Tuva)": ["17"],
    "Udmurt Republic": ["18"],
    "Republic of Khakassia": ["19"],
    "Chechen Republic": ["20", "95"],
    "Chuvash Republic": ["21", "121"],
    "Altai Krai": ["22", "122"],
    "Zabaykalsky Krai": ["75"],
    "Kamchatka Krai": ["41"],
    "Krasnodar Krai": ["23", "93", "123", "193", "323"],
    "Krasnoyarsk Krai": ["24", "84", "88", "124"],
    "Perm Krai": ["59", "81", "159"],
    "Primorsky Krai": ["25", "125"],
    "Stavropol Krai": ["26", "126"],
    "Khabarovsk Krai": ["27"],
    "Amur Oblast": ["28"],
    "Arkhangelsk Oblast": ["29"],
    "Astrakhan Oblast": ["30", "130"],
    "Belgorod Oblast": ["31"],
    "Bryansk Oblast": ["32"],
    "Vladimir Oblast": ["33"],
    "Volgograd Oblast": ["34", "134"],
    "Vologda Oblast": ["35"],
    "Voronezh Oblast": ["36", "136"],
    "Zaporizhzhia Oblast": ["85", "185"],
    "Ivanovo Oblast": ["37"],
    "Irkutsk Oblast": ["38", "85", "138"],
    "Kaliningrad Oblast": ["39", "91"],
    "Kaluga Oblast": ["40"],
    "Kemerovo Oblast": ["42", "142"],
    "Kirov Oblast": ["43"],
    "Kostroma Oblast": ["44"],
    "Kurgan Oblast": ["45"],
    "Kursk Oblast": ["46"],
    "Leningrad Oblast": ["47", "147"],
    "Lipetsk Oblast": ["48"],
    "Magadan Oblast": ["49"],
    "Moscow Oblast": ["50", "90", "150", "190", "250", "550", "750", "790"],
    "Murmansk Oblast": ["51"],
    "Nizhny Novgorod Oblast": ["52", "152", "252"],
    "Novgorod Oblast": ["53"],
    "Novosibirsk Oblast": ["54", "154", "754"],
    "Omsk Oblast": ["55", "155"],
    "Orenburg Oblast": ["56", "156"],
    "Oryol Oblast": ["57"],
    "Penza Oblast": ["58", "158"],
    "Pskov Oblast": ["60"],
    "Rostov Oblast": ["61", "161", "761"],
    "Ryazan Oblast": ["62"],
    "Samara Oblast": ["63", "163", "763"],
    "Saratov Oblast": ["64", "164"],
    "Sakhalin Oblast": ["65"],
    "Sverdlovsk Oblast": ["66", "96", "196"],
    "Smolensk Oblast": ["67"],
    "Tambov Oblast": ["68"],
    "Tver Oblast": ["69"],
    "Tomsk Oblast": ["70"],
    "Tula Oblast": ["71"],
    "Tyumen Oblast": ["72", "172"],
    "Ulyanovsk Oblast": ["73", "173"],
    "Kherson Oblast": ["84", "184"],
    "Chelyabinsk Oblast": ["74", "174", "774"],
    "Yaroslavl Oblast": ["76"],
    "Moscow": ["77", "97", "99", "177", "197", "199", "777", "797", "799", "977"],
    "Saint Petersburg": ["78", "98", "178", "198"],
    "Sevastopol": ["92"],
    "Jewish Autonomous Oblast": ["79"],
    "Nenets Autonomous Okrug": ["83"],
    "Khanty-Mansi Autonomous Okrug": ["86", "186"],
    "Chukotka Autonomous Okrug": ["87"],
    "Yamalo-Nenets Autonomous Okrug": ["89"],
    "Baikonur": ["94"],
    "Occupational Administration of Kharkiv Oblast": ["188"],
}

# ((letters, numbers range (from, to), region code), is it forbidden to buy (bool), do they have an advantage on the road (bool), level of significance (author's opinion))
GOVERNMENT_CODES = {
    # Moscow
    ("AMP", (0, 999), "97"): ("Government of Russia", 1, 1, 10),
    ("AMP", (0, 999), "77"): ("Partially Government of Russia", 0, 1, 8),
    ("EKX", (0, 999), "77"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("EKX", (0, 999), "97"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("EKX", (0, 999), "99"): ("Partially Federal Protective Service (Federal Protective Service)", 0, 1, 6),
    ("KKX", (0, 999), "77"): ("Partially used on vehicles of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("CAC", (500, 999), "77"): ("Former officially 'open' plates of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("CAC", (500, 999), "77"): ("Former officially 'open' plates of Ministry of Security/Federal Counterintelligence Service /Federal Security Service of Russia", 0, 0, 1),
    ("AOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("BOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("MOO", (0, 999), "77"): ("Partially Presidential Administrative Directorate plates", 0, 1, 6),
    ("COO", (0, 999), "77"): ("Partially Administrative Directorate, Federation Council plates", 0, 1, 6),
    ("AMM", (0, 999), "99"): ("Partially plates of Moscow City Duma deputies, police", 0, 1, 4),
    ("CCC", (0, 999), "77"): ("Partially Central Special Communication, Courier Service, Ministry of Communications", 0, 1, 3),
    ("CCC", (0, 999), "99"): ("Partially Tax Police, Customs, Special Communications", 0, 1, 3),
    ("CCC", (0, 999), "97"): ("Partially Central Special Communication, Courier Service, Ministry of Communications", 0, 1, 3),
    ("KKK", (0, 999), "99"): ("Initially belonged to Courier Service, now used among private individuals", 0, 0, 1),
    ("OOO", (0, 999), "77"): ("Initially intended for Federal Security Service", 0, 0, 1),
    ("KMM", (0, 999), "77"): ("Partially Fire Department plates", 0, 1, 3),
    ("MMP", (300, 320), "77"): ("Partially Federal Security Service plates", 0, 1, 4),
    ("MMP", (0, 299), "77"): ("Partially Government of Russia, Federal Security Service, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("MMP", (321, 999), "77"): ("Partially Government of Russia, Federal Security Service, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("PMP", (0, 999), "77"): ("Partially Ministry of Justice plates", 0, 1, 3),
    ("AMO", (0, 999), "77"): ("Partially Moscow City Hall plates", 0, 1, 5),
    ("KOO", (0, 999), "77"): ("Partially Constitutional Court plates", 0, 1, 3),
    ("EPE", (0, 999), "77"): ("Partially State Duma plates", 0, 1, 3),
    ("AAA", (0, 999), "77"): ("Partially Administration of the President plates", 0, 1, 6),
    ("KMP", (0, 999), "77"): ("Partially Government of Russia plates", 0, 1, 3),
    ("TMP", (0, 999), "77"): ("Partially Government of Russia plates, as well as private individuals with connections in the traffic police", 0, 1, 2),
    ("YMP", (0, 999), "77"): ("Partially Government of Russia plates, as well as private individuals with connections in the traffic police", 0, 1, 2),
    ("XXX", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),
    ("YYY", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),
    ("XKX", (0, 999), "77"): ("Partially Federal Security Service and Federal Protective Service plates", 0, 1, 2),
    ("OMP", (0, 999), "77"): ("Partially Government of Russia, banks, and private individuals with connections in the traffic police", 0, 1, 2),
    ("EEE", (0, 999), "77"): ("Private individuals with connections in the traffic police", 0, 1, 2),

    # Moscow Oblast
    ("AMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("BMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("KMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("CMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("OMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("MMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("TMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("HMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("YMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("XMO", (0, 999), "50"): ("Partially various government agencies (administration, ambulance, traffic police, etc.)", 0, 1, 3),
    ("AMM", (0, 999), "50"): ("Partially plates of the regional administration", 0, 1, 5),
    ("AMM", (0, 999), "90"): ("Partially plates of the regional administration", 0, 1, 5),
    ("MMM", (0, 999), "50"): ("Partially plates of law enforcement in the region (prosecutor's office, EMERCOM, traffic police, etc.)", 0, 1, 5),
    ("MMM", (0, 999), "90"): ("Partially plates of law enforcement in the region (prosecutor's office, EMERCOM, traffic police, etc.)", 0, 1, 5),

    # Saint Petersburg
    ("OBO", (0, 999), "78"): ("Partially Departmental Security Service plates", 0, 1, 4),
    ("OBO", (0, 999), "98"): ("Partially Departmental Security Service plates", 0, 1, 4),
    ("OTT", (0, 999), "78"): ("Partially former traffic police plates (now replaced by 98)", 0, 0, 1),
    ("OTT", (0, 999), "98"): ("Partially traffic police plates", 0, 1, 4),
    ("OMM", (0, 999), "78"): ("Partially city district police plates", 0, 1, 3),
    ("OMM", (0, 999), "98"): ("Partially city district police plates", 0, 1, 3),
    ("OOM", (0, 999), "78"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OOM", (0, 999), "98"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OKO", (0, 100), "78"): ("Partially former plates of the prosecutor's office and judicial department (now replaced by 98)", 0, 0, 1),
    ("OKO", (0, 100), "98"): ("Partially plates of the prosecutor's office and judicial department", 0, 1, 3),
    ("OKO", (700, 999), "78"): ("Partially former Federal Security Service plates (now replaced by 98)", 0, 0, 1),
    ("OKO", (700, 999), "98"): ("Partially Federal Security Service plates", 0, 1, 3),
    ("OPP", (0, 999), "78"): ("Partially former plates of the Main Department of Internal Affairs (now replaced by 98)", 0, 0, 1),
    ("OPP", (0, 999), "98"): ("Partially plates of the Main Department of Internal Affairs", 0, 1, 3),
    ("OOH", (0, 999), "78"): ("Partially Federal Drug Control Service and Federal Tax Service plates", 0, 1, 3),
    ("OOH", (0, 999), "98"): ("Partially Federal Drug Control Service and Federal Tax Service plates", 0, 1, 3),
    ("OAO", (0, 999), "78"): ("Partially plates of the city and regional administration", 0, 1, 5),
    ("OAO", (0, 999), "98"): ("Partially plates of the city and regional administration", 0, 1, 5),
    ("AAA", (0, 100), "78"): ("Partially plates of the city and regional administration", 0, 1, 6),
    ("AAA", (0, 100), "98"): ("Partially plates of the city and regional administration", 0, 1, 6),
    ("OOO", (0, 899), "78"): ("Commercial plates", 0, 0, 2),
    ("OOO", (0, 899), "98"): ("Commercial plates", 0, 0, 2),
    ("OOO", (900, 999), "78"): ("Partially Federal Protective Service plates", 0, 1, 3),
    ("OOO", (900, 999), "98"): ("Partially Federal Protective Service plates", 0, 1, 3),
    ("OKC", (0, 999), "98"): ("Partially Constitutional Court of the Russian Federation plates", 0, 1, 3),
    ("OOC", (0, 999), "78"): ("Partially plates of heads of enterprises and organizations", 0, 0, 2),
    ("OOC", (0, 999), "98"): ("Partially plates of heads of enterprises and organizations", 0, 0, 2),
    ("MMM", (0, 999), "78"): ("Commercial plates", 0, 0, 2),
    ("MMM", (0, 999), "98"): ("Commercial plates", 0, 0, 2),

    # Altai Republic
    ("XXX", (0, 999), "04"): ("Widespread 'special' plates", 0, 0, 2),
    ("TTT", (0, 999), "04"): ("Rare 'special' plates", 0, 0, 2),
    ("PPP", (0, 999), "04"): ("Partially prosecutor's office of the republic", 0, 1, 3),
    ("PPA", (0, 999), "04"): ("Partially prosecutor's office of the republic", 0, 1, 3),
    ("MPA", (0, 999), "04"): ("Partially Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("OOO", (0, 999), "04"): ("Partially plates of the government of the republic", 0, 1, 5),
    ("HHH", (0, 999), "04"): ("Partially the republic's tax service plates", 0, 1, 3),
    ("CCC", (0, 999), "04"): ("Partially plates belonging to the republic's judges", 0, 1, 3),

    # Republic of Bashkortostan
    ("PKC", (0, 999), "02"): ("Partially State Assembly (Kurultai) plates", 0, 1, 5),
    ("KKC", (0, 999), "02"): ("Partially State Assembly (Kurultai) plates", 0, 1, 5),
    ("OOO", (0, 999), "02"): ("Partially plates of leaders of large enterprises and ministries", 0, 1, 3),
    ("AAA", (0, 999), "02"): ("Partially plates of the republic's government", 0, 1, 5),

    # Republic of Karelia
    ("TTT", (0, 999), "10"): ("Partially government of the republic and Federal Security Service plates", 0, 1, 5),
    ("HHH", (0, 999), "10"): ("Partially plates of city and district administrations of the republic", 0, 1, 4),
    ("MMM", (0, 999), "10"): ("Partially plates of the Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("EMP", (0, 999), "10"): ("Partially plates of the Ministry of Internal Affairs of the republic", 0, 1, 3),
    ("CCC", (0, 999), "10"): ("Partially plates of the prosecutor's office and judges' vehicles", 0, 1, 3),

    # Komi Republic
    ("TTT", (0, 999), "11"): ("Partially government of the republic and Federal Security Service plates", 0, 1, 5),
    ("OOO", (0, 999), "11"): ("Widespread semi-special plates, leaders of large industrial companies", 0, 1, 3),

    # Sakha Republic
    ("PPP", (0, 999), "14"): ("Partially plates of the republic's prosecutor's office", 0, 1, 3),
    ("AAA", (0, 999), "14"): ("Motor pool of the President, Government, Parliament of the republic, as well as heads of state enterprises", 0, 1, 5),

    # Republic of Tatarstan
    ("OAA", (0, 999), "16"): ("Partially plates of heads of district administrations", 0, 1, 5),
    ("OAA", (0, 999), "116"): ("Partially plates of heads of district administrations", 0, 1, 5),
    ("OAA", (0, 999), "716"): ("Partially plates of heads of district administrations", 0, 1, 5),

    # Krasnodar Krai
    ("PPP", (0, 999), "23"): ("Partially plates of the Krai and city administrations", 0, 1, 5),
    ("HHH", (0, 999), "23"): ("Partially plates of the tax authorities", 0, 1, 3),
    ("OOO", (0, 999), "23"): ("Partially plates of the Krai and city administrations", 0, 1, 5),
    ("KKK", (0, 999), "23"): ("Partially plates of the Krai administration", 0, 1, 5),

    # Krasnoyarsk Krai
    ("KPK", (0, 999), "24"): ("Partially plates of the Krai administration", 0, 1, 5),
    ("OOO", (0, 999), "24"): ("Partially Federal Security Service plates of the Krai", 0, 1, 3),
    ("MKK", (0, 999), "24"): ("Partially former plates of the Ministry of Internal Affairs of the Krai", 0, 0, 1),

    # Primorsky Krai
    ("BOO", (0, 999), "25"): ("Partially military plates", 0, 1, 3),
    ("BOO", (0, 999), "125"): ("Partially city services plates in Vladivostok and districts", 0, 1, 2),
    ("AAA", (0, 999), "25"): ("Issued first in Vladivostok", 0, 0, 2),
    ("AAA", (0, 999), "125"): ("One of the most 'special' series, prosecutor's office", 1, 1, 5),
    ("HHH", (0, 999), "25"): ("Partially plates of the administration and vehicles of City Duma deputies", 0, 1, 3),
    ("MMM", (0, 999), "25"): ("Partially plates of the deputies of the Krai Legislative Assembly", 0, 1, 3),
    ("CCC", (0, 999), "25"): ("Partially plates of the Krai administration", 0, 1, 5),
    ("XXX", (0, 999), "25"): ("Partially plates of the prosecutor's office and the Department of Internal Affairs", 0, 1, 2),
    ("OOO", (0, 999), "25"): ("Partially former plates of the Krai administration (during Governor Evgeny Nazdratenko)", 0, 0, 1),
    ("TTT", (0, 999), "25"): ("Partially former plates of the Vladivostok administration and federal agencies in the Krai (during Mayor Yuri Kopylov)", 0, 0, 1),
    ("MBK", (0, 999), "25"): ("Partially plates for employees of the Department of Internal Affairs", 0, 1, 3),
    ("MBK", (0, 999), "125"): ("Partially plates for employees of the Department of Internal Affairs", 0, 1, 3),
    ("MOO", (0, 999), "25"): ("Partially plates for Krai agencies of the Department of Internal Affairs, EMERCOM, firefighters, etc.", 0, 1, 2),
    ("MOO", (0, 999), "125"): ("Partially plates for Krai agencies of the Department of Internal Affairs, EMERCOM, firefighters, etc.", 0, 1, 2),
    ("HOO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the southeastern region of the Krai (Nakhodka)", 0, 1, 3),
    ("HOO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the southeastern region of the Krai (Nakhodka)", 0, 1, 3),
    ("YOO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the central region of the Krai (Ussuriysk)", 0, 1, 3),
    ("YOO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the central region of the Krai (Ussuriysk)", 0, 1, 3),
    ("COO", (0, 999), "25"): ("Partially plates of the Department of Internal Affairs, traffic police in the northern region of the Krai (Spassk-Dalny)", 0, 1, 3),
    ("COO", (0, 999), "125"): ("Partially plates of the Department of Internal Affairs, traffic police in the northern region of the Krai (Spassk-Dalny)", 0, 1, 3),

    # Vologda Oblast
    ("AAA", (0, 999), "35"): ("Partially plates of the regional government and Vologda city administration", 0, 1, 5),

    # Volgograd Oblast
    ("AAM", (0, 999), "34"): ("Partially plates of the Oblast Duma", 0, 1, 3),
    ("PAA", (0, 999), "34"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAA", (0, 999), "34"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("ACK", (0, 999), "34"): ("Partially plates of the Investigative Committee, Main Department of Internal Affairs", 0, 1, 3),
    ("YYY", (0, 999), "34"): ("Partially Federal Security Service plates", 0, 1, 3),
    ("AAK", (0, 999), "34"): ("Partially plates of the Federal Bailiff Service, Ministry of Justice, and Judicial Department", 0, 1, 3),

    # Voronezh Oblast
    ("ААА", (0, 999), "36"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("BOA", (0, 999), "36"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MMM", (0, 999), "36"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Kaliningrad Oblast
    ("AAK", (0, 999), "39"): ("Partially plates of the Oblast Administration, Federal Security Service, and Prosecutor's Office", 0, 1, 5),
    ("KKK", (0, 999), "39"): ("Partially plates of the Oblast Administration, Federal Security Service, and Prosecutor's Office", 0, 1, 5),
    ("PPP", (0, 999), "39"): ("Partially former plates of the Oblast Administration (during Governor Boos)", 0, 0, 1),

    # Kaluga Oblast
    ("OOO", (0, 999), "40"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("TTT", (0, 999), "40"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "40"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Kurgan Oblast
    ("OOO", (0, 999), "45"): ("Partially former plates of the Oblast Administration", 0, 0, 1),
    ("TTT", (0, 999), "45"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("OKO", (0, 999), "45"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),

    # Novosibirsk Oblast
    ("AAA", (0, 199), "54"): ("Plates for Presidential Plenipotentiaries", 1, 1, 7),
    ("AAA", (200, 999), "54"): ("'Special' plates", 0, 1, 4),
    ("HHH", (0, 999), "54"): ("Partially plates of the Novosibirsk mayor's office, Oblast Administration, and Oblast Council", 0, 1, 5),
    ("ACK", (0, 999), "54"): ("Partially Federal Security Service plates of the Oblast", 0, 1, 3),
    ("AHO", (0, 999), "54"): ("Partially former plates of the Oblast Administration", 0, 0, 1),
    ("AAO", (0, 999), "54"): ("Partially plates of various government agencies, including district administrations of Novosibirsk", 0, 1, 3),
    ("PPP", (0, 999), "54"): ("'Morozov' plates (introduced by former head of traffic police Pyotr Morozov)", 0, 1, 2),
    ("MOP", (0, 999), "54"): ("'Morozov' plates (introduced by former head of traffic police Pyotr Morozov)", 0, 1, 2),

    # Oryol Oblast
    ("AAA", (0, 999), "57"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AOO", (0, 999), "57"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("OAO", (0, 999), "57"): ("Partially plates of directors of public joint-stock companies", 0, 1, 2),

    # Rostov Oblast
    ("APO", (0, 999), "61"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAA", (0, 999), "61"): ("Partially plates of district heads of Rostov-on-Don, mayors of Oblast cities", 0, 1, 5),
    ("APY", (0, 999), "61"): ("Partially plates of the Rostov-on-Don administration", 0, 1, 5),
    ("KKK", (0, 999), "61"): ("Partially former plates for Presidential Plenipotentiaries (during Viktor Kazantsev)", 0, 0, 1),
    ("HHH", (0, 999), "61"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("MMM", (0, 999), "61"): ("Partially plates of the Oblast Department of Internal Affairs", 0, 1, 3),
    ("OOO", (0, 999), "61"): ("Partially plates of the Oblast Legislative Assembly", 0, 1, 4),
    ("BBK", (0, 999), "61"): ("Partially plates of insurance companies in Rostov-on-Don", 0, 1, 1),

    # Saratov Oblast
    ("AAA", (0, 999), "164"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "164"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("XXX", (0, 999), "64"): ("Partially plates of the Oblast courts", 0, 1, 3),
    ("MMM", (0, 999), "64"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("OAA", (0, 999), "64"): ("Partially Federal Security Service plates of the Oblast", 0, 1, 3),

    # Tomsk Oblast
    ("ATO", (0, 999), "70"): ("Partially plates of the Oblast Administration", 0, 1, 5),

    # Tyumen Oblast
    ("ATO", (0, 999), "72"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PTO", (0, 999), "72"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MTO", (0, 999), "72"): ("Partially plates of the Oblast Prosecutor's Office", 0, 1, 3),
    ("HTO", (0, 999), "72"): ("Partially plates of the Tax Service", 0, 1, 3),
    ("CTO", (0, 999), "72"): ("Partially plates of the Oblast courts", 0, 1, 3),
    ("YTO", (0, 999), "72"): ("Partially plates of the bailiff service", 0, 1, 3),
    ("BAA", (0, 999), "72"): ("Partially plates of the Oblast Ministry of Internal Affairs", 0, 1, 3),
    ("KKK", (0, 999), "72"): ("'Gangster' plates", 0, 1, 1),

    # Arkhangelsk Oblast
    ("TTT", (0, 999), "29"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("PPP", (0, 999), "29"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("MAO", (0, 999), "29"): ("Partially plates of the Oblast Ministry of Internal Affairs", 0, 1, 3),

    # Ryazan Oblast
    ("APO", (0, 999), "62"): ("Partially plates of the Oblast Administration", 0, 1, 5),

    # Samara Oblast
    ("PAA", (0, 999), "63"): ("Partially plates of the Oblast Administration", 0, 1, 5),
    ("AAP", (0, 999), "63"): ("Partially plates of the Oblast Administration", 0, 1, 5),
}



# no missing value?
train_csv.isna().sum()


train_csv


train_csv.describe()




# 创建一个包含两张图的窗口，左右并排
plt.figure(figsize=(12, 6))

# 左图：原始价格的直方图
plt.subplot(1, 2, 1)  # 1 行 2 列，第一个子图
sns.histplot(train_csv[train_csv['price'] < 10000000]['price'], kde=True, bins=50)
plt.title("Price Distribution")

# 右图：价格的对数转换后的直方图
plt.subplot(1, 2, 2)  # 1 行 2 列，第二个子图
sns.histplot(np.log10(train_csv['price'] + 1), kde=True, bins=50)
plt.title("Log Transformed Price Distribution")

# 显示图像
plt.tight_layout()  # 自动调整布局，避免重叠
plt.show()




tmp_df = deepcopy(train_csv)

tmp_df['price'] = np.log1p(tmp_df['price'])
# 确保日期列为 datetime 类型
tmp_df['date_by_day'] = pd.to_datetime(train_csv['date']).dt.strftime('%Y-%m-%d')
tmp_df['date_by_day'] = pd.to_datetime(tmp_df['date_by_day'])

# 提取时间特征
tmp_df['year'] = tmp_df['date_by_day'].dt.year
tmp_df['month'] = tmp_df['date_by_day'].dt.month
tmp_df['day'] = tmp_df['date_by_day'].dt.day
tmp_df['weekday'] = tmp_df['date_by_day'].dt.weekday  # 0: Monday, 6: Sunday
tmp_df['is_weekend'] = tmp_df['weekday'].isin([5, 6]).astype(int)  # 是否为周末

# 设置绘图风格
sns.set_style("whitegrid")

# 创建多个子图
fig, axes = plt.subplots(4, 2, figsize=(20, 15))  # 4 行 2 列子图布局
fig.suptitle("Price Trends by Different Time Features", fontsize=18)

# 1️⃣ 按年统计
sns.lineplot(x='year', y='price', data=tmp_df.groupby('year', as_index=False)['price'].sum(), ax=axes[0, 0])
axes[0, 0].set_title("Price by Year")

# 2️⃣ 按月统计
sns.lineplot(x='month', y='price', data=tmp_df.groupby(['month','year'], as_index=False)['price'].sum(), ax=axes[0, 1], hue='year')
axes[0, 1].set_title("Price by Month")

# 3️⃣ 按日统计
sns.lineplot(x='day', y='price', data=tmp_df.groupby('day', as_index=False)['price'].sum(), ax=axes[1, 0])
axes[1, 0].set_title("Price by Day")

# 4️⃣ 按星期统计
sns.lineplot(x='weekday', y='price', data=tmp_df.groupby('weekday', as_index=False)['price'].sum(), ax=axes[1, 1])
axes[1, 1].set_title("Price by Weekday")
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# 5️⃣ 周末 vs. 工作日
sns.barplot(x='is_weekend', y='price', data=tmp_df.groupby('is_weekend', as_index=False)['price'].sum(), ax=axes[2, 0])
axes[2, 0].set_title("Price on Weekends vs. Weekdays")
axes[2, 0].set_xticks([0, 1])
axes[2, 0].set_xticklabels(["Weekday", "Weekend"])

# 6️⃣ 按完整日期绘制价格趋势（原始需求）
sns.lineplot(x='date_by_day', y='price', data=tmp_df.groupby('date_by_day', as_index=False)['price'].sum(), ax=axes[2, 1])
axes[2, 1].set_title("Daily Price Trend")
axes[2, 1].tick_params(axis='x', rotation=45)  # 旋转日期标签

# 7 按节假日价格趋势（原始需求）
sns.lineplot(x='date_by_day', y='price', data=tmp_df.groupby('date_by_day', as_index=False)['price'].sum(), ax=axes[2, 1])
axes[2, 1].set_title("Daily Price Trend")
axes[2, 1].tick_params(axis='x', rotation=45)  # 旋转日期标签

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 让 suptitle 不被覆盖
plt.show()



label = 'price'


!pip install holidays


from sklearn.preprocessing import LabelEncoder
import holidays




tmp = deepcopy(train_csv)
tmp['date'] = pd.to_datetime(tmp['date']).dt.strftime('%Y-%m-%d')
# 创建俄罗斯节假日对象
ru_holidays = holidays.Russia(years=[2021,2022, 2023, 2024, 2025])

tmp['holiday'] = tmp['date'].apply(lambda x:ru_holidays.get(x))

# Crear un objeto LabelEncoder
encoder = LabelEncoder()

# Ajustar y transformar las etiquetas
# tmp['holiday'] = encoder.fit_transform(tmp['holiday'])


# Opcional: para obtener el mapeo de etiquetas a números
# print(encoder.classes_) 
tmp


# 确保你的数据被加载到 tmp DataFrame


# 为节假日的价格做分组
grouped_data = tmp.groupby('holiday', as_index=False)['price'].sum()
np.log(grouped_data.price)
# 绘制箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(x='holiday', y='price', data=grouped_data)
plt.xticks(fontsize=5)
plt.title('按节假日划分的价格趋势')
plt.xlabel('节假日')
plt.ylabel('价格')
plt.show()


import re
def process_data(csv):
    df = deepcopy(csv)
    df[['number', 'region_code']] = df['plate'].apply(lambda x: pd.Series(re.findall(r'\d+', x)))
    df['letter'] = df['plate'].apply(lambda x: pd.Series(''.join(re.findall(r'\D+', x))))
    
    def get_region_key(x):
        for key, value in REGION_CODES.items():
            if x in value:
                return key
        return None  # 防止找不到匹配项时返回 NaN
        
    def func(row):    
        letter = row['letter']
        number = row['number']
        region_code = row['region_code']
        
        for key,value in GOVERNMENT_CODES.items():
            _letter,_numbers_range,_code = key
            forbidden_buy,advantage,significance = value
            if letter == _letter and _numbers_range[0] < number < _numbers_range[1] and region_code == _code:
                return value
    
        
    df['region'] = df['region_code'].apply(get_region_key).astype('category')

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    # df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # 是否为周末
    df['year'] = df['date'].dt.year

    df['number'] = df['number'].astype('category')
    df['letter'] = df['letter'].astype('category')
    df['region_code'] = df['region_code'].astype('category')
    df.drop(columns=['plate','date','region'],inplace=True)
    
    return df
train_df = process_data(train_csv)
test_df = process_data(test_csv)

train_df.isna().sum()



print(train_df.info(), test_df.info())


# import lightgbm as lgb
# lgb_test_preds = []
# # Define the parameters for the LightGBM regressor
# params = {
#     'objective': 'regression',  # Regression task
#     'metric': None,           # None
#     'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
#     # 'num_leaves': 31,           # Number of leaves in a tree
#     'learning_rate': 0.02,     # Learning rate
#     'feature_fraction': 0.9,    # Fraction of features to use for each tree
#     'bagging_fraction': 0.8,    # Fraction of data to use for each tree
#     'bagging_freq': 5,          # Frequency for bagging
#     'verbose': 1,
#     # 'device': 'gpu',
#      'n_estimators' : 10000,
#     'early_stopping_rounds' : 200
# }

# # 自定义 SMAPE 评估函数
# def smape(y_true, y_pred):
#     return np.mean(2 * np.abs(np.exp(y_pred) - np.exp(y_true)) / (np.abs(np.exp(y_true)) + np.abs(np.exp(y_pred)) + 1e-8)) * 100

# def smape_lgbm(preds, dataset):
#     labels = dataset.get_label()
#     smape_val = smape(labels, preds)
#     return 'SMAPE', smape_val, False  # False 代表越小越好


# X = train_df.drop(columns=[label])
# X_test = test_df.drop(columns=[label])
# y = train_df[label]

# cv = KFold(5, shuffle=True, random_state=0)
# cv_splits = cv.split(X, y)
# scores = []

# for train_idx, val_idx in cv_splits:
    
#     X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#     y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

#     y_train_fold, y_val_fold = np.log(y_train_fold), np.log(y_val_fold)
#     train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
#     valid_data = lgb.Dataset(X_val_fold, label=y_val_fold)
#     # Train the model
#     lgb_model = lgb.train(params, train_data, valid_sets = valid_data,feval=smape_lgbm)
    
#     # Make predictions on the test set
#     val_pred = lgb_model.predict(X_val_fold)
    
    
#     # val_pred = np.exp(val_pred)
#     # y_val_fold = np.exp(y_val_fold)

    
#     score = smape(y_val_fold, val_pred)
#     scores.append(score)
#     test_pred = lgb_model.predict(X_test)
#     lgb_test_preds.append(test_pred)
#     print(score)
    
# print(f'Cross-validated smape score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# print(f'Max smape score: {np.max(scores):.3f}')
# print(f'Min smape score: {np.min(scores):.3f}')


# # 获取特征重要性（基于 split）
# importance_split = lgb_model.feature_importance(importance_type='split')

# # 获取特征重要性（基于 gain）
# importance_gain = lgb_model.feature_importance(importance_type='gain')

# # 创建 DataFrame 便于可视化
# feature_importance_df = pd.DataFrame({
#     'Feature':  list(train_df.drop(columns=[label]).columns),
#     'Importance (Split)': importance_split,
#     'Importance (Gain)': importance_gain
# })

# # 按照 Gain 排序
# feature_importance_df = feature_importance_df.sort_values(by='Importance (Gain)', ascending=False)

# # 显示表格
# print(feature_importance_df)

# # 可视化特征重要性
# plt.figure(figsize=(10, 5))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance (Gain)'])
# plt.xlabel('Feature Importance (Gain)')
# plt.ylabel('Features')
# plt.title('LightGBM Feature Importance')
# plt.gca().invert_yaxis()  # 让最重要的特征排在最上方
# plt.show()


%time
!pip install -U autogluon > /dev/null




%time
from autogluon.tabular import TabularDataset, TabularPredictor

from autogluon.core.metrics import make_scorer
import numpy as np

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(np.exp(y_pred) - np.exp(y_true)) / (np.abs(np.exp(y_true)) + np.abs(np.exp(y_pred)) + 1e-8)) * 100

smape_scorer = make_scorer(name = 'smape1',
                           score_func=smape, greater_is_better=False)  # SMAPE 越小越好，所以设为 False
train_df['price'] = np.log(train_df['price'])
predictor = TabularPredictor(label='price',eval_metric=smape_scorer).fit(
                                  train_df,              
                                  # presets="best_quality",
                                  num_bag_folds=5,
    time_limit=3600)
test_data = TabularDataset(test_df)




train_df.to_csv('train_df.csv',index=False)
test_df.to_csv('test_df.csv',index=False)


y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()
y_pred.to_csv('y_pred_v1.csv',index=False)


print(predictor.fit_summary())



predictor.evaluate(train_data, silent=True)


predictor.leaderboard(train_data)


y_pred


submi = pd.DataFrame()
submi['id'] = test_csv['id']
submi[label] = y_pred


submi = pd.DataFrame()
submi['id'] = test_csv['id']
submi[label] = y_pred
submi.to_csv('submi_v2.csv',index=False)

