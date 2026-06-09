import pandas as pd
import numpy as np 
import os 
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns 
import gc 
import time 
from glob import glob 
from IPython.display import display 
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
from wordcloud import WordCloud 
from typing import Dict 

warnings.simplefilter("ignore")
plt.style.use("dark_background")
np.random.seed(42)
debug = True


%%time 


district = pd.read_csv("../input/learnplatform-covid19-impact-on-digital-learning/districts_info.csv")
district.drop("county_connections_ratio", axis=1, inplace=True)
product = pd.read_csv("../input/learnplatform-covid19-impact-on-digital-learning/products_info.csv", usecols=["LP ID", "Product Name", "Primary Essential Function"])

engagement = pd.DataFrame()
for i, f in enumerate(glob("../input/learnplatform-covid19-impact-on-digital-learning/engagement_data/*.csv")):
    df = pd.read_csv(f)
    district_id = f.split("/")[-1].split(".")[0]
    df["district_id"] = district_id 
#     df["engagement_index"] = df["engagement_index"].fillna(df["engagement_index"].mean())
    df["pct_access"] = df["pct_access"].fillna(df["pct_access"].mean())
    engagement = pd.concat([engagement, df])
    if debug:
        if i > 30:
            break 
    else:
        if i > 233:
            break 
        
display(district.head())
display(product.head())
display(engagement.head())



display(district.isnull().sum().to_frame())


'''
district table 
'''

def pct_black(x):
    if x == "[0, 0.2[":
        return float(0.1)
    elif x == "[0.2, 0.4[":
        return float(0.3)
    elif x == "[0.4, 0.6[":
        return float(0.5)
    elif x == "[0.8, 1[":
        return float(0.9)
    else:
        return np.nan
    
def pct_free(x):
    if x == "[0, 0.2[":
        return float(0.1)
    elif x == "[0.2, 0.4[":
        return float(0.3)
    elif x == "[0.4, 0.6[":
        return float(0.5)
    elif x == "[0.6, 0.8[":
        return float(0.7)
    elif x == "[0.8, 1[":
        return float(0.9)
    else:
        return np.nan 
    
def pp_raw(x):
    if x == "missing":
        return x 
    else:
        x = str(x)
        upper_value = int(x.split(",")[0].split("[")[1].strip())
        lower_value = int(x.split(",")[1].split("[")[0].strip())
        return (upper_value+lower_value) // 2 
    
# drppna
district = district[district.state.notna()].reset_index(drop=True)

# take the median as float 
district["pct_black/hispanic"] = district["pct_black/hispanic"].apply(pct_black)
district["pct_free/reduced"] = district["pct_free/reduced"].apply(pct_free)
district["pp_total_raw"] = district["pp_total_raw"].fillna("missing")
district["pp_total_raw"] = district["pp_total_raw"].apply(pp_raw)

# fill values 
district["pct_black/hispanic"] = district["pct_black/hispanic"].fillna(district[district["pct_black/hispanic"].notna()].groupby("locale")["pct_black/hispanic"].mean())
district["pct_black/hispanic"] = district["pct_black/hispanic"].fillna(district["pct_black/hispanic"].mean())
district["pct_free/reduced"] = district["pct_free/reduced"].fillna(district[district["pct_free/reduced"].notna()].groupby("locale")["pct_free/reduced"].mean())
district["pct_free/reduced"] = district["pct_free/reduced"].fillna(district["pct_free/reduced"].mean())
district["pp_total_raw"] = district["pp_total_raw"].apply(lambda x: district.loc[district["pp_total_raw"] != "missing", "pp_total_raw"].mean() if x == "missing" else x)

display(district.isnull().sum().to_frame())



'''
product table 

'''

def get_main(x):
    if type(x) == list and len(x) > 0 :
        return x[0] 
    else:
        return "missing"
    
def get_sub(x):
    if type(x) == list and len(x) > 1:
        return x[1]
    else:
        return "missing"
    
# split main and sub in Primary Essential Function    
product["Primary Essential Function"] = product["Primary Essential Function"].fillna("missing")
product["split"] = product["Primary Essential Function"].apply(lambda x: x.split("-"))
product["main"] = product["split"].apply(get_main)
product["sub"] = product["split"].apply(get_sub)
product.drop(["split", "Primary Essential Function"], axis=1, inplace=True)

display(product.isnull().sum().to_frame())



'''
merge dataframe 

'''

df = pd.merge(engagement, product, how="left", left_on="lp_id", right_on="LP ID")
del engagement, product

df["district_id"] = df["district_id"].astype(int)
district["district_id"] = district["district_id"].astype(int)
df = pd.merge(df, district, how="left", left_on="district_id", right_on="district_id")
del district 
gc.collect()

if debug:
    df = df.dropna()
    
df = df.rename(columns={"Product Name": "product", "pct_black/hispanic": "hispanic", "pct_free/reduced": "reduced", "pp_total_raw": "pp_raw"})
df.drop("LP ID", axis=1, inplace=True)
df.head()


display(df.isnull().sum().to_frame())


def reduce_mem_usage(train_data):
    start_mem = train_data.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in train_data.columns:
        col_type = train_data[col].dtype

        if col_type != object:
            c_min = train_data[col].min()
            c_max = train_data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    train_data[col] = train_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    train_data[col] = train_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    train_data[col] = train_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    train_data[col] = train_data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    train_data[col] = train_data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    train_data[col] = train_data[col].astype(np.float32)
                else:
                    train_data[col] = train_data[col].astype(np.float64)
        else:
            train_data[col] = train_data[col].astype('category')

    end_mem = train_data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return train_data


%%time

df = reduce_mem_usage(df)
df["time"] = pd.to_datetime(df.time)


#hispanic 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

hispanic = df.groupby("product").mean().loc[:, ["hispanic"]].sort_values("hispanic", ascending=False)
hispanic.iloc[:10].sort_values("hispanic").plot(kind="barh", cmap="Greens", ax=ax[0])
ax[0].set_title("high hispanic x Product")
ax[0].set_xticks(np.arange(0, 0.6, 0.1))
hispanic.iloc[-10:].sort_values("hispanic", ascending=False).plot(kind="barh", cmap="Greens", ax=ax[1])
ax[1].set_title("low hispanic x Product")
ax[1].set_xticks(np.arange(0, 0.6, 0.1))
plt.suptitle("hispanic", fontsize=20)
plt.tight_layout()

#reduces 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

hispanic = df.groupby("product").mean().loc[:, ["reduced"]].sort_values("reduced", ascending=False)
hispanic.iloc[:10].sort_values("reduced").plot(kind="barh", cmap="Greens", ax=ax[0])
ax[0].set_title("high reduced x Product")
ax[0].set_xticks(np.arange(0, 0.6, 0.1))
hispanic.iloc[-10:].sort_values("reduced", ascending=False).plot(kind="barh", cmap="Greens", ax=ax[1])
ax[1].set_title("low reduced x Product")
ax[1].set_xticks(np.arange(0, 0.6, 0.1))
plt.suptitle("reduced", fontsize=20)
plt.tight_layout()

#pp_total_raw 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes.ravel()

hispanic = df.groupby("product").mean().loc[:, ["pp_raw"]].sort_values("pp_raw", ascending=False)
hispanic.iloc[:10].sort_values("pp_raw").plot(kind="barh", cmap="Greens", ax=ax[0])
ax[0].set_title("high pp_total_raw x Product")
ax[0].set_xticks(np.arange(0, 16000, 2000))
hispanic.iloc[-10:].sort_values("pp_raw", ascending=False).plot(kind="barh", cmap="Greens", ax=ax[1])
ax[1].set_title("low pp_total_raw x Product")
ax[1].set_xticks(np.arange(0, 16000, 2000))
plt.suptitle("pp_total_raw", fontsize=20)
plt.tight_layout()

del hispanic 
gc.collect()


fig, axes = plt.subplots(1, 3, figsize=(22, 6))
ax = axes.ravel()

grp = df.groupby("locale").mean().loc[:, ["hispanic"]]
grp.plot(kind="barh", cmap="Greens", ax=ax[0])
ax[0].set_title("hispanic")

grp = df.groupby("locale").mean().loc[:, ["reduced"]]
grp.plot(kind="barh", cmap="Greens", ax=ax[1])
ax[1].set_title("reduced")

grp = df.groupby("locale").mean().loc[:, ["pp_raw"]]
grp.plot(kind="barh", cmap="Greens", ax=ax[2])
ax[2].set_title("pp_total_raw")

plt.tight_layout()



locale = df.locale.value_counts().index
fig, axes = plt.subplots(1, 4, figsize=(22, 12))
ax = axes.ravel()

# locale 1
grp = df.loc[df.locale == locale[0], ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10]
grp.sort_values("pct_access").plot(kind="barh", ax=ax[0], cmap="Greens")
ax[0].set_title(locale[0])
ax[0].set_xticks(np.arange(0, 20.0, 2.0))


# locale 2 
grp = df.loc[df.locale == locale[1], ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10]
grp.sort_values("pct_access").plot(kind="barh", ax=ax[1], cmap="Greens")
ax[1].set_title(locale[1])
ax[1].set_xticks(np.arange(0, 20.0, 2.0))


# locale 3 
grp = df.loc[df.locale == locale[2], ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10]
grp.sort_values("pct_access").plot(kind="barh", ax=ax[2], cmap="Greens")
ax[2].set_title(locale[2])
ax[2].set_xticks(np.arange(0, 20.0, 2.0))


# locale 4
grp = df.loc[df.locale == locale[3], ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10]
grp.sort_values("pct_access").plot(kind="barh", ax=ax[3], cmap="Greens")
ax[3].set_title(locale[3])
ax[3].set_xticks(np.arange(0, 20.0, 2.0))

del grp 
gc.collect()
plt.suptitle("most popular Plattform by Locale", fontsize=18)
plt.tight_layout()


grp = df.groupby("product").mean().loc[:, ["pct_access"]]

fig, axes = plt.subplots(1, 2, figsize=(22, 6))
ax = axes.ravel()

grp.sort_values("pct_access", ascending=False)[:10].sort_values("pct_access", ascending=True).plot(kind="barh", ax=ax[0])
ax[0].set_title("most use product by all.")
ax[0].set_xticks(np.arange(0, 20.0, 2.0))

grp.sort_values("pct_access", ascending=True)[:10].sort_values("pct_access", ascending=False).plot(kind="barh", ax=ax[1])
ax[0].set_title("most unuse product by all.")
ax[1].set_xticks(np.arange(0, 0.1, 0.01))

plt.suptitle("populer Plattform.", fontsize=18)
plt.tight_layout()


df["week"] = df.time.dt.dayofweek 

week_name = {
    0: "monday", 
    1: "thesday",
    2: "wednesday", 
    3: "thursday", 
    4: "friday",
    5: "saturday",
    6: "sunday",
}

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
ax = axes.ravel()

for i, week in enumerate(np.argsort(df.week.unique())):
    x = df.loc[df.week == week, ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:5]
    x.sort_values("pct_access", ascending=True).plot(kind="barh", cmap="Greens", ax=ax[i])
    ax[i].set_title(week_name[week])
    ax[i].set_xticks(np.arange(0, 20.0, 2.0))
    
plt.suptitle("Weekly popular platform ", fontsize=18)
plt.tight_layout()


def viz_transition(high, low):
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    ax = axes.ravel()
    color = ["b", "r", "g"]
    for i, p in enumerate(high):
        x = df[df["product"] == p]
        x[["time", "pct_access"]].groupby("time").mean().loc[:, "pct_access"].plot(cmap="Greens", ax=ax[0], color=color[i])
    ax[0].set_title("most popular plattform top3 trends.")
    ax[0].legend(high)
    
    for i, p in enumerate(low):
        x = df[df["product"] == p]
        x[["time", "pct_access"]].groupby("time").mean().loc[:, "pct_access"].plot(cmap="Greens", ax=ax[1], color=color[i])
    ax[1].set_title("not most popular plattform under3 trends.")
    ax[1].legend(low)
    plt.suptitle("Trends.", fontsize=16)
    plt.tight_layout()
    
    del x
    gc.collect()
    

top_3_plattform = df[["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:3].index.values.to_list()
under_3_plattform = df[["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=True)[:3].index.values.to_list()
viz_transition(top_3_plattform, under_3_plattform)


top_10_plattform = df[["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10].index.values.to_list()
under_10_plattform = df[["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[-10:].index.values.to_list()

fig, axes = plt.subplots(1, 2, figsize=(22, 12))
ax = axes.ravel()

grp = df.loc[df["product"].isin(top_10_plattform), ["locale"]].value_counts().to_frame()
grp.plot(kind="barh", cmap="Greens", ax=ax[0])
ax[0].set_title("Top10 Plattfrom")
ax[0].set_xticks(np.arange(0, 40000, 10000))

grp = df.loc[df["product"].isin(under_10_plattform), ["locale"]].value_counts().to_frame()
grp.plot(kind="barh", cmap="Greens", ax=ax[1])
ax[1].set_title("Under10 Plattfrom")
ax[1].set_xticks(np.arange(0, 4000, 1000))

plt.suptitle("Popular platform migration district ", fontsize=18)
plt.tight_layout()

del grp 
gc.collect()


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District Of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

pd.DataFrame({"TOP": top_10_plattform, "UNDER": under_10_plattform})


def show_location(plattform):
    x = df.loc[df["product"].isin(plattform), "state"].value_counts().to_frame().sort_values("state", ascending=False)[:20]
    x = x.rename(columns={"state": "count"})
    x["state"] = x.index 
    x["state"] = x.state.map(us_state_abbrev)
    x = x.reset_index(drop=True)
    data = dict(
        type = 'choropleth',
        colorscale = 'blackbody',
        locations = x["state"],
        locationmode = 'USA-states',
        z = list(x['count']),
        text = x["state"],
        colorbar = {'title':'States'},
      )
    layout = dict(title = 'States',
                  geo = dict(projection = {'type':'mercator'})
                 )
    layout = dict(title= 'Platform popularity usage count ',
                  geo = {'scope':'usa'})

    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap)
    del x
    gc.collect()
    
    
def show_location_locale(locale, plattform):
    x = df.loc[(df["locale"] == locale) & (df["product"].isin(plattform)), ["state", "product"]]
    x = x.groupby("state").count().loc[:, ["product"]]
    x = x[x["product"] != 0]
    x = x.rename(columns={"product": "count"})
    x["state"] = x.index 
    x["state"] = x["state"].map(us_state_abbrev)
    x = x.reset_index(drop=True)
    
    data = dict(
        type = 'choropleth',
        colorscale = 'blackbody',
        locations = x["state"],
        locationmode = 'USA-states',
        z = list(x['count']),
        text = x["state"],
        colorbar = {'title':'States'},
      )
    layout = dict(title = 'States',
                  geo = dict(projection = {'type':'mercator'})
                 )
    layout = dict(title= f'Platform popularity usage count by {locale}',
                  geo = {'scope':'usa'})
    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap)
    del x
    gc.collect()


show_location(top_10_plattform)


show_location(under_10_plattform)


show_location_locale(df.locale.unique()[0], top_10_plattform)


show_location_locale(df.locale.unique()[1], top_10_plattform)


show_location_locale(df.locale.unique()[2], top_10_plattform)


show_location_locale(df.locale.unique()[3], top_10_plattform)


show_location_locale(df.locale.unique()[0], under_10_plattform)


show_location_locale(df.locale.unique()[1], under_10_plattform)


show_location_locale(df.locale.unique()[2], under_10_plattform)


show_location_locale(df.locale.unique()[3], under_10_plattform)


def viz_main(high, low):
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    ax = axes.ravel()
    x = df.loc[df["product"].isin(high), ["main"]].value_counts()
    ax[0].pie(x=x.values, labels=x.index, startangle=90, autopct="%1.1f%%", shadow=True, counterclock=False)
    ax[0].set_title("most popular Plattfrom x main.")
    
    x = df.loc[df["product"].isin(low), ["main"]].value_counts()
    ax[1].pie(x=x.values, labels=x.index, startangle=90, autopct="%1.1f%%", shadow=True, counterclock=False)
    ax[1].set_title("not most popular Plattfrom x main.")
    
    plt.tight_layout()
    del x 
    gc.collect()
    


viz_main(top_10_plattform, under_10_plattform)


def create_vocab(sub) -> Dict[str, int]:
    word2count = {}
    for s in sub:
        s = s.strip()
        for ss in s.split():
            ss = ss.lower()
            if ss == "&": continue 
            if ss not in word2count:
                word2count[ss] = 1 
            else:
                word2count[ss] += 1 
    return word2count 

def viz_sub(high, low):
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    ax = axes.ravel()
    
    x = df.loc[df["product"].isin(high), "sub"].to_list()
    word2count = create_vocab(x)
    word = WordCloud(width=1440, height=1100).generate_from_frequencies(word2count)
    ax[0].imshow(word)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("most popular Plattfrom x sub word")
    
    x = df.loc[df["product"].isin(low), "sub"].to_list()
    word2count = create_vocab(x)
    word = WordCloud(width=1440, height=1100).generate_from_frequencies(word2count)
    ax[1].imshow(word)
    ax[1].set_xticks([])
    ax[1].set_yticks([])    
    ax[1].set_title("not most popular Plattfrom x sub word")


viz_sub(top_10_plattform, under_10_plattform)


df["quarter"] = df.time.dt.quarter 
display(df["quarter"].value_counts().to_frame().sort_index())


before_covid = df[df.quarter == 1]
before_covid = before_covid.pivot_table(values="pct_access", index="product", columns="quarter", aggfunc="mean")
before_covid.columns = ["pct_access"]

before_covid.sort_values("pct_access", ascending=False)[:10].sort_values("pct_access", ascending=True).plot(kind="barh", figsize=(22, 12))
plt.title("before the pandemic most popular Plattfrom.", fontsize=18)
plt.show()


after_covid = df[df.quarter != 1]
after_covid = after_covid.groupby("product").mean().loc[:, ["pct_access"]]

after_covid.sort_values("pct_access", ascending=False)[:10].sort_values("pct_access", ascending=True).plot(kind="barh", figsize=(22, 12))
plt.title("after the pandemic most popular Plattfrom.", fontsize=18)
plt.show()


growth = pd.merge(before_covid.rename(columns={"pct_access": "before_access"}), after_covid.rename(columns={"pct_access": "after_access"}), how="outer", left_index=True, right_index=True)
growth = growth.fillna(0)
growth["growth_access"] = growth["after_access"] - growth["before_access"]
growth = growth[["growth_access"]].sort_values("growth_access", ascending=False)[:10]

growth.sort_values("growth_access", ascending=True).plot(kind="barh", figsize=(22, 12))
plt.title("Access growth potential ", fontsize=18)
plt.show()

del before_covid, after_covid, growth 
gc.collect()


df["google"] = df["product"].apply(lambda x: x.find("Google") >= 0)
df["google"] = df["google"].apply(lambda x: 1 if x is True else 0)
google = df["google"].value_counts()

plt.figure(figsize=(12, 12))
plt.pie(x=google.values, labels=google.index, startangle=90, counterclock=False, autopct="%1.1f%%")
plt.legend(["not Google", "Google"])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(22, 12))
ax = axes.ravel()

google_service = df.loc[df["google"] == 1, ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=False)[:10]
google_service.sort_values("pct_access", ascending=True).plot(kind="barh", ax=ax[0])
ax[0].set_title("most popluar Plattfrom with google")

google_service = df.loc[df["google"] == 1, ["product", "pct_access"]].groupby("product").mean().sort_values("pct_access", ascending=True)[:10]
google_service.sort_values("pct_access", ascending=False).plot(kind="barh", ax=ax[1])
ax[1].set_title("not most popluar Plattfrom with google")

plt.suptitle("Google Service.", fontsize=20)
plt.tight_layout()

del google_service
gc.collect()


def viz_transition_google():
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    ax = axes.ravel()
    
    x = df.loc[df["google"] == 0, ["time", "pct_access"]].groupby("time").mean().sort_index()
    x.plot(ax=ax[0])
    ax[0].set_title("Not google x pct access.")
    
    x = df.loc[df["google"] == 1, ["time", "pct_access"]].groupby("time").mean().sort_index()
    x.plot(ax=ax[1])
    ax[1].set_title("is google x pct access.")
    
    plt.tight_layout()
    del x 
    gc.collect()
    


viz_transition_google()


df.head()

