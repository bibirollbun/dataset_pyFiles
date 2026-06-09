# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


event_attendees=pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/event_attendees.csv.gz", compression = "gzip")
users=pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/users.csv")
user_friends = pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/user_friends.csv.gz", compression = "gzip")
events=pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/events.csv.gz", compression = "gzip")

train=pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/train.csv")
test=pd.read_csv("/kaggle/input/event-recommendation-engine-challenge/test.csv")


#convert to datetime format 
train["timestamp"] = pd.to_datetime(train["timestamp"],format = "ISO8601")
users["joinedAt"] = pd.to_datetime(users["joinedAt"],format = "ISO8601")
events["start_time"] = pd.to_datetime(events["start_time"],format = "ISO8601", errors = "coerce")


def info_describe(df):
    print("{} \n \n {} ".format(df.info(), df.describe(include = "all")))


train[train["invited"]!=0].head()


info_describe(train)


event_attendees.head()


info_describe(event_attendees)


users.head()


info_describe(users)


events.head()


info_describe(events)


user_friends.head()


info_describe(user_friends)


train.groupby(["interested","not_interested"]).size()


# Create function that creates exploded sub table from event_attendees table

def exploded_sub_df( target_col, id, source_table,new_colname):
    sub_df = source_table[[id,target_col]].copy()
    sub_df[new_colname] = sub_df[target_col].str.split()
    sub_df = sub_df.explode(new_colname, ignore_index = True)
    sub_df.dropna(subset=[new_colname], inplace=True)
    sub_df[new_colname] = sub_df[new_colname].astype('int64')
    sub_df.drop([target_col], axis = 1, inplace = True)
    return sub_df


# create table that captures users who said "yes" to an event
event_attendees_yes = exploded_sub_df( "yes", "event", event_attendees, "user")
print(event_attendees_yes.info())
event_attendees_yes.head()


# create table that captures users who said "maybe" to an event
event_attendees_maybe = exploded_sub_df( "maybe", "event", event_attendees,"user")
print(event_attendees_maybe.info())
event_attendees_maybe.head()


# create table that captures users who said "no" to an event
event_attendees_no = exploded_sub_df( "no", "event", event_attendees,"user")
print(event_attendees_no.info())
event_attendees_no.head()


# create table that captures users who said "invited" to an event
event_attendees_invited = exploded_sub_df( "invited", "event", event_attendees,"user")
print(event_attendees_invited.info())
event_attendees_invited.head()


user_friends_df = exploded_sub_df( "friends", "user", user_friends,"friends_uid")
print(user_friends_df.info())
user_friends_df.head()


# check the date column in event table
print("MIN of event[\"start_time\"] : {} \n".format(events["start_time"].min()))
print("MAX of event[\"start_time\"] : {}\n".format(events["start_time"].max()))
print(events.groupby(events["start_time"].dt.strftime('%Y'))['event_id'].nunique())


# check the date column in users table
print("MIN of users[\"joinedAt\"] : {} \n".format(users["joinedAt"].min()))
print("MAX of users[\"joinedAt\"] : {} \n".format(users["joinedAt"].max()))
print(users.groupby(users["joinedAt"].dt.strftime('%Y'))['user_id'].nunique())

# train["timestamp"] = pd.to_datetime(train["timestamp"],format = "ISO8601")
# users["joinedAt"] = pd.to_datetime(users["joinedAt"],format = "ISO8601")
# events["start_time"] = pd.to_datetime(events["start_time"],format = "ISO8601", errors = "coerce")


# check the date column in train table
print("MIN of train[\"timestamp\"] : {} \n".format(train["timestamp"].min()))
print("MAX of train[\"timestamp\"] : {} \n".format(train["timestamp"].max()))
print(train.groupby(train["timestamp"].dt.strftime('%Y'))['user'].nunique())


train_df=pd.merge(train,events[["event_id","start_time"]],left_on="event",right_on="event_id",how="left").drop("event_id",axis=1)
train_df["hrs_to_event"] = (train_df["start_time"]-train_df["timestamp"]).dt.total_seconds()/3600


train_df.head()


print("\n*** df shape \n {} ***".format(train_df.shape))
print("\n*** hrs_to_event analysis \n {} ***".format(train_df["hrs_to_event"].describe()))
print("\n*** Missing count {} ***".format (train_df['hrs_to_event'].isnull().sum()))


fig, axes = plt.subplots(1,3, figsize = (15,4))

train_df["hrs_to_event"].plot.hist(ax=axes[0])
train_df["hrs_to_event"].plot.hist(ax=axes[1], density = True)
train_df["hrs_to_event"].plot.kde(ax=axes[1])
train_df["hrs_to_event"].plot.box(ax=axes[2])

plt.show()


pd.cut(train_df["hrs_to_event"],bins = [-1000,0,21,55,93,500,1000]).value_counts().sort_index()


train_df = pd.merge(train_df,users[["user_id","joinedAt"]],left_on ="user",right_on ="user_id",how="left").drop("user_id",axis=1)
train_df["minsToEvent_frmJoin"] = (train_df["timestamp"] - train_df["joinedAt"]).dt.total_seconds()/60


train_df.head()


print("***Table shape {}***".format(train_df.shape))
print("\n*** minsToEvent_frmJoin univariate analysis \n {}***".format(train_df["minsToEvent_frmJoin"].describe()))
print("\n*** minsToEvent_frmJoin null analysis {}***".format(train_df["minsToEvent_frmJoin"].isnull().sum()))


fig, axes = plt.subplots(1,3,figsize = (15,4))
train_df["minsToEvent_frmJoin"].plot.hist(ax = axes[0])
train_df["minsToEvent_frmJoin"].plot.hist(density = True, ax = axes[1])
train_df["minsToEvent_frmJoin"].plot.kde(ax = axes[1])
train_df["minsToEvent_frmJoin"].plot.box(ax = axes[2])
plt.show()


pd.cut(train_df["minsToEvent_frmJoin"], bins = [-3500,0,0.019633,0.056268,28.509839, 100, 1000 ]).value_counts().sort_index()


# To all the users in the train data get all events that they either chose yes or maybe
train_user = pd.DataFrame(train_df["user"].unique(), columns=['user']) # Get unique user from train
weekday_pref_yes = pd.merge(train_user,event_attendees_yes.astype({'event':'Int64'}),on="user",how="inner")
weekday_pref_maybe = pd.merge(train_user, event_attendees_maybe.astype({'event':'Int64'}),on="user",how = "inner")
print("train_user.shape={}".format(train_user.shape))
print("weekday_pref_yes.shape={}".format(weekday_pref_yes.shape))
print("weekday_pref_maybe.shape={}\n".format(weekday_pref_maybe.shape))

# Concat to get the list of user to yes/maybe events
weekday_pref = pd.concat([weekday_pref_yes, weekday_pref_maybe], axis = 0)

print(weekday_pref.info())
weekday_pref.head()


# join to events table to get event start_time and extract event's weekday info
weekday_pref=pd.merge(weekday_pref,events[["event_id","start_time"]],left_on ="event",right_on="event_id",how="left")
weekday_pref["weekday"] = weekday_pref['start_time'].dt.strftime('%A')
print(weekday_pref.shape)
weekday_pref[weekday_pref["weekday"].notna()].head()


# transpose so that we get 7 column for each weekday and count of events on each week day the user had yes/maybe
weekday_pref_wide = (weekday_pref.groupby(["user","weekday"])['event'].nunique()
                     .unstack(fill_value = 0)
                     .add_prefix("pref_")
                     .reset_index()
                    )
print(weekday_pref_wide.shape)
weekday_pref_wide.head()


# join the new weekday pref datapoint to train_df table
train_df = pd.merge(train_df, weekday_pref_wide, on = "user", how = "left")
print(train_df.info()) 
train_df[train_df["pref_Saturday"].notna()].head()

