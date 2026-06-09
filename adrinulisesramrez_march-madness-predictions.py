# Libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sqlalchemy import text
import csv
from pathlib import Path



# Configure pandas to show all dataframe´s columns
pd.set_option('display.max_columns', None)


from kaggle_secrets import UserSecretsClient
# Connecting to postgresql database using create_engine from sqlalchemy
# Credentials
db_user =  UserSecretsClient().get_secret("db_user")
db_password =  UserSecretsClient().get_secret("db_pass")
db_port =  UserSecretsClient().get_secret("db_port")
db_host =  UserSecretsClient().get_secret("db_host")
db_name =  UserSecretsClient().get_secret("db_name")
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Test conection
with engine.connect() as connection:
    print("Conexión exitosa a postgresql")


# Working directory
data_path = "/kaggle/input/march-machine-learning-mania-2025/"
 
#Check if a string can be converted to an integer type
def to_int32_or_object(string):
    try:
        int(string)
        return 'int32'
    except:
        return 'object'

#Ignore files
ignore_files = ["SeedBenchmarkStage1.csv","MMasseyOrdinals.csv","SampleSubmissionStage1.csv",
                "SampleSubmissionStage2.csv"]

# Load tables in postgresql database
for filename in os.listdir(data_path):
    if filename.endswith('.csv') and filename not in ignore_files:
        column_names = []
        data_types = {}
        table_name = os.path.splitext(filename)[0]
        file_path = os.path.join(data_path, filename)
        # Obtain a dictionary with the data_types for each column name
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            c_row = 0
            for row in csv_reader:
                if c_row < 2:
                    column_names.append(row)
                else:
                    break
                c_row +=1
        column_names[1] = list(map(to_int32_or_object,column_names[1]))
        column_names = list(zip(*column_names))
        data_types = dict(column_names)  
        # Create a SQL table for each file
        df = pd.read_csv(file_path, encoding='cp1252', dtype=data_types)
        df.to_sql(table_name, engine, schema=None , if_exists='replace', index=False)
        print(f"Table '{table_name}' created succesfully.")


# Create "FullDeatailedResults", which is the combination of all the detailed results from men's and women's regular
# season and tournament. We also introduce the "Genre" column, where 1 for men and 0 for women
query = """
   DROP TABLE IF EXISTS "FullDetailedResults";
   CREATE TABLE "FullDetailedResults" AS     
      SELECT *, 1 AS "Genre" 
      FROM 
      (
          SELECT *  FROM "MNCAATourneyDetailedResults"
          UNION
          SELECT * FROM "MRegularSeasonDetailedResults"
       )
       UNION
       SELECT *, 0 AS "Genre" 
       FROM 
       (
          SELECT *  FROM "WNCAATourneyDetailedResults"
          UNION
          SELECT * FROM "WRegularSeasonDetailedResults"
       )
      ;
"""
with engine.begin() as conn:
    conn.execute(text(query))


# Show all the column and table names in the database
query = """
    SELECT DISTINCT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name;
"""
print(pd.read_sql_query(query, engine).to_string())


# Total number of teams
query = """
   SELECT COUNT(*) AS "Total_Teams (M)"
   FROM "MTeams"
"""
display(pd.read_sql_query(query, engine))

# No. of teams playing in 2023
query = """
    SELECT COUNT(DISTINCT "ID") "Total Teams Playing In 2023 (M)"
    FROM
    (
        SELECT DISTINCT "WTeamID" AS "ID"
        FROM "MRegularSeasonCompactResults"
        WHERE"Season" = 2023
        UNION 
        SELECT DISTINCT "LTeamID" AS "ID"
        FROM "MRegularSeasonCompactResults"
        WHERE"Season" = 2023
    )
"""
pd.read_sql_query(query, engine)


# Days in which the Conference tournament and the NCAA tourney


query = """
    SELECT  MIN("DayNum") AS "FirstDayConfTourney", MAX("DayNum") AS "LastDayConfTourney"
    FROM "MConferenceTourneyGames";
"""
display(pd.read_sql_query(query, engine))

query = """
    SELECT  MIN("DayNum") AS "FirstDayNCAATourney", MAX("DayNum") AS "LastDayNCAATourney"
    FROM "MNCAATourneyCompactResults";
"""
display(pd.read_sql_query(query, engine))


# Number of regular season games in a season (2023)
query = """
    SELECT COUNT(*) AS "No. of Regular Season Games (M2023)"
    FROM "MRegularSeasonCompactResults"
    WHERE "DayNum" < 115 AND "Season" = 2023
        
"""

pd.read_sql_query(query, engine)


# Number of wins and losses for each team in the 2023 regular season, ordered by conference
query = """
   SELECT "Conf"."ConfAbbrev", "Winned"."TeamID", "noW", "noL", "noW" + "noL" AS "TotalGames"
   FROM
      (SELECT "WTeamID" AS "TeamID", Count("WTeamID") AS "noW"
      FROM "MRegularSeasonCompactResults"
      WHERE "DayNum" < 115 AND "Season" = 2023
      GROUP BY "WTeamID") "Winned" 
   JOIN
      (SELECT "LTeamID" AS "TeamID", Count(*) AS "noL"
      FROM "MRegularSeasonCompactResults"
      WHERE "DayNum" < 115 AND "Season" = 2023
      GROUP BY "LTeamID") "Lossed"
   ON "Winned"."TeamID" = "Lossed"."TeamID"
   JOIN
       (SELECT  "TeamID", "ConfAbbrev"
       FROM "MTeamConferences"
       WHERE "Season" = 2023) "Conf"
   ON "Conf"."TeamID" = "Lossed"."TeamID"
   ORDER BY "ConfAbbrev"
   
"""
pd.read_sql_query(query, engine).head(20)


# Number of IntraConferenceGames
query1 = """ 
CREATE TEMP TABLE IF NOT EXISTS  "IntraConference" AS  
    SELECT "WConf",  COUNT(*) AS "IntraConferenceGames"
    FROM
    (
        SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConf" , "Lmt"."ConfAbbrev" "LConf"
        FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
             JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
        WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
    )
    WHERE "WConf"="LConf"
    GROUP BY "WConf";
"""

# No. of interconference games where the team in conference team won
query2 = """ 
CREATE TEMP TABLE IF NOT EXISTS "InterConference1" AS  
    SELECT "WConf", COUNT(*) As "IntraConferenceWinnedGames"
    FROM
    (
        SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConf" , "Lmt"."ConfAbbrev" "LConf"
        FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
             JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
        WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
    )
    WHERE "WConf" !="LConf"
    GROUP BY "WConf"
"""

# No. of interconference games where the team in conference team lossed
query3 = """ 
CREATE TEMP TABLE IF NOT EXISTS "InterConference2" AS
    SELECT "LConf", COUNT(*) AS "IntraConferenceLossedGames"
    FROM
    (
        SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConf" , "Lmt"."ConfAbbrev" "LConf"
        FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
             JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
        WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
    )
    WHERE "WConf" !="LConf"
    GROUP BY "LConf"
"""

with engine.begin() as conn:
    conn.execute(text(query1))
    conn.execute(text(query2))
    conn.execute(text(query3))

# No. of intra and interconference games
query4 = """
    SELECT 
        "WConf" ,
        "IntraConferenceGames",
        "IntraConferenceLossedGames" AS "InterConferenceGamesLossed",
        "IntraConferenceWinnedGames" AS "InterConferenceGamesWinned",
        "IntraConferenceWinnedGames" + "IntraConferenceLossedGames" AS "InterConferenceGames",
        "IntraConferenceGames" + "IntraConferenceWinnedGames" + "IntraConferenceLossedGames" AS "TotalGames"
    FROM 
        ("IntraConference" JOIN "InterConference1" USING("WConf")) t1 
    JOIN "InterConference2" t2 ON t1."WConf"= t2."LConf"      
"""
display(pd.read_sql_query(query4, engine).head(10))

# Statistics of Games Played
query5 = """
    SELECT AVG("IntraConferenceGames") AS "AvgNoIntraConfGames", 
    STDDEV("IntraConferenceGames") AS "StdNoIntraConfGames",
    AVG("InterConferenceGames") AS "AvgNoInterConfGames", 
    STDDEV("InterConferenceGames") AS "StdNoInterConfGames",
    AVG("TotalGames") AS "AvgTotalGames",
    STDDEV("TotalGames")  AS "StdTotalGames"
    FROM
    (SELECT "WConf" ,"IntraConferenceGames", "IntraConferenceWinnedGames" + "IntraConferenceLossedGames" AS "InterConferenceGames", 
   "IntraConferenceGames" + "IntraConferenceWinnedGames" + "IntraConferenceLossedGames" AS "TotalGames"
    FROM ("IntraConference" JOIN "InterConference1" USING("WConf")) t1 
    JOIN "InterConference2" t2 ON t1."WConf"= t2."LConf"
    )
      
       
"""
display(pd.read_sql_query(query5, engine))




# No. of Interconference games played per team
query1 = """
    SELECT "WTeamID", COUNT(*) "IntraconferenceGames"
    FROM
    (
        SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConfAbbrev" , "Lmt"."ConfAbbrev" AS "LConfAbbrev" 
        FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
             JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
        WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
        AND "Lmt"."ConfAbbrev" = "Wmt"."ConfAbbrev"
    )  
    GROUP BY "WTeamID"
"""

query2 = """
    SELECT t1."WTeamID", t1."WInterconferenceGames", t2."LInterconferenceGames",  
    t1."WInterconferenceGames" + t2."LInterconferenceGames" AS "InterConferenceGames"
    FROM
        (SELECT "WTeamID", COUNT(*) AS "WInterconferenceGames"
        FROM
            (SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConfAbbrev" , "Lmt"."ConfAbbrev" AS "LConfAbbrev" 
            FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
                 JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
            WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
            AND "Lmt"."ConfAbbrev" != "Wmt"."ConfAbbrev")  
        GROUP BY "WTeamID") t1
    JOIN 
        (SELECT "LTeamID", COUNT(*) AS "LInterconferenceGames"
        FROM
            (SELECT "WTeamID" AS "WTeamID","LTeamID", "Wmt"."ConfAbbrev" AS "WConfAbbrev" , "Lmt"."ConfAbbrev" AS "LConfAbbrev" 
            FROM ("MRegularSeasonCompactResults" "mr" JOIN "MTeamConferences" "Wmt" ON "mr"."WTeamID" = "Wmt"."TeamID")
                 JOIN  "MTeamConferences" "Lmt"   ON "mr"."LTeamID" = "Lmt"."TeamID"
            WHERE "mr"."DayNum" < 115 AND "mr"."Season" = 2023 AND "Lmt"."Season" = 2023 AND "Wmt"."Season" = 2023
            AND "Lmt"."ConfAbbrev" != "Wmt"."ConfAbbrev")  
        GROUP BY "LTeamID") t2
    ON t1."WTeamID" = t2."LTeamID"
    
    
"""


t1 = pd.read_sql_query(query1, engine)
display(t1)

t2 = pd.read_sql_query(query2, engine)
display(t2)

print("Average intraconference games per team: ", t1["IntraconferenceGames"].mean())
print("Average interconference games per team: " , t2["InterConferenceGames"].mean())


print(11.06/(363.-11.)*100)


query = """ SELECT * FROM "FullDetailedResults" LIMIT 3 """
pd.read_sql_query(query, engine)


query = """
       DROP TABLE IF EXISTS "FullDetailedResultsMod";
       CREATE TEMP TABLE "FullDetailedResultsMod" AS
       SELECT concat_ws('_', "Season"::text,  "ATeamID"::text, "BTeamID"::text) AS "ID",*
       FROM
       (
           SELECT
           "Season",
           "DayNum",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WTeamID" ELSE "LTeamID" END "ATeamID",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WScore"  ELSE "LScore"  END "AScore",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFGM"  ELSE "LFGM"  END "AFGM",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFGA"  ELSE "LFGA"  END "AFGA",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFGM3"  ELSE "LFGM3"  END "AFGM3",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFGA3"  ELSE "LFGA3"  END "AFGA3",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFTM"  ELSE "LFTM"  END "AFTM",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WFTA"  ELSE "LFTA"  END "AFTA",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WOR"  ELSE "LOR"  END "AOR",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WDR"  ELSE "LDR"  END "ADR",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WAst"  ELSE "LAst"  END "AAst",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WTO"  ELSE "LTO"  END "ATO",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WStl"  ELSE "LStl"  END "AStl",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WBlk"  ELSE "LBlk"  END "ABlk",
           CASE WHEN "WTeamID" < "LTeamID" THEN "WPF"  ELSE "LPF"  END "APF",

           CASE WHEN "WTeamID" < "LTeamID" THEN "LTeamID" ELSE "WTeamID" END "BTeamID",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LScore"  ELSE "WScore"  END "BScore",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFGM"  ELSE "WFGM"  END "BFGM",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFGA"  ELSE "WFGA"  END "BFGA",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFGM3"  ELSE "WFGM3"  END "BFGM3",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFGA3"  ELSE "WFGA3"  END "BFGA3",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFTM"  ELSE "WFTM"  END "BFTM",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LFTA"  ELSE "WFTA"  END "BFTA",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LOR"  ELSE "WOR"  END "BOR",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LDR"  ELSE "WDR"  END "BDR",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LAst"  ELSE "WAst" END "BAst",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LTO"  ELSE "WTO"  END "BTO",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LStl"  ELSE "WStl"  END "BStl",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LBlk"  ELSE "WBlk"  END "BBlk",
           CASE WHEN "WTeamID" < "LTeamID" THEN "LPF"  ELSE "WPF"  END "BPF",
           "NumOT",
           "Genre",
           CASE WHEN "WTeamID" < "LTeamID" THEN 1 ELSE 0 END "Target",

           
         
           CASE WHEN "WTeamID" < "LTeamID" AND "WLoc"='H' THEN 1
                WHEN "LTeamID" < "WTeamID" AND "WLoc"='A' THEN 1 
                ELSE 0  END "AHomeLoc",
           CASE WHEN "WTeamID" < "LTeamID" AND "WLoc"='A' THEN 1  
                WHEN "LTeamID" < "WTeamID" AND "WLoc"='H' THEN 1
                ELSE 0  END "AAwayLoc",
           CASE WHEN "WLoc"='N' THEN 1  ELSE 0  END "ANeutralLoc"      
           FROM "FullDetailedResults"
           ORDER BY "Season" DESC, "DayNum" DESC
      )
"""

with engine.begin() as conn:
    conn.execute(text(query))

print("The reorganized table is:")
query = """ SELECT * FROM "FullDetailedResultsMod" LIMIT 5 """
pd.read_sql_query(query, engine)


query = """
    CREATE OR REPLACE FUNCTION fnGamesPlayedByTeam(teamID integer)
    RETURNS TABLE
    (   
        "ID" text,
        "Season" integer,
        "DayNum" integer,
        "ATeamID" integer,
        "AScore" integer,
        "AFGM" integer,
        "AFGA" integer,
        "AFGM3" integer,
        "AFGA3" integer,
        "AFTM" integer,
        "AFTA" integer,
        "AOR" integer,
        "ADR" integer,
        "AAst" integer,
        "ATO" integer,
        "AStl" integer,
        "ABlk" integer,
        "APF" integer,
        "BTeamID" integer,
        "BScore" integer,
        "BFGM" integer,
        "BFGA" integer,
        "BFGM3" integer,
        "BFGA3" integer,
        "BFTM" integer,
        "BFTA" integer,
        "BOR" integer,
        "BDR" integer,
        "BAst" integer,
        "BTO" integer,
        "BStl" integer,
        "BBlk" integer,
        "BPF" integer,
        "AHomeLoc" integer,
        "AAwayLoc" integer,
        "ANeutralLoc" integer,
        "NumOT" integer,
        "Genre" integer,
        "Target" integer
    )
    AS
    $$
    BEGIN
        RETURN QUERY
        SELECT 
           t."ID",
           t."Season",
           t."DayNum",
           CASE WHEN t."ATeamID" = teamID THEN t."ATeamID" ELSE t."BTeamID" END "ATeamID",
           CASE WHEN t."ATeamID" = teamID THEN t."AScore"  ELSE t."BScore"  END "AScore",
           CASE WHEN t."ATeamID" = teamID THEN t."AFGM"    ELSE t."BFGM"    END "AFGM",
           CASE WHEN t."ATeamID" = teamID THEN t."AFGA"    ELSE t."BFGA"    END "AFGA",
           CASE WHEN t."ATeamID" = teamID THEN t."AFGM3"   ELSE t."BFGM3"   END "AFGM3",
           CASE WHEN t."ATeamID" = teamID THEN t."AFGA3"   ELSE t."BFGA3"   END "AFGA3",
           CASE WHEN t."ATeamID" = teamID THEN t."AFTM"    ELSE t."BFTM"    END "AFTM",
           CASE WHEN t."ATeamID" = teamID THEN t."AFTA"    ELSE t."BFTA"    END "AFTA",
           CASE WHEN t."ATeamID" = teamID THEN t."AOR"     ELSE t."BOR"     END "AOR",
           CASE WHEN t."ATeamID" = teamID THEN t."ADR"     ELSE t."BDR"     END "ADR",
           CASE WHEN t."ATeamID" = teamID THEN t."AAst"    ELSE t."BAst"    END "AAst",
           CASE WHEN t."ATeamID" = teamID THEN t."ATO"     ELSE t."BTO"     END "ATO",
           CASE WHEN t."ATeamID" = teamID THEN t."AStl"    ELSE t."BStl"    END "AStl",
           CASE WHEN t."ATeamID" = teamID THEN t."ABlk"    ELSE t."BBlk"    END "ABlk",
           CASE WHEN t."ATeamID" = teamID THEN t."APF"     ELSE t."BPF"     END "APF",
           
           CASE WHEN t."ATeamID" = teamID THEN t."BTeamID" ELSE t."ATeamID" END "BTeamID",
           CASE WHEN t."ATeamID" = teamID THEN t."BScore"  ELSE t."AScore"  END "BScore",
           CASE WHEN t."ATeamID" = teamID THEN t."BFGM"    ELSE t."AFGM"    END "BFGM",
           CASE WHEN t."ATeamID" = teamID THEN t."BFGA"    ELSE t."AFGA"    END "BFGA",
           CASE WHEN t."ATeamID" = teamID THEN t."BFGM3"   ELSE t."AFGM3"   END "BFGM3",
           CASE WHEN t."ATeamID" = teamID THEN t."BFGA3"   ELSE t."AFGA3"   END "BFGA3",
           CASE WHEN t."ATeamID" = teamID THEN t."BFTM"    ELSE t."AFTM"    END "BFTM",
           CASE WHEN t."ATeamID" = teamID THEN t."BFTA"    ELSE t."AFTA"    END "BFTA",
           CASE WHEN t."ATeamID" = teamID THEN t."BOR"     ELSE t."AOR"     END "BOR",
           CASE WHEN t."ATeamID" = teamID THEN t."BDR"     ELSE t."ADR"     END "BDR",
           CASE WHEN t."ATeamID" = teamID THEN t."BAst"    ELSE t."AAst"    END "BAst",
           CASE WHEN t."ATeamID" = teamID THEN t."BTO"     ELSE t."ATO"     END "BTO",
           CASE WHEN t."ATeamID" = teamID THEN t."BStl"    ELSE t."AStl"    END "BStl",
           CASE WHEN t."ATeamID" = teamID THEN t."BBlk"    ELSE t."ABlk"    END "BBlk",
           CASE WHEN t."ATeamID" = teamID THEN t."BPF"     ELSE t."APF"     END "BPF",
          /*
           CASE WHEN t."ATeamID" = teamID AND t."Loc"=1 THEN 1
                WHEN t."BTeamID" = teamID AND t."Loc"=1 THEN -1
                WHEN t."ATeamID" = teamID AND t."Loc"=-1 THEN -1  
                WHEN t."BTeamID" = teamID AND t."Loc"=-1 THEN 1
                ELSE 0  END "Loc",*/
            CASE WHEN t."ATeamID" = teamID THEN t."AHomeLoc" ELSE 1 -t."AHomeLoc"       END "AHomeLoc",
            CASE WHEN t."ATeamID" = teamID THEN t."AAwayLoc" ELSE 1 -t."AAwayLoc"       END "AAwayLoc",
            CASE WHEN t."ATeamID" = teamID THEN t."ANeutralLoc" ELSE 1 -t."ANeutralLoc" END "ANeutralLoc",
            t."NumOT",
            t."Genre",
            CASE WHEN t."ATeamID" = teamID THEN t."Target" ELSE 1-t."Target" END "Target"
        FROM "FullDetailedResultsMod" t
        WHERE t."ATeamID" = teamID OR t."BTeamID" = teamID;
    END;
    $$
    LANGUAGE plpgsql;
"""
with engine.begin() as conn:
    conn.execute(text(query))
    
print("Example for ATeamID=1405")
query = """ SELECT * FROM fnGamesPlayedByTeam(1461) LIMIT 5 """
display(pd.read_sql_query(query, engine))
print("We see now that ATeamID=1461 for all columns.")


window = 5

query = f"""

DROP TABLE IF EXISTS "RollingAvg{window}";
CREATE TABLE "RollingAvg{window}" AS

WITH  "teams" AS(
    SELECT DISTINCT "ATeamID" as "TeamID" 
    FROM "FullDetailedResultsMod"
    UNION
    SELECT DISTINCT "BTeamID" as "TeamID"
    FROM "FullDetailedResultsMod"
),

"t2" AS(
   SELECT 
        "ID",
        "Season" ,
        "DayNum" ,
        "ATeamID" ,
        "AScore" ,
        "AFGM" ,
        "AFGA" ,
        "AFGM3" ,
        "AFGA3" ,
        "AFTM" ,
        "AFTA" ,
        "AOR" ,
        "ADR" ,
        "AAst" ,
        "ATO" ,
        "AStl" ,
        "ABlk" ,
        "APF" ,
        "BTeamID" ,
        "BScore" ,
        "BFGM" ,
        "BFGA" ,
        "BFGM3" ,
        "BFGA3" ,
        "BFTM" ,
        "BFTA" ,
        "BOR" ,
        "BDR" ,
        "BAst" ,
        "BTO" ,
        "BStl" ,
        "BBlk" ,
        "BPF" ,
        "AHomeLoc" ,
        "AAwayLoc" ,
        "ANeutralLoc" ,
        "NumOT" ,
        "Genre" ,
        "Target"
   FROM "teams", LATERAL  fnGamesPlayedByTeam("teams"."TeamID")
)

SELECT
        "ID",
        "Season"     ,
        "DayNum"     ,
        "ATeamID"    ,
        "Genre", 
        AVG("AScore") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AScore" ,
        AVG("AScore") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS "AVG_ASeasonScore",
        SUM("Target") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS "ASeasonWins",
        COUNT("Target") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS "ASeasonPlayed",
        AVG("AFGM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFGM"   ,
        AVG("AFGA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFGA"   ,
        AVG("AFGM3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFGM3"  ,
        AVG("AFGA3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFGA3"  ,
        AVG("AFTM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFTM"   ,
        AVG("AFTA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AFTA"   ,
        AVG("AOR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AOR"    ,
        AVG("ADR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_ADR"    ,
        AVG("AAst") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AAst"   ,
        AVG("ATO") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_ATO"    ,
        AVG("AStl") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AStl"   ,
        AVG("ABlk") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_ABlk"   ,
        AVG("APF") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_APF"    ,
        AVG("Target") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_ATarget"    ,
        
        AVG("BScore") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BScore" ,
        AVG("BScore") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS "AVG_BSeasonScore",  
        AVG("BFGM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFGM"   ,
        AVG("BFGA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFGA"   ,
        AVG("BFGM3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFGM3"  ,
        AVG("BFGA3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFGA3"  ,
        AVG("BFTM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFTM"   ,
        AVG("BFTA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BFTA"   ,
        AVG("BOR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BOR"    ,
        AVG("BDR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING)  "AVG_BDR"    ,
        AVG("BAst") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BAst"  ,
        AVG("BTO") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BTO"   ,
        AVG("BStl") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BStl"   ,
        AVG("BBlk") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BBlk"   ,
        AVG("BPF") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_BPF",
        
        AVG("AHomeLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AHomeLoc"  , 
        AVG("AAwayLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_AAwayLoc"  , 
        AVG("ANeutralLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_ANeutralLoc"  , 
        "AHomeLoc"  , 
        "AAwayLoc" , 
        "ANeutralLoc", 
        AVG("NumOT") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING) "AVG_NumOT" ,
        "Target"
FROM "t2"
"""
#pd.read_sql_query(query, engine)
with engine.begin() as conn:
    conn.execute(text(query))
    
print("The rolling avg table is:")
query = """ SELECT * FROM "RollingAvg5" LIMIT 5; """
pd.read_sql_query(query, engine)


query = """
    SELECT 
        "ID",  
        "DayNum",  
        "Season",  
        "Genre",  
        "t1"."ATeamID"      AS "T1_ID",  
        "t1"."AHomeLoc" AS "T1_AHomeLoc", 
        "t1"."AAwayLoc" AS "T1_AAwayLoc", 
        "t1"."ANeutralLoc"    AS "T1_ANeutralLoc", 
        "t1"."AVG_AScore"   AS "AVG_T1_Score",  
        "t1"."AVG_ASeasonScore"   AS "AVG_T1_SeasonScore",  
        "t1"."ASeasonWins"   AS "T1_SeasonWins", 
        "t1"."ASeasonPlayed"   AS "T1_SeasonPlayed", 
        "t1"."AVG_AFGM"     AS "AVG_T1_FGM",  
        "t1"."AVG_AFGA"     AS "AVG_T1_FGA",  
        "t1"."AVG_AFGM3"    AS "AVG_T1_FGM3",  
        "t1"."AVG_AFGA3"    AS "AVG_T1_FGA3",  
        "t1"."AVG_AFTM"     AS "AVG_T1_FTM",  
        "t1"."AVG_AFTA"     AS "AVG_T1_FTA",  
        "t1"."AVG_AOR"      AS "AVG_T1_OR",  
        "t1"."AVG_ADR"      AS "AVG_T1_DR",  
        "t1"."AVG_AAst"     AS "AVG_T1_Ast",  
        "t1"."AVG_ATO"      AS "AVG_T1_TO",  
        "t1"."AVG_AStl"     AS "AVG_T1_Stl",  
        "t1"."AVG_ABlk"     AS "AVG_T1_Blk",  
        "t1"."AVG_APF"      AS "AVG_T1_PF",  
        "t1"."AVG_ATarget"  AS "AVG_T1_Wins",
        "t1"."AVG_AHomeLoc" AS "AVG_T1_AHomeLoc", 
        "t1"."AVG_AAwayLoc" AS "AVG_T1_AAwayLoc", 
        "t1"."AVG_ANeutralLoc"    AS "AVG_T1_ANeutralLoc", 

        "t1"."AVG_NumOT"    AS "AVG_T1_NumOT",  
        "t1"."AVG_BScore"   AS "AVG_T1c_Score",
        "t1"."AVG_BSeasonScore"   AS "AVG_T1c_SeasonScore",
        "t1"."AVG_BFGM"     AS "AVG_T1c_FGM",  
        "t1"."AVG_BFGA"     AS "AVG_T1c_FGA",  
        "t1"."AVG_BFGM3"    AS "AVG_T1c_FGM3",  
        "t1"."AVG_BFGA3"    AS "AVG_T1c_FGA3",  
        "t1"."AVG_BFTM"     AS "AVG_T1c_FTM",  
        "t1"."AVG_BFTA"     AS "AVG_T1c_FTA",  
        "t1"."AVG_BOR"      AS "AVG_T1c_OR",  
        "t1"."AVG_BDR"      AS "AVG_T1c_DR",  
        "t1"."AVG_BAst"     AS "AVG_T1c_Ast",  
        "t1"."AVG_BTO"      AS "AVG_T1c_TO",  
        "t1"."AVG_BStl"     AS "AVG_T1c_Stl",  
        "t1"."AVG_BBlk"     AS "AVG_T1c_Blk",  
        "t1"."AVG_BPF"      AS "AVG_T1c_PF",  


        "t2"."ATeamID"      AS "T2_ID",  
        "t2"."AVG_AScore"   AS "AVG_T2_Score",  
        "t2"."AVG_ASeasonScore"   AS "AVG_T2_SeasonScore", 
        "t2"."ASeasonWins"   AS "T2_SeasonWins",  
        "t2"."ASeasonPlayed"   AS "T2_SeasonPlayed", 
        "t2"."AVG_AFGM"     AS "AVG_T2_FGM",  
        "t2"."AVG_AFGA"     AS "AVG_T2_FGA",  
        "t2"."AVG_AFGM3"    AS "AVG_T2_FGM3",  
        "t2"."AVG_AFGA3"    AS "AVG_T2_FGA3",  
        "t2"."AVG_AFTM"     AS "AVG_T2_FTM",  
        "t2"."AVG_AFTA"     AS "AVG_T2_FTA",  
        "t2"."AVG_AOR"      AS "AVG_T2_OR",  
        "t2"."AVG_ADR"      AS "AVG_T2_DR",  
        "t2"."AVG_AAst"     AS "AVG_T2_Ast",  
        "t2"."AVG_ATO"      AS "AVG_T2_TO",  
        "t2"."AVG_AStl"     AS "AVG_T2_Stl",  
        "t2"."AVG_ABlk"     AS "AVG_T2_Blk",  
        "t2"."AVG_APF"      AS "AVG_T2_PF", 
        "t2"."AVG_ATarget"  AS "AVG_T2_Wins",
        "t2"."AVG_AHomeLoc"       AS "AVG_T2_AHomeLoc", 
        "t2"."AVG_AAwayLoc"       AS "AVG_T2_AAwayLoc", 
        "t2"."AVG_ANeutralLoc"    AS "AVG_T2_ANeutralLoc",  
        "t2"."AVG_NumOT"    AS "AVG_T2_NumOT",  
        "t2"."AVG_BScore"   AS "AVG_T2c_Score",  
        "t2"."AVG_BSeasonScore"   AS "AVG_T2c_SeasonScore",
        "t2"."AVG_BFGM"     AS "AVG_T2c_FGM",  
        "t2"."AVG_BFGA"     AS "AVG_T2c_FGA",  
        "t2"."AVG_BFGM3"    AS "AVG_T2c_FGM3",  
        "t2"."AVG_BFGA3"    AS "AVG_T2c_FGA3",  
        "t2"."AVG_BFTM"     AS "AVG_T2c_FTM",  
        "t2"."AVG_BFTA"     AS "AVG_T2c_FTA",  
        "t2"."AVG_BOR"      AS "AVG_T2c_OR",  
        "t2"."AVG_BDR"      AS "AVG_T2c_DR",  
        "t2"."AVG_BAst"     AS "AVG_T2c_Ast",  
        "t2"."AVG_BTO"      AS "AVG_T2c_TO",  
        "t2"."AVG_BStl"     AS "AVG_T2c_Stl",  
        "t2"."AVG_BBlk"     AS "AVG_T2c_Blk",  
        "t2"."AVG_BPF"      AS "AVG_T2c_PF",  

        "t1"."Target"       AS "Target"  

    FROM "RollingAvg5" "t1" JOIN "RollingAvg5" "t2" USING("ID","DayNum", "Season", "Genre")
    WHERE "t1"."ATeamID" < "t2"."ATeamID" AND "Season" > 2010
    ORDER BY "t1"."ID", "Season", "DayNum"  
"""

df = pd.read_sql_query(query, engine)
display(df.tail(5))



from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


dataset = df
dataset.drop(["ID", "T1_ID", "T2_ID"],axis=1, inplace=True)


cutday = 115  # We only consider days where DayNum>cutday. In practice, the model performs better in this range.
              # Training with the regular season instances reduces performanec
final_regular_season_day = 132
dataset.dropna(inplace=True)
train_set = dataset.query(f'(Season < 2024 and DayNum > {cutday}) or ( Season == 2024 and DayNum <= {final_regular_season_day} and DayNum > {cutday})').reset_index(drop=True)
# We leave the NCAA 2024 data as the test set.
test_set = dataset.query(f'(Season == 2024 and DayNum > {final_regular_season_day})') 
X_train  = train_set.drop(["Target"], axis=1, inplace=False)
X_test  = test_set.drop(["Target"], axis=1, inplace=False)
y_train  = train_set["Target"]
y_test  = test_set["Target"]
seasons = train_set["Season"].unique()


# This class performs incremental season cross validation.
# We start training in the first (regular and NCAA) and second season (regular) and test in the second season (NCAA).
# Then we train in the first, second (regular and NCAA) and third season (regular) and test in the third season (NCAA) 
# and so on.

class IncrementalSeasonCV(BaseCrossValidator):
    def __init__(self, seasons, season_column="Season", day_column="DayNum", final_regular_day=final_regular_season_day):
        """
        seasons: list of all the available seasons
        season_column: name of column with the season information
        """
        self.seasons = sorted(seasons)  
        self.season_column = season_column
        self.day_column = day_column
        self.final_regular_day = final_regular_day

    def split(self, X, y=None, groups=None):
        """
        Generate train and test splits.
        """
        for i in range(len(self.seasons)-2 ):  
            if self.seasons[i+1] == 2020:
                # There was no NCAA tournament in 2020
                pass
            else: 
                train_seasons = self.seasons[:i+1]  # Include all the seasons up to the i-th
                test_season = self.seasons[i+1]  # The i-th season  (NCAA) is the test season 
                train_idx = X[
                     (X[self.season_column].isin(train_seasons)) |
                     ((X[self.season_column] == test_season) & 
                      (X[self.day_column] <= self.final_regular_day ))
                ].index
                # Filtrar los índices de validación
                test_idx = X[
                    (X[self.season_column] == test_season) & 
                    (X[self.day_column] > self.final_regular_day) 
                ].index
                yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return number of splits
        """
        return len(self.seasons) - 3 


# Test of IncrementalSeasonCV
s=0
for i, j in IncrementalSeasonCV(seasons).split(X_train):
    s += 1 
    print("Split: ", s)
    train_seasons = (X_train.iloc[i])["Season"].unique()
    last_season = train_seasons[-1]
    training_days = [[min(X_train.iloc[i][((X_train.iloc[i])["Season"]==season)]["DayNum"]), 
                     max(X_train.iloc[i][((X_train.iloc[i])["Season"]==season)]["DayNum"])] for season in train_seasons]
    test_days = [min(X_train.iloc[j]["DayNum"]), 
                     max(X_train.iloc[j]["DayNum"])]
    #training_days = list(map(list, zip(*training_days)))
    print("TrainSeasons",(X_train.iloc[i])["Season"].unique())
    print("TrainDays")
    for l in training_days:
        print(l)
    print("TestSeasons", (X_train.iloc[j])["Season"].unique())
    print("TestDays")
    print(test_days, "\n")
    
  


#Plot Cross Validation Scores
def show_cv_results(cv_results, display_stats=True):
    x = [year for year  in range(2012,2024) if year !=  2020]
    fig = plt.figure(figsize=(10,7))
    fig, ax = plt.subplots(3, 2,figsize=(10,7))
    keys = [[ i+"_"+j for i in ["train", "test"]] for j in ["accuracy", "precision", "recall", "f1", "brier"]] 

    for i in range(4):
        ax[i//2,i%2].plot(x,cv_results[keys[i][0]], label=keys[i][0])
        ax[i//2,i%2].plot(x,cv_results[keys[i][1]], label=keys[i][1])
        ax[i//2,i%2].legend(loc="upper right")
        ax[i//2,i%2].set_ylim([.5,1])

    ax[2,0].plot(x,-cv_results[keys[4][0]], label=keys[4][0])
    ax[2,0].plot(x,-cv_results[keys[4][1]], label=keys[4][1])
    ax[2,0].legend(loc="upper right")
    ax[2,0].set_ylim([0,.4])
    ax[2,1].axis('off')
    plt.setp(ax, xlabel = "test year")
    plt.tight_layout()
    plt.show() 


    #print(f"{'metric':<15}", "mean", "\t", "std")
    stats_cv_results = []
    for key, value in cv_results.items():
        #print(f"{key:<15}",  abs(value.mean().round(3)), "\t", value.std().round(3))
        stats_cv_results.append([key, abs(value.mean().round(3)),value.std().round(3)])
    if(display_stats):
        display(pd.DataFrame(stats_cv_results, columns=["metric", "mean", "std"]))



# Logistic regression and Icremental Seasonal Cross Validation
log_reg = make_pipeline(StandardScaler(),  LogisticRegression(random_state=42, max_iter=200, penalty='l2'))
scoring = {'accuracy':'accuracy','precision':'precision', 'recall':'recall', 'f1':'f1','brier':'neg_brier_score'}
cv_log_reg = cross_validate(log_reg, X_train, y_train, cv=IncrementalSeasonCV(seasons), scoring=scoring,
                           return_train_score=True)


show_cv_results(cv_log_reg)


param_distribs = {'decision_tree__max_depth': randint(low=3, high=10),
                  'decision_tree__max_leaf_nodes': randint(low=10, high=100),
                  'decision_tree__max_features': randint(low=20, high=len(X_train.columns)),
                 }

pipeline = Pipeline([("decision_tree", DecisionTreeClassifier(random_state=42))])
cv_tree_clf = RandomizedSearchCV(pipeline, param_distributions=param_distribs, n_iter=20, cv=IncrementalSeasonCV(seasons),
                   scoring=scoring, random_state=42,refit='brier')
cv_tree_clf.fit(X_train, y_train)

print(cv_tree_clf.best_params_)
print("Best Score: ", cv_tree_clf.best_score_)
#Note that the Brier score comes with a minus sign.


param_distribs = {'rf__max_depth': randint(low=3, high=10),
                  'rf__max_leaf_nodes': randint(low=10, high=100),
                  'rf__max_features': randint(low=20, high=len(X_train.columns)),
                  'rf__n_estimators': np.arange(50,200,1)
                 }

forest = Pipeline([("rf", RandomForestClassifier(random_state=42))])
cv_forest_clf = RandomizedSearchCV(forest, param_distributions=param_distribs, n_iter=20, cv=IncrementalSeasonCV(seasons),
                   scoring=scoring, random_state=42,refit='brier', n_jobs=-1)
cv_forest_clf.fit(X_train, y_train)

print(cv_forest_clf.best_params_)
print("Best Score: ",cv_forest_clf.best_score_)



param_distribs = {'gb__max_depth': randint(low=3, high=10),
                  'gb__max_leaf_nodes': randint(low=10, high=100),
                  'gb__max_features': randint(low=20, high=len(X_train.columns)),
                  'gb__learning_rate': [.001,.005,.01,.05,.1,.5],
                  'gb__n_estimators': np.arange(50,200,1)
                  
                 }

gb_clf = Pipeline([("gb",  GradientBoostingClassifier(n_iter_no_change=10, random_state=42))])
cv_gb_clf = RandomizedSearchCV(gb_clf, param_distributions=param_distribs, n_iter=20, cv=IncrementalSeasonCV(seasons),
                   scoring=scoring, random_state=42,refit='brier', n_jobs=-1)
cv_gb_clf.fit(X_train, y_train)
print(cv_gb_clf.best_params_)
print("Best Score: ", cv_gb_clf.best_score_)


log_reg = make_pipeline(StandardScaler(),  LogisticRegression(random_state=42, max_iter=200, penalty='l2'))
log_reg.fit( X_train, y_train)
print("Test score: ", brier_score_loss(y_test, log_reg.predict_proba(X_test)[:,1]))



# We train the logistic model in the whole data set
train_set = dataset.query(f'DayNum > {cutday}').reset_index(drop=True)
log_reg = make_pipeline(StandardScaler(),  LogisticRegression(random_state=42, max_iter=200, penalty='l2'))
log_reg.fit(X_train, y_train)
print("Model fitted to the whole dataset.")


window = 5

query = f"""

DROP TABLE IF EXISTS "RollingAvg{window}Current";
CREATE TEMP TABLE "RollingAvg{window}Current" AS

WITH  "teams" AS(
    SELECT DISTINCT "ATeamID" as "TeamID" 
    FROM "FullDetailedResultsMod"
    UNION
    SELECT DISTINCT "BTeamID" as "TeamID"
    FROM "FullDetailedResultsMod"
),

"t2" AS(
   SELECT 
        "ID",
        "Season" ,
        "DayNum" ,
        "ATeamID" ,
        "AScore" ,
        "AFGM" ,
        "AFGA" ,
        "AFGM3" ,
        "AFGA3" ,
        "AFTM" ,
        "AFTA" ,
        "AOR" ,
        "ADR" ,
        "AAst" ,
        "ATO" ,
        "AStl" ,
        "ABlk" ,
        "APF" ,
        "BTeamID" ,
        "BScore" ,
        "BFGM" ,
        "BFGA" ,
        "BFGM3" ,
        "BFGA3" ,
        "BFTM" ,
        "BFTA" ,
        "BOR" ,
        "BDR" ,
        "BAst" ,
        "BTO" ,
        "BStl" ,
        "BBlk" ,
        "BPF" ,
        "AHomeLoc" ,
        "AAwayLoc" ,
        "ANeutralLoc" ,
        "NumOT" ,
        "Genre" ,
        "Target"
   FROM "teams", LATERAL  fnGamesPlayedByTeam("teams"."TeamID")
   WHERE "Season" = 2025
)

SELECT DISTINCT ON ("ATeamID")
        "Season"     ,
        "DayNum"     ,
        "ATeamID"    ,
        "Genre", 
        AVG("AScore") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AScore" ,
        AVG("AScore") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "AVG_ASeasonScore",
        SUM("Target") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "ASeasonWins",
        COUNT("Target") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "ASeasonPlayed",
        AVG("AFGM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFGM"   ,
        AVG("AFGA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFGA"   ,
        AVG("AFGM3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFGM3"  ,
        AVG("AFGA3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFGA3"  ,
        AVG("AFTM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFTM"   ,
        AVG("AFTA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AFTA"   ,
        AVG("AOR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AOR"    ,
        AVG("ADR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_ADR"    ,
        AVG("AAst") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AAst"   ,
        AVG("ATO") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_ATO"    ,
        AVG("AStl") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AStl"   ,
        AVG("ABlk") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_ABlk"   ,
        AVG("APF") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_APF"    ,
        AVG("Target") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_ATarget"    ,
        
        AVG("BScore") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BScore" ,
        AVG("BScore") OVER(PARTITION BY "Season", "ATeamID" ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS "AVG_BSeasonScore",  
        AVG("BFGM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFGM"   ,
        AVG("BFGA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFGA"   ,
        AVG("BFGM3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFGM3"  ,
        AVG("BFGA3") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFGA3"  ,
        AVG("BFTM") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFTM"   ,
        AVG("BFTA") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BFTA"   ,
        AVG("BOR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BOR"    ,
        AVG("BDR") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING)  "AVG_BDR"    ,
        AVG("BAst") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BAst"  ,
        AVG("BTO") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BTO"   ,
        AVG("BStl") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BStl"   ,
        AVG("BBlk") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BBlk"   ,
        AVG("BPF") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_BPF",
        
        AVG("AHomeLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AHomeLoc"  , 
        AVG("AAwayLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_AAwayLoc"  , 
        AVG("ANeutralLoc") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_ANeutralLoc"  , 
        "AHomeLoc"  , 
        "AAwayLoc" , 
        "ANeutralLoc", 
        AVG("NumOT") OVER(ORDER BY "ATeamID", "Season" DESC, "DayNum" DESC ROWS BETWEEN CURRENT ROW AND {window-1} FOLLOWING) "AVG_NumOT" 
        FROM "t2"
        ORDER BY "ATeamID" DESC, "DayNum" DESC 
"""
#pd.read_sql_query(query, engine)
with engine.begin() as conn:
    conn.execute(text(query))


query = """
    SELECT 
        concat_ws('_', "Season"::text, "t1"."ATeamID"::text, "t2"."ATeamID"::text) AS "ID",  
        "t1"."DayNum",  
        "t1"."Season",  
        "t1"."Genre",  
        "t1"."ATeamID"      AS "T1_ID",  
        "t1"."AHomeLoc" AS "T1_AHomeLoc", 
        "t1"."AAwayLoc" AS "T1_AAwayLoc", 
        "t1"."ANeutralLoc"    AS "T1_ANeutralLoc", 
        "t1"."AVG_AScore"   AS "AVG_T1_Score",  
        "t1"."AVG_ASeasonScore"   AS "AVG_T1_SeasonScore",  
        "t1"."ASeasonWins"   AS "T1_SeasonWins", 
        "t1"."ASeasonPlayed"   AS "T1_SeasonPlayed", 
        "t1"."AVG_AFGM"     AS "AVG_T1_FGM",  
        "t1"."AVG_AFGA"     AS "AVG_T1_FGA",  
        "t1"."AVG_AFGM3"    AS "AVG_T1_FGM3",  
        "t1"."AVG_AFGA3"    AS "AVG_T1_FGA3",  
        "t1"."AVG_AFTM"     AS "AVG_T1_FTM",  
        "t1"."AVG_AFTA"     AS "AVG_T1_FTA",  
        "t1"."AVG_AOR"      AS "AVG_T1_OR",  
        "t1"."AVG_ADR"      AS "AVG_T1_DR",  
        "t1"."AVG_AAst"     AS "AVG_T1_Ast",  
        "t1"."AVG_ATO"      AS "AVG_T1_TO",  
        "t1"."AVG_AStl"     AS "AVG_T1_Stl",  
        "t1"."AVG_ABlk"     AS "AVG_T1_Blk",  
        "t1"."AVG_APF"      AS "AVG_T1_PF",  
        "t1"."AVG_ATarget"  AS "AVG_T1_Wins",
        "t1"."AVG_AHomeLoc" AS "AVG_T1_AHomeLoc", 
        "t1"."AVG_AAwayLoc" AS "AVG_T1_AAwayLoc", 
        "t1"."AVG_ANeutralLoc"    AS "AVG_T1_ANeutralLoc", 

        "t1"."AVG_NumOT"    AS "AVG_T1_NumOT",  
        "t1"."AVG_BScore"   AS "AVG_T1c_Score",
        "t1"."AVG_BSeasonScore"   AS "AVG_T1c_SeasonScore",
        "t1"."AVG_BFGM"     AS "AVG_T1c_FGM",  
        "t1"."AVG_BFGA"     AS "AVG_T1c_FGA",  
        "t1"."AVG_BFGM3"    AS "AVG_T1c_FGM3",  
        "t1"."AVG_BFGA3"    AS "AVG_T1c_FGA3",  
        "t1"."AVG_BFTM"     AS "AVG_T1c_FTM",  
        "t1"."AVG_BFTA"     AS "AVG_T1c_FTA",  
        "t1"."AVG_BOR"      AS "AVG_T1c_OR",  
        "t1"."AVG_BDR"      AS "AVG_T1c_DR",  
        "t1"."AVG_BAst"     AS "AVG_T1c_Ast",  
        "t1"."AVG_BTO"      AS "AVG_T1c_TO",  
        "t1"."AVG_BStl"     AS "AVG_T1c_Stl",  
        "t1"."AVG_BBlk"     AS "AVG_T1c_Blk",  
        "t1"."AVG_BPF"      AS "AVG_T1c_PF",  


        "t2"."ATeamID"      AS "T2_ID",  
        "t2"."AVG_AScore"   AS "AVG_T2_Score",  
        "t2"."AVG_ASeasonScore"   AS "AVG_T2_SeasonScore", 
        "t2"."ASeasonWins"   AS "T2_SeasonWins",  
        "t2"."ASeasonPlayed"   AS "T2_SeasonPlayed", 
        "t2"."AVG_AFGM"     AS "AVG_T2_FGM",  
        "t2"."AVG_AFGA"     AS "AVG_T2_FGA",  
        "t2"."AVG_AFGM3"    AS "AVG_T2_FGM3",  
        "t2"."AVG_AFGA3"    AS "AVG_T2_FGA3",  
        "t2"."AVG_AFTM"     AS "AVG_T2_FTM",  
        "t2"."AVG_AFTA"     AS "AVG_T2_FTA",  
        "t2"."AVG_AOR"      AS "AVG_T2_OR",  
        "t2"."AVG_ADR"      AS "AVG_T2_DR",  
        "t2"."AVG_AAst"     AS "AVG_T2_Ast",  
        "t2"."AVG_ATO"      AS "AVG_T2_TO",  
        "t2"."AVG_AStl"     AS "AVG_T2_Stl",  
        "t2"."AVG_ABlk"     AS "AVG_T2_Blk",  
        "t2"."AVG_APF"      AS "AVG_T2_PF", 
        "t2"."AVG_ATarget"  AS "AVG_T2_Wins",
        "t2"."AVG_AHomeLoc"       AS "AVG_T2_AHomeLoc", 
        "t2"."AVG_AAwayLoc"       AS "AVG_T2_AAwayLoc", 
        "t2"."AVG_ANeutralLoc"    AS "AVG_T2_ANeutralLoc",  
        "t2"."AVG_NumOT"    AS "AVG_T2_NumOT",  
        "t2"."AVG_BScore"   AS "AVG_T2c_Score",  
        "t2"."AVG_BSeasonScore"   AS "AVG_T2c_SeasonScore",
        "t2"."AVG_BFGM"     AS "AVG_T2c_FGM",  
        "t2"."AVG_BFGA"     AS "AVG_T2c_FGA",  
        "t2"."AVG_BFGM3"    AS "AVG_T2c_FGM3",  
        "t2"."AVG_BFGA3"    AS "AVG_T2c_FGA3",  
        "t2"."AVG_BFTM"     AS "AVG_T2c_FTM",  
        "t2"."AVG_BFTA"     AS "AVG_T2c_FTA",  
        "t2"."AVG_BOR"      AS "AVG_T2c_OR",  
        "t2"."AVG_BDR"      AS "AVG_T2c_DR",  
        "t2"."AVG_BAst"     AS "AVG_T2c_Ast",  
        "t2"."AVG_BTO"      AS "AVG_T2c_TO",  
        "t2"."AVG_BStl"     AS "AVG_T2c_Stl",  
        "t2"."AVG_BBlk"     AS "AVG_T2c_Blk",  
        "t2"."AVG_BPF"      AS "AVG_T2c_PF"  
    FROM "RollingAvg5Current" "t1" JOIN "RollingAvg5Current" "t2" USING("Season", "Genre")
    WHERE "t1"."ATeamID" < "t2"."ATeamID" 
    ORDER BY "ID"  
    """

df_for_prediction = pd.read_sql_query(query, engine)



df_for_prediction


df2 = df_for_prediction.drop(["ID", "T1_ID", "T2_ID"], axis=1, inplace=False)
# Set all locations to neutral
df2["T1_AHomeLoc"] = 0
df2["T1_AAwayLoc"] = 0
df2["T1_ANeutralLoc"] = 1

#Predicting probabilities
probs = log_reg.predict_proba(df2)[:,1]
#Concatenating the probabilities with the games IDS
result_full = pd.concat([df_for_prediction["ID"], pd.DataFrame(probs, columns=["Pred"])], axis=1)

#Now we have the probabilities from hipotetical matches from teams which played in the 2025 season.
#In the competition we are asked to predict all the probabilities from all the possible teams combinations
# We give a probability of 0.5 for matchups where a team didn´t play in the 2025 season

#
result_full.sort_values(by=['ID'], inplace=True)

output_path = "/kaggle/working/"
result_full.to_csv(output_path+'my_submision.csv', index=False)
print("My predicted probabilities:")
display(result_full)


# Load predictions to database
for filename in [ 'my_submision.csv' ]:
    if filename.endswith('.csv') and filename not in ignore_files:
        column_names = []
        data_types = {}
        table_name = "predictions"
        file_path = os.path.join(output_path, filename)
        column_names = ('ID','Pred')
        data_types = ('object','np.float64')  
        # Create a SQL table for each file
        df = pd.read_csv(file_path)
        df.to_sql(table_name, engine, schema=None , if_exists='replace', index=False)
        print(f"Table '{table_name}' created succesfully.")


# Extra part. Predicting the tournament brackets with the probabilities

predicted_probas = result_full


# Predictions of all possible games
query = """
WITH "Matches" AS
(
    SELECT t1."Seed" AS "Seed1", t2."Seed" AS "Seed2", t1."GameSlot", t1."GameRound" 
    FROM "MNCAATourneySeedRoundSlots" t1 JOIN "MNCAATourneySeedRoundSlots" t2 
    ON t1."GameSlot" = t2."GameSlot" AND  t1."GameRound" = t2."GameRound"
    WHERE t1."Seed" < t2."Seed"
    ORDER BY t1."GameRound"
), 
"MNCAATourneySeeds2025"
AS
(
    SELECT "Seed", "TeamID" FROM "MNCAATourneySeeds" WHERE "Season" = 2025
), "PossibleGames" AS
(
SELECT t0."Seed1", t0."Seed2", t0."GameSlot", t0."GameRound", t1."TeamID" AS "TeamID1", t2."TeamID" AS "TeamID2",
    CASE WHEN t1."TeamID" < t2."TeamID" 
    THEN concat_ws('_', '2025'::text, t1."TeamID"::text, t2."TeamID"::text)  
    ELSE concat_ws('_', '2025'::text,  t2."TeamID"::text, t1."TeamID"::text) 
    END "GameID" 
FROM ("Matches" t0 JOIN "MNCAATourneySeeds2025" t1 ON t0."Seed1" =  t1."Seed")  
JOIN  "MNCAATourneySeeds2025" t2 ON t0."Seed2" =  t2."Seed"
)
SELECT "GameRound", mt1."TeamName" AS "Team1",  mt2."TeamName" AS "Team2", "Pred" AS "ProbTeam1WinsOverTeam2" 
From 
("PossibleGames" JOIN "predictions" ON "PossibleGames"."GameID" = "predictions"."ID") g1
JOIN "MTeams" mt1 ON mt1."TeamID" = g1."TeamID1" JOIN "MTeams" mt2 ON mt2."TeamID" = g1."TeamID2" 
ORDER BY "GameRound"
"""
full_game_predictions = pd.read_sql_query(query, engine)
full_game_predictions


# Create predicted brackets

def eliminite_further_games(row, table):
    if row["ProbTeam1WinsOverTeam2"] > .5:
        table = table[((table.Team2 != row["Team2"]) & (table.Team1 != row["Team2"]) & (table.GameRound>row["GameRound"]))
                     |  (table.GameRound<=row["GameRound"])]
    else: 
        table = table[((table.Team1 != row["Team1"]) & (table.Team2 != row["Team1"]) & (table.GameRound>row["GameRound"]))
                     |  (table.GameRound<=row["GameRound"])]
    return table

fgp = full_game_predictions.copy()
s=0
while True:
    row = fgp.iloc[s]
    if row.GameRound ==6:
        break
    table  = fgp 
    fgp = eliminite_further_games(row, table)
    s += 1
    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(fgp)   




