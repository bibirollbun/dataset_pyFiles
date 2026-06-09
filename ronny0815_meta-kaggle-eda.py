import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#ëª¨ë“  ê²ƒë“¤ì�˜ SQL

kernel_sql = duckdb.query("""SELECT * FROM read_parquet('KernelMerged.parquet') LIMIT 10""")
dataset_sql = duckdb.query("""SELECT * FROM read_parquet('dataset_clean_merged(dict).parquet') LIMIT 10""")
competition_sql = duckdb.query("""SELECT * FROM read_parquet('competitions_merged.parquet') LIMIT 10""")
model_sql = duckdb.query("""SELECT * FROM read_parquet('ModelMerged.parquet') LIMIT 10""")
submissions_sql = duckdb.query("""SELECT * FROM read_parquet('Submissions_clean.parquet') LIMIT 10""")
team_sql = duckdb.query("""SELECT * FROM read_parquet('Team_merged.parquet') LIMIT 10""")
tier0_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier0.parquet') LIMIT 10""")
tier1_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier1.parquet') LIMIT 10""")
tier2_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier2.parquet') LIMIT 10""")
tier3_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier3.parquet') LIMIT 10""")
tier4_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier4.parquet') LIMIT 10""")                                                                                                   
tier5_sql = duckdb.query("""SELECT * FROM read_parquet('Users_Merged_tier5.parquet') LIMIT 10""")


tier2 = pl.read_parquet('Users_Merged_tier2.parquet')


tier2.filter(~tier2['OrganizationId_JoinDate'].is_null())


import pandas as pd
import json

def double_json_load(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            # 1ë‹¨ê³„: ë¬¸ì��ì—´ -> JSON ë¬¸ì��ì—´
            inner = json.loads(x)
            # 2ë‹¨ê³„: JSON ë¬¸ì��ì—´ -> dict
            if isinstance(inner, str):
                return json.loads(inner)
            return inner
        except Exception:
            return x
    return x

cols = ["OrganizationId_JoinDate", "OrganizationId_CreationDate", "OrganizationId_Name", "OrganizationId_Industry"]

df_pd = pd.read_parquet("Users_Merged_tier2.parquet")

for col in cols:
    df_pd[col] = df_pd[col].apply(double_json_load)



def double_json_load(list):
    if isinstance(x, dict):
        return list
    if isinstance()


df = tier2



# ë”•ì…”ë„ˆë¦¬ë¥¼ (key, value) ë¦¬ìŠ¤íŠ¸ë¡œ ë³€í™˜
df_pd["join_date_list"] = df_pd["OrganizationId_JoinDate"].apply(
    lambda d: list(d.items()) if isinstance(d, dict) else []
)

# íŒ�ë‹¤ìŠ¤ -> í�´ë�¼ìŠ¤ ë³€í™˜
df_pl = pl.from_pandas(df_pd)

# explode ë¦¬ìŠ¤íŠ¸ ì»¬ëŸ¼
df_pl = df_pl.explode("join_date_list")

# íŠœí”Œì�„ ê°�ê°� ì»¬ëŸ¼ìœ¼ë¡œ ë¶„ë¦¬
df_pl = df_pl.with_columns([
    pl.col("join_date_list").list.get(0).alias("OrganizationId"),
    pl.col("join_date_list").list.get(1).alias("JoinDate")
]).drop("join_date_list")

print(df_pl)


print(df.select(pl.col("OrganizationId_JoinDate")).filter(pl.col("OrganizationId_JoinDate").is_not_null()).head())



# 1) ì•ˆì „í•œ JSON íŒŒì‹± í•¨ìˆ˜ (íŒŒì�´ì�¬ í•¨ìˆ˜)
def safe_json_decode(s):
    try:
        return json.loads(s) if s is not None else None
    except:
        return None

# 2) ì»¬ëŸ¼ ë�°ì�´í„°ë¥¼ íŒŒì�´ì�¬ ë¦¬ìŠ¤íŠ¸ë¡œ êº¼ëƒ„
data = df["OrganizationId_JoinDate"].to_list()

# 3) ë¦¬ìŠ¤íŠ¸ ì•ˆì�˜ JSON ë¬¸ì��ì—´ë“¤ì�„ ë”•ì…”ë„ˆë¦¬ë¡œ ë³€í™˜
decoded = [safe_json_decode(x) for x in data]

# 4) ë”•ì…”ë„ˆë¦¬ë¥¼ (key, value) ë¦¬ìŠ¤íŠ¸ë¡œ ë³€í™˜
tuple_lists = [list(d.items()) if d else [] for d in decoded]

# 5) ë¦¬ìŠ¤íŠ¸ë¥¼ ë‹¤ì‹œ ì‹œë¦¬ì¦ˆë¡œ ë§Œë“¤ì–´ ì›�ë³¸ dfì—� ë¶™ì�„
df = df.with_columns([
    pl.Series(tuple_lists).alias("join_date_list")
])

# 6) explodeí•´ì„œ (key, value) ê°�ê°� ë¶„ë¦¬
df = df.explode("join_date_list").with_columns([
    pl.col("join_date_list").list.get(0).alias("OrganizationId"),
    pl.col("join_date_list").list.get(1).alias("JoinDate")
]).drop("join_date_list")


def json_decode(list):
    try:
        return json.load(s) if s is not None eles None
    except:
        return None
    



print(
    df
    .filter(pl.col("OrganizationId_JoinDate").is_not_null())
    .select(["OrganizationId_JoinDate", "OrganizationId", "JoinDate"])
    .head(10)
)



df.select(['JoinDate', 'OrganizationId']).filter(pl.col('OrganizationId').is_not_null())














submissions_sql.df().columns


tier0_sql


team_sql.df().columns


competition_sql.df().columns


query = """
SELECT *
FROM read_parquet('Users_Merged_tier1.parquet') u
INNER JOIN read_parquet('Submissions_clean.parquet') s
ON u.UserId = s.SubmittedUserId
LEFT JOIN read_parquet('Team_merged.parquet') t
ON u.UserId = t,UserId
LEFT JOIN read_parquet('competitions_merged.parquet') c
ON t.CompetitionId = c.
WHERE u.AchievementType = 'Competitions'
"""

duckdb.query(query)


query = """
SELECT COUNT(*)
FROM read_parquet('Users_Merged_tier1.parquet') u
LEFT JOIN read_parquet('Team_merged.parquet') s
ON u.UserId = s.UserId
WHERE AchievementType = 'Competitions'
"""

duckdb.query(query)


model_sql.df().columns





kernel_sql.df().columns


duckdb.query("SELECT * FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1.parquet')").df().columns


duckdb.query("SELECT * FROM read_parquet('/home/ronny/Downloads/final_project/REAL_EDA/dataset_clean_merged(dict).parquet')").df().columns


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



duckdb.query("""SELECT * FROM read_parquet('ModelMerged.parquet') LIMIT 10 """).df().info()


test = """
COPY (
SELECT 
    k.KernelVersionId,                          -- k.Id_kv AS KernelVersionId
    k.ScriptLanguageId,
    k.KernelVersionAuthorUserId,                -- k.AuthorUserId_kv AS KernelVersionAuthorUserId
    k.KernelVersionCreationDate,                -- k.CreationDate_kv AS KernelVersionCreationDate
    k.VersionNumber,
    k.KernelVersionTotalVotes,                  -- k.TotalVotes_kv AS KernelVersionTotalVotes
    k.IsInternetEnabled,
    k.RunningTimeInMilliseconds,
    k.AcceleratorTypeId,
    k.KernelId,                                 -- k.Id_k AS KernelId
    k.KernelAuthorUserId,                       -- k.AuthorUserId_k AS KernelAuthorUserId
    k.CurrentKernelVersionId,
    k.ForkParentKernelVersionId,
    k.KernelCreationDate,                       -- k.CreationDate_k AS KernelCreationDate
    k.KernelMedal,                              -- k.Medal AS KernelMedal
    k.KernelMedalDate,                          -- k.MedalAwardDate AS KernelMedalDate
    k.KernelTotalComments,                      -- k.TotalComments AS KernelTotalComments
    k.KernelTotalVotes,                         -- k.TotalVotes_k AS KernelTotalVotes
    k.TagId,
    k.AcceleratorName,                          -- k.Label AS AcceleratorName
    k.LanguageName,                             -- k.Name AS LanguageName
    k.DisplayName,
    k.IsNotebook,
    k.SourceCompetitionId,
    k.SourceDatasetVersionId,
    k.SourceKernelVersionId,
    k.SourceModelVariationId,
    k.UserPerVoteDate,
    m.Id_m AS ModelId,
    m.OwnerUserId AS ModelOwnerUserId, 
    m.OwnerOrganizationId ModelOrganizationId,
    m.CreationDate AS ModelCreationDate,
    m.TotalViews AS ModelTotalViews,
    m.TotalDownloads AS ModelTotalDownloads,
    m.TotalKernels AS ModelTotalKernels,
    m.Id_mv AS ModelVariationId,
    m.ModelFramework,
    m.TagId AS ModelTags,
    m.UserPerVoteDate ModelUserPerVoteDate,
    -- ì‚¬ìš©ì�� ì •ë³´ (Users í…Œì�´ë¸”)
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet') k
LEFT JOIN read_parquet('ModelMerged.parquet') m
ON k.SourceModelVariationid = m.Id_mv
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON k.KernelVersionAuthorUserId = u.UserId
) TO 'dashboard.parquet' (FORMAT 'parquet' ,COMPRESSION zstd);

"""

duckdb.query(test)


test = """
COPY (
SELECT 
    -- Kernel ì •ë³´ (LEFT JOIN)
    k.KernelVersionId,
    k.ScriptLanguageId,
    k.KernelVersionAuthorUserId,
    k.KernelVersionCreationDate,
    k.VersionNumber,
    k.KernelVersionTotalVotes,
    k.IsInternetEnabled,
    k.RunningTimeInMilliseconds,
    k.AcceleratorTypeId,
    k.KernelId,
    k.KernelAuthorUserId,
    k.CurrentKernelVersionId,
    k.ForkParentKernelVersionId,
    k.KernelCreationDate,
    k.KernelMedal,
    k.KernelMedalDate,
    k.KernelTotalComments,
    k.KernelTotalVotes,
    k.TagId,
    k.AcceleratorName,
    k.LanguageName,
    k.DisplayName,
    k.IsNotebook,
    k.SourceCompetitionId,
    k.SourceDatasetVersionId,
    k.SourceKernelVersionId,
    k.SourceModelVariationId,
    k.UserPerVoteDate,

    -- ëª¨ë�¸ ì •ë³´ (ê¸°ì¤€ í…Œì�´ë¸”)
    m.Id_m AS ModelId,
    m.OwnerUserId AS ModelOwnerUserId, 
    m.OwnerOrganizationId AS ModelOrganizationId,
    m.CreationDate AS ModelCreationDate,
    m.TotalViews AS ModelTotalViews,
    m.TotalDownloads AS ModelTotalDownloads,
    m.TotalKernels AS ModelTotalKernels,
    m.Id_mv AS ModelVariationId,
    m.ModelFramework,
    m.TagId AS ModelTags,
    m.UserPerVoteDate AS ModelUserPerVoteDate,

    -- ì‚¬ìš©ì�� ì •ë³´ (INNER JOIN)
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry

FROM read_parquet('ModelMerged.parquet') m

LEFT JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet') k
ON LIST_CONTAINS(k.SourceModelVariationId, m.Id_mv)

LEFT JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON k.KernelVersionAuthorUserId = u.UserId

) TO 'Model_with_kernel_user.parquet' (FORMAT 'parquet', COMPRESSION ZSTD);
"""

duckdb.query(test)



#ì–˜ëŠ” í�¬ëŸ¼ì—� ëŒ€í•œê±°

test = """
COPY
(
SELECT 
    f.Id,
    f.Title,
    f.Id_topics,
    f.ForumId,
    f.KernelId,
    f.CreationDate,
    f.LastCommentDate,
    f.Title_topics,
    f.IsSticky,
    f.TotalViews,
    f.Score,
    f.TotalMessages,
    f.TotalReplies,
    f.ActiveDuration,
    f.UnrepliedMessages,
    f.Id_messages,
    f.ForumTopicId,
    f.PostUserId,
    f.PostDate,
    f.ReplyToForumMessageId,
    f.Message,
    f.Medal,
    f.MedalAwardDate,
    f.HasMedal,
    f.ForumMessageId,
    f.FromId_ToId,
    f.ForumMessageId_reacts,
    f.FromUserId,
    f.ReactionType,
    f.ReactionDate,
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry
FROM read_parquet('Forums_merged.parquet') f
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_with2.parquet') u
ON f.PostUserId = u.UserId
) TO 'User_1to2_with_forum.parquet' (FORMAT 'parquet');
"""

duckdb.query(test)


test = """
COPY (
SELECT 
    -- Kernel ì •ë³´ (LEFT JOIN)
    k.KernelVersionId,
    k.ScriptLanguageId,
    k.KernelVersionAuthorUserId,
    k.KernelVersionCreationDate,
    k.VersionNumber,
    k.KernelVersionTotalVotes,
    k.IsInternetEnabled,
    k.RunningTimeInMilliseconds,
    k.AcceleratorTypeId,
    k.KernelId,
    k.KernelAuthorUserId,
    k.CurrentKernelVersionId,
    k.ForkParentKernelVersionId,
    k.KernelCreationDate,
    k.KernelMedal,
    k.KernelMedalDate,
    k.KernelTotalComments,
    k.KernelTotalVotes,
    k.TagId,
    k.AcceleratorName,
    k.LanguageName,
    k.DisplayName,
    k.IsNotebook,
    k.SourceCompetitionId,
    k.SourceDatasetVersionId,
    k.SourceKernelVersionId,
    k.SourceModelVariationId,
    k.UserPerVoteDate,

    -- ëª¨ë�¸ ì •ë³´ (ê¸°ì¤€ í…Œì�´ë¸”)
    m.Id_m AS ModelId,
    m.OwnerUserId AS ModelOwnerUserId, 
    m.OwnerOrganizationId AS ModelOrganizationId,
    m.CreationDate AS ModelCreationDate,
    m.TotalViews AS ModelTotalViews,
    m.TotalDownloads AS ModelTotalDownloads,
    m.TotalKernels AS ModelTotalKernels,
    m.Id_mv AS ModelVariationId,
    m.ModelFramework,
    m.TagId AS ModelTags,
    m.UserPerVoteDate AS ModelUserPerVoteDate,

    -- ì‚¬ìš©ì�� ì •ë³´ (INNER JOIN)
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry

FROM read_parquet('ModelMerged.parquet') m

LEFT JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet') k
ON LIST_CONTAINS(k.SourceModelVariationId, m.Id_mv)

INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON k.KernelVersionAuthorUserId = u.UserId

) TO 'Model_with_kernel_user.parquet' (FORMAT 'parquet', COMPRESSION ZSTD);
"""

duckdb.query(test)



query = """
COPY (
WITH user_dataset_counts AS (
  SELECT
    CreatorUserId_version AS UserId,
    COUNT(DISTINCT DatasetId) AS NumDatasetsContributed
  FROM read_parquet('dataset_clean_merged(dict).parquet')
  GROUP BY CreatorUserId_version
)
SELECT *
FROM user_dataset_counts
) TO 'dataset_aggregation.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(query)


#ì–˜ëŠ” ë�°ì�´í„°ì…‹ì…‹

test = """
COPY
(
SELECT 
d.DatasetVersionId, 
d.DatasetId, 
d.DatasourceVersionId,
d.CreatorUserId_version,
d.CreationDate_version,
d.VersionNumber,
d.Title,
d.CreatorUserId_data, 
d.OwnerUserId, 
d.OwnerOrganizationId,
d.ForumId, 
d.CreationDate_data, 
d.LastActivityDate, 
d.TotalViews,
d.TotalDownloads, 
d.TotalVotes, 
d.TotalKernels, 
d.Medal,
d.MedalAwardDate, 
d.Taglist, 
d.TagCount, 
d.VoteCount, 
d.FirstVoteDate,
d.LastVoteDate, 
d.VoteList, 
d.OwnerType, 
d.MedalAwardTime,
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry
FROM read_parquet('dataset_clean_merged(dict).parquet') d
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON  d.CreatorUserId_version= u.UserId
) TO 'user_1to4_with_dataset.parquet' (FORMAT 'parquet');
"""

duckdb.query(test)


duckdb.query("SELECT * FROM read_parquet('dataset_clean_merged(dict).parquet')").df().columns














df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)


testquery = """
WITH submission_with_user AS (
  SELECT
    *
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId
),
user_competition_counts AS (
  SELECT
    *
  FROM submission_with_user
)
SELECT *
FROM user_competition_counts
"""

duckdb.query(testquery).df().columns


query = """
COPY (
WITH submission_with_user AS (
  SELECT
    s.Id AS SubmissionId,
    s.TeamId,
    t.CompetitionId,
    t.UserId
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId
),
user_competition_counts AS (
  SELECT
    UserId,
    COUNT(DISTINCT CompetitionId) AS NumCompetitionsParticipated
  FROM submission_with_user
  GROUP BY UserId
)
SELECT *
FROM user_competition_counts
) TO 'user_competition_aggregations.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(query)


df = pl.from_arrow(duckdb.query(test).arrow())
df


df.estimated_size()


import polars as pl
df = pl.scan_parquet('User_1to4_with_kernel.parquet')
df.columns


import polars as pl
import json

def safe_parse_dict_column(s: pl.Series) -> pl.Series:
    return pl.Series([
        list(json.loads(val).items()) if val else []
        for val in s
    ])

def parse_multiple_json_dict_columns(file_path: str, columns: list[str]) -> pl.DataFrame:
    df = pl.read_parquet(file_path)

    for col in columns:
        if col not in df.columns:
            continue
        parsed_col = f"{col}_parsed"
        key_col = col.split("_")[0]
        value_col = col.split("_")[1] + "_value"

        parsed_series = safe_parse_dict_column(df[col])
        df = df.with_columns(pl.Series(parsed_col, parsed_series))
        df = df.explode(parsed_col).with_columns([
            pl.col(parsed_col).list.get(0).alias(key_col),
            pl.col(parsed_col).list.get(1).alias(value_col)
        ]).drop([col, parsed_col])

    return df

# ì‹¤í–‰
dict_cols = [
    "OrganizationId_JoinDate",
    "OrganizationId_Name",
    "OrganizationId_CreationDate",
    "OrganizationId_Industry"
]
df_final = parse_multiple_json_dict_columns("User_1to4_with_kernel.parquet", dict_cols)



df_final.columns


# def parse_multiple_json_dict_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
#     def try_parse(s):
#         try:
#             return json.loads(s) if s else {}
#         except Exception:
#             return {}

#     for col in columns:
#         # JSON ë¬¸ì��ì—´ â†’ dict
#         dicts = [try_parse(val) for val in df[col].to_list()]
#         # dict â†’ (key, value) íŠœí”Œ ë¦¬ìŠ¤íŠ¸ (Noneì�€ ë¹ˆ ë¦¬ìŠ¤íŠ¸ ì²˜ë¦¬)
#         tuple_lists = [list(d.items()) if isinstance(d, dict) else [] for d in dicts]
        
#         # null ë°©ì§€: ì „ì²´ Noneì�„ ë¹ˆ ë¦¬ìŠ¤íŠ¸ë¡œ ë°”ê¿”ì¤Œ
#         safe_tuple_lists = [t if t is not None else [] for t in tuple_lists]

#         # Polars Series ìƒ�ì„±
#         tuple_col = f"{col}_tuple_list"
#         df = df.with_columns([
#             pl.Series(name=tuple_col, values=safe_tuple_lists)
#         ])

#         # explode í›„ key/value ë¶„ë¦¬
#         df = df.explode(tuple_col).with_columns([
#             pl.col(tuple_col).list.get(0).alias(f"{col}_key"),
#             pl.col(tuple_col).list.get(1).alias(f"{col}_value")
#         ]).drop([tuple_col, col])

#     return df


dict_cols = ['OrganizationId_JoinDate', 'OrganizationId_Name', 'OrganizationId_CreationDate', 'OrganizationId_Industry']

df = parse_multiple_json_dict_columns(df, dict_cols)


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

from scipy.stats import shapiro

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df


df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)


df.filter(~df['UserPerVoteDate'].is_null())


df.describe()


# ì»¤ë„�ì�„ ì�‘ì„±í•œ ì´� ìœ ì €
df['KernelVersionAuthorUserId'].n_unique()


# ì�‘ì„±ë�œ ì´� ì»¤ë„�ì�˜ ìˆ˜
df['KernelId'].n_unique()


# ì�‘ì„±ë�œ ì´� ì»¤ë„� ë²„ì „ì�˜ ìˆ˜
df['KernelVersionId'].n_unique()


df


# ì»¬ëŸ¼ë³„ ê³ ìœ  ê°’
for i in df.columns:
    print(f'{i} : {df[i].n_unique()}')


# # # ì»¤
# df_final.write_parquet('1to4_kernel_dict_exploded.parquet')


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('1to4_kernel_dict_exploded.parquet', low_memory=True)


df['UserId'].n_unique()


# í‹°ì–´ê°€ 0ì�¸ ì–˜ë“¤ = 112
df.filter(df['Tier']==0)["UserId"].n_unique()


# í‹°ì–´ê°€ 1ì�¸ ì• ë“¤ = 99297
df.filter(df['Tier']==1)['UserId'].n_unique()


# df = parse_multiple_json_dict_columns(df, ['UserPerVoteDate'])


print(f'ì „ì²´ ì‰�ì�… : {df.shape}')
print(f"ì¡°ì§�ì�´ NULLì�¸ ì‰�ì�… : {df.filter(df['OrganizationId'].is_null()).shape}")
print(f"ì¡°ì§�ì�´ NULLì�´ ì•„ë‹Œ ì‰�ì�… : {df.filter(~df['OrganizationId'].is_null()).shape}")


# ìœ ì € ì•„ì�´ë””ë³„ ì»¤ë„�ì•„ì�´ë”” íƒ‘ 10 ì‹œê°�í™”...
top10_users = (
    df
    .filter(~pl.col('KernelId').is_null())
    .group_by('UserName')
    .agg([
        pl.col('KernelId').n_unique().alias('KernelCounts'),  # ìœ ë‹ˆí�¬ ì»¤ë„� ê°œìˆ˜
        pl.col('KernelTotalVotes').sum().alias('TotalVotes')  # ì´� íˆ¬í‘œìˆ˜
    ])
    .sort('KernelCounts', descending=True)
    .limit(10)
)

top10_users_pd = top10_users.to_pandas()

plt.figure(figsize=(12, 8))
sns.barplot(data=top10_users_pd, x="UserName", y="KernelCounts", order=top10_users_pd['UserName'].tolist())
plt.title("UserNameë³„ ìœ ë‹ˆí�¬ ì»¤ë„� ê°œìˆ˜ Top 10")
plt.show()



df.filter(df['UserName']=='startupsci')


top10_votes = (
    df
    .filter(~pl.col('KernelId').is_null())
    .group_by('UserName')
    .agg([
        pl.col('KernelTotalVotes').max().alias('TotalVotes'),
        pl.col('KernelId').n_unique().alias('KernelCounts'),
    ])
    .sort('TotalVotes', descending=True)
    .limit(10)
)

top10_votes_pd = top10_votes.to_pandas()

plt.figure(figsize=(15, 10))
sns.barplot(
    data=top10_votes_pd,
    x="UserName",
    y="TotalVotes",
    order=top10_votes_pd['UserName'].tolist(),
    palette="viridis"
)

for i, row in top10_votes_pd.iterrows():
    plt.text(
        x=i,
        y=row["TotalVotes"] + 1,
        s=f"{row['TotalVotes']:,}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("ê°€ì�¥ ë§�ì�€ íˆ¬í‘œìˆ˜ë¥¼ ë°›ì�€ ìœ ì € Top 10")
plt.ylabel("ì´� íˆ¬í‘œ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



top10_votes


top10_comments = (
    df
    .filter(~pl.col('KernelId').is_null())
    .group_by('UserName')
    .agg([
        pl.col('KernelTotalComments').mean().alias('TotalComments'),
        pl.col('KernelId').n_unique().alias('KernelCounts'),
    ])
    .sort('TotalComments', descending=True)
    .limit(10)
)

top10_comments_pd = top10_comments.to_pandas()

plt.figure(figsize=(15, 10))
sns.barplot(
    data=top10_comments_pd,
    x="UserName",
    y="TotalComments",
    order=top10_comments_pd['UserName'].tolist(),
    palette="crest"
)

for i, row in top10_comments_pd.iterrows():
    plt.text(
        x=i,
        y=row["TotalComments"] + 1,
        s=f"{row['TotalComments']:.1f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'    
        )

plt.title("ê°€ì�¥ ë§�ì�€ ëŒ“ê¸€ì�„ ë°›ì�€ ìœ ì € Top 10 (í�‰ê· )")
plt.ylabel("ì»¤ë„�ë‹¹ í�‰ê·  ëŒ“ê¸€ ìˆ˜")
plt.tight_layout()
plt.show()



top10_views = (
    df
    .filter(~pl.col('KernelId').is_null())
    .group_by('UserName')
    .agg([
        pl.col('KernelTotalComments').sum().alias('TotalViews'),
        pl.col('KernelId').n_unique().alias('KernelCounts'),
    ])
    .sort('TotalViews', descending=True)
    .limit(10)
)

top10_views_pd = top10_views.to_pandas()

plt.figure(figsize=(15, 10))
sns.barplot(
    data=top10_views_pd,
    x="UserName",
    y="TotalViews",
    order=top10_views_pd['UserName'].tolist(),
    palette="mako"
)

for i, row in top10_views_pd.iterrows():
    plt.text(
        x=i,
        y=row["TotalViews"] + 1,
        s=f"{row['TotalViews']:,}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("ê°€ì�¥ ë§�ì�€ ëŒ“ê¸€ìˆ˜ë¥¼ ê¸°ë¡�í•œ ìœ ì € Top 10")
plt.ylabel("ì´� ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# ë‘�ê°œê°€ ê°™ì�€ í–‰ì�´ë„¤ì—¬...
# df.filter(df['KernelTotalViews'] != df['KernelTotalComments'])


import matplotlib.pyplot as plt
import seaborn as sns

# 1. ìœ ì €ë³„ ì»¤ë„� ìˆ˜ ìƒ�ìœ„ 10ëª… ì§‘ê³„
top10_kernel_users = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelId").n_unique().alias("KernelCount"))
    .sort("KernelCount", descending=True)
    .limit(10)
)

# 2. ê°� ìœ ì €ë³„ Votes, Views ì •ë³´ë�„ ì§‘ê³„
top10_votes = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelTotalVotes").sum().alias("KernelTotalVotes"))
)

top10_views = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelTotalComments").sum().alias("KernelTotalComments"))
)

# 3. ê³µí†µ ìœ ì € í•„í„°ë§� ë°� ë³‘í•©
common_user_df = (
    top10_kernel_users
    .join(top10_views, on='UserName', how='inner')
    .join(top10_votes, on='UserName', how='inner')
    .unique()
)

# 4. Pandasë¡œ ë³€í™˜ í›„ melt í•  ë•Œ KernelCountë�„ í�¬í•¨
common_user_pd = common_user_df.to_pandas()

# KernelCountëŠ” ê°’ ë²”ìœ„ê°€ Votes/Viewsì™€ ë‹¤ë¥´ë‹ˆ ë”°ë¡œ scale ë§�ì¶œ ìˆ˜ë�„ ì�ˆì§€ë§Œ,
# ì�¼ë‹¨ ê°™ì�´ í‘œì‹œí•´ë³´ê³  í•„ìš”í•˜ë©´ scale ì¡°ì •í•˜ì„¸ìš”.

melted = common_user_pd.melt(
    id_vars='UserName',
    value_vars=['KernelCount', 'KernelTotalVotes', 'KernelTotalComments'],
    var_name='Metric',
    value_name='Count'
).sort_values('Count', ascending=False)

# 5. ì‹œê°�í™”
plt.figure(figsize=(15, 10))
ax = sns.barplot(data=melted, x="UserName", y="Count", hue="Metric", palette=["#4e79a7", "#f28e2c", "#e15759"])

# ë§‰ëŒ€ ìœ„ì—� ìˆ˜ì¹˜ í‘œì‹œ (ì •í™•í•œ ë§‰ëŒ€ ìœ„ì¹˜ì—� ì°�ê¸°)
for container in ax.containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + melted["Count"].max() * 0.01,
                f"{int(height):,}",
                ha='center',
                va='bottom',
                fontsize=9
            )

plt.title("ìƒ�ìœ„ ì»¤ë„� ìœ ì €ë“¤ì�˜ ì»¤ë„� ìˆ˜, Comments ë°� Votes ë¹„êµ�")
plt.xlabel("UserName")
plt.ylabel("í•©ê³„")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 1. ìœ ì €ë³„ ì»¤ë„� ìˆ˜ ìƒ�ìœ„ 10ëª… ì§‘ê³„
top10_kernel_users = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelId").n_unique().alias("KernelCount"))
    .sort("KernelCount", descending=True)
    .limit(10)
)

# 2. ê°� ìœ ì €ë³„ Votes, Views ì •ë³´ë�„ ì§‘ê³„
top10_votes = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelTotalVotes").mean().alias("KernelAverageVotes"))
)

top10_views = (
    df.filter(~pl.col("KernelId").is_null())
    .group_by("UserName")
    .agg(pl.col("KernelTotalComments").mean().alias("KernelAverageComments"))
)

# 3. ê³µí†µ ìœ ì € í•„í„°ë§� ë°� ë³‘í•©
common_user_df = (
    top10_kernel_users
    .join(top10_views, on='UserName', how='inner')
    .join(top10_votes, on='UserName', how='inner')
    .unique()
)

# 4. Pandasë¡œ ë³€í™˜ í›„ melt í•  ë•Œ KernelCountë�„ í�¬í•¨
common_user_pd = common_user_df.to_pandas()

# KernelCountëŠ” ê°’ ë²”ìœ„ê°€ Votes/Viewsì™€ ë‹¤ë¥´ë‹ˆ ë”°ë¡œ scale ë§�ì¶œ ìˆ˜ë�„ ì�ˆì§€ë§Œ,
# ì�¼ë‹¨ ê°™ì�´ í‘œì‹œí•´ë³´ê³  í•„ìš”í•˜ë©´ scale ì¡°ì •í•˜ì„¸ìš”.

melted = common_user_pd.melt(
    id_vars='UserName',
    value_vars=['KernelCount', 'KernelAverageVotes', 'KernelAverageComments'],
    var_name='Metric',
    value_name='Count'
).sort_values('Count', ascending=False)

# 5. ì‹œê°�í™”
plt.figure(figsize=(15, 10))
ax = sns.barplot(data=melted, x="UserName", y="Count", hue="Metric", palette=["#4e79a7", "#f28e2c", "#e15759"])

# ë§‰ëŒ€ ìœ„ì—� ìˆ˜ì¹˜ í‘œì‹œ (ì •í™•í•œ ë§‰ëŒ€ ìœ„ì¹˜ì—� ì°�ê¸°)
for container in ax.containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + melted["Count"].max() * 0.01,
                f"{int(height):,}",
                ha='center',
                va='bottom',
                fontsize=9
            )

plt.title("ìƒ�ìœ„ ì»¤ë„� ìœ ì €ë“¤ì�˜ ì»¤ë„� ìˆ˜, í�‰ê·  Comments ë°� Votes ë¹„êµ�")
plt.xlabel("UserName")
plt.ylabel("í�‰ê· ")
plt.tight_layout()
plt.show()


# Labelë³„ KernelId ê³ ìœ  ê°œìˆ˜ ì§‘ê³„
label_dist = (
    df
    .filter(~pl.col('AcceleratorName').is_null())
    .group_by('AcceleratorName')
    .agg(pl.col('KernelId').n_unique().alias('Count'))
    .sort('Count', descending=True)
)

# Pandas ë³€í™˜
label_dist_pd = label_dist.to_pandas()

# ì‹œê°�í™”
plt.figure(figsize=(12, 6))
sns.barplot(data=label_dist_pd, x='AcceleratorName', y='Count')

# ìˆ˜ì¹˜ í‘œì‹œ - xì¢Œí‘œëŠ” ë§‰ëŒ€ index ê¸°ì¤€
for idx, row in label_dist_pd.iterrows():
    plt.text(
        x=idx,
        y=row["Count"] + max(label_dist_pd["Count"]) * 0.01,  # ë§‰ëŒ€ ìœ„ 1% ì •ë�„ ë�„ì›€
        s=f"{row['Count']:,}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("ì „ì²´ ì»¤ë„�ì—�ì„œ Accelerator ì‚¬ìš© ë¹ˆë�„")
plt.xlabel("AcceleratorName")
plt.ylabel("ê³ ìœ  Kernel ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 1. ìœ ì €ë³„ ê³ ìœ  ì»¤ë„� ê°œìˆ˜ ì§‘ê³„
top_users = (
    df
    .filter(~pl.col('UserName').is_null())
    .group_by('UserName')
    .agg(pl.col('KernelId').n_unique().alias('UniqueKernelCount'))
    .sort('UniqueKernelCount', descending=True)
    .limit(10)
)

# 2. ìƒ�ìœ„ 10ëª… ìœ ì € ë¦¬ìŠ¤íŠ¸
top_usernames = top_users.select('UserName').to_series().to_list()

# 3. ìƒ�ìœ„ 10ëª… ìœ ì €ë“¤ì�˜ Labelë³„ ì»¤ë„� ì•„ì�´ë”” ê°œìˆ˜ ì§‘ê³„
top_user_labels = (
    df
    .filter(pl.col('UserName').is_in(top_usernames) & ~pl.col('AcceleratorName').is_null())
    .group_by(['UserName', 'AcceleratorName'])
    .agg(pl.col('KernelId').n_unique().alias('UniqueKernelCount'))
    .sort('UniqueKernelCount', descending=True)
)

# 4. Pandas ë³€í™˜ ë°� ì‹œê°�í™”
top_user_labels_pd = top_user_labels.to_pandas()

plt.figure(figsize=(14, 8))
ax = sns.barplot(data=top_user_labels_pd, x='UserName', y='UniqueKernelCount', hue='AcceleratorName')

# yê°’ í…�ìŠ¤íŠ¸ í‘œì‹œ
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2., 
        height + 0.5,  # ë§‰ëŒ€ ìœ„ë¡œ ì•½ê°„ ë�„ì›Œì„œ
        f'{int(height)}', 
        ha='center', 
        va='bottom', 
        fontsize=9,
        fontweight='bold'
    )

plt.title("Top 10 ìœ ì €ì�˜ Accelerator ì»¤ë„� ì‚¬ìš© ë¹ˆë�„")
plt.tight_layout()
plt.show()



# 'KernelMedalAwardDate' ì»¬ëŸ¼ì—�ì„œ 1980-01-01 ì œì™¸ í•„í„°ë§�
filtered_df = df.filter(
    (pl.col('KernelMedalDate').is_not_null()) &
    (pl.col('KernelMedalDate') != pl.datetime(1980, 1, 1))
)

# ì—°ë�„ ì¶”ì¶œ ë°� ê·¸ë£¹í™”
medal_by_year = (
    filtered_df
    .with_columns(pl.col('KernelMedalDate').dt.year().alias('Year'))
    .group_by('Year')
    .agg(pl.count('KernelMedalDate').alias('MedalCount'))
    .sort('Year')
)

# Pandas ë³€í™˜
medal_by_year_pd = medal_by_year.to_pandas()

# ì—°ë�„ë³„ ë©”ë‹¬ ìˆ˜ ì‹œê°�í™”
plt.figure(figsize=(12, 6))
sns.lineplot(data=medal_by_year_pd, x='Year', y='MedalCount', marker='o')
plt.title('ì—°ë�„ë³„ ë©”ë‹¬ ìˆ˜')
plt.xlabel('ì—°ë�„')
plt.ylabel('ë©”ë‹¬ ìˆ˜')
plt.tight_layout()
plt.show()



def count_unique_ids_by_year(df, date_col, id_col):
    filtered = df.filter(
        (pl.col(date_col).is_not_null()) &
        (pl.col(date_col) != pl.datetime(1980, 1, 1)) &
        (pl.col(id_col).is_not_null())
    )
    yearly_count = (
        filtered
        .with_columns(pl.col(date_col).dt.year().alias('Year'))
        .group_by('Year')
        .agg(pl.col(id_col).n_unique().alias('UniqueCount'))
        .sort('Year')
    )
    return yearly_count


# ê°�ê°� ì§‘ê³„
kernel_versions_yearly = count_unique_ids_by_year(df, 'KernelVersionCreationDate', 'KernelVersionId')
creation_k_yearly = count_unique_ids_by_year(df, 'KernelCreationDate', 'KernelId')

# pandas ë³€í™˜
kernel_versions_yearly_pd = kernel_versions_yearly.to_pandas()
creation_k_yearly_pd = creation_k_yearly.to_pandas()

# ì²« ë²ˆì§¸ ê·¸ë�˜í”„
plt.figure(figsize=(14,6))
sns.lineplot(data=kernel_versions_yearly_pd, x='Year', y='UniqueCount', marker='o')
plt.title('KernelVersionCreationDate ì—°ë�„ë³„ ìƒ�ì„± ê±´ìˆ˜ (1980-01-01 ì œì™¸)')
plt.xlabel('ì—°ë�„')
plt.ylabel('ê±´ìˆ˜')
plt.tight_layout()
plt.show()

# ë‘� ë²ˆì§¸ ê·¸ë�˜í”„
plt.figure(figsize=(14,6))
sns.lineplot(data=creation_k_yearly_pd, x='Year', y='UniqueCount', marker='o')
plt.title('KernelCreationDate ì—°ë�„ë³„ ìƒ�ì„± ê±´ìˆ˜ (1980-01-01 ì œì™¸)')
plt.xlabel('ì—°ë�„')
plt.ylabel('ê±´ìˆ˜')
plt.tight_layout()
plt.show()


def count_unique_ids_by_year_tier(df, date_col, id_col):
    # ìœ íš¨í•œ ê°’ë§Œ í•„í„°ë§�
    filtered = df.filter(
        (pl.col(date_col).is_not_null()) &
        (pl.col(date_col) != pl.datetime(1980, 1, 1)) &
        (pl.col(id_col).is_not_null()) &
        (pl.col("PerformanceTier").is_not_null())
    )

    # ì—°ë�„ ì¶”ì¶œ + ê·¸ë£¹ë°”ì�´(ì—°ë�„, í‹°ì–´)
    yearly_count = (
        filtered
        .with_columns(pl.col(date_col).dt.year().alias('Year'))
        .group_by(['PerformanceTier', 'Year'])
        .agg(pl.col(id_col).n_unique().alias('UniqueCount'))
        .sort(['PerformanceTier', 'Year'])
    )
    return yearly_count


# ê°�ê°� ì§‘ê³„
kernel_versions_by_tier = count_unique_ids_by_year_tier(df, 'KernelVersionCreationDate', 'KernelVersionId')
creation_k_by_tier = count_unique_ids_by_year_tier(df, 'KernelCreationDate', 'KernelId')

# Pandas ë³€í™˜
kver_pd = kernel_versions_by_tier.to_pandas()
kcrt_pd = creation_k_by_tier.to_pandas()

# ---------- KernelVersionCreationDate ì‹œê°�í™” ----------
plt.figure(figsize=(14, 6))
sns.lineplot(data=kver_pd, x="Year", y="UniqueCount", hue="PerformanceTier", marker='o')
plt.title('KernelVersionCreationDate ì—°ë�„ë³„ ìƒ�ì„± ê±´ìˆ˜ (PerformanceTierë³„)')
plt.xlabel("ì—°ë�„")
plt.ylabel("ê±´ìˆ˜")
plt.tight_layout()
plt.legend(title="PerformanceTier")
plt.grid(True)
plt.show()

# ---------- KernelCreationDate ì‹œê°�í™” ----------
plt.figure(figsize=(14, 6))
sns.lineplot(data=kcrt_pd, x="Year", y="UniqueCount", hue="PerformanceTier", marker='o')
plt.title('KernelCreationDate ì—°ë�„ë³„ ìƒ�ì„± ê±´ìˆ˜ (PerformanceTierë³„)')
plt.xlabel("ì—°ë�„")
plt.ylabel("ê±´ìˆ˜")
plt.tight_layout()
plt.legend(title="PerformanceTier")
plt.grid(True)
plt.show()



df


counts = df.select('DisplayName').to_pandas()['DisplayName'].value_counts()
total = counts.sum()

# 1% ì�´ìƒ�ì�¸ í•­ëª©ë§Œ í•„í„°ë§�
counts_filtered = counts[counts / total >= 0.01]

plt.figure(figsize=(6,6))
plt.pie(counts_filtered, labels=counts_filtered.index, autopct='%1.1f%%', startangle=90,
        counterclock=False)
plt.title('DisplayName ë¶„í�¬ ë¹„ìœ¨ (1% ì�´ìƒ�)')
plt.show()



# 1980-01-01 ì œê±° ë°� ì—°ë�„ ì¶”ì¶œ
df_filtered = (
    df
    .filter(pl.col("KernelVersionCreationDate").dt.year() > 1980)
    .with_columns(
        pl.col("KernelVersionCreationDate").dt.year().alias("Year")
    )
    .filter(~pl.col("DisplayName").is_null())
)

# ì§‘ê³„: ì—°ë�„ë³„ DisplayName ì¹´ìš´íŠ¸
df_grouped = (
    df_filtered
    .group_by(["Year", "DisplayName"])
    .agg(pl.count().alias("Count"))
    .sort(["Year", "DisplayName"])
)

# Pandas ë³€í™˜ í›„ pivot
df_pd = df_grouped.to_pandas()
df_pivot = df_pd.pivot(index="Year", columns="DisplayName", values="Count").fillna(0)

# ë¹„ìœ¨ ê³„ì‚°
df_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0)

# ì‹œê°�í™” (stacked area chart)
plt.figure(figsize=(14, 7))
df_pct.plot(kind='area', stacked=True, figsize=(14, 7), cmap="tab20")

plt.title("ì—°ë�„ë³„ DisplayName ì �ìœ ìœ¨ ë³€í™”")
plt.xlabel("ì—°ë�„")
plt.xticks(rotation=45)

plt.ylabel("ì �ìœ ìœ¨")
plt.legend(title="DisplayName", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import pandas as pd

# -1 ì œì™¸
df_filtered = df.filter(pl.col("RunningTimeInMilliseconds") != -1)

# êµ¬ê°„í™” (ì„¸ë¶„í™” í�¬í•¨)
df_categorized = df_filtered.with_columns(
    pl.when(pl.col("RunningTimeInMilliseconds") < 1000).then(pl.lit("0-1ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 5000).then(pl.lit("1-5ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 10000).then(pl.lit("5-10ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 30000).then(pl.lit("10-30ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 60000).then(pl.lit("30ì´ˆ-1ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 120000).then(pl.lit("1-2ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 300000).then(pl.lit("2-5ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 600000).then(pl.lit("5-10ë¶„"))
    # 10ë¶„ ì�´ìƒ� ì„¸ë¶„í™” (600,000 ms = 10ë¶„)
    .when(pl.col("RunningTimeInMilliseconds") < 1800000).then(pl.lit("10ë¶„-30ë¶„"))       # 10~30ë¶„
    .when(pl.col("RunningTimeInMilliseconds") < 3600000).then(pl.lit("30ë¶„-1ì‹œê°„"))      # 30ë¶„~1ì‹œê°„
    .when(pl.col("RunningTimeInMilliseconds") < 10800000).then(pl.lit("1ì‹œê°„-3ì‹œê°„"))    # 1~3ì‹œê°„
    .otherwise(pl.lit("3ì‹œê°„ ì�´ìƒ�"))                                                    # 3ì‹œê°„ ì�´ìƒ�
    .alias("RunningTimeCategory")
)

# ì‹œê°„ ìˆœì„œëŒ€ë¡œ ì¹´í…Œê³ ë¦¬ ì§€ì •
category_order = [
    "0-1ì´ˆ",
    "1-5ì´ˆ",
    "5-10ì´ˆ",
    "10-30ì´ˆ",
    "30ì´ˆ-1ë¶„",
    "1-2ë¶„",
    "2-5ë¶„",
    "5-10ë¶„",
    "10ë¶„-30ë¶„",
    "30ë¶„-1ì‹œê°„",
    "1ì‹œê°„-3ì‹œê°„",
    "3ì‹œê°„ ì�´ìƒ�"
]

# ì§‘ê³„
category_counts = (
    df_categorized
    .group_by("RunningTimeCategory")
    .agg(pl.count().alias("Count"))
)

category_counts_pd = category_counts.to_pandas()
category_counts_pd['RunningTimeCategory'] = pd.Categorical(
    category_counts_pd['RunningTimeCategory'],
    categories=category_order,
    ordered=True
)
category_counts_pd = category_counts_pd.sort_values('RunningTimeCategory').reset_index(drop=True)

max_count = category_counts_pd["Count"].max()
offset = max_count * 0.03

plt.figure(figsize=(14, 7))
sns.barplot(data=category_counts_pd, 
            x="RunningTimeCategory", 
            y="Count", 
            order=category_order)

for i, row in category_counts_pd.iterrows():
    plt.text(
        x=i,
        y=row["Count"] + offset,
        s=f"{row['Count']:,}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("RunningTimeInMilliseconds êµ¬ê°„ë³„ ë¶„í�¬ (ì„¸ë¶„í™”ë�œ ì‹œê°„ êµ¬ê°„)")
plt.xlabel("êµ¬ê°„")
plt.ylabel("ê°œìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -1 ì œì™¸
df_filtered = df.filter(pl.col("RunningTimeInMilliseconds") != -1)

# â–¶ ì»¤ë„� ID ë‹¨ìœ„ë¡œ ëŒ€í‘œ ë ˆì½”ë“œ ì„ íƒ� (ê°€ì�¥ ê¸´ ì‹¤í–‰ ì‹œê°„ ê¸°ì¤€)
df_kernel_level = (
    df_filtered
    .sort("RunningTimeInMilliseconds", descending=True)
    .unique(subset="KernelId", keep="first")
)

# â–¶ êµ¬ê°„í™” (ì„¸ë¶„í™” í�¬í•¨)
df_categorized = df_kernel_level.with_columns(
    pl.when(pl.col("RunningTimeInMilliseconds") < 1000).then(pl.lit("0-1ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 5000).then(pl.lit("1-5ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 10000).then(pl.lit("5-10ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 30000).then(pl.lit("10-30ì´ˆ"))
    .when(pl.col("RunningTimeInMilliseconds") < 60000).then(pl.lit("30ì´ˆ-1ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 120000).then(pl.lit("1-2ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 300000).then(pl.lit("2-5ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 600000).then(pl.lit("5-10ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 1800000).then(pl.lit("10ë¶„-30ë¶„"))
    .when(pl.col("RunningTimeInMilliseconds") < 3600000).then(pl.lit("30ë¶„-1ì‹œê°„"))
    .when(pl.col("RunningTimeInMilliseconds") < 10800000).then(pl.lit("1ì‹œê°„-3ì‹œê°„"))
    .otherwise(pl.lit("3ì‹œê°„ ì�´ìƒ�"))
    .alias("RunningTimeCategory")
)

# â–¶ ë©”ë‹¬ ì—¬ë¶€ í”Œë�˜ê·¸ ì¶”ê°€
df_categorized = df_categorized.with_columns(
    (pl.col("KernelMedal") != 0).cast(pl.Int32).alias("MedalFlag")
)

# â–¶ êµ¬ê°„ë³„ ì§‘ê³„
agg_df = (
    df_categorized
    .group_by("RunningTimeCategory")
    .agg([
        pl.count().alias("TotalKernelCount"),
        pl.sum("MedalFlag").alias("MedalKernelCount")
    ])
    .with_columns(
        (pl.col("MedalKernelCount") / pl.col("TotalKernelCount") * 100).alias("MedalPercent")
    )
)

# â–¶ Pandas ë³€í™˜ ë°� ì •ë ¬
category_order = [
    "0-1ì´ˆ",
    "1-5ì´ˆ",
    "5-10ì´ˆ",
    "10-30ì´ˆ",
    "30ì´ˆ-1ë¶„",
    "1-2ë¶„",
    "2-5ë¶„",
    "5-10ë¶„",
    "10ë¶„-30ë¶„",
    "30ë¶„-1ì‹œê°„",
    "1ì‹œê°„-3ì‹œê°„",
    "3ì‹œê°„ ì�´ìƒ�"
]

agg_pd = agg_df.to_pandas()
agg_pd["RunningTimeCategory"] = pd.Categorical(agg_pd["RunningTimeCategory"], categories=category_order, ordered=True)
agg_pd = agg_pd.sort_values("RunningTimeCategory").reset_index(drop=True)

# â–¶ ì‹œê°�í™”
fig, ax1 = plt.subplots(figsize=(14,7))

# ë°” ì°¨íŠ¸
sns.barplot(
    x="RunningTimeCategory",
    y="TotalKernelCount",
    data=agg_pd,
    order=category_order,
    ax=ax1,
    color='skyblue'
)
ax1.set_xlabel("RunningTimeCategory (êµ¬ê°„)")
ax1.set_ylabel("ì»¤ë„� ìˆ˜", color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# ë‘� ë²ˆì§¸ yì¶•: í�¼ì„¼íŠ¸
ax2 = ax1.twinx()
sns.lineplot(
    x="RunningTimeCategory",
    y="MedalPercent",
    data=agg_pd,
    ax=ax2,
    color='orange',
    marker='o',
    linewidth=2
)
ax2.set_ylabel("ë©”ë‹¬ ë°›ì�€ ì»¤ë„� ë¹„ìœ¨ (%)", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(0, 100)

# ë°” ìœ„ì—� í…�ìŠ¤íŠ¸ í‘œì‹œ
for i, row in agg_pd.iterrows():
    ax1.text(
        i,
        row["TotalKernelCount"] + row["TotalKernelCount"] * 0.03,
        f"{row['TotalKernelCount']:,}",
        ha='center',
        va='bottom',
        color='blue',
        fontsize=10,
        fontweight='bold'
    )

plt.title("RunningTimeInMilliseconds êµ¬ê°„ë³„ ì»¤ë„� ìˆ˜ ë°� ë©”ë‹¬ ë¹„ìœ¨ (ì¤‘ë³µ ì œê±°)")
plt.tight_layout()
plt.show()



import polars as pl

# ìˆ˜ì¹˜í˜• íƒ€ì�… ì •ì�˜
numeric_types = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
}

# ì¡°ê±´ì—� ë§�ëŠ” ì»¬ëŸ¼ëª… í•„í„°ë§�
numeric_cols = [
    col for col, dtype in df.schema.items()
    if dtype in numeric_types and "Id" not in col
]

corr_matrix = [
    [df.select(pl.corr(c1, c2)).item() for c2 in numeric_cols]
    for c1 in numeric_cols
]

# ë”•ì…”ë„ˆë¦¬ í˜•íƒœë¡œ ë³€í™˜
corr_dict = {col: [row[i] for row in corr_matrix] for i, col in enumerate(numeric_cols)}

# DataFrame ìƒ�ì„± í›„ index ì»¬ëŸ¼ ì¶”ê°€
corr_df = pl.DataFrame(corr_dict).with_columns(pl.Series("index", numeric_cols)).select(["index"] + numeric_cols)

print(corr_df)



# polars DataFrame -> pandas DataFrame ë³€í™˜
corr_pd = corr_df.to_pandas().set_index('index')

# ìƒ�ì‚¼ê°� í–‰ë ¬ mask ìƒ�ì„± (Trueë©´ ì•ˆ ë³´ì�„)
mask = np.triu(np.ones_like(corr_pd, dtype=bool))

plt.figure(figsize=(20, 16))
sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, square=True, cbar_kws={"shrink": 0.8})

plt.title("ìƒ�ê´€ê´€ê³„ í�ˆíŠ¸ë§µ (í•˜ì‚¼ê°�)")
plt.tight_layout()
plt.show()


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ìœ ì €ë³„ ê°€ì�…ì�¼(ìµœì†Œ RegisterDate), ë§ˆì§€ë§‰ í™œë�™ì�¼(ìµœëŒ€ CreationDate_KernelVersions) ì§‘ê³„
user_activity = (
    df
    .filter(
        (~pl.col("KernelVersionCreationDate").is_null()) &
        (~pl.col("RegisterDate").is_null())
    )
    .group_by("UserId")  # ìœ ì € ê³ ìœ  ê¸°ì¤€
    .agg([
        pl.col("RegisterDate").min().alias("RegisterDate_min"),
        pl.col("KernelVersionCreationDate").max().alias("LastActivityDate_max"),
    ])
)

# 2. í™œë�™ê¸°ê°„(ì�¼ìˆ˜) ê³„ì‚° (max CreationDate - min RegisterDate)
user_activity = user_activity.with_columns(
    (
        (user_activity["LastActivityDate_max"].cast(pl.Int64)) -
        (user_activity["RegisterDate_min"].cast(pl.Int64))
    ).alias("duration_ns")
).with_columns(
    (pl.col("duration_ns") // 1_000_000_000 // 60 // 60 // 24).alias("ActiveDays")  # ns -> ì�¼ ë‹¨ìœ„ ë³€í™˜
).drop("duration_ns")

# 3. ì�Œìˆ˜ í˜¹ì�€ 0ì�¼ ì�´í•˜ í•„í„°ë§� (ì�´ìƒ�ì¹˜ ì œê±°)
user_activity = user_activity.filter(pl.col("ActiveDays") > 0)

# 4. í™œë�™ê¸°ê°„ êµ¬ê°„í™” (ì�„ì�˜ êµ¬ê°„, í•„ìš”ì‹œ ì¡°ì • ê°€ëŠ¥)
user_activity = user_activity.with_columns(
    pl.when(pl.col("ActiveDays") <= 1).then(pl.lit("1ì�¼ ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 7).then(pl.lit("1ì£¼ì�¼ ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 30).then(pl.lit("1ê°œì›” ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 90).then(pl.lit("3ê°œì›” ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 180).then(pl.lit("6ê°œì›” ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 365).then(pl.lit("1ë…„ ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 365*3).then(pl.lit("1-3ë…„ ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 365*5).then(pl.lit("3-5ë…„ ì�´í•˜"))
    .when(pl.col("ActiveDays") <= 365*10).then(pl.lit("5-10ë…„ ì�´í•˜"))
    .otherwise(pl.lit("10ë…„ ì´ˆê³¼"))
    .alias("ActiveDaysCategory")
)

# 5. ì¹´í…Œê³ ë¦¬ ì •ë ¬ ìˆœì„œ ì •ì�˜
category_order = [
    "1ì�¼ ì�´í•˜",
    "1ì£¼ì�¼ ì�´í•˜",
    "1ê°œì›” ì�´í•˜",
    "3ê°œì›” ì�´í•˜",
    "6ê°œì›” ì�´í•˜",
    "1ë…„ ì�´í•˜",
    "1-3ë…„ ì�´í•˜",
    "3-5ë…„ ì�´í•˜",
    "5-10ë…„ ì�´í•˜",
    "10ë…„ ì´ˆê³¼"
]

# 6. ì§‘ê³„ ë°� Pandas ë³€í™˜
category_counts = (
    user_activity
    .group_by("ActiveDaysCategory")
    .agg(pl.count().alias("UserCount"))
    .to_pandas()
)

category_counts["ActiveDaysCategory"] = pd.Categorical(category_counts["ActiveDaysCategory"], categories=category_order, ordered=True)
category_counts = category_counts.sort_values("ActiveDaysCategory")

# 7. ì‹œê°�í™”
plt.figure(figsize=(12, 9))
ax = sns.barplot(data=category_counts, x="ActiveDaysCategory", y="UserCount", order=category_order)

max_count = category_counts["UserCount"].max()
offset = max_count * 0.02

# x ì¢Œí‘œëŠ” category_order ê¸°ì¤€ ìœ„ì¹˜
for idx, cat in enumerate(category_order):
    val = category_counts.loc[category_counts["ActiveDaysCategory"] == cat, "UserCount"]
    if not val.empty:
        y = val.values[0]
        ax.text(
            idx,
            y + offset,
            f"{y:,}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

plt.title("ìœ ì €ë³„ ê°€ì�…ì�¼ë¶€í„° ë§ˆì§€ë§‰ ì»¤ë„� ìƒ�ì„±ì�¼ê¹Œì§€ í™œë�™ ê¸°ê°„ ë¶„í�¬")
plt.xlabel("í™œë�™ ê¸°ê°„ êµ¬ê°„")
plt.ylabel("ìœ ì € ìˆ˜")
plt.tight_layout()
plt.show()


df.describe()


user_activity.filter(user_activity['ActiveDaysCategory']=='10ë…„ ì´ˆê³¼')


import polars as pl
import pandas as pd
import json

def explode_json_column_safe(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    # 1. pandasë¡œ ë³€í™˜
    pdf = df.to_pandas()
    
    # 2. json ë¬¸ì��ì—´ íŒŒì‹± & explode
    pdf[col_name] = pdf[col_name].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
    pdf = pdf.explode(col_name).reset_index(drop=True)
    
    # 3. dict ì»¬ëŸ¼ì�„ ì—¬ëŸ¬ ì»¬ëŸ¼ìœ¼ë¡œ ë¶„ë¦¬
    json_df = pd.json_normalize(pdf[col_name])
    
    # 4. ì›�ë³¸ dfì™€ í•©ì¹˜ê¸°
    pdf = pd.concat([pdf.drop(columns=[col_name]), json_df], axis=1)
    
    # 5. ë‹¤ì‹œ polarsë¡œ ë³€í™˜ 
    return pl.from_pandas(pdf)



explode_json_column_safe(df, 'UserId_VoteDate')


# df.write_parquet('1to2_kernel_userid_votedate_exploded.parquet')


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)
df


df_preexplode = df.filter(df['AchievementType'] == 'Scripts').select('KernelId', 'UserName', 'KernelVersionId', 'FollowingUserId', 'KernelCreationDate', 'KernelTotalVotes', 'KernelTotalComments')


del df
df_exploded = df_preexplode.explode('FollowingUserId')


df_exploded.write_parquet('1to4_kernel_list_following_exploded_only.parquet')


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df_exploded = pl.read_parquet('1to4_kernel_list_following_exploded_only.parquet', low_memory=True)


df_exploded


# 1.1 íŒ”ë¡œì�‰ ìˆ˜ ì§‘ê³„
top10_following = (
    df_exploded
    .filter(~pl.col("FollowingUserId").is_null())
    .group_by("UserName")
    .agg(pl.col("FollowingUserId").n_unique().alias("FollowingCount"))
    .sort("FollowingCount", descending=True)
    .limit(10)
    .to_pandas()
)

# 1.2 ì‹œê°�í™”
plt.figure(figsize=(12, 8))
sns.barplot(data=top10_following, x="UserName", y="FollowingCount", palette="Blues_d")

for i, row in top10_following.iterrows():
    plt.text(
        x=i,
        y=row["FollowingCount"] + 1,  # ë§‰ëŒ€ ìœ„ ì‚´ì§� ë�„ì›€
        s=f"{row['FollowingCount']:,}",  # ì‰¼í‘œ í�¬í•¨ ìˆ«ì�� í�¬ë§·
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )


plt.title("íŒ”ë¡œì�‰ ìœ ì € ìˆ˜ê°€ ë§�ì�€ ìƒ�ìœ„ 10ëª…")
plt.xlabel("UserName")
plt.ylabel("íŒ”ë¡œì�‰ ìˆ˜")
plt.tight_layout()
plt.show()



# 2.1 ìœ ì €ë³„ íŒ”ë¡œì�‰ ìˆ˜, ì´� ì¡°íšŒìˆ˜, ì´� íˆ¬í‘œìˆ˜ ì§‘ê³„
user_stats = (
    df_exploded
    .filter(~pl.col("FollowingUserId").is_null())
    .group_by("UserName")
    .agg([
        pl.count("FollowingUserId").alias("FollowingCount"),
        pl.sum("KernelTotalComments").alias("KernelTotalComments"),
        pl.sum("KernelTotalVotes").alias("TotalVotes")
    ])
    .filter(pl.col("FollowingCount") > 0)  # 0ìœ¼ë¡œ ë‚˜ëˆ„ê¸° ë°©ì§€
    .with_columns([
        (pl.col("KernelTotalComments") / pl.col("FollowingCount")).alias("CommentsPerFollowing"),
        (pl.col("TotalVotes") / pl.col("FollowingCount")).alias("VotesPerFollowing")
    ])
)

# 2.2 ìƒ�ìœ„ 10ëª… ì¶”ì¶œ (íŒ”ë¡œì�‰ ìˆ˜ ê¸°ì¤€)
top10_following = (
    user_stats
    .sort("FollowingCount", descending=True)
    .limit(10)
    .to_pandas()
)

# 2.3 ì‹œê°�í™” (ì¡°íšŒìˆ˜, íˆ¬í‘œìˆ˜ ë¹„ìœ¨ í•¨ê»˜)
melted = top10_following[["UserName", "CommentsPerFollowing", "VotesPerFollowing"]].melt(
    id_vars="UserName",
    var_name="Metric",
    value_name="Ratio"
)

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=melted,
    x="UserName",
    y="Ratio",
    hue="Metric",
    palette="Set2",
    order=top10_following["UserName"].tolist()  # â†� íŒ”ë¡œì�‰ ë§�ì�€ ìˆœì„œ ë°˜ì˜�
)

# xì¶• ë ˆì�´ë¸” â†” ì�¸ë�±ìŠ¤ ë§¤í•‘
xticks = ax.get_xticks()
xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
x_map = {name: i for i, name in enumerate(top10_following["UserName"])}

# ìˆ˜ì¹˜ í…�ìŠ¤íŠ¸ ì¶œë ¥
for _, row in melted.iterrows():
    x = x_map[row["UserName"]]
    offset = -0.2 if row["Metric"] == "CommentsPerFollowing" else 0.2
    ax.text(
        x + offset,
        row["Ratio"] + (row["Ratio"] * 0.02),  # ë§‰ëŒ€ ìœ„ 2% ì •ë�„ ë�„ì›€
        f"{row['Ratio']:.1f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("íŒ”ë¡œì�‰ ìˆ˜ ìƒ�ìœ„ 10ëª… ê¸°ì¤€ ëŒ“ê¸€ ìˆ˜ / íˆ¬í‘œìˆ˜ ë¹„ìœ¨")
plt.xlabel("UserName")
plt.ylabel("ë¹„ìœ¨")
plt.tight_layout()
plt.show()



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('1to4_kernel_dict_exploded.parquet')

df


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# 1. Scriptsë§Œ í•„í„°ë§�
scripts_df = (
    df
    .filter(pl.col("AchievementType") == "Scripts")
    .filter((~pl.col("TierAchievementDate").is_null()) & (~pl.col("Points").is_null()) & (pl.col('TierAchievementDate')!= pl.datetime(1980, 1, 1)))
)

# 2. TierAchievementDateë¥¼ ì›” ë‹¨ìœ„ë¡œ ë³€í™˜ (ì˜ˆ: 2024-06-01)
scripts_df = scripts_df.with_columns(
    pl.col("TierAchievementDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
)

# 3. ì›”ë³„ Points í•©ê³„ + ìœ ì € ìˆ˜
monthly_stats = (
    scripts_df
    .group_by("Month")
    .agg([
        pl.col("Points").sum().alias("TotalPoints"),
        pl.col("UserId").n_unique().alias("UserCount")  # ë˜�ëŠ” "UserName"
    ])
    .sort("Month")
    .to_pandas()
)

# 4. ì‹œê°�í™” (TotalPoints)
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel("ì›”")
ax1.set_ylabel("ì´� Points", color=color)
ax1.plot(monthly_stats["Month"], monthly_stats["TotalPoints"], marker='o', linewidth=2, color=color, label="ì´� Points")
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

plt.title("ì›”ë³„ Points ì´�í•©")
fig.tight_layout()
plt.show()



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# 1. Scripts + ìœ íš¨ ê°’ í•„í„°ë§�
scripts_df = (
    df
    .filter(pl.col("AchievementType") == "Scripts")
    .filter(
        (~pl.col("TierAchievementDate").is_null()) &
        (pl.col("TierAchievementDate") != pl.datetime(1980, 1, 1)) &
        (~pl.col("Tier").is_null()) &
        (pl.col('Tier')!=0)
    )
    .with_columns(
        pl.col("TierAchievementDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
    )
)

# 2. ì›” + í‹°ì–´ë³„ ìœ ì € ìˆ˜ ì§‘ê³„
monthly_tier_counts = (
    scripts_df
    .group_by(["Month", "Tier"])
    .agg(pl.col("UserId").n_unique().alias("UserCount"))
    .sort(["Month", "Tier"])
)

# 3. Pandas ë³€í™˜ + Pivot (Month x Tier â†’ UserCount)
tier_pivot = monthly_tier_counts.to_pandas().pivot(
    index="Month", columns="Tier", values="UserCount"
).fillna(0).sort_index()

# 4. ëˆ„ì �í•© (cumsum)
tier_cumsum = tier_pivot.cumsum()

# 5. ì‹œê°�í™”: Stacked Area Chart
plt.figure(figsize=(14, 6))
tier_cumsum.plot.area(colormap="Set2", alpha=0.8)
plt.title("Scripts ì—…ì � - í‹°ì–´ë³„ ëˆ„ì � ìœ ì € ìˆ˜ ì¶”ì�´")
plt.xlabel("ì›”")
plt.ylabel("ëˆ„ì � ìœ ì € ìˆ˜")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Tier", loc="upper left")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_valid = df.filter(
    (pl.col("VersionNumber").is_not_null()) &
    (pl.col("VersionNumber") > 0) &
    (pl.col("KernelVersionTotalVotes").is_not_null())
).select([
    pl.col("KernelId"),
    pl.col("VersionNumber"),
    pl.col("KernelVersionTotalVotes")
])

# KernelId ë‹¨ìœ„ë¡œ ìµœëŒ€ ë²„ì „, ì´� íˆ¬í‘œìˆ˜ ì§‘ê³„
agg = (
    df_valid.group_by("KernelId")
    .agg([
        pl.col("VersionNumber").max().alias("MaxVersionNumber"),
        pl.col("KernelVersionTotalVotes").sum().alias("TotalVotes")
    ])
    .to_pandas()
)

# ì‹œê°�í™”
plt.figure(figsize=(10, 6))
sns.scatterplot(data=agg, x="MaxVersionNumber", y="TotalVotes", alpha=0.4)
# plt.xscale("log")
plt.yscale("log")
plt.xlabel("ìµœì¢… VersionNumber")
plt.ylabel("Total Votes (log)")
plt.title("ì»¤ë„� ë²„ì „ ìˆ˜ vs ì´� íˆ¬í‘œ ìˆ˜ (ë¡œê·¸ ìŠ¤ì¼€ì�¼)")
# plt.grid(True)
plt.tight_layout()
plt.show()



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)
df


df_preexplode = df.select('KernelId', 'UserName', 'KernelVersionId', 'SourceCompetitionId', 'KernelVersionCreationDate', 'KernelTotalVotes', 'KernelTotalComments', 'SourceDatasetVersionId', 'SourceKernelVersionId', 'SourceModelVariationId')


df_explode_SourceDatasetVersionId = df_preexplode.explode('SourceDatasetVersionId')
df_explode_SourceKernelVersionId = df_preexplode.explode('SourceKernelVersionId')
df_explode_SourceModelVariationId = df_preexplode.explode('SourceModelVariationId')


import polars as pl
import matplotlib.pyplot as plt

# 1. ìœ íš¨í•œ ë�°ì�´í„° í•„í„°ë§�: ë‚ ì§œì™€ SourceDatasetVersionIdê°€ ì�ˆì–´ì•¼ í•¨
df_valid = df_explode_SourceDatasetVersionId.filter(
    (pl.col("KernelVersionCreationDate").is_not_null()) &
    (pl.col("SourceDatasetVersionId").is_not_null())
)

# 2. ì—°ë�„ ì¶”ì¶œ
df_valid = df_valid.with_columns(
    pl.col("KernelVersionCreationDate").cast(pl.Date).dt.year().alias("Year")
)

# 3. ì—°ë�„ë³„ SourceDatasetVersionId ê°œìˆ˜ ì§‘ê³„
yearly_counts = (
    df_valid
    .group_by("Year")
    .agg(pl.col("SourceDatasetVersionId").n_unique().alias("DatasetVersionCount"))
    .sort("Year")
    .to_pandas()
)

# 4. ì‹œê°�í™”
plt.figure(figsize=(12, 6))
plt.plot(yearly_counts["Year"], yearly_counts["DatasetVersionCount"], marker='o', color='teal', linewidth=2)
plt.title("ì—°ë�„ë³„ SourceDatasetVersionId ê°œìˆ˜ ì¶”ì�´")
plt.xlabel("ì—°ë�„")
plt.ylabel("Dataset Version ì‚¬ìš© íšŸìˆ˜")
plt.grid(True)

# ê°� ì§€ì �ì—� ê°’ í‘œì‹œ
for i, row in yearly_counts.iterrows():
    plt.text(row["Year"], row["DatasetVersionCount"] + 5, f"{row['DatasetVersionCount']:,}", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


df_valid_ks = df_explode_SourceKernelVersionId.filter(
    (pl.col("KernelVersionCreationDate").is_not_null()) &
    (pl.col("SourceKernelVersionId").is_not_null())
).with_columns(
    pl.col("KernelVersionCreationDate").cast(pl.Date).dt.year().alias("Year")
)

yearly_counts_ks = (
    df_valid_ks
    .group_by("Year")
    .agg(pl.col("SourceKernelVersionId").n_unique().alias("UniqueCount"))
    .sort("Year")
    .to_pandas()
)

plt.figure(figsize=(12, 6))
plt.plot(yearly_counts_ks["Year"], yearly_counts_ks["UniqueCount"], marker='o', color='orange', linewidth=2)
plt.title("ì—°ë�„ë³„ SourceKernelVersionId ê°œìˆ˜ ì¶”ì�´")
plt.xlabel("ì—°ë�„")
plt.ylabel("Kernel Version ì‚¬ìš© íšŸìˆ˜")
plt.grid(True)

for i, row in yearly_counts_ks.iterrows():
    plt.text(row["Year"], row["UniqueCount"] + 5, f"{row['UniqueCount']:,}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


df_valid_mv = df_explode_SourceModelVariationId.filter(
    (pl.col("KernelVersionCreationDate").is_not_null()) &
    (pl.col("SourceModelVariationId").is_not_null())
).with_columns(
    pl.col("KernelVersionCreationDate").cast(pl.Date).dt.strftime("%Y-%m").alias("YearMonth")
)

monthly_counts_mv = (
    df_valid_mv
    .group_by("YearMonth")
    .agg(pl.col("SourceModelVariationId").n_unique().alias("UniqueCount"))
    .sort("YearMonth")
    .to_pandas()
)

plt.figure(figsize=(14, 6))
plt.plot(monthly_counts_mv["YearMonth"], monthly_counts_mv["UniqueCount"], marker='o', color='green', linewidth=2)
plt.title("ì›”ë³„ SourceModelVariationId ê°œìˆ˜ ì¶”ì�´ (ì—°ë�„:ì›”)")
plt.xlabel("ì—°ë�„-ì›”")
plt.ylabel("Model Variation ì‚¬ìš© íšŸìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)

for i, row in monthly_counts_mv.iterrows():
    plt.text(i, row["UniqueCount"] + 5, f"{row['UniqueCount']:,}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()



df['FollowingUserId'].len()


df = pl.read_parquet('User_1to4_with_kernel.parquet')


df = df.filter(df['AchievementType']=='Scripts')


df.group_by('UserName').agg(pl.col(''))


max_dates = df.group_by("UserId").agg(
    pl.col("CreationDate_KernelVersions").max().alias("max_date")
)

# 2. ì›�ë³¸ê³¼ ì¡°ì�¸í•˜ì—¬ ìµœëŒ“ê°’ì—� í•´ë‹¹í•˜ëŠ” í–‰ë§Œ í•„í„°
df_max = df.join(
    max_dates,
    left_on=["UserId", "CreationDate_KernelVersions"],
    right_on=["UserId", "max_date"],
    how="inner"
)

# 3. ì™„ì „í�ˆ ê°™ì�€ í–‰ì�´ ì¤‘ë³µë�  ê²½ìš° ì¤‘ë³µ ì œê±°
df_unique = df_max.unique()


df_unique = df_unique.select(['UserName', 'UserId', 'CreationDate_KernelVersions', 'PerformanceTier', 'RegisterDate', 'DaysSinceSignup', 'IsActiveTierUser', 'FirstAchvDate', 'LastAchvDate', 'DaysSinceLastAchv', 'TierProgression', 'CumulativePoints',
                  'AvgPointsPerAchv', 'AchievementType', 'Tier', 'TierAchievementDate', 'Points', 'CurrentRanking', 'HighestRanking', 'TotalGold', 'TotalSilver', 'TotalBronze', 'CurrentRankingStatus', 'HighestRankingStatus'])


df_unique.write_parquet('Kernel_latest_activity.parquet')


df_unique.write_parquet('Kernel_latest_activity_full_columns.parquet')


df_unique


df_unique


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



query = """
COPY
(
SELECT u.FollowingUserId,
        u.UserId,
        k.TotalViews As KernelTotalViews,
        k.TotalVotes_k AS KernelTotalVotes
        FROM read_parquet('KernelMerged.parquet') k
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON k.AuthorUserId_kv = u.UserId
) TO 'User_1to4_for_explode.parquet' (FORMAT 'parquet');
"""

duckdb.query(query)


df = pl.read_parquet('User_1to4_for_explode.parquet', low_memory=True)
df


df = df.explode('FollowingUserId')


top10_following = (
    df
    .filter(~pl.col("FollowingUserId").is_null())
    .group_by("UserId")
    .agg(pl.col("FollowingUserId").n_unique().alias("FollowingCount"))
    .sort("FollowingCount", descending=True)
    .limit(10)
    .to_pandas()
)


import pandas as pd

df_name = pd.read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet', columns=['UserId', 'UserName'])

# ì¤‘ë³µ UserId ì œê±° (UserNameì�€ ì²« ë²ˆì§¸ ê°’ ìœ ì§€)
df_name_unique = df_name.drop_duplicates(subset=["UserId"])

top10_with_names = top10_following.merge(df_name_unique, on="UserId", how="left")

# íŒ”ë¡œì�‰ ìˆ˜ ë‚´ë¦¼ì°¨ìˆœ ì •ë ¬ ê¸°ì¤€ UserId ë¦¬ìŠ¤íŠ¸ (ê·¸ë�˜í”„ ìˆœì„œìš©)
order = top10_with_names.sort_values("FollowingCount", ascending=False)["UserId"].tolist()

# ì‹œê°�í™”
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top10_with_names,
    x="UserId",
    y="FollowingCount",
    palette="Blues_d",
    order=order
)

# xì¶• ë ˆì�´ë¸”ì�„ UserNameìœ¼ë¡œ ë°”ê¾¸ê¸°
plt.xticks(
    ticks=range(len(order)),
    labels=[top10_with_names.loc[top10_with_names["UserId"] == uid, "UserName"].values[0] for uid in order],
)

# íŒ”ë¡œì�‰ ìˆ˜ í…�ìŠ¤íŠ¸ í‘œì‹œ
for i, row in top10_with_names.iterrows():
    x_pos = order.index(row["UserId"])
    plt.text(
        x=x_pos,
        y=row["FollowingCount"] + 1,
        s=f"{row['FollowingCount']}",
        ha='center',
        fontweight='bold'
    )

plt.title("íŒ”ë¡œì�‰ ìˆ˜ ìƒ�ìœ„ 10ëª… (UserName í‘œì‹œ)")
plt.xlabel("UserName")
plt.ylabel("íŒ”ë¡œì�‰ ìˆ˜")
plt.tight_layout()
plt.show()


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)
df


import polars as pl
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ê¸°ì¤€ì�¼
today = datetime.today()

# 2ë‹¬ ì „
start_2months_ago = (today.replace(day=1) - relativedelta(months=2)).date()
end_2months_ago = (today.replace(day=1) - relativedelta(months=1, days=1)).date()

# 1ë‹¬ ì „
start_last_month = (today.replace(day=1) - relativedelta(months=1)).date()
end_last_month = (today.replace(day=1) - timedelta(days=1)).date()

# ì»¤ë„� ìˆ˜ ê³„ì‚°
count_2months_ago = df.filter(
    (pl.col('KernelCreationDate').dt.date() >= pl.lit(start_2months_ago).cast(pl.Date)) &
    (pl.col('KernelCreationDate').dt.date() <= pl.lit(end_2months_ago).cast(pl.Date))
).height

count_last_month = df.filter(
    (pl.col('KernelCreationDate').dt.date() >= pl.lit(start_last_month).cast(pl.Date)) &
    (pl.col('KernelCreationDate').dt.date() <= pl.lit(end_last_month).cast(pl.Date))
).height

# ì¦�ê°�ë¥  ê³„ì‚°
if count_2months_ago > 0:
    change_rate = (count_last_month - count_2months_ago) / count_2months_ago * 100
else:
    change_rate = None  # ë˜�ëŠ” float('nan')

# ê²°ê³¼ ë”•ì…”ë„ˆë¦¬ë¡œ ì €ì�¥
kernel_stats = {
    "2_months_ago": count_2months_ago,
    "last_month": count_last_month,
    "change_percent": round(change_rate, 2) if change_rate is not None else None
}

print(kernel_stats)


import polars as pl
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ê¸°ì¤€ì�¼
today = datetime.today()

# ë‚ ì§œ ë²”ìœ„ ì„¤ì •
start_last_month = (today.replace(day=1) - relativedelta(months=1)).date()
end_last_month = (today.replace(day=1) - timedelta(days=1)).date()

start_2months_ago = (today.replace(day=1) - relativedelta(months=2)).date()
end_2months_ago = (today.replace(day=1) - relativedelta(months=1, days=1)).date()

# ì „ì²´ ìœ ì € ìˆ˜
total_users = df.select("UserId").unique().height

# ì§€ë‚œë‹¬ ì»¤ë„� ì�‘ì„± ìœ ì € ìˆ˜
last_month_users = (
    df.filter(
        (pl.col("KernelCreationDate").dt.date() >= pl.lit(start_last_month).cast(pl.Date)) &
        (pl.col("KernelCreationDate").dt.date() <= pl.lit(end_last_month).cast(pl.Date))
    )
    .select("UserId")
    .unique()
    .height
)

# ì§€ì§€ë‚œë‹¬ ì»¤ë„� ì�‘ì„± ìœ ì € ìˆ˜
two_months_ago_users = (
    df.filter(
        (pl.col("KernelCreationDate").dt.date() >= pl.lit(start_2months_ago).cast(pl.Date)) &
        (pl.col("KernelCreationDate").dt.date() <= pl.lit(end_2months_ago).cast(pl.Date))
    )
    .select("UserId")
    .unique()
    .height
)

# ë¹„ìœ¨ ë°� ì¦�ê°�ë¥  ê³„ì‚°
ratio_to_all_users = last_month_users / total_users * 100 if total_users > 0 else None
ratio_to_prev_month = (
    (last_month_users - two_months_ago_users) / two_months_ago_users * 100
    if two_months_ago_users > 0 else None
)

# ê²°ê³¼ ë”•ì…”ë„ˆë¦¬ ì €ì�¥
user_kernel_stats = {
    "last_month_active_kernel_users": last_month_users,
    "total_users": total_users,
    "2months_ago_active_kernel_users": two_months_ago_users,
    "ratio_to_all_users_percent": round(ratio_to_all_users, 2) if ratio_to_all_users is not None else None,
    "change_percent_from_2months_ago": round(ratio_to_prev_month, 2) if ratio_to_prev_month is not None else None
}

print(user_kernel_stats)



import polars as pl

# 1. ì»¤ë„� ë‹¨ìœ„ë¡œ ê³ ìœ í™” (KernelIdê°€ ì�ˆëŠ” ê²½ìš°ë§Œ ì‚¬ìš©)
df_kernel_unique = (
    df
    .select(["KernelId", "UserId", "KernelCreationDate", "PerformanceTier"])
    .filter(
        pl.col("KernelId").is_not_null() &
        pl.col("UserId").is_not_null() &
        pl.col("KernelCreationDate").is_not_null() &
        pl.col("PerformanceTier").is_not_null()
    )
    .unique(subset=["KernelId"])  # ì»¤ë„� ê¸°ì¤€ ì¤‘ë³µ ì œê±°
    .with_columns(pl.col("KernelCreationDate").cast(pl.Date))
)

# 2. ìœ ì €ë³„ ì—…ë¡œë“œ ê°„ê²© ê³„ì‚°
df_diff = (
    df_kernel_unique
    .select(["UserId", "KernelCreationDate"])
    .sort(["UserId", "KernelCreationDate"])
    .group_by("UserId")
    .agg(
        pl.col("KernelCreationDate").diff().dt.total_days().alias("UploadIntervals")
    )
    .explode("UploadIntervals")
    .filter(pl.col("UploadIntervals").is_not_null())
)

# 3. ìœ ì €ë³„ í‹°ì–´ ì¡°ì�¸
user_tier = df_kernel_unique.select(["UserId", "PerformanceTier"]).unique()
df_with_tier = df_diff.join(user_tier, on="UserId", how="left")

# 4. í‹°ì–´ë³„ í�‰ê·  ì—…ë¡œë“œ ê°„ê²© ì§‘ê³„
result = (
    df_with_tier
    .group_by("PerformanceTier")
    .agg(pl.col("UploadIntervals").mean().alias("AvgUploadInterval"))
    .sort("PerformanceTier")
)

# 5. ë”•ì…”ë„ˆë¦¬ í˜•íƒœë¡œ ì¶œë ¥
upload_duration = {
    row["PerformanceTier"]: row["AvgUploadInterval"]
    for row in result.to_dicts()
}

print(upload_duration)



# 1. ìœ íš¨ ë�°ì�´í„° í•„í„°
df_valid = df.filter(
    (pl.col("KernelVersionCreationDate").is_not_null()) &
    (pl.col("UserId").is_not_null()) &
    (pl.col("PerformanceTier").is_not_null())
).with_columns([
    pl.col("KernelVersionCreationDate").cast(pl.Date)
])

# 2. ìœ ì €ë³„ ì—…ë¡œë“œ ê°„ê²© ê³„ì‚°
df_diff = (
    df_valid
    .select(["UserId", "KernelVersionCreationDate"])
    .sort(["UserId", "KernelVersionCreationDate"])
    .group_by("UserId")
    .agg(pl.col("KernelVersionCreationDate").diff().dt.total_days().alias("UploadIntervals"))
    .explode("UploadIntervals")
    .filter(pl.col("UploadIntervals").is_not_null())
)

# 3. ìœ ì € IDë¡œ PerformanceTier ë¶™ì�´ê¸°
user_tier = df_valid.select(["UserId", "PerformanceTier"]).unique()
df_with_tier = df_diff.join(user_tier, on="UserId", how="left")

# 4. í‹°ì–´ë³„ í�‰ê·  ì—…ë¡œë“œ ê°„ê²© ì§‘ê³„
result = (
    df_with_tier
    .group_by("PerformanceTier")
    .agg(pl.col("UploadIntervals").mean().alias("AvgUploadInterval"))
    .sort("PerformanceTier")
)

upload_duration = {
    row["PerformanceTier"]: row["AvgUploadInterval"]
    for row in result.to_dicts()
}

print(upload_duration)


from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import polars as pl

# ë‚ ì§œ ê¸°ì¤€ ê³„ì‚°
today = datetime.today()
start_last_month = (today.replace(day=1) - relativedelta(months=1)).date()
end_last_month = (today.replace(day=1) - timedelta(days=1)).date()
start_prev_month = (today.replace(day=1) - relativedelta(months=2)).date()
end_prev_month = (start_last_month - timedelta(days=1))

# í•¨ìˆ˜: ê¸°ê°„ ë‚´ ë©”ë‹¬ ê°œìˆ˜ ì§‘ê³„
def count_medals(df, medal_col, date_col, start_date, end_date):
    return (
        df.filter(
            (pl.col(medal_col).is_in([1, 2, 3])) &
            (pl.col(date_col).dt.date() >= pl.lit(start_date)) &
            (pl.col(date_col).dt.date() <= pl.lit(end_date))
        )
        .with_columns(pl.lit(1).alias("dummy"))  # ì�„ì‹œ ì�¸ë�±ìŠ¤ìš©
        .group_by(["dummy", medal_col])
        .agg(pl.len().alias("count"))
        .pivot(values="count", index="dummy", on=medal_col)
        .drop("dummy")
        .fill_null(0)
    )


# ì§‘ê³„
last = count_medals(df, "KernelMedal", "KernelMedalDate", start_last_month, end_last_month)
prev = count_medals(df, "KernelMedal", "KernelMedalDate", start_prev_month, end_prev_month)

# ë”•ì…”ë„ˆë¦¬ í˜•íƒœë¡œ ë³€í™˜
last_dict = last.to_dicts()[0] if last.height > 0 else {1: 0, 2: 0, 3: 0}
prev_dict = prev.to_dicts()[0] if prev.height > 0 else {1: 0, 2: 0, 3: 0}

# ë¹„ìœ¨ ê³„ì‚° ë°� ê²°ê³¼ êµ¬ì„±
medal_stats = {}
for medal in [1, 2, 3]:
    last_count = last_dict.get(medal, 0)
    prev_count = prev_dict.get(medal, 0)
    growth = (
        ((last_count - prev_count) / prev_count * 100)
        if prev_count != 0 else float('inf') if last_count > 0 else 0
    )

    medal_stats[medal] = {
        "last_month": last_count,
        "prev_month": prev_count,
        "change_percent": round(growth, 1)
    }

# ìµœì¢… ë”•ì…”ë„ˆë¦¬ ì¶œë ¥ ì˜ˆì‹œ
print(medal_stats)



import polars as pl
import matplotlib.pyplot as plt

# 0. TagId ë¦¬ìŠ¤íŠ¸ ì •ë³´ ë¡œë“œ ë°� ì •ë¦¬
kernel_tags = pl.read_parquet("temporary_kernel_tags.parquet")
kernel_tags = kernel_tags.filter(
    pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0)
)

# 1. íƒœê·¸ ë©”íƒ€ë�°ì�´í„° ë¡œë“œ
tags_meta = pl.read_parquet("Tags.parquet").select(["Id", "Name"]).rename({"Id": "TagId"})

# 2. ì»¤ë„� ìƒ�ì„±ì�¼ í�¬í•¨í•œ ë�°ì�´í„° ì¤€ë¹„
df_kernel = df.select(["KernelId", "KernelCreationDate"])
df_tags = df_kernel.join(kernel_tags, on="KernelId", how="left")

# 3. ì—°ë�„ ì¶”ì¶œ ë°� explode
df_tags = (
    df_tags
    .filter(pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0))
    .with_columns(pl.col("KernelCreationDate").dt.year().alias("Year"))
    .explode("TagId")
    .with_columns(pl.col("TagId").cast(pl.Int64))
)

# 4. íƒœê·¸ ì�´ë¦„ ì¡°ì�¸
df_joined = df_tags.join(tags_meta, on="TagId", how="left")

# 5. ì „ì²´ ê¸°ê°„ ê¸°ì¤€ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ì¶”ì¶œ
tag_total_count = (
    df_joined
    .group_by("Name")
    .agg(pl.len().alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .limit(10)
)

top_10_tags = tag_total_count["Name"].to_list()

# 6. ì—°ë�„ë³„ í•´ë‹¹ íƒœê·¸ë“¤ë§Œ í•„í„°ë§�í•˜ì—¬ ì§‘ê³„
tag_trend = (
    df_joined
    .filter(pl.col("Name").is_in(top_10_tags))
    .group_by(["Year", "Name"])
    .agg(pl.len().alias("TagCount"))
    .sort(["Year", "TagCount"], descending=[False, True])
)

# 7. Pandas ë³€í™˜ ë°� í”¼ë²— â†’ ì‹œê°�í™”
tag_trend_pd = tag_trend.to_pandas()
pivot_df = tag_trend_pd.pivot(index="Year", columns="Name", values="TagCount").fillna(0)

plt.figure(figsize=(14, 7))
pivot_df.plot(kind='area', stacked=True, figsize=(14, 7), cmap="tab20")
plt.title("ì—°ë�„ë³„ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ë“±ì�¥ íšŸìˆ˜ ë³€í™” (ì „ì²´ ê¸°ì¤€)")
plt.xlabel("ì—°ë�„")
plt.ylabel("ë“±ì�¥ íšŸìˆ˜")
plt.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



import polars as pl
import matplotlib.pyplot as plt

# 0. TagId ë¦¬ìŠ¤íŠ¸ ì •ë³´ ë¡œë“œ ë°� ì •ë¦¬
kernel_tags_orig = pl.read_parquet("temporary_kernel_tags.parquet")
kernel_tags = kernel_tags_orig.filter(
    pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0)
)

# 1. íƒœê·¸ ë©”íƒ€ë�°ì�´í„° ë¡œë“œ
tags_meta = pl.read_parquet("Tags.parquet").select(["Id", "Name"]).rename({"Id": "TagId"})

# 2. ì»¤ë„� ìƒ�ì„±ì�¼ í�¬í•¨í•œ ë�°ì�´í„° ì¤€ë¹„
df_kernel_base = df.select(["KernelId", "KernelCreationDate"])
df_kernel_tags = df_kernel_base.join(kernel_tags, on="KernelId", how="left")

# 3. ì—°ë�„ ì¶”ì¶œ ë°� explode
df_tagged = (
    df_kernel_tags
    .filter(pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0))
    .with_columns(pl.col("KernelCreationDate").dt.year().alias("Year"))
    .explode("TagId")
    .with_columns(pl.col("TagId").cast(pl.Int64))
)

# 4. íƒœê·¸ ì�´ë¦„ ì¡°ì�¸
df_named = df_tagged.join(tags_meta, on="TagId", how="left")

# 5. ì „ì²´ ê¸°ê°„ ê¸°ì¤€ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ì¶”ì¶œ
tag_total_count = (
    df_named
    .group_by("Name")
    .agg(pl.len().alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .limit(10)
)

top_10_tags = tag_total_count["Name"].to_list()

# 6. ì—°ë�„ë³„ í•´ë‹¹ íƒœê·¸ë“¤ë§Œ í•„í„°ë§�í•˜ì—¬ ì§‘ê³„
df_trend = (
    df_named
    .filter(pl.col("Name").is_in(top_10_tags))
    .group_by(["Year", "Name"])
    .agg(pl.len().alias("TagCount"))
    .sort(["Year", "TagCount"], descending=[False, True])
)

df_trend_pd = df_trend.to_pandas()

# 7. ì ˆëŒ“ê°’ ê¸°ë°˜ í”¼ë²—
pivot_count_df = df_trend_pd.pivot(index="Year", columns="Name", values="TagCount").fillna(0)

# 8. ì �ìœ ìœ¨ ê¸°ë°˜ í”¼ë²—
pivot_ratio_df = pivot_count_df.div(pivot_count_df.sum(axis=1), axis=0)

# 9. ì‹œê°�í™” - ì ˆëŒ“ê°’
plt.figure(figsize=(14, 7))
pivot_count_df.plot(kind='area', stacked=True, cmap="tab20", figsize=(14, 7))
plt.title("ì—°ë�„ë³„ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ë“±ì�¥ íšŸìˆ˜ ë³€í™” (ì ˆëŒ“ê°’ ê¸°ì¤€)")
plt.xlabel("ì—°ë�„")
plt.ylabel("ë“±ì�¥ íšŸìˆ˜")
plt.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 10. ì‹œê°�í™” - ì �ìœ ìœ¨
plt.figure(figsize=(14, 7))
pivot_ratio_df.plot(kind='area', stacked=True, cmap="tab20", figsize=(14, 7))
plt.title("ì—°ë�„ë³„ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ì �ìœ ìœ¨ ë³€í™” (ë¹„ìœ¨ ê¸°ì¤€)")
plt.xlabel("ì—°ë�„")
plt.ylabel("ì �ìœ ìœ¨")
plt.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



df = pl.read_parquet("/home/ronny/Downloads/final_project/dashboard/kernel_dashboard.parquet")
df


import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)

# 0. TagId ë¦¬ìŠ¤íŠ¸ ì •ë³´ ë¡œë“œ ë°� ì •ë¦¬
kernel_tags_orig = pl.read_parquet("temporary_kernel_tags.parquet")
kernel_tags = kernel_tags_orig.filter(
    pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0)
)

# 1. íƒœê·¸ ë©”íƒ€ë�°ì�´í„° ë¡œë“œ
tags_meta = pl.read_parquet("Tags.parquet").select(["Id", "Name"]).rename({"Id": "TagId"})

# 2. ì»¤ë„� ìƒ�ì„±ì�¼ í�¬í•¨í•œ ë�°ì�´í„° ì¤€ë¹„
df_kernel_base = df.select(["KernelId", "KernelCreationDate"])
df_kernel_tags = df_kernel_base.join(kernel_tags, on="KernelId", how="left")

# 3. ì¤‘ë³µ ì œê±°, ì—°ë�„ ì¶”ì¶œ ë°� explode
df_tagged = (
    df_kernel_tags
    .unique(subset=["KernelId"])  # ì¤‘ë³µë�œ KernelVersion ì œê±°
    .filter(pl.col("TagId").is_not_null() & (pl.col("TagId").list.len() > 0))
    .with_columns(pl.col("KernelCreationDate").dt.year().alias("Year"))
    .explode("TagId")
    .with_columns(pl.col("TagId").cast(pl.Int64))
)

# 4. íƒœê·¸ ì�´ë¦„ ì¡°ì�¸
df_named = df_tagged.join(tags_meta, on="TagId", how="left")

# 5. ì „ì²´ ê¸°ê°„ ê¸°ì¤€ ìƒ�ìœ„ 5ê°œ íƒœê·¸ ì¶”ì¶œ
tag_total_count = (
    df_named
    .group_by("Name")
    .agg(pl.len().alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .limit(5)
)
top_5_tags = tag_total_count["Name"].to_list()

# 6. ì—°ë�„ë³„ í•´ë‹¹ íƒœê·¸ë“¤ë§Œ í•„í„°ë§�í•˜ì—¬ ì§‘ê³„
df_trend = (
    df_named
    .filter(pl.col("Name").is_in(top_5_tags))
    .group_by(["Year", "Name"])
    .agg(pl.len().alias("TagCount"))
    .sort(["Year", "TagCount"], descending=[False, True])
)

# 7. Pandas ë³€í™˜
df_trend_pd = df_trend.to_pandas()

# 8. í”¼ë²—: ì ˆëŒ“ê°’
pivot_count_df = df_trend_pd.pivot(index="Year", columns="Name", values="TagCount").fillna(0)

# 9. í”¼ë²—: ì �ìœ ìœ¨
pivot_ratio_df = pivot_count_df.div(pivot_count_df.sum(axis=1), axis=0)

# 10. ë�¼ì�¸ê·¸ë�˜í”„ - ì ˆëŒ“ê°’
plt.figure(figsize=(14, 7))
for tag in pivot_count_df.columns:
    plt.plot(pivot_count_df.index, pivot_count_df[tag], marker='o', label=tag)

plt.title("ì—°ë�„ë³„ ìƒ�ìœ„ 5ê°œ íƒœê·¸ ë“±ì�¥ íšŸìˆ˜ ë³€í™” (ì ˆëŒ“ê°’ ê¸°ì¤€)")
plt.xlabel("ì—°ë�„")
plt.ylabel("ë“±ì�¥ íšŸìˆ˜")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 11. ë�¼ì�¸ê·¸ë�˜í”„ - ì �ìœ ìœ¨
plt.figure(figsize=(14, 7))
for tag in pivot_ratio_df.columns:
    plt.plot(pivot_ratio_df.index, pivot_ratio_df[tag], marker='o', label=tag)

plt.title("ì—°ë�„ë³„ ìƒ�ìœ„ 5ê°œ íƒœê·¸ ì �ìœ ìœ¨ ë³€í™” (ë¹„ìœ¨ ê¸°ì¤€)")
plt.xlabel("ì—°ë�„")
plt.ylabel("ì �ìœ ìœ¨")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Tag", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



df["KernelId"].n_unique()
len(df)



import polars as pl
import matplotlib.pyplot as plt

# ì»¤ë„� ìƒ�ì„±ì�¼ì�´ ì¡´ì�¬í•˜ëŠ” ê²½ìš°ë§Œ í•„í„°ë§�
df_kernel_created = df.filter(pl.col("KernelCreationDate").is_not_null())

# ì›” ë‹¨ìœ„ë¡œ ë³€í™˜ (ì˜ˆ: 2023-04-01)
df_kernel_month = df_kernel_created.with_columns(
    pl.col("KernelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("KernelCreationMonth")
)

# ì›”ë³„ ì»¤ë„� ìƒ�ì„± íšŸìˆ˜ ì§‘ê³„
df_kernel_monthly_count = (
    df_kernel_month
    .group_by("KernelCreationMonth")
    .agg(pl.col('KernelId').n_unique().alias("KernelCount"))
    .sort("KernelCreationMonth")
)

# íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜í•´ì„œ ì‹œê°�í™”
df_pd = df_kernel_monthly_count.to_pandas()

plt.figure(figsize=(12, 6))
plt.plot(df_pd["KernelCreationMonth"], df_pd["KernelCount"], marker='o', linewidth=2)
plt.title("ì›”ë³„ ì»¤ë„� ìƒ�ì„± ìˆ˜")
plt.xlabel("ì›”")
plt.ylabel("ì»¤ë„� ìƒ�ì„± íšŸìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()



import polars as pl
import matplotlib.pyplot as plt

# ì»¤ë„� ìƒ�ì„±ì�¼ì�´ ì¡´ì�¬í•˜ëŠ” ê²½ìš°ë§Œ í•„í„°ë§�
df_kernel_created = df.filter(pl.col("KernelVersionCreationDate").is_not_null())

# ì›” ë‹¨ìœ„ë¡œ ë³€í™˜ (ì˜ˆ: 2023-04-01)
df_kernel_month = df_kernel_created.with_columns(
    pl.col("KernelVersionCreationDate").cast(pl.Date).dt.truncate("1mo").alias("KernelCreationMonth")
)

# ì›”ë³„ ì»¤ë„� ìƒ�ì„± íšŸìˆ˜ ì§‘ê³„
df_kernel_monthly_count = (
    df_kernel_month
    .group_by("KernelCreationMonth")
    .agg(pl.col('KernelVersionId').n_unique().alias("KernelCount"))
    .sort("KernelCreationMonth")
)

# íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜í•´ì„œ ì‹œê°�í™”
df_pd = df_kernel_monthly_count.to_pandas()

plt.figure(figsize=(12, 6))
plt.plot(df_pd["KernelCreationMonth"], df_pd["KernelCount"], marker='o', linewidth=2)
plt.title("ì›”ë³„ ì»¤ë„� ë²„ì „ ìƒ�ì„± ìˆ˜")
plt.xlabel("ì›”")
plt.ylabel("ì»¤ë„� ë²„ì „ ìƒ�ì„± íšŸìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()



import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = df.filter(pl.col('AchievementType')=='Scripts')

# 1. ìœ íš¨í•œ ê°’ í•„í„°ë§�
df_valid = df.filter(
    pl.col("KernelId").is_not_null() &
    pl.col("Tier").is_not_null() &
    (pl.col("Tier") != 0)
)

# 2. PerformanceTier + UserId ê¸°ì¤€ìœ¼ë¡œ ê³ ìœ  ìœ ì € ì§‘ê³„
df_kernel_counts = (
    df_valid
    .group_by(["Tier", "UserId"])
    .agg([
        pl.col("UserName").first(),
        pl.col("KernelId").n_unique().alias("KernelCount"),
        pl.col("KernelTotalComments").sum().alias("TotalComments"),
        pl.col("KernelTotalVotes").sum().alias("TotalVotes"),
    ])
)

# 3. í‹°ì–´ë³„ ìƒ�ìœ„ 10ëª… í•„í„°ë§�
df_top10_list = []
for tier in df_kernel_counts.select("Tier").unique().to_series():
    tier_df = (
        df_kernel_counts
        .filter(pl.col("Tier") == tier)
        .sort("KernelCount", descending=True)
        .head(10)
    )
    df_top10_list.append(tier_df)

top10_per_tier = pl.concat(df_top10_list)

# 4. íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜
df_pd = top10_per_tier.to_pandas()
tiers = sorted(df_pd["Tier"].unique())

# ---------- (1) ë°”ê·¸ë�˜í”„ ----------
for tier in tiers:
    subset = df_pd[df_pd["Tier"] == tier].sort_values("KernelCount", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x="KernelCount", y="UserName", palette="Blues_d")
    plt.title(f"Tier {tier} ìƒ�ìœ„ 10ëª… - ì»¤ë„� ìˆ˜ ê¸°ì¤€")
    plt.xlabel("KernelCount")
    plt.ylabel("UserName")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()

# ---------- (2) ì‚°ì �ë�„ ----------
for tier in tiers:
    subset = df_pd[df_pd["Tier"] == tier].copy()
    subset["Size"] = np.sqrt(subset["KernelCount"]) * 5

    plt.figure(figsize=(8, 6))
    plt.scatter(
        subset["TotalComments"],
        subset["TotalVotes"],
        s=subset["Size"],
        alpha=0.7,
        color="orange"
    )
    for _, row in subset.iterrows():
        plt.text(row["TotalComments"], row["TotalVotes"], row["UserName"], fontsize=8)
    plt.title(f"Tier {tier} ìƒ�ìœ„ 10ëª… - ì‚°ì �ë�„")
    plt.xlabel("KernelTotalComments")
    plt.ylabel("KernelTotalVotes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- (3) ê²°ê³¼í‘œ ----------
summary_df = df_pd[["UserId", "UserName", "KernelCount", "TotalComments", "TotalVotes"]]
summary_df



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_kernel.parquet', low_memory=True)


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# 1. íŒŒì�¼ ì�½ê¸°
df = pl.read_parquet("User_1to4_with_kernel.parquet", low_memory=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. ì»¤ë„� ìµœì´ˆ ìƒ�ì„±ì�¼ ê¸°ì¤€ ì§‘ê³„
df_kernel = df.select(["KernelId", "KernelCreationDate"]).filter(
    pl.col("KernelId").is_not_null() & pl.col("KernelCreationDate").is_not_null()
)
df_kernel_created = (
    df_kernel
    .group_by("KernelId")
    .agg(pl.col("KernelCreationDate").min().alias("FirstKernelDate"))
    .with_columns(pl.col("FirstKernelDate").cast(pl.Date).dt.truncate("1mo").alias("Month"))
)
df_kernel_month = (
    df_kernel_created
    .group_by("Month")
    .agg(pl.len().alias("KernelCount"))
    .sort("Month")
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. ì»¤ë„� íˆ¬í‘œ ë‚ ì§œ ë¬¸ì��ì—´ ì •ì œ ë°� ì§‘ê³„
df_votes_raw = df.select(["KernelId", "UserPerVoteDate"]).filter(
    pl.col("UserPerVoteDate").is_not_null() & (pl.col("UserPerVoteDate") != '""{}""')
)
df_votes_exploded = (
    df_votes_raw
    .with_columns([
        pl.col("UserPerVoteDate")
        .str.replace_all(r'^""\{', '')
        .str.replace_all(r'\}""$', '')
        .str.replace_all("'", "")
        .str.split(", ")
        .alias("VoteRawList")
    ])
    .explode("VoteRawList")
    .filter(pl.col("VoteRawList").str.contains(":"))
    .with_columns([
        pl.col("VoteRawList")
        .str.split(":")
        .list.get(1)
        .str.strip_chars()
        .str.strip_chars('"')
        .str.strip_chars("}")
        .str.strip_chars("\\")
        .str.strip_chars("\n")
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("VoteDate")
    ])
    .filter(pl.col("VoteDate").is_not_null())
    .with_columns(pl.col("VoteDate").dt.truncate("1mo").alias("Month"))
)
df_votes_month = (
    df_votes_exploded
    .group_by("Month")
    .agg(pl.len().alias("VoteCount"))
    .sort("Month")
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. ë³‘í•© ë°� ì‹œê°�í™”
df_final = df_kernel_month.join(df_votes_month, on="Month", how="outer").sort("Month")
df_pd = df_final.to_pandas().fillna(0)

plt.figure(figsize=(14, 6))
plt.plot(df_pd["Month"], df_pd["KernelCount"], label="ì»¤ë„� ìƒ�ì„± ìˆ˜", marker="o")
plt.plot(df_pd["Month"], df_pd["VoteCount"], label="íˆ¬í‘œ ìˆ˜", marker="s")
plt.title("ì›”ë³„ ì»¤ë„� ìƒ�ì„± ìˆ˜ ë°� íˆ¬í‘œ ìˆ˜ (ì •ì œë�œ ê¸°ì¤€)")
plt.xlabel("ì›”")
plt.ylabel("ìˆ˜ëŸ‰")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df.filter(pl.col('UserPerVoteDate')!='"{}"')


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# 1. íŒŒì�¼ ì�½ê¸°
df = pl.read_parquet("User_1to4_with_kernel.parquet", low_memory=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. ì»¤ë„� ìƒ�ì„± ìˆ˜ (ìµœì´ˆ ìƒ�ì„±ì�¼ ê¸°ì¤€)
df_kernel = df.select(["KernelId", "KernelCreationDate"]).filter(
    pl.col("KernelId").is_not_null() & pl.col("KernelCreationDate").is_not_null()
)
df_kernel_created = (
    df_kernel
    .group_by("KernelId")
    .agg(pl.col("KernelCreationDate").min().alias("FirstKernelDate"))
    .with_columns(pl.col("FirstKernelDate").cast(pl.Date).dt.truncate("1mo").alias("Month"))
)
df_kernel_month = (
    df_kernel_created
    .group_by("Month")
    .agg(pl.len().alias("KernelCount"))
    .sort("Month")
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. ì»¤ë„� íˆ¬í‘œ ìˆ˜ (UserPerVoteDate ì •ì œ)
df_votes_raw = df.select(["KernelId", "UserPerVoteDate"]).filter(
    pl.col("UserPerVoteDate").is_not_null() & (pl.col("UserPerVoteDate") != '""{}""')
)
df_votes_exploded = (
    df_votes_raw
    .with_columns([
        pl.col("UserPerVoteDate")
        .str.replace_all(r'^""\{', '')
        .str.replace_all(r'\}""$', '')
        .str.replace_all("'", "")
        .str.split(", ")
        .alias("VoteRawList")
    ])
    .explode("VoteRawList")
    .filter(pl.col("VoteRawList").str.contains(":"))
    .with_columns([
        pl.col("VoteRawList")
        .str.split(":")
        .list.get(1)
        .str.strip_chars()
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("VoteDate")
    ])
    .filter(pl.col("VoteDate").is_not_null())
    .with_columns(pl.col("VoteDate").dt.truncate("1mo").alias("Month"))
)
df_votes_month = (
    df_votes_exploded
    .group_by("Month")
    .agg(pl.len().alias("VoteCount"))
    .sort("Month")
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. ë³‘í•© ë°� ê³„ì‚°
df_final = df_kernel_month.join(df_votes_month, on="Month", how="outer").sort("Month")
df_final = df_final.with_columns([
    (pl.col("VoteCount") / pl.col("KernelCount")).alias("AvgVotesPerKernel")
])
df_pd = df_final.to_pandas().fillna(0)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Dual Axis ì‹œê°�í™”
fig, ax1 = plt.subplots(figsize=(14, 6))

# ì¢Œì¸¡: ì»¤ë„� ìƒ�ì„± ìˆ˜
color1 = 'tab:blue'
ax1.set_xlabel("ì›”")
ax1.set_ylabel("ì»¤ë„� ìƒ�ì„± ìˆ˜", color=color1)
ax1.plot(df_pd["Month"], df_pd["KernelCount"], color=color1, marker='o', label="ì»¤ë„� ìƒ�ì„± ìˆ˜")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend(loc='upper left')

# ìš°ì¸¡: ì»¤ë„�ë‹¹ í�‰ê·  íˆ¬í‘œ ìˆ˜
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel("ì»¤ë„�ë‹¹ í�‰ê·  íˆ¬í‘œ ìˆ˜", color=color2)
ax2.plot(df_pd["Month"], df_pd["AvgVotesPerKernel"], color=color2, marker='s', label="ì»¤ë„�ë‹¹ í�‰ê·  íˆ¬í‘œ ìˆ˜")
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right')

# ì œëª© ë°� ë ˆì�´ì•„ì›ƒ
plt.title("ì›”ë³„ ì»¤ë„� ìƒ�ì„± ìˆ˜ ë°� ì»¤ë„�ë‹¹ í�‰ê·  íˆ¬í‘œ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


























































































import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('Model_with_kernel_user.parquet', low_memory=True)


# 1. ëª¨ë�¸ ìƒ�ì„±ì�¼ ê¸°ì¤€ ì›” ë‹¨ìœ„ ë³€í™˜ (ì˜ˆ: 2023-06-01)
df_month = df.filter(pl.col("ModelCreationDate").is_not_null()).with_columns(
    pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
)

# 2. ì›”ë³„ ê³ ìœ  ëª¨ë�¸ ìˆ˜ ì§‘ê³„
df_agg = (
    df_month
    .group_by("Month")
    .agg(pl.col("ModelId").n_unique().alias("ModelCount"))
    .sort("Month")
    .to_pandas()
)

# 3. ì„  ê·¸ë�˜í”„ ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_agg, x="Month", y="ModelCount", marker='o')
plt.title("ì›”ë³„ ìƒ�ì„±ë�œ ê³ ìœ  ëª¨ë�¸ ìˆ˜")
plt.xlabel("ì›”")
plt.ylabel("ëª¨ë�¸ ìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ì›” ë‹¨ìœ„ë¡œ ì •ë¦¬
df_month = df.with_columns(
    pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
)

# 2. ìœ ì €ê°€ ìƒ�ì„±í•œ ëª¨ë�¸ ìˆ˜ (ì¡°ì§� ì •ë³´ ì—†ëŠ” ê²½ìš°)
user_counts = (
    df_month
    .filter(pl.col("OrganizationId_Name").is_null())
    .group_by("Month")
    .agg(pl.count("ModelId").alias("ModelCount"))
    .with_columns(pl.lit("User").alias("CreatorType"))
)

# 3. ì¡°ì§�ì�´ ìƒ�ì„±í•œ ëª¨ë�¸ ìˆ˜
org_counts = (
    df_month
    .filter(pl.col("OrganizationId_Name").is_not_null())
    .group_by("Month")
    .agg(pl.count("ModelId").alias("ModelCount"))
    .with_columns(pl.lit("Organization").alias("CreatorType"))
)

# 4. ë³‘í•© ë°� Pandas ë³€í™˜
df_plot = pl.concat([user_counts, org_counts]).to_pandas()

# 5. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_plot,
    x="Month",
    y="ModelCount",
    hue="CreatorType",
    marker="o"
)
plt.title("ìœ ì € vs ì¡°ì§� - ëª¨ë�¸ ìƒ�ì„± ì¶”ì�´")
plt.xlabel("ì›”")
plt.ylabel("ìƒ�ì„±ë�œ ëª¨ë�¸ ìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



df


test = df.filter((pl.col("ModelCreationDate").is_not_null()) & (pl.col('ModelCreationDate').dt.date() > pl.date(2000,1,1)))
test.describe()


df_pd[df_pd["Month"] < "2023-01"]



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. ë‚ ì§œ í•„í„°ë§�ë¶€í„° ì² ì €í•˜ê²Œ
df_filtered = (
    df
    .filter(pl.col("ModelCreationDate").is_not_null())
    .filter(pl.col("ModelCreationDate") >= pl.datetime(2023, 1, 1))
    .with_columns([
        pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
    ])
)

# 2. ìœ ì € ìƒ�ì„± ëª¨ë�¸ ì§‘ê³„
user_df = (
    df_filtered
    .filter(pl.col("ModelOrganizationId").is_null())
    .group_by("Month")
    .agg([
        pl.sum("ModelTotalDownloads").alias("Downloads_User"),
        pl.sum("ModelTotalViews").alias("Views_User"),
        pl.sum("ModelTotalKernels").alias("Kernels_User"),
    ])
)

# 3. ì¡°ì§� ìƒ�ì„± ëª¨ë�¸ ì§‘ê³„
org_df = (
    df_filtered
    .filter(pl.col("ModelOrganizationId").is_not_null())
    .group_by("Month")
    .agg([
        pl.sum("ModelTotalDownloads").alias("Downloads_Org"),
        pl.sum("ModelTotalViews").alias("Views_Org"),
        pl.sum("ModelTotalKernels").alias("Kernels_Org"),
    ])
)

# 4. ë³‘í•© ë°� Pandas ë³€í™˜
merged_df = user_df.join(org_df, on="Month", how="outer").sort("Month")
merged_df = merged_df.filter(pl.col("Month") >= pl.date(2023, 1, 1))
df_pd = merged_df.to_pandas().fillna(0)
df_pd["Month"] = pd.to_datetime(df_pd["Month"])  # for matplotlib

# 5. ì‹œê°�í™” í•¨ìˆ˜
def plot_metric(df, user_col, org_col, metric_name):
    plt.figure(figsize=(14, 6))
    plt.plot(df["Month"], df[user_col], label="User", marker='o')
    plt.plot(df["Month"], df[org_col], label="Organization", marker='s')
    plt.title(f"ì›”ë³„ {metric_name} (ìœ ì € vs ì¡°ì§�)")
    plt.xlabel("ì›”")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6. ì‹¤í–‰
plot_metric(df_pd, "Downloads_User", "Downloads_Org", "ModelTotalDownloads")
plot_metric(df_pd, "Views_User", "Views_Org", "ModelTotalViews")
plot_metric(df_pd, "Kernels_User", "Kernels_Org", "ModelTotalKernels")



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ì›” ë‹¨ìœ„ ë°� CreatorType ìƒ�ì„±
df_month = df.with_columns([
    pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month"),
    pl.when(pl.col("OrganizationId_Name").is_null())
      .then(pl.lit("User"))
      .otherwise(pl.lit("Organization"))
      .alias("CreatorType")
])

# 2. ìœ íš¨í•œ í”„ë ˆì�„ì›Œí�¬ í•„í„°ë§�
df_valid = df_month.filter(
    pl.col("ModelFramework").is_not_null() &
    pl.col("Month").is_not_null()
)

# 3. ì „ì²´ì—�ì„œ ìƒ�ìœ„ 5ê°œ ModelFramework ì¶”ì¶œ
top5_frameworks = (
    df_valid
    .group_by("ModelFramework")
    .agg(pl.count().alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .limit(5)
    .select("ModelFramework")
    .to_series()
    .to_list()
)

# 4. ìƒ�ìœ„ 5ê°œ í”„ë ˆì�„ì›Œí�¬ë§Œ í•„í„°ë§� í›„ ì§‘ê³„
framework_counts = (
    df_valid
    .filter(pl.col("ModelFramework").is_in(top5_frameworks))
    .group_by(["Month", "CreatorType", "ModelFramework"])
    .agg(pl.col('ModelId').n_unique().alias("ModelCount"))
    .sort(["Month", "CreatorType", "ModelFramework"])
    .to_pandas()
)

# 5. ì‹œê°�í™”
plt.figure(figsize=(16, 8))
sns.lineplot(
    data=framework_counts,
    x="Month",
    y="ModelCount",
    hue="ModelFramework",
    style="CreatorType",
    markers=True,
    dashes=False
)
plt.title("ìœ ì €ì™€ ì¡°ì§�ì�˜ ì›”ë³„ ìƒ�ìœ„ 5ê°œ ëª¨ë�¸ í”„ë ˆì�„ì›Œí�¬ ì„ í˜¸ë�„")
plt.xlabel("ì›”")
plt.ylabel("ëª¨ë�¸ ìˆ˜")
plt.tight_layout()
plt.show()



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. ì›” ë‹¨ìœ„ ë°� CreatorType ìƒ�ì„±
df_month = df.with_columns([
    pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month"),
    pl.when(pl.col("OrganizationId_Name").is_null())
      .then(pl.lit("User"))
      .otherwise(pl.lit("Organization"))
      .alias("CreatorType")
])

# 2. ìœ íš¨í•œ í”„ë ˆì�„ì›Œí�¬ í•„í„°ë§�
df_valid = df_month.filter(
    pl.col("ModelFramework").is_not_null() &
    pl.col("Month").is_not_null()
)

# 3. ìƒ�ìœ„ 5ê°œ í”„ë ˆì�„ì›Œí�¬ ì¶”ì¶œ
top5_frameworks = (
    df_valid
    .group_by("ModelFramework")
    .agg(pl.count().alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .limit(5)
    .select("ModelFramework")
    .to_series()
    .to_list()
)

# 4. ìƒ�ìœ„ 5ê°œ í”„ë ˆì�„ì›Œí�¬ë§Œ í•„í„°ë§� í›„ ì§‘ê³„
df_filtered = (
    df_valid
    .filter(pl.col("ModelFramework").is_in(top5_frameworks))
    .group_by(["Month", "CreatorType", "ModelFramework"])
    .agg(pl.col("ModelId").n_unique().alias("ModelCount"))
    .sort(["Month", "CreatorType", "ModelFramework"])
    .to_pandas()
)

# 5. ë�¼ì�¸ê·¸ë�˜í”„ í˜•íƒœë¡œ ì‹œê°�í™”
for creator in ["User", "Organization"]:
    df_sub = df_filtered[df_filtered["CreatorType"] == creator]

    plt.figure(figsize=(16, 8))
    for framework in top5_frameworks:
        df_line = df_sub[df_sub["ModelFramework"] == framework]
        plt.plot(df_line["Month"], df_line["ModelCount"], marker='o', label=framework)

    plt.title(f"{creator}ì�˜ ì›”ë³„ ëª¨ë�¸ í”„ë ˆì�„ì›Œí�¬ ë¶„í�¬ (ìƒ�ìœ„ 5ê°œ)")
    plt.xlabel("ì›”")
    plt.ylabel("ëª¨ë�¸ ìˆ˜")
    plt.grid(True)
    plt.legend(title="ModelFramework")

    # âœ… xì¶• ë‚ ì§œ í�¬ë§· â†’ YYYY-MM
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



import polars as pl

# 1. í•„ìš”í•œ ì»¬ëŸ¼ë§Œ ì„ íƒ�
df_selected = df.select([
    # k í…Œì�´ë¸”
    "KernelId", "KernelVersionId", "KernelMedal", "KernelMedalDate", "SourceModelVariationId",
    
    # m í…Œì�´ë¸”
    "ModelId", "ModelOwnerUserId", "ModelOrganizationId", "ModelCreationDate",
    "ModelTotalViews", "ModelTotalDownloads", "ModelTotalKernels", "ModelVariationId",
    "ModelFramework", "ModelTags", "ModelUserPerVoteDate",
    
    # u í…Œì�´ë¸”
    "UserId", "AchievementType", "Tier", "TierAchievementDate", "Points", "CurrentRanking",
    "HighestRanking", "TotalGold", "TotalSilver", "TotalBronze", "CurrentRankingStatus",
    "HighestRankingStatus", "UserName", "RegisterDate", "PerformanceTier", "FollowingUserId",
    "DaysSinceSignup", "IsActiveTierUser", "FirstAchvDate", "LastAchvDate",
    "DaysSinceLastAchv", "TierProgression", "CumulativePoints", "AvgPointsPerAchv",
    "OrganizationId_JoinDate", "OrganizationId_Name", "OrganizationId_CreationDate",
    "OrganizationId_Industry"
])

# 2. Tags ë©”íƒ€ë�°ì�´í„° ë¡œë“œ
tags_meta = pl.read_parquet("Tags.parquet").select(["Id", "Name"]).rename({"Id": "TagId"})

# 3. ModelTags explode í›„ íƒœê·¸ ì�´ë¦„ ë§¤í•‘
df_exploded = (
    df_selected
    .filter(pl.col("ModelTags").is_not_null() & (pl.col("ModelTags").list.len() > 0))
    .explode("ModelTags")
    .explode("ModelTags")
    .rename({"ModelTags": "TagId"})
    .join(tags_meta, on="TagId", how="left")
)

# (ì„ íƒ�) í•„ìš” ì‹œ df_exploded â†’ ë³‘í•©ë�œ ë¶„ì„� í…Œì�´ë¸”ë¡œ ì‚¬ìš© ê°€ëŠ¥



df_exploded


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns

# # 1. ì›” ë‹¨ìœ„ ì •ë¦¬
# df_tags = df_exploded.with_columns(
#     pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
# ).filter(
#     pl.col("Month").is_not_null() & pl.col("Name").is_not_null()
# )

# # 2. ì›”ë³„ íƒœê·¸ë³„ ê³ ìœ  ModelId ìˆ˜ ì§‘ê³„
# tag_monthly_trend = (
#     df_tags
#     .group_by(["Month", "Name"])
#     .agg(pl.col("ModelId").n_unique().alias("UniqueModelCount"))
#     .sort("Month")
#     .to_pandas()
# )

# # 3. ì „ì²´ ê¸°ê°„ ê¸°ì¤€ ìƒ�ìœ„ 10ê°œ íƒœê·¸ ì¶”ì¶œ
# top10_tags = (
#     tag_monthly_trend
#     .groupby("Name")["UniqueModelCount"]
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
#     .index.tolist()
# )

# # 4. ìƒ�ìœ„ 10ê°œ íƒœê·¸ë§Œ í•„í„°ë§�
# df_plot = tag_monthly_trend[tag_monthly_trend["Name"].isin(top10_tags)]

# # 5. ì‹œê°�í™”
# plt.figure(figsize=(16, 8))
# sns.lineplot(data=df_plot, x="Month", y="UniqueModelCount", hue="Name", marker="o")

# # ë‚ ì§œ í�¬ë§· YYYY-MM
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.title("ì›”ë³„ íƒœê·¸ë³„ ê³ ìœ  ëª¨ë�¸ ìˆ˜ (ìƒ�ìœ„ 10ê°œ íƒœê·¸)")
# plt.xlabel("ì›”")
# plt.ylabel("ê³ ìœ  ëª¨ë�¸ ìˆ˜")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.legend(title="Tag Name", loc="upper left")
# plt.tight_layout()
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 1. ì›” ë‹¨ìœ„ ì •ë¦¬ ë°� 2023-04 ì�´í›„ í•„í„°
df_tags = df_exploded.with_columns(
    pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
).filter(
    (pl.col("Month") >= pl.date(2023, 4, 1)) &
    (pl.col("Name").is_not_null())
)

# 2. ì›”ë³„ íƒœê·¸ë³„ ê³ ìœ  ModelId ìˆ˜ ì§‘ê³„
tag_monthly_trend = (
    df_tags
    .group_by(["Month", "Name"])
    .agg(pl.col("ModelId").n_unique().alias("UniqueModelCount"))
    .sort("Month")
    .to_pandas()
)

# 3. ì „ì²´ ê¸°ê°„ ê¸°ì¤€ ìƒ�ìœ„ 5ê°œ íƒœê·¸ ì¶”ì¶œ
top5_tags = (
    tag_monthly_trend
    .groupby("Name")["UniqueModelCount"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

# 4. ìƒ�ìœ„ 5ê°œ íƒœê·¸ë§Œ í•„í„°ë§�
df_plot = tag_monthly_trend[tag_monthly_trend["Name"].isin(top5_tags)]

# 5. ì‹œê°�í™”
plt.figure(figsize=(16, 8))
sns.lineplot(data=df_plot, x="Month", y="UniqueModelCount", hue="Name", marker="o")

# xì¶• ë‚ ì§œ í�¬ë§·
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("2023-04 ì�´í›„ ì›”ë³„ íƒœê·¸ë³„ ê³ ìœ  ëª¨ë�¸ ìˆ˜ (ìƒ�ìœ„ 5ê°œ íƒœê·¸)")
plt.xlabel("ì›”")
plt.ylabel("ê³ ìœ  ëª¨ë�¸ ìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title="Tag Name", loc="upper left")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import polars as pl

# 1. Month + CreatorType ìƒ�ì„±
df_month_base = (
    df.with_columns([
        pl.col("ModelCreationDate").cast(pl.Date).dt.truncate("1mo").alias("Month"),
        pl.when(pl.col("ModelOrganizationId").is_null())
          .then(pl.lit("User"))
          .otherwise(pl.lit("Organization"))
          .alias("CreatorType")
    ])
    .filter(pl.col("Month").is_not_null())
)

# 2. ì§‘ê³„: ì›”ë³„ + CreatorType ê¸°ì¤€ ì´�í•©
df_monthly = (
    df_month_base
    .group_by(["Month", "CreatorType"])
    .agg([
        pl.sum("ModelTotalDownloads").alias("Downloads"),
        pl.sum("ModelTotalViews").alias("Views"),
        pl.sum("ModelTotalKernels").alias("Kernels")
    ])
    .pivot(values=["Downloads", "Views", "Kernels"], index="Month", columns="CreatorType")
    .sort("Month")
)


# 2023-04 ì�´ì „ í•„í„°ë§�
df_pd = df_monthly.filter(pl.col("Month") < pl.date(2023, 4, 1)).to_pandas().fillna(0)

# ---------- ì‹œê°�í™” í•¨ìˆ˜ ----------
def plot_metric(df, user_col, org_col, metric_name):
    plt.figure(figsize=(14, 6))
    plt.plot(df["Month"], df[user_col], label="User", marker='o')
    plt.plot(df["Month"], df[org_col], label="Organization", marker='s')
    plt.title(f"ì›”ë³„ {metric_name} (2023-04 ì�´ì „)")
    plt.xlabel("ì›”")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)

    # xì¶• í�¬ë§· YYYY-MM
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- 1. Downloads ----------
plot_metric(df_pd, "Downloads_User", "Downloads_Organization", "ModelTotalDownloads")

# ---------- 2. Views ----------
plot_metric(df_pd, "Views_User", "Views_Organization", "ModelTotalViews")

# ---------- 3. Kernels ----------
plot_metric(df_pd, "Kernels_User", "Kernels_Organization", "ModelTotalKernels")
























import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



duckdb.query("SELECT * FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet') LIMIT 10").df().columns


# ë©”ë‹¬ ì†�ë�„ê°€ ì�ˆëŠ” ê²ƒ
query = """
COPY (
WITH kernel_counts AS (
  SELECT KernelVersionAuthorUserId AS UserId, COUNT(*) AS num_kernels, MAX(KernelVersionCreationDate) AS latest_kernel,
  MIN(KernelVersionCreationDate) AS first_kernel
  FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet')
  GROUP BY KernelVersionAuthorUserId
),
forum_counts AS (
  SELECT f1.PostUserId AS UserId,
  COUNT(*) AS num_forum_messages,
  COUNT(f2.real_post_id) AS num_forum_topics,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.TotalMessages ELSE NULL END) AS avg_messages_on_topic,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.TotalViews ELSE NULL END) AS avg_views_on_topic,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.Score ELSE NULL END) AS avg_scores_on_topic,
  MAX(f1.PostDate) AS latest_forum,
  MIN(f1.PostDate) AS first_forum
  FROM read_parquet('Forums_merged.parquet') f1
  LEFT JOIN(
    SELECT Id_topics,
    MIN(Id_messages) AS real_post_id
    FROM read_parquet('Forums_merged.parquet')
    GROUP BY Id_topics
  ) f2
  ON (f1.Id_topics = f2.Id_topics) AND (f1.Id_messages = f2.real_post_id) 
  GROUP BY f1.PostUserId
),
user_dataset_counts AS (
  SELECT
    CreatorUserId_version AS UserId,
    COUNT(DISTINCT DatasetId) AS NumDatasetsContributed,
    MAX(CreationDate_version) AS latest_dataset,
    MIN(CreationDate_version) AS first_dataset
  FROM read_parquet('dataset_clean_merged(dict).parquet')
  GROUP BY CreatorUserId_version
),
submission_with_user AS (
  SELECT
    s.Id AS SubmissionId,
    s.TeamId,
    t.CompetitionId,
    t.UserId,
    s.SubmissionDate
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId
),
user_competition_counts AS (
  SELECT
    UserId,
    COUNT(DISTINCT CompetitionId) AS NumCompetitionsParticipated,
    MAX(SubmissionDate) AS last_comp_submission,
    MIN(SubmissionDate) AS first_comp_submission
  FROM submission_with_user
  GROUP BY UserId
),
VoteCnt_data AS (
  SELECT UserId, COUNT(*) AS VoteCnt_data, MAX(VoteDate) AS last_vote_date_data
  FROM read_parquet('/home/ronny/Downloads/final_project/DatasetVotes.parquet')
  GROUP BY UserId
),
VoteCnt_forum AS (
  SELECT FromUserId AS UserId, COUNT(*) AS VoteCnt_forum, MAX(VoteDate) AS last_vote_date_forum
  FROM read_parquet('/home/ronny/Downloads/final_project/ForumMessageVotes.parquet')
  GROUP BY FromUserId
),
VoteCnt_kernel AS (
  SELECT UserId, COUNT(*) AS VoteCnt_kernel, MAX(VoteDate) AS last_vote_date_kernel
  FROM read_parquet('/home/ronny/Downloads/final_project/KernelVotes.parquet')
  GROUP BY UserId
),
VoteCnt_model AS (
  SELECT UserId, COUNT(*) AS VoteCnt_model, MAX(VoteDate) AS last_vote_date_model
  FROM read_parquet('/home/ronny/Downloads/final_project/ModelVotes.parquet')
  GROUP BY UserId
)
SELECT 
  u.UserId,
  u.UserName,
  u.AchievementType,
  u.Tier,
  u.TierAchievementDate,
  u.Points,
  u.TotalGold,
  u.TotalSilver,
  u.TotalBronze,
  u.RegisterDate,
  u.PerformanceTier,
  u.FollowingUserId,

  kc.num_kernels, kc.latest_kernel, kc.first_kernel,
  fc.num_forum_messages, fc.num_forum_topics, fc.avg_messages_on_topic,
  fc.avg_views_on_topic, fc.avg_scores_on_topic, fc.latest_forum, fc.first_forum,
  c.NumCompetitionsParticipated, c.last_comp_submission, c.first_comp_submission,
  d.NumDatasetsContributed, d.latest_dataset, d.first_dataset,
  vcd.VoteCnt_data, vcd.last_vote_date_data,
  vcf.VoteCnt_forum, vcf.last_vote_date_forum,
  vck.VoteCnt_kernel, vck.last_vote_date_kernel,
  vcm.VoteCnt_model, vcm.last_vote_date_model,

  -- ìƒˆë¡œ ì¶”ê°€ë�œ 8ê°œ í…Œì�´ë¸” ì»¬ëŸ¼
  dfm.FirstDatasetMedalDate,
  cfm.FirstCompetitionMedalDate,
  kfm.FirstKernelMedalDate,
  ffm.FirstForumMedalDate,

  dts.Tier2Date_Data,
  dts.Tier3Date_Data,
  dts.Tier4Date_Data,

  cts.Tier2Date AS Tier2Date_competition,
  cts.Tier3Date AS Tier3Date_competition,
  cts.Tier4Date AS Tier4Date_competition,

  kts.Tier2Date_kernel,
  kts.Tier3Date_kernel,
  kts.Tier4Date_kernel,

  fts.Tier2Date AS Tier2Date_forum,
  fts.Tier3Date AS Tier3Date_forum,
  fts.Tier4Date AS Tier4Date_forum,

  -- ë§ˆì§€ë§‰/ì²« í™œë�™ì�¼ ê³„ì‚°
  GREATEST(kc.latest_kernel, fc.latest_forum, c.last_comp_submission, d.latest_dataset) AS last_activity_date,
  LEAST(kc.first_kernel, fc.first_forum, c.first_comp_submission, d.first_dataset) AS first_activity_date

FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
LEFT JOIN kernel_counts kc ON kc.UserId = u.UserId
LEFT JOIN forum_counts fc ON fc.UserId = u.UserId
LEFT JOIN user_dataset_counts d ON u.UserId = d.UserId
LEFT JOIN user_competition_counts c ON u.UserId = c.UserId
LEFT JOIN VoteCnt_data vcd ON u.UserId = vcd.UserId
LEFT JOIN VoteCnt_forum vcf ON u.UserId = vcf.UserId
LEFT JOIN VoteCnt_kernel vck ON u.UserId = vck.UserId
LEFT JOIN VoteCnt_model vcm ON u.UserId = vcm.UserId

-- ì¶”ê°€ë�œ ë©”ë‹¬ ë°� í‹°ì–´ ì •ë³´
LEFT JOIN read_parquet('dataset_first_medal.parquet') dfm ON u.UserId = dfm.CreatorUserId_data
LEFT JOIN read_parquet('competition_first_medal.parquet') cfm ON u.UserId = cfm.UserId
LEFT JOIN read_parquet('kernel_first_medal.parquet') kfm ON u.UserId = kfm.KernelAuthorUserId
LEFT JOIN read_parquet('forum_first_medal.parquet') ffm ON u.UserId = ffm.PostUserId
LEFT JOIN read_parquet('dataset_tier_speed.parquet') dts ON u.UserId = dts.CreatorUserId_data
LEFT JOIN read_parquet('competition_tier_speed.parquet') cts ON u.UserId = cts.UserId
LEFT JOIN read_parquet('kernel_tiers_speed.parquet') kts ON u.UserId = kts.KernelAuthorUserId
LEFT JOIN read_parquet('forum_tiers_speed.parquet') fts ON u.UserId = fts.PostUserId
) TO 'User_all_with_all.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(query)


test = """
COPY
(
SELECT k.Id_kv AS KernelVersionId,
    k.KernelId,
    k.ScriptLanguageId,
    k.AuthorUserId_kv AS AuthorUserId_KernelVersions,
    k.CreationDate_kv AS CreationDate_KernelVersions,
    k.VersionNumber,
    k.TotalVotes_kv AS TotalVotes_KernelVersions,
    k.IsInternetEnabled,
    k.RunningTimeInMilliseconds,
    k.AcceleratorTypeId,
    k.Id_k KernelId,
    k.AuthorUserId_k AS AuthorUserId_Kernel,
    k.CurrentKernelVersionId,
    k.ForkParentKernelVersionId,
    k.CreationDate_k,
    k.Medal AS KernelMedal,
    k.MedalAwardDate AS KernelMedalAwardDate,
    k.TotalViews As KernelTotalViews,
    k.TotalComments AS KernelTotalComments,
    k.TotalVotes_k AS KernelTotalVotes,
    k.TagId AS KerenlTag,
    k.Label,
    k.Name,
    k.DisplayName,
    k.IsNotebook,
    k.SourceCompetitionId,
    k.SourceDatasetVersionId,
    k.SourceKernelVersionId,
    k.SourceModelVariationId,
    k.UserPerVoteDate AS UserId_VoteDate,
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.CurrentRankingStatus,
    u.HighestRankingStatus,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.IsActiveTierUser,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.DaysSinceLastAchv,
    u.TierProgression,
    u.CumulativePoints,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry,
    f.Id,
    f.Title,
    f.Id_topics,
    f.ForumId,
    f.KernelId,
    f.CreationDate,
    f.LastCommentDate,
    f.Title_topics,
    f.IsSticky,
    f.TotalViews,
    f.Score,
    f.TotalMessages,
    f.TotalReplies,
    f.ActiveDuration,
    f.UnrepliedMessages,
    f.Id_messages,
    f.ForumTopicId,
    f.PostUserId,
    f.PostDate,
    f.ReplyToForumMessageId,
    f.Message,
    f.Medal,
    f.MedalAwardDate,
    f.HasMedal,
    f.ForumMessageId,
    f.FromId_ToId,
    f.ForumMessageId_reacts,
    f.FromUserId,
    f.ReactionType,
    f.ReactionDate
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_with2.parquet') u
LEFT JOIN read_parquet('KernelMerged.parquet') k
ON k.AuthorUserId_kv = u.UserId
LEFT JOIN read_parquet('Forums_merged.parquet') f
ON f.PostUserId = u.UserId
) TO 'User_1to2_with_kernel_forums.parquet' (FORMAT 'parquet');
"""

duckdb.query(test)


test = """
COPY
(
SELECT k.Id_kv AS KernelVersionId,
    k.Id_kv AS KernelVersionId,      -- ì»¤ë„� ì¡´ì�¬ ìœ ë¬´ â†’ í–‰ë�™ìœ¼ë¡œ ë³¼ ìˆ˜ ì�ˆì�Œ
    k.AuthorUserId_k,
    f.Id_messages,                   -- ë©”ì‹œì§€ ID ì¡´ì�¬ â†’ í–‰ë�™ ë°œìƒ�
    f.PostUserId,
    u.UserId,
    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.RegisterDate,
    u.PerformanceTier,
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_with2.parquet') u
LEFT JOIN read_parquet('KernelMerged.parquet') k
ON k.AuthorUserId_kv = u.UserId
LEFT JOIN read_parquet('Forums_merged.parquet') f
ON f.PostUserId = u.UserId
WHERE AchievementType IN ('Discussion', 'Scripts')
) TO 'User_1to2_with_kernel_forums_part1.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(test)


# GPT COMPACT VERSION > í–‰ë�™ë³„ ë§ˆì§€ë§‰ í™œë�™ì�¼ ë¶™ì�„ì�„ > íˆ¬í‘œì�¼ ë°� íˆ¬í‘œíšŸìˆ˜ë�„ ë¶™ì�„

test = """
COPY (
WITH kernel_counts AS (
  SELECT AuthorUserId_kv AS UserId, COUNT(*) AS num_kernels, MAX(CreationDate_kv) AS latest_kernel,
  MIN(CreationDate_kv) AS first_kernel
  FROM read_parquet('KernelMerged.parquet')
  GROUP BY AuthorUserId_kv
),
forum_counts AS (
  SELECT f1.PostUserId AS UserId,
  COUNT(*) AS num_forum_messages,
  COUNT(f2.real_post_id) AS num_forum_topics,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.TotalMessages ELSE NULL END) AS avg_messages_on_topic,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.TotalViews ELSE NULL END) AS avg_views_on_topic,
  AVG(CASE WHEN f2.real_post_id IS NOT NULL THEN f1.Score ELSE NULL END) AS avg_scores_on_topic,
  MAX(f1.PostDate) AS latest_forum,
  MIN(f1.PostDate) AS first_forum
  FROM read_parquet('Forums_merged.parquet') f1
  LEFT JOIN(
    SELECT Id_topics,
    MIN(Id_messages) AS real_post_id
    FROM read_parquet('Forums_merged.parquet')
    GROUP BY Id_topics
  ) f2
  ON (f1.Id_topics = f2.Id_topics) AND (f1.Id_messages = f2.real_post_id) 
  GROUP BY f1.PostUserId
),
user_dataset_counts AS (
  SELECT
    CreatorUserId_version AS UserId,
    COUNT(DISTINCT DatasetId) AS NumDatasetsContributed,
    MAX(CreationDate_version) AS latest_dataset,
    MIN(CreationDate_version) AS first_dataset
  FROM read_parquet('dataset_clean_merged(dict).parquet')
  GROUP BY CreatorUserId_version
),
submission_with_user AS (
  SELECT
    s.Id AS SubmissionId,
    s.TeamId,
    t.CompetitionId,
    t.UserId,
    s.SubmissionDate
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId
),
user_competition_counts AS (
  SELECT
    UserId,
    COUNT(DISTINCT CompetitionId) AS NumCompetitionsParticipated,
    MAX(SubmissionDate) AS last_comp_submission,
    MIN(SubmissionDate) AS first_comp_submission
  FROM submission_with_user
  GROUP BY UserId
),
VoteCnt_data AS (
  SELECT
    UserId,
    COUNT(*) AS VoteCnt_data,
    MAX(VoteDate) AS last_vote_date_data
  FROM read_parquet('/home/ronny/Downloads/final_project/DatasetVotes.parquet')
  GROUP BY UserId
),
VoteCnt_forum AS (
  SELECT
    FromUserId AS UserId,
    COUNT(*) AS VoteCnt_forum,
    MAX(VoteDate) AS last_vote_date_forum
  FROM read_parquet('/home/ronny/Downloads/final_project/ForumMessageVotes.parquet')
  GROUP BY FromUserId
),
VoteCnt_kernel AS (
  SELECT
    UserId,
    COUNT(*) AS VoteCnt_kernel,
    MAX(VoteDate) AS last_vote_date_kernel
  FROM read_parquet('/home/ronny/Downloads/final_project/KernelVotes.parquet')
  GROUP BY UserId
),
VoteCnt_model AS (
  SELECT
    UserId,
    COUNT(*) AS VoteCnt_model,
    MAX(VoteDate) AS last_vote_date_model
  FROM read_parquet('/home/ronny/Downloads/final_project/ModelVotes.parquet')
  GROUP BY UserId
)
SELECT 
  u.UserId,
  u.UserName,
  u.AchievementType,
  u.Tier,
  u.TierAchievementDate,
  u.Points,
  u.TotalGold,
  u.TotalSilver,
  u.TotalBronze,
  u.RegisterDate,
  u.PerformanceTier,
  u.FollowingUserId,

  kc.num_kernels,
  kc.latest_kernel,
  kc.first_kernel,

  fc.num_forum_messages,
  fc.num_forum_topics,
  fc.avg_messages_on_topic,
  fc.avg_views_on_topic,
  fc.avg_scores_on_topic,
  fc.latest_forum,
  fc.first_forum,

  c.NumCompetitionsParticipated,
  c.last_comp_submission,
  c.first_comp_submission,

  d.NumDatasetsContributed,
  d.latest_dataset,
  d.first_dataset,

  vcd.VoteCnt_data,
  vcd.last_vote_date_data,

  vcf.VoteCnt_forum,
  vcf.last_vote_date_forum,

  vck.VoteCnt_kernel,
  vck.last_vote_date_kernel,

  vcm.VoteCnt_model,
  vcm.last_vote_date_model,

    -- âœ… ë§ˆì§€ë§‰ í™œë�™ì�¼ ê³„ì‚°: ì—¬ëŸ¬ ë‚ ì§œ ì¤‘ ê°€ì�¥ ìµœê·¼ ë‚ ì§œ
    GREATEST(
      kc.latest_kernel,
      fc.latest_forum,
      c.last_comp_submission,
    d.latest_dataset
  ) AS last_activity_date,
    LEAST(
      kc.first_kernel,
      fc.first_forum,
      c.first_comp_submission,
    d.first_dataset
    ) AS first_activity_date

FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_with_additional_var.parquet') u
LEFT JOIN kernel_counts kc ON kc.UserId = u.UserId
LEFT JOIN forum_counts fc ON fc.UserId = u.UserId
LEFT JOIN user_dataset_counts d ON u.UserId = d.UserId
LEFT JOIN user_competition_counts c ON u.UserId = c.UserId
LEFT JOIN VoteCnt_data vcd ON u.UserId = vcd.UserId
LEFT JOIN VoteCnt_forum vcf ON u.UserId = vcf.UserId
LEFT JOIN VoteCnt_kernel vck ON u.UserId = vck.UserId
LEFT JOIN VoteCnt_model vcm ON u.UserId = vcm.UserId
) TO 'User_all_with_all.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');

"""

duckdb.query(test)


# ë§ˆì§€ë§‰ ì�´ê±°ë¡œ ë¬¶ì�„ ìˆ˜ ì�ˆê² ë‹¤!!! 

test = """
COPY
(
WITH kernel_counts AS (
  SELECT AuthorUserId_kv AS UserId, COUNT(*) AS num_kernels
  FROM read_parquet('KernelMerged.parquet')
  GROUP BY AuthorUserId_kv
),
forum_counts AS (
  SELECT PostUserId AS UserId, COUNT(*) AS num_forum_posts
  FROM read_parquet('Forums_merged.parquet')
  GROUP BY PostUserId
)
SELECT 
  u.UserId,
  u.UserName,
  u.AchievementType,
  u.Tier,
  u.TierAchievementDate,
  u.Points,
  u.TotalGold,
  u.TotalSilver,
  u.TotalBronze,
  u.RegisterDate,
  u.PerformanceTier,
  kc.num_kernels,
  fc.num_forum_posts,
  c.NumCompetitionsParticipated,
  d.NumDatasetsContributed
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
LEFT JOIN kernel_counts kc ON kc.UserId = u.UserId
LEFT JOIN forum_counts fc ON fc.UserId = u.UserId
LEFT JOIN read_parquet('user_competition_aggregations.parquet') c ON u.UserId = c.UserId
LEFT JOIN read_parquet('dataset_aggregation.parquet') d ON u.UserId = d.UserId

) TO 'User_1to4_with_all_gpt_mine.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(test)


compquery = """
COPY (
  SELECT
    s.Id AS SubmissionId,
    s.TeamId,
    t.CompetitionId,
    t.UserId,
    s.SubmissionDate,
    c.TotalSubmissions,
    c.AlgorithmCategory,
    c.Title,
    c.HostSegmentTitle,
    u.UserName,
    u.PerformanceTier,
    u.Tier,
    u.AchievementType
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId
  LEFT JOIN read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Organization_Algorithm_cat_withType.parquet') c
    ON t.CompetitionId = c.Id
  LEFT JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_with_additional_var.parquet') u 
    ON t.UserId = u.UserId
) TO 'comp_difficulty.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
"""

duckdb.query(compquery)


duckdb.query("SELECT * FROM read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Organization_Algorithm_cat_withType.parquet') LIMIT 10").df().columns


test = """
SELECT COUNT(*)
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_with2.parquet') u
LEFT JOIN read_parquet('KernelMerged.parquet') k
ON k.AuthorUserId_kv = u.UserId
LEFT JOIN read_parquet('Forums_merged.parquet') f
ON f.PostUserId = u.UserId
WHERE AchievementType IN ('Discussion', 'Scripts')
"""

duckdb.query(test)


test = """
WITH kernel_counts AS (
  SELECT AuthorUserId_kv AS UserId, COUNT(*) AS num_kernels
  FROM read_parquet('KernelMerged.parquet')
  GROUP BY AuthorUserId_kv
),
forum_counts AS (
  SELECT PostUserId AS UserId, COUNT(*) AS num_forum_posts
  FROM read_parquet('Forums_merged.parquet')
  GROUP BY PostUserId
)
SELECT 
COUNT(*)
FROM read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_with2.parquet') u
LEFT JOIN kernel_counts kc ON kc.UserId = u.UserId
LEFT JOIN forum_counts fc ON fc.UserId = u.UserId
WHERE u.AchievementType IN ('Discussion', 'Scripts')
"""

duckdb.query(test)


df = pl.from_arrow(duckdb.query("""  SELECT
    s.Id AS SubmissionId,
    s.TeamId,
    t.CompetitionId,
    t.UserId,
    s.SubmissionDate
  FROM read_parquet('Submissions_clean.parquet') s
  LEFT JOIN read_parquet('Team_merged.parquet') t
    ON s.TeamId = t.TeamId""").to_arrow_table())


comp = """
SELECT 
"""


df


df.group_by('UserId').agg(pl.col('CompetitionId').n_unique()).sort('CompetitionId')


df = pl.scan_parquet('User_1to2_with_kernel_forums_part1.parquet', low_memory=True)


df = pl.read_parquet('User_1to4_with_all.parquet')
df


df.describe()


df.filter(df['PerformanceTier']==1)['UserId'].n_unique()


df.filter(df['PerformanceTier']==2)['UserId'].n_unique()


df.filter((df['Tier']==1) & df['num_kernels'].is_null())['UserId'].n_unique()


df.filter((df['Tier']==1) & df['num_forum_messages'].is_null())['UserId'].n_unique()


df.filter((df['Tier']==2) & df['num_kernels'].is_null())['UserId'].n_unique()


df.filter((df['Tier']==2) & df['num_kernels'].is_null())


df.filter((df['Tier']==2) & df['num_forum_messages'].is_null())['UserId'].n_unique()


# Tierë³„ í�‰ê·  ê³„ì‚°
agg_df = (
    df.group_by("PerformanceTier")
    .agg([
        pl.col("num_kernels").mean().alias("avg_kernels"),
        pl.col("num_forum_messages").mean().alias("avg_forum_posts"),
        pl.col('NumCompetitionsParticipated').mean().alias('avg_comp_join'),
        pl.col('NumDatasetsContributed').mean().alias('avg_dataset_created')
    ])
    .sort("PerformanceTier")
)

tiers = agg_df["PerformanceTier"].to_list()
avg_kernels = agg_df["avg_kernels"].to_list()
avg_forums = agg_df["avg_forum_posts"].to_list()
avg_comp_join = agg_df['avg_comp_join'].to_list()
avg_dataset_created = agg_df['avg_dataset_created'].to_list()

x = np.arange(len(tiers))
width = 0.2

fig, ax = plt.subplots(figsize=(15, 6))
bars1 = ax.bar(x - 1.5*width, avg_kernels, width, label='ì»¤ë„� í�‰ê·  ìˆ˜')
bars2 = ax.bar(x - 0.5*width, avg_forums, width, label='í�¬ëŸ¼ ê¸€ í�‰ê·  ìˆ˜')
bars3 = ax.bar(x + 0.5*width, avg_comp_join, width, label ='í�‰ê·  ëŒ€íšŒ ì°¸ê°€ ìˆ˜')
bars4 = ax.bar(x + 1.5*width, avg_dataset_created, width, label = 'í�‰ê·  ë�°ì�´í„°ì…‹ ì œì�‘ ìˆ˜')

def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x ì¢Œí‘œ: ë§‰ëŒ€ ì¤‘ì•™
            height,                            # y ì¢Œí‘œ: ë§‰ëŒ€ ë†’ì�´ ë°”ë¡œ ìœ„
            f'{height:.2f}',                  # í‘œì‹œí•  ê°’ (ì†Œìˆ˜ì � 2ì��ë¦¬)
            ha='center',                      # ê°€ìš´ë�° ì •ë ¬
            va='bottom',                      # í…�ìŠ¤íŠ¸ê°€ ë§‰ëŒ€ ìœ„ë¡œ ì˜¬ë�¼ê°€ê²Œ
            fontsize=9
        )

# ì˜ˆì‹œ: ê°� ë§‰ëŒ€ì—� ê°’ ì¶”ê°€
add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)

ax.set_xlabel('í‹°ì–´')
ax.set_ylabel('í�‰ê·  ìˆ˜')
ax.set_title('í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ í�‰ê·  í™œë�™ ìˆ˜')
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.legend()

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

# 1. í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ìµœëŒ€ê°’ ê³„ì‚°
metrics = ["num_kernels", "num_forum_messages", "NumCompetitionsParticipated", "NumDatasetsContributed"]
agg_df = (
    df
    .select(["UserId", "PerformanceTier"] + metrics)
    .group_by("PerformanceTier")
    .agg([
        pl.col("num_kernels").max().alias("max_kernels"),
        pl.col("num_forum_messages").max().alias("max_forums"),
        pl.col("NumCompetitionsParticipated").max().alias("max_comp"),
        pl.col("NumDatasetsContributed").max().alias("max_dataset")
    ])
    .sort("PerformanceTier")
)

# 2. ì‹œê°�í™”ìš© ë�°ì�´í„° ì¤€ë¹„
tiers = agg_df["PerformanceTier"].to_list()

max_values = pd.DataFrame({
    'ì»¤ë„�': agg_df["max_kernels"].to_list(),
    'í�¬ëŸ¼': agg_df["max_forums"].to_list(),
    'ëŒ€íšŒ': agg_df["max_comp"].to_list(),
    'ë�°ì�´í„°ì…‹': agg_df["max_dataset"].to_list()
}, index=tiers)

labels_all = ['ì»¤ë„�', 'í�¬ëŸ¼', 'ëŒ€íšŒ', 'ë�°ì�´í„°ì…‹']
colors_map = {
    'ì»¤ë„�': '#4e79a7',
    'í�¬ëŸ¼': '#f28e2c',
    'ëŒ€íšŒ': '#e15759',
    'ë�°ì�´í„°ì…‹': '#76b7b2'
}

# 3. ì •ë ¬ë�œ ìŠ¤íƒ� ë§‰ëŒ€ ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(14, 30))
x = np.arange(len(tiers))
width = 0.6

# ì´ˆê¸° ë°”ë‹¥ ìœ„ì¹˜
bottom = np.zeros(len(tiers))

# ê°� í‹°ì–´ë³„ë¡œ ì •ë ¬í•˜ì—¬ ëˆ„ì � ë§‰ëŒ€ ê·¸ë¦¬ê¸°
for tier_idx, tier in enumerate(tiers):
    tier_data = max_values.loc[tier]
    # ê°’ ê¸°ì¤€ìœ¼ë¡œ ì˜¤ë¦„ì°¨ìˆœ ì •ë ¬
    sorted_items = tier_data.sort_values(ascending=False).items()
    tier_bottom = 0
    for label, val in sorted_items:
        color = colors_map[label]
        bar = ax.bar(
            x[tier_idx],
            val,
            width,
            bottom=tier_bottom,
            color=color,
            label=label if tier_idx == 0 else None
        )

        # í…�ìŠ¤íŠ¸ ìœ„ì¹˜ ì¡°ì •
        if val >= 10:
            text_y = tier_bottom + val / 2
            va = 'center'
            fontsize = 10
        elif val >= 3:
            text_y = tier_bottom + val + 1
            va = 'bottom'
            fontsize = 8
        else:
            text_y = None  # ìƒ�ë�µ

        if text_y is not None:
            ax.text(
                x[tier_idx],
                text_y,
                f"{int(val)}",
                ha='center',
                va=va,
                fontsize=fontsize,
                color='black'
            )

        tier_bottom += val

# 4. ê¾¸ë¯¸ê¸°
ax.set_xlabel('í‹°ì–´', fontsize=12)
ax.set_ylabel('ìˆ˜', fontsize=12)
ax.set_title('í‹°ì–´ë³„ ì»¤ë„�/í�¬ëŸ¼/ëŒ€íšŒ/ë�°ì�´í„°ì…‹ ìµœëŒ€ìˆ˜ ëˆ„ì � ë§‰ëŒ€ê·¸ë�˜í”„ (ì�‘ì�€ ê°’ì�´ ìœ„)', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(tiers, rotation=45)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(False, axis='x')

legend_patches = [Patch(color=colors_map[l], label=l) for l in labels_all]
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()



agg_df


tier_avg = (
    df.group_by("PerformanceTier")
    .agg([
        pl.col("TotalGold").mean().alias("avg_gold"),
        pl.col("TotalSilver").mean().alias("avg_silver"),
        pl.col("TotalBronze").mean().alias("avg_bronze"),
    ])
    .sort("PerformanceTier")
)

import matplotlib.pyplot as plt
import numpy as np

tiers = tier_avg["PerformanceTier"].to_list()
avg_gold = tier_avg["avg_gold"].to_list()
avg_silver = tier_avg["avg_silver"].to_list()
avg_bronze = tier_avg["avg_bronze"].to_list()

x = np.arange(len(tiers))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars_gold = ax.bar(x - width, avg_gold, width, label='í�‰ê·  Gold')
bars_silver = ax.bar(x, avg_silver, width, label='í�‰ê·  Silver')
bars_bronze = ax.bar(x + width, avg_bronze, width, label='í�‰ê·  Bronze')

ax.set_xlabel('í‹°ì–´')
ax.set_ylabel('í�‰ê·  ë©”ë‹¬ ìˆ˜')
ax.set_title('í‹°ì–´ë³„ TotalGold, TotalSilver, TotalBronze í�‰ê·  ë¹„êµ�')
ax.set_xticks(x)
ax.set_xticklabels(tiers, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd

selected_cols = ["UserId", "UserName", "PerformanceTier", "AchievementType", "Points", "Tier"]


# polars -> pandas ë³€í™˜ ì˜ˆì‹œ (í˜•ë‹˜ ìƒ�í™©ì—� ë§�ê²Œ)
df_pd = (
    df
    .select(selected_cols)
    .filter(pl.col('PerformanceTier').is_not_null())
    .filter(pl.col('Tier').is_in([1, 2, 3, 4]))
    .to_pandas()
)

perf_tiers = sorted(df_pd['PerformanceTier'].dropna().unique())
achievement_types = ['Discussion', 'Competitions', 'Datasets', 'Scripts']
tier_order = [1, 2, 3, 4]
colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2']

fig, axs = plt.subplots(1, len(perf_tiers), figsize=(18, 9), sharey=False)

for i, ptier in enumerate(perf_tiers):
    ax = axs[i]
    subset = df_pd[df_pd['PerformanceTier'] == ptier]

    data = (
        subset.groupby(['AchievementType', 'Tier'])
        .size()
        .reset_index(name='count')
        .query("AchievementType in @achievement_types and Tier in @tier_order")
    )

    pivot_df = data.pivot(index='Tier', columns='AchievementType', values='count').fillna(0)
    pivot_df = pivot_df[achievement_types]  # ì»¬ëŸ¼ ìˆœì„œ ë§�ì¶”ê¸°

    bottom = pd.Series([0] * len(pivot_df), index=pivot_df.index)
    bars_list = []
    for j, atype in enumerate(achievement_types):
        bars = ax.bar(
            pivot_df.index,
            pivot_df[atype],
            bottom=bottom,
            color=colors[j],
            label=atype if i == len(perf_tiers) - 1 else None,  # ë§ˆì§€ë§‰ ê·¸ë�˜í”„ì—�ë§Œ ë²”ë¡€ í‘œì‹œ
            width=0.6,
        )
        bars_list.append(bars)
        bottom += pivot_df[atype]

    # ê°’ í‘œì‹œ (ëˆ„ì � ë§‰ëŒ€ ì¤‘ì•™ì—� í‘œì‹œ)
    for bars in bars_list:
        for rect in bars:
            height = rect.get_height()
            if height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + height / 2,
                    f'{int(height)}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='black',
                )

    ax.set_title(f'í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ {ptier}')
    ax.set_xlabel('Tier')
    ax.set_xticks(tier_order)
    ax.set_xticklabels(tier_order)
    if i == 0:
        ax.set_ylabel('ìœ ì € ìˆ˜')

# ë²”ë¡€ë¥¼ ë§ˆì§€ë§‰ ê·¸ë�˜í”„ ì˜¤ë¥¸ìª½ ë°–ì—� ë°°ì¹˜
axs[-1].legend(title='AchievementType', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ AchievementType ë‚´ Tier ë¶„í�¬ (ëˆ„ì � ë§‰ëŒ€)', fontsize=16)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# í•„ìš”í•œ ì»¬ëŸ¼ ì„ íƒ�
selected_cols = ["UserId", "UserName", "PerformanceTier", "AchievementType", "Points", "Tier"]
df_pd = df.filter((pl.col('Tier')!=0)).select(selected_cols).to_pandas()

# jitter ì¶”ê°€
df_pd["TierJitter"] = df_pd["Tier"] + np.random.uniform(-0.2, 0.2, size=len(df_pd))

# AchievementType ê³ ì • palette ì„¤ì •
all_types = sorted(df_pd["AchievementType"].dropna().unique())
palette_fixed = dict(zip(all_types, sns.color_palette("Set2", n_colors=len(all_types))))

perf_tiers = sorted(df_pd['PerformanceTier'].dropna().unique())

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=False)
axs = axs.flatten()

for i, tier in enumerate(perf_tiers):
    ax = axs[i]
    subset = df_pd[df_pd['PerformanceTier'] == tier]

    sns.scatterplot(
        data=subset,
        x="TierJitter",
        y="Points",
        hue="AchievementType",
        palette=palette_fixed,
        alpha=0.7,
        ax=ax
    )

    ax.set_title(f'PerformanceTier {tier}')
    ax.set_xlabel("Tier")
    ax.set_ylabel("Points")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([1, 2, 3, 4])

    if i == len(perf_tiers) - 1:
        ax.legend(title="AchievementType", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()

plt.suptitle("PerformanceTierë³„ AchievementTypeì�˜ Tier vs Points ë¶„í�¬ (jitter ì �ìš©)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# í•„ìš”í•œ ì»¬ëŸ¼ ì„ íƒ�
selected_cols = ["UserId", "UserName", "PerformanceTier", "AchievementType", "Points", "Tier"]
df_pd = df.filter((pl.col('AchievementType')!='Competitions') & (pl.col('Tier')!=0)).select(selected_cols).to_pandas()

# jitter ì¶”ê°€
df_pd["TierJitter"] = df_pd["Tier"] + np.random.uniform(-0.2, 0.2, size=len(df_pd))

# AchievementType ê³ ì • palette ì„¤ì •
all_types = sorted(df_pd["AchievementType"].dropna().unique())
palette_fixed = dict(zip(all_types, sns.color_palette("Set2", n_colors=len(all_types))))

perf_tiers = sorted(df_pd['PerformanceTier'].dropna().unique())

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=False)
axs = axs.flatten()

for i, tier in enumerate(perf_tiers):
    ax = axs[i]
    subset = df_pd[df_pd['PerformanceTier'] == tier]

    sns.scatterplot(
        data=subset,
        x="TierJitter",
        y="Points",
        hue="AchievementType",
        palette=palette_fixed,
        alpha=0.5,
        ax=ax
    )

    ax.set_title(f'PerformanceTier {tier}')
    ax.set_xlabel("Tier")
    ax.set_ylabel("Points")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([1, 2, 3, 4])

    if i == len(perf_tiers) - 1:
        ax.legend(title="AchievementType", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()

plt.suptitle("PerformanceTierë³„ AchievementTypeì�˜ Tier vs Points ë¶„í�¬ (jitter ì �ìš©)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()


df.filter(pl.col('AchievementType')=='Discussion').describe()


df.filter(pl.col('AchievementType')=='Scripts').describe()


df.filter(pl.col('AchievementType')=='Datasets').describe()


df.filter(pl.col('AchievementType')=='Competitions').describe()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. í•„í„°ë§� ë°� ë³€í™˜
df_filtered = (
    df
    .filter(pl.col("TierAchievementDate") != pl.date(1980, 1, 1))
    .filter(pl.col("PerformanceTier").is_not_null())
    .filter(pl.col("Tier").is_in([1, 2, 3, 4]))
    .select(["PerformanceTier", "TierAchievementDate", "Tier"])
    .to_pandas()
)

# 2. ë‚ ì§œ ì •ê·œí™” (ì—°-ì›” ë‹¨ìœ„)
df_filtered["TierAchievementDate"] = pd.to_datetime(df_filtered["TierAchievementDate"])
df_filtered["YearMonth"] = df_filtered["TierAchievementDate"].dt.to_period("M").dt.to_timestamp()

# 3. ê·¸ë£¹ë³„ ê°œìˆ˜ ì§‘ê³„
grouped = (
    df_filtered
    .groupby(["PerformanceTier", "YearMonth", "Tier"])
    .size()
    .reset_index(name="Count")
)

# 4. ì‹œê°�í™”
perf_tiers = sorted(grouped["PerformanceTier"].unique())
palette = sns.color_palette("Set2", n_colors=4)

fig, axs = plt.subplots(len(perf_tiers), 1, figsize=(14, 4 * len(perf_tiers)))

if len(perf_tiers) == 1:
    axs = [axs]

for i, ptier in enumerate(perf_tiers):
    ax = axs[i]
    subset = grouped[grouped["PerformanceTier"] == ptier]
    
    # í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ì—� ë”°ë�¼ í‘œì‹œí•  í‹°ì–´ ì œí•œ
    allowed_tiers = list(range(1, int(ptier) + 1))

    for j, tier in enumerate(allowed_tiers):
        tier_data = subset[subset["Tier"] == tier]
        ax.plot(
            tier_data["YearMonth"],
            tier_data["Count"],
            label=f'Tier {tier}',
            color=palette[tier - 1],
            linewidth=2
        )

    ax.set_title(f"PerformanceTier {ptier} - Tier ë³€í™” ì¶”ì�´")
    ax.set_ylabel("Count")

    # PerformanceTierê°€ 2 ì�´ìƒ�ì�¼ ë•Œë§Œ ë²”ë¡€ í‘œì‹œ
    if ptier > 1:
        ax.legend(title="Tier")

axs[-1].set_xlabel("Year-Month")
plt.suptitle("PerformanceTierë³„ TierAchievementDate ê¸°ì¤€ Tier ì¦�ê°€ ì¶”ì�´", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



from datetime import datetime, timedelta

now = datetime.now() - timedelta(days=1)

max_date = df['TierAchievementDate'].max().date()

newbie_date = max_date - timedelta(days=1)

df.filter((df['RegisterDate'].dt.date() == newbie_date)).select(
    pl.col('UserId').n_unique().alias('ë‰´ë¹„ ì�¸ì›�ìˆ˜')
)


# 1. RegisterDate ê¸°ì¤€ ì�¼ë³„ ê³ ìœ  UserId ìˆ˜ ì„¸ê¸°
daily_counts = (
    df
    .select(["UserId", "RegisterDate"])
    .filter(~pl.col("RegisterDate").is_null())
    .with_columns([
        pl.col("RegisterDate").cast(pl.Date)
    ])
    .group_by("RegisterDate")
    .agg([
        pl.col("UserId").n_unique().alias("UniqueUserCount")
    ])
    .sort("RegisterDate")
)

# 2. Pandas ë³€í™˜ (ì‹œê°�í™” ìš©ë�„)
pd_df = daily_counts.to_pandas()

# 3. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
plt.plot(pd_df["RegisterDate"], pd_df["UniqueUserCount"], linestyle='-')
plt.title("ë‚ ì§œë³„ ì‹ ê·œ ê°€ì�… ìœ ì € ìˆ˜")
plt.xlabel("ê°€ì�… ì�¼ì��")
plt.ylabel("ê³ ìœ  ìœ ì € ìˆ˜")
plt.tight_layout()
plt.show()



import polars as pl
import matplotlib.pyplot as plt

# 1. RegisterDate â†’ 'yyyy-mm' í˜•ì‹� ë¬¸ì��ì—´ë¡œ ë³€í™˜
monthly_counts = (
    df
    .select(["UserId", "RegisterDate"])
    .filter(~pl.col("RegisterDate").is_null())
    .with_columns([
        pl.col("RegisterDate").cast(pl.Date),
        pl.col("RegisterDate").dt.strftime("%Y-%m").alias("RegisterMonth")
    ])
    .group_by("RegisterMonth")
    .agg([
        pl.col("UserId").n_unique().alias("UniqueUserCount")
    ])
    .sort("RegisterMonth")
)

# 2. Pandas ë³€í™˜
pd_df = monthly_counts.to_pandas()

# 3. ì‹œê°�í™”
plt.figure(figsize=(60, 10))
plt.plot(pd_df["RegisterMonth"], pd_df["UniqueUserCount"], marker='o', linestyle='-')
plt.title("ì›”ë³„ ì‹ ê·œ ê°€ì�… ìœ ì € ìˆ˜")
plt.xlabel("ê°€ì�… ì›”")
plt.ylabel("ê³ ìœ  ìœ ì € ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df = pl.read_parquet('User_1to4_with_all_withdate_gpt.parquet', low_memory=True)
df


font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # í˜•ë‹˜ ì‹œìŠ¤í…œì—�ì„œ ì¡´ì�¬í•˜ëŠ” ê²½ë¡œ í™•ì�¸
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# 1. ë§ˆì§€ë§‰ í™œë�™ì�¼ ê³„ì‚°
df_activity = df.with_columns([
    pl.max_horizontal([
        pl.col("latest_kernel"),
        pl.col("latest_forum"),
        pl.col("last_comp_submission"),
        pl.col("latest_dataset")
    ]).alias("last_activity_date")
])

# 2. í™œë�™ ì—°ë�„ ì¶”ì¶œ
df_activity = df_activity.with_columns(
    pl.col("last_activity_date").dt.strftime("%Y").alias("last_activity_year")
)

# 3. ìœ ì € ë‹¨ìœ„ë¡œ ìœ ì�¼í•˜ê²Œ ìœ ì§€
df_unique = (
    df_activity.sort("last_activity_date", descending=True)
    .unique(subset="UserId", keep="first")
)

# â�— 4. ê²°ì¸¡ì¹˜ ì œê±°
df_unique = df_unique.filter(pl.col("last_activity_year").is_not_null())

# 5. ì§‘ê³„ (PerformanceTier + ì—°ë�„ë³„)
agg_df = (
    df_unique.group_by(["PerformanceTier", "last_activity_year"])
    .agg(pl.len().alias("user_count"))
    .sort(["last_activity_year", "PerformanceTier"])
)

# 6. Pandas ë³€í™˜
pdf = agg_df.to_pandas()

# 7. í”¼ë²— í…Œì�´ë¸” ìƒ�ì„± (í–‰: ì—°ë�„, ì—´: í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´)
pivot = (
    pdf.groupby(["last_activity_year", "PerformanceTier"])["user_count"]
    .sum()
    .reset_index()
    .pivot(index="last_activity_year", columns="PerformanceTier", values="user_count")
    .fillna(0)
)

# 8. ì—°ë�„ ì •ë ¬ (ìˆ«ì��í˜• ì—°ë�„ë¡œ ë³€í™˜í•´ì„œ ì •ë ¬)
pivot.index = pivot.index.astype(str)
pivot = pivot.loc[sorted(pivot.index, key=lambda y: int(y))]

# 9. ì‹œê°�í™” (matplotlib ì „ìš©)
plt.figure(figsize=(12, 6))
bar_width = 0.15
years = pivot.index.tolist()
tiers = pivot.columns.tolist()
x = range(len(years))

# ë§‰ëŒ€ ê·¸ë�˜í”„
for i, tier in enumerate(tiers):
    plt.bar(
        [pos + i * bar_width for pos in x],
        pivot[tier],
        width=bar_width,
        label=str(tier)
    )

plt.title("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ë§ˆì§€ë§‰ í™œë�™ ì—°ë�„ ë¶„í�¬", fontsize=15)
plt.xlabel("ì—°ë�„", fontsize=12)
plt.ylabel("ìœ ì € ìˆ˜", fontsize=12)
plt.xticks(
    [pos + bar_width * (len(tiers) - 1) / 2 for pos in x],
    years,
    rotation=45
)
plt.legend(title="í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
plt.tight_layout()
plt.show()


df.filter((df['last_comp_submission'].is_null()) & (df['latest_dataset'].is_null()) & (df['latest_forum'].is_null()) & (df['latest_kernel'].is_null()) & (df['PerformanceTier']==3))
# 2017ë…„ë�„ ëŒ€íšŒì—� ì°¸ì—¬í–ˆì�Œ.
# https://www.kaggle.com/zyzzhaoyuzhe
# ê·¼ë�° ì™œ 2019ë…„ì—� í‹°ì–´ê°€ ì˜¬ë�¼ê°”ì�„ê¹Œ...? > ì§„ì§œ ë©”ë‹¬ì�´ ìˆ˜ë�™ê¸‰ì—¬ì‹œìŠ¤í…œì�¸ê±´ê°€...?



df.filter((df['last_comp_submission'].is_null()) & (df['latest_dataset'].is_null()) & (df['latest_forum'].is_null()) & (df['latest_kernel'].is_null()) & (df['PerformanceTier']==2))
# https://www.kaggle.com/fernando2 
# ì–˜ë�„ ëŒ€íšŒ ë‘�ë²ˆ ì°¸ì—¬í•œ ê¸°ë¡�ì�´ ë³´ì�„. ë�™ë©”ë‹¬ ë‘�ë²ˆ. ê·¼ë�°, ì�´ê±´ 15ë…„ì „ ë�°ì�´í„°ë�¼ì„œ 2010ë…„ì�¸ë�°...ë�°ì�´í„°ì…‹ë�„ í™œë�™ ê¸°ë¡�ì�´ ë”°ë¡œ ì•ˆë³´ì�´ëŠ”ë�° ì–´ì§¸ì„œ ë�°ì�´í„°ì…‹ë§Œë§Œ 2019ë…„ì—� ì—…ë�°ì�´íŠ¸ê°€ ë�˜ì—ˆì�„ê¹Œ?
# 


# ë�°ì�´í„°ì…‹ì—� ê´€í•œê²Œ ë‚˜ì¤‘ì—� ìƒ�ê²¨ì„œ 2019ë…„ 11ì›” 6ì�¼ì�¸ê°€ë´„?
df.filter(df['TierAchievementDate'].dt.date() == pl.datetime(2019, 11, 6))['AchievementType'].value_counts()



# ê·¸ëŸ¼ ì�´ ì „ì—� ìƒ�ì„±í•œ ê³„ì •ì—� í•œí•´ì„œ í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ê°€ ì–´ë–»ê²Œ ì²˜ë¦¬ë�˜ì—ˆì�„ê¹Œ? > ìœ„ë�‘ ë˜‘ê°™ì�€ê±¸ ë³´ë‹ˆ ë‹¤ ì�´ì „ì—� ë§Œë“¤ì–´ì§„ ê³„ì •ì�´ ìš”ëª¨ì–‘ì�¸ë“¯?
df.filter((df['TierAchievementDate'].dt.date() == pl.date(2019, 11, 6)) & (df['RegisterDate'].dt.date() <= pl.date(2019, 11, 6)))['AchievementType'].value_counts()

#ê²°ë¡  : ë�°ì�´í„°ì…‹ì�˜ í‹°ì–´ê°€ ë‚˜ì˜¤ê¸° ì „ì—� í�¼í�¬ë¨¼ìŠ¤í‹°ì–´ê°€ ì�´ë¯¸ 1 ì�´ìƒ�ì�¸ ì‚¬ë�Œë“¤ì—� í•œí•´ì„œëŠ” ë�°ì�´í„°ì…‹ì�˜ ì²˜ë¦¬ê°€ 2019ë…„ 11ì›” 6ì�¼ì—� 1ë¡œ ë�˜ì—ˆë‹¤.> ê·¸ëŸ¼ ë‹¤ë¥¸ê±´...?


df.filter((df['RegisterDate'].dt.date() <= pl.date(2019, 11, 6)) & (df['AchievementType']!=['Datasets'])
          & (df['last_comp_submission'].is_null()) & (df['latest_dataset'].is_null()) & (df['latest_forum'].is_null()) & (df['latest_kernel'].is_null()))['PerformanceTier'].value_counts()


# ê²½ì§„ëŒ€íšŒì—� ì°¸ì—¬í•œ ì‚¬ë�Œ ì¤‘ submissionì�´ ì—†ëŠ” ì‚¬ë�Œë§Œ ì�´ë ‡ê²Œ ë�˜ëŠ” ë“¯.ì „ë¶€ ê²½ì§„ëŒ€íšŒë¡œ ì�¸í•´ í‹°ì–´ê°€ í�¼í�¬ë¨¼ìŠ¤í‹°ì–´ê°€ 2ì�¸ ì‚¬ë�Œë“¤ì�¸ë�°, last_comp_submissionì�´ ê²°ì¸¡ì¹˜ì�„.
# https://www.kaggle.com/competitions/nips-2017-non-targeted-adversarial-attack
# ìœ„ì�˜ ë§�í�¬ëŠ” ìº�ê¸€ì�˜ ì •ìƒ�ì �ì�¸ ì œì¶œ ì ˆì°¨ë¥¼ ë”°ë¥´ì§€ ì•ŠëŠ”ë‹¤ê³  í•¨. ê·¸ë�˜ì„œ ê²°ì¸¡ì�¸ë“¯?
# https://www.kaggle.com/enesmakalic/competitions
# ëª‡ê°€ì§€ ì¼€ì�´ìŠ¤ë¥¼ ë´¤ì�„ ë•Œ TierAchivementDateê°€ 2016ë…„ 7ì›” 15ì—� ìƒ�ê²¼ëŠ”ë�°, ê·¸ ì „ì—� ëŒ€íšŒì—� ì°¸ì—¬í•œ ê¸°ë¡�ì�´ ê²°ì¸¡ì¹˜ë¡œ ë�˜ì–´ì�ˆì–´ì„œ ê·¸ëŸ°ë“¯?

df.filter((df['RegisterDate'].dt.date() <= pl.date(2019, 11, 6)) 
          & (df['last_comp_submission'].is_null()) & (df['latest_dataset'].is_null()) & (df['latest_forum'].is_null()) & (df['latest_kernel'].is_null()) & (df['PerformanceTier']==2) & (df['Tier']==2)
          )


# ì—¬ê¸°ê¹Œì§€ í•˜ê³  VOTE ì¶”ê°€ê°€
df.filter(df['Tier']==0).to_pandas()


df.filter(df['last_activity_date'].is_null())['Tier'].value_counts()


# 1. ë§ˆì§€ë§‰ í™œë�™ì�¼ ê³„ì‚°
df2 = df.with_columns([
    pl.max_horizontal([
        pl.col("latest_kernel"),
        pl.col("latest_forum"),
        pl.col("last_comp_submission"),
        pl.col("latest_dataset")
    ]).alias("last_activity_date")
])

# 2. ë‚ ì§œ ì°¨ì�´ ê³„ì‚° (ë“±ë¡�ì�¼ë¡œë¶€í„°)
df2 = df2.with_columns([
    (pl.col("last_activity_date") - pl.col("RegisterDate"))
    .dt.total_days()
    .cast(pl.Int32)
    .alias("days_to_last_activity")
])

# 3. ê²°ì¸¡ì¹˜ ì œê±°
df_valid = df2.filter(
    (pl.col("days_to_last_activity").is_not_null()) &
    (pl.col("PerformanceTier").is_not_null()) &
    (pl.col("UserId").is_not_null())
)

# 4. ìœ ì €ë³„ ê³ ìœ  row ìœ ì§€
df_unique = df_valid.unique(subset=["UserId", "PerformanceTier", "days_to_last_activity"])

# 5. í™œë�™ ê¸°ê°„ êµ¬ê°„í™”
df_unique = df_unique.with_columns(
    pl.when(pl.col("days_to_last_activity") < 31).then(pl.lit("0~30ì�¼"))
    .when(pl.col("days_to_last_activity") < 91).then(pl.lit("1~3ê°œì›”"))
    .when(pl.col("days_to_last_activity") < 181).then(pl.lit("3~6ê°œì›”"))
    .when(pl.col("days_to_last_activity") < 366).then(pl.lit("6ê°œì›”~1ë…„"))
    .when(pl.col("days_to_last_activity") < 1096).then(pl.lit("1~3ë…„"))
    .when(pl.col("days_to_last_activity") < 1826).then(pl.lit("3~5ë…„"))
    .when(pl.col("days_to_last_activity") < 2920).then(pl.lit("5~8ë…„"))
    .otherwise(pl.lit("8ë…„ ì�´ìƒ�"))
    .alias("í™œë�™ê¸°ê°„êµ¬ê°„")
)

# 6. í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ëª©ë¡� ì •ì�˜
tiers = [1, 2, 3, 4]

# 7. ì‹œê°�í™”
fig, axes = plt.subplots(len(tiers), 1, figsize=(10, 5 * len(tiers)))

for ax, tier in zip(axes, tiers):
    sub_df = df_unique.filter(pl.col("PerformanceTier") == tier)
    total_users = sub_df.select("UserId").n_unique()

    count_df = (
        sub_df.group_by("í™œë�™ê¸°ê°„êµ¬ê°„")
        .agg(pl.n_unique("UserId").alias("user_count"))
        .sort("í™œë�™ê¸°ê°„êµ¬ê°„")
        .with_columns(
            (pl.col("user_count") / total_users * 100).alias("percent")
        )
        .to_pandas()
    )

    ax.bar(count_df["í™œë�™ê¸°ê°„êµ¬ê°„"], count_df["percent"], color='royalblue')
    ax.set_title(f"í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ {tier}", fontsize=14)
    ax.set_ylabel("ë¹„ìœ¨ (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y')

axes[-1].set_xlabel("ê³„ì • ìƒ�ì„±ì�¼ë¡œë¶€í„° ë§ˆì§€ë§‰ í™œë�™ê¹Œì§€ ê¸°ê°„", fontsize=12)

plt.tight_layout()
plt.show()


# 1. ë§ˆì§€ë§‰ í™œë�™ì�¼ ê³„ì‚°
df1 = df.with_columns([
    pl.max_horizontal([
        pl.col("latest_kernel"),
        pl.col("latest_forum"),
        pl.col("last_comp_submission"),
        pl.col("latest_dataset")
    ]).alias("last_activity_date")
])

# 2. ë‚ ì§œ ì°¨ì�´ ê³„ì‚° (ì�¼ ë‹¨ìœ„)
df1 = df1.with_columns([
    (pl.col("last_activity_date") - pl.col("TierAchievementDate")).dt.total_days().cast(pl.Int32).alias("days_to_last_activity")
])

# 3. ê²°ì¸¡ì¹˜ ë°� í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ê²°ì¸¡ ì œê±°
df_valid = df1.filter(
    (pl.col("days_to_last_activity").is_not_null()) &
    (pl.col("PerformanceTier").is_not_null()) &
    (pl.col("UserId").is_not_null())
)

# 4. ìœ ì €ë³„ ê³ ìœ  rowë§Œ ë‚¨ê¸°ê¸°
df_unique = df_valid.unique(subset=["UserId", "PerformanceTier", "days_to_last_activity"])

# 5. í™œë�™ ê¸°ê°„ êµ¬ê°„í™”
df_unique = df_unique.with_columns(
    pl.when(pl.col("days_to_last_activity") < 31).then(pl.lit("0~30ì�¼"))
    .when(pl.col("days_to_last_activity") < 91).then(pl.lit("1~3ê°œì›”"))
    .when(pl.col("days_to_last_activity") < 181).then(pl.lit("3~6ê°œì›”"))
    .when(pl.col("days_to_last_activity") < 366).then(pl.lit("6ê°œì›”~1ë…„"))
    .when(pl.col("days_to_last_activity") < 1096).then(pl.lit("1~3ë…„"))
    .when(pl.col("days_to_last_activity") < 1826).then(pl.lit("3~5ë…„"))
    .when(pl.col("days_to_last_activity") < 2920).then(pl.lit("5~8ë…„"))
    .otherwise(pl.lit("8ë…„ ì�´ìƒ�"))
    .alias("í™œë�™ê¸°ê°„êµ¬ê°„")
)

# 6. í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ëª©ë¡� ì •ì�˜ (ìˆœì„œ ë³´ì�¥)
tiers = [1, 2, 3, 4]

# 7. ì„œë¸Œí”Œë¡¯ ìƒ�ì„±
fig, axes = plt.subplots(len(tiers), 1, figsize=(10, 5 * len(tiers)))

for ax, tier in zip(axes, tiers):
    # í•´ë‹¹ í‹°ì–´ ìœ ì € ë�°ì�´í„° í•„í„°ë§�
    sub_df = df_unique.filter(pl.col("PerformanceTier") == tier)

    total_users = sub_df.select("UserId").n_unique()
    
    # êµ¬ê°„ë³„ ìœ ì € ìˆ˜ ë¹„ìœ¨ ê³„ì‚°
    count_df = (
        sub_df.group_by("í™œë�™ê¸°ê°„êµ¬ê°„")
        .agg(pl.n_unique("UserId").alias("user_count"))
        .sort("í™œë�™ê¸°ê°„êµ¬ê°„")
        .with_columns(
            (pl.col("user_count") / total_users * 100).alias("percent")
        )
        .to_pandas()
    )

    # ê·¸ë�˜í”„
    ax.bar(count_df["í™œë�™ê¸°ê°„êµ¬ê°„"], count_df["percent"], color='royalblue')
    ax.set_title(f"í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ {tier}", fontsize=14)
    ax.set_ylabel("ë¹„ìœ¨ (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y')

axes[-1].set_xlabel("í‹°ì–´ ë‹¬ì„±ì�¼ë¡œë¶€í„° ë§ˆì§€ë§‰ í™œë�™ê¹Œì§€ ê¸°ê°„", fontsize=12)

plt.tight_layout()
plt.show()


# âœ… NumVotes ê³„ì‚°
df_votes = df.with_columns([
    (
        pl.coalesce(pl.col("VoteCnt_data"), pl.lit(0)) +
        pl.coalesce(pl.col("VoteCnt_forum"), pl.lit(0)) +
        pl.coalesce(pl.col("VoteCnt_model"), pl.lit(0)) +
        pl.coalesce(pl.col("VoteCnt_kernel"), pl.lit(0))
    ).alias("NumVotes")
])

# âœ… í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ í�‰ê· ê°’ ì§‘ê³„
agg_df = (
    df_votes.group_by("PerformanceTier")
    .agg([
        pl.mean("num_kernels").alias("ì»¤ë„�"),
        pl.mean("num_forum_messages").alias("í�¬ëŸ¼"),
        pl.mean("NumCompetitionsParticipated").alias("ëŒ€íšŒ"),
        pl.mean("NumDatasetsContributed").alias("ë�°ì�´í„°ì…‹"),
        pl.mean("NumVotes").alias("íˆ¬í‘œ")
    ])
    .sort("PerformanceTier")
    .to_pandas()
)

# âœ… ì‹œê°�í™”: í–‰ë�™ í•­ëª©ë³„ë¡œ í‹°ì–´ í�‰ê·  ë³´ì—¬ì£¼ê¸° (ì„œë¸Œí”Œë¡¯)
categories = ["ì»¤ë„�", "í�¬ëŸ¼", "ëŒ€íšŒ", "ë�°ì�´í„°ì…‹", "íˆ¬í‘œ"]
tiers = agg_df["PerformanceTier"].tolist()

fig, axes = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 4 * len(categories)))

for i, category in enumerate(categories):
    ax = axes[i]
    values = agg_df[category].values
    ax.bar(tiers, values, color='steelblue')
    ax.set_title(f"{category} ìˆ˜ - í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ í�‰ê· ", fontsize=14)
    ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´", fontsize=12)
    ax.set_ylabel("í�‰ê·  ìˆ˜", fontsize=12)
    ax.set_xticks(tiers)
    ax.set_xticklabels([str(t) for t in tiers])

plt.tight_layout()
plt.show()


vote_columns = ["VoteCnt_data", "VoteCnt_forum", "VoteCnt_kernel", "VoteCnt_model"]

means = {}
for col in vote_columns:
    mean_val = df.select(pl.col(col)).drop_nulls().mean()[0, 0]  # ì»¬ëŸ¼ë³„ null ì œê±° í›„ í�‰ê· 
    means[col] = mean_val

# Pandas DataFrame ë³€í™˜
import pandas as pd
vote_means_pd = pd.DataFrame.from_dict(means, orient='index', columns=['í�‰ê·  íˆ¬í‘œ ìˆ˜'])
vote_means_pd.index.name = 'ì¹´í…Œê³ ë¦¬'

# ì‹œê°�í™”
plt.figure(figsize=(8, 5))
bars = plt.bar(vote_means_pd.index, vote_means_pd['í�‰ê·  íˆ¬í‘œ ìˆ˜'], color='cornflowerblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f"{height:.2f}", 
             ha='center', va='bottom', fontsize=10)

plt.title("ì¹´í…Œê³ ë¦¬ë³„ í�‰ê·  íˆ¬í‘œ ìˆ˜ (ê²°ì¸¡ì¹˜ ì œê±° í›„)", fontsize=14)
plt.ylabel("í�‰ê·  íˆ¬í‘œ ìˆ˜", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. nullì�„ 0ìœ¼ë¡œ ì±„ìš°ê¸° (ë¶„ëª¨/ë¶„ì�� ë‘˜ ë‹¤)
df_nonnull = df.with_columns([
    pl.col("VoteCnt_data").fill_null(0),
    pl.col("VoteCnt_forum").fill_null(0),
    pl.col("VoteCnt_kernel").fill_null(0),
    pl.col("NumDatasetsContributed").fill_null(0),
    pl.col("num_forum_messages").fill_null(0),
    pl.col("num_kernels").fill_null(0),
])

# 2. ë¹„ìœ¨ ê³„ì‚° (ë¶„ëª¨ê°€ 0ì�¸ ê²½ìš°ëŠ” NaN ì²˜ë¦¬)
df_nonnull = df_nonnull.with_columns([
    (pl.when(pl.col("NumDatasetsContributed") > 0)
        .then(pl.col("VoteCnt_data") / pl.col("NumDatasetsContributed"))
        .otherwise(None)
     ).alias("ratio_data"),

    (pl.when(pl.col("num_forum_messages") > 0)
        .then(pl.col("VoteCnt_forum") / pl.col("num_forum_messages"))
        .otherwise(None)
     ).alias("ratio_forum"),

    (pl.when(pl.col("num_kernels") > 0)
        .then(pl.col("VoteCnt_kernel") / pl.col("num_kernels"))
        .otherwise(None)
     ).alias("ratio_kernel"),
])

# 3. í�‰ê·  ë¹„ìœ¨ ê³„ì‚° (null ì œì™¸)
ratio_data_mean = df_nonnull.select(pl.col("ratio_data")).drop_nulls().mean()[0, 0]
ratio_forum_mean = df_nonnull.select(pl.col("ratio_forum")).drop_nulls().mean()[0, 0]
ratio_kernel_mean = df_nonnull.select(pl.col("ratio_kernel")).drop_nulls().mean()[0, 0]

# 4. ì‹œê°�í™”
labels = ['Dataset Votes', 'Forum Votes', 'Kernel Votes']
means = [ratio_data_mean, ratio_forum_mean, ratio_kernel_mean]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, means, color='skyblue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom')

plt.title("ì „ì²´ í–‰ë�™ ëŒ€ë¹„ íˆ¬í‘œ ë¹„ìœ¨ í�‰ê· ")
plt.ylabel("í�‰ê·  íˆ¬í‘œ ë¹„ìœ¨")
plt.ylim(0, max(means) * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# 1. AchievementType í�¬í•¨í•œ í‹°ì–´ ë‹¬ì„±ê¹Œì§€ ì�¼ ìˆ˜ ê³„ì‚°
df_tier_pd = (
    df
    .filter(
        (pl.col("TierAchievementDate").dt.date() != pl.date(1980, 1, 1)) &
        (pl.col("TierAchievementDate").is_not_null()) &
        (pl.col("RegisterDate").is_not_null()) &
        (pl.col("PerformanceTier").is_not_null()) &
        (pl.col("AchievementType").is_not_null())
    )
    .with_columns([
        (pl.col("TierAchievementDate") - pl.col("RegisterDate"))
        .dt.total_days()
        .cast(pl.Int32)
        .alias("days_to_achieve_tier")
    ])
    .select(["AchievementType", "PerformanceTier", "days_to_achieve_tier"])
    .to_pandas()
)

# 2. ì‹œê°�í™” - AchievementTypeë³„ë¡œ ë¶„í• ë�œ ë°•ìŠ¤í”Œë¡¯
import seaborn as sns
import matplotlib.pyplot as plt

unique_types = df_tier_pd["AchievementType"].unique()
n_types = len(unique_types)

fig, axs = plt.subplots(n_types, 1, figsize=(10, 6 * n_types), sharey=True)

for i, achv_type in enumerate(unique_types):
    ax = axs[i]
    subset = df_tier_pd[df_tier_pd["AchievementType"] == achv_type]

    sns.boxplot(
        data=subset,
        x="PerformanceTier",
        y="days_to_achieve_tier",
        palette="Set2",
        showfliers=False,
        ax=ax
    )

    # â¬‡ï¸� ì�´ ë¶€ë¶„ì�´ ë£¨í”„ ì•ˆì—� ì�ˆì–´ì•¼ ì •í™•í•œ í†µê³„ ê³„ì‚°ì�´ ë�©ë‹ˆë‹¤
    group_stats = subset.groupby("PerformanceTier")["days_to_achieve_tier"].agg(["mean", "median"])

    for tier, row in group_stats.iterrows():
        y_mean = row["mean"]
        y_median = row["median"]

        mean_offset = 90
        median_offset = -90

        ax.text(
            x=tier - 1,
            y=y_median + mean_offset,
            s=f"í�‰ê· : {y_mean:.0f}ì�¼",
            ha="center",
            fontsize=10,
            color="blue"
        )
        ax.text(
            x=tier - 1,
            y=y_median + median_offset,
            s=f"ì¤‘ì•™ê°’: {y_median:.0f}ì�¼",
            ha="center",
            fontsize=10,
            color="darkgreen"
        )
        

    ax.set_title(f"{achv_type} - í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ë‹¬ì„± ì†Œìš” ì‹œê°„", fontsize=14)
    ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
    ax.set_ylabel("ì†Œìš” ì�¼ìˆ˜")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



df_tier_pd[df_tier_pd['days_to_achieve_tier']==0]


suspicious = (
    df
    .filter(
        (pl.col("PerformanceTier") >= 4) &
        (pl.col("TierAchievementDate").dt.date() != pl.date(1980, 1, 1)) &
        (pl.col("TierAchievementDate").is_not_null()) &
        (pl.col("RegisterDate").is_not_null()) &
        ((pl.col("TierAchievementDate") - pl.col("RegisterDate")).dt.total_days() == 0)
    )
    .select([
        "UserName", "AchievementType", "PerformanceTier", "RegisterDate", "TierAchievementDate"
    ])
)

suspicious

suspicious_users = (
    df
    .filter(
        (pl.col("PerformanceTier") >= 2) &
        (pl.col("TierAchievementDate").dt.date() != pl.date(1980, 1, 1)) &
        (pl.col("RegisterDate").is_not_null()) &
        (pl.col("TierAchievementDate").is_not_null()) &
        ((pl.col("TierAchievementDate") - pl.col("RegisterDate")).dt.total_days() == 0)
    )
    .select(["UserId", "RegisterDate"])
    .unique()
)



import matplotlib.pyplot as plt
import seaborn as sns

# 1. í�‰ê·  í�¬ì�¸íŠ¸ ì§‘ê³„ (ê²°ì¸¡ì¹˜ ì œì™¸)
agg_df = (
    df.filter(
        (pl.col("PerformanceTier").is_not_null()) &
        (pl.col("AchievementType").is_not_null()) &
        (pl.col("Points").is_not_null())
    )
    .group_by(["AchievementType", "PerformanceTier"])
    .agg(pl.mean("Points").alias("avg_points"))
    .sort(["AchievementType", "PerformanceTier"])
    .to_pandas()
)

# 2. ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�… ëª©ë¡�
achievement_types = agg_df["AchievementType"].unique()

# 3. ì„œë¸Œí”Œë¡¯ ìƒ�ì„± (ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�…ë³„)
fig, axes = plt.subplots(len(achievement_types), 1, figsize=(12, 5 * len(achievement_types)))

if len(achievement_types) == 1:
    axes = [axes]

for ax, ach_type in zip(axes, achievement_types):
    data_sub = agg_df[agg_df["AchievementType"] == ach_type]
    sns.barplot(
        data=data_sub,
        x="PerformanceTier",
        y="avg_points",
        palette="muted",
        ax=ax
    )
    ax.set_title(f"ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�…: {ach_type} - í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ í�‰ê·  í�¬ì�¸íŠ¸")
    ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
    ax.set_ylabel("í�‰ê·  í�¬ì�¸íŠ¸")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# 1. í�‰ê·  í�¬ì�¸íŠ¸ ì§‘ê³„
agg_df = (
    df.filter(
        (pl.col("PerformanceTier").is_not_null()) &
        (pl.col("AchievementType").is_not_null()) &
        (pl.col("Points").is_not_null())
    )
    .group_by(["AchievementType", "PerformanceTier"])
    .agg(pl.mean("Points").alias("avg_points"))
    .to_pandas()
)

# 2. AchievementTypeë³„ë¡œ ì •ê·œí™” (í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ë‚´ í�‰ê·  í�¬ì�¸íŠ¸ ëŒ€ë¹„ ë¹„ìœ¨)
agg_df["norm_points"] = agg_df.groupby("AchievementType")["avg_points"].transform(
    lambda x: x / x.sum() * 100
)

# 3. ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�… ëª©ë¡�
achievement_types = agg_df["AchievementType"].unique()

# 4. ì„œë¸Œí”Œë¡¯ ìƒ�ì„± (ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�…ë³„)
fig, axes = plt.subplots(len(achievement_types), 1, figsize=(12, 5 * len(achievement_types)))

if len(achievement_types) == 1:
    axes = [axes]

for ax, ach_type in zip(axes, achievement_types):
    data_sub = agg_df[agg_df["AchievementType"] == ach_type]
    sns.barplot(
        data=data_sub,
        x="PerformanceTier",
        y="norm_points",
        palette="coolwarm",
        ax=ax
    )
    ax.set_title(f"ì–´ì¹˜ë¸Œë¨¼íŠ¸ íƒ€ì�…: {ach_type} - í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ì •ê·œí™” í�‰ê·  í�¬ì�¸íŠ¸ (%)")
    ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
    ax.set_ylabel("í�¬ì�¸íŠ¸ ë¹„ìœ¨ (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()



import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. ì˜¤ëŠ˜ ê¸°ì¤€ ë‚ ì§œ (ë�°ì�´í„°ê°€ ì–¸ì œ ê¸°ì¤€ì�¸ì§€ ëª¨ë¥´ë©´ max ë‚ ì§œë¡œ ëŒ€ì²´ ê°€ëŠ¥)
today = datetime.today()

# 2. last_activity_date ì»¬ëŸ¼ì�´ datetime íƒ€ì�…ì�¸ì§€ í™•ì�¸, ì•„ë‹ˆë©´ ë³€í™˜ í•„ìš”
# (ì˜ˆì‹œ: pl.col("last_activity_date").cast(pl.Datetime) ë“±)

# 3. êµ¬ê°„ ê³„ì‚°
df_intervals = df.with_columns([
    (pl.lit(today) - pl.col("last_activity_date")).dt.total_days().cast(pl.Int32).alias("days_since_last_activity")
    ])

# 4. êµ¬ê°„ë³„ ë�¼ë²¨ë§�
df_intervals = df_intervals.with_columns(
    pl.when(pl.col("days_since_last_activity") <= pl.lit(30)).then(pl.lit("1ë‹¬ ì�´ë‚´"))
    .when(pl.col("days_since_last_activity") <= pl.lit(365)).then(pl.lit("1ë…„ ì�´ë‚´"))
    .when(pl.col("days_since_last_activity") <= pl.lit(365*3)).then(pl.lit("3ë…„ ì�´ë‚´"))
    .when(pl.col("days_since_last_activity") < pl.lit(365*5)).then(pl.lit("5ë…„ ë¯¸ë§Œ"))
    .otherwise(pl.lit("5ë…„ ì�´ìƒ�"))
    .alias("í™œë�™ê¸°ê°„êµ¬ê°„")
)


# 5. êµ¬ê°„ë³„ ìœ ì € ìˆ˜ ì§‘ê³„ (ì¤‘ë³µ ìœ ì € ì œê±°ìš©ìœ¼ë¡œ UserId ê³ ìœ ê°’ ì‚¬ìš©)
agg_counts = (
    df_intervals.select(["UserId", "í™œë�™ê¸°ê°„êµ¬ê°„"])
    .unique()
    .group_by("í™œë�™ê¸°ê°„êµ¬ê°„")
    .agg(pl.n_unique("UserId").alias("user_count"))
    .sort("í™œë�™ê¸°ê°„êµ¬ê°„")
    .to_pandas()
)

# 6. ë¹„ìœ¨ ê³„ì‚°
total_users = agg_counts["user_count"].sum()
agg_counts["percent"] = agg_counts["user_count"] / total_users * 100

# 7. íŒŒì�´ì°¨íŠ¸ ì‹œê°�í™”
plt.figure(figsize=(8, 8))
plt.pie(
    agg_counts["user_count"],
    labels=agg_counts["í™œë�™ê¸°ê°„êµ¬ê°„"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "w"}
)
plt.title("ë§ˆì§€ë§‰ í™œë�™ì�¼ ê¸°ì¤€ ìœ ì € ë¹„ìœ¨ ë¶„í�¬")
plt.show()



import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# 1. ê¸°ì¤€ì�¼ ê³„ì‚°
today = datetime.today()

# 2. í™œë�™ êµ¬ê°„ ê³„ì‚°
df_labeled = df.with_columns([
    (pl.lit(today) - pl.col("last_activity_date")).dt.total_days().cast(pl.Int32).alias("days_since_last_activity")
])

# 3. í™œë�™êµ¬ê°„ ë�¼ë²¨ë§� (pl.litë¡œ ëª…í™•í•˜ê²Œ ë¬¸ì��ì—´ ì²˜ë¦¬)
df_labeled = df_labeled.with_columns(
    pl.when(pl.col("days_since_last_activity") <= 365).then(pl.lit("1ë…„ ì�´ë‚´"))
    .when((pl.col("days_since_last_activity") > 365) & (pl.col("days_since_last_activity") <= 365 * 3)).then(pl.lit("3ë…„ ì�´ë‚´"))
    .when((pl.col("days_since_last_activity") > 365 * 3) & (pl.col("days_since_last_activity") < 365 * 5)).then(pl.lit("5ë…„ ë¯¸ë§Œ"))
    .otherwise(pl.lit("5ë…„ ì�´ìƒ�"))
    .alias("í™œë�™êµ¬ê°„")
)

# 4. ìœ ì € ì¤‘ë³µ ì œê±° í›„ í�¼í�¬ë¨¼ìŠ¤í‹°ì–´ + í™œë�™êµ¬ê°„ ë³„ ìˆ˜ ê³„ì‚°
df_summary = (
    df_labeled.select(["UserId", "PerformanceTier", "í™œë�™êµ¬ê°„"])
    .unique()
    .group_by(["PerformanceTier", "í™œë�™êµ¬ê°„"])
    .agg(pl.count("UserId").alias("user_count"))
    .to_pandas()
)

# 5. í”¼ë²— (í�ˆíŠ¸ë§µìš© í�¬ë§·)
heatmap_df = df_summary.pivot_table(
    index="PerformanceTier",
    columns="í™œë�™êµ¬ê°„",
    values="user_count",
    fill_value=0
)

# 6. ì—´ ìˆœì„œ ê³ ì •
heatmap_df = heatmap_df[["1ë…„ ì�´ë‚´", "3ë…„ ì�´ë‚´", "5ë…„ ë¯¸ë§Œ", "5ë…„ ì�´ìƒ�"]]

# 7. í�ˆíŠ¸ë§µ ì‹œê°�í™”
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='gray'
)
plt.title("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ë§ˆì§€ë§‰ í™œë�™ êµ¬ê°„ë³„ ìœ ì € ìˆ˜", fontsize=14)
plt.xlabel("í™œë�™ êµ¬ê°„")
plt.ylabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
plt.tight_layout()
plt.show()



df.filter(
    (pl.col("RegisterDate").dt.year() == 2019) &
    (pl.col("TierAchievementDate").dt.month() == 11)
)



# 1. 2019ë…„ ë�°ì�´í„° í•„í„°ë§� (RegisterDateì™€ TierAchievementDate ëª¨ë‘�)
df_2019 = df.filter(
    (pl.col("RegisterDate").dt.year() == 2019) |
    (pl.col("TierAchievementDate").dt.year() == 2019)
)

# 2. RegisterDate ì›”ë³„ ê³ ìœ  ìœ ì € ìˆ˜ ì§‘ê³„
register_counts = (
    df_2019.filter(pl.col("RegisterDate").dt.year() == 2019)
    .with_columns(pl.col("RegisterDate").dt.month().alias("month"))
    .group_by("month")
    .agg(pl.col("UserId").n_unique().alias("register_user_count"))
    .sort("month")
    .to_pandas()
)

# 3. TierAchievementDate ì›”ë³„ ê³ ìœ  ìœ ì € ìˆ˜ ì§‘ê³„
tier_counts = (
    df_2019.filter(pl.col("TierAchievementDate").dt.year() == 2019)
    .with_columns(pl.col("TierAchievementDate").dt.month().alias("month"))
    .group_by("month")
    .agg(pl.col("UserId").n_unique().alias("tier_user_count"))
    .sort("month")
    .to_pandas()
)

# 4. ì›” ë¦¬ìŠ¤íŠ¸ (1~12)
months = list(range(1, 13))

# 5. ê·¸ë�˜í”„ ê·¸ë¦¬ê¸°
plt.figure(figsize=(12, 6))
plt.plot(months, register_counts.set_index("month").reindex(months)["register_user_count"], label="RegisterDate ìœ ì € ìˆ˜", marker='o')
plt.plot(months, tier_counts.set_index("month").reindex(months)["tier_user_count"], label="TierAchievementDate ìœ ì € ìˆ˜", marker='o')

plt.xticks(months)
plt.xlabel("2019ë…„ ì›”")
plt.ylabel("ê³ ìœ  ìœ ì € ìˆ˜")
plt.title("2019ë…„ ì›”ë³„ RegisterDate ë°� TierAchievementDate ê¸°ì¤€ ìœ ì € ìˆ˜")
plt.legend()
plt.grid(True)
plt.show()


df


import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. ê°€ì�… í›„ ì²« í™œë�™ê¹Œì§€ ì�¼ìˆ˜ ê³„ì‚°
df_delay = (
    df
    .filter(
        pl.col("RegisterDate").is_not_null() &
        pl.col("first_activity_date").is_not_null() &
        pl.col("PerformanceTier").is_not_null()
    )
    .with_columns([
        (pl.col("first_activity_date") - pl.col("RegisterDate"))
        .dt.total_days()
        .cast(pl.Int32)
        .alias("days_to_first_activity")
    ])
    .filter(pl.col("days_to_first_activity") >= 0)
    .select(["PerformanceTier", "days_to_first_activity"])
    .to_pandas()
)

# 2. ë°•ìŠ¤í”Œë¡¯ ì‹œê°�í™”
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    data=df_delay,
    x="PerformanceTier",
    y="days_to_first_activity",
    palette="Set3",
    showfliers=False
)

# 3. ì¤‘ì•™ê°’ ê³„ì‚° ë°� í…�ìŠ¤íŠ¸ í‘œì‹œ
medians = (
    df_delay.groupby("PerformanceTier")["days_to_first_activity"]
    .median()
)

for tick, median_val in enumerate(medians):
    ax.text(
        tick,
        median_val + 1,
        f'{int(median_val)}ì�¼',
        ha='center',
        va='bottom',
        color='black',
        fontsize=10,
        fontweight='bold'
    )

# 4. ê¾¸ë¯¸ê¸°
plt.xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
plt.ylabel("ê°€ì�… í›„ ì²« í™œë�™ê¹Œì§€ ì†Œìš” ì�¼ìˆ˜")
plt.title("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ê°€ì�… í›„ ì²« í™œë�™ê¹Œì§€ ê±¸ë¦° ì‹œê°„")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_all_withdate_gpt.parquet', low_memory=True)
df


df = df.explode('FollowingUserId')


df['FollowingUserId'].count()


df.columns


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# 1. íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ìˆ˜ (userê°€ íŒ”ë¡œì�‰í•œ ë‹¤ë¥¸ ì‚¬ë�Œ ê¸°ì¤€)
following_counts = (
    df
    .filter(pl.col("FollowingUserId").is_not_null())
    .group_by(["PerformanceTier", "UserId"])
    .agg(pl.count("FollowingUserId").alias("num_following"))
)

# 2. íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ìˆ˜ (ë‹¤ë¥¸ ì‚¬ë�Œì�´ í•´ë‹¹ userë¥¼ íŒ”ë¡œì�‰í•œ íšŸìˆ˜)
followed_counts = (
    df
    .filter(pl.col("FollowingUserId").is_not_null())
    .group_by("FollowingUserId")
    .agg(pl.count("UserId").alias("num_followed"))
    .rename({"FollowingUserId": "UserId"})
)

# 3. ìœ ì €ì •ë³´ ê°€ì ¸ì˜¤ê¸° (UserId, UserName, PerformanceTier)
user_info = df.select(["UserId", "UserName", "PerformanceTier"]).unique()

# 4. íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ë�°ì�´í„°ì—� ìœ ì € ì •ë³´ ì¡°ì�¸
following_with_info = following_counts.join(user_info, on="UserId", how="left")

# 5. íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ë�°ì�´í„°ì—� ìœ ì € ì •ë³´ ì¡°ì�¸
followed_with_info = followed_counts.join(user_info, on="UserId", how="left")

# 6. í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ìƒ�ìœ„ 10ëª… ë½‘ê¸° í•¨ìˆ˜ (apply ì—†ì�´, Noneê°’ í•„í„°ë§� ì¶”ê°€)
def top_n_per_tier(df_pl, count_col, n=10):
    tiers = (
        df_pl
        .filter(pl.col("PerformanceTier").is_not_null())
        .select("PerformanceTier")
        .unique()
        .to_series()
        .to_list()
    )
    dfs = []
    for tier in tiers:
        filtered = df_pl.filter(pl.col("PerformanceTier") == tier)
        sorted_df = filtered.sort(count_col, descending=True).head(n)
        dfs.append(sorted_df)
    return pl.concat(dfs)

top10_following_by_tier = top_n_per_tier(following_with_info, "num_following")
top10_followed_by_tier = top_n_per_tier(followed_with_info, "num_followed")

# 7. Pandas ë³€í™˜
top10_following_pd = top10_following_by_tier.select(["PerformanceTier", "UserName", "num_following"]).to_pandas()
top10_followed_pd = top10_followed_by_tier.select(["PerformanceTier", "UserName", "num_followed"]).to_pandas()

# 8. í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ ìˆœì„œ ì§€ì •
tier_order = [1, 2, 3, 4]
top10_following_pd["PerformanceTier"] = pd.Categorical(top10_following_pd["PerformanceTier"], categories=tier_order, ordered=True)
top10_followed_pd["PerformanceTier"] = pd.Categorical(top10_followed_pd["PerformanceTier"], categories=tier_order, ordered=True)

# 9. ì‹œê°�í™” í•¨ìˆ˜ ìˆ˜ì • (ìˆœì„œëŒ€ë¡œ, ê²½ê³  ì œê±°)
def plot_top10(df_pd, count_col, title):
    df_pd = df_pd.sort_values("PerformanceTier")
    tiers = df_pd["PerformanceTier"].cat.categories
    for tier in tiers:
        data = df_pd[df_pd["PerformanceTier"] == tier].copy()
        # UserNameì�„ num_following/num_followed ê°’ ë‚´ë¦¼ì°¨ìˆœìœ¼ë¡œ ì¹´í…Œê³ ë¦¬ ìˆœì„œ ì§€ì •
        order = data.sort_values(count_col, ascending=False)["UserName"]
        data["UserName"] = pd.Categorical(data["UserName"], categories=order, ordered=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x="UserName", y=count_col, palette="viridis")
        plt.title(f"{title} - í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´: {tier}")
        plt.xlabel("UserName")
        plt.ylabel(count_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# 10. ì‹œê°�í™” ì‹¤í–‰
plot_top10(top10_following_pd, "num_following", "íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ìˆ˜ ìƒ�ìœ„ 10ëª…")
plot_top10(top10_followed_pd, "num_followed", "íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ìˆ˜ ìƒ�ìœ„ 10ëª…")



import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# 0. ë�°ì�´í„°í”„ë ˆì�„ ì˜ˆì‹œ (í˜•ë‹˜ í™˜ê²½ì—�ì„œëŠ” ì�´ë¯¸ dfê°€ ì�ˆë‹¤ê³  ê°€ì •)
# df = pl.read_csv("your_data.csv")  # í•„ìš” ì‹œ

# íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ìˆ˜ êµ¬í•˜ê¸° (explode í›„ ì§‘ê³„)
following_exploded = df.select(["UserId", "FollowingUserId"])
following_counts = (
    following_exploded
    .filter(pl.col("FollowingUserId").is_not_null())
    .group_by("UserId")
    .agg(pl.count("FollowingUserId").alias("num_following"))
)

# íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ìˆ˜ êµ¬í•˜ê¸° (explode í›„ ì§‘ê³„)
followed_exploded = df.select(["UserId", "FollowingUserId"])
followed_counts = (
    followed_exploded
    .filter(pl.col("FollowingUserId").is_not_null())
    .group_by("FollowingUserId")
    .agg(pl.count("UserId").alias("num_followed"))
    .rename({"FollowingUserId": "UserId"})
)



# 4. ìœ ì € ê¸°ë³¸ ì •ë³´ (UserId, UserName, PerformanceTier)
user_info = df.select(["UserId", "UserName", "PerformanceTier"]).unique()

# 5. íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ì •ë³´ì—� ìœ ì € ì •ë³´ ì¡°ì�¸
following_with_info = following_counts.join(user_info, on="UserId", how="left")

# 6. íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ì •ë³´ì—� ìœ ì € ì •ë³´ ì¡°ì�¸
followed_with_info = followed_counts.join(user_info, on="UserId", how="left")

# 7. í‹°ì–´ë³„ ìƒ�ìœ„ Nëª… ì¶”ì¶œ í•¨ìˆ˜
def get_top_n_performance_tier(df_pl: pl.DataFrame, count_col: str, n=10) -> pl.DataFrame:
    tiers = df_pl.select("PerformanceTier").unique().to_series().to_list()
    top_n_list = []
    for tier in tiers:
        filtered = df_pl.filter(pl.col("PerformanceTier") == tier)
        sorted_df = filtered.sort(count_col, descending=True)
        top_n_list.append(sorted_df.head(n))
    return pl.concat(top_n_list)

# 8. íŒ”ë¡œì�‰ ê¸°ì¤€ í‹°ì–´ë³„ ìƒ�ìœ„ 10ëª…
top10_following = get_top_n_performance_tier(following_with_info, "num_following", 10)

# 9. íŒ”ë¡œì�‰ ë‹¹í•œ ê¸°ì¤€ í‹°ì–´ë³„ ìƒ�ìœ„ 10ëª…
top10_followed = get_top_n_performance_tier(followed_with_info, "num_followed", 10)

# 10. ìœ ì € ê¸°ë³¸ í™œë�™ ì§€í‘œ (ì»¤ë„�, í�¬ëŸ¼, ëŒ€íšŒ, ë�°ì�´í„°ì…‹ ìˆ˜)
metrics = ["num_kernels", "num_forum_messages", "NumCompetitionsParticipated", "NumDatasetsContributed"]
user_metrics_df = df.select(["UserId"] + metrics).unique()

# 11. ìƒ�ìœ„ 10ëª…ì—� ê¸°ë³¸ ì§€í‘œ ì¡°ì�¸
top10_following_stats = top10_following.join(user_metrics_df, on="UserId", how="left")
top10_followed_stats = top10_followed.join(user_metrics_df, on="UserId", how="left")

# 12. ë°±ë¶„ìœ„ ê³„ì‚° í•¨ìˆ˜ (í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„, ì§€í‘œë³„ ë°±ë¶„ìœ„)
def calculate_percentiles_for_top10(top10_df: pl.DataFrame, full_df: pl.DataFrame, metrics: list[str]):
    percentiles = {metric: {} for metric in metrics}
    tiers = top10_df.select("PerformanceTier").unique().to_series().to_list()

    for tier in tiers:
        full_tier = full_df.filter(pl.col("PerformanceTier") == tier)
        top10_tier = top10_df.filter(pl.col("PerformanceTier") == tier)

        for metric in metrics:
            # None ì œê±°
            full_values = [v for v in full_tier[metric].to_list() if v is not None]
            if len(full_values) == 0:
                percentiles[metric][tier] = [0] * len(top10_tier)
                continue

            pct_list = []
            for val in top10_tier[metric]:
                if val is None:
                    pct_list.append(0)
                    continue
                count = sum(1 for v in full_values if v <= val)
                pct = (count / len(full_values)) * 100
                pct_list.append(pct)
            percentiles[metric][tier] = pct_list
    return percentiles

# 13. íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ìƒ�ìœ„ 10ëª… ë°±ë¶„ìœ„ ê³„ì‚°
pct_following = calculate_percentiles_for_top10(top10_following_stats, df, metrics)

# 14. íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ìƒ�ìœ„ 10ëª… ë°±ë¶„ìœ„ ê³„ì‚°
pct_followed = calculate_percentiles_for_top10(top10_followed_stats, df, metrics)

# 15. ì‹œê°�í™” í•¨ìˆ˜ - 16x1 ì„œë¸Œí”Œë¡¯ìœ¼ë¡œ í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ì§€í‘œë³„ ë°±ë¶„ìœ„ ì‹œê°�í™”
def plot_percentiles_by_tier(pct_dict: dict, title: str, top10_df: pl.DataFrame):
    tiers = sorted(next(iter(pct_dict.values())).keys())
    metrics = sorted(pct_dict.keys())

    # UserName ë¦¬ìŠ¤íŠ¸ ìƒ�ì„± (í‹°ì–´ë³„ë¡œ ë‚˜ëˆ„ì–´ ì €ì�¥)
    user_names_per_tier = {}
    for tier in tiers:
        tier_users = top10_df.filter(pl.col("PerformanceTier") == tier)
        user_names_per_tier[tier] = tier_users.select("UserName").to_series().to_list()

    fig, axes = plt.subplots(len(tiers) * len(metrics), 1, figsize=(10, 3 * len(tiers) * len(metrics)), sharex=False)

    if len(axes.shape) > 1:
        axes = axes.flatten()
    else:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    ax_idx = 0
    for tier in tiers:
        for metric in metrics:
            ax = axes[ax_idx]
            values = pct_dict[metric][tier]
            ax.plot(range(1, len(values)+1), values, marker="o")
            ax.set_title(f"{title} - í‹°ì–´ {tier} - {metric}")
            ax.set_xlabel("ìƒ�ìœ„ 10ëª… ì‚¬ìš©ì��")
            ax.set_ylabel("ë°±ë¶„ìœ„ (%)")
            ax.set_xticks(range(1, len(values)+1))
            ax.set_xticklabels(user_names_per_tier[tier], rotation=45, ha='right')
            ax.grid(True)
            ax_idx += 1

    plt.tight_layout()
    plt.show()

plot_percentiles_by_tier(pct_following, "íŒ”ë¡œì�‰ í•œ ì‚¬ë�Œ ìƒ�ìœ„ 10ëª…", top10_following_stats)
plot_percentiles_by_tier(pct_followed, "íŒ”ë¡œì�‰ ë‹¹í•œ ì‚¬ë�Œ ìƒ�ìœ„ 10ëª…", top10_followed_stats)



df


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_all_withdate_gpt.parquet')
df


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1. ë�°ì�´í„° ì¤€ë¹„
df_users = df.select(["UserId", "RegisterDate", "PerformanceTier"]).to_pandas()
df_users = df_users.dropna(subset=["RegisterDate", "PerformanceTier"])

df_users["RegisterDate"] = pd.to_datetime(df_users["RegisterDate"])
today = pd.to_datetime(datetime.today().date())
df_users["years_since_register"] = (today - df_users["RegisterDate"]).dt.days / 365.25

# 2. ê°€ì�… ê¸°ê°„ êµ¬ê°„ ì •ì�˜
def assign_range(years):
    if years < 1:
        return "1ë…„ ë¯¸ë§Œ"
    elif years < 3:
        return "1~3ë…„"
    elif years < 5:
        return "3~5ë…„"
    elif years < 10:
        return "5~10ë…„"
    else:
        return "10ë…„ ì�´ìƒ�"

df_users["register_range"] = df_users["years_since_register"].apply(assign_range)

# 3. êµ�ì°¨í‘œ
pivot_df = pd.crosstab(df_users["register_range"], df_users["PerformanceTier"])
pivot_df = pivot_df.reindex(["1ë…„ ë¯¸ë§Œ", "1~3ë…„", "3~5ë…„", "5~10ë…„", "10ë…„ ì�´ìƒ�"], fill_value=0)

# 4. ìŠ¤íƒ� ë§‰ëŒ€ê·¸ë�˜í”„ (ìœ ì € ìˆ˜ ê¸°ì¤€)
ax = pivot_df.plot(
    kind="bar",
    # stacked=True,
    figsize=(16, 12),
    colormap="Set3",
    width=1
)

plt.title("ê°€ì�… ì—°ì°¨ë³„ PerformanceTier ë¶„í�¬ (ìœ ì € ìˆ˜ ê¸°ì¤€)")
plt.xlabel("ê°€ì�… ì—°ì°¨ êµ¬ê°„")
plt.ylabel("ìœ ì € ìˆ˜")
plt.xticks(rotation=0)

# 5. ëª¨ë“  ê°’ ìˆ«ì��ë¡œ í‘œì‹œ
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            if height < 2000:
                y_pos = height + 10     # ì�‘ì�€ ë§‰ëŒ€: ë§‰ëŒ€ ìœ„ìª½
                va = 'bottom'
            else:
                y_pos = height / 2     # ì¶©ë¶„í�ˆ í�° ë§‰ëŒ€: ì¤‘ì•™ í‘œì‹œ
                va = 'center'

            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha='center',
                va=va,
                fontsize=9,
                color='black',
                fontweight='bold'
            )

plt.legend(title="PerformanceTier", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to4_with_all.parquet')
df


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# tier ì»¬ëŸ¼ ì •ì�˜
tier_cols_by_type = {
    "Datasets": {
        "Tier2": "Tier2Date_Data",
        "Tier3": "Tier3Date_Data",
        "Tier4": "Tier4Date_Data"
    },
    "Scripts": {
        "Tier2": "Tier2Date_Kernel",
        "Tier3": "Tier3Date_Kernel",
        "Tier4": "Tier4Date_Kernel"
    },
    "Competitions": {
        "Tier2": "Tier2Date_competition",
        "Tier3": "Tier3Date_competition",
        "Tier4": "Tier4Date_competition"
    },
    "Discussion": {
        "Tier2": "Tier2Date_forum",
        "Tier3": "Tier3Date_forum",
        "Tier4": "Tier4Date_forum"
    }
}

# parquet íŒŒì�¼ ë¶ˆëŸ¬ì˜¤ê¸° (í˜•ë‹˜ í™˜ê²½ì—� ë§�ê²Œ ê²½ë¡œ ìˆ˜ì •)
df = pl.read_parquet("User_1to4_with_all.parquet")

# í•„ìš”í•œ ì»¬ëŸ¼ë§Œ ì„ íƒ�
cols_needed = ["AchievementType", "RegisterDate", "Tier", "TierAchievementDate"] + \
              [col for cat in tier_cols_by_type.values() for col in cat.values()]
df_filtered = df.select(cols_needed)

# ë�„ë‹¬ ì�¼ìˆ˜ ê³„ì‚°
all_rows = []
for atype, cols in tier_cols_by_type.items():
    for tier_name, date_col in cols.items():
        if date_col in df.columns:
            temp_df = df_filtered.filter(pl.col("AchievementType") == atype)

            # Tier4ì�¸ë�° date_colì�´ ê²°ì¸¡ì¹˜ë©´ TierAchievementDate ì‚¬ìš©
            if tier_name == "Tier4":
                temp_df = temp_df.with_columns([
                    pl.when(pl.col(date_col).is_not_null())
                      .then(pl.col(date_col))
                      .otherwise(pl.col("TierAchievementDate"))
                      .alias("TierDate")
                ])
            else:
                temp_df = temp_df.with_columns([
                    pl.col(date_col).alias("TierDate")
                ])

            temp = (
                temp_df
                .filter(pl.col("TierDate").is_not_null())
                .with_columns([
                    (pl.col("TierDate").cast(pl.Date) - pl.col("RegisterDate").cast(pl.Date))
                    .alias("DaysToTier")
                ])
                .with_columns([
                    pl.lit(atype).alias("AchievementType"),
                    pl.lit(tier_name).alias("Tier")
                ])
                .select(["AchievementType", "Tier", "RegisterDate", "DaysToTier"])
            )
            all_rows.append(temp)

# ë³‘í•© í›„ íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜
full_df = pl.concat(all_rows).to_pandas()

# ì „ì²˜ë¦¬: ì�´ìƒ�ì¹˜ ë°� ê²°ì¸¡ ì œê±°
full_df = full_df[
    (full_df["RegisterDate"] != pd.to_datetime("1980-01-01")) &
    (full_df["DaysToTier"].dt.days >= 0) &
    (full_df["DaysToTier"].dt.days <= 365 * 30)  # 30ë…„ ì�´ë‚´
].copy()

# ì�¼ìˆ˜ â†’ ì—°ë�„
full_df["YearsToTier"] = full_df["DaysToTier"].dt.days / 365

# ê³ ìœ  AchievementType ìˆ˜ í™•ì�¸
types = full_df["AchievementType"].unique()
n_types = len(types)

# ì‹œê°�í™”
fig, axs = plt.subplots(n_types, 1, figsize=(6, 6 * n_types), sharey=False)

for i, atype in enumerate(types):
    ax = axs[i]
    subset = full_df[full_df["AchievementType"] == atype]
    subset.boxplot(
        by="Tier",
        column=["YearsToTier"],
        ax=ax,
        grid=False
    )
    ax.set_title(f"{atype}")
    ax.set_xlabel("Tier")
    if i == 0:
        ax.set_ylabel("Years")
    else:
        ax.set_ylabel("")

fig.suptitle("Years to Reach Each Tier by AchievementType", fontsize=16)
plt.tight_layout()
plt.show()



df.describe()


# í™•ì�¸: Tier == 4ì�¸ ìœ ì € ìˆ˜
for atype in ["Kernels", "Forums"]:
    base = df_filtered.filter(pl.col("AchievementType") == atype)
    tier4 = base.filter(pl.col("Tier") == 4)
    total_tier4 = tier4.shape[0]
    non_null_achieve = tier4.filter(pl.col("TierAchievementDate").is_not_null()).shape[0]

    print(f"[{atype}] Tier==4 ìœ ì € ìˆ˜: {total_tier4}, TierAchievementDate ì�ˆëŠ” ì‚¬ë�Œ: {non_null_achieve}")



df.filter(pl.col('UserName')=='shlomoron')


date_cols = [
    "FirstDatasetMedalDate",
    "FirstCompetitionMedalDate",
    "FirstKernelMedalDate",
    "FirstForumMedalDate"
]
cols_needed = ["UserId", "RegisterDate", "PerformanceTier"] + date_cols
df_filtered = df.select(cols_needed)

# ë�„ë‹¬ì�¼ ê³„ì‚° (1980-01-01 ì œê±° í�¬í•¨)
results = []
for col in date_cols:
    temp = (
        df_filtered
        .filter((pl.col(col).cast(pl.Date) != pl.date(1980, 1, 1)) & pl.col(col).is_not_null())
        .with_columns([
            (pl.col(col).cast(pl.Date) - pl.col("RegisterDate").cast(pl.Date)).alias("DaysToFirstMedal"),
            pl.lit(col).alias("MedalType")
        ])
        .select(["PerformanceTier", "DaysToFirstMedal", "MedalType"])
    )
    results.append(temp)

# ë³‘í•© ë°� Pandas ë³€í™˜
final_df = pl.concat(results).to_pandas()
final_df["DaysToFirstMedal"] = final_df["DaysToFirstMedal"].dt.days  # .days ì ˆëŒ€ ì“°ì§€ ë§ˆì‹œê³  ì�´ ë°©ì‹� ìœ ì§€

# ì‹œê°�í™”
tiers = sorted(final_df["PerformanceTier"].dropna().unique())
fig, axs = plt.subplots(len(tiers), 1, figsize=(10, 5 * len(tiers)), sharey=True)

for i, tier in enumerate(tiers):
    ax = axs[i]
    subset = final_df[final_df["PerformanceTier"] == tier]
    subset.boxplot(
        by="MedalType",
        column=["DaysToFirstMedal"],
        ax=ax,
        grid=False
    )
    ax.set_title(f"PerformanceTier {tier}")
    ax.set_xlabel("")
    ax.set_ylabel("Days")

fig.suptitle("Days from RegisterDate to First Medal by MedalType (PerformanceTierë³„)", fontsize=16)
plt.tight_layout()
plt.show()


df


from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df # .filter(pl.col('PerformanceTier')==4) # í˜¹ì‹œ í�¼í�¬ë¨¼ìŠ¤í‹°ì–´ë³„ë¡œ ë³´ê³  ì‹¶ìœ¼ì‹œë©´ ì—¬ê¸°ë¥¼ ì£¼ì„� í•´ì œí•˜ì‹œê³  ì„¤ì •í•˜ì„¸ìš”. 
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >= 45
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(pl.col("last_activity_date") >= pl.date(today.year - 3, today.month, today.day))
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 45+", "ìµœê·¼ 3ë…„ í™œë�™", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”) P=4", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >= 45
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(
    ((pl.col("TotalGold")) +
    (pl.col("TotalSilver")) +
    (pl.col("TotalBronze"))) >= 5
)
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 45+", "ë©”ë‹¬ 5ë§ˆë¦¬", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    ((pl.col("TotalGold")) +
    (pl.col("TotalSilver")) +
    (pl.col("TotalBronze"))) >= 5
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >= 45
)
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ë©”ë‹¬ 5ê°œ", "ìµœê·¼ 3ë…„ í™œë�™", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    ((pl.col("TotalGold")) +
    (pl.col("TotalSilver")) +
    (pl.col("TotalBronze"))) >= 5
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(pl.col("last_activity_date") >= pl.date(today.year - 3, today.month, today.day))
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ë©”ë‹¬ 5ê°œ", "ìµœê·¼ 3ë…„ í™œë�™", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >= 1388
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(pl.col("last_activity_date") >= pl.date(today.year - 3, today.month, today.day))
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 45+", "ìµœê·¼ 3ë…„ í™œë�™", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”) 4", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl
from matplotlib.patches import Polygon

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=46
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=70)
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=220)
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 46+", "ê¸°ì—¬ ëˆ„ì � 70+", 'ê¸°ì—¬ ëˆ„ì � 220+']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1.filter(
    (pl.col("num_kernels") > 0) |
    (pl.col("num_forum_messages") > 0) |
    (pl.col("NumCompetitionsParticipated") > 0) |
    (pl.col("NumDatasetsContributed") > 0)
)
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
pl.col("last_activity_date") >= pl.date(today.year-3, today.month, today.day))
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
pl.col("last_activity_date") >= pl.date(today.year - 1, today.month, today.day))
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(
pl.col("last_activity_date") >= pl.date(today.year, today.month - 3, today.day)
)
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(pl.col("last_activity_date") >= pl.date(today.year, today.month - 2, today.day))
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 45+", "ë©”ë‹¬ 5ë§ˆë¦¬", 'ìµœê·¼ 3ê°œì›” í™œë�™']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



from datetime import date
import polars as pl
from matplotlib.patches import Polygon

# ë‚ ì§œ ì„¤ì •
today = date(2025, 7, 4)

# 1. ê°€ì�…
df_step1 = df
step1 = df_step1["UserId"].n_unique()

# 2. ê¸°ì—¬ 1íšŒ ì�´ìƒ�
df_step2 = df_step1
step2 = df_step2["UserId"].n_unique()

# 3. ë©”ë‹¬ ìˆ˜ìƒ�
df_step3 = df_step2.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=45
)
step3 = df_step3["UserId"].n_unique()

# 4. ê¸°ì—¬ ëˆ„ì � 45+
df_step4 = df_step3.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=230
)
step4 = df_step4["UserId"].n_unique()

# 5. ìµœê·¼ 3ë…„ í™œë�™
df_step5 = df_step4.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=508)
step5 = df_step5["UserId"].n_unique()

# 6. ìµœê·¼ 3ê°œì›” í™œë�™
df_step6 = df_step5.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >=1388)
step6 = df_step6["UserId"].n_unique()

# ì�…ë ¥ ë�°ì�´í„°
labels = ["ê°€ì�…", "ê¸°ì—¬ 1íšŒ ì�´ìƒ�", "ë©”ë‹¬ ìˆ˜ìƒ�", "ê¸°ì—¬ ëˆ„ì � 45+", "ê¸°ì—¬ ëˆ„ì � 230+", 'ê¸°ì—¬ ëˆ„ì � 508+']
values = [step1, step2, step3, step4, step5, step6]
percents = [v / step1 * 100 for v in values]

# ì—­ìˆœìœ¼ë¡œ (1ë‹¨ê³„ê°€ ìœ„ì—�)
labels = labels[::-1]
values = values[::-1]
percents = percents[::-1]

# í�¼ë„� ë„ˆë¹„ ê³„ì‚°
max_val = max(values)
widths = [v / max_val for v in values]

# í�¼ë„� ê·¸ë¦¬ê¸°
fig, ax = plt.subplots(figsize=(8, 6))
height = 1.5
gap = 0.1
colors = plt.cm.Blues([0.2 + 0.15 * i for i in range(len(values))])

for i in range(len(values) - 1):
    top_width = widths[i]
    bottom_width = widths[i + 1]
    y_top = i * (height + gap)
    y_bottom = y_top + height

    # ì¢Œí‘œ ê³„ì‚°
    polygon = Polygon([
        [(1 - top_width) / 2, y_top],
        [(1 + top_width) / 2, y_top],
        [(1 + bottom_width) / 2, y_bottom],
        [(1 - bottom_width) / 2, y_bottom]
    ], closed=True, facecolor=colors[i], edgecolor='black')
    ax.add_patch(polygon)

# ë§ˆì§€ë§‰ ë‹¨ê³„ (ë§ˆê°� ì²˜ë¦¬)
final_width = widths[-1]
y_top = (len(values) - 1) * (height + gap)
polygon = Polygon([
    [(1 - final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top],
    [(1 + final_width) / 2, y_top + height],
    [(1 - final_width) / 2, y_top + height]
], closed=True, facecolor=colors[-1], edgecolor='black')
ax.add_patch(polygon)

# í…�ìŠ¤íŠ¸ ì¶”ê°€
for i, (label, value, pct, w) in enumerate(zip(labels, values, percents, widths)):
    y_center = i * (height + gap) + height / 2
    ax.text(0.5, y_center, f"{label}: {value:,}ëª… ({pct:.1f}%)", ha="center", va="center", fontsize=12, color="black", weight='bold')

# ì¶• ì„¤ì •
ax.set_xlim(0, 1)
ax.set_ylim(0, len(values) * (height + gap))
ax.axis("off")
ax.set_title("ìº�ê¸€ ì‚¬ìš©ì�� í–‰ë�™ íŒ½ì�´ (í�¼ë„�í˜• ì‹œê°�í™”)", fontsize=15)
plt.tight_layout()
plt.show()



print(f'{df.group_by("PerformanceTier").agg((pl.col("num_kernels").mean()+pl.col("num_forum_messages").mean()+pl.col("NumCompetitionsParticipated").mean()+pl.col("NumDatasetsContributed").mean()).alias("behavior_mean_sum")).sort("PerformanceTier")}')


df.select(pl.col('TotalActivitys').mean().alias('TotalUserBehavior_mean'))


df.columns




















df.filter((pl.col('last_activity_date').dt.date() >= pl.date(2025,3,5)) & (pl.col('TotalActivitys')>0))


df.filter(
    (pl.col("num_kernels") + pl.col("num_forum_messages") +
     pl.col("NumCompetitionsParticipated") + pl.col("NumDatasetsContributed")) >= 45
)
# step4 = df_step4["UserId"].n_unique()


df.filter(
    (pl.col("TotalGold") > 0) |
    (pl.col("TotalSilver") > 0) |
    (pl.col("TotalBronze") > 0)
)


df = df.with_columns(
    ((pl.col('TotalGold')) + (pl.col('TotalSilver')) + (pl.col('TotalBronze'))).alias('TotalMedals')
)

df.columns

df.select(
    (pl.col('TotalMedals')/pl.col('TotalActivitys')).alias('ActivityPerMedal')
).describe()

df.select(
    (pl.col('TotalMedals').sum()) / (pl.col('TotalActivitys').sum()).alias('MedalPercentagePerAcitivity')
)





df.filter(df['PerformanceTier']==2).describe() 
# ì»¤ë„� = 165
# ë©”ì‹œì§€ = 46
# ë�°ì�´í„°ì…‹ = 7
# ëŒ€íšŒ = 10


df.filter(df['PerformanceTier']==1).describe()
# ì»¤ë„� = 36
# ë©”ì‹œì§€ = 3
# ë�°ì�´í„°ì…‹ = 3
# ê²½ì§„ëŒ€íšŒ = 3


df.filter(df['PerformanceTier']==2).describe() 
# ì»¤ë„� = 165 - 36 = 129
# ë©”ì‹œì§€ = 46 -3 = 43
# ë�°ì�´í„°ì…‹ = 7 -3 = 4
# ëŒ€íšŒ = 10 -3 = 7
# ìœ„ì�˜ 4ê°œë¥¼ ë‹¤ ë�”í•˜ê³  4ë¡œ ë‚˜ëˆ„ë©´ 183/4 = 45ì •ë�„


df = df.with_columns((pl.col("num_kernels")+
    (pl.col("num_forum_messages"))+
    (pl.col("NumCompetitionsParticipated"))+
    (pl.col("NumDatasetsContributed"))).alias('TotalActivitys')
)
df.describe()


import matplotlib.pyplot as plt

# ë§‰ëŒ€ê·¸ë�˜í”„ ì‹œê°�í™”
plt.figure(figsize=(8, 5))
bars = plt.barh(
    tier_counts["PerformanceTier"].astype(str),
    tier_counts["Count"],
    color="skyblue"
)

# ë ˆì�´ë¸” í‘œì‹œ
for bar in bars:
    width = bar.get_width()
    plt.text(width + 50, bar.get_y() + bar.get_height()/2, f"{int(width)}", va="center")

plt.xlabel("ìœ ì € ìˆ˜")
plt.ylabel("Performance Tier")
plt.title("ìµœê·¼ 3ë…„ ë‚´ í™œë�™ì�� ì¤‘ PerformanceTier ë¶„í�¬ (ë§‰ëŒ€ê·¸ë�˜í”„)")
plt.tight_layout()
plt.show()



# ìµœê·¼ 3ë…„ ê¸°ì¤€ ë‚ ì§œ
recent_threshold = pd.Timestamp.now() - pd.DateOffset(years=3)

# í•„í„°ë§�: ìµœê·¼ 3ë…„ í™œë�™ì�� ì¤‘ í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ 1ë²ˆ
df_filtered = (
    df.filter(
        (pl.col("last_activity_date") >= pl.lit(recent_threshold)) &
        (pl.col("PerformanceTier") == 1)
    )
    .select("RegisterDate")
    .drop_nulls()
    .to_pandas()
)

# RegisterDateë¥¼ ì—°ë�„ë¡œ ë³€í™˜
df_filtered["RegisterYear"] = pd.to_datetime(df_filtered["RegisterDate"]).dt.to_period("Y").astype(str)

# ì‹œê°�í™” (í�ˆìŠ¤í† ê·¸ë�¨ ë˜�ëŠ” ë§‰ëŒ€ê·¸ë�˜í”„)
plt.figure(figsize=(10, 5))
df_filtered["RegisterYear"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Performance Tier 1 ìœ ì €ë“¤ì�˜ ê°€ì�… ì—°ë�„ ë¶„í�¬ (ìµœê·¼ 3ë…„ ë‚´ í™œë�™ì��)")
plt.xlabel("ê°€ì�… ì—°ë�„")
plt.ylabel("ìœ ì € ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df = pl.read_parquet('competitions_merged.parquet')


df


import polars as pl
import matplotlib.pyplot as plt

# ë�°ì�´í„° ë¡œë“œ
# df = pl.read_parquet("competition_table.parquet")

# ì—°ë�„ ë‹¨ìœ„ë¡œ ë³€í™˜
df = df.with_columns([
    pl.col("EnabledDate").cast(pl.Date).dt.truncate("1y").alias("Year")
])

# 1. ì—°ë�„ë³„ ëŒ€íšŒë‹¹ í�‰ê·  ì œì¶œ ìˆ˜
yearly_avg_submissions = (
    df.filter((pl.col("TotalSubmissions").is_not_null()) & (pl.col("EnabledDate").is_not_null()))
      .group_by("Year")
      .agg(pl.col("TotalSubmissions").mean().alias("AvgSubmissions"))
      .sort("Year")
      .to_pandas()
)

# 2. íŒ€ë‹¹ í�‰ê·  ì œì¶œ ìˆ˜ ì»¬ëŸ¼ ì¶”ê°€
df_with_ratio = df.with_columns(
    (pl.col("TotalSubmissions") / pl.col("TotalTeams")).alias("SubmissionPerTeam")
)

yearly_avg_submission_per_team = (
    df_with_ratio.filter(
        (pl.col("SubmissionPerTeam").is_not_null()) &
        (pl.col("EnabledDate").is_not_null())
    )
    .group_by("Year")
    .agg(pl.col("SubmissionPerTeam").mean().alias("AvgSubmissionPerTeam"))
    .sort("Year")
    .to_pandas()
)

# ì‹œê°�í™”
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 1. ëŒ€íšŒë‹¹ í�‰ê·  ì œì¶œ ìˆ˜
axs[0].plot(yearly_avg_submissions["Year"], yearly_avg_submissions["AvgSubmissions"], marker='o', color='blue')
axs[0].set_title("ì—°ë�„ë³„ ëŒ€íšŒë‹¹ í�‰ê·  ì œì¶œ ìˆ˜")
axs[0].set_ylabel("í�‰ê·  ì œì¶œ ìˆ˜")
axs[0].grid(True)

# 2. íŒ€ë‹¹ í�‰ê·  ì œì¶œ ìˆ˜
axs[1].plot(yearly_avg_submission_per_team["Year"], yearly_avg_submission_per_team["AvgSubmissionPerTeam"], marker='s', color='green')
axs[1].set_title("ì—°ë�„ë³„ íŒ€ë‹¹ í�‰ê·  ì œì¶œ ìˆ˜")
axs[1].set_xlabel("ì—°ë�„")
axs[1].set_ylabel("íŒ€ë‹¹ í�‰ê·  ì œì¶œ ìˆ˜")
axs[1].grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df.filter(pl.col('EnabledDate').dt.year() == 2015).sort('TotalSubmissions', descending=True)


# 1. ìœ ì €ë³„ ëŒ€íšŒë³„ ì œì¶œ íšŸìˆ˜ ê³„ì‚°
submissions_per_user_per_comp = (
    df
    .group_by(["CompetitionId", "UserId", "PerformanceTier"])
    .agg(pl.count().alias("NumSubmissions"))
)

# 2. í‹°ì–´ë³„ë¡œ ê°� ëŒ€íšŒë‹¹ í�‰ê·  ì œì¶œ íšŸìˆ˜ ê³„ì‚°
avg_submissions_per_tier = (
    submissions_per_user_per_comp
    .group_by(["PerformanceTier"])
    .agg(pl.col("NumSubmissions").mean().alias("AvgSubmissionsPerComp"))
    .sort("PerformanceTier")
)

# 3. ì‹œê°�í™”
import matplotlib.pyplot as plt

df_plot = avg_submissions_per_tier.to_pandas()

plt.figure(figsize=(10, 6))
bars = plt.bar(
    df_plot["PerformanceTier"],
    df_plot["AvgSubmissionsPerComp"],
    color=plt.cm.viridis(range(len(df_plot)))
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.title("í‹°ì–´ë³„ ìœ ì € 1ì�¸ë‹¹ ëŒ€íšŒë‹¹ í�‰ê·  ì œì¶œ ìˆ˜", fontsize=14)
plt.xlabel("Performance Tier")
plt.ylabel("í�‰ê·  ì œì¶œ ìˆ˜")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



df = df.filter(pl.col('TotalTeams') != 0).with_columns([
    (pl.col("TotalSubmissions") / pl.col("TotalTeams")).alias("SubmissionPerTeam"),
    pl.lit((df["TotalSubmissions"] / df["TotalTeams"]).max()).alias("MaxSubmissionPerTeam")
])

df


df.sort('TotalSubmissions', descending=True)


import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. ìœ ì €-ëŒ€íšŒ ë‹¨ìœ„ ì œì¶œ ìˆ˜ ê³„ì‚°
per_user_comp = (
    df.filter(pl.col('PerformanceTier')!=5)
    .group_by(["PerformanceTier", "UserId", "CompetitionId"])
    .agg(pl.col("SubmissionId").n_unique().alias("SubmissionCount"))
)

# 2. ì „ì²´ ìœ ì €-ëŒ€íšŒ ìˆ˜
total_per_tier = (
    per_user_comp
    .group_by("PerformanceTier")
    .agg(pl.count().alias("TotalUserCompetition"))
)

# 3. 487íšŒ ì�´ìƒ� ì œì¶œí•œ ìœ ì €-ëŒ€íšŒ ìˆ˜
heavy_submitters = (
    per_user_comp
    .filter(pl.col("SubmissionCount") >= 487)
    .group_by("PerformanceTier")
    .agg(pl.count().alias("HeavySubmitters"))
)

# 4. ë³‘í•© ë°� ì •ë ¬
summary_df = (
    total_per_tier
    .join(heavy_submitters, on="PerformanceTier", how="left")
    .fill_null(0)
    .sort("PerformanceTier")
)

# 5. ì‹œê°�í™”ìš© ë�°ì�´í„° ì¶”ì¶œ
tiers = summary_df["PerformanceTier"].to_list()
total = summary_df["TotalUserCompetition"].to_list()
heavy = summary_df["HeavySubmitters"].to_list()
ratios = [h / t * 100 if t > 0 else 0 for h, t in zip(heavy, total)]

x = np.arange(len(tiers))
bar_width = 0.6

# 6. ì‹œê°�í™”
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, total, width=bar_width, color='#d3d3d3', label='ì „ì²´ ìœ ì €-ëŒ€íšŒ ìˆ˜')
ax.bar(x, heavy, width=bar_width * 0.6, color='#e15759', label='487íšŒ ì�´ìƒ� ì œì¶œ')

# í…�ìŠ¤íŠ¸: ìˆ˜ + ë¹„ìœ¨
for i in range(len(tiers)):
    ax.text(x[i], total[i] + 2, str(total[i]), ha='center', va='bottom', fontsize=9, color='black')
    ax.text(
        x[i],
        heavy[i] + 2,
        f"{heavy[i]} ({ratios[i]:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=9,
        color='darkred'
    )

ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
ax.set_ylabel("ìœ ì €-ëŒ€íšŒ ìˆ˜")
ax.set_title("í‹°ì–´ë³„ 487íšŒ ì�´ìƒ� ì œì¶œí•œ ìœ ì €-ëŒ€íšŒ ìˆ˜ (ë¹„ìœ¨ í�¬í•¨)")
ax.legend()
plt.tight_layout()
plt.show()



submission_check = (
    df
    .join(suspicious_users, on="UserId", how="inner")
    .filter(
        pl.col("SubmissionDate").dt.date() == pl.col("RegisterDate").dt.date()
    )
    .select(["UserName", "SubmissionDate", "CompetitionId", "TeamId"])
)

submission_check


df = pl.read_parquet('comp_difficulty.parquet')
df


import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. ìœ ì €-ëŒ€íšŒ ë‹¨ìœ„ë¡œ ì œì¶œ ìˆ˜ ì„¸ê¸°
per_user_comp = (
    df.filter(pl.col('PerformanceTier')!=5)
    .group_by(["PerformanceTier", "UserId", "CompetitionId"])
    .agg(pl.col('SubmissionId').n_unique().alias("SubmissionCount"))
)

# 2. ì „ì²´ ì°¸ê°€ ìˆ˜: ìœ ì €-ëŒ€íšŒ ì¡°í•©ì�˜ ê°œìˆ˜
total_participation = (
    per_user_comp
    .group_by("PerformanceTier")
    .agg(pl.count().alias("TotalParticipation"))
)

# 3. ê·¸ ì¤‘ ì œì¶œ 1íšŒë§Œ í•œ ê²½ìš°
one_submission_participation = (
    per_user_comp
    .filter(pl.col("SubmissionCount") == 1)
    .group_by("PerformanceTier")
    .agg(pl.count().alias("OneTimeSubmission"))
)

# 4. ë³‘í•© ë°� ì •ë ¬
# group_by í›„ ë‹¤ì‹œ PerformanceTier ê¸°ì¤€ìœ¼ë¡œ í•©ì‚°
summary_df = (
    total_participation
    .join(one_submission_participation, on="PerformanceTier", how="outer")
    .group_by("PerformanceTier")
    .agg([
        pl.col("TotalParticipation").sum().alias("TotalParticipation"),
        pl.col("OneTimeSubmission").sum().alias("OneTimeSubmission")
    ])
    .sort("PerformanceTier")
)


# 5. ì‹œê°�í™”ìš© ë�°ì�´í„° ì¶”ì¶œ
tiers = summary_df["PerformanceTier"].to_list()
total = summary_df["TotalParticipation"].to_list()
ones = summary_df["OneTimeSubmission"].to_list()
ratios = [o / t * 100 if t > 0 else 0 for o, t in zip(ones, total)]

x = np.arange(len(tiers))
bar_width = 0.6

# 6. ì‹œê°�í™”
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, total, width=bar_width, color='#d3d3d3', label='ì „ì²´ ëŒ€íšŒ ì°¸ê°€ ìˆ˜')
ax.bar(x, ones, width=bar_width * 0.6, color='#4e79a7', label='1íšŒ ì œì¶œí•œ ì°¸ê°€ ìˆ˜')

# í…�ìŠ¤íŠ¸ í‘œì‹œ (ë¹„ìœ¨ í�¬í•¨)
for i in range(len(tiers)):
    ax.text(x[i], total[i] + 2, str(total[i]), ha='center', va='bottom', fontsize=9, color='black')
    ax.text(
        x[i],
        ones[i] + 2,
        f"{ones[i]} ({ratios[i]:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=9,
        color='blue'
    )

ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_xlabel("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´")
ax.set_ylabel("ëŒ€íšŒ ìˆ˜")
ax.set_title("í‹°ì–´ë³„ ëŒ€íšŒ ì°¸ê°€ ìˆ˜ vs 1íšŒ ì œì¶œ ìˆ˜ (ë¹„ìœ¨ í�¬í•¨)")
ax.legend()
plt.tight_layout()
plt.show()



print(summary_df)
print(summary_df["PerformanceTier"].to_list())
print(summary_df["PerformanceTier"].unique())
print(summary_df["PerformanceTier"].dtype)



import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ì›”ë³„ ê°€ì�…ì�� ìˆ˜ ê³„ì‚° (AchievementType ì œê±°)
df_monthly = (
    df
    .filter(
        (pl.col("PerformanceTier").is_in([1, 2, 3, 4])) &
        (pl.col("RegisterDate").is_not_null()) 
         & (pl.col('PerformanceTier') != 1)
    )
    .with_columns([
        pl.col("RegisterDate").dt.truncate("1y").alias("RegisterMonth")
    ])
    .group_by(["PerformanceTier", "RegisterMonth"])
    .agg(pl.count().alias("UserCount"))
    .sort(["RegisterMonth", "PerformanceTier"])
    .to_pandas()
)

# 2. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_monthly,
    x="RegisterMonth",
    y="UserCount",
    hue="PerformanceTier",
    marker="o",
    palette="Set1"
)

plt.title("í�¼í�¬ë¨¼ìŠ¤ í‹°ì–´ë³„ ê°€ì�… ì›”ë³„ ìœ ì € ìˆ˜")
plt.xlabel("ê°€ì�… ì›”")
plt.ylabel("ìœ ì € ìˆ˜")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(title="Performance Tier")
plt.tight_layout()
plt.show()



import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    kernel['Quarter'] = kernel['KernelCreationDate'].dt.quarter
    kernel['Year'] = kernel['KernleCreationDate'].dt.year

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€



df = pl.read_parquet('User_1to2_with_forum.parquet', low_memory=True)


def parse_multiple_json_dict_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    def safe_json_decode(s):
        try:
            return json.loads(s) if s is not None else None
        except Exception:
            return None

    for col in columns:
        # 1. í•´ë‹¹ ì»¬ëŸ¼ ê°’ ë¦¬ìŠ¤íŠ¸ ì¶”ì¶œ í›„ JSON íŒŒì‹± (ì•ˆì „í•˜ê²Œ)
        data = df[col].to_list()
        decoded = [safe_json_decode(x) for x in data]

        # 2. ë”•ì…”ë„ˆë¦¬ê°€ ì•„ë‹Œ ê²½ìš° ë¹ˆ ë¦¬ìŠ¤íŠ¸ ì²˜ë¦¬ í�¬í•¨í•œ ë°©ì–´ì � ì²˜ë¦¬
        tuple_lists = [list(d.items()) if isinstance(d, dict) else [] for d in decoded]

        # 3. tuple ë¦¬ìŠ¤íŠ¸ë¥¼ ìƒˆë¡œìš´ ì»¬ëŸ¼ìœ¼ë¡œ ì¶”ê°€
        tuple_col = f"{col}_tuple_list"
        # pl.Series ì§�ì ‘ ìƒ�ì„± ëŒ€ì‹  pl.from_dictë¥¼ í™œìš©í•˜ëŠ” ê²ƒë�„ ê³ ë ¤ ê°€ëŠ¥
        df = df.with_columns(pl.Series(tuple_col, tuple_lists))

        # 4. explode ë°� key/value ë¶„ë¦¬
        prefix = col.split('_')[0] if "OrganizationId_" in col else None
        key_alias = prefix if prefix else f"{col}_key"
        value_alias = col.split('_')[1] if "OrganizationId_" in col else f"{col}_value"

        df = df.explode(tuple_col).with_columns([
            pl.col(tuple_col).list.get(0).alias(key_alias),
            pl.col(tuple_col).list.get(1).alias(value_alias)
        ]).drop([tuple_col, col])

    return df

# dict_cols = ['OrganizationId_JoinDate', 'OrganizationId_Name', 'OrganizationId_CreationDate', 'OrganizationId_Industry']

# dict_cols = ['OrganizationId_JoinDate']
# df = parse_multiple_json_dict_columns(df, dict_cols)
# dict_cols = ['OrganizationId_Name']
# df = parse_multiple_json_dict_columns(df, dict_cols)
# dict_cols = ['OrganizationId_CreationDate']
# df = parse_multiple_json_dict_columns(df, dict_cols)
# dict_cols = ['OrganizationId_Industry']


# df = parse_multiple_json_dict_columns(df, dict_cols)


df.describe()


df.filter(df['Tier']==0)['UserId'].n_unique()


# ìœ ì €ì�˜ ê³ ìœ ê°’ > í‹°ì–´ë³„ë¡œë�„ í™•ì�¸í•´ë³´ê¸° > í�¬ëŸ¼ ë©”ì‹œì§€ë¥¼ ì�‘ì„±í•œ ìœ ì € ìˆ˜
df['UserName'].n_unique() # 222361

df.filter(df['PerformanceTier']==1)['UserName'].n_unique() # 205624
df.filter(df['PerformanceTier']==2)['UserName'].n_unique() # 16737


# í�¬ëŸ¼ì•„ì�´ë””ì�˜ ê³ ìœ ê°’ > í‹°ì–´ë³„ë¡œë�„ í™•ì�¸í•´ë³´ê¸°ê¸°
df['ForumId'].value_counts() #10484

df.filter(df['PerformanceTier']==1)['ForumId'].n_unique() # 7701
df.filter(df['PerformanceTier']==2)['ForumId'].n_unique() # 5845


# í�¬ëŸ¼ í† í”½ ì•„ì�´ë””ì�˜ ê³ ìœ ê°’ > í‹°ì–´ë³„ë¡œë�„ í™•ì�¸í•´ë³´ê¸°ê¸°
df['Id_topics'].n_unique() #339004

df.filter(df['PerformanceTier']==1)['Id_topics'].n_unique() # 209487
df.filter(df['PerformanceTier']==2)['Id_topics'].n_unique() # 240618


# ì»¬ëŸ¼ë³„ ê³ ìœ  ê°’
for i in df.columns:
    print(f'{i} : {df[i].n_unique()}')
    print(f"       tier1 : {df.filter(df['PerformanceTier']==1)[i].n_unique()}")
    print(f"       tier2 : {df.filter(df['PerformanceTier']==2)[i].n_unique()}")


df['FollowingUserId'].list.len().max()


pdf = df.select([
    pl.col("PostDate").dt.truncate("1mo").alias("PostMonth"),
    pl.col("PostUserId"),
    pl.col("ForumTopicId"),
    pl.col("PerformanceTier")
]).to_pandas()


# PostUserId ìˆ˜ (ì¤‘ë³µ ì œê±°)
user_counts = (
    pdf.drop_duplicates(subset=["PostMonth", "PostUserId", "PerformanceTier"])
    .groupby(["PostMonth", "PerformanceTier"])
    .agg(UniqueUsers=('PostUserId', 'count'))
    .reset_index()
)

plt.figure(figsize=(14, 6))
sns.lineplot(data=user_counts, x="PostMonth", y="UniqueUsers", hue="PerformanceTier", marker="o")
plt.title("ì›”ë³„ PerformanceTierë³„ ìœ ì € ìˆ˜ ë³€í™” (PostUserId)")
plt.xlabel("PostMonth")
plt.ylabel("Unique Users")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# ForumTopicId ìˆ˜ (ì¤‘ë³µ ì œê±°)
topic_counts = (
    pdf.drop_duplicates(subset=["PostMonth", "ForumTopicId", "PerformanceTier"])
    .groupby(["PostMonth", "PerformanceTier"])
    .agg(UniqueTopics=('ForumTopicId', 'count'))
    .reset_index()
)

plt.figure(figsize=(14, 6))
sns.lineplot(data=topic_counts, x="PostMonth", y="UniqueTopics", hue="PerformanceTier", marker="o")
plt.title("ì›”ë³„ PerformanceTierë³„ í† í”½ ìˆ˜ ë³€í™” (ForumTopicId)")
plt.xlabel("PostMonth")
plt.ylabel("Unique Topics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



pdf = df.select([
    pl.col("Medal"),
    pl.col("PerformanceTier"),
    pl.col("MedalAwardDate").dt.truncate("1mo").alias("AwardMonth")
]).filter(pl.col("Medal") > 0).to_pandas()

medal1 = pdf[pdf["Medal"] == 1]
grouped1 = (
    medal1.groupby(["AwardMonth", "PerformanceTier"])
    .size().reset_index(name="Count")
)

plt.figure(figsize=(14, 5))
sns.lineplot(data=grouped1, x="AwardMonth", y="Count", hue="PerformanceTier", marker="o")
plt.title("Medal 1 - PerformanceTierë³„ ìˆ˜ìƒ� ì¶”ì�´")
plt.xlabel("MedalAward Month")
plt.ylabel("ìˆ˜ìƒ� ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


medal2 = pdf[pdf["Medal"] == 2]
grouped2 = (
    medal2.groupby(["AwardMonth", "PerformanceTier"])
    .size().reset_index(name="Count")
)

plt.figure(figsize=(14, 5))
sns.lineplot(data=grouped2, x="AwardMonth", y="Count", hue="PerformanceTier", marker="o")
plt.title("ğŸ¥ˆ Medal 2 - PerformanceTierë³„ ìˆ˜ìƒ� ì¶”ì�´")
plt.xlabel("MedalAward Month")
plt.ylabel("ìˆ˜ìƒ� ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



medal3 = pdf[pdf["Medal"] == 3]
grouped3 = (
    medal3.groupby(["AwardMonth", "PerformanceTier"])
    .size().reset_index(name="Count")
)

plt.figure(figsize=(14, 5))
sns.lineplot(data=grouped3, x="AwardMonth", y="Count", hue="PerformanceTier", marker="o")
plt.title("ğŸ¥‰ Medal 3 - PerformanceTierë³„ ìˆ˜ìƒ� ì¶”ì�´")
plt.xlabel("MedalAward Month")
plt.ylabel("ìˆ˜ìƒ� ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df_len = df.with_columns([
    pl.col("FollowingUserId").list.len().alias("FollowingCount")
])

result = (
    df_len.filter(pl.col("PerformanceTier").is_in([1, 2]))
      .group_by("PerformanceTier")
      .agg(pl.col("FollowingCount").mean().alias("AvgFollowingCount"))
)

print(result)



import polars as pl

# ìˆ˜ì¹˜í˜• íƒ€ì�… ì •ì�˜
numeric_types = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
}

# ì¡°ê±´ì—� ë§�ëŠ” ì»¬ëŸ¼ëª… í•„í„°ë§�
numeric_cols = [
    col for col, dtype in df.schema.items()
    if dtype in numeric_types and "Id" not in col
]

corr_matrix = [
    [df.select(pl.corr(c1, c2)).item() for c2 in numeric_cols]
    for c1 in numeric_cols
]

# ë”•ì…”ë„ˆë¦¬ í˜•íƒœë¡œ ë³€í™˜
corr_dict = {col: [row[i] for row in corr_matrix] for i, col in enumerate(numeric_cols)}

# DataFrame ìƒ�ì„± í›„ index ì»¬ëŸ¼ ì¶”ê°€
corr_df = pl.DataFrame(corr_dict).with_columns(pl.Series("index", numeric_cols)).select(["index"] + numeric_cols)

print(corr_df)



# polars DataFrame -> pandas DataFrame ë³€í™˜
corr_pd = corr_df.to_pandas().set_index('index')

# ìƒ�ì‚¼ê°� í–‰ë ¬ mask ìƒ�ì„± (Trueë©´ ì•ˆ ë³´ì�„)
mask = np.triu(np.ones_like(corr_pd, dtype=bool))

plt.figure(figsize=(20, 16))
sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, square=True, cbar_kws={"shrink": 0.8})

plt.title("ìƒ�ê´€ê´€ê³„ í�ˆíŠ¸ë§µ (í•˜ì‚¼ê°�)")
plt.tight_layout()
plt.show()


def parse_multiple_json_dict_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    def safe_json_decode(s):
        try:
            return json.loads(s) if s is not None else None
        except Exception:
            return None

    for col in columns:
        # 1. í•´ë‹¹ ì»¬ëŸ¼ ê°’ ë¦¬ìŠ¤íŠ¸ ì¶”ì¶œ í›„ JSON íŒŒì‹± (ì•ˆì „í•˜ê²Œ)
        data = df[col].to_list()
        decoded = [safe_json_decode(x) for x in data]

        # 2. ë”•ì…”ë„ˆë¦¬ê°€ ì•„ë‹Œ ê²½ìš° ë¹ˆ ë¦¬ìŠ¤íŠ¸ ì²˜ë¦¬ í�¬í•¨í•œ ë°©ì–´ì � ì²˜ë¦¬
        tuple_lists = [list(d.items()) if isinstance(d, dict) else [] for d in decoded]

        # 3. tuple ë¦¬ìŠ¤íŠ¸ë¥¼ ìƒˆë¡œìš´ ì»¬ëŸ¼ìœ¼ë¡œ ì¶”ê°€
        tuple_col = f"{col}_tuple_list"
        # pl.Series ì§�ì ‘ ìƒ�ì„± ëŒ€ì‹  pl.from_dictë¥¼ í™œìš©í•˜ëŠ” ê²ƒë�„ ê³ ë ¤ ê°€ëŠ¥
        df = df.with_columns(pl.Series(tuple_col, tuple_lists))

        # 4. explode ë°� key/value ë¶„ë¦¬
        prefix = col.split('_')[0] if "OrganizationId_" in col else None
        key_alias = prefix if prefix else f"{col}_key"
        value_alias = col.split('_')[1] if "OrganizationId_" in col else f"{col}_value"

        df = df.explode(tuple_col).with_columns([
            pl.col(tuple_col).list.get(0).alias(key_alias),
            pl.col(tuple_col).list.get(1).alias(value_alias)
        ]).drop([tuple_col, col])

    return df


df_dic = df.select(['UserName','UserId','RegisterDate','Id_topics','ForumId','CreationDate','TotalMessages','TotalReplies','OrganizationId_JoinDate', 'OrganizationId_Name', 'OrganizationId_CreationDate', 'OrganizationId_Industry'])
dict_cols = ['OrganizationId_JoinDate', 'OrganizationId_Name', 'OrganizationId_CreationDate', 'OrganizationId_Industry']

df_dic = parse_multiple_json_dict_columns(df_dic, dict_cols)


df_dic.write_parquet('user_merged_tier1_with2_dict_explode_specific_columns.parquet')


import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# 1. ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ ì„¤ì •
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 2. font name í™•ì�¸
font_name = font_prop.get_name()

# 3. matplotlib ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# 4. seaborn ì„¤ì •
sns.set_theme(font=font_name)



df_dic = pl.read_parquet('user_merged_tier1_with2_dict_explode_specific_columns.parquet')


# 902ë§Œê°œ > 83178232ê°œê°€ ë�˜ë„¤ìš”.. ì•½ 8300ë§Œê°œ...
df_dic


df_dic.filter(df_dic['OrganizationId'].is_not_null())


df_dic['Name'].value_counts()


df = df_dic.with_columns([
    pl.col("JoinDate").str.to_date().dt.strftime("%Y").alias("JoinYear")
])

# ì›” ëŒ€ì‹  ì—°ë�„ë¡œ ê·¸ë£¹í•‘
monthly_counts = df.group_by("JoinYear").agg(
    pl.col("UserId").n_unique().alias("unique_users")
).sort("JoinYear")

monthly_counts = monthly_counts.filter(
    pl.col("JoinYear").is_not_null()
)

monthly_counts_pd = monthly_counts.to_pandas()

plt.figure(figsize=(15, 7))
plt.plot(monthly_counts_pd["JoinYear"], monthly_counts_pd["unique_users"], marker='o', linestyle='-')

plt.title("ì—°ë�„ë³„ ì¡°ì§� ìœ ì�… ìœ ì € ìˆ˜", fontproperties=font_prop, fontsize=14)
plt.xlabel("ê°€ì�… ì—°ë�„", fontproperties=font_prop, fontsize=12)
plt.ylabel("ê³ ìœ  ìœ ì € ìˆ˜", fontproperties=font_prop, fontsize=12)

plt.xticks(fontproperties=font_prop, fontsize=10)
plt.yticks(fontproperties=font_prop, fontsize=10)
plt.tight_layout()
plt.show()



df = df_dic.with_columns([
    pl.col("JoinDate").str.to_date().alias("JoinDate")
])

# ì—°ë�„ì™€ ë¶„ê¸° ì»¬ëŸ¼ ìƒ�ì„±
df = df.with_columns([
    pl.col("JoinDate").dt.year().cast(pl.Utf8).alias("Year"),
    pl.col("JoinDate").dt.quarter().cast(pl.Utf8).alias("Quarter")
])

# "YYYY-QN" ë¬¸ì��ì—´ ìƒ�ì„±
df = df.with_columns([
    (pl.col("Year") + "-Q" + pl.col("Quarter")).alias("JoinQuarter")
])

# ë¶„ê¸°ë³„ ê³ ìœ  UserId ìˆ˜ ì§‘ê³„
quarterly_counts = df.group_by("JoinQuarter").agg(
    pl.col("UserId").n_unique().alias("unique_users")
).sort("JoinQuarter")

# Null ê°’ ì œê±°
quarterly_counts = quarterly_counts.filter(pl.col("JoinQuarter").is_not_null())

# íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜
quarterly_counts_pd = quarterly_counts.to_pandas()

# ì‹œê°�í™”
plt.figure(figsize=(15, 7))
plt.plot(quarterly_counts_pd["JoinQuarter"], quarterly_counts_pd["unique_users"], marker='o', linestyle='-')

plt.title("ë¶„ê¸°ë³„ ì¡°ì§� ìœ ì�… ìœ ì € ìˆ˜", fontproperties=font_prop, fontsize=14)
plt.xlabel("ê°€ì�… ë¶„ê¸°", fontproperties=font_prop, fontsize=12)
plt.ylabel("ê³ ìœ  ìœ ì € ìˆ˜", fontproperties=font_prop, fontsize=12)

plt.xticks(rotation=45, fontproperties=font_prop, fontsize=10)
plt.yticks(fontproperties=font_prop, fontsize=10)
plt.tight_layout()
plt.show()



# 1) JoinDate -> ì—°-ì›” í˜•íƒœë¡œ ë³€í™˜
df = df_dic.with_columns([
    pl.col("JoinDate").str.to_date().dt.strftime("%Y-%m").alias("JoinMonth"),
    pl.col("Name")  # ì¡°ì§�ëª… ì»¬ëŸ¼
])

# 2) ì›”ë³„ ì¡°ì§�ë³„ ê°€ì�…ì�� ìˆ˜ ì§‘ê³„ (ê³ ìœ  UserId ìˆ˜ ê¸°ì¤€)
monthly_user_count = df.group_by(["JoinMonth", "Name"]).agg(
    pl.col("UserId").n_unique().alias("user_count")
)

# 3) ê²°ì¸¡ì¹˜ ì œê±°
monthly_user_count = monthly_user_count.filter(
    pl.col("JoinMonth").is_not_null() & pl.col("Name").is_not_null()
)

# 4) íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜
monthly_user_count_pd = monthly_user_count.to_pandas()



df_dic.filter(df_dic['Name'].is_not_null()).group_by('Name').agg(
    pl.col('UserId').n_unique().alias('User Count')
).sort('User Count', descending=True).head(10)


# 2. ìƒ�ìœ„ 10ê°œ ì¡°ì§� ì¶”ì¶œ (ê°€ì�…ì�� ìˆ˜ ì´�í•© ê¸°ì¤€)
top_10_orgs = (
    monthly_user_count_pd.groupby("Name")["user_count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# 3. ìƒ�ìœ„ ì¡°ì§�ë§Œ í•„í„°ë§�
filtered_df = monthly_user_count_pd[monthly_user_count_pd["Name"].isin(top_10_orgs)]

# 4. í”¼ë²— í…Œì�´ë¸” ìƒ�ì„± (ì¡°ì§� x ì›” â†’ ê°€ì�…ì�� ìˆ˜)
heatmap_data = filtered_df.pivot_table(
    index="Name",
    columns="JoinMonth",
    values="user_count",
    fill_value=0
)

# 5. í�ˆíŠ¸ë§µ ì‹œê°�í™”
plt.figure(figsize=(20, 6))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    linewidths=0.5,
    annot=True, fmt=".0f",  # ì •ìˆ˜ë¡œ í‘œí˜„
    cbar_kws={'label': 'ê°€ì�…ì�� ìˆ˜'}
)

plt.title("ìƒ�ìœ„ 10ê°œ ì¡°ì§�ì�˜ ì›”ë³„ ê°€ì�…ì�� ìˆ˜ ë³€í™”", fontsize=16)
plt.xlabel("ê°€ì�… ì›”", fontsize=12)
plt.ylabel("ì¡°ì§�ëª…", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# 1. í�°íŠ¸ ì„¤ì •
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 2. JoinMonthë¥¼ ë‚ ì§œ íƒ€ì�…ìœ¼ë¡œ ë³€í™˜ (ì—°-ì›” ê¸°ì¤€)
monthly_user_count_pd["JoinMonth_dt"] = pd.to_datetime(monthly_user_count_pd["JoinMonth"], format="%Y-%m")

# 3. ìƒ�ìœ„ 10ê°œ ì¡°ì§� ì¶”ì¶œ (ê°€ì�…ì�� ìˆ˜ ì´�í•© ê¸°ì¤€)
top_10_orgs = (
    monthly_user_count_pd.groupby("Name")["user_count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# 4. ìƒ�ìœ„ ì¡°ì§�ë§Œ í•„í„°ë§� í›„, JoinMonth_dt ê¸°ì¤€ ì •ë ¬
filtered_df = monthly_user_count_pd[monthly_user_count_pd["Name"].isin(top_10_orgs)]
filtered_df = filtered_df.sort_values(["Name", "JoinMonth_dt"])

# 5. ì‹œê°�í™”
plt.figure(figsize=(18, 8))

for org_name, group in filtered_df.groupby("Name"):
    plt.plot(
        group["JoinMonth_dt"],
        group["user_count"],
        marker='o',
        label=org_name
    )

plt.title("ìƒ�ìœ„ 10ê°œ ì¡°ì§�ì�˜ ì›”ë³„ ê°€ì�…ì�� ìˆ˜ ë³€í™”", fontproperties=font_prop, fontsize=16)
plt.xlabel("ê°€ì�… ì›”", fontproperties=font_prop, fontsize=12)
plt.ylabel("ê°€ì�…ì�� ìˆ˜", fontproperties=font_prop, fontsize=12)

plt.xticks(rotation=45, fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(title="ì¡°ì§�ëª…", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()



import polars as pl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. í�°íŠ¸ ì„¤ì • (í•œê¸€ ê¹¨ì§� ë°©ì§€ìš©)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 2. ì¡°ì§�ëª… ì»¬ëŸ¼ëª…ì—� ë§�ê²Œ ìˆ˜ì • ('Name'ìœ¼ë¡œ ê°€ì •)
# ì¡°ì§�ì—� ì†�í•œ ìœ ì €ì™€ ì†�í•˜ì§€ ì•Šì�€ ìœ ì € ë¹„ìœ¨ ê³„ì‚°
# 'Name'ì�´ null í˜¹ì�€ ë¹ˆ ê°’ì�´ë©´ ì¡°ì§�ì—� ì†�í•˜ì§€ ì•Šì�€ ê²ƒìœ¼ë¡œ ê°„ì£¼

# 'Name' ì»¬ëŸ¼ì—� Null í˜¹ì�€ ë¹ˆ ë¬¸ì��ì—´ ì—¬ë¶€ ì²´í�¬í•´ì„œ 'has_org' ì»¬ëŸ¼ ìƒ�ì„±
df = df_dic.with_columns([
    (pl.col("Name").is_not_null() & (pl.col("Name") != "")).alias("has_org")
])

# ê³ ìœ  ìœ ì € ìˆ˜ ê¸°ì¤€ ì§‘ê³„
user_group_counts = df.group_by("has_org").agg(
    pl.col("UserId").n_unique().alias("unique_users")
).sort("has_org")

# íŒ�ë‹¤ìŠ¤ë¡œ ë³€í™˜
user_group_counts_pd = user_group_counts.to_pandas()

# 3. íŒŒì�´ì°¨íŠ¸ ê·¸ë¦¬ê¸°
labels = ['ì¡°ì§�ì—� ì†�í•¨', 'ì¡°ì§�ì—� ì†�í•˜ì§€ ì•Šì�Œ']
sizes = user_group_counts_pd["unique_users"].values

plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontproperties': font_prop}
)
plt.title("ì¡°ì§� ì†Œì†� ì—¬ë¶€ì—� ë”°ë¥¸ ìœ ì € ë¹„ìœ¨", fontproperties=font_prop, fontsize=16)
plt.axis('equal')  # ì›�í˜• ìœ ì§€
plt.show()



from scipy.stats import anderson, bartlett
import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€


df = pl.read_parquet('User_1to4_with_forum_sentimental.parquet', low_memory=True)
df














from scipy.stats import anderson, bartlett
import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€


duckdb.query("""
             SELECT 
                COUNT(*)
             FROM read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Forums_sentiment_final.parquet') f
            INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
            ON f.PostUserId = u.UserId
             """)


#ì–˜ëŠ” í�¬ëŸ¼ì—� ëŒ€í•œê±° > ì§€í”¼í‹° 

test = """
COPY (
SELECT 
    f.Id,
    f.Title,
    f.Id_topics,
    f.ForumId,
    f.KernelId,
    f.CreationDate,
    f.LastCommentDate,
    f.Title_topics,
    f.IsSticky,
    f.TotalViews,
    f.Score,
    f.TotalMessages,
    f.Id_messages,
    f.ForumTopicId,
    f.PostUserId,
    f.PostDate,
    f.Message,
    f.Medal,
    f.MedalAwardDate,
    f.ForumMessageId,
    
    -- ğŸ”¥ FromId_ToId ê¸¸ì�´ ê³„ì‚°
    (
        SELECT COUNT(*) 
        FROM UNNEST(f.FromId_ToId)
    ) AS votedcnt,

    f.Sentiment_final AS Sentiment,
    f.Confidence,

    u.AchievementType,
    u.Tier,
    u.TierAchievementDate,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.FirstAchvDate,
    u.LastAchvDate,
    u.AvgPointsPerAchv,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry

FROM read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Forums_sentiment_final.parquet') f
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
    ON f.PostUserId = u.UserId
) TO 'User_1to4_with_forum_sentimental_compact.parquet' (FORMAT 'parquet');

"""

duckdb.query(test)


#ì–˜ëŠ” í�¬ëŸ¼ì—� ëŒ€í•œê±°

test = """
COPY
(
SELECT 
    f.Id,
    f.Title,
    f.Id_topics,
    f.ForumId,
    f.CreationDate,
    f.Title_topics,
    f.IsSticky,
    f.TotalViews,
    f.Score,
    f.TotalMessages,
    f.Id_messages,
    f.ForumTopicId,
    f.PostUserId,
    f.PostDate,
    f.Message,
    f.Medal,
    f.MedalAwardDate,
    f.ForumMessageId,
    f.Sentiment_final AS Sentiment,
    f.Confidence,
    u.AchievementType,
    u.Tier,
    u.Points,
    u.CurrentRanking,
    u.HighestRanking,
    u.TotalGold,
    u.TotalSilver,
    u.TotalBronze,
    u.UserName,
    u.RegisterDate,
    u.PerformanceTier,
    u.FollowingUserId,
    u.DaysSinceSignup,
    u.OrganizationId_JoinDate,
    u.OrganizationId_Name,
    u.OrganizationId_CreationDate,
    u.OrganizationId_Industry
FROM read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Forums_sentiment_final.parquet') f
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON f.PostUserId = u.UserId
) TO 'User_1to4_with_forum_sentimental_compact.parquet' (FORMAT 'parquet');
"""

duckdb.query(test)


#ì–˜ëŠ” í�¬ëŸ¼ì—� ëŒ€í•œê±°

test = """
COPY
(
SELECT 
    f.ForumId,  
f.ForumTopicId,  
f.Title_topics,  
f.PostUserId AS UserId,
f.ForumMessageId,
u.UserName,
u.Tier,  
u.PerformanceTier,  
f.Medal,  
f.MedalAwardDate,  
f.PostDate,  
f.Sentiment_final AS sentiment,  
f.Confidence AS confidence_score,
f.Message_clean AS Message
FROM read_parquet('/home/ronny/Downloads/LLM API ì‹¤ìŠµ/Forums_sentiment_final.parquet') f
INNER JOIN read_parquet('/home/ronny/Downloads/final_project/UserMerging/Users_Merged_tier1_to4.parquet') u
ON f.PostUserId = u.UserId
WHERE u.AchievementType = 'Discussion'
) TO 'User_1to4_with_forum_sentimental_compact_mini.parquet' (FORMAT 'parquet');
"""

duckdb.query(test)


df = pl.read_parquet('User_1to4_with_forum_sentimental_compact_mini.parquet', low_memory=True)
df


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ë�°ì�´í„° ë¡œë“œ
# df = pl.read_parquet("forum_sentiment_analysis.parquet")

# df1 = df.filter((pl.col('PostDate').dt.date() < pl.date(2025,4,1)) & (pl.col('PostDate').dt.date() >= pl.date(2025,1,1)))

df1 = df

# 2. í† í”½ë³„ ì²« ë©”ì‹œì§€ (ë³¸ë¬¸) ì¶”ì¶œ
topic_root_msg = (
    df1.group_by("ForumTopicId")
      .agg(pl.col("ForumMessageId").min().alias("TopicRootMessageId"))
)

# 3. ëŒ“ê¸€ë§Œ í•„í„° + ì‹ ë¢°ë�„ + ì œëª© ì •ì œ
df_labeled = df1.join(topic_root_msg, on="ForumTopicId")
df_comments = (
    df_labeled
    .filter(
        (pl.col("ForumMessageId") != pl.col("TopicRootMessageId")) &
        (pl.col("confidence_score") >= 0.7)
    )
    .with_columns([
        pl.col("Title_topics").str.strip_chars().str.to_lowercase()
    ])
    .filter(pl.col("Title_topics") != "untitled")  # ğŸ”¥ 'untitled' ì œì™¸
)

# 4. ê°�ì • ë¶„í�¬ ì§‘ê³„
df_plot = (
    df_comments
    .group_by(["ForumTopicId", "Title_topics", "sentiment"])
    .agg(pl.count().alias("Count"))
    .sort("Count", descending=True)
    .to_pandas()
)

# 5. ìƒ�ìœ„ 10ê°œ í† í”½ (ì •ë ¬ ê¸°ì¤€: ì „ì²´ ëŒ“ê¸€ ìˆ˜ í•©)
top_titles_ordered = (
    df_plot.groupby("Title_topics")["Count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

df_plot_top = df_plot[df_plot["Title_topics"].isin(top_titles_ordered)]

# 6. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_plot_top,
    x="Title_topics",
    y="Count",
    hue="sentiment",
    order=top_titles_ordered,  # âœ… í�° ìˆœì„œëŒ€ë¡œ ì •ë ¬
    estimator=sum,
    ci=None
)
plt.title("ìƒ�ìœ„ 10ê°œ í�¬ëŸ¼ í† í”½ë³„ ê°�ì • ë¶„í�¬ (ëŒ“ê¸€ ê¸°ì¤€, untitled ì œì™¸)")
plt.xlabel("í† í”½ ì œëª©")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# 1. ë¶€ì • ëŒ“ê¸€ë§Œ í•„í„°
df_negative = df_comments.filter(pl.col("sentiment") == "NEGATIVE")

# 2. ë¶€ì • ëŒ“ê¸€ ìˆ˜ ê¸°ì¤€ Top 10 í† í”½ ì œëª© ì¶”ì¶œ
top10_negative_titles = (
    df_negative
    .group_by("Title_topics")
    .agg(pl.count().alias("NegativeCount"))
    .sort("NegativeCount", descending=True)
    .limit(10)
    .select("Title_topics")
    .to_series()
    .to_list()
)

# 3. í•´ë‹¹ 10ê°œ í† í”½ì�˜ ì „ì²´ ê°�ì • ë¶„í�¬ ë‹¤ì‹œ ì§‘ê³„
df_plot_negative_top = (
    df_comments
    .filter(pl.col("Title_topics").is_in(top10_negative_titles))
    .group_by(["Title_topics", "sentiment"])
    .agg(pl.count().alias("Count"))
    .to_pandas()
)

# 4. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_plot_negative_top,
    x="Title_topics",
    y="Count",
    hue="sentiment",
    order=top10_negative_titles,
    estimator=sum,
    ci=None
)
plt.title("ë¶€ì • ëŒ“ê¸€ ê¸°ì¤€ ìƒ�ìœ„ 10ê°œ í�¬ëŸ¼ í† í”½ ê°�ì • ë¶„í�¬")
plt.xlabel("í† í”½ ì œëª©")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# ìœ ì € ê°�ì • ë¶„í�¬ ì§‘ê³„ (ëŒ“ê¸€ ê¸°ì¤€)
df_user = (
    df_comments
    .group_by(["UserName", "sentiment"])
    .agg(pl.count().alias("Count"))
    .to_pandas()
)

# ìƒ�ìœ„ 10ëª… ìœ ì € ì„ íƒ� (ì´� ëŒ“ê¸€ ìˆ˜ ê¸°ì¤€)
top_users = (
    df_user.groupby("UserName")["Count"]
    .sum()
    .nlargest(10)
    .index.tolist()
)

df_user_top10 = df_user[df_user["UserName"].isin(top_users)]

# ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_user_top10,
    x="UserName",
    y="Count",
    hue="sentiment",
    order=top_users,
    estimator=sum,
    ci=None
)
plt.title("ìƒ�ìœ„ 10 ìœ ì €ì�˜ ê°�ì • ì„±í–¥ ë¶„í�¬")
plt.xlabel("UserName")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 1. ë¶€ì • ëŒ“ê¸€ë§Œ í•„í„°
df_negative = df_comments.filter(pl.col("sentiment") == "NEGATIVE")

# 2. ë¶€ì • ëŒ“ê¸€ ìˆ˜ ê¸°ì¤€ ìƒ�ìœ„ 10ëª… ìœ ì € ì¶”ì¶œ
top10_negative_users = (
    df_negative
    .group_by("UserName")
    .agg(pl.count().alias("NegativeCount"))
    .sort("NegativeCount", descending=True)
    .limit(10)
    .select("UserName")
    .to_series()
    .to_list()
)

# 3. ì „ì²´ ê°�ì • ë¶„í�¬ ì�¬ì§‘ê³„ (Top 10 ìœ ì € ê¸°ì¤€)
df_user_neg_top = (
    df_comments
    .filter(pl.col("UserName").is_in(top10_negative_users))
    .group_by(["UserName", "sentiment"])
    .agg(pl.count().alias("Count"))
    .to_pandas()
)

# 4. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_user_neg_top,
    x="UserName",
    y="Count",
    hue="sentiment",
    order=top10_negative_users,
    estimator=sum,
    ci=None
)
plt.title("ë¶€ì • ëŒ“ê¸€ ìˆ˜ ê¸°ì¤€ ìƒ�ìœ„ 10ëª… ìœ ì €ì�˜ ê°�ì • ì„±í–¥ ë¶„í�¬")
plt.xlabel("UserName")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ê°�ì •ë³„ ìœ ì € pivot í…Œì�´ë¸” ìƒ�ì„±
df_pivot = (
    df_comments
    .group_by(["UserName", "sentiment"])
    .agg(pl.len().alias("Count"))
    .pivot(values="Count", index="UserName", on="sentiment")
    .fill_null(0)
)

# 2. ê°�ì • ì»¬ëŸ¼ ëˆ„ë�½ ë°©ì§€
required_sentiments = ["NEGATIVE", "POSITIVE", "NEUTRAL"]
for sent in required_sentiments:
    if sent not in df_pivot.columns:
        df_pivot = df_pivot.with_columns(pl.lit(0).alias(sent))

# 3. ë¹„ìœ¨ ë°� ì´�í•© ê³„ì‚°
user_counts = (
    df_pivot
    .with_columns([
        (
            pl.col("NEGATIVE") / (pl.col("NEGATIVE") + pl.col("POSITIVE") + pl.col("NEUTRAL"))
        ).alias("NegativeRatio"),
        (
            pl.col("NEGATIVE") + pl.col("POSITIVE") + pl.col("NEUTRAL")
        ).alias("Total")
    ])
    .filter(pl.col("Total") >= 10)
    .sort([pl.col("NegativeRatio"), pl.col("Total")], descending=[True, True])
    .select(["UserName", "NegativeRatio", "Total"])
    .limit(10)
)

top10_users = user_counts["UserName"].to_list()

# 4. í‹°ì–´ ì •ë³´ ë¶™ì�´ê¸°
tier_info = (
    df_comments
    .select(["UserName", "PerformanceTier", "Tier"])
    .filter(pl.col("UserName").is_in(top10_users))
    .unique()
)

user_label_map = (
    tier_info
    .to_pandas()
    .set_index("UserName")
    .apply(lambda row: f"{row.name} (P:{row['PerformanceTier']}/T:{row['Tier']})", axis=1)
    .to_dict()
)

# 5. ë�¼ë²¨ ë§¤í•‘ í…Œì�´ë¸” ìƒ�ì„±
df_label_map = pl.DataFrame([
    {"UserName": user, "UserLabel": label}
    for user, label in user_label_map.items()
])

# 6. ê°�ì • ë¶„í�¬ ì�¬ì§‘ê³„
df_user_top_ratio = (
    df_comments
    .filter(pl.col("UserName").is_in(top10_users))
    .join(df_label_map, on="UserName", how="left")
    .group_by(["UserLabel", "sentiment"])
    .agg(pl.count().alias("Count"))
    .to_pandas()
)

# 7. ì‹œê°�í™”ìš© ì´� ëŒ“ê¸€ ìˆ˜ ë§¤í•‘
user_total_map = {
    row["UserLabel"]: row["Total"]
    for row in (
        user_counts
        .with_columns([
            pl.col("UserName").map_elements(lambda name: user_label_map.get(name)).alias("UserLabel")
        ])
        .to_dicts()
    )
}

# 8. ì‹œê°�í™”
ordered_labels = list(user_total_map.keys())
plt.figure(figsize=(14, 6))
ax = sns.barplot(
    data=df_user_top_ratio,
    x="UserLabel",
    y="Count",
    hue="sentiment",
    order=ordered_labels,
    estimator=sum,
    ci=None
)

# í…�ìŠ¤íŠ¸ ì‚½ì�…
for bar in ax.patches:
    height = bar.get_height()
    if height == 0:
        continue
    x = bar.get_x() + bar.get_width() / 2
    sentiment = bar.get_label()

    # í˜„ì�¬ ìœ ì €ë�¼ë²¨ ì¶”ì •
    idx = int(bar.get_x() // (1.0 / len(ordered_labels)))
    if idx >= len(ordered_labels):
        continue
    label = ordered_labels[idx]

    if sentiment == "NEGATIVE":
        total = user_total_map.get(label, 0)
        ax.text(
            x,
            height / 2,
            f"{int(height)}/{total}",
            ha='center',
            va='center',
            fontsize=10,
            color="white",
            fontweight="bold"
        )

plt.title("ë¶€ì • ëŒ“ê¸€ ë¹„ìœ¨ ìƒ�ìœ„ 10 ìœ ì €ì�˜ ê°�ì • ë¶„í�¬ (PerformanceTier / Tier í�¬í•¨)")
plt.xlabel("UserName")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



print(df_comments
    .group_by(["UserName", "sentiment"])
    .agg(pl.count().alias("Count"))
    .pivot(values="Count", index="UserName", columns="sentiment")
    .fill_null(0).columns)



# ê°�ì •ê³¼ Medal (0~3)ì�˜ ê´€ê³„ ì§‘ê³„
df_medal = (
    df_comments
    .group_by(["sentiment", "Medal"])
    .agg(pl.count().alias("Count"))
    .sort(["sentiment", "Medal"])
    .to_pandas()
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_medal,
    x="sentiment",
    y="Count",
    hue="Medal",
    estimator=sum,
    ci=None
)
plt.title("ê°�ì •ê³¼ ë©”ë‹¬ ìˆ˜ì—¬ ê°„ì�˜ ê´€ê³„")
plt.xlabel("ê°�ì •")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.tight_layout()
plt.show()



# 1. ê°�ì • + ë©”ë‹¬ ë³„ Count ì§‘ê³„
df_medal = (
    df_comments
    .group_by(["sentiment", "Medal"])
    .agg(pl.len().alias("Count"))
)

# 2. ê°�ì •ë³„ ì´�í•© ê³„ì‚°
df_sentiment_total = (
    df_medal
    .group_by("sentiment")
    .agg(pl.col("Count").sum().alias("TotalCount"))
)

# 3. ê²°í•© â†’ ë¹„ìœ¨ ê³„ì‚°
df_ratio = (
    df_medal
    .join(df_sentiment_total, on="sentiment")
    .with_columns(
        (pl.col("Count") / pl.col("TotalCount")).alias("Ratio")
    )
    .sort(["sentiment", "Medal"])
    .to_pandas()
)

# 4. ì‹œê°�í™”
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_ratio,
    x="sentiment",
    y="Ratio",
    hue="Medal",
    estimator=sum,
    ci=None
)

# ë¹„ìœ¨ ê°’ í‘œì‹œ (%)
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height * 100:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

plt.title("ê°�ì •ë³„ ë©”ë‹¬ ë¶„í�¬ ë¹„ìœ¨")
plt.xlabel("ê°�ì •")
plt.ylabel("ë¹„ìœ¨ (%)")
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.show()


# ì›” ë‹¨ìœ„ë¡œ ê°�ì • ì§‘ê³„
df_trend = (
    df_comments
    .with_columns(pl.col("PostDate").dt.truncate("1mo").alias("Month"))
    .group_by(["Month", "sentiment"])
    .agg(pl.count().alias("Count"))
    .sort("Month")
    .to_pandas()
)

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_trend,
    x="Month",
    y="Count",
    hue="sentiment",
    marker="o"
)
plt.title("ê°�ì • íŠ¸ë Œë“œ (ì›”ë³„)")
plt.xlabel("Month")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



df

















# ğŸ”¹ 1. í�¬ëŸ¼/í† í”½ë³„ ê°�ì • ë¶„í�¬
df_topic_sentiment = (
    df.filter(pl.col("confidence_score") >= 0.7)
    .group_by(["ForumTopicId", "sentiment"])
    .agg(pl.count().alias("Count"))
    .pivot(index="ForumTopicId", columns="sentiment", values="Count")
    .fill_null(0)
)

# ğŸ”¹ 2. ìœ ì €ë³„ ê°�ì • ì„±í–¥
df_user_sentiment = (
    df.filter(pl.col("confidence_score") >= 0.7)
    .group_by(["UserName", "sentiment"])
    .agg(pl.count().alias("Count"))
    .pivot(index="UserName", columns="sentiment", values="Count")
    .fill_null(0)
)

# ğŸ”¹ 3. ê°�ì • - ì„±ê³¼ ìƒ�ê´€ê´€ê³„
df_sentiment_medal = (
    df.filter(pl.col("confidence_score") >= 0.7)
    .group_by(["sentiment", "Medal"])
    .agg(pl.count().alias("Count"))
    .sort(["sentiment", "Medal"])
)

# ğŸ”¹ 4. ê°�ì • íŠ¸ë Œë“œ (ì›”ë³„)
df_trend = (
    df.filter(pl.col("confidence_score") >= 0.7)
    .with_columns(pl.col("PostDate").cast(pl.Date).dt.truncate("1mo").alias("Month"))
    .group_by(["Month", "sentiment"])
    .agg(pl.count().alias("Count"))
    .sort("Month")
    .to_pandas()
)

# âœ… ì‹œê°�í™” ì˜ˆì‹œ (ê°�ì • íŠ¸ë Œë“œ)
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_trend, x="Month", y="Count", hue="sentiment", marker="o")
plt.title("ì›”ë³„ ê°�ì • íŠ¸ë Œë“œ")
plt.xlabel("Month")
plt.ylabel("ëŒ“ê¸€ ìˆ˜")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ì›” ë‹¨ìœ„ë¡œ ì •ë¦¬
df_month = df_comments.with_columns(
    pl.col("PostDate").cast(pl.Date).dt.truncate("1mo").alias("Month")
)

# 2. ê°�ì •ë³„ ì›”ê°„ ëŒ“ê¸€ ìˆ˜ ì§‘ê³„
df_sentiment_month = (
    df_month
    .group_by(["Month", "sentiment"])
    .agg(pl.len().alias("Count"))
    .pivot(values="Count", index="Month", columns="sentiment")
    .fill_null(0)
)

# 3. ë¹„ìœ¨ ì»¬ëŸ¼ ì¶”ê°€ + ì´�í•© í•„í„°
df_sentiment_month = df_sentiment_month.with_columns([
    (pl.col("NEGATIVE") / (pl.col("NEGATIVE") + pl.col("POSITIVE"))).alias("NEGATIVE_RATIO"),
    (pl.col("POSITIVE") / (pl.col("NEGATIVE") + pl.col("POSITIVE"))).alias("POSITIVE_RATIO"),
    (pl.col("NEGATIVE") + pl.col("POSITIVE")).alias("Total")
]).filter(pl.col("Total") >= 10)  # ğŸ”¥ ëŒ“ê¸€ 10ê°œ ë¯¸ë§Œ ë‹¬ì�€ ì œì™¸


# 4. Pandas ë³€í™˜
df_plot = df_sentiment_month.select(["Month", "NEGATIVE_RATIO", "POSITIVE_RATIO"]).to_pandas()
df_plot = df_plot.melt(id_vars="Month", var_name="sentiment", value_name="Ratio")

# 5. ì‹œê°�í™”
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_plot, x="Month", y="Ratio", hue="sentiment", marker='o')
plt.title("ê°�ì • íŠ¸ë Œë“œ ë¹„ìœ¨ (ì›”ë³„)")
plt.ylabel("ë¹„ìœ¨")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PerformanceTier + sentiment ì¡°í•©ë³„ ê°œìˆ˜
df_tier_sent = (
    df_comments
    .filter(pl.col("sentiment").is_in(["POSITIVE", "NEGATIVE"]))  # ì¤‘ë¦½ ì œì™¸
    .group_by(["PerformanceTier", "sentiment"])
    .agg(pl.len().alias("Count"))
    .sort(["PerformanceTier", "sentiment"])
    .to_pandas()
)

# 2. ì „ì²´ ìˆ˜ ëŒ€ë¹„ ë¹„ìœ¨ ê³„ì‚°
df_total = df_tier_sent.groupby("PerformanceTier")["Count"].sum().reset_index(name="Total")
df_plot = df_tier_sent.merge(df_total, on="PerformanceTier")
df_plot["Ratio"] = df_plot["Count"] / df_plot["Total"]

# 3. ì‹œê°�í™”
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_plot,
    x="PerformanceTier",
    y="Ratio",
    hue="sentiment"
)
plt.title("PerformanceTierë³„ ê°�ì • ë¹„ìœ¨ (POSITIVE vs NEGATIVE)")
plt.xlabel("PerformanceTier")
plt.ylabel("ë¹„ìœ¨")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



# 1. ê°�ì • ì§‘ê³„ í›„ pivot
df_pivot = (
    df_comments
    .group_by(["Tier", "UserName", "sentiment"])
    .agg(pl.len().alias("Count"))
    .pivot(values="Count", index=["Tier", "UserName"], columns="sentiment")
    .fill_null(0)
)

# 2. ëˆ„ë�½ë�œ ê°�ì • ì»¬ëŸ¼ ë³´ì •
required_sentiments = ["NEGATIVE", "POSITIVE", "NEUTRAL"]
for sent in required_sentiments:
    if sent not in df_pivot.columns:
        df_pivot = df_pivot.with_columns(pl.lit(0).alias(sent))

# 3. ë¹„ìœ¨ ë°� ì´�í•© ê³„ì‚°
df_with_ratio = (
    df_pivot
    .with_columns([
        (pl.col("NEGATIVE") / (pl.col("NEGATIVE") + pl.col("POSITIVE") + pl.col("NEUTRAL"))).alias("NegativeRatio"),
        (pl.col("NEGATIVE") + pl.col("POSITIVE") + pl.col("NEUTRAL")).alias("Total")
    ])
    .filter(pl.col("Total") >= 10)
)

# 4. í‹°ì–´ë³„ ìƒ�ìœ„ 10ëª…ì”© ì¶”ì¶œ
top10_per_tier = (
    df_with_ratio
    .sort(["Tier", "NegativeRatio", "Total"], descending=[False, True, True])
    .group_by("Tier")
    .agg([
        pl.col("UserName").head(10),
        pl.col("NegativeRatio").head(10),
        pl.col("Total").head(10)
    ])
    .explode(["UserName", "NegativeRatio", "Total"])
)

# 5. ì‹œê°�í™”ìš© ì •ë¦¬
df_plot = top10_per_tier.to_pandas()
df_plot["NegativeRatioPct"] = df_plot["NegativeRatio"] * 100

# 6. ì‹œê°�í™”
plt.figure(figsize=(16, 8))
sns.barplot(
    data=df_plot,
    x="UserName",
    y="NegativeRatioPct",
    hue="Tier",
    dodge=False,
    palette="Reds_d"
)
plt.title("í‹°ì–´ë³„ ë¶€ì • ëŒ“ê¸€ ë¹„ìœ¨ ìƒ�ìœ„ 10 ìœ ì €")
plt.ylabel("ë¶€ì • ëŒ“ê¸€ ë¹„ìœ¨ (%)")
plt.xlabel("UserName")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()














df['sentiment'].value_counts()


pl.Config.set_fmt_str_lengths(1000)  # ë¬¸ì��ì—´ ê¸¸ì�´ ì œí•œì�„ ì �ìš©í•˜ì§€ ì•Šë�„ë¡� ì„¤ì •

df.filter((df['sentiment']=='NEUTRAL') & (df['IsTopic']==False))



# 1. í† í”½ë³„ ì²« ë©”ì‹œì§€ ID ì¶”ì¶œ
topic_roots = (
    df.group_by("ForumTopicId")
      .agg(pl.col("ForumMessageId").min().alias("TopicRootMessageId"))
)

# 2. ì›�ë³¸ dfì—� TopicRootMessageId ì¡°ì�¸ í›„ ë¹„êµ�
df = df.join(topic_roots, on="ForumTopicId", how="left").with_columns([
    (pl.col("ForumMessageId") == pl.col("TopicRootMessageId")).alias("IsTopic")
]).drop("TopicRootMessageId")






from scipy.stats import anderson, bartlett
import polars as pl
import duckdb
import tqdm
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib.font_manager as fm

# ì„¤ì¹˜ë�œ ë‚˜ëˆ”ê³ ë”• ê²½ë¡œ
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# í�°íŠ¸ ì�´ë¦„ ë¶ˆëŸ¬ì˜¤ê¸°
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlibì—� ê¸°ë³¸ í�°íŠ¸ë¡œ ì„¤ì •
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False  # ë§ˆì�´ë„ˆìŠ¤ ê¸°í˜¸ ê¹¨ì§� ë°©ì§€


def normality_test_for_df(df, status=0):

    num_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64}
    
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in num_types]

    normality, unnormality = [], []

    for j, i in enumerate(numeric_cols):
        data = df.select(pl.col(i).drop_nulls()).to_series().to_numpy()
        result = anderson(data)
        
        if len(data) < 10:
            print(f"[ì œì™¸] {i} â†’ ë�°ì�´í„° ìˆ˜ {len(data)}ê°œë¡œ ì •ê·œì„± ê²€ì • ìƒ�ë�µ")
            continue

        if result.statistic < result.critical_values[2]:  # 5% ìœ ì�˜ìˆ˜ì¤€ ê¸°ì¤€
            normality.append(i)
            print(f"[ì •ê·œì„± ì�ˆì�Œ] {i}")
            print(f"  - í†µê³„ëŸ‰      : {result.statistic:.4f}")
            print(f"  - 5% ì�„ê³„ê°’   : {result.critical_values[2]:.4f}")
            print(f"  - ìœ ì�˜ìˆ˜ì¤€ ëª©ë¡� : {result.significance_level}")
        else:
            unnormality.append(i)
            print(f"[ì •ê·œì„± ì—†ì�Œ] {i}")
            print(f"  - í†µê³„ëŸ‰      : {result.statistic:.4f}")
            print(f"  - 5% ì�„ê³„ê°’   : {result.critical_values[2]:.4f}")
            print(f"  - ìœ ì�˜ìˆ˜ì¤€ ëª©ë¡� : {result.significance_level}")

    if status == 0:
        print("\nâ€» ë°˜í™˜ê°’ ì—†ì�Œ: ì •ê·œì„±ì�„ ë”°ë¥´ëŠ” ì»¬ëŸ¼ì�„ ë°˜í™˜í•˜ë ¤ë©´ status=1, ë”°ë¥´ì§€ ì•ŠëŠ” ì»¬ëŸ¼ì�€ status=2ë¡œ ì„¤ì •í•˜ì„¸ìš”.")
    elif status == 1:
        print("\nâ€» ë°˜í™˜: ì •ê·œì„±ì�„ ë”°ë¥´ëŠ” ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸ ë°˜í™˜")
        return normality
    else:
        print("\nâ€» ë°˜í™˜: ì •ê·œì„±ì�„ ë”°ë¥´ì§€ ì•ŠëŠ” ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸ ë°˜í™˜")
        return unnormality



def total_hypothesis_test(df):
    num_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64}
    
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in num_types]

    print(f'ì•„ë�˜ì™€ ê°™ì�´ ìˆ˜ì¹˜í˜• ì»¬ëŸ¼ì—� ëŒ€í•´ ë¨¼ì € ê²€ì •ì�„ ì‹œì�‘í•©ë‹ˆë‹¤.')
    print(f'ëª©ë¡� : {numeric_cols}')

    normality, unnormality = [], []

    print('ì•¤ë�”ìŠ¨ ê²€ì •ì�„ í†µí•´ ì •ê·œì„±ì�„ í™•ì�¸í•©ë‹ˆë‹¤.')

    for j, i in enumerate(numeric_cols):
        data = df.select(pl.col(i).drop_nulls()).to_series().to_numpy()
        result = anderson(data)
        
        if len(data) < 10:
            print(f"[ì œì™¸] {i} â†’ ë�°ì�´í„° ìˆ˜ {len(data)}ê°œë¡œ ì •ê·œì„± ê²€ì • ìƒ�ë�µ")
            continue

        if result.statistic < result.critical_values[2]:  # 5% ìœ ì�˜ìˆ˜ì¤€ ê¸°ì¤€
            normality.append(i)
            print(f"[ì •ê·œì„± ì�ˆì�Œ] {i}")
            print(f"  - í†µê³„ëŸ‰      : {result.statistic:.4f}")
            print(f"  - 5% ì�„ê³„ê°’   : {result.critical_values[2]:.4f}")
            print(f"  - ìœ ì�˜ìˆ˜ì¤€ ëª©ë¡� : {result.significance_level}")
        else:
            unnormality.append(i)
            print(f"[ì •ê·œì„± ì—†ì�Œ] {i}")
            print(f"  - í†µê³„ëŸ‰      : {result.statistic:.4f}")
            print(f"  - 5% ì�„ê³„ê°’   : {result.critical_values[2]:.4f}")
            print(f"  - ìœ ì�˜ìˆ˜ì¤€ ëª©ë¡� : {result.significance_level}")
    
    if normality == []:
        print('ì •ê·œì„±ì�„ ë§Œì¡±í•˜ëŠ” ì»¬ëŸ¼ì�´ ì—†ìŠµë‹ˆë‹¤.')

    





df = pl.read_parquet('/home/ronny/Downloads/final_project/UserMerging/KernelMerged_dropview.parquet')


normality_test_for_df(df)


df


ì�´ ë’¤ë¡œëŠ” ì•ˆí–ˆìŠˆ

