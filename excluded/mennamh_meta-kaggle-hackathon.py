import pandas as pd


import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


kernel_ver=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')  #read KernelVersions.csv data


kernel_ver.info()   #check number of rows and datatypes


kernel_ver.shape   #check row and column number


kernel_ver.isnull().sum()    #check number of null values in each column


kernels=pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv') #read Kernels.csv dataset


kernels.info()  #check number of rows and datatypes


kernels.isnull().sum()  #check number of null values in each column


#selecting some columns and dropping others
kernel_ver_cleaned= kernel_ver[['AuthorUserId','ScriptId', 'ParentScriptVersionId', 'CreationDate', 'VersionNumber', 'Title',
                                'TotalLines', 'TotalVotes', 'RunningTimeInMilliseconds']]



kernel_ver_cleaned.info() #check number of rows and datatypes


#Selecting 15,000 rows only from dataframe due to limited space
kernel_ver_cleaned=kernel_ver_cleaned.sample(15000)  


#selecting some columns and dropping others
kernels_cleaned= kernels[['AuthorUserId', 'CurrentKernelVersionId', 'ForkParentKernelVersionId',
                          'CreationDate', 'MadePublicDate', 'CurrentUrlSlug', 'Medal', 'MedalAwardDate',
                          'TotalViews', 'TotalComments', 'TotalVotes']]


#Selecting 15,000 rows only from dataframe due to limited space
kernels_cleaned=kernels_cleaned.sample(15000)  


# Merge two datasets (kernels_cleaned, kernel_ver_cleaned) using AuthorUserId column
kernels_and_kernel_ver=pd.merge(kernels_cleaned, kernel_ver_cleaned, on='AuthorUserId')


kernels_and_kernel_ver


comp=pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv') #read Competitions.csv dataset


comp.info()   #check number of rows and datatypes


kernels_and_kernel_ver.sample() #view sample row


kernels_and_kernel_ver['CreationDate_x'].dtype #check datatype


#convert CreationDate_x column to datetime datatype
kernels_and_kernel_ver['CreationDate_x'] = pd.to_datetime(kernels_and_kernel_ver['CreationDate_x'])


#create new column 'create_year' to include year only
kernels_and_kernel_ver['create_year'] = kernels_and_kernel_ver['CreationDate_x'].dt.year


kernels_and_kernel_ver.sample() #view sample row


kernels_and_kernel_ver['create_year'].value_counts()  #check frequency of years


#create line graph for kernels created per year
create_year_plot=kernels_and_kernel_ver['create_year'].value_counts().sort_index().plot(kind='line', figsize=(10, 5), marker='o', color='#22c2b8')
create_year_plot.set_xlabel('Year')
create_year_plot.set_ylabel('Number of Kernels')
create_year_plot.set_title('Number of Kernels created each year');


kernels_and_kernel_ver['MadePublicDate'].dtype  #check datatype


#convert MadePublicDate column to datetime datatype
kernels_and_kernel_ver['MadePublicDate'] = pd.to_datetime(kernels_and_kernel_ver['MadePublicDate'])


#create new column publish_year to include year only
kernels_and_kernel_ver['publish_year'] = kernels_and_kernel_ver['MadePublicDate'].dt.year


kernels_and_kernel_ver['publish_year'].value_counts()  #check frequency of years


#create line graph for publicized kernels per year
publish_year_plot=kernels_and_kernel_ver['publish_year'].value_counts().sort_index().plot(kind='line', figsize=(10, 5), marker='o', color='#16a299')
publish_year_plot.set_xlabel('Year')
publish_year_plot.set_ylabel('Number of Kernels')
publish_year_plot.set_title('Number of Kernels published each year');


#create new dataframe to include rows that has medals
medal= kernels_and_kernel_ver[kernels_and_kernel_ver['Medal'].notnull()] 


medal.info()  #check number of rows and datatypes


#convert MedalAwardDate column to datetime datatype
medal['MedalAwardDate'] = pd.to_datetime(medal['MedalAwardDate'])


#create new column 'medal_year' to include year only
medal['medal_year'] = medal['MedalAwardDate'].dt.year


medal['medal_year'].dtype #check datatype


medal['medal_year'].value_counts() #check frequency of years


#create bar graph for number of medals per year
medal_year_plot = medal['medal_year'].value_counts().sort_index().plot(kind='bar', figsize=(10, 5), color='#ba14ce')
medal_year_plot.set_xlabel('Year')
medal_year_plot.set_ylabel('Number of Medals')
medal_year_plot.set_title('Number of Medals earned each year');


kernels_and_kernel_ver['TotalViews'].dtype  #check datatype


#create line graph for kernel views per year
views_plot=kernels_and_kernel_ver.groupby('publish_year')['TotalViews'].sum().plot(kind='line', figsize=(10, 5), color='#1c0d88', marker='o')
views_plot.set_xlabel('Year')
views_plot.set_ylabel('Number of views')
views_plot.set_title('Number of kernel views each year');


kernels_and_kernel_ver['TotalLines'].dtype  #check datatype


#create line graph for number of code lines per year
lines_plot=kernels_and_kernel_ver.groupby('publish_year')['TotalLines'].sum().plot(kind='line', figsize=(10, 5), color='#0b761a', marker='o')
lines_plot.set_xlabel('Year')
lines_plot.set_ylabel('Number of lines')
lines_plot.set_title('Number of lines each year');


comp.sample(7)  #view 7 random rows


comp.info() #check number of rows and datatypes


comp['EnabledDate'].dtype  #check datatype


#convert EnabledDate column to datetime datatype
comp['EnabledDate'] = pd.to_datetime(comp['EnabledDate'])


#create comp_year column to include year only
comp['comp_year'] = comp['EnabledDate'].dt.year


comp['comp_year'].dtype  #check datatype


comp['comp_year'].value_counts()  #check frequency of years


#check the average maximum team members in competitions per year
comp.groupby('comp_year')['MaxTeamSize'].mean()   


#create line graph for the average maximum team members in competitions per year
team_size_plot=comp.groupby('comp_year')['MaxTeamSize'].mean().plot(kind='line', figsize=(10, 5), color='#4d0723', marker='o')
team_size_plot.set_xlabel('Year')
team_size_plot.set_ylabel('Maximum team size')
team_size_plot.set_title('Average of Maximum team size each year');


comp['RewardType'].value_counts()  #check frequency of rewards


#create bar graph for number of different reward types
comp_reward_plot= comp['RewardType'].value_counts().plot(kind='bar', color=['#0ab6be', '#0c888e'], figsize=(8, 5))
comp_reward_plot.set_xlabel('Reward Type')
comp_reward_plot.set_ylabel('Number of reward')
comp_reward_plot.set_title('Number of Each Reward Type');


#check total number of prizes per year
comp.groupby('comp_year')['NumPrizes'].sum()  


#create line graph for total number of prizes per year
num_prizes_plot=comp.groupby('comp_year')['NumPrizes'].sum().plot(kind='line', figsize=(10, 5), color='#1ebc05', marker='o')
num_prizes_plot.set_xlabel('Year')
num_prizes_plot.set_ylabel('Number of Prizes')
num_prizes_plot.set_title('Number of prizes each year');


comp['comp_year'].value_counts()  #check frequency of years


#create bar graph for number of competitions per year
comp_num_plot= comp['comp_year'].value_counts().sort_index().plot(kind='bar', color=['#d6be22', '#a59115'], figsize=(8, 5))
comp_num_plot.set_xlabel('Year')
comp_num_plot.set_ylabel('Number of Competitions')
comp_num_plot.set_title('Number of Competitions each year');


#check number of total competitors per year
comp.groupby('comp_year')['TotalCompetitors'].sum()


#create bar graph for total number of competitors per year
competitor_plot = comp.groupby('comp_year')['TotalCompetitors'].sum().plot(kind='bar', figsize=(10, 5), color=['#c825d6','#7d1385'])
competitor_plot.set_xlabel('Year')
competitor_plot.set_ylabel('Number of Competitors')
competitor_plot.set_title('Number of Competitors Each Year');


#create bar graph for total number of teams per year
total_teams_plot = comp.groupby('comp_year')['TotalTeams'].sum().plot(kind='bar', figsize=(10, 5), color=['#8533d8','#56119c'])
total_teams_plot.set_xlabel('Year')
total_teams_plot.set_ylabel('Number of Teams')
total_teams_plot.set_title('Number of Teams Each Year');


#create bar graph for total number of submissions per year
comp_subm_plot = comp.groupby('comp_year')['TotalSubmissions'].sum().plot(kind='bar', figsize=(10, 5), color=['#134e85','#1f77ca'])
comp_subm_plot.set_xlabel('Year')
comp_subm_plot.set_ylabel('Number of Submissions')
comp_subm_plot.set_title('Number of Submissions Each Year');




