%%capture
!pip install gradio


import polars as pl
from IPython.display import IFrame, Image
import kagglehub
import os
import glob
import json
from pathlib import Path
import pandas as pd
import graphviz
import folium
import requests
from collections import defaultdict
import re
from difflib import get_close_matches
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


strin = '''
<html>
<head>
<meta charset="UTF-8">
<style>
/* Reset default margins and ensure box-sizing */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Default styles (light mode) */
body {
    font-family: Arial, sans-serif;
    background-color: #ffffff; /* White background */
    color: #333333; /* Dark gray text */
    padding: 20px;
    line-height: 1.6;
}

h2 {
    color: #1a3c6a; /* Dark blue for headers */
    margin-bottom: 10px;
}

p {
    color: #444444; /* Slightly lighter gray for paragraphs */
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    background-color: #f9f9f9; /* Light gray table background */
    border: 1px solid #dcdcdc; /* Light gray border */
}

th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #dcdcdc;
}

th {
    background-color: #4a90e2; /* Blue header background */
    color: #ffffff; /* White text for headers */
}

tr:nth-child(even) {
    background-color: #f1f1f1; /* Slightly darker gray for even rows */
}

hr {
    border: 0;
    border-top: 1px solid #dcdcdc;
    margin: 20px 0;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1e2525; /* Dark gray background */
        color: #e0e0e0; /* Light gray text */
    }

    h2 {
        color: #66b2ff; /* Light blue for headers */
    }

    p {
        color: #cccccc; /* Slightly darker gray for paragraphs */
    }

    table {
        background-color: #2a2f2f; /* Darker gray table background */
        border: 1px solid #444444; /* Dark gray border */
    }

    th, td {
        border-bottom: 1px solid #444444;
    }

    th {
        background-color: #005f99; /* Darker blue header background */
        color: #ffffff; /* White text */
    }

    tr:nth-child(even) {
        background-color: #333838; /* Slightly lighter dark gray for even rows */
    }

    hr {
        border-top: 1px solid #444444;
    }
}
</style>
</head>
<h2>ğŸ“� CompetitionTags.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>126</td><td>1000</td><td>0</td></tr><tr><td>CompetitionId</td><td>3486</td><td>424</td><td>0</td></tr><tr><td>TagId</td><td>14102</td><td>125</td><td>0</td></tr></table><hr><h2>ğŸ“� Competitions.csv</h2><p><strong>Total Columns:</strong> 48</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>2408</td><td>1000</td><td>0</td></tr><tr><td>Slug</td><td>Eurovision2010</td><td>1000</td><td>0</td></tr><tr><td>Title</td><td>Forecast Eurovision Voting</td><td>977</td><td>0</td></tr><tr><td>Subtitle</td><td>This competition requires contestants to forecast the voting for this year's Eurovision Song Contest in Norway on May 25th, 27th and 29th.</td><td>896</td><td>3</td></tr><tr><td>HostSegmentTitle</td><td>Featured</td><td>6</td><td>0</td></tr><tr><td>ForumId</td><td>2.0</td><td>991</td><td>9</td></tr><tr><td>OrganizationId</td><td>18.0</td><td>58</td><td>853</td></tr><tr><td>EnabledDate</td><td>04/07/2010 07:57:43</td><td>995</td><td>0</td></tr><tr><td>DeadlineDate</td><td>05/25/2010 18:00:00</td><td>758</td><td>0</td></tr><tr><td>ProhibitNewEntrantsDeadlineDate</td><td>03/28/2014 23:59:00</td><td>126</td><td>861</td></tr><tr><td>TeamMergerDeadlineDate</td><td>03/28/2014 23:59:00</td><td>123</td><td>866</td></tr><tr><td>TeamModelDeadlineDate</td><td>07/15/2013 23:59:00</td><td>100</td><td>883</td></tr><tr><td>ModelSubmissionDeadlineDate</td><td>04/22/2012 23:59:00</td><td>14</td><td>986</td></tr><tr><td>FinalLeaderboardHasBeenVerified</td><td>True</td><td>2</td><td>0</td></tr><tr><td>HasKernels</td><td>True</td><td>2</td><td>0</td></tr><tr><td>OnlyAllowKernelSubmissions</td><td>False</td><td>2</td><td>0</td></tr><tr><td>HasLeaderboard</td><td>False</td><td>2</td><td>0</td></tr><tr><td>LeaderboardPercentage</td><td>10</td><td>58</td><td>0</td></tr><tr><td>ScoreTruncationNumDecimals</td><td>5</td><td>3</td><td>0</td></tr><tr><td>EvaluationAlgorithmAbbreviation</td><td>AE</td><td>88</td><td>0</td></tr><tr><td>EvaluationAlgorithmName</td><td>Absolute Error</td><td>87</td><td>0</td></tr><tr><td>EvaluationAlgorithmDescription</td><td>Sum of absolute values of all errors.</td><td>80</td><td>3</td></tr><tr><td>EvaluationAlgorithmIsMax</td><td>False</td><td>2</td><td>0</td></tr><tr><td>MaxDailySubmissions</td><td>5</td><td>36</td><td>0</td></tr><tr><td>NumScoredSubmissions</td><td>5</td><td>21</td><td>0</td></tr><tr><td>MaxTeamSize</td><td>20</td><td>9</td><td>0</td></tr><tr><td>BanTeamMergers</td><td>False</td><td>2</td><td>0</td></tr><tr><td>EnableTeamModels</td><td>False</td><td>2</td><td>0</td></tr><tr><td>RewardType</td><td>USD</td><td>6</td><td>380</td></tr><tr><td>RewardQuantity</td><td>1000.0</td><td>55</td><td>191</td></tr><tr><td>NumPrizes</td><td>1</td><td>11</td><td>0</td></tr><tr><td>UserRankMultiplier</td><td>1.0</td><td>8</td><td>0</td></tr><tr><td>CanQualifyTiers</td><td>False</td><td>2</td><td>0</td></tr><tr><td>TotalTeams</td><td>22</td><td>345</td><td>0</td></tr><tr><td>TotalCompetitors</td><td>25</td><td>355</td><td>0</td></tr><tr><td>TotalSubmissions</td><td>22</td><td>596</td><td>0</td></tr><tr><td>LicenseName</td><td>Subject to Competition Rules</td><td>5</td><td>0</td></tr><tr><td>Overview</td><td># Background

The Eurovision Song Contest is an annual competition broadcast worldwide which is open to entrants from active members of the European Broadcasting Union (EBU).Â  This is not defined by Europe's geographical bounds; for example Israel and Cyprus compete in the contest.Â  This year's contest is being held in Oslo, Norway.Â  The location is usually the country of the previous year's winner.Â  The number of entrants changes on a yearly basis, due to flux in EBU membership, political shifts such as the dissolution of the Former Soviet Union and Yugoslavia, and desire (or lack thereof) of countries to participate.Â  Â &lt;br /&gt;&lt;br /&gt;Each country must submit one song to represent it, which must not have been previously commercially released.Â  All songs are, however, made public by a deadline several weeks before the contest, and this year's entrants are all known already with details such as title, language and artist included in the database provided.Â Â  Since 1999 songs may be sung in any language, whereas prior to this point (up to 1998 inclusive) songs had to be sung in one of the entrant country's national languages.&lt;br /&gt;&lt;br /&gt;Although the voting format has changed over time, the modern positional voting system has been in place since 1975.Â  Voting countries award scores of 1, 2, 3, 4, 5, 6, 7, 8, 10, and 12 to competitors, with 12 allocated to the favourite song.Â  Each score can only be awarded once; ie: each voting country can only allocate scores to 10 countries, with all other countries automatically allocated a score of 0.Â  These scores are then added up to determine a competitor's final score.Â  No country can vote for itself.&lt;br /&gt;&lt;br /&gt;Although mass televoting has been the primary mode of final voting since its implementation in 1998, each country also has a back-up jury in the event of failure of the televoting system.Â  Some countries also supplement televotes with sms votes.&lt;br /&gt;&lt;br /&gt;Despite the introduction of semi-finals in 2004, the four same countries - Germany, Spain, United Kingdom and France - automatically qualify for the final each year.Â  This means that they do not compete in the semi-final.Â  Additionally, the host country - this year, Norway - is also exempt from performing in the semi-final, automatically qualifying for the final.Â  However, these five countries do vote in the semi-final, specifically France, Germany and Spain in semi-final 1, and United Kingdom and Norway in semi-final 2 this year. &lt;br /&gt;
----------------------
# Custom

&lt;b&gt;Hint 1&lt;/b&gt;&lt;br /&gt;We suggest browsingÂ Dr. Tim Cocx'sÂ 'Analysis on the Eurovision Songcontest' available at &lt;a href="http://www.liacs.nl/%7Etcocx/songfestival/indexenglish.html" mce_href="http://www.liacs.nl/~tcocx/songfestival/indexenglish.html"&gt;http://www.liacs.nl/~tcocx/songfestival/indexenglish.html&lt;/a&gt;.Â  It contains an interesting document - a 'Distance matrix' - whichÂ represent an abstract (non-geographic) distance that denotes how many points the countries awarded one another in past contests, where a shorter distance implies higher exchanged scores and a larger distanceÂ impliesÂ fewer exchanged points.Â  You may consider it worth utilising or producing your own, up-to-date, version based on a similar concept&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Hint 2&lt;/b&gt;&lt;br /&gt;One way to approach this competition might be to work out that country A normally allocates country B between 4 and 10 votes (with the precise score depending on the performance's quality). Once you have determined these ranges for all countries, you can use betting market data as a proxy for performance quality. &lt;br /&gt;&lt;br /&gt;&lt;b&gt;Hint 3&lt;/b&gt;&lt;br /&gt;If you're thinking about augmenting our dataset, you may be want to try and track down data on expatriate populations in each voting country.&lt;br /&gt;
----------------------
# Custom

Thank you to Stephanie Wayne for putting this competition together.&lt;br /&gt;&lt;br /&gt;We would also like to thank Betfair Pty Ltd for supplyingÂ betting pricesÂ data
----------------------
# Description

&lt;B&gt;The intrigue of Eurovision - the modeller's delight&lt;/B&gt;&lt;BR /&gt;&lt;BR /&gt;Since its inauguration in 1956, the Eurovision Song Contest has transfixed millions of viewers worldwide with its bubblegum pop, quirks, ballads and flops.Â  During its tenure as one of the world's longest-running television programs, it has attracted its fair share of controversy, with claims that the voting outcomes don't simply reflect performance quality but are influenced by factors such as regional politics, expatriate populations, alliances, and artists' sexual appeal.Â  Patterns have appeared over time which support this hypothesis. &lt;BR /&gt;&lt;BR /&gt;Contestants will attempt to exploit these patterns in order to predict the voting for the 2010 Eurovision Song Contest. &lt;BR /&gt;&lt;BR /&gt;&lt;B&gt;Down to the details&lt;/B&gt;&lt;BR /&gt;&lt;BR /&gt;We will provide you with publicly-available data on variables we have deemed potentially important, such as historical voting patterns, betting odds, song details and artist details.Â  You are welcome to supplement this with any data you believe will improve the accuracy of your model.&lt;BR /&gt;&lt;BR /&gt;As the voting method for the Contest has changed over time, we will supply 'Finals' data from 1998 when the current "televoting" format was implemented, and 'semi-finals' data from 2004 when semi-finals were introduced.Â  We will detail Eurovision's voting procedures and their minor changes/exceptions pertinent to the provided data in the 'Data' page of this competition. &lt;BR /&gt;&lt;BR /&gt;Your entry will involve submitting a matrix of the individual votes of each voting country for the competitor countries. &lt;BR /&gt;&lt;BR /&gt;&lt;B&gt;What is your incentive to enter?&lt;/B&gt;&lt;BR /&gt;&lt;BR /&gt;There will be a cash prize of USD1000 for the contestant who produces the most accurate predictions. And just as the Contest has launched the high-flying careers of its own performers - notably ABBA and CÃ©line Dion - creating a strong model for this competition will land you a top ranking as a modeler on the Kaggle website.
----------------------
# Evaluation

&lt;P&gt;Evaluation will be performed using the 'Absolute Error' method.Â A submission's absolute error is calculated by summing the absolute difference between predicted and actual votes.Â  A low score implies a small deviation of predicted votes from actual votes; a higher score implies a larger difference between predicted votes and actual votes.Â Â The most accurate prediction has the lowest Absolute Error score.Â  &lt;BR /&gt;&lt;BR /&gt;Here is an example of how 'Absolute Error' evaluation works:&lt;BR /&gt;&lt;BR /&gt;Predicted voting (submitted by contestant X)&lt;BR /&gt;Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â FranceÂ Â Â Â Â Â Moldova Â Â Â  Norway&lt;BR /&gt;FranceÂ Â Â Â Â Â Â Â Â 0Â Â Â Â Â Â Â Â Â Â Â Â Â  1Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 1&lt;BR /&gt;Moldova Â Â Â Â Â  2Â Â Â Â Â Â Â Â Â Â Â Â Â  0Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 2&lt;BR /&gt;NorwayÂ Â Â Â Â Â Â  1Â Â Â Â Â Â Â Â Â Â Â Â Â  2Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 0&lt;/P&gt;
&lt;P&gt;&lt;BR /&gt;Actual voting&lt;BR /&gt;Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â FranceÂ Â Â Â Â Â Moldova Â Â Â  Norway&lt;BR /&gt;FranceÂ Â Â Â Â Â Â Â Â 0Â Â Â Â Â Â Â Â Â Â Â Â Â Â 1Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 2&lt;BR /&gt;Moldova Â Â Â Â Â  1Â Â Â Â Â Â Â Â Â Â Â Â Â Â 0Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 1&lt;BR /&gt;NorwayÂ Â Â Â Â Â Â Â 2Â Â Â Â Â Â Â Â Â Â Â Â Â Â 2Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â 0 &lt;/P&gt;
&lt;P&gt;ContestantÂ X's Absolute Error score,Â working row-by-row, would be calculated as follows:Â &lt;BR /&gt;Absolute Error(X) = abs(0-0) + abs(1-1) + abs(2-1) + abs(1-2) + abs(0-0) + abs(1-2) + abs(2-1) + abs(2-2) + abs(0-0)&lt;BR /&gt;Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â  = 0 + 0 + 1 + 1 + 0 + 1 + 1+ 0 + 0&lt;BR /&gt;Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â Â  = 4 &lt;/P&gt;
----------------------
# Prizes

The Eurovision Song Contest 2010 forecasting competition will award a prize of US$1000 to the winning entrant.&lt;br /&gt;The prize money will only be awarded once the winner has sent a brief description of the methodology employed to generate his/her submission.
----------------------
# SubmissionInstructions

SubmissionsÂ must be made using theÂ template attached.&lt;br /&gt;&lt;br /&gt;The countries listed in column one are the competitors in the Eurovision Song Contest 2010 final. The countries listed in row one are the voting countries. Only 25 of these 39 competing (column one) countries will progress from the semi-finals to the final, however each voting country listed in (row one) will vote in the final.Â  &lt;br /&gt;&lt;br /&gt;Your task effectively involves  two steps:&lt;br /&gt;1. Firstly you must predict which 25 countries will make it to the final. Keep in mind thatÂ Spain, United
Kingdom, France, Germany and NorwayÂ are guaranteed final competitors.&lt;br /&gt;2. Once you have chosen your 25 finalists you must predict the voting (ie who will vote for who). Make sure you only allocate votes to the 25 countries you predict will make the final. &lt;br /&gt;&lt;br /&gt;Submissions will only be accepted if each voting country (in row one) has allocated all its votes once to the potential final competitors (in column one), ie:Â the scores of 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 have been allocated exactly once per voting country.&lt;br /&gt;Submissions with scores other than 1, 2, 3, 4, 5, 6, 7, 8, 10 or 12 will not be accepted.&lt;br /&gt;Submissions will only be accepted if votes are allocated to â‰¤ 25 countries.Â  Submissions with votes allocated to &gt; 25 countries will not be accepted. Note: all 39 countries vote regardless of whether or not they make the final. &lt;br /&gt;&lt;br /&gt;EachÂ contestant may only enter with one submission. However, a contestant can delete and replace their submission at any time up until the competition's deadline. &lt;br /&gt;&lt;br /&gt;As exemplified by the example submission, please ensure that there are no blank cells; instead, where no vote is allocated, enter a '0'.&lt;br /&gt;&lt;br /&gt;Note: you can open CSV files with Microsoft Excel or OpenOffice for more human-friendly viewing. &lt;br /&gt;</td><td>975</td><td>1</td></tr><tr><td>Rules</td><td># Competition Rules


	&lt;!-- Begin Base Rules --&gt;
    &lt;h3&gt;One account per participant&lt;/h3&gt;
    &lt;p&gt;You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.&lt;/p&gt;
    &lt;h3&gt;No private sharing outside teams&lt;/h3&gt;
    &lt;p&gt;
        Privately sharing code or data outside of teams is not permitted.
        It's okay to share code if made available to all participants on the forums.
    &lt;/p&gt;
    &lt;h3&gt;Team Mergers&lt;/h3&gt;
    &lt;p&gt;
        Team mergers are allowed and can be performed by the team leader. In order to merge, the combined team must have a total submission count less than or equal to the maximum allowed as of the merge date. The maximum allowed is the number of submissions per day multiplied by the number of days the competition has been running.&lt;br/&gt;
    &lt;/p&gt;

    &lt;h3&gt;Team Limits&lt;/h3&gt;
    &lt;p&gt;There is no maximum team size.&lt;/p&gt;
    &lt;h3&gt;Submission Limits&lt;/h3&gt;
    &lt;p&gt;You may submit a maximum of 5 entries per day.&lt;/p&gt;
    &lt;p&gt;You may select up to 5 final submissions for judging.&lt;/p&gt;
    &lt;h3&gt;Competition Timeline&lt;/h3&gt;
    &lt;p&gt;Start Date: &lt;strong&gt;4/7/2010 7:57 AM UTC&lt;/strong&gt;&lt;/p&gt;
    &lt;p&gt;Merger Deadline: &lt;strong&gt;None&lt;/strong&gt;&lt;/p&gt;
    &lt;p&gt;Entry Deadline: &lt;strong&gt;None&lt;/strong&gt;&lt;/p&gt;
    &lt;p&gt;End Date: &lt;strong&gt;5/25/2010 6:00 PM UTC&lt;/strong&gt;&lt;/p&gt;
	&lt;!-- End Base Rules --&gt;
&lt;b&gt;Who may enter&lt;/b&gt;&lt;br /&gt;&lt;br /&gt;Anybody who complies with the rules of the challenge Eurovision Song Contest 2010Â is welcome to participate.Â Â &lt;br /&gt;Only the competition organisers and employees of the European Broadcasting Union are excluded from participating.Â  &lt;br /&gt;Contestants mayÂ enter as individuals or as groups, however the subsequent ranking achieved by their model will be allocated to their individual self/group accordingly.Â  &lt;br /&gt;All entrants must identify themselves by registering on Kaggle website, however personal data will only be utilised by the competition organisers and Kaggle in order to communicate with participants.Â Â &lt;br /&gt;&lt;br /&gt;&lt;b&gt;Submission details&lt;/b&gt;&lt;br /&gt;&lt;br /&gt;Submissions should be made via the 'Make a Submission' page in order to be ranked.Â Â &lt;br /&gt;Each contestant may only enter one submission.Â  &lt;br /&gt;If you encounter anyÂ problems with the submission process, please contact the Kaggle administrator.&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Evaluation and prize-giving&lt;/b&gt;&lt;br /&gt;&lt;br /&gt;The scoring method is detailed on the 'Evaluation' page.Â Â &lt;br /&gt;The winner of the Eurovision Song Contest 2010 will only receive the prize money afterÂ a brief description ofÂ their methodology has been submitted to the competition host.Â Â &lt;br /&gt;We may ask other top-ranking participants to voluntarily fill out a form regarding their modellingÂ methods.</td><td>720</td><td>2</td></tr><tr><td>DatasetDescription</td><td># Dataset Description

&lt;p&gt;&lt;b&gt;Data provided&lt;br /&gt;&lt;/b&gt;&lt;br /&gt;Historical data for 'Finals' is provided from 1998, when mass televoting was first implemented.Â  &lt;br /&gt;Historical data for 'Semi-finals' is provided from 2004, when semi-finals were introduced.Â  Please note that from 2004 to 2007 there was one semi-final; from 2008 onwards there have been two semi-finals.&lt;br /&gt;&lt;br /&gt;The dataset incorporatesÂ 3 sheets.Â  These are provided in a single Microsoft Excel file, or alternatively the identical sheets areÂ available in a different formatÂ asÂ 3 separateÂ csv files.&lt;br /&gt;&lt;br /&gt;TheÂ 3 sheets are as follows:&lt;br /&gt;1)Â  'Final data by year'&lt;br /&gt;2)Â  'Semi-final data by year'&lt;br /&gt;3)Â  '2010 Data'&lt;br /&gt;&lt;br /&gt;&lt;i&gt;Final data by year&lt;/i&gt;&lt;u&gt;&lt;br /&gt;&lt;/u&gt;&lt;/p&gt;
&lt;p&gt;The data in this sheet can be used to calibrate your model.&lt;br /&gt;&lt;/p&gt;
&lt;p&gt;Column A - Year of competition, from 1998 - 2009&lt;br /&gt;Column B - Competitor country&lt;br /&gt;Column C - Region of competitor country (nb: please see 'Region Classifications' below)&lt;br /&gt;Column D - Language of song entered by competitor country&lt;br /&gt;Column E - Name of performing artist&lt;br /&gt;Column F - Name of song&lt;br /&gt;Column G - Name of song, English translation&lt;br /&gt;Column H - Gender of artist (nb: this is designated as "male" or "female" or "both" if a mixed-gender group)&lt;br /&gt;Column I - Group or solo performer&lt;br /&gt;Column J - Place achieved in final (nb: the # of competitors and thus places has changed on a yearly basis)&lt;br /&gt;Column K - Total points received by competing country, an aggregate of votes&lt;br /&gt;Column L - Host country&lt;br /&gt;Column M - Region of host country (nb: please see 'Region Classifications' below)&lt;br /&gt;Column N - Whether the Contest was held in the competitor's own country ("Home") or another country ("Away")&lt;br /&gt;Column O - Whether the Contest was held in the competitor's own region ("Home") or another region ("Away")&lt;br /&gt;Column P - Approximate betting prices (nb: please see 'Approximate Betting Prices data' below)&lt;br /&gt;Columns Q-BK - Voting countries and their allocation of scores to competitor countries listed in Column A (nb: these columns contain the list of all countries which have competed in the Eurovision Song Contest Final from 1998-2009, however not every country voted/entered the Contest every year)&lt;br /&gt;&lt;br /&gt;&lt;i&gt;Semi-final data by year&lt;/i&gt;&lt;u&gt;&lt;br /&gt;&lt;/u&gt;&lt;/p&gt;
&lt;p&gt;&lt;br /&gt;The data in this sheet can also be used to calibrate your model.&lt;/p&gt;
&lt;p&gt;Column A - Year of competition, fromÂ 2004 - 2009, and designation as "SF1" (semi-final 1) or "SF2" (semi-final 2).Â  (nb: for years 2004-2007, there was only one semi-final per year, and these have been designated as "SF1")&lt;br /&gt;Column B - Competitor country&lt;br /&gt;Column C - Region of competitor country (nb: please see 'Region Classifications' below)&lt;br /&gt;Column D - Language of song entered by competitor country&lt;br /&gt;Column E - Name of performing artist&lt;br /&gt;Column F - Name of song&lt;br /&gt;Column G - Name of song, English translation&lt;br /&gt;ColumnÂ H - Place achieved in final (nb: the # of competitors and thus places has changed on a yearly basis)&lt;br /&gt;ColumnÂ I - Total points received by competing country, an aggregate of votes&lt;br /&gt;ColumnÂ J - Host country&lt;br /&gt;ColumnÂ K - Region of host country (nb: please see 'Region Classifications' below)&lt;br /&gt;ColumnÂ L - Whether the Contest was held in the competitor's own country ("Home") or another country ("Away")&lt;br /&gt;ColumnÂ M - Whether the Contest was held in the competitor's own region ("Home") or another region ("Away")&lt;br /&gt;ColumnÂ N - Approximate betting prices (nb: please see 'Approximate Betting Prices data' below)&lt;br /&gt;Columns O-BI - Voting countries and their allocation of scores to competitor countries listed in Column A (nb: these columns contain the list of all countries which have competed in the Eurovision Song Contest semi-finals from 2004-2009, however not every country voted/entered the Contest every year)&lt;br /&gt;&lt;br /&gt;&lt;i&gt;2010 Data&lt;/i&gt;&lt;u&gt;&lt;br /&gt;&lt;/u&gt;&lt;/p&gt;
&lt;p&gt;The data in this sheet can be used to generate your entry.Â &lt;/p&gt;
&lt;p&gt;Column A - Year of competition, ie: 2010&lt;br /&gt;Column B - Competitor country (nb: this contains list of all 2010 Contest contestants, of which only 25 will compete in the final.Â  Germany, United Kingdom, Spain, France and Norway will definitely compete in the Final)&lt;br /&gt;Column C - Region of competitor country (nb: please see 'Region Classifications' below)&lt;br /&gt;Column D - Language of song entered by competitor country&lt;br /&gt;Column E - Name of performing artist&lt;br /&gt;Column F - Name of song&lt;br /&gt;Column G - Name of song, English translation&lt;br /&gt;Column H - Gender of artist (nb: this is designated as "male" or "female" or "both" if a mixed-gender group)&lt;br /&gt;Column I - Group or solo performer&lt;br /&gt;Column J - Place achieved in final&lt;br /&gt;Column K - Total points received by competing country, an aggregate of votes&lt;br /&gt;Column L - Host country&lt;br /&gt;Column M - Region of host country (nb: please see 'Region Classifications' below)&lt;br /&gt;Column N - Whether the ContestÂ is being held in the competitor's own country ("Home") or another country ("Away") (nb: this will read "Home" for Norway and "Away" for every other country)&lt;br /&gt;Column O - Whether the ContestÂ is beingÂ held in the competitor's own region ("Home") or another region ("Away") (nb: this will read "Home" for Scandinavian countries and "Away" for all other countries)&lt;br /&gt;Column P - Approximate betting prices (nb: please see 'Approximate Betting Prices data' below)&lt;br /&gt;&lt;br /&gt;&lt;b&gt;This year's participants&lt;br /&gt;&lt;br /&gt;&lt;/b&gt;39 countries have confirmed their participation in this year's Eurovision Song Contest.Â  &lt;br /&gt;&lt;i&gt;Semi-final 1&lt;/i&gt; - Moldova, Russia, Estonia, Slovakia, Finland, Latvia, Serbia, Bosnia and Herzegovina, Poland, Belgium, Malta, Albania, Greece, Portugal, Macedonia, Belarus, Iceland&lt;br /&gt;&lt;i&gt;Semi-final 2&lt;/i&gt; - Lithuania, Armenia, Israel, Denmark, Switzerland, Sweden, Azerbaijan, Ukraine, Netherlands, Romania, Slovenia, Ireland, Bulgaria, Cyprus, Croatia, Georgia, Turkey&lt;br /&gt;&lt;i&gt;Straight to final&lt;/i&gt; - Spain, Norway, United Kingdom, France, Germany&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Voting procedure in semi-finals&lt;br /&gt;&lt;/b&gt;&lt;br /&gt;Semi-finals were introduced in 2004.Â  There was one semi-final every year from 2004-2007, and two semi-finals per year for 2008 onwards (including this year).Â Â  &lt;br /&gt;&lt;br /&gt;As explained in the 'Background' sheet, a 'positional voting' system is used, whereby each voting country must allocate scores of 1, 2, 3, 4, 5, 6, 7, 8, 10, and 12 to competitor countries.Â  Only the countries competing in that semi-final vote, the exception being the 5 countries which automatically qualify for the final.Â  In 2008, Germany and Spain voted in semi-final 1; France, United Kingdom and Serbia voted in semi-final 2.Â  In 2009, Germany and United Kingdom voted in semi-final 1; France, Spain and Russia voted in semi-final 2.Â  This year, France, Germany and Spain will vote in semi-final 1; Norway and United Kingdom will vote in semi-final 2.&lt;i&gt;&lt;br /&gt;&lt;br /&gt;Voting procedure for 2004-2007&lt;/i&gt; - the top 10 from the previous year's final and this year's semi-final would qualify for this year's final.Â  (eg: in 2004, the final contestants were the top 10 from 2004's semi-final plus the top 10 from 2003's final.)&lt;i&gt;&lt;br /&gt;Voting procedure for 2008-2009&lt;/i&gt; - the top 9 from each semi-final, as determined by televote, would proceed to the final.Â  Additionally, the top country in each semi-finalÂ as voted byÂ the back-up juries, butÂ which was not in the top 9 televotes, would also proceed to the final.Â  In total, this meant 9 televotes + 1 jury wild-card per semi-final, ie: 20 final contestants.&lt;br /&gt;&lt;i&gt;Voting procedure for 2010&lt;/i&gt;Â - replacing the system from the previous 2 years, the top 10 from each semi-final, determined by 50% jury vote and 50% televote, will proceed to the final.&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Voting procedure in the final&lt;br /&gt;&lt;br /&gt;&lt;/b&gt;As explained in the 'Background' sheet, a 'positional voting' system is used, whereby each voting country must allocate scores of 1, 2, 3, 4, 5, 6, 7, 8, 10, and 12 to competitor countries.Â  All countries involved in the Contest,Â whether knocked out after the semi-finals or competing in the final, must vote.Â  Thus, in this year's final, there will be 25 competing countries, and 39 voting countries.&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Region classifications&lt;br /&gt;&lt;br /&gt;&lt;/b&gt;We have classified the countries by region as follows:&lt;br /&gt;&lt;br /&gt;&lt;i&gt;Scandinavia&lt;/i&gt; - Denmark, Finland, Iceland, Norway, Sweden&lt;br /&gt;&lt;i&gt;Western Europe&lt;/i&gt; - Andorra, Austria, Belgium, France, Germany, Greece,Â Italy, Luxembourg, Malta, Monaco, Netherlands, Portugal, San Marino,Â Spain, Switzerland&lt;br /&gt;&lt;i&gt;Independent&lt;/i&gt; - Cyprus, Ireland, Israel, Turkey, United Kingdom&lt;br /&gt;&lt;i&gt;Former Socialist Bloc&lt;/i&gt; - Albania, Armenia, Azerbaijan, Belarus, Bulgaria, Czech Republic, Estonia, Georgia, Hungary, Latvia, Lithuania, Moldova, Poland, Romania,Â Russia, Slovakia, Ukraine&lt;br /&gt;&lt;i&gt;Former Yugoslavia&lt;/i&gt; - Bosnia and Herzegovina, Croatia, Kosovo, Macedonia, Montenegro / Serbia and MontenegroÂ / Serbia, Slovenia&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Approximate Betting Prices data&lt;br /&gt;&lt;br /&gt;&lt;/b&gt;You will notice that Column P 'Approximate Betting Prices' in the '2010 Data' sheet has been filled however this data isÂ obviouslyÂ constantly changing.Â  For the most up-to-date price data, weÂ suggest using theÂ following website:&lt;br /&gt;&lt;a href="http://www.oddschecker.com/specials/tv/eurovision/win-market" mce_href="http://www.oddschecker.com/specials/tv/eurovision/win-market"&gt;&lt;span style="color: rgb(0, 0, 0);" mce_style="FONT-FAMILY: ; COLOR: #000000"&gt;http://www.oddschecker.com/specials/tv/eurovision/win-market&lt;/span&gt;&lt;/a&gt;&lt;br mce_bogus="1" /&gt;&lt;br /&gt;The historical 'Approximate Betting Prices' data (Column P and Column N in the 'Final data by year' and 'Semi-final data by year' sheets respectively) is provided by Betfair, and is an average over time of betting pricesÂ for each competitor country from one particular betting website, Betfair Pty Ltd.Â  In contrast, the above link for oddschecker.com provides spot data which averages betting prices for each competitor country across a range of betting websites.Â Â &lt;br /&gt;&lt;br /&gt;The 'Approximate Betting Prices' data for semi-finals is the same as for finals.&lt;br /&gt;&lt;br /&gt;&lt;b&gt;Anomalies by year including exceptions to televoting rule&lt;/b&gt;&lt;br /&gt;&lt;br /&gt;&lt;i&gt;1998&lt;/i&gt; - Hungary, Ireland, Romania and Turkey used back-up jury vote&lt;br /&gt;&lt;i&gt;1999&lt;/i&gt; - Lithuania, Turkey, Ireland, and Bosnia and Herzegovina used back-up jury vote&lt;br /&gt;&lt;i&gt;2000&lt;/i&gt; - Netherlands, Romania, Russia, Macedonia and Turkey used back-up jury vote&lt;br /&gt;&lt;i&gt;2001&lt;/i&gt; - Bosnia and Herzegovina, Russia and Turkey usedÂ back-up jury vote; Croatia, Malta and Greece used a 50/50 jury/televote system&lt;br /&gt;&lt;i&gt;2002&lt;/i&gt; - Russia, Bosnia and Herzegovina, Turkey and Romania used back-up jury vote; Cyprus, Greece, Croatia and Malta used a 50/50 jury/televote system&lt;br /&gt;&lt;i&gt;2003&lt;/i&gt; - Ireland, Bosnia and Herzegovina and Russia used back-up jury vote&lt;br /&gt;&lt;i&gt;2004&lt;/i&gt; - Turkey did not hear Slovenia's entry due to technical faults and therefore could not vote for it&lt;br /&gt;&lt;i&gt;2005&lt;/i&gt; - Monaco, Andorra and Moldova used back-up jury vote in the final.Â  (Andorra, Monaco and Albania used back-up jury vote in the semi-final)&lt;br /&gt;&lt;i&gt;2006&lt;/i&gt; - Monaco and Albania used back-up jury vote in the final.Â  (Andorra, Monaco and Albania used back-up jury vote in the semi-final)&lt;br /&gt;&lt;i&gt;2007&lt;/i&gt; - Albania and Andorra used back-up jury vote in the final.Â  (Albania and Andorra used back-up jury vote in the semi-final).Â  This was the first year that Serbia and Montenegro competed as separate entities&lt;br /&gt;&lt;i&gt;2008&lt;/i&gt; - Germany, Poland and United Kingdom all received a total of 14 points in the final however have been ranked in the provided data as 23rd, 24th and 25th respectively, as Germany received the most 12s, Poland the most 10s, and United Kingdom the fewest 2009 - Norway used back-upÂ jury vote and Hungary used sms vote only in the final.Â  (Spain and Albania used back-up jury vote in the semi-final).&lt;/p&gt;&lt;p&gt;Note: you can open CSV files with Microsoft Excel or OpenOffice for more human-friendly viewing. &lt;/p&gt;</td><td>867</td><td>64</td></tr><tr><td>TotalCompressedBytes</td><td>800614.0</td><td>904</td><td>14</td></tr><tr><td>TotalUncompressedBytes</td><td>400307.0</td><td>905</td><td>14</td></tr><tr><td>ValidationSetName</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>ValidationSetValue</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>EnableSubmissionModelHashes</td><td>False</td><td>1</td><td>0</td></tr><tr><td>EnableSubmissionModelAttachments</td><td>False</td><td>1</td><td>0</td></tr><tr><td>HostName</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>CompetitionTypeId</td><td>1</td><td>1</td><td>0</td></tr></table><hr><h2>ğŸ“� DatasetTags.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>695474</td><td>1000</td><td>0</td></tr><tr><td>DatasetId</td><td>1051558</td><td>424</td><td>0</td></tr><tr><td>TagId</td><td>16113</td><td>178</td><td>0</td></tr></table><hr><h2>ğŸ“� DatasetTaskSubmissions.csv</h2><p><strong>Total Columns:</strong> 7</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>26</td><td>1000</td><td>0</td></tr><tr><td>DatasetTaskId</td><td>125</td><td>181</td><td>0</td></tr><tr><td>SubmittedUserId</td><td>998023.0</td><td>461</td><td>6</td></tr><tr><td>CreationDate</td><td>12/08/2019 23:38:32</td><td>1000</td><td>0</td></tr><tr><td>KernelId</td><td>6951358.0</td><td>912</td><td>11</td></tr><tr><td>DatasetId</td><td>911790.0</td><td>11</td><td>989</td></tr><tr><td>AcceptedDate</td><td>12/09/2019 10:18:28</td><td>41</td><td>959</td></tr></table><hr><h2>ğŸ“� DatasetTasks.csv</h2><p><strong>Total Columns:</strong> 10</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1</td><td>1000</td><td>0</td></tr><tr><td>DatasetId</td><td>318093</td><td>659</td><td>0</td></tr><tr><td>OwnerUserId</td><td>998023</td><td>477</td><td>0</td></tr><tr><td>CreationDate</td><td>11/05/2019 20:30:00</td><td>1000</td><td>0</td></tr><tr><td>Description</td><td>### Task Details
This dataset has a lot of data about mobile apps, including the rating. What we don't know is when and why an app will get a good rating.

The goal of this task is to train a model that can predict the average user rating of an app. Bonus points if that model isn't just a black box, but can explain why that is. Feel free to use any means for this explanation (correlations, general EDA, model weights, ...).

### Expected Submission
Kaggle Notebooks containing:
- A quick explanation what type of model you're training and any thoughts about your approach
- Steps to actually train the model
- An evaluation of accuracy using **mean absolute error**
- Bonus: Anything you can do to explain which factors are most important (feature selection, EDA, ...)

Feel free to use and join any additional data source beyond the data contained in this dataset.

### Evaluation
Evaluation will be by **mean absolute error**. But subjective bonus points for a good explanation!</td><td>905</td><td>1</td></tr><tr><td>ForumId</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>Title</td><td>Explain user ratings</td><td>915</td><td>0</td></tr><tr><td>Subtitle</td><td>Create a notebook that explains which games get high ratings</td><td>363</td><td>611</td></tr><tr><td>Deadline</td><td>01/27/2019 00:00:00</td><td>141</td><td>747</td></tr><tr><td>TotalVotes</td><td>14</td><td>83</td><td>0</td></tr></table><hr><h2>ğŸ“� DatasetVersions.csv</h2><p><strong>Total Columns:</strong> 14</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>13</td><td>1000</td><td>0</td></tr><tr><td>DatasetId</td><td>13</td><td>354</td><td>0</td></tr><tr><td>DatasourceVersionId</td><td>13</td><td>1000</td><td>0</td></tr><tr><td>CreatorUserId</td><td>1</td><td>189</td><td>0</td></tr><tr><td>LicenseName</td><td>CC0: Public Domain</td><td>12</td><td>0</td></tr><tr><td>CreationDate</td><td>12/18/2015 23:16:00</td><td>1000</td><td>0</td></tr><tr><td>VersionNumber</td><td>1</td><td>111</td><td>0</td></tr><tr><td>Title</td><td>US Baby Names</td><td>349</td><td>0</td></tr><tr><td>Slug</td><td>us-baby-names</td><td>350</td><td>0</td></tr><tr><td>Subtitle</td><td>Explore naming trends from babies born in the US</td><td>254</td><td>140</td></tr><tr><td>Description</td><td>US Social Security applications are a great way to track trends in how babies born in the US are named.

Data.gov releases two datasets that are helplful for this: one at the [national level](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-level-data) and another [at the state level](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-data-by-state-and-district-of-). Note that only names with at least 5 babies born in the same year (/ state) are included in this dataset for privacy.

[![benjamin](https://www.kaggle.io/svf/153725/b6c3f30368aeb2b277016b0582d1eab6/nameOverTime.png)](https://www.kaggle.com/benhamner/d/kaggle/us-baby-names/babies-named-benjamin-over-time)

I've taken the raw files here and combined/normalized them into two CSV files (one for each dataset) as well as a SQLite database with two equivalently-defined tables. The code that did these transformations is [available here](https://github.com/benhamner/us-baby-names/).

*New to data exploration in R? Take the free, interactive DataCamp course, "[Data Exploration With Kaggle Scripts](https://www.datacamp.com/courses/data-exploration-with-kaggle-scripts)," to learn the basics of visualizing data with ggplot. You'll also create your first Kaggle Scripts along the way.*</td><td>350</td><td>236</td></tr><tr><td>VersionNotes</td><td>Initial release</td><td>465</td><td>0</td></tr><tr><td>TotalCompressedBytes</td><td>363493400</td><td>873</td><td>0</td></tr><tr><td>TotalUncompressedBytes</td><td>650553507</td><td>823</td><td>0</td></tr></table><hr><h2>ğŸ“� DatasetVotes.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>900566</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>4178745</td><td>981</td><td>0</td></tr><tr><td>DatasetVersionId</td><td>7651</td><td>9</td><td>0</td></tr><tr><td>VoteDate</td><td>03/15/2021</td><td>737</td><td>0</td></tr></table><hr><h2>ğŸ“� Datasets.csv</h2><p><strong>Total Columns:</strong> 16</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>402034</td><td>1000</td><td>0</td></tr><tr><td>CreatorUserId</td><td>2188609</td><td>81</td><td>0</td></tr><tr><td>OwnerUserId</td><td>2188609.0</td><td>70</td><td>446</td></tr><tr><td>OwnerOrganizationId</td><td>4.0</td><td>24</td><td>554</td></tr><tr><td>CurrentDatasetVersionId</td><td>771250</td><td>1000</td><td>0</td></tr><tr><td>CurrentDatasourceVersionId</td><td>792478</td><td>1000</td><td>0</td></tr><tr><td>ForumId</td><td>414197</td><td>1000</td><td>0</td></tr><tr><td>Type</td><td>Dataset</td><td>1</td><td>0</td></tr><tr><td>CreationDate</td><td>10/31/2019 01:07:32</td><td>998</td><td>0</td></tr><tr><td>LastActivityDate</td><td>10/31/2019</td><td>192</td><td>0</td></tr><tr><td>TotalViews</td><td>2676</td><td>928</td><td>0</td></tr><tr><td>TotalDownloads</td><td>127</td><td>434</td><td>0</td></tr><tr><td>TotalVotes</td><td>1</td><td>120</td><td>0</td></tr><tr><td>TotalKernels</td><td>2</td><td>93</td><td>0</td></tr><tr><td>Medal</td><td>1.0</td><td>3</td><td>825</td></tr><tr><td>MedalAwardDate</td><td>08/05/2021</td><td>119</td><td>826</td></tr></table><hr><h2>ğŸ“� Datasources.csv</h2><p><strong>Total Columns:</strong> 5</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>6</td><td>1000</td><td>0</td></tr><tr><td>CreatorUserId</td><td>1</td><td>635</td><td>0</td></tr><tr><td>CreationDate</td><td>05/01/2017 19:10:03</td><td>1000</td><td>0</td></tr><tr><td>Type</td><td>Dataset</td><td>1</td><td>0</td></tr><tr><td>CurrentDatasourceVersionId</td><td>2158</td><td>1000</td><td>0</td></tr></table><hr><h2>ğŸ“� EpisodeAgents.csv</h2><p><strong>Total Columns:</strong> 10</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>35727</td><td>1000</td><td>0</td></tr><tr><td>EpisodeId</td><td>18132</td><td>504</td><td>0</td></tr><tr><td>Index</td><td>0</td><td>2</td><td>0</td></tr><tr><td>Reward</td><td>0.0</td><td>3</td><td>86</td></tr><tr><td>State</td><td>2</td><td>4</td><td>0</td></tr><tr><td>SubmissionId</td><td>14028642</td><td>383</td><td>0</td></tr><tr><td>InitialConfidence</td><td>141.65896139647967</td><td>802</td><td>130</td></tr><tr><td>InitialScore</td><td>613.5685714971212</td><td>802</td><td>130</td></tr><tr><td>UpdatedConfidence</td><td>131.14294518480975</td><td>871</td><td>62</td></tr><tr><td>UpdatedScore</td><td>501.5042086195842</td><td>871</td><td>62</td></tr></table><hr><h2>ğŸ“� Episodes.csv</h2><p><strong>Total Columns:</strong> 5</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>17217</td><td>1000</td><td>0</td></tr><tr><td>Type</td><td>4</td><td>2</td><td>0</td></tr><tr><td>CompetitionId</td><td>17592</td><td>1</td><td>0</td></tr><tr><td>CreateTime</td><td>01/03/2020 21:08:01</td><td>549</td><td>0</td></tr><tr><td>EndTime</td><td>01/03/2020 21:08:03</td><td>792</td><td>0</td></tr></table><hr><h2>ğŸ“� ForumMessageReactions.csv</h2><p><strong>Total Columns:</strong> 5</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>151412</td><td>1000</td><td>0</td></tr><tr><td>ForumMessageId</td><td>1723</td><td>140</td><td>0</td></tr><tr><td>FromUserId</td><td>22421183</td><td>828</td><td>0</td></tr><tr><td>ReactionType</td><td>ThankYou</td><td>6</td><td>0</td></tr><tr><td>ReactionDate</td><td>02/27/2025</td><td>364</td><td>0</td></tr></table><hr><h2>ğŸ“� ForumMessageVotes.csv</h2><p><strong>Total Columns:</strong> 5</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>3372915</td><td>1000</td><td>0</td></tr><tr><td>ForumMessageId</td><td>1</td><td>511</td><td>0</td></tr><tr><td>FromUserId</td><td>13067417</td><td>449</td><td>0</td></tr><tr><td>ToUserId</td><td>478</td><td>155</td><td>0</td></tr><tr><td>VoteDate</td><td>04/13/2023</td><td>377</td><td>0</td></tr></table><hr><h2>ğŸ“� ForumMessages.csv</h2><p><strong>Total Columns:</strong> 9</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1</td><td>1000</td><td>0</td></tr><tr><td>ForumTopicId</td><td>1</td><td>144</td><td>0</td></tr><tr><td>PostUserId</td><td>478</td><td>135</td><td>0</td></tr><tr><td>PostDate</td><td>04/28/2010 23:13:08</td><td>144</td><td>0</td></tr><tr><td>ReplyToForumMessageId</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>Message</td><td>&lt;div&gt;In response to a comment on the No Free Hunch Blog &lt;a href="http://kaggle.com/blog/2010/04/27/beating-up-on-hiv/" mce_href="http://kaggle.com/blog/2010/04/27/beating-up-on-hiv/"&gt;here &lt;/a&gt;and on my website &lt;a mce_href="http://www.willdampier.info/2010/a-few-ideas/" href="http://www.willdampier.info/2010/a-few-ideas/"&gt;here&lt;/a&gt; ... I'm listing a few ideas that may give you a head-start on how to get started:&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;There are a whole lotta ways to do this sort of prediction, since I come from a machine-learning background I'll describe it from that perspective: &nbsp;So in my mind I need to extract a set of features (observations) from each sequences, then train a SVM, logistic-regression, decision-forest, or ensemble classifier to learn which features are important. &nbsp;Each of those classifiers have advantages and disadvantages that are very well documented ... a safari through Wikipedia should give you a pretty good idea.&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;The hardest part is deciding which features are worth (or even possible) to put into your model:&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;I know people who use "k-mers" as their features ... this involves finding and counting all of the 5 letter instances in the sequence. &nbsp;Then you can use these as features in a prediction model. &nbsp;K-mers are nice because they are easy to pull out with any programming language you can think of. There is also a list of regular-expressions which have some biological meaning here: http://elm.eu.org/browse.html&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;Other people prefer to use the raw sequence. &nbsp;If you can align the sequences (since they don't all start at the same part of the gene) using a program like ClustalW then you can think of each column as a categorical feature. &nbsp;The problem here is that HIV-1 is highly variable and alignments are difficult ... although not impossible.&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;If you wander around the Los Alamos HIV-1 database you can find a list of known resistance mutations: http://www.hiv.lanl.gov/content/sequence/RESDB/. &nbsp;These have been verified to be important in the viral resistance to certain drugs. &nbsp;You can use the presence or absence of these mutations as features to train a model.&lt;/div&gt;&lt;div&gt;&lt;br&gt;&lt;/div&gt;&lt;div&gt;I'm sure there are dozens of ways to extract features that I've never even heard of so don't think that these are your only choices.&lt;/div&gt;</td><td>996</td><td>0</td></tr><tr><td>RawMarkdown</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>Medal</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>MedalAwardDate</td><td>N/A</td><td>0</td><td>1000</td></tr></table><hr><h2>ğŸ“� ForumTopics.csv</h2><p><strong>Total Columns:</strong> 13</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>445373</td><td>1000</td><td>0</td></tr><tr><td>ForumId</td><td>16860</td><td>1</td><td>0</td></tr><tr><td>KernelId</td><td>43959550</td><td>1000</td><td>0</td></tr><tr><td>LastForumMessageId</td><td>2475879</td><td>1000</td><td>0</td></tr><tr><td>FirstForumMessageId</td><td>2471845</td><td>1000</td><td>0</td></tr><tr><td>CreationDate</td><td>10/06/2023 17:07:00</td><td>999</td><td>0</td></tr><tr><td>LastCommentDate</td><td>10/10/2023 08:11:20</td><td>999</td><td>0</td></tr><tr><td>Title</td><td>[Deleted Kernel]</td><td>1</td><td>973</td></tr><tr><td>IsSticky</td><td>False</td><td>1</td><td>0</td></tr><tr><td>TotalViews</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Score</td><td>27</td><td>110</td><td>0</td></tr><tr><td>TotalMessages</td><td>7</td><td>42</td><td>0</td></tr><tr><td>TotalReplies</td><td>7</td><td>42</td><td>0</td></tr></table><hr><h2>ğŸ“� Forums.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>997619</td><td>1000</td><td>0</td></tr><tr><td>ParentForumId</td><td>1023.0</td><td>8</td><td>8</td></tr><tr><td>Title</td><td>jfk_speeches</td><td>993</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelAcceleratorTypes.csv</h2><p><strong>Total Columns:</strong> 2</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>10</td><td>10</td><td>0</td></tr><tr><td>Label</td><td>GPU A100</td><td>9</td><td>1</td></tr></table><hr><h2>ğŸ“� KernelLanguages.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1</td><td>10</td><td>0</td></tr><tr><td>Name</td><td>R</td><td>10</td><td>0</td></tr><tr><td>DisplayName</td><td>R</td><td>2</td><td>0</td></tr><tr><td>IsNotebook</td><td>False</td><td>2</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelTags.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>23526</td><td>1000</td><td>0</td></tr><tr><td>KernelId</td><td>541</td><td>900</td><td>0</td></tr><tr><td>TagId</td><td>13310</td><td>63</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelVersionCompetitionSources.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>937489</td><td>1000</td><td>0</td></tr><tr><td>KernelVersionId</td><td>6672</td><td>1000</td><td>0</td></tr><tr><td>SourceCompetitionId</td><td>4195</td><td>5</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelVersionDatasetSources.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>931763</td><td>1000</td><td>0</td></tr><tr><td>KernelVersionId</td><td>2479803</td><td>1000</td><td>0</td></tr><tr><td>SourceDatasetVersionId</td><td>5511</td><td>22</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelVersionKernelSources.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>40848</td><td>1000</td><td>0</td></tr><tr><td>KernelVersionId</td><td>2554673</td><td>180</td><td>0</td></tr><tr><td>SourceKernelVersionId</td><td>2239493</td><td>144</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelVersionModelSources.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>494830</td><td>1000</td><td>0</td></tr><tr><td>KernelVersionId</td><td>159457945</td><td>623</td><td>0</td></tr><tr><td>SourceModelVariationVersionId</td><td>992</td><td>74</td><td>0</td></tr><tr><td>SourceModelVariationId</td><td>846</td><td>69</td><td>0</td></tr></table><hr><h2>ğŸ“� KernelVersions.csv</h2><p><strong>Total Columns:</strong> 23</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>722456</td><td>1000</td><td>0</td></tr><tr><td>ScriptId</td><td>112370</td><td>650</td><td>0</td></tr><tr><td>ParentScriptVersionId</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>ScriptLanguageId</td><td>9</td><td>7</td><td>0</td></tr><tr><td>AuthorUserId</td><td>368</td><td>303</td><td>0</td></tr><tr><td>CreationDate</td><td>01/18/2017 01:45:51</td><td>1000</td><td>0</td></tr><tr><td>VersionNumber</td><td>36</td><td>122</td><td>0</td></tr><tr><td>Title</td><td>Calculate Velocity Made Good (VMG)</td><td>658</td><td>0</td></tr><tr><td>EvaluationDate</td><td>01/18/2017</td><td>439</td><td>0</td></tr><tr><td>IsChange</td><td>True</td><td>2</td><td>0</td></tr><tr><td>TotalLines</td><td>213.0</td><td>277</td><td>49</td></tr><tr><td>LinesInsertedFromPrevious</td><td>0.0</td><td>75</td><td>53</td></tr><tr><td>LinesChangedFromPrevious</td><td>0.0</td><td>39</td><td>53</td></tr><tr><td>LinesUnchangedFromPrevious</td><td>400.0</td><td>280</td><td>53</td></tr><tr><td>LinesInsertedFromFork</td><td>14.0</td><td>62</td><td>800</td></tr><tr><td>LinesDeletedFromFork</td><td>0.0</td><td>54</td><td>800</td></tr><tr><td>LinesChangedFromFork</td><td>2.0</td><td>52</td><td>800</td></tr><tr><td>LinesUnchangedFromFork</td><td>43.0</td><td>127</td><td>800</td></tr><tr><td>TotalVotes</td><td>0</td><td>13</td><td>0</td></tr><tr><td>IsInternetEnabled</td><td>False</td><td>1</td><td>0</td></tr><tr><td>RunningTimeInMilliseconds</td><td>43866.0</td><td>973</td><td>7</td></tr><tr><td>AcceleratorTypeId</td><td>0</td><td>1</td><td>0</td></tr><tr><td>DockerImage</td><td>gcr.io/kaggle-private-byod/python@sha256:ad25e224a63d8b0766b1ab12510907df6aa6d31e7a7512c5a058fa64d435f7d4</td><td>1</td><td>970</td></tr></table><hr><h2>ğŸ“� KernelVotes.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>753817</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>3321140</td><td>770</td><td>0</td></tr><tr><td>KernelVersionId</td><td>520</td><td>429</td><td>0</td></tr><tr><td>VoteDate</td><td>06/06/2019</td><td>329</td><td>0</td></tr></table><hr><h2>ğŸ“� Kernels.csv</h2><p><strong>Total Columns:</strong> 16</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1</td><td>1000</td><td>0</td></tr><tr><td>AuthorUserId</td><td>2505</td><td>572</td><td>0</td></tr><tr><td>CurrentKernelVersionId</td><td>205.0</td><td>933</td><td>67</td></tr><tr><td>ForkParentKernelVersionId</td><td>154.0</td><td>172</td><td>626</td></tr><tr><td>ForumTopicId</td><td>26670.0</td><td>79</td><td>921</td></tr><tr><td>FirstKernelVersionId</td><td>1</td><td>1000</td><td>0</td></tr><tr><td>CreationDate</td><td>03/25/2015 18:25:32</td><td>1000</td><td>0</td></tr><tr><td>EvaluationDate</td><td>03/23/2018</td><td>90</td><td>67</td></tr><tr><td>MadePublicDate</td><td>03/25/2015</td><td>72</td><td>0</td></tr><tr><td>IsProjectLanguageTemplate</td><td>False</td><td>2</td><td>0</td></tr><tr><td>CurrentUrlSlug</td><td>hello</td><td>640</td><td>0</td></tr><tr><td>Medal</td><td>3.0</td><td>3</td><td>912</td></tr><tr><td>MedalAwardDate</td><td>07/15/2016</td><td>21</td><td>911</td></tr><tr><td>TotalViews</td><td>432</td><td>698</td><td>0</td></tr><tr><td>TotalComments</td><td>0</td><td>18</td><td>0</td></tr><tr><td>TotalVotes</td><td>0</td><td>51</td><td>0</td></tr></table><hr><h2>ğŸ“� ModelTags.csv</h2><p><strong>Total Columns:</strong> 3</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>135890</td><td>1000</td><td>0</td></tr><tr><td>ModelId</td><td>14</td><td>307</td><td>0</td></tr><tr><td>TagId</td><td>14102</td><td>156</td><td>0</td></tr></table><hr><h2>ğŸ“� ModelVariationVersions.csv</h2><p><strong>Total Columns:</strong> 10</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>3728</td><td>1000</td><td>0</td></tr><tr><td>ModelVariationId</td><td>2655</td><td>914</td><td>0</td></tr><tr><td>ModelVersionId</td><td>311</td><td>269</td><td>0</td></tr><tr><td>DatasourceVersionId</td><td>5092433</td><td>1000</td><td>0</td></tr><tr><td>CreationDate</td><td>02/18/2023 18:15:32</td><td>957</td><td>0</td></tr><tr><td>VariationOverview</td><td>EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.</td><td>605</td><td>177</td></tr><tr><td>VariationUsage</td><td>## How do I use this model on an image? 

To load a pretrained model:

```python
import timm
model = timm.create_model('tf_efficientnet_lite4', checkpoint_path='/kaggle/input/tf-efficientnet-lite/pytorch/tf-efficientnet-lite4/1/tf_efficientnet_lite4-741542c3.pth')
model.eval()
```

To load and preprocess the image:
```python
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
```

To get the model predictions:
```python
import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])
```

To get the top-5 predictions class names:
```python
# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
```

Replace the model name with the variant you want to use, e.g. `tf_efficientnet_lite0`. You can find the IDs in the model summaries at the top of this page.

To extract image features with this model, follow the [timm feature extraction examples](https://rwightman.github.io/pytorch-image-models/feature_extraction/), just change the name of the model you want to use.


## How do I finetune this model? 

You can finetune any of the pre-trained models just by changing the classifier (the last layer).
```python
model = timm.create_model('tf_efficientnet_lite4', num_classes=NUM_FINETUNE_CLASSES, checkpoint_path='/kaggle/input/tf-efficientnet-lite/pytorch/tf-efficientnet-lite4/1/tf_efficientnet_lite4-741542c3.pth')
```
To finetune on your own dataset, you have to write a training loop or adapt [timm's training
script](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) to use your dataset.


## How do I train this model? 


You can follow the [timm recipe scripts](https://rwightman.github.io/pytorch-image-models/scripts/) for training a new model afresh.</td><td>730</td><td>141</td></tr><tr><td>FineTunable</td><td>True</td><td>2</td><td>3</td></tr><tr><td>SourceUrl</td><td>https://github.com/rwightman/pytorch-image-models/tree/main/docs/models/tf-efficientnet-lite.md#tf-efficientnet-lite4</td><td>494</td><td>506</td></tr><tr><td>SourceOrganizationName</td><td>TIMM</td><td>2</td><td>515</td></tr></table><hr><h2>ğŸ“� ModelVariations.csv</h2><p><strong>Total Columns:</strong> 8</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1235</td><td>1000</td><td>0</td></tr><tr><td>ModelId</td><td>185</td><td>298</td><td>0</td></tr><tr><td>CurrentVariationSlug</td><td>hrnet-sunrgbd-hrnetv2-w48</td><td>824</td><td>0</td></tr><tr><td>ModelFramework</td><td>TensorFlow2</td><td>17</td><td>0</td></tr><tr><td>CurrentModelVariationVersionId</td><td>1462</td><td>1000</td><td>0</td></tr><tr><td>LicenseName</td><td>Apache 2.0</td><td>18</td><td>4</td></tr><tr><td>BaseModelVariationId</td><td>37999.0</td><td>31</td><td>957</td></tr><tr><td>CurrentDatasourceVersionId</td><td>4906183</td><td>1000</td><td>0</td></tr></table><hr><h2>ğŸ“� ModelVersions.csv</h2><p><strong>Total Columns:</strong> 9</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>14</td><td>1000</td><td>0</td></tr><tr><td>ModelId</td><td>14</td><td>1000</td><td>0</td></tr><tr><td>Title</td><td>bit</td><td>940</td><td>0</td></tr><tr><td>Subtitle</td><td>Collection of BiT models for feature extraction, and image classification on
Imagenet-1k (ILSVRC-2012-CLS) and Imagenet-21k.</td><td>499</td><td>472</td></tr><tr><td>ModelCard</td><td>## Overview

Big Transfer (BiT) is a recipe for pre-training image classification models on
large supervised datasets and efficiently fine-tuning them on any given target
task. The recipe achieves excellent performance on a wide variety of tasks, even
when using very few labeled examples from the target dataset.

*   A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly and N.
    Houlsby:
    [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370).

In the paper, three families of models are presented: "BiT-S", pre-trained on
ImageNet-1k (also known as ILSRCV-2012-CLS); "BiT-M", pre-trained on
ImageNet-21k (also known as the "Full ImageNet, Fall 2011 release"); and
"BiT-L", pre-trained on JFT-300M, a proprietary dataset. This collection
contains the BiT-S and BiT-M families.

Each family is composed of a ResNet-50 (R50x1), a ResNet-50 three times wider
(R50x3), a ResNet-101 (R101x1), a ResNet-101 three times wider (R101x3), and our
flagship architecture, a ResNet-152 four times wider (R152x4). Contrary to the
original ResNet architecture, we used Group Normalization instead of Batch
Normalization, and Weight Standardization of the convolution kernels.

### Models

We have released all the models trained on both of the public datasets as TF2
SavedModels. For each architecture, we provide different SavedModels intended to
use for a) feature extraction or fine-tuning on new tasks, b) image
classification on the popular ImageNet (ILSVRC-2012-CLS) dataset, and c)
multi-label image classification on the bigger ImageNet-21k dataset.

#### BiT-S (pre-trained on ImageNet-1k)

Feature extraction:

*   [R50x1](/models/google/bit/frameworks/TensorFlow2/variations/s-r50x1/versions/1)
*   [R50x3](/models/google/bit/frameworks/TensorFlow2/variations/s-r50x3/versions/1)
*   [R101x1](/models/google/bit/frameworks/TensorFlow2/variations/s-r101x1/versions/1)
*   [R101x3](/models/google/bit/frameworks/TensorFlow2/variations/s-r101x3/versions/1)
*   [R152x4](/models/google/bit/frameworks/TensorFlow2/variations/s-r152x4/versions/1)

ImageNet-1k classification:

*   [R50x1](/models/google/bit/frameworks/TensorFlow2/variations/s-r50x1-ilsvrc2012-classification/versions/1)
*   [R50x3](/models/google/bit/frameworks/TensorFlow2/variations/s-r50x3-ilsvrc2012-classification/versions/1)
*   [R101x1](/models/google/bit/frameworks/TensorFlow2/variations/s-r101x1-ilsvrc2012-classification/versions/1)
*   [R101x3](/models/google/bit/frameworks/TensorFlow2/variations/s-r101x3-ilsvrc2012-classification/versions/1)
*   [R152x4](/models/google/bit/frameworks/TensorFlow2/variations/s-r152x4-ilsvrc2012-classification/versions/1)

#### BiT-M (pre-trained on ImageNet-21k)

Feature extraction:

*   [R50x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1)
*   [R50x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x3/versions/1)
*   [R101x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x1/versions/1)
*   [R101x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x3/versions/1)
*   [R152x4](/models/google/bit/frameworks/TensorFlow2/variations/m-r152x4/versions/1)

ImageNet-1k classification:

*   [R50x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1-ilsvrc2012-classification/versions/1)
*   [R50x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x3-ilsvrc2012-classification/versions/1)
*   [R101x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x1-ilsvrc2012-classification/versions/1)
*   [R101x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x3-ilsvrc2012-classification/versions/1)
*   [R152x4](/models/google/bit/frameworks/TensorFlow2/variations/m-r152x4-ilsvrc2012-classification/versions/1)

ImageNet-21k classification:

*   [R50x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1-imagenet21k-classification/versions/1)
*   [R50x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r50x3-imagenet21k-classification/versions/1)
*   [R101x1](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x1-imagenet21k-classification/versions/1)
*   [R101x3](/models/google/bit/frameworks/TensorFlow2/variations/m-r101x3-imagenet21k-classification/versions/1)
*   [R152x4](/models/google/bit/frameworks/TensorFlow2/variations/m-r152x4-imagenet21k-classification/versions/1)

### Fine-tuning

All the modules can be fine-tuned. We recommend using the models intended for
feature extraction when fine-tuning on new datasets. In the BiT paper, we
propose a recipe, which we call BiT HyperRule, that transfers very well to a
diverse set of tasks with few labeled data examples. We recommend using these
hyperparameter settings as a starting point.</td><td>577</td><td>412</td></tr><tr><td>CreationDate</td><td>01/11/2023 19:46:42</td><td>992</td><td>0</td></tr><tr><td>OriginalPublishDate</td><td>10/06/2020</td><td>104</td><td>727</td></tr><tr><td>CreatorUserId</td><td>13166285</td><td>334</td><td>0</td></tr><tr><td>ProvenanceSources</td><td>https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/google/collections/bit/1.md</td><td>290</td><td>519</td></tr></table><hr><h2>ğŸ“� ModelVotes.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>470</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>8451572</td><td>636</td><td>0</td></tr><tr><td>ModelId</td><td>241</td><td>139</td><td>0</td></tr><tr><td>VoteDate</td><td>03/01/2023</td><td>42</td><td>0</td></tr></table><hr><h2>ğŸ“� Models.csv</h2><p><strong>Total Columns:</strong> 11</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>14</td><td>1000</td><td>0</td></tr><tr><td>OwnerUserId</td><td>767.0</td><td>299</td><td>463</td></tr><tr><td>OwnerOrganizationId</td><td>855.0</td><td>61</td><td>537</td></tr><tr><td>CurrentModelVersionId</td><td>14</td><td>1000</td><td>0</td></tr><tr><td>ForumId</td><td>2839497</td><td>1000</td><td>0</td></tr><tr><td>CreationDate</td><td>01/11/2023 19:46:42</td><td>992</td><td>0</td></tr><tr><td>TotalViews</td><td>8545</td><td>624</td><td>0</td></tr><tr><td>TotalDownloads</td><td>61158</td><td>454</td><td>0</td></tr><tr><td>TotalVotes</td><td>25</td><td>119</td><td>0</td></tr><tr><td>TotalKernels</td><td>3</td><td>60</td><td>0</td></tr><tr><td>CurrentSlug</td><td>bit</td><td>923</td><td>0</td></tr></table><hr><h2>ğŸ“� Organizations.csv</h2><p><strong>Total Columns:</strong> 5</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>2</td><td>1000</td><td>0</td></tr><tr><td>Name</td><td>Facebook</td><td>1000</td><td>0</td></tr><tr><td>Slug</td><td>facebook</td><td>1000</td><td>0</td></tr><tr><td>CreationDate</td><td>07/25/2016</td><td>715</td><td>0</td></tr><tr><td>Description</td><td>Facebook was built to help people connect and share, and over the last decade our tools have played a critical part in changing how people around the world communicate with one another. With over a billion people using the service and more than fifty offices around the globe, a career at Facebook offers countless ways to make an impact in a fast growing organization.

The Facebook Data Science team builds scalable platforms for the collection, management, and analysis of data.

We use these platforms to help drive informed decisions in areas critical to the success of the company, and conduct social science research of both internal and external interest.

We build tools and provide support for anyone at Facebook who would like to use our platforms to help make data-driven decisions or build data-intensive products and services.</td><td>398</td><td>601</td></tr></table><hr><h2>ğŸ“� Submissions.csv</h2><p><strong>Total Columns:</strong> 12</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>2180</td><td>1000</td><td>0</td></tr><tr><td>SubmittedUserId</td><td>647</td><td>149</td><td>0</td></tr><tr><td>TeamId</td><td>496</td><td>157</td><td>0</td></tr><tr><td>SourceKernelVersionId</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>SubmissionDate</td><td>04/29/2010</td><td>105</td><td>0</td></tr><tr><td>ScoreDate</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>IsAfterDeadline</td><td>False</td><td>2</td><td>0</td></tr><tr><td>IsSelected</td><td>False</td><td>2</td><td>0</td></tr><tr><td>PublicScoreLeaderboardDisplay</td><td>55.76919</td><td>855</td><td>0</td></tr><tr><td>PublicScoreFullPrecision</td><td>55.76919937133789</td><td>868</td><td>0</td></tr><tr><td>PrivateScoreLeaderboardDisplay</td><td>56.2139</td><td>917</td><td>0</td></tr><tr><td>PrivateScoreFullPrecision</td><td>56.21390151977539</td><td>927</td><td>0</td></tr></table><hr><h2>ğŸ“� Tags.csv</h2><p><strong>Total Columns:</strong> 9</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1115</td><td>830</td><td>0</td></tr><tr><td>ParentTagId</td><td>12116.0</td><td>62</td><td>10</td></tr><tr><td>Name</td><td>websites</td><td>830</td><td>0</td></tr><tr><td>Slug</td><td>web-sites</td><td>830</td><td>0</td></tr><tr><td>FullPath</td><td>subject &gt; science and technology &gt; internet &gt; websites</td><td>830</td><td>0</td></tr><tr><td>Description</td><td>Websites, like this one, are collections of content that can be found under a specific address. These datasets include information on individual websites or collections of websites.</td><td>149</td><td>681</td></tr><tr><td>DatasetCount</td><td>481</td><td>340</td><td>0</td></tr><tr><td>CompetitionCount</td><td>8</td><td>58</td><td>0</td></tr><tr><td>KernelCount</td><td>302</td><td>322</td><td>0</td></tr></table><hr><h2>ğŸ“� TeamMemberships.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>16458</td><td>1000</td><td>0</td></tr><tr><td>TeamId</td><td>518</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>635</td><td>884</td><td>0</td></tr><tr><td>RequestDate</td><td>02/06/2017</td><td>324</td><td>393</td></tr></table><hr><h2>ğŸ“� Teams.csv</h2><p><strong>Total Columns:</strong> 14</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>882</td><td>1000</td><td>0</td></tr><tr><td>CompetitionId</td><td>2447</td><td>10</td><td>0</td></tr><tr><td>TeamLeaderId</td><td>747</td><td>887</td><td>0</td></tr><tr><td>TeamName</td><td>the_kasparov_of_stats</td><td>928</td><td>1</td></tr><tr><td>ScoreFirstSubmittedDate</td><td>N/A</td><td>0</td><td>1000</td></tr><tr><td>LastSubmissionDate</td><td>08/18/2010</td><td>265</td><td>180</td></tr><tr><td>PublicLeaderboardSubmissionId</td><td>4721.0</td><td>820</td><td>180</td></tr><tr><td>PrivateLeaderboardSubmissionId</td><td>4721.0</td><td>820</td><td>180</td></tr><tr><td>IsBenchmark</td><td>False</td><td>2</td><td>0</td></tr><tr><td>Medal</td><td>3.0</td><td>3</td><td>794</td></tr><tr><td>MedalAwardDate</td><td>09/06/2018</td><td>6</td><td>211</td></tr><tr><td>PublicLeaderboardRank</td><td>223.0</td><td>349</td><td>186</td></tr><tr><td>PrivateLeaderboardRank</td><td>212.0</td><td>351</td><td>186</td></tr><tr><td>WriteUpForumTopicId</td><td>185.0</td><td>4</td><td>996</td></tr></table><hr><h2>ğŸ“� UserAchievements.csv</h2><p><strong>Total Columns:</strong> 11</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>3739827</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>381</td><td>1000</td><td>0</td></tr><tr><td>AchievementType</td><td>Scripts</td><td>3</td><td>0</td></tr><tr><td>Tier</td><td>1</td><td>4</td><td>0</td></tr><tr><td>TierAchievementDate</td><td>07/15/2016</td><td>32</td><td>0</td></tr><tr><td>Points</td><td>0</td><td>17</td><td>0</td></tr><tr><td>CurrentRanking</td><td>5677.0</td><td>13</td><td>987</td></tr><tr><td>HighestRanking</td><td>98.0</td><td>13</td><td>987</td></tr><tr><td>TotalGold</td><td>0</td><td>6</td><td>0</td></tr><tr><td>TotalSilver</td><td>0</td><td>8</td><td>0</td></tr><tr><td>TotalBronze</td><td>0</td><td>16</td><td>0</td></tr></table><hr><h2>ğŸ“� UserFollowers.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>996246</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>9590323</td><td>877</td><td>0</td></tr><tr><td>FollowingUserId</td><td>1185728</td><td>725</td><td>0</td></tr><tr><td>CreationDate</td><td>02/08/2022</td><td>334</td><td>0</td></tr></table><hr><h2>ğŸ“� UserOrganizations.csv</h2><p><strong>Total Columns:</strong> 4</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>6</td><td>1000</td><td>0</td></tr><tr><td>UserId</td><td>2505</td><td>820</td><td>0</td></tr><tr><td>OrganizationId</td><td>4</td><td>353</td><td>0</td></tr><tr><td>JoinDate</td><td>09/23/2016</td><td>375</td><td>0</td></tr></table><hr><h2>ğŸ“� Users.csv</h2><p><strong>Total Columns:</strong> 7</p><table><tr><th>Column Name</th><th>Sample Value</th><th>Unique Values</th><th>Null Values</th></tr><tr><td>Id</td><td>1</td><td>1000</td><td>0</td></tr><tr><td>UserName</td><td>kaggleteam</td><td>1000</td><td>0</td></tr><tr><td>DisplayName</td><td>Kaggle Team</td><td>962</td><td>0</td></tr><tr><td>RegisterDate</td><td>03/24/2011</td><td>112</td><td>0</td></tr><tr><td>PerformanceTier</td><td>5</td><td>6</td><td>0</td></tr><tr><td>Country</td><td>United States</td><td>66</td><td>25</td></tr><tr><td>LocationSharingOptOut</td><td>False</td><td>1</td><td>0</td></tr></table><hr>

</body>
</html>
'''

with open("column_summary.html", "w", encoding="utf-8") as f:
    f.write(strin)
from IPython.display import IFrame
display(IFrame("column_summary.html", width="100%", height="1000"))


relationships = [
    ("Competitions", "CompetitionTags", "Id", "CompetitionId"),
    ("Tags", "CompetitionTags", "Id", "TagId"),
    ("Datasets", "DatasetTags", "Id", "DatasetId"),
    ("Tags", "DatasetTags", "Id", "TagId"),
    ("Datasets", "DatasetTasks", "Id", "DatasetId"),
    ("Users", "DatasetTasks", "Id", "OwnerUserId"),
    ("DatasetTasks", "DatasetTaskSubmissions", "Id", "DatasetTaskId"),
    ("Users", "DatasetTaskSubmissions", "Id", "SubmittedUserId"),
    ("Datasets", "DatasetTaskSubmissions", "Id", "DatasetId"),
    ("Kernels", "KernelTags", "Id", "KernelId"),
    ("Tags", "KernelTags", "Id", "TagId"),
    ("Kernels", "KernelVersions", "Id", "ScriptId"),
    ("KernelVersions", "KernelVersionDatasetSources", "Id", "KernelVersionId"),
    ("KernelVersions", "KernelVersionCompetitionSources", "Id", "KernelVersionId"),
    ("KernelVersions", "KernelVersionModelSources", "Id", "KernelVersionId"),
    ("KernelVersions", "KernelVersionKernelSources", "Id", "KernelVersionId"),
    ("KernelLanguages", "KernelVersions", "Id", "ScriptLanguageId"),
    ("Users", "KernelVersions", "Id", "AuthorUserId"),
    ("Models", "ModelVersions", "Id", "ModelId"),
    ("ModelVariations", "ModelVariationVersions", "Id", "ModelVariationId"),
    ("Users", "ModelVersions", "Id", "CreatorUserId"),
    ("ModelVersions", "ModelVotes", "Id", "ModelVersionId"),
    ("Users", "ModelVotes", "Id", "UserId"),
    ("DatasetVersions", "DatasetVotes", "Id", "DatasetVersionId"),
    ("Users", "DatasetVotes", "Id", "UserId"),
    ("Forums", "ForumTopics", "Id", "ForumId"),
    ("ForumTopics", "ForumMessages", "Id", "ForumTopicId"),
    ("Users", "ForumMessages", "Id", "PostUserId"),
    ("ForumMessages", "ForumMessageVotes", "Id", "ForumMessageId"),
    ("Users", "ForumMessageVotes", "Id", "FromUserId"),
    ("ForumMessages", "ForumMessageReactions", "Id", "ForumMessageId"),
    ("Users", "ForumMessageReactions", "Id", "FromUserId"),
    ("Competitions", "Episodes", "Id", "CompetitionId"),
    ("DatasetVersions", "Datasets", "DatasetId", "Id"),
    ("Datasets", "DatasetVersions", "Id", "DatasetId"),
    ("Users", "DatasetVersions", "Id", "CreatorUserId"),
    ("Users", "Datasets", "Id", "OwnerUserId"),
    ("Users", "Datasets", "Id", "CreatorUserId"),
]


dot = graphviz.Digraph(comment="Meta Kaggle ER Diagram")
dot.attr(rankdir="LR", size="12,10")
entities = set()
for src, dst, src_key, dst_key in relationships:
    entities.update([src, dst])
    dot.node(src, src, shape="box")
    dot.node(dst, dst, shape="box")
    dot.edge(src, dst, label=f"{src_key} â†’ {dst_key}")


dot.render('meta_kaggle_er_diagram', format='png', cleanup=False)
Image("meta_kaggle_er_diagram.png")


users = pl.read_csv(f"{MK_PATH}/Users.csv")
print(f"Shape: {users.shape}")
users.head()


users.select([
    (
        pl.col(col).is_null() | 
        (pl.col(col) == "") if users.schema[col] == pl.Utf8 else pl.col(col).is_null()
    ).sum().alias(col)
    for col in users.columns
])


unique_countries = (
    users
    .filter(pl.col("Country").is_not_null() & (pl.col("Country").str.strip_chars() != ""))
    .select("Country")
    .unique()
    .sort("Country")
)
country_list = unique_countries["Country"].to_list()
print(f"ğŸŒ� Total Unique Countries: {len(country_list)}\n")
for country in country_list:
    print(country)


country_counts = (
    users
    .filter(pl.col("Country").is_not_null() & (pl.col("Country").str.strip_chars() != ""))
    .group_by("Country")
    .agg(pl.len().alias("UserCount"))
    .sort("UserCount", descending=True)
)
print(country_counts)


df_country_counts = country_counts.to_pandas()
print("Countries in your dataset:")
print(df_country_counts["Country"].tolist())
print(f"\nTotal unique ğŸŒ� countries in dataset: {len(df_country_counts)}")


geo_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
geo_json_data = requests.get(geo_url).json()
geojson_countries = [feature["properties"]["name"] for feature in geo_json_data["features"]]
print(f"\nCountries in GeoJSON: {len(geojson_countries)}")


country_mapping = {
    "United States": "United States of America",
    "USA": "United States of America", 
    "usa": "United States of America",
    "US": "United States of America",
    "United Kingdom": "United Kingdom",
    "UK": "United Kingdom",
    "Russia": "Russia",
    "South Korea": "South Korea",
    "North Korea": "North Korea",
    "Vietnam": "Vietnam",
    "Iran": "Iran",
    "Syria": "Syria",
    "Venezuela": "Venezuela",
    "Bolivia": "Bolivia",
    "Tanzania": "Tanzania",
    "Democratic Republic of the Congo": "Democratic Republic of the Congo",
    "Republic of the Congo": "Republic of the Congo",
    "Czechia": "Czech Republic",
    "Myanmar (Burma)": "Myanmar",
    "The Bahamas": "Bahamas",
    "The Gambia": "Gambia",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "CÃ´te d'Ivoire": "Ivory Coast",
    "Eswatini": "Swaziland",
    "North Macedonia": "Macedonia",
    "TÃ¼rkiye": "Turkey",
    "Turkey": "Turkey",
    "Hong Kong": "Hong Kong S.A.R.",
    "Macao": "Macao S.A.R",
    "Taiwan": "Taiwan",
    "Puerto Rico": "Puerto Rico",
    "Palestine": "Palestine",
    "Palestinian Territories": "Palestine",
    "Greenland": "Greenland",
    "French Guiana": "French Guiana",
    "Faroe Islands": "Faeroe Islands",
    
    # Sub-national regions mapped to their countries
    # United States
    "California": "United States of America",
    "Texas": "United States of America",
    "Florida": "United States of America",
    "Washington": "United States of America",
    "District of Columbia": "United States of America",
    
    # Canada
    "Quebec": "Canada",
    
    # Spain
    "Aragon": "Spain",
    "Region of Murcia": "Spain",
    "Canary Islands": "Spain",
    
    # Italy
    "Calabria": "Italy",
    "Lombardy": "Italy",
    "Piedmont": "Italy",
    "Lazio": "Italy",
    
    # China
    "Liaoning": "China",
    
    # India
    "Rajasthan": "India",
    
    # Indonesia
    "East Java": "Indonesia",
    "South Sulawesi": "Indonesia",
    "South East Sulawesi": "Indonesia",
    
    # Brazil
    "State of SÃ£o Paulo": "Brazil",
    "State of Minas Gerais": "Brazil",
    "State of Rio Grande do Sul": "Brazil",
    "State of Santa Catarina": "Brazil",
    "State of Mato Grosso": "Brazil",
    
    # Colombia
    "Caldas": "Colombia",
    
    # Costa Rica
    "San JosÃ© Province": "Costa Rica",
    
    # Portugal
    "Leiria District": "Portugal",
    "Portalegre District": "Portugal",
    "Porto District": "Portugal",
    
    # Croatia
    "Zadar County": "Croatia",
    
    # France (overseas territory)
    "Saint-Paul": "France",
    
    # These are not geographic locations, so we'll filter them out
    "Nationalities": None,
    "Nationalities and Peoples": None,
}



# Function to normalize country names for better matching
def normalize_name(name):
    """Normalize country names for better matching"""
    if pd.isna(name):
        return name
    
    # Convert to string and strip
    name = str(name).strip()
    
    # Remove common prefixes/suffixes
    name = re.sub(r'^(The\s+)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(Republic|State|Province|District|County|Islands?)$', '', name, flags=re.IGNORECASE)
    
    return name


df_country_counts['MappedCountry'] = df_country_counts['Country'].map(
    lambda x: country_mapping.get(x, x)
)
df_country_counts = df_country_counts[df_country_counts['MappedCountry'].notna()]
df_country_counts


unmapped_countries = []
mapped_count = 0

for idx, row in df_country_counts.iterrows():
    mapped_name = row['MappedCountry']
    if mapped_name not in geojson_countries:
        matches = get_close_matches(mapped_name, geojson_countries, n=1, cutoff=0.8)
        if matches:
            df_country_counts.at[idx, 'MappedCountry'] = matches[0]
            mapped_count += 1
            print(f"Fuzzy matched: '{mapped_name}' -> '{matches[0]}'")
        else:
            unmapped_countries.append((row['Country'], mapped_name, row['UserCount']))
print(f"\nFuzzy matching results: {mapped_count} additional mappings")
print(f"Unmapped countries: {len(unmapped_countries)}")


if unmapped_countries:
    print("\nUnmapped countries (Original -> Mapped -> User Count):")
    for orig, mapped, count in sorted(unmapped_countries, key=lambda x: x[2], reverse=True):
        print(f"  {orig} -> {mapped} ({count} users)")


country_user_counts = df_country_counts.groupby('MappedCountry')['UserCount'].sum().reset_index()
country_lookup = dict(zip(country_user_counts['MappedCountry'], country_user_counts['UserCount']))


for feature in geo_json_data["features"]:
    country_name = feature["properties"]["name"]
    user_count = country_lookup.get(country_name, 0)
    feature["properties"]["UserCount"] = user_count


m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
def get_color(user_count):
    if user_count == 0:
        return "#f7f7f7"  # Very light gray for no users
    elif user_count < 10000:
        return "#c6dbef"  # Light blue
    elif user_count < 35000:
        return "#6baed6"  # Medium blue
    elif user_count < 100000:
        return "#2171b5"  # Dark blue
    else:
        return "#08306b"  # Very dark blue


folium.GeoJson(
    geo_json_data,
    name="Users by Country",
    style_function=lambda x: {
        "fillColor": get_color(x["properties"]["UserCount"]),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.8,
        "dashArray": "3" if x["properties"]["UserCount"] == 0 else "0"
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "UserCount"],
        aliases=["Country", "Users"],
        localize=True,
        sticky=False,
        labels=True,
        style="background-color: white; color: #333; font-family: Arial; font-size: 12px; padding: 10px; border-radius: 5px;"
    ),
    popup=folium.GeoJsonPopup(
        fields=["name", "UserCount"],
        aliases=["Country:", "Total Users:"],
        localize=True,
        labels=True,
        style="font-size: 14px; font-weight: bold;"
    )
).add_to(m)


legend_html = '''
<div style="position: fixed; 
     top: 20px; right: 20px; width: 160px; height: 110px; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
<p style="margin: 0 0 8px 0; font-weight: bold;">Users by Country</p>
<p style="margin: 2px 0;"><i class="fa fa-square" style="color:#08306b"></i> 100,000+</p>
<p style="margin: 2px 0;"><i class="fa fa-square" style="color:#2171b5"></i> 35,000-99,999</p>
<p style="margin: 2px 0;"><i class="fa fa-square" style="color:#6baed6"></i> 10,000-34,999</p>
<p style="margin: 2px 0;"><i class="fa fa-square" style="color:#c6dbef"></i> 1-9,999</p>
<p style="margin: 2px 0;"><i class="fa fa-square" style="color:#f7f7f7; border: 1px solid #ccc;"></i> 0</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))


folium.LayerControl().add_to(m)


total_users = country_user_counts['UserCount'].sum()
countries_with_users = len(country_user_counts[country_user_counts['UserCount'] > 0])
print(f"Total users mapped: {total_users:,}")
print(f"Countries with users: {countries_with_users}")
print(f"Top 10 countries by user count:")
top_countries = country_user_counts.sort_values('UserCount', ascending=False).head(10)
for _, row in top_countries.iterrows():
    print(f"  {row['MappedCountry']}: {row['UserCount']:,} users")


m.save('users_by_country_map.html')
print(f"\nMap saved as 'users_by_country_map.html'")
m


users = users.with_columns([
    pl.col("RegisterDate").str.strptime(pl.Date, "%m/%d/%Y").alias("RegisterDateParsed"),
    pl.col("RegisterDate").str.strptime(pl.Date, "%m/%d/%Y").dt.year().alias("RegisterYear")
])
# Group by year and count users
user_growth_per_year = (
    users
    .group_by("RegisterYear")
    .agg(pl.len().alias("UserCount"))
    .sort("RegisterYear")
)


users


df_plot = user_growth_per_year.to_pandas()
df_plot.head()


fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_plot["RegisterYear"],
    y=df_plot["UserCount"],
    mode="lines+markers",
    line=dict(color="royalblue", width=2),
    marker=dict(size=6),
    name="Users"
))
fig1.update_layout(
    title="ğŸ“ˆ User Growth Over the Years",
    xaxis_title="Year",
    yaxis_title="Number of Users",
    template="plotly_white",
    width=1000,
    height=600
)
plot(fig1, filename="eduexp1.html", auto_open=False)
IFrame("eduexp1.html", width=1200, height=600)


users = users.with_columns(
    pl.col("RegisterDate").str.strptime(pl.Date, "%m/%d/%Y", strict=False)
)
earliest_date = users.select(pl.col("RegisterDate").min()).item()
latest_date = users.select(pl.col("RegisterDate").max()).item()
print(f"ğŸ•°ï¸� Earliest Kaggle registration date: {earliest_date}")
print(f"ğŸ•“ Latest Kaggle registration date: {latest_date}")


df_plot = df_plot.sort_values('RegisterYear')
years = df_plot['RegisterYear'].values
user_counts = df_plot['UserCount'].values


print(years)
print(user_counts)


forecast_years = [2025, 2026]


# Method 1: Linear Regression Forecasting
def linear_forecast(x, y, future_years):
    """Simple linear regression forecast"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    forecast = []
    for year in future_years:
        predicted_users = slope * year + intercept
        forecast.append(max(0, int(predicted_users))) 
    return forecast, r_value**2
linear_pred, linear_r2 = linear_forecast(years, user_counts, forecast_years)


# Method 2: Exponential Growth Forecasting
def exponential_forecast(x, y, future_years):
    """Exponential growth forecast using log transformation"""
    try:
        log_y = np.log(y + 1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
        
        forecast = []
        for year in future_years:
            log_predicted = slope * year + intercept
            predicted_users = int(np.exp(log_predicted) - 1)
            forecast.append(max(0, predicted_users))
        
        return forecast, r_value**2
    except:
        # Fallback to linear if exponential fails
        return linear_forecast(x, y, future_years)
exp_pred, exp_r2 = exponential_forecast(years, user_counts, forecast_years)


# Method 3: Polynomial Forecasting (degree 2)
def polynomial_forecast(x, y, future_years, degree=2):
    """Polynomial regression forecast"""
    coeffs = np.polyfit(x, y, degree)      
    poly_func = np.poly1d(coeffs)           

    forecast = []
    for year in future_years:
        predicted_users = int(poly_func(year))
        forecast.append(max(0, predicted_users))    
    y_pred = poly_func(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return forecast, r_squared
poly_pred, poly_r2 = polynomial_forecast(years, user_counts, forecast_years)


models = {
    'Linear': (linear_pred, linear_r2),
    'Exponential': (exp_pred, exp_r2),
    'Polynomial': (poly_pred, poly_r2)
}


best_model = max(models.items(), key=lambda x: x[1][1])
best_predictions = best_model[1][0]
best_r2 = best_model[1][1]


print(f"Model Performance:")
print(f"Linear RÂ²: {linear_r2:.4f}")
print(f"Exponential RÂ²: {exp_r2:.4f}")
print(f"Polynomial RÂ²: {poly_r2:.4f}")


def create_model_chart(model_name, predictions, r_squared, color, filename):
    """Create individual chart for each model"""
    fig_model = go.Figure()
    fig_model.add_trace(go.Scatter(
        x=df_plot["RegisterYear"],
        y=df_plot["UserCount"],
        mode="lines+markers",
        line=dict(color="royalblue", width=3),
        marker=dict(size=8, color="royalblue"),
        name="Historical Data",
        hovertemplate="<b>Year:</b> %{x}<br><b>Users:</b> %{y:,}<extra></extra>"
    ))
    fig_model.add_trace(go.Scatter(
        x=forecast_years,
        y=predictions,
        mode="lines+markers",
        line=dict(color=color, width=3, dash="dash"),
        marker=dict(size=10, color=color, symbol="diamond"),
        name=f"{model_name} Forecast",
        hovertemplate="<b>Year:</b> %{x}<br><b>Predicted Users:</b> %{y:,}<extra></extra>"
    ))
    fig_model.add_trace(go.Scatter(
        x=[years[-1], forecast_years[0]],
        y=[user_counts[-1], predictions[0]],
        mode="lines",
        line=dict(color="orange", width=2, dash="dot"),
        name="Transition",
        showlegend=False,
        hoverinfo='skip'
    ))
    for i, (year, pred) in enumerate(zip(forecast_years, predictions)):
        fig_model.add_annotation(
            x=year,
            y=pred,
            text=f"{pred:,}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            font=dict(size=12, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1
        )
    fig_model.update_layout(
        title={
            'text': f"ğŸ“ˆ {model_name} Model Forecast<br><sub>RÂ² = {r_squared:.4f} | 2025: {predictions[0]:,} | 2026: {predictions[1]:,}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Year",
        yaxis_title="Number of Users",
        template="plotly_white",
        width=1000,
        height=600,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        hovermode='x unified'
    )
    fig_model.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_model.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_model.update_yaxes(tickformat=',')
    plot(fig_model, filename=filename, auto_open=False)
    return fig_model


linear_fig = create_model_chart("Linear", linear_pred, linear_r2, "green", "linear_forecast.html")
exp_fig = create_model_chart("Exponential", exp_pred, exp_r2, "purple", "exponential_forecast.html")
poly_fig = create_model_chart("Polynomial", poly_pred, poly_r2, "orange", "polynomial_forecast.html")


print(f"\n All Model Predictions:")
print(f"{'Model':<12} {'2025':<12} {'2026':<12} {'RÂ²':<8}")
print("-" * 45)
print(f"{'Linear':<12} {linear_pred[0]:<12,} {linear_pred[1]:<12,} {linear_r2:<8.4f}")
print(f"{'Exponential':<12} {exp_pred[0]:<12,} {exp_pred[1]:<12,} {exp_r2:<8.4f}")
print(f"{'Polynomial':<12} {poly_pred[0]:<12,} {poly_pred[1]:<12,} {poly_r2:<8.4f}")


print("1. linear_forecast.html - Linear model only")
display(IFrame("linear_forecast.html", width=1000, height=600))


print("2. exponential_forecast.html - Exponential model only") 
display(IFrame("exponential_forecast.html", width=1000, height=600))


print("3. polynomial_forecast.html - Polynomial model only")
display(IFrame("polynomial_forecast.html", width=1000, height=600))


fig_comparison = go.Figure()
fig_comparison.add_trace(go.Scatter(
    x=df_plot["RegisterYear"],
    y=df_plot["UserCount"],
    mode="lines+markers",
    line=dict(color="royalblue", width=4),
    marker=dict(size=10, color="royalblue"),
    name="Historical Data",
    hovertemplate="<b>Year:</b> %{x}<br><b>Users:</b> %{y:,}<extra></extra>"
))
models_data = [
    ("Linear", linear_pred, "green", "circle"),
    ("Exponential", exp_pred, "purple", "square"),
    ("Polynomial", poly_pred, "orange", "diamond")
]

for model_name, predictions, color, symbol in models_data:
    fig_comparison.add_trace(go.Scatter(
        x=forecast_years,
        y=predictions,
        mode="lines+markers",
        line=dict(color=color, width=3, dash="dash"),
        marker=dict(size=12, color=color, symbol=symbol),
        name=f"{model_name} Forecast",
        hovertemplate=f"<b>{model_name}</b><br><b>Year:</b> %{{x}}<br><b>Predicted Users:</b> %{{y:,}}<extra></extra>"
    ))
fig_comparison.update_layout(
    title={
        'text': "ğŸ“Š All Models Comparison<br><sub>Linear vs Exponential vs Polynomial Forecasts</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title="Year",
    yaxis_title="Number of Users",
    template="plotly_white",
    width=1200,
    height=700,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    ),
    hovermode='x unified'
)

fig_comparison.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig_comparison.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig_comparison.update_yaxes(tickformat=',')
plot(fig_comparison, filename="all_models_comparison.html", auto_open=False)


print("4. all_models_comparison.html - All models on one chart")
display(IFrame("all_models_comparison.html", width=1200, height=700))


import gradio as gr
demo = gr.load("spaces/Asura05/Meta-Kaggle-Dataset-Navigator")
demo.launch()

