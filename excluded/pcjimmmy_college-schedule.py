


import csv
import requests
from datetime import datetime, timedelta
import pytz
import json
import os

def get_college_basketball_games(days=60, save_raw_data=True, raw_data_dir="raw_api_data", group_id=100):
    """Fetches both men's and women's college basketball games for the next specified number of days,
       and optionally saves the raw API responses."""

    today = datetime.now(pytz.utc).date()
    end_date = today + timedelta(days=days)
    games = []

    if save_raw_data and not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    for gender in ['mens', 'womens']:
        try:
            url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/{gender}-college-basketball/scoreboard?dates={{}}&groups={group_id}"

            current_date = today
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                full_url = url.format(date_str)
                response = requests.get(full_url)
                response.raise_for_status()
                data = response.json()

                if save_raw_data:
                    filename = os.path.join(raw_data_dir, f"{gender}_{date_str}.json")
                    with open(filename, "w") as f:
                        json.dump(data, f, indent=4)

                if 'events' in data:
                    for event in data['events']:
                        if 'competitions' in event:
                            for competition in event['competitions']:
                                if 'competitors' in competition and len(competition['competitors']) == 2:
                                    team1 = competition['competitors'][0]
                                    team2 = competition['competitors'][1]

                                    team1_name = team1['team']['name']
                                    team1_school = team1['team']['displayName']
                                    team2_name = team2['team']['name']
                                    team2_school = team2['team']['displayName']

                                    team1_home_away = 'N'
                                    team2_home_away = 'N'

                                    if 'homeAway' in team1:
                                        if team1['homeAway'] == 'home':
                                            team1_home_away = 'H'
                                            team2_home_away = 'A'
                                        elif team1['homeAway'] == 'away':
                                            team1_home_away = 'A'
                                            team2_home_away = 'H'

                                    games.append({
                                        'Gender': gender,
                                        'Team1': team1_name,
                                        'Team1_School': team1_school,
                                        'Team1_H/A/N': team1_home_away,
                                        'Team2': team2_name,
                                        'Team2_School': team2_school,
                                        'Team2_H/A/N': team2_home_away,
                                        'Date': current_date.strftime("%Y-%m-%d")
                                    })
                current_date += timedelta(days=1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {gender} data: {e}")
        except KeyError as e:
            print(f"Error parsing {gender} JSON: {e}")

    return games

def create_csv(games, filename="college_basketball_schedule.csv"):
    """Creates a CSV file from the game data, including gender and school names."""

    if not games:
        print("No games found.")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Gender', 'Team1', 'Team1_School', 'Team1_H/A/N', 'Team2', 'Team2_School', 'Team2_H/A/N', 'Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for game in games:
            writer.writerow(game)

    print(f"CSV file '{filename}' created successfully.")

if __name__ == "__main__":
    games = get_college_basketball_games()
    create_csv(games)

