import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)

class Leaderboard:
    def __init__(self, num_teams, custom_team_sizes=None):
        self.num_teams = num_teams
        self.teams = self._generate_teams(custom_team_sizes)
        self.prize_participants_count = 0

    def _generate_teams(self, custom_team_sizes):
        teams = []
        for i in range(1, self.num_teams + 1):
            if custom_team_sizes and i <= len(custom_team_sizes):
                num_members = len(custom_team_sizes[i - 1])
                teams.append({"place": i, "members": [f"{j}" for j in custom_team_sizes[i-1]]})
            else:
                num_members = 1
                teams.append({"place": i, "members": [f"Member_{i-139}" for j in range(1, num_members + 1)]})
        return teams

    def calculate_prize_participants(self):
        num_teams = len(self.teams)
        if num_teams <= 99:
            bronze_prizes = int(num_teams * 0.40)
            silver_prizes = int(num_teams * 0.20)
            gold_prizes = int(num_teams * 0.10)
        elif 100 <= num_teams <= 249:
            bronze_prizes = int(num_teams * 0.40)
            silver_prizes = int(num_teams * 0.20)
            gold_prizes = 10
        elif 250 <= num_teams <= 999:
            bronze_prizes = 100
            silver_prizes = 50
            gold_prizes = 10 + int(num_teams * 0.002)
        else:
            bronze_prizes = int(num_teams * 0.10)
            silver_prizes = int(num_teams * 0.05)
            gold_prizes = 10 + int(num_teams * 0.002)

        silver_prizes = silver_prizes - gold_prizes
        bronze_prizes = bronze_prizes - gold_prizes - silver_prizes

        total_prizes = gold_prizes + silver_prizes + bronze_prizes
        self.prize_participants_count = sum(
            len(team["members"]) for team in self.teams if team["place"] <= total_prizes
        )

    def merge_teams(self, team1_id, team2_id):
        team1 = self.teams[team1_id]
        team2 = self.teams[team2_id]

        if len(team1["members"]) + len(team2["members"]) > 5:
            return False

        new_team = {
            "place": min(team1["place"], team2["place"]),
            "members": team1["members"] + team2["members"]
        }

        self.teams.pop(max(team1_id, team2_id))
        self.teams.pop(min(team1_id, team2_id))
        self.teams.append(new_team)
        self.update_places()
        return True

    def update_places(self):
        self.teams.sort(key=lambda team: team["place"])
        for index, team in enumerate(self.teams):
            team["place"] = index + 1

    def find_and_merge_teams(self):
        for i in range(4, len(self.teams)):
            team = self.teams[i]
            if len(team["members"]) < 5:
                for j in range(i + 1, len(self.teams)):
                    if len(team["members"]) + len(self.teams[j]["members"]) <= 5:
                        if self.merge_teams(i, j):
                            return True
        return False

    def get_prize_teams(self):
        prize_teams = {
            "Gold": [],
            "Silver": [],
            "Bronze": []
        }

        num_teams = len(self.teams)
        if num_teams <= 99:
            bronze_prizes = int(num_teams * 0.40)
            silver_prizes = int(num_teams * 0.20)
            gold_prizes = int(num_teams * 0.10)
        elif 100 <= num_teams <= 249:
            bronze_prizes = int(num_teams * 0.40)
            silver_prizes = int(num_teams * 0.20)
            gold_prizes = 10
        elif 250 <= num_teams <= 999:
            bronze_prizes = 100
            silver_prizes = 50
            gold_prizes = 10 + int(num_teams * 0.002)
        else:
            bronze_prizes = int(num_teams * 0.10)
            silver_prizes = int(num_teams * 0.05)
            gold_prizes = 10 + int(num_teams * 0.002)

        silver_prizes = silver_prizes - gold_prizes
        bronze_prizes = bronze_prizes - gold_prizes - silver_prizes

        for team in self.teams[:gold_prizes]:
            prize_teams["Gold"].append(team)
        for team in self.teams[gold_prizes: gold_prizes + silver_prizes]:
            prize_teams["Silver"].append(team)
        for team in self.teams[gold_prizes + silver_prizes: gold_prizes + silver_prizes + bronze_prizes]:
            prize_teams["Bronze"].append(team)

        return prize_teams

    def generate_prize_table(self):
        prize_teams = self.get_prize_teams()
        data = []

        for prize, teams in prize_teams.items():
            for team in teams:
                data.append({
                    "Place": team["place"],
                    "Prize": prize,
                    "Team": ", ".join(team["members"])
                })

        return pd.DataFrame(data)

num_teams = 1395
custom_team_sizes = [['kibuna','nagiss','c-number'],['KaizaburoChubachi','SolverWorld','Daniel Phalen'],['Sia','yuanzhe zhou','Ali','Steven_Y'],
                     ['ONODERA','daiwakun'],['CPMP','Horea'],['Shun Fukuda','Yuki.O'],['Takoi','Kotaro','TomFuj','2g'],
                     ['WOOSUNG YOON','Anna','Michael Wu','Mart Preusse','Hikari_30'],['bono'],['Max2020'],['KS'],['chris','Chris Deotte'],
                     ['Yurii Dzeryn','Dewei Chen','Milan Peelman','gzLang','Snorf'], 
                     ['Giba','Bill','Lucien de Rubempre','Egor Trushin','Anil Ozturk'],['Kirill Tushin','Irene'],['Yipeng Liu','juvenjiang'],
                     ['Araik Tamazian','Maham Haroon'],['Tomek'],['adakoda'],['Fu Ryo'],['Liping'],['Felix M Neumann','Tanaka Ai24'],
                     ['Ilya F','Kirill Isupov','arkmartov'],['bsmelbs'],['starpentagon','Ueddy','mzgc','komori3','ocha_heaven'],
                     ['Gasset Mathieu'],['Gilles Vandewiele'],['tetsuya','Kazuo Watanabe'],['Rares Barbantan','Gabriel'],
                     ['hoxosh','yukari17','kumanomi','Prgckwb'],['Hoa Nguyen','Rostyslav Makarenko'],
                     ['Andrew','max','Leon','jakkkc','Dave Greer'],['mrmldjr'],['lhwcv','MartinXH','zby','Roschild.Rui'],['Nick Sarris'],
                     ['alcanta'],['Hew'],['aruaru0'],['akoynk'],['flg'],['shhh39','utm529f'],['Benjamin Kovacs','Roland T Kovacs'],['qnqn1927'],
                     ['akmr'],['sekken','JoshuaHong','Maksim Sinitsyn','klogw'],['PC Jimmmy'],['N.K.'],['Tomasz Kielbasa'],
                     ['colun','siman','Shun_PI'],['denden12','litlit','ISAKA Tsuyoshi','Gassano'],['Agrica'],['Oleg Kokorin'],['kma###'],
                     ['You Lyu'],['yitiaoxiaoyuer'],['liangfeng lin'],['Wun0'],['James Holland'],['Anton Chikin'],['tanadaaa'],['Random Draw'],
                     ['Yuliya'],['rabot','toast-uz','Zach Leee'],['Oleh Uhnivenko'],['Vincent Debout','Skril'],['Victor Ogobi'],['Marc De Vries'],
                     ['Semih Eren'],['gaguudgh','静香さんです','pig-xia','Elizabeth','Task online'],['SpiralTip'],['tyi2000'],
                     ['yinhe wang','I2nfinit3y','Lecheng Yan','Xicheng Han'],['Jonathan Chan','SeshuRaju 🧘‍♂️','Gunes Evitan','lty','samson'],
                     ['kif'],['shanzhong8'],['Glen Koundry'],['east','yuki'],['Md Boktiar Mahbub Murad'],['edteoh'],['YaGana Sheriff-Hussaini'],
                     ['Rjg'],['hasibirok0'],['Alberto MV'],['Mathurin Aché'],['niwakaggler'],['Vicens Gaitan'],['chimaki'],['prvi'],['webmaking'],
                     ['Raki'],['oigckko','guuihuji','lsq','koliyyu'],['NocturneBflat'],['Ezra'],['japanese_tanuki'],['NecroSean38'],
                     ['Mert Bayraktar','ulasdesouza'],['Michael Semenoff'],['Nat Bel ML Fun'],['FrederikZ'],['Tetsuro Tsuda'],['louis-philippe'],
                     ['Pizzaboi'],['Arbidos'],['yashimar'],['JP'],['rf'],['Aatif Fraz'],['lightsource<3'],['Toan Doan'],['qxssxq'],
                     ['Veniamin Nelin'],['Kobelev Maxim'],['Ruslan Vdovychenko'],['Hamza'],['cqr'],['Jagat Kiran'],['Allie K.'],
                     ['Mingxuan Du','Ginto','Georgexzy','Paul2025','Mormaid'],['TeraFlops'],['Mansour'],['JiaPeng'],['Steven Xie','Isaac Qiu'],
                     ['Vladislav Kulikov'],['任文潮'],['kglctf'],['yangjian zhu'],['nora600'],['TAKASHI AZUMA'],['Jiaming Liu'],
                     ['강신성 (HollyRiver)'],['Xinlong Zhang'],['kosirowada','bishopfunc','Amon Kizawa'],['RandR'],['kumk'],['CroDoc'],
                     ['Takahiro Saito'],['heinideyibadiaole'],['Danting Zeng'],['moat']]
lb = Leaderboard(num_teams, custom_team_sizes)

team_counts = []
prize_counts = []
max_prize_count = 0
max_prize_count_teams = 0

start_prize_table = lb.generate_prize_table()

while True:
    lb.calculate_prize_participants()
    team_counts.append(len(lb.teams))
    prize_counts.append(lb.prize_participants_count)

    if lb.prize_participants_count > max_prize_count:
        max_prize_count = lb.prize_participants_count
        max_prize_count_teams = len(lb.teams)

        max_prize_table = lb.generate_prize_table()
    
    if not lb.find_and_merge_teams():
        break


plt.figure(figsize=(10, 6))

plt.plot(range(len(team_counts)), team_counts, label='Number of Teams', color='blue')

plt.plot(range(len(prize_counts)), prize_counts, label='Number of People in Prizes', color='green')

plt.scatter([prize_counts.index(max_prize_count)], [max_prize_count], color='red', zorder=5, label=f'Maximum People in Prizes: {max_prize_count} at Step {max_prize_count_teams} Teams')

plt.legend()

plt.title("Changes in Number of Teams and People in Prizes During Team Merging")
plt.xlabel("Team Merging Steps")
plt.ylabel("Count")

plt.grid(True)
plt.show()

print(f"Maximum People in Prizes: {max_prize_count} at Step {max_prize_count_teams} Teams.")


print(start_prize_table.to_string(index=False))


print(max_prize_table.to_string(index=False))


initial_df = pd.DataFrame(start_prize_table)
final_df = pd.DataFrame(max_prize_table)

initial_participants = []
for index, row in initial_df.iterrows():
    team = row['Team']
    prize = row['Prize']
    place = row['Place']
    for participant in team.split(', '):
        initial_participants.append({
            'Place': place,
            'Prize': prize,
            'Participant': participant
        })

final_participants = []
for index, row in final_df.iterrows():
    team = row['Team']
    prize = row['Prize']
    place = row['Place']
    for participant in team.split(', '):
        final_participants.append({
            'Place': place,
            'Prize': prize,
            'Participant': participant
        })

initial_participants_df = pd.DataFrame(initial_participants)
final_participants_df = pd.DataFrame(final_participants)

medal_changes = {
    'Gold -> Silver': 0,
    'Silver -> Gold': 0,
    'Bronze -> Silver': 0,
    'No Medal -> Gold': 0,
    'No Medal -> Silver': 0,
    'No Medal -> Bronze': 0,
}

improved_positions = 0
unchanged = 0
new_participants = 0

for index, initial_row in initial_participants_df.iterrows():
    participant = initial_row['Participant']
    
    final_row = final_participants_df[final_participants_df['Participant'] == participant]
    
    if not final_row.empty:
        final_row = final_row.iloc[0]
        initial_prize = initial_row['Prize']
        initial_place = initial_row['Place']
        final_prize = final_row['Prize']
        final_place = final_row['Place']
        
        if initial_prize == 'Gold' and final_prize == 'Silver':
            medal_changes['Gold -> Silver'] += 1
        elif initial_prize == 'Silver' and final_prize == 'Gold':
            medal_changes['Silver -> Gold'] += 1
        elif initial_prize == 'Bronze' and final_prize == 'Silver':
            medal_changes['Bronze -> Silver'] += 1
        elif initial_prize == 'No Medal' and final_prize == 'Gold':
            medal_changes['No Medal -> Gold'] += 1
        elif initial_prize == 'No Medal' and final_prize == 'Silver':
            medal_changes['No Medal -> Silver'] += 1
        elif initial_prize == 'No Medal' and final_prize == 'Bronze':
            medal_changes['No Medal -> Bronze'] += 1
        
        if final_place < initial_place:
            improved_positions += 1
        elif final_place == initial_place:
            unchanged += 1

for index, final_row in final_participants_df.iterrows():
    participant = final_row['Participant']
    
    initial_row = initial_participants_df[initial_participants_df['Participant'] == participant]
    
    if initial_row.empty:
        new_participants += 1
        final_prize = final_row['Prize']
        if final_prize == 'Gold':
            medal_changes['No Medal -> Gold'] += 1
        elif final_prize == 'Silver':
            medal_changes['No Medal -> Silver'] += 1
        elif final_prize == 'Bronze':
            medal_changes['No Medal -> Bronze'] += 1

print("Number of people who changed medals:")
for change, count in medal_changes.items():
    print(f"{change}: {count}")

print(f"\nNumber of people who improved their position: {improved_positions+new_participants}")
print(f"Number of people whose medal and position remained the same: {unchanged}")

