# Anime Recommendation Agent
import random

anime_database = {
    "action": ["Attack on Titan", "My Hero Academia", "One Punch Man", "Demon Slayer", "Naruto"],
    "comedy": ["KonoSuba", "One Punch Man", "The Devil is a Part-Timer", "Gintama", "Saiki K."],
    "romance": ["Toradora!", "Your Lie in April", "Fruits Basket", "Clannad", "Kaguya-sama"],
    "slice of life": ["March Comes in Like a Lion", "Barakamon", "Silver Spoon", "Anohana", "Natsume Yuujinchou"],
    "fantasy": ["Re:Zero", "Sword Art Online", "Overlord", "Made in Abyss", "Black Clover"],
    "supernatural": ["Death Note", "Jujutsu Kaisen", "Noragami", "Blue Exorcist", "Mob Psycho 100"]
}

# Pick a random genre and recommend 5 anime
selected_genre = random.choice(list(anime_database.keys()))
recommended = random.sample(anime_database[selected_genre], 5)

print(f"Recommended {selected_genre} anime:")
for a in recommended:
    print("-", a)


