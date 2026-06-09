# Travel Stock – Travel Data Management & Recommendation System

travel_data = []

# Function to add destination
def add_destination():
    name = input("Enter destination name: ")
    country = input("Enter country: ")
    budget = int(input("Enter average budget (in USD): "))
    rating = float(input("Enter rating (1-5): "))

    data = {
        "name": name,
        "country": country,
        "budget": budget,
        "rating": rating
    }
    travel_data.append(data)
    print("\nDestination added successfully!\n")

# Function to view all destinations
def view_destinations():
    if not travel_data:
        print("\nNo data available!\n")
        return
    print("\n--- All Travel Destinations ---")
    for d in travel_data:
        print(f"Name: {d['name']}, Country: {d['country']}, Budget: {d['budget']}, Rating: {d['rating']}")

# Function to search by country
def search_by_country():
    country = input("Enter country name to search: ")
    found = [d for d in travel_data if d["country"].lower() == country.lower()]

    if found:
        print("\n--- Destinations Found ---")
        for d in found:
            print(f"{d['name']} - Budget: {d['budget']}, Rating: {d['rating']}")
    else:
        print("\nNo destinations found in this country.\n")

# Function to filter by budget
def filter_by_budget():
    max_budget = int(input("Enter your maximum budget: "))
    filtered = [d for d in travel_data if d["budget"] <= max_budget]

    if filtered:
        print("\n--- Budget Friendly Options ---")
        for d in filtered:
            print(f"{d['name']} ({d['country']}) - Budget: {d['budget']}")
    else:
        print("\nNo destinations in your budget.\n")

# Top rated destinations
def top_rated():
    if not travel_data:
        print("\nNo data available!\n")
        return

    sorted_data = sorted(travel_data, key=lambda x: x["rating"], reverse=True)
    print("\n--- Top Rated Destinations ---")
    for d in sorted_data[:3]:
        print(f"{d['name']} - Rating: {d['rating']}")

# Recommendation system
def recommend():
    budget = int(input("Enter your budget: "))

    options = [d for d in travel_data if d["budget"] <= budget]
    if not options:
        print("\nNo recommendations available.\n")
        return

    best = max(options, key=lambda x: x["rating"])
    print("\n--- Recommended Destination ---")
    print(f"{best['name']} ({best['country']}) - Budget: {best['budget']}, Rating: {best['rating']}")

# Menu
while True:
    print("\n==== TRAVEL STOCK MENU ====")
    print("1. Add Destination")
    print("2. View All Destinations")
    print("3. Search by Country")
    print("4. Filter by Budget")
    print("5. Top Rated Destinations")
    print("6. Recommendation System")
    print("7. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        add_destination()
    elif choice == 2:
        view_destinations()
    elif choice == 3:
        search_by_country()
    elif choice == 4:
        filter_by_budget()
    elif choice == 5:
        top_rated()
    elif choice == 6:
        recommend()
    elif choice == 7:
        print("Thank you for using Travel Stock!")
        break
    else:
        print("Invalid choice! Try again.")

