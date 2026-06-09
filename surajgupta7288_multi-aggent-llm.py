# ============================================================
# TripGenie â€“ Multi-Agent AI Trip Planner (Single File Version)
# Kaggle / Jupyter / Normal Python Compatible
# ============================================================

import asyncio
import random
import datetime
import nest_asyncio
nest_asyncio.apply()    # <<< FIX for RuntimeError in Kaggle

# ------------------------------------------------------------
# Logging and utilities
# ------------------------------------------------------------

def log(event, **kwargs):
    print(f"[LOG] {event}: {kwargs}")

# Fake tool â€“ simulates hotel booking API
class BookingToolMock:
    async def book(self, location, hotel, price):
        await asyncio.sleep(0.2)
        log("booking_request", location=location, hotel=hotel, price=price)
        return {
            "status": "confirmed",
            "hotel": hotel,
            "location": location,
            "price": price
        }

booking_tool = BookingToolMock()

# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------

class SearchAgent:
    async def search(self, destination):
        await asyncio.sleep(0.5)
        options = [
            {"name": "Sunrise Resort", "price": random.randint(80, 130)},
            {"name": "Ocean View Hotel", "price": random.randint(100, 150)},
        ]
        log("search_results", options=options)
        return options

class PricingAgent:
    async def evaluate(self, hotel):
        await asyncio.sleep(0.3)
        rating = random.choice([4.0, 4.2, 4.5, 4.7, 4.9])
        hotel["rating"] = rating
        log("pricing_eval", hotel=hotel)
        return hotel

class VenueAgent:
    async def recommend(self, destination):
        await asyncio.sleep(0.3)
        venues = [
            f"{destination} Beach",
            f"{destination} Museum",
            f"{destination} Night Market"
        ]
        log("venue_recommendations", venues=venues)
        return venues

class ItineraryComposer:
    async def compose(self, hotel, venues):
        await asyncio.sleep(0.2)
        plan = {
            "hotel": hotel["name"],
            "daily_plan": [
                f"Morning â€“ Visit {venues[0]}",
                f"Afternoon â€“ Explore {venues[1]}",
                f"Evening â€“ Enjoy {venues[2]}"
            ]
        }
        log("itinerary_composed", itinerary=plan)
        return plan

# ------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------

async def demo_run():
    print("\n=== TripGenie AI Trip Planner ===\n")

    destination = "Goa"
    budget = 150

    search_agent = SearchAgent()
    pricing_agent = PricingAgent()
    venue_agent = VenueAgent()
    composer = ItineraryComposer()

    # Search hotels
    hotels = await search_agent.search(destination)

    # Evaluate pricing in parallel
    scored_hotels = await asyncio.gather(
        *[pricing_agent.evaluate(h) for h in hotels]
    )

    # Pick best hotel under budget
    best = max(
        [h for h in scored_hotels if h["price"] <= budget],
        key=lambda x: x["rating"],
        default=None
    )

    if not best:
        print("â�Œ No hotel found within budget.")
        return

    # Recommend places to visit
    venues = await venue_agent.recommend(destination)

    # Compose itinerary
    itinerary = await composer.compose(best, venues)

    # Book
    booking = await booking_tool.book(
        destination,
        best["name"],
        best["price"]
    )

    print("\n=== FINAL TRIP PLAN ===")
    print(f"Hotel: {best['name']} (${best['price']})")
    print(f"Rating: {best['rating']}")
    print("\nDaily Plan:")
    for step in itinerary["daily_plan"]:
        print(" -", step)

    print("\nBooking Status:", booking["status"])
    print("\nTrip planned successfully! ğŸ�‰")

# ------------------------------------------------------------
# MAIN (Kaggle-Safe)
# ------------------------------------------------------------

def main():
    import nest_asyncio
    nest_asyncio.apply()

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return loop.run_until_complete(demo_run())
    except RuntimeError:
        pass

    asyncio.run(demo_run())

if __name__ == "__main__":
    main()


