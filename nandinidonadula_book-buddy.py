# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from dataclasses import dataclass, field
import random

@dataclass
class Book:
    title: str
    author: str
    genre: str
    progress: int = 0  # percent complete
    finished: bool = False
    rating: int = 0  # 0-5
    review: str = ""

class BookBuddyAgent:
    def __init__(self):
        self.books = []
        self.suggestions = [
            ("Atomic Habits", "James Clear", "self-help"),
            ("To Kill a Mockingbird", "Harper Lee", "fiction"),
            ("Sapiens", "Yuval Noah Harari", "history"),
            ("The Alchemist", "Paulo Coelho", "fiction")
        ]

    def add_book(self, title, author, genre):
        self.books.append(Book(title, author, genre))
        return f"Added '{title}' by {author} ({genre})."

    def recommend(self):
        book = random.choice(self.suggestions)
        return f"How about: '{book[0]}' by {book[1]} ({book[2]})?"

    def list_books(self):
        if not self.books:
            return "No books in your list yet."
        return " ".join([f"{idx+1}. {b.title} ({b.genre}) - {b.progress}% read, {'finished' if b.finished else 'reading'}" for idx, b in enumerate(self.books)])

    def update_progress(self, idx, percent):
        if 0 <= idx < len(self.books):
            self.books[idx].progress = percent
            if percent >= 100:
                self.books[idx].finished = True
            return f"Updated progress on '{self.books[idx].title}' to {percent}%."
        return "Invalid book selection."

    def rate_review(self, idx, rating, review):
        if 0 <= idx < len(self.books):
            self.books[idx].rating = rating
            self.books[idx].review = review
            return f"Rated '{self.books[idx].title}' {rating}/5: {review}"
        return "Invalid book selection."

def main():
    agent = BookBuddyAgent()
    print("--- BookBuddy: Book Tracker ---")
    print("Commands: add, rec, list, progress, review, exit")
    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "add":
            title = input("Book title: ")
            author = input("Author: ")
            genre = input("Genre: ")
            print(agent.add_book(title, author, genre))
        elif cmd == "rec":
            print(agent.recommend())
        elif cmd == "list":
            print(agent.list_books())
        elif cmd == "progress":
            idx = int(input("Which book #? ")) - 1
            percent = int(input("Progress (%): "))
            print(agent.update_progress(idx, percent))
        elif cmd == "review":
            idx = int(input("Which book #? ")) - 1
            rating = int(input("Rating (0-5): "))
            review = input("Your review: ")
            print(agent.rate_review(idx, rating, review))
        elif cmd == "exit":
            print("Keep reading! Goodbye!")
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()

