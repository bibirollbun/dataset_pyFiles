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


# -------------------------------
# Class 12 Physics Numerical Bot
# Hindi Medium тАУ Simple AI Logic
# -------------------------------

def solve_newtons_second_law(m, a):
    f = m * a
    steps = f"\nрджрд┐рдпрд╛ рдЧрдпрд╛:\nрджреНрд░рд╡реНрдпрдорд╛рди (m) = {m} kg\nрддреНрд╡рд░рдг (a) = {a} m/s┬▓\n"
    steps += "рд╕реВрддреНрд░: F = m ├Ч a\n"
    steps += f"рд╣рд▓: F = {m} ├Ч {a} = {f} N\n"
    return steps

def solve_ohms_law(v, r):
    i = v / r
    steps = f"\nрджрд┐рдпрд╛ рдЧрдпрд╛:\nрд╡реЛрд▓реНрдЯреЗрдЬ (V) = {v} volt\nрдкреНрд░рддрд┐рд░реЛрдз (R) = {r} ohm\n"
    steps += "рд╕реВрддреНрд░: I = V / R\n"
    steps += f"рд╣рд▓: I = {v} / {r} = {i} ampere\n"
    return steps

def solve_kinetic_energy(m, v):
    ke = 0.5 * m * v * v
    steps = f"\nрджрд┐рдпрд╛ рдЧрдпрд╛:\nрджреНрд░рд╡реНрдпрдорд╛рди (m) = {m} kg\nрд╡реЗрдЧ (v) = {v} m/s\n"
    steps += "рд╕реВрддреНрд░: K.E = 1/2 m v┬▓\n"
    steps += f"рд╣рд▓: K.E = 1/2 ├Ч {m} ├Ч {v}┬▓ = {ke} joule\n"
    return steps

def menu():
    print("\nЁЯФ╖ Class 12 Physics Numerical Practice Bot (Hindi Medium)")
    print("---------------------------------------------------------")
    print("1. рдиреНрдпреВрдЯрди рдХрд╛ рджреВрд╕рд░рд╛ рдирд┐рдпрдо (F = m ├Ч a)")
    print("2. рдУрдо рдХрд╛ рдирд┐рдпрдо (I = V / R)")
    print("3. рдЧрддрд┐рдЬ рдКрд░реНрдЬрд╛ (K.E = 1/2 m v┬▓)")
    print("4. рдмрд╛рд╣рд░ рдирд┐рдХрд▓реЗрдВ")

while True:
    menu()
    choice = input("\nрдХреГрдкрдпрд╛ рд╡рд┐рдХрд▓реНрдк рдЪреБрдиреЗрдВ (1-4): ")

    if choice == "1":
        m = float(input("рджреНрд░рд╡реНрдпрдорд╛рди (kg) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        a = float(input("рддреНрд╡рд░рдг (m/s┬▓) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        print(solve_newtons_second_law(m, a))

    elif choice == "2":
        v = float(input("рд╡реЛрд▓реНрдЯреЗрдЬ (volt) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        r = float(input("рдкреНрд░рддрд┐рд░реЛрдз (ohm) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        print(solve_ohms_law(v, r))

    elif choice == "3":
        m = float(input("рджреНрд░рд╡реНрдпрдорд╛рди (kg) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        v = float(input("рд╡реЗрдЧ (m/s) рджрд░реНрдЬ рдХрд░реЗрдВ: "))
        print(solve_kinetic_energy(m, v))

    elif choice == "4":
        print("рдзрдиреНрдпрд╡рд╛рдж! рдлрд┐рд░ рдорд┐рд▓реЗрдВрдЧреЗ ЁЯШК")
        break
    
    else:
        print("рдЧрд▓рдд рд╡рд┐рдХрд▓реНрдк! рдлрд┐рд░ рд╕реЗ рдкреНрд░рдпрд╛рд╕ рдХрд░реЗрдВред")


def main():
    score = 0

    print("===== Physics Objective Quiz =====\n")

    # Q1
    print("1. Speed ka SI unit kya hai?")
    print(" 1) m/s   2) km/h   3) m^2/s   4) N")
    answer = int(input("Your answer: "))

    if answer == 1:
        score += 1

    # Q2
    print("\n2. Force ka SI unit kya hai?")
    print(" 1) Joule   2) Pascal   3) Newton   4) Watt")
    answer = int(input("Your answer: "))

    if answer == 3:
        score += 1

    # Final score
    print("\nYour total score:", score, "/ 2")


main()


import random

# ---------------------------
# рдкреНрд░рд╢реНрди рдмреИрдВрдХ (Hindi MCQs)
# ---------------------------
questions = [
    {
        "q": "рдХреМрди-рд╕рд╛ рдирд┐рдпрдо рдЧрддрд┐ рдХреЗ рд╕рд╛рде рд╕рдореНрдмрдиреНрдзрд┐рдд рд╣реИ?",
        "options": ["рдХреВрд▓реЙрдореНрдм рдХрд╛ рдирд┐рдпрдо", "рдиреНрдпреВрдЯрди рдХрд╛ рдкреНрд░рдердо рдирд┐рдпрдо", "рд╣реИрдЬрд╝реЗрдирдмрд░реНрдЧ рдЕрдирд┐рд╢реНрдЪрд┐рддрддрд╛", "рдмрд░реНрдиреМрд▓реА рд╕рд┐рджреНрдзрд╛рдВрдд"],
        "answer": 1
    },
    {
        "q": "рдПрдХ рдЪрд╛рд▓реВ рдзрд╛рд░рд╛ I рд╕реЗ рдЧреБрдЬрд░рддреЗ рддрд╛рд░ рдкрд░ рдЪреБрдВрдмрдХреАрдп рдХреНрд╖реЗрддреНрд░ рд╣реЛрддрд╛ рд╣реИред рдЙрд╕рдХреА рджрд┐рд╢рд╛ рд╕рдордЭрд╛рдиреЗ рд╡рд╛рд▓рд╛ рдирд┐рдпрдо рдХреМрди-рд╕рд╛ рд╣реИ?",
        "options": ["рд░рд╛рдЗрдЯ-рд╣реИрдВрдб рдирд┐рдпрдо", "рд▓реЗрдлреНрдЯ-рд╣реИрдВрдб рдирд┐рдпрдо", "рдмрд╛рдпреЛрдЯ-рд╕рд╛рд╡рд░ рдирд┐рдпрдо", "рд╣реБрдХ рдХрд╛ рдирд┐рдпрдо"],
        "answer": 0
    },
    {
        "q": "рдХрд┐рд╕ рдкреНрд░рдХреНрд░рд┐рдпрд╛ рдореЗрдВ рдКрд░реНрдЬрд╛ рдХрд╛ рд╡рд┐рдирд┐рдордп рдмрд┐рдирд╛ рд╡рд░реНрдореАрдп рд╕рдВрдкрд░реНрдХ рдХреЗ рд╣реЛрддрд╛ рд╣реИ?",
        "options": ["рдЪрд╛рд▓рдХрддрд╛ (Conduction)", "рдкреНрд░рд╡рд╣рди (Convection)", "рддрд╛рдк рд╕рдВрдЪрд╛рд░ (Radiation)", "рджреАрдкреНрддрд┐ (Luminescence)"],
        "answer": 2
    },
    {
        "q": "рдлреЛрдЯреЙрди рдХреА рдКрд░реНрдЬрд╛ рдХрд┐рд╕рд╕реЗ рд╕рдВрдмрдВрдзрд┐рдд рд╣реИ?",
        "options": ["рдЖрдпрд╛рдо рд╕реЗ", "рдЖрд╡реГрддреНрддрд┐ рд╕реЗ", "рдЖрдпрддрди рд╕реЗ", "рджрд╛рдм рд╕реЗ"],
        "answer": 1
    },
    {
        "q": "рдХрд┐рд╕реЗ рджреНрд░рд╡реНрдпрдорд╛рди рдХреЗ рд░реВрдк рдореЗрдВ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд┐рдпрд╛ рдЬрд╛рддрд╛ рд╣реИ?",
        "options": ["рдКрд░реНрдЬрд╛ рдХрд╛ рдорд╛рддреНрд░рдХ", "рдкреНрд░рддрд┐рд░реЛрдз рдХрд╛ рдорд╛рддреНрд░рдХ", "рдкрджрд╛рд░реНрде рдХреА рдЬрдбрд╝рддреНрд╡ (inertia)", "рд░рдлреНрддрд╛рд░ рдХрд╛ рдорд╛рддреНрд░рдХ"],
        "answer": 2
    },
    {
        "q": "рдХрд┐рд╕ рдЗрдХрд╛рдИ рдореЗрдВ рд╡рд┐рджреНрдпреБрдд рдзрд╛рд░рд╛ рдорд╛рдкреА рдЬрд╛рддреА рд╣реИ?",
        "options": ["рд╡реЛрд▓реНрдЯ (V)", "рдУрдо (╬й)", "рдПрдореНрдкрд┐рдпрд░ (A)", "рдЬреВрд▓ (J)"],
        "answer": 2
    },
    {
        "q": "рд╣рд╛рдЗрдбреНрд░реЛрд╕реНрдЯреИрдЯрд┐рдХ рдкреНрд░реЗрд╢рд░ рдХрд┐рд╕реА рдмрд┐рдВрджреБ рдкрд░ рдХреНрдпрд╛ рдирд┐рд░реНрднрд░ рдХрд░рддрд╛ рд╣реИ?",
        "options": ["рдкрд╛рдиреА рдХреЗ рддрд╛рдкрдорд╛рди рдкрд░", "рдКрдБрдЪрд╛рдИ рдкрд░", "рд╕рддрд╣ рдХреНрд╖реЗрддреНрд░рдлрд▓ рдкрд░", "рд░рдВрдЧ рдкрд░"],
        "answer": 1
    },
    {
        "q": "рд╕рд╛рдорд╛рдиреНрдп рдкрд░рд┐рд╕реНрдерд┐рддрд┐рдпреЛрдВ рдореЗрдВ рдзреНрд╡рдирд┐ рддрд░рдВрдЧ рдХрд┐рд╕ рдкреНрд░рдХрд╛рд░ рдХреА рд╣реЛрддреА рд╣реИ?",
        "options": ["рд╕рдВрд╡реЗрдЧ рддрд░рдВрдЧ (transverse)", "рджреАрд░реНрдШрд╛рдХрд╛рд░ рддрд░рдВрдЧ", "рд╕рдордорд┐рдд рддрд░рдВрдЧ", "рдЕрд╡рдзрд┐ рддрд░рдВрдЧ (longitudinal)"],
        "answer": 3
    },
    {
        "q": "рдХрд┐рд╕реЗ рд╕рдорддрд╛рдкреА рдкреНрд░рдХреНрд░рд┐рдпрд╛ (isothermal) рдХрд╣рд╛ рдЬрд╛рддрд╛ рд╣реИ?",
        "options": ["рдЬрдм рддрд╛рдк рдмрджрд▓рддрд╛ рд╣реИ", "рдЬрдм рджрд╛рдм рдмрджрд▓рддрд╛ рд╣реИ", "рдЬрдм рддрд╛рдк рдирд┐рдпрдд рд░рд╣рддрд╛ рд╣реИ", "рдЬрдм рдЖрдпрддрди рдирд┐рдпрдд рд░рд╣рддрд╛ рд╣реИ"],
        "answer": 2
    },
    {
        "q": "рд░реЗрдЦреАрдп рдЧрддрд┐ (linear momentum) рдХрд╛ рд╕реВрддреНрд░ рдХреНрдпрд╛ рд╣реИ?",
        "options": ["p = m + v", "p = mv", "p = m/v", "p = v/m"],
        "answer": 1
    },
]

# ---------------------------
# Quiz Logic (Console)
# ---------------------------

print("\n===== Class 12 Physics Quiz (Hindi) =====\n")

score = 0
random.shuffle(questions)

for i, q in enumerate(questions, start=1):
    print(f"\nрдкреНрд░рд╢реНрди {i}: {q['q']}")
    for idx, opt in enumerate(q["options"]):
        print(f" {idx+1}. {opt}")
    
    # User input
    try:
        user = int(input("рдЖрдкрдХрд╛ рдЙрддреНрддрд░ (1-4): ")) - 1
    except:
        user = -1

    # Check answer
    if user == q["answer"]:
        print("тЬФ рд╕рд╣реА рдЙрддреНрддрд░!")
        score += 1
    else:
        correct_opt = q["options"][q["answer"]]
        print(f"тЬШ рдЧрд▓рдд! рд╕рд╣реА рдЙрддреНрддрд░: {correct_opt}")

print("\n---------------------------")
print(f"рдЖрдкрдХрд╛ рдХреБрд▓ рд╕реНрдХреЛрд░: {score} / {len(questions)}")
print("---------------------------\n")

