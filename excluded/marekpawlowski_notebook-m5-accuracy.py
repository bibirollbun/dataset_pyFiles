# Krok 1: Wczytujemy bibliotekę 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Krok 2: Podajemy Pythonowi dokładną ścieżkę do pliku
# W Kaggle te pliki są zawsze w takim specjalnym folderze '/kaggle/input/...'
sciezka_do_pliku = '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv'

# Krok 3: Wczytujemy plik do zmiennej o nazwie 'df_sales'
# 'df' to skrót od DataFrame, czyli takiej tabeli
print("Wczytuję plik... to może chwilę potrwać, jest duży.")
df_sales = pd.read_csv(sciezka_do_pliku)

# Krok 4: Wyświetlamy 5 pierwszych wierszy, żeby sprawdzić, czy działa
print("Wczytywanie zakończone! Oto 5 pierwszych wierszy:")
print(df_sales.head())


# Krok 1: Filtrowanie
# Bierzemy naszą dużą tabelę 'df_sales' i zostawiamy tylko te wiersze,
# w których kolumna 'store_id' ma wartość 'CA_1'.
print("1. Filtruję dane tylko dla sklepu CA_1...")
df_sales_ca1 = df_sales[df_sales['store_id'] == 'CA_1']

# Krok 2: Znalezienie kolumn ze sprzedażą
# Musimy zsumować wszystkie kolumny od 'd_1' do 'd_1913'.
# Ten kod automatycznie znajduje wszystkie nazwy kolumn, które zaczynają się na 'd_'
kolumny_sprzedazy = [kolumna for kolumna in df_sales_ca1.columns if kolumna.startswith('d_')]
print(f"Znalazłem {len(kolumny_sprzedazy)} dni sprzedaży.")

# Krok 3: Agregacja (Sumowanie)
# Sumujemy sprzedaż ze wszystkich wierszy (produktów) dla każdego dnia.
# axis=0 oznacza "sumuj pionowo" (każdą kolumnę osobno).
# Wynikiem będzie seria danych: d_1 = 500, d_2 = 450, itd.
print("2. Sumuję sprzedaż wszystkich produktów w sklepie CA_1 dla każdego dnia...")
aggr_sales_ca1 = df_sales_ca1[kolumny_sprzedazy].sum(axis=0)

# Krok 4: Czyszczenie i "Obracanie" tabeli
# Zamieniamy wynik na ładną tabelę (DataFrame) i poprawiamy nazwy kolumn.
df_aggr = aggr_sales_ca1.to_frame()       # Zamień na tabelę
df_aggr = df_aggr.reset_index()          # Przenieś 'd_1', 'd_2' z indeksu do kolumny
df_aggr.columns = ['dzien', 'laczna_sprzedaz'] # Nadaj kolumnom czytelne nazwy
print("3. Stworzyłem nową tabelę ze sprzedażą zagregowaną:")
print(df_aggr.head()) # Pokaż 5 pierwszych wierszy nowej tabeli

# Krok 5: Wczytanie kalendarza
# Musimy wiedzieć, który dzień ('d_1') to jaka data i jaki dzień tygodnia.
print("\n4. Wczytuję kalendarz...")
sciezka_kalendarza = '/kaggle/input/m5-forecasting-accuracy/calendar.csv'
df_calendar = pd.read_csv(sciezka_kalendarza)

# Krok 6: Połączenie sprzedaży z kalendarzem (Finał Fazy 2)
# Łączymy naszą tabelę sprzedaży z kalendarzem.
# To jak WYSZUKAJ.PIONOWO w Excelu.
# Łączymy kolumnę 'dzien' (np. 'd_1') z kolumną 'd' z kalendarza (też 'd_1').
print("5. Łączę sprzedaż z kalendarzem...")
df_final = pd.merge(df_aggr, df_calendar, left_on='dzien', right_on='d')

# Krok 7: Wyświetlenie ostatecznego wyniku Fazy 2
print("\n--- FAZA 2 ZAKOŃCZONA ---")
print("Oto nasza gotowa tabela do dalszej pracy:")
# Wyświetlamy tylko te kolumny, które nas interesują
print(df_final[['date', 'dzien', 'laczna_sprzedaz', 'weekday', 'event_name_1']].head())


import pandas as pd # Czasem warto powtórzyć import, na wszelki wypadek

print("--- START FAZY 3: Tworzenie 'Wskazówek' (Cech) ---")

# --- Wskazówka 1: Lagi (Opóźnienia) ---
# To najważniejsza wskazówka w prognozowaniu.
# Pytanie: "Ile sprzedało się X dni temu?"
# .shift(28) "przesuwa" całą kolumnę sprzedaży o 28 wierszy w dół.
# Dlaczego 28? Bo prognozujemy 28 dni do przodu. To nasza główna poszlaka.
df_final['lag_28'] = df_final['laczna_sprzedaz'].shift(28)

# Dodajmy też lag 7, aby uchwycić sezonowość tygodniową
# (Sprzedaż w poniedziałek jest podobna do sprzedaży w poprzedni poniedziałek)
df_final['lag_7'] = df_final['laczna_sprzedaz'].shift(7)


# --- Wskazówka 2: Średnie Kroczące (Trendy) ---
# Pytanie: "Jaki był ogólny trend sprzedaży?"
# Obliczamy średnią sprzedaż z 7 dni, ale na danych "przesuniętych" o 28 dni.
# To mówi modelowi: "Jaki był tygodniowy trend miesiąc temu?"
# .rolling(window=7) tworzy "okno" o szerokości 7 dni
# .mean() oblicza średnią z tego okna
df_final['srednia_kroczaca_7_z_lag28'] = df_final['lag_28'].rolling(window=7).mean()

# To samo, ale dla trendu 30-dniowego (miesięcznego)
df_final['srednia_kroczaca_30_z_lag28'] = df_final['lag_28'].rolling(window=30).mean()


# --- Wskazówka 3: Cechy z Kalendarza (Upraszczanie) ---
# Model nie rozumie tekstu (np. "SuperBowl"), ale rozumie liczby (0 lub 1).
# Stwórzmy prostą wskazówkę: 1 jeśli jest święto, 0 jeśli nie ma.

# .notna() sprawdza, czy komórka w 'event_name_1' *nie jest pusta*
# .astype(int) zamienia wynik (True/False) na liczbę (1/0)
df_final['czy_swieto'] = df_final['event_name_1'].notna().astype(int)

# Dzień tygodnia i miesiąc (kolumny 'wday' i 'month') już mamy
# wczytane z kalendarza - są w formacie liczbowym, więc są gotowe!


# --- Krok 4: Wyświetlenie wyników Fazy 3 ---
print("--- FAZA 3 ZAKOŃCZONA ---")
print("Oto nasza tabela z nowymi 'wskazówkami' (cechami):")

# Wybieramy tylko te kolumny, które nas interesują, żeby zobaczyć wynik
kolumny_do_pokazania = [
    'date',                # Data
    'laczna_sprzedaz',     # Nasz CEL (to, co przewidujemy)
    'lag_28',              # Wskazówka 1
    'lag_7',               # Wskazówka 2
    'srednia_kroczaca_7_z_lag28', # Wskazówka 3
    'wday',                # Wskazówka 4 (Dzień tygodnia z kalendarza, 1=Sob)
    'month',               # Wskazówka 5 (Miesiąc z kalendarza)
    'czy_swieto'           # Wskazówka 6 (Nasza nowa)
]

# .head(35) pokaże nam 35 pierwszych wierszy
# Zwróć uwagę na puste pola 'NaN' na początku - to normalne!
# Pojawiają się, bo np. w dniu 3 nie ma danych sprzed 28 dni.
# Nasz model sobie z nimi poradzi (potem je usuniemy).
print(df_final[kolumny_do_pokazania].head(35))


# Import narzędzi do modelowania i rysowania wykresów
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt # To narzędzie do rysowania wykresów

print("--- START FAZY 4 (POPRAWIONA): Modelowanie i Walidacja ---")

# --- Krok 1: Definiowanie Cech (Wskazówek) i Celu ---
# PRZENIESIONE NA POCZĄTEK, ŻEBY POPRAWIĆ KROK 2
print("1. Definiuję, które kolumny to 'wskazówki', a która to 'cel'.")
# Lista naszych 'wskazówek' (muszą pasować do nazw z Fazy 3)
CECHY = [
    'lag_28',
    'lag_7',
    'srednia_kroczaca_7_z_lag28',
    'srednia_kroczaca_30_z_lag28',
    'wday',        # Dzień tygodnia (z kalendarza)
    'month',       # Miesiąc (z kalendarza)
    'czy_swieto'   # Nasza cecha 0 lub 1
]

CEL = 'laczna_sprzedaz' # To, co chcemy przewidzieć


# --- Krok 2: Czyszczenie danych (TERAZ POPRAWIONE) ---
# Prawidłowy sposób: usuwamy puste wiersze TYLKO z kolumn, których używamy.
# Tworzymy listę wszystkich kolumn potrzebnych do modelowania
lista_kolumn_do_sprawdzenia = CECHY + [CEL]

# .dropna(subset=...) usuwa wiersze, które mają 'NaN' TYLKO w podanych kolumnach
df_model = df_final.dropna(subset=lista_kolumn_do_sprawdzenia)
print(f"2. Usunąłem puste wiersze. Zostało nam {len(df_model)} dni do analizy. (Teraz powinno być dużo!)")


# --- Krok 3: Podział na zbiór treningowy i walidacyjny ---
# Ten krok jest taki sam, ale teraz zadziała, bo mamy dużo danych
DNI_WALIDACJI = 28

# Zbiór walidacyjny = ostatnie 28 wierszy
df_walidacyjny = df_model.tail(DNI_WALIDACJI)

# Zbiór treningowy = wszystko OPRÓCZ ostatnich 28 wierszy
df_treningowy = df_model.iloc[:-DNI_WALIDACJI]

print(f"3. Dzielę dane: {len(df_treningowy)} dni na naukę, {len(df_walidacyjny)} dni na sprawdzian.")


# --- Krok 4: Przygotowanie danych dla modelu (format X, y) ---
X_train = df_treningowy[CECHY]
y_train = df_treningowy[CEL]

X_walidacyjny = df_walidacyjny[CECHY]
y_walidacyjny = df_walidacyjny[CEL]


# --- Krok 5: Trenowanie Modelu LightGBM ---
print("4. Rozpoczynam trening 'ucznia' (modelu LightGBM)...")
# Ten kod się nie zmienia
model = lgb.LGBMRegressor(
    objective='regression_l1', 
    metric='rmse',
    n_estimators=500,
    n_jobs=-1,
    learning_rate=0.05
)

model.fit(X_train, y_train,
          eval_set=[(X_walidacyjny, y_walidacyjny)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100)])


# --- Krok 6: Generowanie Predykcji Walidacyjnych ---
print("5. Trening zakończony. Generuję prognozę na dni walidacyjne (sprawdzian)...")
predykcje_walidacyjne = model.predict(X_walidacyjny)


# --- Krok 7: Ocena Wyników i Wizualizacja (Najważniejszy Krok!) ---
print("6. Sprawdzam, jak bardzo model się pomylił...")
rmse = np.sqrt(mean_squared_error(y_walidacyjny, predykcje_walidacyjne))
print(f"\n--- WYNIK WALIDACJI ---")
print(f"Błąd modelu (RMSE): {rmse}")
print("To jest liczba, którą wpiszesz na slajd prezentacji.")
print("\n7. Rysuję wykres (to będzie Twój najważniejszy slajd!)...")

plt.figure(figsize=(15, 6))
plt.plot(df_walidacyjny['date'], y_walidacyjny, label='Rzeczywista sprzedaż')
plt.plot(df_walidacyjny['date'], predykcje_walidacyjne, label='Nasza prognoza', linestyle='--')
plt.title('Wynik Walidacji: Rzeczywistość vs Prognoza (ostatnie 28 dni)')
plt.xlabel('Data')
plt.ylabel('Łączna sprzedaż')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show() # Pokaż wykres!

