# --- FAZA 1: IMPORT I WCZYTANIE DANYCH ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

print("Narzędzia zaimportowane.")

# Definiujemy ścieżki do plików
# (Pamiętaj, że dodałeś folder 'm5-forecasting-accuracy' ręcznie)
sciezka_sprzedazy = '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv'
sciezka_kalendarza = '/kaggle/input/m5-forecasting-accuracy/calendar.csv'

# Wczytujemy pliki
print("Wczytuję plik sprzedaży... (chwilę to potrwa)")
df_sales = pd.read_csv(sciezka_sprzedazy)

print("Wczytuję kalendarz...")
df_calendar = pd.read_csv(sciezka_kalendarza)

print("\n--- Faza 1 Zakończona ---")
print("Dane wczytane. Oto 5 pierwszych wierszy sprzedaży:")
# Ta linia wyświetla tabelę:
print(df_sales.head())


# --- FAZA 2: UPROSZCZENIE I AGREGACJA ---

print("1. Filtruję dane tylko dla sklepu CA_1...")
df_sales_ca1 = df_sales[df_sales['store_id'] == 'CA_1']

# Znajdujemy kolumny ze sprzedażą (wszystkie, które zaczynają się na 'd_')
kolumny_sprzedazy = [kolumna for kolumna in df_sales_ca1.columns if kolumna.startswith('d_')]

print(f"2. Sumuję sprzedaż z {len(df_sales_ca1)} produktów dla każdego z {len(kolumny_sprzedazy)} dni...")
aggr_sales_ca1 = df_sales_ca1[kolumny_sprzedazy].sum(axis=0)

# "Obracamy" tabelę, żeby mieć ładne kolumny
df_aggr = aggr_sales_ca1.to_frame()
df_aggr = df_aggr.reset_index()
df_aggr.columns = ['dzien', 'laczna_sprzedaz']

print("3. Łączę sprzedaż z kalendarzem...")
# Łączymy naszą sprzedaż z kalendarzem (używamy 'df_calendar', który wczytaliśmy w Fazie 1)
df_final = pd.merge(df_aggr, df_calendar, left_on='dzien', right_on='d')

print("\n--- Faza 2 Zakończona ---")
print("Oto nasza gotowa tabela (pierwsze 5 dni):")
print(df_final[['date', 'laczna_sprzedaz', 'weekday', 'event_name_1']].head())


# --- FAZA 3: INŻYNIERIA CECH  ---

print("Tworzę cechy...")

# Wskazówka 1: Lagi (Opóźnienia)
df_final['lag_28'] = df_final['laczna_sprzedaz'].shift(28)
df_final['lag_7'] = df_final['laczna_sprzedaz'].shift(7)

# Wskazówka 2: Średnie Kroczące (Trendy)
df_final['srednia_kroczaca_7_z_lag28'] = df_final['lag_28'].rolling(window=7).mean()
df_final['srednia_kroczaca_30_z_lag28'] = df_final['lag_28'].rolling(window=30).mean()

# Wskazówka 3: Cechy z Kalendarza
df_final['czy_swieto'] = df_final['event_name_1'].notna().astype(int)
# Kolumny 'wday' i 'month' już mamy z kalendarza, więc są gotowe.

print("\n--- Faza 3 Zakończona ---")
print("Tabela wzbogacona o nowe cechy (widać 'NaN' - to normalne):")
kolumny_do_pokazania = [
    'date', 'laczna_sprzedaz', 'lag_28', 'srednia_kroczaca_7_z_lag28', 'wday', 'czy_swieto'
]
print(df_final[kolumny_do_pokazania].head(35))


# --- FAZA 4: MODELOWANIE (WERSJA "UNCERTAINTY") ---

print("Rozpoczynam Fazę 4 (Wersja Uncertainty)...")

# --- Krok 1: Definiowanie Cech i Czyszczenie Danych ---
# (Używamy tej samej poprawionej logiki co poprzednio)

CECHY = [
    'lag_28',
    'lag_7',
    'srednia_kroczaca_7_z_lag28',
    'srednia_kroczaca_30_z_lag28',
    'wday',        # Dzień tygodnia (z kalendarza)
    'month',       # Miesiąc (z kalendarza)
    'czy_swieto'   # Nasza cecha 0 lub 1
]
CEL = 'laczna_sprzedaz'
lista_kolumn_do_sprawdzenia = CECHY + [CEL]

# Czyścimy dane - usuwamy wiersze z 'NaN' TYLKO z kolumn, których używamy
df_model = df_final.dropna(subset=lista_kolumn_do_sprawdzenia)
print(f"1. Przygotowałem {len(df_model)} dni 'czystych' danych.")

# --- Krok 2: Podział na zbiór treningowy i walidacyjny ---
DNI_WALIDACJI = 28
df_walidacyjny = df_model.tail(DNI_WALIDACJI)
df_treningowy = df_model.iloc[:-DNI_WALIDACJI]

X_train = df_treningowy[CECHY]
y_train = df_treningowy[CEL]
X_walidacyjny = df_walidacyjny[CECHY]
y_walidacyjny = df_walidacyjny[CEL]

print(f"2. Dzielę dane: {len(X_train)} dni na naukę, {len(X_walidacyjny)} dni na sprawdzian.")

# --- Krok 3: Definiowanie 9 Kwantyli (Celów) ---
# To jest 9 scenariuszy, które musimy przewidzieć (zgodnie z wymogami konkursu)
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

# Stworzymy 'słownik', aby przechować 9 różnych prognoz
# Oraz jeden model do zapisania (dla mediany)
predykcje_kwantyli = {}
nasz_model_mediany = None # Zapiszemy tu model 0.500

# --- Krok 4: Trenowanie 9 Modeli (w pętli) ---
for q in QUANTILES:
    print(f"\n--- Trenuję model dla kwantyla: {q} ---")
    
    # Tworzymy nowy model dla TEGO kwantyla
    model = lgb.LGBMRegressor(
        objective='quantile',  # Mówimy mu: "przewiduj kwantyle"
        alpha=q,               # Mówimy mu: "przewiduj TEN KONKRETNY kwantyl"
        metric='quantile',     # Używamy też odpowiedniej miary błędu
        n_estimators=300,      # Użyjemy mniej drzew, żeby było szybciej
        n_jobs=-1,
        learning_rate=0.05
    )
    
    model.fit(X_train, y_train)
    
    # Generujemy prognozę i zapisujemy ją
    predykcja = model.predict(X_walidacyjny)
    predykcje_kwantyli[q] = predykcja
    
    # Zapisujemy model dla mediany (0.5), żeby móc go użyć na slajdzie
    if q == 0.500:
        nasz_model_mediany = model

print("\n--- Faza 4 Zakończona: Trening wszystkich 9 modeli zakończony! ---")


# --- FAZA 5: WIZUALIZACJA WYNIKÓW (UNCERTAINTY) ---

print("Rysuję wykres niepewności ...")

# Ustawiamy rozmiar wykresu
plt.figure(figsize=(15, 7))

# --- Rysowanie "Wstęg Niepewności" ---
# Wypełniamy obszar między kwantylami.

# Wstęga 99% (najszersza, najjaśniejsza)
plt.fill_between(
    df_walidacyjny['date'], # Oś X (daty)
    predykcje_kwantyli[0.005], # Dolna granica
    predykcje_kwantyli[0.995], # Górna granica
    color='gray', 
    alpha=0.2, 
    label='99% Przedział Ufności (Scenariusz skrajny)'
)

# Wstęga 95% (trochę węższa, ciemniejsza)
plt.fill_between(
    df_walidacyjny['date'], 
    predykcje_kwantyli[0.025], 
    predykcje_kwantyli[0.975], 
    color='orange', 
    alpha=0.2, 
    label='95% Przedział Ufności'
)

# Wstęga 50% (najwęższa, najciemniejsza)
plt.fill_between(
    df_walidacyjny['date'], 
    predykcje_kwantyli[0.250], 
    predykcje_kwantyli[0.750], 
    color='orange', 
    alpha=0.4, 
    label='50% Przedział Ufności (Najbardziej prawdopodobny zakres)'
)

# --- Rysowanie Linii ---
# Linia niebieska: Prawdziwa sprzedaż (dla porównania)
plt.plot(df_walidacyjny['date'], y_walidacyjny, label='Rzeczywista sprzedaż', color='blue', linewidth=2.0)

# Linia czerwona (przerywana): Nasza prognoza mediany (kwantyl 0.500)
plt.plot(df_walidacyjny['date'], predykcje_kwantyli[0.500], label='Prognoza mediany (Kwantyl 0.5)', linestyle='--', color='red', linewidth=2.0)


# --- Ustawienia wykresu ---
plt.title('Wynik Walidacji: Prognoza Niepewności (9 Kwantyli) dla sklepu CA_1')
plt.xlabel('Data')
plt.ylabel('Łączna sprzedaż')
plt.legend(loc='upper left') # Legenda w lewym górnym rogu
plt.xticks(rotation=45)      # Obróć daty na osi X
plt.grid(True)               # Włącz siatkę
plt.show()                   # Pokaż wykres!

