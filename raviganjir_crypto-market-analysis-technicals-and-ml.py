import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ 'model_lstm' рдКрдкрд░ рдЯреНрд░реЗрди рд╣реЛ рдЪреБрдХрд╛ рд╣реИред

print("\n--- ЁЯдЦ RQC LSTM SIGNAL (Self-Loading) ---")

try:
    # 1. рдбреЗрдЯрд╛ рд▓реЛрдб рдХрд░реЗрдВ (рдЕрдЧрд░ рдореЗрдореЛрд░реА рдореЗрдВ рдирд╣реАрдВ рд╣реИ рддреЛ)
    # рдпрд╣ рд╕рд┐рд░реНрдл рдПрдХ рд╕реБрд░рдХреНрд╖рд╛ рдЙрдкрд╛рдп рд╣реИ
    ticker_symbol = "BTC-USD"
    # рд╣рдо рд╕рд┐рд░реНрдл 60 рджрд┐рди рд╕реЗ рдЬрд╝реНрдпрд╛рджрд╛ рдХрд╛ рдбреЗрдЯрд╛ рд▓реЛрдб рдХрд░реЗрдВрдЧреЗ
    crypto_data = yf.download(ticker_symbol, start="2024-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    
    if crypto_data.empty:
        print("тЭМ ERROR: BTC Data could not be loaded via yfinance.")
        raise Exception("Data Load Failure")
        
    # [***ALERT***: рдмрд╛рдХреА рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ (MA, RSI, BBands) рдХреИрд▓рдХреБрд▓реЗрдЯ рдХрд░рдиреЗ рд╡рд╛рд▓реЗ рдХреЛрдб рдХреЛ рднреА 
    #  рдЖрдкрдХреЛ рдЗрд╕ рд╕реЗрд▓ рдХреЗ рдКрдкрд░ (Prediction рд╕реЗ рдкрд╣рд▓реЗ) рдЪрд▓рд╛рдирд╛ рд╣реЛрдЧрд╛!]

    # 2. рдЖрд╡рд╢реНрдпрдХ рдлреАрдЪрд░реНрд╕ рд▓реЗрдВ рдФрд░ Scale рдХрд░реЗрдВ
    # (рдорд╛рди рд▓реЗрдВ рдХрд┐ рдЖрдкрдХреЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреИрд▓рдХреБрд▓реЗрдЯ рд╣реЛ рдЪреБрдХреЗ рд╣реИрдВ)
    features = [
        'Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band'
    ]
    
    # [***ALERT***: рдпрджрд┐ 'scaler' рдФрд░ 'target_scaler' рдореМрдЬреВрдж рдирд╣реАрдВ рд╣реИрдВ, 
    #  рддреЛ рдпрд╣ рдХреЛрдб рднреА рдПрд░рд░ рджреЗрдЧрд╛ред LSTM рдХреЗ рд▓рд┐рдП, 'Run All' рд╣реА рд╕рдмрд╕реЗ рдЕрдЪреНрдЫрд╛ рд╕рдорд╛рдзрд╛рди рд╣реИред]
    
    # ***рдЖрдкрдХреЛ рдЕрднреА рднреА LSTM рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА рдХреЛрдб (рд╕реНрдХреЗрд▓рд┐рдВрдЧ рд╡рд╛рд▓рд╛) рдЪрд▓рд╛рдирд╛ рд╣реЛрдЧрд╛ рддрд╛рдХрд┐ 'scaler' рдореМрдЬреВрдж рд╣реЛ!***
    print("TIP: рдХреГрдкрдпрд╛ LSTM рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА (рд╕реНрдХреЗрд▓рд┐рдВрдЧ рдФрд░ рд╕реАрдХреНрд╡реЗрдВрд╕) рд╡рд╛рд▓реЗ рд╕реЗрд▓реНрд╕ рдЪрд▓рд╛рдПрдБ!")
    
except NameError as ne:
    print(f"\nтЭМ FINAL SOLUTION: Please press the 'Save & Run All' button to define all variables.")
except Exception as e:
    print(f"\nтЭМ GENERIC ERROR: {e}")




# рдЙрджрд╛рд╣рд░рдг: рдЕрдЧрд░ рдбреЗрдЯрд╛ рдкрд╣рд▓реЗ рд▓реЛрдб рдХрд┐рдпрд╛ рдЧрдпрд╛ рдерд╛, рддреЛ рдпрд╣ рд▓рд╛рдЗрди рдЬрд╝рд░реВрд░ рд╣реЛрдиреА рдЪрд╛рд╣рд┐рдП
crypto_data = pd.read_csv('path/to/your/bitcoin_data.csv') 
# рдпрд╛
# crypto_data = get_binance_klines(...) 



рдХреЛрдб рдХрд╛ рдХрд╛рдо рдЙрджреНрджреЗрд╢реНрдп
1. import yfinance рдФрд░ рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб crypto_data рдмрдирд╛рддрд╛ рд╣реИред
2. Moving Averages, RSI, MACD, BBands рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреЙрд▓рдо рдмрдирд╛рддрд╛ рд╣реИред
3. LSTM рд▓рд╛рдЗрдмреНрд░реЗрд░реАрдЬрд╝ рдЗрдореНрдкреЛрд░реНрдЯ tensorflow рдФрд░ MinMaxScaler рд▓реЛрдб рдХрд░рддрд╛ рд╣реИред
4. LSTM рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА (рд╕реНрдХреЗрд▓рд┐рдВрдЧ рдФрд░ рд╕реАрдХреНрд╡реЗрдВрд╕) X_lstm, y_lstm, scaler, рдФрд░ target_scaler рдмрдирд╛рддрд╛ рд╣реИред (рдпрд╣ рд╕рдмрд╕реЗ рдЬрд╝рд░реВрд░реА рд╣реИ!)
5. LSTM рдореЙрдбрд▓ рдЖрд░реНрдХрд┐рдЯреЗрдХреНрдЪрд░ model_lstm рдмрдирд╛рддрд╛ рд╣реИред
6. LSTM рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ (model_lstm.fit) рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░рддрд╛ рд╣реИред
7. LSTM рдкреНрд░реЗрдбрд┐рдХреНрд╢рди рдХреЛрдб рдЖрдкрдХреЛ BUY/SELL рд╕рд┐рдЧреНрдирд▓ рджреЗрддрд╛ рд╣реИред





import numpy as np
import pandas as pd
# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ 'scaler', 'target_scaler', 'sequence_length', рдФрд░ 'model_lstm' рдКрдкрд░ рдХреЗ рд╕реЗрд▓реНрд╕ рдореЗрдВ рдореМрдЬреВрдж рд╣реИрдВред

print("\n--- ЁЯдЦ ADVANCED RQC LSTM TRADING SIGNAL ---")

try:
    # 1. рдЖрдЬ рдХреЗ рд▓рд┐рдП рдЖрд╡рд╢реНрдпрдХ рдлреАрдЪрд░реНрд╕ рд▓реЗрдВ
    features = [
        'Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band'
    ]
    
    # 2. рд╕рдмрд╕реЗ рдЕрдВрддрд┐рдо 60 рджрд┐рди рдХрд╛ рдбреЗрдЯрд╛ рд▓реЗрдВ (LSTM рд╕реАрдХреНрд╡реЗрдВрд╕ рд▓реЗрдВрде)
    # рдпрд╣ рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░рддрд╛ рд╣реИ рдХрд┐ рдореЙрдбрд▓ рдХреЗ рдкрд╛рд╕ 'рдореЗрдореЛрд░реА' рдХреЗ рд▓рд┐рдП рдкрд░реНрдпрд╛рдкреНрдд рдбреЗрдЯрд╛ рд╣реЛред
    last_60_days = crypto_data[features].iloc[-sequence_length:].values
    
    # 3. рдбреЗрдЯрд╛ рдХреЛ рд╕реНрдХреЗрд▓ рдХрд░реЗрдВ (рдЯреНрд░реЗрдирд┐рдВрдЧ рд╡рд╛рд▓реЗ 'scaler' рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ)
    scaled_input = scaler.transform(last_60_days)
    
    # 4. рдбреЗрдЯрд╛ рдХреЛ LSTM рдХреЗ рд▓рд┐рдП 3D Format рдореЗрдВ рдмрджрд▓реЗрдВ: (1, 60, 7)
    X_predict = np.array([scaled_input])
    
    # 5. LSTM рд╕реЗ рдХрд▓ рдХреЗ Close Price рдХреА рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░реЗрдВ (Scaled Output)
    predicted_price_scaled = model_lstm.predict(X_predict)[0]
    
    # 6. Scaled Output рдХреЛ рдЕрд╕рд▓реА рдХреАрдордд рдореЗрдВ рдмрджрд▓реЗрдВ (Inverse Transform)
    # Target Price (Close Price) рдХреЗ рд▓рд┐рдП Target Scaler рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВ
    
    # ***рдорд╣рддреНрд╡рдкреВрд░реНрдг: рдЪреВрдВрдХрд┐ рдЖрдкрдиреЗ Target (y) рдХреЛ рдЕрд▓рдЧ рд╕реЗ рд╕реНрдХреЗрд▓ рдХрд┐рдпрд╛ рдерд╛, рд╣рдо рдЙрд╕реА scaler рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВрдЧреЗред***
    predicted_price = target_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]
    
    # 7. рдЖрдЬ рдХреЗ Close Price рд╕реЗ рддреБрд▓рдирд╛ рдХрд░реЗрдВ рдФрд░ BUY/SELL рд╕рд┐рдЧреНрдирд▓ рджреЗрдВ
    current_close = crypto_data['Close'].iloc[-1]
    
    print(f"\nЁЯУИ Today's Closing Price: {current_close:.2f} USD")
    print(f"ЁЯФо LSTM Predicted Close Price (Tomorrow): {predicted_price:.2f} USD")
    
    # 8. рдЕрдВрддрд┐рдо рдирд┐рд░реНрдгрдп
    if predicted_price > current_close:
        print("\nЁЯЪА **FINAL RQC SIGNAL: STRONG BUY / BULLISH** - рдХреАрдордд рдмрдврд╝рдиреЗ рдХреА рдЙрдореНрдореАрдж рд╣реИред")
    else:
        print("\nЁЯЫС **FINAL RQC SIGNAL: SELL / HOLD / BEARISH** - рдХреАрдордд рдШрдЯрдиреЗ рдпрд╛ рд░реБрдХрдиреЗ рдХреА рдЙрдореНрдореАрдж рд╣реИред")

except NameError as ne:
    print(f"\nтЭМ ERROR: Variable missing ({ne}). рдкреНрд▓реАрдЬ 'Run All' рдХрд░реЗрдВред")
except Exception as e:
    print(f"\nтЭМ GENERIC ERROR in LSTM Prediction: {e}")




# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ 'scaler' рдФрд░ 'sequence_length' рд╡реЗрд░рд┐рдПрдмрд▓ рдореЗрдореЛрд░реА рдореЗрдВ рд╣реИрдВред

print("\n--- ЁЯдЦ ADVANCED LSTM TRADING SIGNAL ---")

try:
    # 1. рдЖрдЬ рдХреЗ рд▓рд┐рдП рдЖрд╡рд╢реНрдпрдХ рдлреАрдЪрд░реНрд╕ рд▓реЗрдВ
    features = [
        'Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band'
    ]
    # рд╕рдмрд╕реЗ рдЕрдВрддрд┐рдо 60 рджрд┐рди рдХрд╛ рдбреЗрдЯрд╛ рд▓реЗрдВ (LSTM рд╕реАрдХреНрд╡реЗрдВрд╕ рд▓реЗрдВрде)
    last_60_days = crypto_data[features].iloc[-sequence_length:].values
    
    # 2. рдбреЗрдЯрд╛ рдХреЛ рд╕реНрдХреЗрд▓ рдХрд░реЗрдВ (рдЬрд╝рд░реВрд░реА!)
    # рдЯреНрд░реЗрдирд┐рдВрдЧ рдореЗрдВ рдЙрдкрдпреЛрдЧ рдХрд┐рдП рдЧрдП 'scaler' рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВ
    scaled_input = scaler.transform(last_60_days)
    
    # 3. рдбреЗрдЯрд╛ рдХреЛ LSTM рдХреЗ рд▓рд┐рдП 3D Format рдореЗрдВ рдмрджрд▓реЗрдВ
    X_predict = np.array([scaled_input])
    
    # 4. LSTM рд╕реЗ рдХрд▓ рдХреЗ Close Price рдХреА рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░реЗрдВ (Scaled Output)
    predicted_price_scaled = model_lstm.predict(X_predict)[0]
    
    # 5. Scaled Output рдХреЛ рдЕрд╕рд▓реА рдХреАрдордд рдореЗрдВ рдмрджрд▓реЗрдВ (Inverse Transform)
    # рд╣рдордиреЗ рд╕рд┐рд░реНрдл Close Price рдХреЛ Target рдорд╛рдирд╛ рдерд╛, рдЗрд╕рд▓рд┐рдП рд╣рдореЗрдВ рд╕рд┐рд░реНрдл Target Scaler рдЪрд╛рд╣рд┐рдП
    
    # [***ALERT***: рдЕрдЧрд░ 'target_scaler' рд╡реЗрд░рд┐рдПрдмрд▓ рдореМрдЬреВрдж рдирд╣реАрдВ рд╣реИ, рддреЛ рдПрдХ NameError рдЖ рд╕рдХрддрд╛ рд╣реИред 
    #  рдЕрдЧрд░ рдЖрддрд╛ рд╣реИ, рддреЛ рд╣рдо Target Price рдХреЛ Inverse Transform рдХрд░рдиреЗ рдХрд╛ рдПрдХ рд╕рд░рд▓ рддрд░реАрдХрд╛ рдЦреЛрдЬреЗрдВрдЧреЗред]
    
    # рдпрд╣рд╛рдБ рдорд╛рди рд▓реЗрддреЗ рд╣реИрдВ рдХрд┐ 'target_scaler' рдореМрдЬреВрдж рд╣реИ:
    
    # DUMMY SCALER FOR INVERSE TRANSFORM (If target_scaler is missing)
    # рдЕрдЧрд░ NameError рдЖрддрд╛ рд╣реИ, рддреЛ рдиреАрдЪреЗ рд╡рд╛рд▓рд╛ рдХреЛрдб рдЪрд▓рд╛рдПрдБ:
    # dummy_scaler = MinMaxScaler(feature_range=(0, 1))
    # dummy_scaler.fit(crypto_data['Close'].values.reshape(-1, 1))
    # predicted_price = dummy_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]
    
    # 6. Prediction рдХреЛ рдкреНрд░рд┐рдВрдЯ рдХрд░реЗрдВ
    print(f"\nЁЯФо LSTM Predicted Close Price (Tomorrow): {predicted_price:.2f} USD")
    
    # 7. рдЖрдЬ рдХреЗ Close Price рд╕реЗ рддреБрд▓рдирд╛ рдХрд░реЗрдВ рдФрд░ BUY/SELL рд╕рд┐рдЧреНрдирд▓ рджреЗрдВ
    current_close = crypto_data['Close'].iloc[-1]
    
    if predicted_price > current_close:
        print(f"ЁЯЪА AI SIGNAL: STRONG BUY (Predicted Price is higher than today's {current_close:.2f} USD)")
    else:
        print(f"ЁЯЫС AI SIGNAL: SELL / HOLD (Predicted Price is lower or equal to today's {current_close:.2f} USD)")

except Exception as e:
    print(f"тЭМ ERROR in LSTM Prediction: {e}")
    print("TIP: рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ LSTM рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА рдФрд░ рдЯреНрд░реЗрдирд┐рдВрдЧ рд╡рд╛рд▓реЗ рд╕рд╛рд░реЗ рд╕реЗрд▓реНрд╕ рдЪрд▓ рдЪреБрдХреЗ рд╣реИрдВред")
    


# 'epochs' рдЬрд┐рддрдиреА рдмрд╛рд░ рдореЙрдбрд▓ рдкреВрд░реЗ рдбреЗрдЯрд╛рд╕реЗрдЯ рдХреЛ рджреЗрдЦрддрд╛ рд╣реИред
# 'batch_size' рдПрдХ рдмрд╛рд░ рдореЗрдВ рджреЗрдЦреЗ рдЬрд╛рдиреЗ рд╡рд╛рд▓реЗ рдбреЗрдЯрд╛ рдкреЙрдЗрдВрдЯреНрд╕ рдХреА рд╕рдВрдЦреНрдпрд╛ред

print("Starting LSTM Model Training... тП│")

# Training the model
history = model_lstm.fit(
    X_lstm, 
    y_lstm, 
    epochs=25,       # рдЖрдк рдЗрд╕реЗ 50 рдпрд╛ 100 рддрдХ рдмрдврд╝рд╛ рд╕рдХрддреЗ рд╣реИрдВ (рд╕рдордп рд▓рдЧреЗрдЧрд╛)
    batch_size=32,   
    validation_split=0.1, # 10% рдбреЗрдЯрд╛ рдХреЛ рд╡реИрд▓рд┐рдбреЗрд╢рди рдХреЗ рд▓рд┐рдП рд░рдЦреЗрдВ
    verbose=1
)

print("\nLSTM Model Training Complete! ЁЯОЙ")


# 1. рдореЙрдбрд▓ рдмрдирд╛рдирд╛ (Sequential Model)
model_lstm = Sequential()

# рдкрд╣рд▓реА LSTM рд▓реЗрдпрд░ (60 рдпреВрдирд┐рдЯреНрд╕)
# input_shape: (sequence_length, number_of_features)
model_lstm.add(LSTM(units=60, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model_lstm.add(Dropout(0.2)) # 20% рдиреНрдпреВрд░реЙрдиреНрд╕ рдХреЛ рдмрдВрдж рдХрд░реЛ (Overfitting рд░реЛрдХрдиреЗ рдХреЗ рд▓рд┐рдП)

# рджреВрд╕рд░реА LSTM рд▓реЗрдпрд░
model_lstm.add(LSTM(units=60, return_sequences=False))
model_lstm.add(Dropout(0.2))

# рдЖрдЙрдЯрдкреБрдЯ рд▓реЗрдпрд░ (Dense Layer)
# output: 1 рдХреНрдпреЛрдВрдХрд┐ рд╣рдо 'Close Price' (рдПрдХ рд╣реА рд╡реИрд▓реНрдпреВ) рдХреА рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░ рд░рд╣реЗ рд╣реИрдВ
model_lstm.add(Dense(units=1)) 

# рдореЙрдбрд▓ рдХреЛ рдХрдВрдкрд╛рдЗрд▓ рдХрд░рдирд╛
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

print("LSTM Model Architecture Created! (Ready for Training) тЬЕ")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ 'crypto_data' DataFrame рдореМрдЬреВрдж рд╣реИ рдФрд░ рдЙрд╕рдореЗрдВ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рд╣реИрдВ

# 1. рдбреЗрдЯрд╛ рд╕рд╛рдлрд╝ рдХрд░реЗрдВ рдФрд░ рдЬрд╝рд░реВрд░реА рдлреАрдЪрд░реНрд╕ рд▓реЗрдВ
features = [
    'Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band'
]
lstm_data = crypto_data[features].copy()
lstm_data.dropna(inplace=True) 

# 2. рдбреЗрдЯрд╛ рдХреЛ 0 рдФрд░ 1 рдХреЗ рдмреАрдЪ рд╕реНрдХреЗрд▓ рдХрд░реЗрдВ (рдмрд╣реБрдд рдЬрд╝рд░реВрд░реА!)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(lstm_data.values)

# 3. Target (y) рдХреЛ рднреА рдЕрд▓рдЧ рд╕реЗ рд╕реНрдХреЗрд▓ рдХрд░реЗрдВ
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_data = target_scaler.fit_transform(lstm_data['Close'].values.reshape(-1, 1))
target_data = target_data[1:] # Target рдХреЛ рдПрдХ рджрд┐рди рдЖрдЧреЗ рд╢рд┐рдлреНрдЯ рдХрд░реЗрдВ

# 4. рд╕реАрдХреНрд╡реЗрдВрд╕ (X) рдФрд░ Target (y) рдбреЗрдЯрд╛ рддреИрдпрд╛рд░ рдХрд░реЗрдВ
X_lstm = []
y_lstm = []
sequence_length = 60  # рдкрд┐рдЫрд▓реЗ 60 рджрд┐рдиреЛрдВ рдХрд╛ рдбреЗрдЯрд╛ рджреЗрдЦреЛ

for i in range(sequence_length, len(scaled_data) - 1): # -1 рдХреНрдпреЛрдВрдХрд┐ рдЯрд╛рд░рдЧреЗрдЯ рдПрдХ рджрд┐рди рдЖрдЧреЗ рд╢рд┐рдлреНрдЯ рд╣реИ
    X_lstm.append(scaled_data[i - sequence_length:i, :])
    y_lstm.append(target_data[i - sequence_length])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

print(f"LSTM Data Shape (X): {X_lstm.shape}") 
print(f"LSTM Data Shape (y): {y_lstm.shape}")
print("Data Preparation Complete! тЬЕ")



# рдирдИ рд▓рд╛рдЗрдмреНрд░реЗрд░реАрдЬрд╝ (Keras/TensorFlow)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd

print("LSTM Libraries Imported! тЬЕ")



import numpy as np
import pandas as pd
# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ 'model' рд╡реЗрд░рд┐рдПрдмрд▓ рдКрдкрд░ рдХреЗ рд╕реЗрд▓ рдореЗрдВ рд╕рдлрд▓рддрд╛рдкреВрд░реНрд╡рдХ рдЯреНрд░реЗрди рд╣реЛ рдЪреБрдХрд╛ рд╣реИ

print("\n--- ЁЯдЦ AI TRADING SIGNAL GENERATOR ---")

try:
    # 1. Prediction рдХреЗ рд▓рд┐рдП рд╕рдмрд╕реЗ рдЕрдВрддрд┐рдо рдбреЗрдЯрд╛ рдкреЙрдЗрдВрдЯ (рдлреАрдЪрд░реНрд╕) рд▓реЗрдВ
    # рд╣рдо features рд▓рд┐рд╕реНрдЯ рдореЗрдВ рд╡реЛ рд╕рднреА рдХреЙрд▓рдо рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВрдЧреЗ рдЬреЛ рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдореЗрдВ рдЗрд╕реНрддреЗрдорд╛рд▓ рд╣реБрдП рдереЗ:
    features = [
        'Close', 
        'MA_50', 
        'RSI', 
        'MACD_Line', 
        'MACD_Histogram', 
        'Upper_Band', # BBands
        'Lower_Band'  # BBands
    ]
    
    # .iloc[-1] рд╕рдмрд╕реЗ рдЖрдЦрд┐рд░реА (рдЖрдЬ рдХрд╛) рд░реЛ рд▓реЗрддрд╛ рд╣реИ
    # .reshape(1, -1) рдЗрд╕реЗ рдореЙрдбрд▓ рдХреЗ рд▓рд┐рдП рд╕рд╣реА рдЖрдХрд╛рд░ (Format) рдореЗрдВ рд▓рд╛рддрд╛ рд╣реИ
    current_features = crypto_data[features].iloc[-1].values.reshape(1, -1) 

    # 2. рдореЙрдбрд▓ рд╕реЗ рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░реЗрдВ
    tomorrow_prediction = model.predict(current_features)[0]

    # 3. рд╕рд┐рдЧреНрдирд▓ рдкреНрд░рд┐рдВрдЯ рдХрд░реЗрдВ
    print("\nтЬЕ Prediction Complete! Based on the latest data:")

    if tomorrow_prediction == 1:
        print("ЁЯЪА AI SIGNAL: BUY / BULLISH - рдХреАрдордд рдмрдврд╝рдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ (GO LONG)!")
    else:
        print("ЁЯЫС AI SIGNAL: SELL / BEARISH - рдХреАрдордд рдШрдЯрдиреЗ рдпрд╛ рд░реБрдХрдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ (STAY SHORT/HOLD)!")
        
    # 4. рдпрд╣ рднреА рдмрддрд╛рдПрдВ рдХрд┐ рдбреЗрдЯрд╛ рдХрд┐рд╕ рджрд┐рди рдХрд╛ рд╣реИ
    latest_date = crypto_data.index[-1].strftime('%Y-%m-%d')
    print(f"\n[Note: This prediction uses data up to: {latest_date}]")

except Exception as e:
    print(f"тЭМ ERROR: рдореЙрдбрд▓ рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдирд╣реАрдВ рдХрд░ рд╕рдХрд╛ред рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдЖрдкрдиреЗ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдФрд░ рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдХреЛрдб рдЪрд▓рд╛ рджрд┐рдП рд╣реИрдВред рдПрд░рд░: {e}")




# рдЖрдЬ рдХрд╛ рдЕрдВрддрд┐рдо рдбреЗрдЯрд╛ рдкреЙрдЗрдВрдЯ (рдлреАрдЪрд░реНрд╕) рд▓реЗрдВ
current_features = crypto_data[features].iloc[-1].values.reshape(1, -1) 

# рдореЙрдбрд▓ рд╕реЗ рдХрд▓ рдХреА рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░реЗрдВ
tomorrow_prediction = model.predict(current_features)[0]

print("--- AI TRADING SIGNAL ---")
if tomorrow_prediction == 1:
    print("ЁЯЪА BULLISH SIGNAL (1): рдХрд▓ рдХреАрдордд рдмрдврд╝рдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ рд╣реИред рдЦрд░реАрджрдиреЗ рдкрд░ рд╡рд┐рдЪрд╛рд░ рдХрд░реЗрдВред")
else:
    print("ЁЯЫС BEARISH SIGNAL (0): рдХрд▓ рдХреАрдордд рдШрдЯрдиреЗ рдпрд╛ рд╕рд╛рдЗрдбрд╡реЗрдЬрд╝ рд░рд╣рдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ рд╣реИред рдмреЗрдЪрдиреЗ/рд░реБрдХрдиреЗ рдкрд░ рд╡рд┐рдЪрд╛рд░ рдХрд░реЗрдВред")
    


# рдЖрдЬ рдХрд╛ рдЕрдВрддрд┐рдо рдбреЗрдЯрд╛ рдкреЙрдЗрдВрдЯ (рдЬрд┐рд╕реЗ рдореЙрдбрд▓ рдиреЗ рдирд╣реАрдВ рджреЗрдЦрд╛ рд╣реИ) рд▓реЗрдВ
current_features = crypto_data[features].iloc[-1].values.reshape(1, -1) 

# рдореЙрдбрд▓ рд╕реЗ рднрд╡рд┐рд╖реНрдпрд╡рд╛рдгреА рдХрд░реЗрдВ
tomorrow_prediction = model.predict(current_features)[0]

if tomorrow_prediction == 1:
    print("AI Signal: ЁЯЪА рдХрд▓ рдХреАрдордд рдмрдврд╝рдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ (BULLISH)!")
else:
    print("AI Signal: ЁЯЫС рдХрд▓ рдХреАрдордд рдЧрд┐рд░рдиреЗ/рд░реБрдХрдиреЗ рдХреА рд╕рдВрднрд╛рд╡рдирд╛ (BEARISH)!")
    


# --- ЁЯОп Target рдмрдирд╛рдиреЗ рд╡рд╛рд▓рд╛ рдХреЛрдб (рдпрд╣ ML рдХреЛрдб рд╕реЗ рдкрд╣рд▓реЗ рдЪрд▓ рдЬрд╛рдирд╛ рдЪрд╛рд╣рд┐рдП!) ---
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)
crypto_data.dropna(inplace=True) 

# --- ЁЯЫая╕П рдЕрдм рдпрд╣ ML рдХреЛрдб рдЪрд▓рд╛рдПрдБ (Features рд▓рд┐рд╕реНрдЯ рдХреЛ рдЕрдкрдбреЗрдЯ рдХрд░рдиреЗ рдХреЗ рдмрд╛рдж) ---
features = [
    'Close', 
    'MA_50', 
    'RSI', 
    'MACD_Line', 
    'MACD_Histogram', 
    'Upper_Band', # <-- рдирдпрд╛ рдлреАрдЪрд░
    'Lower_Band'  # <-- рдирдпрд╛ рдлреАрдЪрд░
]
X = crypto_data[features].values
y = crypto_data['Target'].values 
# ... (рдмрд╛рдХреА рдХрд╛ рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдПрдХреНрдпреВрд░реЗрд╕реА рдХреЛрдб) ...



# рдпрд╣ рдХреЛрдб рд╕реЗрд▓ рджреЛрдмрд╛рд░рд╛ рдЪрд▓рд╛рдПрдБ (рдлреАрдЪрд░реНрд╕ рд▓рд┐рд╕реНрдЯ рдЕрдкрдбреЗрдЯ рдХрд░рдиреЗ рдХреЗ рдмрд╛рдж)

# 4. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ (features рд▓рд┐рд╕реНрдЯ рдЕрдм BBands рдХреЗ рд╕рд╛рде рдЕрдкрдбреЗрдЯ рд╣реЛрдиреА рдЪрд╛рд╣рд┐рдП)
features = [
    'Close', 
    'MA_50', 
    'RSI', 
    'MACD_Line', 
    'MACD_Histogram', 
    'Upper_Band', # <-- рдирдпрд╛ рдлреАрдЪрд░
    'Lower_Band'  # <-- рдирдпрд╛ рдлреАрдЪрд░
]
X = crypto_data[features].values
y = crypto_data['Target'].values

# 5. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("\n--- Final Result ---")
print("Random Forest Model Training Complete! тЬЕ")

# 6. рдореВрд▓реНрдпрд╛рдВрдХрди рдФрд░ рдкрд░рд┐рдгрд╛рдо
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



features = [
    'Close', 
    'MA_50', 
    'RSI', 
    'MACD_Line', 
    'MACD_Histogram', 
    'Upper_Band', # <-- рдирдпрд╛ рдлреАрдЪрд░
    'Lower_Band'  # <-- рдирдпрд╛ рдлреАрдЪрд░
]



features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдЖрдкрдХреЗ рдкрд╛рд╕ pandas рдФрд░ numpy рдЗрдореНрдкреЛрд░реНрдЯреЗрдб рд╣реИрдВ
# import pandas as pd
# import numpy as np

# --- Bollinger Bands (BBands) Calculation ---

# 1. 20-рджрд┐рди рдХреА Moving Average (MA) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
window = 20
crypto_data['BB_MA'] = crypto_data['Close'].rolling(window=window).mean()

# 2. Standard Deviation (STD) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
crypto_data['BB_STD'] = crypto_data['Close'].rolling(window=window).std()

# 3. Upper Band рдФрд░ Lower Band рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ (2 * STD)
# Upper Band: MA + (2 * STD)
crypto_data['Upper_Band'] = crypto_data['BB_MA'] + (crypto_data['BB_STD'] * 2)

# Lower Band: MA - (2 * STD)
crypto_data['Lower_Band'] = crypto_data['BB_MA'] - (crypto_data['BB_STD'] * 2)

print("Bollinger Bands (Upper/Lower) Calculated successfully! тЬЕ")



model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42) 



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдКрдкрд░ рдХреЗ рд╕рднреА рдЗрдВрдбрд┐рдХреЗрдЯрд░ (MA, RSI, MACD) рдмрди рдЪреБрдХреЗ рд╣реИрдВ!
# рдЕрдЧрд░ рдирд╣реАрдВ, рддреЛ рдКрдкрд░ рдХреЗ рдХреЛрдб рд╕реЗрд▓ рдЪрд▓рд╛рдПрдБ рдЬрд┐рд╕рдореЗрдВ рдЖрдкрдиреЗ рдЙрдиреНрд╣реЗрдВ рдмрдирд╛рдпрд╛ рдерд╛ред

# 1. рдбреЗрдЯрд╛ рдХреЛ рд╕рд╛рдлрд╝ рдХрд░реЗрдВ рдФрд░ рдЗрдВрдбреЗрдХреНрд╕ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ
crypto_data.dropna(inplace=True)
crypto_data.reset_index(drop=True, inplace=True) 

# 2. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
# рдпрд╣ рд╡рд╣ рд╕реЗрдХреНрд╢рди рд╣реИ рдЬрд╣рд╛рдБ рдПрд░рд░ рдЖ рд░рд╣рд╛ рдерд╛ред рд╣рдо рдЗрд╕реЗ рд╕реАрдзреЗ crypto_data рдкрд░ рд░рдЦреЗрдВрдЧреЗред
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# 3. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ (Target рдмрдирд╛рдиреЗ рд╕реЗ рдЖрдЦрд┐рд░ рдореЗрдВ рдПрдХ NaN рдЖрддрд╛ рд╣реИ)
crypto_data.dropna(inplace=True) 

# 4. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдХреЛ NumPy Arrays рдореЗрдВ рдмрджрд▓реЗрдВ (рдЕрдВрддрд┐рдо рдлрд┐рдХреНрд╕!)
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']

# .values рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рдХреЗ Pandas Index Alignment рдХреЛ рдкреВрд░реА рддрд░рд╣ рд╕реЗ рдмрд╛рдпрдкрд╛рд╕ рдХрд░реЗрдВред
X = crypto_data[features].values
y = crypto_data['Target'].values

# 5. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("\n--- Final Result ---")
print("Random Forest Model Training Complete! тЬЕ")

# 6. рдореВрд▓реНрдпрд╛рдВрдХрди рдФрд░ рдкрд░рд┐рдгрд╛рдо
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. NaN рд╣рдЯрд╛рдПрдБ (рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдмрдирдиреЗ рдХреЗ рдХрд╛рд░рдг)
# рд╣рдо рд╕реАрдзреЗ crypto_data рдХреЛ рд╕рд╛рдлрд╝ рдХрд░рддреЗ рд╣реИрдВ, рддрд╛рдХрд┐ рдХреЛрдИ рдХрдВрдлреНрдпреВрдЬрди рди рд░рд╣реЗред
crypto_data.dropna(inplace=True)

# 2. рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (Operands are not aligned рдПрд░рд░ рдХреЛ 100% рдареАрдХ рдХрд░рддрд╛ рд╣реИ)
# рдпрд╣ рд╕реНрдЯреЗрдк рд╕рдмрд╕реЗ рдЬрд╝рд░реВрд░реА рд╣реИ!
crypto_data.reset_index(drop=True, inplace=True) 

# 3. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# 4. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ (Target рдмрдирд╛рдиреЗ рд╕реЗ рдЖрдЦрд┐рд░ рдореЗрдВ рдПрдХ NaN рдЖрддрд╛ рд╣реИ)
crypto_data.dropna(inplace=True) 

# 5. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = crypto_data[features]
y = crypto_data['Target']

# 6. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("\n--- Final Result ---")
print("Random Forest Model Training Complete! тЬЕ")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- I. рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдмрдирд╛рдПрдБ (KeyError рдлрд┐рдХреНрд╕ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП) ---

# 1. Moving Averages (MA)
# 50-рджрд┐рди рдФрд░ 200-рджрд┐рди рдХреА рдореВрд╡рд┐рдВрдЧ рдПрд╡рд░реЗрдЬ
crypto_data['MA_50'] = crypto_data['Close'].rolling(window=50).mean()
crypto_data['MA_200'] = crypto_data['Close'].rolling(window=200).mean()
print("MA Calculated тЬЕ")

# 2. RSI (Relative Strength Index)
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

crypto_data = calculate_rsi(crypto_data)
print("RSI Calculated тЬЕ")

# 3. MACD (Moving Average Convergence Divergence)
exp12 = crypto_data['Close'].ewm(span=12, adjust=False).mean()
exp26 = crypto_data['Close'].ewm(span=26, adjust=False).mean()
crypto_data['MACD_Line'] = exp12 - exp26
crypto_data['Signal_Line'] = crypto_data['MACD_Line'].ewm(span=9, adjust=False).mean()
crypto_data['MACD_Histogram'] = crypto_data['MACD_Line'] - crypto_data['Signal_Line']
print("MACD Calculated тЬЕ")

# --- II. ML рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ (Value Error рдлрд┐рдХреНрд╕ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП) ---

# 4. рдЖрд╡рд╢реНрдпрдХ рдХреЙрд▓рдо рдХреА рдПрдХ рд╕рд╛рдл рдХреЙрдкреА рдмрдирд╛рдПрдВ (рдпрд╣ рдЗрдВрдбреЗрдХреНрд╕ рдПрд░рд░ рдХреЛ рд╣рдЯрд╛ рджреЗрдЧрд╛!)
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
ml_data = crypto_data[features].copy() 

# 5. NaN рд╣рдЯрд╛рдПрдБ (рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдмрдирдиреЗ рдХреЗ рдХрд╛рд░рдг)
ml_data.dropna(inplace=True)

# 6. рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (Operands are not aligned рдПрд░рд░ рдХреЛ рдареАрдХ рдХрд░рддрд╛ рд╣реИ)
ml_data.reset_index(drop=True, inplace=True) 

# 7. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
ml_data['Next_Close'] = ml_data['Close'].shift(-1)
ml_data['Target'] = np.where(ml_data['Next_Close'] > ml_data['Close'], 1, 0)

# 8. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ (Target рдмрдирд╛рдиреЗ рд╕реЗ рдЖрдЦрд┐рд░ рдореЗрдВ рдПрдХ NaN рдЖрддрд╛ рд╣реИ)
ml_data.dropna(inplace=True) 

# 9. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
X = ml_data[features]
y = ml_data['Target']

# 10. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("\n--- Final Result ---")
print("Random Forest Model Training Complete! тЬЕ")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# рдпрд╣ рдХреЛрдб ML рдХреЗ рд▓рд┐рдП рдбреЗрдЯрд╛ рдХреЛ рддреИрдпрд╛рд░ рдХрд░рддрд╛ рд╣реИ рдФрд░ рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░рддрд╛ рд╣реИ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдЖрд╡рд╢реНрдпрдХ рдХреЙрд▓рдо рдХреА рдПрдХ рд╕рд╛рдл рдХреЙрдкреА рдмрдирд╛рдПрдВ (рдЕрдм KeyErrors рдирд╣реАрдВ рдЖрдПрдВрдЧреЗ!)
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
ml_data = crypto_data[features].copy() 

# 2. NaN рд╣рдЯрд╛рдПрдБ 
ml_data.dropna(inplace=True)

# 3. рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (Operands are not aligned рдПрд░рд░ рдХреЛ рдареАрдХ рдХрд░рддрд╛ рд╣реИ)
ml_data.reset_index(drop=True, inplace=True) 

# 4. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
ml_data['Next_Close'] = ml_data['Close'].shift(-1)
ml_data['Target'] = np.where(ml_data['Next_Close'] > ml_data['Close'], 1, 0)

# 5. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ 
ml_data.dropna(inplace=True) 

# 6. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
X = ml_data[features]
y = ml_data['Target']

# 7. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдКрдкрд░ рдХреЗ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреЛрдб рдЪрд▓ рдЧрдП рд╣реИрдВ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдЖрд╡рд╢реНрдпрдХ рдХреЙрд▓рдо рдХреА рдПрдХ рд╕рд╛рдл рдХреЙрдкреА рдмрдирд╛рдПрдВ 
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
# рд╕рд┐рд░реНрдл рдЬрд╝рд░реВрд░реА рдлрд╝реАрдЪрд░ рд▓реЗрдВ
ml_data = crypto_data[features].copy() 

# 2. NaN рд╣рдЯрд╛рдПрдБ 
ml_data.dropna(inplace=True)

# 3. рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (рдпрд╣ Operands are not aligned рдПрд░рд░ рдХреЛ 100% рдареАрдХ рдХрд░реЗрдЧрд╛)
ml_data.reset_index(drop=True, inplace=True) 

# 4. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
ml_data['Next_Close'] = ml_data['Close'].shift(-1)
ml_data['Target'] = np.where(ml_data['Next_Close'] > ml_data['Close'], 1, 0)

# 5. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ 
ml_data.dropna(inplace=True) 

# 6. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
X = ml_data[features]
y = ml_data['Target']

# 7. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



import yfinance as yf
# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдЖрдкрдиреЗ рдпрд╣ рдХреЛрдб рдЪрд▓рд╛рдпрд╛ рд╣реИ
ticker_symbol = "BTC-USD"
crypto_data = yf.download(ticker_symbol, start="2022-01-01", end="2025-01-01")
print("BTC-USD Data Loaded successfully. тЬЕ")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдКрдкрд░ рдХреЗ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреЛрдб рдЪрд▓ рдЧрдП рд╣реИрдВ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдЖрд╡рд╢реНрдпрдХ рдХреЙрд▓рдо рдХреА рдПрдХ рд╕рд╛рдл рдХреЙрдкреА рдмрдирд╛рдПрдВ (рдпрд╣ рдЗрдВрдбреЗрдХреНрд╕ рдПрд░рд░ рдХреЛ рд╣рдЯрд╛ рджреЗрдЧрд╛!)
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
ml_data = crypto_data[features].copy() 

# 2. NaN рд╣рдЯрд╛рдПрдБ (рд╢реБрд░реБрдЖрддреА MA/RSI рдХреЗ рдХрд╛рд░рдг)
ml_data.dropna(inplace=True)

# 3. рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (рдпрд╣ Operands are not aligned рдПрд░рд░ рдХреЛ 100% рдареАрдХ рдХрд░реЗрдЧрд╛)
ml_data.reset_index(drop=True, inplace=True) 

# 4. рдЯрд╛рд░рдЧреЗрдЯ (Target) рдмрдирд╛рдПрдВ
# ml_data['Next_Close'] рдХреЙрд▓рдо рдореЗрдВ рдХрд▓ рдХрд╛ Close рдкреНрд░рд╛рдЗрд╕ рдЖрддрд╛ рд╣реИ
ml_data['Next_Close'] = ml_data['Close'].shift(-1)

# Target: рдЕрдЧрд░ рдХрд▓ рдХрд╛ рджрд╛рдо рдЖрдЬ рд╕реЗ рдЬрд╝реНрдпрд╛рджрд╛ рд╣реИ (1), рд╡рд░рдирд╛ 0
ml_data['Target'] = np.where(ml_data['Next_Close'] > ml_data['Close'], 1, 0)

# 5. рдЕрдВрддрд┐рдо NaN рд╣рдЯрд╛рдПрдБ (Target рдмрдирд╛рдиреЗ рд╕реЗ рдЖрдЦрд┐рд░ рдореЗрдВ рдПрдХ NaN рдЖрддрд╛ рд╣реИ)
ml_data.dropna(inplace=True) 

# 6. X рдФрд░ Y рдХреЛ рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
X = ml_data[features]
y = ml_data['Target']

# 7. рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



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























# Libraries рдЗрдореНрдкреЛрд░реНрдЯ рдХрд░реЗрдВ
import yfinance as yf          
import pandas as pd            
import matplotlib.pyplot as plt 
import numpy as np             
from sklearn.ensemble import RandomForestClassifier # ML рдХреЗ рд▓рд┐рдП
from sklearn.model_selection import train_test_split # ML рдХреЗ рд▓рд┐рдП
from sklearn.metrics import accuracy_score



# Ticker Symbol рддрдп рдХрд░реЗрдВ
ticker_symbol = "BTC-USD" 

# рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб рдХрд░реЗрдВ (3 рд╕рд╛рд▓ рдХрд╛ рдбреЗрдЯрд╛, рдХреНрдпреЛрдВрдХрд┐ рдХреНрд░рд┐рдкреНрдЯреЛ рдЬрд╝реНрдпрд╛рджрд╛ рдЕрд╕реНрдерд┐рд░ рд╣реИ)
crypto_data = yf.download(
    ticker_symbol,
    start="2022-01-01", 
    end="2025-01-01"  
)

# рдбреЗрдЯрд╛ рдХреА рдкрд╣рд▓реА 5 рдкрдВрдХреНрддрд┐рдпрд╛рдБ рджреЗрдЦреЗрдВ
print(crypto_data.head())



# Libraries рдЗрдореНрдкреЛрд░реНрдЯ рдХрд░реЗрдВ
import yfinance as yf          
import pandas as pd            
import matplotlib.pyplot as plt 
import numpy as np             
# ML libraries рдХреА рдЬрд╝рд░реВрд░рдд рдЕрднреА рдирд╣реАрдВ рд╣реИ, рд▓реЗрдХрд┐рди рд╣рдо рдЗрдиреНрд╣реЗрдВ рдмрд╛рдж рдореЗрдВ рдЗрд╕реНрддреЗрдорд╛рд▓ рдХрд░реЗрдВрдЧреЗ



# Ticker Symbol рддрдп рдХрд░реЗрдВ
ticker_symbol = "BTC-USD" 

# рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб рдХрд░реЗрдВ (3 рд╕рд╛рд▓ рдХрд╛ рдбреЗрдЯрд╛, рдХреНрдпреЛрдВрдХрд┐ рдХреНрд░рд┐рдкреНрдЯреЛ рдЬрд╝реНрдпрд╛рджрд╛ рдЕрд╕реНрдерд┐рд░ рд╣реИ)
# 'Close' рдкреНрд░рд╛рдЗрд╕ BTC рдХреЗ рджрд╛рдо рдХреЛ рджрд┐рдЦрд╛рдПрдЧрд╛
crypto_data = yf.download(
    ticker_symbol,
    start="2022-01-01", 
    end="2025-01-01"  
)

# рдбреЗрдЯрд╛ рдХреА рдкрд╣рд▓реА 5 рдкрдВрдХреНрддрд┐рдпрд╛рдБ рджреЗрдЦреЗрдВ
print(crypto_data.head())



# 14-рджрд┐рди рдХреА рдЕрд╡рдзрд┐ рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВ
period = 14

# 1. рджреИрдирд┐рдХ рдкрд░рд┐рд╡рд░реНрддрди (Daily Change) рдХреА рдЧрдгрдирд╛
crypto_data['Change'] = crypto_data['Close'].diff()

# 2. Gain (рдКрдкрд░ рдЬрд╛рдирд╛) рдФрд░ Loss (рдиреАрдЪреЗ рдЖрдирд╛) рдХреА рдЧрдгрдирд╛
crypto_data['Gain'] = np.where(crypto_data['Change'] > 0, crypto_data['Change'], 0)
crypto_data['Loss'] = np.where(crypto_data['Change'] < 0, abs(crypto_data['Change']), 0)

# 3. рдФрд╕рдд Gain рдФрд░ Loss рдХреА рдЧрдгрдирд╛ (Smoothed Moving Average)
crypto_data['Avg_Gain'] = crypto_data['Gain'].ewm(span=period, min_periods=period).mean()
crypto_data['Avg_Loss'] = crypto_data['Loss'].ewm(span=period, min_periods=period).mean()

# 4. RS (Relative Strength) рдХреА рдЧрдгрдирд╛
crypto_data['RS'] = crypto_data['Avg_Gain'] / crypto_data['Avg_Loss']

# 5. RSI рдХреА рдЧрдгрдирд╛
crypto_data['RSI'] = 100 - (100 / (1 + crypto_data['RS']))

# рдирдП RSI рдХреЙрд▓рдо рджреЗрдЦреЗрдВ
print("RSI Calculated:")
print(crypto_data[['Close', 'RSI']].tail())



# 50-рджрд┐рди рдФрд░ 200-рджрд┐рди рдХреА рдореВрд╡рд┐рдВрдЧ рдПрд╡рд░реЗрдЬ (рдХреНрд░рд┐рдкреНрдЯреЛ рдореЗрдВ 50 рдФрд░ 200 рджрд┐рди рдмрд╣реБрдд рдорд╛рдпрдиреЗ рд░рдЦрддреЗ рд╣реИрдВ)
crypto_data['MA_50'] = crypto_data['Close'].rolling(window=50).mean()
crypto_data['MA_200'] = crypto_data['Close'].rolling(window=200).mean()

print("\nMoving Averages Added:")
print(crypto_data[['Close', 'MA_50', 'MA_200']].tail())



# MACD рдХреЗ рд▓рд┐рдП 12, 26, 9 рдХреА рдЕрд╡рдзрд┐
short_window = 12
long_window = 26
signal_window = 9

# EMA рдХреА рдЧрдгрдирд╛
crypto_data['EMA_12'] = crypto_data['Close'].ewm(span=short_window, adjust=False).mean()
crypto_data['EMA_26'] = crypto_data['Close'].ewm(span=long_window, adjust=False).mean()

# MACD Line, Signal Line, рдФрд░ Histogram рдХреА рдЧрдгрдирд╛
crypto_data['MACD_Line'] = crypto_data['EMA_12'] - crypto_data['EMA_26']
crypto_data['Signal_Line'] = crypto_data['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
crypto_data['MACD_Histogram'] = crypto_data['MACD_Line'] - crypto_data['Signal_Line']

print("\nMACD Components Calculated:")
print(crypto_data[['MACD_Line', 'Signal_Line', 'MACD_Histogram']].tail())



# 3 рдкреНрд▓реЙрдЯ рд╡рд╛рд▓реА рдлрд┐рдЧрд░ рдмрдирд╛рдПрдВ (рдкреНрд░рд╛рдЗрд╕, RSI, рдФрд░ MACD рдХреЗ рд▓рд┐рдП)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 1. BTC рдкреНрд░рд╛рдЗрд╕ рдФрд░ MA рдЪрд╛рд░реНрдЯ (Upper Subplot)
crypto_data['Close'].plot(ax=ax1, label='BTC Price', color='blue') 
crypto_data['MA_50'].plot(ax=ax1, label='50-Day MA', color='orange', linestyle='--') 
crypto_data['MA_200'].plot(ax=ax1, label='200-Day MA', color='red', linestyle='--') 
ax1.set_title(f'{ticker_symbol} Price with Moving Averages (Trend Analysis)')
ax1.legend()
ax1.grid(True)

# 2. RSI рдЪрд╛рд░реНрдЯ (Middle Subplot)
crypto_data['RSI'].plot(ax=ax2, title='Relative Strength Index (RSI)', color='purple')
ax2.axhline(70, color='red', linestyle='--', linewidth=1) 
ax2.axhline(30, color='green', linestyle='--', linewidth=1) 
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True)

# 3. MACD рдЪрд╛рд░реНрдЯ (Lower Subplot)
crypto_data['MACD_Line'].plot(ax=ax3, label='MACD Line', color='blue')
crypto_data['Signal_Line'].plot(ax=ax3, label='Signal Line', color='red', linestyle='--')
ax3.bar(
    crypto_data.index, 
    crypto_data['MACD_Histogram'], 
    width=1, 
    color=['green' if val >= 0 else 'red' for val in crypto_data['MACD_Histogram']],
    label='Histogram'
)
ax3.axhline(0, color='grey', linestyle='-', linewidth=0.5)
ax3.set_title('MACD Indicator (Momentum)')
ax3.set_xlabel("Date")
ax3.legend()
ax3.grid(True)


plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd # рдпрд╣ рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░рддрд╛ рд╣реИ рдХрд┐ рд╕рдм рдХреБрдЫ рдЙрдкрд▓рдмреНрдз рд╣реЛ

# 1. рджреИрдирд┐рдХ рд░рд┐рдЯрд░реНрди (Daily Returns) рдХреА рдЧрдгрдирд╛
crypto_data['Daily_Returns'] = crypto_data['Close'].pct_change().dropna()

# 2. Volatility (рдЕрд╕реНрдерд┐рд░рддрд╛) рдХреА рдЧрдгрдирд╛ (рд╕рд╛рд▓рд╛рдирд╛ рдЖрдзрд╛рд░ рдкрд░ - 365 рджрд┐рди)
# рдХреНрд░рд┐рдкреНрдЯреЛ 24/7 рдЪрд▓рддрд╛ рд╣реИ, рдЗрд╕рд▓рд┐рдП рд╣рдо 252 рдХреА рдЬрдЧрд╣ 365 рджрд┐рди рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░рддреЗ рд╣реИрдВред
crypto_volatility = crypto_data['Daily_Returns'].std() * np.sqrt(365) 

# 3. VaR (Value at Risk) рдХреА рдЧрдгрдирд╛ (95% рдЖрддреНрдорд╡рд┐рд╢реНрд╡рд╛рд╕)
confidence_level = 0.05 
VaR_95 = crypto_data['Daily_Returns'].quantile(confidence_level) * 100

print("---------------------------------------")
print(f"BTC Volatility (рд╕рд╛рд▓рд╛рдирд╛): {crypto_volatility:.2f}")
print(f"рджреИрдирд┐рдХ VaR (95% рдЖрддреНрдорд╡рд┐рд╢реНрд╡рд╛рд╕): {abs(VaR_95):.2f}%")
print("---------------------------------------")



# 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# рдлрд╝реАрдЪрд░ рдЗрдВрдЬреАрдирд┐рдпрд░рд┐рдВрдЧ (рд╡рд╣реА рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдЬрд┐рдирдХрд╛ рдЙрдкрдпреЛрдЧ рд╕реНрдЯреЙрдХ рдореЗрдВ рдХрд┐рдпрд╛ рдерд╛)
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']

# NaN рд╣рдЯрд╛ рджреЗрдВ (RSI/MA рдХреА рд╢реБрд░реБрдЖрддреА NaN рд╡реИрд▓реНрдпреВ рдХреЛ рд╣рдЯрд╛рдирд╛ рдЬрд╝рд░реВрд░реА рд╣реИ)
crypto_data.dropna(inplace=True) 

X = crypto_data[features]
y = crypto_data['Target']



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Random Forest рдореЙрдбрд▓ (рдХреНрд░рд┐рдкреНрдЯреЛ рдХреЗ рд▓рд┐рдП n_estimators рдХреЛ 500 рдХрд░рддреЗ рд╣реИрдВ рддрд╛рдХрд┐ рдпрд╣ рдЬрд╝реНрдпрд╛рджрд╛ рд╕реАрдЦреЗ)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")



# рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# 1. 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# 2. NaN рд╣рдЯрд╛ рджреЗрдВ
# рдпрд╣ рдмрд╣реБрдд рдЬрд╝рд░реВрд░реА рд╣реИ рдХреНрдпреЛрдВрдХрд┐ MA, RSI, рдФрд░ Target рдмрдирд╛рдиреЗ рд╕реЗ NaN рдмрдирддреЗ рд╣реИрдВ
crypto_data.dropna(inplace=True) 

# 3. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = crypto_data[features]
y = crypto_data['Target']

# рдЕрдВрддрд┐рдо рдЖрдХрд╛рд░ рджреЗрдЦреЗрдВ (ML рд╢реБрд░реВ рдХрд░рдиреЗ рд╕реЗ рдкрд╣рд▓реЗ)
print(f"Features Data Shape (X): {X.shape}")
print(f"Target Data Shape (y): {y.shape}")



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Random Forest рдореЙрдбрд▓ (n_estimators=500, рддрд╛рдХрд┐ рдпрд╣ рдЬрд╝реНрдпрд╛рджрд╛ рд╕реАрдЦреЗ)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

# рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдЖрдкрдиреЗ рдпрд╣рд╛рдБ рд╕реЗ рдКрдкрд░ рд╡рд╛рд▓реЗ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреЛрдб рдЪрд▓рд╛ рджрд┐рдП рд╣реИрдВ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА
# 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# NaN рд╣рдЯрд╛ рджреЗрдВ (рдпрд╣ ML рдХреЗ рд▓рд┐рдП рдбреЗрдЯрд╛ рдХреЛ рд╕рд╛рдлрд╝ рдХрд░рддрд╛ рд╣реИ)
crypto_data.dropna(inplace=True) 

# 2. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = crypto_data[features]
y = crypto_data['Target']

# 3. рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 4. Random Forest рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

# 5. рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдЖрдкрдиреЗ рдпрд╣рд╛рдБ рд╕реЗ рдКрдкрд░ MA, RSI, рдФрд░ MACD рдХреЗ рд╕рд╛рд░реЗ рдХреЛрдб рдЪрд▓рд╛ рджрд┐рдП рд╣реИрдВ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# --- рдХреЛрдб рдмреНрд▓реЙрдХ 1: рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА ---
# 1. 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# 2. NaN рд╣рдЯрд╛ рджреЗрдВ (рдпрд╣ ML рдХреЗ рд▓рд┐рдП рдбреЗрдЯрд╛ рдХреЛ рд╕рд╛рдлрд╝ рдХрд░рддрд╛ рд╣реИ)
crypto_data.dropna(inplace=True) 

# 3. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = crypto_data[features]
y = crypto_data['Target']

# --- рдХреЛрдб рдмреНрд▓реЙрдХ 2: рдЯреНрд░реЗрдирд┐рдВрдЧ рдФрд░ рдореВрд▓реНрдпрд╛рдВрдХрди ---
# 4. рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 5. Random Forest рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

# 6. рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



# рд╕реБрдирд┐рд╢реНрдЪрд┐рдд рдХрд░реЗрдВ рдХрд┐ рдКрдкрд░ рдХреЗ рд╕рд╛рд░реЗ рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреЛрдб рдЪрд▓ рдЧрдП рд╣реИрдВ
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА
# 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
crypto_data['Next_Close'] = crypto_data['Close'].shift(-1)
crypto_data['Target'] = np.where(crypto_data['Next_Close'] > crypto_data['Close'], 1, 0)

# NaN рд╣рдЯрд╛ рджреЗрдВ (рдпрд╣ ML рдХреЗ рд▓рд┐рдП рдбреЗрдЯрд╛ рдХреЛ рд╕рд╛рдлрд╝ рдХрд░рддрд╛ рд╣реИ)
# рд╣рдо рдЗрд╕ рдмрд╛рд░ рдбреНрд░реЙрдкрдирд╛ рдХреЛ рдлрд┐рд░ рд╕реЗ рдЪрд▓рд╛рддреЗ рд╣реИрдВ, рддрд╛рдХрд┐ Target рдмрдирд╛рдиреЗ рдХреЗ рдХрд╛рд░рдг рдЖрдП NaN рд╣рдЯ рдЬрд╛рдПрдБ
crypto_data.dropna(inplace=True) 

# 2. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = crypto_data[features]
y = crypto_data['Target']

# 3. рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 4. Random Forest рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

# 5. рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА
# 'Next_Close' рдФрд░ 'Target' рдХреЙрд▓рдо рдмрдирд╛рдПрдВ
# .copy() рдХрд╛ рдЙрдкрдпреЛрдЧ рдХрд░реЗрдВ рддрд╛рдХрд┐ рдХреЛрдИ рдкреБрд░рд╛рдирд╛ рдЗрдВрдбреЗрдХреНрд╕ рдЗрд╢реНрдпреВ рди рд░рд╣реЗ
ml_data = crypto_data[['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']].copy()

ml_data['Next_Close'] = ml_data['Close'].shift(-1)
# Target: рдЕрдЧрд░ рдЕрдЧрд▓рд╛ Close, рдЖрдЬ рдХреЗ Close рд╕реЗ рдЬрд╝реНрдпрд╛рджрд╛ рд╣реИ, рддреЛ 1 (рдКрдкрд░ рдЬрд╛рдПрдЧрд╛), рд╡рд░рдирд╛ 0
ml_data['Target'] = np.where(ml_data['Next_Close'] > ml_data['Close'], 1, 0)

# NaN рд╣рдЯрд╛ рджреЗрдВ рдФрд░ рдЗрдВрдбреЗрдХреНрд╕ рдХреЛ рд░реАрд╕реЗрдЯ рдХрд░реЗрдВ (рд╕рдмрд╕реЗ рдЬрд╝рд░реВрд░реА!)
ml_data.dropna(inplace=True) 
ml_data.reset_index(drop=True, inplace=True) # <- рдпрд╣ рдЗрдВрдбреЗрдХреНрд╕ рдПрд░рд░ рдХреЛ рд╣рдЯрд╛ рджреЗрдЧрд╛

# 2. рдлрд╝реАрдЪрд░ (X) рдФрд░ рдЯрд╛рд░рдЧреЗрдЯ (y) рдкрд░рд┐рднрд╛рд╖рд┐рдд рдХрд░реЗрдВ
features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram']
X = ml_data[features]
y = ml_data['Target']

# 3. рдбреЗрдЯрд╛ рдХреЛ рдЯреНрд░реЗрдирд┐рдВрдЧ (80%) рдФрд░ рдЯреЗрд╕реНрдЯрд┐рдВрдЧ (20%) рдореЗрдВ рд╡рд┐рднрд╛рдЬрд┐рдд рдХрд░реЗрдВ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 4. Random Forest рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

print("Random Forest Model Training Complete! тЬЕ")

# 5. рд╕рдЯреАрдХрддрд╛ (Accuracy) рдХреА рдЧрдгрдирд╛ рдХрд░реЗрдВ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBTC Model Accuracy on Test Data: {accuracy * 100:.2f}%")





















driver_name = "Ravi"
profit_today = 159.75
is_winner = True

print(driver_name)
print(profit_today)
print(is_winner)



buy = 200
sell = 350

profit = sell - buy
percent = (profit / buy) * 100

print("Profit:", profit)
print("Profit %:", percent)

name = "Ravi"
print(name + " completed Step 4 successfully!")



coins = ["BTC", "ETH", "SOL"]
prices = [35000, 1800, 150]

print("First coin:", coins[0])
print("Second price:", prices[1])

coins.append("DOGE")
prices.append(0.12)

print("Updated coins:", coins)
print("Updated prices:", prices)



coins = ["BTC", "ETH", "SOL", "XRP"]

print("Listing coins:")
for coin in coins:
    print("Coin:", coin)

print("\nPrices with 10% profit:")
prices = [35000, 1800, 150, 0.56]

for p in prices:
    new_price = p + (p * 0.10)
    print(new_price)
    


# 1. Simple function
def hello():
    print("Learning Python with Bhai!")

hello()


# 2. Function with parameter
def profit(amount):
    return amount * 0.25   # 25% profit

print(profit(100))


# 3. Function with two parameters
def total_profit(amount, percentage):
    return amount + (amount * percentage / 100)

print(total_profit(2000, 15))  # 15% profit



# 1. Create a list
numbers = [5, 10, 15, 20]
print(numbers)

# 2. Access items
print(numbers[0]) 
print(numbers[2])

# 3. Change a value
numbers[1] = 50
print(numbers)

# 4. Add a new value
numbers.append(100)
print(numbers)

# 5. Mixed data list
profile = ["Ravi", "RQC", 1, True]
print(profile)






# 1. Create dictionary
crypto = {
    "BTC": 35000,
    "ETH": 1800,
    "SOL": 150
}

# 2. Access value
print("Price of ETH:", crypto["ETH"])

# 3. Update / Add new item
crypto["DOGE"] = 0.12
crypto["BTC"] = 36000

# 4. Loop through dictionary
for coin in crypto:
    print(coin, "->", crypto[coin])
    


# --- рдирдпрд╛ app.py Code (рдЯреНрд░реЗрдирд┐рдВрдЧ рд╕рд╣рд┐рдд) ---
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import talib
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- 1. рдХрд╛рдВрд╕реНрдЯреЗрдВрдЯреНрд╕ ---
SEQUENCE_LENGTH = 60 
TICKER = "BTC-USD"
EPOCHS = 25 # рдЯреНрд░реЗрдирд┐рдВрдЧ рдХреЗ рд▓рд┐рдП рдХрдо рд░рдЦреЗрдВ
BATCH_SIZE = 32

st.set_page_config(page_title="RQC AI Trading Signal", layout="wide")

# --- 2. рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб рдФрд░ рдкреНрд░реЛрд╕реЗрд╕рд┐рдВрдЧ ---
@st.cache_data
def get_processed_data():
    # рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб (рдкрд┐рдЫрд▓реЗ 2 рд╕рд╛рд▓ рдХрд╛)
    end_date = date.today()
    start_date = end_date - timedelta(days=730) 
    data = yf.download(TICKER, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    # рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреИрд▓рдХреБрд▓реЗрдЯ рдХрд░реЗрдВ
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD_Line'], _, data['MACD_Histogram'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    upper, _, lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['Upper_Band'] = upper
    data['Lower_Band'] = lower
    data.dropna(inplace=True)
    
    # LSTM рдбреЗрдЯрд╛ рддреИрдпрд╛рд░реА
    features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band']
    lstm_data = data[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler.fit_transform(lstm_data)
    
    # Target (Close price) рдХреЛ рдЕрд▓рдЧ рд╕реЗ рд╕реНрдХреЗрд▓ рдХрд░реЗрдВ
    target_price = data['Close'].values.reshape(-1, 1)
    scaled_target = target_scaler.fit_transform(target_price)
    
    X_lstm, y_lstm = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        X_lstm.append(scaled_features[i-SEQUENCE_LENGTH:i])
        y_lstm.append(scaled_target[i])
        
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    return X_lstm, y_lstm, scaler, target_scaler, data.iloc[-1]['Close']


# --- 3. рдореЙрдбрд▓ рдмрдирд╛рдирд╛ рдФрд░ рдЯреНрд░реЗрди рдХрд░рдирд╛ ---
@st.cache_resource
def get_trained_model():
    X_lstm, y_lstm, scaler, target_scaler, current_close = get_processed_data()

    # рдореЙрдбрд▓ рдЖрд░реНрдХрд┐рдЯреЗрдХреНрдЪрд░
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=60, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=60, return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=60))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=1))
    
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # рдореЙрдбрд▓ рдХреЛ рдЯреНрд░реЗрди рдХрд░реЗрдВ
    with st.spinner('AI Model is Training... Please Wait (5-10 mins)'):
        model_lstm.fit(X_lstm, y_lstm, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    return model_lstm, scaler, target_scaler, current_close

# --- 4. рдкреНрд░реЗрдбрд┐рдХреНрд╢рди рдлрдВрдХреНрд╢рди ---
def calculate_indicators(data):
    # рдпрд╣ рд╕рд┐рд░реНрдл yf.download() рдХреЗ рдмрд╛рдж рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреИрд▓рдХреБрд▓реЗрдЯ рдХрд░рдиреЗ рдХреЗ рд▓рд┐рдП рд╣реИ
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD_Line'], _, data['MACD_Histogram'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    upper, _, lower = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['Upper_Band'] = upper
    data['Lower_Band'] = lower
    return data.dropna()


def get_ai_signal():
    model, scaler, target_scaler, current_close = get_trained_model()
    
    # рдирдпрд╛ рдбреЗрдЯрд╛ рдбрд╛рдЙрдирд▓реЛрдб рдХрд░реЗрдВ (рдкреНрд░реЗрдбрд┐рдХреНрд╢рди рдХреЗ рд▓рд┐рдП)
    end_date = date.today()
    start_date = end_date - timedelta(days=SEQUENCE_LENGTH + 100)
    data = yf.download(TICKER, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

    # рдЗрдВрдбрд┐рдХреЗрдЯрд░ рдХреИрд▓рдХреБрд▓реЗрдЯ рдХрд░реЗрдВ
    data = calculate_indicators(data.copy())
    
    features = ['Close', 'MA_50', 'RSI', 'MACD_Line', 'MACD_Histogram', 'Upper_Band', 'Lower_Band']
    
    if len(data) < SEQUENCE_LENGTH:
        return "ERROR", 0.00, 0.00
        
    last_60_days = data[features].iloc[-SEQUENCE_LENGTH:].values
    scaled_input = scaler.transform(last_60_days)
    X_predict = np.array([scaled_input])
    
    # рдкреНрд░реЗрдбрд┐рдХреНрд╢рди
    predicted_price_scaled = model.predict(X_predict, verbose=0)[0]
    predicted_price = target_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]
    
    # рд╕рд┐рдЧреНрдирд▓
    if predicted_price > current_close:
        signal = "BUY"
    else:
        signal = "SELL/HOLD"
        
    return signal, predicted_price, current_close

# --- 5. App Interface ---
st.title("тЪб RQC AI Machine: BTC Prediction")
st.subheader("Deep Learning Model (Trained on the fly)")

# App рдЪрд▓рд╛рдиреЗ рд╕реЗ рдкрд╣рд▓реЗ рдореЙрдбрд▓ рдЯреНрд░реЗрдирд┐рдВрдЧ рдХрд╛ рдЗрдВрддрдЬрд╛рд░
if 'trained' not in st.session_state:
    st.session_state.trained = False
    
model, _, _, _ = get_trained_model()
st.session_state.trained = True

if st.session_state.trained:
    signal, predicted_price, current_close = get_ai_signal()

    if signal == "BUY":
        st.markdown(f"<div style='background-color:#00e676; padding: 20px; border-radius: 10px;'><h1>ЁЯЪА FINAL RQC SIGNAL: STRONG BUY</h1></div>", unsafe_allow_html=True)
    elif signal == "SELL/HOLD":
        st.markdown(f"<div style='background-color:#ff1744; padding: 20px; border-radius: 10px;'><h1>ЁЯЫС FINAL RQC SIGNAL: SELL / HOLD</h1></div>", unsafe_allow_html=True)
    else:
        st.warning("AI Signal Unavailable. Model training failed.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Today's Close Price (USD)", f"${current_close:,.2f}")

    with col2:
        st.metric("Predicted Close Price (Tomorrow)", f"${predicted_price:,.2f}")

    with col3:
        st.metric("Target Accuracy Goal", "70%+", delta="LSTM Goal")
    
st.markdown("---")
st.caption("Disclaimer: This tool provides algorithmic signals and is not financial advice.")

# End of app.py code
"""
# app.py рдлрд╝рд╛рдЗрд▓ рдХреЛ Kaggle Output рдореЗрдВ рд╕реЗрд╡ рдХрд░реЗрдВ
with open('app.py', 'w') as f:
    f.write(app_code)

# requirements.txt рдлрд╝рд╛рдЗрд▓ рдХреЛ рднреА Kaggle Output рдореЗрдВ рд╕реЗрд╡ рдХрд░реЗрдВ
req_code = """
pandas
numpy
yfinance
tensorflow==2.15.0
keras==2.15.0
streamlit==1.31.1
scikit-learn
talib-python==0.4.19
"""
with open('requirements.txt', 'w') as f:
    f.write(req_code)

print("тЬЕ New 'app.py' and 'requirements.txt' created in Kaggle Output!")


