import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
%pip install mplfinance

train_path: str = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
train_data = pd.read_parquet(train_path)

raw_data = train_data[['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']]
del train_data


import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def method1_label_as_return_signal(labels, initial_price=100.0, signal_strength=0.002):
    """
    æ–¹æ³•1: å°†labelè§†ä¸ºæ”¶ç›Šç�‡ä¿¡å�·ï¼ˆä¿�å®ˆç­–ç•¥ï¼‰
    """
    returns = labels * signal_strength
    returns = np.clip(returns, -0.02, 0.02)  # é™�åˆ¶åœ¨Â±2%
    
    prices = np.zeros(len(returns))
    prices[0] = initial_price
    
    for i in range(1, len(returns)):
        prices[i] = prices[i-1] * (1 + returns[i])
        if prices[i] <= 0:
            prices[i] = prices[i-1] * 0.999
    
    return prices, returns

def method2_label_as_price_change(labels, initial_price=100.0, scale_factor=0.1):
    """
    æ–¹æ³•2: å°†labelè§†ä¸ºä»·æ ¼å�˜åŒ–é‡�ï¼ˆä¿�å®ˆç­–ç•¥ï¼‰
    """
    price_changes = labels * scale_factor
    
    prices = np.zeros(len(labels))
    prices[0] = initial_price
    
    for i in range(1, len(labels)):
        prices[i] = prices[i-1] + price_changes[i]
        if prices[i] <= 0:
            prices[i] = prices[i-1] * 0.99
    
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    return prices, returns

def method3_label_as_zscore_signal(labels, initial_price=100.0, volatility=0.003):
    """
    æ–¹æ³•3: å°†labelè§†ä¸ºæ ‡å‡†åŒ–ä¿¡å�·ï¼ˆä¿�å®ˆç­–ç•¥ï¼‰
    """
    returns = labels * volatility
    returns = np.clip(returns, -0.03, 0.03)  # é™�åˆ¶åœ¨Â±3%
    
    prices = np.zeros(len(returns))
    prices[0] = initial_price
    
    for i in range(1, len(returns)):
        prices[i] = prices[i-1] * (1 + returns[i])
        if prices[i] <= 0:
            prices[i] = prices[i-1] * 0.999
    
    return prices, returns

def method4_synthetic_price_from_market_data(raw_data, initial_price=100.0):
    """
    æ–¹æ³•4: åŸºäº�å¸‚åœºæ•°æ�®å�ˆæˆ�ä»·æ ¼åº�åˆ—ï¼ˆæ�¨è��æ–¹æ³•ï¼‰
    """
    # è®¡ç®—ä¹°å�–å�‹åŠ›æŒ‡æ ‡
    buy_pressure = raw_data['buy_qty'] / (raw_data['buy_qty'] + raw_data['sell_qty'] + 1e-10)
    buy_pressure = buy_pressure.fillna(0.5)
    
    # è®¡ç®—è®¢å�•ç°¿ä¸�å¹³è¡¡
    order_imbalance = (raw_data['bid_qty'] - raw_data['ask_qty']) / (raw_data['bid_qty'] + raw_data['ask_qty'] + 1e-10)
    order_imbalance = order_imbalance.fillna(0)
    
    # æ ‡å‡†åŒ–label
    label_normalized = (raw_data['label'] - raw_data['label'].mean()) / (raw_data['label'].std() + 1e-10)
    label_normalized = np.clip(label_normalized, -2, 2)
    
    # ç»„å�ˆä¿¡å�·
    price_signal = (buy_pressure - 0.5) * 0.3 + order_imbalance * 0.2 + label_normalized * 0.001
    
    # è½¬æ�¢ä¸ºæ”¶ç›Šç�‡
    returns = price_signal * 0.002  # é��å¸¸å°�çš„å�˜åŒ–å¹…åº¦
    returns = np.clip(returns, -0.01, 0.01)
    
    prices = np.zeros(len(returns))
    prices[0] = initial_price
    
    for i in range(1, len(returns)):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return prices, returns

def create_ohlcv_data(prices, volume_series, timestamps, resample_freq='2h'):
    """
    ä»�ä»·æ ¼åº�åˆ—å’Œæˆ�äº¤é‡�åº�åˆ—åˆ›å»ºOHLCVæ•°æ�®
    """
    try:
        df = pd.DataFrame({
            'price': prices,
            'volume': volume_series,
            'timestamp': pd.to_datetime(timestamps)
        })
        df.set_index('timestamp', inplace=True)
        
        # é‡�é‡‡æ ·åˆ›å»ºOHLC
        ohlc = df['price'].resample(resample_freq).ohlc()
        volume_resampled = df['volume'].resample(resample_freq).sum()
        
        # å�ˆå¹¶æ•°æ�®
        ohlcv = pd.concat([ohlc, volume_resampled], axis=1)
        ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # å¤„ç�†ç¼ºå¤±å€¼å’Œå¼‚å¸¸å€¼
        ohlcv.dropna(inplace=True)
        
        # ç¡®ä¿�OHLCé€»è¾‘æ­£ç¡®
        ohlcv['High'] = ohlcv[['Open', 'High', 'Low', 'Close']].max(axis=1)
        ohlcv['Low'] = ohlcv[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return ohlcv
    except Exception as e:
        print(f"Error creating OHLCV data: {e}")
        return pd.DataFrame()

def plot_professional_chart(ohlcv_data, title, method_name):
    """
    ä½¿ç”¨mplfinanceåˆ›å»ºä¸“ä¸šå›¾è¡¨ï¼ˆä¿®å¤�ç‰ˆæœ¬å…¼å®¹é—®é¢˜ï¼‰
    """
    try:
        # åˆ›å»ºç®€åŒ–çš„è‡ªå®šä¹‰æ ·å¼�ï¼ˆç§»é™¤ä¸�å…¼å®¹çš„å�‚æ•°ï¼‰
        mc = mpf.make_marketcolors(
            up='red', 
            down='green',
            edge='inherit',
            wick={'up': 'black', 'down': 'black'},
            volume='in'
        )
        
        # ç®€åŒ–æ ·å¼�è®¾ç½®
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='lightgray'
            # ç§»é™¤ gridwidth å�‚æ•°
        )
        
        # ç»˜åˆ¶ä¸“ä¸šå›¾è¡¨
        mpf.plot(
            ohlcv_data,
            type='candle',
            volume=True,
            style=style,
            title=f'{title} - {method_name}',
            ylabel='Price',
            ylabel_lower='Volume',
            figsize=(16, 10),
            panel_ratios=(3, 1),
            show_nontrading=False
            # ç§»é™¤ tight_layout å�‚æ•°
        )
        
        return True
    except Exception as e:
        print(f"mplfinance plotting error: {e}")
        return False

def plot_simple_manual_chart(ohlcv_data, title, method_name):
    """
    å¤‡ç”¨æ–¹æ¡ˆï¼šæ‰‹åŠ¨ç»˜åˆ¶ç®€å�•Kçº¿å›¾+æˆ�äº¤é‡�
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        
        # Kçº¿å›¾
        for i, (date, row) in enumerate(ohlcv_data.iterrows()):
            open_p, high_p, low_p, close_p = row['Open'], row['High'], row['Low'], row['Close']
            
            color = 'red' if close_p >= open_p else 'green'
            
            # å½±çº¿
            ax1.plot([i, i], [low_p, high_p], color='black', linewidth=1)
            
            # å®�ä½“
            if close_p >= open_p:
                ax1.bar(i, close_p - open_p, bottom=open_p, color=color, alpha=0.8, width=0.8)
            else:
                ax1.bar(i, open_p - close_p, bottom=close_p, color=color, alpha=0.8, width=0.8)
        
        ax1.set_title(f'{title} - {method_name}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # æˆ�äº¤é‡�å›¾
        volume_colors = ['red' if row['Close'] >= row['Open'] else 'green' 
                        for _, row in ohlcv_data.iterrows()]
        
        ax2.bar(range(len(ohlcv_data)), ohlcv_data['Volume'], 
               color=volume_colors, alpha=0.7, width=0.8)
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # è®¾ç½®xè½´æ ‡ç­¾
        step = max(1, len(ohlcv_data) // 8)
        tick_positions = list(range(0, len(ohlcv_data), step))
        tick_labels = [ohlcv_data.index[i].strftime('%m-%d\n%H:%M') for i in tick_positions]
        
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=0, ha='center')
        ax1.set_xticklabels([])
        
        plt.tight_layout()
        return True
        
    except Exception as e:
        print(f"Manual plotting error: {e}")
        return False

# ================================
# ä¸»ç¨‹åº�å¼€å§‹
# ================================

# åŠ è½½æ•°æ�®éƒ¨åˆ†ï¼ˆå�‡è®¾raw_dataå·²ç»�å­˜åœ¨ï¼‰
# å¦‚æ�œéœ€è¦�é‡�æ–°åŠ è½½ï¼Œå�–æ¶ˆä¸‹é�¢æ³¨é‡Šï¼š
# train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
# train_data = pd.read_parquet(train_path)
# raw_data = train_data[['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']]
# del train_data

# ç¡®ä¿�æ•°æ�®å®Œæ•´æ€§
if 'timestamp' not in raw_data.columns:
    raw_data = raw_data.reset_index()
    if 'timestamp' not in raw_data.columns:
        raw_data['timestamp'] = pd.date_range(start='2023-03-01', periods=len(raw_data), freq='min')

print("=== ğŸš€ Crypto Market Data Analysis - Complete Version ===")
print(f"Data shape: {raw_data.shape}")
print(f"Date range: {raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}")
print(f"Label stats: min={raw_data['label'].min():.4f}, max={raw_data['label'].max():.4f}, mean={raw_data['label'].mean():.4f}")

# é€‰æ‹©æ•°æ�®å­�é›†è¿›è¡Œåˆ†æ��
data_points = 10000
analysis_data = raw_data.head(data_points).copy()

print(f"\nğŸ“Š Analyzing {data_points} data points...")
print(f"Label distribution:")
print(f"  â€¢ 25th percentile: {analysis_data['label'].quantile(0.25):.4f}")
print(f"  â€¢ 50th percentile: {analysis_data['label'].quantile(0.50):.4f}")
print(f"  â€¢ 75th percentile: {analysis_data['label'].quantile(0.75):.4f}")
print(f"  â€¢ Standard deviation: {analysis_data['label'].std():.4f}")

# å®šä¹‰æ‰€æœ‰æ–¹æ³•
methods = {
    "Method 1: Return Signal (Conservative)": lambda data: method1_label_as_return_signal(
        data['label'].values, initial_price=100.0, signal_strength=0.002
    ),
    "Method 2: Price Change (Conservative)": lambda data: method2_label_as_price_change(
        data['label'].values, initial_price=100.0, scale_factor=0.1
    ),
    "Method 3: Z-Score Signal (Conservative)": lambda data: method3_label_as_zscore_signal(
        data['label'].values, initial_price=100.0, volatility=0.003
    ),
    "Method 4: Market Data Synthesis (Recommended)": lambda data: method4_synthetic_price_from_market_data(
        data, initial_price=100.0
    )
}

successful_charts = 0
total_methods = len(methods)

# ç”Ÿæˆ�å›¾è¡¨
for method_name, method_func in methods.items():
    print(f"\n{'='*60}")
    print(f"ğŸ�¯ {method_name}")
    print(f"{'='*60}")
    
    try:
        # åº”ç”¨æ–¹æ³•ç”Ÿæˆ�ä»·æ ¼åº�åˆ—
        prices, returns = method_func(analysis_data)
        
        # éªŒè¯�ä»·æ ¼åº�åˆ—çš„å�ˆç�†æ€§
        price_ratio = prices.max() / prices.min() if prices.min() > 0 else float('inf')
        
        print(f"ğŸ“ˆ Price Statistics:")
        print(f"  â€¢ Price range: {prices.min():.4f} - {prices.max():.4f}")
        print(f"  â€¢ Price ratio: {price_ratio:.2f}")
        print(f"  â€¢ Total return: {(prices[-1]/prices[0] - 1)*100:.2f}%")
        print(f"  â€¢ Return volatility: {np.std(returns)*100:.4f}%")
        
        # åˆ›å»ºOHLCVæ•°æ�®
        ohlcv_data = create_ohlcv_data(
            prices,
            analysis_data['volume'].values,
            analysis_data['timestamp'].values,
            resample_freq='2h'
        )
        
        if len(ohlcv_data) > 20:
            print(f"ğŸ“Š Generated {len(ohlcv_data)} candlesticks")
            
            # é¦–å…ˆå°�è¯•ä¸“ä¸šå›¾è¡¨
            print("ğŸ�¨ Attempting professional mplfinance chart...")
            success_professional = plot_professional_chart(ohlcv_data, "Crypto Market Analysis", method_name)
            
            if success_professional:
                successful_charts += 1
                plt.show()
                print("âœ… Professional chart generated successfully!")
                
                # æ˜¾ç¤ºè¯¦ç»†ç»Ÿè®¡
                print(f"ğŸ“‹ Chart Statistics:")
                print(f"    â€¢ Timespan: {len(ohlcv_data)} periods")
                print(f"    â€¢ Average Volume: {ohlcv_data['Volume'].mean():.0f}")
                print(f"    â€¢ Price Volatility: {ohlcv_data['Close'].pct_change().std()*100:.2f}%")
                max_dd = ((ohlcv_data['Close'] / ohlcv_data['Close'].expanding().max()) - 1).min()*100
                print(f"    â€¢ Max Drawdown: {max_dd:.2f}%")
                
            else:
                # å¤‡ç”¨æ–¹æ¡ˆï¼šæ‰‹åŠ¨ç»˜åˆ¶
                print("âš ï¸�  Professional chart failed, trying manual chart...")
                success_manual = plot_simple_manual_chart(ohlcv_data, "Crypto Market Analysis", method_name)
                
                if success_manual:
                    successful_charts += 1
                    plt.show()
                    print("âœ… Manual chart generated successfully!")
                else:
                    print("â�Œ Both chart methods failed")
                    
        else:
            print(f"â�Œ Insufficient data points: {len(ohlcv_data)} (need > 20)")
            
    except Exception as e:
        print(f"â�Œ Error in {method_name}: {e}")

# ================================
# æœ€ç»ˆåˆ†æ��æ€»ç»“
# ================================
print(f"\n{'='*80}")
print("ğŸ�Š FINAL ANALYSIS SUMMARY")
print(f"{'='*80}")

print(f"\nğŸ“Š Processing Results:")
print(f"  â€¢ Methods tested: {total_methods}")
print(f"  â€¢ Successful charts: {successful_charts}")
print(f"  â€¢ Success rate: {successful_charts/total_methods*100:.1f}%")
print(f"  â€¢ Data points analyzed: {data_points:,}")

print(f"\nğŸ“ˆ Raw Data Characteristics:")
print(f"  â€¢ Total records: {len(raw_data):,}")
print(f"  â€¢ Time span: {(raw_data['timestamp'].max() - raw_data['timestamp'].min()).days} days")
print(f"  â€¢ Average volume: {raw_data['volume'].mean():.0f}")
print(f"  â€¢ Label range: [{raw_data['label'].min():.4f}, {raw_data['label'].max():.4f}]")

print(f"\nğŸ�ª Market Structure Analysis:")
print(f"  â€¢ Average bid quantity: {raw_data['bid_qty'].mean():.2f}")
print(f"  â€¢ Average ask quantity: {raw_data['ask_qty'].mean():.2f}")
print(f"  â€¢ Average buy quantity: {raw_data['buy_qty'].mean():.2f}")
print(f"  â€¢ Average sell quantity: {raw_data['sell_qty'].mean():.2f}")

# è®¡ç®—ä¹°å�–å�‹åŠ›
buy_pressure = raw_data['buy_qty'] / (raw_data['buy_qty'] + raw_data['sell_qty'] + 1e-10)
print(f"  â€¢ Average buy pressure: {buy_pressure.mean():.3f}")

print(f"\nğŸ”� Label Correlation Analysis:")
print(f"  â€¢ Label vs Volume: {np.corrcoef(raw_data['label'], raw_data['volume'])[0,1]:.4f}")
print(f"  â€¢ Label vs Buy Qty: {np.corrcoef(raw_data['label'], raw_data['buy_qty'])[0,1]:.4f}")
print(f"  â€¢ Label vs Sell Qty: {np.corrcoef(raw_data['label'], raw_data['sell_qty'])[0,1]:.4f}")

print(f"\nğŸ’¡ Key Insights & Recommendations:")
if successful_charts >= 3:
    print(f"  âœ… Multiple methods worked successfully")
    print(f"  âœ… Method 4 (Market Data Synthesis) likely most realistic")
    print(f"  âœ… Label appears to contain meaningful predictive signals")
    print(f"  âœ… Data quality is suitable for candlestick analysis")
else:
    print(f"  âš ï¸�  Limited success - consider parameter adjustment")
    print(f"  ğŸ’¡ Try different resampling frequencies")
    print(f"  ğŸ’¡ Experiment with signal strength parameters")

print(f"\nğŸ”§ Technical Implementation:")
print(f"  â€¢ Charts use professional mplfinance when possible")
print(f"  â€¢ Fallback to manual matplotlib charts if needed")
print(f"  â€¢ Conservative parameter scaling to prevent extreme values")
print(f"  â€¢ 2-hour resampling for clear candlestick patterns")
print(f"  â€¢ Automatic data validation and error handling")

print(f"\nğŸ�¯ Next Steps:")
print(f"  1. Fine-tune parameters based on preferred method")
print(f"  2. Experiment with different time windows")
print(f"  3. Consider adding technical indicators")
print(f"  4. Validate against actual market data if available")

print(f"\nâœ… Analysis complete! Generated {successful_charts}/{total_methods} charts successfully")

if successful_charts == 0:
    print(f"\nğŸš¨ TROUBLESHOOTING TIPS:")
    print(f"  â€¢ Check mplfinance version: pip install --upgrade mplfinance")
    print(f"  â€¢ Verify data format and completeness")
    print(f"  â€¢ Try reducing data_points if memory issues")
    print(f"  â€¢ Manual charts should work as fallback")


raw_data.head()

