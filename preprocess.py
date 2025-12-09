import pandas as pd
import numpy as np
import os

# --- 設定 ---
START_DATE_CHART = '2022-01-01' # 圖表只畫最近幾年，省空間
START_DATE_CALC = '2017-01-01'  # 計算 Correlation 用較長區間

EXPORT_FILE = 'app_data.pkl'

def calculate_macd(series, fast=12, slow=26, signal=9):
    """計算 MACD"""
    series = series.sort_index()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    macd_signal = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - macd_signal
    # 為了省空間，我們只留最近的數據供畫圖
    return pd.DataFrame({'Close': series, 'DIF': dif, 'Signal': macd_signal, 'Hist': hist})

def run_preprocessing():
    print("🚀 開始預處理數據 (Render 瘦身版)...")
    
    files = {
        'price': "收盤價.csv",
        'major': "10%大股東持有數.csv",
        'issued': "發行股數.csv",
        'director_pct': "董監持有股數占比.csv",
        'large': "大戶持股比例.csv"
    }
    
    raw_dfs = {}
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ 找不到 {path}")
            continue
        
        print(f"📖 讀取 {key}...")
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        
        if key == 'large':
            if 'stock_id' not in df.columns: df.rename(columns={df.columns[0]: 'stock_id'}, inplace=True)
            if 'date' not in df.columns: df.rename(columns={df.columns[1]: 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['stock_id'] = df['stock_id'].astype(str)
            
            valid_tiers = [c for c in df.columns if any(x in c for x in ['3000萬', '4000萬', '5000萬', '1億'])]
            large_dfs = {}
            for tier in valid_tiers:
                p = df.pivot_table(index='date', columns='stock_id', values=tier)
                p = p.replace(0, np.nan).ffill() 
                large_dfs[tier] = p
            raw_dfs['large_tiers'] = large_dfs
        else:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            df = df.replace(0, np.nan).ffill()
            raw_dfs[key] = df

    print("📊 執行對齊與計算...")
    
    # 1. 建立週線基準
    price_weekly = raw_dfs['price'].resample('W-FRI').last()
    
    # 2. 對齊數據 (全歷史，用於計算 Correlation)
    aligned_full = {
        'price': price_weekly,
        'director': raw_dfs['director_pct'].reindex(price_weekly.index, method='ffill'),
        'major': (raw_dfs['major'].reindex(price_weekly.index, method='ffill') / 
                  raw_dfs['issued'].reindex(price_weekly.index, method='ffill') * 100),
        'large_tiers': {}
    }
    for tier, df in raw_dfs.get('large_tiers', {}).items():
        aligned_full['large_tiers'][tier] = df.reindex(price_weekly.index, method='ffill')

    # 3. 計算 Correlation (存成輕量 DataFrame)
    print("🧮 計算相關係數...")
    
    # 裁切計算區間 (2017+)
    def slice_calc(df): return df.loc[START_DATE_CALC:]
    
    p_calc = slice_calc(aligned_full['price'])
    
    correlations = {}
    correlations['Major'] = slice_calc(aligned_full['major']).corrwith(p_calc)
    correlations['Director'] = slice_calc(aligned_full['director']).corrwith(p_calc)
    
    for tier, df in aligned_full['large_tiers'].items():
        simple_name = tier.replace('大戶持股比例_', '')
        correlations[f'Large_{simple_name}'] = slice_calc(df).corrwith(p_calc)

    corr_df = pd.DataFrame(correlations)
    
    # 4. 準備圖表數據 (只留最近幾年，極度瘦身)
    print("✂️ 裁切圖表數據...")
    def slice_chart(df): return df.loc[START_DATE_CHART:]
    
    chart_data = {
        'price': slice_chart(aligned_full['price']),
        'director': slice_chart(aligned_full['director']),
        'major': slice_chart(aligned_full['major']),
        'large_tiers': {k: slice_chart(v) for k, v in aligned_full['large_tiers'].items()}
    }
    
    # 5. 準備 MACD 數據 (日線，也只留近期)
    # 我們先預計算好 MACD 的最後一筆值 (for Scanner)，並保留最近 180 天的 Series (for Chart)
    print("📈 預處理 MACD...")
    raw_price_recent = raw_dfs['price'].loc[raw_dfs['price'].index >= '2023-01-01'] # 留兩年算指標比較準
    
    # 為了節省空間，我們不存整個 DataFrame，只存一個 dict
    # key: stock_id, value: small_df (tail 180)
    # 還有一個 summary df 用於掃描
    
    # 掃描用的摘要表 (最新一筆數據)
    macd_scan_list = []
    
    # 圖表用的數據包 (只存最近 180 天)
    macd_chart_data = {} 
    
    target_stocks = raw_price_recent.columns
    total = len(target_stocks)
    
    for i, stock in enumerate(target_stocks):
        if i % 500 == 0: print(f"   處理 MACD: {i}/{total}")
        try:
            series = raw_price_recent[stock].dropna()
            if len(series) < 63: continue
            
            # 計算
            df = calculate_macd(series)
            
            # 存掃描數據 (最新一筆)
            macd_scan_list.append({
                'StockID': stock,
                'Close': df['Close'].iloc[-1],
                'DIF': df['DIF'].iloc[-1],
                'Max_High_63': df['Close'].iloc[-63:].max(),
                'Days_Since_High': (df.index[-1] - df['Close'].iloc[-63:].idxmax()).days
            })
            
            # 存圖表數據 (只留最近 180 天，並且只存需要的欄位以省空間)
            # 使用 JSON 序列化友好的格式或直接 DF
            # 這裡我們只存最近 180 天
            macd_chart_data[stock] = df.tail(180)
            
        except: continue
        
    macd_summary = pd.DataFrame(macd_scan_list).set_index('StockID')

    # 6. 打包存檔
    export_data = {
        'corr_df': corr_df,       # 排行榜用
        'chart_data': chart_data, # Tab 1 畫圖用 (週線)
        'macd_summary': macd_summary, # Tab 4 掃描用
        'macd_chart_data': macd_chart_data # Tab 4 畫圖用 (日線)
    }
    
    print(f"💾 儲存至 {EXPORT_FILE} (請稍候)...")
    pd.to_pickle(export_data, EXPORT_FILE)
    print(f"✅ 完成！檔案大小: {os.path.getsize(EXPORT_FILE) / 1024 / 1024 :.2f} MB")
    print("👉 請將 app_data.pkl, app.py, requirements.txt 上傳至 GitHub。")

if __name__ == "__main__":
    run_preprocessing()