import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 頁面設定 ---
st.set_page_config(page_title="內部人籌碼雷達 (V12 全級距透視版)", layout="wide")

# --- 設定全域時間起點 ---
START_DATE = '2017-01-01'

# --- 2. 定義熱門族群清單 ---
SECTOR_DB = {
    "🔥 CPO (矽光子)": ["3363", "3450", "4908", "4979", "6442", "3081", "3163", "3234", "6451", "2345", "2455"],
    "💻 PCB (印刷電路板)": ["3037", "8046", "3189", "2313", "2368", "3044", "4958", "6269", "5469", "2355", "3715", "6153"],
    "⚡ CCL (銅箔基板)": ["2383", "6213", "6274"],
    "💾 記憶體": ["2408", "2344", "2337", "8299", "3260", "4967", "8271", "3006", "2451", "8112", "3264"],
    "🏭 半導體設備": ["3131", "3583", "6196", "2404", "3680", "6640", "5443", "6667", "2059", "3413"],
    "👕 成衣與紡織": ["1476", "1477", "4401", "1402", "1460"],
    "❄️ 散熱": ["3017", "3324", "3653", "2421", "6230", "8996", "3483", "3338"],
    "🤖 AI 伺服器": ["2382", "2317", "3231", "6669", "2356", "2301"],
    "🧠 IC 設計": ["2454", "3034", "3035", "3529", "4961", "8016", "6138", "3527"],
    "🚢 航運": ["2603", "2609", "2615", "2618", "2610"],
    "⚡ 重電與綠能": ["1513", "1519", "1503", "1504", "6806", "9958"]
}

# --- MACD 計算 ---
def calculate_macd(series, fast=12, slow=26, signal=9):
    series = series.sort_index()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    macd_signal = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - macd_signal
    return pd.DataFrame({'Close': series, 'DIF': dif, 'Signal': macd_signal, 'Hist': hist})

# --- 輔助函式：取得股票名稱 ---
@st.cache_data
def get_stock_name_map():
    possible_paths = ["公司基本資料.csv", "shares/公司基本資料.csv", "stock_names.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={'stock_id': str, 'name': str})
                if 'name' not in df.columns and 'stock_name' in df.columns:
                    df.rename(columns={'stock_name': 'name'}, inplace=True)
                return dict(zip(df['stock_id'], df['name']))
            except:
                continue
    return {}

# --- 核心邏輯函式 ---
@st.cache_data
def load_data_and_calculate_metrics():
    files = {
        'price': "收盤價.csv",
        'major': "10%大股東持有數.csv",
        'issued': "發行股數.csv",
        'director_pct': "董監持有股數占比.csv",
        'large': "大戶持股比例.csv"
    }
    
    filter_files = {
        'cb': "可轉債標的.csv",
        'futures': "股票期貨標的.csv"
    }
    
    filters = {'cb': set(), 'futures': set()}

    try:
        # A. 讀取 CSV
        raw_dfs = {}
        for key, path in files.items():
            if not os.path.exists(path):
                if key == 'large':
                    st.warning("⚠️ 未檢測到 `大戶持股比例.csv`")
                    continue
                else:
                    st.error(f"找不到核心檔案: {path}")
                    return None, None, None

            df = pd.read_csv(path)
            # 強制去除欄位空白
            df.columns = [str(c).strip() for c in df.columns]

            if key == 'large':
                if 'stock_id' not in df.columns: df.rename(columns={df.columns[0]: 'stock_id'}, inplace=True)
                if 'date' not in df.columns: df.rename(columns={df.columns[1]: 'date'}, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'])
                df['stock_id'] = df['stock_id'].astype(str)
                
                valid_tiers = [c for c in df.columns if any(x in c for x in ['3000萬', '4000萬', '5000萬', '1億'])]
                
                large_dfs = {}
                for tier in valid_tiers:
                    pivot_df = df.pivot_table(index='date', columns='stock_id', values=tier)
                    # 修正：0值補前值 (ffill)
                    pivot_df = pivot_df.replace(0, np.nan).ffill()
                    large_dfs[tier] = pivot_df
                
                raw_dfs['large_tiers'] = large_dfs
                
            else:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                
                # 修正：0值補前值
                df = df.replace(0, np.nan).ffill()
                raw_dfs[key] = df

        # B. 建立週線基準
        price_weekly = raw_dfs['price'].resample('W-FRI').last()
        weekly_index = price_weekly.index

        # C. 對齊 (For Charting)
        aligned_price = price_weekly
        issued_weekly = raw_dfs['issued'].reindex(weekly_index, method='ffill')
        major_weekly = raw_dfs['major'].reindex(weekly_index, method='ffill')
        director_weekly = raw_dfs['director_pct'].reindex(weekly_index, method='ffill')
        
        aligned_large_tiers = {}
        if 'large_tiers' in raw_dfs:
            for tier, df in raw_dfs['large_tiers'].items():
                aligned_large_tiers[tier] = df.reindex(weekly_index, method='ffill')

        # D. 讀取篩選清單
        for key, path in filter_files.items():
            if os.path.exists(path):
                try:
                    f_df = pd.read_csv(path, dtype=str)
                    if not f_df.empty:
                        filters[key] = set(f_df.iloc[:, 0].unique())
                except: pass

        name_map = get_stock_name_map()

        # F. 時間裁切 (2017+)
        def slice_data(df):
            return df.loc[START_DATE:]

        final_price = slice_data(aligned_price)
        final_major = slice_data((major_weekly / issued_weekly) * 100)
        final_director = slice_data(director_weekly)
        
        final_large_chart = {} 
        for t, d in aligned_large_tiers.items():
            final_large_chart[t] = slice_data(d)

        # G. 取交集
        valid_stocks = final_price.dropna(axis=1, how='all').columns
        common_stocks = valid_stocks.intersection(final_major.columns).intersection(final_director.columns)
        if final_large_chart:
             first_tier = list(final_large_chart.values())[0]
             common_stocks = common_stocks.intersection(first_tier.columns)

        final_price = final_price[common_stocks]
        final_major = final_major[common_stocks]
        final_director = final_director[common_stocks]
        for t in final_large_chart:
            final_large_chart[t] = final_large_chart[t][common_stocks]

        # H. Correlation 計算
        price_for_major = raw_dfs['price'].reindex(raw_dfs['major'].index, method='ffill').loc[START_DATE:]
        major_raw_sliced = raw_dfs['major'].loc[START_DATE:]
        issued_raw_sliced = raw_dfs['issued'].loc[START_DATE:]
        major_pct_raw = (major_raw_sliced / issued_raw_sliced) * 100
        
        corr_major = major_pct_raw[common_stocks].corrwith(price_for_major[common_stocks])
        
        director_raw_sliced = raw_dfs['director_pct'].loc[START_DATE:]
        price_for_director = raw_dfs['price'].reindex(director_raw_sliced.index, method='ffill').loc[START_DATE:]
        corr_director = director_raw_sliced[common_stocks].corrwith(price_for_director[common_stocks])
        
        corr_large_dict = {}
        tier_stats = {}
        
        if 'large_tiers' in raw_dfs:
            for tier, df_raw in raw_dfs['large_tiers'].items():
                df_raw_sliced = df_raw.loc[START_DATE:]
                price_for_tier = raw_dfs['price'].reindex(df_raw_sliced.index, method='ffill')
                c = df_raw_sliced.corrwith(price_for_tier)
                simple_name = tier.replace('大戶持股比例_', '')
                corr_large_dict[simple_name] = c
                tier_stats[simple_name] = c.mean()

        # I. 建立總表
        stock_names = [name_map.get(s, '') for s in common_stocks]
        display_names = [f"{s} {name_map.get(s, '')}" for s in common_stocks]

        rank_df = pd.DataFrame({
            'StockID': common_stocks,
            'Name': stock_names,
            'DisplayName': display_names,
            'Price': final_price.iloc[-1],
            'Major_Pct': final_major.iloc[-1],
            'Director_Pct': final_director.iloc[-1],
            'Corr_Major': corr_major,
            'Corr_Director': corr_director
        }).set_index('StockID')

        for simple_name, c_series in corr_large_dict.items():
            rank_df[f'Corr_Large_{simple_name}'] = c_series
            tier_full = f'大戶持股比例_{simple_name}'
            if tier_full in final_large_chart:
                rank_df[f'Large_Pct_{simple_name}'] = final_large_chart[tier_full].iloc[-1]

        data_dict = {
            'calc_major': final_major,
            'calc_director': final_director,
            'aligned_price': final_price,
            'aligned_large_tiers': final_large_chart,
            'raw_large_tiers': raw_dfs.get('large_tiers', {}),
            'raw_price': raw_dfs['price'], # 日線資料
            'tier_stats': tier_stats 
        }
        
        return data_dict, rank_df, filters

    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
        return None, None, None

# --- 主程式 ---

st.title("🎯 內部人籌碼雷達 (V12 全級距透視版)")

data_res = load_data_and_calculate_metrics()

if data_res and data_res[0] is not None:
    raw_data, rank_df, stock_filters = data_res
    tier_stats = raw_data.get('tier_stats', {})
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ 全局參數")
        
        # 這裡的選單主要影響 Tab 2 排行榜 與 Tab 3 掃描
        avail_tiers = [k for k in tier_stats.keys()]
        preferred_order = ['1億', '5000萬', '4000萬', '3000萬']
        sorted_tiers = sorted(avail_tiers, key=lambda x: preferred_order.index(x) if x in preferred_order else 99)
        tier_labels = {t: f"{t} (Avg Corr: {tier_stats[t]:.2f})" for t in sorted_tiers}
        
        selected_tier_key_sidebar = st.selectbox(
            "💰 排行榜/掃描 基準門檻", 
            sorted_tiers, 
            format_func=lambda x: tier_labels[x],
            key='sidebar_tier'
        )
        
        st.markdown("---")
        st.header("🛠️ MACD 設定")
        macd_threshold = st.slider("DIF 容許範圍 (±)", 0.5, 10.0, 3.0, 0.5)

    # 用於 Tab 2, 3 的變數
    tier_col_corr_sb = f'Corr_Large_{selected_tier_key_sidebar}'
    tier_col_pct_sb = f'Large_Pct_{selected_tier_key_sidebar}'
    tier_full_name_sb = f'大戶持股比例_{selected_tier_key_sidebar}'

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 個股分析 (全級距)", "🏆 相關性排行", "🚀 連續買超掃描", "🛠️ 技術面掃描 (MACD)"])

    # === Tab 1: 個股分析 (全級距版) ===
    with tab1:
        st.header(f"個股籌碼 & 技術檢測")
        
        option_list = rank_df['DisplayName'].tolist()
        default_idx = rank_df.index.get_loc("2330") if "2330" in rank_df.index else 0
        
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            selected = st.selectbox("搜尋股票", option_list, index=default_idx)
        
        stock_id = selected.split(' ')[0]
        
        if stock_id in rank_df.index:
            info = rank_df.loc[stock_id]
            st.subheader(f"{info['DisplayName']}")
            
            # 1. 關鍵指標 (第一排)
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{info['Price']:.2f}")
            c2.metric("董監連動", f"{info['Corr_Director']:.2f}")
            c3.metric("大股東連動", f"{info['Corr_Major']:.2f}")

            # 2. 大戶連動指標 (第二排，顯示所有級距)
            st.markdown("##### 💰 各級距大戶連動係數")
            cols_tiers = st.columns(len(sorted_tiers))
            for i, t in enumerate(sorted_tiers):
                val = info.get(f'Corr_Large_{t}', 0)
                cols_tiers[i].metric(f">{t}", f"{val:.2f}", delta="High" if val > 0.7 else None)

            st.markdown("---")

            # 3. 圖表控制：選擇要畫哪一條線
            col_chart_ctl, col_chart_space = st.columns([1, 4])
            with col_chart_ctl:
                chart_tier_select = st.radio("選擇走勢圖大戶級距", sorted_tiers, index=0)
            
            # 圖表 A: 籌碼
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=raw_data['aligned_price'].index, y=raw_data['aligned_price'][stock_id], 
                                     name="股價", line=dict(color='#2c3e50', width=2)), secondary_y=False)
            
            # 根據 Radio Button 選擇畫圖
            selected_tier_fullname = f'大戶持股比例_{chart_tier_select}'
            if selected_tier_fullname in raw_data['aligned_large_tiers']:
                tier_data = raw_data['aligned_large_tiers'][selected_tier_fullname]
                if stock_id in tier_data.columns:
                    fig.add_trace(go.Scatter(x=tier_data.index, y=tier_data[stock_id], 
                                             name=f"大戶(>{chart_tier_select})", line=dict(color='#e74c3c', width=2)), secondary_y=True)
            
            fig.add_trace(go.Scatter(x=raw_data['calc_director'].index, y=raw_data['calc_director'][stock_id], 
                                     name="董監", line=dict(color='#f39c12', dash='dot')), secondary_y=True)
            fig.add_trace(go.Scatter(x=raw_data['calc_major'].index, y=raw_data['calc_major'][stock_id], 
                                     name="大股東", line=dict(color='#3498db', width=1)), secondary_y=True)

            fig.update_layout(title=f"籌碼趨勢 (顯示級距: >{chart_tier_select})", height=350, hovermode="x unified", margin=dict(b=0))
            fig.update_yaxes(title_text="股價", secondary_y=False)
            fig.update_yaxes(title_text="持股%", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # 圖表 B: MACD
            if 'raw_price' in raw_data and stock_id in raw_data['raw_price'].columns:
                daily_series = raw_data['raw_price'][stock_id].dropna()
                if not daily_series.empty:
                    last_date = daily_series.index.max()
                    start_lookback = last_date - pd.Timedelta(days=180)
                    daily_series = daily_series.loc[start_lookback:]
                    
                    if not daily_series.empty:
                        macd_df = calculate_macd(daily_series)
                        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        fig2.add_trace(go.Scatter(x=macd_df.index, y=macd_df['Close'], name='收盤價', line=dict(color='#33CC33', width=1.5)), row=1, col=1)
                        fig2.add_trace(go.Scatter(x=macd_df.index, y=macd_df['DIF'], line=dict(color='#FF6B6B', width=2), name='DIF'), row=2, col=1)
                        fig2.add_trace(go.Scatter(x=macd_df.index, y=macd_df['Signal'], line=dict(color='#4ECDC4', width=1), name='Signal'), row=2, col=1)
                        fig2.add_trace(go.Bar(x=macd_df.index, y=macd_df['Hist'], name='Hist', marker_color='gray', opacity=0.3), row=2, col=1)
                        
                        fig2.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)
                        fig2.add_hline(y=macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                        fig2.add_hline(y=-macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)

                        fig2.update_layout(height=400, xaxis_rangeslider_visible=False, template="plotly_dark", title=f"技術指標 (日線)", margin=dict(t=30))
                        st.plotly_chart(fig2, use_container_width=True)

            # 4. 全級距數據表
            st.subheader("近 12 週完整籌碼數據")
            recent_data = pd.DataFrame(index=raw_data['aligned_price'].index)
            recent_data['收盤價'] = raw_data['aligned_price'][stock_id]
            
            # 加入所有大戶級距
            for t in sorted_tiers:
                t_fullname = f'大戶持股比例_{t}'
                if t_fullname in raw_data['aligned_large_tiers']:
                    recent_data[f'>{t} (%)'] = raw_data['aligned_large_tiers'][t_fullname][stock_id]
            
            recent_data['董監(%)'] = raw_data['calc_director'][stock_id]
            recent_data['大股東(%)'] = raw_data['calc_major'][stock_id]
            
            display_recent = recent_data.tail(12).sort_index(ascending=False)
            display_recent.index = display_recent.index.strftime('%Y-%m-%d')
            st.dataframe(display_recent.style.format("{:.2f}"), use_container_width=True)

    # === Tab 2: Ranking ===
    with tab2:
        st.header(f"全市場排行榜")
        
        c1, c2 = st.columns(2)
        with c1:
            filter_mode = st.radio("篩選範圍", ["全市場", "僅限可轉債(CB)", "僅限股票期貨"], horizontal=True, key="rank_filter")
        with c2:
            sort_metric = st.selectbox("排序指標", [f"大戶(>{selected_tier_key_sidebar}) Correlation", "董監持股 Correlation", "10%大股東 Correlation"])

        target_df = rank_df.copy()
        if "CB" in filter_mode and stock_filters['cb']:
            target_df = target_df[target_df.index.isin(stock_filters['cb'])]
        elif "Futures" in filter_mode and stock_filters['futures']:
            target_df = target_df[target_df.index.isin(stock_filters['futures'])]

        col_key = 'Corr_Major'
        if "大戶" in sort_metric: col_key = tier_col_corr_sb
        elif "董監" in sort_metric: col_key = 'Corr_Director'
            
        top_df = target_df.sort_values(by=col_key, ascending=False).head(20)

        display_cols = ['DisplayName', 'Price', col_key, 'Director_Pct', 'Major_Pct']
        renamed = {'DisplayName': '股名', 'Price': '股價', col_key: '相關係數', 'Director_Pct': '董監%', 'Major_Pct': '大股東%'}
        if selected_tier_key_sidebar:
            display_cols.insert(3, tier_col_pct_sb)
            renamed[tier_col_pct_sb] = f'大戶(>{selected_tier_key_sidebar})%'

        subset_cols = list(renamed.values())
        if '股名' in subset_cols: subset_cols.remove('股名')

        st.dataframe(
            top_df[display_cols].rename(columns=renamed)
            .style.background_gradient(subset=['相關係數'], cmap='Reds')
            .format("{:.2f}", subset=subset_cols),
            use_container_width=True,
            height=800
        )

    # === Tab 3: Scanner ===
    with tab3:
        st.header(f"🚀 連續買超掃描")
        
        c_filter, c_thres = st.columns(2)
        with c_filter:
            scan_scope = st.radio("掃描範圍", ["全市場", "僅限可轉債(CB)", "僅限股票期貨"], horizontal=True, key="scan_filter")
        with c_thres:
            growth_thres = st.selectbox("🔥 總成長門檻", [1, 3, 5, 10], format_func=lambda x: f"累計增加 > {x}%")

        if st.button("開始籌碼掃描"):
            raw_large_tiers = raw_data.get('raw_large_tiers', {})
            
            if tier_full_name_sb in raw_large_tiers:
                raw_ts = raw_large_tiers[tier_full_name_sb]
                if len(raw_ts) < 3:
                    st.error("歷史資料不足。")
                else:
                    last_3 = raw_ts.iloc[-3:]
                    w_curr = last_3.iloc[-1]
                    w_prev = last_3.iloc[-2]
                    w_prev2 = last_3.iloc[-3]
                    
                    cond_continuous = (w_curr > w_prev) & (w_prev > w_prev2)
                    diff = w_curr - w_prev2
                    cond_magnitude = diff >= growth_thres
                    
                    candidates = raw_ts.columns[cond_continuous & cond_magnitude]
                    scan_df = rank_df.loc[rank_df.index.intersection(candidates)].copy()
                    
                    if "CB" in scan_scope and stock_filters['cb']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['cb'])]
                    elif "Futures" in scan_scope and stock_filters['futures']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['futures'])]
                    
                    if scan_df.empty:
                        st.warning(f"⚠️ 無符合標的。")
                    else:
                        st.success(f"🎉 發現 {len(scan_df)} 檔潛力股！")
                        
                        res_df = scan_df[['DisplayName', 'Price', tier_col_corr_sb]].copy()
                        res_df['W-2(%)'] = w_prev2[scan_df.index]
                        res_df['W-1(%)'] = w_prev[scan_df.index]
                        res_df['Current(%)'] = w_curr[scan_df.index]
                        res_df['Total Growth'] = diff[scan_df.index]
                        
                        res_df = res_df.sort_values('Total Growth', ascending=False)
                        
                        renamed = {'DisplayName': '股名', 'Price': '股價', tier_col_corr_sb: '相關係數', 'Total Growth': '累計增幅%'}
                        subset_cols = list(renamed.values()) + ['W-2(%)', 'W-1(%)', 'Current(%)']
                        if '股名' in subset_cols: subset_cols.remove('股名')

                        st.dataframe(
                            res_df.rename(columns=renamed)
                            .style.background_gradient(subset=['累計增幅%'], cmap='Reds')
                            .format("{:.2f}", subset=subset_cols),
                            use_container_width=True,
                            height=600
                        )
            else:
                st.error("Data Error.")

    # === Tab 4: MACD Scanner ===
    with tab4:
        st.header("🛠️ 技術面掃描 (MACD)")
        
        all_sectors = list(SECTOR_DB.keys())
        selected_sectors = st.multiselect("選擇掃描板塊", options=all_sectors, default=["🔥 CPO (矽光子)", "💾 記憶體"])
        
        if st.button("🚀 開始 MACD 掃描", type="primary"):
            if not selected_sectors:
                st.warning("請選擇板塊")
            else:
                target_tickers = []
                for s in selected_sectors:
                    target_tickers.extend(SECTOR_DB[s])
                
                target_tickers = [t.replace('.TW', '').replace('.TWO', '') for t in target_tickers]
                target_tickers = list(set(target_tickers))

                results = []
                raw_price_df = raw_data['raw_price']
                
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(target_tickers):
                    progress_bar.progress((i + 1) / len(target_tickers))
                    
                    if ticker in raw_price_df.columns:
                        try:
                            series = raw_price_df[ticker].dropna()
                            if len(series) < 63: continue 

                            macd_df = calculate_macd(series)
                            current_dif = macd_df['DIF'].iloc[-1]
                            current_close = macd_df['Close'].iloc[-1]
                            
                            lookback = macd_df.iloc[-63:]
                            max_high = lookback['Close'].max()
                            max_high_date = lookback['Close'].idxmax()
                            days_since_high = (macd_df.index[-1] - max_high_date).days
                            
                            is_recent_high = days_since_high <= 45
                            is_pullback = current_close < max_high
                            is_near_zero = abs(current_dif) <= macd_threshold
                            
                            if is_recent_high and is_pullback and is_near_zero:
                                my_sectors = []
                                for s_name, s_list in SECTOR_DB.items():
                                    clean_list = [x.replace('.TW', '').replace('.TWO', '') for x in s_list]
                                    if ticker in clean_list: my_sectors.append(s_name.split(' ')[1])
                                
                                results.append({
                                    '代號': ticker,
                                    '名稱': rank_df.loc[ticker, 'Name'] if ticker in rank_df.index else ticker,
                                    '族群': ",".join(my_sectors),
                                    '現價': current_close,
                                    'DIF值': current_dif,
                                    'Data': macd_df
                                })

                        except Exception: continue
                
                progress_bar.empty()
                
                if results:
                    df_res = pd.DataFrame(results)
                    df_res['Abs_DIF'] = df_res['DIF值'].abs()
                    df_res = df_res.sort_values('Abs_DIF').drop(columns=['Abs_DIF', 'Data'])
                    
                    st.success(f"掃描完成！共 {len(results)} 檔。")
                    st.dataframe(df_res, use_container_width=True)
                    
                    st.markdown("---")
                    cols = st.columns(2)
                    for idx, row in enumerate(results):
                        with cols[idx % 2]:
                            stock_code = row['代號']
                            stock_name = row['名稱']
                            df_plot = row['Data'].tail(120)
                            
                            with st.expander(f"{stock_code} {stock_name} (DIF: {row['DIF值']:.2f})", expanded=True):
                                fig_scan = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                                
                                fig_scan.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Close', 
                                                              line=dict(color='#33CC33', width=2)), row=1, col=1)
                                
                                fig_scan.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DIF'], name='DIF', line=dict(color='#FF6B6B')), row=2, col=1)
                                fig_scan.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal'], name='Signal', line=dict(color='#4ECDC4')), row=2, col=1)
                                fig_scan.add_trace(go.Bar(x=df_plot.index, y=df_plot['Hist'], name='Hist', marker_color='gray', opacity=0.3), row=2, col=1)
                                
                                fig_scan.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)
                                fig_scan.add_hline(y=macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                                fig_scan.add_hline(y=-macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                                
                                fig_scan.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                                st.plotly_chart(fig_scan, use_container_width=True)
                else:
                    st.warning("無符合標的。")

else:
    st.info("請將所有 CSV 檔案放置於同一目錄下。")