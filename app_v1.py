import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 頁面設定 ---
st.set_page_config(page_title="內部人籌碼雷達 (輕量旗艦版)", layout="wide")

# --- 讀取預處理資料 (關鍵) ---
@st.cache_data
def load_data():
    file_path = 'app_data.pkl'
    if not os.path.exists(file_path):
        st.error("⚠️ 找不到 `app_data.pkl`。請先在本地端執行 `preprocess.py` 並上傳結果檔案。")
        return None
    return pd.read_pickle(file_path)

@st.cache_data
def get_stock_name_map():
    # 嘗試讀取股名檔
    paths = ["公司基本資料.csv", "shares/公司基本資料.csv", "stock_names.csv"]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str)
                # 簡單容錯
                if 'name' not in df.columns: df.rename(columns={df.columns[-1]: 'name'}, inplace=True)
                if 'stock_id' not in df.columns: df.rename(columns={df.columns[0]: 'stock_id'}, inplace=True)
                return dict(zip(df['stock_id'], df['name']))
            except: continue
    return {}

# --- 族群資料 (寫死在程式裡以免讀檔) ---
SECTOR_DB = {
    "🔥 CPO (矽光子)": ["3363", "3450", "4908", "4979", "6442", "3081", "3163", "3234", "6451", "2345", "2455"],
    "💻 PCB": ["3037", "8046", "3189", "2313", "2368", "3044", "4958", "6269", "5469", "2355", "3715", "6153"],
    "⚡ CCL": ["2383", "6213", "6274"],
    "💾 記憶體": ["2408", "2344", "2337", "8299", "3260", "4967", "8271", "3006", "2451", "8112", "3264"],
    "🏭 半導體設備": ["3131", "3583", "6196", "2404", "3680", "6640", "5443", "6667", "2059", "3413"],
    "👕 成衣": ["1476", "1477", "4401", "1402", "1460"],
    "❄️ 散熱": ["3017", "3324", "3653", "2421", "6230", "8996", "3483", "3338"],
    "🤖 AI 伺服器": ["2382", "2317", "3231", "6669", "2356", "2301"],
    "🧠 IC 設計": ["2454", "3034", "3035", "3529", "4961", "8016", "6138", "3527"],
    "🚢 航運": ["2603", "2609", "2615", "2618", "2610"],
    "⚡ 重電": ["1513", "1519", "1503", "1504", "6806", "9958"]
}

# --- 主程式 ---
st.title("🎯 內部人籌碼雷達 (輕量版)")

data = load_data()
name_map = get_stock_name_map()

if data:
    # 解包資料
    corr_df = data['corr_df']       # 全歷史 Correlation
    chart_data = data['chart_data'] # 最近幾年週線數據
    macd_summary = data['macd_summary'] # MACD 最新掃描結果
    macd_charts = data['macd_chart_data'] # MACD 近半年日線圖
    
    # 建立排行榜主表 (Rank DF)
    # 取最新一筆週線資料
    last_price = chart_data['price'].iloc[-1]
    last_major = chart_data['major'].iloc[-1]
    last_director = chart_data['director'].iloc[-1]
    
    # 合併
    rank_df = corr_df.copy()
    rank_df['Price'] = last_price
    rank_df['Major_Pct'] = last_major
    rank_df['Director_Pct'] = last_director
    
    # 補上大戶持股
    large_tiers_keys = list(chart_data['large_tiers'].keys())
    # 排序 keys: 1億 -> 5000萬...
    def sort_key(x): return ['1億', '5000萬', '4000萬', '3000萬'].index(x.split('_')[1]) if x.split('_')[1] in ['1億', '5000萬', '4000萬', '3000萬'] else 99
    sorted_tier_keys = sorted(large_tiers_keys, key=sort_key)
    
    for k in sorted_tier_keys:
        simple = k.replace('大戶持股比例_', '')
        rank_df[f'Large_Pct_{simple}'] = chart_data['large_tiers'][k].iloc[-1]

    # 補上名稱
    rank_df['Name'] = [name_map.get(x, '') for x in rank_df.index]
    rank_df['DisplayName'] = rank_df.index + " " + rank_df['Name']

    # --- 側邊欄設定 ---
    with st.sidebar:
        st.header("⚙️ 全局參數")
        
        # 用於排行榜的基準
        avail_simples = [k.replace('大戶持股比例_', '') for k in sorted_tier_keys]
        # 建立選項標籤 (含平均 Correlation)
        tier_labels = {}
        for s in avail_simples:
            c = corr_df[f'Large_{s}'].mean()
            tier_labels[s] = f"{s} (Avg Corr: {c:.2f})"
            
        selected_tier = st.selectbox("💰 排行榜基準", avail_simples, format_func=lambda x: tier_labels[x])
        
        st.markdown("---")
        st.header("🛠️ MACD 設定")
        macd_threshold = st.slider("DIF 容許範圍 (±)", 0.5, 10.0, 3.0, 0.5)

    # 變數準備
    col_corr_sel = f'Large_{selected_tier}'
    col_pct_sel = f'Large_Pct_{selected_tier}'
    key_tier_sel = f'大戶持股比例_{selected_tier}'

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 個股分析", "🏆 相關性排行", "🚀 連續買超掃描", "🛠️ 技術面掃描"])

    # === Tab 1: 個股分析 ===
    with tab1:
        st.header("個股全方位檢測")
        
        # 搜尋框
        opt_list = rank_df['DisplayName'].dropna().tolist()
        # 預設台積電
        def_idx = 0
        for i, o in enumerate(opt_list):
            if '2330' in o: 
                def_idx = i
                break
        
        sel_stock_str = st.selectbox("搜尋股票", opt_list, index=def_idx)
        stock_id = sel_stock_str.split(' ')[0]
        
        if stock_id in rank_df.index:
            row = rank_df.loc[stock_id]
            
            # A. 關鍵指標
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{row['Price']:.2f}")
            c2.metric("董監連動", f"{row['Director']:.2f}")
            c3.metric("大股東連動", f"{row['Major']:.2f}")
            
            st.markdown("##### 💰 各級距大戶連動係數")
            cols = st.columns(len(avail_simples))
            for i, s in enumerate(avail_simples):
                val = row.get(f'Large_{s}', 0)
                cols[i].metric(f">{s}", f"{val:.2f}", delta="High" if val>0.7 else None)
            
            st.markdown("---")
            
            # B. 籌碼圖表
            # Radio 切換顯示級距
            chart_tier_simple = st.radio("選擇走勢圖大戶級距", avail_simples, index=0, horizontal=True)
            chart_tier_key = f'大戶持股比例_{chart_tier_simple}'
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 股價
            p_data = chart_data['price'][stock_id].dropna()
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data, name="股價", line=dict(color='#2c3e50', width=2)), secondary_y=False)
            
            # 大戶
            if chart_tier_key in chart_data['large_tiers'] and stock_id in chart_data['large_tiers'][chart_tier_key].columns:
                l_data = chart_data['large_tiers'][chart_tier_key][stock_id].dropna()
                fig.add_trace(go.Scatter(x=l_data.index, y=l_data, name=f">{chart_tier_simple}", line=dict(color='#e74c3c', width=2)), secondary_y=True)
                
            # 董監/大股東
            d_data = chart_data['director'][stock_id].dropna()
            m_data = chart_data['major'][stock_id].dropna()
            
            fig.add_trace(go.Scatter(x=d_data.index, y=d_data, name="董監", line=dict(color='#f39c12', dash='dot')), secondary_y=True)
            fig.add_trace(go.Scatter(x=m_data.index, y=m_data, name="大股東", line=dict(color='#3498db', width=1)), secondary_y=True)
            
            fig.update_layout(title=f"籌碼趨勢 (顯示: >{chart_tier_simple})", height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # C. MACD 圖表 (讀取預存的日線)
            if stock_id in macd_charts:
                df_macd = macd_charts[stock_id]
                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                fig2.add_trace(go.Scatter(x=df_macd.index, y=df_macd['Close'], name='Close', line=dict(color='#33CC33', width=1.5)), row=1, col=1)
                fig2.add_trace(go.Scatter(x=df_macd.index, y=df_macd['DIF'], name='DIF', line=dict(color='#FF6B6B', width=2)), row=2, col=1)
                fig2.add_trace(go.Scatter(x=df_macd.index, y=df_macd['Signal'], name='Signal', line=dict(color='#4ECDC4', width=1)), row=2, col=1)
                fig2.add_trace(go.Bar(x=df_macd.index, y=df_macd['Hist'], name='Hist', marker_color='gray', opacity=0.3), row=2, col=1)
                
                fig2.add_hline(y=0, line_color="white", row=2, col=1)
                fig2.add_hline(y=macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                fig2.add_hline(y=-macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                
                fig2.update_layout(height=400, template="plotly_dark", title="技術指標 (日線)", margin=dict(t=30))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("無近期日線資料，無法顯示 MACD。")

            # D. 詳細表格 (近12週)
            st.subheader("近 12 週籌碼明細")
            # 建立表格
            idx = chart_data['price'].index
            tbl = pd.DataFrame(index=idx)
            tbl['收盤價'] = chart_data['price'][stock_id]
            
            for s in avail_simples:
                k = f'大戶持股比例_{s}'
                if k in chart_data['large_tiers']:
                    tbl[f'>{s} (%)'] = chart_data['large_tiers'][k][stock_id]
            
            tbl['董監(%)'] = chart_data['director'][stock_id]
            tbl['大股東(%)'] = chart_data['major'][stock_id]
            
            # 取最後 12 筆並反轉
            show_tbl = tbl.tail(12).sort_index(ascending=False)
            show_tbl.index = show_tbl.index.strftime('%Y-%m-%d')
            st.dataframe(show_tbl.style.format("{:.2f}"), use_container_width=True)

    # === Tab 2: 排行榜 ===
    with tab2:
        st.header("全市場排行榜")
        
        c1, c2 = st.columns(2)
        with c1:
            # 這裡可以加篩選邏輯，如果 pkl 有存篩選清單的話
            # 為了輕量版簡化，我們先只做全市場排序
            st.info("目前顯示全市場標的 (可轉債/期貨篩選需在預處理階段加入)")
        with c2:
            sort_target = st.selectbox("排序指標", [f"大戶(>{selected_tier}) Corr", "董監 Corr", "大股東 Corr"])
        
        col_map = {
            f"大戶(>{selected_tier}) Corr": col_corr_sel,
            "董監 Corr": 'Director',
            "大股東 Corr": 'Major'
        }
        
        target_col = col_map[sort_target]
        top_df = rank_df.sort_values(target_col, ascending=False).head(20)
        
        # 顯示
        disp_cols = ['DisplayName', 'Price', target_col, 'Director_Pct', 'Major_Pct', col_pct_sel]
        renamed = {
            'DisplayName': '股名', 'Price': '股價', target_col: '相關係數',
            'Director_Pct': '董監%', 'Major_Pct': '大股東%', col_pct_sel: f'大戶(>{selected_tier})%'
        }
        
        st.dataframe(
            top_df[disp_cols].rename(columns=renamed)
            .style.background_gradient(subset=['相關係數'], cmap='Reds')
            .format("{:.2f}", subset=list(renamed.values())[1:]),
            use_container_width=True, 
            height=800
        )

    # === Tab 3: 掃描 (使用 chart_data 的週線數據) ===
    with tab3:
        st.header("🚀 連續買超掃描")
        thres = st.selectbox("累計增幅門檻", [1, 3, 5, 10], format_func=lambda x: f">{x}%")
        
        if st.button("開始掃描"):
            if key_tier_sel in chart_data['large_tiers']:
                df_tier = chart_data['large_tiers'][key_tier_sel]
                # 取最後 3 週 (已ffill過)
                last3 = df_tier.iloc[-3:]
                if len(last3) == 3:
                    w0, w1, w2 = last3.iloc[0], last3.iloc[1], last3.iloc[2] # w0=上上週, w2=本週
                    
                    # 邏輯: 持續增加 且 總量達標
                    cond = (w2 > w1) & (w1 > w0) & ((w2 - w0) >= thres)
                    hits = df_tier.columns[cond]
                    
                    if len(hits) > 0:
                        st.success(f"發現 {len(hits)} 檔！")
                        scan_res = rank_df.loc[hits].copy()
                        scan_res['Growth'] = w2[hits] - w0[hits]
                        scan_res = scan_res.sort_values('Growth', ascending=False)
                        
                        show_res = scan_res[['DisplayName', 'Price', col_corr_sel, 'Growth']]
                        st.dataframe(show_res.style.format("{:.2f}", subset=['Price', col_corr_sel, 'Growth']), use_container_width=True)
                    else:
                        st.warning("無符合標的")
                else:
                    st.error("資料不足")

    # === Tab 4: MACD 掃描 (使用 macd_summary) ===
    with tab4:
        st.header("🛠️ 技術面掃描 (MACD)")
        
        # 這裡需要把 SECTOR_DB 的代號對應到 macd_summary 的 index
        # 簡單做個介面
        sel_sectors = st.multiselect("板塊", list(SECTOR_DB.keys()), default=["🔥 CPO (矽光子)"])
        
        if st.button("MACD 掃描"):
            targets = []
            for s in sel_sectors: 
                # 處理代號 (移除 .TW)
                clean_ids = [x.replace('.TW', '').replace('.TWO', '') for x in SECTOR_DB[s]]
                targets.extend(clean_ids)
            
            # 篩選
            mask = macd_summary.index.isin(targets)
            sub_df = macd_summary[mask].copy()
            
            # 邏輯: DIF 在範圍內 & 收盤價 < 高點 (拉回) & 剛創高
            # is_near_zero
            cond1 = sub_df['DIF'].abs() <= macd_threshold
            # is_pullback
            cond2 = sub_df['Close'] < sub_df['Max_High_63']
            # is_recent_high
            cond3 = sub_df['Days_Since_High'] <= 45
            
            final_hits = sub_df[cond1 & cond2 & cond3]
            
            if not final_hits.empty:
                # 補上名稱
                final_hits['Name'] = [name_map.get(x, x) for x in final_hits.index]
                st.success(f"找到 {len(final_hits)} 檔")
                st.dataframe(final_hits[['Name', 'Close', 'DIF', 'Days_Since_High']], use_container_width=True)
                
                # 畫圖
                cols = st.columns(2)
                for idx, (sid, row) in enumerate(final_hits.iterrows()):
                    with cols[idx % 2]:
                        with st.expander(f"{sid} {row['Name']} (DIF: {row['DIF']:.2f})", expanded=True):
                            if sid in macd_charts:
                                df_plot = macd_charts[sid]
                                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], line=dict(color='#33CC33')), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['DIF'], line=dict(color='#FF6B6B')), row=2, col=1)
                                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal'], line=dict(color='#4ECDC4')), row=2, col=1)
                                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Hist'], marker_color='gray', opacity=0.3), row=2, col=1)
                                fig.add_hline(y=0, line_color="white", row=2, col=1)
                                fig.add_hline(y=macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                                fig.add_hline(y=-macd_threshold, line_dash="dot", line_color="orange", row=2, col=1)
                                fig.update_layout(height=300, template="plotly_dark", margin=dict(t=0, b=0))
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("無符合標的")