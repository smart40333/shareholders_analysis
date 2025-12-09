import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 頁面設定 ---
st.set_page_config(page_title="內部人籌碼雷達 (V5 修正版)", layout="wide")

# --- 設定全域時間起點 ---
START_DATE = '2017-01-01'

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
    # 1. 定義檔案路徑
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
        # A. 讀取並前處理 CSV
        raw_dfs = {}
        for key, path in files.items():
            if not os.path.exists(path):
                if key == 'large':
                    st.warning("⚠️ 未檢測到 `大戶持股比例.csv`，無法使用大戶分析功能。")
                    continue
                else:
                    st.error(f"找不到核心檔案: {path}")
                    return None, None, None

            df = pd.read_csv(path)
            
            # 特殊處理：大戶持股
            if key == 'large':
                df.columns = [c.strip() for c in df.columns]
                if 'stock_id' not in df.columns: df.rename(columns={df.columns[0]: 'stock_id'}, inplace=True)
                if 'date' not in df.columns: df.rename(columns={df.columns[1]: 'date'}, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'])
                df['stock_id'] = df['stock_id'].astype(str)
                
                # 抓出 3000萬以上的級距
                valid_tiers = [c for c in df.columns if any(x in c for x in ['3000萬', '4000萬', '5000萬', '1億'])]
                
                large_dfs = {}
                for tier in valid_tiers:
                    # Pivot Table (原始資料，不做重採樣，保留真實日期)
                    pivot_df = df.pivot_table(index='date', columns='stock_id', values=tier)
                    pivot_df = pivot_df.replace(0, np.nan)
                    large_dfs[tier] = pivot_df
                
                raw_dfs['large_tiers'] = large_dfs
                
            else:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                df.columns = df.columns.astype(str)
                df = df.replace(0, np.nan)
                raw_dfs[key] = df

        # B. 建立週線基準 (Anchor)
        price_weekly = raw_dfs['price'].resample('W-FRI').last()
        weekly_index = price_weekly.index

        # C. 嚴格對齊 (這是給圖表和 Correlation 用的，保證對齊股價)
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

        # F. 時間裁切 (2017+) - 給圖表用
        def slice_and_clean(df_target, df_anchor_price):
            df_subset = df_target.loc[START_DATE:]
            price_subset = df_anchor_price.loc[START_DATE:]
            common_idx = df_subset.index.intersection(price_subset.index)
            return df_subset.loc[common_idx]

        final_price = aligned_price.loc[START_DATE:]
        
        major_pct_raw = (major_weekly / issued_weekly) * 100
        final_major = slice_and_clean(major_pct_raw, final_price)
        final_director = slice_and_clean(director_weekly, final_price)
        
        final_large = {}
        for t, d in aligned_large_tiers.items():
            final_large[t] = slice_and_clean(d, final_price)

        # G. 取最終交集
        valid_stocks = final_price.dropna(axis=1, how='all').columns
        common_stocks = valid_stocks.intersection(final_major.columns).intersection(final_director.columns)
        if final_large:
             first_tier = list(final_large.values())[0]
             common_stocks = common_stocks.intersection(first_tier.columns)

        final_price = final_price[common_stocks]
        final_major = final_major[common_stocks]
        final_director = final_director[common_stocks]
        for t in final_large:
            final_large[t] = final_large[t][common_stocks]

        # H. 計算 Correlation
        corr_major = final_price.corrwith(final_major)
        corr_director = final_price.corrwith(final_director)
        
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

        # 加入各級距大戶指標與 Correlation
        tier_stats = {} 
        for t, d in final_large.items():
            c = final_price.corrwith(d)
            simple_name = t.replace('大戶持股比例_', '')
            
            rank_df[f'Corr_Large_{simple_name}'] = c
            rank_df[f'Large_Pct_{simple_name}'] = d.iloc[-1]
            tier_stats[simple_name] = c.mean()

        data_dict = {
            'calc_major': final_major,
            'calc_director': final_director,
            'aligned_price': final_price,
            'aligned_large_tiers': final_large, # 這是對齊股價後的 (畫圖用)
            'raw_large_tiers': raw_dfs.get('large_tiers', {}), # 這是原始資料 (掃描用) <--- 關鍵新增
            'tier_stats': tier_stats 
        }
        
        return data_dict, rank_df, filters

    except Exception as e:
        import traceback
        st.error(f"資料處理發生錯誤: {e}")
        st.text(traceback.format_exc())
        return None, None, None

# --- 主程式 ---

st.title("🎯 內部人籌碼雷達 (V5 修正版)")

data_res = load_data_and_calculate_metrics()

if data_res:
    raw_data, rank_df, stock_filters = data_res
    tier_stats = raw_data.get('tier_stats', {})
    
    # --- 側邊欄 ---
    with st.sidebar:
        st.header("⚙️ 參數設定")
        
        avail_tiers = [k for k in tier_stats.keys()]
        preferred_order = ['1億', '5000萬', '4000萬', '3000萬']
        sorted_tiers = sorted(avail_tiers, key=lambda x: preferred_order.index(x) if x in preferred_order else 99)
        tier_labels = {t: f"{t} (Avg Corr: {tier_stats[t]:.2f})" for t in sorted_tiers}
        
        selected_tier_key = st.selectbox(
            "💰 選擇大戶門檻", 
            sorted_tiers, 
            format_func=lambda x: tier_labels[x]
        )
        
        st.info(f"已選擇: {selected_tier_key} 級距\n全市場平均相關係數: {tier_stats.get(selected_tier_key, 0):.2f}")

    tier_col_corr = f'Corr_Large_{selected_tier_key}'
    tier_col_pct = f'Large_Pct_{selected_tier_key}'
    tier_full_name = f'大戶持股比例_{selected_tier_key}'

    # --- 頁面分頁 ---
    tab1, tab2, tab3 = st.tabs(["📊 個股分析", "🏆 相關性排行", "🚀 連續買超掃描"])

    # === Tab 1: 個股分析 ===
    with tab1:
        st.header(f"個股籌碼檢測")
        
        option_list = rank_df['DisplayName'].tolist()
        default_idx = rank_df.index.get_loc("2330") if "2330" in rank_df.index else 0
        
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            selected = st.selectbox("搜尋股票", option_list, index=default_idx)
        
        stock_id = selected.split(' ')[0]
        
        if stock_id in rank_df.index:
            info = rank_df.loc[stock_id]
            st.subheader(f"{info['DisplayName']}")
            
            cols = st.columns(4)
            cols[0].metric("最新股價", f"{info['Price']:.2f}")
            val_corr = info.get(tier_col_corr, 0)
            cols[1].metric(f"大戶(>{selected_tier_key}) 連動", f"{val_corr:.2f}", delta="極高" if val_corr > 0.8 else None)
            cols[2].metric("董監連動", f"{info['Corr_Director']:.2f}")
            cols[3].metric("大股東連動", f"{info['Corr_Major']:.2f}")

            # 畫圖
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=raw_data['aligned_price'].index, y=raw_data['aligned_price'][stock_id], 
                                     name="股價", line=dict(color='#2c3e50', width=2)), secondary_y=False)
            
            if tier_full_name in raw_data['aligned_large_tiers']:
                tier_data = raw_data['aligned_large_tiers'][tier_full_name]
                if stock_id in tier_data.columns:
                    fig.add_trace(go.Scatter(x=tier_data.index, y=tier_data[stock_id], 
                                             name=f"大戶(>{selected_tier_key})", line=dict(color='#e74c3c', width=2)), secondary_y=True)
            
            fig.add_trace(go.Scatter(x=raw_data['calc_director'].index, y=raw_data['calc_director'][stock_id], 
                                     name="董監", line=dict(color='#f39c12', dash='dot')), secondary_y=True)
            
            fig.add_trace(go.Scatter(x=raw_data['calc_major'].index, y=raw_data['calc_major'][stock_id], 
                                     name="大股東", line=dict(color='#3498db', width=1)), secondary_y=True)

            fig.update_layout(title=f"{info['DisplayName']} 籌碼趨勢", height=450, hovermode="x unified")
            fig.update_yaxes(title_text="股價", secondary_y=False)
            fig.update_yaxes(title_text="持股比例 (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # === 修正：補回詳細資料表格 ===
            st.subheader("近 12 週詳細數據")
            recent_data = pd.DataFrame(index=raw_data['aligned_price'].index)
            recent_data['收盤價'] = raw_data['aligned_price'][stock_id]
            
            if tier_full_name in raw_data['aligned_large_tiers']:
                 recent_data[f'大戶(>{selected_tier_key})%'] = raw_data['aligned_large_tiers'][tier_full_name][stock_id]
            
            recent_data['董監(%)'] = raw_data['calc_director'][stock_id]
            recent_data['大股東(%)'] = raw_data['calc_major'][stock_id]
            
            display_recent = recent_data.tail(12).sort_index(ascending=False)
            display_recent.index = display_recent.index.strftime('%Y-%m-%d')
            st.dataframe(display_recent.style.format("{:.2f}"), use_container_width=True)

    # === Tab 2: 排行榜 ===
    with tab2:
        st.header(f"全市場籌碼排行榜")
        
        c1, c2 = st.columns(2)
        with c1:
            filter_mode = st.radio("篩選範圍", ["全市場", "僅限可轉債(CB)", "僅限股票期貨"], horizontal=True, key="rank_filter")
        with c2:
            sort_metric = st.selectbox("排序指標", [f"大戶(>{selected_tier_key}) Correlation", "董監持股 Correlation", "10%大股東 Correlation"])

        target_df = rank_df.copy()
        if "可轉債" in filter_mode and stock_filters['cb']:
            target_df = target_df[target_df.index.isin(stock_filters['cb'])]
        elif "股票期貨" in filter_mode and stock_filters['futures']:
            target_df = target_df[target_df.index.isin(stock_filters['futures'])]

        col_key = 'Corr_Major'
        if "大戶" in sort_metric: col_key = tier_col_corr
        elif "董監" in sort_metric: col_key = 'Corr_Director'
            
        top_df = target_df.sort_values(by=col_key, ascending=False).head(20)

        display_cols = ['DisplayName', 'Price', col_key, 'Director_Pct', 'Major_Pct']
        renamed = {'DisplayName': '股名', 'Price': '股價', col_key: '相關係數', 'Director_Pct': '董監%', 'Major_Pct': '大股東%'}
        if selected_tier_key:
            display_cols.insert(3, tier_col_pct)
            renamed[tier_col_pct] = f'大戶(>{selected_tier_key})%'

        subset_cols = list(renamed.values())
        if '股名' in subset_cols: subset_cols.remove('股名')

        st.dataframe(
            top_df[display_cols].rename(columns=renamed)
            .style.background_gradient(subset=['相關係數'], cmap='Reds')
            .format("{:.2f}", subset=subset_cols),
            use_container_width=True,
            height=800
        )

    # === Tab 3: 連續買超掃描 (邏輯修正版) ===
    with tab3:
        st.header(f"🚀 連續兩週大戶買超掃描")
        st.markdown(f"找出大戶 (>{selected_tier_key}) 近期 **連續兩週** 加碼的股票。")
        
        c_filter, c_thres = st.columns(2)
        with c_filter:
            scan_scope = st.radio("掃描範圍", ["全市場", "僅限可轉債(CB)", "僅限股票期貨"], horizontal=True, key="scan_filter")
        with c_thres:
            growth_thres = st.selectbox("🔥 總成長門檻 (兩週合計)", [1, 3, 5, 10], format_func=lambda x: f"累計增加 > {x}%")

        if st.button("開始掃描"):
            # === 關鍵修正：使用原始資料 (Raw Data) ===
            # 不使用 aligned_large_tiers (因為會 ffill)，改用 raw_large_tiers
            raw_large_tiers = raw_data.get('raw_large_tiers', {})
            
            if tier_full_name in raw_large_tiers:
                raw_ts = raw_large_tiers[tier_full_name]
                
                # 取出最後 3 筆 "真實" 數據
                # raw_ts 的 index 是日期，columns 是股票代號
                if len(raw_ts) < 3:
                    st.error("歷史資料不足，無法掃描。")
                else:
                    last_3 = raw_ts.iloc[-3:]
                    
                    # T (最新), T-1, T-2
                    w_curr = last_3.iloc[-1]
                    w_prev = last_3.iloc[-2]
                    w_prev2 = last_3.iloc[-3]
                    
                    # 邏輯: 持續增長且總量達標
                    # 這裡會自動忽略 NaN
                    cond_continuous = (w_curr > w_prev) & (w_prev > w_prev2)
                    diff = w_curr - w_prev2
                    cond_magnitude = diff >= growth_thres
                    
                    # 篩選出的股票代號
                    candidates = raw_ts.columns[cond_continuous & cond_magnitude]
                    
                    # 轉為 DataFrame
                    scan_df = rank_df.loc[rank_df.index.intersection(candidates)].copy()
                    
                    # 範圍篩選
                    if "可轉債" in scan_scope and stock_filters['cb']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['cb'])]
                    elif "股票期貨" in scan_scope and stock_filters['futures']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['futures'])]
                    
                    if scan_df.empty:
                        st.warning(f"⚠️ 在此條件下 (連續2週增長 & 總和>{growth_thres}%) 未發現標的。")
                    else:
                        st.success(f"🎉 掃描完成！共發現 {len(scan_df)} 檔潛力股。")
                        
                        # 準備顯示資料
                        res_df = scan_df[['DisplayName', 'Price', tier_col_corr]].copy()
                        # 補上這三週的真實數值
                        res_df['W-2(%)'] = w_prev2[scan_df.index]
                        res_df['W-1(%)'] = w_prev[scan_df.index]
                        res_df['Current(%)'] = w_curr[scan_df.index]
                        res_df['兩週增幅'] = diff[scan_df.index]
                        
                        res_df = res_df.sort_values('兩週增幅', ascending=False)
                        
                        renamed = {'DisplayName': '股名', 'Price': '最新股價', tier_col_corr: '相關係數', '兩週增幅': '累計增幅%'}
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
                st.error("大戶資料異常，無法掃描。")
else:
    st.info("請將所有 CSV 檔案放置於同一目錄下。")