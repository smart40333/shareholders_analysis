import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 頁面設定 ---
st.set_page_config(page_title="內部人籌碼雷達 (V7 穩定版)", layout="wide")

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
            
            if key == 'large':
                df.columns = [c.strip() for c in df.columns]
                if 'stock_id' not in df.columns: df.rename(columns={df.columns[0]: 'stock_id'}, inplace=True)
                if 'date' not in df.columns: df.rename(columns={df.columns[1]: 'date'}, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'])
                df['stock_id'] = df['stock_id'].astype(str)
                
                valid_tiers = [c for c in df.columns if any(x in c for x in ['3000萬', '4000萬', '5000萬', '1億'])]
                
                large_dfs = {}
                for tier in valid_tiers:
                    # 原始資料 Pivot
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

        # C. 對齊週線 (圖表用)
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
        def slice_data(df):
            return df.loc[START_DATE:]

        final_price = slice_data(aligned_price)
        final_major = slice_data((major_weekly / issued_weekly) * 100)
        final_director = slice_data(director_weekly)
        
        final_large_chart = {} 
        for t, d in aligned_large_tiers.items():
            final_large_chart[t] = slice_data(d)

        # G. 定義全市場交集 (Universe)
        valid_stocks = final_price.dropna(axis=1, how='all').columns
        common_stocks = valid_stocks.intersection(final_major.columns).intersection(final_director.columns)
        if final_large_chart:
             first_tier = list(final_large_chart.values())[0]
             common_stocks = common_stocks.intersection(first_tier.columns)

        # === 關鍵修正：依照交集裁切資料 ===
        # 必須確保所有 DataFrame 的欄位數量與順序完全一致，才能塞入 rank_df
        final_price = final_price[common_stocks]
        final_major = final_major[common_stocks]
        final_director = final_director[common_stocks]
        for t in final_large_chart:
            final_large_chart[t] = final_large_chart[t][common_stocks]

        # H. 計算 Correlation (使用原始頻率)
        
        # 1. 大股東 & 董監
        price_for_major = raw_dfs['price'].reindex(raw_dfs['major'].index, method='ffill').loc[START_DATE:]
        major_raw_sliced = raw_dfs['major'].loc[START_DATE:]
        issued_raw_sliced = raw_dfs['issued'].loc[START_DATE:]
        major_pct_raw = (major_raw_sliced / issued_raw_sliced) * 100
        
        # 只算 common_stocks
        corr_major = major_pct_raw[common_stocks].corrwith(price_for_major[common_stocks])
        
        director_raw_sliced = raw_dfs['director_pct'].loc[START_DATE:]
        price_for_director = raw_dfs['price'].reindex(director_raw_sliced.index, method='ffill').loc[START_DATE:]
        corr_director = director_raw_sliced[common_stocks].corrwith(price_for_director[common_stocks])
        
        # 2. 大戶持股
        corr_large_dict = {}
        tier_stats = {}
        
        if 'large_tiers' in raw_dfs:
            for tier, df_raw in raw_dfs['large_tiers'].items():
                df_raw_sliced = df_raw.loc[START_DATE:]
                price_for_tier = raw_dfs['price'].reindex(df_raw_sliced.index, method='ffill')
                
                # 計算 Correlation
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
            'Price': final_price.iloc[-1], # 因為已裁切，這裡長度會是 2033 (正確)
            'Major_Pct': final_major.iloc[-1],
            'Director_Pct': final_director.iloc[-1],
            'Corr_Major': corr_major, # Series 會自動對齊 index
            'Corr_Director': corr_director
        }).set_index('StockID')

        # 加入大戶指標
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
            'tier_stats': tier_stats 
        }
        
        return data_dict, rank_df, filters

    except Exception as e:
        import traceback
        st.error(f"資料處理發生嚴重錯誤: {e}")
        st.text(traceback.format_exc())
        return None, None, None

# --- 主程式 ---

st.title("🎯 內部人籌碼雷達 (V7 穩定版)")

data_res = load_data_and_calculate_metrics()

# 修正 AttributeError: 必須先檢查 raw_data 是否為 None
if data_res and data_res[0] is not None:
    raw_data, rank_df, stock_filters = data_res
    tier_stats = raw_data.get('tier_stats', {})
    
    # --- Sidebar ---
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
        
        st.info(f"已選擇: {selected_tier_key}\n全市場平均相關係數: {tier_stats.get(selected_tier_key, 0):.2f}")

    tier_col_corr = f'Corr_Large_{selected_tier_key}'
    tier_col_pct = f'Large_Pct_{selected_tier_key}'
    tier_full_name = f'大戶持股比例_{selected_tier_key}'

    # --- Tabs ---
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

            # Chart
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

            fig.update_layout(title=f"{info['DisplayName']} 走勢圖", height=450, hovermode="x unified")
            fig.update_yaxes(title_text="股價", secondary_y=False)
            fig.update_yaxes(title_text="持股比例 (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Table
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

    # === Tab 2: Ranking ===
    with tab2:
        st.header(f"全市場排行榜")
        
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

    # === Tab 3: Scanner ===
    with tab3:
        st.header(f"🚀 連續買超掃描")
        st.markdown(f"掃描條件：大戶 (>{selected_tier_key}) 連續 2 週買進，且總增幅達標。")
        
        c_filter, c_thres = st.columns(2)
        with c_filter:
            scan_scope = st.radio("掃描範圍", ["全市場", "僅限可轉債(CB)", "僅限股票期貨"], horizontal=True, key="scan_filter")
        with c_thres:
            growth_thres = st.selectbox("🔥 總成長門檻", [1, 3, 5, 10], format_func=lambda x: f"累計增加 > {x}%")

        if st.button("開始掃描"):
            raw_large_tiers = raw_data.get('raw_large_tiers', {})
            
            if tier_full_name in raw_large_tiers:
                raw_ts = raw_large_tiers[tier_full_name]
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
                    
                    if "可轉債" in scan_scope and stock_filters['cb']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['cb'])]
                    elif "股票期貨" in scan_scope and stock_filters['futures']:
                        scan_df = scan_df[scan_df.index.isin(stock_filters['futures'])]
                    
                    if scan_df.empty:
                        st.warning(f"⚠️ 無符合標的。")
                    else:
                        st.success(f"🎉 發現 {len(scan_df)} 檔潛力股！")
                        
                        res_df = scan_df[['DisplayName', 'Price', tier_col_corr]].copy()
                        res_df['W-2(%)'] = w_prev2[scan_df.index]
                        res_df['W-1(%)'] = w_prev[scan_df.index]
                        res_df['Current(%)'] = w_curr[scan_df.index]
                        res_df['Total Growth'] = diff[scan_df.index]
                        
                        res_df = res_df.sort_values('Total Growth', ascending=False)
                        
                        renamed = {'DisplayName': '股名', 'Price': '股價', tier_col_corr: '相關係數', 'Total Growth': '累計增幅%'}
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
    st.info("請將所有 CSV 檔案放置於同一目錄下。")