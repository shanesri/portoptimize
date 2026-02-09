import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from scipy.optimize import minimize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- CUSTOM CSS (From your Macro Dashboard) ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
    }
    
    .stMainBlockContainer {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    .nav-item {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-weight: 500;
        color: #8b949e;
        font-size: 16px;
    }
    
    .nav-item a {
        color: inherit;
        text-decoration: none;
        display: block;
        width: 100%;
    }

    .nav-active {
        background-color: #1f2937;
        color: #ffffff !important;
        border-left: 4px solid #4589ff;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }

    /* Custom Danger Red for Warning Bubbles */
    div[data-testid="stNotification"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    div[data-testid="stNotification"] svg {
        fill: white !important;
    }

    /* Card Styling for Results */
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 16px;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-gain {
        color: #26a69a;
    }
    .delta-loss {
        color: #ef5350;
    }
    .delta-neutral {
        color: #ffffff;
    }
    
    /* Special VAR/CVAR Card styling */
    .risk-card {
        background-color: #251212;
        border: 1px solid #632a2a;
        text-align: left;
        padding: 22px;
        min-height: 140px;
    }
    .cvar-card {
        background-color: #350a0a;
        border: 1px solid #8e1e1e;
    }
    .var-text {
        color: #ffffff;
        font-size: 17px;
        line-height: 1.4;
    }
    .var-highlight {
        font-weight: 700;
        color: #ef5350;
    }
    
    /* Success/Failure Card styling */
    .success-card {
        background-color: #102a1e;
        border: 1px solid #1e6341;
    }
    .failure-card {
        background-color: #2a1010;
        border: 1px solid #631e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    st.markdown('<div class="nav-item"><a href="https://shanegreeting.streamlit.app/" target="_blank">👋 Greeting</a></div>', unsafe_allow_html=True)
    # Marked as active since we are on this tool
    st.markdown('<div class="nav-item nav-active">📊 Portfolio Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><a href="https://shanesri.com" target="_blank">🔗 Creator Info</a></div>', unsafe_allow_html=True)


# --- MAIN APP CONTENT ---

st.title("📊 Portfolio Optimizer & Efficient Frontier")
st.markdown("""
This app optimizes a portfolio for **Maximum Sharpe Ratio**, **Minimum Volatility**, and specific **Target Return/Risk** goals using the Efficient Frontier.
""")

# --- HELPER FUNCTIONS ---

@st.cache_data
def get_ticker_info(ticker_list):
    """Fetches short names for tickers to display in the UI."""
    info_map = {}
    for t in ticker_list:
        try:
            # Fetching shortName to ensure we get the full asset title
            name = yf.Ticker(t).info.get('shortName', t)
            info_map[t] = name
        except:
            info_map[t] = t
    return info_map

# --- SECTION 1: DATA & CALCULATIONS (Cached) ---

@st.cache_data
def get_data(tickers, start, end):
    """Fetches closing prices and calculates daily returns."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Using 'Close' as per your update note regarding yfinance auto_adjust
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        
        # If only one ticker is downloaded, yfinance returns a Series or single-column DF
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        daily_returns = data.pct_change().dropna()
        return daily_returns
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_metrics(daily_returns, return_mode, investor_views):
    """Calculates annualized average returns and covariance matrix."""
    cov_matrix = daily_returns.cov() * 252

    if return_mode == 'historical':
        avg_returns = daily_returns.mean() * 252
    else:
        # Match order of tickers to the dataframe columns
        avg_returns = pd.Series([investor_views.get(t, 0.0) for t in daily_returns.columns], index=daily_returns.columns)

    return avg_returns, cov_matrix

def get_portfolio_stats(weights, avg_returns, cov_matrix, rf_rate):
    """Calculates Portfolio Return, Volatility, and Sharpe Ratio."""
    weights = np.array(weights)
    ret = np.sum(avg_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - rf_rate) / vol
    return ret, vol, sharpe

# --- SECTION 2: OPTIMIZATION ENGINE ---

def optimize_portfolio(avg_returns, cov_matrix, rf_rate, tickers, use_constraints, min_w, max_w, target_ret_val, target_vol_val):
    num_assets = len(tickers)
    init_guess = [1/num_assets] * num_assets

    # 1. Define Bounds
    if use_constraints:
        bounds = tuple((min_w, max_w) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))

    # Constraint: Sum of weights = 1
    cons_sum_weight = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Objective Functions
    def min_vol_objective(weights):
        return get_portfolio_stats(weights, avg_returns, cov_matrix, rf_rate)[1]

    def max_sharpe_objective(weights):
        return -get_portfolio_stats(weights, avg_returns, cov_matrix, rf_rate)[2] # Negative for minimization

    def max_return_objective(weights):
        return -get_portfolio_stats(weights, avg_returns, cov_matrix, rf_rate)[0]

    # 2. Standard Optimizations
    opt_sharpe = minimize(max_sharpe_objective, init_guess, method='SLSQP', bounds=bounds, constraints=[cons_sum_weight])
    opt_vol = minimize(min_vol_objective, init_guess, method='SLSQP', bounds=bounds, constraints=[cons_sum_weight])

    # 3. Target Return Optimization
    cons_target_ret = {'type': 'eq', 'fun': lambda x: get_portfolio_stats(x, avg_returns, cov_matrix, rf_rate)[0] - target_ret_val}
    opt_target_ret = minimize(min_vol_objective, init_guess, method='SLSQP', bounds=bounds, constraints=[cons_sum_weight, cons_target_ret])

    # 4. Target Risk Optimization
    cons_target_vol = {'type': 'eq', 'fun': lambda x: get_portfolio_stats(x, avg_returns, cov_matrix, rf_rate)[1] - target_vol_val}
    opt_target_vol = minimize(max_return_objective, init_guess, method='SLSQP', bounds=bounds, constraints=[cons_sum_weight, cons_target_vol])

    return opt_sharpe.x, opt_vol.x, opt_target_ret, opt_target_vol

# --- SECTION 3: CONFIGURATION (MAIN PAGE) ---

# Initialize Session State for Tickers if not present
if 'tickers_list' not in st.session_state:
    st.session_state.tickers_list = ['VTI', 'TLT', 'IEF', 'GLD', 'PDBC']

st.divider()

# --- STEP 1: ASSET CONFIGURATION ---
st.subheader("1. Asset Configuration")
c1_A, c1_B = st.columns([1, 2])

with c1_A:
    # -- Ticker Input Logic --
    c_in, c_add = st.columns([2, 1])
    with c_in:
        new_ticker = st.text_input("Add Ticker", placeholder="Try AAPL", label_visibility="collapsed").strip().upper()
    with c_add:
        if st.button("Add", type="secondary", use_container_width=True):
            if new_ticker and new_ticker not in st.session_state.tickers_list:
                st.session_state.tickers_list.append(new_ticker)
                st.rerun()
    
    # -- Data Range Logic --
    years_back = st.selectbox("Data Range (Years)", [1, 3, 5, 10], index=2)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years_back)
    st.caption(f"From **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")

with c1_B:
    # -- Ticker Table --
    if st.session_state.tickers_list:
        tickers = st.session_state.tickers_list
        ticker_names = get_ticker_info(tickers)
        
        df_data = []
        for t in tickers:
            df_data.append({
                "Active": True, 
                "Ticker": t, 
                "Name": ticker_names.get(t, t)
            })
        df = pd.DataFrame(df_data)

        column_config = {
            "Active": st.column_config.CheckboxColumn("Active", help="Uncheck to remove", default=True, width="small"),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
            "Name": st.column_config.TextColumn("Asset Name", disabled=True),
        }

        edited_df = st.data_editor(
            df, 
            column_config=column_config, 
            use_container_width=True, 
            hide_index=True, 
            key="ticker_editor"
        )

        # Logic to remove unchecked items
        current_active_tickers = edited_df[edited_df["Active"] == True]["Ticker"].tolist()
        
        # If the list size changed, update state and rerun
        if len(current_active_tickers) != len(st.session_state.tickers_list):
            st.session_state.tickers_list = current_active_tickers
            st.rerun()
            
        tickers = st.session_state.tickers_list
    else:
        st.warning("No assets selected.")
        tickers = []

st.divider()

# --- STEP 2: PARAMS & TARGETS ---
st.subheader("2. Params & Targets")

c2_1, c2_2, c2_3 = st.columns(3)

with c2_1:
    rf_input = st.number_input("Risk-Free Rate ($R_f$) (%)", value=4.00, step=0.10, format="%.2f")
    rf_rate = rf_input / 100.0

with c2_2:
    t_ret_input = st.number_input("Target Return (%)", value=7.50, step=0.10, format="%.2f")
    target_ret_val = t_ret_input / 100.0

with c2_3:
    t_vol_input = st.number_input("Target Risk (%)", value=8.00, step=0.10, format="%.2f")
    target_vol_val = t_vol_input / 100.0

# Investor Views Section
st.markdown("")
use_investor_views = st.toggle("Use Investor Views", value=False)
return_mode = "investor_views" if use_investor_views else "historical"
investor_views = {}

st.caption("Enter your expected annual returns below. Defaults are set to 7.00%.")
# Dynamic columns for views to save space
cols_views = st.columns(min(len(tickers), 5)) if tickers else [st.container()]

for i, t in enumerate(tickers):
    col_idx = i % 5
    with cols_views[col_idx]:
        val_input = st.number_input(
            f"{t} (%)", 
            value=7.00, 
            step=0.50, 
            key=f"view_{t}", 
            format="%.2f",
            disabled=not use_investor_views  # Always visible, just disabled if toggle off
        )
        investor_views[t] = val_input / 100.0

st.divider()

# --- STEP 3: CONSTRAINTS ---
st.subheader("3. Constraints")

c3_1, c3_2 = st.columns([1, 3])

with c3_1:
    use_constraints = st.toggle("Use Weight Constraints", value=True)

with c3_2:
    # Sliders in % (0-100), disabled if toggle is False
    c3_sl1, c3_sl2 = st.columns(2)
    with c3_sl1:
        min_w_pct = st.slider("Min Weight per Asset (%)", 0, 100, 5, 1, disabled=not use_constraints)
    with c3_sl2:
        max_w_pct = st.slider("Max Weight per Asset (%)", 0, 100, 100, 1, disabled=not use_constraints)

    # Convert to decimals
    min_w = min_w_pct / 100.0
    max_w = max_w_pct / 100.0

st.markdown("---")
run_btn = st.button("Run Optimization", type="primary", use_container_width=True)

st.divider()

# --- SECTION 4: MAIN EXECUTION & VISUALIZATION ---

if run_btn:
    if not tickers:
        st.error("Please enter at least one valid ticker.")
    else:
        with st.spinner("Fetching data and optimizing..."):
            # 1. Fetch Data
            daily_returns = get_data(tickers, start_date, end_date)
            
            if daily_returns.empty:
                st.error("No data found for the selected tickers/dates.")
            else:
                # 2. Calculate Metrics
                avg_returns, cov_matrix = calculate_metrics(daily_returns, return_mode, investor_views)

                # 3. Optimize
                w_sharpe, w_vol, res_target_ret, res_target_vol = optimize_portfolio(
                    avg_returns, cov_matrix, rf_rate, tickers, use_constraints, min_w, max_w, target_ret_val, target_vol_val
                )

                # 4. Display Results
                
                # --- A. Weights Table ---
                st.subheader("Results: Optimized Weights")
                
                # Check feasibility
                stats_target_ret = get_portfolio_stats(res_target_ret.x, avg_returns, cov_matrix, rf_rate)
                is_feasible_ret = res_target_ret.success and np.isclose(stats_target_ret[0], target_ret_val, atol=0.005)

                stats_target_vol = get_portfolio_stats(res_target_vol.x, avg_returns, cov_matrix, rf_rate)
                is_feasible_vol = res_target_vol.success and np.isclose(stats_target_vol[1], target_vol_val, atol=0.005)

                weights_data = {
                    'Ticker': tickers,
                    'Max Sharpe': (w_sharpe * 100).round(2),
                    'Min Vol': (w_vol * 100).round(2),
                }

                if is_feasible_ret:
                    weights_data[f'Target Ret ({target_ret_val:.1%})'] = (res_target_ret.x * 100).round(2)
                if is_feasible_vol:
                    weights_data[f'Target Risk ({target_vol_val:.1%})'] = (res_target_vol.x * 100).round(2)

                df_weights = pd.DataFrame(weights_data)
                # Set index to start from 1 instead of 0
                df_weights.index = range(1, len(df_weights) + 1)
                st.dataframe(df_weights, use_container_width=True)

                if not is_feasible_ret:
                    st.warning(f"⚠️ Target Return {target_ret_val:.1%} is not feasible with current constraints.")
                if not is_feasible_vol:
                    st.warning(f"⚠️ Target Risk {target_vol_val:.1%} is not feasible with current constraints.")

                # --- B. Performance Summary ---
                st.subheader("Performance Summary")
                stats_sharpe = get_portfolio_stats(w_sharpe, avg_returns, cov_matrix, rf_rate)
                stats_vol = get_portfolio_stats(w_vol, avg_returns, cov_matrix, rf_rate)
                
                sum_col1, sum_col2 = st.columns(2)
                
                with sum_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Max Sharpe Ratio</div>
                        <div class="metric-value">{stats_sharpe[2]:.2f}</div>
                        <div class="metric-delta delta-gain">Ret: {stats_sharpe[0]:.1%} | Vol: {stats_sharpe[1]:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Minimum Volatility</div>
                        <div class="metric-value">{stats_vol[1]:.1%}</div>
                        <div class="metric-delta delta-gain">Ret: {stats_vol[0]:.1%} | Sharpe: {stats_vol[2]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- C. Plots ---
                st.subheader("Visualizations")
                
                # 1. Efficient Frontier (Full Width)
                st.markdown("### Efficient Frontier")
                
                # 1. Monte Carlo Simulation
                n_sim = 1000
                sim_rets = []
                sim_vols = []
                
                for _ in range(n_sim):
                    w = np.random.random(len(tickers))
                    w /= np.sum(w)
                    if use_constraints:
                        if np.any(w < min_w) or np.any(w > max_w):
                            continue
                    
                    r, v, _ = get_portfolio_stats(w, avg_returns, cov_matrix, rf_rate)
                    sim_rets.append(r)
                    sim_vols.append(v)
                
                # 2. Prepare DataFrames for Altair (SIMPLIFIED SINGLE DATASET)
                
                # Cloud Data
                df_cloud = pd.DataFrame({
                    'Volatility': sim_vols, 
                    'Return': sim_rets,
                    'Type': 'Random'
                })
                
                # Special Points Data
                special_points_list = [
                    {'Volatility': stats_sharpe[1], 'Return': stats_sharpe[0], 'Type': 'Max Sharpe'},
                    {'Volatility': stats_vol[1], 'Return': stats_vol[0], 'Type': 'Min Volatility'}
                ]
                
                if is_feasible_ret:
                    special_points_list.append({'Volatility': stats_target_ret[1], 'Return': stats_target_ret[0], 'Type': 'Target Return'})
                
                if is_feasible_vol:
                    special_points_list.append({'Volatility': stats_target_vol[1], 'Return': stats_target_vol[0], 'Type': 'Target Risk'})
                    
                df_special = pd.DataFrame(special_points_list)
                
                # Combine into one main DataFrame
                df_all = pd.concat([df_cloud, df_special], ignore_index=True)

                # 3. Simple Altair Chart (No Layers, Single Mark Point)
                chart = alt.Chart(df_all).mark_point(filled=True, size=60).encode(
                    x=alt.X('Volatility', type='quantitative', title='Annualized Volatility (Risk)', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
                    y=alt.Y('Return', type='quantitative', title='Annualized Return', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
                    color=alt.Color('Type', legend=alt.Legend(title="Type")),
                    tooltip=['Type', alt.Tooltip('Volatility', format='.2%'), alt.Tooltip('Return', format='.2%')]
                ).properties(
                    height=500
                )
                
                st.altair_chart(chart, use_container_width=True)

                st.divider()

                # Split columns for Asset Allocation and Correlation Matrix
                col_alloc, col_corr = st.columns(2)
                
                with col_alloc:
                    st.subheader("Asset Allocation")
                    
                    # Create Tabs for different strategies
                    tab1, tab2, tab3, tab4 = st.tabs(["Max Sharpe", "Min Vol", "Target Return", "Target Risk"])
                    
                    # Helper to draw pie chart
                    def draw_pie(weights, title_suffix=""):
                        df_pie = pd.DataFrame({'Ticker': tickers, 'Weight': weights * 100})
                        # Filter out very small weights
                        df_pie = df_pie[df_pie['Weight'] > 0.01]
                        
                        base = alt.Chart(df_pie).encode(
                            theta=alt.Theta("Weight", stack=True)
                        )
                        
                        pie = base.mark_arc(innerRadius=60).encode(
                            color=alt.Color("Ticker"),
                            order=alt.Order("Weight", sort="descending"),
                            tooltip=["Ticker", alt.Tooltip("Weight", format=".2f")]
                        )
                        
                        text = base.mark_text(radius=140).encode(
                            text=alt.Text("Weight", format=".1f"),
                            order=alt.Order("Weight", sort="descending"),
                            color=alt.value("white")
                        )
                        
                        st.altair_chart((pie + text).properties(height=350), use_container_width=True)

                    with tab1:
                        draw_pie(w_sharpe)
                        
                    with tab2:
                        draw_pie(w_vol)
                        
                    with tab3:
                        if is_feasible_ret:
                            draw_pie(res_target_ret.x)
                        else:
                            st.warning(f"Target Return {target_ret_val:.1%} not feasible.")
                            
                    with tab4:
                        if is_feasible_vol:
                            draw_pie(res_target_vol.x)
                        else:
                            st.warning(f"Target Risk {target_vol_val:.1%} not feasible.")

                with col_corr:
                    st.subheader("Asset Correlation Matrix")
                    # Correlation Matrix as Styled Table
                    corr_matrix = daily_returns.corr()
                    styled_corr = corr_matrix.style.background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1).format("{:.2f}")
                    st.dataframe(styled_corr, use_container_width=True)
else:
    st.info("👈 Configure assets above and click **Run Optimization**.")
