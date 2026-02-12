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
    st.markdown('<div class="nav-item nav-active">⚖️ Weight Optimizer (Efficient Frontier)</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><a href="https://mcsbyshane.streamlit.app/" target="_blank">📈 Portfolio Simulator (Monte Carlo)</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><a href="https://shanesri.com" target="_blank">🔗 Creator Info</a></div>', unsafe_allow_html=True)


# --- MAIN APP CONTENT ---

st.title("⚖️ Weight Optimizer (Efficient Frontier)")
st.markdown("""
Hey there! I'm Shane from Thailand :) I built this app using Python, Streamlit, and GitHub to bring some of the CFA curriculum's finance concepts to life!

**What this app does:** It calculates the mathematically optimal weight of assets in a portfolio to maximize your Sharpe Ratio, minimize volatility, or hit a specific target return/risk using the Efficient Frontier.

**Inputs & Outcomes:**
* **Inputs:** Your chosen tickers, historical data range, risk-free rate, weight constraints, and optional custom expected returns.
* **Outcomes:** The exact optimal percentage weightings for each asset, a performance summary, and a visual plot of the Efficient Frontier.

**Key Features:**
* 📊 **5,000 Simulations** to visually map out the random portfolio universe.
* 🎯 **Target Searching:** Automatically find weights for a specific target risk or return.
* 🧠 **Investor Views:** Override historical data with your own future return expectations.
* 🚧 **Custom Constraints:** Set practical minimum and maximum weight limits for your assets.
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
    # Optimize to minimize risk for the target return
    cons_target_ret = {'type': 'eq', 'fun': lambda x: get_portfolio_stats(x, avg_returns, cov_matrix, rf_rate)[0] - target_ret_val}
    opt_target_ret = minimize(min_vol_objective, init_guess, method='SLSQP', bounds=bounds, constraints=[cons_sum_weight, cons_target_ret])

    # 4. Target Risk Optimization
    # Optimize to maximize return for the target risk
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
        new_ticker = st.text_input(
            "Add Ticker", 
            placeholder="Try AAPL", 
            help="Tickers can be found exactly as they appear on Yahoo Finance.",
            label_visibility="visible"
        ).strip().upper()
    with c_add:
        # Aligning the button with the visible label text input
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Add", type="secondary", use_container_width=True):
            if new_ticker and new_ticker not in st.session_state.tickers_list:
                st.session_state.tickers_list.append(new_ticker)
                st.rerun()
    
    # -- Data Range Logic --
    years_back = st.selectbox(
        "Data Range (Years)", 
        [1, 3, 5, 10], 
        index=2,
        help="This sets the historical timeframe for pulling daily data from Yahoo Finance. We use this to calculate asset correlations and baseline average annual returns."
    )
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
st.subheader("2. Optimization Parameters & Targets")

# Moved Risk-Free Rate to its own line
rf_input = st.number_input(
    "Risk-Free Rate ($R_f$) (%)", 
    value=4.00, 
    step=0.10, 
    format="%.2f",
    help="The theoretical rate of return of an investment with zero risk. This is used as the baseline to calculate the Sharpe Ratio."
)
rf_rate = rf_input / 100.0

st.markdown("<br>", unsafe_allow_html=True)

# Target Return and Target Risk grouped together on a lower line
c2_1, c2_2 = st.columns(2)

with c2_1:
    use_target_ret = st.toggle("Enable Target Return", value=True)
    t_ret_input = st.number_input(
        "Target Return (%)", 
        value=7.50, step=0.10, format="%.2f",
        disabled=not use_target_ret,
        help="Finds the optimal asset weights to achieve a specific target return with the least risk. Returns 'N/A' if the target is unachievable with current constraints."
    )
    target_ret_val = t_ret_input / 100.0

with c2_2:
    use_target_vol = st.toggle("Enable Target Risk", value=True)
    t_vol_input = st.number_input(
        "Target Risk (%)", 
        value=8.00, step=0.10, format="%.2f",
        disabled=not use_target_vol,
        help="Finds the optimal asset weights to maximize return for a specific target risk level. Returns 'N/A' if the target is unachievable with current constraints."
    )
    target_vol_val = t_vol_input / 100.0

# Investor Views Section
st.markdown("")
use_investor_views = st.toggle(
    "Use Investor Views", 
    value=False,
    help="Override historical averages by injecting your own expected annualized returns for each asset."
)
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
    use_constraints = st.toggle(
        "Use Weight Constraints", 
        value=True,
        help="Set the minimum and maximum allocation percentage you want to allow for each asset. If the constraints aren't mathematically practical (e.g., minimums sum to more than 100%), the optimization will fail and show an error."
    )

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

# --- Constraint Validation ---
constraints_valid = True
if use_constraints and len(tickers) > 0:
    num_assets = len(tickers)
    if min_w_pct * num_assets > 100:
        st.error(f"🚨 **Constraint Error:** Minimum weight ({min_w_pct}%) × {num_assets} assets = {min_w_pct * num_assets}%. This exceeds 100% total allocation.")
        constraints_valid = False
    elif max_w_pct * num_assets < 100:
        st.error(f"🚨 **Constraint Error:** Maximum weight ({max_w_pct}%) × {num_assets} assets = {max_w_pct * num_assets}%. The portfolio can never reach a 100% total allocation.")
        constraints_valid = False
    elif min_w_pct > max_w_pct:
        st.error("🚨 **Constraint Error:** Minimum weight cannot exceed maximum weight.")
        constraints_valid = False

st.markdown("---")
run_btn = st.button("Run Optimization", type="primary", use_container_width=True)

st.divider()

# --- SECTION 4: MAIN EXECUTION & VISUALIZATION ---

if run_btn:
    if not tickers:
        st.error("Please enter at least one valid ticker.")
    elif use_constraints and not constraints_valid:
        st.error("⚠️ Please fix the weight constraints above before running the optimization.")
    else:
        with st.spinner("Fetching data and optimizing..."):
            # 1. Fetch Data
            daily_returns = get_data(tickers, start_date, end_date)
            
            if daily_returns.empty:
                st.error("No data found for the selected tickers/dates.")
            else:
                # 1b. Check if there's enough historical data for the selected timeframe
                expected_days = years_back * 252 # Approx trading days per year
                actual_days = len(daily_returns)
                
                if actual_days < expected_days * 0.9: # Give a 10% tolerance for holidays/weekends
                    st.error(f"🚨 **Data Error:** Not enough historical data found for the full {years_back}-year timeframe. (Expected ~{expected_days} trading days, but found {actual_days}). One or more of your selected assets might be too new, which drops the aligned historical data and makes the optimization inappropriate. Please reduce the timeframe or remove newer assets.")
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

                    # Setup DataFrame dict with % formatting applied directly
                    weights_data = {
                        'Ticker': tickers,
                        'Max Sharpe': [f"{(w * 100):.2f}%" for w in w_sharpe],
                        'Min Volatility': [f"{(w * 100):.2f}%" for w in w_vol],
                    }

                    if use_target_ret:
                        if is_feasible_ret:
                            weights_data[f'Target Ret ({target_ret_val:.1%})'] = [f"{(w * 100):.2f}%" for w in res_target_ret.x]
                        else:
                            weights_data[f'Target Ret ({target_ret_val:.1%})'] = ["N/A"] * len(tickers)
                            
                    if use_target_vol:
                        if is_feasible_vol:
                            weights_data[f'Target Risk ({target_vol_val:.1%})'] = [f"{(w * 100):.2f}%" for w in res_target_vol.x]
                        else:
                            weights_data[f'Target Risk ({target_vol_val:.1%})'] = ["N/A"] * len(tickers)

                    df_weights = pd.DataFrame(weights_data)
                    # Set index to start from 1 instead of 0
                    df_weights.index = range(1, len(df_weights) + 1)
                    st.dataframe(df_weights, use_container_width=True)

                    if use_target_ret and not is_feasible_ret:
                        st.warning(f"⚠️ Target Return {target_ret_val:.1%} is not feasible with current constraints.")
                    if use_target_vol and not is_feasible_vol:
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
                            <div class="metric-delta delta-gain">Return: {stats_sharpe[0]:.1%} | Vol: {stats_sharpe[1]:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with sum_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Minimum Volatility</div>
                            <div class="metric-value">{stats_vol[1]:.1%}</div>
                            <div class="metric-delta delta-gain">Return: {stats_vol[0]:.1%} | Sharpe: {stats_vol[2]:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # --- C. Plots ---
                    st.subheader("Visualizations")
                    
                    # 1. Efficient Frontier (Full Width)
                    st.markdown("### Efficient Frontier")
                    
                    # 1. Monte Carlo Simulation (Targeting exactly 5000 valid portfolios)
                    n_target_sims = 5000
                    max_attempts = 250000  # Safety net to prevent infinite loops if constraints are extremely tight
                    sim_rets = []
                    sim_vols = []
                    attempts = 0
                    
                    while len(sim_rets) < n_target_sims and attempts < max_attempts:
                        attempts += 1
                        w = np.random.random(len(tickers))
                        w /= np.sum(w)
                        if use_constraints:
                            if np.any(w < min_w) or np.any(w > max_w):
                                continue
                        
                        r, v, _ = get_portfolio_stats(w, avg_returns, cov_matrix, rf_rate)
                        sim_rets.append(r)
                        sim_vols.append(v)
                    
                    if len(sim_rets) < n_target_sims:
                        st.warning(f"Only found {len(sim_rets)} valid portfolios after {max_attempts} random attempts due to tight constraints.")
                    
                    # 2. Prepare DataFrames for Altair
                    
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
                    
                    if use_target_ret and is_feasible_ret:
                        special_points_list.append({'Volatility': stats_target_ret[1], 'Return': stats_target_ret[0], 'Type': 'Target Return'})
                    
                    if use_target_vol and is_feasible_vol:
                        special_points_list.append({'Volatility': stats_target_vol[1], 'Return': stats_target_vol[0], 'Type': 'Target Risk'})
                        
                    df_special = pd.DataFrame(special_points_list)
                    
                    # Combine into one main DataFrame
                    df_all = pd.concat([df_cloud, df_special], ignore_index=True)

                    # 3. Simple Altair Chart
                    chart = alt.Chart(df_all).mark_point(filled=True, size=60).encode(
                        x=alt.X('Volatility', type='quantitative', title='Annualized Volatility (Risk)', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
                        y=alt.Y('Return', type='quantitative', title='Annualized Return', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False)),
                        color=alt.Color('Type', legend=alt.Legend(title="Type", orient="bottom")),
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
                        
                        # Dynamically generate tabs based on user toggles
                        tab_names = ["Max Sharpe", "Min Volatility"]
                        if use_target_ret:
                            tab_names.append("Target Return")
                        if use_target_vol:
                            tab_names.append("Target Risk")
                            
                        tabs = st.tabs(tab_names)
                        
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

                        with tabs[0]:
                            draw_pie(w_sharpe)
                            
                        with tabs[1]:
                            draw_pie(w_vol)
                            
                        tab_idx = 2
                        if use_target_ret:
                            with tabs[tab_idx]:
                                if is_feasible_ret:
                                    draw_pie(res_target_ret.x)
                                else:
                                    st.warning(f"Target Return {target_ret_val:.1%} not feasible.")
                            tab_idx += 1
                                
                        if use_target_vol:
                            with tabs[tab_idx]:
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
