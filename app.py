import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Portfolio Optimizer & Efficient Frontier")
st.markdown("""
This app optimizes a portfolio for **Maximum Sharpe Ratio**, **Minimum Volatility**, and specific **Target Return/Risk** goals using the Efficient Frontier.
""")

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

# --- SECTION 3: SIDEBAR INPUTS ---

with st.sidebar:
    st.header("1. Asset Selection")
    default_tickers = "VTI, TLT, IEF, GLD, PDBC"
    tickers_input = st.text_input("Tickers (comma separated)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    st.header("2. Parameters")
    rf_rate = st.number_input("Risk-Free Rate (decimal)", value=0.04, step=0.005, format="%.3f")
    return_mode = st.selectbox("Return Assumption", ["historical", "investor_views"])

    investor_views = {}
    if return_mode == "investor_views":
        st.info("Enter expected annual return (decimal) for each ticker:")
        for t in tickers:
            investor_views[t] = st.number_input(f"View: {t}", value=0.07, step=0.01, key=f"view_{t}")

    st.header("3. Constraints")
    use_constraints = st.checkbox("Use Weight Constraints", value=True)
    min_w = st.slider("Min Weight", 0.0, 0.5, 0.05, 0.01)
    max_w = st.slider("Max Weight", 0.0, 1.0, 0.40, 0.01)

    st.header("4. Targets")
    target_ret_val = st.number_input("Target Return Goal", value=0.075, step=0.005, format="%.3f")
    target_vol_val = st.number_input("Target Risk Goal", value=0.080, step=0.005, format="%.3f")

    run_btn = st.button("Run Optimization", type="primary")

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
                st.subheader("1. Optimized Weights")
                
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
                st.dataframe(df_weights, use_container_width=True)

                if not is_feasible_ret:
                    st.warning(f"⚠️ Target Return {target_ret_val:.1%} is not feasible with current constraints.")
                if not is_feasible_vol:
                    st.warning(f"⚠️ Target Risk {target_vol_val:.1%} is not feasible with current constraints.")

                # --- B. Performance Summary ---
                st.subheader("2. Performance Summary")
                stats_sharpe = get_portfolio_stats(w_sharpe, avg_returns, cov_matrix, rf_rate)
                stats_vol = get_portfolio_stats(w_vol, avg_returns, cov_matrix, rf_rate)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Max Sharpe Ratio**
                    - Return: `{stats_sharpe[0]:.2%}`
                    - Volatility: `{stats_sharpe[1]:.2%}`
                    - Sharpe: `{stats_sharpe[2]:.2f}`
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Minimum Volatility**
                    - Return: `{stats_vol[0]:.2%}`
                    - Volatility: `{stats_vol[1]:.2%}`
                    - Sharpe: `{stats_vol[2]:.2f}`
                    """)

                # --- C. Plots ---
                st.subheader("3. Visualizations")
                tab1, tab2 = st.tabs(["Efficient Frontier", "Correlation Matrix"])

                with tab1:
                    # Efficient Frontier Plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Monte Carlo Simulation for cloud
                    n_sim = 1000
                    sim_rets = []
                    sim_vols = []
                    
                    for _ in range(n_sim):
                        w = np.random.random(len(tickers))
                        w /= np.sum(w)
                        # Quick check for constraints to make the cloud relevant
                        if use_constraints:
                            if np.any(w < min_w) or np.any(w > max_w):
                                continue
                        
                        r, v, _ = get_portfolio_stats(w, avg_returns, cov_matrix, rf_rate)
                        sim_rets.append(r)
                        sim_vols.append(v)
                    
                    ax.scatter(sim_vols, sim_rets, color='lightgray', alpha=0.3, s=15, label='Random Portfolios')

                    # Plot Optimized Points
                    ax.scatter(stats_sharpe[1], stats_sharpe[0], color='red', marker='*', s=200, label='Max Sharpe')
                    ax.scatter(stats_vol[1], stats_vol[0], color='blue', marker='*', s=200, label='Min Volatility')

                    if is_feasible_ret:
                        ax.scatter(stats_target_ret[1], stats_target_ret[0], color='green', marker='s', s=100, label=f'Target Ret {target_ret_val:.1%}')
                    
                    if is_feasible_vol:
                        ax.scatter(stats_target_vol[1], stats_target_vol[0], color='orange', marker='D', s=100, label=f'Target Risk {target_vol_val:.1%}')

                    ax.set_xlabel('Annualized Volatility (Risk)')
                    ax.set_ylabel('Annualized Return')
                    ax.set_title('Efficient Frontier Simulation')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with tab2:
                    # Covariance Heatmap
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cov_matrix, annot=True, cmap='RdYlGn_r', fmt='.4f', ax=ax_corr)
                    ax_corr.set_title("Covariance Matrix")
                    st.pyplot(fig_corr)
else:
    st.info("👈 Use the sidebar to configure assets and parameters, then click **Run Optimization**.")
