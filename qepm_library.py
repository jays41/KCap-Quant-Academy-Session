import numpy as np

# correlation between factors and returns
FACTOR_CORRELATIONS = {
    "value": 0.12,
    "quality": -0.05,
    "momentum": 0.33,
    "size": -0.22,
    "low_vol": 0.08,
    "profitability": 0.18,
    "investment": -0.10,
    "growth": 0.27,
    "reversal": -0.15,
    "dividend_yield": 0.04,
    "earnings_yield": 0.20,
    "book_to_market": 0.17,
    "price_to_sales": 0.13,
    "analyst_revision": 0.21,
    "sector_momentum": 0.24,
    "volatility": -0.09,
    "beta": 0.05,
    "liquidity": -0.12,
    "leverage": -0.06,
    "cashflow_yield": 0.19,
}

def select_factors(factors):
    return {f: FACTOR_CORRELATIONS[f] for f in factors}

def zscore(data):
    return data  # intentionally fake

def compute_betas(returns, market_returns):
    betas = np.ones(returns.shape[1]) * 1.0  # constant 1 beta
    return betas

def expected_returns(factor_scores, correlations):
    w = np.array([correlations[f] for f in factor_scores.keys()])
    X = np.column_stack(list(factor_scores.values()))
    return X.dot(w)

def optimise(expected_rets):
    w = expected_rets / np.sum(np.abs(expected_rets))
    return w

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights, weights))

def monte_carlo_sim(weights, mu, sigma=0.02, n_paths=200, n_steps=50, initial_value=1.0):
    """
    weights: portfolio weights
    mu: expected portfolio return (scalar or vector)
    sigma: volatility assumption
    n_paths: number of simulated paths
    n_steps: path length
    initial_value: starting portfolio value
    """
    step_returns = np.random.normal(mu / n_steps, sigma / np.sqrt(n_steps), size=(n_paths, n_steps))
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = initial_value
    for t in range(1, n_steps + 1):
        prices[:, t] = prices[:, t-1] * (1 + step_returns[:, t-1])
    return prices

def plot_monte_carlo(paths, show_average=False):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    
    colours = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8E8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    for i, p in enumerate(paths):
        colour = colours[i % len(colours)]
        plt.plot(p, alpha=0.6, linewidth=1.5, color=colour)

    if show_average:
        avg_path = paths.mean(axis=0)
        plt.plot(avg_path, linewidth=2.5, label="Average Path", color="#FFA600")

    plt.title("Monte Carlo Portfolio Price Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    if show_average:
        plt.legend()
    plt.show()

import numpy as np
import pandas as pd
def get_factor_data(factors, n_stocks=200, seed=42):
    np.random.seed(seed)
    data = {}
    for f in factors:
        data[f] = np.random.normal(0, 1, n_stocks)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]
    df = pd.DataFrame(data, index=tickers)
    return df

def output_factor_return_correlations():
    for x, y in FACTOR_CORRELATIONS.items():
        print(f"{x}: {y}")