# !wget https://raw.githubusercontent.com/jays41/KCap-Quant-Academy-Session/main/qepm_library.py -O qepm_library.py
from qepm_library import *
print("Demo library loaded successfully!")

# 1. Choose factors
output_factor_return_correlations()
my_factors = ["value", "momentum", "quality", "dividend_yield"]
factor_corrs = select_factors(my_factors)

# 2. Load dataset
df = get_factor_data(my_factors, n_stocks=150)

# 3. Compute betas and expected returns
betas = get_betas(df)
factor_scores = get_z_scores(my_factors, df)
mu = expected_returns(factor_scores, factor_corrs)

# 4. Optimise portfolio
weights = optimise(mu, betas)
print("Weights:", weights[:10], "...")
expected_portfolio_return = np.dot(weights, mu)
print(f"Expected portfolio return: {expected_portfolio_return}")

# 5. Monte Carlo
paths = monte_carlo_sim(weights, mu.mean())
plot_monte_carlo(paths, show_average=False)

print("Simulated portfolio mean return:", paths.mean())