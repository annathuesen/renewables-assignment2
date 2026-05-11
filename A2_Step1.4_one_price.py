# Import packages
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# LOAD DATA
P_nom = 500
P_rodsand = 207

# Wind data
wind_df = pd.read_csv("Wind_Farm_Generation_Data.csv", skiprows=1)
wind_df = wind_df.iloc[:, 1:].to_numpy(dtype=float).T

wind_df = (wind_df / P_rodsand) * P_nom
wind_df = np.clip(wind_df, 0, P_nom)

# Price data
price_df = pd.read_csv("Day_Ahead_Market_Price_Data_2.csv", skiprows=1)
price_df = price_df.iloc[:, 1:].to_numpy(dtype=float).T

# System imbalance scenarios
n_si = 4
rng = np.random.default_rng(42)

imbalance_df = rng.binomial(
    n=1,
    p=0.5,
    size=(n_si, 24)
)

# BUILD SCENARIOS

wind_scenarios = []
price_scenarios = []
imbalance_scenarios = []

for i in range(wind_df.shape[0]):
    for j in range(price_df.shape[0]):
        for k in range(imbalance_df.shape[0]):
            wind_scenarios.append(wind_df[i])
            price_scenarios.append(price_df[j])
            imbalance_scenarios.append(imbalance_df[k])

wind_scenarios = np.array(wind_scenarios)
price_scenarios = np.array(price_scenarios)
imbalance_scenarios = np.array(imbalance_scenarios)

W = wind_scenarios.shape[0]
T = range(24)
probability = 1 / W


# Balancing price
balancing_price = np.where(
    imbalance_scenarios == 1,
    1.25 * price_scenarios,
    0.85 * price_scenarios
)


# CVaR PARAMETERS
alpha = 0.90
beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# Helper to compute CVaR from the profit directly
def compute_cvar(profits, alpha):
    n = len(profits)
    n_tail = max(1, int(np.floor((1 - alpha) * n)))
    sorted_profits = np.sort(profits)
    return sorted_profits[:n_tail].mean()


# ONE-PRICE SCHEME WITH CVaR


results_exp = []
results_cvar = []
results_offers = []
results_profits = []

for beta in beta_values:
    model_1 = gp.Model("One_Price_CVaR")
    model_1.Params.OutputFlag = 0

    # Decision variable 
    P_DA_1 = model_1.addVars(T, lb=0, ub=P_nom, vtype=GRB.CONTINUOUS, name="P_DA_1")

    # CVaR variables: zeta (VaR, free) and eta (shortfall per scenario, >= 0)
    zeta = model_1.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zeta")
    eta = model_1.addVars(range(W), lb=0, vtype=GRB.CONTINUOUS, name="eta")

    # Profit per scenario 
    profit = {}
    for w in range(W):
        profit[w] = gp.quicksum(
            price_scenarios[w, t] * P_DA_1[t]
            + balancing_price[w, t] * (wind_scenarios[w, t] - P_DA_1[t])
            for t in T
        )

    # Expected profit
    expected_profit_1 = gp.quicksum(
        probability * profit[w] for w in range(W)
    )

    # CVaR = zeta - 1/(1-alpha) * sum_w pi_w * eta_w
    cvar = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        probability * eta[w] for w in range(W)
    )

    # CVaR constraints: eta_w >= zeta - Profit_w
    model_1.addConstrs(
        eta[w] >= zeta - profit[w]
        for w in range(W)
    )

    # Objective: (1-beta)*E[Profit] + beta*CVaR
    model_1.setObjective(
        (1 - beta) * expected_profit_1 + beta * cvar,
        GRB.MAXIMIZE
    )
    model_1.optimize()

    P_DA_one_price = np.array([P_DA_1[t].X for t in T])
    scenario_profits = np.array([profit[w].getValue() for w in range(W)])

    results_exp.append(scenario_profits.mean())
    results_cvar.append(compute_cvar(scenario_profits, alpha))
    results_offers.append(P_DA_one_price)
    results_profits.append(scenario_profits)

    print(f"beta={beta:.1f}: E[Profit]={scenario_profits.mean():>12,.2f} EUR, "
          f"CVaR={compute_cvar(scenario_profits, alpha):>12,.2f} EUR")


# RESULTS
print("\n=== RESULTS ===")
print(f"\n{'beta':>6}  {'E[Profit]':>14}  {'CVaR':>14}")
print("-" * 40)
for i, beta in enumerate(beta_values):
    print(f"{beta:6.2f}  {results_exp[i]:>14,.2f}  {results_cvar[i]:>14,.2f}")

print("\nHourly DA offers (beta=0 vs beta=1):")
for t in T:
    print(
        f"Hour {t:02d}: "
        f"beta=0 = {results_offers[0][t]:8.2f} MW | "
        f"beta=1 = {results_offers[-1][t]:8.2f} MW"
    )


# PLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Efficient frontier
ax = axes[0]
ax.plot(results_cvar, results_exp, "bo-", markersize=6)
for i, beta in enumerate(beta_values):
    if beta in [0.0, 0.5, 1.0]:
        ax.annotate(rf"$\beta$={beta}", (results_cvar[i], results_exp[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.set_xlabel("CVaR (EUR)", fontsize=12)
ax.set_ylabel("Expected Profit (EUR)", fontsize=12)
ax.set_title("Efficient Frontier - One-Price Scheme", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# DA offers for selected betas
ax = axes[1]
mean_wind = wind_scenarios.mean(axis=0)
for beta_val, color in zip([0.0, 0.3, 0.6, 1.0], ["green", "orange", "red", "purple"]):
    idx = beta_values.index(beta_val)
    ax.plot(list(T), results_offers[idx],
            "-o", markersize=3, color=color, label=rf"$\beta$={beta_val}")
ax.plot(list(T), mean_wind, "k--", linewidth=2, label="Mean wind")
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Day-ahead offer [MW]", fontsize=12)
ax.set_title(r"DA Offers for Different $\beta$ - One-Price", fontsize=14)
ax.legend(fontsize=8, frameon=True)
ax.set_xticks(list(T))
ax.grid(True, linestyle='--', alpha=0.6)

# Profit distribution: beta=0 vs beta=1
ax = axes[2]
ax.hist(results_profits[0], bins=40, alpha=0.5, color="green",
        label=r"$\beta$=0", edgecolor="black", linewidth=0.3)
ax.hist(results_profits[-1], bins=40, alpha=0.5, color="purple",
        label=r"$\beta$=1", edgecolor="black", linewidth=0.3)
ax.set_xlabel("Profit [EUR]", fontsize=12)
ax.set_ylabel("Number of Scenarios", fontsize=12)
ax.set_title("Profit Distribution - One-Price Scheme", fontsize=14)
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("Step1_4_One_Price.pdf", bbox_inches="tight")
plt.show()

# Computational Aspects
print(f"\nNumber of constraints: {model_1.NumConstrs}")
print(f"Number of variables: {model_1.NumVars}")
print(f"Runtime (last beta): {model_1.Runtime:.4f} seconds")
