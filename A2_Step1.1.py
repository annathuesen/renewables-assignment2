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
price_df = pd.read_csv("Day_Ahead_Market_Price_Data.csv", skiprows=1)
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

# ONE-PRICE SCHEME

model_1 = gp.Model("One_Price_Scheme")
model_1.Params.OutputFlag = 0

P_DA_1 = model_1.addVars(T, lb=0, ub=P_nom, vtype=GRB.CONTINUOUS, name="P_DA_1")

expected_profit_1 = gp.quicksum(
    probability * gp.quicksum(
        price_scenarios[w, t] * P_DA_1[t]
        + balancing_price[w, t] * (wind_scenarios[w, t] - P_DA_1[t])
        for t in T
    )
    for w in range(W)
)

model_1.setObjective(expected_profit_1, GRB.MAXIMIZE)
model_1.optimize()

P_DA_one_price = np.array([P_DA_1[t].X for t in T])
expected_profit_one_price = model_1.ObjVal


print("\n=== RESULTS ===")
print(f"Expected profit — one-price: {expected_profit_one_price:.2f} EUR")

print("\nHourly DA offers:")
for t in T:
    print(
        f"Hour {t:02d}: "
        f"One-price = {P_DA_one_price[t]:8.2f} MW | "
    )


# PROFIT DISTRIBUTIONS
scenario_profits_one = np.zeros(W)

for w in range(W):
    scenario_profits_one[w] = sum(
        price_scenarios[w, t] * P_DA_one_price[t]
        + balancing_price[w, t] * (wind_scenarios[w, t] - P_DA_one_price[t])
        for t in T
    )

# PLOTS
plt.figure()
plt.plot(list(T), P_DA_one_price, label="One-price")
plt.xlabel("Hour", fontsize = 12)
plt.xticks(list(T), rotation=45)
plt.ylabel("Day-ahead offer [MW]", fontsize = 12)
plt.title("Optimal Hourly Day-Ahead Offers - One-Price Scheme", fontsize = 14)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("One_Scheme_Offers.pdf", bbox_inches="tight")
plt.show()


plt.figure()
plt.hist(scenario_profits_one, bins=40, edgecolor= 'black')
plt.xlabel("Profit [EUR]", fontsize = 12)
plt.ylabel("Number of Scenarios", fontsize = 12)
plt.title("Profit Distribution - One-Price Scheme", fontsize = 14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("One_Scheme_Histogram.pdf", bbox_inches="tight")
plt.show()

#Computational Aspects
print(f"Number of constraints: {model_1.NumConstrs}")
print(f"Number of variables: {model_1.NumVars}")
print(f"Runtime: {model_1.Runtime:.4f} seconds")

# plt.figure()
# plt.hist(scenario_profits_one, bins=40, alpha=0.5, label="One-price")
# plt.hist(scenario_profits_two, bins=40, alpha=0.5, label="Two-price")
# plt.xlabel("Profit [EUR]")
# plt.ylabel("Number of scenarios")
# plt.title("Profit distribution comparison")
# plt.legend()
# plt.grid(True)
# plt.show()