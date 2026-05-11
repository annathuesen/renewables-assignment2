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

#Two-Price Scheme Balancing Price (if system imbalance is desired or not)
# If system imbalance is desired, balancing price is equal to Day-Ahead price. If undesired, it is less than Day-Ahead
lambda_up = np.where(
    imbalance_scenarios == 1,
    price_scenarios, #Day-Ahead Price
    balancing_price
) 

lambda_down = np.where(
    imbalance_scenarios == 1,
    balancing_price, 
    price_scenarios #Day-Ahead Price
)

# TWO-PRICE SCHEME
model_2 = gp.Model("Two_Price_Scheme")
model_2.Params.OutputFlag = 0

P_DA_2 = model_2.addVars(T, lb=0, ub=P_nom, vtype=GRB.CONTINUOUS, name="P_DA_2")
Delta_Up = model_2.addVars(range(W), T, lb=0, vtype=GRB.CONTINUOUS, name="Delta_Up")
Delta_Down = model_2.addVars(range(W), T, lb=0, vtype=GRB.CONTINUOUS, name="Delta_Down")

model_2.addConstrs(
    wind_scenarios[w, t] - P_DA_2[t] == Delta_Up[w, t] - Delta_Down[w, t]
    for w in range(W)
    for t in T
)

expected_profit_2 = gp.quicksum(
    probability * gp.quicksum(
        price_scenarios[w, t] * P_DA_2[t]
        + lambda_up[w, t] * Delta_Up[w, t]
        - lambda_down[w, t] * Delta_Down[w, t]
        for t in T
    )
    for w in range(W)
)

model_2.setObjective(expected_profit_2, GRB.MAXIMIZE)
model_2.optimize()

P_DA_two_price = np.array([P_DA_2[t].X for t in T])
expected_profit_two_price = model_2.ObjVal


print("\n=== RESULTS ===")
print(f"Expected profit — two-price: {expected_profit_two_price:.2f} EUR")


print("\nHourly DA offers:")
for t in T:
    print(
        f"Hour {t:02d}: "
        f"Two-price = {P_DA_two_price[t]:8.2f} MW"
    )

# PROFIT DISTRIBUTIONS
scenario_profits_two = np.zeros(W)

for w in range(W):
    scenario_profits_two[w] = sum(
        price_scenarios[w, t] * P_DA_two_price[t]
        + lambda_up[w, t] * Delta_Up[w, t].X
        - lambda_down[w, t] * Delta_Down[w, t].X
        for t in T
    )
    
# PLOTS
plt.figure()
plt.plot(list(T), P_DA_two_price, linewidth=2, color = 'orange', label="Two-price")
plt.xlabel("Hour")
plt.xticks(list(T), rotation=45)
plt.ylabel("Day-ahead offer [MW]")
plt.title("Optimal hourly day-ahead offers")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.hist(scenario_profits_two, bins=40, edgecolor= 'black', color='orange', label="Two-Price")
plt.xlabel("Profit [EUR]")
plt.ylabel("Number of scenarios")
plt.title("Profit distribution — two-price scheme")
plt.grid(True)
plt.show()

#Computational Aspects
print(f"Number of constraints: {model_2.NumConstrs}")
print(f"Number of variables: {model_2.NumVars}")
print(f"Runtime: {model_2.Runtime:.4f} seconds")