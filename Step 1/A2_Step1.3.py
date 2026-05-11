import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# ============================================================
# LOAD DATA
# ============================================================

P_nom = 500
P_rodsand = 207

# Wind data
wind_df = pd.read_csv("Wind_Farm_Generation_Data.csv", skiprows=1)
wind_df = wind_df.iloc[:, 1:].to_numpy(dtype=float).T

wind_df = (wind_df / P_rodsand) * P_nom

# Price data
price_df = pd.read_csv("Day_Ahead_Market_Price_Data_2.csv", skiprows=1)
price_df = price_df.iloc[:, 1:].to_numpy(dtype=float).T

# Imbalance scenarios
n_si = 4
rng = np.random.default_rng(42)

imbalance_df = rng.binomial(
    n=1,
    p=0.5,
    size=(n_si, 24)
)

# ============================================================
# BUILD 1600 SCENARIOS
# ============================================================

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

# ============================================================
# CROSS VALIDATION
# ============================================================
n_folds = 8 #Change depending on input
in_sample_size = 200
out_sample_size = W - in_sample_size

Two_Price_Scheme = True

in_sample_profits = []
out_sample_profits = []

for fold in range(n_folds):
    split_start = fold * in_sample_size
    split_end = split_start + in_sample_size
    
    #In Sample Variables (200,24)
    wind_in_sample = wind_scenarios[split_start : split_end]
    price_in_sample = price_scenarios[split_start : split_end]
    balancing_price_in_sample = balancing_price[split_start : split_end]
    lambda_down_in_sample = lambda_down[split_start : split_end]
    lambda_up_in_sample = lambda_up[split_start : split_end]
    probability_in_sample = 1/in_sample_size
    
    #Step 1 (In-Sample Analysis) - Solve optimization problem and determine optimal quantity offer (MW) in the day-ahead market
    model_3 = gp.Model("Two_Price_Scheme")
    model_3.Params.OutputFlag = 0

    if Two_Price_Scheme == True:
        P_DA_in_sample = model_3.addVars(T, lb=0, ub=P_nom, vtype=GRB.CONTINUOUS, name="P_DA_in_sample")
        Delta_Up = model_3.addVars(range(in_sample_size), T, lb=0, name="Delta_Up")
        Delta_Down = model_3.addVars(range(in_sample_size), T,  lb=0, name="Delta_Down")

        model_3.addConstrs(
            wind_in_sample[w, t] - P_DA_in_sample[t] == Delta_Up[w, t] - Delta_Down[w, t]
            for w in range(in_sample_size)
            for t in T
        )

        expected_profit_in_sample = gp.quicksum(
            probability_in_sample * gp.quicksum(
                price_in_sample[w, t] * P_DA_in_sample[t]
                + lambda_up_in_sample[w, t] * Delta_Up[w, t]
                - lambda_down_in_sample[w, t] * Delta_Down[w, t]
                for t in T
            )
            for w in range(in_sample_size)
        )

    else:
        P_DA_in_sample = model_3.addVars(T, lb=0, ub=P_nom, vtype=GRB.CONTINUOUS, name="P_DA_in_sample")


        expected_profit_in_sample = gp.quicksum(
            probability_in_sample * gp.quicksum(
                price_in_sample[w, t] * P_DA_in_sample[t]
                + balancing_price_in_sample[w, t] * (wind_in_sample[w,t] - P_DA_in_sample[t])
                for t in T
            )
            for w in range(in_sample_size)
        )
               
    model_3.setObjective(expected_profit_in_sample, GRB.MAXIMIZE)
    model_3.optimize()

    P_DA_opt = np.array([P_DA_in_sample[t].X for t in T])
    in_sample_profits.append(model_3.ObjVal)

    #Out of sample Analysis Variables
    wind_out_sample = np.concatenate([wind_scenarios[:split_start], wind_scenarios[split_end:]])
    price_out_sample = np.concatenate([price_scenarios[:split_start], price_scenarios[split_end:]])
    balancing_price_out_sample = np.concatenate([balancing_price[:split_start], balancing_price[split_end:]])
    lambda_down_out_sample = np.concatenate([lambda_down[:split_start], lambda_down[split_end:]])
    lambda_up_out_sample = np.concatenate([lambda_up[:split_start], lambda_up[split_end:]])
    
    #Step 2 (Out-of-sample Analysis) - For a given offering decision, calculate the imbalance and corresponding cost incurred
    imbalance_out = np.zeros((out_sample_size, len(T)))

    for o in range(out_sample_size):
        for t in T:
            imbalance_out[o, t] = wind_out_sample[o, t] - P_DA_opt[t]
    
    delta_up_out   = np.maximum(imbalance_out, 0)
    delta_down_out = np.maximum(-imbalance_out, 0)                                                                                                                                                                                           
    
    #Pay-off
    if Two_Price_Scheme == True:
        balancing_payoff = np.zeros(out_sample_size)
        for o in range(out_sample_size):
            balancing_payoff[o] = sum(
                lambda_up_out_sample[o, t]   * delta_up_out[o, t]
                - lambda_down_out_sample[o, t] * delta_down_out[o, t]
                for t in T
            )
        
    else:
        balancing_payoff = np.zeros(out_sample_size)
        for o in range(out_sample_size):
            balancing_payoff[o] = sum(
                balancing_price_out_sample[o,t] * imbalance_out[o, t]
                for t in T
            )
            
    #Step 3 (Out-of-sample Profit) - Day-Ahead Profit plus average payoff from Step 2
    #Day-Ahead Profit
    da_profit = probability_in_sample * sum(
        price_out_sample[w, t] * P_DA_opt[t]
        for w in range(in_sample_size)
        for t in T
    )
    out_sample_profit = da_profit + balancing_payoff.mean()
    
    out_sample_profits.append(out_sample_profit)
    
    print(f"Fold {fold+1}: in-sample = {model_3.ObjVal:.2f} €, out-of-sample = {out_sample_profit:.2f} €")
    
print(f"Avg in-sample profit:     {np.mean(in_sample_profits):.2f} €")
print(f"Avg out-of-sample profit: {np.mean(out_sample_profits):.2f} €")  
    
#Plots
x = np.arange(n_folds)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_in  = ax.bar(x - width/2, in_sample_profits,  width, label='In-sample', edgecolor= 'black')
bars_out = ax.bar(x + width/2, out_sample_profits, width, label='Out-of-sample', edgecolor= 'black')

ax.set_xlabel('Fold', fontsize = 12)
ax.set_ylabel('Expected Profit (€)', fontsize = 12)
ax.set_title(f'Cross-Validation: In-sample vs Out-of-sample Profit - Two-Price-Scheme', fontsize = 14)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
ax.legend(frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'two_price_cross_validation_{in_sample_size}.pdf', dpi=150)
plt.show()