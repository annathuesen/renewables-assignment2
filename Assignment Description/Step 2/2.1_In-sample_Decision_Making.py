#%%
"""
Task 2.1) In-sample Decision Making: Offering Strategy Under the P90 Re-
quirement
Using the 100 in-sample profiles, determine the optimal FCR-D UP reserve bid (in kW) satis-
fying Energinet’s P90 requirement. Solve the problem using both ALSO-X and CVaR.


"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


N_PROFILES = 300
N_MINUTES = 60
P_MIN = 220.0
P_MAX = 600.0
MAX_RAMP = 35.0
N_IN_SAMPLE = 100
P90_LEVEL = 0.90
SEED = 42


def generate_load_profiles():
    rng = np.random.default_rng(SEED)
    profiles = np.zeros((N_PROFILES, N_MINUTES))

    for s in range(N_PROFILES):
        profiles[s, 0] = rng.uniform(P_MIN, P_MAX)

        for t in range(1, N_MINUTES):
            low = max(P_MIN, profiles[s, t - 1] - MAX_RAMP)
            high = min(P_MAX, profiles[s, t - 1] + MAX_RAMP)
            profiles[s, t] = rng.uniform(low, high)

    return profiles


def scenario_reserve_capacities(profiles):
    return np.min(profiles, axis=1)


def solve_also_x_gurobi(capacities):
    n = len(capacities)
    allowed_violations = int(np.floor((1.0 - P90_LEVEL) * n))

    model = gp.Model("also_x_p90")
    model.Params.OutputFlag = 0

    bid = model.addVar(lb=0.0, ub=P_MAX, name="bid")
    violation = model.addVars(n, vtype=GRB.BINARY, name="violation")

    for s in range(n):
        model.addConstr(bid <= capacities[s] + P_MAX * violation[s])

    model.addConstr(gp.quicksum(violation[s] for s in range(n)) <= allowed_violations)

    model.setObjective(bid, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(bid.X)


def solve_cvar_gurobi(capacities):
    capacities = np.asarray(capacities)
    n = len(capacities)
    alpha = 1.0 - P90_LEVEL

    model = gp.Model("cvar_p90")
    model.Params.OutputFlag = 0

    bid = model.addVar(lb=0.0, ub=P_MAX, name="bid")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    shortfall = model.addVars(n, lb=0.0, name="shortfall")

    for s in range(n):
        model.addConstr(shortfall[s] >= bid - capacities[s] - eta)

    model.addConstr(
        eta + (1.0 / (alpha * n)) * gp.quicksum(shortfall[s] for s in range(n)) <= 0.0
    )

    model.setObjective(bid, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed: {model.Status}")

    return float(bid.X)


def plot_in_sample_profiles(in_sample_profiles):
    minutes = np.arange(N_MINUTES)

    plt.figure(figsize=(10, 5))
    for profile in in_sample_profiles[:25]:
        plt.plot(minutes, profile, alpha=0.5)

    plt.xlabel("Minute")
    plt.ylabel("Consumption (kW)")
    plt.title("Task 2.1: Example In-Sample Load Profiles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bid_vs_capacity(capacities, also_x_bid, cvar_bid):
    sorted_capacities = np.sort(capacities)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_capacities, marker="o", label="Scenario reserve capacity")
    plt.axhline(also_x_bid, linestyle="--", label=f"ALSO-X bid: {also_x_bid:.2f} kW")
    plt.axhline(cvar_bid, linestyle=":", label=f"CVaR bid: {cvar_bid:.2f} kW")

    plt.xlabel("Sorted in-sample scenario")
    plt.ylabel("Reserve capacity [kW]")
    plt.title("Task 2.1: In-Sample Reserve Bid vs Scenario Capacities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    profiles = generate_load_profiles()
    in_sample_profiles = profiles[:N_IN_SAMPLE]
    capacities = scenario_reserve_capacities(in_sample_profiles)

    also_x_bid = solve_also_x_gurobi(capacities)
    cvar_bid = solve_cvar_gurobi(capacities)

    print("Task 2.1 — In-Sample Decision Making")
    print("====================================")
    print(f"ALSO-X bid: {also_x_bid:.2f} kW")
    print(f"CVaR bid:   {cvar_bid:.2f} kW")
    print(f"ALSO-X in-sample feas: {np.mean(capacities >= also_x_bid):.2%}")
    print(f"CVaR in-sample feas:   {np.mean(capacities >= cvar_bid):.2%}")

    plot_in_sample_profiles(in_sample_profiles)
    plot_bid_vs_capacity(capacities, also_x_bid, cvar_bid)


if __name__ == "__main__":
    main()