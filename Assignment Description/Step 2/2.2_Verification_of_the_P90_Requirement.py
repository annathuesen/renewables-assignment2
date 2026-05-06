#%%
"""
Task 2.2) Verification of the P90 Requirement Using Out-of-Sample Analysis
Using the 200 out-of-sample profiles, verify whether the P90 requirement is met for each solution
method. No optimization is needed—compare the chosen reserve bid with the minute-level
consumption profiles to evaluate possible shortfalls.
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
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(bid.X)


def verify_p90(profiles, bid):
    capacities = scenario_reserve_capacities(profiles)
    shortfalls = np.maximum(0.0, bid - capacities)
    feasible = shortfalls == 0.0

    return {
        "feasible_count": int(np.sum(feasible)),
        "total_count": len(profiles),
        "empirical_feasibility": float(np.mean(feasible)),
        "p90_met": bool(np.mean(feasible) >= P90_LEVEL),
        "expected_shortfall": float(np.mean(shortfalls)),
        "max_shortfall": float(np.max(shortfalls)),
        "shortfalls": shortfalls,
        "capacities": capacities,
    }


def print_verification(method, bid, result):
    print(f"\n{method}")
    print("-" * len(method))
    print(f"Bid: {bid:.2f} kW")
    print(f"Feasible profiles: {result['feasible_count']} / {result['total_count']}")
    print(f"Out-of-sample feasibility: {result['empirical_feasibility']:.2%}")
    print(f"P90 met: {result['p90_met']}")
    print(f"Expected shortfall: {result['expected_shortfall']:.2f} kW")
    print(f"Maximum shortfall: {result['max_shortfall']:.2f} kW")


def plot_out_sample_profiles(out_sample_profiles, also_x_bid, cvar_bid):
    minutes = np.arange(N_MINUTES)

    plt.figure(figsize=(10, 5))
    for profile in out_sample_profiles[:25]:
        plt.plot(minutes, profile, alpha=0.5)

    plt.axhline(also_x_bid, linestyle="--", label=f"ALSO-X bid = {also_x_bid:.2f} kW")
    plt.axhline(cvar_bid, linestyle=":", label=f"CVaR bid = {cvar_bid:.2f} kW")

    plt.xlabel("Minute")
    plt.ylabel("Consumption (kW)")
    plt.title("Task 2.2: Out-of-Sample Profiles with Reserve Bids")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_out_sample_shortfalls(also_x_result, cvar_result):
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(also_x_result["shortfalls"]), marker="o", label="ALSO-X shortfall")
    plt.plot(np.sort(cvar_result["shortfalls"]), marker="x", label="CVaR shortfall")

    plt.xlabel("Sorted out-of-sample scenario")
    plt.ylabel("Shortfall (kW)")
    plt.title("Task 2.2: Out-of-Sample Reserve Shortfalls")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_capacity_histogram(also_x_bid, cvar_bid, out_sample_capacities):
    plt.figure(figsize=(10, 5))
    plt.hist(out_sample_capacities, bins=20, alpha=0.7)
    plt.axvline(also_x_bid, linestyle="--", label=f"ALSO-X bid = {also_x_bid:.2f} kW")
    plt.axvline(cvar_bid, linestyle=":", label=f"CVaR bid = {cvar_bid:.2f} kW")

    plt.xlabel("Out-of-sample reserve capacity (kW)")
    plt.ylabel("Number of profiles")
    plt.title("Task 2.2: Distribution of Out-of-Sample Reserve Capacities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    profiles = generate_load_profiles()

    in_sample_profiles = profiles[:N_IN_SAMPLE]
    out_sample_profiles = profiles[N_IN_SAMPLE:]

    in_sample_capacities = scenario_reserve_capacities(in_sample_profiles)

    also_x_bid = solve_also_x_gurobi(in_sample_capacities)
    cvar_bid = solve_cvar_gurobi(in_sample_capacities)

    also_x_result = verify_p90(out_sample_profiles, also_x_bid)
    cvar_result = verify_p90(out_sample_profiles, cvar_bid)

    print("Task 2.2 — Out-of-Sample P90 Verification")
    print("=========================================")

    print_verification("ALSO-X", also_x_bid, also_x_result)
    print_verification("CVaR", cvar_bid, cvar_result)

    plot_out_sample_profiles(out_sample_profiles, also_x_bid, cvar_bid)
    plot_out_sample_shortfalls(also_x_result, cvar_result)
    plot_capacity_histogram(
        also_x_bid,
        cvar_bid,
        also_x_result["capacities"],
    )


if __name__ == "__main__":
    main()