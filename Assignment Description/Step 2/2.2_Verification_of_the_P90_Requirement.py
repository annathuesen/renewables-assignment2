#%%
"""
Task 2.2) Verification of the P90 Requirement Using Out-of-Sample Analysis

Minute-level implementation matching the report notation. The reserve bids are first obtained
from the 100 in-sample profiles using the same ALSO-X and CVaR formulations as Task 2.1.
They are then tested on 200 out-of-sample profiles by comparing c^up with F^up_{omega,m}.
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
EPSILON = 1.0 - P90_LEVEL
SEED = 42
BIG_M = P_MAX


def generate_load_profiles():
    rng = np.random.default_rng(SEED)
    F_up = np.zeros((N_PROFILES, N_MINUTES))

    for omega in range(N_PROFILES):
        F_up[omega, 0] = rng.uniform(P_MIN, P_MAX)

        for m in range(1, N_MINUTES):
            low = max(P_MIN, F_up[omega, m - 1] - MAX_RAMP)
            high = min(P_MAX, F_up[omega, m - 1] + MAX_RAMP)
            F_up[omega, m] = rng.uniform(low, high)

    return F_up


def scenario_reserve_capacities(F_up):
    return np.min(F_up, axis=1)


def solve_also_x_gurobi(F_up_in_sample):
    n_omega, n_m = F_up_in_sample.shape
    q = int(np.floor(EPSILON * n_omega * n_m))

    model = gp.Model("also_x_p90_minute_level")
    model.Params.OutputFlag = 0

    c_up = model.addVar(lb=0.0, ub=P_MAX, name="c_up")
    y = model.addVars(n_omega, n_m, vtype=GRB.BINARY, name="y")

    for omega in range(n_omega):
        for m in range(n_m):
            model.addConstr(c_up - F_up_in_sample[omega, m] <= BIG_M * y[omega, m])

    model.addConstr(gp.quicksum(y[omega, m] for omega in range(n_omega) for m in range(n_m)) <= q)
    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(c_up.X)


def solve_cvar_gurobi(F_up_in_sample):
    n_omega, n_m = F_up_in_sample.shape
    n_pairs = n_omega * n_m

    model = gp.Model("cvar_p90_minute_level")
    model.Params.OutputFlag = 0

    c_up = model.addVar(lb=0.0, ub=P_MAX, name="c_up")
    beta_P90 = model.addVar(lb=-GRB.INFINITY, name="beta_P90")
    zeta_P90 = model.addVars(n_omega, n_m, lb=0.0, name="zeta_P90")

    for omega in range(n_omega):
        for m in range(n_m):
            model.addConstr(zeta_P90[omega, m] >= c_up - F_up_in_sample[omega, m] - beta_P90)

    model.addConstr(
        beta_P90
        + (1.0 / (EPSILON * n_pairs))
        * gp.quicksum(zeta_P90[omega, m] for omega in range(n_omega) for m in range(n_m))
        <= 0.0
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(c_up.X)


def verify_p90_minute_level(F_up_out_sample, c_up):
    shortfall = np.maximum(0.0, c_up - F_up_out_sample)
    violated = shortfall > 1e-8

    # Global minute-based compliance across all out-of-sample minute-scenario pairs
    total_minutes = F_up_out_sample.size
    overbid_minutes = int(np.sum(violated))
    minute_compliance = 1.0 - overbid_minutes / total_minutes

    # Profile-based P90 compliance: no more than 6 violated minutes in a 60-minute profile
    max_violated_minutes = int(np.floor(EPSILON * N_MINUTES))
    violated_minutes_by_profile = np.sum(violated, axis=1)
    profile_feasible = violated_minutes_by_profile <= max_violated_minutes

    return {
        "feasible_profiles": int(np.sum(profile_feasible)),
        "total_profiles": int(F_up_out_sample.shape[0]),
        "profile_compliance": float(np.mean(profile_feasible)),
        "overbid_minutes": overbid_minutes,
        "total_minutes": int(total_minutes),
        "minute_compliance": float(minute_compliance),
        "p90_met_profile_based": bool(np.mean(profile_feasible) >= P90_LEVEL),
        "p90_met_minute_based": bool(minute_compliance >= P90_LEVEL),
        "expected_shortfall": float(np.mean(shortfall)),
        "max_shortfall": float(np.max(shortfall)),
        "shortfall": shortfall,
        "violated": violated,
        "capacities": scenario_reserve_capacities(F_up_out_sample),
    }


def print_verification(method, c_up, result):
    print(f"\n{method}")
    print("-" * len(method))
    print(f"Bid: {c_up:.2f} kW")
    print(f"Profiles satisfying P90: {result['feasible_profiles']} / {result['total_profiles']}")
    print(f"Profile-based compliance: {result['profile_compliance']:.2%}")
    print(f"Overbid minutes: {result['overbid_minutes']} / {result['total_minutes']}")
    print(f"Minute-based compliance: {result['minute_compliance']:.2%}")
    print(f"P90 met, profile-based: {result['p90_met_profile_based']}")
    print(f"P90 met, minute-based: {result['p90_met_minute_based']}")
    print(f"Expected shortfall: {result['expected_shortfall']:.4f} kW")
    print(f"Maximum shortfall: {result['max_shortfall']:.4f} kW")


def plot_out_sample_profiles(F_up_out_sample, also_x_bid, cvar_bid):
    minutes = np.arange(N_MINUTES)

    plt.figure(figsize=(10, 5))
    for profile in F_up_out_sample[:25]:
        plt.plot(minutes, profile, alpha=0.5)

    plt.axhline(also_x_bid, linestyle="--", label=f"ALSO-X bid = {also_x_bid:.2f} kW")
    plt.axhline(cvar_bid, linestyle=":", label=f"CVaR bid = {cvar_bid:.2f} kW")

    plt.xlabel("Minute")
    plt.ylabel("Consumption (kW)")
    plt.title("Task 2.2: Out-of-Sample Profiles with Reserve Bids")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.2_out_of_sample_profile_and_bids.png", dpi=300)
    plt.show()


def plot_out_sample_shortfalls(also_x_result, cvar_result):
    also_profile_shortfall = np.max(also_x_result["shortfall"], axis=1)
    cvar_profile_shortfall = np.max(cvar_result["shortfall"], axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(also_profile_shortfall), marker="o", label="ALSO-X max profile shortfall")
    plt.plot(np.sort(cvar_profile_shortfall), marker="x", label="CVaR max profile shortfall")

    plt.xlabel("Sorted out-of-sample scenario")
    plt.ylabel("Shortfall (kW)")
    plt.title("Task 2.2: Out-of-Sample Reserve Shortfalls")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.2_shortfalls.png", dpi=300)
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
    plt.savefig("2.2_distribution_reserve_capacities.png", dpi=300)
    plt.show()


def main():
    F_up = generate_load_profiles()
    F_up_in_sample = F_up[:N_IN_SAMPLE]
    F_up_out_sample = F_up[N_IN_SAMPLE:]

    also_x_bid = solve_also_x_gurobi(F_up_in_sample)
    cvar_bid = solve_cvar_gurobi(F_up_in_sample)

    also_x_result = verify_p90_minute_level(F_up_out_sample, also_x_bid)
    cvar_result = verify_p90_minute_level(F_up_out_sample, cvar_bid)

    print("Task 2.2 — Out-of-Sample P90 Verification")
    print("=========================================")
    print_verification("ALSO-X", also_x_bid, also_x_result)
    print_verification("CVaR", cvar_bid, cvar_result)

    plot_out_sample_profiles(F_up_out_sample, also_x_bid, cvar_bid)
    plot_out_sample_shortfalls(also_x_result, cvar_result)
    plot_capacity_histogram(also_x_bid, cvar_bid, also_x_result["capacities"])


if __name__ == "__main__":
    main()
