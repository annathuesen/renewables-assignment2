#%%
"""
Task 2.1) In-sample Decision Making: Offering Strategy Under the P90 Requirement

Minute-level implementation matching the report notation:
- F_up[omega, m] is the available flexible load in scenario omega and minute m.
- c_up is the reserve capacity bid.
- ALSO-X uses binary variables y[omega, m] to indicate violated minute-scenario pairs.
- CVaR uses beta_P90 and zeta_P90[omega, m] to control tail shortfall risk.
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


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def generate_load_profiles():
    """Generate F^up_{omega,m} profiles satisfying load and ramp constraints."""
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
    """c^up_omega = min_m F^up_{omega,m}, used only for plotting/interpretation."""
    return np.min(F_up, axis=1)


# -----------------------------------------------------------------------------
# Optimization models
# -----------------------------------------------------------------------------
def solve_also_x_gurobi(F_up_in_sample):
    """
    ALSO-X minute-level formulation:
        max c^up
        s.t. c^up - F^up_{omega,m} <= M y_{omega,m}
             sum_{omega,m} y_{omega,m} <= q
             y_{omega,m} in {0,1}

    q = floor(epsilon * |Omega_IS| * |T|)
    """
    n_omega, n_m = F_up_in_sample.shape
    q = int(np.floor(EPSILON * n_omega * n_m))

    model = gp.Model("also_x_p90_minute_level")
    model.Params.OutputFlag = 1

    c_up = model.addVar(lb=0.0, ub=P_MAX, name="c_up")
    y = model.addVars(n_omega, n_m, vtype=GRB.BINARY, name="y")

    for omega in range(n_omega):
        for m in range(n_m):
            model.addConstr(
                c_up - F_up_in_sample[omega, m] <= BIG_M * y[omega, m],
                name=f"reserve_availability_{omega}_{m}",
            )

    model.addConstr(
        gp.quicksum(y[omega, m] for omega in range(n_omega) for m in range(n_m)) <= q,
        name="violation_budget",
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(c_up.X), model.Runtime, model.NumVars, model.NumConstrs


def solve_cvar_gurobi(F_up_in_sample):
    """
    CVaR minute-level formulation using report notation:
        max c^up
        s.t. zeta^P90_{omega,m} >= c^up - F^up_{omega,m} - beta^P90
             zeta^P90_{omega,m} >= 0
             beta^P90 + 1/(epsilon*|Omega_IS|*|T|) * sum zeta^P90_{omega,m} <= 0

    beta^P90 is the VaR auxiliary variable and zeta^P90 controls excess shortfall.
    """
    n_omega, n_m = F_up_in_sample.shape
    n_pairs = n_omega * n_m

    model = gp.Model("cvar_p90_minute_level")
    model.Params.OutputFlag = 1

    c_up = model.addVar(lb=0.0, ub=P_MAX, name="c_up")
    beta_P90 = model.addVar(lb=-GRB.INFINITY, name="beta_P90")
    zeta_P90 = model.addVars(n_omega, n_m, lb=0.0, name="zeta_P90")

    for omega in range(n_omega):
        for m in range(n_m):
            model.addConstr(
                zeta_P90[omega, m] >= c_up - F_up_in_sample[omega, m] - beta_P90,
                name=f"excess_shortfall_{omega}_{m}",
            )

    model.addConstr(
        beta_P90
        + (1.0 / (EPSILON * n_pairs))
        * gp.quicksum(zeta_P90[omega, m] for omega in range(n_omega) for m in range(n_m))
        <= 0.0,
        name="cvar_constraint",
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    return float(c_up.X), model.Runtime, model.NumVars, model.NumConstrs


# -----------------------------------------------------------------------------
# Evaluation and plotting
# -----------------------------------------------------------------------------
def evaluate_minute_level(F_up, c_up):
    shortfall = np.maximum(0.0, c_up - F_up)
    violated = shortfall > 1e-8
    minute_feasibility = 1.0 - np.mean(violated)

    # Per-profile P90: at least 90% of minutes feasible in each one-hour profile
    max_violated_minutes = int(np.floor(EPSILON * N_MINUTES))
    profile_feasible = np.sum(violated, axis=1) <= max_violated_minutes

    return {
        "minute_feasibility": float(minute_feasibility),
        "profile_feasibility": float(np.mean(profile_feasible)),
        "feasible_profiles": int(np.sum(profile_feasible)),
        "total_profiles": int(F_up.shape[0]),
        "mean_shortfall": float(np.mean(shortfall)),
        "max_shortfall": float(np.max(shortfall)),
        "shortfall": shortfall,
        "violated": violated,
    }


def plot_in_sample_profiles(F_up_in_sample):
    minutes = np.arange(N_MINUTES)

    plt.figure(figsize=(10, 5))
    for profile in F_up_in_sample[:25]:
        plt.plot(minutes, profile, alpha=0.5)

    plt.xlabel("Minute")
    plt.ylabel("Consumption (kW)")
    plt.title("Task 2.1: Example In-Sample Load Profiles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.1_in-sample_load_profile.png", dpi=300)
    plt.show()


def plot_bid_vs_capacity(F_up_in_sample, also_x_bid, cvar_bid):
    sorted_capacities = np.sort(scenario_reserve_capacities(F_up_in_sample))

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
    plt.savefig("2.1_in-sample_bid_vs_cap.png", dpi=300)
    plt.show()


def plot_overbid_frequency(F_up_in_sample, also_x_bid, cvar_bid):
    also_eval = evaluate_minute_level(F_up_in_sample, also_x_bid)
    cvar_eval = evaluate_minute_level(F_up_in_sample, cvar_bid)

    also_freq = np.mean(also_eval["violated"], axis=1) * 100
    cvar_freq = np.mean(cvar_eval["violated"], axis=1) * 100

    plt.figure(figsize=(10, 5))
    plt.hist(also_freq, bins=20, alpha=0.7, label=f"ALSO-X mean = {np.mean(also_freq):.2f}%")
    plt.hist(cvar_freq, bins=20, alpha=0.7, label=f"CVaR mean = {np.mean(cvar_freq):.2f}%")
    plt.axvline(10, linestyle="--", label="P90 target (10%)")

    plt.xlabel("Overbid frequency per profile (%)")
    plt.ylabel("Number of profiles")
    plt.title("Task 2.1: In-Sample Overbid Frequency by Scenario")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.1_overbid_frequency.png", dpi=300)
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    F_up = generate_load_profiles()
    F_up_in_sample = F_up[:N_IN_SAMPLE]

    also_x_bid, also_runtime, also_vars, also_constrs = solve_also_x_gurobi(F_up_in_sample)
    cvar_bid, cvar_runtime, cvar_vars, cvar_constrs = solve_cvar_gurobi(F_up_in_sample)

    also_eval = evaluate_minute_level(F_up_in_sample, also_x_bid)
    cvar_eval = evaluate_minute_level(F_up_in_sample, cvar_bid)

    print("Task 2.1 — In-Sample Decision Making")
    print("====================================")
    print(f"ALSO-X bid: {also_x_bid:.2f} kW")
    print(f"CVaR bid:   {cvar_bid:.2f} kW")
    print()
    print("Minute-level feasibility")
    print(f"ALSO-X: {also_eval['minute_feasibility']:.2%}")
    print(f"CVaR:   {cvar_eval['minute_feasibility']:.2%}")
    print()
    print("Profile-level P90 feasibility")
    print(f"ALSO-X: {also_eval['feasible_profiles']} / {also_eval['total_profiles']} = {also_eval['profile_feasibility']:.2%}")
    print(f"CVaR:   {cvar_eval['feasible_profiles']} / {cvar_eval['total_profiles']} = {cvar_eval['profile_feasibility']:.2%}")
    print()
    print("Computational aspects")
    print(f"ALSO-X: runtime={also_runtime:.4f}s, variables={also_vars}, constraints={also_constrs}")
    print(f"CVaR:   runtime={cvar_runtime:.4f}s, variables={cvar_vars}, constraints={cvar_constrs}")

    plot_in_sample_profiles(F_up_in_sample)
    plot_bid_vs_capacity(F_up_in_sample, also_x_bid, cvar_bid)
    plot_overbid_frequency(F_up_in_sample, also_x_bid, cvar_bid)


if __name__ == "__main__":
    main()
