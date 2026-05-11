#%%
"""
Task 2.3) Energinet Perspective

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


def solve_also_x_gurobi(F_up_in_sample, reliability_level):
    reliability_level = min(max(float(reliability_level), 0.0), 1.0)
    epsilon = 1.0 - reliability_level

    n_omega, n_m = F_up_in_sample.shape
    q = int(np.floor(epsilon * n_omega * n_m))
    q = max(0, q)

    model = gp.Model("also_x_reliability_sweep_minute_level")
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


def evaluate_out_of_sample(F_up_out_sample, c_up, reliability_level):
    epsilon = 1.0 - reliability_level
    shortfall = np.maximum(0.0, c_up - F_up_out_sample)
    violated = shortfall > 1e-8

    # Minute-based feasibility across all OOS minute-scenario pairs
    out_sample_feasibility = 1.0 - np.mean(violated)

    # Profile-based P-requirement: each profile may have at most floor(epsilon*60) violated minutes
    max_violated_minutes = int(np.floor(epsilon * N_MINUTES))
    profile_feasible = np.sum(violated, axis=1) <= max_violated_minutes

    return {
        "out_sample_feasibility": float(out_sample_feasibility),
        "profile_feasibility": float(np.mean(profile_feasible)),
        "expected_shortfall": float(np.mean(shortfall)),
        "max_shortfall": float(np.max(shortfall)),
    }


def run_reliability_sweep(F_up_in_sample, F_up_out_sample):
    reliability_levels = np.round(np.linspace(0.80, 1.00, 21), 2)
    results = []

    for reliability in reliability_levels:
        c_up = solve_also_x_gurobi(F_up_in_sample, reliability)
        metrics = evaluate_out_of_sample(F_up_out_sample, c_up, reliability)

        results.append({"reliability": reliability, "bid": c_up, **metrics})

    return results


def print_results(results):
    print("Task 2.3 — Energinet Perspective using ALSO-X")
    print("============================================")
    print(
        f"{'Reliability':>12} | {'Bid kW':>10} | {'OOS minute feasible':>19} | "
        f"{'OOS profile feasible':>20} | {'Exp. shortfall kW':>18} | {'Max shortfall kW':>17}"
    )
    print("-" * 110)

    for row in results:
        print(
            f"{row['reliability']:>11.0%} | "
            f"{row['bid']:>10.2f} | "
            f"{row['out_sample_feasibility']:>18.2%} | "
            f"{row['profile_feasibility']:>19.2%} | "
            f"{row['expected_shortfall']:>18.4f} | "
            f"{row['max_shortfall']:>17.4f}"
        )


def plot_bid_vs_reliability(results):
    reliability = [row["reliability"] * 100 for row in results]
    bids = [row["bid"] for row in results]

    plt.figure(figsize=(10, 5))
    plt.plot(reliability, bids, marker="o")
    plt.xlabel("Reliability requirement (%)")
    plt.ylabel("Optimal reserve bid (kW)")
    plt.title("Task 2.3: Reserve Bid vs Reliability Requirement")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.3_bid_vs_reliability.png", dpi=300)
    plt.show()


def plot_shortfall_vs_reliability(results):
    reliability = [row["reliability"] * 100 for row in results]
    expected_shortfalls = [row["expected_shortfall"] for row in results]
    max_shortfalls = [row["max_shortfall"] for row in results]

    plt.figure(figsize=(10, 5))
    plt.plot(reliability, expected_shortfalls, marker="o", label="Expected shortfall")
    plt.plot(reliability, max_shortfalls, marker="x", label="Maximum shortfall")
    plt.xlabel("Reliability requirement (%)")
    plt.ylabel("Out-of-sample shortfall (kW)")
    plt.title("Task 2.3: Out-of-Sample Shortfall vs Reliability Requirement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.3_shortfall_vs_reliability.png", dpi=300)
    plt.show()


def plot_oos_feasibility_vs_reliability(results):
    reliability = [row["reliability"] * 100 for row in results]
    feasibility = [row["out_sample_feasibility"] * 100 for row in results]

    plt.figure(figsize=(10, 5))
    plt.plot(reliability, feasibility, marker="o")
    plt.axhline(90, linestyle="--", label="P90 target")
    plt.xlabel("Reliability requirement (%)")
    plt.ylabel("Out-of-sample feasibility (%)")
    plt.title("Task 2.3: Out-of-Sample Feasibility vs Reliability Requirement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.3_feasibility_vs_realiability.png", dpi=300)
    plt.show()


def main():
    F_up = generate_load_profiles()
    F_up_in_sample = F_up[:N_IN_SAMPLE]
    F_up_out_sample = F_up[N_IN_SAMPLE:]

    results = run_reliability_sweep(F_up_in_sample, F_up_out_sample)
    print_results(results)

    plot_bid_vs_reliability(results)
    plot_shortfall_vs_reliability(results)
    plot_oos_feasibility_vs_reliability(results)


if __name__ == "__main__":
    main()
