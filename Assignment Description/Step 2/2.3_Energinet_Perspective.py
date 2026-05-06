#%%
"""
Task 2.3) Energinet Perspective
Analyze how modifying the P90 requirement (e.g., threshold between 80% and 100%) affects the
optimal reserve bid (in-sample) and the expected reserve shortfall (out-of-sample) when using
ALSO-X. Discuss whether a trade-off emerges between higher reliability and reduced reserve
provision.
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


def solve_also_x_gurobi(capacities, reliability_level):
    capacities = np.asarray(capacities)
    n = len(capacities)

    reliability_level = min(max(reliability_level, 0.0), 1.0)
    allowed_violations = int(np.floor((1.0 - reliability_level) * n))
    allowed_violations = max(0, allowed_violations)

    model = gp.Model("also_x_reliability_sweep")
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


def evaluate_out_of_sample(out_sample_profiles, bid):
    capacities = scenario_reserve_capacities(out_sample_profiles)
    shortfalls = np.maximum(0.0, bid - capacities)
    feasible = shortfalls == 0.0

    return {
        "out_sample_feasibility": float(np.mean(feasible)),
        "expected_shortfall": float(np.mean(shortfalls)),
        "max_shortfall": float(np.max(shortfalls)),
    }


def run_reliability_sweep(in_sample_profiles, out_sample_profiles):
    in_sample_capacities = scenario_reserve_capacities(in_sample_profiles)
    reliability_levels = np.round(np.linspace(0.80, 1.00, 21), 2)

    results = []

    for reliability in reliability_levels:
        bid = solve_also_x_gurobi(in_sample_capacities, reliability)
        metrics = evaluate_out_of_sample(out_sample_profiles, bid)

        results.append(
            {
                "reliability": reliability,
                "bid": bid,
                **metrics,
            }
        )

    return results


def print_results(results):
    print("Task 2.3 — Energinet Perspective using ALSO-X")
    print("============================================")
    print(
        f"{'Reliability':>12} | {'Bid kW':>10} | "
        f"{'OOS feasible':>12} | {'Exp. shortfall kW':>18} | {'Max shortfall kW':>17}"
    )
    print("-" * 84)

    for row in results:
        print(
            f"{row['reliability']:>11.0%} | "
            f"{row['bid']:>10.2f} | "
            f"{row['out_sample_feasibility']:>11.2%} | "
            f"{row['expected_shortfall']:>18.2f} | "
            f"{row['max_shortfall']:>17.2f}"
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
    plt.show()


def print_discussion():
    print("\nDiscussion")
    print("----------")
    print(
        "As the reliability threshold increases from 80% to 100%, ALSO-X allows "
        "fewer in-sample violations."
    )
    print(
        "The optimal reserve bid therefore decreases or stays constant because "
        "the bid must be feasible for more scenarios."
    )
    print(
        "The expected out-of-sample shortfall generally decreases when the "
        "reliability requirement becomes stricter."
    )
    print(
        "A trade-off appears: higher reliability reduces shortfall risk, but it "
        "also reduces the amount of reserve capacity offered."
    )


def main():
    profiles = generate_load_profiles()

    in_sample_profiles = profiles[:N_IN_SAMPLE]
    out_sample_profiles = profiles[N_IN_SAMPLE:]

    results = run_reliability_sweep(in_sample_profiles, out_sample_profiles)

    print_results(results)
    print_discussion()

    plot_bid_vs_reliability(results)
    plot_shortfall_vs_reliability(results)
    plot_oos_feasibility_vs_reliability(results)


if __name__ == "__main__":
    main()