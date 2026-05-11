# Renewables Assignment 2

Optimization and market analysis project focused on renewable energy participation in electricity markets using stochastic programming, balancing market analysis, and reliability-based decision making.

---

# Project Overview

This repository contains coursework and numerical experiments related to:

- Day-ahead electricity market participation
- Wind power generation forecasting and scenario modeling
- One-price and two-price imbalance settlement schemes
- Stochastic optimization using Gurobi
- P90 reliability requirements
- In-sample and out-of-sample verification
- Renewable energy offering strategies

The implementation is primarily written in Python and uses mathematical optimization techniques for market participation analysis.

---

# Technologies Used

- Python 3
- Gurobi Optimizer (`gurobipy`)
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install numpy pandas matplotlib gurobipy
```

> Note: Gurobi requires a valid license.

---

# Repository Structure

```text
renewables-assignment2/
├── README.md
├── .gitignore
├── Assignment Description/
│   ├── Assignment 2, Rubric table.docx
│   └── Assignment_2__renewables_in_electricity_markets_2026___Copy_.pdf
│
├── Step 1/
│   ├── A2_Step1.1.py
│   ├── A2_Step1.2.py
│   ├── A2_Step1.4_one_price.py
│   ├── A2_Step1.4_two_price.py
│   ├── Day_Ahead_Market_Price_Data.csv
│   ├── Day_Ahead_Market_Price_Data_2.csv
│   ├── Wind_Farm_Generation_Data.csv
│   └── additional scenario/result files
│
└── Step 2/
    ├── 2.1_In-sample_Decision_Making.py
    ├── 2.2_Verification_of_the_P90_Requirement.py
    └── 2.3_Energinet_Perspective.py
```

---

# Step 1 — Market Participation Modeling

This section focuses on stochastic electricity market participation for a renewable energy producer.

## Main Topics

- Wind generation scenario construction
- Electricity price scenario modeling
- System imbalance simulation
- Day-ahead offering optimization
- Balancing market settlement analysis

## Key Scripts

### `A2_Step1.1.py`
Implements scenario generation and stochastic optimization for day-ahead bidding strategies.

### `A2_Step1.2.py`
Extends the optimization framework with additional market assumptions and scenario handling.

### `A2_Step1.4_one_price.py`
Analyzes balancing market participation under a one-price settlement mechanism.

### `A2_Step1.4_two_price.py`
Analyzes balancing market participation under a two-price settlement mechanism.

---

# Step 2 — Reliability-Constrained Decision Making

This section introduces reliability-based operational constraints using P90 requirements.


## Key Scripts

### `2.1_In-sample_Decision_Making.py`
Implements stochastic optimization models for offering strategies under P90 constraints.

Key concepts:

- CVaR formulation
- ALSO-X formulation
- Ramp-limited load profiles
- Reliability-constrained optimization

### `2.2_Verification_of_the_P90_Requirement.py`
Performs out-of-sample testing to verify whether optimized decisions satisfy the required reliability level.

### `2.3_Energinet_Perspective.py`
Explores the optimization problem from the system operator or Energinet perspective.



# How to Run

Run individual scripts directly from their respective folders.

Example:

```bash
cd "Step 1"
python A2_Step1.1.py
```

or:

```bash
cd "Step 2"
python 2.1_In-sample_Decision_Making.py
```

---


# License

This repository currently does not specify a license.

Consider adding one if the project will be shared publicly.

