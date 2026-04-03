# SEIR parameter sweep (Mesa 3.3 compatible) — 125 experiments, 1 run per (BETA,SIGMA,GAMMA)
# Population CSV: df_88225.csv
# Output per experiment: seir_beta{b}_sigma{s}_gamma{g}.csv
# Cumulative plot: seir_infected_all_sets.png

import os
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.datacollection import DataCollector

# -------------------
# Fixed simulation controls
# -------------------
DAILY_CONTACTS = 5
TOTAL_STEPS = 120
INPUT_CSV = "df_88225.csv"

# Parameter grid (your request)
BETA_LIST  = [0.18, 0.19, 0.20, 0.21, 0.22]
SIGMA_LIST = [0.35, 0.40, 0.45, 0.50, 0.55]
GAMMA_LIST = [0.10, 0.125, 0.15, 0.175, 0.20]

VALID_STATES = {"S", "E", "I", "R"}

def clean_state(x):
    if isinstance(x, str):
        s = x.strip().upper()
        return s if s in VALID_STATES else "S"
    return "S"

def fmt_component(x):
    """Make a safe filename component: 0.125 -> '0p125'"""
    return str(x).replace('.', 'p')

def seed_from_params(beta, sigma, gamma):
    """Deterministic seed from tuple of floats."""
    key = f"{beta:.6f}|{sigma:.6f}|{gamma:.6f}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**32 - 1)

# -------------------
# Load population
# -------------------
print("Loading df...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded df shape: {df.shape}")
if "Infection_Status" not in df.columns:
    raise ValueError("Input CSV must include 'Infection_Status' column.")
print("Initial infection status counts:")
print(df["Infection_Status"].value_counts(dropna=False))
print()

# -------------------
# Agent (Mesa 3.3 signature)
# -------------------
class SEIRAgent(Agent):
    def __init__(self, model, row, infection_status, external_id=None):
        super().__init__(model)  # Mesa 3.3: only model here
        self.external_id = external_id
        # Optional attributes from CSV (Series.get works)
        self.age = row.get("Age", None)
        self.gender = row.get("Gender", None)
        self.family_id = row.get("Family_ID", None)
        self.work_id = row.get("Work_ID", None)
        self.school_id = row.get("School_ID", None)

        self.infection_status = infection_status
        self.next_status = infection_status

# -------------------
# Model
# -------------------
class SEIRModel(Model):
    def __init__(self, seed, beta, sigma, gamma):
        # Mesa 3.3 requires keyword seed
        super().__init__(seed=seed)
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

        random.seed(seed)
        np.random.seed(seed)

        self.running = True
        self.agents_list = []

        for i, row in df.iterrows():
            status = clean_state(row["Infection_Status"])
            a = SEIRAgent(self, row, status, external_id=i)
            self.agents_list.append(a)

        # Initialize next_status
        for a in self.agents_list:
            a.next_status = a.infection_status

        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: sum(a.infection_status == "S" for a in m.agents_list),
                "Exposed":     lambda m: sum(a.infection_status == "E" for a in m.agents_list),
                "Infected":    lambda m: sum(a.infection_status == "I" for a in m.agents_list),
                "Removed":     lambda m: sum(a.infection_status == "R" for a in m.agents_list),
            }
        )

    def step(self):
        # Reset next_status
        for a in self.agents_list:
            a.next_status = a.infection_status

        # 1) E -> I transitions (process only exposed agents)
        for a in self.agents_list:
            if a.infection_status == "E":
                if random.random() < self.sigma:
                    a.next_status = "I"

        # 2) Infection + Recovery for infectious agents (process only I agents)
        infectious_agents = [a for a in self.agents_list if a.infection_status == "I"]

        # Infection attempts
        for inf in infectious_agents:
            for _ in range(DAILY_CONTACTS):
                target = random.choice(self.agents_list)
                if target.infection_status == "S" and random.random() < self.beta:
                    target.next_status = "E"

        # Recovery for I
        for inf in infectious_agents:
            if random.random() < self.gamma:
                inf.next_status = "R"

        # 3) Apply transitions
        for a in self.agents_list:
            a.infection_status = a.next_status

        # 4) Collect
        self.datacollector.collect(self)

# -------------------
# Run one experiment
# -------------------
def run_experiment(beta, sigma, gamma, steps=TOTAL_STEPS):
    seed = seed_from_params(beta, sigma, gamma)
    model = SEIRModel(seed=seed, beta=beta, sigma=sigma, gamma=gamma)

    out_rows = []
    peak_I, peak_day = -1, -1

    for t in range(steps):
        model.step()
        # Counts
        S = sum(a.infection_status == "S" for a in model.agents_list)
        E = sum(a.infection_status == "E" for a in model.agents_list)
        I = sum(a.infection_status == "I" for a in model.agents_list)
        R = sum(a.infection_status == "R" for a in model.agents_list)
        out_rows.append({"Step": t, "Susceptible": S, "Exposed": E, "Infected": I, "Removed": R})
        if I > peak_I:
            peak_I, peak_day = I, t

    df_out = pd.DataFrame(out_rows)
    return df_out, peak_I, peak_day

# -------------------
# Sweep
# -------------------
os.makedirs(".", exist_ok=True)

all_series = []   # (label, Step array, Infected array)
n_total = len(BETA_LIST) * len(SIGMA_LIST) * len(GAMMA_LIST)
k = 0

for beta in BETA_LIST:
    for sigma in SIGMA_LIST:
        for gamma in GAMMA_LIST:
            k += 1
            label = f"β={beta}, σ={sigma}, γ={gamma}"
            print(f"[{k}/{n_total}] Running {label}")

            run_df, peak_I, peak_day = run_experiment(beta, sigma, gamma, steps=TOTAL_STEPS)

            # Save CSV named by params
            fname = f"seir_beta{fmt_component(beta)}_sigma{fmt_component(sigma)}_gamma{fmt_component(gamma)}.csv"
            run_df.to_csv(fname, index=False)
            print(f"    Saved: {fname} | Peak I = {peak_I} on day {peak_day}")

            # Keep for cumulative plot
            all_series.append((label, run_df["Step"].to_numpy(), run_df["Infected"].to_numpy()))

# -------------------
# Cumulative plot: Infected per set
# -------------------
if all_series:
    fig, ax = plt.subplots(figsize=(12, 7))
    # Too many lines for a readable legend; plot thin, semi-transparent lines
    for (label, x, y) in all_series:
        ax.plot(x, y, linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected individuals")
    ax.set_title(f"SEIR — Infected trajectories for {len(all_series)} parameter sets")

    # Legend with many entries is unreadable; include only if <= 20 sets
    if len(all_series) <= 20:
        ax.legend([lab for (lab, _, __) in all_series], fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig("seir_infected_all_sets.png", dpi=144)
    # plt.show()  # enable if you have a GUI backend

print("Done.")
