# SEIR agent-based model (Mesa 3.3 compatible) without scheduler (manual loop)
# States: S (susceptible), E (exposed), I (infectious), R (removed)
# Transitions per day:
#   S -> E via contact with I (prob BETA per contact)
#   E -> I with prob SIGMA
#   I -> R with prob GAMMA
#
# Outputs:
#   scenario_1.csv ... scenario_10.csv  (per-run counts per day)
#   average_scenario_rounded.csv        (mean over runs, rounded)
#   seir_average.png                    (plot of averaged trajectories)

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.datacollection import DataCollector

# -------------------
# Parameters
# -------------------
DAILY_CONTACTS = 5      # contacts per infectious agent per day
BETA = 0.20             # infection probability per contact
SIGMA = 1 / 3.0         # ~3-day average incubation
GAMMA = 1 / 7.0         # ~7-day average infectious period
TOTAL_STEPS = 120       # <<< reduced from 180 to 120 days >>>
INPUT_CSV = "df_88225.csv"

VALID_STATES = {"S", "E", "I", "R"}

def clean_state(x):
    if isinstance(x, str):
        s = x.strip().upper()
        return s if s in VALID_STATES else "S"
    return "S"

# -------------------
# Load population
# -------------------
print("Loading df...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded df shape: {df.shape}")
print("Columns:", df.columns.tolist())
if "Infection_Status" not in df.columns:
    raise ValueError("Input CSV must include 'Infection_Status' column.")
print("Initial infection status counts:")
print(df["Infection_Status"].value_counts(dropna=False))

# -------------------
# Agent
# -------------------
class SEIRAgent(Agent):
    def __init__(self, model, row, infection_status, external_id=None):
        super().__init__(model)         # Mesa 3.3 expects only 'model' here
        self.external_id = external_id  # keep DataFrame index if needed

        # Optional attributes from your CSV
        self.age = row.get("Age", None)
        self.gender = row.get("Gender", None)
        self.family_id = row.get("Family_ID", None)
        self.work_id = row.get("Work_ID", None)
        self.school_id = row.get("School_ID", None)

        # Epidemiological state
        self.infection_status = infection_status
        self.next_status = infection_status

    def step(self):
        if self.infection_status == "E":
            if random.random() < SIGMA:
                self.next_status = "I"

        elif self.infection_status == "I":
            for _ in range(DAILY_CONTACTS):
                target = random.choice(self.model.agents_list)
                if target.infection_status == "S" and random.random() < BETA:
                    target.next_status = "E"
            if random.random() < GAMMA:
                self.next_status = "R"

    def advance(self):
        self.infection_status = self.next_status


# -------------------
# Model
# -------------------
class SEIRModel(Model):
    def __init__(self, seed=42):
        super().__init__(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        self.running = True
        self.agents_list = []

        for i, row in df.iterrows():
            status = clean_state(row["Infection_Status"])
            a = SEIRAgent(self, row, status, external_id=i)
            self.agents_list.append(a)

        for a in self.agents_list:
            a.next_status = a.infection_status

        print("Unique infection statuses in agents:", {a.infection_status for a in self.agents_list})
        print(f"Model initialized with {len(self.agents_list)} agents")

        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: sum(a.infection_status == "S" for a in m.agents_list),
                "Exposed":     lambda m: sum(a.infection_status == "E" for a in m.agents_list),
                "Infected":    lambda m: sum(a.infection_status == "I" for a in m.agents_list),
                "Removed":     lambda m: sum(a.infection_status == "R" for a in m.agents_list),
            }
        )

    def step(self):
        for a in self.agents_list:
            a.next_status = a.infection_status
        for a in self.agents_list:
            a.step()
        for a in self.agents_list:
            a.advance()
        self.datacollector.collect(self)


# -------------------
# Simulation (10 runs)
# -------------------
os.makedirs(".", exist_ok=True)
paths = []

for iteration in range(1, 11):
    seed = iteration
    print("=" * 50)
    print(f"Iteration {iteration}")
    print("=" * 50)

    model = SEIRModel(seed=seed)

    intermediate = []
    peak_I = -1
    peak_day = -1

    for t in range(TOTAL_STEPS):
        model.step()
        S = sum(a.infection_status == "S" for a in model.agents_list)
        E = sum(a.infection_status == "E" for a in model.agents_list)
        I = sum(a.infection_status == "I" for a in model.agents_list)
        R = sum(a.infection_status == "R" for a in model.agents_list)
        intermediate.append({"Step": t, "Susceptible": S, "Exposed": E, "Infected": I, "Removed": R})
        if I > peak_I:
            peak_I, peak_day = I, t

    out = f"scenario_{iteration}.csv"
    pd.DataFrame(intermediate).to_csv(out, index=False)
    paths.append(out)
    print(f"[Iter {iteration}] Peak infected: {peak_I} on day {peak_day}")

# -------------------
# Aggregate & Plot
# -------------------
all_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
avg = all_df.groupby("Step", as_index=False)[["Susceptible", "Exposed", "Infected", "Removed"]].mean()
avg_rounded = avg.round(0)
avg_rounded.to_csv("average_scenario_rounded.csv", index=False)

if not avg.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg["Step"], avg["Susceptible"], label="Susceptible")
    ax.plot(avg["Step"], avg["Exposed"], label="Exposed")
    ax.plot(avg["Step"], avg["Infected"], label="Infected")
    ax.plot(avg["Step"], avg["Removed"], label="Removed")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of individuals")
    ax.set_title("SEIR Model Simulation (average of 10 runs, 120 days)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("seir_average.png", dpi=144)
    # plt.show()  # enable if you have a GUI backend
