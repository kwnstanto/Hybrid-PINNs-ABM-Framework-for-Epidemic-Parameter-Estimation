# SEIR uniform-β parameter sweep (Mesa 3.x)
# Runs 125 experiments (BETA×SIGMA×GAMMA), saves per-run CSVs, and a cumulative "Infected" plot.

import os, random, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mesa import Agent, Model

# -------------------
# Global simulation controls
# -------------------
TOTAL_STEPS     = 120
INPUT_CSV       = "df_88225.csv"
RANDOM_CONTACTS = 5  # uniform random population contacts per infectious agent per day

AGE_GROUPS = {
    '0-4':   {'contacts_per_day': 10.21, 'interaction_probability': 0.093},
    '5-9':   {'contacts_per_day': 14.81, 'interaction_probability': 0.135},
    '10-14': {'contacts_per_day': 18.22, 'interaction_probability': 0.166},
    '15-19': {'contacts_per_day': 17.58, 'interaction_probability': 0.160},
    '20-29': {'contacts_per_day': 13.57, 'interaction_probability': 0.124},
    '30-39': {'contacts_per_day': 14.14, 'interaction_probability': 0.129},
    '40-49': {'contacts_per_day': 13.83, 'interaction_probability': 0.126},
    '50-59': {'contacts_per_day': 12.30, 'interaction_probability': 0.112},
    '60-69': {'contacts_per_day':  9.21, 'interaction_probability': 0.084},
    '70+':   {'contacts_per_day':  6.89, 'interaction_probability': 0.063},
}

# Parameter grid
BETA_LIST  = [0.05, 0.06, 0.07, 0.08, 0.09]
SIGMA_LIST = [0.16, 0.18, 0.20, 0.22, 0.24]
GAMMA_LIST = [0.08, 0.09, 0.10, 0.11, 0.12]

# -------------------
# Helpers
# -------------------
def clean_state(x):
    if isinstance(x, str):
        s = x.strip().upper()
        return s if s in {"S", "E", "I", "R"} else "S"
    return "S"

def age_group_of(age):
    if age is None or pd.isna(age):
        return '20-29'
    age = int(age)
    if age <= 4:   return '0-4'
    if age <= 9:   return '5-9'
    if age <= 14:  return '10-14'
    if age <= 19:  return '15-19'
    if age <= 29:  return '20-29'
    if age <= 39:  return '30-39'
    if age <= 49:  return '40-49'
    if age <= 59:  return '50-59'
    if age <= 69:  return '60-69'
    return '70+'

def fmt(x):  # for filenames
    return str(x).replace('.', 'p')

def seed_from_params(beta, sigma, gamma):
    key = f"{beta:.6f}|{sigma:.6f}|{gamma:.6f}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**32 - 1)

# -------------------
# Agent & Model (Mesa 3.x)
# -------------------
class SEIRAgent(Agent):
    def __init__(self, model, unique_id, age, gender, family_id, work_id, school_id, infection_status):
        super().__init__(model)
        self.external_id = unique_id

        self.age       = int(age) if not pd.isna(age) else None
        self.gender    = gender
        self.family_id = int(family_id) if not pd.isna(family_id) else -999
        self.work_id   = int(work_id)   if not pd.isna(work_id)   else -999
        self.school_id = int(school_id) if not pd.isna(school_id) else -999

        self.infection_status = clean_state(infection_status)
        self.next_status      = self.infection_status

    def step(self):
        # E -> I
        if self.infection_status == 'E':
            if random.random() < self.model.SIGMA:
                self.next_status = 'I'
            return

        # I: per-contact infection trials across all layers + recovery
        if self.infection_status == 'I':
            beta = self.model.GLOBAL_BETA

            # family
            if self.family_id != -999:
                for a in self.model.by_family.get(self.family_id, []):
                    if a is self: continue
                    if a.infection_status == 'S' and a.next_status != 'E':
                        if random.random() < beta:
                            a.next_status = 'E'

            # work
            if self.work_id != -999:
                for a in self.model.by_work.get(self.work_id, []):
                    if a is self: continue
                    if a.infection_status == 'S' and a.next_status != 'E':
                        if random.random() < beta:
                            a.next_status = 'E'

            # school
            if self.school_id != -999:
                for a in self.model.by_school.get(self.school_id, []):
                    if a is self: continue
                    if a.infection_status == 'S' and a.next_status != 'E':
                        if random.random() < beta:
                            a.next_status = 'E'

            # random population contacts
            if self.model.agent_list:
                for _ in range(RANDOM_CONTACTS):
                    target = random.choice(self.model.agent_list)
                    if target is self: continue
                    if target.infection_status == 'S' and target.next_status != 'E':
                        if random.random() < beta:
                            target.next_status = 'E'

            # same-age contacts (sample ~contacts_per_day)
            if self.age is not None:
                gkey = age_group_of(self.age)
                K_age  = max(0, int(round(AGE_GROUPS[gkey]['contacts_per_day'])))
                p_gate = AGE_GROUPS[gkey]['interaction_probability']
                same_age = self.model.by_age.get(self.age, [])
                if same_age:
                    for _ in range(K_age):
                        peer = random.choice(same_age)
                        if peer is self: continue
                        if random.random() < p_gate:
                            if peer.infection_status == 'S' and peer.next_status != 'E':
                                if random.random() < beta:
                                    peer.next_status = 'E'

            # recovery
            if random.random() < self.model.GAMMA:
                self.next_status = 'R'

    def advance(self):
        self.infection_status = self.next_status

class SEIRModel(Model):
    def __init__(self, seed, beta, sigma, gamma, df):
        super().__init__(seed=seed)
        random.seed(seed); np.random.seed(seed)
        self.running = True

        # parameters
        self.GLOBAL_BETA = beta
        self.SIGMA       = sigma
        self.GAMMA       = gamma

        # storage & indices
        self.agent_list = []
        self.by_family, self.by_work, self.by_school, self.by_age = {}, {}, {}, {}

        # build agents
        for i, row in df.iterrows():
            a = SEIRAgent(
                model=self,
                unique_id=i,
                age=row.get('Age', None),
                gender=row.get('Gender', None),
                family_id=row.get('Family_ID', -999),
                work_id=row.get('Work_ID', -999),
                school_id=row.get('School_ID', -999),
                infection_status=row.get('Infection_Status', 'S'),
            )
            self.add_agent(a)

    def add_agent(self, a: SEIRAgent):
        self.agent_list.append(a)
        if a.family_id != -999: self.by_family.setdefault(a.family_id, []).append(a)
        if a.work_id   != -999: self.by_work.setdefault(a.work_id,   []).append(a)
        if a.school_id != -999: self.by_school.setdefault(a.school_id, []).append(a)
        if a.age is not None:   self.by_age.setdefault(a.age,        []).append(a)

    def one_step(self):
        for ag in self.agent_list: ag.next_status = ag.infection_status
        for ag in self.agent_list: ag.step()
        for ag in self.agent_list: ag.advance()

# -------------------
# Load population once
# -------------------
df = pd.read_csv(INPUT_CSV)
if "Infection_Status" not in df.columns:
    raise ValueError("Input CSV must include 'Infection_Status'.")
print("Initial infection status counts:")
print(df["Infection_Status"].value_counts(dropna=False))

# -------------------
# Sweep & outputs
# -------------------
os.makedirs(".", exist_ok=True)
series_for_plot = []  # (label, t array, I array)

def run_experiment(beta, sigma, gamma):
    seed = seed_from_params(beta, sigma, gamma)
    model = SEIRModel(seed=seed, beta=beta, sigma=sigma, gamma=gamma, df=df)

    rows = []
    for t in range(TOTAL_STEPS):
        model.one_step()
        S = sum(a.infection_status == 'S' for a in model.agent_list)
        E = sum(a.infection_status == 'E' for a in model.agent_list)
        I = sum(a.infection_status == 'I' for a in model.agent_list)
        R = sum(a.infection_status == 'R' for a in model.agent_list)
        rows.append({"Step": t, "Susceptible": S, "Exposed": E, "Infected": I, "Recovered": R})
    return pd.DataFrame(rows)

exp_count = 0
for beta in BETA_LIST:
    for sigma in SIGMA_LIST:
        for gamma in GAMMA_LIST:
            exp_count += 1
            label = f"β={beta}, σ={sigma}, γ={gamma}"
            print(f"[{exp_count}/125] Running {label}")

            df_run = run_experiment(beta, sigma, gamma)

            fname = f"seir_u_beta{fmt(beta)}_sigma{fmt(sigma)}_gamma{fmt(gamma)}.csv"
            df_run.to_csv(fname, index=False)
            print(f"    Saved {fname} | Peak I {df_run['Infected'].max()} at day {int(df_run['Infected'].idxmax())}")

            series_for_plot.append((label, df_run["Step"].to_numpy(), df_run["Infected"].to_numpy()))

# -------------------
# Cumulative plot: Infected per set
# -------------------
fig, ax = plt.subplots(figsize=(12, 7))
for (label, x, y) in series_for_plot:
    ax.plot(x, y, linewidth=0.8, alpha=0.55)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Infected individuals")
ax.set_title(f"SEIR uniform-β multi-layer — Infected trajectories (n={len(series_for_plot)})")
# Too many labels for a legend; omit by default. Uncomment next line if you want it for small subsets.
# ax.legend([lab for (lab, _, __) in series_for_plot], fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("seir_uniform_beta_infected_all_sets.png", dpi=144)
# plt.show()
print("Done.")
