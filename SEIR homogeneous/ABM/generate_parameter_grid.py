# generate_parameter_grid.py
import pandas as pd

BETA_LIST  = [0.18, 0.19, 0.20, 0.21, 0.22]
SIGMA_LIST = [0.35, 0.40, 0.45, 0.50, 0.55]
GAMMA_LIST = [0.10, 0.125, 0.15, 0.175, 0.20]

def fmt(x): return str(x).replace('.', 'p')

rows = []
exp_id = 0

for beta in BETA_LIST:
    for sigma in SIGMA_LIST:
        for gamma in GAMMA_LIST:
            exp_id += 1
            filename = f"seir_beta{fmt(beta)}_sigma{fmt(sigma)}_gamma{fmt(gamma)}.csv"
            rows.append({
                "Experiment": exp_id,
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "Filename": filename
            })

df = pd.DataFrame(rows)
df.to_csv("parameter_grid.csv", index=False)
print("✅ Saved parameter_grid.csv with", len(df), "rows")
print(df.head())

