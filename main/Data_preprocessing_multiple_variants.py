# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:49:45 2026

@author: Konstanto
"""

import pandas as pd
import matplotlib.pyplot as plt

#  Load and clean data
data = pd.read_csv(r'...\Real_data\df_greece_world_in_data.csv')

#  Keep the desirable columns 
data_1=data.iloc[:,4:18]
data_2=data.iloc[:,26:44]; data_2 = data_2.drop(['tests_units','total_boosters'], axis=1)
data_3=data.iloc[:,45:48]
df_1=pd.concat([data_1,data_2,data_3,data['population']],axis=1,join='outer') # ad the info about population
df_1['date'] = pd.to_datetime(df_1['date'])
df_1 = df_1.sort_values('date') # sort by date
df_1.drop(df_1.index[:56],inplace=True); df_1=df_1.reset_index(drop=True) # drop the first rows (no actual info-everything zeros)
# df_1.to_csv(r'...\Real_data\Greece_SEIR\df_greece.csv', index=False)

df_1 = pd.read_csv(r'...\Real_data\Greece_SEIR\df_greece.csv') # store csv

# Data Cleaning, fill missing values with previous known value - no report
cumulative_cols_1 = ['total_cases','total_deaths','people_vaccinated']
for col in cumulative_cols_1:
    df_1[col] = df_1[col].fillna(method='ffill').fillna(0)
    
# Data Cleaning, fill missing values with 0 
cumulative_cols_2 = ['new_cases','new_deaths','new_vaccinations']
for col in cumulative_cols_2:
    df_1[col] = df_1[col].fillna(0)  

cols=['date','population','reproduction_rate'] + cumulative_cols_1 + cumulative_cols_2

df = df_1[cols].copy() # daily dataset
# df.to_csv(r'...\Real_data\Greece_SEIR\test_daily.csv',index=False) # store csv

df_weekly = df.iloc[::7].reset_index(drop=True).copy() # weekly dataset
# df_weekly.to_csv(r'...\Real_data\Greece_SEIR\test_weekly.csv',index=False)

variants = [
    {"name": "initial", "start": 0, "end": 45, "recovery": 12, "incubation": 5},
    {"name": "alpha",   "start": 45, "end": 69, "recovery": 7,  "incubation": 5},
    {"name": "delta",   "start": 69, "end": 93, "recovery": 6,  "incubation": 4},
    {"name": "omicron", "start": 93, "end": 218, "recovery": 6,  "incubation": 3},
]

df_weekly['Recovered'] = 0

for v in variants:
    
    start = v["start"]
    end = v["end"]
    r = v["recovery"]

    recovered = (
        df_weekly['total_cases'].shift(r)
        - df_weekly['total_deaths'].shift(r)
    )

    df_weekly.loc[start:end, 'Recovered'] = recovered.loc[start:end]

df_weekly['Recovered'] = df_weekly['Recovered'].clip(lower=0).fillna(0)


df_weekly['Infected'] = (df_weekly['total_cases']- df_weekly['Recovered']- df_weekly['total_deaths']).clip(lower=0)

df_weekly['Deaths'] = df_weekly['total_deaths']

# For SEIR the 'true' recovered are: R = Recovered + Deaths
df_weekly['Removed'] = df_weekly['Recovered'] + df_weekly['Deaths']

# Estimation of Exposed (E), people in 'Exposed' today will become 'Infected' over the an assumed n-day incubation period. 
df_weekly['Exposed'] = 0

for v in variants:

    start = v["start"]
    end = v["end"]
    incubation = v["incubation"]

    exposed = (
        df_weekly['new_cases']
        .shift(-incubation)
        .rolling(incubation)
        .sum()
    )

    df_weekly.loc[start:end, 'Exposed'] = exposed.loc[start:end]

df_weekly['Exposed'] = df_weekly['Exposed'].bfill().fillna(0)

# Define Vaccinated (V) and Deaths (D)
def get_vaccinated(df_data):
    df_data['Vaccinated'] = df_data['people_vaccinated']
    return df_data


def get_susceptible_seir(df_data):
# Estimate Susceptible S= Population - (E + I + R + V + Deaths), ensuring also the conservation of the population: N = S + E + I + R + V + Deaths
    df_data['Susceptible'] = df_data['population'] - (df_data['Exposed'] + df_data['Infected'] + df_data['Recovered'] + df_data['Deaths'])
    df_data['Susceptible'] = df_data['Susceptible'].clip(lower=0)
    columns = ['date', 'population', 'Susceptible', 'Exposed', 'Infected', 'Recovered', 'Deaths', 'Removed']
    df_data = df_data[columns]
    return df_data

seir_df = get_susceptible_seir(df_weekly)

check = (
    seir_df['Susceptible']
    + seir_df['Exposed']
    + seir_df['Infected']
    + seir_df['Recovered']
    + seir_df['Deaths']
)

print("Max population error:", (check - seir_df['population']).abs().max())

t = seir_df.index.values.astype(float).reshape(-1,1)
S = seir_df['Susceptible'].values.astype(float).reshape(-1,1)
E = seir_df['Exposed'].values.astype(float).reshape(-1,1)
I = seir_df['Infected'].values.astype(float).reshape(-1,1)
R = seir_df['Removed'].values.astype(float).reshape(-1,1)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1); plt.plot(S, lw=3, label='Data'); plt.title("S"); plt.xlabel('Time(Weeks)'); plt.grid(True); plt.legend(); plt.ylabel('Individuals')
plt.subplot(2,2,2); plt.plot(E, lw=3, label='Data'); plt.title("E"); plt.xlabel('Time(Weeks)'); plt.grid(True); plt.legend()
plt.subplot(2,2,3); plt.plot(I, lw=3, label='Data'); plt.title("I"); plt.xlabel('Time(Weeks)'); plt.grid(True); plt.legend()
plt.subplot(2,2,4); plt.plot(R, lw=3, label='Data'); plt.title("R"); plt.xlabel('Time(Weeks)'); plt.grid(True); plt.legend(); plt.ylabel('Individuals')
plt.tight_layout(); plt.show()

# seir_df.to_csv(r'...\Real_data\Greece_SEIR\seir_weekly_processed_data(multiple_variants).csv', index=False)
