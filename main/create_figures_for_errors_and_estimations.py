# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:13:16 2026

@author: Konstanto
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statistics
from decimal import Decimal
from sklearn.metrics import r2_score, mean_absolute_error

df_errors = pd.read_csv(r'...\SEIR homogeneous\results\relative_errors.csv')
df_estimations = pd.read_csv(r'...\SEIR homogeneous\results\estimations.csv')

    ## --------- Statistics Table --------- ##

beta_mean = format(Decimal(df_errors['beta'].mean()), '.2f')
sigma_mean = format(Decimal(df_errors['sigma'].mean()), '.2f')
gamma_mean = format(Decimal(df_errors['gamma'].mean()), '.2f')

beta_median = statistics.median(df_errors['beta'])
sigma_median = statistics.median(df_errors['sigma'])
gamma_median = statistics.median(df_errors['gamma'])

beta_std = format(Decimal(df_errors['beta'].std()), '.2f')
sigma_std = format(Decimal(df_errors['sigma'].std()), '.2f')
gamma_std = format(Decimal(df_errors['gamma'].std()), '.2f')

beta_max = df_errors['beta'].max()
sigma_max = df_errors['sigma'].max()
gamma_max = df_errors['gamma'].max()

beta_min = df_errors['beta'].min()
sigma_min = df_errors['sigma'].min()
gamma_min = df_errors['gamma'].min()


df_errors['beta'].values[111]
df_errors.iloc[111]

cases_of_beta = (df_errors['beta']<10).sum()
cases_of_beta_perc = (df_errors['beta']<10).sum()/125*100
cases_of_sigma = (df_errors['sigma']<1).sum()
cases_of_sigma_perc = (df_errors['sigma']<1).sum()/125*100
cases_of_gamma = (df_errors['gamma']<2).sum()
cases_of_gamma_perc = (df_errors['gamma']<2).sum()/125*100

data_stat = {
        'Mean':[beta_mean, sigma_mean, gamma_mean], 
        'Median':[beta_median, sigma_median, gamma_median], 
        'Std':[beta_std, sigma_std, gamma_std],
        'Maximum':[beta_max, sigma_max, gamma_max], 
        'Minimum':[beta_min, sigma_min, gamma_min],
        }

rows_stat = ['β','σ','γ']

table_stat = pd.DataFrame(data_stat,index=rows_stat)

print('statistics',table_stat)

    ## --------- Cases Table --------- ##

data_cases = {
        'Count' : [cases_of_beta, cases_of_sigma, cases_of_gamma],
        'Percentage': [cases_of_beta_perc, cases_of_sigma_perc, cases_of_gamma_perc],
        }

rows_cases = ['β error < 10%','σ error < 1%','γ error < 2%']
table_cases = pd.DataFrame(data_cases,index=rows_cases)

print('cases', table_cases)

beta_true=np.array(df_estimations['beta_true'])
sigma_true=np.array(df_estimations['sigma_true'])
gamma_true=np.array(df_estimations['gamma_true'])

beta_pred=np.array(df_estimations['beta'])
sigma_pred=np.array(df_estimations['sigma'])
gamma_pred=np.array(df_estimations['gamma'])

    ## --------- Paramaeters Recovery figure --------- ##

def recovery_plot(true, pred, name, ax):

    ax.scatter(true, pred, s=35, alpha=0.7)
    
    # identity line
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    
    ax.text(0.05, 0.95,
            f"$R^2$={r2:.3f}\nMAE={mae:.2e}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white"))
    
    ax.set_xlabel(f"True {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(f"{name} recovery")
    ax.grid(True)

fig, axs = plt.subplots(1, 3, figsize=(15,4))

recovery_plot(beta_true, beta_pred, r'$\beta$', axs[0])
recovery_plot(sigma_true, sigma_pred, r'$\sigma$', axs[1])
recovery_plot(gamma_true, gamma_pred, r'$\gamma$', axs[2])

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\parameter_recovery.pdf", dpi=600)
plt.show() 

    ## --------- Boxplot --------- ##
err_beta = np.abs(beta_pred  - beta_true) / np.abs(beta_true)
err_sigma = np.abs(sigma_pred - sigma_true) / np.abs(sigma_true)
err_gamma = np.abs(gamma_pred - gamma_true) / np.abs(gamma_true)

data_errors = [err_beta, err_sigma, err_gamma]

plt.figure(figsize=(6,5))

plt.boxplot(
    data_errors,
    labels=[r'$\beta$', r'$\sigma$', r'$\gamma$'],
    showfliers=True
)

plt.ylabel("Relative error") 
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\error_distribution.pdf", dpi=600)
plt.show()

    ## ---------- identifiability figure--------- ##
fig, axs = plt.subplots(1, 3, figsize=(15,4))

axs[0].scatter(beta_true, err_beta, alpha=0.7, s=35)
axs[1].scatter(sigma_true, err_sigma, alpha=0.7, s=35)
axs[2].scatter(gamma_true, err_gamma, alpha=0.7, s=35)


axs[0].set_title(r'$\beta$ identifiability')
axs[1].set_title(r'$\sigma$ identifiability')
axs[2].set_title(r'$\gamma$ identifiability')

for ax in axs:
    ax.set_ylabel("Relative error")
    ax.set_xlabel("True value")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\identifiability_regions.pdf", dpi=600)
plt.show()

      #------------- Heatmap ----------------#
total_error = (err_beta + err_gamma + err_sigma) / 3
nbins = 6
beta_bins = np.linspace(min(beta_true), max(beta_true),  nbins)
gamma_bins = np.linspace(min(gamma_true), max(gamma_true), nbins)

heatmap = np.zeros((nbins-2, nbins-2))
counts = np.zeros((nbins-2, nbins-2))

for b, g, e in zip(beta_true, gamma_true, total_error):

    i = np.digitize(b, beta_bins) - 1
    j = np.digitize(g, gamma_bins) - 1

    if 0 <= i < nbins-1 and 0 <= j < nbins-1:
        heatmap[j, i] += e
        counts[j, i] += 1

heatmap = np.divide(heatmap, counts, where=counts>0)
heatmap[counts==0] = np.nan

plt.figure(figsize=(6,5))

im = plt.imshow(
    heatmap,
    origin='lower',
    aspect='auto',
    extent=[beta_bins[0], beta_bins[-1], gamma_bins[0], gamma_bins[-1]]
)

plt.colorbar(im, label="Mean relative error")
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.title("SEIR identifiability map")

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\identifiability_heatmap.png", dpi=300)
plt.show()

#----- Corelation analysis ---------#

log_eb = np.log10(err_beta)
log_eg = np.log10(err_gamma)
log_es = np.log10(err_sigma)

corr_bg = np.corrcoef(err_beta, err_gamma)[0,1]
corr_bs = np.corrcoef(err_beta, err_sigma)[0,1]
corr_gs = np.corrcoef(err_gamma, err_sigma)[0,1]

print("corr(β,γ) =", corr_bg)
print("corr(β,σ) =", corr_bs)
print("corr(γ,σ) =", corr_gs)

fig, axs = plt.subplots(1,3, figsize=(14,4))

axs[0].scatter(err_beta, err_gamma, alpha=0.6)
axs[0].set_title("β–γ coupling")

axs[1].scatter(err_beta, err_sigma, alpha=0.6)
axs[1].set_title("β–σ coupling")

axs[2].scatter(err_gamma, err_sigma, alpha=0.6)
axs[2].set_title("γ–σ coupling")

for ax in axs:
    ax.set_xlabel("Error")
    ax.set_ylabel("Error")
    ax.grid(True)

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\error_correlation.png", dpi=600)
plt.show()


##--------- Ro Recovery Plot ---------##
R0_true = beta_true / gamma_true
R0_pred = beta_pred / gamma_pred

err_R0 = np.abs(R0_pred - R0_true) / np.abs(R0_true)

plt.figure(figsize=(5,5))

plt.scatter(R0_true, R0_pred, alpha=0.7, s=35)

min_val = min(min(R0_true), min(R0_pred))
max_val = max(max(R0_true), max(R0_pred))

plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

r2 = r2_score(R0_true, R0_pred)
mae = mean_absolute_error(R0_true, R0_pred)

plt.text(0.05, 0.95,
         f"$R^2$={r2:.4f}\nMAE={mae:.3e}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white"))

plt.xlabel(r"True $R_0$")
plt.ylabel(r"Predicted $R_0$")
plt.title(r"$R_0$ recovery")

plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\R0_recovery.pdf", dpi=600)
plt.show()


plt.figure(figsize=(4,5))
plt.boxplot(err_R0, labels=[r"$R_0$"])
plt.ylabel("Relative error")
plt.tight_layout()
# plt.savefig(r"F:\Users\Xristos\Downloads\R0_error.pdf", dpi=600)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FuncFormatter

csv_path = r'C:\Users\Konstanto.DESKTOP-GJ4K86G\Desktop\New\Real_data\Δεδομένα covid\Greece_SEIR(V)\seir_weekly_processed_data(multiple_variants)-test.csv'
df = pd.read_csv(csv_path)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Variant periods
variant_periods = {
    "Initial": (df.iloc[:45]['date'].iloc[0], df.iloc[:45]['date'].iloc[-1]),
    "Alpha": (df.iloc[46:69]['date'].iloc[0], df.iloc[46:69]['date'].iloc[-1]),
    "Delta": (df.iloc[70:93]['date'].iloc[0], df.iloc[70:93]['date'].iloc[-1]),
    "Omicron": (df.iloc[94:]['date'].iloc[0], df.iloc[94:]['date'].iloc[-1]),
}

variant_colors = {
    "Initial": "#e0e0e0",
    "Alpha": "#9ecae1",
    "Delta": "#a1d99b",
    "Omicron": "#fdae6b"
}

dates = df['date']
S = df['Susceptible']
E = df['Exposed']
I = df['Infected']
R = df['Recovered']

# Global style (cleaner for papers)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

fig, axs = plt.subplots(2, 2, figsize=(14,10), sharex=True)

# Colors
colors = {
    "S": "#1f77b4",
    "E": "#ff7f0e",
    "I": "#d62728",
    "R": "#2ca02c"
}

axs[0,0].plot(dates, S, lw=2.5, color=colors["S"], label='Susceptible')
axs[0,0].set_ylabel("Individuals")
axs[0,0].legend()

axs[0,1].plot(dates, E, lw=2.5, color=colors["E"], label='Exposed')
axs[0,1].legend()

axs[1,0].plot(dates, I, lw=2.5, color=colors["I"], label='Infected')
axs[1,0].set_ylabel("Individuals")
axs[1,0].legend()

axs[1,1].plot(dates, R, lw=2.5, color=colors["R"], label='Removed')
axs[1,1].legend()

formatter = FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')

axs[0,0].yaxis.set_major_formatter(formatter)
axs[1,0].yaxis.set_major_formatter(formatter)
axs[1,1].yaxis.set_major_formatter(formatter)

# Date formatting
locator = mdates.MonthLocator(interval=3)  # every 3 months
formatter = mdates.DateFormatter('%b %Y')

for ax in axs.flat:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Date")

for ax in axs.flat:

    # Remove margins
    ax.margins(x=0)
    ax.set_xlim(dates.min(), dates.max())

    # Shaded variant periods
    for variant, (start, end) in variant_periods.items():
        ax.axvspan(start, end, color=variant_colors[variant], alpha=0.15)

    # Vertical dashed lines
    boundaries = [v[0] for v in variant_periods.values()][1:]
    for b in boundaries:
        ax.axvline(b, color='black', linestyle='--', linewidth=1)

    # Variant labels
    for variant, (start, end) in variant_periods.items():
        mid = start + (end - start)/2
        ax.text(mid, 0.90, variant,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=10, fontweight='bold', alpha=0.85)
    
plt.tight_layout(rect=[0,0,1,0.97])
# plt.savefig(r"F:\Users\Xristos\Downloads\Covid-Greece.pdf", dpi=400)
plt.show()