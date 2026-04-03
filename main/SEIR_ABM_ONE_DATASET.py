# -*- coding: utf-8 -*-
"""
Created on Thu Nov 6 09:48:26 2025

@author: Konstanto
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import os
import re
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

csv_path = r'...SEIR homogeneous\Datasets\seir_beta0p18_sigma0p45_gamma0p2.csv' ## change for a different dataset
out_dir = os.path.dirname(os.path.abspath(csv_path))
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# Training hyperparams
adam_epochs = 25000
adam_lr = 0.001
lbfgs_max_iter = 1500
lbfgs_lr = 0.1
n_collocation = 70
hidden=32
nlayers=4

# Loss/penalty weights
w_init = 1.0        # weight for exact initial condition enforcement
w_cons = 1.0        # weight for population conservation
w_data = 1.0        # data loss weight
w_phys = 1.0        # physics loss weight

# ---------- Load data ----------
df = pd.read_csv(csv_path)
t = df['Step'].values.astype(float).reshape(-1,1)
S = df['Susceptible'].values.astype(float).reshape(-1,1)
E = df['Exposed'].values.astype(float).reshape(-1,1)
I = df['Infected'].values.astype(float).reshape(-1,1)
R = df['Removed'].values.astype(float).reshape(-1,1)

##---------- Automatically save β,γ,σ -----------##
filename = os.path.basename(csv_path)

pattern = r'beta(\d+p\d+)_sigma(\d+p\d+)_gamma(\d+p\d+)'
match = re.search(pattern, filename)

# True values for plotting
if match:
    beta_true  = 5*float(match.group(1).replace('p', '.'))
    sigma_true = float(match.group(2).replace('p', '.'))
    gamma_true = float(match.group(3).replace('p', '.'))
else:
    raise ValueError("Parameters not found in filename")

t_min, t_max = float(t.min()), float(t.max())
N_pop = float((S+E+I+R)[0].sum()) # total population

# Normalize fractions
S_frac = S / N_pop
E_frac = E / N_pop
I_frac = I / N_pop
R_frac = R / N_pop

t_train = torch.tensor(t, dtype=torch.float32, device=device)
S_train = torch.tensor(S_frac, dtype=torch.float32, device=device)
E_train = torch.tensor(E_frac, dtype=torch.float32, device=device)
I_train = torch.tensor(I_frac, dtype=torch.float32, device=device)
R_train = torch.tensor(R_frac, dtype=torch.float32, device=device)

# Initial values
S0_val = float(S[0,0]/ N_pop)  
E0_val = float(E[0,0]/ N_pop)  
I0_val = float(I[0,0]/ N_pop)   
R0_val = float(R[0,0]/ N_pop)   

# ---------- NN Model ----------
class PINN(nn.Module):
    def __init__(self, hidden, nlayers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden)) # First layer - time
        layers.append(nn.Softplus())
        for i in range(nlayers-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Softplus())   
        layers.append(nn.Linear(hidden, 4))  # 4 layers for outputs S,E,I,R
        self.net = nn.Sequential(*layers) # all layers to a sequential model
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # Initializes weights using Xavier initialization (helps stable gradients)
                nn.init.zeros_(m.bias) # Biases set to zero
                
            
    def forward(self, t): # defines how data passes through the network
        return self.net(t)
    

model = PINN(hidden=hidden, nlayers=nlayers).to(device)

# trainable parameters to be optimized
raw_beta = nn.Parameter(torch.tensor(np.log(np.exp(0.4)-1), dtype=torch.float32, device=device))
raw_sigma = nn.Parameter(torch.tensor(np.log(np.exp(0.1)-1), dtype=torch.float32, device=device))
raw_gamma = nn.Parameter(torch.tensor(np.log(np.exp(0.2)-1), dtype=torch.float32, device=device))

# ---------- Physics residuals ----------
def sample_collocation(n):
    ts = np.random.randint(t_min, t_max, size=(n,1)) # n uniform collocation points
    return torch.tensor(ts, dtype=torch.float32, device=device)


def physics_residuals(model, t_coll):
    t_coll = t_coll.clone().detach().requires_grad_(True)
    out = model(t_coll)
    S_p = out[:,0:1]; E_p = out[:,1:2]; I_p = out[:,2:3]; R_p = out[:,3:4]
    
    dS_dt = torch.autograd.grad(S_p, t_coll, torch.ones_like(S_p), create_graph=True)[0] # create_graph=True ensures we can differentiate again
    dE_dt = torch.autograd.grad(E_p, t_coll, torch.ones_like(E_p), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_p, t_coll, torch.ones_like(I_p), create_graph=True)[0]
    dR_dt = torch.autograd.grad(R_p, t_coll, torch.ones_like(R_p), create_graph=True)[0]

    beta = torch.nn.functional.softplus(raw_beta)
    sigma = torch.nn.functional.softplus(raw_sigma)
    gamma = torch.nn.functional.softplus(raw_gamma)

    # Using fraction variables so no N factor
    res_S = dS_dt + beta * S_p * I_p 
    res_E = dE_dt - beta * S_p * I_p  + sigma * E_p
    res_I = dI_dt - sigma * E_p + gamma * I_p 
    res_R = dR_dt - gamma * I_p

    return res_S, res_E, res_I, res_R, beta, sigma, gamma

# ---------- Loss function ----------
mse = nn.MSELoss()
def loss_total(model, t_data, S_d, E_d, I_d, R_d, n_collocation, w_data, w_phys):
    preds = model(t_data)
    S_p, E_p, I_p, R_p = preds[:,0:1], preds[:,1:2], preds[:,2:3], preds[:,3:4]
    loss_data = mse(S_p, S_d) + mse(E_p, E_d) + mse(I_p, I_d) + mse(R_p, R_d) # MSE between predicted and true data
    t_col = sample_collocation(n_collocation)
    res_S, res_E, res_I, res_R, beta, sigma, gamma = physics_residuals(model, t_col)
    loss_phys = mse(res_S, torch.zeros_like(res_S)) + mse(res_E, torch.zeros_like(res_E)) + mse(res_I, torch.zeros_like(res_I)) + mse(res_R, torch.zeros_like(res_R))
    # initial conditions
    S0 = torch.tensor([[S0_val]], dtype=torch.float32, device=device)
    E0 = torch.tensor([[E0_val]], dtype=torch.float32,device=device)
    I0 = torch.tensor([[I0_val]], dtype=torch.float32,device=device)
    R0 = torch.tensor([[R0_val]], dtype=torch.float32,device=device)
    t0 = torch.tensor([[t_min]], dtype=torch.float32, device=device)
    pred0 = model(t0)
    loss_init = mse(pred0, torch.cat([S0, E0, I0, R0], dim=1))

    total_loss = w_data*loss_data + w_phys*loss_phys + w_init*loss_init 
    return total_loss, loss_data.detach(), loss_phys.detach(), loss_init, beta.detach(), sigma.detach(), gamma.detach()

# ---------- Training config (adjustable) ----------
print_every = 200
params = list(model.parameters()) + [raw_beta, raw_sigma, raw_gamma]
opt = torch.optim.Adam(params, lr=adam_lr)

loss_hist, ldata_hist, lphys_hist = [], [], []
beta_hist, sigma_hist, gamma_hist = [], [], []

# Adam training
print("Starting Adam training...")
start_time = time.time()
for epoch in range(1, adam_epochs+1):
    opt.zero_grad() # clears accumulated gradients from previous iterations
    loss, ldata, lphys, linit, b, s, g = loss_total(
        model, 
        t_train, S_train, E_train, I_train, R_train, 
        n_collocation, w_data, w_phys)
    loss.backward() # backpropagation: computes gradients of loss w.r.t. all trainable parameters (network weights and β,σ,γ)
    opt.step()
    
    # ---- store losses ----
    loss_hist.append(loss.item())
    ldata_hist.append(ldata.item())
    lphys_hist.append(lphys.item())
    
    beta_hist.append(b.item())
    sigma_hist.append(s.item())
    gamma_hist.append(g.item())
    
    if epoch % print_every == 0 or epoch == 1:
        print(f"Adam epoch {epoch}/{adam_epochs} "
              f"Loss={loss.item():.3e} "
              f"Data={ldata.item():.3e} "
              f"Phys={lphys.item():.3e} "
              f"beta={b.item():.4f} "
              f"sigma={s.item():.4f} "
              f"gamma={g.item():.4f} "
              )

# Refine with LBFGS (quasi-Newton optimizer)
use_lbfgs = True
if use_lbfgs:
    optimizer_lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, lr=lbfgs_lr)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, ldata, lphys, linit, b, s, g = loss_total(model, t_train, S_train, E_train, I_train, R_train, n_collocation, w_data, w_phys)
        loss.backward()
        return loss
    print("Starting L-BFGS refine...")
    optimizer_lbfgs.step(closure)
    print("L-BFGS done.")
elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start_time))

# Final evaluation
loss, ldata, lphys, linit, b, s, g = loss_total(model, t_train, S_train, E_train, I_train, R_train, n_collocation, w_data, w_phys)
print("Final losses: Total={:.3e}, Data={:.3e}, Phys={:.3e}".format(loss.item(), ldata.item(), lphys.item()))
print("Learned parameters: beta={:.6f} sigma={:.6f}, gamma={:.6f}".format(b.item(), s.item(), g.item()))
print('Training time:',elapsed,'minutes.')

# Save predictions on fine grid
t_test = np.linspace(t_min, t_max, 120).reshape(-1,1)
t_test_t = torch.tensor(t_test, dtype=torch.float32, device=device)
with torch.no_grad():
    preds = model(t_test_t).cpu().numpy()
S_pred = preds[:,0].reshape(-1,1) * N_pop
E_pred = preds[:,1].reshape(-1,1) * N_pop
I_pred = preds[:,2].reshape(-1,1) * N_pop
R_pred = preds[:,3].reshape(-1,1) * N_pop

pred_df = pd.DataFrame(np.hstack([t_test, S_pred, E_pred, I_pred, R_pred]), columns=['t','S_pred','E_pred','I_pred','R_pred'])
pred_path = os.path.join(out_dir, "seir_pinn_predictions.csv")
pred_df.to_csv(pred_path, index=False)

params_df = pd.DataFrame([{'beta':float(b.item()), 'sigma':float(s.item()), 'gamma':float(g.item())}])
params_path = os.path.join(out_dir, "seir_pinn_learned_params.csv")
params_df.to_csv(params_path, index=False)

# save model state and raw params
state = {'beta': b.detach().cpu().numpy().tolist(),
         'sigma': s.detach().cpu().numpy().tolist(),
         'gamma': g.detach().cpu().numpy().tolist(),
         'N_pop': N_pop, 't_min': t_min, 't_max': t_max}
with open(os.path.join(out_dir, "seir_pinn_model_state.pkl"), 'wb') as f:
    pickle.dump(state, f)

print("Saved:", pred_path, params_path)

# plot S, E, I, R
try:
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1); plt.plot(t_test, S_pred, ls='-', lw='3', label='Predicted'); plt.plot(t, S, ls='--', lw='3', label='Actual'); plt.title("S"); plt.grid(True); legend=plt.legend()
    plt.subplot(2,2,2); plt.plot(t_test, E_pred, ls='-', lw='3', label='Predicted'); plt.plot(t, E, ls='--', lw='3', label='Actual'); plt.title("E"); plt.grid(True); legend=plt.legend()
    plt.subplot(2,2,3); plt.plot(t_test, I_pred, ls='-', lw='3', label='Predicted'); plt.plot(t, I, ls='--', lw='3', label='Actual'); plt.title("I"); plt.grid(True); legend=plt.legend() 
    plt.subplot(2,2,4); plt.plot(t_test, R_pred, ls='-', lw='3', label='Predicted'); plt.plot(t, R, ls='--', lw='3', label='Actual'); plt.title("R"); plt.grid(True); legend=plt.legend()
    # plt.savefig(r'F:\Users\Xristos\Downloads\Figures\Epidemic PINNs\SEIR\.eps', format='eps')
    plt.tight_layout(); plt.show();   
except Exception:
    pass

# plot losses
plt.figure(figsize=(8,6))
plt.subplot(3,1,1); plt.semilogy(loss_hist); plt.ylabel("Total loss"); plt.grid(True, which="both")
plt.subplot(3,1,2); plt.semilogy(ldata_hist); plt.ylabel("Data loss"); plt.grid(True, which="both")
plt.subplot(3,1,3); plt.semilogy(lphys_hist); plt.xlabel("Epoch"); plt.ylabel("Physics loss"); plt.grid(True, which="both")
plt.tight_layout()
plt.show()

# plot parameter convergence
plt.figure(figsize=(10,8))
plt.subplot(2,2,1); plt.plot(beta_hist, label="Learned β"); plt.axhline(beta_true, linestyle="--", color="r", label="true β"); plt.ylabel("β"); plt.grid(True); plt.legend()
plt.subplot(2,2,2); plt.plot(sigma_hist, label="Learned σ"); plt.axhline(sigma_true, linestyle="--", color="r", label="true σ"); plt.ylabel("β"); plt.grid(True); plt.legend()
plt.subplot(2,2,3); plt.plot(gamma_hist, label="Learned γ"); plt.axhline(gamma_true, linestyle="--", color="r", label="true γ"); plt.xlabel("Epoch"); plt.ylabel("γ"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

print("Relative errors:")
print(f"β error  = {abs(b.item()-beta_true)/beta_true:.2%}")
print(f"σ error  = {abs(s.item()-sigma_true)/sigma_true:.2%}")
print(f"γ error  = {abs(g.item()-gamma_true)/gamma_true:.2%}")
