import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc
from BurgersNet import BurgersNet
from Graphics import error_plot, solution_plot

matplotlib.use('TkAgg')
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data Generation

# Initial & boundary data
N_ic, N_bc1, N_bc2 = 50, 25, 25
lhs = qmc.LatinHypercube(d=1)

t_d = lhs.random(n=N_bc1 + N_bc2) # 50 t-Werte --> t_bc
t_d = np.append(np.zeros((N_ic, 1)), t_d, axis=0) # 50 mal 0 + vorherige 50 t-Werte

x_d = lhs.random(n=N_ic) #50 x-Werte
x_d = 2 * (x_d - 0.5) #Intervall von x-Werten auf [-1,1] anpassen --> x_init
x_d = np.append(x_d, -1 * np.ones((N_bc1, 1)), axis=0) #vorherige 50 x-Werte + 25 mal -1
x_d = np.append(x_d, +1 * np.ones((N_bc2, 1)), axis=0) #vorherige 75 x-Werte + 25 mal 1

u_d = np.zeros_like(x_d) #100 mal 0
u_d[:N_ic] = -np.sin(np.pi * x_d[:N_ic])
# zu den ersten 50 Werten gehören die ersten 50 aus x_d und t_d, zu den zweiten 50 die zweiten 50 aus x_d und t_d
# erste 50: u von Anfangsbedingung, x in [-1,1], t = 0
# zweite 50: u = 0, x = -1 oder x = 1, t in [0,1]

# Main data for burgers equation residual
N_c = 10000
lhs_2d = qmc.LatinHypercube(d=2)
points = lhs_2d.random(n=N_c)
points[:, 1] = 2 * (points[:, 1] - 0.5)

t_c = points[:, [0]] # [0,1) --> 10000 Punkte
x_c = points[:, [1]] #[-1,1) --> 10000 Punkte

x_d = torch.tensor(x_d, device=device)
t_d = torch.tensor(t_d, device=device)
u_d = torch.tensor(u_d, device=device)
x_c = torch.tensor(x_c, device=device)
t_c = torch.tensor(t_c, device=device)

### Model
model = BurgersNet().to(device)

### Training
if os.path.exists("burgers_net.pth"):
    model.load_state_dict(torch.load("burgers_net.pth"))
    print("Loaded trained model from file")
else:
    print("Training model...")
    loss_history = model.fit(x_d, t_d, x_c, t_c, u_d, epochs=15000)
    torch.save(model.state_dict(), "burgers_net.pth")

    error_plot(loss_history)

print("Model parameters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad))+"\n")

### Solution Plot
solution_plot(model, device)
