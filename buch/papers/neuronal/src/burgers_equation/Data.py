from scipy.stats import qmc
import torch
import numpy as np

lhs = qmc.LatinHypercube(d=1)
lhs_2d = qmc.LatinHypercube(d=2)
N_ic, N_bc1, N_bc2 = 2000, 1000, 1000 #50, 25, 25
N_c = 5000 #10000

def get_train(device):
    # Initial & boundary data
    t_d = lhs.random(n=N_bc1 + N_bc2)  # 50 t-Werte --> t_bc
    t_d = np.append(np.zeros((N_ic, 1)), t_d, axis=0)  # 50 mal 0 + vorherige 50 t-Werte

    x_d = lhs.random(n=N_ic)  # 50 x-Werte
    x_d = 2 * (x_d - 0.5)  # Intervall von x-Werten auf [-1,1] anpassen --> x_init
    x_d = np.append(x_d, -1 * np.ones((N_bc1, 1)), axis=0)  # vorherige 50 x-Werte + 25 mal -1
    x_d = np.append(x_d, +1 * np.ones((N_bc2, 1)), axis=0)  # vorherige 75 x-Werte + 25 mal 1

    u_d = np.zeros_like(x_d)  # 100 mal 0
    u_d[:N_ic] = -np.sin(np.pi * x_d[:N_ic])
    # zu den ersten 50 Werten gehören die ersten 50 aus x_d und t_d, zu den zweiten 50 die zweiten 50 aus x_d und t_d
    # erste 50: u von Anfangsbedingung, x in [-1,1], t = 0
    # zweite 50: u = 0, x = -1 oder x = 1, t in [0,1]

    # Main data for burgers equation residual
    points = lhs_2d.random(n=N_c)
    points[:, 1] = 2 * (points[:, 1] - 0.5)

    t_c = points[:, [0]]  # [0,1) --> 10000 Punkte
    x_c = points[:, [1]]  # [-1,1) --> 10000 Punkte

    x_d = torch.tensor(x_d, device=device)
    t_d = torch.tensor(t_d, device=device)
    u_d = torch.tensor(u_d, device=device)
    x_c = torch.tensor(x_c, device=device)
    t_c = torch.tensor(t_c, device=device)

    return x_d, t_d, x_c, t_c, u_d

def get_test(device):
    # Gleich wie get_train, aber nur 20% der Datensatz-Grössen
    # Initial & boundary data
    t_d = lhs.random(n=int(0.2 * (N_bc1 + N_bc2)))  # 50 t-Werte --> t_bc
    t_d = np.append(np.zeros((int(0.2 * N_ic), 1)), t_d, axis=0)  # 50 mal 0 + vorherige 50 t-Werte

    x_d = lhs.random(n=int(0.2 * N_ic))  # 50 x-Werte
    x_d = 2 * (x_d - 0.5)  # Intervall von x-Werten auf [-1,1] anpassen --> x_init
    x_d = np.append(x_d, -1 * np.ones((int(0.2 * N_bc1), 1)), axis=0)  # vorherige 50 x-Werte + 25 mal -1
    x_d = np.append(x_d, +1 * np.ones((int(0.2 * N_bc2), 1)), axis=0)  # vorherige 75 x-Werte + 25 mal 1

    u_d = np.zeros_like(x_d)  # 100 mal 0
    u_d[:int(0.2 * N_ic)] = -np.sin(np.pi * x_d[:int(0.2 * N_ic)])
    # zu den ersten 50 Werten gehören die ersten 50 aus x_d und t_d, zu den zweiten 50 die zweiten 50 aus x_d und t_d
    # erste 50: u von Anfangsbedingung, x in [-1,1], t = 0
    # zweite 50: u = 0, x = -1 oder x = 1, t in [0,1]

    # Main data for burgers equation residual
    points = lhs_2d.random(n=int(0.2 * N_c))
    points[:, 1] = 2 * (points[:, 1] - 0.5)

    t_c = points[:, [0]]  # [0,1) --> 10000 Punkte
    x_c = points[:, [1]]  # [-1,1) --> 10000 Punkte

    x_d = torch.tensor(x_d, device=device)
    t_d = torch.tensor(t_d, device=device)
    u_d = torch.tensor(u_d, device=device)
    x_c = torch.tensor(x_c, device=device)
    t_c = torch.tensor(t_c, device=device)

    return x_d, t_d, x_c, t_c, u_d