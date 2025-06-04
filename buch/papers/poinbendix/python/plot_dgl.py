import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from enable_export import enable_export

enable_export()

# def system(t, z):
#     x, y = z
#     dxdt = y
#     dydt = -x + 0.1*(1 - x**2)*y  # Van der Pol example
#     return [dxdt, dydt]


alpha = 1 / 8
epsilon = 0.1
b = 2
gamma = 3 / 4
r = 1 / 4


class RechargeOscillator():
    def __init__(self, alpha, epsilon, b, gamma,  r):
        self.alpha = alpha
        self.epsilon = epsilon
        self.b = b
        self.gamma = gamma
        self.r = r
    def run(self, t, z):
        x, y = z
        dxdt = -x + self.gamma*(self.b*x + y) - self.epsilon*(self.b*x + y)**3
        dydt = -self.r*y - self.alpha* self.b * x
        return [dxdt, dydt]

# Time span and initial conditions
ro = RechargeOscillator(alpha, epsilon, b, gamma, r)
t_span = [0, 1000]
t_eval = np.linspace(*t_span, 10000)
# initial_conditions = [[0, 0], [0.2, 0.7], [1, -1], [-2, 0]]
initial_conditions = [[1, -1], [1, 2], [-0.25, 2], [-0.75, 1], [-1, -1]]

# Plotting
fig, ax = plt.subplots()
for z0 in initial_conditions:
    sol = solve_ivp(ro.run, t_span, z0, t_eval=t_eval)
    ax.plot(sol.y[0], sol.y[1])

plt.grid()
fig.set_size_inches(w=4.5, h=2.5)
fig.savefig("../images/recharge_oscillator.pgf")

# Experiment 2: wiggle on parameter alpha
z0 = [1, -1]
alphas = [alpha, alpha + 0.01, alpha - 0.01]
fig, ax = plt.subplots()
for alph in alphas:
    ro = RechargeOscillator(alph, epsilon, b, gamma, r)
    sol = solve_ivp(ro.run, t_span, z0, t_eval=t_eval)
    ax.plot(sol.y[0], sol.y[1], label=f"$\\alpha={alph:.3f}$")
plt.grid()
plt.legend()
fig.set_size_inches(w=4.5, h=2.5)
fig.savefig("../images/recharge_oscillator_alpha.pgf")

# Experiment 3: wiggle on parameter gamma
z0 = [1, -1]
gammas = [gamma, gamma + 0.02, gamma - 0.02]
fig, ax = plt.subplots()
for gamm in gammas:
    ro = RechargeOscillator(alpha, epsilon, b, gamm, r)
    sol = solve_ivp(ro.run, t_span, z0, t_eval=t_eval)
    ax.plot(sol.y[0], sol.y[1], label=f"$\\gamma={gamm:.3f}$")
plt.grid()
plt.legend()
fig.set_size_inches(w=4.5, h=2.5)
fig.savefig("../images/recharge_oscillator_gamma.pgf")
