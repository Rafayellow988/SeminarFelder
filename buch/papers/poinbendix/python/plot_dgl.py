import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, z):
    x, y = z
    dxdt = y
    dydt = -x + 0.1*(1 - x**2)*y  # Van der Pol example
    return [dxdt, dydt]


alpha = 1 / 8
epsilon = 0.1
b = 2
gamma = 3 / 4
r = 1 / 4
def recharge_oscillator(t, z):
    x, y = z
    dxdt = -x + gamma*(b*x + y) - epsilon*(b*x + y)**3
    dydt = -r*y - alpha* b * x
    return [dxdt, dydt]

# Time span and initial conditions
t_span = [0, 10000]
t_eval = np.linspace(*t_span, 100000)
# initial_conditions = [[0, 0], [0.2, 0.7], [1, -1], [-2, 0]]
initial_conditions = [[1, -1]]

# Plotting
for z0 in initial_conditions:
    sol = solve_ivp(recharge_oscillator, t_span, z0, t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Portrait')
plt.grid()
plt.show()

