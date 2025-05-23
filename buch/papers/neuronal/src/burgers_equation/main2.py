import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc

tf.keras.backend.set_floatx('float64')
matplotlib.use('TkAgg')

### Data Generation

# Number of initial and boundary condition data points
N_ic, N_bc1, N_bc2 = 50, 25, 25

# Latin Hypercube Sampling (LHS)
lhs = qmc.LatinHypercube(d=1)

# Time coordinates
t_d = lhs.random(n=N_bc1 + N_bc2)
t_d = np.append(np.zeros((N_ic, 1)), t_d, axis=0)

# Space coordinates
x_d = lhs.random(n=N_ic)
x_d = 2 * (x_d - 0.5)  # scale from [0, 1] to [-1, 1]
x_d = np.append(x_d, -1 * np.ones((N_bc1, 1)), axis=0)
x_d = np.append(x_d, +1 * np.ones((N_bc2, 1)), axis=0)

# Visualize initial/boundary points
plt.figure(figsize=(6, 3))
plt.scatter(t_d, x_d, marker="x", c="k")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Initial and Boundary Data Points")
plt.tight_layout()
plt.show()

# Data values
y_d = np.zeros_like(x_d)
y_d[:N_ic] = -np.sin(np.pi * x_d[:N_ic])

# Collocation points
N_c = 10000
lhs_2d = qmc.LatinHypercube(d=2)
points = lhs_2d.random(n=N_c)
points[:, 1] = 2 * (points[:, 1] - 0.5)

t_c = points[:, [0]]
x_c = points[:, [1]]

# Convert to tensors
x_d, t_d, y_d, x_c, t_c = map(tf.convert_to_tensor, [x_d, t_d, y_d, x_c, t_c])

### Neural Network Model

neurons = 20
hidden_layers = 8

inputs = tf.keras.layers.Input(shape=(2,))
x = inputs
for _ in range(hidden_layers):
    x = tf.keras.layers.Dense(neurons, activation="tanh")(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

### PDE Solution u(x, t)
@tf.function
def u(t, x):
    return model(tf.concat([t, x], axis=1))

### Physics-Informed Loss
@tf.function
def F(t, x):
    u0 = u(t, x)
    u_t = tf.gradients(u0, t)[0]
    u_x = tf.gradients(u0, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    return tf.reduce_mean(tf.square(u_t + u0 * u_x - (0.01 / np.pi) * u_xx))

@tf.function
def mse(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

### Training Loop

epochs = 15000
opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
loss_history = []

if os.path.exists("burgers_net.h5"):
    model = tf.keras.models.load_model('burgers_net.h5')
    print("Loaded trained model from file")
else:
    print("Training model...")

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = u(t_d, x_d)
            loss = F(t_c, x_c) + mse(y_d, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        loss_history.append(loss)

    model.save('burgers_net.h5')

# Plot training loss
plt.figure(figsize=(6, 3))
plt.semilogy(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.show()

def initial_condition_loss(x_init):
    t0 = np.zeros_like(x_init)
    X0, T0 = np.meshgrid(x_init, t0)
    X_flat = tf.convert_to_tensor(X0.reshape(-1, 1))
    T_flat = tf.convert_to_tensor(T0.reshape(-1, 1))

    u0_pred = u(T_flat, X_flat)

    u0_true = tf.sin(-np.pi * X_flat)
    return tf.reduce_mean(tf.square(u0_pred + u0_true))

def boundary_condition_loss(t_bc):
    x_left = np.zeros_like(t_bc) - 1
    x_right = np.zeros_like(t_bc) + 1

    X_left, t_bcl = np.meshgrid(x_left, t_bc)
    X_left_flat = tf.convert_to_tensor(X_left.reshape(-1, 1))
    T_left_flat = tf.convert_to_tensor(t_bcl.reshape(-1, 1))

    X_right, t_bcr = np.meshgrid(x_right, t_bc)
    X_right_flat = tf.convert_to_tensor(X_right.reshape(-1, 1))
    T_right_flat = tf.convert_to_tensor(t_bcr.reshape(-1, 1))

    u_pred_left = u(T_left_flat, X_left_flat)
    u_pred_right = u(T_right_flat, X_right_flat)

    return tf.reduce_mean(tf.square(0.5 * u_pred_left + 0.5 * u_pred_right))


### Solution Plot

n, m = 100, 200
X = np.linspace(-1, +1, m)
T = np.linspace(0, 1, n)
X0, T0 = np.meshgrid(X, T)

X_flat = tf.convert_to_tensor(X0.reshape(-1, 1))
T_flat = tf.convert_to_tensor(T0.reshape(-1, 1))

U = u(T_flat, X_flat).numpy().reshape(n, m)

pde_loss = F(T_flat, X_flat)
ic_loss = initial_condition_loss(X)
bc_loss = boundary_condition_loss(T)
total_loss = (m*n - m - 2*n)/(m*n) * pde_loss + m/(m*n) * ic_loss + 2*n/(m*n) * bc_loss
print(total_loss)

plt.figure(figsize=(8, 4), dpi=150)
plt.pcolormesh(T0, X0, U, cmap='coolwarm', shading='auto')
plt.colorbar(label="u(x, t)")
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("t")
plt.ylabel("x")
plt.title(f"1D Burgers-Gleichung - Mittlerer Fehler: {total_loss}")
plt.tight_layout()
plt.show()
