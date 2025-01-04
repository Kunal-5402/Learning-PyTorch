import numpy as np
import plotly.graph_objects as go

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

def gradient_f(x, y):
    norm = np.sqrt(x**2 + y**2)
    
    if norm == 0: 
        return np.array([0, 0])
    df_dx = (x * np.cos(norm)) / norm
    df_dy = (y * np.cos(norm)) / norm
    return np.array([df_dx, df_dy])

EPOCHS = 100
BETA_1 = 0.9
BETA_2 = 0.999
ALPHA = 0.01

x = np.random.uniform(-5, 5)
y = np.random.uniform(-5, 5)
print(f"Initial X: {x}, Initial Y: {y}")

M_t = np.zeros(2)
V_t = np.zeros(2)
path = [(x, y, f(x, y))]  


for i in range(EPOCHS):
    gradient = gradient_f(x, y)

    M_t = BETA_1 * M_t + (1 - BETA_1) * gradient
    V_t = BETA_2 * V_t + (1 - BETA_2) * gradient**2

    corrected_M_t = M_t / (1 - BETA_1**(i + 1))
    corrected_V_t = V_t / (1 - BETA_2**(i + 1))

    x = x - ALPHA * corrected_M_t[0] / (np.sqrt(corrected_V_t[0]) + 1e-8)
    y = y - ALPHA * corrected_M_t[1] / (np.sqrt(corrected_V_t[1]) + 1e-8)

    print(f"Epoch: {i+1}, X: {x}, Y: {y}")

    path.append((x, y, f(x, y)))

