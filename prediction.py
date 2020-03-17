import numpy as np
import load_data as ld
import math
import matplotlib.pyplot as plt

# motion model with Gaussion noise
def motion_model(px, u):
    u = u + np.random.normal(np.array([0, 0, 0]).T, np.array([1, 1, 0.2]).T, (3, 1))
    px += u
    return px

# predict
def prediction(particle, u):
    shape = np.shape(particle)
    Np = shape[1]
    for j in range(Np):
        particle[:, j:j + 1] = motion_model(particle[:, j:j + 1], u)
    return particle
