#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:23:01 2025

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path("Data/")
file_to_open = data_folder / "Pairing_in_Square_L_x_70_L_y_70_B_0.4_theta_1.571_phi_0_mu_-40_Lambda_0.56.npz"
Data = np.load(file_to_open)

s_wave_pairing = Data["s_wave_pairing"]
p_wave_pairing_horizontal = Data["p_wave_pairing_horizontal"]
p_wave_pairing_vertical = Data["p_wave_pairing_vertical"]
L_x = Data["L_x"]
L_y = Data["L_y"]

pairing_s_wave_amplitude = np.abs(s_wave_pairing)

fig, ax = plt.subplots()
image = ax.imshow(pairing_s_wave_amplitude.T, origin="lower",
                  cmap="coolwarm")
fig.colorbar(image)

#%% Plot of p-wave pairing density
from matplotlib import colors
import matplotlib as mpl

# Lattice parameters
spacing = 1 # Spacing between lattice points

# Create a colormap
cmap = mpl.cm.coolwarm
norm = colors.Normalize(min(np.min(np.abs(p_wave_pairing_horizontal)),
                               np.min(np.abs(p_wave_pairing_vertical))),
                        max(np.max(np.abs(p_wave_pairing_horizontal)),
                        np.max(np.abs(p_wave_pairing_vertical))))

norm = colors.Normalize(min(np.min(np.abs(p_wave_pairing_horizontal)),
                               np.min(np.abs(p_wave_pairing_vertical))),
                        max(np.max(np.abs(p_wave_pairing_horizontal)),
                        np.max(np.abs(p_wave_pairing_vertical))))
# Create a ScalarMappable
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

# Generate lattice points
x = np.arange(0, L_x * spacing, spacing)
y = np.arange(0, L_y * spacing, spacing)
X, Y = np.meshgrid(x, y)

# Plot lattice points
fig, ax = plt.subplots()

#Plot bonds (horizontal and vertical)
for i in range(L_y - 1):  
    for j in range(L_x):
        Z = np.abs(p_wave_pairing_vertical)[j, i]
        ax.plot([X[i,j], X[i+1,j]], [Y[i,j], Y[i+1,j]],
                       color=sm.to_rgba(Z),
                       linewidth=2)
for i in range(L_y):
    for j in range(L_x - 1):
        Z = np.abs(p_wave_pairing_horizontal)[j, i]
        ax.plot([X[i,j], X[i,j+1]], [Y[i,j], Y[i,j+1]],
                color=sm.to_rgba(Z),
                linewidth=2)

fig.colorbar(sm, ax=ax)
# Set aspect ratio and display
ax.axis('equal')

ax.set_title("p-wave pairing")
plt.show()