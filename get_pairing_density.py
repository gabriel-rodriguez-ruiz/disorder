#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:36:59 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductor
import random
from disorder import ZKMBDisorderedSuperconductor, get_components
from pairing import Pairing

L_x = 30
L_y = 30
t = 10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = np.pi/4
B = 2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40
mu_0 = 0#mu/10#mu/2


# U = eigenvectors.T.conj()      # eigenvectors in rows
# D = U @ S.matrix @ np.linalg.inv(U)

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda}

system_params = {"L_x": L_x, "L_y": L_y, "theta": theta,
                 "phi": phi, "mu_0": mu_0
                 }

S = ZKMBDisorderedSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
                              B_x, B_y, B_z, mu_0)
Delta = Pairing(S)

s_wave_pairing = Delta.get_s_wave_pairing()

p_wave_pairing_horizontal, p_wave_pairing_vertical = Delta.get_p_wave_pairing()

#%% Plot of s-wave pairing density
import matplotlib.pyplot as plt

pairing_s_wave_amplitude = np.abs(s_wave_pairing)

fig, ax = plt.subplots()
image = ax.imshow(pairing_s_wave_amplitude,
                  origin="lower",
                  cmap="coolwarm")
fig.colorbar(image)

#%% Plot of p-wave pairing density
from matplotlib import colors
import matplotlib as mpl

# Lattice parameters
size = Delta.L_x   # Size of the lattice (size x size)
spacing = 1 # Spacing between lattice points

# Create a colormap
cmap = mpl.cm.coolwarm
norm = colors.Normalize(min(np.min(np.abs(p_wave_pairing_horizontal)),
                               np.min(np.abs(p_wave_pairing_vertical))),
                        max(np.max(np.abs(p_wave_pairing_horizontal)),
                        np.max(np.abs(p_wave_pairing_vertical))))
# Create a ScalarMappable
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

# Generate lattice points
x = np.arange(0, size * spacing, spacing)
y = np.arange(0, size * spacing, spacing)
X, Y = np.meshgrid(x, y)

# Plot lattice points
fig, ax = plt.subplots()

# Plot bonds (horizontal and vertical)
for i in range(size):
    for j in range(size):
        if i < size - 1: # Vertical bonds
            Z = np.abs(p_wave_pairing_vertical.T[i,j])
            line = ax.plot([X[i,j], X[i+1,j]], [Y[i,j], Y[i+1,j]],
                           color=sm.to_rgba(Z),
                           linewidth=2)
        if j < size - 1: # Horizontal bonds
            Z = np.abs(p_wave_pairing_horizontal.T[i,j])
            ax.plot([X[i,j], X[i,j+1]], [Y[i,j], Y[i,j+1]],
                     color=sm.to_rgba(Z),
                     linewidth=2)

fig.colorbar(sm, ax=ax)
# Set aspect ratio and display
ax.axis('equal')

ax.set_title("p-wave pairing")
plt.show()