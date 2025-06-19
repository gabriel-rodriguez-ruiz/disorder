#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:36:59 2025

@author: gabriel
"""

import numpy as np
from disordered_superconductivity import ZKMBDisorderedSuperconductor, ZKMBDisorderedSuperconductorPeriodicInY
from pairing import Pairing
from pathlib import Path

L_x = 30
L_y = 30
t = 10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -38
mu_0 = 0  #mu/10

# U = eigenvectors.T.conj()      # eigenvectors in rows
# D = U @ S.matrix @ np.linalg.inv(U)
# eigenvalues, eigenvectors = np.linalg.eigh(S.matrix)

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda}

system_params = {"L_x": L_x, "L_y": L_y, "theta": theta,
                 "phi": phi, "mu_0": mu_0
                 }

S = ZKMBDisorderedSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
                              B_x, B_y, B_z, mu_0)
# S = ZKMBDisorderedSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                               B_x, B_y, B_z, mu_0)

Delta = Pairing(S)

s_wave_pairing = Delta.get_local_s_wave_pairing()
non_local_pairing_horizontal, non_local_pairing_vertical = Delta.get_non_local_pairing()


#%% Plot of local s-wave pairing density
import matplotlib.pyplot as plt

pairing_s_wave_amplitude = np.abs(s_wave_pairing)

fig, ax = plt.subplots()
image = ax.imshow(pairing_s_wave_amplitude.T, origin="lower",
                  cmap="coolwarm")
ax.set_title("Local s-wave pairing")
fig.colorbar(image)

#%% Plot of p-wave pairing density
from matplotlib import colors
import matplotlib as mpl

fig, axs = plt.subplots(2, 2)

for n in range(2):
    for m in range(2): 
    
        # Lattice parameters
        spacing = 1 # Spacing between lattice points
        
        # Create a colormap
        cmap = mpl.cm.coolwarm
        norm = colors.Normalize(min(np.min(np.abs(non_local_pairing_horizontal[:, :, n, m])),
                                       np.min(np.abs(non_local_pairing_vertical[:, :, n, m]))),
                                max(np.max(np.abs(non_local_pairing_horizontal[:, :, n, m])),
                                np.max(np.abs(non_local_pairing_vertical[:, :, n, m]))))
        # Create a ScalarMappable
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        # Generate lattice points
        x = np.arange(0, Delta.L_x * spacing, spacing)
        y = np.arange(0, Delta.L_y * spacing, spacing)
        X, Y = np.meshgrid(x, y)
        
        # Plot bonds (horizontal and vertical)
        
        # for i in range(Delta.L_y - 1):  
        #     for j in range(Delta.L_x):
        #         Z = np.abs(non_local_pairing_vertical)[j, i, n, m]
        #         axs[n, m].plot([X[i,j], X[i+1,j]], [Y[i,j], Y[i+1,j]],
        #                        color=sm.to_rgba(Z),
        #                        linewidth=2)
        for i in range(Delta.L_y):
            for j in range(Delta.L_x - 1):
                Z = np.abs(non_local_pairing_horizontal)[j, i, n, m]
                axs[n, m].plot([X[i,j], X[i,j+1]], [Y[i,j], Y[i,j+1]],
                        color=sm.to_rgba(Z),
                        linewidth=2)
        axs[n, m].set_xlabel("x")
        axs[n, m].set_ylabel("y")
        axs[n, m].set_aspect('equal')
        fig.colorbar(sm, ax=axs[n, m])

fig.suptitle("Non local pairing "
             + r"$B$" + f"={B}; "
          + r" $\lambda=$"
          + f"{Lambda};"
          + r" $\theta=$"
          + f"{np.round(theta, 3)};"
          + r" $\varphi=$"
          + f"{np.round(phi, 3)}"
          + r"; $\mu$"
          + f"={mu}")

axs[0, 0].set_title(r"$|\langle c_{\uparrow} c_{\uparrow}\rangle|$")
axs[0, 1].set_title(r"$|\langle c_{\uparrow} c_{\downarrow}\rangle|$")
axs[1, 0].set_title(r"$|\langle c_{\downarrow} c_{\uparrow}\rangle|$")
axs[1, 1].set_title(r"$|\langle c_{\downarrow} c_{\downarrow}\rangle|$")

plt.tight_layout()
plt.show()

#%%
data_folder = Path("Data/")
name = f"Pairing_in_Square_L_x_{L_x}_L_y_{L_y}_B_{B}_theta_{np.round(theta, 3)}_phi_{phi}_mu_{mu}_Lambda_{Lambda}"
file = data_folder / name
np.savez(file, s_wave_pairing=s_wave_pairing,
         non_local_pairing_horizontal=non_local_pairing_horizontal,
         non_local_pairing_vertical=non_local_pairing_vertical,
         **superconductor_params, **system_params)