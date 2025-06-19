#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:41:31 2025

@author: gabriel
"""

import numpy as np

class Pairing():
    """ A class representing the pairing field in 2D for a Superconductor S.
    """
    def __init__(self, S):
        self.S = S
        self.L_x = S.L_x
        self.L_y = S.L_y
    def _get_occupied_eigenstates(self):
        r"""Returns the positive energy eigenvectors as the columns of
        an array U^-1 of shape (4*L_x*L_y, N) where N is the number of positive
        energies such that
        
        .. math ::
            
            D = U H U^{-1} \\
            \Gamma = U c  \\
            c = (c_{1,1}, c_{1,2}, ..., c_{1,L_y}, c_{2,1}, ..., c_{L_x,L_y})^T
            
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.S.matrix) 
        positive_energy_eigenvectors = eigenvectors[:, np.where(eigenvalues>0)[0]]  #where returns a tuple
        return positive_energy_eigenvectors
    def _get_components(self, state):
        r"""
        Get the numeric components of the state: for a given
        column state (column of U^-1). Returns 4 arrays of shape (L_x, L_y)
        
        .. math ::
            
            \Gamma_\nu = \sum_\sigma \sum_{n,m} \left( A_{n,m\sigma}^\nu c^\dagger_{n,m\sigma} 
            + B_{n,m\sigma}^\nu  c_{n,m\sigma} \right)
            
        """
        A_up = -state[3::4].conj().reshape((self.L_x, self.L_y))
        A_down = state[2::4].conj().reshape((self.L_x, self.L_y))
        B_up = state[0::4].conj().reshape((self.L_x, self.L_y))
        B_down = state[1::4].conj().reshape((self.L_x, self.L_y))
        return (A_up,
                A_down,
                B_up,
                B_down)
    def get_local_pairing(self):
        r"""Returns a list [up_down, down_up] with an array whose elements are:
            
        .. math ::
            \Delta^{\text{s-wave}}_{n,m} = \langle c_{n,m\uparrow}c_{n,m\downarrow} \rangle
            = \sum_{E_\nu>0} (B_{n,m\uparrow}^\nu)^* A_{n,m\downarrow}^\nu \\
                where \\
            c_{n,m\sigma} = \sum_{E_\nu>0} A_{n,m\downarrow}^\nu \Gamma^\dagger_\nu + (B_{n,m\uparrow}^\nu)^* \Gamma_\nu  
               \\ and \\
            \Gamma_\nu = \sum_\sigma \sum_{n,m} \left( A_{n,m\sigma}^\nu c^\dagger_{n,m\sigma} 
            + B_{n,m\sigma}^\nu  c_{n,m\sigma} \right)

        """
        positive_energy_eigenvectors = self._get_occupied_eigenstates()
        local_pairing_up_down = np.zeros((self.S.L_x, self.S.L_y), dtype=complex)
        local_pairing_down_up = np.zeros((self.S.L_x, self.S.L_y), dtype=complex)
        for nu in range(np.shape(positive_energy_eigenvectors)[1]):
            A_up, A_down, B_up, B_down = \
                self._get_components(positive_energy_eigenvectors[:, nu])
            local_pairing_up_down_nu = B_up.conj() * A_down
            local_pairing_down_up_nu = B_down.conj() * A_up
            local_pairing_up_down += local_pairing_up_down_nu
            local_pairing_down_up += local_pairing_down_up_nu
        return [local_pairing_up_down, local_pairing_down_up]
    def get_non_local_pairing(self):
        r"""Returns 2 arrays corresponding to
        
        .. math ::
            \Delta^{\text{horiz}}_{n,m} = \langle c_{n,m\uparrow}c_{n+1,m\downarrow}
            - c_{n,m\downarrow}c_{n+1,m\uparrow} \rangle \\
            = \sum_{E_\nu>0} (B_{n,m\uparrow}^\nu)^* A_{n+1,m\downarrow}^\nu
            - (B_{n,m\downarrow}^\nu)^* A_{n+1,m\uparrow}^\nu \\
            \Delta^{\text{vert}}_{n,m} = \langle c_{n,m\uparrow}c_{n,m+1\downarrow}
            - c_{n,m\downarrow}c_{n,m+1\uparrow} \rangle \\
            = \sum_{E_\nu>0} (B_{n,m\uparrow}^\nu)^* A_{n,m+1\downarrow}^\nu
            - (B_{n,m\downarrow}^\nu)^* A_{n,m+1\uparrow}^\nu \\
        """
        positive_energy_eigenvectors = self._get_occupied_eigenstates()
        non_local_pairing_horizontal = np.zeros((self.L_x-1, self.L_y, 2, 2), dtype=complex)
        non_local_pairing_vertical = np.zeros((self.L_x, self.L_y-1, 2, 2), dtype=complex)
        for nu in range(np.shape(positive_energy_eigenvectors)[1]):
            A_up, A_down, B_up, B_down = \
                self._get_components(positive_energy_eigenvectors[:, nu])
            non_local_pairing_horizontal_nu = np.zeros_like(non_local_pairing_horizontal)
            for i in range(self.L_x-1):
                for j in range(self.L_y):
                    for k in range(2):
                        for l in range(2):
                            non_local_pairing_horizontal_nu[i, j, k, l] = np.array([[B_up[i, j].conj()*A_up[i+1, j], B_up[i, j].conj()*A_down[i+1, j]],
                                                                                         [B_down[i, j].conj()*A_up[i+1, j], B_down[i, j].conj()*A_down[i+1, j]]])[k, l]                                                                 
            non_local_pairing_horizontal += non_local_pairing_horizontal_nu
            non_local_pairing_vertical_nu = np.zeros_like(non_local_pairing_vertical)
            for i in range(self.L_x):
                for j in range(self.L_y-1):
                    for k in range(2):
                        for l in range(2):
                            non_local_pairing_vertical_nu[i, j, k, l] = np.array([[B_up[i, j].conj()*A_up[i, j+1], B_up[i, j].conj()*A_down[i, j+1]],
                                                                                      [B_down[i, j].conj()*A_up[i, j+1], B_down[i, j].conj()*A_down[i, j+1]]])[k, l]
            non_local_pairing_vertical += non_local_pairing_vertical_nu
        return non_local_pairing_horizontal, non_local_pairing_vertical
    def get_local_s_wave_pairing(self):
        local_pairing = self.get_local_pairing()
        local_s_wave_pairing = 1/np.sqrt(2)*(local_pairing[0] - local_pairing[1]) # antisimetric in spin
        return local_s_wave_pairing
