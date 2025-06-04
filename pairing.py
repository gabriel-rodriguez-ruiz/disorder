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
    def get_occupied_eigenstates(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.S.matrix) 
        positive_energy_eigenvectors = eigenvectors[:, np.where(eigenvalues>0)[0]]  #where returns a tuple
        return positive_energy_eigenvectors
    def get_components(self, state):
        """
        Get the components of the state: creation_up,
        creation_down, destruction_down, destruction_up for a given
        column state. Returns an array of shape (L_y, L_x).
        """
        destruction_up = state[0::4].reshape((self.L_x, self.L_y))
        destruction_down = state[1::4].reshape((self.L_x, self.L_y))
        creation_down = state[2::4].reshape((self.L_x, self.L_y))
        creation_up = state[3::4].reshape((self.L_x, self.L_y))
        return (np.flip(destruction_up.T, axis=0),
                np.flip(destruction_down.T, axis=0),
                np.flip(creation_down.T, axis=0),
                np.flip(creation_up.T, axis=0))
    def get_s_wave_pairing(self):
        r"""Returns an array whose elements are:
            
        .. math ::
            \Delta^{\text{s-wave}}_{n,m} = \langle c_{n,m\uparrow}c_{n,m\downarrow} \rangle
            = \sum_{E_\nu>0} (B_{n,m\uparrow}^\nu)^* A_{n,m\downarrow}^\nu \\
                where \\
            c_{n,m\sigma} = \sum_{E_\nu>0} A_{n,m\downarrow}^\nu \Gamma^\dagger_\nu + (B_{n,m\uparrow}^\nu)^* \Gamma_\nu  
               \\ and \\
            \Gamma_\nu = \sum_\sigma \sum_{n,m} \left( A_{n,m\sigma}^\nu c^\dagger_{n,m\sigma} 
            + B_{n,m\sigma}^\nu  c_{n,m\sigma} \right)

        """
        positive_energy_eigenvectors = self.get_occupied_eigenstates()
        pairing_s_wave = np.zeros((self.S.L_x, self.S.L_y), dtype=complex)
        for nu in range(np.shape(positive_energy_eigenvectors)[1]):
            creation_up, creation_down, destruction_down, destruction_up = \
                self.get_components(positive_energy_eigenvectors[:, nu])
            A_down = creation_down
            B_up = destruction_up
            pairing_up_down = B_up.conj() * A_down
            pairing_nu = pairing_up_down
            pairing_s_wave += pairing_nu
        return pairing_s_wave
    def get_p_wave_pairing(self):
        positive_energy_eigenvectors = self.get_occupied_eigenstates()
        pairing_p_wave_horizontal = np.zeros((self.L_x-1, self.L_y), dtype=complex)
        pairing_p_wave_vertical = np.zeros((self.L_x, self.L_y-1), dtype=complex)
        for nu in range(np.shape(positive_energy_eigenvectors)[1]):
            creation_up, creation_down, destruction_down, destruction_up = \
                self.get_components(positive_energy_eigenvectors[:, nu])
            A_up = creation_up
            A_down = creation_down
            B_down = destruction_down
            B_up = destruction_up
            pairing_p_wave_horizontal_nu = np.zeros_like(pairing_p_wave_horizontal)
            for i in range(self.L_x-1):
                for j in range(self.L_y):
                    pairing_p_wave_horizontal_nu[i, j] = (B_up[i, j].conj()*A_down[i+1, j]
                                                       - B_down[i, j].conj()*A_up[i+1, j])    
            pairing_p_wave_horizontal += pairing_p_wave_horizontal_nu
            pairing_p_wave_vertical_nu = np.zeros_like(pairing_p_wave_vertical)
            for i in range(self.L_x):
                for j in range(self.L_y-1):
                    pairing_p_wave_vertical_nu[i, j] = (B_up[i, j].conj()*A_down[i, j+1]
                                                       - B_down[i, j].conj()*A_up[i, j+1])
            pairing_p_wave_vertical += pairing_p_wave_vertical_nu
        return pairing_p_wave_horizontal, pairing_p_wave_vertical

