#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:11:02 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductor, ZKMBSuperconductivity
from hamiltonian import Hamiltonian
import random
import scipy
from pauli_matrices import *

class DisorderedHamiltonian(Hamiltonian):
    def __init__(self, L_x:int, L_y:int,
                 hopping_x, hopping_y):
        self.L_x = L_x
        self.L_y = L_y
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix().toarray()
    def _get_random_mu(self):
        """Returns random uniform value of mu."""
        w = self.mu + self.mu_0 * random.uniform(-1, 1)
        return w
    def _get_matrix(self):
        r"""
        Matrix of the BdG-Hamiltonian.        
        
        Returns
        -------
        M : ndarray
            Matrix of the BdG-Hamiltonian.
        .. math ::
            \text{matrix space}
            
            (c_{11} &... c_{1L_y})
                             
            (c_{L_x1} &... c_{L_xL_y})
        """
        L_x = self.L_x
        L_y = self.L_y
        M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
        #onsite
        for i in range(1, L_x+1):    
            for j in range(1, L_y+1):
                w = self._get_random_mu()
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i , j, alpha), self._index(i, j, beta)]\
                            = 1/2*self._get_onsite(w)[alpha, beta]
                            # factor 1/2 in the diagonal because I multiplicate
                            # with the transpose conjugate matrix
        #hopping_x
        for i in range(1, L_x):
            for j in range(1, L_y+1):    
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i+1, j, beta)]\
                        = self.hopping_x[alpha, beta]
        #hopping_y
        for i in range(1, L_x+1):
            for j in range(1, L_y): 
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i, j+1, beta)]\
                        = self.hopping_y[alpha, beta]
        return M + M.conj().T
    
class ZKMBDisorderedSuperconductor(ZKMBSuperconductivity,
                                  DisorderedHamiltonian):
    def __init__(self, L_x:int, L_y: int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float, mu_0:float):
        self.mu_0 = mu_0
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        DisorderedHamiltonian.__init__(self, L_x, L_y, 
                             self._get_hopping_x(),
                            self._get_hopping_y())
    def _get_onsite(self, mu):
        return 1/2*(-mu*np.kron(tau_z, sigma_0)
                    + self.Delta_0*np.kron(tau_x, sigma_0)
                    -self.B_x*np.kron(tau_0, sigma_x)
                    -self.B_y*np.kron(tau_0, sigma_y)
                    -self.B_z*np.kron(tau_0, sigma_z))
    
def get_components(state, L_x, L_y):
    """
    Get the components of the state: creation_up,
    creation_down, destruction_down, destruction_up for a given
    column state. Returns an array of shape (L_y, L_x)
    """
    destruction_up = state[0::4].reshape((L_x, L_y))
    destruction_down = state[1::4].reshape((L_x, L_y))
    creation_down = state[2::4].reshape((L_x, L_y))
    creation_up = state[3::4].reshape((L_x, L_y))
    return (np.flip(destruction_up.T, axis=0),
            np.flip(destruction_down.T, axis=0),
            np.flip(creation_down.T, axis=0),
            np.flip(creation_up.T, axis=0))

