#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 16:16:51 2025

@author: gabriel
"""

from hamiltonian import Hamiltonian
import random
import scipy

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

class PeriodicDisorderedHamiltonianInY(DisorderedHamiltonian):
    def __init__(self, L_x:int, L_y:int, hopping_x, hopping_y):
        super().__init__(L_x, L_y, hopping_x, hopping_y)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()