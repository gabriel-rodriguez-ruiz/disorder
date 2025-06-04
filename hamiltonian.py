# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:37:41 2023

@author: gabri
"""
import numpy as np
from pauli_matrices import tau_y, sigma_y
import scipy

class Hamiltonian(object):
    r"""A class for 2D Bogoliubov-de-Gennes Hamiltonians.

        Parameters
        ----------
        
        L_x : int
            Number of sites in x-direction (horizontal).
        L_y : int
            Number of sites in y-direction (vertical).
        onsite : ndarray
            4x4 matrix representing the onsite term of the Hamiltonian.
        hopping_x : ndarray
            4x4 matrix representing the hopping term in x of the Hamiltonian.
        hopping_y : ndarray
            4x4 matrix representing the hopping term in y of the Hamiltonian.
    
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
              
        H = \sum_i^{L_x}\sum_j^{L_y} \mathbf{c}^\dagger_{i,j}\left[ 
                    \text{onsite} \right] \mathbf{c}_{i,j}\nonumber
        				+ 
                    \sum_i^{L_x}\sum_j^{L_y-1}\left[\mathbf{c}^\dagger_{i,j}
                    \left(\text{hopping_y} \right)\mathbf{c}_{i,j+1}
                    + H.c.\right]
                    +\sum_i^{L_x-1}\sum_j^{L_y}\left[\mathbf{c}^\dagger_{i,j}
                     \left(\text{hopping_x} \right)\mathbf{c}_{i+1,j}
                    + H.c.\right]
    """
    def __init__(self, L_x:int, L_y:int, onsite,
                 hopping_x, hopping_y):
        self.L_x = L_x
        self.L_y = L_y
        self.onsite = onsite
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix().toarray()
    def _index(self, i:int , j:int, alpha:int):    
        #protected method, accesible from derived class but not from object
        r"""Return the index of basis vector given the site (i,j)
        and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
        j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real
        space position.
         
            Parameters
            ----------
            i : int
                Site index in x direction. 1<=i<=L_x
            j : int
                Positive site index in y direction. 1<=j<=L_y
            alpha : int
                Spin index. 0<=alpha<=3        
        .. math ::
            \text{Basis vector} = 
           (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
           
           \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) +
                                              + j-1\right)
           
           \text{real space}
           
           (c_{1L_y} &... c_{L_xL_y})
                            
           (c_{11} &... c_{L_x1})

        """
        if (i>self.L_x or j>self.L_y):
            raise Exception("Site index should not be greater than \
                            samplesize.")
        if (i<1 or j<1):
            raise Exception("Site index should be a positive integer")
        return alpha + 4*( self.L_y*(i-1) + j-1 )
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
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i , j, alpha), self._index(i, j, beta)]\
                            = 1/2*self.onsite[alpha, beta]
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
    def is_charge_conjugation(self):
        """
        Check if charge conjugation is present.

        Parameters
        ----------
        H : Hamltonian
            H_BdG Hamiltonian.

        Returns
        -------
        True or false depending if the symmetry is present or not.

        """
        C = np.kron(tau_y, sigma_y)     #charge conjugation operator
        M = np.kron(np.eye(self.L_x*self.L_y), C)      
        return np.all(np.linalg.inv(M) @ self.matrix @ M
                      == -self.matrix.conj())
    def _get_matrix_periodic_in_y(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the y direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y
        for i in range(1, self.L_x+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(i, self.L_y, alpha),
                      self._index(i, 1, beta)] =\
                        self.hopping_y[alpha, beta]
        return M + M.conj().T
    def _get_matrix_periodic_in_x(self):     #self is the Hamiltonian class
        """The part of the tight binding matrix which connects the first
        and last site in the x direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y
        for j in range(1, self.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(self.L_x, j, alpha),
                      self._index(1, j, beta)] =\
                        self.hopping_x[alpha, beta]
        return M + M.conj().T
        
class PeriodicHamiltonianInY(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()

class PeriodicHamiltonianInYandX(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix().toarray()\
                        + super()._get_matrix_periodic_in_y().toarray()\
                        + super()._get_matrix_periodic_in_x().toarray()

class SparseHamiltonian(Hamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        self.L_x = L_x      #Do not use super().__init__ because it is sparse
        self.L_y = L_y
        self.onsite = onsite
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix()
        
class SparsePeriodicHamiltonianInY(SparseHamiltonian):
    def __init__(self, L_x:int, L_y:int, onsite, hopping_x, hopping_y):
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)
        self.matrix = super()._get_matrix() +\
                        super()._get_matrix_periodic_in_y()
