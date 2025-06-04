#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:05:57 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y,\
                            tau_0, sigma_z
from hamiltonian import Hamiltonian, PeriodicHamiltonianInY,\
                        SparseHamiltonian, SparsePeriodicHamiltonianInY

class ZKMBSuperconductivity():
    r"""Topological superconductor with local and extended s-wave pairing,
     spin-orbit coupling and magnetic field.
    
    Parameters
    ----------
    t : float
        Hopping amplitude in x and y directions. Positive.
    mu : float
        Chemical potential.
    Delta_0 : float
        Local s-wave pairing potential.
    Delta_1 : float
        Extended s-wave pairing potential.
    Lambda : float
        Spin-orbit coupling.
    B_x : float
        Magnetic field in x.
    B_x : float
        Magnetic field in y.
    B_x : float
        Magnetic field in z.
        
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} \vec{c}^\dagger_{n,m} 
       \left(-\mu 
          \tau_z\sigma_0 +\Delta_0\tau_x\sigma_0 
          -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)\right) \vec{c}_{n,m}
       + \frac{1}{2}
       \sum_n^{L_x-1}\sum_m^{L_y}\left[\mathbf{c}^\dagger_{n,m}\left(
           -t\tau_z\sigma_0 +
           i\lambda\tau_z\sigma_y + \Delta_1 \tau_x\sigma_0 \right)\mathbf{c}_{n+1,m}
       + H.c.\right]
       + \frac{1}{2}
       \sum_n^{L_x}\sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{n,m}
       \left(-t\tau_z\sigma_0 -
       i\lambda\tau_z\sigma_x + \Delta_1 \tau_x\sigma_0 \right)\mathbf{c}_{n,m+1}
       + H.c.\right]
    """
    def __init__(self, t:float, mu:float, Delta_0:float, Delta_1: float,
                 Lambda:float, B_x:float, B_y:float, B_z:float):
        self.t = t
        self.mu = mu
        self.Delta_0 = Delta_0
        self.Delta_1 = Delta_1
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
        self.B_z = B_z
    def _get_onsite(self):
        return 1/2*(-self.mu*np.kron(tau_z, sigma_0)
                    + self.Delta_0*np.kron(tau_x, sigma_0)
                    -self.B_x*np.kron(tau_0, sigma_x)
                    -self.B_y*np.kron(tau_0, sigma_y)
                    -self.B_z*np.kron(tau_0, sigma_z))
    def _get_hopping_x(self):
        return 1/2*( -self.t*np.kron(tau_z, sigma_0)
                    +1j*self.Lambda*np.kron(tau_z, sigma_y)
                    +self.Delta_1*np.kron(tau_x, sigma_0) )
    def _get_hopping_y(self):
        return 1/2*( -self.t*np.kron(tau_z, sigma_0)
                    -1j*self.Lambda*np.kron(tau_z, sigma_x)
                    +self.Delta_1*np.kron(tau_x, sigma_0) )

class ZKMBSuperconductor(ZKMBSuperconductivity,
                                  Hamiltonian):
    def __init__(self, L_x:int, L_y: int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        Hamiltonian.__init__(self, L_x, L_y, self._get_onsite(), 
                             self._get_hopping_x(),
                            self._get_hopping_y()) 

class ZKMBSparseSuperconductor(ZKMBSuperconductivity,
                                  SparseHamiltonian):
    def __init__(self, L_x:int, L_y: int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        SparseHamiltonian.__init__(self, L_x, L_y, self._get_onsite(), 
                             self._get_hopping_x(),
                            self._get_hopping_y())
        
class ZKMBSparseSuperconductorPeriodicInY(ZKMBSuperconductivity,
                                       SparsePeriodicHamiltonianInY):
    def __init__(self, L_x:int, L_y:int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                      B_x, B_y, B_z)
        SparsePeriodicHamiltonianInY.__init__(self, L_x, L_y,
                                              self._get_onsite(), 
                                              self._get_hopping_x(),
                                              self._get_hopping_y())    

class ZKMBSuperconductorKY(ZKMBSuperconductivity, Hamiltonian):
    r"""ZKM superconductor for a given k in the y direction and magnetic field.
    
    .. math::

        H_{ZKMB} = \frac{1}{2}\sum_k H_{ZKMB,k}
        
        H_{ZKMB,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 + \left(\Delta_0 +2\Delta_1\cos(k) \right) \tau_x\sigma_0
            -2\lambda sin(k) \tau_z\sigma_x
            -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)
            \right]\vec{c}_n +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 
            -i\lambda \tau_z\sigma_y
            +\Delta_1\tau_x\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    def __init__(self,  k:float, L_x:int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        self.k = k
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        Hamiltonian.__init__(self, L_x, 1, self._get_onsite(),
                             self._get_hopping_x(), np.zeros((4, 4)))
    def _get_onsite(self):
        chi_k = -2*self.t*np.cos(self.k)-self.mu
        return 1/2*( chi_k*np.kron(tau_z, sigma_0) +
                (self.Delta_0+2*self.Delta_1*np.cos(self.k))*np.kron(tau_x, sigma_0) +
                -2*self.Lambda*np.sin(self.k)*np.kron(tau_z, sigma_x)
                -self.B_x*np.kron(tau_0, sigma_x)
                -self.B_y*np.kron(tau_0, sigma_y)
                -self.B_z*np.kron(tau_0, sigma_z))
    def _get_hopping_x(self):
        return 1/2*( -self.t*np.kron(tau_z, sigma_0) -
                    1j*self.Lambda*np.kron(tau_z, sigma_y)
                    +self.Delta_1*np.kron(tau_x, sigma_0))

class ZKMBSuperconductorKX(ZKMBSuperconductivity, Hamiltonian):
    r"""ZKM superconductor for a given k in the x direction and magnetic field.
    
    .. math::

        H_{ZKMB} = \frac{1}{2}\sum_k H_{ZKMB,k}
        
        H_{ZKMB,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 + \left(\Delta_0 +2\Delta_1\cos(k) \right) \tau_x\sigma_0
            +2\lambda sin(k) \tau_z\sigma_y
            -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)
            \right]\vec{c}_n +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 
            +i\lambda \tau_z\sigma_x
            +\Delta_1\tau_x\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    def __init__(self,  k:float, L_y:int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        self.k = k
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        Hamiltonian.__init__(self, 1, L_y, self._get_onsite(),
                             np.zeros((4, 4)), self._get_hopping_y())
    def _get_onsite(self):
        chi_k = -2*self.t*np.cos(self.k)-self.mu
        return 1/2*( chi_k*np.kron(tau_z, sigma_0) +
                (self.Delta_0+2*self.Delta_1*np.cos(self.k))*np.kron(tau_x, sigma_0) +
                +2*self.Lambda*np.sin(self.k)*np.kron(tau_z, sigma_y)
                -self.B_x*np.kron(tau_0, sigma_x)
                -self.B_y*np.kron(tau_0, sigma_y)
                -self.B_z*np.kron(tau_0, sigma_z))
    def _get_hopping_y(self):
        return 1/2*( -self.t*np.kron(tau_z, sigma_0)
                    +1j*self.Lambda*np.kron(tau_z, sigma_x)
                    +self.Delta_1*np.kron(tau_x, sigma_0))

class ZKMBSuperconductorKXKY(ZKMBSuperconductivity, Hamiltonian):
    r"""ZKM superconductor for a given k_x and k_y and magnetic field.
    
    .. math::

        H_{ZKMB} = \frac{1}{2}\sum_k H_{ZKMB,k}
        
        H_{ZKMB,k} = 
            \xi_k\tau_z\sigma_0 + \left(\Delta_0 +2\Delta_1(\cos(k_x) + \cos(k_y) \right) \tau_x\sigma_0
            +2\lambda sin(k_x) \tau_z\sigma_y
            -2\lambda sin(k_y) \tau_z\sigma_x
            -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)
            
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2t(cos(k_x)+\cos(k_y)) - \mu
    """
    def __init__(self,  k_x:float, k_y:float, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        self.k_x = k_x
        self.k_y = k_y
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        Hamiltonian.__init__(self, 1, 1, self._get_onsite(),
                             np.zeros((4, 4)), np.zeros((4, 4)))
    def _get_onsite(self):
        chi_k = -2*self.t * (np.cos(self.k_x) + np.cos(self.k_y)) - self.mu
        return 1/2*( chi_k*np.kron(tau_z, sigma_0) +
                (self.Delta_0 + 2*self.Delta_1* (np.cos(self.k_x) +
                                                 np.cos(self.k_y)))*np.kron(tau_x, sigma_0)
                + 2*self.Lambda*np.sin(self.k_x)*np.kron(tau_z, sigma_y)
                - 2*self.Lambda*np.sin(self.k_y)*np.kron(tau_z, sigma_x)
                - self.B_x*np.kron(tau_0, sigma_x)
                - self.B_y*np.kron(tau_0, sigma_y)
                - self.B_z*np.kron(tau_0, sigma_z))
