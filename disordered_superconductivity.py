#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:11:02 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductivity
from disordered_hamiltonian import DisorderedHamiltonian, PeriodicDisorderedHamiltonianInY
from pauli_matrices import tau_z, sigma_0, tau_x, sigma_x, tau_0, sigma_y, sigma_z


class ZKMBDisorderedSuperconductivity(ZKMBSuperconductivity):
    def __init__(self, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        ZKMBSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
    def _get_onsite(self, mu):
        return 1/2*(-mu*np.kron(tau_z, sigma_0)
                    + self.Delta_0*np.kron(tau_x, sigma_0)
                    -self.B_x*np.kron(tau_0, sigma_x)
                    -self.B_y*np.kron(tau_0, sigma_y)
                    -self.B_z*np.kron(tau_0, sigma_z))
    
class ZKMBDisorderedSuperconductor(ZKMBDisorderedSuperconductivity,
                                  DisorderedHamiltonian):
    def __init__(self, L_x:int, L_y: int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float, mu_0:float):
        self.mu_0 = mu_0
        ZKMBDisorderedSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                       B_x, B_y, B_z)
        DisorderedHamiltonian.__init__(self, L_x, L_y, 
                             self._get_hopping_x(),
                            self._get_hopping_y())
    
class ZKMBDisorderedSuperconductorPeriodicInY(ZKMBDisorderedSuperconductivity,
                                       PeriodicDisorderedHamiltonianInY):
    def __init__(self, L_x:int, L_y:int, t:float, mu:float,
                 Delta_0:float, Delta_1:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float, mu_0:float):
        self.mu_0 = mu_0
        ZKMBDisorderedSuperconductivity.__init__(self, t, mu, Delta_0, Delta_1, Lambda,
                                                 B_x, B_y, B_z)
        PeriodicDisorderedHamiltonianInY.__init__(self, L_x, L_y,
                                                  self._get_hopping_x(),
                                                  self._get_hopping_y())  

