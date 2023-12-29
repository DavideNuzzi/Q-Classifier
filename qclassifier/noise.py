import torch
import numpy as np
from torch import nn


# --------------------------- Depolarizing channel --------------------------- #
# Simple depolarizing channel parametrized by the probability p. TODO: calculate
# the probability p using the gate times T1, T2, Tg and the parameter epsilon
class DepolarizingChannel(nn.Module):
    
    def __init__(self, p = 0, device = 'cpu'):

        super().__init__()

        self.p = p
        self.device = device

        self.K = torch.zeros((2,2,4), dtype = torch.cfloat, device = device)
        self.K[:,:,0] = np.sqrt(1 - 3/4 * p) * torch.tensor([[1,0],[0,1]])
        self.K[:,:,1] = np.sqrt(p/4)         * torch.tensor([[0,1],[1,0]])
        self.K[:,:,2] = np.sqrt(p/4)         * torch.tensor([[0,-1j],[1j,0]])
        self.K[:,:,3] = np.sqrt(p/4)         * torch.tensor([[1,0],[0,-1]])

    def forward(self,rho):
        
        # Applico il canale
        # Deve funzionare sia per il caso multi-qubit che per il singolo
        if rho.dim() == 3:
            return torch.einsum('imk, bmn , jnk -> bij', self.K, rho, torch.conj(self.K))
        else:
            return torch.einsum('imk, bcmn , jnk -> bcij', self.K, rho, torch.conj(self.K))


# ------------------------ Thermal Relaxation Channel ------------------------ #
# Simple thermal relaxation channel, parametrized by the phase flip probability
# p_z, and the reset probabilities p_r0 and p_r1. TODO: calculate the
# probabilities using the gate times T1, T2, Tg and the temperature.
class ThermalRelaxationChannel(nn.Module):
    
    def __init__(self, p_z = 0, p_r0 = 0, p_r1 = 0, device = 'cpu'):

        super().__init__()

        self.p_z = p_z
        self.p_r0 = p_r0
        self.p_r1 = p_r1
        self.device = device

        self.K = torch.zeros((2,2,6), dtype = torch.cfloat, device = device)
        self.K[:,:,0] = np.sqrt(1 - p_z - p_r0 - p_r1)  * torch.tensor([[1,0],[0,1]])
        self.K[:,:,1] = np.sqrt(p_z)                    * torch.tensor([[1,0],[0,-1]])
        self.K[:,:,2] = np.sqrt(p_r0)                   * torch.tensor([[1,0],[0,0]])
        self.K[:,:,3] = np.sqrt(p_r0)                   * torch.tensor([[0,1],[0,0]])
        self.K[:,:,4] = np.sqrt(p_r1)                   * torch.tensor([[0,0],[1,0]])
        self.K[:,:,5] = np.sqrt(p_r1)                   * torch.tensor([[0,0],[0,1]])

    def forward(self,rho):
        
        # Applico il canale
        # Deve funzionare sia per il caso multi-qubit che per il singolo
        if rho.dim() == 3:
            return torch.einsum('imk, bmn , jnk -> bij', self.K, rho, torch.conj(self.K))
        else:
            return torch.einsum('imk, bcmn , jnk -> bcij', self.K, rho, torch.conj(self.K))

