import torch
import numpy as np
from abc import ABC
from torch import nn


class NoiseChannel(nn.Module, ABC):

    def __init__(self, device='cpu'):

        super().__init__()

        self.device = device

    def forward(self, rho, K):

        # Check if we are working with more than one qubit and apply
        # the channel accordingly
        if rho.dim() == 3:
            return torch.einsum('imk, bmn , jnk -> bij',
                                K, rho, torch.conj(K))
        else:
            return torch.einsum('imk, bcmn , jnk -> bcij',
                                K, rho, torch.conj(K))


class DepolarizingChannel(NoiseChannel):

    # Simple depolarizing channel parametrized by the probability p. TODO:
    # calculate the probability p using the gate times T1, T2, Tg and the
    # parameter epsilon

    def __init__(self, p=0, device='cpu'):

        super().__init__(device=device)

        self.p = p

        self.K = torch.zeros((2, 2, 4), dtype=torch.cfloat, device=device)
        self.K[:, :, 0] = np.sqrt(1 - 3/4 * p) * torch.tensor([[1, 0], [0, 1]])
        self.K[:, :, 1] = np.sqrt(p/4) * torch.tensor([[0, 1], [1, 0]])
        self.K[:, :, 2] = np.sqrt(p/4) * torch.tensor([[0, -1j], [1j, 0]])
        self.K[:, :, 3] = np.sqrt(p/4) * torch.tensor([[1, 0], [0, -1]])

    def forward(self, rho):
        return super().forward(rho, self.K)


class ThermalRelaxationChannel(NoiseChannel):

    # Simple thermal relaxation channel, parametrized by the phase flip
    # probability p_z, and the reset probabilities p_r0 and p_r1. TODO:
    # calculate the probabilities using the gate times T1, T2, Tg and the
    # temperature.

    def __init__(self, p_z=0, p_r0=0, p_r1=0, device='cpu'):

        super().__init__(device=device)

        self.p_z = p_z
        self.p_r0 = p_r0
        self.p_r1 = p_r1

        self.K = torch.zeros((2, 2, 6), dtype=torch.cfloat, device=device)
        self.K[:, :, 0] = np.sqrt(1 - p_z - p_r0 - p_r1) * \
            torch.tensor([[1, 0], [0, 1]])
        self.K[:, :, 1] = np.sqrt(p_z) * torch.tensor([[1, 0], [0, -1]])
        self.K[:, :, 2] = np.sqrt(p_r0) * torch.tensor([[1, 0], [0, 0]])
        self.K[:, :, 3] = np.sqrt(p_r0) * torch.tensor([[0, 1], [0, 0]])
        self.K[:, :, 4] = np.sqrt(p_r1) * torch.tensor([[0, 0], [1, 0]])
        self.K[:, :, 5] = np.sqrt(p_r1) * torch.tensor([[0, 0], [0, 1]])

    def forward(self, rho):
        return super().forward(rho, self.K)
