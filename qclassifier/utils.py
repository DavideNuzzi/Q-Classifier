import torch
import numpy as np


def cartesian_to_density_matrix(p):

    x, y, z = p
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    # psi = torch.zeros(2,dtype = torch.cfloat)
    psi = torch.tensor([np.cos(theta/2), np.sin(theta/2)*np.exp(1j * phi)])
    return psi


def state_to_cartesian(psi):

    rho = torch.einsum('bi,bj -> bji', torch.conj(psi), psi)
    return density_matrix_to_cartesian(rho)


def density_matrix_to_cartesian(rho):

    u = 2 * torch.real(rho[:, 0, 1])
    v = 2 * torch.imag(rho[:, 1, 0])
    w = torch.real(rho[:, 0, 0] - rho[:, 1, 1])

    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    w = w.detach().cpu().numpy()

    return [u, v, w]
