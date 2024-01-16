import torch
import numpy as np


def cartesian_to_density_matrix(p):

    x, y, z = p
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

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


def get_maximally_orthogonal_states(C, device):

    # Base computazionale
    if C == 2:
        psi_c = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.cfloat, device=device)

    # Triangolo equilatero
    if C == 3:
        psi_c = torch.tensor([[1, 1/2,          1/2],
                              [0, np.sqrt(3)/2, -np.sqrt(3)/2]],
                             dtype=torch.cfloat, device=device)

    # Tetraedro
    if C == 4:
        psi_c = torch.zeros((2, 4), dtype=torch.cfloat, device=device)
        psi_c[:, 0] = cartesian_to_density_matrix((0, 0, 1))
        psi_c[:, 1] = cartesian_to_density_matrix((np.sqrt(8/9), 0, -1/3))
        psi_c[:, 2] = cartesian_to_density_matrix((-np.sqrt(2/9),
                                                   np.sqrt(2/3), -1/3))
        psi_c[:, 2] = cartesian_to_density_matrix((-np.sqrt(2/9),
                                                   -np.sqrt(2/3), -1/3))

        # psi_c = torch.tensor([[1,1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
        #                       [0,np.sqrt(2/3),np.sqrt(2/3) *
        #                       np.exp(1j * 2 * np.pi/3),np.sqrt(2/3) *
        #                       np.exp(-1j * 2 * np.pi/3)]],
        #                      dtype=torch.cfloat, device=self.device)

    # # Distribuzione bipiramidale
    # # https://arxiv.org/pdf/0906.0937.pdf
    if C == 5:
        psi_c = torch.zeros((2, 5), dtype=torch.cfloat, device=device)

        psi_c[:, 0] = cartesian_to_density_matrix((0, 0, 1))
        psi_c[:, 1] = cartesian_to_density_matrix((0, 0, -1))
        psi_c[:, 2] = cartesian_to_density_matrix((1, 0, 0))
        psi_c[:, 3] = cartesian_to_density_matrix((-1/2, np.sqrt(3)/2, 0))
        psi_c[:, 4] = cartesian_to_density_matrix((-1/2, -np.sqrt(3)/2, 0))

    # Gyroelongated square bipyramid
    # https://polytope.miraheze.org/wiki/Gyroelongated_square_bipyramid
    # https://en.wikipedia.org/wiki/Thomson_problem
    if C == 10:
        psi_c = torch.zeros(
            (2, 10), dtype=torch.cfloat, device=device)

        a = np.sqrt(2) / 2
        b = np.power(8, 1/4)/4

        psi_c[:, 0] = cartesian_to_density_matrix((0, 0, a+b))
        psi_c[:, 1] = cartesian_to_density_matrix((0, 0, -a-b))
        psi_c[:, 2] = cartesian_to_density_matrix((1/2, 1/2, b))
        psi_c[:, 3] = cartesian_to_density_matrix((1/2, -1/2, b))
        psi_c[:, 4] = cartesian_to_density_matrix((-1/2, 1/2, b))
        psi_c[:, 5] = cartesian_to_density_matrix((-1/2, -1/2, b))
        psi_c[:, 6] = cartesian_to_density_matrix((0, a, -b))
        psi_c[:, 7] = cartesian_to_density_matrix((0, -a, -b))
        psi_c[:, 8] = cartesian_to_density_matrix((a, 0, -b))
        psi_c[:, 9] = cartesian_to_density_matrix((-a, 0, -b))

    return psi_c
