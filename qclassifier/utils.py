import torch
import numpy as np


def cartesian_to_state(p):
    """Convert cartesian coordinates (x,y,z) inside the Bloch sphere to a
    quantum state.

    Parameters
    ----------
    p : Tuple/List of floats
        Point coordinates (x,y,z)

    Returns
    -------
    psi : Tensor
        Quantum state
    """
    x, y, z = p
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    psi = torch.tensor([np.cos(theta/2), np.sin(theta/2)*np.exp(1j * phi)])
    return psi


def density_matrix_to_cartesian(rho):
    """Convert a density matrix to a point in the Bloch sphere in cartesian
    coordinates.

    Parameters
    ----------
    rho : Tensor
        Density matrix

    Returns
    -------
    p : List of floats
        Point coordinates (x,y,z)
    """
    u = 2 * torch.real(rho[:, 0, 1])
    v = 2 * torch.imag(rho[:, 1, 0])
    w = torch.real(rho[:, 0, 0] - rho[:, 1, 1])

    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    w = w.detach().cpu().numpy()

    return [u, v, w]


def state_to_cartesian(psi):
    """Convert a quantum state to a point in the Bloch sphere in cartesian
    coordinates

    Parameters
    ----------
    psi : Tensor
        Quantum state

    Returns
    -------
    p : List of floats
        Point coordinates (x,y,z)
    """

    # Convert the pure state to a density matrix and then apply the
    # density_matrix_to_cartesian function
    rho = torch.einsum('bi,bj -> bji', torch.conj(psi), psi)
    return density_matrix_to_cartesian(rho)


def get_maximally_orthogonal_states(C, device):
    """Get a set of C maximally orthogonal states on the Bloch sphere.
    For C = 2,3,4,6,8,12,20 the optimal solution is known (for C > 3 it is
    given by the vertices of the Platonic solids). For other values of C the
    solution is an approximation given by the best numerical solution to the
    Thomson_problem (https://en.wikipedia.org/wiki/Thomson_problem). Currently
    it implements only C = 2,3,4,5,10

    Parameters
    ----------
    C : int
        Number of classes
    device : string
        Device to initialize the vectors on (cpu or gpu)

    Returns
    -------
    psi_c : Tensor
        Set of maximally orthogonal vectors
    """

    # Computational basis
    if C == 2:
        psi_c = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.cfloat, device=device)

    # Equilateral triangle
    if C == 3:
        psi_c = torch.tensor([[1, 1/2,          1/2],
                              [0, np.sqrt(3)/2, -np.sqrt(3)/2]],
                             dtype=torch.cfloat, device=device)

    # Tetraedron
    if C == 4:
        psi_c = torch.zeros((2, 4), dtype=torch.cfloat, device=device)
        psi_c[:, 0] = cartesian_to_state((0, 0, 1))
        psi_c[:, 1] = cartesian_to_state((np.sqrt(8/9), 0, -1/3))
        psi_c[:, 2] = cartesian_to_state((-np.sqrt(2/9), np.sqrt(2/3), -1/3))
        psi_c[:, 3] = cartesian_to_state((-np.sqrt(2/9), -np.sqrt(2/3), -1/3))

    # Bipyramid
    # https://arxiv.org/pdf/0906.0937.pdf
    if C == 5:
        psi_c = torch.zeros((2, 5), dtype=torch.cfloat, device=device)

        psi_c[:, 0] = cartesian_to_state((0, 0, 1))
        psi_c[:, 1] = cartesian_to_state((0, 0, -1))
        psi_c[:, 2] = cartesian_to_state((1, 0, 0))
        psi_c[:, 3] = cartesian_to_state((-1/2, np.sqrt(3)/2, 0))
        psi_c[:, 4] = cartesian_to_state((-1/2, -np.sqrt(3)/2, 0))

    # Gyroelongated square bipyramid
    # https://polytope.miraheze.org/wiki/Gyroelongated_square_bipyramid
    if C == 10:
        psi_c = torch.zeros(
            (2, 10), dtype=torch.cfloat, device=device)

        a = np.sqrt(2) / 2
        b = np.power(8, 1/4)/4

        psi_c[:, 0] = cartesian_to_state((0, 0, a+b))
        psi_c[:, 1] = cartesian_to_state((0, 0, -a-b))
        psi_c[:, 2] = cartesian_to_state((1/2, 1/2, b))
        psi_c[:, 3] = cartesian_to_state((1/2, -1/2, b))
        psi_c[:, 4] = cartesian_to_state((-1/2, 1/2, b))
        psi_c[:, 5] = cartesian_to_state((-1/2, -1/2, b))
        psi_c[:, 6] = cartesian_to_state((0, a, -b))
        psi_c[:, 7] = cartesian_to_state((0, -a, -b))
        psi_c[:, 8] = cartesian_to_state((a, 0, -b))
        psi_c[:, 9] = cartesian_to_state((-a, 0, -b))

    return psi_c
