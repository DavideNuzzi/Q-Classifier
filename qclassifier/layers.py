import torch
import numpy as np
from torch import nn, Tensor
from abc import ABC, abstractmethod


class QLayerBase(nn.Module, ABC):

    """Abstract base class for a generic quantum layer.
    Every layer must implement the forward and init_parameters methods

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    noise_channels : List of NoiseChannel
        Noise channels to apply after each forward pass
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=1, noise_channels=[], device='cpu'):

        super().__init__()

        self.D = D
        self.noise_channels = noise_channels
        self.device = device

    @abstractmethod
    def forward(self, rho: Tensor, x: Tensor) -> Tensor:
        """Forward pass of the layer. Depends on the specific implementation.
        Then the input batch 'x' is uploaded into the weights of the layer and
        the density matrix is processed by the corresponding operator(s).
        The noise channels are applied after the unitary operator(s).
        It returns the processed density matrix.

        Parameters
        ----------
        rho : Tensor
            Input density matrix
        x : Tensor
            Input batch of shape (batch_size x D)

        Returns
        -------
        rho : Tensor
            Output density matrix
        """
        pass

    @abstractmethod
    def init_parameters(self):
        """Reset the parameters of the level"""
        pass


class QLayerChunked(QLayerBase):

    """Based on the <<Data re-uploading for a universal quantum classifier>>
    paper implementation. The D-dimensional input is split into blocks of
    size 3. One operator is constructed for each block and they are applied in
    sequence. The noise channels are applied after each operator. If the input
    dimension D is not divisible by 3, the input is expanded to the nearest
    multiple of 3.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    noise_channels : List of NoiseChannel
        Noise channels to apply after each forward pass
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=3, noise_channels=[], device='cpu'):

        super().__init__(D=D, noise_channels=noise_channels, device=device)

        # Evaluate the nearest multiple of 3 for D
        self.D3 = ((self.D - 1) // 3 + 1) * 3

        # Initialize the parameters
        self.w = nn.Parameter(torch.zeros(D))
        self.theta = nn.Parameter(torch.zeros(self.D3))
        self.init_parameters()

    def init_parameters(self):

        torch.nn.init.normal_(self.w, std=np.sqrt(2 / 3))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    def forward(self, rho, x):

        """Forward pass of the layer. The input batch x is uploaded into the
        weights of the layer, which are then split into blocks of size 3, each
        used to construct an operator. The density matrix is  then processed by
        them and the noise channels are applied at the end. It returns the
        processed density matrix.

        Parameters
        ----------
        rho : Tensor
            Input density matrix of shape (batch_size x 2 x 2)
        x : Tensor
            Input batch of shape (batch_size x D)

        Returns
        -------
        rho : Tensor
            Output density matrix
        """

        # Get the batch size
        batch_size = x.size(0)

        # Create the angles from the weights (theta, w) and input batch x
        # phi = torch.zeros((batch_size, self.D3), device=self.device)
        phi = torch.tile(self.theta, (batch_size, 1))
        phi[:, 0:self.D] = phi[:, 0:self.D] + \
            torch.einsum('j,bj -> bj', self.w, x)

        # Create D//3 operators and apply them
        for i in range(self.D3//3):

            U = torch.zeros((batch_size, 2, 2),
                            dtype=torch.cfloat, device=self.device)

            U[:, 0, 0] = torch.cos(phi[:, i*3]/2) * \
                torch.exp(1j * (phi[:, i*3+1] + phi[:, i*3+2])/2)
            U[:, 0, 1] = -torch.sin(phi[:, i*3]/2) * \
                torch.exp(1j * (phi[:, i*3+1] - phi[:, i*3+2])/2)
            U[:, 1, 0] = torch.sin(phi[:, i*3]/2) * \
                torch.exp(-1j * (phi[:, i*3+1] - phi[:, i*3+2])/2)
            U[:, 1, 1] = torch.cos(phi[:, i*3]/2) * \
                torch.exp(-1j * (phi[:, i*3+1] + phi[:, i*3+2])/2)

            rho = torch.einsum('bim, bmn, bjn -> bij', U, rho, torch.conj(U))

            # Apply noise
            if len(self.noise_channels) > 0:
                for channel in self.noise_channels:
                    rho = channel(rho)

        return rho


class QLayer(QLayerBase):

    """Improved layer that maps any input dimension D into a single operator.
    First the operator acts on the density matrix, then the noise channels are
    applied.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    noise_channels : List of NoiseChannel
        Noise channels to apply after each forward pass
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=3, noise_channels=[], device='cpu'):

        super().__init__(D=D, noise_channels=noise_channels, device=device)

        # Initialize the parameters
        self.w = nn.Parameter(torch.zeros(3, D))
        self.theta = nn.Parameter(torch.zeros(3))
        self.init_parameters()

    def init_parameters(self):

        # (Kaiming)
        torch.nn.init.normal_(self.w, std=np.sqrt(2 / self.D))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    def forward(self, rho, x):

        """Forward pass of the layer. The input batch x is uploaded into the
        weights of the layer, which are then mapped into 3 angles used to
        define an unitary operator. The density matrix is then processed by
        it and the noise channels are applied at the end. It returns the
        processed density matrix.

        Parameters
        ----------
        rho : Tensor
            Input density matrix of shape (batch_size x 2 x 2)
        x : Tensor
            Input batch of shape (batch_size x D)

        Returns
        -------
        rho : Tensor
            Output density matrix
        """

        # Get the batch size
        batch_size = x.size(0)

        # Create the angles of shape (batch_size x 3)
        phi = self.theta + torch.einsum('ij,bj -> bi', self.w, x)

        # Create the operator and apply it
        U = torch.zeros((batch_size, 2, 2),
                        dtype=torch.cfloat, device=self.device)

        U[:, 0, 0] = torch.cos(phi[:, 0]/2) * \
            torch.exp(1j * (phi[:, 1] + phi[:, 2])/2)
        U[:, 0, 1] = -torch.sin(phi[:, 0]/2) * \
            torch.exp(1j * (phi[:, 1] - phi[:, 2])/2)
        U[:, 1, 0] = torch.sin(phi[:, 0]/2) * \
            torch.exp(-1j * (phi[:, 1] - phi[:, 2])/2)
        U[:, 1, 1] = torch.cos(phi[:, 0]/2) * \
            torch.exp(-1j * (phi[:, 1] + phi[:, 2])/2)

        rho = torch.einsum('bim, bmn, bjn -> bij', U, rho, torch.conj(U))

        # Apply noise
        if len(self.noise_channels) > 0:
            for channel in self.noise_channels:
                rho = channel(rho)

        return rho


class QLayerMultiQubit(QLayerBase):

    """Variation of the QLayer that works with many qubits, one for each class
    of the problem. It maps any input dimension D into a C operators, one for
    qubit. Then the operators act on the density matrix and finally the noise
    channels are applied.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    C : int
        Number of classes
    noise_channels : List of NoiseChannel
        Noise channels to apply after each forward pass
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=3, C=2, noise_channels=[], device='cpu'):

        super().__init__(D=D, noise_channels=noise_channels, device=device)

        # Save the number of classes
        self.C = C

        # Initialize the parameters
        self.w = nn.Parameter(torch.zeros(C, 3, D))
        self.theta = nn.Parameter(torch.zeros(C, 3))
        self.init_parameters()

    def init_parameters(self):

        # (Kaiming)
        torch.nn.init.normal_(self.w, std=np.sqrt(2 / self.D))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    def forward(self, rho, x):

        """Forward pass of the layer. The input batch x is uploaded into the
        weights of the layer, which are then mapped into 3 angles for each
        qubit, so 3 x C angles in total. They are then used to define C
        unitary operators. The density matrix is then processed by them and the
        noise channels are applied at the end. It returns the processed density
        matrix.

        Parameters
        ----------
        rho : Tensor
            Input density matrix of shape (batch_size x C x 2 x 2)
        x : Tensor
            Input batch of shape (batch_size x D)

        Returns
        -------
        rho : Tensor
            Output density matrix
        """

        # Get the batch size
        batch_size = x.size(0)

        # Create the angles of shape (batch_size x C x 3)
        phi = self.theta + torch.einsum('cij,bj -> bci', self.w, x)

        # Create the operators and apply them
        U = torch.zeros((batch_size, self.C, 2, 2),
                        dtype=torch.cfloat, device=self.device)

        U[:, :, 0, 0] = torch.cos(phi[:, :, 0]/2) * \
            torch.exp(1j * (phi[:, :, 1] + phi[:, :, 2])/2)
        U[:, :, 0, 1] = -torch.sin(phi[:, :, 0]/2) * \
            torch.exp(1j * (phi[:, :, 1] - phi[:, :, 2])/2)
        U[:, :, 1, 0] = torch.sin(
            phi[:, :, 0]/2)*torch.exp(-1j * (phi[:, :, 1] - phi[:, :, 2])/2)
        U[:, :, 1, 1] = torch.cos(
            phi[:, :, 0]/2)*torch.exp(-1j * (phi[:, :, 1] + phi[:, :, 2])/2)

        rho = torch.einsum('bcim, bcmn, bcjn -> bcij', U, rho, torch.conj(U))

        # Apply noise
        if len(self.noise_channels) > 0:
            for channel in self.noise_channels:
                rho = channel(rho)

        return rho
