import torch
from torch import nn
from abc import ABC, abstractmethod
from .utils import get_maximally_orthogonal_states
from torch import Tensor


class QuantumClassifier(nn.Module, ABC):

    """Abstract base class for a generic quantum classifier.
    Every classifier must implement the forward, loss and predict methods.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    L : int
        Number of layers in the network
    C : int
        Number of classes
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=1, L=1, C=2, device='cpu'):

        super().__init__()

        self.D = D
        self.L = L
        self.C = C
        self.device = device

    @abstractmethod
    def loss(self, rho: Tensor, labels: Tensor) -> Tensor:
        """Loss of the classifier. Depends on the specific implementation.
        It includes gradients and can be used for GD optimization.

        Parameters
        ----------
        rho : Tensor
            Density matrix for all the batches
        labels : Tensor
            True labels for the batch

        Returns
        -------
        Tensor
            Loss
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model. Depends on the specific implementation.
        The density matrix is initialized here. Then the input batch "x" is
        uploaded into the weights of each layer and the density matrix is
        processed by them. It returns the final density matrix.

        Parameters
        ----------
        x : Tensor
            Input batch of shape: batch_size x D

        Returns
        -------
        Tensor
        rho : Tensor
            Output density matrix
        """
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Predicts the labels of a given batch of data.

        Parameters
        ----------
        x : Tensor
            Input batch of shape: batch_size x D

        Returns
        -------
        y : Tensor
            Predicted labels
        """
        pass

    @abstractmethod
    def reset_parameters() -> None:
        """Resets the parameters of the model without initializing it."""
        pass


class SingeQubitClassifierBase(QuantumClassifier, ABC):

    """Abstract base class for all classifiers using a single qubit.
    It implements a specific type of loss function, and forward pass, but no
    specific layer type.

    Details: a set of maximally orthogonal vectors is assigned to all classes.
    Then, for each input sample, the fidelity between the output of the model
    and each class-vector is compared to the expected fidelity for the true
    class. Finally, the error is computed as the squared difference between
    the two fidelities. A set of trainable weight "alpha" is used to weight
    the fidelity of each class.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    L : int
        Number of layers in the network
    C : int
        Number of classes
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=1, L=1, C=2, device='cpu'):

        super().__init__(D=D, L=L, C=C, device=device)

        # Initialize the parameters
        self.alpha = nn.Parameter(torch.ones(C))
        nn.init.ones_(self.alpha)

        # Initialize the set of maximally orthogonal vectors
        self.psi_c = get_maximally_orthogonal_states(C, device)

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize the density matrix
        rho = torch.zeros((batch_size, 2, 2),
                          dtype=torch.cfloat, device=self.device)
        rho[:, 0, 0] = 1

        # Do a forward pass
        for i in range(self.L):
            layer = self.layers[i]
            rho = layer(rho, x)

        return rho

    def loss(self, rho, labels):

        # Evaluate true and actual fidelity
        F_true = self.get_expected_fidelity(labels)
        F = self.get_fidelity(rho)

        # Evaluate loss
        loss = torch.mean(torch.sum(torch.square(F - F_true), axis=1))

        return loss

    def predict(self, x):

        # Do a forward pass
        rho = self.forward(x)

        # Calculate the probability of each class
        p = self.get_fidelity(rho)

        return torch.argmax(p, dim=1)

    def get_fidelity(self, rho):

        # Calculate the fidelity of the output by projecting it onto the set of
        # maximally orthogonal vectors
        return torch.real(torch.einsum('c,ic,bij,jc -> bc',
                                       self.alpha,
                                       torch.conj(self.psi_c),
                                       rho,
                                       self.psi_c))

    def get_expected_fidelity(self, labels):

        # Calculate the expected fidelity based on the true class labels
        psi_l = self.psi_c[:, labels]
        return torch.square(torch.abs(torch.einsum('ic,ib -> bc',
                                                   torch.conj(self.psi_c),
                                                   psi_l)))

    def reset_parameters(self):

        nn.init.ones_(self.alpha)
        for layer in self.layers:
            layer.init_parameters()


class MultiQubitClassifierBase(QuantumClassifier, ABC):

    """Abstract base class for all classifiers using one qubit per class.
    It implements a specific type of loss function, and forward pass, but no
    specific layer type.

    Details: each qubit encodes the probability of each class. At the end of
    the forward process each of them is measured indipendently and the class
    with the highest probability becomes the prediction.

    Parameters
    ----------
    D : int
        Dimensionality of the input data
    L : int
        Number of layers in the network
    C : int
        Number of classes
    device : string
        Device to run the model on (cpu or gpu)
    """

    def __init__(self, D=3, L=1, C=2, device='cpu'):

        super().__init__(D=D, L=L, C=C, device=device)

        self.alpha = nn.Parameter(torch.ones(C))
        nn.init.ones_(self.alpha)

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize the density matrix
        rho = torch.zeros((batch_size, self.C, 2, 2),
                          dtype=torch.cfloat, device=self.device)
        rho[:, :, 0, 0] = 1

        # Do a forward pass
        for i in range(self.L):
            layer = self.layers[i]
            rho = layer(rho, x)

        return rho

    def loss(self, rho, labels):

        # Evaluate true and actual fidelity
        F_true = self.get_expected_fidelity(labels)
        F = self.get_fidelity(rho)

        # Evaluate the loss
        loss = torch.mean(torch.sum(torch.square(F - F_true), axis=1))

        return loss

    def predict(self, x):

        # Do a forward pass
        rho = self.forward(x)

        # Calculate the probability of each class
        F = self.get_fidelity(rho)

        return torch.argmax(F, dim=1)

    def get_expected_fidelity(self, labels):

        # Calculate the expected fidelity based on the true class labels
        # All the fidelities are zero except for the one corresponding
        # to the true label, which is one
        F = torch.zeros((labels.size(0), self.C), device=self.device)

        for i, label in enumerate(labels):
            F[i, label] = 1

        return F

    def get_fidelity(self, rho):

        # Get the fidelity for each qubit (probability of the up state)
        return torch.real(rho[:, :, 1, 1])

        # return torch.einsum('c,bc -> bc', self.alpha,
        #                     torch.real(rho[:, :, 1, 1]))

    def reset_parameters(self):

        nn.init.ones_(self.alpha)
        for layer in self.layers:
            layer.init_parameters()
