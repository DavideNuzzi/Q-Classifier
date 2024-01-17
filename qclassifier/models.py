import torch
from torch import nn
from qiskit import QuantumCircuit
from .layers import QLayerChunked, QLayer, QLayerMultiQubit
from .models_base import SingeQubitClassifierBase, MultiQubitClassifierBase


class ClassifierSingleQubitChunked(SingeQubitClassifierBase):
    """Based on the paper: 'Data re-uploading for a universal quantum
    classifier'. The input is split in blocks of dimension 3, each block
    is then used to construct an unitary operator that is applied to the qubit

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
    noise_channels : List of NoiseChannel
        List of channels to emulate noise
    """
    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__(D=D, L=L, C=C, device=device)
        self.noise_channels = noise_channels

        # Create the layers
        layers = nn.ModuleList()

        for _ in range(L):
            layers.append(QLayerChunked(
                self.D,
                noise_channels=noise_channels,
                device=device))

        self.layers = layers

        # Move the model on the GPU if needed
        self.to(device)


class ClassifierSingleQubit(SingeQubitClassifierBase):

    """Improved version of the ClassifierSingleQubitChunked that does not need
    to split the input data into blocks.

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
    noise_channels : List of NoiseChannel
        List of channels to emulate noise
    """

    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__(D=D, L=L, C=C, device=device)
        self.noise_channels = noise_channels

        # Create the layers
        layers = nn.ModuleList()

        for _ in range(L):
            layers.append(QLayer(
                self.D,
                noise_channels=noise_channels,
                device=device))

        self.layers = layers

        # Move the model on the GPU if needed
        self.to(device)


class ClassifierMultiQubit(MultiQubitClassifierBase):

    """Implementation of the ClassifierSingleQubit with multiple qubits, one
    for each class of the problem.

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
    noise_channels : List of NoiseChannel
        List of channels to emulate noise
    """

    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__(D=D, L=L, C=C, device=device)
        self.noise_channels = noise_channels

        # Create the layers
        layers = nn.ModuleList()

        for _ in range(L):
            layers.append(
                QLayerMultiQubit(self.D, self.C,
                                 noise_channels=noise_channels,
                                 device=device))

        self.layers = layers

        # Move the model on the GPU if needed
        self.to(device)

    def convert_to_qiskit_circuit(self, x, measure=True):

        circuit = QuantumCircuit(self.C, self.C)

        # Per ogni livello creo un unitario generico con tre rotazioni
        for layer in self.layers:

            phi = layer.theta + torch.einsum('cij,bj -> bci', layer.w, x)

            # Non funziona a batch, quindi anche se la "x" è data come batched
            # utilizzo solo il primo elemento
            phi = (phi[0, :, :]).detach().cpu().numpy()

            for i in range(self.C):
                circuit.rz(phi[i, 2], i)
                circuit.ry(phi[i, 0], i)
                circuit.rz(phi[i, 1], i)

        if measure:
            for i in range(self.C):
                circuit.measure(i, i)

        return circuit
