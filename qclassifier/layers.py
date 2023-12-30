import torch
import numpy as np
from torch import nn


# --------------------------------------------------------------------------- #
#                           Original implementation                           #
# --------------------------------------------------------------------------- #
# Basata su "Data re-uploading for a universal quantum classifier", l'input
# D-dimensionale viene diviso in blocchi da 3 e viene costruito un unitario per
# ogni blocco, da applicare in sequenza. Sono necessari 2D parametri a livello.
# Poiché vengono applicati (D - 1)//3 + 1 unitari a livello, gli eventuali
# rumori vengono passati direttamente alla funzione forward in modo che possano
# essere applicati dopo ogni operazione invece che solo una volta alla fine

class QLayerOriginal(nn.Module):

    def __init__(self, D=3, device='cpu'):

        super().__init__()

        self.D = D
        self.device = device

        # Calcolo il più vicino multiplo di 3 per D
        self.D3 = ((self.D - 1) // 3 + 1) * 3

        # Creo i parametri
        self.w = nn.Parameter(torch.zeros(D))
        self.theta = nn.Parameter(torch.zeros(self.D3))

        # Inizializzo
        self.init_parameters()

    def init_parameters(self):

        # (Kaiming)
        torch.nn.init.normal_(self.w, std=np.sqrt(2 / 3))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    def forward(self, rho, x, noise_channels=[]):

        # rho   - Matrice densità per il batch (batch_size x 2 x 2)
        # x     - Batch di dati classico (batch_size x D)

        batch_size = x.size(0)

        # Angoli generalizzati (batch_size x D) Tengo conto che se D non è un
        # multiplo di 3 devo aggiungere degli angoli nulli
        phi = torch.zeros((batch_size, self.D3), device=self.device)
        phi = torch.tile(self.theta, (batch_size, 1))
        phi[:, 0:self.D] = phi[:, 0:self.D] + \
            torch.einsum('j,bj -> bj', self.w, x)

        # Creo D//3 unitari differenti e li applico
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

            # Se ci sono dei canali per il rumore, li applico ora
            if len(noise_channels) > 0:
                for channel in noise_channels:
                    rho = channel(rho)

        return rho


# --------------------------------------------------------------------------- #
#                           Improved implementation                           #
# --------------------------------------------------------------------------- #
# Mappo l'intero input in 3 angoli distinti, definendo quindi un solo unitario.
# Sono necessari 3 * (D + 1) parametri a livello. La funzione di loss è
# comunque definita come nel modello precedente, quindi lavoro su un solo qubit

class QLayer(nn.Module):

    def __init__(self, D=3, device='cpu'):

        super().__init__()

        self.D = D
        self.device = device

        # Creo i parametri
        self.w = nn.Parameter(torch.zeros(3, D))
        self.theta = nn.Parameter(torch.zeros(3))

        # Inizializzo
        self.init_parameters()

    def init_parameters(self):

        # (Kaiming)
        torch.nn.init.normal_(self.w, std=np.sqrt(2 / self.D))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    # Implemento la rotazione
    def forward(self, rho, x):

        # rho   - Matrice densità per il batch (batch_size x 2 x 2)
        # x     - Batch di dati classico (batch_size x D)

        batch_size = x.size(0)

        # Angoli generalizzati (batch_size x 3)
        phi = self.theta + torch.einsum('ij,bj -> bi', self.w, x)

        # Gate (batch_size x 2 x 2)
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

        return torch.einsum('bim, bmn, bjn -> bij', U, rho, torch.conj(U))


# --------------------------------------------------------------------------- #
#                          Multi-qubit implementation                         #
# --------------------------------------------------------------------------- #
# Aggiungo la possibilità di lavorare su più qubit in contemporanea, utile
# principalmente per un classificatore con un qubit per classe in output. In
# futuro si potrebbe implementare l'entanglement. Sono necessari 3 * C * (D +
# 1) parametri a livello.

class QLayerMultiQubit(nn.Module):

    def __init__(self, D=3, C=2, device='cpu'):

        super().__init__()

        self.D = D
        self.C = C
        self.device = device

        # Creo i parametri
        self.w = nn.Parameter(torch.zeros(C, 3, D))
        self.theta = nn.Parameter(torch.zeros(C, 3))

        # Inizializzo
        self.init_parameters()

    def init_parameters(self):

        # (Kaiming)
        torch.nn.init.normal_(self.w, std=np.sqrt(2 / self.D))  # Weights
        torch.nn.init.zeros_(self.theta)  # Biases

    # Implemento la rotazione
    def forward(self, rho, x):

        # rho   - Matrice densità per il batch (batch_size x C x 2 x 2)
        # x     - Batch di dati classico (batch_size x D)

        batch_size = x.size(0)

        # Angoli generalizzati (batch_size x C x 3)
        phi = self.theta + torch.einsum('cij,bj -> bci', self.w, x)

        # Gate (batch_size x C x 2 x 2)
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

        return torch.einsum('bcim, bcmn, bcjn -> bcij', U, rho, torch.conj(U))
