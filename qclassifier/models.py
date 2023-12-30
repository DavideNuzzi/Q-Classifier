import torch
import numpy as np
from torch import nn
from .layers import QLayerOriginal, QLayer, QLayerMultiQubit
from .utils import cartesian_to_density_matrix


# --------------------------------------------------------------------------- #
#                           Single-qubit Classifier                           #
# --------------------------------------------------------------------------- #
# Classe astratta, descrive un generico classificatore con singolo qubit, senza
# specificare il tipo di livello. Implementa una funzione di loss basata sulla
# scelta di C vettori "massimalmente ortogonali" (con C numero di classi),
# basato su "Data re-uploading for a universal quantum classifier". Attualmente
# implementa solo i casi C = 2,3,4,5,10.

class SingeQubitClassifier(nn.Module):

    def __init__(self, D=3, L=1, C=2, device='cpu'):

        super().__init__()

        self.D = D
        self.L = L
        self.C = C
        self.device = device

        # Ulteriori parametri per pesare le classi
        self.alpha = nn.Parameter(torch.ones(C))
        nn.init.ones_(self.alpha)

        # Inizializzo i vettori massimalmente ortogonali così da non doverli
        # ricalcolare ogni volta
        self.psi_c = self.get_maximally_orthogonal_states()

    def loss(self, rho, labels):

        # Calcolo per ogni label del batch, la fidelity attesa rispetto a tutti
        # gli stati
        Y = self.get_expected_fidelity(labels)

        # Vedo la fidelity del mio output
        F = self.get_fidelity(rho)

        # Calcolo la loss
        loss = torch.mean(torch.sum(torch.square(F - Y), axis=1))

        return loss

    def get_fidelity(self, rho):
        return torch.real(torch.einsum('c,ic,bij,jc -> bc',
                                       self.alpha,
                                       torch.conj(self.psi_c),
                                       rho,
                                       self.psi_c))

    def get_expected_fidelity(self, labels):

        psi_l = self.psi_c[:, labels]
        return torch.square(torch.abs(torch.einsum('ic,ib -> bc',
                                                   torch.conj(self.psi_c),
                                                   psi_l)))

    def get_maximally_orthogonal_states(self):

        # Base computazionale
        if self.C == 2:
            psi_c = torch.tensor(
                [[1, 0], [0, 1]], dtype=torch.cfloat, device=self.device)

        # Triangolo equilatero
        if self.C == 3:
            psi_c = torch.tensor([[1, 1/2,          1/2],
                                  [0, np.sqrt(3)/2, -np.sqrt(3)/2]],
                                 dtype=torch.cfloat, device=self.device)

        # Tetraedro
        if self.C == 4:
            psi_c = torch.zeros((2, 4), dtype=torch.cfloat, device=self.device)
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
        if self.C == 5:
            psi_c = torch.zeros((2, 5), dtype=torch.cfloat, device=self.device)

            psi_c[:, 0] = cartesian_to_density_matrix((0, 0, 1))
            psi_c[:, 1] = cartesian_to_density_matrix((0, 0, -1))
            psi_c[:, 2] = cartesian_to_density_matrix((1, 0, 0))
            psi_c[:, 3] = cartesian_to_density_matrix((-1/2, np.sqrt(3)/2, 0))
            psi_c[:, 4] = cartesian_to_density_matrix((-1/2, -np.sqrt(3)/2, 0))

        # Gyroelongated square bipyramid
        # https://polytope.miraheze.org/wiki/Gyroelongated_square_bipyramid
        # https://en.wikipedia.org/wiki/Thomson_problem
        if self.C == 10:
            psi_c = torch.zeros(
                (2, 10), dtype=torch.cfloat, device=self.device)

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

    def predict(self, x):

        # Faccio un forward pass e poi valuto tutte le probabilità
        rho = self.forward(x)

        # Vedo le probabilità per ogni classe e restituisco la massima
        p = self.get_fidelity(rho)

        return torch.argmax(p, dim=1)

    def reset_parameters(self):

        nn.init.ones_(self.alpha)
        for layer in self.layers:
            layer.init_parameters()


# --------------------------------------------------------------------------- #
#                             Original Classifier                             #
# --------------------------------------------------------------------------- #
# Implementazione del classificatore descritto in "Data re-uploading for a
# universal quantum classifier". Lavora con un solo qubit. Se D > 3, con D
# dimensione dell'input, esso viene diviso in blocchi 3-dimensionali
# indipendenti e, per ognuno di essi, vengono poi eventualmente applicati i
# canali che descrivono il rumore.

class ClassifierOriginal(SingeQubitClassifier):

    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__(D=D, L=L, C=C, device=device)

        self.noise_channels = noise_channels

        layers = nn.ModuleList()
        for _ in range(L):
            layers.append(QLayerOriginal(self.D, device=device))
        self.layers = layers

        # Se necessario sposto il modello sulla gpu
        self.to(device)

    def forward(self, x):

        batch_size = x.size(0)

        # Costruisco uno stato per ogni qubit (uno per sample del batch)
        rho = torch.zeros((batch_size, 2, 2),
                          dtype=torch.cfloat, device=self.device)
        rho[:, 0, 0] = 1

        # Propago in avanti tra i livelli
        for i in range(self.L):
            layer = self.layers[i]

            # Per come funziona questo modello devo passare i canali di rumore
            # a ogni livello in modo che vengano applicati dopo ogni unitario
            rho = layer(rho, x, self.noise_channels)

        return rho


# --------------------------------------------------------------------------- #
#                             Improved ClassifierX                            #
# --------------------------------------------------------------------------- #
# La mia versione migliorata, che invece di dividere gli input in blocchi da 3
# lavora direttamente su tutto l'input, mappandolo su 3 angoli distinti e
# costruendo un solo unitario. Questa cosa aumenta il numero di parametri ma al
# contempo riduce notevolmente il numero di operazioni da fare (sia al livello
# di rete neurale che su un computer quantistico) e conseguentemente oltre a
# risparmiare tempo minimizza anche il numero di volte in cui possono esserci
# errori dovuti al rumore.

class ClassifierImproved(SingeQubitClassifier):

    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__(D=D, L=L, C=C, device=device)

        self.noise_channels = noise_channels

        layers = nn.ModuleList()
        for _ in range(L):
            layers.append(QLayer(self.D, device=device))
        self.layers = layers

        # Se necessario sposto il modello sulla gpu
        self.to(device)

    def forward(self, x):

        batch_size = x.size(0)

        # Costruisco uno stato per ogni qubit (uno per sample del batch)
        rho = torch.zeros((batch_size, 2, 2),
                          dtype=torch.cfloat, device=self.device)
        rho[:, 0, 0] = 1

        # Propago in avanti tra i livelli
        for i in range(self.L):
            layer = self.layers[i]

            rho = layer(rho, x)

            # Applico il noise
            if len(self.noise_channels) > 0:
                for channel in self.noise_channels:
                    rho = channel(rho)
        return rho


# --------------------------------------------------------------------------- #
#                            Multi-qubit classifier                           #
# --------------------------------------------------------------------------- #
# Lavora con C qubit, uno per classe, per poi proiettare sulla base
# computazionale e ottenere una probabilità per ognuna delle classi. La
# probabilità più alta indica la classe vincente. Può essere facilmente
# generalizzato a una classificazione multi-label

class ClassifierMultiQubit(nn.Module):

    def __init__(self, D=3, L=1, C=2, device='cpu', noise_channels=[]):

        super().__init__()

        self.D = D
        self.L = L
        self.C = C
        self.device = device
        self.noise_channels = noise_channels

        self.alpha = nn.Parameter(torch.ones(C))
        nn.init.ones_(self.alpha)

        layers = nn.ModuleList()

        for _ in range(L):
            layers.append(QLayerMultiQubit(self.D, self.C, device=device))

        self.layers = layers

    def forward(self, x):

        batch_size = x.size(0)

        # Costruisco uno stato per ogni qubit (uno per classe per sample del
        # batch)
        rho = torch.zeros((batch_size, self.C, 2, 2),
                          dtype=torch.cfloat, device=self.device)
        rho[:, :, 0, 0] = 1

        for i in range(self.L):
            layer = self.layers[i]
            rho = layer(rho, x)

            if len(self.noise_channels) > 0:
                for channel in self.noise_channels:
                    rho = channel(rho)

        return rho

    def loss(self, rho, labels):

        batch_size = rho.size(0)

        # Le fidelity attese sono banalmente tutte zero tranne una sola, quella
        # corrispondente alla classe giusta Y_c   = (batch_size x C)
        Y_c = self.get_expected_fidelity(labels)

        # Valuto la loss
        p = torch.real(rho[:, :, 0, 0])
        F = torch.einsum('c,bc -> bc', self.alpha, p)

        loss = torch.sum(torch.square(F - Y_c)) / (batch_size)

        return loss

    def predict(self, x):

        # Faccio un forward pass e poi valuto tutte le probabilità
        rho = self.forward(x)

        # Calcolo le probabilità per ogni classe e ogni sample del batch
        p = torch.real(rho[:, :, 0, 0])
        F = torch.einsum('c,bc -> bc', self.alpha, p)

        return torch.argmax(F, dim=1)

    def get_prob(self, x):

        # Faccio un forward pass e poi valuto tutte le probabilità
        rho = self.forward(x)

        # Calcolo le probabilità per ogni classe e ogni sample del batch
        p = torch.real(rho[:, :, 0, 0])
        F = torch.einsum('c,bc -> bc', self.alpha, p)

        return F

    def get_expected_fidelity(self, labels):

        batch_size = labels.size(0)

        Y_c = torch.zeros((batch_size, self.C), device=self.device)

        for i, label in enumerate(labels):
            Y_c[i, label] = 1

        return Y_c

    def reset_parameters(self):

        nn.init.ones_(self.alpha)
        for layer in self.layers:
            layer.init_parameters()
