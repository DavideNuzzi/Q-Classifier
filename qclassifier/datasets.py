import os
import torch
import numpy as np
import matplotlib.image as img
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
#                              Dataset Generation                             #
# --------------------------------------------------------------------------- #
def get_sample_data(dataset_type, N_train, N_test):

    if dataset_type == '1 circle':
        data_train = one_circle(N_train)
        data_test = one_circle(N_test)

    elif dataset_type == '3 circles':
        data_train = three_circles(N_train)
        data_test = three_circles(N_test)

    elif dataset_type == 'diagonal band':
        data_train = diagonal_band(N_train)
        data_test = diagonal_band(N_test)

    elif dataset_type == 'one piece':
        data_train = one_piece(N_train)
        data_test = one_piece(N_test)

    elif dataset_type == 'mnist 2':
        data_train, data_test = mnist(N_train, N_test, 2)

    elif dataset_type == 'mnist 3':
        data_train, data_test = mnist(N_train, N_test, 3)

    elif dataset_type == 'mnist':
        data_train, data_test = mnist(N_train, N_test)

    return data_train, data_test


# --------------------------------- 1 Circle -------------------------------- #
def one_circle(N):

    D = 2
    X = torch.zeros((N, D))
    Y = torch.zeros(N, dtype=torch.int)

    for n in range(N):

        X[n, :] = 2 * torch.rand((1, D)) - 1
        r = torch.sqrt(X[n, 0]**2 + X[n, 1] ** 2)

        if r < np.sqrt(2/torch.pi):
            Y[n] = 0
        else:
            Y[n] = 1

    return X, Y


# -------------------------------- 3 Circles -------------------------------- #
def three_circles(N):

    D = 2
    X = torch.zeros((N, D))
    Y = torch.zeros(N, dtype=torch.int)

    for n in range(N):

        X[n, :] = 2 * torch.rand((1, D)) - 1

        d1 = torch.sqrt((X[n, 0] + 1)**2 + (X[n, 1] - 1)**2)
        d2 = torch.sqrt((X[n, 0] - 1)**2 + (X[n, 1])**2)
        d3 = torch.sqrt((X[n, 0] + 0.5)**2 + (X[n, 1] + 0.5)**2)

        if d1 < 1:
            Y[n] = 1
        elif d2 < np.sqrt(6/np.pi - 1):
            Y[n] = 2
        elif d3 < 1/2:
            Y[n] = 3
        else:
            Y[n] = 0

    return X, Y


# ------------------------------ Diagonal band ------------------------------ #
def diagonal_band(N):

    D = 2
    X = torch.zeros((N, D))
    Y = torch.zeros(N, dtype=torch.int)

    for n in range(N):

        X[n, :] = 2 * torch.rand((1, D)) - 1

        x = X[n, 0]
        y = X[n, 1]

        y1 = x + np.sqrt(2) - 2
        y2 = x - np.sqrt(2) + 2

        if (y >= y1) and (y <= y2):
            Y[n] = 0
        else:
            Y[n] = 1

    return X, Y


# ------------------------------ One Piece Logo ----------------------------- #
def one_piece(N):

    X = torch.zeros((N, 2))
    Y = torch.zeros(N, dtype=torch.int)

    image = img.imread(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'img', 'logo.png'))

    cols = np.zeros((5, 3))
    cols[0, :] = [0, 0, 0]  # Background
    cols[1, :] = [1, 1, 1]  # Skull
    cols[2, :] = [0, 0, 1]  # Circle
    cols[3, :] = [1, 1, 0]  # Hat

    for n in range(N):

        X[n, :] = 2 * torch.rand((1, 2)) - 1

        i = int(np.floor((1 - X[n, 1]) / 2 * image.shape[1]))
        j = int(np.floor((X[n, 0] + 1) / 2 * image.shape[0]))

        c = image[i, j, :]
        c = c[np.newaxis, :]
        c = np.tile(c, (5, 1))

        Y[n] = np.argmin(np.sum(np.square((cols - c)), axis=1))

    return X, Y


# ---------------------------------- MNIST ---------------------------------- #
def mnist(N_train, N_test, classes=10):

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((10, 10), antialias=True)])
    mnist_ds = datasets.MNIST(
        root='./data', train=True, download=True, transform=transf)

    X = torch.stack([sample[0].view(1, -1) for sample in mnist_ds])
    X = torch.reshape(X, (60000, 10*10))
    Y = mnist_ds.targets

    if classes < 10:
        X = X[Y < classes, :]
        Y = Y[Y < classes]

    X_train = X[0:N_train, :]
    Y_train = Y[0:N_train]
    X_test = X[N_train:(N_train + N_test), :]
    Y_test = Y[N_train:(N_train + N_test)]

    return (X_train, Y_train), (X_test, Y_test)
