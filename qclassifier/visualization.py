import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from .utils import state_to_cartesian, density_matrix_to_cartesian
import qutip


# -------------------------- Plot example datasets -------------------------- #

def plot_problem(data, labels, problem):

    if problem == '1 circle':
        plot_one_circle(data, labels)
    elif problem == '3 circles':
        plot_three_circles(data, labels)
    elif problem == 'one piece':
        plot_one_piece(data, labels)
    elif problem == 'mnist':
        plot_mnist(data, labels)


def plot_one_circle(data, labels, C=2, ax=None, colors=None, legend=None):

    # If there is no axis, create a new figure
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Plot the data points
    plot_data_2D(data, labels, C, ax, colors)

    # Add the circle
    ax.add_patch(Circle((0, 0), np.sqrt(2/np.pi),
                 edgecolor='k', facecolor='w', linewidth=1))

    # Adjust the axis and add the legend
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    if legend is None:
        legend = [0, 1]

    plt.legend(legend, markerscale=3, framealpha=1)


def plot_three_circles(data, labels, C=4, ax=None, colors=None, legend=None):

    # If there is no axis, create a new figure
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Plot the data points
    plot_data_2D(data, labels, C, ax, colors)

    # Add the circles
    ax.add_patch(Circle((-0.5, -0.5), 0.5, edgecolor='k',
                        facecolor='w', linewidth=1))
    ax.add_patch(Circle((-1, 1), 1, edgecolor='k', 
                        facecolor='w', linewidth=1))
    ax.add_patch(Circle((1, 0), np.sqrt(6/np.pi - 1), edgecolor='k',
                        facecolor='w', linewidth=1))

    # Adjust the axis and add the legend
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    if legend is None:
        legend = [0, 1, 2, 3]

    plt.legend(legend, markerscale=3, framealpha=1)


def plot_one_piece(data, labels, C=4, ax=None, colors=None, legend=None):

    # If there is no axis, create a new figure
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Plot the data points
    plot_data_2D(data, labels, C, ax, colors)

    # Adjust the axis and add the legend
    plt.axis('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    if legend is None:
        legend = [0, 1, 2, 3]

    plt.legend(legend, markerscale=3, framealpha=1)


def plot_mnist(data, labels):

    # Limit the number of images
    N = data.shape[0] if data.shape[0] < 20 else 20

    # Create a new figure
    plt.figure(figsize=(10, (N//4) * 2.5))

    for i in range(N):
        plt.subplot(N//4, 4, i+1)
        plt.imshow(np.reshape(data[i, :].numpy(), (10, 10)))
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.title(labels[i].numpy())


def plot_data_2D(X, labels, classes, ax=None, colors=None):

    plt.sca(ax)

    for i in range(classes):

        if colors is None:
            plt.plot(X[labels == i, 0], X[labels == i, 1], '.', markersize=2)
        else:
            plt.plot(X[labels == i, 0], X[labels == i, 1], '.',
                     color=colors[i], markersize=2)


# -------------------------- Plot model predictions ------------------------- #
def plot_prediction(data, labels_true, labels_predicted, problem=None):

    if problem == '1 circle':
        plot_prediction_one_circle(data, labels_true, labels_predicted)
    elif problem == '3 circles':
        plot_prediction_three_circles(data, labels_true, labels_predicted)
    elif problem == 'one piece':
        plot_prediction_one_piece(data, labels_true, labels_predicted)
    elif problem == 'mnist':
        plot_prediction_mnist(data, labels_true, labels_predicted)


def plot_prediction_one_circle(data, labels_true, labels_predicted):

    plt.figure(figsize=(10, 5))

    # PLOT 1 - Class predictions
    ax = plt.subplot(1, 2, 1)
    plot_one_circle(data, labels_predicted, ax=ax)

    # PLOT 2 - Prediction correctness
    ax = plt.subplot(1, 2, 2)
    correct = 1 * (labels_true == labels_predicted)
    plot_one_circle(data, correct, ax=ax, C=2, colors=['r', 'g'],
                    legend=['Wrong', 'Correct'])


def plot_prediction_three_circles(data, labels_true, labels_predicted):

    plt.figure(figsize=(10, 5))

    # PLOT 1 - Class predictions
    ax = plt.subplot(1, 2, 1)
    plot_three_circles(data, labels_predicted, ax=ax)

    # PLOT 2 - Prediction correctness
    ax = plt.subplot(1, 2, 2)
    correct = 1 * (labels_true == labels_predicted)
    plot_three_circles(data, correct, ax=ax, C=2, colors=['r', 'g'],
                       legend=['Wrong', 'Correct'])


def plot_prediction_one_piece(data, labels_true, labels_predicted):

    plt.figure(figsize=(10, 5))

    # PLOT 1 - Class predictions
    ax = plt.subplot(1, 2, 1)
    plot_one_piece(data, labels_predicted, ax=ax)

    # PLOT 2 - Prediction correctness
    ax = plt.subplot(1, 2, 2)
    correct = 1 * (labels_true == labels_predicted)
    plot_one_piece(data, correct, ax=ax, C=2, colors=['r', 'g'],
                   legend=['Wrong', 'Correct'])


def plot_prediction_mnist(data, labels_true, labels_predicted):

    # Limit the number of images
    N = data.shape[0] if data.shape[0] < 20 else 20

    # Create a new figure
    plt.figure(figsize=(10, (N//4) * 3))

    for i in range(N):
        plt.subplot(N//4, 4, i+1)
        plt.imshow(np.reshape(data[i, :].numpy(), (10, 10)))
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.title(f'True: {labels_true[i].numpy()}' +
                  f'\nPredicted: {labels_predicted[i].numpy()}')


# ------------------------ Visualize density matrices ----------------------- #

def plot_multiqubit_on_bloch_sphere(rho, labels):

    """Plots states on the Bloch sphere, one for each qubit.
    Each qubit is assigned a different color to help distinguish them.

    Parameters
    ----------
    rho : Tensor
        Density matrix of shape (N x C x 2 x 2), where N is the number of
        samples and C is the number of qubits.
    labels : Tensor
        Real labels for each sample, shape (N x 1)
    """

    # Cycle over all qubits/classes
    C = rho.shape[1]

    for i in range(C):

        # Convert each qubit density matrix to cartesian coordinates
        u, v, w = density_matrix_to_cartesian(rho[:, i, :, :])

        # Create the sphere and set the visualization parameters
        b = qutip.Bloch()
        b.make_sphere()
        point_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        b.point_color = point_color
        b.point_size = [2]
        b.frame_color = (0, 0, 0)
        b.sphere_color = point_color[i]
        b.sphere_alpha = 0.05

        # Loop over the real labels and only add the points corresponding to
        # that class onto the current Bloch sphere
        for j in range(C):
            mask = labels == j
            points = [u[mask], v[mask], w[mask]]
            b.add_points(points)

        b.render()


def plot_on_bloch_sphere(rho, labels, base, view=[-60, 30]):

    """Plots states on the Bloch sphere. Each point is assigned a color
    based on the true label for that sample. It also shows the set of
    maximally orthogonal states as vectors.

    Parameters
    ----------
    rho : Tensor
        Density matrix of shape (N x 2 x 2), where N is the number of samples.
    labels : Tensor
        Real labels for each sample, shape (N x 1).
    base : Tensor
        Set of maximally orthogonal vectors for this problem.
    """

    # Convert each density matrix to cartesian coordinates
    u, v, w = density_matrix_to_cartesian(rho)

    # Create the sphere and set the visualization parameters
    b = qutip.Bloch(view=view)
    b.make_sphere()
    point_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    b.point_color = point_color
    b.point_size = [1]
    b.vector_color = point_color
    b.frame_color = (0, 0, 0)

    C = torch.max(labels) + 1

    # Loop over the real labels and plot the corresponding points, using a
    # different color for each class.
    for i in range(C):
        mask = labels == i
        points = [u[mask], v[mask], w[mask]]
        b.add_points(points)

        # Show the vector corresponding to this class
        psi = torch.transpose(base[:, i:(i+1)], 1, 0)
        p = state_to_cartesian(psi)
        p = [p[0].item(), p[1].item(), p[2].item()]
        b.add_vectors(p)

    b.render()
