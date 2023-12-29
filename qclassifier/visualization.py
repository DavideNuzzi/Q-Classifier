import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from utils import convert_state_to_cartesian, convert_density_to_cartesian
import qutip


# ---------------------------------------------------------------------------- #
#                    Visualization of the model predictions                    #
# ---------------------------------------------------------------------------- #
def plot_prediction(data, Y_pred, problem = None):

    if problem == '1 circle':
        plot_one_circle(data, Y_pred)

    if problem == '3 circles':
        plot_three_circle(data, Y_pred)

    if problem == 'one piece logo':
        plot_onepiece(data, Y_pred)

    if problem == 'MNIST':
         plot_mnist(data, Y_pred)

    if problem == '2D':
        plot_2d(data,Y_pred)


# ------------------------------ Generic 2D plot ----------------------------- #
# Only plots the 2D coordinates and the class labels
def plot_2d(data, Y_pred):
    
    X = data[0]
    Y_true = data[1]

    # Coloro i punti in base alla classe predetta
    plt.subplot(1,2,1)
    C = len(np.unique(Y_true))
    for i in range(C):
        plt.plot(X[Y_pred == i,0],X[Y_pred == i,1],'.', markersize = 2)
    
    plt.axis('equal')
    plt.title('Class prediction')
    plt.legend([f'Class {i}' for i in range(C)])

    # Mostro i punti classificati correttamente o meno
    plt.subplot(1,2,2)

    plt.plot(X[Y_pred == Y_true,0],X[Y_pred == Y_true,1],'.g', markersize = 2)
    plt.plot(X[Y_pred != Y_true,0],X[Y_pred != Y_true,1],'.r', markersize = 2)
    
# ---------------------------- One-circle problem ---------------------------- #
def plot_one_circle(data, Y_pred):

    X = data[0]
    Y_true = data[1]

    # Coloro i punti in base alla classe predetta
    plt.subplot(1,2,1)
    C = len(np.unique(Y_true))
    for i in range(C):
        plt.plot(X[Y_pred == i,0],X[Y_pred == i,1],'.', markersize = 2)
    
    ax = plt.gca()
    ax.add_patch(Circle((0,0), np.sqrt(2/np.pi),edgecolor = 'k',facecolor = 'w', linewidth = 1))
    plt.axis('equal')
    plt.title('Class prediction')
    plt.legend([f'Class {i}' for i in range(C)])

    # Mostro i punti classificati correttamente o meno
    plt.subplot(1,2,2)

    plt.plot(X[Y_pred == Y_true,0],X[Y_pred == Y_true,1],'.g', markersize = 2)
    plt.plot(X[Y_pred != Y_true,0],X[Y_pred != Y_true,1],'.r', markersize = 2)
    
    ax = plt.gca()
    ax.add_patch(Circle((0,0), np.sqrt(2/np.pi),edgecolor = 'k',facecolor = 'w', linewidth = 1))
    plt.axis('equal')
    plt.title('Classification result')
    plt.legend(['Correct','Incorrect'])     


# --------------------------- Three-circles problem -------------------------- #
def plot_three_circle(data, Y_pred):
        
    X = data[0]
    Y_true = data[1]

    # Coloro i punti in base alla classe predetta
    plt.subplot(1,2,1)
    C = len(np.unique(Y_true))
    for i in range(C):
        plt.plot(X[Y_pred == i,0],X[Y_pred == i,1],'.', markersize = 2)
    
    ax = plt.gca()
    ax.add_patch(Circle((-0.5,-0.5),0.5,edgecolor = 'k',facecolor = 'w', linewidth = 1))
    ax.add_patch(Circle((-1,1),1,edgecolor = 'k',facecolor = 'w', linewidth = 1))
    ax.add_patch(Circle((1,0), np.sqrt(6/np.pi - 1),edgecolor = 'k',facecolor = 'w', linewidth = 1))

    plt.axis('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title('Class prediction')
    plt.legend([f'Class {i}' for i in range(C)])

    # Mostro i punti classificati correttamente o meno
    plt.subplot(1,2,2)

    plt.plot(X[Y_pred == Y_true,0],X[Y_pred == Y_true,1],'.g', markersize = 2)
    plt.plot(X[Y_pred != Y_true,0],X[Y_pred != Y_true,1],'.r', markersize = 2)
    
    ax = plt.gca()
    ax.add_patch(Circle((-0.5,-0.5),0.5,edgecolor = 'k',facecolor = 'w', linewidth = 1))
    ax.add_patch(Circle((-1,1),1,edgecolor = 'k',facecolor = 'w', linewidth = 1))
    ax.add_patch(Circle((1,0), np.sqrt(6/np.pi - 1),edgecolor = 'k',facecolor = 'w', linewidth = 1))

    plt.axis('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title('Classification result')
    plt.legend(['Correct','Incorrect'])     

def plot_onepiece(data, Y_pred):

    X = data[0]
    Y_true = data[1]

    # Coloro i punti in base alla classe predetta
    plt.subplot(1,2,1)
    C = len(np.unique(Y_true))
    for i in range(C):
        plt.plot(X[Y_pred == i,0],X[Y_pred == i,1],'.', markersize = 5)
    
    plt.axis('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title('Class prediction')
    plt.legend([f'Class {i}' for i in range(C)])

    plt.subplot(1,2,2)

    plt.plot(X[Y_pred == Y_true,0],X[Y_pred == Y_true,1],'.g', markersize = 5)
    plt.plot(X[Y_pred != Y_true,0],X[Y_pred != Y_true,1],'.r', markersize = 5)
    
    plt.axis('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title('Classification result')
    plt.legend(['Correct','Incorrect'])   

# ------------------------------- MNIST dataset ------------------------------ #
def plot_mnist(data, Y_pred):
        
    X = data[0]
    Y_true = data[1]
    
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(np.reshape(X[i,:].numpy(),(10,10)))
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Pred = {Y_pred[i].item()}, True = {Y_true[i]}')
     

# ---------------------------------------------------------------------------- #
#                              Other utility plots                             #
# ---------------------------------------------------------------------------- #
        
# Plotto le previsioni del modello multiqubit
def plot_multiqubit_on_bloch_sphere(rho, labels):

    # rho : B x C x 2 x 2 

    # In questo caso ho C qubit, quindi C sfere di Bloch distinte
    # Ogni sample avrà la sua classe reale, da mostrare comunque come un colore
    B, C, _, _ = rho.shape

    for i in range(C):

        # Mappo gli elementi delle rho sulla sfera
        u,v,w = convert_density_to_cartesian(rho[:,i,:,:])

        # Creo la sfera e setto i parametri
        b = qutip.Bloch()
        b.make_sphere()
        point_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        b.point_color = point_color
        b.point_size = [2]
        b.frame_color = (0,0,0)      
        b.sphere_color = point_color[i]
        b.sphere_alpha = 0.05

        # Ciclo sulle classi reali
        for j in range(C):
            mask = labels == j
            points = [u[mask], v[mask], w[mask]]
            b.add_points(points)

        b.render()
        # b.show()

def plot_on_bloch_sphere(rho, labels, base, view = [-60,30]):

    # rho : B x 2 x 2  (più sample contemporaneamente)

    # Mappo gli elementi delle rho sulla sfera
    # u = 2 * torch.real(rho[:,0,1])
    # v = 2 * torch.imag(rho[:,1,0])
    # w = torch.real(rho[:,0,0] - rho[:,1,1])
    u,v,w = convert_density_to_cartesian(rho)

    b = qutip.Bloch(view=view)
    b.make_sphere()
    point_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    b.point_color = point_color
    b.point_size = [1]
    b.vector_color = point_color
    b.frame_color = (0,0,0)

    C = torch.max(labels) + 1

    for i in range(C):
        mask = labels == i
        points = [u[mask], v[mask], w[mask]]
        b.add_points(points)

        psi = torch.transpose(base[:,i:(i+1)],1,0)
        p = convert_state_to_cartesian(psi)
        p = [p[0].item(),p[1].item(),p[2].item()]
        b.add_vectors(p)

    b.render()
    # b.show()
