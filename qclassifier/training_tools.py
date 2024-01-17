import torch
import numpy as np
import copy
from tqdm import tqdm


def best_initial_conditions(model, data, epochs=5, batch_size=20, samples=20):
    """Train the model multiple times starting from different initial weights
    distributions, for a limited number of epochs, and return the model with
    the lower training loss.

    Parameters
    ----------
    model : QuantumClassifier
        Model to train.
    data : Tuple
        Training set. Must be provided as a tuple (X_train,Y_train).
    epochs : int, optional
        Epochs for training, by default 5.
    batch_size : int, optional
        Batch size, by default 20.
    samples : int, optional
        Number of times the model is randomized and tested, by default 20.

    Returns
    -------
    best_model : QuantumClassifier
        Model with the least loss. It is returned as an untrained model,
        starting from its initial weights distribution.
    """

    best_loss = 1e10
    best_model = None

    pbar = tqdm(range(samples))

    # Samples loop
    for _ in pbar:

        # Initialize the model parameters and save a copy before optimization
        model.reset_parameters()
        model_untrained = copy.deepcopy(model)

        # Initialize a default optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Trainining
        result = train(model, optimizer, data, epochs=epochs,
                       batch_size=batch_size, show_progress=False)

        # If the training loss is better, save a copy of this model
        # (before optimization)
        loss = result['Train loss'][-1]

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model_untrained)

        pbar.set_postfix({'Lowest loss': best_loss})

    return best_model


def train(model, optimizer, data_train, data_test=None, epochs=500,
          batch_size=20, patience=0, show_progress=True):

    """Train the model with the given optimizer. If a test set is given
    the test loss and accuracy are evaluated after each epoch. If a patience
    greater than zero is given, the model will stop when the solution is not
    improving for at least 'patience' epochs (according to the test loss, if
    the test set is provided, otherwise according to the training loss).

    Parameters
    ----------
    model : QuantumClassifier
        Model to train.
    optimizer : Optimizer
        Optimizer to use for the training (es. SGD, Adam, etc.).
    data_train : tuple
        Training set (X_train,Y_train).
    data_test : tuple, optional
        Test set (X_test,Y_test), by default None.
    epochs : int, optional
        Epochs for training, by default 5.
    batch_size : int, optional
        Batch size, by default 20.
    patience : int, optional
        Patience for the early stopping, by default 0.
    show_progress : bool, optional
        Wheter to show the progress bar during training, by default True.

    Returns
    -------
    results: Dict
        Dictionary containing the history of the training/test loss and
        accuracy during the training process. The keys are:
        'Train loss', 'Test loss', 'Train accuracy', 'Test accuracy'.
    """

    X_train, Y_train = data_train
    N_train = X_train.size(0)

    # Check if we need to work on the test set
    if data_test is not None:
        do_test = True
        X_test, Y_test = data_test
        N_test = X_test.size(0)
    else:
        do_test = False

    batch_num = N_train//batch_size
    device = model.device

    # Initialize the progress bar if needed
    if show_progress:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)

    # Initialize the array that will collect the training/test results history
    loss_train_history = np.zeros(epochs)
    loss_test_history = np.zeros(epochs)
    accuracy_train_history = np.zeros(epochs)
    accuracy_test_history = np.zeros(epochs)

    # Check if we need to do early stopping
    if patience > 0:
        best_loss = 1e20
        epochs_no_improvement = 0
        best_model = None

    # Epochs loop
    for e in pbar:

        # Random permutation of the training set indices
        inds = torch.randperm(N_train)

        loss_epoch = 0
        accuracy_epoch = 0

        # Batch loop
        for i in range(batch_num):

            # Get batch indices
            batch_inds = inds[(i*batch_size):((i+1)*batch_size)]

            # Reset gradients
            optimizer.zero_grad()

            # Load batch data and, if needed, move it on the GPU
            x = X_train[batch_inds, :]
            y = Y_train[batch_inds]

            x = x.to(device)
            y = y.to(device)

            # Do a forward pass
            rho = model(x)

            # Evaluate the loss and propagate the gradients backward
            loss = model.loss(rho, y)
            loss.backward()

            # Do a step of optimization
            optimizer.step()

            # Predict the labels
            y_pred = model.predict(x)

            # Keep track of the loss/accuracy for this epoch
            loss_epoch += loss.item()
            accuracy_epoch += torch.sum(y == y_pred)

        # Save statistics
        loss_train_history[e] = loss_epoch / batch_num
        accuracy_train_history[e] = accuracy_epoch / (batch_size * batch_num)

        # If needed, calculate loss and accuracy on test set
        if do_test:

            Y_pred = torch.zeros(N_test)
            test_loss = 0

            for n in range(N_test//batch_size):

                x = X_test[(n * batch_size):((n+1)*batch_size), :]
                y = Y_test[(n * batch_size):((n+1)*batch_size)]

                Y_pred[(n * batch_size):((n+1)*batch_size)] = model.predict(x)

                rho = model(x)
                test_loss += model.loss(rho, y)

            loss_test_history[e] = test_loss / (N_test//batch_size)
            accuracy_test_history[e] = torch.sum(Y_pred == Y_test) / N_test

        # If early stopping is on, check if we need to stop the training
        if patience > 0:

            if do_test:
                loss = loss_test_history[e]
            else:
                loss = loss_train_history[e]

            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(model)
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

                # Patience has run out
                if epochs_no_improvement >= patience:

                    # Get the best model until now
                    model = best_model

                    # Cut the history array up to the current epoch
                    loss_train_history = loss_train_history[0:e]
                    loss_test_history = loss_test_history[0:e]
                    accuracy_train_history = accuracy_train_history[0:e]
                    accuracy_test_history = accuracy_test_history[0:e]

                    break

        # If there is a progress bar, show the statistics of the last epoch
        if show_progress:
            stats = dict()
            stats['Train loss'] = loss_train_history[e]
            stats['Train accuracy'] = accuracy_train_history[e]

            if do_test:
                stats['Test loss'] = loss_test_history[e]
                stats['Test accuracy'] = accuracy_test_history[e]

            pbar.set_postfix(stats)

    return {
            'Train loss': loss_train_history,
            'Test loss': loss_test_history,
            'Train accuracy':  accuracy_train_history,
            'Test accuracy': accuracy_test_history
            }


def test(model, data, batch_size=20):
    """Test the model and return the accuracy.

    Parameters
    ----------
    model : QuantumClassifier
        Model to test.
    data : Tuple
        Dataset (X,Y).
    batch_size : int, optional
        Batch size, by default 20.

    Returns
    -------
    accuracy: float
        Accuracy of the predictions.
    """
    X = data[0]
    Y_true = data[1]
    Y_pred = classify(model, X, batch_size=batch_size)
    Y_true = Y_true.to(model.device)

    return torch.sum(Y_pred == Y_true) / X.size(0)


def classify(model, X, batch_size=20):
    """Predict the label for each element in the set X.

    Parameters
    ----------
    model : QuantumClassifier
        Model to use for the predictions.
    X : Tensor
        Input data of shape (N,D).
    batch_size : int, optional
        Batch size, by default 20.

    Returns
    -------
    Y_pred : Tensor
        Array of label predictions of shape (N,1).
    """
    device = model.device
    N_test = X.size(0)

    Y_pred = torch.zeros(N_test, dtype=torch.int32, device=device)

    for n in range(N_test//batch_size):

        x = X[(n * batch_size):((n+1)*batch_size), :]
        x = x.to(device)
        Y_pred[(n * batch_size):((n+1)*batch_size)] = model.predict(x)

    return Y_pred
