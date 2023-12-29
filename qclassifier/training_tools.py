import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import copy


def find_best_initial_conditions(model, data, epochs = 5, batch_size = 20, samples = 20, lr = 1e-2):

    # Inizializzo casualmente il modello per "samples" volte e per ognuna faccio
    # un training veloce Poi uso l'accuratezza per scegliere il modello
    # migliore. Attualmente non faccio specificare all'utente quale
    # ottimizzatore preferisce, perché sarebbe difficile poi reinizializzarne i
    # parametri per ogni sample. Voglio salvare il modello migliore PRIMA della
    # ottimizzazione e non dopo, in modo da poter ripetere il training dalla
    # prima epoca

    best_loss = 1e10
    best_model = None

    pbar = tqdm(range(samples))

    for n in pbar:

        # torch.ma  nual_seed(n)

        # Reinizializzo il modello
        model.reset_parameters()

        model_untrained = copy.deepcopy(model)

        # Ricreo l'ottimizzatore
        optimizer = torch.optim.Adam(model.parameters(),lr = lr)
        
        result = train(model, optimizer, data, data, epochs = epochs, batch_size = batch_size, test_accuracy = False, show_progress = False)
        loss = result['Train loss'][-1]
        
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model_untrained)

        pbar.set_postfix({'Lowest loss' : best_loss})

    return best_model


def train(model, optimizer, data_train, data_test = None, epochs = 500, batch_size = 20, patience = 0, show_progress = True):
    
    X_train, Y_train = data_train
    N_train = X_train.size(0)

    if data_test is not None:
        do_test = True
        X_test, Y_test = data_test
        N_test = X_test.size(0)
    else:
        do_test = False

    batch_num = N_train//batch_size
    device = model.device

    # Barra di caricamento
    if show_progress: pbar = tqdm(range(epochs))
    else:             pbar = range(epochs)

    # Statistiche sull'andamento del training
    loss_train_history = np.zeros(epochs)
    accuracy_train_history = np.zeros(epochs)
    
    if do_test:
        loss_test_history = np.zeros(epochs)
        accuracy_test_history = np.zeros(epochs)

    # Parametri per l'early stopping
    if patience > 0:
        best_loss = 1e20
        epochs_no_improvement = 0
        best_model = None

    # Fisso il seed per come verranno mischiati i batch
    # torch.manual_seed(0)

    for e in pbar:

        # Permuto gli indici in modo da non usare sempre gli stessi batch
        inds = torch.randperm(N_train)
        
        loss_epoch = 0
        accuracy_epoch = 0

        # Ciclo sui batch
        for i in range(batch_num):

            # Indici di questo batch
            batch_inds = inds[(i*batch_size):((i+1)*batch_size)]

            # Resetto i gradienti dell'ottimizzatore
            optimizer.zero_grad()

            # Carico il batch
            x = X_train[batch_inds,:]
            y = Y_train[batch_inds]

            # Se necessario, sposto i tensori su GPU
            x = x.to(device)
            y = y.to(device)

            # Applico tutti gli operatori e ottengo lo stato finale
            rho = model(x)

            # Valuto la loss e la sommo a quella del batch
            loss = model.loss(rho, y)

            # Propago indietro i gradienti
            loss.backward()
        
            # Faccio uno step di ottimizzazione
            optimizer.step()

            # Faccio una previsione per stimare l'accuratezza durante il training
            y_pred = model.predict(x)

            # Salvo le informazioni di questa epoca
            loss_epoch += loss.item()
            accuracy_epoch += torch.sum(y == y_pred)

        # Salvo le statistiche sul training set
        loss_train_history[e] = loss_epoch / batch_num
        accuracy_train_history[e] = accuracy_epoch / (batch_size * batch_num)


        # Se necessario, valuto le performance sul test set
        if do_test:
            
            Y_pred = torch.zeros(N_test)
            test_loss = 0

            for n in range(N_test//batch_size):

                x = X_test[(n * batch_size):((n+1)*batch_size),:]
                y = Y_test[(n * batch_size):((n+1)*batch_size)]

                # Prevedo le label
                Y_pred[(n * batch_size):((n+1)*batch_size)] = model.predict(x)

                # Calcolo la loss
                rho = model(x)
                test_loss += model.loss(rho, y)     

            loss_test_history[e] = test_loss / (N_test//batch_size)
            accuracy_test_history[e] = torch.sum(Y_pred == Y_test) / N_test

        # Scrivo sulla barra le statistiche attuali
        stats = dict()
        stats['Train loss'] = loss_train_history[e]
        stats['Train accuracy'] = accuracy_train_history[e]

        if do_test:
            stats['Test loss'] = loss_test_history[e]
            stats['Test accuracy'] = accuracy_test_history[e]

        # In caso sia abilitato l'early stopping, controllo a che punto sta
        if patience > 0:
            
            if do_test: l = loss_test_history[e]
            else:       l = loss_train_history[e]
          
            if l < best_loss:
                best_loss = l
                best_model = copy.deepcopy(model)
                epochs_no_improvement = 0            
            else:
                # In caso non ci siano miglioramenti aumento il counter
                epochs_no_improvement += 1

                # Se ho finito la pazienza
                if epochs_no_improvement >= patience:

                    # Recupero il miglior modello fino a questo momento
                    model = best_model

                    # Taglio tutte le statistiche in modo che si fermino a quest'epoca
                    loss_train_history  = loss_train_history[0:e]
                    loss_test_history   = loss_test_history[0:e]
                    accuracy_train_history  = accuracy_train_history[0:e]
                    accuracy_test_history   = accuracy_test_history[0:e]

                    # Fermo il training
                    break

        if show_progress: pbar.set_postfix(stats)


    return {'Train loss' : loss_train_history, 'Test loss' : loss_test_history, 
            'Train accuracy' :  accuracy_train_history, 'Test accuracy' : accuracy_test_history}


def test(model, data, batch_size = 20):
    
    X = data[0]
    Y_true = data[1]
    Y_pred = classify(model, X, batch_size = batch_size)
    Y_true = Y_true.to(model.device)

    return torch.sum(Y_pred == Y_true) / X.size(0)


def classify(model, X_test, batch_size = 20):

    device = model.device
    N_test = X_test.size(0)

    Y_pred = torch.zeros(N_test, dtype = torch.int32, device = device)

    for n in range(N_test//batch_size):

        x = X_test[(n * batch_size):((n+1)*batch_size),:]
        x = x.to(device)
        Y_pred[(n * batch_size):((n+1)*batch_size)] = model.predict(x)
        
    return Y_pred
