# This file contains the functions used to train and evaluate the LGGNet model.
from torch.utils.data import DataLoader, random_split
from networks import LGGNet
from networks_explore import LGGNetnew
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import netfunctions as nf
import os
import torch.optim.lr_scheduler as lr_scheduler

def dataset_to_loader(dataset, splits= [0.8, 0.15,0.05], batch_size = 10, manual_seed=42):
    """
    This function takes a dataset and splits it into training, validation, and testing sets.
    :param dataset: The dataset to be split
    :param splits: A list of floats that represent the proportions of the dataset to be used for training, validation, and testing
    :param shuffle: A boolean that determines whether the dataset will be shuffled before splitting
    :param manual_seed: An integer that is used to set the seed for the random number generator

    :return: train_loader, val_loader, test_loader

    """

    train_size = int(splits[0] * len(dataset))  # 75% of the dataset size
    val_size =  int(splits[1] * len(dataset))   # 15% for validation
    test_size =  int(splits[2] * len(dataset))  # 10% for testing

    # Splitting the dataset
    generator1 = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size], generator=generator1)

    # Creating DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader

def initialize_lggnet( raw_combined,lggnet_params=None,optimize_params=None,graph_general=None,cuda=True,threeD=False,temporal_windows = [0.15, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.056 , 0.3515]):
    """
    This function initializes the LGGNet model.

    :param raw_combined: The combined raw data
    :param lggnet_params: The parameters of the LGGNet model
    :param graph_general: The general graph definition for DEAP
    :param cude: A boolean that determines whether CUDA will be used


    :return: LGG: The initialized LGGNet model

    """
    torch.manual_seed(42)
    graph_idx = graph_general   # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(raw_combined.ch_names.index(chan))
    if threeD:
        inputsize = (3, int(lggnet_params['number_of_channels']/3) , lggnet_params['input_length'] )
    else:
        inputsize = (1, lggnet_params['number_of_channels'], lggnet_params['input_length'])
    LGG = LGGNet(
        num_classes=lggnet_params['num_classes'],
        input_size=inputsize,
        sampling_rate=raw_combined.info['sfreq'],
        num_T=optimize_params['num_T'],  # num_T controls the number of temporal filters
        out_graph=optimize_params['out_graph'],  # out_graph controls the number of graph filters
        pool=optimize_params['pool'],
        pool_step_rate=optimize_params['pool_step_rate'],
        idx_graph=num_chan_local_graph,
        dropout_rate=optimize_params['dropout_rate'],
        window = temporal_windows
        )
    
    if cuda:
        LGG = LGG.to('cuda')
    else:
        LGG = LGG.to('cpu')

    return LGG

def initialize_lggnet_unity( chnames,lggnet_params=None,optimize_params=None,graph_general=None,cuda=True,threeD=False,temporal_windows = [0.15, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.056 , 0.3515]):
    """
    This function initializes the LGGNet model.

    :param raw_combined: The combined raw data
    :param lggnet_params: The parameters of the LGGNet model
    :param graph_general: The general graph definition for DEAP
    :param cude: A boolean that determines whether CUDA will be used


    :return: LGG: The initialized LGGNet model

    """
    torch.manual_seed(42)
    graph_idx = graph_general   # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(chnames.index(chan))
    if threeD:
        inputsize = (3, int(lggnet_params['number_of_channels']/3) , lggnet_params['input_length'] )
        LGG = LGGNetnew(
        num_classes=lggnet_params['num_classes'],
        input_size=inputsize,
        sampling_rate= 256,
        num_T=optimize_params['num_T'],  # num_T controls the number of temporal filters
        out_graph=optimize_params['out_graph'],  # out_graph controls the number of graph filters
        pool=optimize_params['pool'],
        pool_step_rate=optimize_params['pool_step_rate'],
        idx_graph=num_chan_local_graph,
        dropout_rate=optimize_params['dropout_rate'],
        window= temporal_windows
        )
    else:
        inputsize = (1, lggnet_params['number_of_channels'], lggnet_params['input_length'])
        LGG = LGGNet(
        num_classes=lggnet_params['num_classes'],
        input_size=inputsize,
        sampling_rate= 256,
        num_T=optimize_params['num_T'],  # num_T controls the number of temporal filters
        out_graph=optimize_params['out_graph'],  # out_graph controls the number of graph filters
        pool=optimize_params['pool'],
        pool_step_rate=optimize_params['pool_step_rate'],
        idx_graph=num_chan_local_graph,
        dropout_rate=optimize_params['dropout_rate'],
        window= temporal_windows
        )
    
    
    
    if cuda:
        LGG = LGG.to('cuda')
    else:
        LGG = LGG.to('cpu')

    return LGG

def initialize_lggnetnew( raw_combined,lggnet_params=None,optimize_params=None,graph_general=None,cuda=True,threeD=False,temporal_windows = [0.15, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.056 , 0.3515]):
    """
    This function initializes the LGGNet model.

    :param raw_combined: The combined raw data
    :param lggnet_params: The parameters of the LGGNet model
    :param graph_general: The general graph definition for DEAP
    :param cude: A boolean that determines whether CUDA will be used


    :return: LGG: The initialized LGGNet model

    """
    torch.manual_seed(42)
    graph_idx = graph_general   # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(raw_combined.ch_names.index(chan))
    if threeD:
        inputsize = (3, int(lggnet_params['number_of_channels']/3) , lggnet_params['input_length'] )
    else:
        inputsize = (1, lggnet_params['number_of_channels'], lggnet_params['input_length'])

    LGG = LGGNetnew(
        num_classes=lggnet_params['num_classes'],
        input_size=inputsize,
        sampling_rate=raw_combined.info['sfreq'],
        num_T=optimize_params['num_T'],  # num_T controls the number of temporal filters
        out_graph=optimize_params['out_graph'],  # out_graph controls the number of graph filters
        pool=optimize_params['pool'],
        pool_step_rate=optimize_params['pool_step_rate'],
        idx_graph=num_chan_local_graph,
        dropout_rate=optimize_params['dropout_rate'],
        window= temporal_windows
        )
    
    if cuda:
        LGG = LGG.to('cuda')
    else:
        LGG = LGG.to('cpu')

    return LGG

def run_model(LGG=LGGNet,train_dataset=None,test_loader=None, criterion=None, optimizer=None, lggnet_params=None,ifplot=False,iftest=False,kfolds=5,ifresult=False,threeD=False):
    """
    This function trains and evaluates the LGGNet model.
    :param LGG: The LGGNet model
    :param train_loader: The DataLoader for the training set
    :param val_loader: The DataLoader for the validation set
    :param test_loader: The DataLoader for the testing set
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param lggnet_params: The parameters of the LGGNet model
    :param ifplot: A boolean that determines whether the training and validation loss and accuracy will be plotted
    :param iftest: A boolean that determines whether the model will be tested
    :param kfolds: The number of folds for k-fold cross validation
    :param ifresult: A boolean that determines whether the results will be printed
    
    :return: train_losses, val_losses, train_acc, val_acc, test_loss, test_acc


    """
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    test_loss, test_acc = 0, 0

    """
    min_val = float('inf')
    max_val = float('-inf')
    for data, _ in train_loader:
        min_batch = torch.min(data)
        max_batch = torch.max(data)
        min_val = min(min_val, min_batch.item())
        max_val = max(max_val, max_batch.item())
    
    def min_max_normalize(tensor, min_val, max_val):
        return (tensor - min_val) / (max_val - min_val)
    """

    results = {}

    
    if kfolds == 0:
        kfold = KFold(n_splits = 9 ,  shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        
        train_subsampler = Subset(train_dataset, train_ids)
        val_subsampler = Subset(train_dataset, test_ids)
        train_loader = DataLoader(train_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=lggnet_params['batch_size'], shuffle=False)

        early_stopping = EarlyStoppingVal(patience=5, verbose=True)
        # plt.ion()
        # Training the model
        for epoch in range(lggnet_params['num_epochs']):

            LGG.train()
            pred_train = []
            act_train = []
            
            correct = 0
            train_loss = 0
            total = 0
            for data, label in train_loader:
                if threeD:
                    split_data = torch.split(data, 1, dim=1)
                    normalized_splits = []
                    for tensor in split_data:
                        normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                        normalized_splits.append(normalized_tensor)
                    data = torch.cat(normalized_splits, dim=1)
                    data = torch.nn.functional.normalize(data,p = 1.0,dim =3)
                    data = data.repeat(1, 1, 3, 1, 1)
                else:
                    data = torch.nn.functional.normalize(data,p = 1.0,dim =2)

                # data = min_max_normalize(data, min_val, max_val)
                label = label.long()
                outputs = LGG(data)
                loss = criterion(outputs, label)
                _, pred = torch.max(outputs, 1)
                pred_train.extend(pred.data.tolist())
                act_train.extend(label.data.tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += label.size(0)
                correct += (pred == label).sum().item()
                #print(f'Loss: {loss.item()}')
            #print(f'Epoch {epoch}, Train Loss: {train_loss / total}, Accuracy: {correct / total * 100}%')   
            train_losses.append(train_loss / total)
            train_acc.append(correct / total * 100)

            # Validation step
            LGG.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for input, label in val_loader:
                    #data = min_max_normalize(data, min_val, max_val)
                    if threeD:
                        split_input = torch.split(input, 1, dim=1)
                        normalized_splits = []
                        for tensor in split_input:
                            normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                            normalized_splits.append(normalized_tensor)
                        input = torch.cat(normalized_splits, dim=1)
                        input = torch.nn.functional.normalize(input,p = 1.0,dim =3)
                        input = input.repeat(1, 1, 3, 1, 1)
                    else:
                        input = torch.nn.functional.normalize(input,p = 1.0,dim =2)

                    label = label.long()
                    outputs = LGG(input)
                    val_loss += criterion(outputs, label).sum().item()
                    _, predicted = torch.max(outputs, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            early_stopping(val_loss, LGG, optimizer)
            print(f'Epoch {epoch}, Validation Loss: {val_loss / total}, Accuracy: {correct / total * 100}%')
            val_losses.append(val_loss / total)
            val_acc.append(correct / total * 100)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        results[fold] = {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_acc, 'val_acc': val_acc}
        if ifplot:
            # Plotting after training
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()

            plt.plot(train_acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy %')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.show()
        if kfolds == 0 and fold == 0:
            break
    if ifresult:
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {kfolds} FOLDS')
        print('--------------------------------')
        avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc = 0.0, 0.0, 0.0, 0.0
        for key, value in results.items():
            print(f'Fold {key}: Train Loss: {value["train_loss"][-1]}, Val Loss: {value["val_loss"][-1]}, Train Acc: {value["train_acc"][-1]}%, Val Acc: {value["val_acc"][-1]}%')
            avg_train_loss += value["train_loss"][-1]
            avg_val_loss += value["val_loss"][-1]
            avg_train_acc += value["train_acc"][-1]
            avg_val_acc += value["val_acc"][-1]

        num_folds = len(results.items())
        print(f'Average Train Loss: {avg_train_loss / num_folds}')
        print(f'Average Val Loss: {avg_val_loss / num_folds}')
        print(f'Average Train Acc: {avg_train_acc / num_folds}%')
        print(f'Average Val Acc: {avg_val_acc / num_folds}%')
        print('--------------------------------')
    
    if iftest:
        # Test step
        LGG.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for input, label in test_loader:
                if threeD:
                    split_input = torch.split(input, 1, dim=1)
                    normalized_splits = []
                    for tensor in split_input:
                        normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                        normalized_splits.append(normalized_tensor)
                    input = torch.cat(normalized_splits, dim=1)
                    input = torch.nn.functional.normalize(input,p = 1.0,dim =3)
                    input = input.repeat(1, 1, 3, 1, 1)
                else:
                    input = torch.nn.functional.normalize(input,p = 1.0,dim =2)

                label = label.long()
                outputs = LGG(input)
                test_loss += criterion(outputs, label).sum().item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            test_acc = correct / total * 100
            print(f'Test Loss: {test_loss / total}, Accuracy: {test_acc}%')
        

    return train_losses, val_losses, train_acc, val_acc, test_loss/total, test_acc


def run_model_new(raw_combined=None,train_dataset=None,test_loader=None,optimize_params=None, lggnet_params=None,ifplot=False,iftest=False,kfolds=5,ifresult=False,threeD=False,window =[0.15, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.056 , 0.3515], graph_general=None,run_number=int,model_directory=None,):
    """
    This function trains and evaluates the LGGNet model.
    :param LGG: The LGGNet model
    :param train_loader: The DataLoader for the training set
    :param val_loader: The DataLoader for the validation set
    :param test_loader: The DataLoader for the testing set
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param lggnet_params: The parameters of the LGGNet model
    :param ifplot: A boolean that determines whether the training and validation loss and accuracy will be plotted
    :param iftest: A boolean that determines whether the model will be tested
    :param kfolds: The number of folds for k-fold cross validation
    :param ifresult: A boolean that determines whether the results will be printed
    
    :return: train_losses, val_losses, train_acc, val_acc, test_loss, test_acc
    """

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    test_loss, test_acc = 0, 0
    torch.manual_seed(42)
    results = {}
    
    if kfolds == 0:
        kfold = KFold(n_splits = 9 ,  shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        # window =  [1, 0.50, 0.125, 0.6667, 0.25, 0.40, 0.19, 1.5] #[0.4994, 0.7819, 0.6219, 0.2443, 1.0, 0.1957, 0.7723, 0.7708] #[0.8966, 0.6767, 1.4293, 1.382, 0.416, 1.4833, 0.0791, 1.4945] # 
        if threeD:
            LGG = nf.initialize_lggnetnew(raw_combined, lggnet_params=lggnet_params,optimize_params=optimize_params, graph_general=graph_general,cuda=True,threeD=threeD,temporal_windows=window)
        else:
            LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params=optimize_params, graph_general=graph_general,cuda=True,threeD=False,temporal_windows=window)
      
        manual_seed = 42
        # # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss() #LabelSmoothing(smoothing=0.1) #
        optimizer = torch.optim.Adam(LGG.parameters(), lr=optimize_params['learning_rate']) 
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        train_subsampler = Subset(train_dataset, train_ids)
        val_subsampler = Subset(train_dataset, test_ids)
        train_loader = DataLoader(train_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True,generator=torch.Generator().manual_seed(manual_seed))
        val_loader = DataLoader(val_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True,generator=torch.Generator().manual_seed(manual_seed))

        early_stopping = EarlyStoppingAcc(patience=10, verbose=True)
        # plt.ion()
        # Training the model
        for epoch in range(lggnet_params['num_epochs']):

            LGG.train()
            pred_train = []
            act_train = []
            
            correct = 0
            train_loss = 0
            total = 0
            for data, label in train_loader:
                if threeD:
                    split_data = torch.split(data, 1, dim=1)
                    normalized_splits = []
                    for tensor in split_data:
                        normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                        normalized_splits.append(normalized_tensor)
                    data = torch.cat(normalized_splits, dim=1)
                    data = torch.nn.functional.normalize(data,p = 1.0,dim =3)
                    data = data.repeat(1, 1, 3, 1, 1)
                else:
                    data = torch.nn.functional.normalize(data,p = 1.0,dim =2)

                # data = min_max_normalize(data, min_val, max_val)
                label = label.long()
                outputs = LGG(data)
                loss = criterion(outputs, label)
                _, pred = torch.max(outputs, 1)
                pred_train.extend(pred.data.tolist())
                act_train.extend(label.data.tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += label.size(0)
                correct += (pred == label).sum().item()
                #print(f'Loss: {loss.item()}')
            #print(f'Epoch {epoch}, Train Loss: {train_loss / total}, Accuracy: {correct / total * 100}%')   
            train_losses.append(train_loss / total)
            train_acc.append(correct / total * 100)
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

            # Validation step
            LGG.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for input, label in val_loader:
                    #data = min_max_normalize(data, min_val, max_val)
                    if threeD:
                        split_input = torch.split(input, 1, dim=1)
                        normalized_splits = []
                        for tensor in split_input:
                            normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                            normalized_splits.append(normalized_tensor)
                        input = torch.cat(normalized_splits, dim=1)
                        input = torch.nn.functional.normalize(input,p = 1.0,dim =3)
                        input = input.repeat(1, 1, 3, 1, 1)
                    else:
                        input = torch.nn.functional.normalize(input,p = 1.0,dim =2)

                    label = label.long()
                    outputs = LGG(input)
                    val_loss += criterion(outputs, label).sum().item()
                    _, predicted = torch.max(outputs, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            early_stopping(correct/total, LGG, optimizer, run_number, model_directory)
            print(f'Epoch {epoch}, Validation Loss: {val_loss / total}, Accuracy: {correct / total * 100}%')
            val_losses.append(val_loss / total)
            val_acc.append(correct / total * 100)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        results[fold] = {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_acc, 'val_acc': val_acc}
        if ifplot:
            # Plotting after training
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()

            plt.plot(train_acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy %')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.show()
        if kfolds == 0 and fold == 0:
            break
    if ifresult:
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {kfolds} FOLDS')
        print('--------------------------------')
        avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc = 0.0, 0.0, 0.0, 0.0
        for key, value in results.items():
            print(f'Fold {key}: Train Loss: {value["train_loss"][-1]}, Val Loss: {value["val_loss"][-1]}, Train Acc: {value["train_acc"][-1]}%, Val Acc: {value["val_acc"][-1]}%')
            avg_train_loss += value["train_loss"][-1]
            avg_val_loss += value["val_loss"][-1]
            avg_train_acc += value["train_acc"][-1]
            avg_val_acc += value["val_acc"][-1]

        num_folds = len(results.items())
        print(f'Average Train Loss: {avg_train_loss / num_folds}')
        print(f'Average Val Loss: {avg_val_loss / num_folds}')
        print(f'Average Train Acc: {avg_train_acc / num_folds}%')
        print(f'Average Val Acc: {avg_val_acc / num_folds}%')
        print('--------------------------------')
    
    if iftest:
        # Test step
        LGG = load_best_modelAcc(LGG, optimizer,run_number=run_number, model_directory= model_directory,test=False).to('cuda' if torch.cuda.is_available() else 'cpu')
        LGG = train_on_all(LGG, train_dataset, criterion, optimizer, lggnet_params, ifplot=False, iftest=False, threeD=threeD,run_number=run_number, model_directory= model_directory)
        LGG = load_best_modelAcc(LGG, optimizer,run_number=run_number, model_directory= model_directory,test=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        LGG.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for input, label in test_loader:
                if threeD:
                    split_input = torch.split(input, 1, dim=1)
                    normalized_splits = []
                    for tensor in split_input:
                        normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                        normalized_splits.append(normalized_tensor)
                    input = torch.cat(normalized_splits, dim=1)
                    input = torch.nn.functional.normalize(input,p = 1.0,dim =3)
                    input = input.repeat(1, 1, 3, 1, 1)
                else:
                    input = torch.nn.functional.normalize(input,p = 1.0,dim =2)

                label = label.long()
                outputs = LGG(input)
                test_loss += criterion(outputs, label).sum().item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            test_acc = correct / total * 100
            print(f'Test Loss: {test_loss / total}, Accuracy: {test_acc}%')
        

    return train_losses, val_losses, train_acc, val_acc, test_loss/total, test_acc

def train_on_all(LGG=LGGNet,train_dataset=None, criterion=None, optimizer=None, lggnet_params=None,ifplot=False,iftest=False,threeD=False,run_number=int,model_directory=None):
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        torch.manual_seed(42)
        kfold = KFold(n_splits = 30 ,  shuffle=True, random_state=42)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
            
            train_subsampler = Subset(train_dataset, train_ids)
            val_subsampler = Subset(train_dataset, test_ids)
            train_loader = DataLoader(train_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=lggnet_params['batch_size'], shuffle=False)
            early_stopping = EarlyStoppingAcc(patience=10, verbose=True)
            # Training the model
            for epoch in range(lggnet_params['num_epochs']):

                LGG.train()
                pred_train = []
                act_train = []
                
                correct = 0
                train_loss = 0
                total = 0
                for data, label in train_loader:
                    if threeD:
                        split_data = torch.split(data, 1, dim=1)
                        normalized_splits = []
                        for tensor in split_data:
                            normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                            normalized_splits.append(normalized_tensor)
                        data = torch.cat(normalized_splits, dim=1)
                        data = torch.nn.functional.normalize(data,p = 1.0,dim =3)
                        data = data.repeat(1, 1, 3, 1, 1)
                    else:
                        data = torch.nn.functional.normalize(data,p = 1.0,dim =2)

                    # data = min_max_normalize(data, min_val, max_val)
                    label = label.long()
                    outputs = LGG(data)
                    loss = criterion(outputs, label)
                    _, pred = torch.max(outputs, 1)
                    pred_train.extend(pred.data.tolist())
                    act_train.extend(label.data.tolist())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    total += label.size(0)
                    correct += (pred == label).sum().item()
                    #print(f'Loss: {loss.item()}')
                #print(f'Epoch {epoch}, Train Loss: {train_loss / total}, Accuracy: {correct / total * 100}%')   
                train_losses.append(train_loss / total)
                train_acc.append(correct / total * 100)

                # Validation step
                LGG.eval()
                with torch.no_grad():
                    val_loss = 0
                    correct = 0
                    total = 0
                    for input, label in val_loader:
                        #data = min_max_normalize(data, min_val, max_val)
                        if threeD:
                            split_input = torch.split(input, 1, dim=1)
                            normalized_splits = []
                            for tensor in split_input:
                                normalized_tensor = torch.nn.functional.normalize(tensor, p=1.0, dim=3)
                                normalized_splits.append(normalized_tensor)
                            input = torch.cat(normalized_splits, dim=1)
                            input = torch.nn.functional.normalize(input,p = 1.0,dim =3)
                            input = input.repeat(1, 1, 3, 1, 1)
                        else:
                            input = torch.nn.functional.normalize(input,p = 1.0,dim =2)

                        label = label.long()
                        outputs = LGG(input)
                        val_loss += criterion(outputs, label).sum().item()
                        _, predicted = torch.max(outputs, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                early_stopping(correct/ total, LGG, optimizer,run_number=run_number, model_directory=model_directory,test=True)
                print(f'Epoch {epoch}, Validation Loss: {val_loss / total}, Accuracy: {correct / total * 100}%')
                val_losses.append(val_loss / total)
                val_acc.append(correct / total * 100)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            if fold == 0:
                break
        return LGG

class LabelSmoothing(torch.nn.Module):
        """NLL loss with label smoothing.
        refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
        """
        def __init__(self, smoothing=0.0):
            """Constructor for the LabelSmoothing module.
            :param smoothing: label smoothing factor
            """
            super(LabelSmoothing, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing

        def forward(self, x, target):
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()

class EarlyStoppingVal:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, run_number,model_directory='save_to_path',test=False):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer,run_number, model_directory=model_directory, test=test)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer,run_number, model_directory=model_directory, test=test)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer,run_number,model_directory='save_to_path', test=False):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint_model.pth')  # Save the model

        # Check if the directory exists, and create it if it does not
        if not os.path.exists(os.path.join(model_directory, str(run_number))):
            os.makedirs(os.path.join(model_directory, str(run_number)))
        if test:
            if not os.path.exists(os.path.join(model_directory, str(run_number),"train_96")):
                os.makedirs(os.path.join(model_directory, str(run_number),"train_96"))
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss}, 
                    f"{model_directory}/{run_number}/train_96/model_{val_loss:.4f}.pth")
        else:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss}, 
                        f"{model_directory}/{run_number}/model_{val_loss:.4f}.pth")

        self.val_loss_min = val_loss


def load_best_modelVal(model, optimizer,run_number, model_directory='save_to_path',test=False):
    if test:
        model_files = [f for f in os.listdir(os.path.join(model_directory, str(run_number),"train_96")) if f.endswith('.pth')]
    else:
        model_files = [f for f in os.listdir(os.path.join(model_directory, str(run_number))) if f.endswith('.pth')]

    # Finding the file with the lowest validation loss
    lowest_loss = float('inf')
    best_model_file = None

    for filename in model_files:
        # Extract loss from filename assuming format "model_{val_loss}.pth"
        loss_value = float(filename.split('_')[1].split('.pth')[0])
        if loss_value < lowest_loss:
            lowest_loss = loss_value
            best_model_file = filename

    # Load the model with the lowest validation loss
    if best_model_file:
        if test:
            best_model_path = os.path.join(model_directory,str(run_number),"train_96", best_model_file)
        else:
            best_model_path = os.path.join(model_directory,str(run_number), best_model_file)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model with loss: {lowest_loss}")
    else:
        print("No model files found.")
    return model


import statistics

def print_results(results_all, kfolds):
    """
    This function prints the results of the k-fold nested cross validation.
    """
    avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, ave_test_loss, avg_test_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    test_acc_values = []  # List to store test accuracy values for standard deviation calculation

    print('----------------------------------------------------------------')
    print(f'K-FOLD NESTED CROSS VALIDATION RESULTS FOR {kfolds} FOLDS')
    print('----------------------------------------------------------------')

    for key, value in results_all.items():
        print(f'Fold {key}: Train Loss: {value["train_loss"][-1]}, Val Loss: {value["val_loss"][-1]}, Train Acc: {value["train_acc"][-1]}%, Val Acc: {value["val_acc"][-1]}%, Test Loss: {value["test_loss"]}, Test Acc: {value["test_acc"]}')
        avg_train_loss += value["train_loss"][-1]
        avg_val_loss += value["val_loss"][-1]
        avg_train_acc += value["train_acc"][-1]
        avg_val_acc += value["val_acc"][-1]
        ave_test_loss += value["test_loss"]
        avg_test_acc += value["test_acc"]
        test_acc_values.append(value["test_acc"])  # Append test accuracy to the list

    num_folds = len(results_all)
    avg_test_acc /= num_folds  # Average test accuracy
    std_dev_test_acc = statistics.stdev(test_acc_values)  # Standard deviation of test accuracy

    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(f'5x5 NCV Average Train Loss: {avg_train_loss / num_folds}')
    print(f'5x5 NCV Average Val Loss: {avg_val_loss / num_folds}')
    print(f'5x5 NCV Average Train Acc: {avg_train_acc / num_folds}%')
    print(f'5x5 NCV Average Val Acc: {avg_val_acc / num_folds}%')
    print(f'5x5 NCV Average Test Loss: {ave_test_loss / num_folds}')
    print('----------------------------------------------------------------')
    print(f'5x5 NCV Average Test Acc: {avg_test_acc}%')
    print(f'5x5 NCV Test Acc Standard Deviation: {std_dev_test_acc}%')
    print('--------------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


class EarlyStoppingAcc:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = float('-inf')
        self.delta = delta

    def __call__(self, val_accuracy, model, optimizer, run_number, model_directory='save_to_path', test=False):

        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, run_number, model_directory=model_directory, test=test)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, run_number, model_directory=model_directory, test=test)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model, optimizer, run_number, model_directory='save_to_path', test=False):
        '''Saves model when validation accuracy increase.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_accuracy:.6f}). Saving model ...')
        
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(os.path.join(model_directory, str(run_number))):
            os.makedirs(os.path.join(model_directory, str(run_number)))
        if test:
            if not os.path.exists(os.path.join(model_directory, str(run_number),"train_96")):
                os.makedirs(os.path.join(model_directory, str(run_number),"train_96"))
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy}, 
                    f"{model_directory}/{run_number}/train_96/model_{val_accuracy:.4f}.pth")
        else:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': val_accuracy}, 
                        f"{model_directory}/{run_number}/model_{val_accuracy:.4f}.pth")

        self.val_acc_max = val_accuracy


def load_best_modelAcc(model, optimizer, run_number, model_directory='save_to_path', test=False):
    if test:
        model_files = [f for f in os.listdir(os.path.join(model_directory, str(run_number), "train_96")) if f.endswith('.pth')]
    else:
        model_files = [f for f in os.listdir(os.path.join(model_directory, str(run_number))) if f.endswith('.pth')]

    # Finding the file with the highest validation accuracy
    highest_accuracy = float('-inf')  # Initialize with negative infinity
    best_model_file = None

    for filename in model_files:
        # Extract accuracy from filename assuming format "model_{val_accuracy}.pth"
        accuracy_value = float(filename.split('_')[1].split('.pth')[0])
        if accuracy_value > highest_accuracy:
            highest_accuracy = accuracy_value
            best_model_file = filename

    # Load the model with the highest validation accuracy
    if best_model_file:
        if test:
            best_model_path = os.path.join(model_directory, str(run_number), "train_96", best_model_file)
        else:
            best_model_path = os.path.join(model_directory, str(run_number), best_model_file)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model with accuracy: {highest_accuracy}")
    else:
        print("No model files found.")
    return model
