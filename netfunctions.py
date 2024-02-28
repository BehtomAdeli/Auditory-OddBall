# This file contains the functions used to train and evaluate the LGGNet model.
from torch.utils.data import DataLoader, random_split
from networks import LGGNet
import torch
import matplotlib.pyplot as plt

def dataset_to_loader(dataset, splits= [0.8, 0.15,0.05], batch_size = 2, manual_seed=42):
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

def initialize_lggnet( raw_combined,lggnet_params=None,optimize_params=None,graph_general_DEAP=None,cuda=True):
    """
    This function initializes the LGGNet model.

    :param raw_combined: The combined raw data
    :param lggnet_params: The parameters of the LGGNet model
    :param graph_general_DEAP: The general graph definition for DEAP
    :param cude: A boolean that determines whether CUDA will be used


    :return: LGG: The initialized LGGNet model

    """
    
    graph_idx = graph_general_DEAP   # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(raw_combined.ch_names.index(chan))
    
    LGG = LGGNet(
        num_classes=lggnet_params['num_classes'],
        input_size=(1, lggnet_params['number_of_channels'] , lggnet_params['input_length'] ),
        sampling_rate=raw_combined.info['sfreq'],
        num_T=optimize_params['num_T'],  # num_T controls the number of temporal filters
        out_graph=optimize_params['out_graph'],  # out_graph controls the number of graph filters
        pool=optimize_params['pool'],
        pool_step_rate=optimize_params['pool_step_rate'],
        idx_graph=num_chan_local_graph,
        dropout_rate=optimize_params['dropout_rate'],
        )
    if cuda:
        LGG = LGG.to('cuda')
    else:
        LGG = LGG.to('cpu')

    return LGG


def run_model(LGG,dataset, criterion, optimizer,optimize_params, lggnet_params,ifplot=False,iftest=False):
    """
    This function trains the LGGNet model and evaluates it on the test set.
    :param LGG: The LGGNet model
    :param dataset: The dataset
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param lggnet_params: The parameters of the LGGNet model
    
    :return: train_losses, val_losses, train_acc, val_acc, test_loss, test_acc

    """
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    test_loss, test_acc = 0, 0
    train_loader, val_loader, test_loader = dataset_to_loader(dataset,lggnet_params['splits'] , optimize_params['batch_size'], manual_seed=42)

    #Training the model
    for epoch in range(lggnet_params['num_epochs']):
        print(f'Epoch {epoch}')
        LGG.train()
        pred_train = []
        act_train = []
        
        correct = 0
        train_loss = 0
        total = 0
        for data, label in train_loader:
            data = torch.nn.functional.normalize(data,p = 1.0,dim =2)
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
        train_losses.append(train_loss / len(train_loader))
        train_acc.append(correct / total * 100)

        # Validation step
        LGG.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for input, label in val_loader:
                input = torch.nn.functional.normalize(input,p = 1.0,dim =2)
                label = label.long()
                outputs = LGG(input)
                val_loss += criterion(outputs, label).sum().item()
                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print(f'Epoch {epoch}, Validation Loss: {val_loss / total}, Accuracy: {correct / total * 100}%')
            val_losses.append(val_loss / len(val_loader))
            val_acc.append(correct / total * 100)
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
    
    if iftest:
        # Test step
        LGG.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for input, label in test_loader:
                input = torch.nn.functional.normalize(input,p = 1.0,dim =2)
                label = label.long()
                outputs = LGG(input)
                test_loss += criterion(outputs, label).sum().item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            test_acc = correct / total * 100
            print(f'Test Loss: {test_loss / total}, Accuracy: {test_acc}%')
        

    return train_losses, val_losses, train_acc, val_acc, test_loss, test_acc
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