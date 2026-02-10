import mne
import torch
import midfunctions as mf
import netfunctions as nf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

raw_combined = mne.io.read_raw_fif('C:\\Users\\\ data.fif', preload=True)

event_id = dict(deviant40=2,standard1k=1)
mapping={2:'deviant40',1:'standard1k'}
picks = mne.pick_types(raw_combined.info, eeg=True,fnirs=False, stim=False, exclude='bads')
tmin =-0.5
tmax = 1.5

epochs_concat, labels , number_of_channels, input_length, Number_trials_per_class = mf.epoch_data(raw_combined, tmin =tmin, tmax =tmax, event_id =event_id, mapping = mapping, stim=['StimulusCode'] , picks=picks)

# Create LGGNet parameters
lggnet_params = {
                 'num_classes': 2, 
                 'batch_size': 10,
                 'num_epochs': 250,
                 'number_trials_per_class': Number_trials_per_class,
                 'number_of_channels': number_of_channels, 
                 'input_length': input_length, 
                 }
graph_general_eeg = [['AFz', 'F5'], ['F6', 'FC4'], ['AFz', 'FCz','Cz'],
                 ['C6', 'TTP8'],
                 ['C5', 'TTP7'],
                 ['Pz','Fpz']
                 ]

dataset = mf.create_dataset(epochs_concat, labels, lggnet_params)

# Define the graph
original_order = raw_combined.ch_names
original_order.remove('StimulusCode')

graph_general_All = [['S1_D1 hbo','S1_D1 hbr', 'AFz', 'F5'], ['F6','S3_D2 hbo','S3_D2 hbr' , 'FC4'], ['AFz', 'FCz', 'S2_D1 hbo','S2_D1 hbr', 'S2_D2 hbo', 'S2_D2 hbr'],
                     ['C6', 'TTP8', 'S6_D6 hbo', 'S6_D7 hbo','S6_D6 hbr', 'S6_D7 hbr'], ['TPP7', 'S4_D5 hbo', 'S5_D5 hbo','S4_D5 hbr', 'S5_D5 hbr'],
                     ['C5', 'TTP7', 'S4_D3 hbo', 'S4_D4 hbo','S4_D3 hbr', 'S4_D4 hbr'],
                     ['TPP8','S6_D8 hbr', 'S7_D8 hbr','S6_D8 hbo', 'S7_D8 hbo'],
                     ['Pz'],['Fpz'], ['Cz']]


optimize_params = {'learning_rate': 0.001, 'dropout_rate': 0.5, 'num_T': 64, 'out_graph': 32, 'pool': 16, 'pool_step_rate': 0.3}


kfolds=5
manual_seed = 42
results_all = {}
kfold = KFold(n_splits=kfolds, shuffle=True)


for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    train_subsampler = Subset(dataset, train_ids)
    test_subsampler = Subset(dataset, test_ids)
    # train_loader = DataLoader(train_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True)


    LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params=optimize_params, graph_general=graph_general_eeg,cuda=True)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss() #LabelSmoothing(smoothing=0.1) #
    optimizer = torch.optim.Adam(LGG.parameters(), lr=optimize_params['learning_rate'])  

    # Train,Validate, and Test the model
    train_losses, val_losses, train_acc, val_acc,test_loss, test_acc = nf.run_model(LGG,train_subsampler,test_loader, criterion, optimizer, lggnet_params,ifplot=False, iftest=True, kfolds=5, ifresult=True)
    results_all[fold] = {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_acc, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc}

print(f'K-FOLD CROSS VALIDATION RESULTS FOR {kfolds} FOLDS')
print('--------------------------------')
print('--------------------------------')
avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc,ave_test_loss,avg_test_acc =0.0, 0.0, 0.0, 0.0, 0.0, 0.0
for key, value in results_all.items():
    print(f'Fold {key}: Train Loss: {value["train_loss"][-1]}, Val Loss: {value["val_loss"][-1]}, Train Acc: {value["train_acc"][-1]}%, Val Acc: {value["val_acc"][-1]}%, Test Loss: {value["test_loss"]}, Test Acc: {value["test_acc"]}')
    avg_train_loss += value["train_loss"][-1]
    avg_val_loss += value["val_loss"][-1]
    avg_train_acc += value["train_acc"][-1]
    avg_val_acc += value["val_acc"][-1]
    ave_test_loss += value["test_loss"]
    avg_test_acc += value["test_acc"]

num_folds = len(results_all.items())
print(f'Average Train Loss: {avg_train_loss / num_folds}')
print(f'Average Val Loss: {avg_val_loss / num_folds}')
print(f'Average Train Acc: {avg_train_acc / num_folds}%')
print(f'Average Val Acc: {avg_val_acc / num_folds}%')
print(f'Average Test Loss: {ave_test_loss / num_folds}')
print(f'Average Test Acc: {avg_test_acc / num_folds}%')
print('--------------------------------')

print('--------------------------------')
