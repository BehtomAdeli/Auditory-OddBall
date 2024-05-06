import importlib
import mne
import torch
import midfunctions as mf
import netfunctions as nf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
importlib.reload(nf)

def test_5x5_ncv(tmin=0.01, tmax=1.99, window= [1, 0.40, 0.125, 0.0625], eeg=True, fnirs=False, threeD=False, model_directory = "run 1"):

    # Load the data
    raw_combined = mne.io.read_raw_fif('C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\MNE_ica01excluded-All_NEW.fif', preload=True)
    event_id = dict(deviant40=2,standard1k=1)
    mapping={2:'deviant40',1:'standard1k'}

    if threeD and eeg and fnirs:
        picks = mne.pick_types(raw_combined.info, eeg=True, fnirs=True, stim=False, exclude=['bads', 'Cz'])
    elif (threeD and eeg) or (threeD and fnirs):
        TypeError('Please select both EEG and fNIRS data for 3D model')
    else:
        picks = mne.pick_types(raw_combined.info, eeg=eeg, fnirs=fnirs, stim=False, exclude=['bads'])

    mapping_fnirs = [2, 1, 4, 3, 8, 13, 9, 6, 11, 5, 10, 7, 12, 14]
    epochs_concat, labels , number_of_channels, input_length, Number_trials_per_class = mf.epoch_data(raw_combined, tmin =tmin, tmax =tmax, event_id =event_id, mapping = mapping, stim=['StimulusCode'] , picks=picks,mapping_fnirs=mapping_fnirs,threeD=threeD)

    # Create LGGNet parameters
    lggnet_params = {
                    'num_classes': 2, 
                    'batch_size': 10,
                    'num_epochs': 50,
                    'number_trials_per_class': Number_trials_per_class,
                    'number_of_channels': number_of_channels, 
                    'input_length': input_length, 
                    }
    optimize_params = {'learning_rate': 0.001, 'dropout_rate': 0.5, 'num_T': 64, 'out_graph': 4, 'pool': 16, 'pool_step_rate': 0.3}
    #[0.4994, 0.7819, 0.6219, 0.2443, 1.0, 0.1957, 0.7723, 0.7708] #[0.8966, 0.6767, 1.4293, 1.382, 0.416, 1.4833, 0.0791, 1.4945]

    dataset = mf.create_dataset(epochs_concat, labels, lggnet_params, threeD=threeD)

    # Define the graph
    original_order = raw_combined.ch_names
    original_order.remove('StimulusCode')

    if eeg and not fnirs:
        graph_general_list = [['F5','FC3'],['F6', 'FC4'], ['Fpz','AFz','FCz'],
                              ['C6','TTP8','TPP8'],
                              ['C5','TTP7','TPP7'],
                              ['Pz','Cz']
                             ]
    elif fnirs and not eeg:
        graph_general_list = [['S1_D1 hbo','S1_D1 hbr'], ['S3_D2 hbo','S3_D2 hbr' ], ['S2_D1 hbo','S2_D1 hbr', 'S2_D2 hbo', 'S2_D2 hbr'],
                              ['S6_D6 hbo', 'S6_D6 hbr', 'S6_D7 hbo', 'S6_D7 hbr'], ['S6_D8 hbr', 'S6_D8 hbo', 'S7_D8 hbr'], ['S7_D8 hbo','S7_D7 hbo', 'S7_D7 hbr'], 
                              ['S4_D3 hbo', 'S4_D4 hbo','S4_D3 hbr', 'S4_D4 hbr'],  ['S4_D5 hbo', 'S4_D5 hbr', 'S5_D5 hbo'], ['S5_D5 hbr','S5_D4 hbo', 'S5_D4 hbr'],
                             ]
    elif eeg and fnirs and not threeD:
        graph_general_list = [['S1_D1 hbo','S1_D1 hbr', 'AFz', 'F5'], ['F6','S3_D2 hbo','S3_D2 hbr' , 'FC4'], ['AFz', 'FCz', 'S2_D1 hbo','S2_D1 hbr', 'S2_D2 hbo', 'S2_D2 hbr'],
                              ['C6', 'TTP8', 'S6_D6 hbo', 'S6_D7 hbo','S6_D6 hbr', 'S6_D7 hbr'], ['TPP7', 'S4_D5 hbo', 'S5_D5 hbo','S4_D5 hbr', 'S5_D5 hbr'],
                              ['C5', 'TTP7', 'S4_D3 hbo', 'S4_D4 hbo','S4_D3 hbr', 'S4_D4 hbr'],
                              ['TPP8','S6_D8 hbr', 'S7_D8 hbr','S6_D8 hbo', 'S7_D8 hbo'],
                              ['Pz'], ['Cz']
                             ]
    elif eeg and fnirs and threeD:
        graph_general_list = [['F5','FC3'],['F6', 'FC4'], ['AFz','FCz','Fpz','Pz'],
                              ['C6','TTP8','TPP8'],
                              ['C5','TTP7','TPP7'],
                              
                             ]

    kfolds=5
    torch.manual_seed(42)
    manual_seed = 42
    results_all = {}
    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=manual_seed)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = Subset(dataset, train_ids)
        test_subsampler = Subset(dataset, test_ids)
        test_loader = DataLoader(test_subsampler, batch_size=lggnet_params['batch_size'], shuffle=True,generator=torch.Generator().manual_seed(manual_seed))

        # Train,Validate, and Test the model
        train_losses, val_losses, train_acc, val_acc,test_loss, test_acc = nf.run_model_new(raw_combined, train_subsampler, test_loader,optimize_params, lggnet_params,
                                                                                            ifplot=False, iftest=True, kfolds=5, ifresult=False, threeD=threeD, window = window,
                                                                                            graph_general=graph_general_list,run_number= fold+1, model_directory=model_directory )
        results_all[fold] = {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_acc, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc}
    
    nf.print_results(results_all, kfolds)

    return results_all
