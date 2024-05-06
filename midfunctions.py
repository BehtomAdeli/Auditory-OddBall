import functions as fn
from EEG_fNIRS_Info import EEG_fNIRS_Info
import mne
import numpy as np
import torch

def import_data(info=EEG_fNIRS_Info):
    # Check if the info dictionary is provided
    # using the info file to import the data
    #
    # Parameters
    # ----------
    # info : EEG_fNIRS_Info
    #     The info dictionary containing the information about the data
    #
    # Returns
    # -------
    # raw_combined : mne.io.Raw
    #     The combined raw data containing the EEG and fNIRS data
    #

    if info == None:
        raise ValueError('Please provide the info dictionary')
    if not isinstance(info, EEG_fNIRS_Info):
        raise ValueError('Please provide the info dictionary')
    
    # Import the EEG data to MNE
    mat_files_data = fn.find_and_load_files(info.eeg_dir,'.mat')
    raw_eeg_concatenated = fn.import_eeg_to_mne(mat_files_data, key=info.eeg_key, channels=info.eeg_ch_numbers,
                                                eeg_channels=info.eeg_ch_names, stimulus= info.eeg_stimulus_name, stimulusCode=info.eeg_stimulus_no, fs=info.eeg_fs)


    ica, raw_eeg_preprocessed =fn.preprocess_eeg(raw_eeg_concatenated, lowcut=info.eeg_lowcut, highcut= info.eeg_highcut, notch_freq=info.eeg_notch,
                                                ica = info.ica , method = 'infomax', exclude = info.ica_exclude, plot=False, start=100, stop=200)
    # Import the fNIRS data to MNE
    hbo_data = fn.find_and_load_files(info.hbo_dir,filetype='.txt')

    raw_hbo_combined = fn.import_fnirs_to_mne(hbo_data,type= 'hbo',ch_names= info.hbo_ch_names, fs=info.hbo_fs)

    # Importing the Hbr data
    hbr_data = fn.find_and_load_files(info.hbr_dir,filetype='.txt')
    raw_hbr_combined = fn.import_fnirs_to_mne(hbr_data,type= 'hbr',ch_names= info.hbr_ch_names, fs=info.hbr_fs)

    # Merging EEG and fNIRS data
    raw_combined = fn.merging_eeg_fnirs(raw_eeg_preprocessed, raw_hbo_combined, raw_hbr_combined)
    # Set the montage for the combined data
    fn.set_combined_montage(montage = 'standard_1005', sources = info.sources, detectors = info.detectors, eeg_ch_names = info.eeg_ch_names)
    fn.edit_raw_metadata(raw_combined, subject_id = info.subject_id, subject_sex = info.subject_sex,
                        subject_hand = info.subject_hand, exprimenter = info.exprimenter, 
                        experiment_description = info.experiment_description)
    return raw_combined


def epoch_data(raw_combined, tmin =None, tmax =None, event_id =None, mapping = None, stim="StimulusCode", picks=None,fs=None,mapping_fnirs=None,threeD=False):
    # Epoching the combined data
    # 
    # Parameters
    # ----------
    # raw_combined : mne.io.Raw
    #     The combined raw data containing the EEG and fNIRS data
    # tmin : float
    #     The start time of the epoch in seconds
    # tmax : float
    #     The end time of the epoch in seconds
    # event_id : dict
    #     The dictionary containing the event ids
    # mapping : dict
    #     The dictionary containing the mapping of the event ids
    # stim : str
    #     The name of the stimulus channel
    # picks : list
    #     The list of the channels to be picked
    # 
    # Returns
    # -------
    # epochs_concat : np.array
    #     The concatenated epochs
    # labels : np.array
    #     The labels of the epochs
    # 
    # 
    # Epoching combined data

    if threeD:
        if mapping_fnirs == None:
            mapping_fnirs = [2, 1, 4, 3, 8, 13, 9, 6, 11, 5, 10, 7, 12, 14]
        else:
            raw_combined = rearrange_channels(raw_combined, mapping_fnirs)
    np.random.seed(42)
    epochs_combined, all_labels = fn.epoching(raw_combined,tmin = tmin, tmax = tmax, event_id = event_id, mapping = mapping,picks = picks)
    all_labels = all_labels - 1
    label_zeros = all_labels[all_labels == 0]
    label_ones = all_labels[all_labels == 1]
    trials_per_class = epochs_combined['deviant40'].__len__()
    trials_standard = epochs_combined['standard1k'].__len__()
    number_of_channels = epochs_combined.ch_names.__len__()
    input_length = ((epochs_combined['deviant40'].tmax - epochs_combined['deviant40'].tmin)*raw_combined.info['sfreq'] + 1).astype(int)
    class_deviant =epochs_combined["deviant40"]
    random_choose = get_random_choose(0,7,trials_per_class,trials_standard,seed=42)
    class_standard =epochs_combined["standard1k"][random_choose]    #.drop_channels("StimulusCode")
    label_zeros_needed=label_zeros[random_choose]
        
    #concatenate the epochs and lables
    epochs_concat = np.vstack([class_deviant, class_standard])
    epochs_concat.astype(float)
    labels = np.hstack((label_ones,label_zeros_needed))
    #print(labels)
    print("number of trials per class:",labels.shape[0]/2)

    return epochs_concat, labels,number_of_channels, input_length, trials_per_class

def get_random_choose(start, end, trials_per_class,max, seed=42):
    
    np.random.seed(seed=seed)
    random_choose = np.random.randint(start,end,trials_per_class)
    cumulative_sum = []
    current_sum = 0

    for i,  num in enumerate(random_choose):
        
        current_sum = num +i*np.random.randint(6,7,1)[0]
        cumulative_sum.append(current_sum)

    return cumulative_sum

def create_dataset(epochs_concat, labels, lggnet_params=None,threeD=False):
    """
    This function creates a dataset from the concatenated epochs and labels
    :param epochs_concat: The concatenated epochs
    :param labels: The labels of the epochs
    :param lggnet_params: The parameters of the LGGNet model
    :return: dataset
    """
    class EEGDataset():
        def __init__(self, data, labels):

            self.data = data.astype('float32')  # Your EEG data, shaped (N, 1, channels, data_points) where N is the number of epochs
            self.data = torch.from_numpy(self.data)
            self.data = self.data.unsqueeze(1)
            self.labels = labels.astype('float')
            self.labels = torch.from_numpy(labels)  # Corresponding labels for each epoch

            CUDA = torch.cuda.is_available()
            if CUDA:
                self.labels = self.labels.to('cuda')
                self.data = self.data.to('cuda')
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
          
    epochs_concat = epochs_concat.astype(float)
    dataset = EEGDataset(epochs_concat, labels)
    if threeD:
        dataset.data = dataset.data.reshape(lggnet_params['number_trials_per_class']*2,3, int(lggnet_params['number_of_channels']/3), lggnet_params['input_length'])
        dataset.data = dataset.data.unsqueeze(2)
    
    else:
        dataset.data = dataset.data.reshape(lggnet_params['number_trials_per_class']*2,1,lggnet_params['number_of_channels'], lggnet_params['input_length'])
    
    print("Shape of the dataset:",dataset.data.shape)

    return dataset


def rearrange_channels(raw_combined, mapping):

    # Extract EEG, HBO, and HBR channels
    picks = mne.pick_types(raw_combined.info, eeg=True, fnirs=False, stim=False)
    eeg_channels = raw_combined.info.ch_names[:16]
    hbo_channels = raw_combined.info.ch_names[16:30]
    hbr_channels = raw_combined.info.ch_names[30:]

    # Rearrange HBO and HBR channels
    rearranged_hbo = [hbo_channels[i - 1] for i in mapping]
    rearranged_hbr = [hbr_channels[i - 1] for i in mapping]

    # Combined rearranged channels
    rearranged_channels = eeg_channels + rearranged_hbo + rearranged_hbr

    print(rearranged_channels)

    raw_combined_rearranged = raw_combined.copy().reorder_channels(rearranged_channels)
    # Check the new order of channels
    print(raw_combined_rearranged.info.ch_names)
    
    return raw_combined_rearranged