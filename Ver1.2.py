import pandas as pd
from scipy.io import loadmat
import mne
import numpy as np
from scipy.signal import iirnotch ,filtfilt, butter , sosfiltfilt
import os
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import PyQt5
import darkdetect
import mne_nirs
import torch
from networks import LGGNet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import functions as fn


directory_path = 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\HSO_00\\'
mat_files_data = fn.find_and_load_files(directory_path,'.mat')

# Apply the notch filter to all datasets

fs = 256  # Sample rate, Hz
notch_freq = 60  # Notch frequency, Hz
quality_factor = 30  # Quality factor
# b_notch, a_notch = design_notch_filter(notch_freq, quality_factor, fs)
# filtered_data_list = apply_notch_filter_to_data(mat_files_data, b_notch, a_notch, key='signal', channels=slice(0,15))

# Apply the iir bandpass filter to all datasets

lowcut = 0.5  # Low cut-off frequency, Hz
highcut = 30  # High cut-off frequency, Hz
# sos = design_bandpass_filter(lowcut, highcut, fs, order=4)
# double_filtered_eeg_data = apply_iir_filter_to_data(filtered_data_list,sos, key='signal', channels=slice(0,15))
# double_filtered_eeg_data.plot.line(subplots=True,figsize=(20,18))

# Import the EEG data to MNE
eeg_ch_names = [
            'Fpz','AFz','F5','F6','FCz','FC3','FC4','Cz','C5','C6','TTP7','TTP8','TPP7','TPP8','Pz'
            ]
raw_eeg_concatenated = fn.import_eeg_to_mne(mat_files_data, key='signal', 
                                         channels=slice(0, 15),eeg_channels=eeg_ch_names,
                                         stimulus='StimulusCode', stimulusCode=32, fs=256)
fn.edit_eeg_raw_metadata(raw_eeg_concatenated, line_freq= 60, subject_id = 00, subject_sex = 'M',
                          subject_hand = 1, exprimenter = 'NeuralPC Lab URI', experiment_description = 'Auditory Oddball Task')

ica, raw_eeg_preprocessed =fn.preprocess_eeg(raw_eeg_concatenated, lowcut=0.5, highcut= 30, notch_freq=60, ica = True, method = 'infomax', exclude = [0,1], plot=False, start=100, stop=200)
# Import the fNIRS data to MNE
hbo_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_Ox'
hbo_data = fn.find_and_load_files(hbo_address,filetype='.txt')
ch_names =['S1_D1','S2_D1', 'S2_D2', 'S3_D2', 'S4_D3', 'S4_D4', 'S4_D5', 'S5_D4', 'S5_D5', 'S6_D6', 'S6_D7','S6_D8', 'S7_D7','S7_D8']
raw_hbo_combined = fn.import_fnirs_to_mne(hbo_data,type= 'hbo',ch_names= ch_names, fs=7.8125)

# Importing the Hbr data
hbr_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_DeOx'
hbr_data = fn.find_and_load_files(hbr_address,filetype='.txt')
raw_hbr_combined = fn.import_fnirs_to_mne(hbr_data,type= 'hbr',ch_names= ch_names, fs=7.8125)

# Merging EEG and fNIRS data
raw_combined = fn.merging_eeg_fnirs(raw_eeg_preprocessed, raw_hbo_combined, raw_hbr_combined)
# Set the montage for the combined data
sources = {'S1': 'F3', 'S2': 'Fz', 'S3':'F4','S4':'TP7','S5':'P5','S6':'TP8','S7':'P6'}
detectors = {'D1': 'F1', 'D2': 'F2','D3':'T7','D4':'CP5','D5':'P7','D6':'T8','D7':'CP6','D8':'P8'}
fn.set_combined_montage(montage = 'standard_1005', sources = sources, detectors = detectors)
fn.edit_eeg_raw_metadata(raw_combined, line_freq= 60, subject_id = 00, subject_sex = 'M',
                          subject_hand = 1, exprimenter = 'NeuralPC Lab URI', experiment_description = 'Auditory Oddball Task')

# Epoching combined data
picks = mne.pick_types(raw_combined.info, meg=False, eeg=True, stim=False,fnirs=True, eog=False)
event_id = dict(deviant40=2,standard1k=1)
mapping={2:'deviant40',1:'standard1k'}
epochs_combined, labels = fn.epocking(raw_combined,tmin = -0.5, tmax = 1.5, event_id = event_id, mapping = mapping,stim_channel=['StimulusCode'],picks = picks)



class_deviant =epochs_combined["deviant40"]#.drop_channels("StimulusCode")
class_standard =epochs_combined["standard1k"][np.random.randint(0,735,120)]#.drop_channels("StimulusCode")
# making a lable vector
labels = np.zeros(len(class_deviant.events) + len(class_standard.events), int)
labels[0:len(class_deviant)] = 0
labels[len(class_deviant):240] = 1
print(labels)
print(labels.shape)
#concatenate the epochs
epochs_concat = np.vstack([class_deviant, class_standard])
epochs_concat.astype(float)

# Define the graph
original_order = raw_combined.ch_names
# original_order.remove('StimulusCode')

graph_general_DEAP = [['S1_D1 hbo','S1_D1 hbr', 'AFz', 'F5'], ['F6','S3_D2 hbo','S3_D2 hbr' , 'FC4'], ['AFz', 'FCz', 'S2_D1 hbo','S2_D1 hbr', 'S2_D2 hbo', 'S2_D2 hbr'],
                     ['C6', 'TTP8', 'S6_D6 hbo', 'S6_D7 hbo','S6_D6 hbr', 'S6_D7 hbr'], ['TPP7', 'S4_D5 hbo', 'S5_D5 hbo','S4_D5 hbr', 'S5_D5 hbr'],
                     ['C5', 'TTP7', 'S4_D3 hbo', 'S4_D4 hbo','S4_D3 hbr', 'S4_D4 hbr'],
                     ['TPP8','S6_D8 hbr', 'S7_D8 hbr','S6_D8 hbo', 'S7_D8 hbo'],
                     ['Pz'],['Fpz'], ['Cz'],
                     ]

graph_idx = graph_general_DEAP   # The general graph definition for DEAP is used as an example.
idx = []
num_chan_local_graph = []
for i in range(len(graph_idx)):
    num_chan_local_graph.append(len(graph_idx[i]))
    for chan in graph_idx[i]:
        idx.append(original_order.index(chan))

LGG = LGGNet(
    num_classes=2,
    input_size=(1, 43, 513),
    sampling_rate=256,
    num_T=4,  # num_T controls the number of temporal filters
    out_graph=39,
    pool=16,
    pool_step_rate=0.25,
    idx_graph=num_chan_local_graph,
    dropout_rate=0.5
).to('cuda')

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype('float32')  # Your EEG data, shaped (N, 1, 43, data_points) where N is the number of epochs
        self.data = torch.from_numpy(self.data)
        self.data = self.data.unsqueeze(0).unsqueeze(0)
        self.data = self.data.to('cuda')
        self.labels = labels.astype('float')
        self.labels = torch.from_numpy(labels)  # Corresponding labels for each epoch
    
        self.labels = self.labels.to('cuda')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

epochs_concat = epochs_concat.astype(float)
dataset = EEGDataset(epochs_concat, labels)

dataset.data = dataset.data.reshape(240,1,43,513)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(LGG.parameters(), lr=0.0001)  # Adjust learning rate as needed


train_size = int(0.75 * len(dataset))  # 75% of the dataset size
val_size =  int(0.15 * len(dataset))   # 15% for validation
test_size =  int(0.10 * len(dataset))  # 10% for testing

# Splitting the dataset
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size], generator=generator1)

# Creating DataLoaders for both training and validation sets
train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True) 
val_loader = DataLoader(val_dataset,batch_size=10, shuffle=False) 
test_loader = DataLoader(test_dataset, shuffle=False)

num_epochs = 40
#Training the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    LGG.train()
    pred_train = []
    act_train = []
    for data, label in train_loader:
        # data = data.unsqueeze(0)
        label = label.long()
        outputs = LGG(data)
        loss = criterion(outputs, label)
        _, pred = torch.max(outputs, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(label.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')
        
        # Validation step
        LGG.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for input, label in val_loader:
                label = label.long()
                outputs = LGG(input)
                val_loss += criterion(outputs, label).item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print(f'Epoch {epoch}, Validation Loss: {val_loss / total}, Accuracy: {correct / total * 100}%')
        
# Test step
LGG.eval()
with torch.no_grad():
    test_loss = 0
    correct = 0
    total = 0
    for input, label in test_loader:
        label = label.long()
        outputs = LGG(input)
        test_loss += criterion(outputs, label).item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f'Test Loss: {test_loss / total}, Accuracy: {correct / total * 100}%')
