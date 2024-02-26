from scipy.io import loadmat
import mne
from scipy.signal import iirnotch ,filtfilt, butter , sosfiltfilt
import os
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import torch
from networks import LGGNet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import functions as fn
from EEG_fNIRS_Info import EEG_fNIRS_Info
import midfunctions as mf

eeg_dir = 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\HSO_00\\'
hbr_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_DeOx'
hbo_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_Ox'

eeg_ch_numbers = [0,15] #Column numbers for the EEG data
eeg_fs = 256  # Sample rate, Hz
notch_freq = 60  # Notch frequency, Hz
quality_factor = 30  # Quality factor
lowcut = 0.5  # Low cut-off frequency, Hz
highcut = 30  # High cut-off frequency, Hz
stimuluscode_no = 32 #StimulusCode Column number in the data
stimuluscode = 'StimulusCode' #StimulusCode Column name in the data
eeg_ch_names = [
            'Fpz','AFz','F5','F6','FCz','FC3','FC4','Cz','C5','C6','TTP7','TTP8','TPP7','TPP8','Pz'
            ] #EEG channel names
key = 'signal' #Key for the EEG data in the mat file
fnirs_ch_names =[
    'S1_D1','S2_D1', 'S2_D2', 'S3_D2', 'S4_D3', 'S4_D4', 'S4_D5', 'S5_D4', 'S5_D5', 'S6_D6', 'S6_D7','S6_D8', 'S7_D7','S7_D8'
    ]
fnirs_fs = 7.8125 # fNIRS sample rate
source_locations = [ 'F3', 'Fz','F4','TP7','P5','TP8','P6'] #fNIRS source locations
detector_locations = ['F1', 'F2','T7','CP5','P7','T8','CP6','P8'] #fNIRS detector locations
subject_id = 00
ica = True
ica_exclude = [0,1]



info_ef = EEG_fNIRS_Info(eeg_dir= eeg_dir, eeg_ch_names= eeg_ch_names, eeg_ch_numbers= eeg_ch_numbers,eeg_key = key ,
                  eeg_stimulus_name=stimuluscode, eeg_stimulus_no=stimuluscode_no, eeg_lowcut=lowcut, eeg_highcut=highcut, eeg_notch=notch_freq, eeg_fs=eeg_fs, 
                  hbo_dir=hbo_address, hbr_dir=hbr_address, hbo_fs=fnirs_fs, hbr_fs=fnirs_fs, hbo_lowcut=None, hbr_lowcut=None,  
                  source_locations=source_locations, detector_locations=detector_locations,
                  hbo_highcut=None, hbr_highcut=None, hbo_ch_names=fnirs_ch_names, hbr_ch_names=fnirs_ch_names, ica = ica, ica_exclude = ica_exclude,
                  subject_id = subject_id, subject_sex = 'M',subject_hand = 'right', exprimenter = 'NeuralPC Lab URI', experiment_description = 'Auditory Oddball Task' )

raw_combined = mf.import_data(info=info_ef)


event_id = dict(deviant40=2,standard1k=1)
mapping={2:'deviant40',1:'standard1k'}
picks = mne.pick_types(raw_combined.info, eeg=True, eog=False, stim=False, exclude='bads')
tmin =-0.5
tmax = 1.5

epochs_concat, labels , number_of_channels, input_length, Number_trials_per_class = mf.epock_data(raw_combined, tmin =tmin, tmax =tmax, event_id =event_id, mapping = mapping, stim=info_ef.eeg_stimulus_name, picks=picks,fs=info_ef.eeg_fs)

dataset = mf.create_dataset(epochs_concat, labels, Number_trials_per_class, number_of_channels, input_length)


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
    input_size=(1, number_of_channels, input_length),
    sampling_rate=info_ef.eeg_fs,
    num_T=4,  # num_T controls the number of temporal filters
    out_graph=39,
    pool=16,
    pool_step_rate=0.25,
    idx_graph=num_chan_local_graph,
    dropout_rate=0.5
).to('cuda')

#%%
criterion = nn.CrossEntropyLoss()#LabelSmoothing(smoothing=0.1) #
optimizer = optim.Adam(LGG.parameters(), lr=0.001)  # Adjust learning rate as needed

train_size = int(0.80 * len(dataset))  # 75% of the dataset size
val_size =  int(0.15 * len(dataset))   # 15% for validation
test_size =  int(0.05 * len(dataset))  # 10% for testing

# Splitting the dataset
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size], generator=generator1)

# Creating DataLoaders for both training and validation sets
train_loader = DataLoader(train_dataset,batch_size=2, shuffle=True) 
val_loader = DataLoader(val_dataset,batch_size=2, shuffle=False) 
test_loader = DataLoader(test_dataset, shuffle=False)


#%%

num_epochs = 1000
#Training the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    LGG.train()
    pred_train = []
    act_train = []
    for data, label in train_loader:
        
        label = label.long()
        outputs = LGG(data)
        loss = criterion(outputs, label)
        _, pred = torch.max(outputs, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(label.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f'Loss: {loss.item()}')
       
        # Validation step
    LGG.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for input, label in val_loader:
            label = label.long()
            outputs = LGG(input)
            val_loss += criterion(outputs, label).sum().item()
            _, predicted = torch.max(outputs, 1)
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





class LabelSmoothing(nn.Module):
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
