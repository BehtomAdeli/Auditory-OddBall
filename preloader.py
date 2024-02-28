from EEG_fNIRS_Info import EEG_fNIRS_Info
import midfunctions as mf
"""
This script is used to preprocess the data and save it as a .fif file. The .fif file can be used to be further processed in the main-process.py file.
EDIT THE FOLLOWING PARAMETERS BASED ON YOUR DATA:
parameters:
eeg_dir: The directory containing the EEG data
hbo_address: The directory containing the fNIRS hbo data
hbr_address: The directory containing the fNIRS hbr data
eeg_ch_names: The names of the EEG channels
fnirs_ch_names: The names of the fNIRS channels
source_locations: The locations of the fNIRS sources
detector_locations: The locations of the fNIRS detectors
subject_id: The ID of the subject
ica: A boolean that determines whether ICA will be used for preprocessing the EEG data
ica_exclude: A list of integers that represent the indices of the ICA components to be excluded
EEG_fNIRS_Info: The EEG_fNIRS_Info object (Please edit this part based on your data)


Behtom Adeli, 2024 URI
"""

# Directory containing the EEG and fNIRS data
eeg_dir = 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\HSO_00\\'
hbr_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_DeOx'
hbo_address= 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\H_SO_00\\All_Ox'
 #EEG channel names
eeg_ch_names = ['Fpz','AFz','F5','F6','FCz','FC3','FC4','Cz','C5','C6','TTP7','TTP8','TPP7','TPP8','Pz']
fnirs_ch_names =['S1_D1','S2_D1', 'S2_D2', 'S3_D2', 'S4_D3', 'S4_D4', 'S4_D5', 'S5_D4', 'S5_D5', 'S6_D6', 'S6_D7','S6_D8', 'S7_D7','S7_D8']
#fNIRS source locations
source_locations = [ 'F3', 'Fz','F4','TP7','P5','TP8','P6'] 
#fNIRS detector locations
detector_locations = ['F1', 'F2','T7','CP5','P7','T8','CP6','P8'] 
# Indipeendent Component Analysis
ica = True
ica_exclude = [0,1]

# Create the EEG_fNIRS_Info object Please edit this part based on your data
info_ef = EEG_fNIRS_Info(eeg_dir= eeg_dir, eeg_ch_names= eeg_ch_names, eeg_ch_numbers= [0,15],eeg_key = 'signal' ,
                  eeg_stimulus_name= 'StimulusCode', eeg_stimulus_no= 32 , eeg_lowcut= 0.5, eeg_highcut=30, eeg_notch=60, eeg_fs = 256, 
                  hbo_dir=hbo_address, hbr_dir=hbr_address, hbo_fs=7.8125, hbr_fs=7.8125, hbo_lowcut=None, hbr_lowcut=None,  
                  source_locations=source_locations, detector_locations=detector_locations,
                  hbo_highcut=None, hbr_highcut=None, hbo_ch_names=fnirs_ch_names, hbr_ch_names=fnirs_ch_names, ica = ica, ica_exclude = ica_exclude,
                  subject_id = 00, subject_sex = 'M',subject_hand = 'right', exprimenter = 'NeuralPC Lab URI', experiment_description = 'Auditory Oddball Task' )

raw_combined = mf.import_data(info=info_ef)

save_dir = 'C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\HSO_00\\'
save_name = 'MNE_ica01excluded.fif'

raw_combined.save(save_dir+save_name, overwrite=True)
