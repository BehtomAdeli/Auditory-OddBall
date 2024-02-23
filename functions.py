import pandas as pd
from scipy.io import loadmat
import mne
import numpy as np
from scipy.signal import iirnotch ,filtfilt, butter , sosfiltfilt
import os
from mne.preprocessing import ICA

# Importing Data
def find_and_load_files(directory,filetype):
    """
    Find and load all .mat files in a directory.

    Parameters:
    - directory: The directory to search for .mat files.
    - filetype: The type of file to search for. E.G. '.mat', '.csv', etc.

    Returns:
    - data_list: A list of dictionaries, each containing data from a .mat file.
    
    See Also:
    - loadmat
    """
    data_list = []
    if not isinstance(filetype, str):
        raise ValueError("filetype must be a string")
    elif not filetype.startswith('.'):
        raise ValueError("filetype must start with a period (.)")
    elif not os.path.isdir(directory):
        raise ValueError("directory must be a valid directory")
    elif isinstance(filetype, str) and filetype=='.mat':
        print(f"Searching for {filetype} files in {directory}...")
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(filetype):
                    file_path = os.path.join(root, file)
                    print(f"Loading {file_path}")
                    
                    mat_data = loadmat(file_path)
                    data_list.append(mat_data)
    elif isinstance(filetype, str) and filetype=='.csv':
        print(f"Searching for {filetype} files in {directory}...")
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(filetype):
                    file_path = os.path.join(root, file)
                    print(f"Loading {file_path}")
                    
                    csv_data = pd.read_csv(file_path)
                    data_list.append(csv_data)
                    
    elif isinstance(filetype, str) and filetype=='.txt':
        print(f"Searching for {filetype} files in {directory}...")
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(filetype):
                    file_path = os.path.join(root, file)
                    print(f"Loading {file_path}")
                    
                    txt_data = pd.read_csv(file_path, delim_whitespace=True, header=None)
                    data_list.append(txt_data)
    
    return data_list

# Filter designs
def design_notch_filter(notch_freq, quality_factor, fs):
    """
    Design a notch filter using the iirnotch function.

    Parameters:
    - notch_freq: The target frequency to notch out (e.g., 60 Hz).
    - Quality_factor: The quality factor of the notch filter.
    - fs: The sampling frequency of the data.

    Returns:
    - b, a: Numerator (b) and denominator (a) polynomials of the IIR filter.
    
    See Also:
    - scipy.signal.iirnotch

    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    return b, a

def design_bandpass_filter(lowcut, highcut, fs, order=5):
    """
    Design a bandpass filter using the butter function.

    Parameters:
    - lowcut: Low cut-off frequency, Hz.
    - highcut: High cut-off frequency, Hz.
    - fs: The sampling frequency of the data.
    - order: The order of the filter. Default is 5.

    Returns:
    sos : ndarray
    Second-order sections representation of the IIR filter. See Notes for more information.

    Notes:
    The sos parameter was added in scipy version 0.16.0 (2015.06.15).
    Read Butterworth filter for more information on the sos parameter.

    See also:
    butter, sosfilt, sosfiltfilt, sosfreqz, sosfreqz_zpk, sos2tf, zpk2sos
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output='sos')
    return sos

# Apply filters

def apply_notch_filter_to_data(data_list, b, a, key='data', channels=slice(0,15)):
    """
    Apply a filter to each dataset in the list.

    Parameters:
    - data_list: List of dictionaries, each containing data from a .mat file.
    - b, a: Numerator (b) and denominator (a) polynomials of the IIR filter.
    - key: The key in the dictionary that corresponds to the data to be filtered. E.G. 'data', 'signal', etc.
    - channels: The channels to be filtered. Default is all channels. Can be a slice object or a list of integers. E.G. slice(0, 15), [0, 1, 2, 3, 4].
    

    Returns:
    - filtered_data_list: List of dictionaries with the filtered data.
    """
    filtered_data_list = []
    for data in data_list:
        # Make a copy of the data to avoid modifying the original
        filtered_data = data.copy()
        if not isinstance(data[key], np.ndarray) and data[key].ndim == 2 and data[key].shape[1] >= channels.stop:
                print(f"Filtering {key}...")
        # Apply the notch filter
        if key in data:
            filtered_data[key][:,channels] = filtfilt(b, a, data[key][:,channels].T).T
        
        filtered_data_list.append(filtered_data)
    
    return filtered_data_list

def apply_iir_filter_to_data(data_list, sos, key='data', channels=slice(0,15)):
    """
    Apply a IIR filter to each dataset in the list.

    Parameters:
    - data_list: List of dictionaries, each containing data from a .mat file.
    - sos: Second-order sections representation of the IIR filter.
    - key: The key in the dictionary that corresponds to the data to be filtered. E.G. 'data', 'signal', etc.
    - channels: The channels to be filtered. Default is all channels. Can be a slice object or a list of integers. E.G. slice(0, 15), [0, 1, 2, 3, 4].

    Returns:
    - filtered_data_list: List of dictionaries with the filtered data.

    Notes:  
    The sos parameter was added in scipy version 0.16.0 (2015.06.15).

    See also:
    butter, sosfilt, sosfiltfilt, sosfreqz, sosfreqz_zpk, sos2tf, zpk2sos
    """
    filtered_data_list = []
    for data in data_list:
        # Make a copy of the data to avoid modifying the original
        filtered_data = data.copy()
        if not isinstance(data[key], np.ndarray) and data[key].ndim == 2 and data[key].shape[1] >= channels.stop:
                print(f"Filtering {key}...")
        # Apply the notch filter
        if key in data:
            filtered_data[key][:,channels] = sosfiltfilt(sos, data[key][:,channels].T).T
        
        filtered_data_list.append(filtered_data)
    
    return filtered_data_list


# Import EEG data to MNE
def import_eeg_to_mne(double_filtered_eeg_data, key='signal', channels=slice(0, 15),eeg_channels = [any], stimulus = 'StimulusCode', stimulusCode=32, fs=256):
    """
    Import All sessions of EEG data into MNE.

    Parameters:
    - double_filtered_eeg_data: List of dictionaries, each containing data from all .mat files for all sessions.
    - key: The key in the dictionary that corresponds to the data to be filtered. E.G. 'signal', etc.
    - channels: The channels to be filtered. Default is all channels. Can be a slice object or a list of integers. E.G. slice(0, 15), [0, 1, 2, 3, 4].
    - eeg_channels: The EEG channels names to be imported. This should be the locations of the electrodes. E.G. ['Fpz', 'AFz', 'F5', 'F6', 'FCz', 'FC3', 'FC4', 'Cz', 'C5', 'C6', 'TTP7', 'TTP8', 'TPP7', 'TPP8', 'Pz']
    - stimulus: The name of the stimulus channel. Default is 'StimulusCode'.
    - stimulusCode: The index of the stimulus channel. Default is 32.
    - lowcut: Low cut-off frequency, Hz. Default is None.
    - highcut: High cut-off frequency, Hz. Default is None.
    - notch_freq: The target frequency to notch out (e.g., 60 Hz). Default is None.
    - fs: The sampling frequency of the data. Default is 256 Hz.

    Returns:
    - raw_eeg_concatenated: A single MNE Raw object containing all EEG data from all sessions.

    Notes:
    Montage is set to standard 1005 system to cover all the possible electrode locations.

    See also:
    mne.channels.make_standard_montage, mne.create_info, mne.io.RawArray, mne.concatenate_raws

    """
    easycap_montage = mne.channels.make_standard_montage("standard_1005")
    ch_types = ['eeg'] * 15 + ['stim']
    ch_names = eeg_channels + [stimulus]
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    info.set_montage(easycap_montage)

    # Initialize an empty list to hold RawArray objects
    raw_eeg_list = []

    for data in double_filtered_eeg_data:
        # Extract the required columns
        columns = np.r_[channels, stimulusCode]
        eeg_data = data[key][:, columns].T  # Transpose to match MNE's expected shape
        # Create a RawArray and set the montage
        raw = mne.io.RawArray(eeg_data, info)
        raw.apply_function(lambda x: x*1e-6, picks="eeg")
        raw.set_montage(easycap_montage)

        # Append the RawArray to the list
        raw_eeg_list.append(raw)

    # Concatenate all RawArray objects into a single Raw object
    if raw_eeg_list:
        raw_eeg_concatenated = mne.concatenate_raws(raw_eeg_list)
    else:
        raise ValueError("No EEG data was provided.")

    return raw_eeg_concatenated

def preprocess_eeg(raw, lowcut=None, highcut=None, notch_freq=None,ica = True, method = 'infomax', exclude = [0,1], plot=False, start=100, stop=200):
    """
    Apply preprocessing steps to the EEG data.

    Parameters:
    - raw: The MNE Raw object containing the EEG data.
    - lowcut: Low cut-off frequency, Hz. Default is None.
    - highcut: High cut-off frequency, Hz. Default is None.
    - notch_freq: The target frequency to notch out (e.g., 60 Hz). Default is None.
    - plot: Whether to plot the ica after preprocessing. Default is False.
    - start: Default is 100.
    - stop: Default is 200.

    Returns:
    - raw: The MNE Raw object containing the preprocessed EEG data.

    Notes:
    The data is high-pass filtered at 0.5 Hz, low-pass filtered at 40 Hz, and notch filtered at 60 Hz.
    """
    if lowcut is not None:
        raw.filter(l_freq=lowcut, h_freq=highcut, picks='eeg', fir_design='firwin')
    if notch_freq is not None:
        raw.notch_filter(freqs=60, picks='eeg', fir_design='firwin')
    if ica:
        eeg_ch_number = raw.info.get_channel_types(picks='eeg').__len__()
        ica = ICA(n_components=eeg_ch_number,method= method,fit_params=dict(extended=True), max_iter='auto' , random_state=24)
        ica.fit(raw)
        explained_var_ratio = ica.get_explained_variance_ratio(
            raw, components=[1], ch_type="eeg"
        )
        # This time, print as percentage.
        ratio_percent = round(100 * explained_var_ratio["eeg"])
        print(
            f"Fraction of variance in EEG signal explained by first component: "
            f"{ratio_percent}%"
        )
        if exclude is not None:
            ica.exclude = exclude
            ica.apply(raw)
    if plot:
        ica.plot_sources(raw, show_scrollbars=False,start =start , stop = stop);
    
    return ica , raw

def import_fnirs_to_mne(txt_data,type= 'hbo',ch_names= [any],lowcut=None,highcut=None,notch_freq=None,plot = False, plot_duration=100, fs=7.8125):
    """
    Import fNIRS data to MNE.

    Parameters:
    - txt_data: List of pandas dataframes, each containing data from a .txt file.
    - type: The type of fNIRS data. Default is 'hbo'. Can be 'hbo' or 'hbr'.
    - ch_names: The channel names of the fNIRS data. Default is None.
    - lowcut: Low cut-off frequency, Hz. Default is None.
    - highcut: High cut-off frequency, Hz. Default is None.
    - notch_freq: The target frequency to notch out (e.g., 60 Hz). Default is None.
    - fs: The sampling frequency of the data. Default is 7.8125 Hz.

    Returns:
    - raw_fnirs_concatenated: A single MNE Raw object containing all fNIRS data from all sessions.

    Notes:
    Montage is set to not set.

    See also:
    mne.create_info, mne.io.RawArray, mne.concatenate_raws
    """
    raw_list = []
    for data in txt_data:
        n_channels = data.shape[1]
        if ch_names is None:
            ch_names =['S1_D1','S2_D1', 'S2_D2', 'S3_D2', 'S4_D3', 'S4_D4', 'S4_D5', 'S5_D4', 'S5_D5', 'S6_D6', 'S6_D7','S6_D8', 'S7_D7','S7_D8']
        if type == 'hbo':    
            ch_types = ['hbo'] * n_channels
        elif type == 'hbr':
            ch_types = ['hbr'] * n_channels
        info_hbo = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        data = data.to_numpy().T
        raw = mne.io.RawArray(data, info_hbo)
        raw_list.append(raw)
    # Concatenate all RawArray objects into a single Raw object
    if raw_list:
        raw_fnirs_concatenated = mne.concatenate_raws(raw_list)
    else:
        raise ValueError("No fNIRS data was provided.")
    if plot:
        raw_fnirs_concatenated.plot(
            n_channels=len(raw_fnirs_concatenated.ch_names), duration=plot_duration, show_scrollbars=False,scalings=dict(hbo='10e-4',hbr='10e-4')
        )
        
    return raw_fnirs_concatenated

# Edit eeg raw file metadata
def edit_raw_metadata(raw, line_freq= None, subject_id = None, subject_sex = None, subject_birthday = None,
                          subject_hand = None, exprimenter = None, experiment_date = None, experiment_time = None, experiment_description = None):
    """
    Edit the metadata of an MNE Raw object.

    Parameters:
    - raw: The MNE Raw object to be edited.
    - lowpass: The lowpass frequency of the data. Default is 40 Hz.
    - highpass: The highpass frequency of the data. Default is 0.5 Hz.
    - subject_id: The ID of the subject. Default is 0.
    - subject_sex: 'M' for Male, 'F' for female, 'U' for unknown. Default is 'M'.
    - subject_birthday: Set birthday as (Year, Month, Day)
    - subject_hand: 1 for male, 2 for female, 0 if unknown
    - exprimenter
    - experiment_date
    - experiment_time
    - experiment_description
    - experiment_code
    - line_freq: The frequency of the power line noise. Default is 60 Hz.

    Returns:
    - raw: The edited MNE Raw object.

    See Also:
    - mne.utils._dt_to_stamp

    """
    # Change participant information
    if 'subject_info' not in raw.info or raw.info['subject_info'] is None:
        raw.info['subject_info'] = {}
        if subject_id is not None:
            raw.info['subject_info']['his_id'] = subject_id  # Change participant ID
        if subject_birthday is not None:
            raw.info['subject_info']['birthday'] = subject_birthday  # Set birthday as (Year, Month, Day)
        if subject_sex is not None:
            raw.info['subject_info']['sex'] = subject_sex # 1 for male, 2 for female, 0 if unknown
        if subject_hand is not None:
            raw.info['subject_info']['hand'] = subject_hand  # 1 for right-handed, 2 for left-handed, 0 if unknown


    if exprimenter is not None:
            raw.info['experimenter'] = exprimenter
    if experiment_date is not None:
        raw.info['meas_date'] = mne.utils._dt_to_stamp(experiment_date + ' ' + experiment_time)  # Set the experiment date and time
    if experiment_description is not None:
        raw.info['description'] = experiment_description
    if line_freq is not None:
        raw.info['line_freq'] = line_freq

def merging_eeg_fnirs(raw_eeg, raw_hbo, raw_hbr):
    """
    Merge EEG and fNIRS data into a single MNE Raw object.

    Parameters:
    - raw_eeg: The MNE Raw object containing the EEG data.
    - raw_hbo: The MNE Raw object containing the fNIRS data.
    - raw_hbr: The MNE Raw object containing the fNIRS data.

    Returns:
    - raw: A single MNE Raw object containing all EEG and fNIRS data with combined channel names.

    Notes:
    The data is concatenated along the channel dimension and fNIRS data is resampled to match the EEG data's sampling frequency. But the montage is not set. It should be set using the set_combined_montage function.
    See also:
    mne.io.RawArray, mne.concatenate_raws, set_combined_montage
    """
    # Resample the fNIRS data to match the EEG data's sampling frequency
    raw_hbo_resampled = raw_hbo.resample(raw_eeg.info['sfreq'])
    min_duration = min(raw_eeg.times[-1], raw_hbo_resampled.times[-1])
    raw_hbo_resampled.crop(tmax=min_duration)

    raw_hbr_resampled = raw_hbr.resample(raw_eeg.info['sfreq'])
    raw_hbr_resampled.crop(tmax=min_duration)

    # Combine the channel names
    hbo_ch_names = [ch + ' hbo' for ch in raw_hbo_resampled.ch_names]
    hbr_ch_names = [ch + ' hbr' for ch in raw_hbr_resampled.ch_names]

    ch_names = raw_eeg.ch_names + hbo_ch_names + hbr_ch_names

    eeg_ch_number = raw_eeg.info.get_channel_types(picks='eeg').__len__()
    stim_ch_number = raw_eeg.info.get_channel_types(picks='stim').__len__()
    hbo_ch_number = raw_hbo_resampled.info.get_channel_types(picks='hbo').__len__()
    hbr_ch_number = raw_hbr_resampled.info.get_channel_types(picks='hbr').__len__()

    info_combined = mne.create_info(ch_names=ch_names,
                                sfreq=raw_eeg.info['sfreq'],
                                ch_types= ['eeg']*eeg_ch_number +['stim']*stim_ch_number + ['hbo'] *hbo_ch_number +['hbr'] *hbr_ch_number,
                                )

    # Combine the data arrays
    data_combined = np.concatenate([raw_eeg.get_data(), raw_hbo_resampled.get_data(),raw_hbr_resampled.get_data()], axis=0)

    raw_combined = mne.io.RawArray(data_combined, info_combined)

    return raw_combined

def set_combined_montage(montage = 'standard_1005', sources = None, detectors = None,eeg_ch_names = [any]):
    # Combine the info objects from both datasets, making sure to update the channel names to avoid duplicates
    # This might require modifying the channel names of the fNIRS data
    # Define your source and detector names and positions
    # For demonstration, we use standard EEG locations for simplicity
    if sources is None:
        sources = {'S1': 'F3', 'S2': 'Fz', 'S3':'F4','S4':'TP7','S5':'P5','S6':'TP8','S7':'P6'}
    
    if detectors is None:
        detectors = {'D1': 'F1', 'D2': 'F2','D3':'T7','D4':'CP5','D5':'P7','D6':'T8','D7':'CP6','D8':'P8'}
    # Standard montage for reference
    standard_montage = mne.channels.make_standard_montage(montage)
    # Extract positions from the standard montage
    source_positions = {s: standard_montage.get_positions()['ch_pos'][loc] for s, loc in sources.items()}
    detector_positions = {d: standard_montage.get_positions()['ch_pos'][loc] for d, loc in detectors.items()}

    eeg_coords={}        
    for channel in eeg_ch_names:
        if channel in standard_montage.ch_names:
            # Get the index of the EEG label in the standard montage
            idx = standard_montage.ch_names.index(channel)
            # Use this index to get the 3D coordinates from the standard montage
            coord = standard_montage.dig[idx+3]['r']  # Offset by 3 to skip the fiducials
            # Assign these coordinates to your NIRS channel
            eeg_coords[channel] = coord
        else:
            print("Label not found in standard montage.")

    # Combine source and detector positions
    montage_positions = {**source_positions, **detector_positions,**eeg_coords}
    mne.channels.make_dig_montage(ch_pos=montage_positions,nasion=(0, 0.1, 0), lpa=(-0.1, 0, 0), rpa=(0.1, 0, 0), coord_frame='head')
    print(montage_positions)

def epocking(raw_combined = any, tmin = -0.5, tmax = 1.5, event_id = None, mapping = None, stim_channel=['StimulusCode'], picks = None):
    """
    Epoch the combined data.

    Parameters:
    - raw_combined: The MNE Raw object containing the combined EEG and fNIRS data.
    - tmin: The start time of each epoch, in seconds. Default is -0.5.
    - tmax: The end time of each epoch, in seconds. Default is 1.5.
    - event_id: The event IDs and their corresponding descriptions. Default is {'deviant40': 2, 'standard1k': 1}.
    - mapping: The mapping of event IDs to their descriptions. Default is {2: 'deviant40', 1: 'standard1k'}.
    - stim_channel: The name of the stimulus channel. Default is ['StimulusCode'].
    - picks: The channels to be included in the epochs. Default is all channels.

    Returns:
    - epochs: The MNE Epochs object containing the epoched data.
    - labels: The labels of the epochs.

    Notes:
    The annotations are set based on the event IDs and their corresponding descriptions.
    The event IDs are set based on the stimulus channel and the event IDs and their corresponding descriptions.
    The labels are extracted from the event IDs.

    See also:
    mne.find_events, mne.Epochs
    """
    if event_id is None:
        event_id = dict(
                        deviant40=2,
                        standard1k=1
                        )
        mapping={2:'deviant40',
                 1:'standard1k'
                }
    #raw.set_eeg_reference("average")
    events = mne.find_events(raw_combined, stim_channel=stim_channel)
    onsets = events[:, 0] / raw_combined.info['sfreq']
    durations = np.zeros_like(onsets) 
    descriptions = [mapping[event_id] for event_id in events[:, 2]]
    annot_from_events = mne.Annotations(onset=onsets, duration=durations,
                                        description=descriptions,
                                        orig_time=raw_combined.info['meas_date'])
    raw_combined.set_annotations(annot_from_events)

    epochs = mne.Epochs(raw_combined, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True) #event_repeated='merge'

    labels = epochs.events[:, -1]
    
    return epochs, labels


