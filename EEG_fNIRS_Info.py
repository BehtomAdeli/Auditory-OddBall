class EEG_fNIRS_Info:
    def __init__(self, eeg_dir=None, eeg_ch_names=None, eeg_ch_numbers=None,eeg_key = None ,
                  eeg_stimulus_no=None,eeg_stimulus_name=None, eeg_lowcut=None, eeg_highcut=None, eeg_notch=None, eeg_fs=None, 
                  hbo_dir=None, hbr_dir=None, hbo_fs=None, hbr_fs=None, hbo_lowcut=None, hbr_lowcut=None,  source_locations=None, detector_locations=None,
                  hbo_highcut=None, hbr_highcut=None, hbo_ch_names=None, hbr_ch_names=None, ica = None, ica_exclude = None,
                  subject_id = 00, subject_sex = 'M',subject_hand = 'right', exprimenter = 'NeuralPC Lab URI', experiment_description = 'Auditory Oddball Task'):
        self.eeg_dir = eeg_dir
        self.eeg_ch_names = eeg_ch_names
        self.eeg_ch_numbers = slice(eeg_ch_numbers[0],eeg_ch_numbers[1],1) if eeg_ch_numbers is not None else [1,15]
        self.eeg_key = eeg_key
        self.eeg_stimulus_no = eeg_stimulus_no
        self.eeg_stimulus_name = eeg_stimulus_name
        self.eeg_lowcut = eeg_lowcut
        self.eeg_highcut = eeg_highcut
        self.eeg_notch = eeg_notch
        self.eeg_fs = eeg_fs
        self.hbo_dir = hbo_dir
        self.hbr_dir = hbr_dir
        self.hbo_fs = hbo_fs
        self.hbr_fs = hbr_fs
        self.hbo_lowcut = hbo_lowcut
        self.hbr_lowcut = hbr_lowcut
        self.hbo_highcut = hbo_highcut
        self.hbr_highcut = hbr_highcut
        self.hbo_ch_names = hbo_ch_names
        self.hbr_ch_names = hbr_ch_names
        self.sources = self.set_fnirs_source_locations(source_locations)
        self.detectors = self.set_fnirs_detector_locations(detector_locations)
        self.subject_id = subject_id
        self.subject_sex = subject_sex
        self.subject_hand = self.set_hand(subject_hand)
        self.exprimenter = exprimenter
        self.experiment_description = experiment_description
        self.ica = ica
        self.ica_exclude = ica_exclude
    
    # EEG Setters
    def set_eeg_dir(self, value):
        self.eeg_dir = value

    def set_eeg_ch_names(self, value):
        self.eeg_ch_names = value

    def set_eeg_ch_numbers(self, value):
        self.eeg_ch_numbers = value

    def set_eeg_key(self, value):
        self.eeg_key = value

    def set_eeg_stimulus(self, value):
        self.eeg_stimulus = value

    def set_eeg_lowcut(self, value):
        self.eeg_lowcut = value

    def set_eeg_highcut(self, value):
        self.eeg_highcut = value

    def set_eeg_notch(self, value):
        self.eeg_notch = value

    def set_eeg_fs(self, value):
        self.eeg_fs = value

    # HbO Setters
    def set_hbo_dir(self, value):
        self.hbo_dir = value

    def set_hbo_fs(self, value):
        self.hbo_fs = value

    def set_hbo_lowcut(self, value):
        self.hbo_lowcut = value

    def set_hbo_highcut(self, value):
        self.hbo_highcut = value

    def set_hbo_ch_names(self, value):
        self.hbo_ch_names = value

    # HbR Setters
    def set_hbr_dir(self, value):
        self.hbr_dir = value

    def set_hbr_fs(self, value):
        self.hbr_fs = value

    def set_hbr_lowcut(self, value):
        self.hbr_lowcut = value

    def set_hbr_highcut(self, value):
        self.hbr_highcut = value

    def set_hbr_ch_names(self, value):
        self.hbr_ch_names = value
    
    def set_sources(self, value):
        self.sources = self.set_fnirs_source_locations(self,value)
    
    def set_detectors(self, value):
        self.detectors = self.set_fnirs_detector_locations(self,value)
    
    def set_subject_id(self, value):
        self.subject_id = value
    
    def set_subject_sex(self, value):
        self.subject_sex = value
    
    def set_subject_hand(self, value):
        self.subject_hand = self.set_hand(self,value)

    def set_exprimenter(self, value):
        self.exprimenter = value
    
    def set_experiment_description(self, value):
        self.experiment_description = value


    # Print
    def print_info(self):
        attributes = [
            ('EEG Directory', self.eeg_dir),
            ('EEG Channel Names', self.eeg_ch_names),
            ('EEG Channel Numbers', self.eeg_ch_numbers),
            ('EEG Directory key', self.eeg_key),
            ('EEG Stimulus', self.eeg_stimulus),
            ('EEG Low Cut Frequency', self.eeg_lowcut),
            ('EEG High Cut Frequency', self.eeg_highcut),
            ('EEG Notch Filter Setting', self.eeg_notch),
            ('EEG Sampling Frequency', self.eeg_fs),
            ('HbO Directory', self.hbo_dir),
            ('HbR Directory', self.hbr_dir),
            ('HbO Sampling Frequency', self.hbo_fs),
            ('HbR Sampling Frequency', self.hbr_fs),
            ('HbO Low Cut Frequency', self.hbo_lowcut),
            ('HbR Low Cut Frequency', self.hbr_lowcut),
            ('HbO High Cut Frequency', self.hbo_highcut),
            ('HbR High Cut Frequency', self.hbr_highcut),
            ('HbO Channel Names', self.hbo_ch_names),
            ('HbR Channel Names', self.hbr_ch_names)
            ('Subject ID', self.subject_id),
            ( 'Subject Sex', self.subject),
            ('Subject Hand', self.subject_hand),
            ('Experimenter', self.exprimenter),
            ('Experiment Description', self.experiment_description)
        ]

        for attr_name, attr_value in attributes:
            if attr_value is not None:
                print(f"{attr_name}: {attr_value}")
        
    def print_info2(self):
        for attribute, value in self.__dict__.items():
            if value is not None:
                print(f"{attribute}: {value}")

    def set_fnirs_source_locations(self,source_locations=None):
        source_dict = {}
        for i, loc in enumerate(source_locations, start=1):
            source_dict[f'S{i}'] = loc
        return source_dict
    
    def set_fnirs_detector_locations(self,detector_locations=None):
        detector_dict = {}
        for i, loc in enumerate(detector_locations, start=1):
            detector_dict[f'D{i}'] = loc
        return detector_dict
    
    def set_hand(self,hand):
        if isinstance(hand,str):
            if hand.lower() == 'right':
                return 1
            elif hand.lower() == 'left':
                return 0
            else:
                return 1
        elif isinstance(hand,int):
            if hand == 1:
                return 1
            elif hand == 0:
                return 0
            else:
                return 1