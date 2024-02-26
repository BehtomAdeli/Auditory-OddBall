This is the code to import the EEG-fNIRS fusion dataset into MNE, preprocess, epoch, and then feed it to a deep neural net using Torch.
Start from the main.py file and edit the variables as needed.
functions.py contains all the functions needed to load, import, merge, edit and or preprocess the eeg/fnirs data.
midfunctions.py is the secondry level functions that are called in main.py that would use  functions in functions.py.
EEG-fNIRS_Info.py is the class designed to hold info related to the database.
Please see the Wiki page for more information.
Thank you
