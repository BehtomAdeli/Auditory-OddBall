This is the code to import the EEG-fNIRS fusion dataset into MNE, preprocess, epoch, and then feed it to a deep neural net using Torch.
Start from the main.py file and edit the variables as needed.
functions.py contains all the functions needed to load, import, merge, edit and or preprocess the eeg/fnirs data.
midfunctions.py is the secondry level functions that are called in main.py that would use  functions in functions.py.
EEG-fNIRS_Info.py is the class designed to hold info related to the database.
If you want to use this code you should start with the preloader.py to import your EEG and or fNIRS data and save it. Then you can go to the main-ga.py to find the best hyperparameters foryour problem or go directly to the 5x5ncv.py. Bare in mind that main-gy uses genetic algorithm and you can change the hyperpararmeters at the end of the code or the number of population, mutation and much more. But all of them are time-gpu consuming. You need 30xx or 40xx RTX series or equivalant. 
Please see the Wiki page for more information.
Thank you
