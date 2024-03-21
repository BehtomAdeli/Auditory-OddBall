import mne
import torch
import midfunctions as mf
import netfunctions as nf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

raw_combined = mne.io.read_raw_fif('C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\MNE_ica01excluded-All.fif', preload=True)

event_id = dict(deviant40=2,standard1k=1)
mapping={2:'deviant40',1:'standard1k'}
picks = mne.pick_types(raw_combined.info, eeg=True,fnirs=False, stim=False, exclude='bads')
tmin =-0.5
tmax = 1.5

epochs_concat, labels , number_of_channels, input_length, Number_trials_per_class = mf.epoch_data(raw_combined, tmin =tmin, tmax =tmax, event_id =event_id, mapping = mapping, stim=['StimulusCode'] , picks=picks)

# Create LGGNet parameters
lggnet_params = {
                 'learning_rate': 0.001,
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

"""
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


    LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params=optimize_params, graph_general_DEAP=graph_general_eeg,cuda=True)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss() #LabelSmoothing(smoothing=0.1) #
    optimizer = torch.optim.Adam(LGG.parameters(), lr=optimize_params['learning_rate'])  

    # Train,Validate, and Test the model
    train_losses, val_losses, train_acc, val_acc,test_loss, test_acc = nf.run_model(LGG,train_subsampler,test_loader, criterion, optimizer, lggnet_params,ifplot=False, iftest=True)
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

"""

import numpy as np 
import random

def initialize_population(size, param_bounds):
    population = []
    for _ in range(size):
        individual = {key: random.uniform(bounds[0], bounds[1]) if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else random.choice(bounds) for key, bounds in param_bounds.items()}
        population.append(individual)
    return population

def evaluate_individual(individual):
    # Initialize the model with current parameters
    print(individual)
    LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params =individual, graph_general=graph_general_eeg, cuda=True)
    
    # Define the loss function and optimizer with current learning rate
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LGG.parameters(), lr=lggnet_params['learning_rate'])
    
    # Train, validate, and test the model
    _, val_losses, _, val_acc, _, _ = nf.run_model(LGG, dataset,test_loader=None, criterion=criterion, optimizer=optimizer,lggnet_params=lggnet_params, ifplot=False, iftest=False, kfolds=2, ifresult=False)
    
    # Here, we use the mean of validation accuracy as the fitness score
    fitness_score = np.mean(val_losses)
    return fitness_score

def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        max_fitness_idx = np.argmax(fitness_scores)
        parents.append(population[max_fitness_idx])
        fitness_scores[max_fitness_idx] = -np.inf  # Exclude this index in the next iteration
    return parents

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = {key: parent1[key] if random.random() < 0.5 else parent2[key] for key in parent1}
        offspring.append(child)
    return offspring

def mutate(offspring, mutation_rate, param_bounds):
    for child in offspring:
        if random.random() < mutation_rate:
            mutation_key = random.choice(list(child.keys()))
            child[mutation_key] = random.uniform(param_bounds[mutation_key][0], param_bounds[mutation_key][1]) if isinstance(param_bounds[mutation_key], (list, tuple)) and len(param_bounds[mutation_key]) == 2 else random.choice(param_bounds[mutation_key])
    return offspring

def genetic_algorithm(param_bounds, population_size, num_generations, mutation_rate):
    population = initialize_population(population_size, param_bounds)
    for generation in range(num_generations):
        fitness_scores = [evaluate_individual(individual) for individual in population]
        parents = select_parents(population, fitness_scores, population_size // 2)
        offspring = crossover(parents, population_size - len(parents))
        offspring = mutate(offspring, mutation_rate, param_bounds)
        population = parents + offspring
    best_individual_idx = np.argmax([evaluate_individual(individual) for individual in population])
    return population[best_individual_idx]

# Define parameter bounds
param_bounds = {
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7,0.8, 0.9],
    'num_T': [1,2,3,4,5,6,8,10], 
    'out_graph': [1,2,3,4,5,6], 
    'pool': [4,8,16,32,64,128], 
    'pool_step_rate': [0.2,0.3, 0.4, 0.5, 0.7,0.8, 0.9]
}

# Run GA
best_params = genetic_algorithm(param_bounds, population_size=100, num_generations=30, mutation_rate=0.1)
print("Best Parameters Found:", best_params)
