import mne
import torch
import midfunctions as mf
import netfunctions as nf

raw_combined = mne.io.read_raw_fif('C:\\Users\\behtu\\URI_Backup\\1.NeuroLab\Database\\Auditory Oddball Data\\Shared_Oddball_Raw\\Mat Files\\HSO_00\\MNE_ica01excluded.fif', preload=True)

event_id = dict(deviant40=2,standard1k=1)
mapping={2:'deviant40',1:'standard1k'}
picks = mne.pick_types(raw_combined.info, eeg=True,fnirs=False, stim=False, exclude='bads')
tmin =-0.5
tmax = 1.5

epochs_concat, labels , number_of_channels, input_length, Number_trials_per_class = mf.epoch_data(raw_combined, tmin =tmin, tmax =tmax, event_id =event_id, mapping = mapping, stim=['StimulusCode'] , picks=picks)

# Create LGGNet parameters
lggnet_params = {
                 'num_classes': 2, 
                 'batch_size': 10,
                 'splits': [0.7, 0.15,0.15],
                 'learning_rate': 0.001,
                 'num_epochs': 100,
                 'number_trials_per_class': Number_trials_per_class,
                 'number_of_channels': number_of_channels, 
                 'input_length': input_length, 
                 'num_T': 4, 
                 'out_graph': 6, 
                 'pool': 16, 
                 'pool_step_rate': 0.5, 
                 'dropout_rate': 0.5
                 }

dataset = mf.create_dataset(epochs_concat, labels, lggnet_params)

# Define the graph
original_order = raw_combined.ch_names
original_order.remove('StimulusCode')
"""
graph_general_DEAP = [['S1_D1 hbo','S1_D1 hbr', 'AFz', 'F5'], ['F6','S3_D2 hbo','S3_D2 hbr' , 'FC4'], ['AFz', 'FCz', 'S2_D1 hbo','S2_D1 hbr', 'S2_D2 hbo', 'S2_D2 hbr'],
                     ['C6', 'TTP8', 'S6_D6 hbo', 'S6_D7 hbo','S6_D6 hbr', 'S6_D7 hbr'], ['TPP7', 'S4_D5 hbo', 'S5_D5 hbo','S4_D5 hbr', 'S5_D5 hbr'],
                     ['C5', 'TTP7', 'S4_D3 hbo', 'S4_D4 hbo','S4_D3 hbr', 'S4_D4 hbr'],
                     ['TPP8','S6_D8 hbr', 'S7_D8 hbr','S6_D8 hbo', 'S7_D8 hbo'],
                     ['Pz'],['Fpz'], ['Cz']]"""

graph_general = [['AFz', 'F5'], ['F6', 'FC4'], ['AFz', 'FCz','Cz'],
                 ['C6', 'TTP8'],
                 ['C5', 'TTP7'],
                 ['Pz','Fpz']
                 ]
"""
#optimize_params = {'batch_size': 10, 'num_epochs': 150, 'dropout_rate': 0.10843596987181164, 'num_T': 1, 'out_graph': 4, 'pool': 16, 'pool_step_rate': 0.31535928497575016}
optimize_params = {'batch_size': 10, 'num_epochs': 50, 'dropout_rate': 0.1, 'num_T': 1, 'out_graph': 4, 'pool': 16, 'pool_step_rate': 0.3}

LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params=optimize_params, graph_general_DEAP=graph_general,cuda=True)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss() #LabelSmoothing(smoothing=0.1) #
optimizer = torch.optim.Adam(LGG.parameters(), lr=0.001)  # Adjust learning rate as needed

# Train,Validate, and Test the model
train_losses, val_losses, train_acc, val_acc = nf.run_model(LGG,dataset, criterion, optimizer,optimize_params, lggnet_params)
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
    LGG = nf.initialize_lggnet(raw_combined, lggnet_params=lggnet_params,optimize_params =individual, graph_general_DEAP=graph_general, cuda=True)
    
    # Define the loss function and optimizer with current learning rate
    criterion = torch.nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(LGG.parameters(), lr=0.001)
    
    # Train, validate, and test the model
    _, val_losses, _, val_acc, _, _ = nf.run_model(LGG, dataset, criterion, optimizer, individual,lggnet_params, ifplot=False, iftest=False)
    
    # Here, we use the mean of validation accuracy as the fitness score
    fitness_score = np.mean(val_acc)+np.mean(val_losses)
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
    'batch_size': [1,2, 10, 20],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7,0.8, 0.9],
    'num_T': [1,2,3,4,5,6], 
    'out_graph': [2,4,6], 
    'pool': [4,8,16,32,64,128], 
    'pool_step_rate': [0.1, 0.2,0.3, 0.4, 0.5, 0.7,0.8, 0.9]
}

# Run GA
best_params = genetic_algorithm(param_bounds, population_size=10, num_generations=30, mutation_rate=0.1)
print("Best Parameters Found:", best_params)

