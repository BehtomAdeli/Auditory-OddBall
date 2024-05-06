import torch
import netfunctions as nf
import numpy as np 
import random

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

    return dataset


epochs_concat = np.load('lggnet/epochs_concat_all.npy')
labels = np.load('lggnet/labels.npy')

lggnet_params = torch.load('lggnet/lggnet_params.pth')

dataset = create_dataset(epochs_concat, labels, lggnet_params, threeD=False)

graph_general_eeg = [['F5','FC3'], ['F6', 'FC4'], ['Fpz','AFz'],['FCz','Cz'],
                    ['C6','TTP8','TPP8'],
                    ['C5','TTP7','TPP7'],
                    ['Pz','Cz']
                    ]

# Define the graph
chnames = ['Fpz', 'AFz', 'F5', 'F6', 'FCz', 'FC3', 'FC4', 'Cz', 'C5', 'C6', 'TTP7', 'TTP8', 'TPP7', 'TPP8', 'Pz',
            'StimulusCode', 'S1_D1 hbo', 'S2_D1 hbo', 'S2_D2 hbo', 'S3_D2 hbo', 'S4_D3 hbo', 'S4_D4 hbo', 'S4_D5 hbo',
              'S5_D4 hbo', 'S5_D5 hbo', 'S6_D6 hbo', 'S6_D7 hbo', 'S6_D8 hbo', 'S7_D7 hbo', 'S7_D8 hbo', 'S1_D1 hbr', 
              'S2_D1 hbr', 'S2_D2 hbr', 'S3_D2 hbr', 'S4_D3 hbr', 'S4_D4 hbr', 'S4_D5 hbr', 'S5_D4 hbr', 'S5_D5 hbr', 
              'S6_D6 hbr', 'S6_D7 hbr', 'S6_D8 hbr', 'S7_D7 hbr', 'S7_D8 hbr'
              ]

original_order = chnames
original_order.remove('StimulusCode')


def initialize_population(size, param_bounds):
    population = []
    for _ in range(size):
        individual = [round(random.uniform(bounds[0], bounds[1]),4) for key, bounds in param_bounds.items()]
        population.append(individual)
    return population

def evaluate_individual(individual):
    # Initialize the model with current parameters
    print(individual)
    optimize_params = {'learning_rate': 0.001, 'dropout_rate': 0.5, 'num_T': 64, 'out_graph': 4, 'pool': 16, 'pool_step_rate': 0.3}
    LGG = nf.initialize_lggnet_unity(chnames, lggnet_params=lggnet_params,optimize_params =optimize_params, graph_general=graph_general_eeg, cuda=True,threeD=True, temporal_windows = individual)
    
    # Define the loss function and optimizer with current learning rate
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LGG.parameters(), lr=lggnet_params['learning_rate'])
    
    # Train, validate, and test the model
    _, _, val_loss, _, _,_ = nf.run_model(LGG, dataset,test_loader=None, criterion=criterion, optimizer=optimizer,lggnet_params=lggnet_params, ifplot=False, iftest=False, kfolds=0  , ifresult=False,threeD=False)
    
    # Here, we use the mean of validation accuracy as the fitness score
    fitness_score = np.mean(val_loss)
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
        # Ensure we are choosing the midpoint for crossover
        midpoint = random.randint(1, len(parent1) - 1)  # Choose a crossover point
        child = parent1[:midpoint] + parent2[midpoint:]
        offspring.append(child)
    return offspring

def mutate(offspring, mutation_rate, param_bounds):
    for child in offspring:
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, len(child) - 1)  # Randomly choose which parameter to mutate
            # Mutate the chosen parameter by assigning a new value within bounds
            bounds = list(param_bounds.values())[mutation_index]
            child[mutation_index] = round(random.uniform(bounds[0], bounds[1]), 4)
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
                '0' : (0.01, 1.5),
                '1' : (0.01, 1.5),
                '2' : (0.01, 1.5),
                '3' : (0.01, 1.5),
                '4' : (0.01, 1.5),
                '5' : (0.01, 1.5),
                '6' : (0.01, 1.5),
                '7' : (0.01, 1.5)
                }



# Run GA
best_params = genetic_algorithm(param_bounds, population_size=300, num_generations=10, mutation_rate=0.1)

print("Best Parameters Found:", best_params)

with open('//home//behtom_uri_edu//LGGNet//best_params.txt', 'w') as file:
    for param, value in best_params.items():
        file.write(f'{param}: {value}\n')
