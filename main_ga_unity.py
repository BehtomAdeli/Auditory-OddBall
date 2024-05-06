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


epochs_concat = np.load('epochs_concat.npy')
labels = np.load('labels.npy')

lggnet_params = torch.load('lggnet_params.pth')

dataset = create_dataset(epochs_concat, labels, lggnet_params, threeD=False)

graph_general_eeg = [['AFz', 'F5'], ['F6', 'FC4'], ['AFz', 'FCz','Cz'],
                    ['C6', 'TTP8'],
                    ['C5', 'TTP7'],
                    ['Pz','Fpz']
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
        individual = {key: random.uniform(bounds[0], bounds[1]) if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else random.choice(bounds) for key, bounds in param_bounds.items()}
        population.append(individual)
    return population

def evaluate_individual(individual):
    # Initialize the model with current parameters
    print(individual)
    LGG = nf.initialize_lggnet_unity(chnames, lggnet_params=lggnet_params,optimize_params =individual, graph_general=graph_general_eeg, cuda=True,threeD=False)
    
    # Define the loss function and optimizer with current learning rate
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LGG.parameters(), lr=lggnet_params['learning_rate'])
    
    # Train, validate, and test the model
    _, _, val_loss, _, _,_ = nf.run_model(LGG, dataset,test_loader=None, criterion=criterion, optimizer=optimizer,lggnet_params=lggnet_params, ifplot=False, iftest=False, kfolds=5, ifresult=False,threeD=False)
    
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
    'num_T': [1,2,4,8,16,32,64,128], 
    'out_graph': [1,2,3,4,5,6,7,8], 
    'pool': [4,8,16,32,64,128], 
    'pool_step_rate': [0.2,0.3, 0.4, 0.5, 0.7,0.8, 0.9]
}

# Run GA
best_params = genetic_algorithm(param_bounds, population_size=300, num_generations=30, mutation_rate=0.1)

print("Best Parameters Found:", best_params)

with open('//home//behtom_uri_edu//LGGNet//best_params.txt', 'w') as file:
    for param, value in best_params.items():
        file.write(f'{param}: {value}\n')
