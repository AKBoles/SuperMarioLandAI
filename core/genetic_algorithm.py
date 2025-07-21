import numpy as np
import torch
import torch.nn as nn
from core.utils import mario_controls

# Network parameters
input_size = 400  # Approximate size for flattened game area (20x20)
hidden_size_1 = 80
hidden_size_2 = 20
output_size = 9  # Number of possible actions (from mario_controls())

# Evolution parameters
elitism_pct = 0.2
mutation_prob = 0.3
weights_init_min = -1
weights_init_max = 1
weights_mutate_power = 0.5

device = 'cpu'

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Build a simple feedforward network
        self.hidden1 = nn.Linear(input_size, hidden_size_1)
        self.hidden2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.uniform_(module.weight, a=weights_init_min, b=weights_init_max)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Ensure input is a tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Reshape if necessary
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        
        return x
    
    def get_action(self, x):
        """Get the best action for the given input"""
        with torch.no_grad():
            output = self.forward(x)
            action = torch.argmax(output, dim=-1)
            return action.item() if output.shape[0] == 1 else action.numpy()

class Genome:
    def __init__(self, network=None):
        if network is None:
            self.phenome = Network()
        else:
            self.phenome = network
        self.fitness = 0.0
    
    def copy(self):
        """Create a copy of this genome"""
        new_network = Network()
        new_network.load_state_dict(self.phenome.state_dict())
        return Genome(new_network)

class Population:
    def __init__(self, size=20):
        self.size = size
        self.genomes = [Genome() for _ in range(size)]
        self.fitnesses = np.zeros(size)
        self.generation = 0
    
    def evaluate(self, fitnesses):
        """Set fitness values for the population"""
        self.fitnesses = np.array(fitnesses)
        for i, genome in enumerate(self.genomes):
            genome.fitness = fitnesses[i]
    
    def create_new_generation(self):
        """Create a new generation using selection, crossover, and mutation"""
        # Sort genomes by fitness (descending)
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        
        new_genomes = []
        elite_size = int(self.size * elitism_pct)
        
        # Elitism: keep the best genomes
        for i in range(elite_size):
            new_genomes.append(self.genomes[sorted_indices[i]].copy())
        
        # Fill the rest with crossover and mutation
        while len(new_genomes) < self.size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            self._mutate(child)
            
            new_genomes.append(child)
        
        self.genomes = new_genomes
        self.generation += 1
        self.fitnesses = np.zeros(self.size)
    
    def _tournament_selection(self, tournament_size=3):
        """Select a parent using tournament selection"""
        # Choose random candidates
        candidates = np.random.choice(len(self.genomes), tournament_size, replace=False)
        # Return the best one
        best_idx = candidates[np.argmax(self.fitnesses[candidates])]
        return self.genomes[best_idx]
    
    def _crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        child = Genome()
        
        # Uniform crossover for each layer
        for child_param, p1_param, p2_param in zip(
            child.phenome.parameters(), 
            parent1.phenome.parameters(), 
            parent2.phenome.parameters()
        ):
            # Create a random mask for crossover
            mask = torch.rand_like(p1_param) > 0.5
            child_param.data = torch.where(mask, p1_param.data, p2_param.data)
        
        return child
    
    def _mutate(self, genome):
        """Mutate a genome"""
        for param in genome.phenome.parameters():
            # Add random noise to weights with some probability
            mutation_mask = torch.rand_like(param) < mutation_prob
            noise = torch.randn_like(param) * weights_mutate_power
            param.data += mutation_mask.float() * noise
    
    def get_best_genome(self):
        """Get the genome with the highest fitness"""
        best_idx = np.argmax(self.fitnesses)
        return self.genomes[best_idx]

def get_action(phenome, inputs):
    """
    Get action from neural network given game inputs
    
    Args:
        phenome: The neural network (Network instance)
        inputs: Flattened game area or state
    
    Returns:
        Action index (0-8 corresponding to mario_controls())
    """
    try:
        # Ensure inputs are the right size
        if len(inputs) > input_size:
            # Resize inputs if they're too large
            inputs = inputs[:input_size]
        elif len(inputs) < input_size:
            # Pad inputs if they're too small
            padded = np.zeros(input_size)
            padded[:len(inputs)] = inputs
            inputs = padded
        
        # Get action from the network
        action = phenome.get_action(inputs)
        
        # Ensure action is in valid range
        return int(action) % len(mario_controls())
        
    except Exception as e:
        print(f"Error in get_action: {e}")
        return 0  # Default action (no action)
