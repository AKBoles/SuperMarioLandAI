#!/usr/bin/env python3

from multiprocessing import Pool
import numpy as np
from core.genetic_algorithm import get_action, Population
from core.utils import *
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Create a downloadable demo ROM path or use a fallback
ROM_PATH = 'SuperMarioLand.gb'

# If ROM doesn't exist, we'll create a simple fallback
import os
if not os.path.exists(ROM_PATH):
    print("Warning: SuperMarioLand.gb not found. You need to provide a Game Boy ROM file.")
    print("Please place 'SuperMarioLand.gb' in the current directory.")
    print("Exiting...")
    exit(1)

def eval_network(phenome):
    pyboy = None
    try:
        # Initialize PyBoy with correct parameters for v2.x
        pyboy = PyBoy(ROM_PATH, window='null', sound_emulated=False)
        
        # Start the game properly
        pyboy.game_wrapper.start_game()
        
        # Skip the intro sequence (optional but recommended for training)
        for _ in range(100):
            pyboy.tick()
            
        fitness = 0
        max_steps = 1000  # Limit steps to prevent infinite loops
        steps = 0
        
        while not pyboy.game_wrapper.game_over() and steps < max_steps:
            # Get the current game state
            game_area = pyboy.game_area()
            
            # Flatten the game area for the neural network
            inputs = game_area.flatten()
            
            # Get action from the neural network
            action = get_action(phenome, inputs)
            
            # Map action to game controls
            controls = mario_controls()
            if action in controls:
                # Release all buttons first
                for event in [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, 
                             WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B]:
                    pyboy.send_input(event)
                
                # Press the chosen buttons
                for event in controls[action]:
                    pyboy.send_input(event)
            
            # Advance the game by a few frames
            for _ in range(4):  # Hold input for 4 frames
                pyboy.tick()
            
            # Calculate fitness based on Mario's progress
            try:
                mario_x = pyboy.memory[0xC202]  # Mario's X position in memory
                fitness = max(fitness, mario_x)  # Track furthest progress
            except:
                # Fallback fitness calculation
                fitness = steps * 0.1
            
            steps += 1
            
        return fitness
        
    except Exception as e:
        print(f"Error in eval_network: {e}")
        return 0
    finally:
        if pyboy:
            pyboy.stop()

def main():
    population = Population()
    
    print("Created first population!")
    
    # Use a smaller number of processes to avoid resource conflicts
    num_processes = min(4, os.cpu_count())
    
    for generation in range(100):  # Run for 100 generations
        print(f"Generation {generation}")
        
        # Evaluate fitness for each genome in the population
        with Pool(processes=num_processes) as pool:
            # Submit all networks for evaluation
            results = []
            for i, genome in enumerate(population.genomes):
                result = pool.apply_async(eval_network, (genome.phenome,))
                results.append(result)
            
            # Collect results
            for i, result in enumerate(results):
                try:
                    population.fitnesses[i] = result.get(timeout=60)  # 60 second timeout
                except Exception as e:
                    print(f"Error evaluating genome {i}: {e}")
                    population.fitnesses[i] = 0
        
        # Check for completion
        max_fitness = max(population.fitnesses) if population.fitnesses else 0
        avg_fitness = sum(population.fitnesses) / len(population.fitnesses) if population.fitnesses else 0
        
        print(f"Max fitness: {max_fitness:.2f}, Average fitness: {avg_fitness:.2f}")
        
        # Create next generation
        population.create_new_generation()
        
        # Optional: Save best genome periodically
        if generation % 10 == 0:
            print(f"Saving checkpoint at generation {generation}")
            # You could save the population state here
            
    print("Training completed!")

if __name__ == "__main__":
    main()
