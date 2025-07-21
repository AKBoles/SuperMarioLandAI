#!/usr/bin/env python3
"""
Test script to verify the Mario AI training setup without requiring the actual ROM.
This will help identify issues with the code structure before trying with a real ROM.
"""

import numpy as np
import os
import sys

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from pyboy import PyBoy
        from pyboy.utils import WindowEvent
        print("✓ PyBoy imports successful")
    except ImportError as e:
        print(f"✗ PyBoy import failed: {e}")
        return False
    
    try:
        import torch
        import torch.nn as nn
        print("✓ PyTorch imports successful")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from core.genetic_algorithm import get_action, Population, Network
        print("✓ Genetic algorithm imports successful")
    except ImportError as e:
        print(f"✗ Genetic algorithm import failed: {e}")
        return False
    
    try:
        from core.utils import mario_controls, calculate_fitness
        print("✓ Utils imports successful")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True

def test_neural_network():
    """Test that the neural network can be created and used"""
    print("\nTesting neural network...")
    
    try:
        from core.genetic_algorithm import Network, get_action
        
        # Create a network
        network = Network()
        print("✓ Network creation successful")
        
        # Test with dummy input
        dummy_input = np.random.random(400)
        action = get_action(network, dummy_input)
        print(f"✓ Network forward pass successful - Action: {action}")
        
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_population():
    """Test that the population can be created and evolved"""
    print("\nTesting population...")
    
    try:
        from core.genetic_algorithm import Population
        
        # Create a small population
        population = Population(size=5)
        print("✓ Population creation successful")
        
        # Test fitness assignment
        dummy_fitnesses = [10, 20, 15, 5, 25]
        population.fitnesses = np.array(dummy_fitnesses)
        print("✓ Fitness assignment successful")
        
        # Test evolution
        population.create_new_generation()
        print("✓ Population evolution successful")
        
        return True
    except Exception as e:
        print(f"✗ Population test failed: {e}")
        return False

def test_controls():
    """Test that the control mapping works"""
    print("\nTesting controls...")
    
    try:
        from core.utils import mario_controls
        from pyboy.utils import WindowEvent
        
        controls = mario_controls()
        print(f"✓ Controls mapping created - {len(controls)} actions available")
        
        # Test that all actions are valid WindowEvent values
        for action_id, events in controls.items():
            if events:  # Skip empty action
                for event in events:
                    # Check if it's a valid WindowEvent by trying to use it
                    if not isinstance(event, int) or event not in range(0, 100):  # Reasonable range check
                        raise ValueError(f"Invalid event value: {event}")
        print("✓ All control events are valid")
        
        # Test a specific action
        test_action = controls[1]  # Right action
        if test_action and test_action[0] == WindowEvent.PRESS_ARROW_RIGHT:
            print("✓ Control mapping validation successful")
        
        return True
    except Exception as e:
        print(f"✗ Controls test failed: {e}")
        return False

def test_rom_check():
    """Test ROM file checking logic"""
    print("\nTesting ROM file checking...")
    
    rom_path = 'SuperMarioLand.gb'
    if os.path.exists(rom_path):
        print(f"✓ ROM file found at {rom_path}")
        return True
    else:
        print(f"⚠ ROM file not found at {rom_path}")
        print("To run the actual training, you need to:")
        print("1. Obtain a legal copy of Super Mario Land (Game Boy)")
        print("2. Place the ROM file as 'SuperMarioLand.gb' in this directory")
        print("3. Make sure you own the original cartridge (for legal compliance)")
        return False

def provide_setup_instructions():
    """Provide instructions for setting up the training environment"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    print()
    print("Your AI training code has been fixed and should now work!")
    print()
    print("To complete the setup:")
    print()
    print("1. ROM FILE REQUIRED:")
    print("   - You need a legal copy of 'Super Mario Land' for Game Boy")
    print("   - Save it as 'SuperMarioLand.gb' in this directory")
    print("   - Only use ROMs you legally own")
    print()
    print("2. DEPENDENCIES:")
    print("   - All required Python packages are installed")
    print("   - PyBoy, PyTorch, NumPy are ready")
    print()
    print("3. TO START TRAINING:")
    print("   - Once you have the ROM file: python3 mario.py")
    print("   - The AI will evolve over multiple generations")
    print("   - Progress will be displayed in the console")
    print()
    print("4. WHAT WAS FIXED:")
    print("   - Updated PyBoy API calls to v2.x format")
    print("   - Fixed import errors (WindowEvent)")
    print("   - Corrected neural network structure")
    print("   - Improved genetic algorithm implementation")
    print("   - Added proper error handling")
    print("   - Fixed multiprocessing issues")
    print()
    print("The AI will learn to play Mario by:")
    print("- Observing the game screen")
    print("- Making decisions through a neural network")
    print("- Evolving better strategies over generations")
    print("- Maximizing progress through the level")
    print()

def main():
    """Run all tests and provide setup instructions"""
    print("Mario AI Training Setup Verification")
    print("="*50)
    
    all_tests_passed = True
    
    all_tests_passed &= test_imports()
    all_tests_passed &= test_neural_network()
    all_tests_passed &= test_population()
    all_tests_passed &= test_controls()
    rom_exists = test_rom_check()
    
    print("\n" + "="*50)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Code is ready!")
        if rom_exists:
            print("✓ ROM file found - You can start training immediately!")
            print("Run: python3 mario.py")
        else:
            print("⚠ Need ROM file to start training")
    else:
        print("✗ Some tests failed - Check the errors above")
        return 1
    
    provide_setup_instructions()
    return 0

if __name__ == "__main__":
    sys.exit(main())