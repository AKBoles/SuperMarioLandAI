import numpy as np
from pyboy.utils import WindowEvent

# Action Map
do_action_map_all = {

# following functions are for getting screen information

def convert_area(area_mapping, screen_area):
    """
    Converts the game's screen area based on the provided area mapping.
    
    Args:
        area_mapping: A mapping dictionary or list for converting game screen values
        screen_area: The current game screen area as a numpy array
    
    Returns:
        Converted screen area based on the mapping
    """
    if area_mapping is None:
        return screen_area
    
    if isinstance(area_mapping, dict):
        # Convert using dictionary mapping
        converted = np.copy(screen_area)
        for original, mapped in area_mapping.items():
            converted[screen_area == original] = mapped
        return converted
    elif isinstance(area_mapping, (list, np.ndarray)):
        # Convert using array indexing
        return np.array([area_mapping[val] if val < len(area_mapping) else val for val in screen_area.flatten()]).reshape(screen_area.shape)
    else:
        # No valid mapping provided, return original
        return screen_area

def get_screen_area(pyboy_instance, x_start=0, y_start=0, width=None, height=None):
    """
    Extract a specific area of the game screen.
    
    Args:
        pyboy_instance: Instance of PyBoy emulator
        x_start: Starting x coordinate (default: 0)
        y_start: Starting y coordinate (default: 0)
        width: Width of area to extract (default: full width)
        height: Height of area to extract (default: full height)
    
    Returns:
        Extracted screen area as numpy array
    """
    game_area = pyboy_instance.game_area()
    
    if width is None:
        width = game_area.shape[1] - x_start
    if height is None:
        height = game_area.shape[0] - y_start
    
    return game_area[y_start:y_start+height, x_start:x_start+width]

def get_mario_position(pyboy_instance):
    """
    Get Mario's position from memory.
    
    Args:
        pyboy_instance: Instance of PyBoy emulator
    
    Returns:
        Tuple of (x, y) position
    """
    try:
        # For Super Mario Land, these memory addresses typically contain Mario's position
        mario_x = pyboy_instance.memory[0xC202]
        mario_y = pyboy_instance.memory[0xC201]
        return mario_x, mario_y
    except:
        # Fallback if memory addresses don't work
        return 0, 0

def get_mario_level_progress(pyboy_instance):
    """
    Get Mario's level progress from memory.
    
    Args:
        pyboy_instance: Instance of PyBoy emulator
    
    Returns:
        Level progress value
    """
    try:
        # This is an approximation - the actual memory address may vary
        return pyboy_instance.memory[0xC202]
    except:
        return 0

def calculate_fitness(pyboy_instance, prev_fitness=0):
    """
    Calculate a basic fitness score for Mario based on his x position.
    
    Args:
        pyboy_instance: Instance of PyBoy emulator
        prev_fitness: Previous fitness score
    
    Returns:
        Current fitness score
    """
    # This is a basic fitness function - you might want to customize this
    # based on Mario's position, score, lives, etc.
    
    # Try to get Mario's x position from memory (address varies by game)
    # For Super Mario Land, Mario's x position is typically around 0xC201-0xC203
    try:
        mario_x = pyboy_instance.memory[0xC202]  # This might need adjustment
        fitness = mario_x + prev_fitness * 0.9  # Small momentum from previous fitness
        return max(fitness, prev_fitness)  # Ensure fitness doesn't decrease much
    except:
        # Fallback: just use a basic scoring mechanism
        return prev_fitness + 1

def mario_controls():
    """
    Returns a dictionary mapping action indices to PyBoy WindowEvent controls.
    
    Returns:
        Dictionary with action mappings
    """
    return {
        0: [],  # No action
        1: [WindowEvent.PRESS_ARROW_RIGHT],  # Move right
        2: [WindowEvent.PRESS_ARROW_LEFT],   # Move left  
        3: [WindowEvent.PRESS_BUTTON_A],     # Jump
        4: [WindowEvent.PRESS_BUTTON_B],     # Run/Fire
        5: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],  # Right + Jump
        6: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],   # Left + Jump
        7: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],  # Right + Run
        8: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],   # Left + Run
    }

# Legacy functions for compatibility (if needed by other parts)
def get_mario(pyboy):
    """Legacy function - get mario location directly from memory"""
    try:
        mario_x = pyboy.memory[0xC202]
        mario_y = pyboy.memory[0xC201]
        return [mario_x, mario_y]
    except:
        return [0, 0]

def fitness_calc(score, level_progress, time_left):
    """Legacy fitness calculation function"""
    level_progress = level_progress - 250
    return level_progress**2
