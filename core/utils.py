import numpy as np
from pyboy import WindowEvent

# Action map
action_map = {
  'Left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
  'Right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
  'Up': [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
  'Down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
  'A': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
  'B': [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B]
}

do_action_map = {
  0 : [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
  1 : [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
  2 : [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
  3 : [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
  4 : [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
  5 : [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B]
}

# List of Enemies
enemies = {
  'goomba': [144],
  'koopa': [150, 151, 152, 153],
  'plant': [146, 147, 148, 149],
  'moth': [160, 161, 162, 163, 176, 177, 178, 179],
  'flying_moth': [192, 193, 194, 195, 208, 209, 210, 211],
  'sphinx': [164, 165, 166, 167, 180, 181, 182, 183],
  'big_sphinx': [198, 199, 201, 202, 203, 204, 205, 214, 215, 217, 218, 219],
  'fist': [240, 241, 242, 243],
  'bill': [249],
  'projectiles': [172, 188, 196, 197, 212, 213, 226, 227],
  'shell': [154, 155],
  'explosion': [157, 158],
  'spike': [237]
}

# solid blocks
solid_blocks = [
129, 142, 143, 221, 222, 231, 232, 233, 234, 235, 236, 301, 302, 303, 304, 319, 340, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362, 381, 382, 383
]
pipes = list(range(368, 381))
solid_blocks.extend(pipes)

# following functions help mario move

def do_action(action, pyboy):
  # given action output (format: [0,0,0,0,0,0] where a 0 will be replaced with a 1 for an action)
  #test[np.where(a == 1)[0].item(0)]
  pyboy.send_input(do_action_map[np.where(action == 1)[0].item(0)][0])
  ticks = 30
  while ticks > 0:
    pyboy.tick()
    ticks = ticks - 1
  pyboy.send_input(do_action_map[np.where(action == 1)[0].item(0)][1])
  pyboy.tick()

def move_right(pyboy):
  pyboy.send_input(action_map['Right'][0])
  pyboy.tick()
  pyboy.send_input(action_map['Right'][1])
  pyboy.tick()

def move_left(pyboy):
  pyboy.send_input(action_map['Left'][0])
  pyboy.tick()
  pyboy.send_input(action_map['Left'][1])
  pyboy.tick()

def jump(pyboy):
  pyboy.send_input(action_map['A'][0])
  ticks = 60
  while ticks > 0:
    pyboy.tick()
    ticks = ticks - 1
  pyboy.send_input(action_map['A'][1])
  pyboy.tick()

def right_jump(pyboy):
  pyboy.send_input(action_map['Right'][0])
  jump(pyboy)
  pyboy.send_input(action_map['Right'][1])
  pyboy.tick()

def left_jump(pyboy):
  pyboy.send_input(action_map['Left'][0])
  jump(pyboy)
  pyboy.send_input(action_map['Left'][1])
  pyboy.tick()


# following functions are for getting screen information

def convert_area(area, pyboy):
  # mario is a 2
  mario_loc = get_mario(pyboy)
  area[mario_loc[1]][mario_loc[0]] = 2
  # solid blocks are a 1
  for value in solid_blocks:
    area = np.where((area == value), 1, area)
  # enemy blocks are a -1
  for value in sum(enemies.values(), []):
    area = np.where((area == value), -1, area)
  # everything else is empty space
  area = np.where(area > 2, 0, area)
  # now need to only return array with cols > col where mario is --> only area in front of mario
#  return area[:, mario_loc[0]:]
  return area

def get_loc_enemies(area):
  enemy_loc = sum(enemies.values(),[])
  loc = []
  for value in enemy_loc:
    if np.where(area == value)[0].size > 0:
      loc.append(np.asarray(np.where(area == value)))
  return loc

def get_mario(pyboy):
  # get mario location directly from memory
  mario_x = pyboy.get_memory_value(0xC202)
  mario_y = pyboy.get_memory_value(0xC201)
  mario_x = int((mario_x - 8) / 8)
  mario_y = int((mario_y - 24) / 8)
  # return as a [x,y] list of coordinates
  # note: x --> col, y--> row
  return([mario_x, mario_y])

