import numpy as np
import io
import os
import torch
import logging
import neat
from datetime import datetime
import pickle
from core.genetic_algorithm import get_action_neat, Population
from core.utils import do_action, do_action_multiple, fitness_calc
from pyboy import PyBoy
from multiprocessing import Pool, cpu_count

# logging information
logger = logging.getLogger('mario')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs.out')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

epochs = 50
population = None
run_per_child = 1
max_fitness = 0
pop_size = 10
max_score = 999999
#n_workers = cpu_count()
n_workers = 10

def eval_genome(genome, config):
  global max_fitness

  pyboy = PyBoy('SuperMarioLand.gb', game_wrapper=True) #, window_type="headless")
  pyboy.set_emulation_speed(0)
  mario = pyboy.game_wrapper()
  mario.start_game()
  # start off with just one life each
  mario.set_lives_left(0)
 
  run = 0
  scores = []
  fitness_scores = []
  level_progress = []
  time_left = []

  # create NN from neat
  model = neat.nn.FeedForwardNetwork.create(genome, config)
  child_fitness = 0

  while run < run_per_child:

    # do some things
    action = get_action_neat(pyboy, mario, model)
    action = np.asarray([np.mean(val) for val in action])
    action = np.where(action < np.max(action), 0, action)
    action = np.where(action == np.max(action), 1, action)
    action = action.astype(int)
    action = action.reshape((2,))
    do_action(action, pyboy)

    # Game over:
    if mario.game_over() or mario.score == max_score:
      scores.append(mario.score)
      fitness_scores.append(fitness_calc(mario.score, mario.level_progress, mario.time_left))
      level_progress.append(mario.level_progress)
      time_left.append(mario.time_left)
      if run == run_per_child - 1:
        pyboy.stop()
      else:
        mario.reset_game()
      run += 1

  child_fitness = np.average(fitness_scores)
  #logger.info("-" * 20)
  #logger.info("Iteration %s - child %s" % (epoch, child_index))
  #logger.info("Score: %s, Level Progress: %s, Time Left %s" % (scores, level_progress, time_left))
  #logger.info("Fitness: %s" % child_fitness)

  return child_fitness

def run(config_path):
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

  p = neat.Population(config)
  # Uncomment to load from checkpoint
  #p = neat.Checkpointer().restore_checkpoint('checkpoint/neat-checkpoint-2')
  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)
  p.add_reporter(neat.Checkpointer(1, filename_prefix='neat_checkpoint/neat-checkpoint-'))

  pe = neat.ParallelEvaluator(n_workers, eval_genome)
  winner = p.run(pe.evaluate, epochs)

  # show final stats
  print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, 'config', 'config-neat.txt')
  run(config_path)
