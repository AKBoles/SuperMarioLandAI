import numpy as np
import io
import torch
import logging
from datetime import datetime
import pickle
from core.genetic_algorithm import get_action, Population
from core.utils import do_action,move_right, move_left, jump, right_jump, left_jump
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

epochs = 5
population = None
run_per_child = 1
max_fitness = 0
pop_size = 10
max_score = 999999
#n_workers = cpu_count()
n_workers = 3

def eval_network(epoch, child_index, child_model):
  pyboy = PyBoy('SuperMarioLand.gb', game_wrapper=True) #, window_type="headless")
  pyboy.set_emulation_speed(3)
  mario = pyboy.game_wrapper()
  mario.start_game()
  #assert mario.lives_left == 0

  run = 0
  scores = []
  fitness_scores = []
  level_progress = []
  time_left = []
  #prev_action = np.asarray([0,0,0,0,0,0])

  while run < run_per_child:

    # do some things
    action = get_action(pyboy, mario, child_model)
    action = action.detach().numpy()
    action = np.where(action < np.max(action), 0, action)
    action = np.where(action == np.max(action), 1, action)
    action = action.astype(int)
    action = action.reshape((3,))
    #do_action(prev_action,action, pyboy) # simultaneous action is not implemented yet..
    do_action(action, pyboy)
    #prev_action = action

    # Game over:
    if mario.game_over() or mario.score == max_score:
      scores.append(mario.score)
      fitness_scores.append(mario.fitness)
      level_progress.append(mario.level_progress)
      time_left.append(mario.time_left)
      if run == run_per_child - 1:
        pyboy.stop()
      else:
        mario.reset_game()
      run += 1

  child_fitness = np.average(fitness_scores)
  logger.info("-" * 20)
  logger.info("Iteration %s - child %s" % (epoch, child_index))
  logger.info("Score: %s, Level Progress: %s, Time Left %s" % (scores, level_progress, time_left))
  logger.info("Fitness: %s" % child_fitness)
  #logger.info("Output weight:")
  #weights = {}
  #for i, j in zip(feature_names, child_model.output.weight.data.tolist()[0]):
  #  weights[i] = np.round(j, 3)
  #logger.info(weights)

  return child_fitness

if __name__ == '__main__':
  e = 0
  p = Pool(n_workers)

  while e < epochs:
    start_time = datetime.now()
    if population is None:
      if e == 0:
        population = Population(size=pop_size)
        print('Created first population!')
      else:
        with open('checkpoint/checkpoint-%s.pkl' % (e - 1), 'rb') as f:
          population = pickle.load(f)
    else:
      population = Population(size=pop_size, old_population=population)

    result = [0] * pop_size
    for i in range(pop_size):
      result[i] = p.apply_async(eval_network,(e, i, population.models[i]))

    for i in range(pop_size):
      population.fitnesses[i] = result[i].get()

    # Saving population
    with open('checkpoint/checkpoint-%s.pkl' % e, 'wb') as f:
      pickle.dump(population, f)

    if np.max(population.fitnesses) >= max_fitness:
      max_fitness = np.max(population.fitnesses)
      file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(np.round(max_fitness, 2))
      # Saving best model
      torch.save(population.models[np.argmax(population.fitnesses)].state_dict(),'models/%s' % file_name)
    e += 1

  #p.join()
  p.close()
  p.join()
