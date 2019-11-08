import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 
                '../gym_subgridworld/envs')
                )
from subgridworld_env import SubGridWorldEnv

# Example: 10 x 10 grid walls
PLANE_WALLS = [[0, x] for x in [5]] + [[1, x] for x in [0,1,4,5,7,8]] \
                + [[2, x] for x in [1,2,4,7]] + [[3,x] for x in [4,5,6,7,9]] \
                + [[4, x] for x in [1,2]] + [[5,x] for x in [1,6,7,8,9]] \
                + [[6,x] for x in [1,2,3,4,7]] + [[7,x] for x in [2]] \
                + [[8,x] for x in [2,3,4,5,6,7,8,9]]

ENV_PARAMS = {"grid_size": [10, 10, 10], 
              "plane": [1,1,0], 
              "plane_walls": PLANE_WALLS, 
              "max_steps": 400,
              'reward_pos': [9,9]}

env = SubGridWorldEnv(random_grid=False, check_valid_run=False, **ENV_PARAMS)

# Play a few episodes of a random game and render.
for i in range(10):
    observation = env.reset()
    print(f'Starting from position {env.idealize_observation()[0]}, '+
          f'the optimal path to the reward at {env._reward_pos} is...\n' 
          f'{env.get_optimal_path()}.')
    done = False
    env.render()
    while not done:
        (observation, reward, done) = env.step(np.random.choice(range(6)))
        # print(f'Idealized observation (x,y,z): {env.idealize_observation()}')
        env.render()
