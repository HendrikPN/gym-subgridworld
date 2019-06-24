import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../gym_subgridworld/envs'))
from subgridworld_env import SubGridWorldEnv

PLANE_WALLS = [[1,2],[1,3],[1,4],[1,5],[4,3], [5,3], [6,3], [6,4], [6,5]]
WALLS = [[*PLANE_WALLS[i], j] for i in range(len(PLANE_WALLS)) for j in range(10)]
ENV_PARAMS = {"grid_size": [10, 10, 10], "plane": [1,1,0], "walls_coord": WALLS, "max_steps": 400}

env = SubGridWorldEnv(**ENV_PARAMS)

# Play a few episodes of a random game, and render.
for i in range(3):
    observation = env.reset()
    done = False
    env.render()
    while not done:
        (observation, reward, done) = env.step(np.random.choice(range(6)))
        # print(f'Idealized observation: {env.idealize_observation()} as (x,y,z)')
        env.render()