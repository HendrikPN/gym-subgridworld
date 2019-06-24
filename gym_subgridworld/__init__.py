from gym.envs.registration import register

register(
    id='subgridworld-v0',
    entry_point='gym_subgridworld.envs:SubGridWorldEnv',
)
