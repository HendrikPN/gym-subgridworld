# gym-subgridworld

An openAI gym environment for representation learning within a reinforcement 
learning (RL) setting.
This environment is build in accordance with the 
[openAI gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md#how-to-create-new-environments-for-gym)
policy for standardized RL environments.

# Environment description

This environment is a N x M x L gridworld with walls where the optimal 
policy requires only knowledge of a 2D subspace, e.g. the (x,y)-plane. 
The agent perceives the whole world and its position in it. 
It does not see the reward, which is fixed to a position in a corner 
of the same plane as the agent's starting position. The agents goal is 
to find the reward within the minimum number of steps. An episode ends 
if the reward has been obtained or the maximum number of steps is 
exceeded. By default there is no restriction to the number of steps.
At each reset, the agent is moved to a random position in the cube and 
the reward is positioned in the corner of the same plane.

## Simplifications

We provide two methods that may simplify the environment by simplifying the
observation. By default, the observation is a 3D pixel image.

+ `simplify_observation()`: This function provides the observation 
                            represented by a vector of length N x M x L
                            with three 1's at positions 
                            (x, N + y, N + M + z) where (x,y,z) is the 
                                position of the agent.
+ `idealize_observation()`: This function provies an ideal observation,
                           i.e. the vector (x,y,z) which represents the
                           position of the agent.

# Installation

```
git clone -b master https://github.com/HendrikPN/gym-subgridworld.git
cd gym-subgridworld
pip install --user -e .
``` 

