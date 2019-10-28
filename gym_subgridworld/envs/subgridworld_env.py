import gym
from gym import spaces
import numpy as np
import typing
from typing import List, Tuple
import cv2

from gym_subgridworld.utils.a_star_path_finding import AStar 

class SubGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        """
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

        NOTE:
        If you specify walls, make sure that the list is only two dimensional,
        that is, e.g., [[1,1], [2,3], ...]. We automatically expand the wall
        throughout the dimension orthorgonal to the specified plane.

        Args:
            **kwargs:
                grid_size (:obj:`list` of :obj:`int`): The size of the grid. 
                                                       Defaults to [50, 50, 50].
                plane (:obj:`list` of :obj:`int`): The plane in which the reward 
                                                   and agent will be located.
                                                   Defaults to [1,1,0], i.e. the 
                                                   (x,y)-plane
                max_steps (int): The maximum number of allowed time steps. 
                                 Defaults to 0, i.e. no restriction.
                plane_walls (:obj:`list`): The coordinates of walls within the 
                                           gridworld. Defaults to [].
        """
        if 'grid_size' in kwargs and type(kwargs['grid_size']) is list:
            setattr(self, '_grid_size', kwargs['grid_size'])
        else:
            setattr(self, '_grid_size', [50, 50, 50])
        if 'plane' in kwargs and type(kwargs['plane']) is list:
            setattr(self, '_plane', kwargs['plane'])
        else:
            setattr(self, '_plane', [1,1,0])
        if 'max_steps' in kwargs and type(kwargs['max_steps']) is int:
            setattr(self, '_max_steps', kwargs['max_steps'])
        else:
            setattr(self, '_max_steps', 0)
        if 'plane_walls' in kwargs and type(kwargs['plane_walls']) is list:
            setattr(self, '_plane_walls', kwargs['plane_walls'])
        else:
            setattr(self, '_plane_walls', [])

        if any(x <= 1 for x in self._grid_size):
            raise ValueError('The gridworld needs to be 3D. Instead, received '+ 
                             f'grid of size {self._grid_size}'
                             )
        if any((x != 0 and x != 1) for x in self._plane):
            raise ValueError('The plane you specified is invalid: '+
                             f'{self._plane}. Expected list of 0 or 1.'
                             )
        for wall in self._plane_walls:
            if not len(wall) == 2:
                raise ValueError('You specified a wall which is not 2D: '+
                                 f'{wall}.')

        #int: image size is [x-size * 7, y-size * 7, z-size * 7] pixels.
        self._img_size = np.array(self._grid_size) * 7 

        #:class:`gym.Box`: Image properties to be used as observation.
        self.observation_space=gym.spaces.Box(low=0, high=1, 
                                              shape=(self._img_size[0], 
                                              self._img_size[1], 
                                              self._img_size[2]), 
                                              dtype=np.float32)
        #:class:`gym.Discrete`: The space of actions available to the agent.
        self.action_space=gym.spaces.Discrete(6)

        #list: The coordinates of the walls throughout the 3D gridworld.
        self._walls_coord = self._expand_walls()

        #list of int: The current position of the agent. Not initial position.
        self._agent_pos = [0, 0, 0]
        #list of int: The position of the reward. Not initial position.
        self._reward_pos = [v * (self._grid_size[i] - 1) 
                            for i, v in enumerate(self._plane)]
        
        #function: Sets the static part of the observed image, i.e. walls.
        self._get_static_image()
        #numpy.ndarray of float: The currently observed image.
        self._img = np.zeros(self.observation_space.shape)
        #int: Number of time steps since last reset.
        self._time_step = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        An action moves the agent in one direction. The agent cannot cross walls 
        or borders of the grid. If the maximum number of steps is exceeded, the 
        agent receives a negative reward, and the game resets.
        Once the reward is hit, the agent receives the positive reward.

        Args:
            action (int): The index of the action to be taken.

        Returns:
            observation (numpy.ndarray): An array representing the current image 
                                         of the environment.
            reward (float): The reward given after this time step.
            done (bool): The information whether or not the episode is finished.

        """
        # Move according to action.
        if action == 0:
            new_pos = [self._agent_pos[0] + 1, *self._agent_pos[1:3]]
        elif action == 1:
            new_pos = [self._agent_pos[0] - 1, *self._agent_pos[1:3]]
        elif action == 2:
            new_pos = [self._agent_pos[0], 
                       self._agent_pos[1] + 1, 
                       self._agent_pos[2]
                       ]
        elif action == 3:
            new_pos = [self._agent_pos[0], 
                       self._agent_pos[1] - 1, 
                       self._agent_pos[2]
                       ]
        elif action == 4:
            new_pos = [*self._agent_pos[0:2], self._agent_pos[2] + 1]
        elif action == 5:
            new_pos = [*self._agent_pos[0:2], self._agent_pos[2] - 1]
        else:
            raise TypeError('The action is not valid. The action should be an '+ 
                            f'integer 0 <= action <= {self.action_space.n}'
                            )

        # If agent steps on a wall, it is not moved at all.
        if new_pos not in self._walls_coord:
            self._agent_pos = new_pos
            
        # If agent moves beyond the borders, it is not moved at all instead.
        for index, pos in enumerate(self._agent_pos):
            if pos >= self._grid_size[index]:
                self._agent_pos[index] = self._grid_size[index] - 1
            elif pos < 0:
                self._agent_pos[index] = 0

        # Check whether reward was found. Last step may get rewarded.
        self._time_step += 1
        if self._agent_pos == self._reward_pos:
            reward = 1.
            done = True
        # Check whether maximum number of time steps has been reached.
        elif self._max_steps and self._time_step >= self._max_steps:
            reward = -1.
            done = True
        # Continue otherwise.
        else:
            reward = 0.
            done = False

        # Create a new image and observation.
        self._img = self._get_image()
        observation = self._get_observation()

        return (observation, reward, done)

    def reset(self) -> np.ndarray:
        """
        Agent is reset to a random position in the cube. 
        Reward is placed in the corner of the respective plane.

        Returns:
            observation (numpy.ndarray): An array representing the current and 
                                         the previous image of the environment.
        """
        # Reset internal timer.
        self._time_step = 0

        # Initialize agent and reward position to some value.
        self._agent_pos = [0, 0, 0]
        self._reward_pos = [0 ,0 ,0]

        while self._agent_pos == self._reward_pos:
            # Place the agent randomly within the cube.
            for i in range(3):
                self._agent_pos[i] = np.random.choice(range(self._grid_size[i]))
            while self._agent_pos in self._walls_coord:
                for i in range(3):
                    choices = range(self._grid_size[i])
                    self._agent_pos[i] = np.random.choice(choices)
            
            # Place reward in the top corner of the same plane as the agent.
            for index, v in enumerate(self._plane):
                if v:
                    self._reward_pos[index] = self._grid_size[index] - 1 
                else:
                    self._reward_pos[index] = self._agent_pos[index]

        # Create initial image.
        self._img = self._get_image()

        return self._get_observation()

    def render(self, mode: str ='human') -> None:
        """
        Renders part of the current state of the environment as an image in a 
        popup window. Since the environment is 3D, only the 2D slice where the 
        reward is located is imaged.

        Args:
            mode (str): The mode in which the image is rendered. 
                        Defaults to 'human' for human-friendly. 
                        Currently, only 'human' is supported.
        """
        image = self._img.copy()
        #np.ndarray: the coordinate for the position of the reward in the image
        reward_coord = np.array(self._reward_pos) * 7
        # Draw reward into image.
        image[reward_coord[0]:reward_coord[0]+7, 
              reward_coord[1]:reward_coord[1]+7, 
              reward_coord[2]:reward_coord[2]+7] = self._img_reward
        slices = ()
        for index, v in enumerate(self._plane):
            if v != 0:
                slices += (slice(0, self._img_size[index]),)
            else:
                slices += (slice(reward_coord[index], reward_coord[index]+1),)
        image = image[slices].squeeze()

        if mode == 'human':
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600,600)
            cv2.imshow('image',np.uint8(image * 255))
            cv2.waitKey(50)
        else:
            raise NotImplementedError('We only support `human` render mode.')
    
    def idealize_observation(self) -> np.ndarray:
        """
        Calculates an ideal representation of the observation. 
        The representation is given by the position of the agent, 
        i.e. (x, y ,z).

        Returns:
            observation (numpy.ndarray): The ideal observation.
        """
        observation = np.array([[*self._agent_pos]])
        return observation

    def simplify_observation(self) -> np.ndarray:
        """
        Calculates a simplified representation consisting of 3 concatenated
        lists, each representing the position of the agent in the respective
        grid dimension.

        Returns:
            observation (numpy.ndarray): The simplified observation.
        """
        x_dim = np.eye(self._grid_size[0])[self._agent_pos[0]]
        y_dim = np.eye(self._grid_size[1])[self._agent_pos[1]]
        z_dim = np.eye(self._grid_size[2])[self._agent_pos[2]]

        observation = np.concatenate((x_dim, y_dim, z_dim))
        return observation

    def get_optimal_path(self):
        """
        Calculates the optimal path for the current position of the agent
        using the A* search algorithm.
        
        Returns:
            optimal_path (`list` of `tuple`): The optimal path of 2D positions.
        """
        # Reduces to 2D gridworld.
        dim_keep = np.array(self._plane).nonzero()[0]
        dim_remove = np.asarray(np.array(
                                self._plane, 
                                copy=False) == 0).nonzero()[0].item(0)
        walls = []
        if len(self._walls_coord) > 0:
            walls = np.array(self._walls_coord)
            walls = np.delete(walls, dim_remove, 1)
            walls = list(map(tuple, walls))
        start_pos = np.array(self._agent_pos)
        start_pos = list(np.delete(start_pos, dim_remove, 0))
        end_pos = np.array(self._reward_pos)
        end_pos = list(np.delete(end_pos, dim_remove, 0))

        # Runs the A* algorithm.
        a_star_alg = AStar()
        a_star_alg.init_grid(self._grid_size[dim_keep[0]], 
                             self._grid_size[dim_keep[0]], 
                             walls, 
                             start_pos, 
                             end_pos
                            )
        optimal_path = a_star_alg.solve()

        return optimal_path

    # ----------------- helper methods -----------------------------------------

    def _get_static_image(self) -> None:
        """
        Generate the static part of the gridworld image, i.e. walls, image of 
        the agent and reward.
        """
        # Empty world.
        gridworld = np.zeros(self.observation_space.shape)

        # Draw walls.
        wall_draw = np.ones((7,7,7))
        for wall in self._walls_coord:
            wall_coord = np.array(wall) * 7
            gridworld[wall_coord[0]:wall_coord[0]+7, 
                      wall_coord[1]:wall_coord[1]+7, 
                      wall_coord[2]:wall_coord[2]+7] = wall_draw
        
        #array of float: The static part of the gridworld image, i.e. walls.
        self._img_static = gridworld

        # Draw 2D agent image.
        agent_draw_2d = np.zeros((7,7))
        agent_draw_2d[0, 3] = 0.8
        agent_draw_2d[1, 0:7] = 0.9
        agent_draw_2d[2, 2:5] = 0.9
        agent_draw_2d[3, 2:5] = 0.9
        agent_draw_2d[4, 2] = 0.9
        agent_draw_2d[4, 4] = 0.9
        agent_draw_2d[5, 2] = 0.9
        agent_draw_2d[5, 4] = 0.9
        agent_draw_2d[6, 1:3] = 0.9
        agent_draw_2d[6, 4:6] = 0.9

        # Draw 3D agent image.
        agent_draw = np.zeros((7,7,7))
        agent_draw[0] = agent_draw_2d
        agent_draw[6] = agent_draw_2d
        agent_draw[:, 0, :] = agent_draw_2d
        agent_draw[:, 6, :] = agent_draw_2d
        agent_draw[:, :, 0] = agent_draw_2d
        agent_draw[:, :, 6] = agent_draw_2d

        #array of float: The static 7 x 7 x 7 image of the agent.
        self._img_agent = agent_draw

        # Draw 2D reward image.
        reward_draw_2d = np.zeros((7,7))
        for i in range(7):
            reward_draw_2d[i, i] = 0.7
            reward_draw_2d[i, 6-i] = 0.7

        # Draw 3D reward image.
        reward_draw = np.zeros((7,7,7))
        reward_draw[0] = reward_draw_2d
        reward_draw[6] = reward_draw_2d
        reward_draw[:, 0, :] = reward_draw_2d
        reward_draw[:, 6, :] = reward_draw_2d
        reward_draw[:, :, 0] = reward_draw_2d
        reward_draw[:, :, 6] = reward_draw_2d

        #array of float: The static 7 x 7 x 7 image of the reward.
        self._img_reward = reward_draw
    
    def _get_image(self) -> np.ndarray:
        """
        Generate an image from the current state of the environment.

        Returns:
            image (numpy.ndarray): An array representing an environment image.
        """
        image = self._img_static.copy()
        #np.ndarray: the coordinate for the position of the agent in the image
        agent_coord = np.array(self._agent_pos) * 7

        # Draw agent into static image.
        image[agent_coord[0]:agent_coord[0]+7, 
              agent_coord[1]:agent_coord[1]+7,  
              agent_coord[2]:agent_coord[2]+7] = self._img_agent

        return image

    def _get_observation(self) -> np.ndarray:
        """
        Generates an observation from two sequenced images.

        Returns:
            observation (numpy.ndarray): An 1 x (grid_size * 7) ** 3 array of 
                                         the gridworld.
        """
        observation = self._img
        observation = np.reshape(observation, 
                                 (1, *self.observation_space.shape)
                                 )
        return observation
    
    def _expand_walls(self) -> list:
        """
        Expands the walls specified for a plane through the whole 3D grid.
        """
        dim_expand = np.asarray(np.array(
                                self._plane, 
                                copy=False) == 0).nonzero()[0].item(0)
        walls = []
        for wall in np.array(self._plane_walls):
            for i in range(self._grid_size[dim_expand]):
                wall_expand = list(np.insert(wall, dim_expand, i))
                walls.append(wall_expand)
        return walls
