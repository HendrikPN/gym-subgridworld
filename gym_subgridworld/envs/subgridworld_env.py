import gym
from gym import spaces
import numpy as np
import typing
from typing import List, Tuple
import cv2

from gym_subgridworld.utils.a_star_path_finding import AStar 

class SubGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, random_grid=False, check_valid=True, 
                 check_valid_run=False, **kwargs):
        """
        This environment is a N x M x L gridworld with walls where the optimal 
        policy requires only knowledge of a 2D subspace, e.g. the (x,y)-plane. 
        The agent perceives the whole world and its position in it. 
        It does not see the reward, which is fixed to a given position in the 
        same plane as the agent's starting position. The agents goal is to find
        the reward within the minimum number of steps. An episode ends if the
        reward has been obtained or the maximum number of steps is exceeded.
        By default there is no restriction to the number of steps.
        At each reset, the agent is moved to a random position in the cube and 
        the reward is placed at a specified position of the same plane.

        NOTE:

        You can create your own grid or a randomized version. You can also 
        choose whether or not you want an A* algorithm to (i) check for valid
        positions at the beginning for each grid point, or (ii) at each reset,
        or (iii) not at all. For larger grids, it is recommended to choose
        (iii) over (ii) and (ii) over (i). 

            (i) check_valid = True (default)
            (ii) check_valid_run = True
            (iii) check_valid = False and check_valid_run = False

        If check_valid is True and you choose to generate a random grid, 
        it will try to rebuild the grid until the number of valid positions is 
        sufficiently large.

        If you specify walls, make sure that the list is only two dimensional,
        that is, e.g., [[1,1], [2,3], ...]. We automatically expand the wall
        throughout the dimension orthorgonal to the specified plane.

        Args:
            random_grid (bool): Whether or not to create a randomized grid.
                                Defaults to False.
            check_valid (bool): Whether or not to run A* algorithm to check
                                that a grid has enough paths to the 
                                reward at the beginning. This may be 
                                computationally expensive. Defaults to True.
            check_valid_run (bool): Overwrites `check_valid` and checks 
                                    validity at runtime, i.e. at each reset.
                                    Defaults to False.

            **kwargs:
                grid_size (:obj:`list` of :obj:`int`): The size of the grid. 
                                                       Defaults to [50, 50, 50].
                plane (:obj:`list` of :obj:`int`): The plane in which the reward 
                                                   and agent will be located.
                                                   Defaults to [1,1,0], i.e. the 
                                                   (x,y)-plane
                reward_pos (:obj:`list` of :obj:`int`): The position of the 
                                                        reward in the respective
                                                        plane.
                                                        Defaults to [0, 0].
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
        if 'reward_pos' in kwargs and type(kwargs['reward_pos']) is list:
            setattr(self, '_reward_pos_plane', kwargs['reward_pos'])
        else:
            setattr(self, '_reward_pos_plane', [0,0])
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
        if self._reward_pos_plane in self._plane_walls:
            raise ValueError('The reward is located in a wall: '+
                             f'{self._reward_pos_plane}.')
        
        self.check_valid = check_valid
        self.check_valid_run = check_valid_run

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

        #list of int: Axis specifying the relevant plane.
        self._dim_keep = list(np.array(self._plane).nonzero()[0])
        #int: Axis which is orthorgonal to the relevant plane.
        self._dim_expand = np.asarray(np.array(
                          self._plane, 
                          copy=False) == 0).nonzero()[0].item(0)

        #list: The coordinates of the walls throughout the 3D gridworld.
        self._walls_coord = self._expand_walls()
        if random_grid:
            self.generate_random_walls()

        if not random_grid and self.check_valid and not self.check_valid_run:
            #function: Sets the valid and invalid positions in the plane.
            self._get_valid_pos()

        #list of int: The current position of the agent. Not initial position.
        self._agent_pos = [0, 0, 0]

        #list of int: The position of the reward. Not initial position.
        self._reward_pos = list(np.insert(self._reward_pos_plane, 
                                          self._dim_expand, 
                                          0))
        
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
        Agent is reset to a random position in the cube from where it can reach
        the reward. 
        Reward is placed in the respective plane.

        Returns:
            observation (numpy.ndarray): An array representing the current and 
                                         the previous image of the environment.
        """
        # Reset internal timer.
        self._time_step = 0

        # Initialize agent and reward position to some value.
        self._agent_pos = [0, 0, 0]
        self._reward_pos = [0 ,0 ,0]

        is_valid_pos = False
        while self._agent_pos == self._reward_pos or not is_valid_pos:
            # Place the agent randomly within the cube.
            for i in range(3):
                self._agent_pos[i] = np.random.choice(range(self._grid_size[i]))
            while self._agent_pos in self._walls_coord:
                for i in range(3):
                    choices = range(self._grid_size[i])
                    self._agent_pos[i] = np.random.choice(choices)
            
            # Place reward in the same plane as the agent.
            self._reward_pos = list(np.insert(self._reward_pos_plane, 
                                            self._dim_expand, 
                                            self._agent_pos[self._dim_expand]))

            # Check whether there exists a path to the reward if required.
            if self.check_valid_run:
                is_valid_pos = self.get_optimal_path() is not None
            elif self.check_valid:
                plane_pos = np.delete(self._agent_pos, self._dim_expand, 0)
                is_valid_pos = list(plane_pos) in self._valid_pos
                

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
        walls = []
        if len(self._walls_coord) > 0:
            walls = np.array(self._walls_coord)
            walls = np.delete(walls, self._dim_expand, 1)
            walls = list(map(tuple, walls))
        start_pos = np.array(self._agent_pos)
        start_pos = list(np.delete(start_pos, self._dim_expand, 0))
        end_pos = self._reward_pos_plane
        # Runs the A* algorithm.
        if end_pos != start_pos:
            a_star_alg = AStar()
            a_star_alg.init_grid(self._grid_size[self._dim_keep[0]], 
                                self._grid_size[self._dim_keep[1]], 
                                walls, 
                                start_pos, 
                                end_pos
                                )
            optimal_path = a_star_alg.solve()
        else:
            optimal_path = []

        return optimal_path

    def generate_random_walls(self, prob=0.3, max_ratio=0.2) -> None:
        """
        Clears the current set of walls and generates walls at random.
        If check_valid is True the process is repeated until the ratio of 
        invalid to valid positions is larger than a given value. 
        Valid positions are those that have a path to the reward.

        Args:
            prob (float): Probability of placing a a wall at any given point
                          in the plane. Defaults to 0.3.
            max_ratio (float): The maximum allowed ratio between invalid and 
                               valid positions on the grid. Defaults to 0.2.
        """

        self._valid_pos = []
        self._invalid_pos = []
        while len(self._valid_pos) == 0 \
              or len(self._invalid_pos)/len(self._valid_pos) > max_ratio:
            # Reset walls
            self._walls_coord = []
            self._plane_walls = []

            # Place walls at random.
            for i in range(self._grid_size[self._dim_keep[0]]):
                for j in range(self._grid_size[self._dim_keep[1]]):
                    if not (i == self._reward_pos_plane[0] 
                            and j == self._reward_pos_plane[1]):
                        choice = np.random.choice(2, 1, p=[1-prob, prob])[0]
                        if choice:
                            self._plane_walls.append([i,j])
            self._walls_coord = self._expand_walls()

            # Get valid positions if required.
            if self.check_valid and not self.check_valid_run:
                self._get_valid_pos()
            else:
                break              

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

        Returns:
            walls (`list` of `int`): The walls expanded throughout the 3rd dim.
        """
        walls = []
        for wall in np.array(self._plane_walls):
            for i in range(self._grid_size[self._dim_expand]):
                wall_expand = list(np.insert(wall, self._dim_expand, i))
                walls.append(wall_expand)
        return walls
    
    def _get_valid_pos(self) -> None:
        """
        Gets the valid and invalid positions with and without path to the 
        reward.
        """
        #list: list of valid positions with path to reward.
        self._valid_pos = []
        #list: list of invalid positions without path to reward.
        self._invalid_pos = []
        for i in range(self._grid_size[self._dim_keep[0]]):
            for j in range(self._grid_size[self._dim_keep[1]]):
                if [i,j] not in self._plane_walls \
                    and [i,j] != self._reward_pos_plane:
                    self._agent_pos = np.insert([i, j], self._dim_expand, 0)
                    if self.get_optimal_path() is not None:
                        self._valid_pos.append([i,j])
                    else:
                        self._invalid_pos.append([i,j])
