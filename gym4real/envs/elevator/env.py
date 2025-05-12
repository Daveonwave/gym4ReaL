from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import numpy as np
from .simulator.elevator import Elevator
from .simulator.passenger import generate_arrival_distribution


class ElevatorEnv(Env):
    metadata = {"render_modes": []}
    
    def __init__(self, 
                 settings:dict,
                 **kwargs):
        super().__init__()
        
        self._elevator = Elevator(max_floor=settings['max_floor'],
                                  min_floor=settings['min_floor'],
                                  max_capacity=settings['max_capacity'],
                                  movement_speed=settings['movement_speed'],
                                  floor_height=settings['floor_height'],
                                  max_arrivals=settings['max_arrivals'],
                                  max_queue_length=settings['max_queue_length'],
                                  )
        
        self.init_elevator_pos = settings['init_elevator_pos']
        self._arrival_distributions = settings['arrival_distributions']
        self._goal_floor = settings['goal_floor']
        self._reward_coeff = settings['reward_coeff']
        
        self.current_time = 0
        self.duration = settings['duration']
        
        self._seed = settings['arrival_distributions']['seed']
        self._rng = np.random.default_rng(self._seed)
        
        self.observation_space = spaces.Dict({
            'current_position': spaces.Discrete(int(self._elevator.floor_height * self._elevator.max_floor / self._elevator.movement_speed) + 1),
            'n_passengers': spaces.Discrete(self._elevator.max_capacity + 1),
            #'speed': spaces.Discrete(3),  # 0: down, 1: still, 2: up (multiply by movement speed)
            'floor_queues': spaces.MultiDiscrete(np.array([settings['max_queue_length'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor)),
            'arrivals': spaces.MultiDiscrete(np.array([settings['max_arrivals'] + 1] * (self._elevator.max_floor - self._elevator.min_floor)), start=[0] * (self._elevator.max_floor - self._elevator.min_floor))
        })
       
        # Action space -> 1: stay still, 2: move up, 0: move down
        self.action_space = spaces.Discrete(3)

    def set_state(self, state):
        self._elevator.set_status(state, self.current_time)
    
    def _get_obs(self):
        return {
            'current_position': int(self._elevator.vertical_position // self._elevator.movement_speed),
            'n_passengers': len(self._elevator.passengers),
            #'speed': int(self._elevator.speed // self._elevator.movement_speed) + 1,            
            'floor_queues': np.array([len(queue) for queue in self._elevator.queues if queue.floor != 0]).astype(np.int64),  
            'arrivals': [len(queue.futures) for queue in self._elevator.queues if queue.floor != 0],
        }
    
    def _get_info(self):
        return {
            'elevator_status': self._elevator.status(),
            'current_time': self.current_time,
            'goal_floor': self._goal_floor,
            'floor_height': self._elevator.floor_height,
            'movement_speed': self._elevator.movement_speed,
            'vertical_position': self._elevator.vertical_position,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #print(self._get_obs(), self._get_info())
        
        self.current_time = 0
        
        # Set the elevator position
        if self.init_elevator_pos is not None:
            self._elevator.reset(initial_position=self.init_elevator_pos)
        else:
            self._elevator.reset(initial_position=self.observation_space['current_position'].sample() * self._elevator.movement_speed)
        
        for queue in self._elevator.queues:
            if queue.floor != self._goal_floor:
                lambd = self._rng.uniform(self._arrival_distributions['lambda_min'], self._arrival_distributions['lambda_max'])
                queue.set_arrivals(arrivals=generate_arrival_distribution(lambd=lambd,
                                                                        total_time=self.duration, 
                                                                        floor=queue.floor,
                                                                        goal_floor=self._goal_floor,
                                                                        seed=self._seed))
                 
        self._elevator.update_queues(current_time=self.current_time)
        return self._get_obs(), self._get_info()
        
    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (np.ndarray): The action to take.
            0: move down
            1: stay still
            2: move up
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
                observation (dict): The current state of the environment.
                reward (float): The reward received after taking the action.
                done (bool): Whether the episode has ended.
                truncated (bool): Whether the episode was truncated.
                info (dict): Additional information about the step.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        served = []
        
        # 0: move down, 1: stay still, 2: move up
        if action == 2:
            self._elevator.move('up')
        elif action == 0:
            self._elevator.move('down')
        elif action == 1: 
            served = self._elevator.open_doors()
        else:
            raise ValueError(f"Invalid action {action}")
        
        # Calculate the reward: we penalize the waiting time of passengers
        reward = 0
        if len(served) > 0:
            reward += len(served) * self._reward_coeff
        else:
            reward -= 1 * (len(self._elevator.passengers) + sum([len(queue) for queue in self._elevator.queues]))
        
        # Check arrivals at the current time
        self.current_time += 1
        self._elevator.update_queues(current_time=self.current_time)
        
        done = self.current_time >= self.duration
        
        return self._get_obs(), reward, done, False, self._get_info()
        

