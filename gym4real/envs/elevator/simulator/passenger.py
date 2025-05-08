import numpy as np

def generate_arrival_distribution(lambd: float, 
                                  total_time: int, 
                                  floor: int,
                                  goal_floor: int = None,
                                  max_floor: int = None,
                                  seed: int = None,
                                  ) -> list:
    """
    Generate a list of random arrival times based on a Poisson distribution.

    :param lambd: The lambda term for the Poisson distribution.
    :param total_time: The total time period over which to generate arrivals.
    :param floor: The floor number.
    :param goal_floor: The goal floor for the passengers (if specified).
    :param max_floor: The maximum floor number (if specified).
    :param seed: Random seed for reproducibility.
    
    :return: A list of random arrival times.
    """
    person_list = []
    person_id = 0

    rng = np.random.default_rng(seed)
    
    for t in range(total_time):
        # Draw the number of people arriving at time t from a Poisson distribution
        num_new = rng.poisson(lambd)
        
        for _ in range(num_new):
            person = Passenger(_id=f"{floor}_{person_id}", 
                               arrival_time=t, 
                               goal_floor=goal_floor if goal_floor is not None else np.random.randint(0, max_floor))
            person_list.append(person)
            person_id += 1
    
    return person_list


def generate_generic_passengers(n: int, global_goal_floor: int = None) -> list:
    """
    Generate a list of generic passengers.

    :param n: The number of passengers to generate.
    :param global_goal_floor: The goal floor for all passengers (if specified).
    :return: A list of Passenger instances.
    """
    person_list = []
    
    for i in range(n):
        person = Passenger(_id=-i, arrival_time=-1, goal_floor=np.random.randint(0, 4) if global_goal_floor is None else global_goal_floor)
        person_list.append(person)
    
    return person_list


class Passenger:
    def __init__(self, _id:int, arrival_time:int, goal_floor:int):
        """
        Initialize a Passenger instance.

        :param _id: The unique identifier for the passenger.
        :param arrival_time: The time at which the passenger arrives.
        :param goal_floor: The floor the passenger wants to go to.
        """
        self._id = _id
        self.arrival_time = arrival_time
        self.goal_floor = goal_floor

    def __repr__(self):
        """
        Return a string representation of the Passenger instance.
        """
        return f"Passenger(id={self._id}, arrival_time={self.arrival_time}, goal_floor={self.goal_floor})"
    
    @property
    def id(self):
        """
        Return the ID of t.
        """
        return self._id
        
        
class FloorQueue:
    def __init__(self, floor:int, max_queue_length:int = 5, max_arrivals:int = 5):
        """
        Initialize a FloorQueue instance.
        """
        self.floor = floor
        self.max_queue_length = max_queue_length
        self.max_arrivals = max_arrivals
        
        self.waitings = []
        self.futures = []
        self.arrivals = {}
    
    def __repr__(self):
        """
        Return a string representation of the PassengerQueue instance.
        """
        return f"FloorQueue(floor={self.floor}, waiting={self.waitings}, futures={self.futures})"
    
    def __len__(self):
        """
        Return the number of passengers in the queue.
        """
        return len(self.waitings)
    
    def reset(self):
        """
        Reset the queue to an empty state.
        """
        self.waitings = []
        self.futures = []
        self.arrivals = {}
        
    def set_queue(self, queue: list):
        """
        Set the queue for the floor.

        :param queue: A list of Passenger instances.
        """
        assert len(queue) <= self.max_queue_length, "Queue length exceeds maximum limit."
        self.waitings = queue
    
    def set_arrivals(self, arrivals: list):
        """
        Set the arrivals for the queue.

        :param arrivals: A list of Passenger instances.
        """
        times = set([person.arrival_time for person in arrivals])
        self.arrivals = {time:[person for person in arrivals if person.arrival_time == time] for time in times}
        
        # Pop elements if sometime they are more than max_arrivals
        for time in self.arrivals.keys():
            if len(self.arrivals[time]) > self.max_arrivals:
                self.arrivals[time] = self.arrivals[time][:self.max_arrivals]
                
    def update_waitings(self):
        """
        Update the queue by moving passengers from the futures list to the waiting list.
        """
        for i, _ in enumerate(self.futures):
            if len(self) >= self.max_queue_length:
                break
            self.waitings.append(self.futures.pop(i))
            #print(f"Queue {self.floor}: passenger {self.waitings[-1].id} in queue at {self.waitings[-1].arrival_time} seconds")
    
        self.futures = []
                
    def check_arrivals(self, current_time: int):
        """
        Update the queue by removing passengers who have arrived.

        :param current_time: The current time in the simulation.
        """
        if current_time in self.arrivals.keys():
            for person in self.arrivals[current_time]:
                self.futures.append(person)