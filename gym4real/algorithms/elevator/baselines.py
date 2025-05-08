from rich.pretty import pprint


def random_policy(env):
    n_episodes = 10
    
    for episode in range(n_episodes):
        env.reset()
        done = False
        cumulated_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, Reward: {reward}, Cumulative Reward: {cumulated_reward}")


def longest_queue_first(env):
    n_episodes = 30
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        
        done = False
        cumulated_reward = 0
        
        while not done:
            if obs['n_passengers'] > 0:
                if obs['current_position'] * info['movement_speed'] / info['floor_height'] == info['goal_floor']:
                    action = 1  # stay still
                else:  
                    action = 0  # move down
            else:
                # Check if any passengers are waiting on the current floor
                len_max_queue, idx_max_queue = max((queue, idx+1) for idx, queue in enumerate(obs['floor_queues']))
                
                if len_max_queue == 0:
                    action = 1  # stay still
                else:
                    if obs['current_position'] * info['movement_speed'] == (idx_max_queue) * info['floor_height']:
                        action = 1  # stay still
                    elif obs['current_position'] * info['movement_speed'] < (idx_max_queue) * info['floor_height']:
                        action = 2  # move up            
                    elif obs['current_position'] * info['movement_speed'] > (idx_max_queue) * info['floor_height']:
                        action = 0
                    else:
                        action = 1  # stay still
                    
            obs, reward, terminated, truncated, info = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, "
                   f"Reward: {reward}, Cumulative Reward: {cumulated_reward}, "
                   f"Current Pos: {obs['current_position']}, Passengers: {obs['n_passangers']}, "
                   f"Queues: {obs['floor_queues']}")
            
            
def shortest_queue_first(env):
    n_episodes = 30
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        
        done = False
        cumulated_reward = 0
        
        while not done:
            if obs['n_passengers'] > 0:
                if obs['current_position'] * info['movement_speed'] / info['floor_height'] == info['goal_floor']:
                    action = 1  # stay still
                else:  
                    action = 0  # move down
            else:
                # Check if any passengers are waiting on the current floor
                len_min_queue, idx_min_queue = min((queue, idx+1) for idx, queue in enumerate(obs['floor_queues']) if queue > 0)
                
                if obs['current_position'] * info['movement_speed'] == (idx_min_queue) * info['floor_height']:
                    action = 1  # stay still
                elif obs['current_position'] * info['movement_speed'] < (idx_min_queue) * info['floor_height']:
                    action = 2  # move up            
                elif obs['current_position'] * info['movement_speed'] > (idx_min_queue) * info['floor_height']:
                    action = 0
                else:
                    action = 1
                    
            obs, reward, terminated, truncated, info = env.step(action)
            
            cumulated_reward += reward
            done = terminated or truncated
            pprint(f"Episode: {episode}, Step: {info['current_time']}, Action: {action}, "
                   f"Reward: {reward}, Cumulative Reward: {cumulated_reward}, "
                   f"Current Pos: {obs['current_position']}, Passengers: {obs['n_passangers']}, "
                   f"Queues: {obs['floor_queues']}")
            