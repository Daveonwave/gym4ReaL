from gymnasium import Env
from gymnasium.spaces import Box

import numpy as np
from scipy.spatial.transform import Rotation
from .src import robot_simulator
import math

#ex env 10

class robotEnv(Env):
    def __init__(self,config_file, sim_seed=None):
        """
        Env config:
        - ROS_ID: ID of ROS node
        - LOG_FOLDER: Folder where to save the log file (optional)
        """    
        # Init the simulation
        super(robotEnv, self).__init__()
        seed = sim_seed if sim_seed is not None else 123
        self.simulator = robot_simulator(config_file, seed=seed) # use as seed the ROS_ID        
        
        #Spaces
        # Set the action space to [x,y,rotation]
        self.action_space = Box(low=-1, high=1.0, shape=(3,)) 
        # Set the observation space to [rgb] gray immage array
        self.observation_space = Box(low=0, high=1, shape=(1,self.simulator.configs["OBSERVATION_IMAGE_DIM"],self.simulator.configs["OBSERVATION_IMAGE_DIM"]))  

        # Set the Episode length
        self.max_episode_steps = 2
        self.curr_num_episode = 0

        self.reset()
        
        # Init the log file
        # self.is_log_set = ("LOG_FOLDER" in env_config)
        # if(self.is_log_set):
        #     self.log_file = open(env_config["LOG_FOLDER"]+"/sim{}.csv".format(env_config["ROS_ID"]), "w")
        #     self.log_file.write("X,Y,Rotation,rew\n")

    # END INIT
    
    def step(self, action):
        """
        - action[0]: x coordinate of the object in the image
        - action[1]: y coordinate of the object in the image
        - action[2]: rotation of the object in the image
        """

        self.curr_num_episode += 1
        
        # Compute the rotation from [-1,1] to [0,pi]
        action[2] = (action[2]+1)*np.pi/2 
        # Switch from Pixel to World Coordinates and Map the action from [-1,1] to [0,300]
        action[0:2] = (action[0:2]+1)*self.simulator.configs["OBSERVATION_IMAGE_DIM"]/2 
        
        #Compute the position of the object in the world frame
        obj_prediction = self.simulator.pixel2World(pixelCoordinates=action[0:2]) 
        
        # Simulate the action
        resultIMG, self.rew = self.simulator.simulate_pick(np.append(obj_prediction,0.11),action[2])
        #resultIMG, self.rew = self.simulator.simulate_pick(np.append(obj_prediction,0.0),action[2])
        
        if (self.rew is None): # If the action is not feasible
            self.rew = -1
            done = (self.max_episode_steps == self.curr_num_episode)
            if(self.is_log_set): 
                self.log_file.write(f"{action},NONVALID\n")
                
            self.rew  /= self.max_episode_steps
            return self.current_obs,self.rew, done,done,{} # If the action is not feasible

        # 3.1 reward
        #From the simulator: 1 if the object is picked, -1 if the object is not picked (now improve the reward)
        # If pick try near the objects reward can go from -1 to 0
        
        if(self.rew == -1): #not picked
            #Comput the distance from the nearest object 
            distance = [0]*self.simulator.configs["NUMBER_OF_OBJECTS"]
            for i in range(self.simulator.configs["NUMBER_OF_OBJECTS"]): distance[i]=np.linalg.norm(obj_prediction[0:2]-(self.simulator.initialObjPos[i*7:i*7+2]))
            distance_index = np.argmin(distance)
            distance = distance[distance_index]
            
            #Compute real obj rotation of the nearest object
            obj_real_rot = Rotation.from_quat(self.simulator.initialObjPos[distance_index*7+3:distance_index*7+7]).as_euler('xyz')
            obj_real_rot = self.normalizeAngle(2.35+obj_real_rot[0])

            deltarot = abs(obj_real_rot-action[2])
            if(deltarot>2.6): deltarot = np.pi-deltarot # Complemetary angle for boundary conditions

            if(self.simulator.objPicked[distance_index]==0): # if the object is not picked (or not considered in the reward)
                #Reward Parameters
                REW_COEFF = 1
                DISTANCE_LIMIT = 0.012
                ROTATION_LIMIT = 0.35

                alpha_d = -math.log(0.5) / (DISTANCE_LIMIT ** 2)
                alpha_theta = -math.log(0.5) / (ROTATION_LIMIT ** 2)

                # Distance reward (always computed)
                r_d = 0.85 * math.exp(-alpha_d * (distance ** 2))
                
                r_theta = 0.15 * math.exp(-alpha_theta * (deltarot ** 2))

                self.rew = self.rew + r_d + r_theta

        # 3.2. current_state
        self.current_obs = self.rgb2gray(resultIMG)
        # 3.3. done
        done = (self.max_episode_steps == self.curr_num_episode) or (sum(self.simulator.objPicked) == self.simulator.configs["NUMBER_OF_OBJECTS"])   
        self.rew  /= self.max_episode_steps
        # if(self.is_log_set): self.log_file.write(str(action[0])+","+str(action[1])+","+str(action[2])+","+str(self.rew)+"\n")
        return self.current_obs, self.rew, done, done,{}
    # END STEP
    
    # Inizialize a new episode
    def reset(self,*,seed=None,options=None):
        self.simulator.reset()
        self.current_obs = self.rgb2gray(self.simulator.get_state())
        
        self.curr_num_episode = 0 
        self.simulator.last_pos = [0.0,0.0,0.0,0.0,0.0,0.0]
        return self.current_obs,{}
    # END RESET
            
    def rgb2gray(self,rgb):
        gray = np.dot(rgb[...,:3], np.array([0.2989, 0.5870, 0.1140],dtype=np.float32)).reshape(1,self.simulator.configs["OBSERVATION_IMAGE_DIM"],self.simulator.configs["OBSERVATION_IMAGE_DIM"]) #shape = (1,h,w)
        gray = gray/255.0 #Normalize 
        return gray

    def close(self):
        print("CLOSE")
        self.simulator.close()

    def normalizeAngle(self,angle):
        if(angle>np.pi): angle -=np.pi
        elif(angle<0): angle += np.pi
        return angle
        