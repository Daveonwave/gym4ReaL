from gymnasium import Env
from gymnasium.spaces import Box

import numpy as np
from scipy.spatial.transform import Rotation

from .src import robot_simulator
import os 
import torch # This import is needed to run onnx on GPU 
import onnxruntime as rt
import cv2 
import math
"""
This version includes the following changes:
- Object Detection network that feed the network with obj small images
- Goal of the env is learn to pick the objects starting from a small image of the object

"""
class robotEnv(Env):

    def __init__(self, config_file, sim_seed=None):
        super(robotEnv, self).__init__()

        providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        current_dir = os.path.dirname(__file__)
        # Load the object detection network
        model_dir = os.path.join(current_dir, "utils", "objDetectionNetwork/")
        self.ort_sess = rt.InferenceSession(model_dir + 'objDetection.onnx',providers=providers)
        
        # Init the simulation
        seed = sim_seed if sim_seed is not None else 123
        self.simulator = robot_simulator(config_file, seed=seed) # use as numpy random seed the ROS_ID
        
        # Get Simulation data
        self.inv_camera_matrix = self.simulator.inv_camera_matrix
        #Spaces
        self.action_space = Box(low=-1, high=1.0, shape=(3,))  # Set the action space to [x,y,rotation]
        
        self.CROP_DIM = 50
        self.observation_space = Box(low=0, high=1, shape=(1,self.CROP_DIM,self.CROP_DIM))  # Set the observation space to [rgb] gray immage array

        # Set the Episode length
        self.max_episode_steps = 2
        self.curr_num_episode = 0

        self.is_log_set = False

        #get the first obs
        self.reset()
    # END INIT
    
    def step(self, action):
        self.curr_num_episode += 1
        self.rew = -1

        # 1. Map the action: Switch from Pixel to World Coordinates and compute the rotation
        #convert the action from [-1,1] to the dimension of the [-self.CROP_DIM/2: self.CROP_DIM/2]
        # and  trasform the action from local coordinates to global coordinates (pixel)
        target_pos = self.obsCenter + (action[0:2])*self.CROP_DIM/2 
        rotation = (action[2]+1)*np.pi/2 
        
        # Simulate the action
        obj_position = self.simulator.pixel2World(pixelCoordinates=target_pos)     
        resultIMG, self.rew = self.simulator.simulate_pick(np.append(obj_position,0.11),rotation)
        
        if (self.rew is None): # If the action is not feasible
            self.rew = -1
            done = (self.max_episode_steps == self.curr_num_episode)
            if(self.is_log_set): 
                self.log_file.write(f"{action},NONVALID\n")
            
            self.rew /= self.max_episode_steps
            return self.current_obs,self.rew, done,done,{} # If the action is not feasible
        
        # 3.1 reward
        if(self.rew == -1): #From the simulator: 1 if the object is picked, -1 if the object is not picked (now imprrove the reward)
            
            #Comput the distance from the nearest object 
            distance = [0]*self.simulator.configs["NUMBER_OF_OBJECTS"]
            for i in range(self.simulator.configs["NUMBER_OF_OBJECTS"]): distance[i]=np.linalg.norm(obj_position[0:2]-(self.simulator.initialObjPos[i*7:i*7+2]))
            
            distance_index = np.argmin(distance)
            distance = distance[distance_index]
            
            #Compute real obj rotation of the nearest object
            rot = Rotation.from_quat(self.simulator.initialObjPos[distance_index*7+3:distance_index*7+7]).as_euler('xyz')
            rot = self.normalizeAngle(2.35+rot[0])
            deltarot = abs(rot-rotation)
            if(deltarot>2.6): deltarot = np.pi-deltarot # Complemetary angle for boundary conditions

            if(self.simulator.objPicked[distance_index]==0): # if the object is not picked (or not considered in the reward)
                #Reward Parameters
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
        self.rew /= self.max_episode_steps
        return self.current_obs, self.rew, done, done,{}
    # END STEP
    
    # Inizialize a new episode
    def reset(self,*,seed=None,options=None):
        self.simulator.reset()
        self.curr_num_episode = 0
        self.current_obs = self.rgb2gray(self.simulator.get_state())
        self.simulator.last_pos = [0.0,0.0,0.0,0.0,0.0,0.0]
        return self.current_obs,{}
    # END RESET
        
    def rgb2gray(self,rgb):
        try:
            results_ort = self.ort_sess.run(None, {"input_tensor": rgb[np.newaxis, :]})
            vertexs = results_ort[1][0][0]*self.simulator.configs["OBSERVATION_IMAGE_DIM"] #pick the first object recognized (and rescale the vertex)
            x_min,y_min,x_max,y_max = int(vertexs[1]),int(vertexs[0]),int(vertexs[3]),int(vertexs[2])

            self.obsCenter = np.array([x_min+(x_max-x_min)/2,y_min+(y_max-y_min)/2],dtype=np.float32) # set the center of the img to set the initial position of the action
            


            deltax,deltay = round(-((x_max-x_min)-self.CROP_DIM)/2),round(-((y_max-y_min)-self.CROP_DIM)/2)
            errorx,errory = 0,0
            
            if (x_min - deltax <0 or x_max + deltax >self.simulator.configs["OBSERVATION_IMAGE_DIM"]): 
                errorx = x_min - deltax if (x_min- deltay<0) else x_max + deltax-self.simulator.configs["OBSERVATION_IMAGE_DIM"]
            if (y_min - deltay <0 or y_max + deltay >self.simulator.configs["OBSERVATION_IMAGE_DIM"]): 
                errory = y_min - deltay if (y_min- deltay<0) else y_max + deltay -self.simulator.configs["OBSERVATION_IMAGE_DIM"]

            cropped = rgb[y_min-deltay-errory :y_max+deltay-errory, x_min-deltax -errorx :x_max+deltax- errorx] #y,x
            cropped = cv2.resize(cropped, (self.CROP_DIM, self.CROP_DIM), interpolation = cv2.INTER_NEAREST)  # o cosi o controllando i numeri dispari (efficienza)
            cropped = np.dot(cropped[...,:3], np.array([0.2989, 0.5870, 0.1140],dtype=np.float32)).reshape(1,self.CROP_DIM,self.CROP_DIM) #shape = (1,h,w)
            cropped = cropped/255.0 #Normalize the image

        except Exception as e:
            print("Exception:",e)
            self.obsCenter = np.array([0,0])
            if(self.is_log_set): self.log_file.write(f"Error in Crop Image: Shape:{cropped.shape},Y:({y_min-deltay-errory}->{y_max+deltay-errory}),X:({x_min-deltax -errorx}->{x_max+deltax- errorx})\n")
            cropped = np.zeros((1,self.CROP_DIM,self.CROP_DIM),dtype=np.float32)
        return cropped

    def close(self):
        print("CLOSE")
        self.simulator.close()
        #self.simulator.planner.shutdown()
        #self.log_file.close()

    def normalizeAngle(self,angle):
        if(angle>np.pi): angle -=np.pi
        elif(angle<0): angle += np.pi
        return angle        