from gymnasium import Env
from gymnasium.spaces import Box,Discrete

import numpy as np
from scipy.spatial.transform import Rotation

from .src import robot_simulator
import os
import torch # This import is needed to run onnx on GPU 
import onnxruntime as rt
import cv2 

"""
This version includes the following changes:
 - Use OBJ detection network to get small images of the objects to pick
 - USE PPO network to get the pick action  
 - Model should learn which images are valid pick and which are not
 - Action space is composed by IMAGE_NUM + 1 actions: IMAGE_NUM pick actions and 1 reset action
 - black images when numObjPicked < IMAGE_NUM

"""

class robotEnv(Env):
    def __init__(self,config_file):
        super(robotEnv, self).__init__()

        # Init the ONNX runtime session
        providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        current_dir = os.path.dirname(__file__)
        # Load the object detection network
        model_dir = os.path.join(current_dir, "utils", "objDetectionNetwork/")
        pretrained_ppo_dir = os.path.join(current_dir, "utils", "Pretrained/")        
        self.ort_sess_fe = rt.InferenceSession(model_dir + 'objDetection.onnx',providers=providers)
        self.ort_sess_ppo = rt.InferenceSession(pretrained_ppo_dir + 'robofeeder-picking.onnx',providers=providers)
        
        # Init the simulation
        self.simulator = robot_simulator(config_file,seed=123) # use as numpy random seed the ROS_ID
        
        #Spaces
        self.IMAGE_NUM = 3
        self.CROP_DIM = 50
        self.action_space = Discrete(self.IMAGE_NUM+1) # Reset + IMAGE_NUM possible pick actions
        self.observation_space = Box(low=0, high=1, shape=(self.IMAGE_NUM,self.CROP_DIM,self.CROP_DIM))  # Set the observation space to [rgb] gray immage array

        # Set the Episode length
        self.max_episode_steps = 5
        self.curr_num_episode = 0
        #self.total_rew = 0

        self.is_log_set = False

        #get the first obs
        self.reset()
    # END INIT
    
    def step(self, action):
        self.curr_num_episode += 1
        self.rew = -1

        if(action == 0): # Reset
            self.rew = 0  
            for index in range(self.simulator.configs["NUMBER_OF_OBJECTS"]):
                if (self.simulator.objPicked[index] == 0):                                      # If the object is not picked
                    initialObjPos = self.simulator.data.qpos[0+7*index+3:7*(index+1)].copy()    # Get the rotation of the object
                    rot = Rotation.from_quat(initialObjPos).as_euler('xyz')                     # Convert the rotation to euler
                    check = rot[2]<0.7 and rot[2]>0                                             # Check if the object is in the correct orientation
                    self.rew += -int(check)                                                     # if a valid object is not picked give an extra -1
            done = True

            if(self.is_log_set): self.log_file.write(f"reset,{self.rew}\n")
            return self.current_obs, self.rew, done,done, {}
        
        #if not vibration, the action is a pick action
        result_ppo = self.ort_sess_ppo.run(None, {"input": self.current_obs[action-1].reshape(1,1,self.CROP_DIM,self.CROP_DIM)})[0][0] #get the vector of the action (x,y,rotation)

        # Map the action: Switch from Pixel to World Coordinates and compute the rotation
        target_pos = (result_ppo[0:2])*self.CROP_DIM/2      #convert the action from [-1,1] to the dimension of the [-self.CROP_DIM/2: self.CROP_DIM/2]
        target_pos = self.obsCenter[action-1] + target_pos    #trasform the action from local coordinates to global coordinates (pixel)        
        rotation = (result_ppo[2]+1)*np.pi/2                # Convert the rotation from [-1,1] to [0,pi]
        
        # Simulate the action
        obj_position = self.simulator.pixel2World(pixelCoordinates=target_pos)     
        resultIMG, self.rew = self.simulator.simulate_pick(np.append(obj_position,0.11),rotation)
        
        if (self.rew is None): # If the action is not feasible
            self.rew = -1
            done = (self.max_episode_steps == self.curr_num_episode)
            if(self.is_log_set): self.log_file.write(f"{action},NONVALID\n")
            return self.current_obs,self.rew, done,done,{} # If the action is not feasible


        # 3.2. current_state
        self.current_obs = self.rgb2gray(resultIMG)
        # 3.3. done
        done = (self.max_episode_steps == self.curr_num_episode)

        return self.current_obs, self.rew, done,done, {}
    # END STEP
    
    # Inizialize a new episode
    def reset(self,*,seed=None,options=None):
        # reset simulation, episode number and get a new observation
        self.simulator.reset()
        self.curr_num_episode = 0
        #self.total_rew = 0
        self.current_obs = self.rgb2gray(self.simulator.get_state())
        return self.current_obs,{}
    # END RESET

    def close(self):
        self.log_file.close()
    # END CLOSE

    ### UTILITY FUNCTIONS ###
    def rgb2gray(self,rgb):
        results_ort = self.ort_sess_fe.run(None, {"input_tensor": rgb[np.newaxis, :]})
        self.obsCenter = []
        result = np.zeros((0,self.CROP_DIM,self.CROP_DIM),dtype=np.float32)
        for i in range(self.IMAGE_NUM): 
            #get a number of images equals to the number of objects to pick, otherwise get a "empty" image
            if(i<(self.simulator.configs["NUMBER_OF_OBJECTS"] - sum(self.simulator.objPicked))):

                vertexs = results_ort[1][0][i]*self.simulator.configs["OBSERVATION_IMAGE_DIM"] #pick the first object recognized (and rescale the vertex)
                x_min,y_min,x_max,y_max = int(vertexs[1]),int(vertexs[0]),int(vertexs[3]),int(vertexs[2])
                 
                # set the center of the img to set the initial position of the action
                self.obsCenter.append(np.array([x_min+(x_max-x_min)/2,y_min+(y_max-y_min)/2],dtype=np.float32))

                deltax,deltay = round(-((x_max-x_min)-self.CROP_DIM)/2),round(-((y_max-y_min)-self.CROP_DIM)/2)
                errorx,errory = 0,0
                

                if (x_min - deltax <0 or x_max + deltax >self.simulator.configs["OBSERVATION_IMAGE_DIM"]): 
                    errorx = x_min - deltax if (x_min- deltay<0) else x_max + deltax-self.simulator.configs["OBSERVATION_IMAGE_DIM"]
                if (y_min - deltay <0 or y_max + deltay >self.simulator.configs["OBSERVATION_IMAGE_DIM"]): 
                    errory = y_min - deltay if (y_min- deltay<0) else y_max + deltay -self.simulator.configs["OBSERVATION_IMAGE_DIM"]

                cropped = rgb[y_min-deltay-errory :y_max+deltay-errory, x_min-deltax -errorx :x_max+deltax- errorx] #y,x
                cropped = cv2.resize(cropped, (self.CROP_DIM, self.CROP_DIM), interpolation = cv2.INTER_NEAREST) #INTER_AREA is slower

                cropped = np.dot(cropped[...,:3], np.array([0.2989, 0.5870, 0.1140],dtype=np.float32)).reshape(1,self.CROP_DIM,self.CROP_DIM) #shape = (h,w)
                cropped = cropped/255.0 #Normalize the image
                result = np.append(result,cropped.reshape(1,self.CROP_DIM,self.CROP_DIM),axis=0)

            else:
                self.obsCenter.append(np.array([0,0],dtype=np.float32)) # set the center of the img to set the initial position of the action
                result = np.append(result,np.zeros((1,self.CROP_DIM,self.CROP_DIM),dtype=np.float32),axis=0)

        self.obsCenter = np.array(self.obsCenter)   

        # shuffle elements to add noise
        p = np.random.permutation(len(self.obsCenter))
        self.obsCenter = self.obsCenter[p]
        result = result[p]      
        return result

    def normalizeAngle(self,angle):
        if(angle>np.pi): angle -=np.pi
        elif(angle<0): angle += np.pi
        return angle
    
    def vibrate(self):
        self.simulator.vibrate_sinuisodal()
        self.current_obs = self.rgb2gray(self.simulator.get_state())
        return self.current_obs

    def close(self):
        print("CLOSE")
        self.simulator.close()