import random
import os
import subprocess
import numpy as np
import time
import yaml
import imageio
import mujoco
import mujoco.viewer
import cv2

#import ros_planner
from . import obj_configurator
# Open the file and load the file
from .planner import PlanningClass



class robot_simulator:
    def __init__(self,config_file,seed=None):

        # Load the constants from the configuration file
        with open(config_file) as f: self.configs = yaml.load(f, Loader=yaml.SafeLoader)

        try:
            # set the objects in the simulation
            obj_configurator.set_XML_obj("staubli/urdf/",self.configs["NUMBER_OF_OBJECTS"])
            
            #Inizialize the Mujoco model
            xml_path = 'staubli/urdf/tx2_60.xml'
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model,height=self.configs["OBSERVATION_IMAGE_DIM"],width=self.configs["OBSERVATION_IMAGE_DIM"]) 
            self.isRealTime = self.configs["IS_SIMULATION_REAL_TIME"]
        except Exception as e:
            print("Error loading Mujoco model:", e)
            raise

        #intialize the Planner
        self.planner=PlanningClass()
        self.counter = 0
        
        #Initilize data
        if(seed is not None): np.random.seed(seed)
        self.last_pos= [0.0,0.0,0.0,0.0,0.0,0.0]
        self.rewCheck= np.array([-0.19096066,  0.47406576,  0.41207368]) #np.array([-0.0882, 0.4929, 0.4])
        self.possibleOrietnation = np.arange(-1,1.1,0.2)
        self.objPicked = [0]*self.configs["NUMBER_OF_OBJECTS"]

        #perform one step to initialize the data
        mujoco.mj_step(self.model, self.data) 
        self.inv_camera_matrix = self.get_inverse_camera_matrix(self.renderer,self.data)    

        self.viewer = None
        # Viewer Setting if the simulation is showed 
        if(self.configs["IS_SIMULATION_SHOWED"]):
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]=0
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
            self.viewer = viewer
    # END INIT  

    def reset(self):
        self.objPicked = [0]*self.configs["NUMBER_OF_OBJECTS"]
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.last_pos = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.counter = 0
        
        #not overlapping objects
        repeat = True
        while(repeat):
            x= np.random.uniform(-0.8,0.8,self.configs["NUMBER_OF_OBJECTS"]) # avoid [-1,1] to avoid not feasible picks
            y= np.random.uniform(-0.75,0.1,self.configs["NUMBER_OF_OBJECTS"]) # avoid [-1,1] to avoid not feasible picks
            
            repeat = False
            for i in range(self.configs["NUMBER_OF_OBJECTS"]):
                for j in range (i+1,self.configs["NUMBER_OF_OBJECTS"]):
                    if(np.linalg.norm(np.array([x[i],y[i]])-np.array([x[j],y[j]]))<0.27):
                        repeat= True
                        break

        x,y = (x+1)*self.configs["OBSERVATION_IMAGE_DIM"]/2,(y+1)*self.configs["OBSERVATION_IMAGE_DIM"]/2

        if(self.configs["SHUFFLE_OBJECTS"]):
            for i in range(self.configs["NUMBER_OF_OBJECTS"]):
                #random position and orientation
                res = self.pixel2Wolrd(np.array([x[i],y[i]]))
                randomlist=random.choices(self.possibleOrietnation,k=4)
                self.data.qpos[i*7:i*7+3] = [res[0], res[1], 0.21]
                if(self.configs["OBJ_CORRECT_ORIENTATION"]): # the first obj is always in the right orientation
                    # the rotation of the obj is correct on z axis 
                    self.data.qpos[i*7+3]= randomlist[0]
                    self.data.qpos[i*7+6]= randomlist[1]
                else:
                    self.data.qpos[i*7+3:i*7+7] = randomlist # the rotation of the obj is totally random
                
        #to stabilize the simulation
        mujoco.mj_step(self.model, self.data,900) 
        if(self.viewer is not None): self.viewer.sync()
    # END RESET

    def simulate_pick(self,coordinates,rotation):
        self.step_count = 0 
        self.step_time = 0
        self.limit = self.configs["ACTION_SHORT_PAUSE"]
        self.closeTime = False
        self.openTime = False
        
        self.isFinished = False
        self.finalObjsPos= []
        self.initialObjPos= []
        self.reward = -1
        frames_robot = []

        # Compute the Path 
        computed_plan,computed_plan_index = self.planner.planFunction(initial_joint_position=self.last_pos, obj_position=coordinates)
        if(computed_plan is None): return None,None
        
        #Function that is performed ad the start of the simulation
        def init_controller(model, data):
                data.ctrl[6:8]=[0.008,0.008]
                data.ctrl[5] = rotation
        # END INIT CONTROLLER

        #Function that is performed at each step of the simulation
        def controller(model, data):
            # perform simulation step after a delay in simulation time (stability reason)
            if(data.time - self.step_time>self.limit and not self.isFinished):
                if (self.step_count <= computed_plan_index[-1] and not self.openTime and not self.closeTime):
                    # perform arm movement
                    data.ctrl[0:5] = computed_plan[self.step_count][1:6] #6 dof [index 5] is moved at initial timestamp 
                    self.limit= self.configs["ACTION_LONG_PAUSE"] if(self.step_count in computed_plan_index) else self.configs["ACTION_SHORT_PAUSE"]
                    self.closeTime = (self.step_count +1 == computed_plan_index[1])
                    self.openTime = (self.step_count +1 == computed_plan_index[-1])

                    if(self.openTime):self.finalObjsPos= [self.data.qpos[i*7:i*7+3].copy() for i in range(self.configs["NUMBER_OF_OBJECTS"])]
                    self.step_count += 1
                    
                else:
                    if(self.closeTime):
                        # perform gripper movement (close)
                        self.limit = self.configs["ACTION_SHORT_PAUSE"]
                        data.ctrl[6:8]=[data.ctrl[6]-self.configs["DELTA_GRIPPER"],data.ctrl[7]-self.configs["DELTA_GRIPPER"]]

                        if(data.ctrl[6]<=self.configs["CLOSE_GRIPPER"]):
                            self.limit = self.configs["ACTION_LONG_PAUSE"]
                            self.closeTime = False
                            
                    if(self.openTime):
                        # perform gripper movement (open)
                        self.limit = self.configs["ACTION_SHORT_PAUSE"]
                        data.ctrl[6:8]=[data.ctrl[6]+self.configs["DELTA_GRIPPER"],data.ctrl[7]+self.configs["DELTA_GRIPPER"]]
                        
                        if(data.ctrl[6]>=self.configs["OPEN_GRIPPER"]):
                            self.last_pos = computed_plan[-1][1:7].tolist()
                            self.openTime = False
                            self.isFinished = True     
                self.step_time = data.time
        # END CONTROLLER  

        ## SIMULATION START ##
        init_controller(self.model, self.data)
        mujoco.set_mjcb_control(controller)

        #get the initial position of the objects for the reward computation
        for i in range(self.configs["NUMBER_OF_OBJECTS"]):
            # get the position of the obj from the site in the right position (+1 due to the target site)
            self.initialObjPos.extend(self.data.site(1+i).xpos.copy())
            #get the rotation of the obj
            self.initialObjPos.extend(self.data.qpos[0+7*i+3:7*(i+1)].copy())
        
        while  not self.isFinished:
            step_start = time.time()           
            mujoco.mj_step(self.model, self.data)

            if self.configs["IS_SIMULATION_RECORD"]:
                # Render and store frame
                self.renderer.update_scene(self.data, camera="angled_side_view")
                frame = self.renderer.render()
                # Convert to RGB format
                frame_rgb = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                frames_robot.append(frame_rgb)

            if(self.viewer is not None): self.viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock. Sync simulation with real time
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0 and self.isRealTime: time.sleep(time_until_next_step)
                

        self.counter +=1
        if self.configs["IS_SIMULATION_RECORD"]:
            self.save_video(frames_robot)
        
        self.compute_reward()
        
        mujoco.mj_step(self.model, self.data,400) #to stabilize the simulation
        if(self.viewer is not None): self.viewer.sync()
        # SIMULATION END #

        # Render the scene from the top-down camera and return the frame and the reward
        self.renderer.update_scene(self.data, camera="top_down")
        frame = self.renderer.render()
        return frame,self.reward
    # END SIMULATE
    #### UTILITY FUNCTIONS ####

    def get_state(self):
        mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera="top_down")
        frame = self.renderer.render()
        return frame
    # END GET STATE

    def get_inverse_camera_matrix(self,renderer, data):
        renderer.update_scene(data,camera="top_down")
        """
        Having two cameras in a Mujoco scene serves the purpose of simulating a stereoscopic view, similar to how human eyes perceive depth.
        In stereoscopic vision, each eye captures a slightly different perspective of the same scene. 
        These two different perspectives are then combined by the brain to perceive depth and three-dimensional information.
        """
        pos = self.model.cam('top_down').pos #np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
        z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
        y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
        rot = np.vstack((np.cross(y, z), y, z))
        fov = self.model.cam('top_down').fovy[0]

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (renderer.width - 1) / 2.0
        image[1, 2] = (renderer.height - 1) / 2.0

        camera_matrix = image @ focal @ rotation @ translation
        """
        https://stackoverflow.com/questions/48104143/inverse-of-a-4x3-matrix-in-python
        4x3 matrix in 3D graphics is usually 4x4 uniform transform matrix where the last row or column (depends on the convention used) is 
        omitted and represents (0,0,0,1) vector (no projection). This is done usually to preserve space.

        """
        full_cam_matrix = np.vstack((camera_matrix, np.array([0, 0, 0, 1])))
        return np.linalg.inv(full_cam_matrix)
    
    def pixel2Wolrd(self,pixelCoordinates):
        depth =  0.42078046 #static value of the camera depth
        x,y= pixelCoordinates[0]*(-depth),pixelCoordinates[1]*(-depth)

        result = self.inv_camera_matrix @ np.array([x, y, -depth, 1])  
        return result[0:2]
    
    def compute_reward(self): 
        # The reward is 1 if the object is in the goal position (site of the drop Pose).
        for i in range(self.configs["NUMBER_OF_OBJECTS"]):
            if(np.linalg.norm(self.rewCheck-self.finalObjsPos[i])<0.085 and self.objPicked[i]==0):
                self.reward=1
                self.objPicked[i]=1
                break
    
    def vibrate_sinuisodal(self,duration=850,frequency=0.045,amplitude=0.0135):
        #Perform Vibrations
        for _ in range(duration):
            step_start = time.time() #for real time simulation

            sim_time = round((self.data.time)*100,1)
            val = abs(np.cos(2 * np.pi * frequency* sim_time)) * amplitude

            self.data.ctrl[8] = val
            mujoco.mj_step(self.model, self.data)

            if(self.viewer is not None): self.viewer.sync()
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0 and self.isRealTime: time.sleep(time_until_next_step)

        self.data.ctrl[8] = 0 #reset the table position 

        #to stabilize the simulation      
        if(self.viewer is not None): 
            for _ in range(1200):
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)        
                self.viewer.sync()
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0 and self.isRealTime: time.sleep(time_until_next_step)
        else: 
            mujoco.mj_step(self.model, self.data,1200)
        return 

    def save_video(self,frames):  
        # Save video
        video_name = f"simulated_pick_{self.counter}.mp4"
        video_path = os.path.join(self.configs["RECORD_FOLDER"], video_name)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")


    def close(self):
        # Properly close viewer and renderer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.model = None
        self.data = None
        self.renderer = None

        # if self.renderer is not None:
        #     self.renderer.close()
        #     self.renderer = None

    # def __del__(self):
    #     self.close()