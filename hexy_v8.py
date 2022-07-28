import numpy as np
import matplotlib.pyplot as plt
from gym import utils
from gym.envs.mujoco import mujoco_env


# if you want to receive pixel data from camera using render("rgb_array",_,_,_,_)
# you should change below line <site_packages>/gym/envs/mujoco/mujoco_env.py to:
# self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)



DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}


class HexyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Hexy_ver_2.3/hexy-v2.3.xml', ):
        utils.EzPickle.__init__(**locals())
        self.time_step = 0

        self.interval = 2

        #### For Following Agent
        self.xlist_1 = []
        self.ylist_1 = []

        #### For Reference Agent
        self.xlist_2 = []
        self.ylist_2 = []

        ## For vel
        self.xy_vel = []
        self.xy_1_vel = 0

        ## Distance between 2 agents
        self.dist_between_agents = 0

        # angle diff
        self.angle_diff = 0
        self.rel_desired_heading_vec = np.array([0,0])

        # Body name for collision detection
        self.target_body_array   = ["T_FRf1", "T_FRf2", "T_FRf3", "T_FRf4",
                                   "T_FLf1", "T_FLf2", "T_FLf3", "T_FLf4",
                                   "T_MRf1", "T_MRf2", "T_MRf3", "T_MRf4",
                                   "T_MLf1", "T_MLf2", "T_MLf3", "T_MLf4",
                                   "T_BRf1", "T_BRf2", "T_BRf3", "T_BRf4",
                                   "T_BLf1", "T_BLf2", "T_BLf3", "T_BLf4",
                                   "T_FRt1", "T_FRt2",
                                   "T_FLt1", "T_FLt2",
                                   "T_MRt1", "T_MRt2",
                                   "T_MLt1", "T_MLt2",
                                   "T_BRt1", "T_BRt2",
                                   "T_BLt1", "T_BLt2",
                                   "T_FRs1",
                                   "T_FLs1",
                                   "T_MRs1",
                                   "T_MLs1",
                                   "T_BRs1",
                                   "T_BLs1",
                                   "Torso_2"]

        self.follower_body_array = ["F_FRf1", "F_FRf2", "F_FRf3", "F_FRf4",
                                   "F_FLf1", "F_FLf2", "F_FLf3", "F_FLf4",
                                   "F_MRf1", "F_MRf2", "F_MRf3", "F_MRf4",
                                   "F_MLf1", "F_MLf2", "F_MLf3", "F_MLf4",
                                   "F_BRf1", "F_BRf2", "F_BRf3", "F_BRf4",
                                   "F_BLf1", "F_BLf2", "F_BLf3", "F_BLf4",
                                   "F_FRt1", "F_FRt2",
                                   "F_FLt1", "F_FLt2",
                                   "F_MRt1", "F_MRt2",
                                   "F_MLt1", "F_MLt2",
                                   "F_BRt1", "F_BRt2",
                                   "F_BLt1", "F_BLt2",
                                   "F_FRs1",
                                   "F_FLs1",
                                   "F_MRs1",
                                   "F_MLs1",
                                   "F_BRs1",
                                   "F_BLs1",
                                   "Torso"]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)


    @property
    def is_healthy(self):
        # if height of hexy is too low or distance between two agent is too far, reset!!

        # Initialize Healthy condition
        is_healthy = (self.state_vector()[2]) > -0.05 and self.dist_between_agents < 0.65 and self.xy_1_vel < 0.5  and abs(self.angle_diff) < 2  #1.8?

        for i in range(self.sim.data.ncon):
             sim_contact = self.sim.data.contact[i]
             for j in range(len(self.target_body_array)):
                 if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == self.target_body_array[j]):
                    if str(self.sim.model.geom_id2name(sim_contact.geom2)) in self.follower_body_array :
                        is_healthy = False
                        print("Collision! : RESET")
                        return is_healthy

                 if (str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.target_body_array[j]):
                     if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.follower_body_array:
                         is_healthy = False
                         print("Collision! : RESET")
                         return is_healthy

        return is_healthy


    @property
    def done(self):
        done = not self.is_healthy
        return done


    def step(self, action):
        # print(action.shape)
        action = action[0:18]

        # Turn Right
        Act1 = np.array([0.0, -0.75, 0.4,
                         -0.3, -0.75, 0.4,
                         0.3, -0.75, 0.8,
                         0.6, -0.75, 0.8,
                         -0.6, -0.75, 0.4,
                         0.0, -0.75, 0.4])

        Act3 = np.array([-0.3, -0.75, 0.8,
                         0.3, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.6, -0.75, 0.4,
                         -0.6, -0.75, 0.8])

        Act2 = (Act1 + Act3) / 2
        Act2[4] += 0.75
        Act2[10] += 0.75
        Act2[16] += 0.75

        Act4 = (Act1 + Act3) / 2
        Act4[1] += 0.5
        Act4[7] += 0.5
        Act4[13] += 0.5

        Turn_Right = [Act1, Act2, Act3, Act4]


        # Turn Left

        Act1 = np.array([0.0, -0.75, 0.4,
                         -0.6, -0.75, 0.4,
                         0.6, -0.75, 0.8,
                         0.3, -0.75, 0.8,
                         -0.3, -0.75, 0.4,
                         0.0, -0.75, 0.4])

        Act3 = np.array([-0.6, -0.75, 0.8,
                         0.6, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.3, -0.75, 0.4,
                         -0.3, -0.75, 0.8])

        Act2 = (Act1 + Act3) / 2
        Act2[4] += 0.5
        Act2[10] += 0.5
        Act2[16] += 0.5

        Act4 = (Act1 + Act3) / 2
        Act4[1] += 0.75
        Act4[7] += 0.75
        Act4[13] += 0.75

        Turn_Left = [Act1, Act2, Act3, Act4]


        # Go-straight
        Act1 = np.array([0.0, -0.75, 0.4,
                         -0.6, -0.75, 0.4,
                         0.6, -0.75, 0.8,
                         0.6, -0.75, 0.8,
                         -0.6, -0.75, 0.4,
                         0.0, -0.75, 0.4])

        Act3 = np.array([-0.6, -0.75, 0.8,
                         0.6, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.0, -0.75, 0.4,
                         0.6, -0.75, 0.4,
                         -0.6, -0.75, 0.8])

        Act2 = (Act1 + Act3) / 2
        Act2[4] += 0.5
        Act2[10] += 0.5
        Act2[16] += 0.5

        Act4 = (Act1 + Act3) / 2
        Act4[1] += 0.5
        Act4[7] += 0.5
        Act4[13] += 0.5

        Go_straight = [Act1, Act2, Act3, Act4]


        #### Designing Action Sequence
        Action_dct = {}
        Action_dct["Go_straight"] = Go_straight
        Action_dct["Turn_Right"] = Turn_Right
        Action_dct["Turn_Left"] = Turn_Left

        Action_sequence = ["Go_straight"] *10 + ["Turn_Left"]*10 +["Go_straight"] *10  +["Turn_Right"] * 20 +["Go_straight"] *10 + ["Turn_Left"]*20 +["Go_straight"] *10+ ["Turn_Right"] * 20 +["Go_straight"] *10 + ["Turn_Left"]*20 +["Go_straight"] *20
        #Action_sequence = ["Go_straight"] * 160
        #Action_sequence = ["Go_straight"] * 10 + ["Turn_Left"] * 20 + ["Go_straight"] * 10 + ["Turn_Right"] * 20 + ["Go_straight"] * 10 + ["Turn_Left"] * 10 + ["Go_straight"] * 20 + ["Turn_Right"] * 10 + ["Go_straight"] * 10 + ["Turn_Left"] * 30 + ["Go_straight"] * 10

        ## How to calculate time-step ?  =>  (160) * 4 * Interval(2) = 1280

        motion = Action_dct[Action_sequence [  (self.time_step // (4*self.interval)) ]  ]  [(self.time_step%(4*self.interval))//(self.interval)]
        # motion = [0]*18

        # self.set_state(np.hstack((self.sim.get_state().qpos[0:30] ,motion))  , np.hstack((self.sim.get_state().qvel[0:30],[0.4]*18)))


        #### Get initial INFO
        x_1_init = self.state_vector()[0]
        y_1_init = self.state_vector()[1]

        x_2_init = self.state_vector()[24]+0.45
        y_2_init = self.state_vector()[25]


        #### Do Simulation
        self.do_simulation(np.hstack((action,motion)), self.frame_skip)


        #### Calculate rewards and costs

        ## Position
        x_1_pos = self.state_vector()[0]
        y_1_pos = self.state_vector()[1]
        x_2_pos = self.state_vector()[24]+0.45
        y_2_pos = self.state_vector()[25]

        ## Velocity
        x_1_vel = (x_1_pos - x_1_init) / self.dt
        y_1_vel = (y_1_pos - y_1_init) / self.dt
        x_2_vel = (x_2_pos - x_2_init) / self.dt
        y_2_vel = (y_2_pos - y_2_init) / self.dt

        ## Planar velocity
        xy_1_vel = np.linalg.norm(np.array([x_1_vel, y_1_vel]))
        xy_2_vel = np.linalg.norm(np.array([x_2_vel, y_2_vel]))

        self.xy_1_vel = xy_1_vel

        ## Check coordinate
        # print("Following :", x_1_pos, y_1_pos)
        # print("Ref :", x_2_pos, y_2_pos )

        ## Distance between two agents (initially d=0.5)
        Distance_between_two_agents = ((x_1_pos-x_2_pos)**2 + (y_1_pos - y_2_pos)**2)**(0.5)
        self.dist_between_agents = Distance_between_two_agents


        #### Rewards (사이의 거리, 속도, 충돌)
        ## Distance reward
        # Desired heading vector
        desired_heading_vec = np.array([x_2_pos - x_1_pos , y_2_pos -  y_1_pos])
        desired_heading_unit_vec= desired_heading_vec / np.linalg.norm(desired_heading_vec)

        # Follower's rotation matrix
        rot_ang = self.state_vector()[5]
        rot_matrix = np.array([[np.cos(rot_ang), np.sin(rot_ang)] , [-np.sin(rot_ang), np.cos(rot_ang)]])

        # Desired heading vector w.r.t Follower's frame
        rel_desired_heading_vec = rot_matrix @ desired_heading_vec
        self.rel_desired_heading_vec = rel_desired_heading_vec

        # Angle difference ver 1
        ref_unit_vec = np.array([1, 0])
        cos_theta = np.dot(desired_heading_unit_vec, ref_unit_vec)
        if desired_heading_vec[1]<0:
            theta = -np.arccos(cos_theta)
        else:
            theta = np.arccos(cos_theta)
        z_1_theta = self.state_vector()[5]
        if theta > 0 :
            if z_1_theta > 0:
                angle_diff = z_1_theta - theta
            else :
                angle_diff = abs(z_1_theta) + theta
        else:
            if z_1_theta > 0:
                angle_diff = z_1_theta + abs(theta)
            else :
                angle_diff = abs(z_1_theta) - abs(theta)

        self.angle_diff = abs(angle_diff)

        # Angle difference ver 2
        target_ang = self.state_vector()[29]
        follower_ang = self.state_vector()[5]

        if follower_ang > 0:
            if target_ang > 0:
                angle_diff_2 = abs(follower_ang - target_ang)
            else:
                angle_diff_2 = follower_ang + abs(target_ang)
        else:
            if target_ang > 0:
                angle_diff_2 = abs(follower_ang) + target_ang
            else:
                angle_diff_2 = abs(follower_ang - target_ang)


        A = 10
        B = 0.2
        C = 10

        dist_reward = A - ((((rel_desired_heading_vec[0]-0.40)**2 + (rel_desired_heading_vec[1]) **2)**(0.5)) / B   +  C * angle_diff_2)

        # print(angle_diff_2 * 180 / np.pi)

        # Collision reward
        col_reward = 0

        for i in range(self.sim.data.ncon):
             sim_contact = self.sim.data.contact[i]
             for j in range(len(self.target_body_array)):
                 if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == self.target_body_array[j]):
                    if str(self.sim.model.geom_id2name(sim_contact.geom2)) in self.follower_body_array :
                        col_reward = -100
                        print("Collision! : Reward -= 50")
                        break

                 if (str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.target_body_array[j]):
                     if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.follower_body_array:
                        col_reward = -100
                        print("Collision! : Reward -= 50")
                        break

        # Goal in reward
        goal_reward = 0
        if self.time_step == 1240 - 1 :     # max_episode_steps - 1
            goal_reward = 1000
            print("!!!!!!!!!!!!! GOAL IN !!!!!!!!!!!!!")

        # Reward sum
        reward = dist_reward + col_reward + goal_reward

        # print("dist : " ,dist_reward, "vel :", vel_reward,"ser :", ser_reward, "sum : ", reward)

        #### Append postion of Two agents
        # For reference agent
        self.xlist_2.append(x_2_pos)
        self.ylist_2.append(y_2_pos)

        # For following agent
        self.xlist_1.append(x_1_pos)
        self.ylist_1.append(y_1_pos)

        # For Velocity check
        self.xy_vel.append(xy_2_vel)


        ## Plotting Trajectory
        if self.time_step == 280 - 1 :     # max_episode_steps - 1
            plt.plot(self.xlist_2[2:], self.ylist_2[2:], 'r-', label='reference_agent')
            plt.plot(self.xlist_1[2:], self.ylist_1[2:], 'b-', label='following_agent')
            #plt.plot(self.xy_vel)
            plt.legend()
            plt.show()

        #### Update time step
        self.time_step += 1

        #### Return INFOs
        done = self.done
        observation = self._get_obs()
        info = {

            'total reward': reward
        }

        return observation, reward, done, info


    def _get_obs(self):

        ## 1. For Training
        #camera_data = np.array(self.render("rgb_array", 148, 148, 2))
        #CHW = np.transpose(camera_data, (2, 0, 1))

        # print(camera_data, CHW)

        ## If you wanna check the input image
        # plt.imshow(camera_data)
        # plt.show()

        ## 2. For rendering check
        data = self._get_viewer("rgb_array").read_pixels(148, 148, depth=False)
        CHW = np.transpose(data[::-1, :, :] , (2, 0, 1))

        obs_dct = {}
        obs_dct['image'] = np.array(CHW)/255.0
        obs_dct['vector'] = self.state_vector()[6:24]

        return obs_dct


    def reset_model(self):

        ## Reset all Joint to zero position
        qpos = np.array(
            [0, 0.0, -0.005, 0, 0, 0, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6,
             0, 0.0, -0.005, 0, 0, 0, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6])
        qvel = np.array([0]*48)
        self.set_state(qpos, qvel)

        ## Update obervation
        observation = self._get_obs()

        ## Initialize timestep
        self.time_step = 0

        # Clear the batch
        self.xlist_1 = []
        self.ylist_1 = []
        self.xlist_2 = []
        self.ylist_2 = []
        self.xy_vel = []

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


