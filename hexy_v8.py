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

        self.interval = 5

        #### For Following Agent
        self.xlist_1 = []
        self.ylist_1 = []

        #### For Reference Agent
        self.xlist_2 = []
        self.ylist_2 = []

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)


    @property
    def is_healthy(self):
        # if hexy was tilted or changed position too much, reset environments
        is_healthy = (self.state_vector()[2]) > -0.05

        Map_array = ["curved_map1", "curved_map2", "curved_map3", "curved_map4"]


        # 충돌 + MAX time step is over

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
        ## How to calculate time-step ?  =>  (160) * 4 * Interval(5) = 3200

        motion = Action_dct[Action_sequence [  (self.time_step // (4*self.interval)) ]  ]  [(self.time_step%(4*self.interval))//(self.interval)]

        # self.set_state(np.hstack((self.sim.get_state().qpos[0:30] ,motion))  , np.hstack((self.sim.get_state().qvel[0:30],[0.4]*18)))


        #### Get initial INFO
        x_1_init = self.state_vector()[0]
        y_1_init = self.state_vector()[1]

        x_2_init = self.state_vector()[24]+0.5
        y_2_init = self.state_vector()[25]


        #### Do Simulation
        self.do_simulation(np.hstack((action,motion)), self.frame_skip)


        #### Calculate rewards and costs

        # Position
        x_1_pos = self.state_vector()[0]
        y_1_pos = self.state_vector()[1]
        x_2_pos = self.state_vector()[24]+0.5
        y_2_pos = self.state_vector()[25]

        # Velocity
        x_1_vel = (x_1_pos - x_1_init) / self.dt
        y_1_vel = (y_1_pos - y_1_init) / self.dt
        x_2_vel = (x_2_pos - x_2_init) / self.dt
        y_2_vel = (y_2_pos - y_2_init) / self.dt

        # Planar velocity
        xy_1_vel = np.sqrt(np.mean(np.square(np.array([x_1_vel, y_1_vel]))))
        xy_2_vel = np.sqrt(np.mean(np.square(np.array([x_2_vel, y_2_vel]))))

        ## Check coordinate
        # print("Following :", x_1_pos, y_1_pos)
        # print("Ref :", x_2_pos, y_2_pos )

        # Distance between two agents
        Distance_between_two_agents = ((x_1_pos-x_2_pos)**2 + (y_1_pos - y_2_pos)**2)**(0.5)

        # Rewards (사이의 거리, 충돌, 속도)
        reward = 10 * x_1_vel

        # ## Plotting Trajectory
        # # For reference agent
        # self.xlist_2.append(x_2_pos)
        # self.ylist_2.append(y_2_pos)
        #
        # # For following agent
        # self.xlist_1.append(x_1_pos)
        # self.ylist_1.append(y_1_pos)
        #
        # if self.time_step == 3000 - 1 :     # max_episode_steps - 1 = 3000 - 1
        #     plt.plot(self.xlist_2[2:],self.ylist_2[2:], 'r-')
        #     plt.plot(self.xlist_1[2:],self.ylist_1[2:], 'b-')
        #     plt.show()



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
        # take account of history
        return np.concatenate([self.state_vector()[6:24]])



    def reset_model(self):

        qpos = np.array(
            [0, 0.0, -0.005, 0, 0, 0, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8,0.6,
             0, 0.0, -0.005, 0, 0, 0, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6, 0, -0.8, 0.6])

        qvel = np.array([0]*48)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.time_step = 0

        return observation



    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


