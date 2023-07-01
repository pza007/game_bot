from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from keras.layers import Dense, Flatten
import tensorflow as tf
import numpy as np
import math
import gym
from gym.wrappers import FlattenObservation
from keras.optimizers import Adam   #from tensorflow.keras.optimizers.legacy import Adam
from spectator_lib import Spectator
from global_vars import SCREEN_DIAG
import matplotlib.pyplot as plt
import datetime
import json


class MyEnv(gym.Env):
    def __init__(self):
        # (0,1)             18 possible actions
        self.action_space = gym.spaces.Discrete(18)
        # out[0]      float [0.0; 1.0]     bot health = current / max
        # out[1]      float [0.0; 1.0]     bot mana = current / max
        # out[2]      float [0.0; 1.0]     distance on minimap between gate and bot's icon = distance/100
        # out[3]      float [0.0; 1.0]     distance on screen between bot and closest BLUE minion = 1/distance
        # out[4]      float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        # out[5..10]  6*int [0 or 1]       6*cooldown value
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 11), dtype=np.float32)
        self.spectator = Spectator()
        self.buffer = {1:
                           {'xp_total': 0,
                            'num_steps': 0,
                            'steps': {1: {'reward': 0, 'action_result': 0, 'action_result_desc': '', 'action_name': '', 'observation': [0]*11}}
                            }
                       }
        self.episode_num = 1
        self.step_num = 1

        self._prev_observation = None
        self._observation = None
        self._action = None
        self._action_name = None
        self._action_result = None
        self._action_result_desc = None
        self._reward = 0
        self._xp_prev = None
        self._xp_total = 0
        self._done = False
        self._done_reason = None

    def reset_parameters(self):
        self._prev_observation = None
        self._observation = None
        self._action = None
        self._action_name = None
        self._action_result = None
        self._action_result_desc = None
        self._reward = 0
        self._xp_prev = None
        self._xp_total = 0
        self._done = False
        self._done_reason = None

    def get_info(self):
        global model
        """
        self.buffer = {1:
                        {'xp_total': 0,
                         'num_steps': 0,
                         'steps': {1: {'reward': 0, 'action_result': 0, 'action_result_desc': '', 'action_name': '', 'observation': [0]*11}}
                         }
        """
        if self._done:
            # xp_total
            self.buffer[self.episode_num]['xp_total'] = self._xp_total
            # save weights if xp_total is high
            if self._xp_total > 1000:
                model.save_weights(f'rl/models/weights_ep{self.episode_num}_xp{self._xp_total}.h5', overwrite=True)

            # create new episode entry
            self.episode_num += 1
            self.step_num = 1
            self.buffer[self.episode_num] = \
                {'xp_total': 0,
                 'num_steps': 0,
                 'steps': {1: {'reward': 0, 'action_result': 0, 'action_result_desc': '', 'action_name': '', 'observation': [0]*11}}
                 }
        else:
            # fill episode parameters
            self.buffer[self.episode_num]['num_steps'] = self.step_num
            self.buffer[self.episode_num]['steps'][self.step_num] = \
                {'reward': self._reward,
                 'action_result': self._action_result,
                 'action_result_desc': self._action_result_desc,
                 'action_name': self._action_name,
                 'observation': self._observation}

            self.step_num += 1

        return {'info': ''}

    def render(self, mode="human"):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)    # reset seed if you want to use here: self.np_random.integers(0, 1)
        self.reset_parameters()
        self.spectator.init_game()
        return [0]*11, {}

    def step(self, action):
        # xp_before
        if self._xp_prev is None:
            xp_before, err_desc = self.spectator.get_xp_value()
        else:
            xp_before = self._xp_prev

        # done
        self._done, self._done_reason = self.spectator.is_game_finished()

        # action, result
        if not self._done:  # wait for action to be finished
            self._action = action
            self._action_name = self.get_action_name(action)
            self._action_result, self._action_result_desc, _ = self.spectator.execute_action(self._action_name)
        else:
            self._action, self._action_name, self._action_result, self._action_result_desc = None, None, None, None

        # xp_after
        xp_after, err_desc = self.spectator.get_xp_value()
        self._xp_prev = xp_after   # could be None

        # observation
        self._observation = self.spectator.get_game_state()

        # reward
        if None not in [xp_before, xp_after]:
            self._reward = max(xp_after-xp_before, 0)
        else:
            self._reward = -1

        # xp_total
        self._xp_total += self._reward

        return self._observation, self._reward, self._done, False, self.get_info()

    # OBSOLETE function
    def get_reward(self, step_reward):
        #self._observation = [0]*11
        # out[0]      float [0.0; 1.0]     bot health = current / max
        # out[1]      float [0.0; 1.0]     bot mana = current / max
        # out[2]      float [0.0; 1.0]     distance on minimap between gate and bot's icon = distance/100
        # out[3]      float [0.0; 1.0]     distance on screen between bot and closest BLUE minion = 1/distance
        # out[4]      float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        # out[5..10]  6*int [0 or 1]       6*cooldown value
        reward = 0

        # ++++++ REWARD
        # xp was gained
        if step_reward is not None:
            reward += step_reward
        # ACTION
        if self._action is not None:
            # action was successful
            if self._action_result > 0:
                reward += 5
                # action involved attack
                if self._action_name in ['basic_attack', 'q_attack', 'w_attack']:
                    reward += 50
                # action involved gathering xp
                if self._action_name in ['collect_globes', 'run_middle', 'hide_in_bushes']:
                    reward += 15
        # POSITION
        # bot is close to minions
        dist_blue = self._observation[3]  # float [0.0; 1.0]
        reward += min(5, 5*dist_blue*200)
        dist_red = self._observation[4]  # float [0.0; 1.0]
        reward += min(10, 10*dist_red*200)
        # SPEED
        #if self._action_name is not None and step_reward is not None:
        #    t_max = self.spectator.actions.objects[self._action_name].TIMEOUT
        #    t0 = self.spectator.actions.objects[self._action_name].t0
        #    t_diff = time.time() - t0
        #    factor = (t_max - t_diff) / (3*t_max)
        #    reward += factor*step_reward    # short action with big xp_gain is promoted


        # ------ REWARD
        # bot is dead
        if self._done_reason == 'bot_dead':
            reward -= 2000
        # action was not successful
        if self._action_result is not None and self._action_result < 0:
            reward -= 100
        # 'incorrect' actions, based on situation (bot played defensive when it has more than 10% of health)
        if self._action_name in ['hide_behind_gate', 'escape_behind_gate', 'use_well', 'use_spell_d'] \
                and self._observation[0] > 0.1:
            reward -= 1000
        # bot was close to RED minion but currently lost it from screen
        #prev_dist = self._prev_observation[4]  # float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        #curr_dist = self._observation[4]  # float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        #if prev_dist > 1/SCREEN_DIAG > curr_dist:  # 1/SCREEN_DIAG used, to compare to value 0.0
        #    reward -= 50
        # bot moved behind the gate, even though bots'HP > 10%
        #if self._action_name in ['hide_behind_gate', 'escape_behind_gate'] and self._observation[0] > 0.1:
        #    reward -= 500

        # OBSOLETE code
        """
        def get_dist_change(prev_vector, curr_vector):
            if prev_vector != (0, 0) and curr_vector != (0, 0):
                prev_dist = math.sqrt(prev_vector[0] ** 2 + prev_vector[1] ** 2)
                curr_dist = math.sqrt(curr_vector[0] ** 2 + curr_vector[1] ** 2)
                return curr_dist - prev_dist
            else:
                return None
                
        # COMPARE TO PREVIOUS OBSERVATION
        #   bot moved closer to BLUE minion or away from it?
        prev_vector = self._prev_observation[6:8]   # (dx, dy)
        curr_vector = self._observation[6:8]        # (dx, dy)
        dist_change = get_dist_change(prev_vector, curr_vector)
        if dist_change is not None:
            if dist_change > 0:   # moved closer
                reward += 8
            else:   # moved away
                reward -= 12
        #   bot moved closer to RED minion or away from it?
        prev_vector = self._prev_observation[8:10]   # (dx, dy)
        curr_vector = self._observation[8:10]        # (dx, dy)
        dist_change = get_dist_change(prev_vector, curr_vector)
        if dist_change is not None:
            if dist_change > 0:   # moved closer
                reward += 10
            else:   # moved away
                reward -= 15

        # bot's health did not decrease?
        prev_health = self._prev_observation[0:2]   # (curr, max)
        health = self._observation[0:2]   # (curr, max)
        if prev_health != (0, 0) and health != (0, 0):
            if health[0]/health[1] >= prev_health[0]/prev_health[1]:
                reward += 5
            else: reward -= 5

        # bot moved closer to blue minion?
        prev_vct = self._prev_observation[6:8]   # (dx, dy)
        vct = self._observation[6:8]   # (dx, dy)
        if prev_vct != (0, 0) and vct != (0, 0):
            prev_dist = math.sqrt(prev_vct[0] ** 2 + prev_vct[1] ** 2)
            dist = math.sqrt(vct[0] ** 2 + vct[1] ** 2)
            if dist < prev_dist:
                reward += 5
            #else: reward -= 5

        # bot moved closer to red minion?
        prev_vct = self._prev_observation[8:10]   # (dx, dy)
        vct = self._observation[8:10]   # (dx, dy)
        if prev_vct != (0, 0) and vct != (0, 0):
            prev_dist = math.sqrt(prev_vct[0] ** 2 + prev_vct[1] ** 2)
            dist = math.sqrt(vct[0] ** 2 + vct[1] ** 2)
            if dist < prev_dist:
                reward += 10
            #else: reward -= 10
        """

        return reward

    @staticmethod
    def get_action_name(number):
        if number == 0:
            return 'move_up'
        elif number == 1:
            return 'move_down'
        elif number == 2:
            return 'move_right'
        elif number == 3:
            return 'move_left'
        elif number == 4:
            return 'move_up-right'
        elif number == 5:
            return 'move_down-right'
        elif number == 6:
            return 'move_up-left'
        elif number == 7:
            return 'move_down-left'
        elif number == 8:
            return 'run_middle'
        elif number == 9:
            return 'collect_globes'
        elif number == 10:
            return 'basic_attack'
        elif number == 11:
            return 'q_attack'
        elif number == 12:
            return 'w_attack'
        elif number == 13:
            return 'use_well'
        elif number == 14:
            return 'hide_in_bushes'
        elif number == 15:
            return 'hide_behind_gate'
        elif number == 16:
            return 'escape_behind_gate'
        elif number == 17:
            return 'use_spell_d'
        else:
            raise Exception(f'Number: {number} not defined.')


def build_model(in_states, in_actions):
    out_model = tf.keras.Sequential()
    out_model.add(Flatten(input_shape=(1, in_states)))
    out_model.add(Dense(64, activation='relu'))
    out_model.add(Dense(32, activation='relu'))
    out_model.add(Dense(16, activation='relu'))
    out_model.add(Dense(in_actions, activation='linear'))
    out_model.add(Flatten())
    return out_model


def build_agent(in_model, in_actions, num_steps):
    # warmup = 20% of num_steps
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    out_agent = DQNAgent(in_model, memory=memory, policy=policy, nb_actions=in_actions, nb_steps_warmup=int(0.2*num_steps), target_model_update=1e-2)
    return out_agent


def train_agent(in_agent, in_env, num_steps):
    in_agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    """
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
    """
    history = in_agent.fit(in_env, nb_steps=num_steps, visualize=False, verbose=1)
    return in_agent, history


def save_model(in_hist, ):
    try:
        max_xp_ep = int(max(in_hist.history['episode_reward']))
        print('History', in_hist.history)
        if max_xp_ep > 0:
            max_xp_ep = 'plus' + str(max_xp_ep)
        else:
            max_xp_ep = 'minus' + str(abs(max_xp_ep))
    except Exception:
        max_xp_ep = 'none'
    model_path = f'rl/models/{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}_{max_xp_ep}.h5'
    model.save(model_path)   # model.save_weights(f'rl/models/{name}_weights.h5', overwrite=True)

    # save JSON file
    filename = model_path.replace('.h5', '_data.json')
    with open(filename, "w") as file:
        json.dump(env.buffer, file)


def plot_training_history(filename):
    import json
    """
    self.buffer = {1:
                       {'reward_total': 0.0,
                        'num_steps': 0,
                        'steps': {1: {'reward': 0, 'action_result': 0, 'action_name': '', 'observation': [0]*11}}
                        }
                   }
    """
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    # episodes
    x = list(int(val)for val in data.keys())
    # reward_total in each episode
    #y11 = [episode_obj['reward_total'] for episode_obj in data.values()]
    # reward_total in each episode
    y12 = [episode_obj['xp_total'] for episode_obj in data.values()]

    # num_steps in each episode
    y21 = [episode_obj['num_steps'] for episode_obj in data.values()]
    # num_steps ended with success in each episode
    y22 = [0]*len(x)
    for i, episode_obj in enumerate(data.values()):
        for step_obj in episode_obj['steps'].values():
            if step_obj['action_result'] > 0:
                y22[i] += 1
    # num_steps ended with fail in each episode
    y23 = [step_all - step_succ for step_all, step_succ in zip(y21, y22)]

    #plot
    fig, axs = plt.subplots(2, 1)
    #axs[0].plot(x, y11, 'b', x, y12, 'g', linewidth=1.0)
    axs[0].plot(x, y12, 'g', linewidth=1.0)
    axs[0].set_xlabel('Episode number')
    #axs[0].set_ylabel('Reward/XP')
    axs[0].set_ylabel('XP')
    axs[0].grid(True)

    axs[1].plot(x, y21, 'b', x, y22, 'g', x, y23, 'r', linewidth=1.0)
    axs[1].set_xlabel('Episode number')
    axs[1].set_ylabel('Number of all/succ/fail steps')
    axs[1].grid(True)

    plt.show()

"""
env = MyEnv()
env = FlattenObservation(env)

num_steps = 10000
# load model
#model = keras.models.load_model('rl/models/230623_133436_plus1757.h5')
states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions)
# load weights
#model.load_weights('rl/models/230626_weights_ep27_xp2830.h5')
agent = build_agent(model, actions, num_steps)
"""