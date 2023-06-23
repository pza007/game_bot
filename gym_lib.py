import time

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

import spectator_lib
from spectator_lib import Spectator


class MyEnv(gym.Env):
    def __init__(self):
        # (0,1)             18 possible actions
        self.action_space = gym.spaces.Discrete(18)
        # (uint)            bot health current, max
        # (uint)
        # (uint)            bot mana current, max
        # (uint)
        # (int16)           vector (dx,dy) from gate to bot's icon  !(minimap)!
        # (int16)
        # (int16)           vector (dx,dy) from bot to the closest blue minion (screen)
        # (int16)
        # (int16)           vector (dx,dy) from bot to the closest red minion (screen)
        # (int16)
        # (0,1)             bot cooldowns
        # (0,1)
        # (0,1)
        # (0,1)
        # (0,1)
        # (0,1)
        self.observation_space = gym.spaces.Box(low=-2000, high=2000, shape=(1, 16), dtype=np.int16)  # Integer (-32768 to 32767)
        self.spectator = Spectator()

        self._prev_observation = None
        self._observation = None
        self._action = None
        self._action_name = None
        self._action_result = None
        self._action_result_desc = None
        self._reward = 0
        self._reward_total = 0
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
        self._reward_total = 0
        self._done = False
        self._done_reason = None

    def get_info(self):
        if self._done:
            out_str = f' Reward={self._reward}, Total={self._reward_total}. Game finished: reason={self._done_reason}, '
        else:
            if self._action_result > 0:  # success
                out_str = f' Reward={self._reward}. Action={self._action_name}'
            else:  # fail
                out_str = f' Reward={self._reward}.\tfailed action={self._action_name}, reason={self._action_result_desc}.'
        print(out_str)
        return {'info': out_str}

    def render(self, mode="human"):
        pass

    def reset(self, seed=None, options=None):
        print('......Reseting......')
        super().reset(seed=seed)    # reset seed if you want to use here: self.np_random.integers(0, 1)
        self.reset_parameters()
        self.spectator.init_game()
        return [0]*16, {}

    def step(self, action):
        step_reward = None

        # done
        self._done, self._done_reason = self.spectator.is_game_finished()

        # action, result
        if not self._done:  # wait for action to be finished
            self._action = action
            self._action_name = self.get_action_name(action)
            self._action_result, self._action_result_desc, step_reward = self.spectator.execute_action(self._action_name)
        else:
            self._action, self._action_name, self._action_result, self._action_result_desc = None, None, None, None

        # observations
        if self._prev_observation is None:
            self._prev_observation = self.spectator.get_game_state()
        else:
            self._prev_observation = self._observation
        self._observation = self.spectator.get_game_state()

        # reward
        self._reward = self.get_reward(step_reward)
        self._reward_total += self._reward

        return self._observation, self._reward, self._done, False, self.get_info()

    def get_reward(self, step_reward):
        """
        out: reward
        """
        def get_dist_change(prev_vector, curr_vector):
            if prev_vector != (0, 0) and curr_vector != (0, 0):
                prev_dist = math.sqrt(prev_vector[0] ** 2 + prev_vector[1] ** 2)
                curr_dist = math.sqrt(curr_vector[0] ** 2 + curr_vector[1] ** 2)
                return curr_dist - prev_dist
            else:
                return None

        #self._observation = [0]*16
        # (uint), (uint)        bot health current, max
        # (uint), (uint)        bot mana current, max
        # (int16), (int16)      vector (dx,dy) from gate to bot's icon  !(minimap)!
        # (int16), (int16)      vector (dx,dy) from bot to the closest blue minion (screen)
        # (int16), (int16)      vector (dx,dy) from bot to the closest red minion (screen)
        # (0,0,0,0,0,0)         bot cooldowns
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
        # bot is behind blue minion
        blue_minion = self._observation[6:8]    # (dx, dy)
        if None not in blue_minion and blue_minion[0] > 0:
            reward += 5
        # bot is close to red minion
        red_minion = self._observation[8:10]    # (dx, dy)
        if None not in red_minion:
            reward += 10
        # SPEED
        #if self._action_name is not None and step_reward is not None:
        #    t_max = self.spectator.actions.objects[self._action_name].TIMEOUT
        #    t0 = self.spectator.actions.objects[self._action_name].t0
        #    t_diff = time.time() - t0
        #    factor = (t_max - t_diff) / (3*t_max)
        #    reward += factor*step_reward    # short action with big xp_gain is promoted


        # ------ REWARD
        # bot is dead
        if self._done_reason is not None and self._done_reason == 'bot_dead':
            reward -= 1000
        # action was not successful
        if self._action_result is not None and self._action_result < 0:
            reward -= 15
        # bot was close to red minion but now he lost it from sight
        prev_red_minion = self._prev_observation[8:10]    # (dx, dy)
        red_minion = self._prev_observation[8:10]    # (dx, dy)
        if None not in prev_red_minion and None in red_minion:
            reward -= 50
        # bot moved behind the gate, even though bots'HP > 10%
        if None not in [self._action_name, self._observation[0], self._observation[1]]:
            if self._action_name in ['hide_behind_gate', 'escape_behind_gate'] \
                    and self._observation[0] / self._observation[0] > 0.1:
                reward -= 500

        """
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
    out_model.add(Dense(24, activation='relu'))
    out_model.add(Dense(24, activation='relu'))
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


env = MyEnv()
env = FlattenObservation(env)
