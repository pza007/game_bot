# TODO: Spectator
#   - record inputs at the moment that action was accepted
#   - record actions (only the ones that started)
#   * if bot is dead -> wait for alive, move to gate, demount (press y)
#   * if end time reached:
#       - if bot alive -> move to gate, check xp gain, refresh forts
#       - if bot dead -> wait for alive, move to gate, demount (press y), check xp gain, refresh forts
#       - save records to file, clear data
import random

from window_capture import WindowCapture
import cv2 as cv
from functions import *
import datetime


class Spectator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.window = WindowCapture('Heroes of the Storm')
        self.actions = Actions()
        self.data = {}
        self.buffer = []
        self.t_start = None
        self.iteration = None

        # do not reset!
        self.__xp = self.get_current_xp()
        self.__TIMEOUT = 30   # sec
        self.__max_iterations = 2

    def run(self):
        # init values
        self.t_start = time.time()
        self.iteration = 0

        while self.iteration < self.__max_iterations:

            self.get_data()

            # end conditions
            if self.data['bot_dead']:
                self.cleanup('dead')
                self.t_start = time.time()
                self.iteration += 1

            elif time.time() - self.t_start >= self.__TIMEOUT:
                self.cleanup('timeout')
                self.t_start = time.time()
                self.iteration += 1

            else:
                self.execute_action()


    def get_data(self):
        # bot_dead          True or False
        # bot_pos_frame     {'health_bar': (x, y), 'bounding_box': (x, y), 'circle': (x, y)}
        # bot_pos_minimap   (x, y)
        # bot_health        (curr, max)
        # bot_mana          (curr, max)
        # cooldowns         {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}
        # minions           {color: [(x_center, y_center), ...] }

        frame = self.window.get_screenshot()
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # screenshot = cv.imread("imgs\\in_img.png", cv.IMREAD_UNCHANGED)
        # cv.imshow('Computer Vision', screenshot)

        # BOT
        bot_dead = is_bot_dead(frame)
        if not bot_dead:
            bot_pos_frame, desc = get_bot_positions(frame_hsv)
            if not bot_pos_frame:
                bot_pos_frame = {'health_bar': (None, None), 'bounding_box': (None, None), 'circle': (None, None)}

            bot_pos_minimap = get_bot_icon_position(frame)   # (x, y) or (None, None)

            bot_health = get_bot_health_value(frame)   # (curr, max) or (None, None)

            bot_mana = get_bot_mana_value(frame)   # (curr, max) or (None, None)

            cooldowns = get_cooldowns(frame)    # {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}
        else:
            bot_pos_frame = {'health_bar': (None, None), 'bounding_box': (None, None), 'circle': (None, None)}
            bot_pos_minimap = (None, None)
            bot_health = (None, None)
            bot_mana = (None, None)
            cooldowns = {'Q': True, 'W': True, 'E': True, 'R': True, 'D': True, 'well': True}

        # MINIONS
        minions = get_minions_positions(frame, frame_hsv)
        if not minions['red']:
            minions['red'] = [(None, None)]

        self.data = {
            'frame': frame,
            'frame_hsv': frame_hsv,
            'bot_dead': bot_dead,
            'bot_pos_frame': bot_pos_frame,
            'bot_pos_minimap': bot_pos_minimap,
            'bot_health': bot_health,
            'bot_mana': bot_mana,
            'cooldowns': cooldowns,
            'minions': minions
        }

    def execute_action(self):
        if not self.actions.current_action:
            list_of_actions = self.actions.get_available_actions(**self.data)
            action_name = list_of_actions[random.randint(0, len(list_of_actions)-1)]

            if self.actions.start(action_name, **self.data):
                self.actions.process(**self.data)
                self.record(action_name)
                print(f'Started action: {action_name}')

        else:
            self.actions.process(**self.data)
            #self.actions.printout()

    def record(self, action_name):
        """
        Record only if bot is alive and action started
        """
        # bot_pos_frame     (x, y)
        # bot_pos_minimap   (x, y)
        # bot_health        (curr, max)
        # bot_mana          (curr, max)
        # cooldowns         0 or 1   for 'Q','W','E','R','D','well'   e.g (0,0,0,0,0,1)
        # minions           (x1, y1, x2, y2, x3, y3, ...)
        # action            number:  0 = 'move_up'
        #                            1 = 'move_down'
        #                            2 = 'move_right'
        #                            3 = 'move_left'
        #                            4 = 'move_up-right'
        #                            5 = 'move_down-right'
        #                            6 = 'move_up-left'
        #                            7 = 'move_down-left'
        #                            8 = 'run_middle'
        #                            9 = 'collect_globes'
        #
        #                           10 = 'basic_attack'
        #                           11 = 'q_attack'
        #                           12 = 'w_attack'
        #
        #                           13 = 'use_well'
        #                           14 = 'hide_in_bushes'
        #                           15 = 'hide_behind_gate'
        #                           16 = 'escape_behind_gate'
        #                           17 = 'use_spell_d'

        bot_pos_frame = self.data['bot_pos_frame']['circle'][1]
        bot_pos_minimap = self.data['bot_pos_minimap']
        bot_health = self.data['bot_health']
        bot_mana = self.data['bot_mana']
        cooldowns = tuple([int(val) for val in self.data['cooldowns'].values()])
        minions = []
        for x, y in self.data['minions']['red']:
            minions.append(x)
            minions.append(y)
        minions = tuple(minions)
        action = ['move_up', 'move_down', 'move_right', 'move_left', 'move_up-right', 'move_down-right',
                  'move_up-left', 'move_down-left', 'run_middle', 'collect_globes',
                  'basic_attack', 'q_attack', 'w_attack',
                  'use_well', 'hide_in_bushes', 'hide_behind_gate', 'escape_behind_gate', 'use_spell_d'].index(action_name)

        self.buffer.append([action, bot_pos_frame, bot_pos_minimap, bot_health, bot_mana, cooldowns, minions])

    def store_data(self):
        xp_gain = self.get_xp_gain()
        filename = f'logs\\{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}_{xp_gain}.txt'
        header = '\t'.join(['action', 'bot_pos_frame', 'bot_pos_minimap', 'bot_health', 'bot_mana', 'cooldowns', 'minions'])
        lines = []
        for data in self.buffer:
            lines.append('\t'.join([str(val) for val in data]) + '\n')

        with open(filename, mode='w') as f:
            f.write(header + '\n')
            f.writelines(lines)

        # reset
        self.data = {}
        self.buffer = []

    def get_current_xp(self):
        # Note: to read value, need to move mouse to position (x=960, y=44) and wait 1.5 sec
        pyautogui.moveTo(960, 44)
        pyautogui.click(button='left')
        time.sleep(1.5)
        frame = self.window.get_screenshot()
        return get_total_xp_value(frame)

    # function changes variable: self.__xp !!
    def get_xp_gain(self):
        xp_current = self.get_current_xp()
        xp_gain = xp_current - self.__xp
        self.__xp = xp_current
        return xp_gain

    def cleanup(self, reason):
        def wait_bot_stopped(max_time=100):
            if self.actions.current_action is not None:
                self.actions.current_action.set_result(-1, 'Stopped.')
            self.actions.current_action = None

            pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
            pyautogui.click(button='left')
            pyautogui.press('s')

            prev_pos = (0, 0)
            cnt = 1
            for _ in range(max_time):
                self.get_data()
                new_pos = self.data['bot_pos_minimap']
                if (abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1])) < 5:  # pixels
                    break
                prev_pos = new_pos
                cnt += 1
            if cnt == 100:
                raise Exception(f'Bot still not stopped after {max_time} frames of waiting')

        def wait_bot_alive(max_time=100):
            for _ in range(max_time):
                self.get_data()
                if not self.data['bot_dead']:
                    break
            if self.data['bot_dead']:
                raise Exception(f'Bot still dead after {max_time} frames of waiting')

        def refresh_forts():
            pyautogui.moveTo(132, 442)
            pyautogui.click(button='left')
            time.sleep(4)

        def wait_bot_at_gate(max_time=100):
            self.get_data()
            self.actions = Actions()
            self.actions.start('hide_behind_gate', **self.data)
            for _ in range(max_time):
                self.actions.process(**self.data)
                self.get_data()
                if self.actions.current_action is None:
                    break
            if self.actions.current_action is not None:
                raise Exception(f'Bot still not at gate after {max_time} frames of waiting')

        print('>> cleanup', reason)
        self.store_data()

        if reason == 'dead':
            wait_bot_alive(200)         # wait until bot is alive again
            refresh_forts()             # refresh_forts
            pyautogui.press('y')        # demount
            wait_bot_at_gate(200)       # wait until bot is at gate
            return

        else:   # timeout
            wait_bot_stopped(100)       # wait until bot is stopped
            refresh_forts()             # refresh_forts
            wait_bot_at_gate(200)       # wait until bot is at gate
            return
