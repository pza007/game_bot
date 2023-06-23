# TODO: Spectator
#   - record inputs at the moment that action was accepted
#   - record actions (only the ones that started)
#   * if bot is dead -> wait for alive, move to gate, demount (press y)
#   * if end time reached:
#       - if bot alive -> move to gate, check xp gain, refresh forts
#       - if bot dead -> wait for alive, move to gate, demount (press y), check xp gain, refresh forts
#       - save records to file, clear data
import random
import time

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
        self.t_start = None

        # do not change!
        self.__TIMEOUT = 30   # sec

    def init_game(self):
        def wait_bot_alive(max_time=30):
            if self.data['bot_dead']:
                t0 = time.time()
                while time.time() - t0 < max_time:
                    self.get_data()
                    if not self.data['bot_dead']:
                        pyautogui.press('y')  # demount
                        return
                raise Exception(f'Bot still dead after {max_time} sec of waiting')
            else:
                return

        def wait_bot_stopped(max_time=10):
            # stop command
            pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
            pyautogui.click(button='left')
            pyautogui.press('s')

            t0 = time.time()
            prev_pos = self.data['bot_pos_minimap']
            while time.time() - t0 < max_time:
                self.get_data()
                new_pos = self.data['bot_pos_minimap']
                if (abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1])) < 5:  # pixels
                    return
                prev_pos = new_pos
            raise Exception(f'Bot still not stopped after {max_time} sec of waiting')

        def reset_level():
            pyautogui.moveTo(221, 70)   # "Set Level" button
            pyautogui.click(button='left')
            time.sleep(1)
            pyautogui.click(button='left')
            time.sleep(1)
            pyautogui.press('y')  # demount

        def wait_bot_at_gate(max_time=10):
            self.get_data()
            self.actions = Actions()
            #f_can, f_can_desc = self.actions.objects['hide_behind_gate'].can_be_started(**self.data)
            f_started, f_started_desc = self.actions.start('hide_behind_gate')
            if not f_started:
                raise Exception(f_started_desc)

            t0 = time.time()
            while time.time() - t0 < max_time:
                self.actions.process(**self.data)
                self.get_data()
                if self.actions.current_action is None:
                    return
            raise Exception(f'Bot still not at gate after {max_time} sec of waiting')

        def refresh_forts(sleep_time=2):
            pyautogui.moveTo(132, 442)
            pyautogui.click(button='left')
            time.sleep(sleep_time)

        def wait_blue_minion_in_front(max_time=10):
            t0 = time.time()
            while time.time() - t0 < max_time:
                self.get_data()
                try:
                    for (x, y) in self.data['minions']['blue']:
                        if x > self.data['bot_pos_frame']['bounding_box'][0]:
                            return
                except Exception:
                    pass
            raise Exception(f'Blue minion still not in front of bot after {max_time} sec of waiting')

        self.get_data()

        #wait_bot_alive() and  wait_bot_stopped() -> simplified: reset_level()
        reset_level()
        # bot at gate?
        wait_bot_at_gate()
        # refresh_forts
        refresh_forts()
        # blue minion in front of bot?
        wait_blue_minion_in_front()

        # set initial values
        self.actions = Actions()
        self.data = {}
        self.t_start = time.time()

    def get_data(self):
        # bot_dead          True or False
        # bot_pos_frame     {'health_bar': (x, y), 'bounding_box': (x, y), 'circle': (x, y)}
        # bot_pos_minimap   (x, y)
        # bot_health        (curr, max)
        # bot_mana          (curr, max)
        # cooldowns         {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}
        # minions           {'blue': [(x_center, y_center), ...], 'red': [(x_center, y_center), ...]}
        # xp                value

        frame = self.window.get_screenshot()
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # screenshot = cv.imread("imgs\\in_img.png", cv.IMREAD_UNCHANGED)
        # cv.imshow('Computer Vision', screenshot)

        # BOT
        bot_dead, err_desc = is_bot_dead(frame)
        if not bot_dead:
            bot_pos_frame, desc = get_bot_positions(frame_hsv)
            if type(bot_pos_frame) is dict and len(bot_pos_frame) > 0:
                bot_pos_frame = {'health_bar': bot_pos_frame['health_bar'][1],
                                 'bounding_box': bot_pos_frame['bounding_box'][1],
                                 'circle': bot_pos_frame['circle'][1]}
            else:
                bot_pos_frame = {'health_bar': (None, None),
                                 'bounding_box': (None, None),
                                 'circle': (None, None)}
            bot_pos_minimap, desc = get_bot_icon_position(frame)   # (x, y) or (None, None)
            if bot_pos_minimap is None:
                bot_pos_minimap = (None, None)  # error
            bot_health, err_desc = get_bot_health_value(frame)   # (curr, max) or (None, None)
            bot_mana, err_desc = get_bot_mana_value(frame)   # (curr, max) or (None, None)
            cooldowns, err_desc = get_cooldowns(frame)    # {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}
        else:
            bot_pos_frame = {'health_bar': (None, None), 'bounding_box': (None, None), 'circle': (None, None)}
            bot_pos_minimap = (None, None)
            bot_health = (None, None)
            bot_mana = (None, None)
            cooldowns = {'Q': True, 'W': True, 'E': True, 'R': True, 'D': True, 'well': True}

        # MINIONS
        minions, err_desc = get_minions_positions(frame, frame_hsv)   # {'blue': [(x_center, y_center), ...], 'red': [(x_center, y_center), ...]}

        # XP
        xp, err_desc = get_xp_from_level(frame)

        self.data = {
            'frame': frame,
            'frame_hsv': frame_hsv,
            'bot_dead': bot_dead,
            'bot_pos_frame': bot_pos_frame,
            'bot_pos_minimap': bot_pos_minimap,
            'bot_health': bot_health,
            'bot_mana': bot_mana,
            'cooldowns': cooldowns,
            'minions': minions,
            'xp': xp
        }

    def execute_action(self, action_name):
        """
        out: result, description, xp_gain
        """
        self.get_data()
        xp_prev = self.data['xp']

        result, description = None, None
        is_available, err_desc = self.actions.objects[action_name].is_available(**self.data)
        if is_available:
            started, err_desc = self.actions.start(action_name)
            if started:
                while self.actions.current_action is not None:
                    self.actions.process(**self.data)
                    self.get_data()
                    if result is None and self.actions.current_action.result in [-1, 1]:
                        result = self.actions.current_action.result
                        description = self.actions.current_action.description
                # exit when self.actions.current_action is None
                xp_current, err_desc = get_xp_from_level(self.data['frame'], xp_prev=xp_prev)
                if xp_current is None:
                    return -1, f'Action failed during process. Reason:{err_desc}', 0
                return result, description, xp_current-xp_prev
            else:
                return -1, f'Action failed at starting. Reason:{err_desc}', 0
        else:
            return -1, f'Action is not available. Reason:{err_desc}', 0
        # self.actions.printout()
        #list_of_actions = self.actions.get_available_actions(**self.data)

    def is_game_finished(self):
        """
        out:  finished, reason
        """
        self.get_data()
        if self.data['bot_dead']:
            return True, 'bot_dead'
        if time.time() - self.t_start >= self.__TIMEOUT:
            return True, 'timeout'
        else:
            return False, None

    def get_game_state(self):
        def get_vector_closest_minion(color):
            x_bot, y_bot = self.data['bot_pos_frame']['bounding_box']
            out = []
            for x_minion, y_minion in self.data['minions'][color]:    # [(x_center, y_center), ...]
                dist, err_desc = get_distance_between_points((x_bot, y_bot), (x_minion, y_minion))
                out.append((dist, (x_bot-x_minion, y_bot-y_minion)))
            out.sort()
            return out[0][1]

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
        out = [0]*16
        # bot health
        if None not in self.data['bot_health']:
            out[0] = self.data['bot_health'][0]     # current
            out[1] = self.data['bot_health'][1]     # max
        # bot mana
        if None not in self.data['bot_mana']:
            out[2] = self.data['bot_mana'][0]     # current
            out[3] = self.data['bot_mana'][1]     # max
        # vector (dx,dy) from gate to bot's icon  !(minimap)!
        if None not in self.data['bot_pos_minimap']:
            gate_pos_minimap = (1654, 912)
            bot_pos_minimap = self.data['bot_pos_minimap']  # (x,y)
            out[4] = gate_pos_minimap[0] - bot_pos_minimap[0]   # dx
            out[5] = gate_pos_minimap[1] - bot_pos_minimap[1]   # dy
        if None not in self.data['bot_pos_frame']['bounding_box']:
            # vector (dx,dy) from bot to the closest blue minion (screen)
            if self.data['minions']['blue']:
                out[6], out[7] = get_vector_closest_minion('blue')
            # vector (dx,dy) from bot to the closest red minion (screen)
            if self.data['minions']['red']:
                out[8], out[9] = get_vector_closest_minion('red')
        # bot cooldowns
        out[10:] = tuple([int(val) for val in self.data['cooldowns'].values()])

        return out

    # obsolete functions
    """
    def get_current_xp(self):
        # Note: to read value, need to move mouse to position (x=960, y=44) and wait 1.5 sec
        pyautogui.moveTo(960, 44)
        pyautogui.click(button='left')
        time.sleep(1.5)
        frame = self.window.get_screenshot()
        return get_total_xp_value(frame)
        
    def get_xp_gain(self):
        return self.get_current_xp() - self.xp_start
        
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
    """


