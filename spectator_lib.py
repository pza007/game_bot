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

import functions
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
        self.__TIMEOUT = 60   # [sec] time of game

    def init_game(self):
        """
        Initializes the game. After the function is over:
        - the level is reset
        - bot is at gate
        - forts are refreshed
        - blue minion is in front of bot
        - actions, data, t_start is reset

        Args:
            self

        Returns:
            None

        Raises:
            Exception in: wait_bot_at_gate()
            Exception in: wait_blue_minion_in_front()
        """
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

        def wait_bot_at_gate():
            # trigger actions
            self.actions = Actions()
            self.get_data()
            available, err_desc = self.actions.objects['hide_behind_gate'].is_available(**self.data)
            if not available:
                raise Exception(err_desc)
            started, err_desc = self.actions.start('hide_behind_gate')
            if not started:
                raise Exception(err_desc)

            # wait for action's result
            while self.actions.current_action is not None:
                self.get_data()
                result, err_desc = self.actions.process(**self.data)
                if result is None or result == -1:  # None = error, -1 = finished unsuccessfully
                    raise Exception(err_desc)
                if result == 1:     # 1 = finished successfully,
                    break  # action finished
                # timeout included in actions.process()

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
        """
        Get information from game's screenshot (frame). Fills variable: self.data with values.

        Args:
            self

        Returns:
            None
            self.data = {
            'frame':            None or  np.ndarray
            'frame_hsv':        None or  np.ndarray
            'bot_dead':         None or  bool
            'bot_pos_frame':    None or  {'health_bar': tuple[int, int], 'bounding_box': tuple[int, int], 'circle': tuple[int, int]}
            'bot_pos_minimap':  None or  tuple[int, int]
            'bot_health':       None or  tuple[int, int]
            'bot_mana':         None or  tuple[int, int]
            'cooldowns':        None or  {'Q': bool, 'W': bool, 'E': bool, 'R': bool, 'D': bool, 'well': bool}
            'minions':          None or  {'blue': list[tuple[int, int]], 'red': list[tuple[int, int]]}
            }

        Raises:
            Exception in: wait_bot_at_gate()
            Exception in: wait_blue_minion_in_front()
        """
        # initial values
        self.data = {
            'frame': None,
            'frame_hsv': None,
            'bot_dead': None,
            'bot_pos_frame': None,
            'bot_pos_minimap': None,
            'bot_health': None,
            'bot_mana': None,
            'cooldowns': None,
            'minions': None
        }

        # frame, frame_hsv
        frame = self.window.get_screenshot()
        if frame is None:
            return  # error ! no other data can be extracted in "frame" is None
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # bot_dead
        bot_dead, err_desc = is_bot_dead(frame)
        # bot_pos_frame
        bot_pos_frame, err_desc = get_bot_positions(frame_hsv)
        if bot_pos_frame is not None:
            bot_pos_frame = {'health_bar': bot_pos_frame['health_bar'][1],
                             'bounding_box': bot_pos_frame['bounding_box'][1],
                             'circle': bot_pos_frame['circle'][1]}
        # bot_pos_minimap
        bot_pos_minimap, desc = get_bot_icon_position(frame)
        # bot_health, bot_mana
        bot_health, err_desc = get_bot_health_value(frame)
        bot_mana, err_desc = get_bot_mana_value(frame)
        # cooldowns
        cooldowns, err_desc = get_cooldowns(frame)
        # minions
        minions, err_desc = get_minions_positions(frame, frame_hsv)

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

    def execute_action(self, action_name: str) -> tuple[int, str, int] | tuple[int, None, int]:
        """
        Execute action and wait for it's result

        Args:
            action_name (str): name of the action to be executed

        Returns:
            tuple[int, str, int] | tuple[int, None, int]: action_result[-1,1], action_description[str or None], xp_gain

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # inputs
        if not isinstance(action_name, str):
            return -1, f'action_name: {action_name} is not str', 0
        if action_name not in ['move_up', 'move_down', 'move_right', 'move_left', 'move_up-right', 'move_down-right',
                               'move_up-left', 'move_down-left', 'run_middle', 'collect_globes', 'basic_attack',
                               'q_attack', 'w_attack', 'use_well', 'hide_in_bushes', 'hide_behind_gate',
                               'escape_behind_gate', 'use_spell_d']:
            return -1, f'action_name: {action_name} not defined', 0

        self.get_data()
        xp_prev, err_desc = self.get_xp_value()
        if xp_prev is None:
            return -1, f'xp_prev is None, Reason:{err_desc}', 0

        action_result, action_description = None, None
        available, err_desc = self.actions.objects[action_name].is_available(**self.data)
        if available:
            started, err_desc = self.actions.start(action_name)
            if started:
                while self.actions.current_action is not None:
                    self.get_data()
                    action_result, action_description = self.actions.process(**self.data)
                    if action_result is None:
                        return -1, f'Action failed during process. Reason:{action_description}', 0
                    if action_result in [-1, 1]:  # 1 = finished successfully, -1 = finished unsuccessfully
                        break   # action finished

                # when current_action is None -> get current xp
                xp_current, err_desc = self.get_xp_value()
                if xp_current is None:
                    return -1, f'xp_current is None, Reason:{err_desc}', 0
                else:
                    # final output
                    return action_result, action_description, max(xp_current-xp_prev, 0)
            else:
                return -1, f'Action failed at starting. Reason:{err_desc}', 0
        else:
            return -1, f'Action is not available. Reason:{err_desc}', 0

    def is_game_finished(self) -> tuple[bool, None] | tuple[bool, str]:
        """
        Get if game is finished or not and the reason for finish

        Args:
            self

        Returns:
            tuple[bool, None] | tuple[bool, str]: game_finished, finish_reason

        Raises:
            None: This function does not raise any specific exceptions.
        """
        self.get_data()
        if self.data['bot_dead']:
            return True, 'bot_dead'
        if time.time() - self.t_start >= self.__TIMEOUT:
            return True, 'timeout'
        else:
            return False, None

    def get_game_state(self) -> list[float]:
        """
        Get game's state based on self.data gathered in self.get_data()

        Args:
            self

        Returns: list[0..10] of float[0.0; 1.0]
            - out[0]      float [0.0; 1.0]     bot health = current / max
            - out[1]      float [0.0; 1.0]     bot mana = current / max
            - out[2]      float [0.0; 1.0]     distance on minimap between gate and bot's icon = distance/100
            - out[3]      float [0.0; 1.0]     distance on screen between bot and closest BLUE minion = 1/distance
            - out[4]      float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
            - out[5..10]  6*int [0 or 1]       6*cooldown value

        Raises:
            None: This function does not raise any specific exceptions.
        """

        def get_distance_to_closest_minion(color: str) -> float | None:
            """
            Calculate the distance to the closest minion of defined color

            Args:
                color (str): 'red' or 'blue'

            Returns:
                float | None: The distance to the closest minion or None in case there is no minion or error occurred

            Raises:
                None: This function does not raise any specific exceptions.
            """
            # input - color
            if not isinstance(color, str):
                return None
            if color not in ['red', 'blue']:
                return None
            # input - bot_pos_frame
            if self.data['bot_pos_frame'] is None:
                return None
            # input - minions
            if self.data['minions'] is None:
                return None
            if len(self.data['minions'][color]) == 0:   # no minions
                return None

            bot_pos_screen = self.data['bot_pos_frame']['bounding_box']
            distances = []
            for minion_pos_screen in self.data['minions'][color]:  # [(x_center, y_center), ...]
                dist, err_desc = get_distance_between_points(bot_pos_screen, minion_pos_screen)
                if dist is None:
                    return None  # error!
                distances.append(dist)
            if len(distances) == 0:
                return None
            else:
                distances.sort()
                return distances[0]

        out = [0.0] * 11
        # bot health = current / max
        if self.data['bot_health'] is not None:
            out[0] = self.data['bot_health'][0] / self.data['bot_health'][1]
        #  bot mana = current / max
        if self.data['bot_mana'] is not None:
            out[1] = self.data['bot_mana'][0] / self.data['bot_mana'][1]
        # distance on minimap from gate to bot's icon = distance/100
        if self.data['bot_pos_minimap'] is not None:
            gate_pos_minimap = (1654, 912)
            bot_pos_minimap = self.data['bot_pos_minimap']  # (x,y)
            dist, err_desc = get_distance_between_points(gate_pos_minimap, bot_pos_minimap)
            if dist is not None:
                out[2] = dist / 100
        # distance on screen from bot to the closest blue minion = 1/distance
        dist = get_distance_to_closest_minion('blue')
        if dist is not None and dist != 0:
            out[3] = 1/dist
        # distance on screen from bot to the closest red minion = 1/distance
        dist = get_distance_to_closest_minion('red')
        if dist is not None and dist != 0:
            out[4] = 1/dist
        # cooldowns
        if self.data['cooldowns'] is not None:
            out[5:] = tuple([int(val) for val in self.data['cooldowns'].values()])

        return out

    def get_xp_value(self) -> tuple[int, None] | tuple[None, str]:
        """
        Trigger 'TAB' key press, read xp value, press 'TAB' key again to hide window

        Args:
            self

        Returns:
            tuple[int, None] | tuple[None, str]: xp value[int or None], error description[str or None]

        Raises:
            None: This function does not raise any specific exceptions.
        """
        pyautogui.moveTo(SCREEN_W//2, SCREEN_H//2)
        pyautogui.click(button='left')

        pyautogui.press('tab')
        time.sleep(0.2)

        frame = self.window.get_screenshot()
        if frame is None:
            return None, 'frame is None'  # error !

        xp, err_desc = functions.get_xp_from_tab(frame)

        pyautogui.press('tab')
        time.sleep(0.1)

        return xp, err_desc


    # OBSOLETE functions
    """
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


