import time
import json
from spectator_lib import Spectator
from functions import get_distance_between_points, Actions


class AI:
    """
    AI ALGORITHM
    # if no red minion is available:
    #   1. 'hide_in_bushes' and wait

    # if red minions are available:
    #   * 1,2 minions: 'basic_attack'
    #   * 3>= minions: 'q_attack'
    #   * 4>= minions: 'w_attack'

    # every 10 sec:
    #   1. 'collect_globes'
    #   2. 'run_middle' + 'collect_globes'
    #   [finally] move behind the closest blue minion OR 'hide_in_bushes'

    # if bot's mana < 5%:
    #   1. 'use_spell_d'
    #   2. 'use_well'

    # if bot's health < 15%:
    #   1. 'use_spell_d'
    #   2. 'hide_behind_gate' + 'use_spell_d'
    #   3. 'escape_behind_gate' + 'use_spell_d'
    #   4. 'use_well'
    """
    def __init__(self):
        self.spectator = Spectator()
        self.xp_start = None
        self.xp_total = None
        self.action_names = []      # filled inside: self.execute_actions()
        self.action_results = []    # filled inside: self.execute_actions()
        # not cleared
        self.buffer = {}

    def init_game(self):
        self.spectator.init_game()
        self.xp_start = None
        self.xp_total = None
        self.action_names = []
        self.action_results = []

    def play_game(self, num_episodes):
        for i in range(num_episodes):
            print(f'-----------------------------------\nEpisode = {i + 1}\n-----------------------------------')
            self.init_game()
            self.xp_start, err_desc = self.spectator.get_xp_value()
            finished = False
            finished_desc = None

            while not finished:
                self.spectator.get_data()
                action_names = self.get_action_names()
                self.execute_actions(action_names)
                finished, finished_desc = self.spectator.is_game_finished()

            xp, err_desc = self.spectator.get_xp_value()
            if None not in [xp, self.xp_start]:
                self.xp_total = xp - self.xp_start
            else:
                self.xp_total = 0

            # fill buffer
            self.buffer.update({i+1: {'xp_total': self.xp_total,
                                      'num_steps': len(self.action_names),
                                      'steps': {idx+1: {'action_name': action_name, 'action_result': action_result} for idx, (action_name, action_result) in enumerate(zip(self.action_names, self.action_results))}
                                      }})

            # printout
            print(f'Game lasted = {(time.time() - self.spectator.t_start):.2f} sec. Reason = {finished_desc}')
            print(f'Bot executed {len(self.action_names)} actions. Success rate = {((self.action_results.count(1)/len(self.action_results))*100):.2f} %')
            print(f'Total experience gained = {self.xp_total}')

        self.save_data()

    def save_data(self):
        # save JSON file
        xp_max = max([value['xp_total'] for value in self.buffer.values()])
        filename = f'ai/ai_XP{xp_max}_data.json'
        with open(filename, "w") as file:
            json.dump(self.buffer, file)

    def get_action_names(self) -> list[str | None]:
        data = self.spectator.data
        """
        self.spectator.data = {
            'frame': None or np.ndarray
            'frame_hsv': None or np.ndarray
            'bot_dead': None or bool
            'bot_pos_frame': None or {'health_bar': tuple[int, int], 'bounding_box': tuple[int, int], 'circle': tuple[int, int]}
            'bot_pos_minimap': None or tuple[int, int]
            'bot_health': None or tuple[int, int]
            'bot_mana': None or tuple[int, int]
            'cooldowns': None or {'Q': bool, 'W': bool, 'E': bool, 'R': bool, 'D': bool, 'well': bool}
            'minions': None or {'blue': list[tuple[int, int]], 'red': list[tuple[int, int]]}
        }
        """
        def get_move_actions():
            # inputs
            if data['minions'] is None:
                return None
            if len(data['minions']['blue']) == 0:
                return None
            if data['bot_pos_frame'] is None:
                return None

            move_names = []
            scale = 16; move_dx = 8; move_dy = 8
            bot_x, bot_y = data['bot_pos_frame']['circle']
            tmp = [(get_distance_between_points((bot_x, bot_y), (x, y))[0], (x,y)) for x,y in data['minions']['blue']]
            tmp.sort()
            minion_x, minion_y = tmp[0][1]
            dx, dy = (bot_x - minion_x)//scale, (bot_y - minion_y)//scale
            if dx < 0:  # right
                for _ in range(dx // move_dx + 1): move_names.append('move_right')
            if dx > 0:  # left
                for _ in range(dx // move_dx + 1): move_names.append('move_left')
            if dy < 0:  # down
                for _ in range(dy // move_dy + 1): move_names.append('move_down')
            if dy > 0:  # up
                for _ in range(dy // move_dy + 1): move_names.append('move_up')
            return move_names

        action_names = [None]
        # if no red minion is available:
        #   1. 'hide_in_bushes' and wait
        if data['minions'] is not None and len(data['minions']['red']) == 0:
            available, err_desc = self.spectator.actions.objects['hide_in_bushes'].is_available(**data)
            if available:
                action_names = ['hide_in_bushes']

        # if red minions are available:
        #   * 1,2 minions: 'basic_attack'
        #   * 3>= minions: 'q_attack'
        #   * 4>= minions: 'w_attack'
        if data['minions'] is not None and len(data['minions']['red']) > 0:
            action_names = ['basic_attack']
            if 1 <= len(data['minions']['red']) <= 2:
                available, err_desc = self.spectator.actions.objects['basic_attack'].is_available(**data)
                if available:
                    action_names = ['basic_attack']
            if 3 <= len(data['minions']['red']):
                available, err_desc = self.spectator.actions.objects['q_attack'].is_available(**data)
                if available:
                    action_names = ['q_attack']
            if 4 <= len(data['minions']['red']):
                available, err_desc = self.spectator.actions.objects['w_attack'].is_available(**data)
                if available:
                    action_names = ['w_attack']

        # every 10 sec:
        #   1. 'collect_globes'
        #   2. 'run_middle' + 'collect_globes'
        #   [finally] move behind the closest blue minion OR 'hide_in_bushes'
        if self.action_names.count('collect_globes') < (time.time() - self.spectator.t_start) // 10:
            # 1.
            available, err_desc = self.spectator.actions.objects['collect_globes'].is_available(**data)
            if available:
                action_names = ['collect_globes']
            # 2.
            else:
                available, err_desc = self.spectator.actions.objects['run_middle'].is_available(**data)
                if available:
                    action_names = ['run_middle', 'collect_globes']
            # [finally]
            if 'collect_globes' in action_names:
                move_actions = get_move_actions()
                if move_actions is not None:
                    action_names += move_actions    # move behind the closest blue minion
                else:
                    action_names += ['hide_in_bushes']  # OR 'hide_in_bushes'

        # if bot's mana < 5%:
        #   1. 'use_spell_d'
        #   2. 'use_well'
        if data['bot_mana'] is not None and data['bot_mana'][0] / data['bot_mana'][1] < 0.05:
            available, err_desc = self.spectator.actions.objects['use_spell_d'].is_available(**data)
            if available:
                action_names = ['use_spell_d']
            else:
                available, err_desc = self.spectator.actions.objects['use_well'].is_available(**data)
                if available:
                    action_names = ['use_well']

        # if bot's health < 15%:
        #   1. 'use_spell_d'
        #   2. 'hide_behind_gate' + 'use_spell_d'
        #   3. 'escape_behind_gate' + 'use_spell_d'
        #   4. 'use_well'
        if data['bot_health'] is not None and data['bot_health'][0] / data['bot_health'][1] < 0.15:
            available, err_desc = self.spectator.actions.objects['use_spell_d'].is_available(**data)
            if available:
                action_names = ['use_spell_d']
            else:
                available, err_desc = self.spectator.actions.objects['hide_behind_gate'].is_available(**data)
                if available:
                    action_names = ['hide_behind_gate', 'use_spell_d']
                else:
                    available, err_desc = self.spectator.actions.objects['escape_behind_gate'].is_available(**data)
                    if available:
                        action_names = ['escape_behind_gate', 'use_spell_d']
                    else:
                        available, err_desc = self.spectator.actions.objects['use_well'].is_available(**data)
                        if available:
                            action_names = ['use_well']

        return action_names

    def execute_actions(self, action_names: list[str | None]) -> None:
        data = self.spectator.data
        self.spectator.actions = Actions()

        if action_names[0] is None:
            return

        for idx in range(len(action_names)):
            action_name = action_names[idx]
            self.action_names.append(action_name)

            # check preconditions
            if idx == 0:
                available = True    # first action is always available -> checked in: get_action_names()
                started, err_desc = self.spectator.actions.start(action_name)
            else:
                available, err_desc = self.spectator.actions.objects[action_name].is_available(**data)
                started, err_desc = self.spectator.actions.start(action_name)
            if not available or not started:
                result = -1  # error
                self.action_results.append(result)
            else:
                # wait until action has been finished with result
                result = -1
                while self.spectator.actions.current_action is not None:
                    self.spectator.get_data()
                    data = self.spectator.data
                    result, err_desc = self.spectator.actions.process(**data)
                    if result is None:  # None = error
                        break
                    if result == -1:
                        break
                    elif result in [-1, 1]:  # 1 = finished successfully, -1 = finished unsuccessfully
                        break
                # action finished
                self.action_results.append(result)
