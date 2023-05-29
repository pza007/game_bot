import pyautogui
from global_vars import *
import time


def get_bot_positions(in_hsv):
    """
    out: [dict] or None, description [str]

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
out, desc = get_bot_positions(hsv)
(x_min, y_min, x_max, y_max), (x_center, y_center) = out['health_bar']
cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
cv.rectangle(frame, (x_center-1, y_center-1), (x_center+1, y_center+1), (0, 0, 255), 1)
(x_min, y_min, x_max, y_max), (x_center, y_center) = out['bounding_box']
cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
cv.rectangle(frame, (x_center-1, y_center-1), (x_center+1, y_center+1), (0, 0, 255), 1)
(x_min, y_min, x_max, y_max), (x_center, y_center) = out['circle']
cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
cv.rectangle(frame, (x_center-1, y_center-1), (x_center+1, y_center+1), (0, 0, 255), 1)
cv.imwrite('imgs\\_result.jpg', frame)
    """
    out = {
        'health_bar':     ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        'bounding_box':   ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        'circle':         ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
    }
    _HEALTH_BAR_COLOR_THR = [np.array([50, 100, 150]), np.array([55, 255, 255])]
    _HEALTH_BAR_BOX = [8, 11, 2, 11]     # height_min, height_max, width_min, width_max
    _BOT_BOX_W_MOUNT = 100
    _BOT_BOX_H_MOUNT = 150
    _BOT_BOX_H_SHIFT_MOUNT = 90
    _BOT_BOX_X_SHIFT_MOUNT = -10
    _BOT_BOX_W_UNMOUNT = 80
    _BOT_BOX_H_UNMOUNT = 110
    _BOT_BOX_H_SHIFT_UNMOUNT = 44
    _BOT_BOX_X_SHIFT_UNMOUNT = -5
    _BOT_CIRCLE_SHIFT_MOUNT = 62
    _BOT_CIRCLE_SHIFT_UNMOUNT = 44
    _BOT_CIRCLE_W = 43*2
    _BOT_CIRCLE_H = 36*2

    def get_health_bar():
        """ find bot's health bar, depending on it's position """
        def execute():
            hsv = in_hsv[shift_y_min:shift_y_max, shift_x_min:shift_x_max]
            mask = cv.inRange(hsv, _HEALTH_BAR_COLOR_THR[0], _HEALTH_BAR_COLOR_THR[1])
            contours = get_contours(mask)
            if len(contours) > 0:
                contours = filter_by_dimensions(contours)
                if len(contours) > 0:
                    set_positions(contours)
                    return True
            return False

        def get_contours(in_mask):
            in_contours, in_hierarchy = cv.findContours(in_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            out_contours = []
            for contour in in_contours:
                x_min = min([_contour[0][0] for _contour in contour])  # point X
                x_max = max([_contour[0][0] for _contour in contour])  # point X
                y_min = min([_contour[0][1] for _contour in contour])  # point Y
                y_max = max([_contour[0][1] for _contour in contour])  # point Y
                out_contours.append([(x_min, y_min), (x_max, y_max)])

            return out_contours

        def filter_by_dimensions(in_contours):
            out_contours = []
            for [(x_min, y_min), (x_max, y_max)] in in_contours:
                height = y_max - y_min + 1
                width = x_max - x_min + 1
                #print(f'height={height}, width={width}')
                if _HEALTH_BAR_BOX[0] <= height <= _HEALTH_BAR_BOX[1] and _HEALTH_BAR_BOX[2] <= width <= _HEALTH_BAR_BOX[3]:
                    out_contours.append([(x_min, y_min), (x_max, y_max)])

            return out_contours

        def set_positions(in_contours):
            # health bar
            y_min = min([y_min for [(x_min, y_min), (x_max, y_max)] in in_contours])
            pos = (shift_x_min, shift_y_min+y_min, shift_x_max, shift_y_max+y_min)
            center = (pos[0]+(pos[2]-pos[0])//2, pos[1]+(pos[3]-pos[1])//2)
            out['health_bar'] = (pos, center)

        # ON GROUND?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 346, 905+121, 346+9
        if execute(): return

        # ON HORSE?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 261, 905+121, 261+9
        if execute(): return

        # IN BUSHES?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 150, 905+121, 450
        if execute(): return

        return f'Cannot find position of bot health bar'

    def get_bounding_box():
        x_hb_min = out['health_bar'][0][0]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        y_hb_min = out['health_bar'][0][1]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        x_hb_middle = out['health_bar'][1][0]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))

        # bot mounted
        if y_hb_min < int(2 * SCREEN_H / 8):
            x_hb_middle += _BOT_BOX_X_SHIFT_MOUNT
            x_min = x_hb_middle - _BOT_BOX_W_MOUNT // 2
            x_max = x_hb_middle + _BOT_BOX_W_MOUNT // 2
            y_min = y_hb_min + _BOT_BOX_H_SHIFT_MOUNT
            y_max = y_min + _BOT_BOX_H_MOUNT
        else:
            # bot unmounted = on ground
            x_hb_middle += _BOT_BOX_X_SHIFT_UNMOUNT
            x_min = x_hb_middle - _BOT_BOX_W_UNMOUNT // 2
            x_max = x_hb_middle + _BOT_BOX_W_UNMOUNT // 2
            y_min = y_hb_min + _BOT_BOX_H_SHIFT_UNMOUNT
            y_max = y_min + _BOT_BOX_H_UNMOUNT

        pos = (x_min, y_min, x_max, y_max)
        center = (x_min+(pos[2]-pos[0])//2, y_min+(pos[3]-pos[1])//2)
        out['bounding_box'] = (pos, center)

    def get_circle():
        y_hb_min = out['health_bar'][0][1]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        x_center, y_center = out['bounding_box'][1]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))

        # bot mounted
        if y_hb_min < int(2 * SCREEN_H / 8):
            center = (x_center, y_center+_BOT_CIRCLE_SHIFT_MOUNT)
            x_min = center[0]-_BOT_CIRCLE_W//2
            y_min = center[1]-_BOT_CIRCLE_H//2
            x_max = center[0]+_BOT_CIRCLE_W//2
            y_max = center[1]+_BOT_CIRCLE_H//2
        else:
            # bot unmounted = on ground
            center = (x_center, y_center+_BOT_CIRCLE_SHIFT_UNMOUNT)
            x_min = center[0]-_BOT_CIRCLE_W//2
            y_min = center[1]-_BOT_CIRCLE_H//2
            x_max = center[0]+_BOT_CIRCLE_W//2
            y_max = center[1]+_BOT_CIRCLE_H//2

        pos = (x_min, y_min, x_max, y_max)
        center = (x_min+(pos[2]-pos[0])//2, y_min+(pos[3]-pos[1])//2)
        out['circle'] = (pos, center)

    description = get_health_bar()
    if description is not None:
        return None, description
    get_bounding_box()
    get_circle()

    return out, ''


def get_bot_health_value(in_frame):
    """
    out: (val_current, val_max)

from functions import get_bot_health_value
file_path = f'imgs\\detect\\health\\'
for cnt in range(1, 7):
    frame = cv.imread(f'{file_path}health_{cnt}.png', cv.IMREAD_UNCHANGED)
    print(f'health_{cnt}.png', get_bot_health_value(frame))
    """

    # get part of image
    x_min, y_min, x_max, y_max = 206, 992, 420, 1010
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get position of slash character
    template = template_H10_slash
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= res.max())
    if len(loc[0]) == 0:
        return None, None   # could not find slash position
    x_slash, y_slash = loc[::-1][0][0], loc[::-1][1][0]

    # get positions of numbers
    out = []  # (x,y,number)
    threshold = 0.8
    for idx, template in enumerate([template_H10_num_0, template_H10_num_1, template_H10_num_2,
                                    template_H10_num_3, template_H10_num_4, template_H10_num_5,
                                    template_H10_num_6, template_H10_num_7, template_H10_num_8,
                                    template_H10_num_9]):
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            out.append((pt[0], pt[1], idx))
    #   sort numbers (by x position)
    out.sort(key=lambda x: x[0])

    # create final output values
    val_current_str = ''.join([str(num) for (x, y, num) in out if x < x_slash])   # numbers before slash
    val_max_str = ''.join([str(num) for (x, y, num) in out if x > x_slash])   # numbers after slash
    if not val_current_str.isdigit() or not val_max_str.isdigit():
        return None, None   # could not detect numbers
    val_current = int(val_current_str)
    val_max = int(val_max_str)
    if val_current > val_max:
        return None, None   # error when detecting numbers

    return val_current, val_max


def get_bot_mana_value(in_frame):
    """
    out: (val_current, val_max)

from functions import get_bot_mana_value
file_path = f'imgs\\detect\\mana\\'
for cnt in range(1, 5):
    frame = cv.imread(f'{file_path}mana_{cnt}.png', cv.IMREAD_UNCHANGED)
    print(f'mana_{cnt}.png', get_bot_mana_value(frame))
    """

    # get part of image
    x_min, y_min, x_max, y_max = 198, 1013, 403, 1030
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get position of slash character
    template = template_H08_slash
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= res.max())
    if len(loc[0]) == 0:
        return None, None   # could not find slash position
    x_slash, y_slash = loc[::-1][0][0], loc[::-1][1][0]

    # get positions of numbers
    out = []  # (x,y,number)
    threshold = 0.8
    for idx, template in enumerate([template_H08_num_0, template_H08_num_1, template_H08_num_2,
                                    template_H08_num_3, template_H08_num_4, template_H08_num_5,
                                    template_H08_num_6, template_H08_num_7, template_H08_num_8,
                                    template_H08_num_9]):
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            out.append((pt[0], pt[1], idx))
    #   sort numbers (by x position)
    out.sort(key=lambda x: x[0])

    # create final output values
    val_current_str = ''.join([str(num) for (x, y, num) in out if x < x_slash])   # numbers before slash
    val_max_str = ''.join([str(num) for (x, y, num) in out if x > x_slash])   # numbers after slash
    if not val_current_str.isdigit() or not val_max_str.isdigit():
        return None, None   # could not detect numbers
    val_current = int(val_current_str)
    val_max = int(val_max_str)
    if val_current > val_max:
        return None, None   # error when detecting numbers

    return val_current, val_max


def get_cooldowns(in_frame):
    """
    out: {skill_key: is_on_cooldown}    e.g: {'Q': False}

file_path = f'imgs\\detect\\cooldowns.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_cooldowns(frame))
    """
    def get_well_cooldown():
        # get part of image
        x_mini, y_mini, x_maxi, y_maxi = 182, 1038, 182+34, 1038+17
        img = in_frame[y_mini:y_maxi, x_mini:x_maxi]
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # template
        thr = 0.8
        template = template_plus_symbol
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= thr)
        # result
        if len(loc[0]) > 0:
            return False  # found plus sign = no cooldown
        else:
            return True

    threshold = 60  # min number of pixels == cooldown number is present

    out = {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}
    for i, key in enumerate(out.keys()):
        x_min, y_min, x_max, y_max = 762 + 85 * i, 1019, 762 + 85 * i + 57, 1019 + 24
        hsv = cv.cvtColor(in_frame[y_min:y_max, x_min:x_max], cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 0, 200), (1, 1, 255))
        num_pixels = len(np.where(mask >= 200)[0])
        if num_pixels >= threshold:
            out[key] = True  # is on cooldown

    out['well'] = get_well_cooldown()

    return out


def get_total_xp_value(in_frame):
    """
    out: xp_value
    # Note: to read value, need to move mouse to position (x=960, y=44) and wait 1.5 sec

file_path = f'imgs\\detect\\xp_total.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_total_xp_value(frame))
    """

    # get part of image
    x_min, y_min, x_max, y_max = 875, 243, 875+70, 243+18
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of numbers
    out = []  # (x,y,number)
    threshold = 0.8
    for idx, template in enumerate([template_H10_num_0, template_H10_num_1, template_H10_num_2,
                                    template_H10_num_3, template_H10_num_4, template_H10_num_5,
                                    template_H10_num_6, template_H10_num_7, template_H10_num_8,
                                    template_H10_num_9]):
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            out.append((pt[0], pt[1], idx))
    #   sort numbers (by x position)
    out.sort(key=lambda x: x[0])

    # create final value
    return int(''.join([str(num) for (x, y, num) in out]))


def is_bot_hidden(in_frame):
    """
    out: True or False

file_path = f'imgs\\detect\\hidden.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(is_bot_hidden(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 930, 250, 930+60, 250+90
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.5
    template = template_hidden_symbol
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # final output
    if len(loc[0]) == 0:
        return False
    else:
        return True


def is_bot_dead(in_frame):
    """
    out: True or False

file_path = f'imgs\\detect\\death.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(is_bot_dead(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 945, 696, 945+30, 696+30
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.8
    template = template_death_icon
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # final output
    if len(loc[0]) == 0:
        return False
    else:
        return True


def get_bot_icon_position(in_frame):
    """
    out: (x, y) or (None, None)

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_bot_icon(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 1558, 863, 1558+366, 863+183
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.62
    template = template_bot_icon
    x_shift, y_shit = template.shape[1]//2, template.shape[0]//2
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None, None   # could not find icon's position
    else:
        x_out, y_out = x_min + loc[::-1][0][0] + x_shift, y_min + loc[::-1][1][0] + y_shit
        return x_out, y_out


def is_bot_icon_in_place(in_frame, in_place):
    """
    out: True or False

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(verify_bot_icon_place(frame, 'gate'))
    """
    x, y = get_bot_icon_position(in_frame)
    if None in [x, y]:
        return False    # could not find icon's position

    if in_place == 'gate':
        x_min, y_min, x_max, y_max = 1651, 899, 1651+6, 899+17
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    if in_place == 'bush':
        x_min, y_min, x_max, y_max = 1712, 873, 1712+23, 873+3
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False


def get_well_position(in_frame):
    """
    out: (x, y) or (None, None)
    # Note: to get value, bot need to be at place: 'gate'

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_well_position(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 550, 250, 550+350, 250+300
    img = in_frame[y_min:y_max, x_min:x_max]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # get mask from color
    thr = [np.array([60, 60, 200]), np.array([180, 255, 255])]
    mask = cv.inRange(hsv, thr[0], thr[1])
    kernel = np.ones((6, 6), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    # get circle
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=300, param2=10, minRadius=12, maxRadius=25)  # param2 - minimal number of edges
    if circles is not None and circles.shape == (1, 1, 3):
        circle_x, circle_y, circle_r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
        out_x, out_y = x_min + circle_x, y_min + circle_y
        return out_x, out_y
    else:
        return None, None   # could not find circle

    # old code - calculate fill level
    """
    # is circle shape filled with pixels?
    threshold = 0.9     # 90%
    circle_area = math.pi * math.pow(circle_r, 2)
    shape = mask[circle_y - circle_r:circle_y + circle_r, circle_x - circle_r:circle_x + circle_r]
    shape_fill_val = len(np.where(shape >= 255)[0])
    if shape_fill_val >= threshold * circle_area:
        out_x, out_y = x_min + circle_x, y_min + circle_y
        return out_x, out_y
    else:
        out_x, out_y = x_min + circle_x, y_min + circle_y
        print(f'{cnt} ERR - not filled enough! {shape_fill_val} < {threshold * circle_area}')
        cv.circle(mask, (circle_x, circle_y), circle_r, (0, 0, 255), 1)
        cv.imwrite(f'imgs\\{cnt}_mask.jpg', mask)
        cv.circle(in_frame, (out_x, out_y), circle_r, (0, 0, 255), 1)
        cv.rectangle(in_frame, (out_x - 1, out_y - 1), (out_x + 1, out_y + 1), color=(0, 0, 255), thickness=1)
        cv.imwrite(f'imgs\\{cnt}_img.jpg', in_frame)
        cnt += 1
        return None, None   # not filled enough
    """

    # old code - use template
    """
    # get part of image
    x_min, y_min, x_max, y_max = 725, 297, 725+136, 297+242
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.45
    template = template_well
    x_shift, y_shit = template.shape[1]//2, template.shape[0]//2
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None, None   # could not find icon's position
    else:
        x_out, y_out = x_min + loc[::-1][0][0] + x_shift, y_min + loc[::-1][1][0] + y_shit
        return x_out, y_out
    """


class Actions(dict):
    def __init__(self):
        args = tuple([{
            'use_well': ActionUseWell(),
            'hide_in_bushes': ActionHideInBushes(),
            'hide_behind_gate': ActionHideBehindGate(),
            'escape_behind_gate': ActionEscapeBehindGate(),
            'use_spell_d': ActionUseSpellD()
        }])
        super().__init__(*args, **{})

        self.current_action = None

    def start(self, in_action_name):
        if self.current_action is None:
            self.current_action = in_action_name
            self[self.current_action].start()

    def process(self, in_frame):
        if self.current_action is not None:
            self[self.current_action].process(in_frame)
            if self[self.current_action].result in [-1, 1]:
                # reset action
                self.printout()
                self[self.current_action].__init__()
                self.current_action = None

    def printout(self):
        if self.current_action is not None:
            self[self.current_action].printout()


class Action:
    def __init__(self):
        self.result = None
        # None - not started
        # 0    - in progress
        # 1    - finished successfully
        # -1   - finished unsuccessfully
        self.description = ''
        self.steps = []
        self.step_idx = None

    def start(self):
        self.result = 0

    def stop(self, in_result=-1, in_description='stopped'):
        pyautogui.press('s')
        self.set_result(in_result, in_description)

    def set_result(self, in_result, in_description):
        self.result, self.description = in_result, in_description

    def printout(self):
        if len(self.steps) > 0 and self.step_idx is not None:
            text = f'Action={self.__class__.__name__.replace("Action","")}, ' \
                   f'Step={self.steps[self.step_idx]}, ' \
                   f'Result={self.result}'
            if len(self.description) > 0:
                text += f', Description={self.description}'
            print(text)


class ActionUseWell(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_gate',
            'at_gate',
            'find_well',
            'click_well',
            'well_cooldown'
        ]
        self.well_x, self.well_y = None, None
        self.t0 = None
        self.timeout = 30    # [sec]
        self.timeout_short = 5    # [sec]

    def start(self):
        super().start()
        self.step_idx = 0

    def process(self, in_frame):
        if self.result != [-1, 1]:
            if self.steps[self.step_idx] == 'click_gate':
                pyautogui.moveTo(1654, 912)
                pyautogui.click(button='right')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'at_gate':
                if is_bot_icon_in_place(in_frame, 'gate'):
                    self.step_idx += 1
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Bot did not reach gate within {self.timeout} sec.')
                        self.t0 = None

            if self.steps[self.step_idx] == 'find_well':
                self.well_x, self.well_y = get_well_position(in_frame)
                if None not in [self.well_x, self.well_y]:
                    self.step_idx += 1
                else:
                    self.stop(-1, 'Could not find well')
                    return

            if self.steps[self.step_idx] == 'click_well':
                pyautogui.moveTo(self.well_x, self.well_y)
                pyautogui.click(button='right')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'well_cooldown':
                cooldowns = get_cooldowns(in_frame)
                if cooldowns['well']:
                    self.set_result(1, '')
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout_short:
                        self.set_result(-1, f'Could not detect well cooldown for {self.timeout_short} sec.')
                        self.t0 = None


class ActionHideInBushes(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_bushes',
            'at_bushes',
            'hidden'
        ]
        self.hidden_values = [None]*10
        self.t0 = None
        self.timeout = 30    # [sec]

    def start(self):
        super().start()
        self.step_idx = 0

    def process(self, in_frame):
        if self.result != [-1, 1]:
            if self.steps[self.step_idx] == 'click_bushes':
                pyautogui.moveTo(1722, 879)
                pyautogui.click(button='right')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'at_bushes':
                if is_bot_icon_in_place(in_frame, 'bush'):
                    self.step_idx += 1
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Bot did not reach bushes within {self.timeout} sec.')
                        self.t0 = None

            if self.steps[self.step_idx] == 'hidden':
                try:
                    idx = self.hidden_values.index(None)
                    val = is_bot_hidden(in_frame)
                    if val:
                        self.set_result(1, '')  # bot is hidden
                    self.hidden_values[idx] = val
                except ValueError:
                    self.set_result(-1, "Bot is not hidden in bushes")


class ActionHideBehindGate(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_gate',
            'at_gate'
        ]
        self.diff_values = [None]*10
        self.t0 = None
        self.timeout = 30    # [sec]

    def start(self):
        super().start()
        self.step_idx = 0

    def process(self, in_frame):
        if self.result != [-1, 1]:
            if self.steps[self.step_idx] == 'click_gate':
                pyautogui.moveTo(1654, 912)
                pyautogui.click(button='right')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'at_gate':
                if is_bot_icon_in_place(in_frame, 'gate'):
                    self.set_result(1, '')  # finished
                    self.t0 = None
                else:
                    bot_x, bot_y = get_bot_icon_position(in_frame)
                    if None not in [bot_x, bot_y]:
                        try:
                            idx = self.diff_values.index(None)
                            val = abs(1654-bot_x) + abs(912-bot_y)
                            self.diff_values[idx] = val
                        except ValueError:
                            # all values are gathered
                            if self.diff_values[-1] >= self.diff_values[0]:
                                self.set_result(-1, f'Bot is not moving towards gate since {len(self.diff_values)} frames')
                                self.t0 = None
                            self.diff_values = [None]*len(self.diff_values)    # reset
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Bot did not reach gate within {self.timeout} sec.')
                        self.t0 = None


class ActionUseSpellD(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'press_button',
            'spell_blocked',
            'spell_cooldown'
        ]
        self.t0 = None
        self.timeout = 5    # [sec]

    def start(self):
        super().start()
        self.step_idx = 0

    def process(self, in_frame):
        if self.result != [-1, 1]:
            if self.steps[self.step_idx] == 'press_button':
                pyautogui.moveTo(SCREEN_W//2, SCREEN_H//2)
                pyautogui.click(button='left')
                pyautogui.press('d')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'spell_blocked':
                # get part of image
                x_min, y_min, x_max, y_max = 1006, 1010, 1006 + 144, 1010 + 40
                img = in_frame[y_min:y_max, x_min:x_max]
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # get positions of template
                threshold = 0.8
                template = template_blocked_symbol
                x_shift, y_shit = template.shape[1] // 2, template.shape[0] // 2
                res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                if len(loc[0]) > 0:
                    self.step_idx += 1  # found symbol
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Could not detect spell_blocked symbol for {self.timeout} sec.')
                        self.t0 = None

            if self.steps[self.step_idx] == 'spell_cooldown':
                cooldowns = get_cooldowns(in_frame)
                if cooldowns['D']:
                    self.set_result(1, '')
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Could not detect spell cooldown for {self.timeout} sec.')
                        self.t0 = None


class ActionEscapeBehindGate(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'press_button',
            'wait',
            'spell_cooldown',
            'click_gate',
            'at_gate'
        ]
        self.diff_values = [None]*10
        self.t0 = None
        self.timeout = 30    # [sec]
        self.timeout_short = 5    # [sec]

    def start(self):
        super().start()
        self.step_idx = 0

    def process(self, in_frame):
        if self.result != [-1, 1]:
            if self.steps[self.step_idx] == 'press_button':
                # get mouse position between bot and gate
                bot_icon_pos = get_bot_icon_position(in_frame)
                if None in bot_icon_pos:
                    self.set_result(-1, f'Could not detect bot icon position')
                    return
                in_hsv = cv.cvtColor(in_frame, cv.COLOR_BGR2HSV)
                positions, description = get_bot_positions(in_hsv)
                if positions is None:
                    self.set_result(-1, description)
                    return
                circle_pos = positions['circle'][1]
                gate_icon_pos = (1654, 912)
                vector = (gate_icon_pos[0]-bot_icon_pos[0], gate_icon_pos[1]-bot_icon_pos[1])
                mouse_pos = (circle_pos[0]+vector[0], circle_pos[1]+vector[1])
                # button
                pyautogui.moveTo(mouse_pos[0], mouse_pos[1])
                pyautogui.click(button='left')
                pyautogui.press('e')
                pyautogui.click(button='left')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'wait':
                if self.t0 is None:
                    self.t0 = time.time()
                if time.time() - self.t0 >= 1.0:  # sec
                    self.step_idx += 1
                    self.t0 = None

            if self.steps[self.step_idx] == 'spell_cooldown':
                cooldowns = get_cooldowns(in_frame)
                if cooldowns['E']:
                    self.step_idx += 1
                    self.t0 = None
                else:
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout_short:
                        self.set_result(-1, f'Could not detect spell cooldown for {self.timeout_short} sec.')
                        self.t0 = None

            if self.steps[self.step_idx] == 'click_gate':
                pyautogui.moveTo(1654, 912)
                pyautogui.click(button='right')
                self.step_idx += 1

            if self.steps[self.step_idx] == 'at_gate':
                if is_bot_icon_in_place(in_frame, 'gate'):
                    self.set_result(1, '')  # finished
                    self.t0 = None
                else:
                    bot_x, bot_y = get_bot_icon_position(in_frame)
                    if None not in [bot_x, bot_y]:
                        try:
                            idx = self.diff_values.index(None)
                            val = abs(1654-bot_x) + abs(912-bot_y)
                            self.diff_values[idx] = val
                        except ValueError:
                            # all values are gathered
                            if self.diff_values[-1] >= self.diff_values[0]:
                                self.set_result(-1, f'Bot is not moving towards gate since {len(self.diff_values)} frames')
                                self.t0 = None
                            self.diff_values = [None]*len(self.diff_values)    # reset
                    # timeout?
                    if self.t0 is None:
                        self.t0 = time.time()
                    if time.time() - self.t0 >= self.timeout:
                        self.set_result(-1, f'Bot did not reach gate within {self.timeout} sec.')
                        self.t0 = None




def action_use_well():
    # if bot's health < max and well is not on cooldown:
    #   - 'move to gate'
    #   - find well's position
    #   - right-click well
    #   - check if health++
    pass


action = None
clicked = None
def process_action(in_frame, in_action):
    global action, clicked

    if action is None:
        action = in_action

    finished = False
    # move to gate
    if action == 'move to gate':
        if clicked is None:
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            clicked = True
        if is_bot_icon_in_place(in_frame, 'gate'):
            finished = True  # finished action

    # move to bush
    if action == 'move to bush':
        if clicked is None:
            pyautogui.moveTo(1722, 879)
            pyautogui.click(button='right')
            clicked = True
        if is_bot_icon_in_place(in_frame, 'bush'):
            finished = True  # finished action


    if finished:
        action = None
        clicked = None

    return action, clicked






#import cv2 as cv
#from image_objects_lib import ImageObjects
#from tracker_lib import TrackerClass
#from painter_lib import PainterClass

#ImgObjs = ImageObjects()
#Tracker = TrackerClass()
#Painter = PainterClass()

"""
class MinionClass:
    def __init__(self, point_center):
        self.img = None
        self.color = None
        self.health_bar = None
        self.health = None
        self.bounding_box = None
        self.point_center = point_center

mns = MinionsClass(None, None, 'blue')
mns.objects = []
mns.health_bars = None
inputs = [
    [MinionClass((821, 315)), MinionClass((884, 314)), MinionClass((961, 305)), MinionClass((1002, 290)), MinionClass((1125, 289)), MinionClass((1064, 289)), MinionClass((925, 281))],
    [MinionClass((835, 311)), MinionClass((898, 313)), MinionClass((975, 304)), MinionClass((1017, 289)), MinionClass((1139, 288)), MinionClass((1078, 288)), MinionClass((939, 280))],
    [MinionClass((841, 310)), MinionClass((905, 313)), MinionClass((982, 303)), MinionClass((1024, 288)), MinionClass((1146, 287)), MinionClass((946, 280))]
]
cnt = 0
"""




"""
def detect_objects(img):
    ImgObjs.img = img
    ImgObjs.hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for obj in ImgObjs.detected_objects:
        obj.get_from_image(ImgObjs.img, ImgObjs.hsv)

    ImgObjs.minimap.get_from_image(ImgObjs.img, ImgObjs.hsv, ImgObjs.bot, ImgObjs.minions_blue, ImgObjs.minions_red)


def track_objects():
    Tracker.update()


def paint_objects():
    img = Painter.update()
    return img



def save_image(img, file_path):
    cv.imwrite(file_path, img)
"""