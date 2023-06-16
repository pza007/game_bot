import pyautogui
from global_vars import *
import time
import math

from itertools import product, combinations


def get_distance_between_points(point1, point2):
    return float(np.linalg.norm(np.array((point1[0], point1[1])) - np.array((point2[0], point2[1]))))


def get_centroid_point(list_of_points: list) -> tuple:
    # link: https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi
    arr = np.array(list_of_points)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return int(sum_x / length), int(sum_y / length)


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_inside_ellipse(point, point_center, ellipse_type):
    ELLIPSE_Q_WITH = 222
    ELLIPSE_Q_HEIGHT = 180
    ELLIPSE_W_WITH = 385
    ELLIPSE_W_HEIGHT = 305

    if ellipse_type == 'Q':
        a = ELLIPSE_Q_WITH // 2
        b = ELLIPSE_Q_HEIGHT // 2
    else:  # 'W'
        a = ELLIPSE_W_WITH // 2
        b = ELLIPSE_W_HEIGHT // 2
    phi = math.atan2(a * point[1], b * point[0])
    point_on_ellipse = (point_center[0] + a * math.cos(phi), point_center[1] + b * math.sin(phi))
    dist_to_point = get_distance_between_points(point_center, point)
    dist_to_point_on_ellipse = get_distance_between_points(point_center, point_on_ellipse)
    if dist_to_point <= dist_to_point_on_ellipse:
        return True
    else:
        return False


def get_minions_positions(in_img, in_hsv):
    """
    out: [dict] {color: [(x_center, y_center), ...] }

window = WindowCapture('Heroes of the Storm')
img = window.get_screenshot()
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
print(get_minions_positions(img, hsv))
    """

    def get_health_bars(in_color):
        HB_COLOR_THR_RED = [np.array([150, 170, 80]), np.array([190, 255, 230])]
        HB_COLOR_THR_BLUE = [np.array([90, 120, 90]), np.array([120, 220, 255])]
        HB_BOX_H = 4
        HB_BOX_MIN_W = 1
        HB_BOX_MAX_W = 51
        f_debug = False

        def get_mask():
            # set color thresholds
            if in_color == 'red':
                thr = HB_COLOR_THR_RED
            else:  # 'blue'
                thr = HB_COLOR_THR_BLUE
            # create mask, based on thresholds
            return cv.inRange(in_hsv, thr[0], thr[1])

        def transform_mask(in_mask):
            kernel = np.ones((3, 3), np.uint8)
            result = cv.morphologyEx(in_mask, cv.MORPH_OPEN, kernel)
            return result

        def get_contours(in_mask):
            return cv.findContours(in_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        def filter_1(in_contours):
            # filter contours with 4 points (rectangle)
            out_contours = []
            for contour in in_contours:
                if len(contour) == 4:
                    x_min = min([_contour[0][0] for _contour in contour])  # point X
                    x_max = max([_contour[0][0] for _contour in contour])  # point X
                    y_min = min([_contour[0][1] for _contour in contour])  # point Y
                    y_max = max([_contour[0][1] for _contour in contour])  # point Y
                    out_contours.append((x_min, y_min, x_max, y_max))
            return out_contours

        def filter_2(in_contours):
            # filter contours with correct dimensions
            out_contours = []
            for (x_min, y_min, x_max, y_max) in in_contours:
                height = y_max - y_min + 1
                width = x_max - x_min + 1
                if height == HB_BOX_H and HB_BOX_MIN_W <= width <= HB_BOX_MAX_W:
                    out_contours.append((x_min, y_min, x_max, y_max))
            return out_contours

        def filter_3(in_contours):
            # filter contours located outside skip areas
            out_contours = []
            for (x_min, y_min, x_max, y_max) in in_contours:
                success = True
                for box in SKIP_AREAS:
                    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                    polygon = [box[0], (box[1][0], box[0][1]), box[1], (box[0][0], box[1][1])]
                    for point in points:
                        if point_inside_polygon(point[0], point[1], polygon):
                            success = False
                            break
                    if not success:
                        break
                if success:
                    out_contours.append((x_min, y_min, x_max, y_max))
            return out_contours

        def filter_4(in_contours):
            # filter contours with black border outside
            out_contours = []
            for (x_min, y_min, x_max, y_max) in in_contours:
                # min,max points of border inside window?
                b_x_min, b_y_min = x_min - 1, y_min - 1
                b_x_max, b_y_max = x_max + HB_BOX_MAX_W - (x_max - x_min) + 1, y_max + 1
                if 0 <= b_x_min <= SCREEN_W - 1 and 0 <= b_y_min <= SCREEN_H - 1 and \
                        0 <= b_x_max <= SCREEN_W - 1 and 0 <= b_y_max <= SCREEN_H - 1:

                    # border color thresholds
                    b_thr, g_thr, r_thr = 0, 0, 0  # B,G,R
                    if in_color == 'red':
                        b_thr, g_thr, r_thr = 10, 10, 90
                    else:  # blue
                        b_thr, g_thr, r_thr = 90, 10, 10

                    # borders
                    cnt_top, cnt_bottom = 0, 0
                    for x in range(b_x_min, b_x_max + 1):
                        # top border
                        b, g, r, *a = in_img[b_y_min][x]  # B,G,R,A
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_top += 1
                        # bottom border
                        b, g, r, *a = in_img[b_y_max][x]
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_bottom += 1
                    cnt_left, cnt_right = 0, 0
                    for y in range(b_y_min, b_y_max + 1):
                        # left border
                        b, g, r, *a = in_img[y][b_x_min]
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_left += 1
                        # right border
                        b, g, r, *a = in_img[y][b_x_max]
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_right += 1

                    # final check
                    # print(cnt_top, cnt_bottom, cnt_left, cnt_right)
                    # color: _HB_BOX_MAX_W + 1 while pixel + 2 black pixels for border
                    if cnt_top == HB_BOX_MAX_W + 3 and cnt_bottom == HB_BOX_MAX_W + 3:
                        # color: _HB_BOX_H + 2 black pixels for border
                        if cnt_left == HB_BOX_H + 2:
                            out_contours.append((x_min, y_min, x_max, y_max))
                            # print('\t added')

            return out_contours

        if f_debug:
            import copy
            cv.imwrite('imgs\\_output1.jpg', in_img)
            mask = get_mask()
            cv.imwrite('imgs\\_output2.jpg', mask)
            mask = transform_mask(mask)
            cv.imwrite('imgs\\_output3.jpg', mask)
            contours, _ = get_contours(mask)
            img = copy.deepcopy(in_img)
            for contour in contours:
                x_min = min([_contour[0][0] for _contour in contour])  # point X
                x_max = max([_contour[0][0] for _contour in contour])  # point X
                y_min = min([_contour[0][1] for _contour in contour])  # point Y
                y_max = max([_contour[0][1] for _contour in contour])  # point Y
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output4.jpg', img)
            contours = filter_1(contours)
            img = copy.deepcopy(in_img)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output5.jpg', img)
            contours = filter_2(contours)
            img = copy.deepcopy(in_img)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output6.jpg', img)
            contours = filter_3(contours)
            img = copy.deepcopy(in_img)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output7.jpg', img)
            contours = filter_4(contours)
            img = copy.deepcopy(in_img)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output8.jpg', img)

        mask = get_mask()
        mask = transform_mask(mask)
        contours, _ = get_contours(mask)
        contours = filter_1(contours)
        contours = filter_2(contours)
        contours = filter_3(contours)
        contours = filter_4(contours)
        contours = [(int(x_min), int(y_min), int(x_max), int(y_max)) for (x_min, y_min, x_max, y_max) in contours]

        return contours

    def get_points(in_dict):
        HB_BOX_MAX_W = 51
        MN_BOX_W = 65
        MN_BOX_H = 73
        f_debug = False

        out = {}  # {color: [(x,y), ...] }
        for color, list_of_positions in in_dict.items():
            new_list = []
            for positions in list_of_positions:  # (x_min, y_min, x_max, y_max)

                x_hb_min = positions[0]
                y_hb_min = positions[1]
                x_hb_max = positions[2]
                y_hb_max = positions[3]
                x_hb_middle = x_hb_min + int(HB_BOX_MAX_W // 2)

                x_min = x_hb_middle - MN_BOX_W
                x_max = x_hb_middle + MN_BOX_W
                y_min = y_hb_max
                y_max = y_min + MN_BOX_H + 30

                x_new = x_min + (x_max - x_min) // 2
                y_new = y_min + (y_max - y_min) // 2
                new_list.append((int(x_new), int(y_new)))  # (x,y) of center

            out[color] = new_list

        if f_debug:
            import copy
            img = copy.deepcopy(in_img)
            for color, list_of_points in out.items():
                for (x, y) in list_of_points:
                    if color == 'red':
                        cv.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), 2)
                    else:  # blue
                        cv.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (255, 0, 0), 2)
                cv.imwrite(f'imgs\\_output_{color}.jpg', img)

        return out

    out = {}  # {color: [(x_min, y_min, x_max, y_max), ...] }
    for color in ['red', 'blue']:
        out[color] = get_health_bars(color)
    out = get_points(out)  # {color: [(x_center, y_center), ...] }

    return out


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
        'health_bar': ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        'bounding_box': ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        'circle': ((None, None, None, None), (None, None)),  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
    }
    _HEALTH_BAR_COLOR_THR = [np.array([50, 100, 150]), np.array([55, 255, 255])]
    _HEALTH_BAR_BOX = [8, 11, 2, 11]  # height_min, height_max, width_min, width_max
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
    _BOT_CIRCLE_W = 43 * 2
    _BOT_CIRCLE_H = 36 * 2

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
                out_contours.append([(int(x_min), int(y_min)), (int(x_max), int(y_max))])

            return out_contours

        def filter_by_dimensions(in_contours):
            out_contours = []
            for [(x_min, y_min), (x_max, y_max)] in in_contours:
                height = y_max - y_min + 1
                width = x_max - x_min + 1
                # print(f'height={height}, width={width}')
                if _HEALTH_BAR_BOX[0] <= height <= _HEALTH_BAR_BOX[1] and _HEALTH_BAR_BOX[2] <= width <= \
                        _HEALTH_BAR_BOX[3]:
                    out_contours.append([(x_min, y_min), (x_max, y_max)])

            return out_contours

        def set_positions(in_contours):
            # health bar
            y_min = min([y_min for [(x_min, y_min), (x_max, y_max)] in in_contours])
            pos = (shift_x_min, shift_y_min + y_min, shift_x_max, shift_y_max + y_min)
            center = (pos[0] + (pos[2] - pos[0]) // 2, pos[1] + (pos[3] - pos[1]) // 2)
            out['health_bar'] = (pos, center)

        # ON GROUND?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 346, 905 + 121, 346 + 9
        if execute(): return

        # ON HORSE?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 261, 905 + 121, 261 + 9
        if execute(): return

        # IN BUSHES?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 150, 905 + 121, 450
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
        center = (x_min + (pos[2] - pos[0]) // 2, y_min + (pos[3] - pos[1]) // 2)
        out['bounding_box'] = (pos, center)

    def get_circle():
        y_hb_min = out['health_bar'][0][1]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))
        x_center, y_center = out['bounding_box'][1]  # ((x_min, y_min, x_max, y_max), (x_center,y_center))

        # bot mounted
        if y_hb_min < int(2 * SCREEN_H / 8):
            center = (x_center, y_center + _BOT_CIRCLE_SHIFT_MOUNT)
            x_min = center[0] - _BOT_CIRCLE_W // 2
            y_min = center[1] - _BOT_CIRCLE_H // 2
            x_max = center[0] + _BOT_CIRCLE_W // 2
            y_max = center[1] + _BOT_CIRCLE_H // 2
        else:
            # bot unmounted = on ground
            center = (x_center, y_center + _BOT_CIRCLE_SHIFT_UNMOUNT)
            x_min = center[0] - _BOT_CIRCLE_W // 2
            y_min = center[1] - _BOT_CIRCLE_H // 2
            x_max = center[0] + _BOT_CIRCLE_W // 2
            y_max = center[1] + _BOT_CIRCLE_H // 2

        pos = (x_min, y_min, x_max, y_max)
        center = (x_min + (pos[2] - pos[0]) // 2, y_min + (pos[3] - pos[1]) // 2)
        out['circle'] = (pos, center)

    description = get_health_bar()
    if description is not None:
        return None, description
    get_bounding_box()
    get_circle()

    return out, ''


def get_bot_health_value(in_frame):
    """
    out: (val_current, val_max) or (None, None)

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
        return None, None  # could not find slash position
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
    val_current_str = ''.join([str(num) for (x, y, num) in out if x < x_slash])  # numbers before slash
    val_max_str = ''.join([str(num) for (x, y, num) in out if x > x_slash])  # numbers after slash
    if not val_current_str.isdigit() or not val_max_str.isdigit():
        return None, None  # could not detect numbers
    val_current = int(val_current_str)
    val_max = int(val_max_str)
    if val_current > val_max:
        return None, None  # error when detecting numbers

    return val_current, val_max


def get_bot_mana_value(in_frame):
    """
    out: (val_current, val_max) or (None, None)

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
        return None, None  # could not find slash position
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
    val_current_str = ''.join([str(num) for (x, y, num) in out if x < x_slash])  # numbers before slash
    val_max_str = ''.join([str(num) for (x, y, num) in out if x > x_slash])  # numbers after slash
    if not val_current_str.isdigit() or not val_max_str.isdigit():
        return None, None  # could not detect numbers
    val_current = int(val_current_str)
    val_max = int(val_max_str)
    if val_current > val_max:
        return None, None  # error when detecting numbers

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
        x_mini, y_mini, x_maxi, y_maxi = 182, 1038, 182 + 34, 1038 + 17
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
    x_min, y_min, x_max, y_max = 875, 243, 875 + 70, 243 + 18
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
    x_min, y_min, x_max, y_max = 930, 250, 930 + 60, 250 + 90
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.45
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
    x_min, y_min, x_max, y_max = 945, 696, 945 + 30, 696 + 30
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
print(get_bot_icon_position(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 1558, 863, 1558 + 366, 863 + 183
    img = in_frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.62
    template = template_bot_icon
    x_shift, y_shit = template.shape[1] // 2, template.shape[0] // 2 + 6
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None, None  # could not find icon's position
    else:
        x_out, y_out = x_min + loc[::-1][0][0] + x_shift, y_min + loc[::-1][1][0] + y_shit
        return int(x_out), int(y_out)


def is_bot_icon_in_place(position: tuple, in_place: str) -> bool:
    """
    out: True or False
    """
    x, y = position
    if type(x) is not int or type(y) is not int:
        raise Exception(f'Invalid input type: x={x}, y={y}')

    if in_place == 'gate':
        x_min, y_min, x_max, y_max = 1651, 899, 1651 + 6, 899 + 17
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    elif in_place == 'bush':
        x_min, y_min, x_max, y_max = 1712, 880, 1712 + 23, 880 + 4
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    elif in_place == 'middle':
        x_min, y_min, x_max, y_max = 1723-5, 896-5, 1723+5, 896+5
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    else:
        raise Exception(f'Invalid input: in_place={in_place}')


def get_well_position(in_frame):
    """
    out: (x, y) or (None, None)
    # Note: to get value, bot need to be at place: 'gate'

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_well_position(frame))
    """
    # get part of image
    x_min, y_min, x_max, y_max = 550, 250, 550 + 350, 250 + 300
    img = in_frame[y_min:y_max, x_min:x_max]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # get mask from color
    thr = [np.array([60, 60, 200]), np.array([180, 255, 255])]
    mask = cv.inRange(hsv, thr[0], thr[1])
    kernel = np.ones((6, 6), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    # get circle
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=300, param2=10, minRadius=12,
                              maxRadius=25)  # param2 - minimal number of edges
    if circles is not None and circles.shape == (1, 1, 3):
        circle_x, circle_y, circle_r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
        out_x, out_y = x_min + circle_x, y_min + circle_y
        return int(out_x), int(out_y)
    else:
        return None, None  # could not find circle

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


def get_center_point_for_spell(minions: dict, bot_pos: dict, spell_type: str, **kwargs) -> tuple:
    """
    out: (x, y)

    - get points that are close to each other
    - calculate all centroids for 2,3,4,5... points
    - choose center point of centroid that ellipse includes the most of points

file_path = f'imgs\\attack\\attack_q.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
minions = get_minions_positions(frame, frame_hsv)
bot, desc = get_bot_positions(frame_hsv)
print(get_center_point_for_spell(minions, bot, 'Q', frame=frame))
    """
    def get_points(in_points):
        """ get the group of points with most points that are close to each other (max_dist == ellipse width//2) """
        if spell_type == 'Q':
            ELLIPSE_WITH = 222
        else:   # spell_type == 'W'
            ELLIPSE_WITH = 385

        tups = [sorted(sub) for sub in product(in_points, repeat=2) if
                get_distance_between_points(*sub) <= ELLIPSE_WITH // 2]

        res_dict = {point: {point} for point in in_points}
        out_points = []
        for tup1, tup2 in tups:
            res_dict[tup1] |= res_dict[tup2]
            res_dict[tup2] = res_dict[tup1]
            for val in [tup1, tup2]:
                if len(res_dict[val]) > len(out_points):
                    out_points = res_dict[val]

        return list(out_points)

    f_debug = False

    # get points
    points = get_points(minions['red'])
    #   1 point -> return point that is closest to bot
    if len(points) == 1:
        return min((get_distance_between_points(bot_pos, pos), pos) for pos in minions['red'])[1]
    #   2 points -> return point between them
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return int(x1 + (x1 - x2)), int(y1 + (y1 - y2))

    # get centroids
    centroids = set()
    for num_tuples in range(2, len(points) + 1):
        for obj in combinations(points, num_tuples):
            centroids.add(get_centroid_point(list(obj)))

    # get how many points are inside ellipse which center is defined by centroid
    centroids_ellipse = []
    max_val = -1
    for pos_centroid in centroids:
        val = sum(1 for pos_point in points if point_inside_ellipse(pos_point, pos_centroid, spell_type))
        if val > max_val:
            centroids_ellipse.append((pos_centroid, val))
            max_val = val

    # choose centroid that includes the most of the points
    centroids_ellipse.sort(key=lambda x: x[1])
    point_center = centroids_ellipse[-1][0]

    if f_debug:
        print('number of points inside=', centroids_ellipse[-1][1])
        frame = kwargs.get('frame')
        for (x, y) in points:
            cv.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), 2)
        cv.rectangle(frame, (point_center[0] - 1, point_center[1] - 1), (point_center[0] + 1, point_center[1] + 1), (255, 0, 0), 2)
        if spell_type == 'Q':
            ELLIPSE_WITH = 222
            ELLIPSE_HEIGHT = 180
        else:   # spell_type == 'W'
            ELLIPSE_WITH = 385
            ELLIPSE_HEIGHT = 305
        cv.ellipse(frame, point_center, (ELLIPSE_WITH // 2, ELLIPSE_HEIGHT // 2), 0, 0, 360, (255, 0, 0), 1)
        cv.imwrite('imgs\\_output.jpg', frame)

    return int(point_center[0]), int(point_center[1])


def get_move_position(position, in_direction):
    """
    out: (x, y) or (None, None)

file_path = f'imgs\\detect\\gate.png'
frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
print(get_move_position(frame, 'up'))
    """
    dx = 8
    dy = 8
    dxy = 5
    x_shift = SKIP_AREA_BOTTOM_RIGHT[0][0]
    y_shift = SKIP_AREA_BOTTOM_RIGHT[0][1]
    x, y = position
    if type(x) is not int or type(y) is not int:
        raise Exception(f'Invalid input type: x={x}, y={y}')

    if in_direction == 'up':
        new_x, new_y = x, y - dy
    elif in_direction == 'down':
        new_x, new_y = x, y + dy
    elif in_direction == 'right':
        new_x, new_y = x + dx, y
    elif in_direction == 'left':
        new_x, new_y = x - dx, y
    elif in_direction == 'up-right':
        new_x, new_y = x + dxy, y - dxy
    elif in_direction == 'down-right':
        new_x, new_y = x + dxy, y + dxy
    elif in_direction == 'up-left':
        new_x, new_y = x - dxy, y - dxy
    else:  # 'down-left'
        new_x, new_y = x - dxy, y + dxy

    if 0 > new_x - x_shift or new_x - x_shift > template_minimap.shape[1]-1:    # x
        return None, None
    elif 0 > new_y - y_shift or new_y - y_shift > template_minimap.shape[0]-1:    # y
        return None, None
    elif template_minimap[new_y - y_shift, new_x - x_shift] != 255:
        return None, None
    else:
        return int(new_x), int(new_y)


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
        self.t0 = None
        self.TIMEOUT = 30   # sec

    def can_be_started(self, *args, **kwargs):
        raise NotImplemented

    def start(self):
        self.result = 0
        self.step_idx = 0
        self.t0 = time.time()

    def process(self, *args, **kwargs):
        raise NotImplemented

    def set_result(self, in_result, in_description):
        self.result, self.description = in_result, in_description


class Actions:
    def __init__(self):
        self.objects = {
            'move_up': ActionMove('up'),
            'move_down': ActionMove('down'),
            'move_right': ActionMove('right'),
            'move_left': ActionMove('left'),
            'move_up-right': ActionMove('up-right'),
            'move_down-right': ActionMove('down-right'),
            'move_up-left': ActionMove('up-left'),
            'move_down-left': ActionMove('down-left'),
            'run_middle': ActionRunMiddle(),
            'collect_globes': ActionCollectGlobes(),

            'basic_attack': ActionBasicAttack(),
            'q_attack': ActionQAttack(),
            'w_attack': ActionWAttack(),

            'use_well': ActionUseWell(),
            'hide_in_bushes': ActionHideInBushes(),
            'hide_behind_gate': ActionHideBehindGate(),
            'escape_behind_gate': ActionEscapeBehindGate(),
            'use_spell_d': ActionUseSpellD()
        }
        self.current_action = None

    def get_available_actions(self, *args, **kwargs):
        out = []
        for name, action in self.objects.items():
            if action.can_be_started(*args, **kwargs):
                out.append(name)
        return out

    def start(self, action_name, *args, **kwargs):
        if self.current_action is None:
            action = self.objects[action_name]

            # Reset all values
            if action_name.find('move_') >= 0:  # special for move actions
                direction = action_name[5:]
                action.__init__(direction)
            else:
                action.__init__()

            if action.can_be_started(*args, **kwargs):
                self.current_action = action
                self.current_action.start()
                return True
        return False

    def process(self, *args, **kwargs):
        if self.current_action is not None:
            # finished?
            if self.current_action.result in [-1, 1]:
                self.printout()
                self.current_action = None
            # timeout?
            elif time.time() - self.current_action.t0 >= self.current_action.TIMEOUT:
                self.current_action.set_result(-1, f'Reached timeout = {self.current_action.TIMEOUT} sec.')
                self.printout()
                self.current_action = None
            # process...
            else:
                self.current_action.process(*args, **kwargs)

    def printout(self):
        if self.current_action is not None:
            if len(self.current_action.steps) > 0 and self.current_action.step_idx is not None:
                text = f'Action={self.current_action.__class__.__name__.replace("Action", "")}, ' \
                       f'Step={self.current_action.steps[self.current_action.step_idx]}, ' \
                       f'Result={self.current_action.result}'
                if len(self.current_action.description) > 0:
                    text += f', Description={self.current_action.description}'
                print(text)


class ActionBasicAttack(Action):
    def __init__(self):
        super().__init__()
        self.MIN_DIST_TO_TARGET = 250
        self.steps = [
            'at_range',
            'click_attack'
        ]

    def can_be_started(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion

        try:
            int(kwargs.get('bot_pos_frame')['bounding_box'][1][0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('minions')['red'][0][0])   # x
        except Exception:
            return False

        return True

    def process(self, *args, **kwargs):
        bot_x, bot_y = kwargs.get('bot_pos_frame')['bounding_box'][1]    # (center x, center y)
        minions = kwargs.get('minions')['red']

        # get the position of the closest red minion == target
        distances = [(get_distance_between_points((bot_x, bot_y), minion_pos), minion_pos) for minion_pos in minions]
        distances.sort()
        distance, target = distances[0]  # float dist, (x, y)

        # evaluate if bot is in range to attack target position
        if self.steps[self.step_idx] == 'at_range':
            if distance > self.MIN_DIST_TO_TARGET:
                # bot needs to move closer to target position, to be in range
                new_pos = bot_x + (target[0] - bot_x) // 2, \
                          bot_y + (target[1] - bot_y) // 2
                pyautogui.moveTo(*new_pos)
                pyautogui.click(button='right')
            else:   # already close to target
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'click_attack':
            pyautogui.moveTo(target)
            pyautogui.press('a')
            self.set_result(1, '')  # finish


class ActionQAttack(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'choose_target',
            'click_attack',
            'spell_cooldown'
        ]
        self.point_attack = None

    def can_be_started(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion
        # - Q spell available

        try:
            int(kwargs.get('bot_pos_frame')['bounding_box'][1][0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('minions')['red'][0][0])   # x
        except Exception:
            return False

        if kwargs.get('cooldowns')['Q']:
            return False

        return True

    def process(self, *args, **kwargs):
        minions = kwargs.get('minions')
        bot_pos = kwargs.get('bot_pos_frame')['bounding_box'][1]    # (center_x, center_y)
        cooldowns = kwargs.get('cooldowns')

        if self.point_attack is None:
            self.point_attack = get_center_point_for_spell(minions, bot_pos, 'Q')

        if self.steps[self.step_idx] == 'choose_target':
            pyautogui.moveTo(self.point_attack)
            pyautogui.click(button='left')
            pyautogui.press('q')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'click_attack':
            pyautogui.click(button='left')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'spell_cooldown':
            if cooldowns['Q']:
                self.set_result(1, '')  # finish


class ActionWAttack(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'choose_target',
            'click_attack',
            'spell_cooldown'
        ]
        self.point_attack = None

    def can_be_started(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion
        # - W spell available

        try:
            int(kwargs.get('bot_pos_frame')['bounding_box'][1][0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('minions')['red'][0][0])   # x
        except Exception:
            return False

        if kwargs.get('cooldowns')['W']:
            return False

        return True

    def process(self, *args, **kwargs):
        minions = kwargs.get('minions')
        bot_pos = kwargs.get('bot_pos_frame')['bounding_box'][1]    # (center_x, center_y)
        cooldowns = kwargs.get('cooldowns')

        if self.point_attack is None:
            self.point_attack = get_center_point_for_spell(minions, bot_pos, 'W')

        if self.steps[self.step_idx] == 'choose_target':
            pyautogui.moveTo(self.point_attack)
            pyautogui.click(button='left')
            pyautogui.press('w')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'click_attack':
            pyautogui.click(button='left')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'spell_cooldown':
            if cooldowns['W']:
                self.set_result(1, '')  # finish


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

    def can_be_started(self, *args, **kwargs):
        # - frame is available
        # - bot_pos_minimap is available
        # - well not on cooldown
        # - bot's health < max

        frame = kwargs.get('frame')
        if type(frame) is not np.ndarray:
            return False

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        if kwargs.get('cooldowns')['well']:
            return False

        try:
            int(kwargs.get('bot_health')[0])    # current
        except Exception:
            return False
        if int(kwargs.get('bot_health')[0]) > 0.95*int(kwargs.get('bot_health')[1]):
            return False

        return True

    def process(self, *args, **kwargs):
        frame = kwargs.get('frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        cooldowns = kwargs.get('cooldowns')

        if self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            if is_bot_icon_in_place(bot_pos_minimap, 'gate'):
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'find_well':
            self.well_x, self.well_y = get_well_position(frame)
            if None not in [self.well_x, self.well_y]:
                self.step_idx += 1
            else:
                self.set_result(-1, 'Could not find well')

        elif self.steps[self.step_idx] == 'click_well':
            pyautogui.moveTo(self.well_x, self.well_y)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'well_cooldown':
            if cooldowns['well']:
                self.set_result(1, '')


class ActionHideInBushes(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_bushes',
            'at_bushes',
            'hidden'
        ]
        self.hidden_values = [None] * 10

    def can_be_started(self, *args, **kwargs):
        # - frame is available
        # - bot_pos_minimap is available
        # - bot not in bushes

        frame = kwargs.get('frame')
        if type(frame) is not np.ndarray:
            return False

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        if is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'bush'):
            return False

        return True

    def process(self, *args, **kwargs):
        frame = kwargs.get('frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_bushes':
            pyautogui.moveTo(1722, 879)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_bushes':
            if is_bot_icon_in_place(bot_pos_minimap, 'bush'):
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'hidden':
            try:
                idx = self.hidden_values.index(None)
                val = is_bot_hidden(frame)
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
        self.diff_values = [None] * 10

    def can_be_started(self, *args, **kwargs):
        # - bot pos icon is available
        # - bot not at gate

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        if is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'gate'):
            return False

        return True

    def process(self, *args, **kwargs):
        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            if is_bot_icon_in_place(bot_pos_minimap, 'gate'):
                self.set_result(1, '')  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1654 - bot_pos_minimap[0]) + abs(912 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        self.set_result(-1, f'Bot is not moving towards gate since {len(self.diff_values)} frames')


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
        self.diff_values = [None] * 10
        self.t1 = None
        self.timeout1 = 5  # [sec]

    def can_be_started(self, *args, **kwargs):
        # - bot position available
        # - bot position minimap available
        # - E spell not on cooldown
        # - bot not at gate

        try:
            int(kwargs.get('bot_pos_frame')['circle'][1][0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        if kwargs.get('cooldowns')['E']:
            return False

        if is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'gate'):
            return False

        return True

    def process(self, *args, **kwargs):
        bot_pos_frame = kwargs.get('bot_pos_frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        cooldowns = kwargs.get('cooldowns')

        if self.steps[self.step_idx] == 'press_button':
            # get mouse position between bot and gate
            circle_pos = bot_pos_frame['circle'][1]
            gate_icon_pos = (1654, 912)
            vector = (gate_icon_pos[0] - bot_pos_minimap[0], gate_icon_pos[1] - bot_pos_minimap[1])
            mouse_pos = (circle_pos[0] + vector[0], circle_pos[1] + vector[1])
            # button
            pyautogui.moveTo(mouse_pos[0], mouse_pos[1])
            pyautogui.click(button='left')
            pyautogui.press('e')
            pyautogui.click(button='left')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'wait':
            if self.t1 is None:
                self.t1 = time.time()
            if time.time() - self.t1 >= 1.0:  # sec
                self.step_idx += 1
                self.t1 = None

        elif self.steps[self.step_idx] == 'spell_cooldown':
            if cooldowns['E']:
                self.step_idx += 1
                self.t1 = None
            else:
                # timeout?
                if self.t1 is None:
                    self.t1 = time.time()
                if time.time() - self.t1 >= self.timeout1:
                    self.set_result(-1, f'Could not detect spell cooldown for {self.timeout1} sec.')
                    self.t1 = None

        elif self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            if is_bot_icon_in_place(bot_pos_minimap, 'gate'):
                self.set_result(1, '')  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1654 - bot_pos_minimap[0]) + abs(912 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        self.set_result(-1, f'Bot is not moving towards gate since {len(self.diff_values)} frames')


class ActionUseSpellD(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'press_button',
            'spell_blocked',
            'spell_cooldown'
        ]
        self.t1 = None
        self.timeout1 = 5  # [sec]

    def can_be_started(self, *args, **kwargs):
        # - frame is available
        # - spell D not on cooldown
        # - bots' health < max

        frame = kwargs.get('frame')
        if type(frame) is not np.ndarray:
            return False

        if kwargs.get('cooldowns')['D']:
            return False

        try:
            int(kwargs.get('bot_health')[0])    # current
        except Exception:
            return False
        if int(kwargs.get('bot_health')[0]) > 0.95*int(kwargs.get('bot_health')[1]):
            return False

        return True

    def process(self, *args, **kwargs):
        frame = kwargs.get('frame')
        cooldowns = kwargs.get('cooldowns')

        if self.steps[self.step_idx] == 'press_button':
            pyautogui.moveTo(SCREEN_W // 2, SCREEN_H // 2)
            pyautogui.click(button='left')
            pyautogui.press('d')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'spell_blocked':
            # get part of image
            x_min, y_min, x_max, y_max = 1006, 1010, 1006 + 144, 1010 + 40
            img = frame[y_min:y_max, x_min:x_max]
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # get positions of template
            threshold = 0.8
            template = template_blocked_symbol
            x_shift, y_shit = template.shape[1] // 2, template.shape[0] // 2
            res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            if len(loc[0]) > 0:
                self.step_idx += 1  # found symbol
                self.t1 = None
            else:
                # timeout?
                if self.t1 is None:
                    self.t1 = time.time()
                if time.time() - self.t1 >= self.timeout1:
                    self.set_result(-1, f'Could not detect spell_blocked symbol for {self.timeout1} sec.')

        elif self.steps[self.step_idx] == 'spell_cooldown':
            if cooldowns['D']:
                self.set_result(1, '')


class ActionMove(Action):
    def __init__(self, direction):
        super().__init__()
        self.direction = direction
        self.steps = [
            'click_position',
            'at_position'
        ]
        self.x, self.y = None, None
        self.t1 = None
        self.timeout1 = 3  # [sec]

    def can_be_started(self, *args, **kwargs):
        # - bot position minimap available
        # - move direction can be reached

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        try:
            int(get_move_position(kwargs.get('bot_pos_minimap'), self.direction)[0])  # x
        except Exception:
            return False

        return True

    def process(self, *args, **kwargs):
        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.x is None or self.y is None:
            self.x, self.y = get_move_position(bot_pos_minimap, self.direction)

        if self.steps[self.step_idx] == 'click_position':
            pyautogui.moveTo(self.x, self.y)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_position':
            x, y = bot_pos_minimap
            if self.x - 4 <= x <= self.x + 4 and self.y - 4 <= y <= self.y + 4:
                self.set_result(1, '')  # finished

            # timeout?
            if self.t1 is None:
                self.t1 = time.time()
            if time.time() - self.t1 >= self.timeout1:
                self.set_result(-1, f'Reached timeout: {self.timeout1} sec.')


class ActionRunMiddle(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_middle',
            'at_middle'
        ]
        self.diff_values = [None] * 10

    def can_be_started(self, *args, **kwargs):
        # - bot pos icon is available
        # - bot not at middle

        try:
            int(kwargs.get('bot_pos_minimap')[0])  # x
        except Exception:
            return False

        if is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'middle'):
            return False

        return True

    def process(self, *args, **kwargs):
        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_middle':
            pyautogui.moveTo(1723, 896)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_middle':
            if is_bot_icon_in_place(bot_pos_minimap, 'middle'):
                self.set_result(1, '')  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1723 - bot_pos_minimap[0]) + abs(896 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        self.set_result(-1, f'Bot is not moving towards middle since {len(self.diff_values)} frames')


class ActionCollectGlobes(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_target',
            'at_target'
        ]
        self.point_target = None
        self.diff_values = [None] * 10

    def can_be_started(self, *args, **kwargs):
        # - bot position available
        # - bot icon position available
        # - at least one red minion

        try:
            int(kwargs.get('bot_pos_frame')['bounding_box'][1][0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('bot_pos_minimap')[0])   # x
        except Exception:
            return False

        try:
            int(kwargs.get('minions')['red'][0][0])   # x
        except Exception:
            return False

        return True

    def process(self, *args, **kwargs):
        bot_pos = kwargs.get('bot_pos_frame')['bounding_box'][1]    # (center_x, center_y)
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        minions = kwargs.get('minions')

        if self.point_target is None:
            # get position in the center of the group
            point_center = get_center_point_for_spell(minions, bot_pos, 'Q')
            diff = (point_center[0]-bot_pos[0]) // 16, (point_center[1]-bot_pos[1]) // 16,
            self.point_target = (bot_pos_minimap[0] + diff[0], bot_pos_minimap[1] + diff[1])

        if self.steps[self.step_idx] == 'click_target':
            pyautogui.moveTo(self.point_target)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_target':
            x, y = bot_pos_minimap
            if self.point_target[0]-4 <= x <= self.point_target[0]+4 and self.point_target[1]-4 <= y <= self.point_target[1]+4:
                self.set_result(1, '')  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(self.point_target[0] - x) + abs(self.point_target[1] - y)
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        self.set_result(-1, f'Bot is not moving towards target since {len(self.diff_values)} frames')


# import cv2 as cv
# from image_objects_lib import ImageObjects
# from tracker_lib import TrackerClass
# from painter_lib import PainterClass

# ImgObjs = ImageObjects()
# Tracker = TrackerClass()
# Painter = PainterClass()

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
