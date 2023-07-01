from typing import List, Tuple, Union
import pyautogui
from global_vars import *
import time
import math
from itertools import product, combinations


def get_distance_between_points(point1: Tuple[int, int],
                                point2: Tuple[int, int]) -> tuple[float, None] | tuple[None, str]:
    """
    Calculate the distance between two points in a 2D space.

    Args:
        point1 (Tuple[int, int]): The coordinates of the first point.
        point2 (Tuple[int, int]): The coordinates of the second point.

    Returns:
        tuple[float, None] | tuple[None, str]: The distance between the points if inputs are valid, otherwise None.

    Raises:
        None: This function does not raise any specific exceptions.
    """
    for point in [point1, point2]:
        if not isinstance(point, tuple) or len(point) != 2 or not all(isinstance(coord, int) for coord in point):
            return None, f'{point} should be a tuple(int, int)'

    return float(np.linalg.norm(np.array(point1) - np.array(point2))), None


def get_centroid_point(list_of_points: List[Tuple[int, int]]) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Calculate the centroid point of a list of 2D points.

    Args:
        list_of_points (List[Tuple[int, int]]): The list of points.

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: The centroid point as a tuple of integers if inputs are valid,
        otherwise returns None and an error description.

    Raises:
        None: This function does not raise any specific exceptions.
    """
    if not isinstance(list_of_points, list):
        return None, "list_of_points should be a list"
    if len(list_of_points) == 0:
        return None, "list_of_points should not be empty"
    for item in list_of_points:
        if not isinstance(item, tuple) or len(item) != 2 or not all(isinstance(coord, int) for coord in item):
            return None, f'{item} should not be tuple(int, int)'

    try:
        arr = np.array(list_of_points)
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return (int(sum_x / length), int(sum_y / length)), None
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return None, str(e)


def point_inside_polygon(x: int, y: int, poly: List[Union[int, Tuple[int, int]]]) \
        -> tuple[bool, None] | tuple[None, str]:
    """
    Check if a point is inside a polygon.

    Args:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        poly ([ int, Tuple[int, int], int, Tuple[int, int] ]): polygon

    Returns:
        tuple[bool, None] | tuple[None, str]: True if the point is inside the polygon, False otherwise,
        or returns None and an error description if inputs are invalid.

    Raises:
        None: This function does not raise any specific exceptions.
    """
    if not isinstance(x, int):
        return None, "x should be an integer"
    if not isinstance(y, int):
        return None, "y should be an integer"
    if not isinstance(poly, list):
        return None, "poly should be a list"
    if len(poly) != 4:
        return None, "poly should have 4 values"
    if not isinstance(poly[0], int):
        return None, "poly[0] should be an integer"
    if not isinstance(poly[1], tuple):
        return None, "poly[1] should be a tuple"
    if not isinstance(poly[2], int):
        return None, "poly[2] should be an integer"
    if not isinstance(poly[3], tuple):
        return None, "poly[3] should be a tuple"

    try:
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
        return inside, None
    except (TypeError, ZeroDivisionError) as e:
        return None, str(e)


def point_inside_ellipse(point: Tuple[int, int], point_center: Tuple[int, int], ellipse_type: str) \
        -> tuple[bool, None] | tuple[None, str]:
    """
    Check if a point is inside an ellipse.

    Args:
        point (Tuple[int, int]): The coordinates of the point.
        point_center (Tuple[int, int]): The coordinates of the ellipse center.
        ellipse_type (str): The type of ellipse ('Q' or 'W').

    Returns:
        tuple[bool, None] | tuple[None, str]: True if the point is inside the ellipse, False otherwise,
        or returns None and an error description if inputs are invalid.

    Raises:
        None: This function does not raise any specific exceptions.
    """
    for item in [point, point_center]:
        if not isinstance(item, tuple) or len(item) != 2 or not all(isinstance(coord, int) for coord in item):
            return None, f'{item} should be a tuple(int, int)'
    if not isinstance(ellipse_type, str) or ellipse_type not in ['Q', 'W']:
        return None, f'{ellipse_type} should be a str: Q or W'

    ELLIPSE_Q_WITH = 222
    ELLIPSE_Q_HEIGHT = 180
    ELLIPSE_W_WITH = 385
    ELLIPSE_W_HEIGHT = 305

    if ellipse_type == 'Q':
        a = ELLIPSE_Q_WITH // 2
        b = ELLIPSE_Q_HEIGHT // 2
    else:  # ellipse_type == 'W':
        a = ELLIPSE_W_WITH // 2
        b = ELLIPSE_W_HEIGHT // 2

    phi = math.atan2(a * point[1], b * point[0])
    point_on_ellipse = (int(point_center[0] + a * math.cos(phi)), int(point_center[1] + b * math.sin(phi)))
    dist_to_point, err_desc = get_distance_between_points(point_center, point)
    if dist_to_point is None:
        return None, err_desc
    dist_to_point_on_ellipse, err_desc = get_distance_between_points(point_center, point_on_ellipse)
    if dist_to_point_on_ellipse is None:
        return None, err_desc
    return dist_to_point <= dist_to_point_on_ellipse, None


def get_minions_positions(frame: np.ndarray, hsv_frame: np.ndarray) -> tuple[dict, None] | tuple[None, str]:
    """
    Get positions of minions in frame

    Args:
        frame (np.ndarray)
        hsv_frame (np.ndarray)

    Returns:
        tuple[dict, None] | tuple[None, str]: dictionary of positions if inputs are valid, otherwise returns
        None and an error description.

        - {'blue': [(x_center, y_center), ...],
        - 'red': [(x_center, y_center), ...]}

    Example:
        window = WindowCapture('Heroes of the Storm')
        img = window.get_screenshot()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        print(get_minions_positions(img, hsv))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'
    if not isinstance(hsv_frame, np.ndarray):
        return None, f'hsv_frame should be np.ndarray'

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
            return cv.inRange(hsv_frame, thr[0], thr[1])

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
                        inside, err_desc = point_inside_polygon(point[0], point[1], polygon)
                        if inside:
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
                        b, g, r, *a = frame[b_y_min][x]  # B,G,R,A
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_top += 1
                        # bottom border
                        b, g, r, *a = frame[b_y_max][x]
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_bottom += 1
                    cnt_left, cnt_right = 0, 0
                    for y in range(b_y_min, b_y_max + 1):
                        # left border
                        b, g, r, *a = frame[y][b_x_min]
                        if b <= b_thr and g <= g_thr and r <= r_thr:
                            cnt_left += 1
                        # right border
                        b, g, r, *a = frame[y][b_x_max]
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
            cv.imwrite('imgs\\_output1.jpg', frame)
            mask = get_mask()
            cv.imwrite('imgs\\_output2.jpg', mask)
            mask = transform_mask(mask)
            cv.imwrite('imgs\\_output3.jpg', mask)
            contours, _ = get_contours(mask)
            img = copy.deepcopy(frame)
            for contour in contours:
                x_min = min([_contour[0][0] for _contour in contour])  # point X
                x_max = max([_contour[0][0] for _contour in contour])  # point X
                y_min = min([_contour[0][1] for _contour in contour])  # point Y
                y_max = max([_contour[0][1] for _contour in contour])  # point Y
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output4.jpg', img)
            contours = filter_1(contours)
            img = copy.deepcopy(frame)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output5.jpg', img)
            contours = filter_2(contours)
            img = copy.deepcopy(frame)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output6.jpg', img)
            contours = filter_3(contours)
            img = copy.deepcopy(frame)
            for (x_min, y_min, x_max, y_max) in contours:
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_output7.jpg', img)
            contours = filter_4(contours)
            img = copy.deepcopy(frame)
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
            img = copy.deepcopy(frame)
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

    return out, None


def get_bot_positions(hsv_frame: np.ndarray) -> tuple[dict, None] | tuple[None, str]:
    """
    Get positions of bot in hsv frame

    Args:
        hsv_frame (np.ndarray)

    Returns:
        tuple[dict, None] | tuple[None, str]: dictionary of positions if inputs are valid, otherwise returns
        None and an error description.

        - {'health_bar':  ((x_min, y_min, x_max, y_max), (x_center,y_center)),
        - 'bounding_box': ((x_min, y_min, x_max, y_max), (x_center,y_center)),
        - 'circle':      ((x_min, y_min, x_max, y_max), (x_center,y_center))}

    Example:
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
    if not isinstance(hsv_frame, np.ndarray):
        return None, f'hsv_frame should be np.ndarray'

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
            hsv_small = hsv_frame[shift_y_min:shift_y_max, shift_x_min:shift_x_max]
            mask = cv.inRange(hsv_small, _HEALTH_BAR_COLOR_THR[0], _HEALTH_BAR_COLOR_THR[1])
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

    # final check before return
    for key, value in out.items():
        for val in [j for i in value for j in i]:   # =all values from "value"
            if not isinstance(val, int):
                return None, f'value:{val} inside output is not integer'

    return out, None


def get_bot_health_value(frame: np.ndarray) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get value of bot's mana in the frame

    Args:
        frame (np.ndarray)

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: current value of mana, max value of mana if inputs are valid,
        otherwise returns None and an error description.

    Example:
        from functions import get_bot_health_value
        file_path = f'imgs\\detect\\health\\'
        for cnt in range(1, 7):
            frame = cv.imread(f'{file_path}health_{cnt}.png', cv.IMREAD_UNCHANGED)
            print(f'health_{cnt}.png', get_bot_health_value(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 206, 992, 420, 1010
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get position of slash character
    template = template_H10_slash
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= res.max())
    if len(loc[0]) == 0:
        return None, f'Did not find slash-character position'
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
        return None, f'Did not detect numbers (H08)'
    if int(val_current_str) > int(val_max_str):
        return None, f'Current value={val_current_str} greater than max value={val_max_str}'

    return (int(val_current_str), int(val_max_str)), None


def get_bot_mana_value(frame: np.ndarray) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get value of bot's mana in the frame

    Args:
        frame (np.ndarray)

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: current value of mana, max value of mana if inputs are valid,
        otherwise returns None and an error description.

    Example:
        from functions import get_bot_mana_value
        file_path = f'imgs\\detect\\mana\\'
        for cnt in range(1, 5):
            frame = cv.imread(f'{file_path}mana_{cnt}.png', cv.IMREAD_UNCHANGED)
            print(f'mana_{cnt}.png', get_bot_mana_value(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 198, 1013, 403, 1030
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get position of slash character
    template = template_H08_slash
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= res.max())
    if len(loc[0]) == 0:
        return None, f'Did not find slash-character position'
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
        return None, f'Did not detect numbers (H08)'
    if int(val_current_str) > int(val_max_str):
        return None, f'Current value={val_current_str} greater than max value={val_max_str}'

    return (int(val_current_str), int(val_max_str)), None


def get_cooldowns(frame: np.ndarray) -> tuple[dict, None] | tuple[None, str]:
    """
    Get cooldowns in the frame

    Args:
        frame (np.ndarray)

    Returns:
        tuple[dict, None] | tuple[None, str]: dictionary of cooldown names and bolean values if cooldown is True or False if inputs are valid, otherwise returns
        None and an error description.
        {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}

    Example:
        file_path = f'imgs\\detect\\cooldowns.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_cooldowns(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    out = {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'well': False}

    # Q, W, E, R, D
    threshold = 60  # min number of pixels == cooldown number is present
    for i, key in enumerate(out.keys()):
        x_min, y_min, x_max, y_max = 762 + 85 * i, 1019, 762 + 85 * i + 57, 1019 + 24
        hsv = cv.cvtColor(frame[y_min:y_max, x_min:x_max], cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 0, 200), (1, 1, 255))
        num_pixels = len(np.where(mask >= 200)[0])
        if num_pixels >= threshold:
            out[key] = True  # found number -> is on cooldown

    # well
    #   get part of image
    x_mini, y_mini, x_maxi, y_maxi = 182, 1038, 182 + 34, 1038 + 17
    img = frame[y_mini:y_maxi, x_mini:x_maxi]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #   template
    thr = 0.8
    template = template_plus_symbol
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= thr)
    #   result
    if len(loc[0]) == 0:
        out['well'] = True  # did not find plus sign -> is on cooldown

    return out, None


def get_xp_from_tab(frame: np.ndarray) -> tuple[int, None] | tuple[None, str]:
    """
    Get experience (XP) value in the frame, after 'TAB' button is pressed
    !IMPORTANT!: press 'TAB' to read values from window. After reading, press 'TAB' again, to hide window.

    Args:
        frame (np.ndarray)

    Returns:
        tuple[int, None] | tuple[None, str]: (xp value, None) or (None, error description) if error occurred.

    Example:
        import functions
        import cv2 as cv
        file_path = f'imgs\\detect\\xp\\xp_tab_1.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(functions.get_xp_from_tab(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 1355, 250, 1355 + 100, 250 + 30
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of numbers
    out = []  # (x,y,number)
    threshold = 0.8
    for idx, template in enumerate([template_H13_num_0, template_H13_num_1, template_H13_num_2,
                                    template_H13_num_3, template_H13_num_4, template_H13_num_5,
                                    template_H13_num_6, template_H13_num_7, template_H13_num_8,
                                    template_H13_num_9]):
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            out.append((pt[0], pt[1], idx))
    if len(out) == 0:
        return None, f'No numbers (H13) found'
    #   sort numbers (by x position)
    out.sort(key=lambda x: x[0])

    # filter only unique matches (if the same number is detected too close, means it is the same number)
    out_filtered = []
    prev_x, prev_y, prev_num = None, None, None
    for x, y, num in out:
        if prev_num is None:
            prev_x, prev_y, prev_num = x, y, num
            out_filtered.append((x, y, num))  # first number
        else:
            if num != prev_num:
                out_filtered.append((x, y, num))  # different number
            else:  # same number
                if abs(x - prev_x) >= 5:  # x-shifted in image ==> different number
                    out_filtered.append((x, y, num))
            prev_x, prev_y, prev_num = x, y, num
    if len(out_filtered) == 0:
        return None, f'No numbers (H13), after filtering, found'

    # create final value
    return int(''.join([str(num) for (x, y, num) in out_filtered])), None


# OBSOLETE function
def get_xp_from_level(frame: np.ndarray, **kwargs) -> tuple[int, None] | tuple[None, str]:
    """
    Get value of XP in the frame. XP value is calculated based on XP bar and XP number

    Args:
        frame (np.ndarray)
        kwargs: 'xp_prev' = previous value of XP

    Returns:
        tuple[int, None] | tuple[None, str]: XP value if inputs are valid, otherwise returns
        None and an error description. If 'xp_prev' is available then returns greater value

    Example:
        import cv2 as cv
        from functions import get_xp_from_level
        file_path = f'imgs\\detect\\xp\\780.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_xp_from_level(frame, xp_prev=0))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    def calculate_xp_value(in_level: int, in_progress: float) -> int:
        """
        in: in_level - number of level [0,1,2,...30]
        in: progress - current progress, based on XP bar [0.0; 1.0]
        out: xp_value [int]
        """
        if in_level == 1:
            curr_sum, next_sum = 0, 2010
        elif in_level == 2:
            curr_sum, next_sum = 2010, 2154
        elif in_level == 3:
            curr_sum, next_sum = 4164, 2154
        elif in_level == 4:
            curr_sum, next_sum = 6318, 2154
        elif in_level == 5:
            curr_sum, next_sum = 8472, 2154
        elif in_level == 6:
            curr_sum, next_sum = 10626, 3303
        elif in_level == 7:
            curr_sum, next_sum = 13929, 3303
        elif in_level == 8:
            curr_sum, next_sum = 17232, 3303
        elif in_level == 9:
            curr_sum, next_sum = 20535, 3303
        elif in_level == 10:
            curr_sum, next_sum = 23838, 3303
        elif in_level == 11:
            curr_sum, next_sum = 27141, 4452
        elif in_level == 12:
            curr_sum, next_sum = 31593, 4452
        elif in_level == 13:
            curr_sum, next_sum = 36045, 4452
        elif in_level == 14:
            curr_sum, next_sum = 40497, 4452
        elif in_level == 15:
            curr_sum, next_sum = 44949, 4452
        elif in_level == 16:
            curr_sum, next_sum = 49401, 5600
        elif in_level == 17:
            curr_sum, next_sum = 55001, 5600
        elif in_level == 18:
            curr_sum, next_sum = 60601, 5600
        elif in_level == 19:
            curr_sum, next_sum = 66201, 5600
        elif in_level == 20:
            curr_sum, next_sum = 71801, 9000
        elif in_level == 21:
            curr_sum, next_sum = 80801, 10000
        elif in_level == 22:
            curr_sum, next_sum = 90801, 11500
        elif in_level == 23:
            curr_sum, next_sum = 102301, 13000
        elif in_level == 24:
            curr_sum, next_sum = 115301, 15000
        elif in_level == 25:
            curr_sum, next_sum = 130301, 17000
        elif in_level == 26:
            curr_sum, next_sum = 147301, 19500
        elif in_level == 27:
            curr_sum, next_sum = 166801, 22000
        elif in_level == 28:
            curr_sum, next_sum = 188801, 25000
        elif in_level == 29:
            curr_sum, next_sum = 213801, 28000
        else:  # in_level == 30
            curr_sum, next_sum = 241801, 0

        return int('%d' % (curr_sum + in_progress * next_sum))  # get values before comma

    # LEVEL detection  -----------------------------------
    # get part of image
    x_min, y_min, x_max, y_max = 865, 25, 920, 64
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # get positions of numbers
    out = []  # (x,y,number)
    threshold = 0.8
    for idx, template in enumerate([template_H33_num_0, template_H33_num_1, template_H33_num_2,
                                    template_H33_num_3, template_H33_num_4, template_H33_num_5,
                                    template_H33_num_6, template_H33_num_7, template_H33_num_8,
                                    template_H33_num_9]):
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            out.append((pt[0], pt[1], idx))
    if len(out) == 0:
        return None, f'No numbers (H33) detected'
    # sort numbers (by x position)
    out.sort(key=lambda x: x[0])
    # filter only unique matches (if the same number is detected too close, means it is the same number)
    out_filtered = []
    prev_x, prev_y, prev_num = None, None, None
    for x, y, num in out:
        if prev_num is None:
            prev_x, prev_y, prev_num = x, y, num
            out_filtered.append((x, y, num))  # first number
        else:
            if num != prev_num:
                out_filtered.append((x, y, num))  # different number
            else:  # same number
                if abs(x - prev_x) >= 10:  # x-shifted in image ==> different number
                    out_filtered.append((x, y, num))
            prev_x, prev_y, prev_num = x, y, num
    if len(out_filtered) > 2:
        return None, f'More than 2 numbers (H33) detected'
    # final value
    level = int(''.join([str(num) for (x, y, num) in out_filtered]))

    # BAR detection -----------------------------------
    # get part of image
    x_min, y_min, x_max, y_max = 838, 5, 838 + 81, 5 + 12
    img = frame[y_min:y_max, x_min:x_max]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    color_thr = [np.array([90, 80, 90]), np.array([180, 255, 255])]
    mask = cv.inRange(hsv, color_thr[0], color_thr[1])
    # get rid of 'edges' pixels (h,w)
    blank_pixels_left = [(2, 0), (3, 0), (4, 0), (4, 0), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (7, 0), (7, 1),
                         (7, 2), (7, 3), (8, 0), (8, 1), (8, 2), (8, 3), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
                         (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (11, 0), (11, 1), (11, 2), (11, 3), (11, 4),
                         (11, 5)]
    blank_pixels_right = [(0, 75), (0, 76), (0, 77), (0, 78), (0, 79), (0, 80), (1, 76), (1, 77), (1, 78), (1, 79),
                          (1, 80), (2, 76), (2, 77), (2, 78), (2, 79), (2, 80), (3, 77), (3, 78), (3, 79), (3, 80),
                          (4, 77), (4, 78), (4, 79), (4, 80), (5, 78), (5, 79), (5, 80), (6, 78), (6, 79), (6, 80),
                          (7, 79), (7, 80), (8, 79), (8, 80), (9, 80)]
    for h, w in blank_pixels_left: mask[h, w] = 0
    for h, w in blank_pixels_right: mask[h, w] = 0
    # find the lowest, most right pixel in the mask
    idx = None
    for h in range(mask.shape[0] - 1, 0, -1):
        for w in range(mask.shape[1] - 1, 0, -1):
            if mask[h][w] > 100:
                # check if there is SOLID line of pixels reaching the left blank edge
                left_w = None
                tmp = [pixel_w for pixel_h, pixel_w in blank_pixels_left if pixel_h == h]  # pixels with the same height
                if len(tmp) > 0:
                    left_w = max(tmp)  # pixel closest to center
                else:
                    left_w = -1
                if all(val > 100 for val in mask[h][left_w + 1: w + 1]):
                    idx = w
            if idx:
                break
        if idx:
            break
    # compensate idx value
    if idx is None:
        idx = 0  # did not find the bar's pixel color -> just started number
    if idx + 1 <= mask.shape[1]:
        idx += 1
    # final value
    progress = idx / mask.shape[1]

    # XP value -----------------------------------
    xp = calculate_xp_value(level, progress)
    xp_prev = kwargs.get('xp_prev')
    if xp_prev is None:
        return xp, None
    else:
        return max(xp, xp_prev), None


# OBSOLETE function
def get_total_xp_value(frame: np.ndarray) -> tuple[int, None] | tuple[None, str]:
    """
    Get total value of XP in the frame
    @Note: to read value, need to move mouse to position (x=960, y=44) and wait 1.5 sec

    Args:
        frame (np.ndarray)

    Returns:
        tuple[int, None] | tuple[None, str]: total value of XP if inputs are valid, otherwise returns
        None and an error description.

    Example:
        file_path = f'imgs\\detect\\xp_total.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_total_xp_value(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 875, 243, 875 + 70, 243 + 18
    img = frame[y_min:y_max, x_min:x_max]
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

    if len(out) == 0:
        return None, f'No XP numbers found'

    #   sort numbers (by x position)
    out.sort(key=lambda x: x[0])
    # create final value
    return int(''.join([str(num) for (x, y, num) in out])), None


def is_bot_hidden(frame: np.ndarray) -> tuple[bool, None] | tuple[None, str]:
    """
    Checks if bot is hidden in the frame

    Args:
        frame (np.ndarray)

    Returns:
        tuple[bool, None] | tuple[None, str]: True if bot is hidden or False if is not hidden, if inputs are invalid returns
        None and an error description.

    Example:
        file_path = f'imgs\\detect\\hidden.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(is_bot_hidden(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 930, 250, 930 + 60, 250 + 90
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.45
    template = template_hidden_symbol
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # final output
    return len(loc[0]) > 0, None


def is_bot_dead(frame: np.ndarray) -> tuple[bool, None] | tuple[None, str]:
    """
    Check if bot is dead in the frame

    Args:
        frame (np.ndarray)

    Returns:
        tuple[bool, None] | tuple[None, str]: True if bot is dead or False if is not dead, if inputs are invalid returns
        None and an error description.

    Example:
        file_path = f'imgs\\detect\\death.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(is_bot_dead(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 945, 696, 945 + 30, 696 + 30
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.8
    template = template_death_icon
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # final output
    return len(loc[0]) > 0, None


def get_bot_icon_position(frame: np.ndarray) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get the position of the bot icon in the frame.

    Args:
        frame (np.ndarray)

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: The coordinates (x, y) of the bot icon position if inputs are valid,
        otherwise returns None and an error description.

    Example:
        file_path = f'imgs\\detect\\gate.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_bot_icon_position(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 1558, 863, 1558 + 366, 863 + 183
    img = frame[y_min:y_max, x_min:x_max]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get positions of template
    threshold = 0.58
    template = template_bot_icon
    x_shift, y_shit = template.shape[1] // 2, template.shape[0] // 2 + 6
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None, f"could not find bot's icon position"
    else:
        x_out, y_out = x_min + loc[::-1][0][0] + x_shift, y_min + loc[::-1][1][0] + y_shit
        return (int(x_out), int(y_out)), None


def is_bot_icon_in_place(position: tuple[int, int], place: str) -> tuple[bool, None] | tuple[None, str]:
    """
    Check if the bot icon is in the specified position.

    Args:
        position (tuple[int, int]): The position coordinates (x, y).
        place (str): The place to check. Possible values: 'gate', 'bush', 'middle'.

    Returns:
        tuple[bool, None] | tuple[None, str]: True if the bot icon is in the specified place, False otherwi. If
        inputs are invalid returns None and an error description.

    Raises:
        None: This function does not raise any specific exceptions.
    """
    if not isinstance(position, tuple) or len(position) != 2 or not all(isinstance(item, int) for item in position):
        return None, f'position={position} should be tuple[int, int]'
    if not isinstance(place, str) or place not in ['gate', 'bush', 'middle']:
        return None, f'place={place} should be str'

    x, y = position

    if place == 'gate':
        x_min, y_min, x_max, y_max = 1651, 899, 1651 + 6, 899 + 17
        return x_min <= x <= x_max and y_min <= y <= y_max, None

    elif place == 'bush':
        x_min, y_min, x_max, y_max = 1712, 880, 1712 + 23, 880 + 4
        return x_min <= x <= x_max and y_min <= y <= y_max, None

    else:  # place == 'middle':
        x_min, y_min, x_max, y_max = 1723 - 5, 896 - 5, 1723 + 5, 896 + 5
        return x_min <= x <= x_max and y_min <= y <= y_max, None


def get_well_position(frame: np.ndarray) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get the well's position from frame.
    !Note! to get position, bot need to be at place: 'gate'

    Args:
        frame (np.ndarray)

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: The well's position (x, y) if inputs are valid,
        otherwise returns None and an error description.

    Raises:
        None: This function does not raise any specific exceptions.

    Example:
        file_path = f'imgs\\detect\\gate.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_well_position(frame))
    """
    if not isinstance(frame, np.ndarray):
        return None, f'frame should be np.ndarray'

    # get part of image
    x_min, y_min, x_max, y_max = 550, 250, 550 + 350, 250 + 300
    img = frame[y_min:y_max, x_min:x_max]
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
        return (int(out_x), int(out_y)), None
    else:
        return None, "Could not find well's circle"

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


def get_center_point_for_spell(minions: dict, bot_pos: tuple[int, int], spell_type: str, **kwargs) -> \
        tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get the center point for a spell based on minions positions and bot position.
    - get points that are close to each other
    - calculate all centroids for 2,3,4,5... points
    - choose center point of centroid that ellipse includes the most points

    Args:
        minions (dict): Dictionary containing minion positions.
        bot_pos (tuple[int, int]):  bot position (x,y).
        spell_type (str): Type of the spell.
        **kwargs: 'frame'

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: The center point (x, y) if inputs are valid,
        otherwise returns None and an error description.

    Raises:
        None: This function does not raise any specific exceptions.

    Example:
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
        else:  # spell_type == 'W'
            ELLIPSE_WITH = 385

        tups = [sorted(sub) for sub in product(in_points, repeat=2) if
                get_distance_between_points(*sub)[0] <= ELLIPSE_WITH // 2]

        res_dict = {point: {point} for point in in_points}
        out_points = []
        for tup1, tup2 in tups:
            res_dict[tup1] |= res_dict[tup2]
            res_dict[tup2] = res_dict[tup1]
            for val in [tup1, tup2]:
                if len(res_dict[val]) > len(out_points):
                    out_points = res_dict[val]

        return list(out_points)

    try:
        int(minions['red'][0][0])  # x
    except Exception as e:
        return None, f'minions["red"]={minions["red"]} shall be List[Tuple[int, int]]'
    if not isinstance(bot_pos, tuple) or len(bot_pos) != 2 or not all(isinstance(item, int) for item in bot_pos):
        return None, f'bot_pos={bot_pos} shall be Tuple[int, int]'
    if not isinstance(spell_type, str) or spell_type not in ['Q', 'W']:
        return None, f'{spell_type} should be a str: Q or W'

    # get points
    points = get_points(minions['red'])
    #   1 point -> return point that is closest to bot
    if len(points) == 1:
        distances = []
        for minion_pos in minions['red']:
            dist, err_desc = get_distance_between_points(bot_pos, minion_pos)
            if dist is None:
                return None, err_desc
            distances.append((dist, minion_pos))
        if len(distances) == 0:
            return None, 'No distances to minions'
        distances.sort()
        return distances[0][1], None  # pos
    #   2 points -> return point between them
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return (int(x1 + (x1 - x2)), int(y1 + (y1 - y2))), None

    # get centroids
    centroids = set()  # set( tuple(int, int) )
    for num_tuples in range(2, len(points) + 1):
        for obj in combinations(points, num_tuples):
            pos, err_desc = get_centroid_point(list(obj))
            if pos is None:
                return None, err_desc
            centroids.add(pos)

    # get how many points are inside ellipse which center is defined by centroid
    centroids_ellipse = []
    max_val = -1
    for pos_centroid in centroids:
        val = 0
        for pos_point in points:
            inside, err_desc = point_inside_ellipse(pos_point, pos_centroid, spell_type)
            if inside is None:
                return None, err_desc
            if inside:
                val += 1
        if val > max_val:
            centroids_ellipse.append((pos_centroid, val))
            max_val = val

    # choose centroid that includes the most of the points
    centroids_ellipse.sort(key=lambda x: x[1])
    point_center = centroids_ellipse[-1][0]

    frame = kwargs.get('frame')  # FOR DEBUGGING
    if frame is not None:
        print('number of points inside=', centroids_ellipse[-1][1])
        for (x, y) in points:
            cv.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), 2)
        cv.rectangle(frame, (point_center[0] - 1, point_center[1] - 1), (point_center[0] + 1, point_center[1] + 1),
                     (255, 0, 0), 2)
        if spell_type == 'Q':
            ELLIPSE_WITH = 222
            ELLIPSE_HEIGHT = 180
        else:  # spell_type == 'W'
            ELLIPSE_WITH = 385
            ELLIPSE_HEIGHT = 305
        cv.ellipse(frame, point_center, (ELLIPSE_WITH // 2, ELLIPSE_HEIGHT // 2), 0, 0, 360, (255, 0, 0), 1)
        cv.imwrite('imgs\\_output.jpg', frame)

    return (int(point_center[0]), int(point_center[1])), None


def get_move_position(position: tuple[int, int], direction: str) -> tuple[tuple[int, int], None] | tuple[None, str]:
    """
    Get the new position based on the given direction.

    Args:
        position (tuple[int, int]): The current position (x, y).
        direction (str): The direction to move.

    Returns:
        tuple[tuple[int, int], None] | tuple[None, str]: The new position (x, y) if inputs are valid,
        otherwise returns None and an error description.

    Raises:
        None: This function does not raise any specific exceptions.

    Example:
        file_path = f'imgs\\detect\\gate.png'
        frame = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        print(get_move_position(frame, 'up'))
    """
    if not isinstance(position, tuple) or len(position) != 2 or not all(isinstance(item, int) for item in position):
        return None, f'position={position} should be tuple(int, int)'
    if not isinstance(direction, str) or direction not in ['up', 'down', 'right', 'left', 'up-right', 'down-right',
                                                           'up-left', 'down-left']:
        return None, f'direction={direction} should be str'

    dx = 8
    dy = 8
    dxy = 5
    x_shift = SKIP_AREA_BOTTOM_RIGHT[0][0]
    y_shift = SKIP_AREA_BOTTOM_RIGHT[0][1]
    x, y = position

    if direction == 'up':
        new_x, new_y = x, y - dy
    elif direction == 'down':
        new_x, new_y = x, y + dy
    elif direction == 'right':
        new_x, new_y = x + dx, y
    elif direction == 'left':
        new_x, new_y = x - dx, y
    elif direction == 'up-right':
        new_x, new_y = x + dxy, y - dxy
    elif direction == 'down-right':
        new_x, new_y = x + dxy, y + dxy
    elif direction == 'up-left':
        new_x, new_y = x - dxy, y - dxy
    else:  # direction == 'down-left':
        new_x, new_y = x - dxy, y + dxy

    if not (0 <= new_x - x_shift < template_minimap.shape[1]) or not (0 <= new_y - y_shift < template_minimap.shape[0]):
        return None, 'New position outside of minimap'
    elif template_minimap[new_y - y_shift, new_x - x_shift] != 255:
        return None, 'New position not accessible of minimap'
    else:
        return (int(new_x), int(new_y)), None


def label_frame(frame: np.ndarray):
    """
    Label frame with rectangles and information

    Args:
        frame (np.ndarray)

    Returns:
        frame (np.ndarray): labeled frame

    Raises:
        None: This function does not raise any specific exceptions.

    Example:
        labeled_frame = label_frame(frame)
    """
    #'bot_dead':         None or  bool
    #'bot_pos_frame':    None or  {'health_bar': tuple[int, int], 'bounding_box': tuple[int, int], 'circle': tuple[int, int]}
    #'bot_pos_minimap':  None or  tuple[int, int]
    #'bot_health':       None or  tuple[int, int]
    #'bot_mana':         None or  tuple[int, int]
    #'cooldowns':        None or  {'Q': bool, 'W': bool, 'E': bool, 'R': bool, 'D': bool, 'well': bool}
    #'minions':          None or  {'blue': list[tuple[int, int]], 'red': list[tuple[int, int]]}

    # gather data
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    bot_dead, err_desc = is_bot_dead(frame)
    bot_pos_frame, err_desc = get_bot_positions(frame_hsv)  # 'bounding_box': ((x_min, y_min, x_max, y_max), (x_center,y_center)),
    bot_pos_minimap, desc = get_bot_icon_position(frame)
    bot_health, err_desc = get_bot_health_value(frame)
    bot_mana, err_desc = get_bot_mana_value(frame)
    cooldowns, err_desc = get_cooldowns(frame)
    minions, err_desc = get_minions_positions(frame, frame_hsv)  # {'blue': [(x_center, y_center), ...]

    # draw rectangles
    #   bot
    ((x_min, y_min, x_max, y_max), (x_center,y_center)) = bot_pos_frame['bounding_box']
    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    #   minions
    for color_name, color in [('red', (0, 0, 255)), ('blue',(255, 0, 0))]:
        for x_center, y_center in minions[color_name]:
            x_min, y_min, x_max, y_max = x_center-70//2, y_center-76//2, x_center+70//2, y_center+76//2
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # draw box
    x_min, y_min, x_max, y_max = 10, 10, 380, 240
    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)    # white background
    #text = f'Bot position = {bot_pos_frame["bounding_box"][1]}'
    text = f'Bot position = {bot_pos_minimap}'
    y = y_min + 25; dy = 32
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = f'Bot health = {bot_health[0]} / {bot_health[1]}'; y+=dy
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = f'Bot mana = {bot_mana[0]} / {bot_mana[1]}'; y+=dy
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = f'Num. BLUE minions = {len(minions["blue"])}'; y+=dy
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = f'Num. RED minions = {len(minions["red"])}'; y+=dy
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = f'Cooldowns:'; y+=dy
    cv.putText(img=frame, text=text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    text = ', '.join([name for name, val  in cooldowns.items() if val]); y+=dy
    if not text: text = '-'
    cv.putText(img=frame, text='  ' + text, org=(x_min + 5, y), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

    return frame


class Action:
    def __init__(self):
        self.steps = []
        self.step_idx = None
        self.t0 = None
        self.TIMEOUT = 30  # sec

    def is_available(self, *args, **kwargs):
        raise NotImplemented

    def start(self):
        self.step_idx = 0
        self.t0 = time.time()

    def process(self, *args, **kwargs):
        raise NotImplemented


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

    def start(self, action_name):
        """! IMPORTANT !: make sure that action is available to start"""
        if self.current_action is None:
            action = self.objects[action_name]

            # Reset all values
            if action_name.find('move_') >= 0:  # special for move actions
                direction = action_name[5:]
                action.__init__(direction)
            else:
                action.__init__()

            # start action
            self.current_action = action
            self.current_action.start()
            return True, ''

        else:
            return False, 'current_action is not None'

    def process(self, *args, **kwargs):
        """
        out: action_result, action_description
        action_result:
        # None - error
        # 0    - in progress
        # 1    - finished successfully
        # -1   - finished unsuccessfully
        """
        if self.current_action is not None:
            # result [-1, 0, 1], description [None or str]
            result, description = self.current_action.process(*args, **kwargs)

            # if finished or timeout -> clear action
            if result in [-1, 1]:
                if time.time() - self.current_action.t0 >= self.current_action.TIMEOUT:
                    description = f'Reached timeout = {self.current_action.TIMEOUT} sec.'
                self.current_action = None

            return result, description
        else:
            return None, f'self.current_action is None'

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

    def is_available(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion

        if kwargs.get('bot_pos_frame'):
            if kwargs.get('bot_pos_frame')['bounding_box']:
                if type(kwargs.get('bot_pos_frame')['bounding_box'][0]) is int:
                    pass
                else:
                    return False, "bot_pos_frame['bounding_box'][0] is not integer"
            else:
                return False, "bot_pos_frame['bounding_box'] is None"
        else:
            return False, "bot_pos_frame is None"

        if kwargs.get('minions'):
            if kwargs.get('minions')['red']:
                if kwargs.get('minions')['red'][0]:
                    if type(kwargs.get('minions')['red'][0][0]) is int:
                        pass
                    else:
                        return False, "minions['red'][0][0] is not integer"
                else:
                    return False, "minions['red'][0] is None"
            else:
                return False, "minions['red'] is None"
        else:
            return False, "minions is None"

        return True, ''

    def process(self, *args, **kwargs) -> tuple[int, str] | tuple[int, None]:
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_frame') is None:
            return -1, f'bot_pos_frame is None'
        if kwargs.get('minions') is None:
            return -1, f'minions is None'
        if len(kwargs.get('minions')['red']) == 0:
            return -1, f'no RED minions found'

        bot_x, bot_y = kwargs.get('bot_pos_frame')['bounding_box']  # (center x, center y)
        minions = kwargs.get('minions')['red']

        # get the position of the closest red minion == target
        distances = []
        for minion_pos in minions:
            dist, err_desc = get_distance_between_points((bot_x, bot_y), minion_pos)
            if dist is None:
                return -1, err_desc  # error!
            distances.append((dist, minion_pos))
        if len(distances) == 0:
            return -1, 'No distances to minions'  # error!
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
            else:  # already close to target
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'click_attack':
            pyautogui.moveTo(target)
            pyautogui.press('a')
            return 1, None  # finished

        return 0, None


class ActionQAttack(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'choose_target',
            'click_attack',
            'spell_cooldown'
        ]
        self.point_attack = None

    def is_available(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion
        # - Q spell available

        if kwargs.get('bot_pos_frame'):
            if kwargs.get('bot_pos_frame')['bounding_box']:
                if type(kwargs.get('bot_pos_frame')['bounding_box'][0]) is int:
                    pass
                else:
                    return False, "bot_pos_frame['bounding_box'][0] is not integer"
            else:
                return False, "bot_pos_frame['bounding_box'] is None"
        else:
            return False, "bot_pos_frame is None"

        if kwargs.get('minions'):
            if kwargs.get('minions')['red']:
                if kwargs.get('minions')['red'][0]:
                    if type(kwargs.get('minions')['red'][0][0]) is int:
                        pass
                    else:
                        return False, "minions['red'][0][0] is not integer"
                else:
                    return False, "minions['red'][0] is None"
            else:
                return False, "minions['red'] is None"
        else:
            return False, "minions is None"

        if kwargs.get('cooldowns'):
            if not kwargs.get('cooldowns')['Q']:
                pass
            else:
                return False, "cooldowns['Q'] is True"
        else:
            return False, "cooldowns is None"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_frame') is None:
            return -1, f'bot_pos_frame is None'
        if kwargs.get('minions') is None:
            return -1, f'minions is None'
        if kwargs.get('cooldowns') is None:
            return -1, f'cooldowns is None'

        bot_pos = kwargs.get('bot_pos_frame')['bounding_box']  # (center_x, center_y)
        minions = kwargs.get('minions')
        cooldowns = kwargs.get('cooldowns')

        if self.point_attack is None:
            point_attack, err_desc = get_center_point_for_spell(minions, bot_pos, 'Q')
            if point_attack is None:
                return -1, err_desc  # error!
            self.point_attack = point_attack

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
                return 1, None  # finished

        return 0, None


class ActionWAttack(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'choose_target',
            'click_attack',
            'spell_cooldown'
        ]
        self.point_attack = None

    def is_available(self, *args, **kwargs):
        # - bot position available
        # - at least one red minion
        # - W spell available

        if kwargs.get('bot_pos_frame'):
            if kwargs.get('bot_pos_frame')['bounding_box']:
                if type(kwargs.get('bot_pos_frame')['bounding_box'][0]) is int:
                    pass
                else:
                    return False, "bot_pos_frame['bounding_box'][0] is not integer"
            else:
                return False, "bot_pos_frame['bounding_box'] is None"
        else:
            return False, "bot_pos_frame is None"

        if kwargs.get('minions'):
            if kwargs.get('minions')['red']:
                if kwargs.get('minions')['red'][0]:
                    if type(kwargs.get('minions')['red'][0][0]) is int:
                        pass
                    else:
                        return False, "minions['red'][0][0] is not integer"
                else:
                    return False, "minions['red'][0] is None"
            else:
                return False, "minions['red'] is None"
        else:
            return False, "minions is None"

        if kwargs.get('cooldowns'):
            if not kwargs.get('cooldowns')['W']:
                pass
            else:
                return False, "cooldowns['W'] is True"
        else:
            return False, "cooldowns is None"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_frame') is None:
            return -1, f'bot_pos_frame is None'
        if kwargs.get('minions') is None:
            return -1, f'minions is None'
        if kwargs.get('cooldowns') is None:
            return -1, f'cooldowns is None'

        bot_pos = kwargs.get('bot_pos_frame')['bounding_box']  # (center_x, center_y)
        minions = kwargs.get('minions')
        cooldowns = kwargs.get('cooldowns')

        if self.point_attack is None:
            point_attack, err_desc = get_center_point_for_spell(minions, bot_pos, 'W')
            if point_attack is None:
                return -1, err_desc  # error!
            self.point_attack = point_attack

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
                return 1, None  # finished

        return 0, None


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

    def is_available(self, *args, **kwargs):
        # - frame is available
        # - bot_pos_minimap is available
        # - well not on cooldown
        # - bot's health < 0.9 *max

        if kwargs.get('frame') is not None:
            if type(kwargs.get('frame')) is np.ndarray:
                pass
            else:
                return False, 'frame is not np.ndarray'
        else:
            return False, 'frame is None'

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        if kwargs.get('cooldowns'):
            if not kwargs.get('cooldowns')['well']:
                pass
            else:
                return False, "cooldowns['well'] is True"
        else:
            return False, "cooldowns is None"

        if kwargs.get('bot_health'):
            if type(kwargs.get('bot_health')[0]) is int:  # current
                if int(kwargs.get('bot_health')[0]) < 0.9 * int(kwargs.get('bot_health')[1]):  # current < max
                    pass
                else:
                    return False, "bot_health: current value > 0.9*max value"
            else:
                return False, "bot_health[0] is not integer"
        else:
            return False, "bot_health is None"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('frame') is None:
            return -1, f'frame is None'
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'
        if kwargs.get('cooldowns') is None:
            return -1, f'cooldowns is None'

        frame = kwargs.get('frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        cooldowns = kwargs.get('cooldowns')

        if self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            in_place, err_desc = is_bot_icon_in_place(bot_pos_minimap, 'gate')
            if in_place is None:
                return -1, err_desc  # error!
            if in_place:
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'find_well':
            well_pos, err_desc = get_well_position(frame)
            if well_pos is None:
                return -1, err_desc  # error!
            self.well_x, self.well_y = well_pos
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'click_well':
            pyautogui.moveTo(self.well_x, self.well_y)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'well_cooldown':
            if cooldowns['well']:
                return 1, None  # finished

        return 0, None


class ActionHideInBushes(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_bushes',
            'at_bushes',
            'hidden'
        ]
        self.hidden_values = [None] * 10

    def is_available(self, *args, **kwargs):
        # - frame is available
        # - bot_pos_minimap is available
        # - bot not in bushes

        if kwargs.get('frame') is not None:
            if type(kwargs.get('frame')) is np.ndarray:
                pass
            else:
                return False, 'frame is not np.ndarray'
        else:
            return False, 'frame is None'

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        in_place, err_desc = is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'bush')
        if in_place is None:
            return False, err_desc
        if not in_place:
            pass
        else:
            return False, "bot is already in bushes"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('frame') is None:
            return -1, f'frame is None'
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'

        frame = kwargs.get('frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_bushes':
            pyautogui.moveTo(1722, 879)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_bushes':
            in_place, err_desc = is_bot_icon_in_place(bot_pos_minimap, 'bush')
            if in_place is None:
                return -1, err_desc  # error!
            if in_place:
                self.step_idx += 1

        elif self.steps[self.step_idx] == 'hidden':
            try:
                idx = self.hidden_values.index(None)
                is_hidden, err_desc = is_bot_hidden(frame)
                if is_hidden is None:
                    return -1, err_desc  # error!
                if is_hidden:
                    return 1, None  # finished
                self.hidden_values[idx] = is_hidden
            except ValueError:
                return -1, f'Bot is not hidden in bushes'

        return 0, None


class ActionHideBehindGate(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_gate',
            'at_gate'
        ]
        self.diff_values = [None] * 10

    def is_available(self, *args, **kwargs):
        # - bot pos icon is available
        # - bot not at gate

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        in_place, err_desc = is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'gate')
        if in_place is None:
            return False, err_desc
        if not in_place:
            pass
        else:
            return False, "bot is already at gate"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'

        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            in_place, err_desc = is_bot_icon_in_place(bot_pos_minimap, 'gate')
            if in_place is None:
                return -1, err_desc  # error!
            if in_place:
                return 1, None  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1654 - bot_pos_minimap[0]) + abs(912 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        return -1, f'Bot is not moving towards gate since {len(self.diff_values)} frames'

        return 0, None


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

    def is_available(self, *args, **kwargs):
        # - bot position available
        # - bot position minimap available
        # - E spell not on cooldown
        # - bot not at gate

        if kwargs.get('bot_pos_frame'):
            if kwargs.get('bot_pos_frame')['circle']:
                if type(kwargs.get('bot_pos_frame')['circle'][0]) is int:  # x
                    pass
                else:
                    return False, "bot_pos_frame['circle'][0] is not integer"
            else:
                return False, "bot_pos_frame['circle'] is None"
        else:
            return False, "bot_pos_frame is None"

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        if kwargs.get('cooldowns'):
            if not kwargs.get('cooldowns')['E']:
                pass
            else:
                return False, "cooldowns['E'] is True"
        else:
            return False, "cooldowns is None"

        in_place, err_desc = is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'gate')
        if in_place is None:
            return False, err_desc
        if not in_place:
            pass
        else:
            return False, "bot is already at gate"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_frame') is None:
            return -1, f'bot_pos_frame is None'
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'
        if kwargs.get('cooldowns') is None:
            return -1, f'cooldowns is None'

        bot_pos_frame = kwargs.get('bot_pos_frame')
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        cooldowns = kwargs.get('cooldowns')

        if self.steps[self.step_idx] == 'press_button':
            # get mouse position between bot and gate
            circle_pos = bot_pos_frame['circle']
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
                    return -1, f'Could not detect spell cooldown for {self.timeout1} sec.'  # error!

        elif self.steps[self.step_idx] == 'click_gate':
            pyautogui.moveTo(1654, 912)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_gate':
            in_place, err_desc = is_bot_icon_in_place(bot_pos_minimap, 'gate')
            if in_place is None:
                return -1, err_desc  # error!
            if in_place:
                return 1, None  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1654 - bot_pos_minimap[0]) + abs(912 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        return -1, f'Bot is not moving towards gate since {len(self.diff_values)} frames'  # error!

        return 0, None


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

    def is_available(self, *args, **kwargs):
        # - frame is available
        # - spell D not on cooldown

        if kwargs.get('frame') is not None:
            if type(kwargs.get('frame')) is np.ndarray:
                pass
            else:
                return False, 'frame is not np.ndarray'
        else:
            return False, 'frame is None'

        if kwargs.get('cooldowns'):
            if not kwargs.get('cooldowns')['D']:
                pass
            else:
                return False, "cooldowns['D'] is True"
        else:
            return False, "cooldowns is None"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('frame') is None:
            return -1, f'frame is None'
        if kwargs.get('cooldowns') is None:
            return -1, f'cooldowns is None'

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
                    return -1, f'Could not detect spell_blocked symbol for {self.timeout1} sec.'  # error!

        elif self.steps[self.step_idx] == 'spell_cooldown':
            if cooldowns['D']:
                return 1, None  # finished

        return 0, None


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

    def is_available(self, *args, **kwargs):
        # - bot position minimap available
        # - move direction can be reached

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        pos, err_desc = get_move_position(kwargs.get('bot_pos_minimap'), self.direction)
        if pos is None:
            return False, err_desc

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'

        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.x is None or self.y is None:
            pos, err_desc = get_move_position(bot_pos_minimap, self.direction)
            if pos is None:
                return -1, err_desc  # error!
            self.x, self.y = pos

        if self.steps[self.step_idx] == 'click_position':
            pyautogui.moveTo(self.x, self.y)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_position':
            x, y = bot_pos_minimap
            if self.x - 4 <= x <= self.x + 4 and self.y - 4 <= y <= self.y + 4:
                return 1, None  # finished

            # timeout?
            if self.t1 is None:
                self.t1 = time.time()
            if time.time() - self.t1 >= self.timeout1:
                return -1, f'Reached timeout: {self.timeout1} sec.'  # error!

        return 0, None


class ActionRunMiddle(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_middle',
            'at_middle'
        ]
        self.diff_values = [None] * 10

    def is_available(self, *args, **kwargs):
        # - bot pos icon is available
        # - bot not at middle

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        in_place, err_desc = is_bot_icon_in_place(kwargs.get('bot_pos_minimap'), 'middle')
        if in_place is None:
            return False, err_desc
        if not in_place:
            pass
        else:
            return False, "bot is already in middle"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'

        bot_pos_minimap = kwargs.get('bot_pos_minimap')

        if self.steps[self.step_idx] == 'click_middle':
            pyautogui.moveTo(1723, 896)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_middle':
            in_place, err_desc = is_bot_icon_in_place(bot_pos_minimap, 'middle')
            if in_place is None:
                return -1, err_desc  # error!
            if in_place:
                return 1, None  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(1723 - bot_pos_minimap[0]) + abs(896 - bot_pos_minimap[1])
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        return -1, f'Bot is not moving towards middle since {len(self.diff_values)} frames'  # error!

        return 0, None


class ActionCollectGlobes(Action):
    def __init__(self):
        super().__init__()
        self.steps = [
            'click_target',
            'at_target'
        ]
        self.point_target = None
        self.diff_values = [None] * 10

    def is_available(self, *args, **kwargs):
        # - bot position available
        # - bot icon position available
        # - at least one red minion

        if kwargs.get('bot_pos_frame'):
            if kwargs.get('bot_pos_frame')['bounding_box']:
                if type(kwargs.get('bot_pos_frame')['bounding_box'][0]) is int:  # x
                    pass
                else:
                    return False, "bot_pos_frame['bounding_box'][0] is not integer"
            else:
                return False, "bot_pos_frame['bounding_box'] is None"
        else:
            return False, "bot_pos_frame is None"

        if kwargs.get('bot_pos_minimap'):
            if type(kwargs.get('bot_pos_minimap')[0]) is int:  # x
                pass
            else:
                return False, "bot_pos_minimap[0] is not integer"
        else:
            return False, "bot_pos_minimap is None"

        if kwargs.get('minions'):
            if kwargs.get('minions')['red']:
                if kwargs.get('minions')['red'][0]:
                    if type(kwargs.get('minions')['red'][0][0]) is int:
                        pass
                    else:
                        return False, "minions['red'][0][0] is not integer"
                else:
                    return False, "minions['red'][0] is None"
            else:
                return False, "minions['red'] is None"
        else:
            return False, "minions is None"

        return True, ''

    def process(self, *args, **kwargs):
        """
        Process action and return it's result

        Args:
            kwargs: values that are set in: spectator_lib.Spectator.data

        Returns:
            tuple[int, str] | tuple[int, None]: result [-1, 0, 1], description [None or str]
            0=action still in process, -1=action finished with error, 1=action finished with success

        Raises:
            None: This function does not raise any specific exceptions.
        """
        # check inputs
        if kwargs.get('bot_pos_frame') is None:
            return -1, f'bot_pos_frame is None'
        if kwargs.get('bot_pos_minimap') is None:
            return -1, f'bot_pos_minimap is None'
        if kwargs.get('minions') is None:
            return -1, f'minions is None'

        bot_pos = kwargs.get('bot_pos_frame')['bounding_box']  # (center_x, center_y)
        bot_pos_minimap = kwargs.get('bot_pos_minimap')
        minions = kwargs.get('minions')

        if self.point_target is None:
            # get position in the center of the group
            point_center, err_desc = get_center_point_for_spell(minions, bot_pos, 'Q')
            if point_center is None:
                return -1, err_desc  # error!
            diff = (point_center[0] - bot_pos[0]) // 16, (point_center[1] - bot_pos[1]) // 16,
            self.point_target = (bot_pos_minimap[0] + diff[0], bot_pos_minimap[1] + diff[1])

        if self.steps[self.step_idx] == 'click_target':
            pyautogui.moveTo(self.point_target)
            pyautogui.click(button='right')
            self.step_idx += 1

        elif self.steps[self.step_idx] == 'at_target':
            x, y = bot_pos_minimap
            if self.point_target[0] - 4 <= x <= self.point_target[0] + 4 and self.point_target[1] - 4 <= y <= \
                    self.point_target[1] + 4:
                return 1, None  # finished
            else:
                try:
                    idx = self.diff_values.index(None)
                    val = abs(self.point_target[0] - x) + abs(self.point_target[1] - y)
                    self.diff_values[idx] = val
                except ValueError:
                    # all values are gathered
                    if self.diff_values[-1] >= self.diff_values[0]:
                        return -1, f'Bot is not moving towards target since {len(self.diff_values)} frames'   # error!

        return 0, None



# OBSOLETE code
"""
# import cv2 as cv
# from image_objects_lib import ImageObjects
# from tracker_lib import TrackerClass
# from painter_lib import PainterClass

# ImgObjs = ImageObjects()
# Tracker = TrackerClass()
# Painter = PainterClass()

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
