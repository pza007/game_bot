import cv2 as cv
import numpy as np
from global_vars import _HB_BOX_MIN_H, _HB_BOX_MAX_H, _HB_BOX_MIN_W, _HB_BOX_MAX_W, _HB_SKIP_AREAS, _HB_COLOR_THRS_RED,\
    _HB_COLOR_THRS_BLUE, _MN_HAT_MAGE_RED, _MN_HAT_MAGE_BLUE, _BOT_CIRCLE_COLOR_THR, _BOT_HB_COLOR_THR, _BOT_HB_BOX_MIN_H, \
    _BOT_HB_BOX_MAX_H, _BOT_HB_BOX_MIN_W, _BOT_HB_BOX_MAX_W, _BOT_HB_MAX_Y_SHIFT


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


class HealthBarsClass:
    def __init__(self, img, hsv, color):
        self.objects = []   # <list of HealthBarClass>
        self.img = img     # image with B,G,R values
        self.hsv = hsv     # image with H,S,V values
        self.color = color  # <str> 'red' or 'blue'
        self.color_thresholds = []  # <list of np.array([int, int, int])>  H,S,V values
        self.mask = None  # output array of the same size as src and CV_8U type
        self.contours = []  # [[(x_min, y_min), (x_max, y_max)], ...]

    class HealthBarClass:
        def __init__(self, position):
            self.position = position  # <int> [(x_min, y_min), (x_max, y_max)]
            self.fill_lvl = self.set_fill_lvl()  # <int> 0-100 %

        def set_fill_lvl(self):
            [(x_min, y_min), (x_max, y_max)] = self.position
            width = x_max - x_min + 1
            return int((width / _HB_BOX_MAX_W) * 100)

    def get_from_image(self):
        self.filter_by_color()
        self.transform_mask()
        self.get_contours()
        self.filter_contours_by_borders()
        self.filter_contours_by_color_fill()
        self.filter_contours_by_dimensions()
        self.filter_contours_by_position_in_image()

    def filter_by_color(self):
        # set color thresholds
        if self.color == 'red':
            self.color_thresholds = _HB_COLOR_THRS_RED
        elif self.color == 'blue':
            self.color_thresholds = _HB_COLOR_THRS_BLUE

        # create mask, based on thresholds
        self.mask = cv.inRange(self.hsv, self.color_thresholds[0], self.color_thresholds[1])

    def transform_mask(self):
        # open
        kernel = np.ones((4, 4), np.uint8)
        result = cv.morphologyEx(self.mask, cv.MORPH_OPEN, kernel)
        result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
        result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
        # dilate
        kernel = np.ones((4, 4), np.uint8)
        result = cv.dilate(result, kernel, iterations=2)
        self.mask = result

    def get_contours(self):
        contours, hierarchy = cv.findContours(self.mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        out_contours = []
        for contour in contours:
            x_min = min([_contour[0][0] for _contour in contour])  # point X
            x_max = max([_contour[0][0] for _contour in contour])  # point X
            y_min = min([_contour[0][1] for _contour in contour])  # point Y
            y_max = max([_contour[0][1] for _contour in contour])  # point Y
            out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    def filter_contours_by_borders(self):
        thr_value_for_color_change = 50  # %

        def color_changed(list_1, list_2):
            # when comparing hsv arrays and the change of values is more than threshold -> then the color has been changed
            for val1, val2 in zip(list_1, list_2):
                if int(val1) == 0:  # case: divide by 0
                    return False
                elif ((int(val1) - int(val2)) / int(val1)) * 100 >= thr_value_for_color_change:
                    return True
            return False

        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            found_top, found_bottom, found_left, found_right = False, False, False, False
            x_min_new, y_min_new, x_max_new, y_max_new = None, None, None, None

            # find starting point
            x_start, y_start = None, None
            found_start = False
            #   travel through diagonal: y=y+1, x=x+1
            y, x = y_min, x_min
            while y <= y_max and x <= x_max:
                h, s, v = self.hsv[y][x]
                if self.color_thresholds[0][0] < h < self.color_thresholds[1][0] and \
                        self.color_thresholds[0][1] < s < self.color_thresholds[1][1] and \
                        self.color_thresholds[0][2] < v < self.color_thresholds[1][2]:
                    found_start = True
                    y_start = y
                    x_start = x
                    break
                y, x = y+1, x+1

            # find borders if starting point exists
            if found_start:
                # find top border
                for y in range(y_start, y_start - _HB_BOX_MAX_H, -1):
                    if color_changed(self.hsv[y][x_start], self.hsv[y - 1][x_start]):
                        found_top = True
                        y_min_new = y - 1
                        break
                # bottom border
                for y in range(y_start, y_start + _HB_BOX_MAX_H):
                    if color_changed(self.hsv[y][x_start], self.hsv[y + 1][x_start]):
                        found_bottom = True
                        y_max_new = y + 1
                        break
                # left border
                for x in range(x_start, x_start - _HB_BOX_MAX_W, -1):
                    if color_changed(self.hsv[y_start][x], self.hsv[y_start][x - 1]):
                        found_left = True
                        x_min_new = x - 1
                        break
                # right border
                for x in range(x_start, x_start + _HB_BOX_MAX_W):
                    if color_changed(self.hsv[y_start][x], self.hsv[y_start][x + 1]):
                        found_right = True
                        x_max_new = x + 1
                        break

                # add contour if all borders are found
                if found_top and found_bottom and found_left and found_right:
                    if x_min_new > x_max_new or y_min_new > y_max_new:
                        raise Exception(f'Incorrect detection of borders: min=({x_min_new, y_min_new}), max=({x_max_new, y_max_new})')
                    out_contours.append([(x_min_new, y_min_new), (x_max_new, y_max_new)])
            #print('\t', [(x_min, y_min), (x_max, y_max)], found_top, found_bottom, found_left, found_right, [[(x_min_new, y_min_new), (x_max_new, y_max_new)]])
        self.contours = out_contours

    """
    # based on contours min,max values, go all the way on each polygon side
    def filter_contours_by_borders(self):
        thr_value_for_color_change = 50  # %
        thr_number_of_pixels_defining_border = 50  # %

        def color_changed(list_1, list_2):
            # when comparing hsv arrays and the change of values is more than threshold -> then the color has been changed
            for val1, val2 in zip(list_1, list_2):
                if int(val1) == 0:  # case: divide by 0
                    return False
                elif ((int(val1) - int(val2)) / int(val1)) * 100 >= thr_value_for_color_change:
                    return True
            return False

        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            found_top, found_bottom, found_left, found_right = False, False, False, False
            x_min_new, y_min_new, x_max_new, y_max_new = None, None, None, None
            x_s = x_min + (x_max - x_min) // 2
            y_s = y_min + (y_max - y_min) // 2
            x_step = (x_s - x_min) // 2 + 1
            y_step = (y_s - y_min) // 2 + 1

            # top border
            for y in range(y_s, y_min - y_step, -1):
                cnt = 0
                for x in range(x_min, x_max):
                    if color_changed(self.hsv[y][x], self.hsv[y - 1][x]):
                        cnt += 1
                if cnt / len(range(x_min, x_max)) * 100 >= thr_number_of_pixels_defining_border:
                    found_top = True
                    y_min_new = y - 1
                    break
            # bottom border
            for y in range(y_s, y_max + y_step):
                cnt = 0
                for x in range(x_min, x_max):
                    if color_changed(self.hsv[y][x], self.hsv[y + 1][x]):
                        cnt += 1
                if cnt / len(range(x_min, x_max)) * 100 >= thr_number_of_pixels_defining_border:
                    found_bottom = True
                    y_max_new = y + 1
                    break
            # left border
            for x in range(x_min + x_step, x_min - x_step, -1):
                cnt = 0
                for y in range(y_min, y_max):
                    if color_changed(self.hsv[y][x], self.hsv[y][x - 1]):
                        cnt += 1
                if cnt / len(range(y_min, y_max)) * 100 >= thr_number_of_pixels_defining_border:
                    found_left = True
                    x_min_new = x - 1
                    break
            # right border
            for x in range(x_max - x_step, x_max + x_step):
                cnt = 0
                for y in range(y_min, y_max):
                    if color_changed(self.hsv[y][x], self.hsv[y][x + 1]):
                        cnt += 1
                if cnt / len(range(y_min, y_max)) * 100 >= thr_number_of_pixels_defining_border:
                    found_right = True
                    x_max_new = x + 1
                    break

            # add contour if all borders are found
            if found_top and found_bottom and found_left and found_right:
                if x_min_new > x_max_new or y_min_new > y_max_new:
                    raise Exception(
                        f'Incorrect detection of borders: min=({x_min_new, y_min_new}), max=({x_max_new, y_max_new})')
                out_contours.append([(x_min_new, y_min_new), (x_max_new, y_max_new)])
            # print('\t', [(x_min, y_min), (x_max, y_max)], found_top, found_bottom, found_left, found_right)
        self.contours = out_contours

        # draw
        if len(self.contours) > 0:
            for [(x_min, y_min), (x_max, y_max)] in self.contours:
                cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv.imwrite('imgs\\_4_contours_2.jpg', self.img)
    """
    def filter_contours_by_color_fill(self):
        thr_value_for_color_fill = 60  # %

        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cnt = 0
            for y in range(y_min + 1, y_max):  # ignore the outside, black frame (1 pixel wide)
                for x in range(x_min + 1, x_max):  # ignore the outside, black frame (1 pixel wide)
                    h, s, v = self.hsv[y][x]
                    if self.color_thresholds[0][0] < h < self.color_thresholds[1][0] and \
                            self.color_thresholds[0][1] < s < self.color_thresholds[1][1] and \
                            self.color_thresholds[0][2] < v < self.color_thresholds[1][2]:
                        cnt += 1
            # add contour if most area is filled with color
            if (cnt / (len(range(y_min + 1, y_max)) * len(range(x_min + 1, x_max)))) * 100 >= thr_value_for_color_fill:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    def filter_contours_by_dimensions(self):
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            if _HB_BOX_MIN_H <= height <= _HB_BOX_MAX_H and _HB_BOX_MIN_W <= width <= _HB_BOX_MAX_W:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    def filter_contours_by_position_in_image(self):
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            success = True
            for box in _HB_SKIP_AREAS:
                points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                polygon = [box[0], (box[1][0], box[0][1]), box[1], (box[0][0], box[1][1])]
                for point in points:
                    if point_inside_polygon(point[0], point[1], polygon):
                        success = False
                        break
                if not success:
                    break
            if success:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    def fill_objects(self):
        self.objects = []
        for contour in self.contours:
            self.objects.append(HealthBarsClass.HealthBarClass(contour))


class MinionsClass:
    def __init__(self, img, hsv, color):
        self.objects = []   # <list of MinionClass>
        self.img = img     # image with B,G,R values
        self.hsv = hsv     # image with H,S,V values
        self.color = color  # <str> 'red' or 'blue'
        self.health_bars = None  # <HealthBarsClass>

    class MinionClass:
        def __init__(self, img, color, health_bar):
            self.img = img
            self.color = color
            self.position = self.set_position(health_bar)    # <int> [(x_min, y_min), (x_max, y_max)]
            self.point_center = self.set_point_center()    # <int> (x,y)
            self.type = self.set_type()    # <str> 'mage', 'any'
            self.health_bar = health_bar    # <HealthBarClass>
            self.health = health_bar.fill_lvl  # <int> 0-100 %

        def set_position(self, health_bar):
            [(x_min, y_min), (x_max, y_max)] = health_bar.position
            x_min_new = x_min - 30
            x_max_new = x_min + _HB_BOX_MAX_W + 30
            y_min_new = y_max + 1
            y_max_new = y_min_new + 130
            return [(x_min_new, y_min_new), (x_max_new, y_max_new)]

        def set_point_center(self):
            [(x_min, y_min), (x_max, y_max)] = self.position
            return x_min + (x_max - x_min)//2, y_min + (y_max - y_min)//2

        def set_type(self):
            out_type = 'any'
            [(x_min, y_min), (x_max, y_max)] = self.position
            cropped_image = self.img[y_min:y_max, x_min:x_max]
            if self.color == 'red':
                hat = _MN_HAT_MAGE_RED
            else:   # blue
                hat = _MN_HAT_MAGE_BLUE

            match_result = cv.matchTemplate(cropped_image, hat, cv.TM_CCOEFF_NORMED)  # cv.TM_SQDIFF_NORMED
            threshold = 0.7
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)
            #print('Best match top left position: %s' % str(max_loc))
            #print('Best match confidence: %s' % max_val)
            if max_val >= threshold:
                out_type = 'mage'

            return out_type

    def get_from_image(self):
        self.get_health_bars()
        self.fill_objects()

    def get_health_bars(self):
        self.health_bars = HealthBarsClass(self.img, self.hsv, self.color)
        self.health_bars.get_from_image()
        self.health_bars.fill_objects()

    def fill_objects(self):
        self.objects = []
        for health_bar in self.health_bars.objects:
            self.objects.append(MinionsClass.MinionClass(self.img, self.color, health_bar))

    def draw(self, in_img):
        for minion in self.objects:
            if self.color == 'red':
                col = (0, 0, 255)
            else:   # blue
                col = (255, 0, 0)
            # draw rectangles for minion's health bar
            #[(x_min, y_min), (x_max, y_max)] = minion.health_bar.position
            #cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), col, -1)
            #cv.rectangle(in_img, (x_min, y_min), (x_min+_HB_BOX_MAX_W, y_max), col, 2)
            # draw line for minion's position
            #cv.line(in_img, minion.point_center, (minion.point_center[0], y_max), col, 2)

            # draw rectangle for center position
            cv.rectangle(in_img,
                         (minion.point_center[0] - 28, minion.point_center[1] - 10),
                         (minion.point_center[0] + 15, minion.point_center[1] + 19),
                         col, -1)
            # draw text for minion's type and %health
            cv.putText(
                img=in_img,
                text=minion.type.upper(),
                org=(minion.point_center[0] - 25, minion.point_center[1] + 2),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(0, 0, 0),
                thickness=1)
            cv.putText(
                img=in_img,
                text=str(minion.health)+'%',
                org=(minion.point_center[0] - 25, minion.point_center[1] + 14),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(0, 0, 0),
                thickness=1)

        return in_img


class BotClass:
    def __init__(self, img, hsv):
        self.img = img     # image with B,G,R values
        self.hsv = hsv     # image with H,S,V values
        self.circle_x, self.circle_y, self.circle_r = None, None, None   # <int>
        self.health_bar = None  # <HealthBarsClass.HealthBarClass>
        self.health = None  # <int> 0-100 %
        self.point_center = (None, None)    # <int> (x,y)

    def get_from_image(self):
        self.get_circle()
        self.get_health_bar()
        self.get_center_point()

    def get_circle(self):
        self.circle_x, self.circle_y = None, None
        #   mask
        mask = cv.inRange(self.hsv, _BOT_CIRCLE_COLOR_THR[0], _BOT_CIRCLE_COLOR_THR[1])
        #   transform
        kernel = np.ones((4, 4), np.uint8)
        result = cv.dilate(mask, kernel, iterations=1)
        #   circle
        #       param1 - sensitivity
        #       param2 - minimal number of edges
        circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=40, param2=10, minRadius=20, maxRadius=200)
        if circles is not None:
            position = np.uint16(np.around(circles[0]))[0]
            self.circle_x, self.circle_y, self.circle_r = position[0], position[1], position[2]
        else:
            print('BOT: circle not found!')

    def get_health_bar(self):
        #   mask
        mask = cv.inRange(self.hsv, _BOT_HB_COLOR_THR[0], _BOT_HB_COLOR_THR[1])
        #   transform
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=3)
        #   contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        out_contours = []
        for contour in contours:
            x_min = min([_contour[0][0] for _contour in contour])  # point X
            x_max = max([_contour[0][0] for _contour in contour])  # point X
            y_min = min([_contour[0][1] for _contour in contour])  # point Y
            y_max = max([_contour[0][1] for _contour in contour])  # point Y
            out_contours.append([(x_min, y_min), (x_max, y_max)])
        contours = out_contours
        # debug only ----
        #for [(x_min, y_min), (x_max, y_max)] in contours:
        #    cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        #cv.imwrite('imgs\\output1.jpg', self.img)
        # ----------------
        #   filter contours -> is contour located in rectangle, above circle's center?
        out_contours = []
        search_polygon = [(self.circle_x-self.circle_r-_BOT_HB_BOX_MAX_W, self.circle_y-_BOT_HB_MAX_Y_SHIFT),
                          (self.circle_x+self.circle_r+_BOT_HB_BOX_MAX_W, self.circle_y-_BOT_HB_MAX_Y_SHIFT),
                          (self.circle_x+self.circle_r+_BOT_HB_BOX_MAX_W, self.circle_y),
                          (self.circle_x-self.circle_r-_BOT_HB_BOX_MAX_W, self.circle_y)]
        # debug only ----
        #cv.rectangle(self.img, (search_polygon[0][0], search_polygon[0][1]), (search_polygon[2][0], search_polygon[2][1]), (255, 0, 0), 2)
        #cv.imwrite('imgs\\output2.jpg', self.img)
        # ----------------
        for [(x_min, y_min), (x_max, y_max)] in contours:
            points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            for point in points:
                if point_inside_polygon(point[0], point[1], search_polygon):
                    out_contours.append([(x_min, y_min), (x_max, y_max)])
                    break
        contours = out_contours
        #   filter contours -> do contour dimensions fit health bar dimensions?
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in contours:
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            if _BOT_HB_BOX_MIN_H <= height <= _BOT_HB_BOX_MAX_H and _BOT_HB_BOX_MIN_W <= width <= _BOT_HB_BOX_MAX_W:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        contours = out_contours
        #   if there is one contour -> add health bar
        if len(contours) == 1:
            self.health_bar = HealthBarsClass.HealthBarClass(contours[0])
            # fill_lvl
            [(x_min, y_min), (x_max, y_max)] = self.health_bar.position
            width = x_max - x_min + 1
            self.health_bar.fill_lvl = int((width / _BOT_HB_BOX_MAX_W) * 100)
            # health
            self.health = self.health_bar.fill_lvl
        else:
            print('BOT: health bar not found!')

    def get_center_point(self):
        if self.health_bar is not None:
            x_min = self.health_bar.position[0][0]     # [(x_min, y_min), (x_max, y_max)]
            y_min = self.health_bar.position[0][1]
            x_new = x_min + int(_BOT_HB_BOX_MAX_W//2)
            y_new = self.circle_y - int((self.circle_y - y_min)*0.25)
            self.point_center = (x_new, y_new)
        else:
            print('BOT: center point not found!')

    def draw(self, in_img):
        if None not in [self.circle_x, self.circle_y, self.health_bar, self.point_center[0], self.point_center[1]]:
            # draw rectangles for health bar
            #[(x_min, y_min), (x_max, y_max)] = self.health_bar.position
            #cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
            #cv.rectangle(in_img, (x_min, y_min), (x_min+_BOT_HB_BOX_MAX_W, y_max), (0, 255, 255), 2)
            # draw line for center position
            #cv.line(in_img, self.point_center, (self.point_center[0], y_max), (0, 255, 255), 2)

            # draw rectangle for center position
            cv.rectangle(in_img,
                         (self.point_center[0] - 20, self.point_center[1] - 10),
                         (self.point_center[0] + 16, self.point_center[1] + 19),
                         (0, 255, 255), -1)
            # draw text for type and %health
            cv.putText(
                img=in_img,
                text='BOT',
                org=(self.point_center[0] - 20, self.point_center[1] + 3),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1)
            cv.putText(
                img=in_img,
                text=str(self.health)+'%',
                org=(self.point_center[0] - 20, self.point_center[1] + 16),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1)

            return in_img
