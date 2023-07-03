import cv2 as cv
import numpy as np
import time
from global_vars import _SKIP_AREAS, _SCREEN_W, _SCREEN_H,\
    _HB_BOX_H, _HB_BOX_MIN_W, _HB_BOX_MAX_W, _HB_COLOR_THRS_RED, _HB_COLOR_THRS_BLUE,\
    _MN_BOX_W, _MN_BOX_H, _MN_BOX_H_SHIFT, _MN_BOX_X_SHIFT,\
    _BOT_HB_COLOR_THR, _BOT_HB_BOX_MIN_H, _BOT_HB_BOX_MAX_H, _BOT_HB_BOX_MIN_W, _BOT_HB_BOX_W, _BOT_HB_BOX_MAX_W,\
    _BOT_BOX_W_MOUNT, _BOT_BOX_H_MOUNT, _BOT_BOX_H_SHIFT_MOUNT, _BOT_BOX_X_SHIFT_MOUNT,\
    _BOT_BOX_W_UNMOUNT, _BOT_BOX_H_UNMOUNT, _BOT_BOX_H_SHIFT_UNMOUNT, _BOT_BOX_X_SHIFT_UNMOUNT,\
    _WELL_HB_BOX_H, _WELL_HB_BOX_W, _WELL_HB_COLOR_THR


# alternative: pointPolygonTest	(	InputArray 	contour, Point2f 	pt, bool 	measureDist)
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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageObjects(metaclass=Singleton):
    def __init__(self):
        self.img = None     # image with B,G,R,A values
        self.hsv = None     # image with H,S,V values

        self.bot = BotsClass()
        self.minions_red = MinionsClass('red')      # MinionsClass
        self.minions_blue = MinionsClass('blue')    # MinionsClass
        self.minimap = MapClass()

        self.detected_objects = [self.bot, self.minions_red, self.minions_blue]

        # OTHER
        self.minions_ghost = MinionsGhostClass()
        self.manas = ManasClass()


class HealthBarsClass:
    def __init__(self, img, hsv, color):
        self.objects = []   # <list of HealthBarClass>
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
        self.color = color  # <str> 'red' or 'blue'
        self.color_thresholds = []  # <list of np.array([int, int, int])>  H,S,V values
        self.mask = None  # output array of the same size as src and CV_8U type
        self.contours = []  # [[(x_min, y_min), (x_max, y_max)], ...]

    class HealthBarClass:
        def __init__(self, position):
            self.position = position  # <int> [(x_min, y_min), (x_max, y_max)]
            self.fill_lvl = None  # <int> 0-100 %

    def get_mask(self):
        # set color thresholds
        if self.color == 'red':
            self.color_thresholds = _HB_COLOR_THRS_RED
        elif self.color == 'blue':
            self.color_thresholds = _HB_COLOR_THRS_BLUE
        # create mask, based on thresholds
        self.mask = cv.inRange(self.hsv, self.color_thresholds[0], self.color_thresholds[1])

    def transform_mask(self):
        kernel = np.ones((3, 3), np.uint8)
        result = cv.morphologyEx(self.mask, cv.MORPH_OPEN, kernel)
        self.mask = result

    def get_contours(self):
        self.contours, hierarchy = cv.findContours(self.mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # filter contours with 4 points (rectangle)
    def filter_1(self):
        out_contours = []
        for contour in self.contours:
            if len(contour) == 4:
                x_min = min([_contour[0][0] for _contour in contour])  # point X
                x_max = max([_contour[0][0] for _contour in contour])  # point X
                y_min = min([_contour[0][1] for _contour in contour])  # point Y
                y_max = max([_contour[0][1] for _contour in contour])  # point Y
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    # filter contours with correct dimensions
    def filter_2(self):
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            if height == _HB_BOX_H and _HB_BOX_MIN_W <= width <= _HB_BOX_MAX_W:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        self.contours = out_contours

    # filter contours located outside skip areas
    def filter_3(self):
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            success = True
            for box in _SKIP_AREAS:
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

    # filter contours with black border outside
    def filter_4(self):
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            # min,max points of border inside window?
            b_x_min, b_y_min = x_min-1, y_min-1
            b_x_max, b_y_max = x_max+_HB_BOX_MAX_W-(x_max - x_min)+1, y_max+1
            if 0 <= b_x_min <= _SCREEN_W-1 and 0 <= b_y_min <= _SCREEN_H-1 and \
                    0 <= b_x_max <= _SCREEN_W-1 and 0 <= b_y_max <= _SCREEN_H-1:

                # border color thresholds
                b_thr, g_thr, r_thr = 0, 0, 0     # B,G,R
                if self.color == 'red':
                    b_thr, g_thr, r_thr = 10, 10, 90
                else:   # blue
                    b_thr, g_thr, r_thr = 90, 10, 10

                # borders
                cnt_top, cnt_bottom = 0, 0
                for x in range(b_x_min, b_x_max + 1):
                    # top border
                    [b, g, r, a] = self.img[b_y_min][x]    # B,G,R,A
                    if b <= b_thr and g <= g_thr and r <= r_thr:
                        cnt_top += 1
                    # bottom border
                    [b, g, r, a] = self.img[b_y_max][x]
                    if b <= b_thr and g <= g_thr and r <= r_thr:
                        cnt_bottom += 1
                cnt_left, cnt_right = 0, 0
                for y in range(b_y_min, b_y_max + 1):
                    # left border
                    [b, g, r, a] = self.img[y][b_x_min]
                    if b <= b_thr and g <= g_thr and r <= r_thr:
                        cnt_left += 1
                    # right border
                    [b, g, r, a] = self.img[y][b_x_max]
                    if b <= b_thr and g <= g_thr and r <= r_thr:
                        cnt_right += 1

                # final check
                #print(cnt_top, cnt_bottom, cnt_left, cnt_right)
                # color: _HB_BOX_MAX_W + 1 while pixel + 2 black pixels for border
                if cnt_top == _HB_BOX_MAX_W + 3 and cnt_bottom == _HB_BOX_MAX_W + 3:
                    # color: _HB_BOX_H + 2 black pixels for border
                    if cnt_left == _HB_BOX_H + 2:
                        out_contours.append([(x_min, y_min), (x_max, y_max)])
                        #print('\t added')

        self.contours = out_contours

    def get_from_image(self):
        #----debug only
        #cv.imwrite('imgs\\_output1.jpg', self.img)
        #----------------

        self.get_mask()
        #----debug only
        #cv.imwrite('imgs\\_output2.jpg', self.mask)
        #----------------

        self.transform_mask()
        #----debug only
        #cv.imwrite('imgs\\_output3.jpg', self.mask)
        #----------------

        self.get_contours()
        #----debug only
        #img = copy.deepcopy(self.img)
        #for contour in self.contours:
        #    x_min = min([_contour[0][0] for _contour in contour])  # point X
        #    x_max = max([_contour[0][0] for _contour in contour])  # point X
        #    y_min = min([_contour[0][1] for _contour in contour])  # point Y
        #    y_max = max([_contour[0][1] for _contour in contour])  # point Y
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #cv.imwrite('imgs\\_output4.jpg', img)
        #----------------

        self.filter_1()
        #----debug only
        #img = copy.deepcopy(self.img)
        #for [(x_min, y_min), (x_max, y_max)] in self.contours:
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #cv.imwrite('imgs\\_output5.jpg', img)
        #----------------

        self.filter_2()
        #----debug only
        #img = copy.deepcopy(self.img)
        #for [(x_min, y_min), (x_max, y_max)] in self.contours:
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #cv.imwrite('imgs\\_output5.jpg', img)
        #----------------

        self.filter_3()
        #----debug only
        #img = copy.deepcopy(self.img)
        #for [(x_min, y_min), (x_max, y_max)] in self.contours:
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #cv.imwrite('imgs\\_output6.jpg', img)
        #----------------

        self.filter_4()
        #----debug only
        #img = copy.deepcopy(self.img)
        #for [(x_min, y_min), (x_max, y_max)] in self.contours:
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #cv.imwrite('imgs\\_output7.jpg', img)
        #----------------

    def fill_objects(self):
        self.objects = []
        for contour in self.contours:
            # health bar object
            self.objects.append(HealthBarsClass.HealthBarClass(contour))
            # health bar - fill level
            [(x_min, y_min), (x_max, y_max)] = contour
            width = x_max - x_min + 1
            self.objects[-1].fill_lvl = int((width / _HB_BOX_MAX_W) * 100)
            if self.objects[-1].fill_lvl > 100:
                self.objects[-1].fill_lvl = 100


class MinionsClass:
    def __init__(self, color):
        self.objects = []   # <list of MinionClass>
        self.img = None     # image with B,G,R,A values
        self.hsv = None     # image with H,S,V values
        self.color = color  # <str> 'red' or 'blue'
        self.health_bars = None  # <HealthBarsClass>

    class MinionClass:
        def __init__(self, img, color, health_bar):
            self.img = img
            self.color = color
            self.health_bar = health_bar  # <HealthBarClass>
            self.health = health_bar.fill_lvl  # <int> 0-100 %
            self.bounding_box = self.get_bounding_box()    # <int> [(x_min, y_min), (x_max, y_max)]
            self.point_center = self.get_point_center()    # <int> (x,y)

        def get_bounding_box(self):
            x_hb_min = self.health_bar.position[0][0]     # [(x_min, y_min), (x_max, y_max)]
            y_hb_min = self.health_bar.position[0][1]
            x_hb_max = self.health_bar.position[1][0]
            y_hb_max = self.health_bar.position[1][1]
            x_hb_middle = x_hb_min + int(_HB_BOX_MAX_W // 2)

            x_min = x_hb_middle - _MN_BOX_W
            x_max = x_hb_middle + _MN_BOX_W
            y_min = y_hb_max
            y_max = y_min + _MN_BOX_H + 30
            return [(x_min, y_min), (x_max, y_max)]
            """
            x_hb_min = self.health_bar.position[0][0]     # [(x_min, y_min), (x_max, y_max)]
            y_hb_min = self.health_bar.position[0][1]
            x_hb_middle = x_hb_min + int(_HB_BOX_MAX_W // 2) + _MN_BOX_X_SHIFT

            x_min = x_hb_middle - _MN_BOX_W // 2
            x_max = x_hb_middle + _MN_BOX_W // 2
            y_min = y_hb_min + _MN_BOX_H_SHIFT
            y_max = y_min + _MN_BOX_H
            return [(x_min, y_min), (x_max, y_max)]
            """

        def get_point_center(self):
            [(x_min, y_min), (x_max, y_max)] = self.bounding_box
            return x_min + (x_max - x_min)//2, y_min + (y_max - y_min)//2

        """ get_type: to be changed! -> cv.matchTemplate cannot be used anymore! """
        # def get_type(self):
        #     out_type = 'any'
        #     [(x_min, y_min), (x_max, y_max)] = self.position
        #     cropped_image = self.img[y_min:y_max, x_min:x_max]
        #     if self.color == 'red':
        #         hat = _MN_HAT_MAGE_RED
        #     else:   # blue
        #         hat = _MN_HAT_MAGE_BLUE
        #
        #     match_result = cv.matchTemplate(cropped_image, hat, cv.TM_CCOEFF_NORMED)  # cv.TM_SQDIFF_NORMED
        #     threshold = 0.7
        #     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)
        #     #print('Best match top left position: %s' % str(max_loc))
        #     #print('Best match confidence: %s' % max_val)
        #     if max_val >= threshold:
        #         out_type = 'mage'
        #
        #     return out_type

    def get_from_image(self, img, hsv):
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
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

    """  draw() - obsolete, handled in painter class """
    # def draw(self, in_img):
    #     for minion in self.objects:
    #         if self.color == 'red':
    #             col = (0, 0, 255)
    #         else:   # blue
    #             col = (255, 0, 0)
    #         # draw rectangles for minion's health bar
    #         #[(x_min, y_min), (x_max, y_max)] = minion.health_bar.position
    #         #cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), col, -1)
    #         #cv.rectangle(in_img, (x_min, y_min), (x_min+_HB_BOX_MAX_W, y_max), col, 2)
    #         # draw line for minion's position
    #         #cv.line(in_img, minion.point_center, (minion.point_center[0], y_max), col, 2)
    #
    #         # draw rectangle for bounding box
    #         [(x_min, y_min), (x_max, y_max)] = minion.bounding_box
    #         cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), col, 1)
    #         # draw circle for center
    #         cv.circle(in_img, (minion.point_center[0]-1, minion.point_center[1]-1), 2, col, 2)
    #         # draw rectangle + text for object type and %health
    #         cv.rectangle(in_img, (x_min, y_max), (x_min+100, y_max+14), col, -1)
    #         cv.putText(
    #             img=in_img,
    #             text='MN hp:'+str(minion.health)+'%',
    #             org=(x_min+1, y_max+12),
    #             fontFace=cv.FONT_HERSHEY_COMPLEX,
    #             fontScale=0.45,
    #             color=(0, 0, 0),
    #             thickness=1)
    #
    #     return in_img


class MinionsGhostClass:
    def __init__(self):
        self.objects = []   # <list of MinionClass>
        # ---------
        self.timeout = 8.0

    class MinionGhostClass:
        def __init__(self, in_bot, in_minion):
            self.bot_point_center_screen = in_bot.screen_pos     # <int> (x,y)
            self.bot_point_center_minimap = in_bot.minimap_pos   # <int> (x,y)
            self.point_center_screen = in_minion.screen_pos      # <int> (x,y)
            self.point_center_minimap = in_minion.minimap_pos    # <int> (x,y)
            self.timestamp = time.time()
            # ----- for painter --------------------
            self.point_center = self.point_center_screen
            self.bounding_box = self.set_bounding_box()    # <int> [(x_min, y_min), (x_max, y_max)]

        def set_bounding_box(self):
            x, y = self.point_center
            x_min = x - 60
            if x_min < 0: x_min = 0
            y_min = y - 40
            if y_min < 0: y_min = 0
            x_max = x + 60
            if x_max > _SCREEN_W: x_max = _SCREEN_W
            y_max = y + 40
            if y_max > _SCREEN_H: y_max = _SCREEN_H
            return [(x_min, y_min), (x_max, y_max)]

    def add_objects(self, in_bot, in_minions):
        for in_minion in in_minions:
            self.objects.append(self.MinionGhostClass(in_bot, in_minion))

    def update_objects(self, in_bot):
        to_delete = []
        for idx, obj in enumerate(self.objects):
            # recalculate minion's screen position in regard to current bot's position
            dist_minimap = (
                obj.point_center_minimap[0] - in_bot.minimap_pos[0],
                obj.point_center_minimap[1] - in_bot.minimap_pos[1])
            dist_screen = (round(dist_minimap[0]*16), round(dist_minimap[1]*16))
            new_point_center_screen = (
                in_bot.screen_pos[0] + dist_screen[0],
                in_bot.screen_pos[1] + dist_screen[1])

            bot_diff = (
                obj.bot_point_center_minimap[0] - in_bot.minimap_pos[0],
                obj.bot_point_center_minimap[1] - in_bot.minimap_pos[1])
            bot_diff = (bot_diff[0]*12, bot_diff[1]*12)
            new_point_center_screen = (
                obj.point_center_screen[0] + bot_diff[0],
                obj.point_center_screen[1] + bot_diff[1])


            #   update values
            obj.bot_point_center_screen = in_bot.screen_pos     # <int> (x,y)
            obj.bot_point_center_minimap = in_bot.minimap_pos   # <int> (x,y)
            obj.point_center_screen = new_point_center_screen   # <int> (x,y)
            # point_center_minimap -> the same
            obj.point_center = obj.point_center_screen
            obj.bounding_box = obj.set_bounding_box()

            # remove old ghosts
            time_diff = time.time() - obj.timestamp
            if time_diff >= self.timeout:
                to_delete.append(idx)
        if len(to_delete) > 0:
            self.objects = [obj for idx, obj in enumerate(self.objects) if idx not in to_delete]


class BotsClass:
    def __init__(self):
        self.objects = []   # <list of BotClass>
        self.img = None     # image with B,G,R,A values
        self.hsv = None     # image with H,S,V values
        self.health_bar = None  # <HealthBarsClass.HealthBarClass>

    class BotClass:
        def __init__(self, img, health_bar):
            self.img = img
            self.health_bar = health_bar  # <HealthBarsClass.HealthBarClass>
            self.health = self.health_bar.fill_lvl  # <int> 0-100 %
            self.bounding_box = self.get_bounding_box()    # <int> [(x_min, y_min), (x_max, y_max)]
            self.point_center = self.get_point_center()    # <int> (x,y)

        def get_bounding_box(self):
            x_hb_min = self.health_bar.position[0][0]  # [(x_min, y_min), (x_max, y_max)]
            y_hb_min = self.health_bar.position[0][1]
            x_hb_middle = x_hb_min + int(_BOT_HB_BOX_MAX_W // 2)

            # bot mounted
            if y_hb_min < int(2 * _SCREEN_H / 8):
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

            return [(x_min, y_min), (x_max, y_max)]

        def get_point_center(self):
            [(x_min, y_min), (x_max, y_max)] = self.bounding_box
            return x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

    def get_from_image(self, img, hsv):
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
        self.get_health_bar()
        self.fill_object()

    def get_health_bar(self):
        """ find bot's health bar, depending on it's position """
        def execute():
            hsv = self.hsv[shift_y_min:shift_y_max, shift_x_min:shift_x_max]
            mask = cv.inRange(hsv, _BOT_HB_COLOR_THR[0], _BOT_HB_COLOR_THR[1])
            contours = get_contours(mask)
            if len(contours) > 0:
                contours = filter_by_dimensions(contours)
                if len(contours) > 0:
                    set_health_bar(contours)
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
                if 8 <= height <= 11 and 2 <= width <= 11:
                    out_contours.append([(x_min, y_min), (x_max, y_max)])

            return out_contours

        def set_health_bar(in_contours):
            # health bar
            y_min = min([y_min for [(x_min, y_min), (x_max, y_max)] in in_contours])
            self.health_bar = HealthBarsClass.HealthBarClass([(shift_x_min, shift_y_min+y_min), (shift_x_max, shift_y_max+y_min)])
            # fill lvl
            x_max = max([x_max for [(x_min, y_min), (x_max, y_max)] in in_contours])
            self.health_bar.fill_lvl = int((x_max / 122) * 100)

        # ON GROUND?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 347, 905+122, 347+10
        if execute(): return

        # ON HORSE?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 262, 905+122, 262+10
        if execute(): return

        # IN BUSHES?
        shift_x_min, shift_y_min, shift_x_max, shift_y_max = 905, 150, 905+122, 450
        if execute(): return

        raise Exception(f'Bot: cannot find health bar position!')

    def fill_object(self):
        self.objects = [self.BotClass(self.img, self.health_bar)]

    """ old code """
    # def get_mask():
    #     mask = cv.inRange(self.hsv, _BOT_HB_COLOR_THR[0], _BOT_HB_COLOR_THR[1])
    #     #   transform
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv.dilate(mask, kernel, iterations=3)
    #     return mask
    #
    #
    # # filter contours: contour located outside skip areas?
    # def filter_1(contours):
    #     out_contours = []
    #     for [(x_min, y_min), (x_max, y_max)] in contours:
    #         success = True
    #         for box in _SKIP_AREAS:
    #             points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    #             polygon = [box[0], (box[1][0], box[0][1]), box[1], (box[0][0], box[1][1])]
    #             for point in points:
    #                 if point_inside_polygon(point[0], point[1], polygon):
    #                     success = False
    #                     break
    #             if not success:
    #                 break
    #         if success:
    #             out_contours.append([(x_min, y_min), (x_max, y_max)])
    #     contours = out_contours
    #
    #     if len(contours) == 0:
    #         cv.imwrite('imgs\\_0_img.jpg', self.img)
    #         cv.imwrite('imgs\\_1_mask.jpg', mask)
    #         raise Exception(f'Bot: All contours inside SKIP AREAS!')
    #
    #     return contours

    # filter contours: contour filled with health bar colors?
    # out_contours = []
    # thr_fill = 60  # %
    # cnt = 0
    # cnt_all = 0
    # for [(x_min, y_min), (x_max, y_max)] in contours:
    #     for y in range(y_min + 2, y_max):
    #         for x in range(x_min + 2, x_max):
    #             if 50 <= self.hsv[y][x][0] <= 55 and \
    #                     100 <= self.hsv[y][x][1] <= 255 and \
    #                     150 <= self.hsv[y][x][2] <= 255:
    #                 cnt += 1
    #             cnt_all += 1
    #     print(f'\t\t fill={(cnt / cnt_all) * 100}')
    #     if (cnt / cnt_all) * 100 >= thr_fill:
    #         out_contours.append([(x_min, y_min), (x_max, y_max)])
    # contours = out_contours
    # print(f'\t3: {contours}')
    # ----debug only
    #img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    #for [(x_min, y_min), (x_max, y_max)] in contours:
    #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    #cv.imwrite('imgs\\_output_4.jpg', img)
    # ----------------


    # # filter contours -> contour has border?
    # def filter_3(contours):
    #     # parameters of border
    #     b_color_thr = [np.array([0, 0, 0]), np.array([180, 255, 180])]
    #     b_min_width = 2
    #     b_height = 10
    #
    #     out_contours = []
    #     dimensions = []
    #     for [(x_min, y_min), (x_max, y_max)] in contours:
    #         b_hsv = cv.cvtColor(self.img[y_min:y_max, x_min:x_max], cv.COLOR_BGR2HSV)
    #         b_mask = cv.inRange(b_hsv, b_color_thr[0], b_color_thr[1])
    #         b_inv_mask = cv.bitwise_not(b_mask)
    #         b_contours, _ = cv.findContours(b_inv_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    #         for contour in b_contours:
    #             if len(contour) >= 4:
    #                 x_points = [value[0][0] for value in contour]
    #                 y_points = [value[0][1] for value in contour]
    #                 width = max(x_points) - min(x_points) + 1
    #                 height = max(y_points) - min(y_points) + 1
    #
    #                 cnt = 0
    #                 cnt_all = 0
    #                 # correct dimension of border
    #                 if width >= b_min_width and height == b_height:
    #                     # inside border is filled with color?
    #                     for y in range(min(y_points), max(y_points)):
    #                         for x in range(min(x_points), max(x_points)):
    #                             if _BOT_HB_COLOR_THR[0][0] <= b_hsv[y][x][0] <= _BOT_HB_COLOR_THR[1][0] and \
    #                                _BOT_HB_COLOR_THR[0][1] <= b_hsv[y][x][1] <= _BOT_HB_COLOR_THR[1][1] and \
    #                                _BOT_HB_COLOR_THR[0][2] <= b_hsv[y][x][2] <= _BOT_HB_COLOR_THR[1][2]:
    #                                 cnt += 1
    #                             cnt_all += 1
    #                     if cnt == cnt_all:
    #                         out_contours.append([(x_min, y_min), (x_max, y_max)])
    #                     break
    #                 dimensions.append((height, width, (cnt, cnt_all)))
    #
    #     contours = out_contours
    #     if len(contours) == 0:
    #         cv.imwrite('imgs\\_0_img.jpg', self.img)
    #         cv.imwrite('imgs\\_1_mask.jpg', mask)
    #         raise Exception(f'Bot: Incorrect dimensions of border contours! dimensions = {dimensions}')
    #
    #     return contours
    #
    # # filter contours -> only one contour?
    # def filter_4(contours):
    #     if len(contours) > 1:
    #         for [(x_min, y_min), (x_max, y_max)] in contours:
    #             cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    #         cv.imwrite('imgs\\_0_img.jpg', self.img)
    #         cv.imwrite('imgs\\_1_mask.jpg', mask)
    #         cv.imwrite('imgs\\_2_contours.jpg', self.img)
    #         raise Exception(f'Bot: Too many contours! contours= {contours}')
    #
    #     return contours
    #
    # self.health_bar = None
    #
    # mask = get_mask()
    # contours = get_contours()
    # contours = filter_1(contours)
    # contours = filter_2(contours)
    # contours = filter_3(contours)
    # contours = filter_4(contours)
    #
    # # only one contour -> add health bar
    # self.health_bar = HealthBarsClass.HealthBarClass(contours[0])
    # [(x_min, y_min), (x_max, y_max)] = self.health_bar.position
    # width = x_max - x_min + 1
    # self.health_bar.fill_lvl = int((width / _BOT_HB_BOX_W) * 100)
    # if self.health_bar.fill_lvl > 100:
    #     self.health_bar.fill_lvl = 100

    """ get_circle() is not reliable! can be covered by other objects """
    # def get_circle(self):
    #     self.circle_x, self.circle_y = None, None
    #     #   mask
    #     mask = cv.inRange(self.hsv, _BOT_CIRCLE_COLOR_THR[0], _BOT_CIRCLE_COLOR_THR[1])
    #     #   transform
    #     kernel = np.ones((4, 4), np.uint8)
    #     result = cv.dilate(mask, kernel, iterations=1)
    #     #   circle
    #     #       param1 - sensitivity
    #     #       param2 - minimal number of edges
    #     circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=40, param2=10, minRadius=20, maxRadius=200)
    #     if circles is not None:
    #         position = np.uint16(np.around(circles[0]))[0]
    #         self.circle_x, self.circle_y, self.circle_r = position[0], position[1], position[2]
    #     else:
    #         print('BOT: circle not found!')

    """  draw() - obsolete, handled in painter class """
    # def draw(self, in_img):
    #     if None not in [self.health_bar, self.bounding_box, self.point_center]:
    #         # draw rectangles for health bar
    #         #[(x_min, y_min), (x_max, y_max)] = self.health_bar.position
    #         #cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
    #         #cv.rectangle(in_img, (x_min, y_min), (x_min+_BOT_HB_BOX_MAX_W, y_max), (0, 255, 255), 1)
    #         # draw line for center position
    #         #cv.line(in_img, self.point_center, (self.point_center[0], y_max), (0, 255, 255), 2)
    #
    #         # draw rectangle for bounding box
    #         [(x_min, y_min), (x_max, y_max)] = self.bounding_box
    #         cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
    #
    #         # draw rectangle + text for object type and %health
    #         cv.rectangle(in_img, (x_min, y_max), (x_min+100, y_max+14), (0, 255, 255), -1)
    #         cv.putText(
    #             img=in_img,
    #             text='BOT hp:'+str(self.health)+'%',
    #             org=(x_min+1, y_max+12),
    #             fontFace=cv.FONT_HERSHEY_COMPLEX,
    #             fontScale=0.45,
    #             color=(0, 0, 0),
    #             thickness=1)
    #
    #         return in_img
    #     else:
    #         print('BOT: nothing to draw')
    #         return None


class ManasClass:
    """
    if minion dies, in his bounding box there could be MANA sphere (or XP sphere)
    for red minion: 4 sec mana blue -> 4 sec mana pink -> disappear
    for blue minion: 4 sec mana red -> 4 sec mana pink -> disappear

    Note: bot can collect blue and pink spheres, not red ones!
    """

    def __init__(self):
        self.objects = []   # <list of ManaClass>
        self.img = None     # image with B,G,R,A values
        self.hsv = None     # image with H,S,V values
        self.boxes = None   # list of <list of int> [[(x_min, y_min), (x_max, y_max)], [(x_min, y_min), (x_max, y_max)], ...]

        self.color_thr = [np.array([137, 142, 158]), np.array([165, 255, 255])]

    class ManaClass:
        def __init__(self, point_center, bounding_box):
            self.point_center = point_center    # <int> (x,y)
            self.bounding_box = bounding_box    # <int> [(x_min, y_min), (x_max, y_max)]

    def get_from_image(self, img, hsv, bounding_box):
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
        self.get_boxes(bounding_box)
        self.fill_objects()

    def get_boxes(self, bounding_box):
        self.boxes = []
        [(x_min, y_min), (x_max, y_max)] = bounding_box
        mask = cv.inRange(self.hsv[y_min:y_max, x_min:x_max], self.color_thr[0], self.color_thr[1])
        y, x = np.where(mask == 255)
        if x.size != 0 or y.size != 0:
            x_min_new, y_min_new, x_max_new, y_max_new = x.min(), y.min(), x.max(), y.max()
            area = (y_max_new - y_min_new + 1) * (x_max_new - x_min_new + 1)
            if (float(y.size) / float(area)) * 100 > 30:    # at least 30%
                self.boxes.append([(x_min_new, y_min_new), (x_max_new, y_max_new)])  # <int> [(x_min, y_min), (x_max, y_max)]

            name1=f'imgs\\frames\\{"%.2f" % (time.time())}_{(x_min,y_min)}_img.jpg'
            name2=f'imgs\\frames\\{"%.2f" % (time.time())}_{(x_min,y_min)}_mask.jpg'
            cv.imwrite(name1, self.img[y_min:y_max, x_min:x_max])
            cv.imwrite(name2, mask)
            print(f'mana_area = {(float(y.size) / float(area)) * 100}%')
    """
    def get_boxes(self, in_minions_ghost):
        self.boxes = []
        for obj in in_minions_ghost.objects:
            [(x_min, y_min), (x_max, y_max)] = obj.bounding_box
            mask = cv.inRange(self.hsv[y_min:y_max, x_min:x_max], self.color_thr[0], self.color_thr[1])
            y, x = np.where(mask == 255)
            if x.size != 0 or y.size != 0:
                x_min_new, y_min_new, x_max_new, y_max_new = x.min(), y.min(), x.max(), y.max()
                area = (y_max_new - y_min_new + 1) * (x_max_new - x_min_new + 1)
                if (float(y.size) / float(area)) * 100 > 30:    # at least 30%
                    self.boxes.append([(x_min_new, y_min_new), (x_max_new, y_max_new)])  # <int> [(x_min, y_min), (x_max, y_max)]

                name1=f'imgs\\frames\\{"%.2f" % (time.time())}_{(x_min,y_min)}_img.jpg'
                name2=f'imgs\\frames\\{"%.2f" % (time.time())}_{(x_min,y_min)}_mask.jpg'
                cv.imwrite(name1, self.img[y_min:y_max, x_min:x_max])
                cv.imwrite(name2, mask)
                print(f'mana_area = {(float(y.size) / float(area)) * 100}%')
    """

    def fill_objects(self):
        self.objects = []
        for [(x_min, y_min), (x_max, y_max)] in self.boxes:
            x, y = ((x_max - x_min + 1) // 2, (y_max - y_min + 1) // 2)  # <int> (x,y)
            point_center = (x, y)
            bounding_box = [(x_min, y_min), (x_max, y_max)]
            self.objects.append(self.ManaClass(point_center, bounding_box))

    """ old code """
    # def get_circles(self):
    #     for [(box_x_min, box_y_min), (box_x_max, box_y_max)] in self.boxes:
    #         # crop main image to box size
    #         img = self.img[box_y_min:box_y_max, box_x_min:box_x_max]
    #         hsv = self.hsv[box_y_min:box_y_max, box_x_min:box_x_max]
    #
    #
    #         t1 = np.array([80, 80, 120])  # 90,  110, 210
    #         t2 = np.array([150, 220, 255])  # 120, 200, 255
    #         color_thresholds = [t1, t2]
    #         mask = cv.inRange(hsv, color_thresholds[0], color_thresholds[1])
    #         cv.imwrite(f"imgs\\_output_1.jpg", mask)
    #
    #         # kernel = np.ones((3, 3), np.uint8)
    #         # result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    #         # result = cv.dilate(mask, kernel, iterations=1)
    #         # save_image(result, f"imgs\\_output_2.jpg")
    #         result = mask
    #
    #         # minDist - minimum distance between the centers of the detected circles
    #         # param1 - sensitivity
    #         # param2 - minimal number of edges
    #         #img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    #         circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=11, param1=300, param2=10, minRadius=14, maxRadius=20)
    #         out_circles = []
    #         if circles is not None:
    #             circles = np.uint16(np.around(circles))
    #             for i in circles[0, :]:
    #                 # draw the outer circle
    #                 cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
    #                 # draw the center of the circle
    #                 cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
    #                 out_circles.append([i[0], i[1], i[2]])  # circle_x, circle_y, circle_r
    #             circles = out_circles
    #             cv.imwrite(f'imgs\\_output_2.jpg', img)
    #
    #         # circle fill lvl?
    #         img = deepcopy(self.img[box_y_min:box_y_max, box_x_min:box_x_max])
    #         out_circles = []
    #         if circles is not None:
    #             for [circle_x, circle_y, circle_r] in circles:
    #                 # adjust values to fit image box dimensions
    #                 if int(circle_x)-int(circle_r) < 0:
    #                     circle_r = circle_x
    #                 if int(circle_y)-int(circle_r) < 0:
    #                     circle_r = circle_y
    #                 if int(circle_x)+int(circle_r) > img.shape[1]:  # width
    #                     circle_r = img.shape[1] - circle_x
    #                 if int(circle_y)+int(circle_r) > img.shape[0]:  # high
    #                     circle_r = img.shape[0] - circle_y
    #
    #                 circle_mask = np.zeros((2 * circle_r, 2 * circle_r), dtype="uint8")  # tuple(size_y, size_x)
    #                 x_min, y_min = circle_x - circle_r, circle_y - circle_r
    #                 x_max = x_min + circle_mask.shape[1]  # width
    #                 y_max = y_min + circle_mask.shape[0]  # high
    #
    #                 cv.circle(circle_mask, (circle_r, circle_r), circle_r, 255, -1)
    #                 cv.imwrite(f'imgs\\_output_3.jpg', circle_mask)
    #                 img_cropped = img[y_min:y_max, x_min:x_max]
    #                 cv.imwrite(f'imgs\\_output_4.jpg', img_cropped)
    #                 img_masked = cv.bitwise_and(img_cropped, img_cropped, mask=circle_mask)
    #                 cv.imwrite(f'imgs\\_output_5.jpg', img_masked)
    #                 hsv_cropped = cv.cvtColor(img_masked, cv.COLOR_BGR2HSV)
    #
    #                 thr_fill = 40  # %
    #                 cnt = 0
    #                 for y in range(0, hsv_cropped.shape[0]):
    #                     for x in range(0, hsv_cropped.shape[1]):
    #                         h, s, v = hsv_cropped[y][x]
    #                         if color_thresholds[0][0] <= h <= color_thresholds[1][0] and \
    #                                 color_thresholds[0][1] <= s <= color_thresholds[1][1] and \
    #                                 color_thresholds[0][2] <= v <= color_thresholds[1][2]:
    #                             cnt += 1
    #
    #                 # check
    #                 circle_area = math.pi * circle_r * circle_r
    #                 if (cnt / circle_area) * 100 >= thr_fill:
    #                     self.circles.append([box_x_min + circle_x,
    #                                          box_y_min + circle_y,
    #                                          circle_r])
    #                     # draw the outer circle
    #                     #cv.circle(img, (circle_x, circle_y), circle_r, (0, 255, 0), 1)
    #                     # draw the center of the circle
    #                     #cv.circle(img, (circle_x, circle_y), 2, (0, 0, 255), 2)
    #
    # def fill_objects(self):
    #     self.objects = []
    #     for x, y, r in self.circles:
    #         self.objects.append(self.ManaClass(self.color, x, y, r))

    """ old code - main.py """
    # f_path = "imgs\\test_images\\collectibles_1.jpg"
    # # f_path="imgs\\test_images\\collectibles_1_2.jpg"
    # # f_path="imgs\\test_images\\collectibles_3_1.jpg"
    # img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #
    #
    # t1 = np.array([80, 80, 120])  # 90,  110, 210
    # t2 = np.array([150, 220, 255])  # 120, 200, 255
    # color_thresholds = [t1, t2]
    # mask = cv.inRange(hsv, color_thresholds[0], color_thresholds[1])
    # cv.imwrite(mask, f"imgs\\_output_1.jpg")
    #
    # # kernel = np.ones((3, 3), np.uint8)
    # # result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # # result = cv.dilate(mask, kernel, iterations=1)
    # # save_image(result, f"imgs\\_output_2.jpg")
    # result = mask
    #
    # # minDist - minimum distance between the centers of the detected circles
    # # param1 - sensitivity
    # # param2 - minimal number of edges
    # img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    # circle_x, circle_y, circle_r = None, None, None
    # circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=11, param1=300, param2=10, minRadius=14,
    #                           maxRadius=20)
    # out_circles = []
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # draw the outer circle
    #         cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
    #         # draw the center of the circle
    #         cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
    #         out_circles.append([i[0], i[1], i[2]])  # circle_x, circle_y, circle_r
    #     circles = out_circles
    #     cv.imwrite(img, f"imgs\\_output_2.jpg")
    #
    # # circle fill lvl?
    # img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    # out_circles = []
    # for [circle_x, circle_y, circle_r] in circles:
    #     x_min, y_min = circle_x - circle_r, circle_y - circle_r
    #     x_max, y_max = circle_x + circle_r, circle_y + circle_r
    #
    #     circle_mask = np.zeros((2 * circle_r + 1, 2 * circle_r + 1), dtype="uint8")  # tuple(size_y, size_x)
    #     cv.circle(circle_mask, (circle_r, circle_r), circle_r, 255, -1)
    #     # save_image(circle_mask, f'imgs\\_output_3.jpg')
    #     img_cropped = img[y_min:y_max + 1, x_min:x_max + 1]
    #     # save_image(img_cropped, f'imgs\\_output_4.jpg')
    #     img_masked = cv.bitwise_and(img_cropped, img_cropped, mask=circle_mask)
    #     # save_image(img_masked, f'imgs\\_output_5.jpg')
    #     hsv_cropped = cv.cvtColor(img_masked, cv.COLOR_BGR2HSV)
    #
    #     thr_fill = 40  # %
    #     cnt = 0
    #     for y in range(0, hsv_cropped.shape[0]):
    #         for x in range(0, hsv_cropped.shape[1]):
    #             h, s, v = hsv_cropped[y][x]
    #             if color_thresholds[0][0] <= h <= color_thresholds[1][0] and \
    #                     color_thresholds[0][1] <= s <= color_thresholds[1][1] and \
    #                     color_thresholds[0][2] <= v <= color_thresholds[1][2]:
    #                 cnt += 1
    #
    #     # check
    #     circle_area = math.pi * circle_r * circle_r
    #     if (cnt / circle_area) * 100 >= thr_fill:
    #         out_circles.append([circle_x, circle_y, circle_r])
    #         # draw the outer circle
    #         cv.circle(img, (circle_x, circle_y), circle_r, (0, 255, 0), 1)
    #         # draw the center of the circle
    #         cv.circle(img, (circle_x, circle_y), 2, (0, 0, 255), 2)
    #
    #     print(f'circle: ({circle_x},{circle_y}) r={circle_r}, area={(cnt / circle_area) * 100}')
    # cv.imwrite(img, f'imgs\\_output_3.jpg')


class MapClass:
    def __init__(self):
        self.objects = {
            'bot': (None, None),  # (x,y)
            'minions_blue': [],  # [(x, y), ...],
            'minions_red': [],  # [(x, y), ...],
            'well': (None, None),  # (x,y)
            'gate': [],  # [(x, y), ...],
            'bush': (None, None),  # (x,y)
        }
        self.img = None  # image with B,G,R,A values
        self.hsv = None  # image with H,S,V values
        # ------------------------------------------ #
        self.bot_x_shift = 11
        self.bot_y_shift = 16
        self.bot_mask = cv.imread("imgs\\test_images\\bot_mask.png", cv.IMREAD_GRAYSCALE)
        self.bot_mask_sum = int(np.sum(self.bot_mask))
        self.bot_mask_inv = cv.bitwise_not(self.bot_mask)
        self.bot_thr = [np.array([0, 0, 125]), np.array([255, 255, 255])]
        self.minimap_img = cv.imread("imgs\\test_images\\minimap.jpg", cv.IMREAD_UNCHANGED)
        self.minimap_img = cv.cvtColor(self.minimap_img, cv.COLOR_RGB2RGBA)

    def get_from_image(self, img, hsv, in_bot, in_minions_blue, in_minions_red):

        def get_bot():
            def crop_image():
                # crop image
                [(x_min, y_min), (x_max, y_max)] = _SKIP_AREAS[3]
                x_min = x_min - self.bot_mask.shape[1]
                if x_min < 0: x_min = 0
                y_min = y_min - self.bot_mask.shape[0]
                if y_min < 0: y_min = 0
                x_max = x_max + self.bot_mask.shape[1]
                if x_max > self.img.shape[1]: x_max = self.img.shape[1]
                y_max = y_max + self.bot_mask.shape[0]
                if y_max > self.img.shape[0]: y_max = self.img.shape[0]
                out_x = x_min
                out_y = y_min
                img = self.img[y_min:y_max, x_min:x_max]
                hsv = self.hsv[y_min:y_max, x_min:x_max]
                return img, hsv, out_x, out_y

            img, hsv, out_x, out_y = crop_image()

            # check if color-based detection is possible
            mask = cv.inRange(hsv, _BOT_HB_COLOR_THR[0], _BOT_HB_COLOR_THR[1])
            y, x = np.where(mask == 255)
            if x.size != 0 or y.size != 0:
                # narrow down region of interest based on color threshold
                x_min_new = x.min() - self.bot_mask.shape[1] - 5
                if x_min_new < 0: x_min_new = 0
                y_min_new = y.min() - self.bot_mask.shape[0] - 5
                if y_min_new < 0: y_min_new = 0
                x_max_new = x.max() + self.bot_mask.shape[1] + 5
                if x_max_new > self.img.shape[1]: x_max_new = self.img.shape[1]
                y_max_new = y.max() + self.bot_mask.shape[0] + 5
                if y_max_new > self.img.shape[0]: y_max_new = self.img.shape[0]
                out_x += x_min_new
                out_y += y_min_new
                mask = cv.inRange(
                    self.hsv[out_y:out_y + y_max_new - y_min_new + 1, out_x:out_x + x_max_new - x_min_new + 1],
                    self.bot_thr[0], self.bot_thr[1])
                img = self.img[out_y:out_y + y_max_new - y_min_new + 1, out_x:out_x + x_max_new - x_min_new + 1]

                # find matching pattern
                tmp_x, tmp_y, tmp_sum = None, None, 45  # at least 45% needs to match pattern
                for y in range(0, mask.shape[0] - self.bot_mask.shape[0] + 1):
                    for x in range(0, mask.shape[1] - self.bot_mask.shape[1] + 1):
                        cropped = mask[y:y + self.bot_mask.shape[0], x:x + self.bot_mask.shape[1]]
                        compare = cv.bitwise_and(self.bot_mask, cropped)
                        _sum_valid = int(np.sum(compare))
                        # compare = cv.bitwise_and(self.bot_mask_inv, cropped)
                        # _sum_invalid = int(np.sum(compare))
                        _sum_invalid = 0
                        _sum = ((_sum_valid - _sum_invalid) / self.bot_mask_sum) * 100
                        if _sum > tmp_sum:
                            tmp_x, tmp_y, tmp_sum = x + self.bot_x_shift, y + self.bot_y_shift, _sum

                if None not in [tmp_x, tmp_y]:
                    out_x += tmp_x
                    out_y += tmp_y
                    self.objects['bot'] = (out_x, out_y)
                else:
                    cv.imwrite(f'imgs\\_1_img.jpg', img)
                    cv.imwrite(f'imgs\\_2_img.jpg', mask)
                    raise Exception(f'Minimap:Bot: Cannot find bot!')

            # try to detect based on previous position
            else:
                out_x, out_y = get_bot_from_prev_pos()
                if None not in [out_x, out_y]:
                    self.objects['bot'] = (out_x, out_y)
                else:
                    raise Exception(f'Minimap:Bot: Cannot find bot!')

        def get_bot_from_prev_pos():
            out_x, out_y = None, None

            # get last bot's position
            bot_x, bot_y = self.objects['bot']
            if None in [bot_x, bot_y]:
                return out_x, out_y

            # set region of interest
            x_min = bot_x - self.bot_mask.shape[1] - 5
            if x_min < 0: x_min = 0
            y_min = bot_y - self.bot_mask.shape[0] - 5
            if y_min < 0: y_min = 0
            x_max = bot_x + self.bot_mask.shape[1] + 5
            if x_max > self.img.shape[1]: x_max = self.img.shape[1]
            y_max = bot_y + self.bot_mask.shape[0] + 5
            if y_max > self.img.shape[0]: y_max = self.img.shape[0]
            out_x = x_min
            out_y = y_min
            hsv = self.hsv[y_min:y_max, x_min:x_max]
            mask = cv.inRange(hsv, self.bot_thr[0], self.bot_thr[1])
            img = self.img[y_min:y_max, x_min:x_max]

            # find matching pattern
            tmp_x, tmp_y, tmp_sum = None, None, 45  # at least 45% needs to match pattern
            for y in range(0, mask.shape[0] - self.bot_mask.shape[0] + 1):
                for x in range(0, mask.shape[1] - self.bot_mask.shape[1] + 1):
                    cropped = mask[y:y + self.bot_mask.shape[0], x:x + self.bot_mask.shape[1]]
                    compare = cv.bitwise_and(self.bot_mask, cropped)
                    _sum_valid = int(np.sum(compare))
                    # compare = cv.bitwise_and(self.bot_mask_inv, cropped)
                    # _sum_invalid = int(np.sum(compare))
                    _sum_invalid = 0
                    _sum = ((_sum_valid - _sum_invalid) / self.bot_mask_sum) * 100
                    if _sum > tmp_sum:
                        tmp_x, tmp_y, tmp_sum = x + self.bot_x_shift, y + self.bot_y_shift, _sum

            if None not in [tmp_x, tmp_y]:
                out_x += tmp_x
                out_y += tmp_y
            else:
                out_x, out_y = None, None
                cv.imwrite(f'imgs\\_1_img.jpg', img)
                cv.imwrite(f'imgs\\_2_img.jpg', mask)

            return out_x, out_y

        def set_minions():
            bot_x, bot_y = in_bot.objects[0].point_center
            out = []
            for obj in in_minions_blue.objects:
                x, y = obj.point_center
                out.append((self.objects['bot'][0] + round((x - bot_x) / 16),
                            self.objects['bot'][1] + round((y - bot_y) / 16)))
            self.objects['minions_blue'] = out
            out = []
            for obj in in_minions_red.objects:
                x, y = obj.point_center
                out.append((self.objects['bot'][0] + round((x - bot_x) / 16),
                            self.objects['bot'][1] + round((y - bot_y) / 16)))
            self.objects['minions_red'] = out

        self.img = img  # image with B,G,R,A values
        self.hsv = hsv  # image with H,S,V values

        get_bot()
        set_minions()

    # map's bot detection - obsolete
    """
    def get_first_bot(self):
        out_x, out_y = None, None

        # crop image
        [(x_min, y_min), (x_max, y_max)] = _SKIP_AREAS[3]
        x_min = x_min - self.bot_mask.shape[1]
        if x_min < 0: x_min = 0
        y_min = y_min - self.bot_mask.shape[0]
        if y_min < 0: y_min = 0
        x_max = x_max + self.bot_mask.shape[1]
        if x_max > self.img.shape[1]: x_max = self.img.shape[1]
        y_max = y_max + self.bot_mask.shape[0]
        if y_max > self.img.shape[0]: y_max = self.img.shape[0]
        out_x = x_min
        out_y = y_min
        img = self.img[y_min:y_max, x_min:x_max]
        hsv = self.hsv[y_min:y_max, x_min:x_max]
        mask = cv.inRange(hsv, self.bot_thr[0], self.bot_thr[1])

        # find matching pattern
        tmp_x, tmp_y, tmp_sum = None, None, 45  # at least 45% needs to match pattern
        for y in range(img.shape[0] - self.bot_mask.shape[0]):
            for x in range(img.shape[1] - self.bot_mask.shape[1]):
                cropped = mask[y:y + self.bot_mask.shape[0], x:x + self.bot_mask.shape[1]]
                compare = cv.bitwise_and(self.bot_mask, cropped)
                _sum_valid = int(np.sum(compare))
                compare = cv.bitwise_and(self.bot_mask_inv, cropped)
                _sum_invalid = int(np.sum(compare))
                _sum = ((_sum_valid - _sum_invalid) / self.bot_mask_sum) * 100
                if _sum > tmp_sum:
                    tmp_x, tmp_y, tmp_sum = x + self.bot_x_shift, y + self.bot_y_shift, _sum

        if None not in [tmp_x, tmp_y]:
            out_x += tmp_x
            out_y += tmp_y
        else:
            out_x, out_y = None, None

        #cv.rectangle(self.img, (out_x, out_y), (out_x+1, out_y+1), (0, 0, 255), -1)
        #f_path = f'imgs\\frames\\__frame_0.jpg'
        #cv.imwrite(f_path, self.img)
        return out_x, out_y

    def get_next_bot(self):
        out_x, out_y = None, None

        # get last bot's position
        bot_x, bot_y = self.objects['bot']

        # set region of interest
        x_min = bot_x - 2*self.bot_mask.shape[1]
        if x_min < 0: x_min = 0
        y_min = bot_y - 2*self.bot_mask.shape[0]
        if y_min < 0: y_min = 0
        x_max = bot_x + 2*self.bot_mask.shape[1]
        if x_max > self.img.shape[1]: x_max = self.img.shape[1]
        y_max = bot_y + 2*self.bot_mask.shape[0]
        if y_max > self.img.shape[0]: y_max = self.img.shape[0]
        out_x = x_min
        out_y = y_min
        hsv = self.hsv[y_min:y_max, x_min:x_max]
        mask = cv.inRange(hsv, self.bot_thr[0], self.bot_thr[1])

        # narrow down region of interest using circle detection
        circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=300, param2=5, minRadius=9, maxRadius=12)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            xs, ys, r = circles[0][0]
            x_min_new = xs - r - 5
            if x_min_new < 0: x_min_new = 0
            y_min_new = ys - r - 5
            if y_min_new < 0: y_min_new = 0
            x_max_new = xs + r + 5
            if x_max_new > mask.shape[1]: x_max_new = mask.shape[1]
            y_max_new = ys + r + 5
            if y_max_new > mask.shape[0]: y_max_new = mask.shape[0]
            mask = mask[y_min_new:y_max_new, x_min_new:x_max_new]
            out_x += x_min_new
            out_y += y_min_new

        # find matching pattern
        tmp_x, tmp_y, tmp_sum = None, None, 45  # at least 45% needs to match pattern
        for y in range(0, mask.shape[0]-self.bot_mask.shape[0]+1):
            for x in range(0, mask.shape[1]-self.bot_mask.shape[1]+1):
                cropped = mask[y:y + self.bot_mask.shape[0], x:x + self.bot_mask.shape[1]]
                compare = cv.bitwise_and(self.bot_mask, cropped)
                _sum_valid = int(np.sum(compare))
                #compare = cv.bitwise_and(self.bot_mask_inv, cropped)
                #_sum_invalid = int(np.sum(compare))
                _sum_invalid = 0
                _sum = ((_sum_valid - _sum_invalid) / self.bot_mask_sum) * 100
                if _sum > tmp_sum:
                    tmp_x, tmp_y, tmp_sum = x + self.bot_x_shift, y + self.bot_y_shift, _sum

        if None not in [tmp_x, tmp_y]:
            out_x += tmp_x
            out_y += tmp_y
        else:
            out_x, out_y = None, None

        #cv.rectangle(self.img, (out_x, out_y), (out_x+1, out_y+1), (0, 0, 255), -1)
        #f_path = f'imgs\\frames\\__frame_1.jpg'
        #cv.imwrite(f_path, self.img)
        return out_x, out_y
    """

    # map's bot detection - used in main.py
    """
    import time
    from global_vars import _SKIP_AREA_BOTTOM_RIGHT

    x_shift = 11
    y_shift = 16
    bot_mask = cv.imread("imgs\\test_images\\bot_mask.png", cv.IMREAD_GRAYSCALE)
    sum_bot_mask = int(np.sum(bot_mask))
    bot_mask_inv = cv.bitwise_not(bot_mask)
    thr = [np.array([0, 0, 125]), np.array([255, 255, 255])]

    t1 = time.time_ns()

    # search bot's first occurrence
    i = 0
    f_path = f'imgs\\frames\\frame_{"%03d" % i}.jpg'
    img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, thr[0], thr[1])

    out_x, out_y, out_sum = None, None, 45
    for y in range(img.shape[0]-bot_mask.shape[0]):
        for x in range(img.shape[1]-bot_mask.shape[1]):
            cropped = mask[y:y+bot_mask.shape[0], x:x+bot_mask.shape[1]]
            compare = cv.bitwise_and(bot_mask, cropped)
            _sum_valid = int(np.sum(compare))
            compare = cv.bitwise_and(bot_mask_inv, cropped)
            _sum_invalid = int(np.sum(compare))

            _sum = ((_sum_valid-_sum_invalid) / sum_bot_mask) * 100

            if _sum > out_sum:
                out_x, out_y, out_sum = x+x_shift, y+y_shift, _sum
    print(f'frame_{"%03d" % i}, x={out_x}, y={out_y}, sum={"%.2f" % out_sum} %')
    t2 = time.time_ns()

    # search bot's next occurrences
    for i in range(1, 96):
        f_path = f'imgs\\frames\\frame_{"%03d" % i}.jpg'
        img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, thr[0], thr[1])

        # define region of interest
        if out_y - bot_mask.shape[0] < 0:
            y_min = 0
        else:
            y_min = out_y - bot_mask.shape[0]
        if out_y + bot_mask.shape[0] > img.shape[0]:
            y_max = img.shape[0]
        else:
            y_max = out_y + bot_mask.shape[0]
        if out_x - bot_mask.shape[1] < 0:
            x_min = 0
        else:
            x_min = out_x - bot_mask.shape[1]
        if out_x + bot_mask.shape[1] > img.shape[1]:
            x_max = img.shape[1]
        else:
            x_max = out_x + bot_mask.shape[1]

        #contours, hierarchy = cv.findContours(mask[y_min:y_max, x_min:x_max], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(mask, contours, -1, (0, 255, 0), 1)
        #cv.imwrite(f'imgs\\frames\\__frame_{"%03d" % i}.jpg', mask)

        # narrow down region of interest
        circles = cv.HoughCircles(mask[y_min:y_max, x_min:x_max], cv.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=300, param2=5, minRadius=9, maxRadius=12)
        out_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            xs, ys, r = circles[0][0]
            x_min_new = x_min + xs - r - 5
            y_min_new = y_min + ys - r - 5
            x_max_new = x_min + xs + r + 5
            y_max_new = y_min + ys + r + 5
            if x_min_new < x_min:
                x_min_new = x_min
            if y_min_new < y_min:
                y_min_new = y_min
            if x_max_new > x_max:
                x_max_new = x_max
            if y_max_new > y_max:
                y_max_new = y_max
            x_min, y_min, x_max, y_max = x_min_new, y_min_new, x_max_new, y_max_new
            #for i in circles[0, :]:
                # draw the outer circle
                #cv.circle(img, (x_min+i[0], y_min+i[1]), i[2], (255, 0, 0), 1)
                # draw the center of the circle
                #cv.circle(img, (x_min+i[0], y_min+i[1]), 2, (255, 0, 0), 2)
            #cv.imwrite(f'imgs\\frames\\__frame_1.jpg', img)

        # find matching pattern
        out_x, out_y, out_sum = None, None, 45
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                cropped = mask[y:y+bot_mask.shape[0], x:x+bot_mask.shape[1]]
                compare = cv.bitwise_and(bot_mask, cropped)
                _sum_valid = int(np.sum(compare))
                #compare = cv.bitwise_and(bot_mask_inv, cropped)
                #_sum_invalid = int(np.sum(compare))
                _sum_invalid = 0

                _sum = ((_sum_valid-_sum_invalid) / sum_bot_mask) * 100
                #print(f'y={y}, x={x}, _sum={"%.2f" % _sum} %')

                if _sum > out_sum:
                    out_x, out_y, out_sum = x+x_shift, y+y_shift, _sum

        print(f'frame_{"%03d" % i}, x={out_x}, y={out_y}, sum={"%.2f" % out_sum} %')
        #cv.rectangle(img, (out_x, out_y), (out_x+1, out_y+1), (0, 0, 255), -1)
        #f_path = f'imgs\\frames\\_frame_{"%03d" % i}.jpg'
        #cv.imwrite(f_path, img)

    print('----------------------')
    print(f't_all={((time.time_ns() - t1)/1000000):.2f} ms')
    print(f't_1_2={((t2 - t1)/1000000):.2f} ms')
    print(f't_one_frame={((time.time_ns() - t2)/1000000/95):.2f} ms')
    """




class WellsClass:
    def __init__(self, img, hsv):
        self.objects = []   # <list of WellClass>
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
        self.health_bar = None  # <HealthBarsClass.HealthBarClass>

    class WellClass:
        def __init__(self, img, health_bar):
            self.img = img
            self.health_bar = health_bar  # <HealthBarsClass.HealthBarClass>
            # same as health_bar
            self.health = self.get_health()  # <int> 0-100 %
            self.bounding_box = self.get_bounding_box()    # <int> [(x_min, y_min), (x_max, y_max)]
            self.point_center = self.get_point_center()    # (x,y)

        def get_health(self):
            if self.health_bar is not None:
                return self.health_bar.fill_lvl
            else:
                print('Well: health not found!')
                return None

        def get_bounding_box(self):
            if self.health_bar is not None:
                return self.health_bar.position
            else:
                print('Well: bounding box not found!')
                return None

        def get_point_center(self):
            if self.bounding_box is not None:
                [(x_min, y_min), (x_max, y_max)] = self.bounding_box
                return x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
            else:
                print('Well: no point center')
                return None

    def get_from_image(self):
        self.get_health_bar()
        self.fill_object()

    def get_health_bar(self):
        self.health_bar = None
        # mask
        mask = cv.inRange(self.hsv, _WELL_HB_COLOR_THR[0], _WELL_HB_COLOR_THR[1])
        # ----debug only
        #cv.imwrite(f'imgs\\_output_1.jpg', mask)
        #----------------

        # transform
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # ----debug only
        #cv.imwrite(f'imgs\\_output_2.jpg', mask)
        #----------------

        # contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #   filter contours with 4 points (rectangle)
        out_contours = []
        for contour in contours:
            if len(contour) == 4:
                x_min = min([_contour[0][0] for _contour in contour])  # point X
                x_max = max([_contour[0][0] for _contour in contour])  # point X
                y_min = min([_contour[0][1] for _contour in contour])  # point Y
                y_max = max([_contour[0][1] for _contour in contour])  # point Y
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        contours = out_contours
        # ----debug only
        #img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
        #for contour in contours:
        #    (x_min, y_min), (x_max, y_max) = contour
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        #cv.imwrite(f'imgs\\_output_3.jpg', img)
        #----------------

        # filter contours with correct dimensions
        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in contours:
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            if height == _WELL_HB_BOX_H and width == _WELL_HB_BOX_W:
                out_contours.append([(x_min, y_min), (x_max, y_max)])
        contours = out_contours
        # ----debug only
        #img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
        #for contour in contours:
        #    (x_min, y_min), (x_max, y_max) = contour
        #    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        #cv.imwrite(f'imgs\\_output_4.jpg', img)
        #----------------

        #   if there is one contour -> add health bar
        if len(contours) == 1:
            self.health_bar = HealthBarsClass.HealthBarClass(contours[0])
            self.health_bar.fill_lvl = 100
        else:
            print('Well: health bar not found!')

    def fill_object(self):
        self.objects = []
        if self.health_bar is not None:
            self.objects = [self.WellClass(self.img, self.health_bar)]
        else:
            print('Well: health bar not found!')




"""
from global_vars import _SKIP_AREA_BOTTOM_RIGHT
f_path = "imgs\\input_0.jpg"
img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
[(x_min, y_min), (x_max, y_max)] = _SKIP_AREA_BOTTOM_RIGHT
img = img[y_min:y_max, x_min:x_max]
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

thr = [np.array([34, 70, 125]), np.array([100, 255, 255])]
cv.imwrite('imgs\\result1.jpg', img)
mask = cv.inRange(hsv, thr[0], thr[1])
cv.imwrite('imgs\\result2.jpg', mask)
#kernel = np.ones((2, 2), np.uint8)
#result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
#cv.imwrite('imgs\\result3.jpg', result)
result = mask

# minDist - minimum distance between the centers of the detected circles
# param1 - sensitivity
# param2 - minimal number of edges
img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
[(x_min, y_min), (x_max, y_max)] = _SKIP_AREA_BOTTOM_RIGHT
img = img[y_min:y_max, x_min:x_max]
circle_x, circle_y, circle_r = None, None, None
circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=999, param1=300, param2=10, minRadius=10, maxRadius=13)
out_circles = []
col = (0, 0, 255)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        circle_x, circle_y, circle_r = i[0], i[1], i[2]
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], col, 1)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 1, col, 1)
        out_circles.append([i[0], i[1], i[2]])  # circle_x, circle_y, circle_r
    circles = out_circles

    # draw bot position
    cv.circle(img, (circles[0][0], circles[0][1]+5), 1, (255, 0, 0), 1)


    cv.imwrite(f"imgs\\result4.jpg", img)
"""



# OLD CODE - FOR REUSE
"""
class HealthBarsClass:
    def get_from_image(self):
        #----debug only
        cv.imwrite('imgs\\_output1.jpg', self.img)
        #----------------

        self.filter_by_color()
        #----debug only
        cv.imwrite('imgs\\_output2.jpg', self.mask)
        #----------------

        self.transform_mask()
        #----debug only
        cv.imwrite('imgs\\_output3.jpg', self.mask)
        #----------------

        self.get_contours()
        #----debug only
        img = copy.deepcopy(self.img)
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv.imwrite('imgs\\_output4.jpg', img)
        #----------------

        self.filter_contours_by_borders()
        #----debug only
        img = copy.deepcopy(self.img)
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv.imwrite('imgs\\_output5.jpg', img)
        #----------------

        self.filter_contours_by_color_fill()
        #----debug only
        img = copy.deepcopy(self.img)
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv.imwrite('imgs\\_output6.jpg', img)
        #----------------

        self.filter_contours_by_dimensions()
        #----debug only
        img = copy.deepcopy(self.img)
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv.imwrite('imgs\\_output7.jpg', img)
        #----------------

        self.filter_contours_by_position_in_image()
        #----debug only
        img = copy.deepcopy(self.img)
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv.imwrite('imgs\\_output8.jpg', img)
        #----------------
        a=1

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
        kernel = np.ones((3, 3), np.uint8)
        result = cv.morphologyEx(self.mask, cv.MORPH_OPEN, kernel)
        #result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
        #result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
        # dilate
        #kernel = np.ones((3, 3), np.uint8)
        #result = cv.dilate(self.mask, kernel, iterations=2)
        # close
        #kernel = np.ones((2, 2), np.uint8)
        #result = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, kernel)

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

        def get_adjusted_range(x_first=None, x_last=None, y_first=None, y_last=None, step=None):
            # make sure that values are within window's range (last is 1 pixel before window's edge)
            if None not in [x_first, x_last]:
                if x_first < 0: x_first = 0
                if x_first > _SCREEN_W: x_first = _SCREEN_W
                if x_last < 0: x_last = 1
                if x_last > _SCREEN_W: x_last = _SCREEN_W - 1
                return range(x_first, x_last, step)
            else:   # [y_first, y_last]:
                if y_first < 0: y_first = 0
                if y_first > _SCREEN_H: y_first = _SCREEN_H
                if y_last < 0: y_last = 1
                if y_last > _SCREEN_H: y_last = _SCREEN_H - 1
                return range(y_first, y_last, step)

        out_contours = []
        for [(x_min, y_min), (x_max, y_max)] in self.contours:
            found_top, found_bottom, found_left, found_right = False, False, False, False
            x_min_new, y_min_new, x_max_new, y_max_new = None, None, None, None

            # find starting point - point, closest to (x_min, y_min), that has color in range
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
                for y in get_adjusted_range(y_first=y_start, y_last=y_start-_HB_BOX_MAX_H, step=-1):
                    if color_changed(self.hsv[y][x_start], self.hsv[y - 1][x_start]):
                        found_top = True
                        y_min_new = y - 1
                        break
                # bottom border
                for y in get_adjusted_range(y_first=y_start, y_last=y_start+_HB_BOX_MAX_H, step=1):
                    if color_changed(self.hsv[y][x_start], self.hsv[y + 1][x_start]):
                        found_bottom = True
                        y_max_new = y + 1
                        break
                # left border
                for x in get_adjusted_range(x_first=x_start, x_last=x_start-_HB_BOX_MAX_W, step=-1):
                    if color_changed(self.hsv[y_start][x], self.hsv[y_start][x - 1]):
                        found_left = True
                        x_min_new = x - 1
                        break
                # right border
                for x in get_adjusted_range(x_first=x_start, x_last=x_start+_HB_BOX_MAX_W, step=1):
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


    # based on contours min,max values, go all the way on each polygon side
    # def filter_contours_by_borders(self):
    #     thr_value_for_color_change = 50  # %
    #     thr_number_of_pixels_defining_border = 50  # %
    # 
    #     def color_changed(list_1, list_2):
    #         # when comparing hsv arrays and the change of values is more than threshold -> then the color has been changed
    #         for val1, val2 in zip(list_1, list_2):
    #             if int(val1) == 0:  # case: divide by 0
    #                 return False
    #             elif ((int(val1) - int(val2)) / int(val1)) * 100 >= thr_value_for_color_change:
    #                 return True
    #         return False
    # 
    #     out_contours = []
    #     for [(x_min, y_min), (x_max, y_max)] in self.contours:
    #         found_top, found_bottom, found_left, found_right = False, False, False, False
    #         x_min_new, y_min_new, x_max_new, y_max_new = None, None, None, None
    #         x_s = x_min + (x_max - x_min) // 2
    #         y_s = y_min + (y_max - y_min) // 2
    #         x_step = (x_s - x_min) // 2 + 1
    #         y_step = (y_s - y_min) // 2 + 1
    # 
    #         # top border
    #         for y in range(y_s, y_min - y_step, -1):
    #             cnt = 0
    #             for x in range(x_min, x_max):
    #                 if color_changed(self.hsv[y][x], self.hsv[y - 1][x]):
    #                     cnt += 1
    #             if cnt / len(range(x_min, x_max)) * 100 >= thr_number_of_pixels_defining_border:
    #                 found_top = True
    #                 y_min_new = y - 1
    #                 break
    #         # bottom border
    #         for y in range(y_s, y_max + y_step):
    #             cnt = 0
    #             for x in range(x_min, x_max):
    #                 if color_changed(self.hsv[y][x], self.hsv[y + 1][x]):
    #                     cnt += 1
    #             if cnt / len(range(x_min, x_max)) * 100 >= thr_number_of_pixels_defining_border:
    #                 found_bottom = True
    #                 y_max_new = y + 1
    #                 break
    #         # left border
    #         for x in range(x_min + x_step, x_min - x_step, -1):
    #             cnt = 0
    #             for y in range(y_min, y_max):
    #                 if color_changed(self.hsv[y][x], self.hsv[y][x - 1]):
    #                     cnt += 1
    #             if cnt / len(range(y_min, y_max)) * 100 >= thr_number_of_pixels_defining_border:
    #                 found_left = True
    #                 x_min_new = x - 1
    #                 break
    #         # right border
    #         for x in range(x_max - x_step, x_max + x_step):
    #             cnt = 0
    #             for y in range(y_min, y_max):
    #                 if color_changed(self.hsv[y][x], self.hsv[y][x + 1]):
    #                     cnt += 1
    #             if cnt / len(range(y_min, y_max)) * 100 >= thr_number_of_pixels_defining_border:
    #                 found_right = True
    #                 x_max_new = x + 1
    #                 break
    # 
    #         # add contour if all borders are found
    #         if found_top and found_bottom and found_left and found_right:
    #             if x_min_new > x_max_new or y_min_new > y_max_new:
    #                 raise Exception(
    #                     f'Incorrect detection of borders: min=({x_min_new, y_min_new}), max=({x_max_new, y_max_new})')
    #             out_contours.append([(x_min_new, y_min_new), (x_max_new, y_max_new)])
    #         # print('\t', [(x_min, y_min), (x_max, y_max)], found_top, found_bottom, found_left, found_right)
    #     self.contours = out_contours
    # 
    #     # draw
    #     if len(self.contours) > 0:
    #         for [(x_min, y_min), (x_max, y_max)] in self.contours:
    #             cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    #         cv.imwrite('imgs\\_4_contours_2.jpg', self.img)

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
            for box in _SKIP_AREAS:
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
"""