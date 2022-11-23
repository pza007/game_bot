import cv2 as cv
import numpy as np
import math
from global_vars import _SKIP_AREAS, _SCREEN_W, _SCREEN_H,\
    _HB_BOX_H, _HB_BOX_MIN_W, _HB_BOX_MAX_W, _HB_COLOR_THRS_RED, _HB_COLOR_THRS_BLUE,\
    _MN_BOX_W, _MN_BOX_H, _MN_BOX_H_SHIFT, _MN_BOX_X_SHIFT,\
    _BOT_HB_COLOR_THR, _BOT_HB_BOX_MIN_H, _BOT_HB_BOX_MAX_H, _BOT_HB_BOX_MIN_W, _BOT_HB_BOX_W, _BOT_HB_BOX_MAX_W,\
    _BOT_BOX_W_MOUNT, _BOT_BOX_H_MOUNT, _BOT_BOX_H_SHIFT_MOUNT, _BOT_BOX_X_SHIFT_MOUNT,\
    _BOT_BOX_W_UNMOUNT, _BOT_BOX_H_UNMOUNT, _BOT_BOX_H_SHIFT_UNMOUNT, _BOT_BOX_X_SHIFT_UNMOUNT

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
    def __init__(self, img, hsv, color):
        self.objects = []   # <list of MinionClass>
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
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

            # draw rectangle for bounding box
            [(x_min, y_min), (x_max, y_max)] = minion.bounding_box
            cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), col, 1)
            # draw circle for center
            cv.circle(in_img, (minion.point_center[0]-1, minion.point_center[1]-1), 2, col, 2)
            # draw rectangle + text for object type and %health
            cv.rectangle(in_img, (x_min, y_max), (x_min+100, y_max+14), col, -1)
            cv.putText(
                img=in_img,
                text='MN hp:'+str(minion.health)+'%',
                org=(x_min+1, y_max+12),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=(0, 0, 0),
                thickness=1)

        return in_img


class ManaCollectible:
    # if minion dies, in his bounding box there could be MANA sphere (or XP sphere)
    # for red minion: 4 sec mana blue -> 4 sec mana pink -> disappear
    # for blue minion: 4 sec mana red -> 4 sec mana pink -> disappear
    pass
    """
    f_path = "imgs\\test_images\\collectibles_1.jpg"
    # f_path="imgs\\test_images\\collectibles_1_2.jpg"
    # f_path="imgs\\test_images\\collectibles_3_1.jpg"
    img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


    t1 = np.array([80, 80, 120])  # 90,  110, 210
    t2 = np.array([150, 220, 255])  # 120, 200, 255
    color_thresholds = [t1, t2]
    mask = cv.inRange(hsv, color_thresholds[0], color_thresholds[1])
    cv.imwrite(mask, f"imgs\\_output_1.jpg")

    # kernel = np.ones((3, 3), np.uint8)
    # result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # result = cv.dilate(mask, kernel, iterations=1)
    # save_image(result, f"imgs\\_output_2.jpg")
    result = mask

    # minDist - minimum distance between the centers of the detected circles
    # param1 - sensitivity
    # param2 - minimal number of edges
    img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    circle_x, circle_y, circle_r = None, None, None
    circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, dp=1.2, minDist=11, param1=300, param2=10, minRadius=14,
                              maxRadius=20)
    out_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
            out_circles.append([i[0], i[1], i[2]])  # circle_x, circle_y, circle_r
        circles = out_circles
        cv.imwrite(img, f"imgs\\_output_2.jpg")

    # circle fill lvl?
    img = cv.imread(f_path, cv.IMREAD_UNCHANGED)
    out_circles = []
    for [circle_x, circle_y, circle_r] in circles:
        x_min, y_min = circle_x - circle_r, circle_y - circle_r
        x_max, y_max = circle_x + circle_r, circle_y + circle_r

        circle_mask = np.zeros((2 * circle_r + 1, 2 * circle_r + 1), dtype="uint8")  # tuple(size_y, size_x)
        cv.circle(circle_mask, (circle_r, circle_r), circle_r, 255, -1)
        # save_image(circle_mask, f'imgs\\_output_3.jpg')
        img_cropped = img[y_min:y_max + 1, x_min:x_max + 1]
        # save_image(img_cropped, f'imgs\\_output_4.jpg')
        img_masked = cv.bitwise_and(img_cropped, img_cropped, mask=circle_mask)
        # save_image(img_masked, f'imgs\\_output_5.jpg')
        hsv_cropped = cv.cvtColor(img_masked, cv.COLOR_BGR2HSV)

        thr_fill = 40  # %
        cnt = 0
        for y in range(0, hsv_cropped.shape[0]):
            for x in range(0, hsv_cropped.shape[1]):
                h, s, v = hsv_cropped[y][x]
                if color_thresholds[0][0] <= h <= color_thresholds[1][0] and \
                        color_thresholds[0][1] <= s <= color_thresholds[1][1] and \
                        color_thresholds[0][2] <= v <= color_thresholds[1][2]:
                    cnt += 1

        # check
        circle_area = math.pi * circle_r * circle_r
        if (cnt / circle_area) * 100 >= thr_fill:
            out_circles.append([circle_x, circle_y, circle_r])
            # draw the outer circle
            cv.circle(img, (circle_x, circle_y), circle_r, (0, 255, 0), 1)
            # draw the center of the circle
            cv.circle(img, (circle_x, circle_y), 2, (0, 0, 255), 2)

        print(f'circle: ({circle_x},{circle_y}) r={circle_r}, area={(cnt / circle_area) * 100}')
    cv.imwrite(img, f'imgs\\_output_3.jpg')
    """


class BotClass:
    def __init__(self, img, hsv):
        self.img = img     # image with B,G,R,A values
        self.hsv = hsv     # image with H,S,V values
        self.health_bar = None  # <HealthBarsClass.HealthBarClass>
        self.health = None  # <int> 0-100 %
        self.bounding_box = None    # <int> [(x_min, y_min), (x_max, y_max)]
        self.point_center = None    # <int> (x,y)

    def get_from_image(self):
        self.get_health_bar()
        self.get_bounding_box()
        self.get_point_center()

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
        #----debug only
        #for [(x_min, y_min), (x_max, y_max)] in contours:
        #    cv.rectangle(self.img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        #cv.imwrite('imgs\\output1.jpg', self.img)
        #----------------
        #   filter contours -> contour located in center of image?
        out_contours = []
        search_polygon = [(_SCREEN_W // 2 - _BOT_HB_BOX_MAX_W, _SCREEN_H // 8),
                          (_SCREEN_W // 2 + _BOT_HB_BOX_MAX_W, _SCREEN_H // 8),
                          (_SCREEN_W // 2 + _BOT_HB_BOX_MAX_W, _SCREEN_H // 2),
                          (_SCREEN_W // 2 - _BOT_HB_BOX_MAX_W, _SCREEN_H // 2)]
        for [(x_min, y_min), (x_max, y_max)] in contours:
            points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            for point in points:
                if point_inside_polygon(point[0], point[1], search_polygon):
                    out_contours.append([(x_min, y_min), (x_max, y_max)])
                    break
        contours = out_contours
        #----debug only
        #cv.rectangle(self.img, (search_polygon[0][0], search_polygon[0][1]), (search_polygon[2][0], search_polygon[2][1]), (255, 0, 0), 2)
        #cv.imwrite('imgs\\output2.jpg', self.img)
        #----------------
        #   filter contours -> contour fits health bar dimensions?
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
            self.health_bar.fill_lvl = int((width / _BOT_HB_BOX_W) * 100)
            if self.health_bar.fill_lvl > 100:
                self.health_bar.fill_lvl = 100
            # health
            self.health = self.health_bar.fill_lvl
        else:
            print('BOT: health bar not found!')

    def get_bounding_box(self):
        if self.health_bar is not None:
            x_hb_min = self.health_bar.position[0][0]     # [(x_min, y_min), (x_max, y_max)]
            y_hb_min = self.health_bar.position[0][1]
            x_hb_middle = x_hb_min + int(_BOT_HB_BOX_MAX_W // 2)

            # bot mounted
            if y_hb_min < int(2*_SCREEN_H/8):
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

            self.bounding_box = [(x_min, y_min), (x_max, y_max)]
        else:
            print('BOT: bounding box not found!')

    def get_point_center(self):
        if self.bounding_box is not None:
            [(x_min, y_min), (x_max, y_max)] = self.bounding_box
            self.point_center = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
        else:
            print('BOT: no point center')

    def draw(self, in_img):
        if self.health_bar is not None and self.bounding_box is not None:
            # draw rectangles for health bar
            #[(x_min, y_min), (x_max, y_max)] = self.health_bar.position
            #cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
            #cv.rectangle(in_img, (x_min, y_min), (x_min+_BOT_HB_BOX_MAX_W, y_max), (0, 255, 255), 1)
            # draw line for center position
            #cv.line(in_img, self.point_center, (self.point_center[0], y_max), (0, 255, 255), 2)

            # draw rectangle for bounding box
            [(x_min, y_min), (x_max, y_max)] = self.bounding_box
            cv.rectangle(in_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)

            # draw rectangle + text for object type and %health
            cv.rectangle(in_img, (x_min, y_max), (x_min+100, y_max+14), (0, 255, 255), -1)
            cv.putText(
                img=in_img,
                text='BOT hp:'+str(self.health)+'%',
                org=(x_min+1, y_max+12),
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=(0, 0, 0),
                thickness=1)

            return in_img
        else:
            print('BOT: nothing to draw')
            return None


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