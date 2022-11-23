import numpy as np
import cv2 as cv
###################################################
#        GENERAL
# -------------------------------------------------
# Skip areas (do not detect object inside these areas)
_SKIP_AREA_TOP_MIDDLE =    [(630, 0), (1200, 77)]
_SKIP_AREA_BOTTOM_LEFT =   [(0, 894), (432, 1080)]
_SKIP_AREA_BOTTOM_MIDDLE = [(657, 982), (1263, 1080)]
_SKIP_AREA_BOTTOM_RIGHT =  [(1544, 838), (1920, 1080)]
_SKIP_AREAS = [_SKIP_AREA_TOP_MIDDLE, _SKIP_AREA_BOTTOM_LEFT, _SKIP_AREA_BOTTOM_MIDDLE, _SKIP_AREA_BOTTOM_RIGHT]
_SCREEN_W = 1920
_SCREEN_H = 1080



###################################################
#        MINIONS
# -------------------------------------------------
# Health Bar - dimensions
_HB_BOX_H = 4
_HB_BOX_MIN_W = 1
_HB_BOX_MAX_W = 51
# Health Bar - color red
#                          min: 171 199 99           max: 173 255 207
_HB_COLOR_THRS_RED = [np.array([150, 170, 80]), np.array([190, 255, 230])]
#_HB_COLOR_THRS_RED = [np.array([170, 198, 98]), np.array([175, 256, 210])]    # old values
# Health Bar - color blue
#                           min: 107 141  99         max: 109  206  237
_HB_COLOR_THRS_BLUE = [np.array([90, 120, 90]), np.array([120, 220, 255])]
#_HB_COLOR_THRS_BLUE = [np.array([103, 135, 93]), np.array([115, 211, 240])]    # old values


# Minion - dimensions
_MN_BOX_W = 65
_MN_BOX_H = 73
_MN_BOX_H_SHIFT = 21    # shift bounding box to bottom (in respect to y_min of health bar)
_MN_BOX_X_SHIFT = -20    # shift bounding box to left (in respect to x_middle of health bar)
# Minion - mage image to comparison
_MN_HAT_MAGE_RED =  cv.imread('imgs\\minion_mage_hat_red.jpg', cv.IMREAD_UNCHANGED)
_MN_HAT_MAGE_BLUE = cv.imread('imgs\\minion_mage_hat_blue.jpg', cv.IMREAD_UNCHANGED)



###################################################
#        COLLECTIBLES
# -------------------------------------------------
# Mana sphere - color blue
#                             min: 90, 110,210         max: 120, 200, 255
_MANA_COLOR_THRS_BLUE = [np.array([80, 80, 120]), np.array([150, 220, 255])]
# Mana sphere - circle parameters
_MANA_CIRCLE_MIN_DIST = 11    # minDist - minimum distance between the centers of the detected circles
_MANA_CIRCLE_SENSITIVITY = 300  # param1 - sensitivity
_MANA_CIRCLE_MIN_EDGES = 10  # param2 - minimal number of edges
_MANA_CIRCLE_MIN_RADIUS = 14
_MANA_CIRCLE_MAX_RADIUS = 20


# param2

###################################################
#        BOT
# -------------------------------------------------
# Circle - color
#                             min: 58, 200, 120         max: 61, 235, 230
_BOT_CIRCLE_COLOR_THR = [np.array([58, 200, 120]), np.array([61, 245, 240])]
# Health Bar - dimensions
_BOT_HB_BOX_MIN_H = 6
_BOT_HB_BOX_H = 9
_BOT_HB_BOX_MAX_H = 14
_BOT_HB_BOX_MIN_W = 2
_BOT_HB_BOX_W = 124
_BOT_HB_BOX_MAX_W = 130
# Health Bar - color
#                         min: 50, 200, 200         max: 55, 230, 210
_BOT_HB_COLOR_THR = [np.array([50, 200, 200]), np.array([55, 230, 210])]
# Model - dimensions
_BOT_BOX_W_MOUNT = 100
_BOT_BOX_H_MOUNT = 150
_BOT_BOX_H_SHIFT_MOUNT = 90
_BOT_BOX_X_SHIFT_MOUNT = -10
_BOT_BOX_W_UNMOUNT = 80
_BOT_BOX_H_UNMOUNT = 110
_BOT_BOX_H_SHIFT_UNMOUNT = 44
_BOT_BOX_X_SHIFT_UNMOUNT = -5
