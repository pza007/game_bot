import cv2 as cv
import numpy as np

###################################################
#        GENERAL
# -------------------------------------------------
SCREEN_W = 1920
SCREEN_H = 1080
# Skip areas (do not detect bot or minions inside these areas)
SKIP_AREA_TOP_MIDDLE =    [(630, 0), (1200, 77)]
SKIP_AREA_BOTTOM_LEFT =   [(0, 894), (432, 1080)]
SKIP_AREA_BOTTOM_MIDDLE = [(657, 982), (1263, 1080)]
SKIP_AREA_BOTTOM_RIGHT =  [(1565, 872), (1916, 1036)]
SKIP_AREAS = [SKIP_AREA_TOP_MIDDLE, SKIP_AREA_BOTTOM_LEFT, SKIP_AREA_BOTTOM_MIDDLE, SKIP_AREA_BOTTOM_RIGHT]


###################################################
#        TEMPLATES
# -------------------------------------------------
template_H08_slash = cv.imread(f'imgs\\templates\\numbers\\H08_slash.png', cv.IMREAD_GRAYSCALE)
template_H08_num_0 = cv.imread(f'imgs\\templates\\numbers\\H08_num_0.png', cv.IMREAD_GRAYSCALE)
template_H08_num_1 = cv.imread(f'imgs\\templates\\numbers\\H08_num_1.png', cv.IMREAD_GRAYSCALE)
template_H08_num_2 = cv.imread(f'imgs\\templates\\numbers\\H08_num_2.png', cv.IMREAD_GRAYSCALE)
template_H08_num_3 = cv.imread(f'imgs\\templates\\numbers\\H08_num_3.png', cv.IMREAD_GRAYSCALE)
template_H08_num_4 = cv.imread(f'imgs\\templates\\numbers\\H08_num_4.png', cv.IMREAD_GRAYSCALE)
template_H08_num_5 = cv.imread(f'imgs\\templates\\numbers\\H08_num_5.png', cv.IMREAD_GRAYSCALE)
template_H08_num_6 = cv.imread(f'imgs\\templates\\numbers\\H08_num_6.png', cv.IMREAD_GRAYSCALE)
template_H08_num_7 = cv.imread(f'imgs\\templates\\numbers\\H08_num_7.png', cv.IMREAD_GRAYSCALE)
template_H08_num_8 = cv.imread(f'imgs\\templates\\numbers\\H08_num_8.png', cv.IMREAD_GRAYSCALE)
template_H08_num_9 = cv.imread(f'imgs\\templates\\numbers\\H08_num_9.png', cv.IMREAD_GRAYSCALE)

template_H10_slash = cv.imread(f'imgs\\templates\\numbers\\H10_slash.png', cv.IMREAD_GRAYSCALE)
template_H10_num_0 = cv.imread(f'imgs\\templates\\numbers\\H10_num_0.png', cv.IMREAD_GRAYSCALE)
template_H10_num_1 = cv.imread(f'imgs\\templates\\numbers\\H10_num_1.png', cv.IMREAD_GRAYSCALE)
template_H10_num_2 = cv.imread(f'imgs\\templates\\numbers\\H10_num_2.png', cv.IMREAD_GRAYSCALE)
template_H10_num_3 = cv.imread(f'imgs\\templates\\numbers\\H10_num_3.png', cv.IMREAD_GRAYSCALE)
template_H10_num_4 = cv.imread(f'imgs\\templates\\numbers\\H10_num_4.png', cv.IMREAD_GRAYSCALE)
template_H10_num_5 = cv.imread(f'imgs\\templates\\numbers\\H10_num_5.png', cv.IMREAD_GRAYSCALE)
template_H10_num_6 = cv.imread(f'imgs\\templates\\numbers\\H10_num_6.png', cv.IMREAD_GRAYSCALE)
template_H10_num_7 = cv.imread(f'imgs\\templates\\numbers\\H10_num_7.png', cv.IMREAD_GRAYSCALE)
template_H10_num_8 = cv.imread(f'imgs\\templates\\numbers\\H10_num_8.png', cv.IMREAD_GRAYSCALE)
template_H10_num_9 = cv.imread(f'imgs\\templates\\numbers\\H10_num_9.png', cv.IMREAD_GRAYSCALE)

template_death_icon = cv.imread(f'imgs\\templates\\death_icon.png', cv.IMREAD_GRAYSCALE)
template_bot_icon = cv.imread(f'imgs\\templates\\bot_icon.png', cv.IMREAD_GRAYSCALE)
template_well = cv.imread(f'imgs\\templates\\well.png', cv.IMREAD_GRAYSCALE)
template_plus_symbol = cv.imread(f'imgs\\templates\\plus_symbol.png', cv.IMREAD_GRAYSCALE)
template_hidden_symbol = cv.imread(f'imgs\\templates\\hidden_symbol.png', cv.IMREAD_GRAYSCALE)
template_blocked_symbol = cv.imread(f'imgs\\templates\\blocked_symbol.png', cv.IMREAD_GRAYSCALE)
template_minimap = cv.imread(f'imgs\\templates\\minimap.png', cv.IMREAD_GRAYSCALE)







###################################################
#        GENERAL
# -------------------------------------------------



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



###################################################
#        BOT
# -------------------------------------------------
# Circle - color
#                             min: 58, 200, 120         max: 61, 235, 230
_BOT_CIRCLE_COLOR_THR = [np.array([58, 200, 120]), np.array([61, 245, 240])]
# Health Bar - dimensions
_BOT_HB_BOX_MIN_H = 15
_BOT_HB_BOX_H = 16
_BOT_HB_BOX_MAX_H = 17
_BOT_HB_BOX_MIN_W = 2
_BOT_HB_BOX_W = 124
_BOT_HB_BOX_MAX_W = 130
# Health Bar - color
#                         min: 50, 100, 150         max: 55, 255, 255
_BOT_HB_COLOR_THR = [np.array([50, 100, 150]), np.array([55, 255, 255])]
# Model - dimensions
_BOT_BOX_W_MOUNT = 100
_BOT_BOX_H_MOUNT = 150
_BOT_BOX_H_SHIFT_MOUNT = 90
_BOT_BOX_X_SHIFT_MOUNT = -10
_BOT_BOX_W_UNMOUNT = 80
_BOT_BOX_H_UNMOUNT = 110
_BOT_BOX_H_SHIFT_UNMOUNT = 44
_BOT_BOX_X_SHIFT_UNMOUNT = -5


###################################################
#        WELL
# -------------------------------------------------
# Health Bar - dimensions
_WELL_HB_BOX_H = 12
_WELL_HB_BOX_W = 200
# Health Bar - color
_WELL_HB_COLOR_THR = [np.array([66, 0, 134]), np.array([100, 255, 255])]
