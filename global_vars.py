import numpy as np
import cv2 as cv
###################################################
#        MINIONS
# -------------------------------------------------
# Health Bar - dimensions
_HB_BOX_MIN_H = 8
_HB_BOX_MAX_H = 10
_HB_BOX_MIN_W = 2
_HB_BOX_MAX_W = 77
# Health Bar - color
#                          min: 171 199 99           max: 173 255 207
_HB_COLOR_THRS_RED = [np.array([170, 198, 98]), np.array([175, 256, 210])]
#                           min: 107 141 99            max: 109 206 237
_HB_COLOR_THRS_BLUE = [np.array([103, 135, 93]), np.array([115, 211, 240])]
# Health Bar - skip areas (do not detect health bars inside these areas)
_HB_SKIP_AREA_TOP_MIDDLE =    [(891, 0), (1698, 110)]
_HB_SKIP_AREA_BOTTOM_LEFT =   [(0, 1262), (616, 1528)]
_HB_SKIP_AREA_BOTTOM_MIDDLE = [(927, 1391), (1786, 1528)]
_HB_SKIP_AREA_BOTTOM_RIGHT =  [(2165, 1140), (2715, 1528)]
_HB_SKIP_AREAS = [_HB_SKIP_AREA_TOP_MIDDLE, _HB_SKIP_AREA_BOTTOM_LEFT, _HB_SKIP_AREA_BOTTOM_MIDDLE, _HB_SKIP_AREA_BOTTOM_RIGHT]

# Minion - mage image to comparison
_MN_HAT_MAGE_RED =  cv.imread('imgs\\minion_mage_hat_red.jpg', cv.IMREAD_UNCHANGED)
_MN_HAT_MAGE_BLUE = cv.imread('imgs\\minion_mage_hat_blue.jpg', cv.IMREAD_UNCHANGED)



###################################################
#        BOT
# -------------------------------------------------
# Circle - color
#                             min: 58, 200, 120         max: 61, 235, 230
_BOT_CIRCLE_COLOR_THR = [np.array([58, 200, 120]), np.array([61, 245, 240])]

# Health Bar - dimensions
_BOT_HB_BOX_MIN_H = 10
_BOT_HB_BOX_MAX_H = 17
_BOT_HB_BOX_MIN_W = 2
_BOT_HB_BOX_MAX_W = 180
# Health Bar - shift from circle's center
_BOT_HB_MAX_Y_SHIFT = 400
# Health Bar - color
#                         min: 50, 200, 200         max: 55, 230, 210
_BOT_HB_COLOR_THR = [np.array([50, 200, 200]), np.array([55, 230, 210])]
