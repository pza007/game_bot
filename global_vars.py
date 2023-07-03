import cv2 as cv
import numpy as np

###################################################
#        GENERAL
# -------------------------------------------------
SCREEN_W = 1920
SCREEN_H = 1080
SCREEN_DIAG = 2202  # diagonal
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

template_H13_num_0 = cv.imread(f'imgs\\templates\\numbers\\H13_num_0.png', cv.IMREAD_GRAYSCALE)
template_H13_num_1 = cv.imread(f'imgs\\templates\\numbers\\H13_num_1.png', cv.IMREAD_GRAYSCALE)
template_H13_num_2 = cv.imread(f'imgs\\templates\\numbers\\H13_num_2.png', cv.IMREAD_GRAYSCALE)
template_H13_num_3 = cv.imread(f'imgs\\templates\\numbers\\H13_num_3.png', cv.IMREAD_GRAYSCALE)
template_H13_num_4 = cv.imread(f'imgs\\templates\\numbers\\H13_num_4.png', cv.IMREAD_GRAYSCALE)
template_H13_num_5 = cv.imread(f'imgs\\templates\\numbers\\H13_num_5.png', cv.IMREAD_GRAYSCALE)
template_H13_num_6 = cv.imread(f'imgs\\templates\\numbers\\H13_num_6.png', cv.IMREAD_GRAYSCALE)
template_H13_num_7 = cv.imread(f'imgs\\templates\\numbers\\H13_num_7.png', cv.IMREAD_GRAYSCALE)
template_H13_num_8 = cv.imread(f'imgs\\templates\\numbers\\H13_num_8.png', cv.IMREAD_GRAYSCALE)
template_H13_num_9 = cv.imread(f'imgs\\templates\\numbers\\H13_num_9.png', cv.IMREAD_GRAYSCALE)

template_H33_num_0 = cv.imread(f'imgs\\templates\\numbers\\H33_num_0.png', cv.IMREAD_GRAYSCALE)
template_H33_num_1 = cv.imread(f'imgs\\templates\\numbers\\H33_num_1.png', cv.IMREAD_GRAYSCALE)
template_H33_num_2 = cv.imread(f'imgs\\templates\\numbers\\H33_num_2.png', cv.IMREAD_GRAYSCALE)
template_H33_num_3 = cv.imread(f'imgs\\templates\\numbers\\H33_num_3.png', cv.IMREAD_GRAYSCALE)
template_H33_num_4 = cv.imread(f'imgs\\templates\\numbers\\H33_num_4.png', cv.IMREAD_GRAYSCALE)
template_H33_num_5 = cv.imread(f'imgs\\templates\\numbers\\H33_num_5.png', cv.IMREAD_GRAYSCALE)
template_H33_num_6 = cv.imread(f'imgs\\templates\\numbers\\H33_num_6.png', cv.IMREAD_GRAYSCALE)
template_H33_num_7 = cv.imread(f'imgs\\templates\\numbers\\H33_num_7.png', cv.IMREAD_GRAYSCALE)
template_H33_num_8 = cv.imread(f'imgs\\templates\\numbers\\H33_num_8.png', cv.IMREAD_GRAYSCALE)
template_H33_num_9 = cv.imread(f'imgs\\templates\\numbers\\H33_num_9.png', cv.IMREAD_GRAYSCALE)

template_death_icon = cv.imread(f'imgs\\templates\\death_icon.png', cv.IMREAD_GRAYSCALE)
template_bot_icon = cv.imread(f'imgs\\templates\\bot_icon.png', cv.IMREAD_GRAYSCALE)
template_well = cv.imread(f'imgs\\templates\\well.png', cv.IMREAD_GRAYSCALE)
template_plus_symbol = cv.imread(f'imgs\\templates\\plus_symbol.png', cv.IMREAD_GRAYSCALE)
template_hidden_symbol = cv.imread(f'imgs\\templates\\hidden_symbol.png', cv.IMREAD_GRAYSCALE)
template_blocked_symbol = cv.imread(f'imgs\\templates\\blocked_symbol.png', cv.IMREAD_GRAYSCALE)
template_minimap = cv.imread(f'imgs\\templates\\minimap.png', cv.IMREAD_GRAYSCALE)
