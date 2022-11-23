import cv2 as cv
import numpy as np
from time import time
import math
from functions import detect_objects, draw_objects, save_image
from window_capture import WindowCapture

# TODO:
# - detect collectibles: class ManaCollective
# - track objects


window = WindowCapture('Heroes of the Storm')
loop_time = time()
while(True):
    # get screenshot
    screenshot = window.get_screenshot()
    #screenshot = cv.imread("imgs\\output.jpg", cv.IMREAD_UNCHANGED)

    # detect
    bot, minions_red, minions_blue = detect_objects(screenshot)
    # draw
    l_screenshot = draw_objects(screenshot, bot, minions_red, minions_blue)
    # show
    cv.imshow('Computer Vision', l_screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

save_image(l_screenshot, f"imgs\\output.jpg")
print('Done.')


# GET MIN MAX COLORS FROM IMGAE
"""
    screenshot = cv.imread("imgs\\tmp.jpg", cv.IMREAD_UNCHANGED)
    hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
    
    min_h, min_s, min_v = 300, 300, 300
    max_h, max_s, max_v = -1, -1, -1
    for i in range(0, len(screenshot)):
        for j in range(0, len(screenshot[0])):
            h = hsv[i][j][0]
            s = hsv[i][j][1]
            v = hsv[i][j][2]
            if h < min_h: min_h = h
            if h > max_h: max_h = h
            if s < min_s: min_s = s
            if s > max_s: max_s = s
            if v < min_v: min_v = v
            if v > max_v: max_v = v
"""
# PYAUTOGUI
"""
import pyautogui
screenWidth, screenHeight = pyautogui.size()    # Get the size of the primary monitor.
print(f'screenWidth={screenWidth}, screenHeight={screenHeight}')
#currentMouseX, currentMouseY = pyautogui.position()    # Get the XY position of the mouse.
#print(currentMouseX, currentMouseY)
time.sleep(2)
pyautogui.moveTo(214, 98) # Move the mouse to XY coordinates.
pyautogui.click()          # Click the mouse.
"""
# MATCH TEMPLATE
"""
img = cv.imread('imgs\\test_1.jpg', cv.IMREAD_UNCHANGED)  # cv.IMREAD_UNCHANGED, cv.IMREAD_REDUCED_COLOR_2

hat = cv.imread('imgs\\mage_hat_blue.jpg', cv.IMREAD_UNCHANGED)
hat = cv.imread('imgs\\mage_hat_red.jpg', cv.IMREAD_UNCHANGED)
hat = cv.imread('imgs\\archer_hat.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(img, hat, cv.TM_CCOEFF_NORMED)    # cv.TM_SQDIFF_NORMED
threshold = 0.57
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))
print(locations)
if locations:
    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + hat.shape[1], top_left[1] + hat.shape[0])
        cv.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=1, lineType=cv.LINE_4)
else:
    print('Hat not found :(')
cv.imwrite('imgs\\result.jpg', img)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)
if max_val >= threshold:
    print('Found hat :)')
    top_left = max_loc
    bottom_right = (top_left[0] + hat.shape[1], top_left[1] + hat.shape[0])
    cv.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=1, lineType=cv.LINE_4)
    #cv.imshow('Result', img)
    #cv.waitKey()
    cv.imwrite('imgs\\result.jpg', img)
else:
    print('Hat not found :(')
"""
# CONVERT VIDEO TO FRAMES
"""
vidcap = cv.VideoCapture('C:\\Users\\Lordor\\Videos\\Captures\\vid_1.mp4')
success, image = vidcap.read()
count_all = 0
count_s = 0
while success:
  cv.imwrite("imgs\\frames\\frame%d.jpg" % count_all, image)     # save frame as JPEG file
  success, image = vidcap.read()
  if success: count_s+=1
  count_all += 1
print(f'number of frames={count_all}, success={count_s} ({(count_s/count_all)*100}%)')

# LABEL EACH FRAME
#for i in range(649):
  file_path = f"imgs\\frames\\frame{i}.jpg"

  img = cv.imread(file_path, cv.IMREAD_UNCHANGED)  # cv.IMREAD_UNCHANGED, cv.IMREAD_REDUCED_COLOR_2
  # detect
  bot, minions_red, minions_blue = detect_objects(img)
  # draw
  l_img = draw_objects(img, bot, minions_red, minions_blue)
  # save
  file_path = f"imgs\\l_frames\\frame{i}.jpg"
  save_image(l_img, file_path)
  print(f'frame {i}')
"""
