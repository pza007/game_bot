import cv2 as cv
import numpy as np
from functions import detect_objects, draw_objects, save_image

"""
# PYAUTOGUI
import pyautogui
screenWidth, screenHeight = pyautogui.size()    # Get the size of the primary monitor.
print(f'screenWidth={screenWidth}, screenHeight={screenHeight}')
#currentMouseX, currentMouseY = pyautogui.position()    # Get the XY position of the mouse.
#print(currentMouseX, currentMouseY)
time.sleep(2)
pyautogui.moveTo(214, 98) # Move the mouse to XY coordinates.
pyautogui.click()          # Click the mouse.
"""
"""
# COMPARE BY PART OF IMAGE
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
#k = cv.waitKey(5) & 0xFF
#cv.destroyAllWindows()

#img = cv.imread('imgs\\bot_health_bar.jpg', cv.IMREAD_UNCHANGED)  # cv.IMREAD_UNCHANGED, cv.IMREAD_REDUCED_COLOR_2
#img = cv.imread('imgs\\bot_ring.jpg', cv.IMREAD_UNCHANGED)  # cv.IMREAD_UNCHANGED, cv.IMREAD_REDUCED_COLOR_2


"""
# CONVERT VIDEO TO FRAMES
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

file_path = f"imgs\\test_3.jpg"
img = cv.imread(file_path, cv.IMREAD_UNCHANGED)  # cv.IMREAD_UNCHANGED, cv.IMREAD_REDUCED_COLOR_2
# detect
bot, minions_red, minions_blue = detect_objects(img)
# draw
l_img = draw_objects(img, bot, minions_red, minions_blue)
# save
save_image(l_img, f"imgs\\l_frames\\test_3_labeled.jpg")