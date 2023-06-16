# TODO: plan for game
#   0. No ally, no enemy, toggle ON minions, control panel hidden, talent tree hidden
#   1. Refresh forts (unhide, hide control panel)
#   2. Move to gate, demount
#   3. Perform all actions
#   ...
#   When dead -> check total xp and start from step 1)
# ----------------------------------------------------------------------------------------------------------




# Implemented:
# 'basic_attack' (move & attack closest red minion)
# 'q_attack' (move & attack with spell 'Q')
# 'use_well'
# 'hide_in_bushes'
# 'hide_behind_gate
# 'escape_behind_gate' (with spell 'E')
# 'use_spell_d' (recover health)
# 'move' (up, right, down, left, up-right, down-right, left-down, up-left)
# ----------------------------------------------------------------------------------------------------------
from spectator_lib import Spectator

spec = Spectator()
spec.run()


# GET H,S,V VALUES OF IMAGE
"""
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

frame = cv.imread("imgs\\tmp_2.jpg", cv.IMREAD_UNCHANGED)
[(x_min, y_min), (x_max, y_max)] = _SKIP_AREA_BOTTOM_RIGHT
frame = frame[y_min:y_max, :]
while True:
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)

    if cv.waitKey(30) == ord('q'):
        cv.destroyAllWindows()
        break
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
