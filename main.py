# AI
#import gym_lib
#gym_lib.plot_training_history('rl/models/ai_XP1180_data.json')
from ai_lib import AI
env = AI()
env.play_game(10)


# TEST ENV
"""
episodes = 2
for episode in range(1, episodes + 1):
    print('Reset.....')
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        score += reward
        print(info['info'])
    print('Episode:{} Score:{}'.format(episode, score))
"""


# MACHINE LEARNING
"""
import os
import gym_lib

# plot data
gym_lib.plot_training_history('rl/models/230628_030928/230628_030928_plus1040_data.json')

# train model
gym_lib.num_steps = 8000    # = 5 hours of work
agent, hist = gym_lib.train_agent(gym_lib.agent, gym_lib.env, gym_lib.num_steps)

# save model
gym_lib.save_model(hist)

# shutdown PC
os.system("shutdown /s /t 1")

# test model
#from keras.optimizers import Adam
#agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
#scores = agent.test(env, nb_episodes=2, visualize=False)
#print(np.mean(scores.history['episode_reward']))
#scores = model.test(env, nb_episodes=2, visualize=False)
#print(np.mean(scores.history['episode_reward']))
"""






# GET H,S,V VALUES OF IMAGE
"""
import cv2 as cv
import numpy as np
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
"""
# LABEL EACH FRAME
"""
import cv2 as cv
import spectator_lib
import functions

s = spectator_lib.Spectator()

i = 0
while 1:
  frame = s.window.get_screenshot()
  file_path = f'C:\\Users\\Lordor\\Videos\\Captures\\frames\\frame{i}.jpg'
  cv.imwrite(file_path, frame)
  # draw
  frame = functions.label_frame(frame)
  file_path = f'C:\\Users\\Lordor\\Videos\\Captures\\l_frames\\frame{i}.jpg'
  cv.imwrite(file_path, frame)
  print(f'frame_{i}')
  i+=1
"""