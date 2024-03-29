# Heroes of the Storm - robot player
<p align="center">
<img src="https://github.com/pza007/game_bot/blob/main/gifs/frame30.jpg" width="640" height="360">
</p>


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Setup](#setup)
* [Object Detection](#object-detection)
* [Control of bot](#control-of-bot)
* [Machine learning](#machine-learning)
  * [Strict Learning](#strict-learning)
  * [Free Learning](#free-learning)
* [Programming AI](#programming-ai)
* [Results Summary](#results-summary)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)
* [License](#license)


## General Information
The idea for this project came to me after playing the online video game: "Heroes of the Storm™". 
In the game, players form into five-player teams and fight against another team.
Team consists of characters of different roles.
One of the tasks of a player is to control the situation on the map's lane, by killing minions and collecting experience globes.
I wanted to automize this process and started a project, to create robot player (bot) that will do it for me, as best as possible and without external help. 

As the project envolved, I divided it into following sections:
- Object detection
- Control of bot
- Machine learning - bot as an agent
- Programming AI


## Technologies Used
- Object detection: _opencv, win32com_
- Control of bot: _pyautogui_
- Machine learning: _keras, tensorflow, gym, json_


## Features
- Detecting in-game objects for further evaluation
- Controlling movements and actions of the bot to gather game's experience points
- Executing escape maneuvers when the bot is at risk af dying


## Setup
1. Create the account on: battle.net 
2. Download and install the game: "Heroes of the Storm™"
3. Start game's launcher, start game
4. Choose: "Collections" then "Heroes". Select hero: "Mei" and choose "Try" button 
5. In-game: choose Ally Hero to "None" and Enemy Hero to "None" 
6. Clone this repository (git clone https://github.com/pza007/game_bot.git)
7. Install all packages, defined in requirements.txt 
8. Run the project while still in game


## Object Detection
Before object detection:

![](https://github.com/pza007/game_bot/blob/main/gifs/game_run.gif)


After object detection:

![](https://github.com/pza007/game_bot/blob/main/gifs/game_run_with_labels.gif)


Object detection starts with capturing frames of game's screen, using _win32com_ packages.
With the help of _opencv_ package, I managed to detect position of the bot and minions. 
Please see below, the instructions that cover detecting the position of the minion, based on it's health-bar position:
1. Convert RGB fame to HSV frame
2. Create image mask, based on threshold color
3. Transform mask (morphology)
4. Detect contours of objects in the mask
5. Execute filters:
   - filter contours with 4 points (rectangle)
   - filter contours with correct dimensions
   - filter contours located outside skip areas
   - filter contours with black border outside

Using combination of techniques mentioned above and 'matchTemplate' function, the following information can be extracted from the single frame:
```bash
'bot_dead': bool
'bot_pos_frame': {'health_bar': tuple[int, int], 'bounding_box': tuple[int, int], 'circle': tuple[int, int]}
'bot_pos_minimap': tuple[int, int]
'bot_health': tuple[int, int]
'bot_mana': tuple[int, int]
'cooldowns': {'Q': bool, 'W': bool, 'E': bool, 'R': bool, 'D': bool, 'well': bool}
'minions': {'blue': list[tuple[int, int]], 'red': list[tuple[int, int]]}
```


## Control of bot
To control the bot, package _pyautogui_ came in handy. 
Using it's functions I could command the movement of the bot by simulating mouse clicks on the game's screen.
It also helped me to simulate more complex bot's actions, using key presses.


## Machine learning
Another step in the project was to teach the bot how to efficiently play the game.
For this purpose, I used _keras, tensorflow_ and _gym_ packages.
I decided to use reinforced learning techniques. In this spirit, the bot (agent) will perform steps (actions) in the environment (game) and will get reward (experience) for it's steps.
The results of the agent's learning will be transferred to model, that will update it's weights accordingly. Used parameters for the rl-machine-learning are presented below.
```bash
inputs:
# [0]      float [0.0; 1.0]     bot health = current / max
# [1]      float [0.0; 1.0]     bot mana = current / max
# [2]      float [0.0; 1.0]     distance on minimap between gate and bot's icon = distance/100
# [3]      float [0.0; 1.0]     distance on screen between bot and closest BLUE minion = 1/distance
# [4]      float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
# [5..10]  6*int [0 or 1]       6*cooldown value
actions:  
#          18*int [0 or 1]      18 possible actions:
# [0..10]  'move_up', 'move_down', 'move_right', 'move_left', 'move_up-right', 'move_down-right', 'move_up-left', 'move_down-left', 
# [9..11]  'run_middle', 'collect_globes', 'hide_in_bushes',
# [12..14] 'basic_attack', 'q_attack', 'w_attack', 
# [15..17] 'use_well', 'hide_behind_gate', 'escape_behind_gate', 'use_spell_d'		
agent:
# policy = BoltzmannQPolicy()
# memory = SequentialMemory(limit=50000, window_length=1)
# agent = DQNAgent(...)
model:
# Flatten(input_shape=(1, 11))
# Dense(64, activation='relu')
# Dense(32, activation='relu')
# Dense(16, activation='relu')
# Dense(18, activation='linear')
# Flatten()
```
### Strict Learning
<p align="center">
<img src="https://github.com/pza007/game_bot/blob/main/gifs/strict_learning_plots.png" width="640" height="480">
</p>
This agent was trained for 8000 steps. The end of session could occur because timeout (60sec) or bot's death.
The rewards (upper graph, blue line) were designed to guide the agent to choose the most efficient actions:

```bash
reward += amount of experience gained during step
reward += 5 # if action was successful
reward += 50 # if action involved attack
reward += 15 # if action involved gathering experience
reward += 5 # if bot is close to blue minion
reward += 10 # if bot is close to red minion

reward -= 2000 # if bot is dead
reward -= 100 # if action was not successful
reward -= 1000 # for incorrect actions, based on situation (bot played defensive when it has more than 10% of health)
```
The highest user-defined reward for the game was 1771.
The highest game experience reward for the game was 1060 (episode 160 of training).
The lower graph represents the number of all (blue line), successful (green line) and failed (red line) steps taken for each episode.

### Free Learning
<p align="center">
<img src="https://github.com/pza007/game_bot/blob/main/gifs/free_learning_plots.png" width="640" height="480">
</p>
This agent was trained for 8000 steps. The end of session could occur because timeout (60sec) or bot's death.
The rewards (upper graph, green line) were based only on the game experience, gained during each episode:

```bash
reward += amount of experience gained during step
```
The highest game experience reward for the game was 1040 (episode 94 of training).
The lower graph represents the number of all (blue line), successful (green line) and failed (red line) steps taken for each episode.


## Programming AI
Alternative to machine learning is writing algorithm, using the game's knowledge.
Code below chooses the action to be taken by bot, based on the following principles:
```bash
if no red minion is available:
  1) 'hide_in_bushes' and wait

if red minions are available:
  * 1,2 minions: 'basic_attack'
  * 3>= minions: 'q_attack'
  * 4>= minions: 'w_attack'

every 10 sec:
  1) 'collect_globes'
  2) 'run_middle' + 'collect_globes'
  [finally] move behind the closest blue minion OR 'hide_in_bushes'

if bot mana < 5%:
  1) 'use_spell_d'
  2) 'use_well'

if bot health < 15%:
  1) 'use_spell_d'
  2) 'hide_behind_gate' + 'use_spell_d'
  3) 'escape_behind_gate' + 'use_spell_d'
  4) 'use_well'
```
<p align="center">
<img src="https://github.com/pza007/game_bot/blob/main/gifs/ai_plots.png" width="526" height="272">
</p>
The bot was tested for 10 episodes. The end of each game occurred after 60 sec (timeout).
The rewards (upper graph, green line) were based only on game experience gained during each episode.
The highest reward for the game was 1180 and happened in the first episode of testing.
The lower graph represents the number of all (blue line), successful (green line) and failed (red line) steps taken during each episode of testing.

## Results Summary
The total game experience, gathered by the bot, looks similar for each of the three techniques.
Unfortunately, some results were compromised by issue, observed at the end of the project.
Not-picked experience globes were not cleared, after the game was finished.
As a result, the bot from the next episode would collect the "experience from previous episode" and current episode.
This situation was clearly observed, when the ai-driven bot collected all globes available for the episode and the value of rewards were similar for each episode.
To overcome this issue, the waiting period would need to be implemented (around 1 min), after the game is finished.
Unfortunately, this change will greatly increase the learning time of agents.

My personal winner is the ai-driven bot. It maximizes the damage done to the minions, the success rate of taken action is 97% and it uses escape or recovery techniques when at risk of dying.

## Project Status
The project is: _no longer being worked on_. 
The basic functionality was successfully implemented. 
There is plenty room for improvements to enrich the bot with further functionalities.


## Room for Improvement
- Object Detection
  - detect how the camera view distorts the environment when the bot is moving and compensate it (camera is not located directly above the bot but also shifted in x,y,z axis)
  - detect experience globes and bigger "health/mana" globes
  - detect fortification structure: gate, towers, forts
- Control of bot
  - improve center of attack for W spell
  - implement new actions to fight against structures and heroes
- Machine Learning
  - improve type of model and(or) agent to get better results (higher value of game experience points)
- Other
  - improve resetting game's state -> no experience globes are available at the start of new episode

## Contact
Created by: przemyslaw.zawadzki@hotmail.com. Feel free to contact me!


## License
MIT

Copyright to all images and texts belongs to:
Heroes of the Storm™
©2014 Blizzard Entertainment, Inc. All rights reserved. Heroes of the Storm is a trademark of Blizzard Entertainment, Inc.