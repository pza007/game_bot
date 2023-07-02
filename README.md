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
* [Programming AI](#programming-ai)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)
* [License](#license)


## General Information
Idea to start this project came to me after playing the online video game - "Heroes of the Storm™". 
In the game, players form into five-player teams and fight against another team.
Team consists of characters different roles.
One of the tasks of player is to control the situation on map's lane, by killing minions and collecting experience globes.
I wanted to automize this process and started a project to create robot player (bot) that will do it for me, as best as possible and without external help. 

As the project envolved, I've divided it into following sections:
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
- Controlling movements and action of bot to gather game's experience
- Executing escape maneuvers when bot's health is getting low and could die


## Setup
1. Create the account on: battle.net 
2. Download and install the game: "Heroes of the Storm™"
3. Start game's launcher, start game
4. Choose: "Collections" then "Heroes". Select hero: "Mei" and choose "Try" button 
5. In-game: choose Ally Hero to "None" and Enemy Hero to "None" 
6. Clone this repository (git clone https://github.com/pza007/game_bot.git)
7. Install all packages defined in requirements.txt 
8. Run the project while still in game


## Object Detection
Before object detection:
<p align="center">
![](https://github.com/pza007/game_bot/blob/main/gifs/game_run.gif)
</p>

After object detection:
<p align="center">
![](https://github.com/pza007/game_bot/blob/main/gifs/game_run_with_labels.gif)
</p>

To start the object detection i've captured each frame of game's screen using _win32com_ package.
Using _opencv_ package, I've managed to detect position of bot and minions. 
Please see below the example algorithm to detect position of the minion, based on position of it's health-bar:
1. Convert RGB fame to HSV frame
2. Create image mask, based on threshold color
3. Transform mask (morphology)
4. Detect contours of objects in the mask
5. Execute filters:
   - filter contours with 4 points (rectangle)
   - filter contours with correct dimensions
   - filter contours located outside skip areas
   - filter contours with black border outside

Using combination of techniques mentioned above and 'matchTemplate' function, the following information can be extracted from single frame:
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
TODO

## Machine learning
TODO

## Programming AI
TODO


## Project Status
The project is: _no longer being worked on_. 
The basic fuctionality was successfully implemented. 
There is plenty room for improvements to enrich the bot with further functionalities.


## Room for Improvement
- Object Detection
  - detect how the camera view distorts the environment when the bot is moving and compensate it (camera is not located directly above the bot but also shifted in x,y,z axis)
  - detect experience globes, big health-mana globes
  - detect fortification structure: gate, towers, forts
- 


## Contact
Created by: przemyslaw.zawadzki@hotmail.com. Feel free to contact me!


## License
MIT

Copyright to all images and texts belongs to:
Heroes of the Storm™
©2014 Blizzard Entertainment, Inc. All rights reserved. Heroes of the Storm is a trademark of Blizzard Entertainment, Inc.