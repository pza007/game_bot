import cv2 as cv
import numpy as np
from image_objects import BotClass, MinionsClass


def detect_objects(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    bot = BotClass(img, hsv)
    bot.get_from_image()

    minions_red = MinionsClass(img, hsv, 'red')
    minions_red.get_from_image()

    minions_blue = MinionsClass(img, hsv, 'blue')
    minions_blue.get_from_image()

    return bot, minions_red, minions_blue


def draw_objects(img, bot, minions_red, minions_blue):
    out_img = bot.draw(img)
    if out_img is None:
        out_img = img

    out_img = minions_red.draw(out_img)
    if out_img is None:
        out_img = img

    out_img = minions_blue.draw(out_img)
    if out_img is None:
        out_img = img

    return out_img


def save_image(img, file_path):
    cv.imwrite(file_path, img)
