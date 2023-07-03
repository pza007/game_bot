import cv2 as cv
from image_objects_lib import ImageObjects, BotsClass, MinionsClass
from tracker_lib import TrackerClass
from global_vars import _SKIP_AREAS


class PainterClass:
    """
    - paint objects from ImageObjects()
    Note!! will change value of ImageObjects.img
    """

    def draw(self, img, obj, color, text_desc):
        # draw rectangle for bounding box
        [(x_min, y_min), (x_max, y_max)] = obj.bounding_box
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
        # draw circle for center
        cv.circle(img, (obj.point_center[0], obj.point_center[1]), 1, color, 2)
        # draw rectangle + text for object type and %health
        cv.rectangle(img, (x_min, y_max), (x_min + 100, y_max + 14), color, -1)
        cv.putText(
            img=img,
            text=text_desc,
            org=(x_min + 1, y_max + 12),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=0.45,
            color=(0, 0, 0),
            thickness=1)
        # draw rectangle + text for object ID
        # cv.rectangle(img, (x_max - 50, y_min), (x_max, y_min + 14), col, -1)
        # cv.putText(
        #     img=img,
        #     text='ID:' + str(track_id),
        #     org=(x_max-49, y_min+12),
        #     fontFace=cv.FONT_HERSHEY_COMPLEX,
        #     fontScale=0.45,
        #     color=(0, 0, 0),
        #     thickness=1)

    def draw_minimap(self, ImgObjs, img):
        [(x_min, y_min), (x_max, y_max)] = _SKIP_AREAS[3]
        # minimap
        img[y_min-16:y_max+16, x_min-16:x_max+4] = ImgObjs.minimap.minimap_img

        # minions
        for (x, y) in ImgObjs.minimap.objects['minions_blue']:
            cv.rectangle(img, (x-1, y-1), (x+1, y+1), (255, 0, 0), -1)
        for (x, y) in ImgObjs.minimap.objects['minions_red']:
            cv.rectangle(img, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)

        # bot
        x, y = ImgObjs.minimap.objects['bot']
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)  # inner circle
        cv.circle(img, (x, y), 3, (0, 0, 0), 1)  # outer circle

        return img

    def update(self):
        ImgObjs = ImageObjects()
        img = ImgObjs.img

        for group in ImgObjs.detected_objects:
            for obj in group.objects:
                # set parameters for type of object
                #   BOT
                if type(obj) == BotsClass.BotClass:
                    text_desc = 'BOT hp:' + str(obj.health) + '%'
                    col = (0, 255, 255)
                #   WELL
                # elif type(obj) == WellsClass.WellClass:
                #    text_desc = 'WELL hp:' + str(obj.health) + '%'
                #    col = (0, 255, 255)
                #   MINION
                elif type(obj) == MinionsClass.MinionClass:
                    text_desc = 'MN hp:' + str(obj.health) + '%'
                    if obj.color == 'red':
                        col = (0, 0, 255)
                    else:  # blue
                        col = (255, 0, 0)
                # draw
                self.draw(img, obj, col, text_desc)

        for obj in ImgObjs.minions_ghost.objects:
            text_desc = 'GHOST'
            col = (255, 255, 255)
            self.draw(img, obj, col, text_desc)

        img = self.draw_minimap(ImgObjs, img)

        return img


    @staticmethod
    def draw_minion_ghost(img, track_obj):
        obj = track_obj.object
        text_desc = 'GHOST'
        col = (255, 255, 255)

        # draw rectangle for bounding box
        [(x_min, y_min), (x_max, y_max)] = obj.bounding_box
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), col, 1)
        # draw circle for center
        cv.circle(img, (obj.point_center[0] - 1, obj.point_center[1] - 1), 2, col, 2)
        # draw rectangle + text for object type and %health
        cv.rectangle(img, (x_min, y_max), (x_min + 100, y_max + 14), col, -1)
        cv.putText(
            img=img,
            text=text_desc,
            org=(x_min + 1, y_max + 12),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=0.45,
            color=(0, 0, 0),
            thickness=1)
        # draw rectangle + text for object ID
        # cv.rectangle(img, (x_max - 50, y_min), (x_max, y_min + 14), col, -1)
        # cv.putText(
        #     img=img,
        #     text='ID:' + str(track_id),
        #     org=(x_max-49, y_min+12),
        #     fontFace=cv.FONT_HERSHEY_COMPLEX,
        #     fontScale=0.45,
        #     color=(0, 0, 0),
        #     thickness=1)

        return img

    @staticmethod
    def draw_minion(img, track_obj):
        obj = track_obj.object
        text_desc = 'MN hp:' + str(obj.health) + '%'
        if obj.color == 'red':
            col = (0, 0, 255)
        else:  # blue
            col = (255, 0, 0)

        # draw rectangle for bounding box
        [(x_min, y_min), (x_max, y_max)] = obj.bounding_box
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), col, 1)
        # draw circle for center
        cv.circle(img, (obj.point_center[0] - 1, obj.point_center[1] - 1), 2, col, 2)
        # draw rectangle + text for object type and %health
        cv.rectangle(img, (x_min, y_max), (x_min + 100, y_max + 14), col, -1)
        cv.putText(
            img=img,
            text=text_desc,
            org=(x_min + 1, y_max + 12),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=0.45,
            color=(0, 0, 0),
            thickness=1)
        # draw rectangle + text for object ID
        # cv.rectangle(img, (x_max - 50, y_min), (x_max, y_min + 14), col, -1)
        # cv.putText(
        #     img=img,
        #     text='ID:' + str(track_id),
        #     org=(x_max-49, y_min+12),
        #     fontFace=cv.FONT_HERSHEY_COMPLEX,
        #     fontScale=0.45,
        #     color=(0, 0, 0),
        #     thickness=1)

        return img

    @staticmethod
    def draw_bot(img, track_obj):
        obj = track_obj.object
        text_desc = 'BOT hp:'+str(obj.health)+'%'
        col = (0, 255, 255)

        # draw rectangle for bounding box
        [(x_min, y_min), (x_max, y_max)] = obj.bounding_box
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), col, 1)
        # draw rectangle + text for object type and %health
        cv.rectangle(img, (x_min, y_max), (x_min+100, y_max+14), col, -1)
        cv.putText(
            img=img,
            text=text_desc,
            org=(x_min+1, y_max+12),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=0.45,
            color=(0, 0, 0),
            thickness=1)
        # draw rectangle + text for object ID
        # cv.rectangle(img, (x_max - 50, y_min), (x_max, y_min + 14), col, -1)
        # cv.putText(
        #     img=img,
        #     text='ID:' + str(track_id),
        #     org=(x_max-49, y_min+12),
        #     fontFace=cv.FONT_HERSHEY_COMPLEX,
        #     fontScale=0.45,
        #     color=(0, 0, 0),
        #     thickness=1)

        return img
