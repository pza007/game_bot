import numpy as np
from image_objects_lib import ImageObjects


class TrackerClass:
    def __init__(self):
        self.curr = {
            'bot': None,                # self.ObjectClass()
            'minions_blue': [],         # [self.ObjectClass(), ...]
            'minions_red': [],          # [self.ObjectClass(), ...]
        }
        self.prev = {
            'bot': None,                # self.ObjectClass()
            'minions_blue': [],         # [self.ObjectClass(), ...]
            'minions_red': [],          # [self.ObjectClass(), ...]
        }

    class ObjectClass:
        screen_pos = (None, None)
        minimap_pos = (None, None)

    def update(self):
        def set_curr():
            # bot
            obj = self.ObjectClass()
            obj.screen_pos = ImgObjs.bot.objects[0].point_center    # (x, y)
            obj.minimap_pos = ImgObjs.minimap.objects['bot']        # (x, y)
            self.curr['bot'] = obj

            # minions_blue
            screen_minions = ImgObjs.minions_blue.objects
            minimap_minions = ImgObjs.minimap.objects['minions_blue']
            out = []
            for i in range(len(screen_minions)):
                obj = self.ObjectClass()
                obj.screen_pos = screen_minions[i].point_center    # (x, y)
                obj.minimap_pos = minimap_minions[i]  # (x, y)
                out.append(obj)
            self.curr['minions_blue'] = out

            # minions_blue
            screen_minions = ImgObjs.minions_red.objects
            minimap_minions = ImgObjs.minimap.objects['minions_red']
            out = []
            for i in range(len(screen_minions)):
                obj = self.ObjectClass()
                obj.screen_pos = screen_minions[i].point_center    # (x, y)
                obj.minimap_pos = minimap_minions[i]  # (x, y)
                out.append(obj)
            self.curr['minions_red'] = out

        ImgObjs = ImageObjects()
        set_curr()
        """
        for minion_type in ['minions_blue', 'minions_red']:
            # number of detected minions decreased?
            diff = len(self.prev[minion_type]) - len(self.curr[minion_type])
            if diff > 0:
                # find 'lost' minion(s)
                out = []    # [[ObjectClass(), distance], ...]
                for obj_prev in self.prev[minion_type]:
                    distances = []
                    for obj_curr in self.curr[minion_type]:
                        # distance on minimap
                        x_prev, y_prev = obj_prev.minimap_pos
                        x_curr, y_curr = obj_curr.minimap_pos
                        distances.append(get_distance_between_points((x_prev, y_prev), (x_curr, y_curr)))
                    # in case there are no current minions
                    if len(distances) == 0:
                        distances = [9999]
                    obj = self.ObjectClass()
                    obj.screen_pos = obj_prev.screen_pos
                    obj.minimap_pos = obj_prev.minimap_pos
                    out.append([obj, min(distances)])
                out.sort(key=lambda values: values[1], reverse=True)
                lost = [obj for [obj, dist] in out][:diff]  # [ObjectClass(), ...]
                print(f'lost={lost}')

                # if 'lost' minion(s) is out of bot's vision -> do not evaluate (remove from lost)
                out = []
                for obj_prev in lost:
                    # distance on minimap
                    x_prev, y_prev = obj_prev.minimap_pos
                    x_bot, y_bot = self.curr['bot'].minimap_pos
                    dist = get_distance_between_points((x_prev, y_prev), (x_bot, y_bot))
                    if dist < 38:
                        out.append(obj_prev)
                lost = out  # [ObjectClass(), ...]
                print(f'lost={lost}')

                # add 'lost' minions to MinionsGhost
                bot = self.curr['bot']
                minions_ghost = ImgObjs.minions_ghost
                minions_ghost.add_objects(bot, lost)


        # update minions ghost
        minions_ghost = ImgObjs.minions_ghost
        minions_ghost.update_objects(self.curr['bot'])
        """

        for minion_type in ['minions_red']:
            # number of detected minions decreased?
            diff = len(self.prev[minion_type]) - len(self.curr[minion_type])
            if diff > 0:
                # search mana inside bounding boxes of minions ghost
                x_all, y_all = [], []
                for obj in self.prev['minions_red']:
                    x, y = obj.screen_pos
                    x_all.append(x-60)
                    x_all.append(x+60)
                    y_all.append(y-40)
                    y_all.append(y+40)
                for obj in self.curr['minions_red']:
                    x, y = obj.screen_pos
                    x_all.append(x-60)
                    x_all.append(x+60)
                    y_all.append(y-40)
                    y_all.append(y+40)
                x_min, y_min, x_max, y_max = min(x_all), min(y_all), max(x_all), max(y_all)
                bounding_box = [(x_min, y_min), (x_max, y_max)]

                manas = ImgObjs.manas
                manas.get_from_image(ImgObjs.img, ImgObjs.hsv, bounding_box)


        # prev = curr
        for key, value in self.curr.items():
            self.prev.update({key: value})

        """
        ImgObjs = ImageObjects()
        in_bot = ImgObjs.bot
        in_minions_red, in_minions_blue = ImgObjs.minions_red, ImgObjs.minimap_minions_blue

        # bot not detected or minions not detected
        if len(in_bot.objects) == 0 or \
                len(in_minions_red.objects) == 0 and len(in_minions_blue.objects) == 0:
            self.reset()
            return

        # MINIONS
        #found_ghosts = []
        for in_group in [in_minions_red, in_minions_blue]:
            track_group = self.get_track_group(in_group)
            # number of detected minions is the same or increased? -> update track_group
            if len(in_group.objects) >= len(track_group):
                track_group.clear()
                for in_obj in in_group.objects:
                    new_obj = self.TrackObjectClass(in_obj)
                    track_group.append(new_obj)
        """

        """
            # number of detected minions decreased -> found ghost!
            else:
                # find out which object is 'missing'
                self.get_min_distances(track_group, in_group)
                for track_object in track_group:
                    if track_object.min_dist > self._DIST_THR:
                        # add to ghosts
                        self.ghosts.add(track_object.object, self.ghost_id)
                        found_ghosts.append(self.ghost_id)
                        self.ghost_id += 1
                        #print('\t found ghost!')
                # update track_group
                track_group.clear()
                for in_obj in in_group.objects:
                    new_obj = self.TrackObjectClass(in_obj)
                    track_group.append(new_obj)
        """

        # GHOSTS
        #self.ghosts.detect_inner_objects()
        #self.ghosts.update(found_ghosts)


""" previous TrackerClass """
# class TrackerClass:
#     def __init__(self):
#         self.tr_minions_red = []     # list of TrackObjectClass
#         self.tr_minions_blue = []    # list of TrackObjectClass
#         self.ghosts = self.GhostsClass()
#
#         self.ghost_id = 0
#
#     _DIST_THR = 35  # pixels
#     _GHOST_TIMEOUT = 3.0  # seconds
#
#     class TrackObjectClass:
#         def __init__(self, obj):
#             self.object = obj
#             self.min_dist = None
#
#     class GhostsClass:
#         def __init__(self):
#             self.objects = []   # to fit structure from image_objects_lib.py
#             self.identifiers = []
#             self.timestamps = []
#
#         def add(self, obj, id_value):
#             self.objects.append(obj)
#             self.identifiers.append(id_value)
#             self.timestamps.append(time.time())
#
#         def detect_inner_objects(self):
#             ImgObjs = ImageObjects()
#
#             # Mana collectibles
#             ghost_boxes = [ghost_obj.bounding_box for ghost_obj in self.objects]  # <int> [(x_min, y_min), (x_max, y_max)]
#             ImgObjs.manas_blue.get_from_image(ImgObjs.img, ImgObjs.hsv, ghost_boxes)
#             print(f'mana={ImgObjs.manas_blue.objects}')
#
#
#
#         def update(self, found_ids):
#             """
#             - check if ghost reach timeout -> delete
#             - update ImageObjects()
#             """
#             remove_idx = []
#             for idx, id_val in enumerate(self.identifiers):
#                 if id_val not in found_ids:
#                     # timeout? -> remove 'old' ghost object
#                     time_diff = time.time() - self.timestamps[idx]
#                     if time_diff >= TrackerClass._GHOST_TIMEOUT:
#                         remove_idx.append(idx)
#             # reassign
#             self.objects =      [obj for idx, obj in enumerate(self.objects) if idx not in remove_idx]
#             self.identifiers =  [obj for idx, obj in enumerate(self.identifiers) if idx not in remove_idx]
#             self.timestamps =   [obj for idx, obj in enumerate(self.timestamps) if idx not in remove_idx]
#
#             # update ImageObjects
#             ImgObjs = ImageObjects()
#             ImgObjs.minions_ghost = self
#
#
#     # get screen shift:
#     #   select point, click point to move
#     #   when bot is stopped -> press space + calculate screen shift
#
#     @staticmethod
#     def get_min_distances(track_group, in_group):
#         """
#         calculate the change in distance between old(track_group) and new(in_group) objects
#         save the minimal one
#         """
#         for track_object in track_group:
#             distances = []
#             for in_obj in in_group.objects:
#                 distances.append(get_distance_between_points(track_object.object.point_center, in_obj.point_center))
#             if len(distances) == 0:
#                 # no objects detected (in_group is empty)
#                 track_object.min_dist = 999999  # max value
#             else:
#                 track_object.min_dist = min(distances)
#
#     def get_track_group(self, in_group):
#         # Minions
#         if type(in_group) == MinionsClass:
#             if in_group.color == 'red':
#                 return self.tr_minions_red
#             else:              # 'blue'
#                 return self.tr_minions_blue
#         # Bot
#         else:
#             return self.bot
#
#     def reset(self):
#         self.tr_minions_red.clear()
#         self.tr_minions_blue.clear()
#         self.ghosts.update([])
#
#     def update(self):
#         """
#         Minions: detect "ghosts" by decreased number of detected minions, update ImageObjects()
#         """
#         ImgObjs = ImageObjects()
#         in_bot = ImgObjs.bot
#         in_minions_red, in_minions_blue = ImgObjs.minions_red, ImgObjs.minions_blue
#
#         # bot not detected or minions not detected
#         if len(in_bot.objects) == 0 or \
#                 len(in_minions_red.objects) == 0 and len(in_minions_blue.objects) == 0:
#             self.reset()
#             return
#
#         # MINIONS
#         #found_ghosts = []
#         for in_group in [in_minions_red, in_minions_blue]:
#             track_group = self.get_track_group(in_group)
#             # number of detected minions is the same or increased? -> update track_group
#             if len(in_group.objects) >= len(track_group):
#                 track_group.clear()
#                 for in_obj in in_group.objects:
#                     new_obj = self.TrackObjectClass(in_obj)
#                     track_group.append(new_obj)
#
#             """
#             # number of detected minions decreased -> found ghost!
#             else:
#                 # find out which object is 'missing'
#                 self.get_min_distances(track_group, in_group)
#                 for track_object in track_group:
#                     if track_object.min_dist > self._DIST_THR:
#                         # add to ghosts
#                         self.ghosts.add(track_object.object, self.ghost_id)
#                         found_ghosts.append(self.ghost_id)
#                         self.ghost_id += 1
#                         #print('\t found ghost!')
#                 # update track_group
#                 track_group.clear()
#                 for in_obj in in_group.objects:
#                     new_obj = self.TrackObjectClass(in_obj)
#                     track_group.append(new_obj)
#             """
#
#         # GHOSTS
#         #self.ghosts.detect_inner_objects()
#         #self.ghosts.update(found_ghosts)

""" old TrackerClass """
# class OldTrackObjectClass:
#     def __init__(self):
#         self.object = None
#         self.time_last_update = None
#         self.is_ghost = None
#         self.dist_angle = None
#
#
# class OldTrackerClass:
#     def __init__(self):
#         self.nextObjectID = 0
#         self.dist_thr = 30  # pixels
#         self.ghost_timeout = 3.0  # sec
#
#         self.bot = {}           # {ID: OldTrackObjectClass}
#         self.minions_red = {}   # {ID: OldTrackObjectClass}
#         self.minions_blue = {}  # {ID: OldTrackObjectClass}
#         self.all = [self.bot, self.minions_red, self.minions_blue]
#
#     def calc_dist_angle(self, in_group, i):
#         dist_angle = []
#         for j in range(len(in_group.objects)):
#             if i != j:
#                 point_i = in_group.objects[i].point_center
#                 point_j = in_group.objects[j].point_center
#                 dist = get_distance_between_points(point_i, point_j)
#                 new_point_j = (point_j[0]+point_i[0]*-1, point_j[1]+point_i[1]*-1)
#                 angle = angle_between((0, 0), new_point_j)
#                 dist_angle.append((dist, angle))
#         dist_angle.sort(key=lambda x: x[0])
#         #print(i, [('%.02f'%dist, '%.02f'%angle) for dist, angle in dist_angle])
#         return dist_angle
#
#     def calc_angle_diff(self, angle1, angle2):
#         diff1 = abs(angle1 - angle2)
#         if 360-angle1 < 360-angle2:
#             diff2 = 360-angle1 + angle2
#         else:
#             diff2 = 360-angle1 + angle1
#         return min([diff1, diff2])
#
#     def update_minions(self, in_group):
#         out_dict = self.get_dictionary(in_group)
#         if len(out_dict) == 0:
#             for i in range(len(in_group.objects)):
#                 in_minion = in_group.objects[i]
#                 new_obj = OldTrackObjectClass()
#                 new_obj.object = in_minion
#                 new_obj.time_last_update = time()
#                 new_obj.is_ghost = False
#                 new_obj.dist_angle = self.calc_dist_angle(in_group, i)
#                 new_id = self.get_next_id()
#                 out_dict.update({new_id: new_obj})
#             return
#
#         # -- debug only
#         #all_ids = []
#         #for dictionary in self.all:
#         #    for obj_id in list(dictionary.keys()):
#         #        all_ids.append(obj_id)
#         #print('all_ids', all_ids)
#         # -- debug only
#
#         # try to find match with current objects
#         dist_thr = 20       # 5
#         angle_thr = 30      # 10
#         found_ids = []
#         for i in range(len(in_group.objects)):
#             obj = in_group.objects[i]
#             in_dist_angle = self.calc_dist_angle(in_group, i)
#             if len(in_dist_angle) == 0:
#                 # only one object detected -> update recently added object
#                 #self.nextObjectID
#                 pass
#
#             matches = []    # (id, number of found connections)
#             for track_id, track_obj in out_dict.items():
#                 num_connections = 0
#                 for dist, angle in track_obj.dist_angle:
#                     for in_dist, in_angle in in_dist_angle:
#                         if abs(in_dist-dist) <= dist_thr and self.calc_angle_diff(in_angle, angle) <= angle_thr:
#                             num_connections += 1
#                             break
#                 matches.append((track_id, num_connections))
#                 #print(f'\tID={track_id}, connections={num_connections}')
#
#             matches.sort(key=lambda x: x[1], reverse=True)
#             track_id, num_connections = matches[0]
#
#             if num_connections > 0:
#                 print(f'Input: {i}, num_conn={num_connections}, ID={track_id}')
#                 # found existing object -> update its values
#                 old_obj = out_dict[track_id]
#                 old_obj.object = obj
#                 old_obj.time_last_update = time()
#                 old_obj.is_ghost = False
#                 old_obj.dist_angle = in_dist_angle
#                 out_dict.update({track_id: old_obj})
#                 found_ids.append(track_id)
#
#             else:
#                 print(f'Input: {i}, num_conn={num_connections}, new object!')
#                 # cannot match existing object -> assign new ID, add to dictionary
#                 new_obj = OldTrackObjectClass()
#                 new_obj.object = obj
#                 new_obj.time_last_update = time()
#                 new_obj.is_ghost = False
#                 new_obj.dist_angle = in_dist_angle
#                 new_id = self.get_next_id()
#                 out_dict.update({new_id: new_obj})
#                 found_ids.append(new_id)
#
#         #print('found_ids=', found_ids)
#
#         # -----------------------------------------------------------------------------------------------
#         # check existing group
#         for obj_id in list(out_dict.keys()):
#             if obj_id not in found_ids and not out_dict[obj_id].is_ghost:
#                 # object not detected -> change 'is_ghost' flag
#                 out_dict[obj_id].is_ghost = True
#                 # print(f'--- ID={obj_id} not detected -> mark as ghost')
#
#             t_diff = time() - out_dict[obj_id].time_last_update
#             # timeout ghost -> free it's ID and remove from dictionary
#             if out_dict[obj_id].is_ghost and t_diff >= self.ghost_timeout:
#                 # print(f'--- ID={obj_id} timeout -> delete')
#                 self.nextObjectID = obj_id
#                 del out_dict[obj_id]
#
#
#     def update(self, in_group):
#         """
#         :param   in_group:  bot, minions_red, minions_blue
#         """
#         # check input group
#         out_dict = self.get_dictionary(in_group)
#         #   first update?
#         if len(out_dict) == 0:
#             out_dict = self.first_update(in_group, out_dict)
#             return
#         #   calculate the distances between input objects to common existing objects
#         out_list = self.get_min_distances(in_group, out_dict)
#         #   try to assign ID
#         found_ids = []
#         for (obj, min_dist, id_candidate) in out_list:
#             # found existing object -> update its values
#             if min_dist <= self.dist_thr:
#                 old_obj = out_dict[id_candidate]
#                 old_obj.object = obj
#                 old_obj.time_last_update = time()
#                 old_obj.is_ghost = False
#                 out_dict.update({id_candidate: old_obj})
#                 found_ids.append(id_candidate)
#             # cannot match existing object -> assign new ID, add to dictionary
#             else:
#                 new_obj = OldTrackObjectClass()
#                 new_obj.object = obj
#                 new_obj.time_last_update = time()
#                 new_obj.is_ghost = False
#                 new_id = self.get_next_id()
#                 out_dict.update({new_id: new_obj})
#                 found_ids.append(new_id)
#         # -----------------------------------------------------------------------------------------------
#
#         # check existing group
#         for obj_id in list(out_dict.keys()):
#             if obj_id not in found_ids and not out_dict[obj_id].is_ghost:
#                 # object not detected -> change 'is_ghost' flag
#                 out_dict[obj_id].is_ghost = True
#                 #print(f'--- ID={obj_id} not detected -> mark as ghost')
#
#             t_diff = time() - out_dict[obj_id].time_last_update
#             # timeout ghost -> free it's ID and remove from dictionary
#             if out_dict[obj_id].is_ghost and t_diff >= self.ghost_timeout:
#                 #print(f'--- ID={obj_id} timeout -> delete')
#                 self.nextObjectID = obj_id
#                 del out_dict[obj_id]
#
#     def get_dictionary(self, in_group):
#         """
#         :param   in_group:      bot, minions_red, minions_blue
#         :return: self.minions_red or self.minions_blue or self.bot
#         """
#         # Minions
#         if type(in_group) == MinionsClass:
#             if in_group.color == 'red':
#                 return self.minions_red
#             else:              # 'blue'
#                 return self.minions_blue
#         # Bot
#         else:
#             return self.bot
#
#     def first_update(self, in_group, dictionary):
#         # Minions
#         if type(in_group) == MinionsClass:
#             for in_minion in in_group.objects:  # <list of MinionClass>
#                 new_obj = OldTrackObjectClass()
#                 new_obj.object = in_minion
#                 new_obj.time_last_update = time()
#                 new_obj.is_ghost = False
#                 new_id = self.get_next_id()
#                 dictionary.update({new_id: new_obj})
#         # Bot
#         else:
#             new_obj = OldTrackObjectClass()
#             new_obj.object = in_group
#             new_obj.time_last_update = time()
#             new_obj.is_ghost = False
#             new_id = self.get_next_id()
#             dictionary.update({new_id: new_obj})
#
#         return dictionary
#
#     def get_min_distances(self, in_group, dictionary):
#         """
#         :param   in_group:      bot, minions_red, minions_blue
#         :param   dictionary:    self.minions_red or self.minions_blue or self.bot
#         :return: out_list:      [(object, min_dist, id candidate),  ]
#         """
#         out_list = []    # (object, min_dist, id candidate)
#
#         # Minions
#         if type(in_group) == MinionsClass:
#             for in_minion in in_group.objects:   # <list of MinionClass>
#                 distances, ids = [], []
#                 for track_id, track_object in dictionary.items():  # {ID: OldTrackObjectClass}
#                     distances.append(get_distance_between_points(in_minion.point_center, track_object.object.point_center))
#                     ids.append(track_id)
#                 min_dist_val = min(distances)
#                 id_val = ids[distances.index(min_dist_val)]
#                 out_list.append((in_minion, min_dist_val, id_val))
#
#         # Bot
#         else:
#             if in_group.point_center is not None:
#                 id_val = list(dictionary.keys())[0]
#                 min_dist_val = get_distance_between_points(in_group.point_center, dictionary[id_val].object.point_center)
#                 out_list.append((in_group, min_dist_val, id_val))
#
#         return out_list
#
#     def get_next_id(self):
#         all_ids = []
#         for dictionary in self.all:
#             for obj_id in list(dictionary.keys()):
#                 all_ids.append(obj_id)
#         # ensure to include 0 to force to use low numbers
#         if len(all_ids) > 0 and all_ids[0] != 0:
#             all_ids = [0] + all_ids
#
#         if self.nextObjectID not in all_ids:
#             return self.nextObjectID
#         else:
#             # find free ID in between current values
#             a = np.array(all_ids)
#             a2 = np.sort(a)
#             a3 = np.diff(a2)
#             a4 = np.where(a3 > 1)[0]
#             if len(a4) > 0:
#                 idx = a4[0]
#                 self.nextObjectID = a2[idx]+1
#             else:
#                 # pick highest + 1
#                 self.nextObjectID = a2[-1] + 1
#             return self.nextObjectID
#
#     def print_all(self):
#         print('Bot:')
#         if len(self.bot) > 0:
#             obj = self.bot[list(self.bot.keys())[0]]
#             print(f'\t ID={list(self.bot.keys())[0]}, center={obj.object.point_center}, ghost={obj.is_ghost}')
#         else:
#             print('\t-')
#
#         print('Minions RED:')
#         if len(self.minions_red) > 0:
#             for track_id, track_object in self.minions_red.items():
#                 print(f'\t ID={track_id},  center={track_object.object.point_center}, ghost={track_object.is_ghost}')
#         else:
#             print('\t-')
#
#         print('Minions BLUE:')
#         if len(self.minions_blue) > 0:
#             for track_id, track_object in self.minions_blue.items():
#                 print(f'\t ID={track_id},  center={track_object.object.point_center}, ghost={track_object.is_ghost}')
#         else:
#             print('\t-')


def get_distance_between_points(point_1, point_2):
    return np.sqrt((point_2[0]-point_1[0])**2 + (point_2[1]-point_1[1])**2)


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))  # angle 0-360

