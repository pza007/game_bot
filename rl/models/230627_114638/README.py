num_steps = 10000
game_timeout = 60sec

    def get_reward(self, step_reward):
        #self._observation = [0]*11
        # out[0]      float [0.0; 1.0]     bot health = current / max
        # out[1]      float [0.0; 1.0]     bot mana = current / max
        # out[2]      float [0.0; 1.0]     distance on minimap between gate and bot's icon = distance/100
        # out[3]      float [0.0; 1.0]     distance on screen between bot and closest BLUE minion = 1/distance
        # out[4]      float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        # out[5..10]  6*int [0 or 1]       6*cooldown value
        reward = 0

        # ++++++ REWARD
        # xp was gained
        if step_reward is not None:
            reward += step_reward
        # ACTION
        if self._action is not None:
            # action was successful
            if self._action_result > 0:
                reward += 5
                # action involved attack
                if self._action_name in ['basic_attack', 'q_attack', 'w_attack']:
                    reward += 50
                # action involved gathering xp
                if self._action_name in ['collect_globes', 'run_middle', 'hide_in_bushes']:
                    reward += 15
        # POSITION
        # bot is close to minions
        dist_blue = self._observation[3]  # float [0.0; 1.0]
        reward += min(5, 5*dist_blue*200)
        dist_red = self._observation[4]  # float [0.0; 1.0]
        reward += min(10, 10*dist_red*200)
        # SPEED
        #if self._action_name is not None and step_reward is not None:
        #    t_max = self.spectator.actions.objects[self._action_name].TIMEOUT
        #    t0 = self.spectator.actions.objects[self._action_name].t0
        #    t_diff = time.time() - t0
        #    factor = (t_max - t_diff) / (3*t_max)
        #    reward += factor*step_reward    # short action with big xp_gain is promoted


        # ------ REWARD
        # bot is dead
        if self._done_reason == 'bot_dead':
            reward -= 2000
        # action was not successful
        if self._action_result is not None and self._action_result < 0:
            reward -= 100
        # 'incorrect' actions, based on situation (bot played defensive when it has more than 10% of health)
        if self._action_name in ['hide_behind_gate', 'escape_behind_gate', 'use_well', 'use_spell_d'] \
                and self._observation[0] > 0.1:
            reward -= 1000
        # bot was close to RED minion but currently lost it from screen
        #prev_dist = self._prev_observation[4]  # float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        #curr_dist = self._observation[4]  # float [0.0; 1.0]     distance on screen between bot and closest RED minion = 1/distance
        #if prev_dist > 1/SCREEN_DIAG > curr_dist:  # 1/SCREEN_DIAG used, to compare to value 0.0
        #    reward -= 50
        # bot moved behind the gate, even though bots'HP > 10%
        #if self._action_name in ['hide_behind_gate', 'escape_behind_gate'] and self._observation[0] > 0.1:
        #    reward -= 500

        # OBSOLETE code
        """
        def get_dist_change(prev_vector, curr_vector):
            if prev_vector != (0, 0) and curr_vector != (0, 0):
                prev_dist = math.sqrt(prev_vector[0] ** 2 + prev_vector[1] ** 2)
                curr_dist = math.sqrt(curr_vector[0] ** 2 + curr_vector[1] ** 2)
                return curr_dist - prev_dist
            else:
                return None
                
        # COMPARE TO PREVIOUS OBSERVATION
        #   bot moved closer to BLUE minion or away from it?
        prev_vector = self._prev_observation[6:8]   # (dx, dy)
        curr_vector = self._observation[6:8]        # (dx, dy)
        dist_change = get_dist_change(prev_vector, curr_vector)
        if dist_change is not None:
            if dist_change > 0:   # moved closer
                reward += 8
            else:   # moved away
                reward -= 12
        #   bot moved closer to RED minion or away from it?
        prev_vector = self._prev_observation[8:10]   # (dx, dy)
        curr_vector = self._observation[8:10]        # (dx, dy)
        dist_change = get_dist_change(prev_vector, curr_vector)
        if dist_change is not None:
            if dist_change > 0:   # moved closer
                reward += 10
            else:   # moved away
                reward -= 15

        # bot's health did not decrease?
        prev_health = self._prev_observation[0:2]   # (curr, max)
        health = self._observation[0:2]   # (curr, max)
        if prev_health != (0, 0) and health != (0, 0):
            if health[0]/health[1] >= prev_health[0]/prev_health[1]:
                reward += 5
            else: reward -= 5

        # bot moved closer to blue minion?
        prev_vct = self._prev_observation[6:8]   # (dx, dy)
        vct = self._observation[6:8]   # (dx, dy)
        if prev_vct != (0, 0) and vct != (0, 0):
            prev_dist = math.sqrt(prev_vct[0] ** 2 + prev_vct[1] ** 2)
            dist = math.sqrt(vct[0] ** 2 + vct[1] ** 2)
            if dist < prev_dist:
                reward += 5
            #else: reward -= 5

        # bot moved closer to red minion?
        prev_vct = self._prev_observation[8:10]   # (dx, dy)
        vct = self._observation[8:10]   # (dx, dy)
        if prev_vct != (0, 0) and vct != (0, 0):
            prev_dist = math.sqrt(prev_vct[0] ** 2 + prev_vct[1] ** 2)
            dist = math.sqrt(vct[0] ** 2 + vct[1] ** 2)
            if dist < prev_dist:
                reward += 10
            #else: reward -= 10
        """

        return reward