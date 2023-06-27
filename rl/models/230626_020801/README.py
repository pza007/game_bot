num_steps = 5000
game_timeout = 30 sec

def get_reward(self, step_reward):
	"""
	out: reward
	"""
	def get_dist_change(prev_vector, curr_vector):
		if prev_vector != (0, 0) and curr_vector != (0, 0):
			prev_dist = math.sqrt(prev_vector[0] ** 2 + prev_vector[1] ** 2)
			curr_dist = math.sqrt(curr_vector[0] ** 2 + curr_vector[1] ** 2)
			return curr_dist - prev_dist
		else:
			return None

	#self._observation = [0]*16
	# (uint), (uint)        bot health current, max
	# (uint), (uint)        bot mana current, max
	# (int16), (int16)      vector (dx,dy) from gate to bot's icon  !(minimap)!
	# (int16), (int16)      vector (dx,dy) from bot to the closest blue minion (screen)
	# (int16), (int16)      vector (dx,dy) from bot to the closest red minion (screen)
	# (0,0,0,0,0,0)         bot cooldowns
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
	# bot is behind blue minion
	blue_minion = self._observation[6:8]    # (dx, dy)
	if None not in blue_minion and blue_minion[0] > 0:
		reward += 5
	# bot is close to red minion
	red_minion = self._observation[8:10]    # (dx, dy)
	if None not in red_minion:
		reward += 10
	# SPEED
	#if self._action_name is not None and step_reward is not None:
	#    t_max = self.spectator.actions.objects[self._action_name].TIMEOUT
	#    t0 = self.spectator.actions.objects[self._action_name].t0
	#    t_diff = time.time() - t0
	#    factor = (t_max - t_diff) / (3*t_max)
	#    reward += factor*step_reward    # short action with big xp_gain is promoted


	# ------ REWARD
	# bot is dead
	if self._done_reason is not None and self._done_reason == 'bot_dead':
		reward -= 1000
	# action was not successful
	if self._action_result is not None and self._action_result < 0:
		reward -= 15
	# bot was close to red minion but now he lost it from sight
	prev_red_minion = self._prev_observation[8:10]    # (dx, dy)
	red_minion = self._prev_observation[8:10]    # (dx, dy)
	if None not in prev_red_minion and None in red_minion:
		reward -= 50
	# bot moved behind the gate, even though bots'HP > 10%
	if None not in [self._action_name, self._observation[0], self._observation[1]]:
		if self._action_name in ['hide_behind_gate', 'escape_behind_gate'] \
				and self._observation[0] / self._observation[0] > 0.1:
			reward -= 500

	"""
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