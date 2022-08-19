
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/

import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE=10
HM_EPISODES=10000
MAX_ITER = 200 # actually the no of steps the player can take
SHOW_EVERY=200 # change to 1 if start_q_table has a filepath

MOVE_PENALTY=1
ENEMY_PENALTY=300
FOOD_REWARD=25

epsilon=1 # change to 0.0 if start_q_table has a filepath
EPSILON_DECAY=0.9998

start_q_table = None # or a filename, you can throw a q_Table already saved, and train from there

LEARNING_RATE=0.1 # alpha
DISCOUNT=0.95 	 #gamma

PLAYER_N=1
FOOD_N=2
ENEMY_N=3
# assign colors according to dict
d = {1: (255, 175, 0), 
	 2: (0, 255, 0), 
	 3: (0, 0, 255)}


class Blob:
	def __init__(self):
		self.x = np.random.randint(0, SIZE)
		self.y = np.random.randint(0, SIZE)

	def __str__(self):
		return f"{self.x}, {self.y}"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def action(self, choice):
		if choice == 0:
			self.move(x=1, y=1)
		elif choice == 1:
			self.move(x=-1, y=-1)
		elif choice == 2:
			self.move(x=-1, y=1)
		elif choice == 3:
			self.move(x=1, y=-1)

	def move(self, x=False, y=False):
		if not x:
			self.x += np.random.randint(-1, 2)
		else:
			self.x += x 

		if not y:
			self.y += np.random.randint(-1, 2)
		else:
			self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > SIZE-1:
			self.x = SIZE-1

		if self.y < 0:
			self.y = 0
		elif self.y > SIZE-1:
			self.y = SIZE-1


if start_q_table is None:
	# initialize the q-table#
	q_table = {} # here we take q_table as a dict
	for x1 in range(-SIZE+1, SIZE):
		for y1 in range(-SIZE+1, SIZE): # x1, y1 and x2, y2 are tuples representing the difference b/w player - food and player -enemy coords
			for x2 in range(-SIZE+1, SIZE):
					for y2 in range(-SIZE+1, SIZE):
						q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)] # this means we take 4 uniform values b/w [-5, 1] and store in array for the 4 actions that we can take
else:
	with open(start_q_table, "rb") as f:
		q_table = pickle.load(f)


all_episode_rewards = []

for episode in range(HM_EPISODES+1):
	player = Blob()
	food = Blob()
	enemy = Blob()

	episode_reward = 0

	if episode % SHOW_EVERY == 0:
		show = True
	else:
		show=False

	for i in range(MAX_ITER):
		obs = (player - food, player - enemy)

		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0, 4)

		player.action(action)

		
		# food.move()
		# enemy.move()  # will be implemented later
		

		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY
		elif player.x == food.x and player.y ==food.y:
			reward = FOOD_REWARD
		else:
			reward = -MOVE_PENALTY

		new_obs = (player - food, player - enemy)

		# q_table start ------------------------------------------>
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == FOOD_REWARD:
			new_q = FOOD_REWARD
		elif reward == -ENEMY_PENALTY:
			new_q = -ENEMY_PENALTY
		else:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		q_table[obs][action] = new_q

		# q_table ends ------------------------------------------->
		episode_reward += reward

		if show:
			env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) # starts an rbg of our size
			env[food.y][food.x] = d[FOOD_N] # sets the food location tile to green color
			env[player.y][player.x] = d[PLAYER_N] # sets the player tile to blue
			env[enemy.y][enemy.x] = d[ENEMY_N] # sets the enemy location to red

			img = Image.fromarray(env, 'RGB') # reading to rgb. Apparently. Even tho color definitions are bgr. ???
			img = img.resize((SIZE*30, SIZE*30), resample=Image.BOX) # resizing so we can see our agent in all its glory.
			cv2.imshow("image", np.array(img)) # show it!

			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: # crummy code to hang at the end if we reach abrupt end for good reasons or not.
				if cv2.waitKey(500) & 0xFF == ord('q'):
					break
			else:
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

		
		if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
			break

	all_episode_rewards.append(episode_reward)
	epsilon *= EPSILON_DECAY

	if episode % SHOW_EVERY == 0:
		print(f"Episode No. -> {episode}, Epsilon -> {epsilon}")
		print(f"Avg rewards -> {np.mean(all_episode_rewards[-SHOW_EVERY:])}")
	

moving_avg = np.convolve(all_episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward every {SHOW_EVERY} episodes")
plt.xlabel("Episode #")
plt.show()

with open(f"{SIZE}X{SIZE}_qtable-{int(time.time())}.pickle", "wb") as f:
	pickle.dump(q_table, f)