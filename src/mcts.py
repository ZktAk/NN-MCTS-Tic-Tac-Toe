#
# MCTS algorithm implementation
#

# packages
import math
import random

#from numpy import float128

from NeuralNetwork import _sigmoid
from NeuralNetwork import *


def _adjusted_sigmoid(x):
	test = (2 / (1 + exp(-x))) - 1
	return (2 / (1 + exp(-x, dtype=numpy.longdouble))) - 1

# tree node class definition
# Each TreeNode is a State
# Each TreeNode has children which are Actions
class TreeNode():
	# class constructor (create tree node class instance)
	def __init__(self, board, parent):
		# init associated board state
		self.board = board

		self.is_Init, self.is_fully_expanded = False, False

		self.is_terminal = None

		# init parent node if available
		self.parent = parent

		if parent is not None:

			# init the number of node visits
			self.N = 0

			# init the Prior Probability
			self.P = 0

			# init the Intermediary Value
			self.W = 0

			# init the Action-Value
			self.Q = 0

		# init the total score (number of wins) of the node
		#self.V = 0

		# init current node's children:
		# These are the "edges": The Actions that can be taken from this State
		self.children = []


	# init state's children
	def init(self, policy_output):

		if self.is_Init:
			print("\nSHOULD NOT GET HERE!!!!!!!!!!!!!!!!!!\n")

		# init is node terminal flag
		if self.board.getReward() is not None:
			# we have a terminal node
			self.is_terminal = True

		# otherwise
		else:
			# we have a non-terminal node
			self.is_terminal = False

		# init is fully expanded flag
		self.is_fully_expanded = self.is_terminal if self.is_fully_expanded is False else True

		# init children
		for n in range(9):
			self.children.append(TreeNode(None, self))


		# init prior prob of each child node
		for n in range(len(self.children)):
			#print(f"\ntolist = {policy_output.tolist()}\n{n+1} of {len(self.children)}\n")
			newP = policy_output.tolist()[n]
			self.children[n].P = newP
			#self.children[n].board = self.board.getMove(n)

		self.is_Init = True


# MCTS class definition
class MCTS():

	def __init__(self):
		self.Policy_Net = NeuralNetwork(18, 25, 2, 9)
		self.Value_Net = NeuralNetwork(18, 25, 2, 1)

		self.input_states = []

		self.game_target_values = []

		self.master_target_values = [] # self.master_target_values.extend(self.game_target_values)

		self.target_policies = []


	def update_target_vals(self, reward):

		corrected_val = 1

		# only need to change values if game is drawn
		if reward == 1:
			# update values from end of game to the beginning
			for n in range(len(self.game_target_values)-1, -1, -1): #-1
				self.game_target_values[n] = corrected_val
				corrected_val *= -1

		# append game values to end of master values
		self.master_target_values.extend(self.game_target_values)

		# reset game values list
		self.game_target_values = []

	def get_targets(self):
		return array(self.input_states), array(self.target_policies), array([self.master_target_values]).T

	def reset_targets(self):

		self.input_states = []

		self.game_target_values = []

		self.master_target_values = []  # self.master_target_values.extend(self.game_target_values)

		self.target_policies = []

	# search for the best move in the current position
	def search(self, initial_state):
		# create root node
		self.root = TreeNode(initial_state, None)
		self.root.init(self.Policy_Net.predict(initial_state.getArrBoard().copy())[-1].copy())

		self.input_states.append(initial_state.getArrBoard().copy())
		self.game_target_values.append(0)

		# Allowed thinking time/iterations: walk through 1000 iterations
		for iteration in range(300):
			# select a node (selection phase)
			node = self.select(self.root)

			# score current node
			NN_state = node.board.getArrBoard().copy()
			#print(f"node.board.position.copy() = {NN_state}")
			score = self.Value_Net.predict(NN_state, _adjusted_sigmoid)[-1].copy()

			# backpropagate results
			self.backpropagate(node, score)

		return self.get_move(self.root, 0.2)


	# select most promising node
	def select(self, node):
		# make sure that we're dealing with non-terminal nodes
		while not node.is_terminal:
			# case where the node is fully expanded
			if node.is_fully_expanded:
				node = self.choose_action(node, math.sqrt(2))

			# case where the node is not fully expanded
			else:
				# otherwise expand the node (expansion phase)
				return self.expand(node)

		# return node
		return node


	def expand(self, state):

		if state.is_fully_expanded:
			print("\nSTATE IS ALREADY FULL EXPANDED\n")

		# get the current state position
		pos = state.board.position.copy()

		best_val = float('-inf')
		len_best_val = 0

		numLegal = 0
		numChildren = 0

		# get the prior probabilities of our edges
		#policy_output = self.Policy_Net.predict(pos)[-1].copy()

		for n in range(len(pos)): # -1
			# exclude illegal moves and moves that have already been initialized
			if pos[n] == 0:
				numLegal += 1

				if not state.children[n].is_Init:

					if state.children[n].P > best_val:
						best_val = state.children[n].P
						len_best_val = n

		# init best valued child of state
		new_board = state.board.getMove(len_best_val)
		state.children[len_best_val].board = new_board

		new_board_pos = new_board.getArrBoard().copy()
		policy = self.Policy_Net.predict(new_board_pos)[-1].copy()

		# init the prior policies of the children
		#print(f"policy = {policy}")
		state.children[len_best_val].init(policy)

		# get number of initialized children
		for child in state.children:
			if child.is_Init:
				numChildren += 1

		# case when node is fully expanded
		if numLegal == numChildren:
			state.is_fully_expanded = True

		return state.children[len_best_val]

	# backpropagate the number of visits and score up to the root node
	def backpropagate(self, node, value):

		V = value
		player = -1

		while node.parent is not None:
			# update node's visits
			node.N += 1

			# update intermediary value
			node.W += V * player
			player *= -1

			# update action-value
			node.Q = node.W / node.N

			# set node to parent
			node = node.parent


	# select the best node basing on PUCT formula
	def choose_action(self, state, c): # c is exploration_constant


		best_node_score = float('-inf')
		best_moves = []

		# get the sum of N
		sum_N = 0
		for child in state.children:
			sum_N += child.N

		for child_node in state.children:
			if child_node.is_Init:
				node_score = child_node.Q + (c * child_node.P * math.sqrt(math.log(sum_N)) / (1 + child_node.N))
			else:
				node_score = float('-inf')

			# better move has been found
			if node_score > best_node_score:
				best_node_score = node_score
				best_moves = [child_node]

			# found as good move as already available
			elif node_score == best_node_score:
				best_moves.append(child_node)

		# return one of the best moves randomly
		return random.choice(best_moves)


	# select the move we actualy want ot play
	def get_move(self, state, t):
		"""The temperature parameter t describes how much you
			want to value the visit count of a given edge.
			The closer to zero this is set, the more chance
			there is of choosing greedily with respect to visit
			count, where a temperature parameter of one will
			more readily choose less-explored actions."""

		# get the sum of N
		sum_N = 0
		for child in state.children:
			sum_N += pow(child.N, 1/t)

		best_action_val = float('-inf')
		best_actions = []

		search_probabilities = []

		for n in range(len(state.children)): #-1
			action = state.children[n]
			search_prob = _sigmoid(pow(action.N, 1/t) / sum_N)
			search_probabilities.append(search_prob)

			if search_prob > best_action_val:
				best_action_val = search_prob
				best_actions = [n]

			elif search_prob == best_action_val:
				best_actions.append(n)

		self.target_policies.append(search_probabilities.copy()) # append a list of floats

		return random.choice(best_actions)
