#
# MCTS algorithm implementation
#

# packages
import math
import random
from NeuralNetwork import *

# tree node class definition
# Each TreeNode is a State
# Each TreeNode has children which are Actions
class TreeNode():
	# class constructor (create tree node class instance)
	def __init__(self, board, parent):
		# init associated board state
		self.board = board

		# init is node terminal flag
		if self.board.is_win() or self.board.is_draw():
			# we have a terminal node
			self.is_terminal = True

		# otherwise
		else:
			# we have a non-terminal node
			self.is_terminal = False

		# init is fully expanded flag
		self.is_fully_expanded = self.is_terminal

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
		self.children = {}

# MCTS class definition
class MCTS():

	def __init__(self):
		self.Policy_Net = NeuralNetwork(9, 12, 2, 9)
		self.Value_Net = NeuralNetwork(9, 12, 1, 1)

	# search for the best move in the current position
	def search(self, initial_state):
		# create root node
		self.root = TreeNode(initial_state, None)

		# Allowed thinking time/iterations: walk through 1000 iterations
		for iteration in range(1000):
			# select a node (selection phase)
			node, isTerminal = self.select(self.root)

			if not isTerminal:
				# expand and score the node
				node = self.expand(node)

			# score current node (simulation phase)
			NN_state = self.arrayTo1D(node.board)
			score = self.Value_Net.predict(NN_state)[-1]  # self.rollout(node.board)


			# backpropagate results
			self.backpropagate(node, score)

		# pick up the best move in the current position
		try:
			return self.play(self.root, 0.1)  # self.get_best_move(self.root, 0)

		except:
			pass

	# select most promising node
	def select(self, node):
		# make sure that we're dealing with non-terminal nodes
		while not node.is_terminal:
			# case where the node is fully expanded
			if node.is_fully_expanded:
				node = self.get_best_move(node, 2)

			# case where the node is not fully expanded
			else:
				# otherwise expand the node
				return node, False # self.expand(node)

		# return node
		return node, True

	# convert 2D state array to 1D array
	def arrayTo1D(self, array2D):

		array1D = []

		if array2D is not None:
			for player in array2D.position.values():
				if player == "x":
					val = 1
				elif player == "o":
					val = -1
				elif player == ".":
					val = 0
				else:
					print("SHOULD NOT GET HERE")

				array1D.append(val)
		else:
			array1D.append(None)

		return array1D

	# expand node
	def expand(self, state):

		# convert board state to 1D array to feed to Neural Net
		NN_state = self.arrayTo1D(state.board)

		policy_output = self.Policy_Net.predict(NN_state)[-1]  # the prior probabilities of our edges

		# generate legal actions (formerly states) for the given state (formerly node)
		actions = state.board.generate_actions()

		lenActions = 0

		best_val = float('-inf')
		best_move = None

		for n in range(len(actions)):
			# make sure that current action (state) is not present in child nodes
			if actions[n] is not None:
				lenActions += 1
				policy = policy_output[n]

				if str(actions[n].position) not in state.children:
					if policy > best_val:
						best_val = policy
						best_move = n

		# create a new node
		new_node = TreeNode(actions[best_move], state)

		# add child node to parent's node children list (dict)
		state.children[str(actions[best_move].position)] = new_node

		# case when node is fully expanded
		if lenActions == len(state.children):
			state.is_fully_expanded = True

		# return newly created node
		return new_node

	# backpropagate the number of visits and score up to the root node
	def backpropagate(self, node, value):
		# update nodes's up to root node

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
	def get_best_move(self, state, exploration_constant):

		# define best score & best moves
		best_score = float('-inf')
		best_moves = []

		# get the sum
		total_n = 1
		for child in state.children.values():
			total_n += child.N


		# loop over child nodes
		for child_node in state.children.values():

			# define current player
			current_player = 1 if child_node.board.player_2 == 'x' else -1

			# get move score based on Polynomial Upper Confidence Tree (PUCT)
			move_score = current_player * child_node.Q + (exploration_constant * child_node.P * math.sqrt(math.log(total_n) / (1 + child_node.N)))

			# better move has been found
			if move_score > best_score:
				best_score = move_score
				best_moves = [child_node]

			# found as good move as already available
			elif move_score == best_score:
				best_moves.append(child_node)

		# return one of the best moves randomly
		return random.choice(best_moves)

	# select the move we actualy want ot play
	def play(self, state, t):

		"""The temperature parameter t describes how much you
		want to value the visit count of a given edge.
		The closer to zero this is set, the more chance
		there is of choosing greedily with respect to visit
		count, where a temperature parameter of one will
		more readily choose less-explored actions."""

		# get the sum
		sumN = 1
		for child in state.children.values():
			sumN += pow(child.N, 1/t)

		best_action_val = float('-inf')
		best_actions = []
		search_probabilities = []

		for action in state.children.values():
			search_probabilities.append(pow(action.N, 1/t) / sumN)
			if (pow(action.N, 1/t) / sumN) >= best_action_val:
				best_action_val = pow(action.N, 1/t) / sumN
				best_actions.append(action)

		return random.choice(best_actions), search_probabilities
