#
# AI that learns to play Tic Tac Toe using
#        reinforcement learning
#                (MCTS)
#

# packages
from copy import deepcopy
from MCTS import *
from MCTS import _adjusted_sigmoid
import matplotlib.pyplot as plt
from params import Training_sets, Training_reps, Tournament_length


# Tic Tac Toe board class
class Board():
	# create constructor (init board class instance)
	def __init__(self, board=None):
		# define players
		self.player_1 = 1
		self.player_2 = -1
		self.empty_square = 0

		# define board position
		self.position = []

		# init (reset) board
		self.init_board()

		# define wining states
		self.xWinPositions = ["0b111000000000000000", "0b000111000000000000", "0b000000111000000000",
		                      "0b100100100000000000", "0b010010010000000000", "0b001001001000000000",
		                      "0b100010001000000000", "0b001010100000000000"]

		self.oWinPositions = ["0b000000000111000000", "0b000000000000111000", "0b000000000000000111",
		                      "0b000000000100100100", "0b000000000010010010", "0b000000000001001001",
		                      "0b000000000100010001", "0b000000000001010100"]

		# create a copy of a previous board state if available
		if board is not None:
			self.__dict__ = deepcopy(board.__dict__)


	# init (reset) board
	def init_board(self):
		self.position = []

		for n in range(9):
			self.position.append(self.empty_square)

		self.player_1 = 1
		self.player_2 = -1


	# print board state
	def print(self):
		# define board string representation
		board_string = ''

		# loop over board rows
		for row in range(3):
			# loop over board columns
			for col in range(3):
				if self.position[col + (row * 3)] == 0:
					board_string += ' %s' % "."
				elif self.position[col + (row * 3)] == 1:
					board_string += ' %s' % "x"
				elif self.position[col + (row * 3)] == -1:
					board_string += ' %s' % "o"

			# print new line every row
			board_string += '\n'

		# prepend side to move
		if self.player_1 == 1:
			board_string = '\n--------------\n "x" to move:\n--------------\n\n' + board_string

		elif self.player_1 == -1:
			board_string = '\n--------------\n "o" to move:\n--------------\n\n' + board_string

		# return board string
		print(board_string)


	# get the len 18 array representation of board state
	def getArrBoard(self):
		arrayBoard = []
		for n in self.position:
			if n == 1:
				arrayBoard.append(1)
			else:
				arrayBoard.append(0)

		for n in self.position:
			if n == -1:
				arrayBoard.append(1)
			else:
				arrayBoard.append(0)

		return arrayBoard

	# get binary (bitboard) representation of board state
	def getBitboard(self):
		bitArray = self.getArrBoard()
		bitBoard = "0b"

		for n in bitArray:
			if n == 1:
				bitBoard += "1"
			elif n == 0:
				bitBoard += "0"

		#print(f"Bitboard = {bitBoard}")

		return bitBoard


	# Return reward if game has reached a terminal state, else return None
	def getReward(self):

		bitBoard = self.getBitboard()

		# check if x won
		for board in self.xWinPositions:

			temp = bin(int(bitBoard, 2) & int(board, 2))

			while len(temp[2:]) < 18:
				temp = temp[:2] + "0" + temp[2:]

			if (temp == board):
				return 1

		# check if o won
		for board in self.oWinPositions:

			temp = bin(int(bitBoard, 2) & int(board, 2))

			while len(temp[2:]) < 18:
				temp = temp[:2] + "0" + temp[2:]

			if (temp == board):
				return 1

		# check if game is drawn
		if (bin(int(bitBoard[2:11], 2) | int(bitBoard[11:], 2)) == "0b111111111"):
			return 0.5

		# if game has not reached a terminal state
		return None


	# make move
	def makeMove(self, index):
		# make move
		self.position[index] = self.player_1


	def getMove(self, index):
		newBoard = Board(self)
		newBoard.makeMove(index)
		return newBoard

	# train the neural networks
	def train(self, sets, reps, len_tourny, data_mcts, trainee_mcts):
		# create MCTS instances

		Val_errors = []
		Policy_errors = []

		print("\nData generation networks initialized")
		data_mcts = data_mcts
		print("Trainee network initialized")
		trainee_mcts = trainee_mcts

		# Train n times

		for n in range(sets):
			print("\nGenerating Data...")

			# Generate data
			for game in range(reps):  # set number of training steps


				if game % math.ceil(reps / 100) == 0:
					percent = int(((n * reps) + game) / (sets * reps) * 100)
					print(f"Training set {n+1} of {sets} | Game {game} of {reps} | {percent}% complete")


				move_num = 0
				self.init_board()

				#print(f"Board = {self.position}")
				#self.print()

				# game loop
				while True:

					# move
					move = data_mcts.search(self)  # returns the index of self.position to change
					move_num += 1
					#print(f"Move = {move+1}")
					self.makeMove(move)

					#print(self.getBitboard())

					# swap players
					(board.player_1, board.player_2) = (board.player_2, board.player_1)

					#self.print()
					#print(f"Board = {self.position}")

					# check if the game is terminal
					reward = self.getReward()
					if reward is not None:

						"""if reward == 0.5:
							print("Game is Drawn")
						elif reward == 1:
							winner = "x" if self.player_1 == -1 else "o"
							print(f"{winner} won!")"""

						data_mcts.update_target_vals(reward)
						break

			percent = int((n+1) / sets * 100)
			print(f"Training set {n + 1} of {sets} | Game {reps} of {reps} | {percent}% complete")
			print("Data Generation Complete")

			# get the training data
			inputs, target_policies, target_values = data_mcts.get_targets()
			data_mcts.reset_targets()

			# train the network
			print(f"\nRound {n+1} Training Started...")
			Val_errors.extend(trainee_mcts.Value_Net.train2(inputs, target_values, 10000, _adjusted_sigmoid))
			Policy_errors.extend(trainee_mcts.Policy_Net.train2(inputs, target_policies, 10000))
			print(f"Round {n+1} Training Complete")

			print("\nTournament Started...")

			print(f"Tournament Complete | Winner = {self.NN_Tournament(data_mcts, trainee_mcts, len_tourny)}")

		print("\n*******************************")
		print(" Neural Network Fully Trained")
		print("*******************************\n")

		plt.plot(Val_errors)
		plt.plot(Policy_errors)
		plt.xlabel("Iterations")
		plt.ylabel("Error for all training instances")
		plt.savefig("cumulative_error.png")

		print(f"\nPolicy Error at End {Policy_errors[-1]}")
		print(f"\nValue Error at End {Val_errors[-1]}")

	def NN_Tournament(self, dataMCTS, traineeMCTS, numGames):

		traineeMCTS_Wins = 0
		dataMCTS_Wins = 0

		for n in range(numGames):

			self.init_board()

			# Determine which MCTS will play the first move.
			# PlayerX always plays first.
			first_to_move = random.choice([1,2])

			if first_to_move == 1:
				PlayerX = dataMCTS
				PlayerO = traineeMCTS
			else:  # first_to_move == 2
				PlayerX = traineeMCTS
				PlayerO = dataMCTS

			while True:
				# X to move
				move = PlayerX.search(self)  # returns the index of self.position to change
				self.makeMove(move)

				#self.print()

				# check if the game is terminal
				reward = self.getReward()
				if reward is not None:

					if first_to_move == 2:
						if reward == 1:
							traineeMCTS_Wins += 1
					elif first_to_move == 1:
						if reward == 1:
							dataMCTS_Wins += 1
					break

				# O to move
				move = PlayerO.search(self)  # returns the index of self.position to change
				self.makeMove(move)

				#self.print()

				# check if the game is terminal
				reward = self.getReward()
				if reward is not None:

					if first_to_move == 1:
						if reward == 1:
							traineeMCTS_Wins += 1
					elif first_to_move == 2:
						if reward == 1:
							dataMCTS_Wins += 1
					break

		dataMCTS.reset_targets()
		traineeMCTS.reset_targets()

		# If the Trainee MCTS won at least 55% of the games
		if traineeMCTS_Wins / numGames >= 0.60:
			trainee_prob_policy = traineeMCTS.Policy_Net.layers
			trainee_val_policy = traineeMCTS.Value_Net.layers

			dataMCTS.Policy_Net.set_policy(trainee_prob_policy)
			dataMCTS.Value_Net.set_policy(trainee_val_policy)

			return f"Trainee MCTS | {traineeMCTS_Wins} out of {numGames} | Draws: {numGames - (traineeMCTS_Wins + dataMCTS_Wins)} | Win Rate: {100 * traineeMCTS_Wins / numGames}%"

		return f"Data Generator MCTS | {dataMCTS_Wins} out of {numGames} | Draws: {numGames - (traineeMCTS_Wins + dataMCTS_Wins)} | Win Rate: {100 * dataMCTS_Wins / numGames}%"


	def game_loop(self, mcts):
		for n in range(1):
			self.init_board()

			# print(f"Board = {self.position}")
			self.print()

			# game loop
			while True:

				# move
				move = mcts.search(self)  # returns the index of self.position to change
				print(f"Move = {move+1}")
				self.makeMove(move)

				# swap players
				(board.player_1, board.player_2) = (board.player_2, board.player_1)

				self.print()

				# check if the game is terminal
				reward = self.getReward()
				if reward is not None:
					if reward == 0.5:
						print("Game is Drawn")
					elif reward == 1:
						winner = "x" if self.player_1 == -1 else "o"
						print(f"{winner} won!")

					mcts.reset_targets()
					break

# main driver
if __name__ == '__main__':
	# create board instance
	board = Board()

	# create MCTS instances
	player = MCTS()
	trainer = MCTS()

	# Train the AI
	board.train(Training_sets, Training_reps, Tournament_length, player, trainer)

	# Let the AI player 1 game against itself to reveal if it has learned anything
	board.game_loop(player)
