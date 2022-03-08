#
# MCTS algorithm implementation
#

# packages
import math
import random
from NeuralNetwork import *

# tree node class definition
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
        
        # init the number of node visits
        self.visits = 0
        
        # init the total score (number of wins) of the node
        self.score = 0
        
        # init current node's children
        self.children = {}

# MCTS class definition
class MCTS():

    def __init__(self):
        self.W = dict()  # total reward of taking action from state
        self.Q = dict()  # average reward of taking action from state
        self.N = dict()  # total visit count of taking action from state
        self.actions = dict()  # possible actions for each state, each state corresponding to a list
        self.policy = dict()

        self.Net = NeuralNetwork(9, 12, 2, 10)

    # search for the best move in the current position
    def search(self, initial_state):
        # create root node
        self.root = TreeNode(initial_state, None)

        # walk through 1000 iterations
        for iteration in range(1000):
            # select a node (selection phase)
            node, isTerminal = self.select(self.root)

            if not isTerminal:
                # expand and score the node
                node = self.expand(node)

            
            # score current node (simulation phase)
            score = self.rollout(node.board)
            
            # backpropagate results
            self.backpropagate(node, score)
        
        # pick up the best move in the current position
        try:
            return self.get_best_move(self.root, 0)
        
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

        for row in array2D:
            for col in row:
                array1D.append(array2D[row, col])

        return array1D

    # expand node
    def expand(self, state):

        # convert board state to 1D array to feed to Neural Net
        NN_state = self.arrayTo1D(state.board)

        # get Neural Network predictions
        output = self.Net.predict(NN_state)
        value_output = output[-1]
        output.pop(-1)
        policy_output = output.copy()  # the prior probabilities of our edges

        # generate legal actions (formerly states) for the given state (formerly node)
        actions = state.board.generate_actions()

        best_val = '-inf'
        best_move = None

        # loop over generated actions (states)
        for n in range(len(actions)):
            # make sure that current action (state) is not present in child nodes
            if str(actions[n].position) not in state.children:

                if policy_output[n] > best_val:
                    best_val = policy_output
                    best_move = n

        # create a new node
        new_node = TreeNode(actions[best_move], state)
                
        # add child node to parent's node children list (dict)
        state.children[str(actions[best_move].position)] = new_node
                
        # case when node is fully expanded
        if len(actions) == len(state.children):
            state.is_fully_expanded = True
                
        # return newly created node
        return new_node
    
    # simulate the game via making random moves until reach end of the game
    def rollout(self, board):
        # make random moves for both sides until terminal state of the game is reached
        while not board.is_win():
            # try to make a move
            try:
                # make the on board
                board = random.choice(board.generate_states())
                
            # no moves available
            except:
                # return a draw score
                return 0
        
        # return score from the player "x" perspective
        if board.player_2 == 'x': return 1
        elif board.player_2 == 'o': return -1
                
    # backpropagate the number of visits and score up to the root node
    def backpropagate(self, node, score):
        # update nodes's up to root node
        while node is not None:
            # update node's visits
            node.visits += 1
            
            # update node's score
            node.score += score
            
            # set node to parent
            node = node.parent
    
    # select the best node basing on UCB1 formula
    def get_best_move(self, state, exploration_constant):

        # define best score & best moves
        best_score = float('-inf')
        best_moves = []

        # get the sum
        total_n = 1
        actionList = state.children.values()
        for action in actionList:
            total_n += self.N[state][action]
        
        # loop over child nodes
        for child_node in state.children.values():
            # define current player
            if child_node.board.player_2 == 'x': current_player = 1
            elif child_node.board.player_2 == 'o': current_player = -1
            
            # get move score
            move_score = current_player * self.Q[state][child_node] + (exploration_constant * self.policy[state][child_node] * math.sqrt(math.log(total_n) / (1 + self.N[state][child_node])))

            # better move has been found
            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]
            
            # found as good move as already available
            elif move_score == best_score:
                best_moves.append(child_node)
            
        # return one of the best moves randomly
        return random.choice(best_moves)




























