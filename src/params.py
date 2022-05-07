"""This file contains all parameters that are arbitrarily set and
   that impact how much time it requires to run the code."""


# The following params set the size and shape of the two Neural Networks

Policy_NN_Size = 25  # sets the size of each hidden layer in Policy Neural Network
Value_NN_Size = 25  # sets the size of each hidden layer in Value Neural Network

Policy_NN_Layers = 2  # sets the number hidden layer in Policy Neural Network
Value_NN_Layers = 2  # sets the number hidden layer in Value Neural Network

#-----------------------------------------------------------------------------------

# The following param sets how many iterations the algorithm is allowed to think about each move
Thinking_time = 300

#-----------------------------------------------------------------------------------

# The following params set how many reps and set the Neural Networks are trained

Training_sets = 50  # how many sets the NN's are trained for
Training_reps = 100  # how many reps are in each training set

"""After each training set, the newly trained MCTS + NN is pitted against
   the previous version of itself in a tournament of n number of rounds in 
   order to determine which version performs the best. The following param 
   sets the value of n."""

Tournament_length = 100
