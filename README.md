# Neural Net Tic-Tac-Toe
This project uses a Neural Network guided Monte-Carlo Tree-Search to play Tic-Tac-Toe.

### Usage
```
pip install matplotlib, numpy, pickle
```
```
python3 Tic-Tac-Toe.py
```

# Completed Tasks
1. Implement Monte-Carlo Tree Search Algorithm
2. Rework MCTS framwork to work with Neural Network.
3. Add Neural Network class
4. Elimate all errors and bugs with MCTS + NN framwork
5. Add policy.bin file to store NN policy
6. Implement NN training
7. Preliminary training to assess if NN is learning

# Results of Training
The following image shows the cumulative error of the Neural Network over 50 training rounds of 100 games (5000 games total). Better choices result in lower values on the graph, with valuees < 1 being very accurate. 

Substancial debugging is evidently required.

![cumulative_error](https://user-images.githubusercontent.com/95774165/168077612-991b30fb-d599-4e9e-a296-2dbef1e3642a.png)
