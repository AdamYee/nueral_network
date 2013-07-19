Machine Learning class project 2.

Part 1: Implementing a Multi-Layer Network
Implement the standard backpropagation algorithm for multi-layer neural network learning. Your
program should have inputs allowing the user to specify the number of hidden layers (between 0 and 3)
and the number of hidden nodes in each layer (arbitrary).

Your program should have an option to dump the learned weights to the screen (or to a file if you
prefer) in a format that makes it easy to understand how the weights apply to the different links in the
network.

You will need to tune parameters such as the learning rate (and whether / how quickly it decreases from
epoch to epoch). Be sure to describe in your analysis document how you determined the best values for
these parameters.

Part 2: Analysis
2a: Tuning the Parameters
Split your training data set into a grow set (90%) and a tuning set (10%). Use the tuning set to
determine the optimal number of training epochs. Graph the results.

Vary the number of hidden layers in your multi-layer network and the number of nodes in each hidden
layer and graph the error rates of the resulting networks. Determine which combination gives you the
best performance on each of the data sets.

Part 2b: Comparing the Results
Use 10-fold cross validation to determine the error rate of your best-sized neural network on the data
set used in project 1 and specify a 95% confidence bound. Compare the results to the results of your
best decision tree from project 1 and to a naïve classifier that simply guessed the majority answer 
(or arandomly selected answer if there are equal numbers of training samples of each output) each time. 
Is one approach clearly better? How does the runtime for each of the learning algorithms compare?

Graph ROC and precision-recall points for your neural network on the same graphs you submitted for
project 1. Describe how the neural network performs relative to the decision tree algorithms.