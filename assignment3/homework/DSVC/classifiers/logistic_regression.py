import numpy as np
import random
import math


class LogisticRegression(object):

    def __init__(self):
        self.w = None

    def sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################

        #         %calculation J
        #         z = X*theta;
        #         t = -y' * log(sigmoid(z)) - (1 - y') *  log (1 - sigmoid(z));
        #         J = t / m;

        # the loss
        m = X_batch.shape[0]
        z = X_batch.dot(self.w)
        J = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1 - y_batch.T).dot(np.log(1 - self.sigmoid(z)))) / m

        # derivative
        derivative = np.dot(X_batch.T, self.sigmoid(z) - y_batch) / m

        return J, derivative
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
              batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            sample_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[sample_indices, :]
            y_batch = y[sample_indices]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, derivative = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= derivative * learning_rate
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        z = X.dot(self.w)
        h = self.sigmoid(z)
        for i in range(X.shape[0]):
            if h[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
       
        if self.mw is not None:
            lables_num = self.mw.shape[1]
            h = self.sigmoid(X.dot(self.mw))
            y_pred = np.argmax(h,axis=1)
        
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """

        lables = list(set(y))
        m, n = X.shape
        loss_history = np.zeros((len(lables), num_iters))
        self.mw = np.zeros((n, len(lables)))
        for l in range(len(lables)):
            index = np.where(y == lables[l])
            train_y = np.zeros_like(y)
            train_y[index] = 1
            self.w = None
            loss_history[l] = self.train(X, train_y, learning_rate, num_iters, batch_size)
            self.mw[:, l] = self.w.T
        #print(self.mw)