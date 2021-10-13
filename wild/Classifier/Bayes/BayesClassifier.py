# Abstract classifier
from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import KFold, train_test_split

'''
Bayes classifier based on the maximum likelihood
@param prior: 1-D array of prior probability
@param attributes: 2-D array of samples [classes, attributes in classes] 
@param labels: 1-D array of samples labels
@param test_size: percentage of test samples
'''


class BayesClassifier(ABC):

    def __init__(self, prior, attributes, labels, test_size):
        # safety check
        prior = np.array(prior)
        self.classes = np.array(list(set(labels)))
        # classes
        if prior.shape[0] != len(self.classes):
            raise Exception('The length of prior probability could not correspond to labels.')
        # prior
        if np.sum(prior) != 1:
            raise Exception('The prior probabilities are illegal.')
        # test_size
        if test_size < 0 or test_size > 1:
            raise Exception('The test size is illegal.')
        self.prior = prior
        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size

        # record number of classes and attributes per class
        self.cNumb = self.classes.shape[0]
        self.attNumb = attributes.shape[1]

    def leave_out(self, test_size=0.25):
        """
        :param test_size: size of test samples
        :return: X_train, X_test, y_train, y_test
        """
        return train_test_split(self.attributes, self.labels, test_size=test_size)

    '''
        Validation of the model
    '''

    # cross validation
    def cross_val(self, splits=10):
        kf = KFold(n_splits=splits)
        rn = range(0, self.attributes.shape[0])
        test_results = []
        for train_index, test_index in kf.split(rn):
            x_train, x_test = self.attributes.iloc[train_index, :], self.attributes.iloc[test_index, :]
            y_train, y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
            self.train(x_train, y_train)
            test_result = self.test(x_test, y_test)
            test_results.append(test_result)
        return test_results

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the classifier
        :param x_train: 2-D array of training samples [classes, attributes in classes]
        :param y_train: 1-D array of training samples labels
        :return:
        """

    @abstractmethod
    def classify(self, x_):
        """
        Get the class it belongs to.
        :param x_: attributes
        :return: class
        """

    def test(self, x_test, y_test):
        """
        Give the accuracy of classification regarding the provided test data and labels
        :param x_test: 2-D array of test samples [classes, attributes in classes]
        :param y_test: 1-D array of test samples labels
        :return: accuracy
        """
        correct = 0
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        for i in range(0, x_test.shape[0]):
            if self.classify(x_test[i]) == y_test[i]:
                correct += 1
        return correct / x_test.shape[0]