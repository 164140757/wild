import scipy.stats as stats

from main.Classifier.Bayes.BayesClassifier import BayesClassifier
import numpy as np


class GaussianNB(BayesClassifier):

    def __init__(self, prior, attributes, labels, test_size):
        BayesClassifier.__init__(self, prior, attributes, labels, test_size)

    def classify(self, atr):
        posts = []
        # get conditional class probability
        # iterate in classes
        for i in range(0, self.cNumb):
            # set up posterior probability in a class
            post = []
            prior = self.prior[i]
            mu = self.mean[i]
            sigma = self.var[i] ** (1 / 2)
            # iterate attributes
            for j in range(0, self.attNumb):
                post.append(stats.norm.pdf(atr[j], mu[j], sigma[j]))
            posts.append(post)
        posts = np.array(posts)
        # no need to compute denominator as what is going do is comparison of different posterior probability and the
        # denominator is the same
        # add log() for each columns
        tmp = np.zeros(self.cNumb)
        for i in range(0, self.attNumb):
            tmp = tmp + np.log(posts[:, i])

        return self.classes[np.argmax(np.multiply(tmp, self.prior))]

    def train(self, x_train, y_train):
        mean = []
        var = []
        for c in self.classes:
            mean.append(np.mean(x_train[y_train == c], axis=0))
            var.append(np.var(x_train[y_train == c], axis=0))

        # unbiased variance
        N = x_train.shape[0]
        for v in var:
            v = v * N / (N - 1)
        # to numpy array
        self.mean = np.array(mean)
        self.var = np.array(var)
