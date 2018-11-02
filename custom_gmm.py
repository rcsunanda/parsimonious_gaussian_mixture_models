# This file implements a Gaussian Mixture Model (GMM) and the Expectation-Maximization (EM) algorithm to learn it
# The interface and internals are based on (similar to) the sklearn.mixture.GaussianMixture class

import numpy as np
from scipy.stats import multivariate_normal

class GMM():

    """The following methods are the interface of the class"""

    def __init__(self, n_components=1, covariance_type='full', max_iter=100,
                 weights_init=None, means_init=None, precisions_init=None):

        # Some assertions to enforce limitations of the model (may be able to remove some as implementation progresses)
        assert covariance_type == 'full';
        # Following 3 initial parameters need to be given
        assert weights_init != None;    # This may not be needed
        assert means_init != None;
        assert precisions_init is not None

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter

        # Note: precision is the inverse matrix of the covariance matrix
        covariances_init = precisions_init.copy()
        for j, precision_mat in enumerate(precisions_init):
            covariances_init[j] = np.linalg.inv(precision_mat)

        # Following are model parameters that will be set once it is fitted
        self.weights_ = weights_init    # The weights of each mixture component
        self.means_ = means_init  # The mean of each mixture components
        self.covariances_ = covariances_init    # The covariance of each mixture component


    def fit(self, X):
        """Estimate model parameters (train the model) with the EM algorithm"""

        num_samples = X.shape[0]
        data_dimension = X.shape[1]

        # self.weights = np.zeros(self.n_components)
        # self.means = np.zeros([self.n_components, data_dimension])
        # self.covariances = np.zeros([self.n_components, data_dimension, data_dimension])

        # We initialize the 'w' matrix here and pass to e_step
        # This avoids unnecessary memory allocs/deallocs (that may occur if w were to be created and as destroyed in _e_step() at each iteration)
        w = np.zeros([num_samples, self.n_components])  # MxK matrix

        for n_iter in range(1, self.max_iter + 1):
            print("\nEM iteration={}".format(n_iter))
            self._e_step(X, w)
            self._m_step(X, w)
            print("weights={}, means={}, covs={}".format(self.weights_, self.means_, self.covariances_))

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data"""
        pass


    def predict(self, X):
        """Predict the labels for the data samples in X using trained model"""
        pass


    def bic(self, X):
        """Bayesian information criterion for the current model on the input X"""
        return 100


    def aic(self, X):
        """Akaike information criterion for the current model on the input X"""
        return 100


    """The following methods are the internals of the class"""

    def __repr__(self):
        str = "GMM(n_components={}, covariance_type={}, max_iter={}, " \
              "weights_init={}, means_init={}, precisions_init={})".\
            format(self.n_components, self.covariance_type, self.max_iter,
                   self.weights_init, self.means_init, self.precisions_init)
        return str


    def _e_step(self, X, w):
        """E step.
        We compute the soft weights (probabilities) for each sample (i) and each component (j) according to below equation
        w(i,j)  = p(z(i)=j | x(i), current_params)
                = (p(x(i,j) | z(i)) * weights(j)) / sum_over_j(numerator)
                where  p(x(i,j) | z(i)) ~ Gaussian(means(j), covariances(j))
                = (this is a MxK matrix where M = no. of samples, K = no. of Gaussian components)
        """

        prob_x_given_z = []
        for j, (mean, cov) in enumerate(zip(self.means_, self.covariances_)):
            # print("j={}, weights={}, mean={}, cov={}".format(j, self.weights, mean, cov))
            prob_x_given_z.append(multivariate_normal(mean=mean, cov=cov, allow_singular=True))

        # prob_x_given_z = [multivariate_normal(mean=mean, cov=cov)
        #                   for mean, cov in zip(self.means, self.covariances)]

        for i, sample in enumerate(X):
            denominator_sum = 0
            for j, component_dist in enumerate(prob_x_given_z):
                denominator_sum = denominator_sum + component_dist.pdf(sample) * self.weights_[j]

            for j, component_dist in enumerate(prob_x_given_z):
                numerator = component_dist.pdf(sample) * self.weights_[j]
                w[i][j] = numerator / denominator_sum


    def _m_step(self, X, w):
        """M step.
        We compute the parameters of each Guassian component (j) according to below equations

        weights(j)  = p(z(j)) = coefficient of used to multiply jth Gaussuan component = mean of w(i,j) over i
                = sum_over_i(w(i),j) / M    where M = no. of samples
                = (this is a scalar) --> K such scalars in a Kx1 vector

        means(j)   = mean of jth Gaussian component = mean of x w.r.t probability distribution w(j)
                = sum_over_i(w(i,j) * x(i)) / sum_over_i(w(i,j)
                = (this is a 1xD vector where D = dimension of a data sample x(i)) --> K such vectors in a KxD matrix

        covariances(j)  = covariance of jth Gaussian component = covariance of x w.r.t probability distribution w(j)
                = sum_over_i(w(i,j) * (x(i) - means(j)) * transpose((x(i) - means(j))) / sum_over_i(w(i,j)
                = (this is a DxD matrix where D = dimension of a data sample x(i)) --> K such matrices in a KxDxD matrix
        """

        num_samples = X.shape[0]
        data_dimension = X.shape[1]

        for j, weight in enumerate(self.weights_):
            w_sum = 0
            mean_numerator = np.zeros([data_dimension])
            cov_numerator_total = np.zeros([data_dimension, data_dimension])

            for i, sample in enumerate(X):
                w_sum = w_sum + w[i][j]
                mean_numerator = mean_numerator + w[i][j] * sample

            self.weights_[j] = w_sum / num_samples
            self.means_[j] = mean_numerator / w_sum

            for i, sample in enumerate(X):
                cov_numerator = w[i][j] * np.matmul(sample - self.means_[j], np.transpose(sample - self.means_[j]))
                cov_numerator_total = cov_numerator_total + cov_numerator

            # print("j={}, w_sum={}, cov_numerator={}".format(j, w_sum, cov_numerator_total))
            self.covariances_[j] = cov_numerator_total / w_sum

        # print(self.means[j].shape)
