# This file implements a Gaussian Mixture Model (GMM) and the Expectation-Maximization (EM) algorithm to learn it
# The interface and internals are based on (similar to) the sklearn.mixture.GaussianMixture class

import numpy as np
from scipy.stats import multivariate_normal

class GMM():

    """The following methods are the interface of the class"""

    def __init__(self, n_components=1, covariance_type='full', max_iter=100,
                 weights_init=None, means_init=None, precisions_init=None, verbose=0):

        # Some assertions to enforce limitations of the model (may be able to remove some as implementation progresses)
        assert covariance_type == 'full';
        # Following 3 initial parameters need to be given
        assert weights_init != None;    # This may not be needed
        assert means_init != None;
        assert precisions_init is not None

        assert(len(weights_init) == n_components)
        assert(len(means_init) == n_components)
        assert(len(precisions_init) == n_components)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.verbose = verbose

        # Note: precision is the inverse matrix of the covariance matrix
        covariances_init = [np.linalg.inv(precision_mat) for precision_mat in precisions_init]

        # Following are model parameters that will be set once it is fitted
        self.weights_ = np.array(weights_init)    # The weights of each mixture component
        self.means_ = np.array(means_init)  # The mean of each mixture components
        self.covariances_ = np.array(covariances_init)    # The covariance of each mixture component


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

        print("Initial values: weights={}, means={}, covs={}".format(self.weights_, self.means_, self.covariances_))

        for n_iter in range(1, self.max_iter + 1):
            self._e_step(X, w)
            self._m_step(X, w)

            if (self.verbose > 0):
                print("\nEM iteration={} \nweights={} \nmeans={} \ncovs={} \n".format(n_iter, self.weights_, self.means_, self.covariances_))


    def predict_proba(self, X):
        """Predict posterior probability of each component given the data"""
        num_samples = X.shape[0]
        probabilities = np.zeros([num_samples, self.n_components])  # MxK matrix
        self.sample_component_prob(X, probabilities)
        return probabilities


    def predict(self, X):
        """Predict the labels for the data samples in X using trained model"""
        pred_probabilities = self.predict_proba(X)
        return pred_probabilities.argmax(axis=1)    #argmax(axis=1) gives the index of the column with max value, for each row


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
        self.sample_component_prob(X, w)    # The actual calculation is delegated to this method (to reuse)


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

        # First set new weights

        for j, weight in enumerate(self.weights_):

            w_sum = 0

            for i, sample in enumerate(X):
                w_sum = w_sum + w[i][j]

            self.weights_[j] = w_sum / num_samples


        # Then set new means

        means = np.zeros((self.n_components, data_dimension))   # Compute means in new variable and assign to self.means_ (to fix bug: unclear why this is the case)

        for j, weight in enumerate(self.weights_):
            w_sum = 0
            mean_numerator = np.zeros([data_dimension])

            for i, sample in enumerate(X):
                w_sum = w_sum + w[i][j]
                mean_numerator += w[i][j] * sample

            means[j] = mean_numerator / w_sum

        self.means_ = means


        # Then set new covariance matrix

        for j, weight in enumerate(self.weights_):

            cov_numerator_total = np.zeros([data_dimension, data_dimension])

            for i, sample in enumerate(X):
                diff = sample - self.means_[j]
                diff_reshaped = np.reshape(diff, [len(diff), 1])
                cov_numerator = w[i][j] * np.dot(diff_reshaped, diff_reshaped.T)
                cov_numerator_total = cov_numerator_total + cov_numerator

            # print("j={}, w_sum={}, cov_numerator={}".format(j, w_sum, cov_numerator_total))
            self.covariances_[j] = cov_numerator_total / w[:, j].sum()


        # print(self.means[j].shape)

        # Correctness check
        weights_sum = self.weights_.sum()
        assert (abs(weights_sum - 1) < 1e-8)


    def sample_component_prob(self, X, probs_matrix):
        """
        Compute the probability that each sample in X came from each component (under the current model params)
        Calculated values are set in probs_matrix (MxK matrix where M = no. of samples in X, K = no. of Gaussian components)
        """
        prob_x_given_z = [multivariate_normal(mean=mean, cov=cov)
                          for mean, cov in zip(self.means_, self.covariances_)]

        for i, sample in enumerate(X):
            for j, component_dist in enumerate(prob_x_given_z):
                numerator = component_dist.pdf(sample) * self.weights_[j]
                probs_matrix[i][j] = numerator

        probs_matrix /= probs_matrix.sum(axis=1, keepdims=True)

        # Correctness check
        expected = np.ones([len(X), 1])  # MxK matrix
        prob_z_given_x = probs_matrix.sum(axis=1)
        is_equal = np.allclose(prob_z_given_x, expected)
        assert(is_equal)



def em_gmm_orig(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)

        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (2,1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j,:].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas