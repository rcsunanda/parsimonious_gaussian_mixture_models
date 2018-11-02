# This file implements a Gaussian Mixture Model (GMM) and the Expectation-Maximization (EM) algorithm to learn it
# The interface is based on (similar to) that of sklearn.mixture.GaussianMixture

class GMM():
    def __init__(self, n_components=1, covariance_type='full', max_iter=100,
                 weights_init=None, means_init=None, precisions_init=None):
        assert covariance_type == 'full';

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter

        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        # Following are model parameters that will be set once it is fitted
        self.weights_ = None    # The weights of each mixture components
        self.means_ = None  # The mean of each mixture component
        self.covariances_ = None    # The covariance of each mixture component


    def __repr__(self):
        str = "GMM(n_components={}, covariance_type={}, max_iter={}, " \
              "weights_init={}, means_init={}, precisions_init={})".\
            format(self.n_components, self.covariance_type, self.max_iter,
                   self.weights_init, self.means_init, self.precisions_init)
        return str


    def fit(self, X):
        """Estimate model parameters (train the model) with the EM algorithm"""
        pass


    def predict_proba(self, X):
        """Predict posterior probability of each component given the data"""
        pass


    def predict(self, X):
        """Predict the labels for the data samples in X using trained model"""
        pass


    def bic(self, X):
        """Bayesian information criterion for the current model on the input X"""
        pass


    def aic(self, X):
        """Akaike information criterion for the current model on the input X"""
        pass
